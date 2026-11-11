"""
scSimilarity experiment for hierarchical cell-type annotation.

scSimilarity uses a metric-learning encoder (MLP + BatchNorm + PReLU + L2-norm)
to produce cell embeddings. The pretrained model expects ~20k genes; our MERFISH
panel has only 140 genes (0.7% coverage), far below the 5000-gene minimum the
authors enforce in align_dataset. Zero-filling 99.3% of the input would produce
uninformative embeddings.

Instead, we instantiate the scSimilarity Encoder architecture with n_genes=140
and train from scratch on our data. Two modes:

  (a) freeze  -- Train encoder+decoder (reconstruction), freeze encoder,
                  extract embeddings, train sklearn LogisticRegression.
  (b) full    -- Encoder + 3 classification heads, end-to-end fine-tuning
                  via ScGPTTrainerEngine.
"""
import sys
import os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scimilarity.nn_models import Encoder, Decoder

from src.utils.dataloader import build_dataloaders
from src.utils.logger import get_logger
from src.utils.args import get_public_config


# ── Classification head (same as Stack) ─────────────────────────────────────

class ClsDecoder(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers += [nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU()]
        layers.append(nn.Linear(d_model, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Full fine-tune model ────────────────────────────────────────────────────

class ScSimilarityForAnnotation(nn.Module):
    """
    scSimilarity Encoder (from scratch, 140 genes) + classification heads.
    Forward returns (logits_list, cell_emb) matching ScGPTTrainerEngine API.
    """

    def __init__(
        self,
        n_genes: int = 140,
        latent_dim: int = 128,
        hidden_dim: list = None,
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        output_num: list = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [512, 512]
        if output_num is None:
            output_num = [3, 24, 137]

        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.freeze_backbone = freeze_backbone

        # scSimilarity encoder: MLP -> L2-normalized embeddings
        self.encoder = Encoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_dropout=input_dropout,
        )

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[scSimilarity] Encoder frozen.")

        # Classification heads
        self.cls_head_class = ClsDecoder(latent_dim, output_num[0])
        self.cls_head_subclass = ClsDecoder(latent_dim, output_num[1])
        self.cls_head_supertype = ClsDecoder(latent_dim, output_num[2])

        print(
            f"[scSimilarity] n_genes={n_genes}, latent_dim={latent_dim}, "
            f"hidden_dim={hidden_dim}, freeze={freeze_backbone}"
        )

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, 140) raw gene expression.
        Returns:
            logits: list of [(B, 3), (B, 24), (B, 137)]
            cell_emb: (B, latent_dim)
        """
        x = torch.log1p(X)

        if self.freeze_backbone:
            with torch.no_grad():
                cell_emb = self.encoder(x)
        else:
            cell_emb = self.encoder(x)

        logit_c = self.cls_head_class(cell_emb)
        logit_sc = self.cls_head_subclass(cell_emb)
        logit_st = self.cls_head_supertype(cell_emb)

        return [logit_c, logit_sc, logit_st], cell_emb


# ── Freeze mode: pretrain encoder, then sklearn LR ─────────────────────────

class ScSimilarityAutoencoder(nn.Module):
    """Encoder + Decoder for reconstruction pretraining."""

    def __init__(self, n_genes=140, latent_dim=128, hidden_dim=None,
                 dropout=0.5, input_dropout=0.4):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [512, 512]
        self.encoder = Encoder(
            n_genes=n_genes, latent_dim=latent_dim,
            hidden_dim=hidden_dim, dropout=dropout, input_dropout=input_dropout,
        )
        self.decoder = Decoder(
            n_genes=n_genes, latent_dim=latent_dim,
            hidden_dim=list(reversed(hidden_dim)), dropout=dropout,
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def pretrain_autoencoder(ae_model, train_loader, val_loader, device, logger, args):
    """Train encoder+decoder with MSE reconstruction loss."""
    optimizer = torch.optim.AdamW(
        ae_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.ae_epochs, eta_min=1e-6,
    )
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(args.save_dir, 'ae_best.pth')

    for epoch in range(1, args.ae_epochs + 1):
        # Train
        ae_model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"AE Train {epoch}"):
            X = torch.log1p(batch['X'].to(device))
            optimizer.zero_grad()
            z, x_hat = ae_model(X)
            loss = F.mse_loss(x_hat, X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        ae_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"AE Val {epoch}"):
                X = torch.log1p(batch['X'].to(device))
                z, x_hat = ae_model(X)
                val_loss += F.mse_loss(x_hat, X).item()
        val_loss /= len(val_loader)
        scheduler.step()

        logger.info(
            f"AE Epoch [{epoch:03d}/{args.ae_epochs}] | "
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ae_model.state_dict(), save_path)
            logger.info("  Best AE model saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("  AE early stopping.")
                break

    ae_model.load_state_dict(torch.load(save_path, map_location=device))
    return ae_model


@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    """Extract L2-normalized embeddings from frozen encoder."""
    encoder.eval()
    all_emb, all_y = [], {'class': [], 'subclass': [], 'supertype': []}
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        X = torch.log1p(batch['X'].to(device))
        emb = encoder(X)
        all_emb.append(emb.cpu().numpy())
        all_y['class'].append(batch['y_class'].numpy())
        all_y['subclass'].append(batch['y_subclass'].numpy())
        all_y['supertype'].append(batch['y_supertype'].numpy())
    emb = np.concatenate(all_emb)
    labels = {k: np.concatenate(v) for k, v in all_y.items()}
    return emb, labels


def run_freeze_mode(args, train_loader, val_loader, test_loader, device, logger):
    """Freeze mode: pretrain AE, extract embeddings, train sklearn LR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    logger.info("=" * 60)
    logger.info("MODE: freeze (AE pretrain -> sklearn LR)")
    logger.info("=" * 60)

    ae = ScSimilarityAutoencoder(
        n_genes=args.input_dim, latent_dim=args.scsi_latent_dim,
        hidden_dim=args.scsi_hidden_dim, dropout=args.scsi_dropout,
        input_dropout=args.scsi_input_dropout,
    ).to(device)

    ae = pretrain_autoencoder(ae, train_loader, val_loader, device, logger, args)
    encoder = ae.encoder

    # Extract embeddings
    logger.info("Extracting train embeddings...")
    train_emb, train_labels = extract_embeddings(encoder, train_loader, device)
    logger.info("Extracting test embeddings...")
    test_emb, test_labels = extract_embeddings(encoder, test_loader, device)
    logger.info(f"Train: {train_emb.shape}, Test: {test_emb.shape}")

    # Save embeddings
    emb_path = os.path.join(args.save_dir, 'embeddings.npz')
    np.savez_compressed(emb_path, train_emb=train_emb, test_emb=test_emb)
    logger.info(f"Embeddings saved to {emb_path}")

    # Train sklearn LR for each hierarchy level
    levels = [('class', 3), ('subclass', 24), ('supertype', 137)]
    for level_name, n_cls in levels:
        logger.info(f"\n--- {level_name} ({n_cls} classes) ---")
        solver = 'saga' if n_cls > 30 else 'lbfgs'
        clf = LogisticRegression(
            max_iter=1000, solver=solver, multi_class='multinomial',
            C=1.0, n_jobs=10, verbose=0,
        )
        clf.fit(train_emb, train_labels[level_name])
        y_pred = clf.predict(test_emb)
        y_prob = clf.predict_proba(test_emb)

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            test_labels[level_name], y_pred, average='macro', zero_division=0)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            test_labels[level_name], y_pred, average='micro', zero_division=0)
        logger.info(
            f"[{level_name.upper()}] Macro  - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
        logger.info(
            f"[{level_name.upper()}] Micro  - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")

        # Per-class
        per_p, per_r, _, support = precision_recall_fscore_support(
            test_labels[level_name], y_pred, average=None, zero_division=0)
        parts = [f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}"
                 for i in range(len(per_p)) if support[i] > 0]
        logger.info(f"[{level_name.upper()}] Per-class: {' | '.join(parts)}")

        # AUC-ROC
        y_onehot = np.zeros_like(y_prob)
        y_onehot[np.arange(len(test_labels[level_name])), test_labels[level_name]] = 1
        micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())
        valid_aucs = []
        for c in range(n_cls):
            y_bin = (test_labels[level_name] == c).astype(int)
            if len(np.unique(y_bin)) == 2:
                valid_aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        logger.info(
            f"[{level_name.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | "
            f"Micro: {micro_auc:.4f} | (Valid: {len(valid_aucs)}/{n_cls})")
        logger.info("-" * 50)


# ── Full fine-tune mode ────────────────────────────────────────────────────

def run_full_mode(args, train_loader, val_loader, test_loader, device, logger):
    """Full fine-tune: encoder + cls heads, end-to-end."""
    from src.engines.scgpt_trainer import ScGPTTrainerEngine

    logger.info("=" * 60)
    logger.info("MODE: full (end-to-end fine-tuning)")
    logger.info("=" * 60)

    model = ScSimilarityForAnnotation(
        n_genes=args.input_dim,
        latent_dim=args.scsi_latent_dim,
        hidden_dim=args.scsi_hidden_dim,
        dropout=args.scsi_dropout,
        input_dropout=args.scsi_input_dropout,
        output_num=args.output_num,
        freeze_backbone=False,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6,
    )

    engine = ScGPTTrainerEngine(model, optimizer, scheduler, device, logger, args)

    if args.mode == 'train':
        engine.train(train_loader, val_loader)
        engine.test(test_loader)
    else:
        engine.test(test_loader)


# ── CLI ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = get_public_config()
    parser.set_defaults(model_name="scSimilarity", dropout=0.5)

    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze mode: AE pretrain + sklearn LR.')
    parser.add_argument('--ae_epochs', type=int, default=100,
                        help='Max epochs for AE pretraining (freeze mode).')

    # scSimilarity architecture
    parser.add_argument('--scsi_latent_dim', type=int, default=128)
    parser.add_argument('--scsi_hidden_dim', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--scsi_dropout', type=float, default=0.5)
    parser.add_argument('--scsi_input_dropout', type=float, default=0.4)

    args = parser.parse_args()

    ft_tag = "freeze" if args.freeze_backbone else "full"
    ckpt_tag = f"{ft_tag}_s{args.seed}"
    args.ckpt_tag = ckpt_tag

    log_dir = "./experiments/{}/{}/".format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, f"record_{ckpt_tag}.log")
    logger.info(args)

    return args, log_dir, logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args, log_dir, logger = get_args()
    args.log_dir = log_dir
    args.save_dir = os.path.join(log_dir, "checkpoints", args.ckpt_tag)
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    print("====== scSimilarity ======")

    # Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        npz_path=args.npz_path, batch_size=args.bs,
        seed=args.seed, num_workers=args.num_workers,
    )

    if args.freeze_backbone:
        run_freeze_mode(args, train_loader, val_loader, test_loader, device, logger)
    else:
        run_full_mode(args, train_loader, val_loader, test_loader, device, logger)
