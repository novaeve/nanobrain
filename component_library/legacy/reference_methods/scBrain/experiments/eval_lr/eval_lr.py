#!/usr/bin/env python
"""
Evaluate MJM encoder features with sklearn LogisticRegression (CellTypist-style).

For each trained MJM model:
  1. Load checkpoint, extract encoder features (h1, h2, H) on train+test sets
  2. Train sklearn LogisticRegression on train features
  3. Evaluate on test features
  4. Log classification metrics (P/R/F1, AUC-ROC)

Usage:
    python experiments/eval_lr/eval_lr.py --device cuda:4
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.utils.dataloader import build_dataloaders
from src.utils.dataloader_v2 import build_dataloaders_v2
from src.utils.logger import get_logger


# ── Model registry ──────────────────────────────────────────────────────────

MODELS = [
    # (name, model_class_path, checkpoint_path, dataloader_version, extra_model_kwargs)
    # mjm (original, dec256)
    {
        'name': 'mjm_dec256',
        'module': 'src.models.mjm', 'cls': 'mjm',
        'ckpt': 'experiments/mjm/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec256_z20_s3028/best_model.pth',
        'dl': 'v1', 'kwargs': {'dec_hidden_dim': 256},
    },
    # mjm_1 variants
    {
        'name': 'mjm_1_dec128_feature',
        'module': 'src.models.mjm', 'cls': 'mjm_1',
        'ckpt': 'experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_resfeature/best_model.pth',
        'dl': 'v1', 'kwargs': {'dec_hidden_dim': 128, 'residual_mode': 'feature'},
    },
    {
        'name': 'mjm_1_dec128_logit',
        'module': 'src.models.mjm', 'cls': 'mjm_1',
        'ckpt': 'experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_reslogit/best_model.pth',
        'dl': 'v1', 'kwargs': {'dec_hidden_dim': 128, 'residual_mode': 'logit'},
    },
    {
        'name': 'mjm_1_dec128_none',
        'module': 'src.models.mjm', 'cls': 'mjm_1',
        'ckpt': 'experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_resnone/best_model.pth',
        'dl': 'v1', 'kwargs': {'dec_hidden_dim': 128, 'residual_mode': 'none'},
    },
    {
        'name': 'mjm_1_dec256_feature',
        'module': 'src.models.mjm', 'cls': 'mjm_1',
        'ckpt': 'experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec256_z20_s3028_resfeature/best_model.pth',
        'dl': 'v1', 'kwargs': {'dec_hidden_dim': 256, 'residual_mode': 'feature'},
    },
    {
        'name': 'mjm_1_dec64_feature',
        'module': 'src.models.mjm', 'cls': 'mjm_1',
        'ckpt': 'experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec64_z20_s3028_resfeature/best_model.pth',
        'dl': 'v1', 'kwargs': {'dec_hidden_dim': 64, 'residual_mode': 'feature'},
    },
    # mjm_2 variants
    {
        'name': 'mjm_2a_depth',
        'module': 'src.models.mjm_2', 'cls': 'mjm_2a',
        'ckpt': 'experiments/mjm_2/SEA_AD_MTP_ST/checkpoints/a_e3_d1_enc256_dec128_z20_s3028_resfeature_ldepth0.1/best_model.pth',
        'dl': 'v2', 'kwargs': {},
    },
    {
        'name': 'mjm_2b_volume',
        'module': 'src.models.mjm_2', 'cls': 'mjm_2b',
        'ckpt': 'experiments/mjm_2/SEA_AD_MTP_ST/checkpoints/b_e3_d1_enc256_dec128_z20_s3028_resfeature_lvol0.1/best_model.pth',
        'dl': 'v2', 'kwargs': {},
    },
    {
        'name': 'mjm_2c_film',
        'module': 'src.models.mjm_2', 'cls': 'mjm_2c',
        'ckpt': 'experiments/mjm_2/SEA_AD_MTP_ST/checkpoints/c_e3_d1_enc256_dec128_z20_s3028_resfeature_film64/best_model.pth',
        'dl': 'v2', 'kwargs': {'film_hidden': 64},
    },
    # mjm_attn_dec (Transformer encoder + cascaded decoder)
    {
        'name': 'mjm_attn_dec',
        'module': 'src.models.mjm_attn', 'cls': 'mjm_attn_dec',
        'ckpt': 'experiments/mjm_attn/SEA_AD_MTP_ST/checkpoints/dec_e3_d1_enc256_dec256_z20_h8_bs4096_s3028/best_model.pth',
        'dl': 'v1', 'kwargs': {'n_heads': 8, 'dec_hidden_dim': 256},
    },
]


# ── Feature extraction ──────────────────────────────────────────────────────

def normalize_log1p(x):
    return torch.log1p(x)


def extract_encoder_features(model, model_name, dataloader, device, dl_version):
    """Extract h1, h2, H from encoder stages. Returns numpy arrays."""
    model.eval()
    all_h1, all_h2, all_H = [], [], []
    all_y = {'class': [], 'subclass': [], 'supertype': []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Extracting {model_name}"):
            X = normalize_log1p(batch['X'].to(device))

            # Extract encoder features based on model type
            cls_name = model.__class__.__name__

            if cls_name == 'mjm':
                # mjm original: single encoder → z
                z = model.encoder(X)  # [B, latent_dim]
                all_h1.append(z.cpu().numpy())
                all_h2.append(z.cpu().numpy())
                all_H.append(z.cpu().numpy())

            elif cls_name == 'mjm_2b':
                # mjm_2b: input fusion then E1/E2/E3
                vol = batch['cell_volume_norm'].to(device).unsqueeze(-1)  # [B,1]
                sp = batch['spatial_tiled'].to(device)
                x_aug = torch.cat([X, vol, sp], dim=-1)
                x_proj = model.input_proj(x_aug)
                h1 = model.E1(x_proj)
                h2 = model.E2(h1)
                H = model.E3(h2)
                all_h1.append(h1.cpu().numpy())
                all_h2.append(h2.cpu().numpy())
                all_H.append(H.cpu().numpy())

            elif cls_name == 'mjm_2c':
                # mjm_2c: FiLM on h1
                h1 = model.E1(X)
                cps = batch['cps'].to(device).unsqueeze(-1)  # [B,1]
                sp = batch['spatial_tiled'].to(device)
                context = torch.cat([cps, sp], dim=-1)
                film_params = model.film_net(context)
                gamma, beta = film_params.chunk(2, dim=-1)
                h1_mod = (1.0 + gamma) * h1 + beta
                h2 = model.E2(h1_mod)
                H = model.E3(h2)
                all_h1.append(h1_mod.cpu().numpy())
                all_h2.append(h2.cpu().numpy())
                all_H.append(H.cpu().numpy())

            elif cls_name == 'mjm_attn_dec':
                # Transformer encoder
                spatial = batch['spatial'].to(device)
                batch_id = batch['batch_id'].to(device)
                h = model.input_proj(X)
                cos, sin = model.rope(spatial)
                donor_mask = (batch_id.unsqueeze(0) == batch_id.unsqueeze(1))
                for block in model.E1:
                    h = block(h, cos, sin, donor_mask)
                h1 = h
                for block in model.E2:
                    h = block(h, cos, sin, donor_mask)
                h2 = h
                for block in model.E3:
                    h = block(h, cos, sin, donor_mask)
                H = model.latent_proj(h)
                all_h1.append(h1.cpu().numpy())
                all_h2.append(h2.cpu().numpy())
                all_H.append(H.cpu().numpy())

            else:
                # mjm_1, mjm_2a: standard E1/E2/E3
                h1 = model.E1(X)
                h2 = model.E2(h1)
                H = model.E3(h2)
                all_h1.append(h1.cpu().numpy())
                all_h2.append(h2.cpu().numpy())
                all_H.append(H.cpu().numpy())

            all_y['class'].append(batch['y_class'].numpy())
            all_y['subclass'].append(batch['y_subclass'].numpy())
            all_y['supertype'].append(batch['y_supertype'].numpy())

    features = {
        'h1': np.concatenate(all_h1),
        'h2': np.concatenate(all_h2),
        'H': np.concatenate(all_H),
    }
    labels = {k: np.concatenate(v) for k, v in all_y.items()}
    return features, labels


# ── LR evaluation ───────────────────────────────────────────────────────────

def train_and_eval_lr(train_feats, train_labels, test_feats, test_labels,
                      level_name, n_classes, logger):
    """Train sklearn LogisticRegression and evaluate."""
    logger.info(f"\n--- {level_name} ({n_classes} classes) ---")
    logger.info(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")

    # Use saga solver for large multi-class problems (much faster than lbfgs)
    solver = 'saga' if n_classes > 30 else 'lbfgs'
    clf = LogisticRegression(
        max_iter=1000, solver=solver, multi_class='multinomial',
        C=1.0, n_jobs=10, verbose=0,
    )
    clf.fit(train_feats, train_labels)

    y_pred = clf.predict(test_feats)
    y_prob = clf.predict_proba(test_feats)

    # Macro/Micro F1
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average='micro', zero_division=0)

    logger.info(f"[{level_name.upper()}] Macro  - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
    logger.info(f"[{level_name.upper()}] Micro  - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")

    # Per-class
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        test_labels, y_pred, average=None, zero_division=0)
    parts = []
    for i in range(len(per_p)):
        if support[i] > 0:
            parts.append(f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}")
    logger.info(f"[{level_name.upper()}] Per-class P/R: {' | '.join(parts)}")

    # AUC-ROC
    y_onehot = np.zeros_like(y_prob)
    y_onehot[np.arange(len(test_labels)), test_labels] = 1
    micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())
    valid_aucs = []
    for c in range(n_classes):
        y_bin = (test_labels == c).astype(int)
        if len(np.unique(y_bin)) == 2:
            valid_aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    logger.info(f"[{level_name.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid: {len(valid_aucs)}/{n_classes})")
    logger.info("-" * 50)

    return {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'macro_auc': macro_auc}


# ── Main ────────────────────────────────────────────────────────────────────

def load_model(entry, device):
    """Dynamically import and instantiate model, load checkpoint."""
    import importlib
    mod = importlib.import_module(entry['module'])
    cls = getattr(mod, entry['cls'])

    base_kwargs = dict(
        input_dim=140, latent_dim=20, e_layers=3,
        enc_hidden_dim=256, expansion_factor=2.67, dropout=0.3,
        output_num=[3, 24, 137],
    )
    # Add d_layers for models that need it
    if entry['cls'] in ('mjm', 'mjm_1', 'mjm_2a', 'mjm_2b', 'mjm_2c', 'mjm_attn_dec'):
        base_kwargs['d_layers'] = 1
    if entry['cls'] not in ('mjm',):
        base_kwargs.setdefault('dec_hidden_dim', 128)
    # Override with entry-specific kwargs
    base_kwargs.update(entry.get('kwargs', {}))

    # For mjm_attn_dec, add n_heads
    if entry['cls'] == 'mjm_attn_dec':
        base_kwargs['n_heads'] = 8

    model = cls(**base_kwargs).to(device)
    ckpt_path = entry['ckpt']
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--npz_v1', type=str, default='./data/global_data.npz')
    parser.add_argument('--npz_v2', type=str, default='./data/global_data_v2.npz')
    parser.add_argument('--bs', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--model_idx', type=int, default=-1,
                        help='Run only this model index (0-based). -1 = all sequentially.')
    args = parser.parse_args()

    log_dir = './experiments/eval_lr/SEA_AD_MTP_ST/'
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device(args.device)

    # Select models to run
    if args.model_idx >= 0:
        entries = [MODELS[args.model_idx]]
        log_name = f'record_lr_{entries[0]["name"]}_s{args.seed}.log'
    else:
        entries = MODELS
        log_name = f'record_lr_eval_s{args.seed}.log'

    logger = get_logger(log_dir, __name__, log_name)

    # Build dataloaders (only the version needed)
    need_v1 = any(e['dl'] == 'v1' for e in entries)
    need_v2 = any(e['dl'] == 'v2' for e in entries)

    train_v1, test_v1, train_v2, test_v2 = None, None, None, None
    if need_v1:
        logger.info("Building v1 dataloaders...")
        train_v1, _, test_v1 = build_dataloaders(
            args.npz_v1, batch_size=args.bs, seed=args.seed, num_workers=8)
    if need_v2:
        logger.info("Building v2 dataloaders...")
        train_v2, _, test_v2 = build_dataloaders_v2(
            args.npz_v2, batch_size=args.bs, seed=args.seed, num_workers=8)

    output_num = [3, 24, 137]
    level_map = {
        'h1': ('class', output_num[0]),
        'h2': ('subclass', output_num[1]),
        'H': ('supertype', output_num[2]),
    }

    for entry in entries:
        name = entry['name']
        ckpt = entry['ckpt']
        if not os.path.exists(ckpt):
            logger.info(f"\nSKIPPING {name}: checkpoint not found at {ckpt}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {name}")
        logger.info(f"Checkpoint: {ckpt}")

        model, n_params = load_model(entry, device)
        logger.info(f"Params: {n_params:,}")

        dl_train = train_v2 if entry['dl'] == 'v2' else train_v1
        dl_test = test_v2 if entry['dl'] == 'v2' else test_v1

        # Extract features
        logger.info("Extracting train features...")
        train_feats, train_labels = extract_encoder_features(
            model, name, dl_train, device, entry['dl'])
        logger.info("Extracting test features...")
        test_feats, test_labels = extract_encoder_features(
            model, name, dl_test, device, entry['dl'])

        # Free GPU memory before CPU-heavy LR training
        del model
        torch.cuda.empty_cache()

        # Train and evaluate LR for each level
        logger.info(f"\n========== {name} LR Classification ==========")
        for feat_key, (level_name, n_cls) in level_map.items():
            train_and_eval_lr(
                train_feats[feat_key], train_labels[level_name],
                test_feats[feat_key], test_labels[level_name],
                level_name, n_cls, logger,
            )

        del train_feats, test_feats

    logger.info("\n\nDone!")


if __name__ == '__main__':
    main()
