#!/usr/bin/env python
"""
Novae (Nature Methods 2025) spatial foundation model benchmark.

Approach:
  1. Load SEA-AD MERFISH data, create per-donor AnnData objects
  2. Build spatial neighbor graphs (required by Novae)
  3. Use pretrained Novae to extract cell embeddings (64-dim)
  4. Train sklearn LogisticRegression heads on embeddings (freeze mode)
  5. Report Macro P/R/F1 and Micro F1 for Class/Subclass/Supertype

Uses the same train/test donor split (seed=3028) as all other models.
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import time
import argparse
import numpy as np
import anndata

import novae
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.utils.logger import get_logger


# ── Args ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Novae benchmark")
    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    parser.add_argument('--h5ad_path', type=str,
                        default='./data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad')
    parser.add_argument('--model_name', type=str, default='MICS-Lab/novae-human-0',
                        help='HuggingFace model name for Novae pretrained weights')
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--accelerator', type=str, default='cuda',
                        help='Device for Novae inference: cuda, cpu, auto')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune Novae on training data before extracting embeddings')
    parser.add_argument('--ft_epochs', type=int, default=10,
                        help='Max epochs for fine-tuning')
    return parser.parse_args()


# ── Donor split (same as dataloader.py) ─────────────────────────────────────

def get_donor_split(batch_donor, seed=3028):
    """Reproduce the exact same split as dataloader.py."""
    unique_donors = np.unique(batch_donor)
    np.random.seed(seed)
    np.random.shuffle(unique_donors)
    test_donors  = set(unique_donors[:6].tolist())
    val_donors   = set(unique_donors[6:9].tolist())
    train_donors = set(unique_donors[9:].tolist())
    return train_donors, val_donors, test_donors


# ── Build per-donor AnnData + spatial graph ─────────────────────────────────

def build_donor_adata(X_donor, spatial_donor, gene_names):
    """Create an AnnData for one donor with spatial coordinates."""
    adata = anndata.AnnData(
        X=X_donor.astype(np.float32),
        obs={},
    )
    adata.var_names = gene_names
    adata.obsm['spatial'] = spatial_donor.astype(np.float64)
    return adata


# ── Metrics ─────────────────────────────────────────────────────────────────

def log_metrics(y_true, y_pred, y_prob, level_name, n_classes, logger):
    """Log classification metrics for one hierarchy level."""
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0)

    logger.info(f"[{level_name.upper()}] Macro  - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
    logger.info(f"[{level_name.upper()}] Micro  - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")

    # Per-class
    per_p, per_r, _, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    parts = [f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}"
             for i in range(len(per_p)) if support[i] > 0]
    logger.info(f"[{level_name.upper()}] Per-class: {' | '.join(parts)}")

    # AUC-ROC
    y_onehot = np.zeros_like(y_prob)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())
    valid_aucs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if len(np.unique(y_bin)) == 2:
            valid_aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    logger.info(f"[{level_name.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid: {len(valid_aucs)}/{n_classes})")
    logger.info("-" * 50)


def train_and_eval_lr(train_X, train_y, test_X, test_y, level_name, n_classes, logger):
    """Train SGDClassifier (log loss = logistic regression) and evaluate."""
    logger.info(f"\n--- {level_name} ({n_classes} classes) ---")
    logger.info(f"  Train: {train_X.shape}, Test: {test_X.shape}")

    # Scale features for SGD convergence
    scaler = StandardScaler()
    train_X_sc = scaler.fit_transform(train_X)
    test_X_sc = scaler.transform(test_X)

    # SGDClassifier with log_loss = logistic regression via SGD (much faster)
    clf = SGDClassifier(
        loss='log_loss', alpha=1e-4, max_iter=100,
        n_jobs=-1, verbose=0, random_state=42,
    )
    clf.fit(train_X_sc, train_y)

    y_pred = clf.predict(test_X_sc)

    # For probability estimates, use decision_function + softmax
    decision = clf.decision_function(test_X_sc)
    if decision.ndim == 1:
        # Binary case
        y_prob = np.column_stack([1 - 1/(1+np.exp(-decision)), 1/(1+np.exp(-decision))])
    else:
        # Multi-class: softmax
        exp_d = np.exp(decision - decision.max(axis=1, keepdims=True))
        y_prob = exp_d / exp_d.sum(axis=1, keepdims=True)

    log_metrics(test_y, y_pred, y_prob, level_name, n_classes, logger)
    return clf


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    log_dir = './experiments/Novae/SEA_AD_MTP_ST/'
    os.makedirs(log_dir, exist_ok=True)
    ft_tag = "finetune" if args.fine_tune else "freeze"
    logger = get_logger(log_dir, __name__, f'record_{ft_tag}_s{args.seed}.log')
    logger.info(f"Args: {args}")

    # ── Check for cached embeddings ───────────────────────────────────────────
    save_dir = os.path.join(log_dir, 'checkpoints', ft_tag)
    os.makedirs(save_dir, exist_ok=True)
    emb_path = os.path.join(save_dir, 'embeddings.npz')

    if os.path.exists(emb_path):
        logger.info(f"Loading cached embeddings from {emb_path}")
        cached = np.load(emb_path)
        train_emb = cached['train_emb']
        test_emb = cached['test_emb']
        train_labels = {k: cached[f'train_{k}'] for k in ['class', 'subclass', 'supertype']}
        test_labels = {k: cached[f'test_{k}'] for k in ['class', 'subclass', 'supertype']}
        logger.info(f"Train embeddings: {train_emb.shape}, Test: {test_emb.shape}")
    else:
        # ── 1. Load data ────────────────────────────────────────────────────────
        logger.info("Loading data...")
        data = np.load(args.npz_path, allow_pickle=True)
        X_all       = data['X']
        spatial_all = data['spatial']
        batch_all   = data['batch_donor']
        y_c_all     = data['y_class']
        y_sc_all    = data['y_subclass']
        y_st_all    = data['y_supertype']
        logger.info(f"Total cells: {X_all.shape[0]}, genes: {X_all.shape[1]}")

        # Gene names from h5ad (filter Blank)
        h5ad = anndata.read_h5ad(args.h5ad_path, backed='r')
        gene_names = [g for g in h5ad.var_names if not g.startswith('Blank')]
        assert len(gene_names) == X_all.shape[1], f"Gene count mismatch: {len(gene_names)} vs {X_all.shape[1]}"
        logger.info(f"Gene names: {len(gene_names)} (first 5: {gene_names[:5]})")

        # ── 2. Donor split ──────────────────────────────────────────────────────
        train_donors, val_donors, test_donors = get_donor_split(batch_all, args.seed)
        logger.info(f"Train donors: {len(train_donors)}, Val: {len(val_donors)}, Test: {len(test_donors)}")

        # ── 3. Load pretrained Novae ────────────────────────────────────────────
        logger.info(f"Loading Novae model: {args.model_name}")
        model = novae.Novae.from_pretrained(args.model_name)
        logger.info(f"Novae output_size={model.hparams.get('output_size')}, "
                    f"vocab={len(model.hparams.get('var_names', []))}")

        # ── 4. Build per-donor AnnData, compute spatial graph + embeddings ──
        # Process donors one at a time to manage memory
        unique_donors = np.unique(batch_all)
        donor_embeddings = {}  # donor_id -> novae_latent array
        donor_indices = {}     # donor_id -> boolean mask into X_all

        t0 = time.time()
        for i, donor_id in enumerate(unique_donors):
            mask = batch_all == donor_id
            donor_indices[donor_id] = mask
            n_cells = mask.sum()

            split = "train" if donor_id in train_donors else ("val" if donor_id in val_donors else "test")
            logger.info(f"[{i+1}/{len(unique_donors)}] Donor {donor_id} ({split}): {n_cells} cells")

            # Build AnnData for this donor
            adata_d = build_donor_adata(X_all[mask], spatial_all[mask], gene_names)

            # Compute spatial neighbor graph (required by Novae)
            novae.spatial_neighbors(adata_d)

            # Fine-tune on training donors only (if requested)
            if args.fine_tune and donor_id in train_donors:
                model.fine_tune(adata_d, max_epochs=args.ft_epochs,
                                accelerator=args.accelerator, num_workers=4)

            # Compute representations (embeddings stored in obsm['novae_latent'])
            # zero_shot=False avoids expensive kmeans prototype assignment
            model.compute_representations(adata_d, zero_shot=False,
                                          accelerator=args.accelerator, num_workers=4)

            # Extract embeddings
            emb = adata_d.obsm.get('novae_latent', None)
            if emb is None:
                logger.error(f"  No embeddings for donor {donor_id}!")
                donor_embeddings[donor_id] = np.zeros((n_cells, 64), dtype=np.float32)
            else:
                donor_embeddings[donor_id] = emb.astype(np.float32)
                logger.info(f"  Embedding shape: {emb.shape}")

            del adata_d  # free memory

        elapsed = time.time() - t0
        logger.info(f"Embedding extraction done in {elapsed:.1f}s")

        # ── 5. Assemble train/test feature matrices ─────────────────────────
        logger.info("Assembling train/test splits...")

        train_emb, train_yc, train_ysc, train_yst = [], [], [], []
        test_emb, test_yc, test_ysc, test_yst = [], [], [], []

        for donor_id in unique_donors:
            mask = donor_indices[donor_id]
            emb = donor_embeddings[donor_id]

            if donor_id in train_donors or donor_id in val_donors:
                train_emb.append(emb)
                train_yc.append(y_c_all[mask])
                train_ysc.append(y_sc_all[mask])
                train_yst.append(y_st_all[mask])
            elif donor_id in test_donors:
                test_emb.append(emb)
                test_yc.append(y_c_all[mask])
                test_ysc.append(y_sc_all[mask])
                test_yst.append(y_st_all[mask])

        train_emb = np.concatenate(train_emb)
        test_emb  = np.concatenate(test_emb)
        train_labels = {
            'class': np.concatenate(train_yc),
            'subclass': np.concatenate(train_ysc),
            'supertype': np.concatenate(train_yst),
        }
        test_labels = {
            'class': np.concatenate(test_yc),
            'subclass': np.concatenate(test_ysc),
            'supertype': np.concatenate(test_yst),
        }

        logger.info(f"Train embeddings: {train_emb.shape}, Test: {test_emb.shape}")

        # Save embeddings
        np.savez_compressed(emb_path,
                            train_emb=train_emb, test_emb=test_emb,
                            **{f'train_{k}': v for k, v in train_labels.items()},
                            **{f'test_{k}': v for k, v in test_labels.items()})
        logger.info(f"Embeddings saved to {emb_path}")

    # ── 6. Train LR heads ───────────────────────────────────────────────────
    logger.info("\n========== Novae + LR Classification ==========")
    output_num = {'class': 3, 'subclass': 24, 'supertype': 137}

    for level_name, n_cls in output_num.items():
        train_and_eval_lr(
            train_emb, train_labels[level_name],
            test_emb, test_labels[level_name],
            level_name, n_cls, logger,
        )

    logger.info("Done!")


if __name__ == '__main__':
    main()