#!/usr/bin/env python
"""
CellTypist benchmark for hierarchical cell annotation on SEA-AD MERFISH.

Two modes:
  1. Pre-trained (zero-shot): Use CellTypist's Adult_Human_MTG model
  2. Retrained: Train sklearn LogisticRegression on our training data

Usage:
    python experiments/CellTypist/main.py
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import time
import random
import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import joblib

from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


DEFAULT_H5AD_CANDIDATES = [
    os.environ.get("SEAAD_MTG_H5AD", "").strip(),
    './data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad',
    './data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad',
]


def resolve_h5ad_path():
    for candidate in DEFAULT_H5AD_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "SEA-AD MERFISH h5ad not found. Checked: "
        + ", ".join([p for p in DEFAULT_H5AD_CANDIDATES if p])
    )


def load_and_split(npz_path, seed=3028):
    """Same donor-stratified split as the PyTorch pipeline."""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y_c = data['y_class']
    y_sc = data['y_subclass']
    y_st = data['y_supertype']
    batch = data['batch_donor']
    meta = data['meta'].item()

    unique_donors = np.unique(batch)
    np.random.seed(seed)
    np.random.shuffle(unique_donors)

    test_donors  = unique_donors[:6]
    val_donors   = unique_donors[6:9]
    train_donors = unique_donors[9:]

    splits = {}
    for name, donors in [('train', train_donors), ('val', val_donors), ('test', test_donors)]:
        mask = np.isin(batch, donors)
        splits[name] = {
            'X': X[mask], 'y_class': y_c[mask], 'y_subclass': y_sc[mask],
            'y_supertype': y_st[mask],
        }
    return splits, meta

def get_gene_names():
    """Get the 140 non-Blank gene names from the original h5ad."""
    h5ad_path = resolve_h5ad_path()
    adata = ad.read_h5ad(h5ad_path, backed='r')
    genes = [g for g in adata.var_names if not g.startswith('Blank')]
    adata.file.close()
    return genes


def build_label_maps(h5ad_path):
    """Build integer->string label maps from the h5ad for Class/Subclass/Supertype.

    The npz integer labels correspond to sorted unique string labels from the h5ad.
    """
    adata = ad.read_h5ad(h5ad_path, backed='r')
    maps = {}
    for col, key in [('Class', 'class'), ('Subclass', 'subclass'), ('Supertype', 'supertype')]:
        labels_sorted = sorted(adata.obs[col].cat.categories.tolist())
        maps[key] = {i: name for i, name in enumerate(labels_sorted)}
    adata.file.close()
    return maps


def log_metrics(logger, y_true, y_pred, num_classes, task_name):
    """Compute and log classification metrics."""
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0)
    per_p, per_r, _, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    logger.info(f"[{task_name.upper()}] Macro  - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
    logger.info(f"[{task_name.upper()}] Micro  - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")

    per_class_logs = [
        f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}"
        for i in range(len(per_p)) if support[i] > 0 or per_p[i] > 0
    ]
    logger.info(f"[{task_name.upper()}] Per-class: {' | '.join(per_class_logs)}")
    logger.info("-" * 50)
    return {'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1, 'micro_f1': micro_f1}


# ---------------------------------------------------------------------------
# Mode 1: Pre-trained (zero-shot) using CellTypist Adult_Human_MTG model
# ---------------------------------------------------------------------------

def run_pretrained(splits, meta, gene_names, label_maps, logger):
    """Run CellTypist pre-trained MTG model on test data (zero-shot)."""
    import celltypist

    logger.info("=" * 60)
    logger.info("MODE 1: Pre-trained CellTypist (Adult_Human_MTG, zero-shot)")
    logger.info("=" * 60)

    X_test = splits['test']['X']
    logger.info(f"Test cells: {X_test.shape[0]}")

    # Build AnnData with log1p normalization (CellTypist expects log-normalized)
    X_log = np.log1p(X_test)
    adata_test = ad.AnnData(
        X=sp.csr_matrix(X_log),
        var={"gene_names": gene_names},
    )
    adata_test.var_names = gene_names

    # Load pre-trained model
    logger.info("Loading Adult_Human_MTG.pkl model...")
    model = celltypist.models.Model.load(model='Adult_Human_MTG.pkl')
    pretrained_types = list(model.cell_types)
    logger.info(f"Pre-trained model has {len(pretrained_types)} cell types")

    # Run prediction
    logger.info("Running CellTypist prediction...")
    t0 = time.time()
    predictions = celltypist.annotate(adata_test, model=model, majority_voting=False)
    pred_time = time.time() - t0
    logger.info(f"Prediction time: {pred_time:.1f}s")

    pred_labels = predictions.predicted_labels['predicted_labels'].values
    logger.info(f"Unique predicted labels: {len(set(pred_labels))}")

    # The pre-trained model predicts Supertype-level labels.
    # Map predictions to our label hierarchy.
    st_name_to_int = {v: k for k, v in label_maps['supertype'].items()}
    sc_name_to_int = {v: k for k, v in label_maps['subclass'].items()}
    c_name_to_int  = {v: k for k, v in label_maps['class'].items()}

    # Build supertype -> subclass -> class mapping from h5ad
    h5ad_path = resolve_h5ad_path()
    adata_ref = ad.read_h5ad(h5ad_path, backed='r')
    st_to_sc = dict(zip(adata_ref.obs['Supertype'], adata_ref.obs['Subclass']))
    st_to_c  = dict(zip(adata_ref.obs['Supertype'], adata_ref.obs['Class']))
    adata_ref.file.close()

    # Map predicted string labels to integer labels at each level
    y_test_st = splits['test']['y_supertype']
    y_test_sc = splits['test']['y_subclass']
    y_test_c  = splits['test']['y_class']

    # For supertype: direct match where possible, -1 for unknown
    y_pred_st = np.full(len(pred_labels), -1, dtype=int)
    y_pred_sc = np.full(len(pred_labels), -1, dtype=int)
    y_pred_c  = np.full(len(pred_labels), -1, dtype=int)

    matched_st, matched_sc, matched_c = 0, 0, 0
    for i, plabel in enumerate(pred_labels):
        # Supertype level
        if plabel in st_name_to_int:
            y_pred_st[i] = st_name_to_int[plabel]
            matched_st += 1
        # Subclass level: derive from supertype mapping
        if plabel in st_to_sc:
            sc_name = st_to_sc[plabel]
            if sc_name in sc_name_to_int:
                y_pred_sc[i] = sc_name_to_int[sc_name]
                matched_sc += 1
        # Class level: derive from supertype mapping
        if plabel in st_to_c:
            c_name = st_to_c[plabel]
            if c_name in c_name_to_int:
                y_pred_c[i] = c_name_to_int[c_name]
                matched_c += 1

    logger.info(f"Label matching: ST={matched_st}/{len(pred_labels)}, "
                f"SC={matched_sc}/{len(pred_labels)}, C={matched_c}/{len(pred_labels)}")

    # Only evaluate on cells where we could map the prediction
    for task, y_true, y_pred, n_cls in [
        ('class', y_test_c, y_pred_c, meta['num_class']),
        ('subclass', y_test_sc, y_pred_sc, meta['num_subclass']),
        ('supertype', y_test_st, y_pred_st, meta['num_supertype']),
    ]:
        valid = y_pred >= 0
        logger.info(f"[{task.upper()}] Evaluating on {valid.sum()}/{len(y_pred)} mapped cells")
        if valid.sum() > 0:
            log_metrics(logger, y_true[valid], y_pred[valid], n_cls, f"pretrained_{task}")
        else:
            logger.info(f"[{task.upper()}] No valid predictions — skipping")


# ---------------------------------------------------------------------------
# Mode 2: Retrained — sklearn LogisticRegression on our training data
# ---------------------------------------------------------------------------

def run_retrained(splits, meta, logger, seed=3028):
    """Train LR classifiers on our training data, evaluate on test."""
    logger.info("=" * 60)
    logger.info("MODE 2: Retrained CellTypist (sklearn LR on our data)")
    logger.info("=" * 60)

    X_train = np.log1p(splits['train']['X'])
    X_test  = np.log1p(splits['test']['X'])
    logger.info(f"Train: {X_train.shape[0]} cells, Test: {X_test.shape[0]} cells, "
                f"Features: {X_train.shape[1]}")

    save_dir = os.path.join('experiments', 'CellTypist', 'retrained', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    tasks = [
        ('class',     meta['num_class'],     'lbfgs'),
        ('subclass',  meta['num_subclass'],  'lbfgs'),
        ('supertype', meta['num_supertype'], 'saga'),
    ]

    logger.info("========== Retrained Test Classification Metrics ==========")
    for task_name, n_classes, solver in tasks:
        y_key = f'y_{task_name}'
        y_train = splits['train'][y_key]
        y_test  = splits['test'][y_key]

        logger.info(f"\n--- Training {task_name} ({n_classes} classes, solver={solver}) ---")
        t0 = time.time()
        clf = LogisticRegression(
            max_iter=1000, solver=solver, multi_class='multinomial',
            C=1.0, random_state=seed, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        logger.info(f"Training time: {train_time:.1f}s")

        joblib.dump(clf, os.path.join(save_dir, f'{task_name}_model.pkl'))

        y_pred = clf.predict(X_test)
        log_metrics(logger, y_test, y_pred, n_classes, f"retrained_{task_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    SEED = 3028
    NPZ_PATH = './data/global_data.npz'
    H5AD_PATH = resolve_h5ad_path()

    set_seed(SEED)

    log_dir = './experiments/CellTypist/logs/'
    logger = get_logger(log_dir, 'celltypist', f'record_s{SEED}.log')
    logger.info("CellTypist Benchmark — SEA-AD MERFISH MTG")
    logger.info(f"Seed: {SEED}")

    # Load data
    logger.info("Loading data...")
    splits, meta = load_and_split(NPZ_PATH, seed=SEED)
    logger.info(f"Train: {splits['train']['X'].shape[0]}, "
                f"Val: {splits['val']['X'].shape[0]}, "
                f"Test: {splits['test']['X'].shape[0]} cells")

    # Gene names
    logger.info("Loading gene names...")
    gene_names = get_gene_names()
    logger.info(f"Genes: {len(gene_names)}")

    # Label maps for pre-trained evaluation
    logger.info("Building label maps...")
    label_maps = build_label_maps(H5AD_PATH)

    # --- Mode 1: Pre-trained ---
    try:
        run_pretrained(splits, meta, gene_names, label_maps, logger)
    except Exception as e:
        logger.error(f"Pre-trained mode failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # --- Mode 2: Retrained ---
    try:
        run_retrained(splits, meta, logger, seed=SEED)
    except Exception as e:
        logger.error(f"Retrained mode failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("=" * 60)
    logger.info("CellTypist benchmark complete.")
