#!/usr/bin/env python
"""
TACCO benchmark for hierarchical cell annotation on SEA-AD MERFISH.

TACCO (Nature Biotechnology 2023) transfers annotations from a reference
(training cells) to query (test cells) via optimal transport.

Methods benchmarked:
  - OT (default, flagship): semi-balanced optimal transport
  - projection: fast profile-based projection

Usage:
    python experiments/TACCO/main.py
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import time
import random
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.metrics import precision_recall_fscore_support
import tacco as tc

from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 3028
NPZ_PATH = './data/global_data.npz'
DEFAULT_H5AD_CANDIDATES = [
    os.environ.get("SEAAD_MTG_H5AD", "").strip(),
    './data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad',
    './data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad',
]

# Max reference cells for OT (memory/time constraint)
OT_MAX_REF = 200_000


def resolve_h5ad_path():
    for candidate in DEFAULT_H5AD_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "SEA-AD MERFISH h5ad not found. Checked: "
        + ", ".join([p for p in DEFAULT_H5AD_CANDIDATES if p])
    )


H5AD_PATH = resolve_h5ad_path()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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
    adata = ad.read_h5ad(H5AD_PATH, backed='r')
    genes = [g for g in adata.var_names if not g.startswith('Blank')]
    adata.file.close()
    return genes


def build_label_maps():
    """Build integer->string label maps from the h5ad."""
    adata = ad.read_h5ad(H5AD_PATH, backed='r')
    maps = {}
    for col, key in [('Class', 'class'), ('Subclass', 'subclass'), ('Supertype', 'supertype')]:
        labels_sorted = sorted(adata.obs[col].cat.categories.tolist())
        maps[key] = {i: name for i, name in enumerate(labels_sorted)}
    adata.file.close()
    return maps


def make_anndata(X, gene_names, labels_dict=None):
    """Create AnnData from expression matrix. X is raw counts (float32)."""
    adata = ad.AnnData(X=sp.csr_matrix(X), var=pd.DataFrame(index=gene_names))
    if labels_dict is not None:
        for key, vals in labels_dict.items():
            adata.obs[key] = pd.Categorical(vals)
    return adata


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


def subsample_stratified(X, labels_dict, max_n, seed=3028):
    """Stratified subsample keeping class proportions."""
    n = X.shape[0]
    if n <= max_n:
        return X, labels_dict
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, max_n, replace=False)
    idx.sort()
    X_sub = X[idx]
    labels_sub = {k: v[idx] for k, v in labels_dict.items()}
    return X_sub, labels_sub


def fractions_to_labels(fractions_df, int_to_str_map):
    """Convert TACCO fraction DataFrame to integer label predictions.

    fractions_df columns are string label names. We pick argmax per cell,
    then map back to integer labels using the inverse of int_to_str_map.
    """
    str_to_int = {v: k for k, v in int_to_str_map.items()}
    pred_str = fractions_df.idxmax(axis=1).values
    pred_int = np.array([str_to_int.get(s, -1) for s in pred_str])
    return pred_int


def run_tacco_method(adata_ref, adata_query, annotation_key, method,
                     logger, bisections=None, max_annotation=1, multi_center=None,
                     verbose=1):
    """Run TACCO annotate and return fractions DataFrame."""
    logger.info(f"  Running tc.tl.annotate(method='{method}', annotation_key='{annotation_key}', "
                f"bisections={bisections}, max_annotation={max_annotation}, "
                f"multi_center={multi_center}) ...")
    t0 = time.time()
    kwargs = dict(
        annotation_key=annotation_key,
        method=method,
        bisections=bisections,
        max_annotation=max_annotation,
        assume_valid_counts=True,
        verbose=verbose,
    )
    if multi_center is not None:
        kwargs['multi_center'] = multi_center
    fractions = tc.tl.annotate(adata_query, adata_ref, **kwargs)
    elapsed = time.time() - t0
    logger.info(f"  Annotation time: {elapsed:.1f}s")
    return fractions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    set_seed(SEED)

    log_dir = './experiments/TACCO/logs/'
    logger = get_logger(log_dir, 'tacco', f'record_s{SEED}.log')
    logger.info("TACCO Benchmark — SEA-AD MERFISH MTG")
    logger.info(f"Seed: {SEED}")

    # Load data
    logger.info("Loading data...")
    splits, meta = load_and_split(NPZ_PATH, seed=SEED)
    logger.info(f"Train: {splits['train']['X'].shape[0]}, "
                f"Val: {splits['val']['X'].shape[0]}, "
                f"Test: {splits['test']['X'].shape[0]} cells")

    # Gene names and label maps
    logger.info("Loading gene names and label maps...")
    gene_names = get_gene_names()
    label_maps = build_label_maps()
    logger.info(f"Genes: {len(gene_names)}")

    # Build string labels for reference (TACCO needs categorical obs)
    train_labels = {}
    for level in ['class', 'subclass', 'supertype']:
        int_labels = splits['train'][f'y_{level}']
        str_labels = np.array([label_maps[level][i] for i in int_labels])
        train_labels[level] = str_labels

    X_train = splits['train']['X']
    X_test = splits['test']['X']

    # ===================================================================
    # Method 1: TACCO projection (fast, profile-based)
    # ===================================================================
    logger.info("=" * 60)
    logger.info("METHOD 1: TACCO projection (full reference)")
    logger.info("=" * 60)

    adata_ref_full = make_anndata(X_train, gene_names, train_labels)
    adata_query = make_anndata(X_test, gene_names)

    for level in ['class', 'subclass', 'supertype']:
        logger.info(f"\n--- {level} ---")
        n_classes = meta[f'num_{level}']
        y_test = splits['test'][f'y_{level}']

        try:
            fractions = run_tacco_method(
                adata_ref_full, adata_query.copy(),
                annotation_key=level, method='projection',
                logger=logger, bisections=0, max_annotation=1,
            )
            y_pred = fractions_to_labels(fractions, label_maps[level])
            valid = y_pred >= 0
            logger.info(f"  Valid predictions: {valid.sum()}/{len(y_pred)}")
            if valid.sum() > 0:
                log_metrics(logger, y_test[valid], y_pred[valid], n_classes,
                            f"projection_{level}")
        except Exception as e:
            logger.error(f"  projection/{level} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    del adata_ref_full
    import gc; gc.collect()

    # ===================================================================
    # Method 2: TACCO OT (flagship, subsampled reference)
    # ===================================================================
    logger.info("=" * 60)
    logger.info(f"METHOD 2: TACCO OT (reference subsampled to {OT_MAX_REF})")
    logger.info("=" * 60)

    X_train_sub, train_labels_sub = subsample_stratified(
        X_train, train_labels, OT_MAX_REF, seed=SEED)
    logger.info(f"Subsampled reference: {X_train_sub.shape[0]} cells")

    adata_ref_sub = make_anndata(X_train_sub, gene_names, train_labels_sub)

    # Process test in chunks to manage memory
    CHUNK_SIZE = 50_000
    n_test = X_test.shape[0]
    n_chunks = (n_test + CHUNK_SIZE - 1) // CHUNK_SIZE

    for level in ['class', 'subclass', 'supertype']:
        logger.info(f"\n--- {level} ---")
        n_classes = meta[f'num_{level}']
        y_test = splits['test'][f'y_{level}']

        all_preds = []
        try:
            t0_total = time.time()
            for ci in range(n_chunks):
                start = ci * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, n_test)
                logger.info(f"  Chunk {ci+1}/{n_chunks}: cells {start}-{end}")

                adata_chunk = make_anndata(X_test[start:end], gene_names)
                fractions = run_tacco_method(
                    adata_ref_sub, adata_chunk,
                    annotation_key=level, method='OT',
                    logger=logger, bisections=4, max_annotation=1,
                    multi_center=20,
                )
                chunk_preds = fractions_to_labels(fractions, label_maps[level])
                all_preds.append(chunk_preds)
                del adata_chunk, fractions
                gc.collect()

            y_pred = np.concatenate(all_preds)
            elapsed_total = time.time() - t0_total
            logger.info(f"  Total OT time for {level}: {elapsed_total:.1f}s")

            valid = y_pred >= 0
            logger.info(f"  Valid predictions: {valid.sum()}/{len(y_pred)}")
            if valid.sum() > 0:
                log_metrics(logger, y_test[valid], y_pred[valid], n_classes,
                            f"OT_{level}")
        except Exception as e:
            logger.error(f"  OT/{level} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("=" * 60)
    logger.info("TACCO benchmark complete.")
