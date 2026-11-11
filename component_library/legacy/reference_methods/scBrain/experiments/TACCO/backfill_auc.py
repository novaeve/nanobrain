#!/usr/bin/env python
"""Backfill AUC-ROC metrics for TACCO OT without changing the training setup."""

import gc
import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.logger import get_logger
from main import (
    OT_MAX_REF,
    SEED,
    NPZ_PATH,
    build_label_maps,
    get_gene_names,
    load_and_split,
    make_anndata,
    run_tacco_method,
    set_seed,
    subsample_stratified,
)


LOG_DIR = "./experiments/TACCO/logs/"
CHUNK_SIZE = 50_000


def compute_auc(y_true, y_prob, n_classes):
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())

    valid_aucs = []
    for class_idx in range(n_classes):
        y_bin = (y_true == class_idx).astype(int)
        if len(np.unique(y_bin)) == 2:
            valid_aucs.append(roc_auc_score(y_bin, y_prob[:, class_idx]))
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return macro_auc, float(micro_auc), len(valid_aucs)


def aligned_prob(fractions_df, int_to_str_map, n_classes):
    str_to_int = {label: idx for idx, label in int_to_str_map.items()}
    prob = np.zeros((len(fractions_df), n_classes), dtype=np.float32)
    for label_name in fractions_df.columns:
        class_idx = str_to_int.get(label_name)
        if class_idx is not None:
            prob[:, class_idx] = fractions_df[label_name].to_numpy(dtype=np.float32, copy=False)
    return prob


if __name__ == "__main__":
    set_seed(SEED)
    logger = get_logger(LOG_DIR, "tacco_auc_backfill", f"record_s{SEED}.log")
    logger.info("=" * 60)
    logger.info("AUC-ROC BACKFILL: TACCO OT (test-only)")
    logger.info("=" * 60)

    splits, meta = load_and_split(NPZ_PATH, seed=SEED)
    gene_names = get_gene_names()
    label_maps = build_label_maps()

    train_labels = {}
    for level in ["class", "subclass", "supertype"]:
        int_labels = splits["train"][f"y_{level}"]
        train_labels[level] = np.array([label_maps[level][idx] for idx in int_labels])

    x_train = splits["train"]["X"]
    x_test = splits["test"]["X"]
    x_train_sub, train_labels_sub = subsample_stratified(x_train, train_labels, OT_MAX_REF, seed=SEED)
    adata_ref_sub = make_anndata(x_train_sub, gene_names, train_labels_sub)

    n_test = x_test.shape[0]
    n_chunks = (n_test + CHUNK_SIZE - 1) // CHUNK_SIZE

    logger.info("========== OT Test AUC-ROC Metrics ==========")
    for level in ["class", "subclass", "supertype"]:
        logger.info(f"--- OT {level} AUC backfill ---")
        n_classes = meta[f"num_{level}"]
        y_true = splits["test"][f"y_{level}"]

        prob_chunks = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n_test)
            logger.info(f"  Chunk {chunk_idx + 1}/{n_chunks}: cells {start}-{end}")
            adata_chunk = make_anndata(x_test[start:end], gene_names)
            fractions = run_tacco_method(
                adata_ref_sub,
                adata_chunk,
                annotation_key=level,
                method="OT",
                logger=logger,
                bisections=4,
                max_annotation=1,
                multi_center=20,
            )
            prob_chunks.append(aligned_prob(fractions, label_maps[level], n_classes))
            del adata_chunk, fractions
            gc.collect()

        y_prob = np.concatenate(prob_chunks, axis=0)
        macro_auc, micro_auc, n_valid = compute_auc(y_true, y_prob, n_classes)
        logger.info(
            f"[OT_{level.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid: {n_valid}/{n_classes})"
        )
        del prob_chunks, y_prob
        gc.collect()

    logger.info("=" * 60)
    logger.info("TACCO AUC-ROC backfill complete.")
