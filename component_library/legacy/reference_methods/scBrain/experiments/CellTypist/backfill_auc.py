#!/usr/bin/env python
"""Backfill AUC-ROC metrics for retrained CellTypist without retraining."""

import os
import sys

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.logger import get_logger
from main import load_and_split, set_seed


SEED = 3028
NPZ_PATH = "./data/global_data.npz"
LOG_DIR = "./experiments/CellTypist/logs/"
CKPT_DIR = "./experiments/CellTypist/retrained/checkpoints"


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


def aligned_proba(clf, x_test, n_classes):
    raw_prob = clf.predict_proba(x_test).astype(np.float32, copy=False)
    prob = np.zeros((x_test.shape[0], n_classes), dtype=np.float32)
    prob[:, clf.classes_.astype(int)] = raw_prob
    return prob


if __name__ == "__main__":
    set_seed(SEED)
    logger = get_logger(LOG_DIR, "celltypist_auc_backfill", f"record_s{SEED}.log")
    logger.info("=" * 60)
    logger.info("AUC-ROC BACKFILL: Retrained CellTypist (test-only)")
    logger.info("=" * 60)

    splits, meta = load_and_split(NPZ_PATH, seed=SEED)
    x_test = np.log1p(splits["test"]["X"])

    tasks = [
        ("class", "RETRAINED_CLASS", meta["num_class"]),
        ("subclass", "RETRAINED_SUBCLASS", meta["num_subclass"]),
        ("supertype", "RETRAINED_SUPERTYPE", meta["num_supertype"]),
    ]

    logger.info("========== Retrained Test AUC-ROC Metrics ==========")
    for task_name, log_name, n_classes in tasks:
        model_path = os.path.join(CKPT_DIR, f"{task_name}_model.pkl")
        clf = joblib.load(model_path)
        y_true = splits["test"][f"y_{task_name}"]
        y_prob = aligned_proba(clf, x_test, n_classes)
        macro_auc, micro_auc, n_valid = compute_auc(y_true, y_prob, n_classes)
        logger.info(
            f"[{log_name}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid: {n_valid}/{n_classes})"
        )

    logger.info("=" * 60)
    logger.info("CellTypist AUC-ROC backfill complete.")
