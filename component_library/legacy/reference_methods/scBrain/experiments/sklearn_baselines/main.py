#!/usr/bin/env python
"""
Sklearn-based baselines for hierarchical cell annotation.
Supports: Logistic Regression (CellTypist-style), SVM, XGBoost.

Usage (run from project root with conda stack env):
    python experiments/sklearn_baselines/main.py --method lr
    python experiments/sklearn_baselines/main.py --method svm
    python experiments/sklearn_baselines/main.py --method xgboost
"""
import sys, os
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import argparse
import random
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib

from src.utils.logger import get_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_and_split(npz_path, seed=3028):
    """Load NPZ and apply the same donor-stratified split as the PyTorch pipeline."""
    data = np.load(npz_path, allow_pickle=True)
    X = np.log1p(data['X'].astype(np.float32))
    y_c = data['y_class']
    y_sc = data['y_subclass']
    y_st = data['y_supertype']
    batch = data['batch_donor']
    spatial = data['spatial']
    cps = data['cps']
    confidence = data['y_supertype_confidence']
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
            'y_supertype': y_st[mask], 'confidence': confidence[mask],
            'spatial': spatial[mask], 'batch': batch[mask], 'cps': cps[mask],
        }
    return splits, meta


def build_classifier(method, n_classes, seed=3028):
    if method == 'lr':
        return LogisticRegression(
            max_iter=1000, solver='lbfgs', multi_class='multinomial',
            C=1.0, random_state=seed, n_jobs=-1,
        )
    elif method == 'svm':
        base = LinearSVC(max_iter=2000, C=1.0, random_state=seed, dual='auto')
        return CalibratedClassifierCV(base, cv=3, n_jobs=-1)
    elif method == 'xgboost':
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=n_classes,
            tree_method='hist', device='cuda',
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def log_metrics(logger, y_true, y_pred, y_prob, num_classes, task_name, prefix="Test"):
    """Compute and log classification + AUC-ROC metrics (same format as TrainerEngine)."""
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

    # AUC-ROC
    if y_prob is not None:
        y_onehot = np.zeros((len(y_true), num_classes))
        y_onehot[np.arange(len(y_true)), y_true] = 1
        micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())
        valid_aucs = []
        for c in range(num_classes):
            binary = (y_true == c).astype(int)
            if len(np.unique(binary)) == 2:
                valid_aucs.append(roc_auc_score(binary, y_prob[:, c]))
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        logger.info(f"[{task_name.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid: {len(valid_aucs)}/{num_classes})")
    logger.info("-" * 50)


def get_proba(clf, X, method, n_classes):
    """Get probability predictions, handling models that don't natively support predict_proba."""
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X)
        # Ensure correct number of columns (some classes may be missing in training)
        if proba.shape[1] < n_classes:
            full_proba = np.zeros((len(X), n_classes))
            full_proba[:, clf.classes_] = proba
            return full_proba
        return proba
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='lr', choices=['lr', 'svm', 'xgboost'])
    parser.add_argument('--npz_path', type=str, default='./data/global_data.npz')
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    set_seed(args.seed)

    method_names = {'lr': 'LogisticRegression', 'svm': 'SVM', 'xgboost': 'XGBoost'}
    log_dir = f'./experiments/sklearn_baselines/{args.method}/SEA_AD_MTP_ST/'
    logger = get_logger(log_dir, __name__, f'record_s{args.seed}.log')
    logger.info(f"Method: {method_names[args.method]}, Seed: {args.seed}")

    # Load data
    splits, meta = load_and_split(args.npz_path, seed=args.seed)
    X_train, X_test = splits['train']['X'], splits['test']['X']
    logger.info(f"Train: {X_train.shape[0]} cells, Test: {X_test.shape[0]} cells")

    tasks = [
        ('class',     meta['num_class']),
        ('subclass',  meta['num_subclass']),
        ('supertype', meta['num_supertype']),
    ]

    save_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    logger.info("========== Test Classification Metrics ==========")
    for task_name, n_classes in tasks:
        y_key = f'y_{task_name}'
        y_train = splits['train'][y_key]
        y_test  = splits['test'][y_key]

        logger.info(f"\n--- Training {task_name} ({n_classes} classes) ---")
        t0 = time.time()
        clf = build_classifier(args.method, n_classes, seed=args.seed)
        clf.fit(X_train, y_train)
        logger.info(f"Training time: {time.time() - t0:.1f}s")

        # Save model
        joblib.dump(clf, os.path.join(save_dir, f'{task_name}_model.pkl'))

        # Predict
        y_pred = clf.predict(X_test)
        y_prob = get_proba(clf, X_test, args.method, n_classes)

        log_metrics(logger, y_test, y_pred, y_prob, n_classes, task_name)

    logger.info("Done.")
