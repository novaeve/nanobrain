#!/usr/bin/env python3
"""Run reproducible baseline and nanoBrain pilot evaluations.

The script deliberately targets the fixed ``nanobrain_v3_100k`` materialized
subset.  Every fit uses train rows only and every reported value is computed
on validation rows with the corresponding task mask.  It writes one standard
NumPy prediction artifact per method/task, task metric receipts, and a
long-format table.  These pilot results are kept separate from the formal
42.8M-row benchmark until the same runners are promoted to the full corpus.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainwm_benchmark import evaluate  # noqa: E402
from nanobrain.data import DatasetSplits, NanoDataset, load_v3_subset, verify_subset_manifest  # noqa: E402


SPLIT_ID = "BRAINWM-HUMAN-GLOBAL-6-2-2-V3"
TASKS = ("annotation", "integration", "spatial_niche", "spatial_completion", "development")
METRIC_DIRECTION: dict[str, str] = {
    "rmse": "min", "mae": "min", "nonzero_rmse": "min",
    "heldout_rmse": "min", "heldout_mae": "min", "row_mse": "min", "row_mae": "min",
    "population_mse": "min", "mean_error": "min", "variance_error": "min",
    "mmd_rbf": "min", "gearys_c": "min", "ece": "min", "nll": "min",
    "brier_multiclass": "min", "zero_calibration_error": "min",
}


def _array(value: Any, dtype: Any | None = None) -> np.ndarray:
    """Convert a tensor/memmap to a NumPy view without needless copies."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no native bfloat16 dtype.  CUDA evaluation intentionally
        # uses BF16, so promote only at the framework boundary before export.
        if str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    out = np.asarray(value)
    return out.astype(dtype, copy=False) if dtype is not None else out


def _labels(dataset: NanoDataset, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = _array(dataset.labels[name])
    mask = _array(dataset.masks[name]).astype(bool, copy=False)
    if values.ndim == 1:
        valid = mask & (values >= 0)
    else:
        valid = mask.reshape(len(values), -1).all(axis=1) & np.isfinite(values).all(axis=1)
    return values, valid


def _features(dataset: NanoDataset) -> np.ndarray:
    return _array(dataset.features, np.float32)


def _coords(dataset: NanoDataset) -> np.ndarray:
    if dataset.coordinates is None:
        return np.full((dataset.size, 2), np.nan, dtype=np.float32)
    return _array(dataset.coordinates, np.float32)


def _metadata(dataset: NanoDataset, name: str, default: int = -1) -> np.ndarray:
    value = (dataset.metadata or {}).get(name)
    if value is None or not isinstance(value, np.ndarray):
        return np.full(dataset.size, default, dtype=np.int64)
    return np.asarray(value)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _class_probabilities(model: Any, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    output = np.zeros((len(x), len(classes)), dtype=np.float64)
    positions = {int(value): index for index, value in enumerate(classes)}
    for index, value in enumerate(np.asarray(model.classes_)):
        target = positions.get(int(value))
        if target is not None:
            output[:, target] = probabilities[:, index]
    normalizer = output.sum(axis=1, keepdims=True)
    # A class absent from the training split receives a zero probability.  A
    # row predicted by a degenerate estimator still gets a valid distribution.
    output /= np.maximum(normalizer, 1.0e-12)
    return output


def _decision_probabilities(model: Any, x: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Turn a classifier decision function into the shared probability schema.

    LinearSVC deliberately has no ``predict_proba`` method.  A temperature-free
    softmax over its decision scores is sufficient for comparable ranking,
    calibration and top-k diagnostics while keeping the estimator itself
    deterministic and train-only.
    """
    scores = np.asarray(model.decision_function(x), dtype=np.float64)
    model_classes = np.asarray(model.classes_)
    if scores.ndim == 1:
        if len(model_classes) != 2:
            scores = scores[:, None]
        else:
            scores = np.stack((-scores, scores), axis=1)
    output = np.full((len(x), len(classes)), -np.inf, dtype=np.float64)
    positions = {int(value): index for index, value in enumerate(classes)}
    for index, value in enumerate(model_classes):
        target = positions.get(int(value))
        if target is not None:
            output[:, target] = scores[:, index]
    output -= np.max(output, axis=1, keepdims=True)
    probabilities = np.exp(output)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1.0e-12)
    return probabilities


def _global_classes(train_y: np.ndarray, eval_y: np.ndarray) -> np.ndarray:
    values = np.concatenate((train_y, eval_y))
    return np.arange(int(values.max()) + 1, dtype=np.int64) if len(values) else np.empty(0, dtype=np.int64)


def _fit_pca(x: np.ndarray, components: int, seed: int):
    from sklearn.decomposition import PCA

    components = max(1, min(int(components), x.shape[1], max(1, x.shape[0] - 1)))
    return PCA(n_components=components, svd_solver="randomized", random_state=int(seed)).fit(x)


def _neighbor_smooth(x: np.ndarray, dataset: NanoDataset) -> np.ndarray:
    if dataset.neighbors is None or dataset.neighbor_mask is None:
        return x.copy()
    neighbors = _array(dataset.neighbors).astype(np.int64, copy=False)
    mask = _array(dataset.neighbor_mask).astype(bool, copy=False)
    safe = np.clip(neighbors, 0, len(x) - 1)
    valid = mask & (safe != np.arange(len(x), dtype=np.int64)[:, None])
    gathered = x[safe]
    denominator = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    mean = (gathered * valid[..., None]).sum(axis=1) / denominator
    return np.where(valid.any(axis=1, keepdims=True), 0.5 * (x + mean), x)


def _metric_rows(
    *,
    task: str,
    artifact: Path,
    method: str,
    family: str,
    regime: str,
    fit_rows: int,
    eval_rows: int,
    runtime: float,
    notes: str,
) -> list[dict[str, Any]]:
    """Evaluate one artifact and turn scalar metrics into table rows."""
    base = {
        "split_id": SPLIT_ID,
        "subset": "nanobrain_v3_100k",
        "task": task,
        "method": method,
        "method_family": family,
        "training_regime": regime,
        "fit_rows": int(fit_rows),
        "eval_rows": int(eval_rows),
        "status": "complete",
        "runtime_seconds": float(runtime),
        "prediction_artifact": str(artifact),
        "notes": notes,
    }
    rows: list[dict[str, Any]] = []
    try:
        with np.load(artifact, allow_pickle=False) as payload:
            metrics = evaluate(task, payload)
            # Keep the evaluator contract as the source of truth while adding
            # task-specific held-out diagnostics below.
            if task == "spatial_completion":
                target = np.asarray(payload["target"], dtype=float)
                prediction = np.asarray(payload["prediction"], dtype=float)
                heldout = np.asarray(payload["mask"], dtype=bool)
                if heldout.shape == target.shape and heldout.any():
                    error = prediction[heldout] - target[heldout]
                    metrics["heldout_rmse"] = float(np.sqrt(np.mean(error * error)))
                    metrics["heldout_mae"] = float(np.mean(np.abs(error)))
                    if len(error) > 1 and np.std(target[heldout]) > 0 and np.std(prediction[heldout]) > 0:
                        metrics["heldout_pearson"] = float(np.corrcoef(target[heldout], prediction[heldout])[0, 1])
                    else:
                        metrics["heldout_pearson"] = math.nan
            elif task == "development":
                target = np.asarray(payload["target"], dtype=float)
                prediction = np.asarray(payload["prediction"], dtype=float)
                unpaired = bool(
                    np.asarray(payload["population_unpaired"]).reshape(-1)[0]
                ) if "population_unpaired" in payload else False
                if not unpaired:
                    metrics["row_mse"] = float(np.mean(np.square(prediction - target)))
                    metrics["row_mae"] = float(np.mean(np.abs(prediction - target)))
    except Exception as exc:  # pragma: no cover - exercised by optional backends
        base.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return [base]
    # Keep a task-local receipt beside every prediction artifact.  This makes
    # a single method independently auditable without parsing the aggregate
    # CSV and preserves undefined metrics as JSON null values.
    receipt_path = artifact.with_suffix(".metrics.json")
    receipt_path.write_text(
        json.dumps(
            {
                "format": "brainwm_v3_metric_receipt_v1",
                "split_id": SPLIT_ID,
                "subset": "nanobrain_v3_100k",
                "task": task,
                "method": method,
                "fit_rows": int(fit_rows),
                "eval_rows": int(eval_rows),
                "metrics": {key: (float(value) if np.isfinite(value) else None) for key, value in metrics.items()},
                "prediction_artifact": str(artifact),
                "test_opened": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    for metric, value in metrics.items():
        row = dict(base)
        row.update({
            "metric": metric,
            "value": float(value) if np.isfinite(value) else None,
            "direction": METRIC_DIRECTION.get(metric, "max"),
            "stratum_type": "overall",
            "stratum_value": "overall",
        })
        rows.append(row)
    return rows


def _append_group_classification_rows(
    rows: list[dict[str, Any]], *, artifact: Path, method: str, family: str,
    regime: str, fit_rows: int, runtime: float, notes: str, dataset: NanoDataset,
) -> None:
    """Add compact life-stage/anatomy accuracy and macro-F1 strata."""
    from sklearn.metrics import accuracy_score, f1_score

    try:
        with np.load(artifact, allow_pickle=False) as payload:
            y_true, y_pred = payload["y_true"], payload["y_pred"]
            eval_indices = payload["eval_indices"].astype(np.int64, copy=False)
    except Exception:
        return
    for stratum_type, values in (("life_stage", _metadata(dataset, "life_stage")), ("anatomy", _metadata(dataset, "anatomy"))):
        selected_values = values[eval_indices]
        for group in np.unique(selected_values):
            if int(group) < 0:
                continue
            take = selected_values == group
            if int(take.sum()) < 2:
                continue
            for metric, value in (
                ("accuracy", accuracy_score(y_true[take], y_pred[take])),
                ("f1_macro", f1_score(y_true[take], y_pred[take], average="macro", zero_division=0)),
            ):
                rows.append({
                    "split_id": SPLIT_ID, "subset": "nanobrain_v3_100k",
                    "task": "annotation", "method": method, "method_family": family,
                    "training_regime": regime, "fit_rows": int(fit_rows),
                    "eval_rows": int(take.sum()), "status": "complete",
                    "runtime_seconds": float(runtime), "prediction_artifact": str(artifact),
                    "notes": notes, "metric": metric, "value": float(value),
                    "direction": "max", "stratum_type": stratum_type,
                    "stratum_value": str(int(group)),
                })


def _run_annotation(
    train: NanoDataset, evaluation: NanoDataset, output: Path, seed: int,
    methods: set[str], include_slow: bool,
) -> list[dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

    train_values, train_valid = _labels(train, "annotation")
    eval_values, eval_valid = _labels(evaluation, "annotation")
    x_train, x_eval = _features(train), _features(evaluation)
    train_x, train_y = x_train[train_valid], train_values[train_valid].astype(np.int64)
    eval_indices = np.flatnonzero(eval_valid)
    eval_x, eval_y = x_eval[eval_valid], eval_values[eval_valid].astype(np.int64)
    classes = _global_classes(train_y, eval_y)
    if not len(train_y) or not len(eval_y):
        return []
    result: list[dict[str, Any]] = []

    def emit(name: str, family: str, regime: str, pred: np.ndarray, prob: np.ndarray, started: float, notes: str) -> None:
        artifact = output / "predictions" / f"{_safe_name(name)}__annotation.npz"
        _write_npz(artifact, y_true=eval_y, y_pred=pred.astype(np.int64), y_prob=prob.astype(np.float32), classes=classes, eval_indices=eval_indices)
        rows = _metric_rows(task="annotation", artifact=artifact, method=name, family=family, regime=regime,
                            fit_rows=len(train_y), eval_rows=len(eval_y), runtime=time.perf_counter() - started, notes=notes)
        _append_group_classification_rows(rows, artifact=artifact, method=name, family=family, regime=regime,
                                           fit_rows=len(train_y), runtime=time.perf_counter() - started, notes=notes, dataset=evaluation)
        result.extend(rows)

    if "majority" in methods:
        started = time.perf_counter()
        counts = np.bincount(train_y, minlength=len(classes))
        label = int(np.argmax(counts))
        prob = np.tile(counts / max(counts.sum(), 1), (len(eval_y), 1))
        emit("majority", "classical", "fit", np.full(len(eval_y), label), prob, started, "train-label mode")

    if "nearest_centroid" in methods:
        started = time.perf_counter()
        model = NearestCentroid()
        model.fit(train_x, train_y)
        pred = model.predict(eval_x)
        # NearestCentroid has no calibrated probabilities; use a one-hot
        # distribution so all classification metrics remain defined.
        prob = np.zeros((len(pred), len(classes)), dtype=np.float64)
        prob[np.arange(len(pred)), pred.astype(int)] = 1.0
        emit("nearest_centroid", "classical", "fit", pred, prob, started, "Euclidean train centroids")

    if "knn" in methods:
        started = time.perf_counter()
        model = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
        model.fit(train_x, train_y)
        pred = model.predict(eval_x)
        emit("knn", "classical", "fit", pred, _class_probabilities(model, eval_x, classes), started, "15-neighbor distance-weighted classifier")

    if "linear_sgd" in methods:
        started = time.perf_counter()
        model = SGDClassifier(loss="log_loss", max_iter=40, tol=1e-3, random_state=int(seed), n_jobs=-1)
        model.fit(train_x, train_y)
        pred = model.predict(eval_x)
        emit("linear_sgd", "classical", "fit", pred, _class_probabilities(model, eval_x, classes), started, "multinomial log-loss linear probe")

    if "multinomial_nb" in methods:
        started = time.perf_counter()
        # The V3 panel is z-normalized.  MultinomialNB requires non-negative
        # inputs, so use a shift computed from train rows only and clip rare
        # validation underflows at zero.
        shift = np.maximum(0.0, -np.min(train_x, axis=0)) + 1.0e-3
        model = MultinomialNB()
        model.fit(np.maximum(train_x + shift, 0.0), train_y)
        pred = model.predict(np.maximum(eval_x + shift, 0.0))
        emit(
            "multinomial_nb", "classical", "fit", pred,
            _class_probabilities(model, np.maximum(eval_x + shift, 0.0), classes),
            started, "MultinomialNB with train-derived non-negative feature shift",
        )

    if "linear_svm" in methods:
        started = time.perf_counter()
        # A hinge-loss SGD probe has the same linear SVM objective but avoids
        # the very slow one-vs-rest coordinate solver on the 180-class panel.
        # It is deterministic here and exposes decision_function just like
        # LinearSVC, so the artifact schema remains unchanged.
        model = SGDClassifier(
            loss="hinge", max_iter=40, tol=1e-3, random_state=int(seed),
            n_jobs=4, alpha=1.0e-4,
        )
        model.fit(train_x, train_y)
        pred = model.predict(eval_x)
        emit(
            "linear_svm", "classical", "fit", pred,
            _decision_probabilities(model, eval_x, classes),
            started, "linear hinge-loss SGD SVM; softmaxed decision scores for probability metrics",
        )

    if "logistic_regression" in methods and include_slow:
        started = time.perf_counter()
        model = LogisticRegression(solver="saga", max_iter=80, tol=1e-3, C=1.0, random_state=int(seed), n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_x, train_y)
        pred = model.predict(eval_x)
        emit("logistic_regression", "classical", "fit", pred, _class_probabilities(model, eval_x, classes), started, "saga multinomial logistic regression")
    return result


def _run_integration(train: NanoDataset, evaluation: NanoDataset, output: Path, seed: int, methods: set[str]) -> list[dict[str, Any]]:
    train_ann, train_ann_valid = _labels(train, "annotation")
    train_batch, train_batch_valid = _labels(train, "integration")
    eval_ann, eval_ann_valid = _labels(evaluation, "annotation")
    eval_batch, eval_batch_valid = _labels(evaluation, "integration")
    train_valid = train_ann_valid & train_batch_valid
    eval_valid = eval_ann_valid & eval_batch_valid
    x_train, x_eval = _features(train), _features(evaluation)
    pca = _fit_pca(x_train, min(32, x_train.shape[1]), seed)
    train_embedding = pca.transform(x_train).astype(np.float32)
    eval_embedding = pca.transform(x_eval).astype(np.float32)
    result: list[dict[str, Any]] = []
    for name in ("raw_pca", "batch_centered_pca"):
        if name not in methods:
            continue
        started = time.perf_counter()
        embedding = eval_embedding.copy()
        notes = "randomized PCA fit on all train rows"
        if name == "batch_centered_pca":
            global_mean = train_embedding.mean(axis=0)
            batch_means = {int(batch): train_embedding[train_batch == batch].mean(axis=0)
                           for batch in np.unique(train_batch[train_valid])}
            eval_batches = eval_batch
            embedding = np.stack([embedding[i] - batch_means.get(int(batch), global_mean) + global_mean
                                  for i, batch in enumerate(eval_batches)], axis=0)
            notes = "PCA followed by train-derived per-batch centering; unseen batches use train global mean"
        indices = np.flatnonzero(eval_valid)
        artifact = output / "predictions" / f"{name}__integration.npz"
        _write_npz(artifact, embedding=embedding[indices], biology_label=eval_ann[indices].astype(np.int64),
                   batch_label=eval_batch[indices].astype(np.int64), eval_indices=indices)
        result.extend(_metric_rows(task="integration", artifact=artifact, method=name, family="classical",
                                   regime="fit", fit_rows=int(train_valid.sum()), eval_rows=int(len(indices)),
                                   runtime=time.perf_counter() - started, notes=notes))
    return result


def _run_spatial(train: NanoDataset, evaluation: NanoDataset, output: Path, seed: int, methods: set[str]) -> list[dict[str, Any]]:
    from sklearn.cluster import MiniBatchKMeans

    train_label, train_label_valid = _labels(train, "niche")
    eval_label, eval_label_valid = _labels(evaluation, "niche")
    train_coord, eval_coord = _coords(train), _coords(evaluation)
    train_valid = train_label_valid & np.isfinite(train_coord[:, :2]).all(axis=1)
    eval_valid = eval_label_valid & np.isfinite(eval_coord[:, :2]).all(axis=1)
    x_train, x_eval = _features(train), _features(evaluation)
    pca = _fit_pca(x_train, min(24, x_train.shape[1]), seed)
    train_base, eval_base = pca.transform(x_train).astype(np.float32), pca.transform(x_eval).astype(np.float32)
    representations: dict[str, tuple[np.ndarray, np.ndarray, str]] = {
        "spatial_no_structure": (train_base, eval_base, "PCA embedding; coordinates are used only by the evaluator"),
        "spatial_neighbor_mean": (_neighbor_smooth(train_base, train), _neighbor_smooth(eval_base, evaluation), "PCA embedding with bounded in-section neighbor mean"),
        "spatial_coordinate_kmeans": (np.nan_to_num(train_coord[:, :2], nan=0.0), np.nan_to_num(eval_coord[:, :2], nan=0.0), "coordinate-only MiniBatchKMeans control"),
    }
    n_clusters = max(2, min(int(len(np.unique(train_label[train_valid]))), 64))
    result: list[dict[str, Any]] = []
    indices = np.flatnonzero(eval_valid)
    for name, (fit_rep, eval_rep, notes) in representations.items():
        if name not in methods:
            continue
        started = time.perf_counter()
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, n_init=3, max_iter=100, random_state=int(seed))
        model.fit(fit_rep[train_valid])
        clusters = model.predict(eval_rep[indices])
        artifact = output / "predictions" / f"{name}__spatial_niche.npz"
        _write_npz(artifact, embedding=eval_rep[indices], coordinates=eval_coord[indices, :2],
                   labels=eval_label[indices].astype(np.int64), clusters=clusters.astype(np.int64), eval_indices=indices)
        result.extend(_metric_rows(task="spatial_niche", artifact=artifact, method=name, family="classical",
                                   regime="fit", fit_rows=int(train_valid.sum()), eval_rows=len(indices),
                                   runtime=time.perf_counter() - started, notes=notes))
    return result


def _run_completion(train: NanoDataset, evaluation: NanoDataset, output: Path, methods: set[str]) -> list[dict[str, Any]]:
    x_train, x_eval = _features(train), _features(evaluation)
    mask = _array(evaluation.completion_mask).astype(bool, copy=False)
    target = _array(evaluation.completion_target if evaluation.completion_target is not None else evaluation.features, np.float32)
    train_mean = x_train.mean(axis=0)
    result: list[dict[str, Any]] = []
    for name in ("zero_fill", "gene_mean", "row_mean"):
        if name not in methods:
            continue
        started = time.perf_counter()
        observed = ~mask
        if name == "zero_fill":
            fill = np.zeros_like(x_eval)
            notes = "held-out entries set to zero; observed entries copied"
        elif name == "gene_mean":
            fill = np.broadcast_to(train_mean, x_eval.shape).copy()
            notes = "held-out entries filled with train gene means"
        else:
            row_sum = (x_eval * observed).sum(axis=1, keepdims=True)
            row_count = np.maximum(observed.sum(axis=1, keepdims=True), 1)
            row_mean = row_sum / row_count
            fill = np.broadcast_to(row_mean, x_eval.shape).copy()
            notes = "held-out entries filled with each row's observed mean"
        prediction = np.where(mask, fill, x_eval)
        artifact = output / "predictions" / f"{name}__spatial_completion.npz"
        _write_npz(artifact, target=target, prediction=prediction.astype(np.float32), mask=mask)
        result.extend(_metric_rows(task="spatial_completion", artifact=artifact, method=name, family="classical",
                                   regime="fit", fit_rows=len(x_train), eval_rows=len(x_eval),
                                   runtime=time.perf_counter() - started, notes=notes))
    return result


def _run_development(
    splits: DatasetSplits, output: Path, seed: int, methods: set[str]
) -> list[dict[str, Any]]:
    from sklearn.linear_model import Ridge

    train, evaluation = splits.train, splits.validation
    if splits.development_train is not None and splits.development_validation is not None:
        train_pairs = list(splits.development_train.iter_batches(max_cells=64, seed=seed))
        validation_pairs = list(
            splits.development_validation.iter_batches(max_cells=64, seed=seed)
        )
        train_source_means = np.stack(
            [_array(pair.source.features, np.float32).mean(0) for pair in train_pairs]
        )
        train_target_means = np.stack(
            [_array(pair.target.features, np.float32).mean(0) for pair in train_pairs]
        )
        train_delta = train_target_means - train_source_means
        interval_features = np.asarray(
            [[pair.source_time_days / 365.25, (pair.target_time_days - pair.source_time_days) / 365.25]
             for pair in train_pairs],
            dtype=np.float32,
        )
        ridge = Ridge(alpha=1.0).fit(
            np.concatenate((train_source_means, interval_features), axis=1), train_delta
        )
        global_delta = train_delta.mean(0)
        result: list[dict[str, Any]] = []
        population_methods = (
            "development_identity", "development_global_delta", "development_ridge_delta"
        )
        for name in population_methods:
            if name not in methods:
                continue
            started = time.perf_counter()
            targets: list[np.ndarray] = []
            predictions: list[np.ndarray] = []
            pointer = [0]
            for pair in validation_pairs:
                source = _array(pair.source.features, np.float32)
                target = _array(pair.target.features, np.float32)
                if name == "development_identity":
                    prediction = source
                    notes = "source population identity forecast"
                elif name == "development_global_delta":
                    prediction = source + global_delta
                    notes = "train-derived global population mean delta"
                else:
                    features = np.concatenate((
                        source.mean(0),
                        np.asarray([
                            pair.source_time_days / 365.25,
                            (pair.target_time_days - pair.source_time_days) / 365.25,
                        ], dtype=np.float32),
                    ))[None, :]
                    prediction = source + ridge.predict(features)[0]
                    notes = "fully fitted Ridge population-delta forecast"
                targets.append(target)
                predictions.append(np.asarray(prediction, dtype=np.float32))
                pointer.append(pointer[-1] + len(target))
            artifact = output / "predictions" / f"{name}__development.npz"
            _write_npz(
                artifact,
                target=np.concatenate(targets),
                prediction=np.concatenate(predictions),
                group_ptr=np.asarray(pointer, dtype=np.int64),
                population_unpaired=np.asarray([True]),
            )
            result.extend(_metric_rows(
                task="development", artifact=artifact, method=name,
                family="classical", regime="fit", fit_rows=sum(p.cells for p in train_pairs),
                eval_rows=sum(p.cells for p in validation_pairs),
                runtime=time.perf_counter() - started, notes=notes,
            ))
        return result

    train_target = _array(train.labels["development"], np.float32).reshape(train.size, -1)
    eval_target = _array(evaluation.labels["development"], np.float32).reshape(evaluation.size, -1)
    train_mask = _array(train.masks["development"]).astype(bool, copy=False).reshape(train.size, -1).all(axis=1) & np.isfinite(train_target).all(axis=1)
    eval_mask = _array(evaluation.masks["development"]).astype(bool, copy=False).reshape(evaluation.size, -1).all(axis=1) & np.isfinite(eval_target).all(axis=1)
    train_condition = _array(train.development_condition, np.float32).reshape(train.size, -1)[:, 0] if train.development_condition is not None else np.zeros(train.size)
    eval_condition = _array(evaluation.development_condition, np.float32).reshape(evaluation.size, -1)[:, 0] if evaluation.development_condition is not None else np.zeros(evaluation.size)
    x_train, x_eval = _features(train), _features(evaluation)
    fit_y = train_target[train_mask]
    eval_indices = np.flatnonzero(eval_mask)
    result: list[dict[str, Any]] = []
    for name in ("global_mean", "condition_mean", "ridge_features"):
        if name not in methods:
            continue
        started = time.perf_counter()
        if name == "global_mean":
            prediction = np.broadcast_to(fit_y.mean(axis=0), (len(eval_indices), fit_y.shape[1])).copy()
            notes = "global train population mean"
        elif name == "condition_mean":
            global_mean = fit_y.mean(axis=0)
            by_condition = {float(value): fit_y[train_condition[train_mask] == value].mean(axis=0)
                            for value in np.unique(train_condition[train_mask])}
            prediction = np.stack([by_condition.get(float(value), global_mean) for value in eval_condition[eval_indices]], axis=0)
            notes = "train-derived mean age by observed life-stage condition"
        else:
            model = Ridge(alpha=1.0)
            model.fit(x_train[train_mask], fit_y)
            prediction = np.asarray(model.predict(x_eval[eval_indices]))
            if prediction.ndim == 1:
                prediction = prediction[:, None]
            notes = "Ridge regression from the 256-gene panel"
        artifact = output / "predictions" / f"{name}__development.npz"
        _write_npz(artifact, target=eval_target[eval_indices], prediction=np.asarray(prediction, dtype=np.float32), eval_indices=eval_indices)
        result.extend(_metric_rows(task="development", artifact=artifact, method=name, family="classical",
                                   regime="fit", fit_rows=int(train_mask.sum()), eval_rows=len(eval_indices),
                                   runtime=time.perf_counter() - started, notes=notes))
    return result


def _find_default_checkpoint() -> Path | None:
    root = ROOT / "results" / "nanobrain"
    candidates = sorted(root.glob("campaign-*/candidate_*/candidate-*.pt"))
    if not candidates:
        return None
    # Select by the recorded validation quality, rather than directory order;
    # candidate_03 is not necessarily the best candidate in a generation.
    scored: list[tuple[float, float, Path]] = []
    for checkpoint in candidates:
        receipts = sorted(checkpoint.parent.glob("receipt-*.json"))
        quality = float("-inf")
        mtime = checkpoint.stat().st_mtime
        for receipt in receipts:
            try:
                payload = json.loads(receipt.read_text(encoding="utf-8"))
                quality = max(quality, float(payload.get("quality", float("-inf"))))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        scored.append((quality, mtime, checkpoint))
    return max(scored, key=lambda value: (value[0], value[1]))[2]


def _run_nanobrain(
    splits: DatasetSplits, checkpoint: Path, output: Path, device: str, batch_size: int,
    *, evaluation_seed: int | None = None,
) -> list[dict[str, Any]]:
    import torch
    from nanobrain.genome import CandidateGenome
    from nanobrain.model import NanoBrainModel, NanoBrainModelConfig

    if checkpoint.suffix == ".json":
        import hashlib
        bundle = json.loads(checkpoint.read_text())
        if bundle.get("format") != "nanobrain_task_checkpoints_v1":
            raise ValueError("unknown checkpoint bundle")
        rows = []
        for task, record in bundle["tasks"].items():
            path = Path(record["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"task checkpoint checksum mismatch: {task}")
            task_payload = torch.load(path, map_location="cpu", weights_only=False)
            if task_payload.get("train", {}).get("task") != task or task_payload["genome"] != bundle["genome"]:
                raise ValueError("task checkpoint provenance mismatch")
            rows.extend(_run_nanobrain(splits, path, output, device, batch_size, evaluation_seed=evaluation_seed))
        return rows
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_task = payload.get("train", {}).get("task", "joint")
    from nanobrain.task_training import TASKS
    if training_task not in ("joint", *TASKS):
        raise ValueError("invalid checkpoint training task")
    selected_tasks = set(TASKS) if training_task == "joint" else {training_task}
    regime = "joint_from_scratch" if training_task == "joint" else "independent_task_from_scratch"
    config = NanoBrainModelConfig(**payload["config"])
    genome = CandidateGenome.from_dict(payload["genome"])
    from nanobrain.integration_components import prepare_component_views
    prepare_component_views(splits, genome)
    population_seed = genome.seed if evaluation_seed is None else int(evaluation_seed)
    model = NanoBrainModel(config, genome)
    model.load_state_dict(payload["model"])
    requested = str(device).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    torch_device = torch.device(requested)
    model.to(torch_device).eval()
    eval_set = splits.validation
    collected: dict[str, list[np.ndarray]] = {"embedding": [], "annotation": [], "niche": [], "integration": [], "development": [], "completion": []}
    started = time.perf_counter()
    with torch.no_grad():
        if selected_tasks != {"development"} or splits.development_validation is None:
            for batch in eval_set.iter_batches(int(batch_size), shuffle=False):
                batch = batch.to(torch_device)
                if training_task == "joint":
                    prediction = model(batch)
                    for key, attribute in (("embedding", "embedding"), ("annotation", "annotation_logits"),
                                           ("niche", "niche_logits"), ("integration", "integration_prediction"),
                                           ("development", "development_prediction"), ("completion", "completion_prediction")):
                        collected[key].append(_array(getattr(prediction, attribute)))
                else:
                    from nanobrain.task_training import task_input
                    embedding = model.encode(task_input(batch, training_task))
                    collected["embedding"].append(_array(embedding))
                    head = {"annotation": ("annotation", model.annotation_head),
                            "spatial_niche": ("niche", model.niche_head),
                            "spatial_completion": ("completion", model.completion_head)}.get(training_task)
                    if head:
                        collected[head[0]].append(_array(head[1](embedding)))
    outputs = {key: np.concatenate(value, axis=0) for key, value in collected.items() if value}
    candidate_id = genome.genome_id
    candidate_seed = str(payload.get("genome", {}).get("seed", checkpoint.stem))
    method = f"nanobrain_candidate-{candidate_id}"
    result: list[dict[str, Any]] = []

    def emit(task: str, artifact: Path, fit_rows: int, eval_rows: int, notes: str) -> None:
        result.extend(_metric_rows(task=task, artifact=artifact, method=method, family="ours",
                                   regime=regime, fit_rows=fit_rows, eval_rows=eval_rows,
                                   runtime=time.perf_counter() - started, notes=notes))

    if "annotation" in selected_tasks:
        ann, ann_valid = _labels(eval_set, "annotation")
        indices = np.flatnonzero(ann_valid)
        logits = outputs["annotation"][indices]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True)); probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        artifact = output / "predictions" / f"{method}__annotation.npz"
        _write_npz(artifact, y_true=ann[indices].astype(np.int64), y_pred=logits.argmax(axis=1), y_prob=probs.astype(np.float32), classes=np.arange(logits.shape[1]), eval_indices=indices)
        emit("annotation", artifact, splits.train.size, len(indices), f"checkpoint={checkpoint}; candidate_seed={candidate_seed}")
        _append_group_classification_rows(result, artifact=artifact, method=method, family="ours",
                                           regime=regime, fit_rows=splits.train.size, runtime=time.perf_counter() - started,
                                           notes=f"checkpoint={checkpoint}", dataset=eval_set)

    if "spatial_niche" in selected_tasks:
        niche, niche_valid = _labels(eval_set, "niche")
        indices = np.flatnonzero(niche_valid & np.isfinite(_coords(eval_set)[:, :2]).all(axis=1))
        niche_logits = outputs["niche"][indices]
        artifact = output / "predictions" / f"{method}__spatial_niche.npz"
        _write_npz(artifact, embedding=outputs["embedding"][indices], coordinates=_coords(eval_set)[indices, :2], labels=niche[indices].astype(np.int64), clusters=niche_logits.argmax(axis=1).astype(np.int64), eval_indices=indices)
        emit("spatial_niche", artifact, splits.train.size, len(indices), f"checkpoint={checkpoint}; niche head")

    if "integration" in selected_tasks:
        integ, integ_valid = _labels(eval_set, "integration")
        ann_values, ann_valid = _labels(eval_set, "annotation")
        indices = np.flatnonzero(integ_valid & ann_valid)
        artifact = output / "predictions" / f"{method}__integration.npz"
        _write_npz(artifact, embedding=outputs["embedding"][indices], biology_label=ann_values[indices].astype(np.int64), batch_label=integ[indices].astype(np.int64), eval_indices=indices)
        emit("integration", artifact, splits.train.size, len(indices), f"checkpoint={checkpoint}; shared embedding")

    if "spatial_completion" in selected_tasks:
        target = _array(eval_set.completion_target if eval_set.completion_target is not None else eval_set.features, np.float32)
        mask = _array(eval_set.completion_mask).astype(bool, copy=False)
        artifact = output / "predictions" / f"{method}__spatial_completion.npz"
        _write_npz(artifact, target=target, prediction=outputs["completion"].astype(np.float32), mask=mask)
        emit("spatial_completion", artifact, splits.train.size, eval_set.size, f"checkpoint={checkpoint}; completion head")

    if "development" in selected_tasks:
        artifact = output / "predictions" / f"{method}__development.npz"
        if splits.development_validation is not None:
            targets: list[np.ndarray] = []
            predictions: list[np.ndarray] = []
            pointer = [0]
            with torch.no_grad():
                for population in splits.development_validation.iter_batches(
                    max_cells=64, seed=population_seed
                ):
                    population = population.to(torch_device)
                    with torch.autocast(
                        device_type=torch_device.type,
                        dtype=torch.bfloat16,
                        enabled=torch_device.type == "cuda",
                    ):
                        prediction, _ = model.predict_development_population(
                            population.source,
                            population.source_time_days,
                            population.target_time_days,
                        )
                    target = _array(population.target.features, np.float32)
                    targets.append(target)
                    predictions.append(_array(prediction, np.float32))
                    pointer.append(pointer[-1] + len(target))
            _write_npz(
                artifact,
                target=np.concatenate(targets),
                prediction=np.concatenate(predictions),
                group_ptr=np.asarray(pointer, dtype=np.int64),
                population_unpaired=np.asarray([True]),
            )
            emit(
                "development", artifact, sum(
                    int(group["cells"]) for group in splits.development_train.groups
                ) if splits.development_train is not None else splits.train.size,
                pointer[-1], f"checkpoint={checkpoint}; conditional population rollout; evaluation_seed={population_seed}",
            )
        else:
            dev, dev_valid = _labels(eval_set, "development")
            indices = np.flatnonzero(dev_valid)
            _write_npz(artifact, target=dev[indices].astype(np.float32), prediction=outputs["development"][indices].astype(np.float32), eval_indices=indices)
            emit("development", artifact, splits.train.size, len(indices), f"checkpoint={checkpoint}; legacy row head")
    return result


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split_id", "subset", "task", "method", "method_family", "training_regime", "fit_rows", "eval_rows", "status", "runtime_seconds", "prediction_artifact", "metric", "value", "direction", "stratum_type", "stratum_value", "notes", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "assets" / "nanobrain_v3_100k")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--population-evaluation-seed", type=int, default=None,
                        help="fixed population sample shared across candidates and training seeds")
    parser.add_argument("--include-slow", action="store_true", help="also run saga logistic regression")
    parser.add_argument("--skip-baselines", action="store_true", help="evaluate only the supplied nanoBrain checkpoint")
    parser.add_argument("--skip-nanobrain", action="store_true")
    parser.add_argument(
        "--update-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="update the global LATEST_PILOT pointer after a standalone run",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--methods", default="all", help="comma-separated method names or all")
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required for the single frozen-protocol test evaluation",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.evaluation_split == "test" and not args.allow_test:
        parser.error("test evaluation requires --allow-test after protocol freeze")
    if args.allow_test and args.evaluation_split != "test":
        parser.error("--allow-test is only valid with --evaluation-split test")
    loaded_splits = ("train", args.evaluation_split)
    manifest = verify_subset_manifest(
        args.data,
        verify_hashes=False,
        data_splits=loaded_splits,
    )
    source_splits = load_v3_subset(
        args.data,
        verify=False,
        include_test=args.evaluation_split == "test",
    )
    if args.evaluation_split == "test":
        if source_splits.test is None:
            parser.error("test split is unavailable")
        splits = DatasetSplits(
            train=source_splits.train,
            validation=source_splits.test,
            test=None,
            development_train=source_splits.development_train,
            development_validation=source_splits.development_test,
            development_test=None,
        )
    else:
        splits = source_splits
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or (ROOT / "results" / "nanobrain" / f"benchmark-pilot-{stamp}")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_methods = {
        "majority", "nearest_centroid", "knn", "linear_sgd", "multinomial_nb", "linear_svm", "logistic_regression",
        "raw_pca", "batch_centered_pca", "spatial_no_structure", "spatial_neighbor_mean",
        "spatial_coordinate_kmeans", "zero_fill", "gene_mean", "row_mean",
        "global_mean", "condition_mean", "ridge_features",
        "development_identity", "development_global_delta", "development_ridge_delta",
    }
    methods = all_methods if args.methods.strip().lower() == "all" else {item.strip() for item in args.methods.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if not args.skip_baselines:
        # Use explicit task prefixes to keep similarly named methods unambiguous.
        rows.extend(_run_annotation(splits.train, splits.validation, output, args.seed, methods, args.include_slow))
        rows.extend(_run_integration(splits.train, splits.validation, output, args.seed, methods))
        rows.extend(_run_spatial(splits.train, splits.validation, output, args.seed, methods))
        rows.extend(_run_completion(splits.train, splits.validation, output, methods))
        rows.extend(_run_development(splits, output, args.seed, methods))
    checkpoint = None if args.skip_nanobrain else (args.checkpoint or _find_default_checkpoint())
    if not args.skip_nanobrain and checkpoint is not None and checkpoint.is_file():
        rows.extend(_run_nanobrain(splits, checkpoint.resolve(), output, args.device, args.batch_size,
                                   evaluation_seed=args.population_evaluation_seed))
    elif not args.skip_nanobrain:
        warnings.warn("no nanoBrain checkpoint found; baseline-only pilot", RuntimeWarning)
    _write_rows(output / "benchmark_table.csv", rows)
    manifest_out = {
        "format": "nanobrain_pilot_benchmark_v1",
        "split_id": SPLIT_ID,
        "data": str(Path(args.data).resolve()),
        "data_manifest_sha256": __import__("hashlib").sha256((Path(args.data) / "manifest.json").read_bytes()).hexdigest(),
        "rows": {split: int(manifest["splits"][split]["rows"]) for split in ("train", "validation", "test")},
        "fit_split": "train", "evaluation_split": args.evaluation_split,
        "test_opened": args.evaluation_split == "test",
        "population_evaluation_seed": args.population_evaluation_seed,
        "graph_metric_version": "graph_metrics_v2",
        "tasks_evaluated": sorted({row["task"] for row in rows}),
        "training_regimes": sorted({row["training_regime"] for row in rows}),
        "methods_requested": sorted(methods), "checkpoint": str(checkpoint.resolve()) if checkpoint and checkpoint.is_file() else None,
        "runtime_seconds": float(time.perf_counter() - started), "metric_rows": len(rows),
        "prediction_artifacts": sorted(str(path.relative_to(output)) for path in (output / "predictions").glob("*.npz")),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # A stable pointer makes the latest pilot discoverable without changing
    # the immutable, timestamped run directory.
    if args.update_latest:
        pointer = ROOT / "results" / "nanobrain" / "LATEST_PILOT.json"
        pointer.write_text(json.dumps({"path": str(output), **manifest_out}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metric_rows": len(rows), "methods": sorted(methods), "checkpoint": str(checkpoint) if checkpoint else None}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
