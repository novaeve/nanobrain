"""Task-neutral NumPy prediction evaluators for BrainWM Human V3."""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, normalized_mutual_info_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def _finite(name: str, value, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(value)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {out.ndim}")
    if np.issubdtype(out.dtype, np.number) and not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def _corr(a: np.ndarray, b: np.ndarray, method="pearson") -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(fn(a, b).statistic)


def _mean_axis_corr(a: np.ndarray, b: np.ndarray, axis: int, method="pearson") -> float:
    values = [_corr(x, y, method) for x, y in zip(np.moveaxis(a, axis, 0), np.moveaxis(b, axis, 0))]
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else math.nan


def annotation(data: Mapping[str, np.ndarray]) -> dict[str, float]:
    y = _finite("y_true", data["y_true"], 1)
    p = _finite("y_pred", data["y_pred"], 1)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred shapes differ")
    result = {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "mcc": matthews_corrcoef(y, p),
        "cohen_kappa": cohen_kappa_score(y, p),
    }
    for average in ("macro", "micro", "weighted"):
        result[f"precision_{average}"] = precision_score(y, p, average=average, zero_division=0)
        result[f"recall_{average}"] = recall_score(y, p, average=average, zero_division=0)
        result[f"f1_{average}"] = f1_score(y, p, average=average, zero_division=0)
    if "y_prob" in data:
        prob = _finite("y_prob", data["y_prob"], 2).astype(float)
        if len(prob) != len(y):
            raise ValueError("y_prob row count differs")
        prob = prob / prob.sum(1, keepdims=True).clip(1e-12)
        classes = np.asarray(data.get("classes", np.arange(prob.shape[1])))
        onehot = (y[:, None] == classes[None, :]).astype(float)
        confidence, predicted = prob.max(1), classes[prob.argmax(1)]
        bins = np.minimum((confidence * 15).astype(int), 14)
        ece = sum(np.mean(bins == i) * abs(np.mean(predicted[bins == i] == y[bins == i]) - np.mean(confidence[bins == i])) for i in np.unique(bins))
        result.update({
            "nll": log_loss(y, prob, labels=classes),
            "brier_multiclass": np.mean(np.sum((prob - onehot) ** 2, axis=1)),
            "ece": ece,
            "top3_accuracy": np.mean([truth in classes[np.argpartition(row, -min(3, len(row)))[-min(3, len(row)):]] for truth, row in zip(y, prob)]),
        })
        try:
            result["auroc_macro_ovr"] = roc_auc_score(onehot, prob, average="macro", multi_class="ovr")
            result["auprc_macro"] = average_precision_score(onehot, prob, average="macro")
        except ValueError:
            result["auroc_macro_ovr"] = result["auprc_macro"] = math.nan
    return {k: float(v) for k, v in result.items()}


def completion(data: Mapping[str, np.ndarray]) -> dict[str, float]:
    truth = _finite("target", data["target"], 2).astype(float)
    pred = _finite("prediction", data["prediction"], 2).astype(float)
    if truth.shape != pred.shape:
        raise ValueError("target and prediction shapes differ")
    error = pred - truth
    nz = truth != 0
    result = {
        "rmse": np.sqrt(np.mean(error ** 2)), "mae": np.mean(np.abs(error)),
        "pearson_global": _corr(truth.ravel(), pred.ravel()),
        "pearson_cell": _mean_axis_corr(truth, pred, 0),
        "pearson_gene": _mean_axis_corr(truth, pred, 1),
        "spearman_gene": _mean_axis_corr(truth, pred, 1, "spearman"),
        "cosine": np.mean(np.sum(truth * pred, 1) / (np.linalg.norm(truth, axis=1) * np.linalg.norm(pred, axis=1)).clip(1e-12)),
        "nonzero_rmse": np.sqrt(np.mean(error[nz] ** 2)) if nz.any() else math.nan,
        "nonzero_pearson": _corr(truth[nz], pred[nz]) if nz.sum() > 1 else math.nan,
    }
    if "zero_probability" in data:
        zp = _finite("zero_probability", data["zero_probability"], 2)
        result["zero_calibration_error"] = np.mean(np.abs(zp - (truth == 0)))
    return {k: float(v) for k, v in result.items()}


def _rbf_mmd(a: np.ndarray, b: np.ndarray) -> float:
    # MMD is a diagnostic, not a reason to materialize an O(N^2) kernel over
    # the whole held-out corpus.  Use a deterministic bounded sample for both
    # distributions; this keeps the metric stable across runs and prevents a
    # 20k-row pilot from allocating tens of gigabytes.
    limit = 2048
    a = np.asarray(a)[:limit]
    b = np.asarray(b)[:limit]
    merged = np.concatenate([a, b])
    sample = merged[:min(len(merged), limit)]
    distance = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=2)
    positive = distance[distance > 0]
    gamma = 1.0 / max(float(np.median(positive)) if positive.size else 1.0, 1e-12)
    def kernel(x, z): return np.exp(-gamma * np.sum((x[:, None] - z[None, :]) ** 2, axis=2))
    return float(kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean())


def _development_distribution(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if truth.shape[1] != pred.shape[1]:
        raise ValueError("target and prediction feature counts differ")
    n = min(len(truth), len(pred), 4096)
    if n < 2:
        raise ValueError("development population metrics require at least two cells")
    truth, pred = truth[:n], pred[:n]
    mean_error = np.mean((truth.mean(0) - pred.mean(0)) ** 2)
    variance_error = np.mean((truth.var(0) - pred.var(0)) ** 2)
    return {
        "population_mse": mean_error,
        "mean_error": mean_error,
        "variance_error": variance_error,
        "mmd_rbf": _rbf_mmd(truth, pred),
        "gene_pearson": _corr(truth.mean(0), pred.mean(0)),
        "gene_spearman": _corr(truth.mean(0), pred.mean(0), "spearman"),
        "output_floor_fraction": np.mean(pred <= 0),
    }


def development(data: Mapping[str, np.ndarray]) -> dict[str, float]:
    truth = _finite("target", data["target"], 2).astype(float)
    pred = _finite("prediction", data["prediction"], 2).astype(float)
    group_ptr = data.get("group_ptr")
    if group_ptr is None:
        result = _development_distribution(truth, pred)
    else:
        pointer = np.asarray(group_ptr, dtype=np.int64)
        if pointer.ndim != 1 or len(pointer) < 2 or pointer[0] != 0:
            raise ValueError("development group_ptr must be rank-1 and start at zero")
        if pointer[-1] != len(truth) or pointer[-1] != len(pred):
            raise ValueError("development group_ptr must cover target and prediction rows")
        if np.any(np.diff(pointer) < 2):
            raise ValueError("every development population group requires at least two cells")
        per_group = [
            _development_distribution(truth[start:end], pred[start:end])
            for start, end in zip(pointer[:-1], pointer[1:])
        ]
        result = {}
        for name in per_group[0]:
            values = np.asarray([metrics[name] for metrics in per_group], dtype=float)
            finite = values[np.isfinite(values)]
            result[name] = float(finite.mean()) if finite.size else math.nan
    if "target_cell_type" in data and "prediction_cell_type" in data:
        yt, yp = np.asarray(data["target_cell_type"]), np.asarray(data["prediction_cell_type"])
        labels = np.union1d(yt, yp)
        result["celltype_proportion_error"] = np.mean([abs(np.mean(yt == x) - np.mean(yp == x)) for x in labels])
    return {k: float(v) for k, v in result.items()}


def _knn_graph(x: np.ndarray, k: int = 15) -> tuple[np.ndarray, sparse.csr_matrix]:
    k = min(k, len(x) - 1)
    if k < 1: raise ValueError("at least two observations are required")
    # With X=None sklearn already excludes the query row. Asking for k+1
    # then slicing again skipped the nearest neighbor and failed for n<=k+1.
    idx = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(return_distance=False)
    rows = np.repeat(np.arange(len(x)), k)
    graph = sparse.csr_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=(len(x), len(x)))
    return idx, graph.maximum(graph.T)


def integration(data: Mapping[str, np.ndarray]) -> dict[str, float]:
    x = _finite("embedding", data["embedding"], 2).astype(float)
    biology, batch = np.asarray(data["biology_label"]), np.asarray(data["batch_label"])
    if len(x) != len(biology) or len(x) != len(batch): raise ValueError("integration row counts differ")
    idx, graph = _knn_graph(x)
    cell_asw = (silhouette_score(x, biology, sample_size=min(len(x), 10000), random_state=0) + 1) / 2
    batch_asw = 1 - (silhouette_score(x, batch, sample_size=min(len(x), 10000), random_state=0) + 1) / 2
    connect = []
    for label in np.unique(biology):
        take = np.flatnonzero(biology == label)
        if len(take) > 1:
            _, component = connected_components(graph[take][:, take])
            connect.append(np.bincount(component).max() / len(take))
    graph_connectivity = float(np.mean(connect))
    local_batch = batch[idx]
    ilisi = np.mean([1 / np.sum((np.unique(row, return_counts=True)[1] / len(row)) ** 2) for row in local_batch])
    ilisi /= max(1, len(np.unique(batch)))
    same_biology = biology[idx] == biology[:, None]
    isolated = f1_score(np.ones(same_biology.size), same_biology.ravel(), zero_division=0)
    biology_score = np.mean([cell_asw, graph_connectivity, isolated])
    batch_score = np.mean([batch_asw, ilisi])
    return {k: float(v) for k, v in {
        "celltype_asw": cell_asw, "batch_asw_inverse": batch_asw,
        "graph_connectivity": graph_connectivity, "isolated_label_f1": isolated,
        "ilisi_normalized": ilisi, "biology_conservation": biology_score,
        "batch_removal": batch_score, "integration_composite": .6 * biology_score + .4 * batch_score,
    }.items()}


def spatial_niche(data: Mapping[str, np.ndarray]) -> dict[str, float]:
    x = _finite("embedding", data["embedding"], 2).astype(float)
    coord = _finite("coordinates", data["coordinates"], 2).astype(float)
    labels = np.asarray(data["labels"])
    clusters = np.asarray(data["clusters"])
    idx, graph = _knn_graph(coord)
    same = labels[idx] == labels[:, None]
    centered = x[:, 0] - x[:, 0].mean()
    w = graph.sum()
    moran = len(x) / w * float(centered @ graph @ centered) / float(centered @ centered)
    geary = (len(x) - 1) / (2 * w) * float(sum((centered[i] - centered[j]) ** 2 for i, j in zip(*graph.nonzero()))) / float(centered @ centered)
    return {"ari": float(adjusted_rand_score(labels, clusters)),
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
            "celltype_asw": float((silhouette_score(x, labels, sample_size=min(len(x),10000),random_state=0)+1)/2),
            "morans_i": moran, "gearys_c": geary,
            "neighbor_label_enrichment": float(same.mean())}


EVALUATORS = {"annotation": annotation, "integration": integration,
              "spatial_niche": spatial_niche, "spatial_completion": completion,
              "development": development}


def evaluate(task: str, data: Mapping[str, np.ndarray]) -> dict[str, float]:
    if task not in EVALUATORS: raise ValueError(f"unknown task: {task}")
    return EVALUATORS[task](data)
