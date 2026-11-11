"""Task-neutral NumPy prediction evaluators for BrainWM Human V3."""
from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np
from scipy import sparse, stats
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, normalized_mutual_info_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from .protocol import (
    DEVELOPMENT_EVAL_CELLS, EVALUATION_SEED, SCIB_MAX_CELLS, SCIB_NEIGHBORS,
    SCIB_VERSION, SINKHORN_BLUR, SINKHORN_SCALING,
)


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
        # The macro class universe must not change with a candidate's predictions.
        kwargs = {"labels": np.unique(y)} if average == "macro" else {}
        result[f"precision_{average}"] = precision_score(y, p, average=average, zero_division=0, **kwargs)
        result[f"recall_{average}"] = recall_score(y, p, average=average, zero_division=0, **kwargs)
        result[f"f1_{average}"] = f1_score(y, p, average=average, zero_division=0, **kwargs)
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
    # A fixed bandwidth in the shared feature space cannot be enlarged by a
    # candidate to make its own discrepancy disappear. Avoid N x N x D arrays.
    a, b = a / np.sqrt(a.shape[1]), b / np.sqrt(b.shape[1])
    def kernel(x, z): return np.exp(-cdist(x, z, "sqeuclidean"))
    return float(kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean())


def _population_sample(x: np.ndarray) -> np.ndarray:
    if len(x) <= DEVELOPMENT_EVAL_CELLS:
        return x
    indices = np.sort(np.random.default_rng(EVALUATION_SEED).choice(
        len(x), DEVELOPMENT_EVAL_CELLS, replace=False))
    return x[indices]


def sinkhorn_divergence(truth: np.ndarray, pred: np.ndarray) -> float:
    """Debiased distribution error in a common, dimension-normalized gene space."""
    import torch
    from geomloss import SamplesLoss

    truth, pred = _population_sample(truth), _population_sample(pred)
    scale = np.sqrt(truth.shape[1])
    with torch.no_grad():
        loss = SamplesLoss("sinkhorn", p=2, blur=SINKHORN_BLUR,
                           scaling=SINKHORN_SCALING, debias=True, backend="tensorized")
        value = float(loss(torch.tensor(truth / scale, dtype=torch.float64),
                           torch.tensor(pred / scale, dtype=torch.float64)))
    if not math.isfinite(value) or value < -1e-7:
        raise ValueError(f"invalid Sinkhorn divergence: {value}")
    return max(0.0, value)


def wasserstein2(truth: np.ndarray, pred: np.ndarray) -> float:
    """Exact empirical W2; no row pairing or model-specific feature space."""
    from scipy.optimize import linear_sum_assignment
    truth, pred = _population_sample(truth), _population_sample(pred)
    cost = cdist(truth, pred, "sqeuclidean")
    if len(truth) == len(pred):
        rows, columns = linear_sum_assignment(cost)
        squared = cost[rows, columns].mean()
    else:
        from ot.lp import emd2
        squared = emd2(np.full(len(truth), 1 / len(truth)),
                       np.full(len(pred), 1 / len(pred)), cost)
    return float(np.sqrt(max(0.0, squared)))


def _development_distribution(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if truth.shape[1] != pred.shape[1]:
        raise ValueError("target and prediction feature counts differ")
    if min(len(truth), len(pred)) < 2:
        raise ValueError("development population metrics require at least two cells")
    truth, pred = _population_sample(truth), _population_sample(pred)
    mean_error = np.mean((truth.mean(0) - pred.mean(0)) ** 2)
    variance_error = np.mean((truth.var(0) - pred.var(0)) ** 2)
    return {
        "wasserstein2": wasserstein2(truth, pred),
        "sinkhorn_divergence": sinkhorn_divergence(truth, pred),
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


def _integration_sample(biology: np.ndarray, batch: np.ndarray) -> np.ndarray:
    if len(biology) <= SCIB_MAX_CELLS:
        return np.arange(len(biology))
    # Selection depends only on the fixed labels/row order, never on embeddings.
    _, groups = np.unique(np.column_stack([biology, batch]), axis=0, return_inverse=True)
    rng = np.random.default_rng(EVALUATION_SEED)
    required = np.array([rng.choice(np.flatnonzero(groups == g)) for g in np.unique(groups)])
    if len(required) > SCIB_MAX_CELLS:
        raise ValueError("more integration strata than the frozen sample budget")
    rest = np.setdiff1d(np.arange(len(biology)), required)
    extra = rng.choice(rest, SCIB_MAX_CELLS - len(required), replace=False)
    return np.sort(np.concatenate([required, extra]))


def integration(data: Mapping[str, np.ndarray], *, diagnostics=True) -> dict[str, float]:
    """Fixed official scIB subset; no approximate lookalikes or silent fallback."""
    import os
    from importlib.metadata import version

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if version("scib-metrics") != SCIB_VERSION:
        raise RuntimeError(f"this protocol requires scib-metrics=={SCIB_VERSION}")
    import jax
    import scib_metrics as scib
    from scib_metrics.nearest_neighbors import NeighborsResults

    x = _finite("embedding", data["embedding"], 2).astype(float)
    biology, batch = np.asarray(data["biology_label"]), np.asarray(data["batch_label"])
    if len(x) != len(biology) or len(x) != len(batch): raise ValueError("integration row counts differ")
    take = _integration_sample(biology, batch)
    x, biology, batch = x[take], biology[take], batch[take]
    if not 1 < len(np.unique(biology)) < len(x) or len(np.unique(batch)) < 2:
        raise ValueError("scIB requires multiple biological labels and batches")
    if not any(1 < len(np.unique(batch[biology == label])) < int((biology == label).sum())
               for label in np.unique(biology)):
        raise ValueError("no cell type has evaluable within-type batch replication")
    neighbors = min(SCIB_NEIGHBORS, len(x))
    distances, indices = NearestNeighbors(n_neighbors=neighbors).fit(x).kneighbors(x)
    knn = NeighborsResults(indices=indices, distances=distances)
    with jax.default_device(jax.devices("cpu")[0]):
        clustering = scib.nmi_ari_cluster_labels_kmeans(x, biology)
        metrics = {
            "scib_label_asw": scib.silhouette_label(x, biology),
            "scib_isolated_label_asw": scib.isolated_labels(x, biology, batch),
            "scib_nmi": clustering["nmi"],
            "scib_ari": clustering["ari"],
            "scib_batch_asw": scib.silhouette_batch(x, biology, batch),
            "scib_ilisi": scib.ilisi_knn(knn, batch),
            "scib_graph_connectivity": scib.graph_connectivity(knn.subset_neighbors(min(15, neighbors)), biology),
        }
        conditional = []
        if diagnostics:
            metrics["bras"] = scib.bras(x, biology, batch)
        for label in np.unique(biology) if diagnostics else []:
            take = np.flatnonzero(biology == label)
            if len(take) < 6 or len(np.unique(batch[take])) < 2:
                continue
            distances, indices = NearestNeighbors(n_neighbors=min(SCIB_NEIGHBORS, len(take))).fit(x[take]).kneighbors(x[take])
            value = scib.ilisi_knn(NeighborsResults(indices=indices, distances=distances), batch[take])
            if not np.isfinite(value):
                raise ValueError("undefined conditional iLISI")
            conditional.append(float(value))
        if diagnostics and conditional:
            metrics["cilisi"] = float(np.mean(conditional))
    for name, value in metrics.items():
        low = -1 if name == "scib_ari" else 0
        if not np.isfinite(value) or not low - 1e-5 <= value <= 1 + 1e-5:
            raise ValueError(f"undefined or out-of-range scIB component: {name}={value}")
        metrics[name] = float(np.clip(value, low, 1))
    bio = np.mean([metrics["scib_label_asw"], metrics["scib_isolated_label_asw"],
                   metrics["scib_nmi"], max(0.0, metrics["scib_ari"])])
    mixing = np.mean([metrics["scib_batch_asw"], metrics["scib_ilisi"], metrics["scib_graph_connectivity"]])
    metrics.update(biology_conservation=float(bio), batch_removal=float(mixing),
                   scib_composite=float(0.6 * bio + 0.4 * mixing))
    if diagnostics:
        metrics["cilisi_evaluable_labels"] = float(len(conditional))
        metrics.setdefault("cilisi", math.nan)
    return metrics


def niche_composition(data: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Paired center-cell targets; equal weighting of prespecified radii."""
    truth = _finite("target_composition", data["target_composition"], 3).astype(float)
    pred = _finite("prediction_composition", data["prediction_composition"], 3).astype(float)
    mask = np.asarray(data.get("composition_mask", np.ones(truth.shape[:2])), dtype=bool)
    if truth.shape != pred.shape or mask.shape != truth.shape[:2]:
        raise ValueError("niche composition shape mismatch")
    if np.any(pred < -1e-7) or not np.allclose(pred.sum(-1), 1, atol=1e-5):
        raise ValueError("predicted compositions must be probability vectors")
    if not mask.any() or not np.allclose(truth[mask].sum(-1), 1, atol=1e-5) or np.any(truth < 0):
        raise ValueError("niche composition has no valid normalized targets")
    maes, mses, jsds = [], [], []
    result = {}
    for scale in range(truth.shape[1]):
        take = mask[:, scale]
        if not take.any():
            raise ValueError(f"no valid niche targets at scale {scale}")
        y, p = truth[take, scale], pred[take, scale]
        error = np.abs(y - p)
        mid = (y + p) / 2
        jsd = .5 * np.sum(y * np.log(y.clip(1e-12) / mid.clip(1e-12))
                           + p * np.log(p.clip(1e-12) / mid.clip(1e-12)), axis=1)
        maes.append(float(error.mean())); mses.append(float(np.square(y - p).mean())); jsds.append(float(jsd.mean()))
        result[f"composition_mae_scale_{scale}"] = maes[-1]
        for cell_type, value in enumerate(error.mean(0)):
            result[f"composition_mae_scale_{scale}_type_{cell_type}"] = float(value)
    result.update(composition_mae=float(np.mean(maes)), composition_mse=float(np.mean(mses)),
                  composition_jsd=float(np.mean(jsds)))
    if "section_id" in data:
        sections = np.asarray(data["section_id"])
        if sections.shape != (len(truth),):
            raise ValueError("niche section identifiers do not match centers")
        scores = []
        for section in np.unique(sections[mask.any(1)]):
            values = []
            for s in range(truth.shape[1]):
                take = (sections == section) & mask[:, s]
                if take.any():
                    values.append(float(np.abs(truth[take, s] - pred[take, s]).mean()))
            if values:
                scores.append(float(np.mean(values)))
                result[f"composition_mae_section_{section}"] = scores[-1]
        if scores:
            result["section_macro_mae"] = float(np.mean(scores))
    if "target_density" in data and "prediction_density" in data:
        from sklearn.metrics import r2_score
        density = _finite("target_density", data["target_density"], 2)
        estimate = _finite("prediction_density", data["prediction_density"], 2)
        valid = np.asarray(data.get("density_mask", mask), dtype=bool)
        if density.shape != mask.shape or estimate.shape != density.shape or valid.shape != density.shape:
            raise ValueError("niche density shape mismatch")
        if np.any(estimate < 0):
            raise ValueError("density predictions must be nonnegative")
        result["density_mae"] = float(np.mean([np.abs(density[valid[:, s], s] - estimate[valid[:, s], s]).mean()
                                                for s in range(density.shape[1]) if valid[:, s].any()]))
        r2 = [r2_score(density[valid[:, s], s], estimate[valid[:, s], s]) for s in range(density.shape[1])
              if valid[:, s].sum() > 1 and np.std(density[valid[:, s], s]) > 0]
        result["density_r2"] = float(np.mean(r2)) if r2 else math.nan
    return result


def spatial_niche(data: Mapping[str, np.ndarray]) -> dict[str, float]:
    if "target_composition" in data:
        return niche_composition(data)
    x = _finite("embedding", data["embedding"], 2).astype(float)
    labels = np.asarray(data["labels"])
    clusters = np.asarray(data["clusters"])
    if len(x) != len(labels) or labels.shape != clusters.shape:
        raise ValueError("spatial row counts differ")
    result = {"ari": float(adjusted_rand_score(labels, clusters)),
              "nmi": float(normalized_mutual_info_score(labels, clusters))}
    if "section_id" in data:
        sections = np.asarray(data["section_id"])
        if sections.shape != labels.shape:
            raise ValueError("spatial section identifiers do not match labels")
        per_section = []
        for section in np.unique(sections):
            take = sections == section
            if section == -1 or take.sum() < 2 or len(np.unique(labels[take])) < 2:
                continue
            per_section.append((adjusted_rand_score(labels[take], clusters[take]),
                                normalized_mutual_info_score(labels[take], clusters[take])))
        if per_section:
            result["section_mean_ari"], result["section_mean_nmi"] = map(float, np.mean(per_section, axis=0))
    # Cluster IDs have no semantic class meaning. Never score them as classes
    # or fit a cluster-to-region map using evaluation labels.
    if "y_pred" in data:
        result.update(annotation({"y_true": labels, "y_pred": data["y_pred"]}))
    return result


EVALUATORS = {"annotation": annotation, "integration": integration,
              "spatial_niche": spatial_niche, "spatial_completion": completion,
              "development": development}


def evaluate(task: str, data: Mapping[str, np.ndarray]) -> dict[str, float]:
    if task not in EVALUATORS: raise ValueError(f"unknown task: {task}")
    return EVALUATORS[task](data)
