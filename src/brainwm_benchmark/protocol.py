"""Frozen metric definitions for the independent four-task protocol."""
from __future__ import annotations

METRIC_PROTOCOL_ID = "NANOBRAIN-METRICS-V3-20261102"
REPORTING_REVISION = "standard-task-metrics-20261103"
EVALUATION_SEED = 20260915
DEVELOPMENT_EVAL_CELLS = 256
SINKHORN_BLUR = 0.05
SINKHORN_SCALING = 0.9
SCIB_MAX_CELLS = 5000
SCIB_NEIGHBORS = 90
SCIB_VERSION = "0.6.0"

# Scalar objectives used internally for checkpoint selection and search.
# The integration aggregate is a project utility, not an official scIB total.
SELECTION_METRICS = (
    ("annotation", "f1_macro", "max"),
    ("integration", "scib_composite", "max"),
    ("spatial_niche", "composition_mae", "min"),
    ("spatial_completion", "heldout_rmse", "min"),
    ("development", "wasserstein2", "min"),
)

# Backward-compatible name used by the existing search/gating code only.
PRIMARY_METRICS = SELECTION_METRICS

INTEGRATION_REPORT_METRICS = (
    ("scib_label_asw", "Label ASW", "max"),
    ("scib_isolated_label_asw", "Isolated-label ASW", "max"),
    ("scib_nmi", "KMeans NMI", "max"),
    ("scib_ari", "KMeans ARI", "max"),
    ("scib_batch_asw", "Batch ASW within cell type", "max"),
    ("scib_ilisi", "iLISI", "max"),
    ("scib_graph_connectivity", "Graph connectivity", "max"),
)

# Formal task results use established individual metrics, never the search sum.
BENCHMARK_METRICS = (
    ("annotation", "f1_macro", "max"),
    *(("integration", metric, direction) for metric, _, direction in INTEGRATION_REPORT_METRICS),
    ("spatial_niche", "composition_mae", "min"),
    ("development", "wasserstein2", "min"),
)

METRIC_REFERENCES = {
    "annotation": "https://www.nature.com/articles/s41467-024-51059-5",
    "integration": "https://www.nature.com/articles/s41592-021-01336-8",
    "spatial_niche": "https://www.nature.com/articles/s41592-025-02814-z",
    "development": "https://doi.org/10.1016/j.cell.2019.01.006",
}


def metric_role(task: str, metric: str) -> str:
    if any(task == t and metric == m for t, m, _ in BENCHMARK_METRICS):
        return "benchmark"
    if task == "integration" and metric in {"scib_composite", "biology_conservation", "batch_removal"}:
        return "selection_only"
    return "diagnostic"


def metric_protocol() -> dict:
    return {
        "id": METRIC_PROTOCOL_ID,
        "reporting_revision": REPORTING_REVISION,
        "primary_metrics": [list(x) for x in BENCHMARK_METRICS],
        "selection_metrics": [list(x) for x in SELECTION_METRICS],
        "references": dict(METRIC_REFERENCES),
        "selection_policy": "project utilities select validation checkpoints and architectures; they are not scientific benchmark metrics",
        "classification": "macro F1 over ground-truth-supported evaluation classes; unknown truths excluded upstream",
        "spatial_task": {
            "target": "true neighborhood cell-type count proportions, not anatomy",
            "input": "center expression only; target-neighbor context is hidden",
            "scales": [10, 20, 50, 100],
            "radii": "per-source fixed radii calibrated from full training sections only",
            "normalization": "counts / total neighbors; unknown types retain an explicit bin",
            "loss": "composition MSE; separate density probe on detached embeddings with log1p MSE (default weight 0.1)",
            "aggregation": "equal mean over four radii; per-type and per-section diagnostics",
        },
        "integration": {
            "implementation": f"scib-metrics=={SCIB_VERSION}",
            "biology": ["silhouette_label", "isolated_labels_ASW", "kmeans_NMI", "kmeans_ARI"],
            "batch": ["silhouette_batch_within_label", "ilisi_knn", "graph_connectivity"],
            "weights": [0.6, 0.4],
            "aggregate_role": "selection_only; selected subset, not an official scIB total",
            "ari_zero_floor": "internal aggregate only; benchmark ARI remains unmodified",
            "max_cells": SCIB_MAX_CELLS,
            "sampling": "fixed label-by-batch stratified sample; mandatory one per stratum",
            "neighbors": SCIB_NEIGHBORS,
            "seed": EVALUATION_SEED,
            "cross_method_minmax_scaling": False,
            "robustness_diagnostics": ["BRAS", "cell-type-conditional iLISI"],
            "note": "fixed subset of official scIB metrics, not the entire original 14-metric suite",
        },
        "development": {
            "implementation": "exact uniform-mass OT; scipy assignment for equal populations; POT for unequal populations",
            "p": 2,
            "cost": "squared Euclidean distance; square root after transport averaging",
            "space": "shared train-normalized gene panel; never a model-specific latent space",
            "max_cells_per_population": DEVELOPMENT_EVAL_CELLS,
            "seed": EVALUATION_SEED,
            "aggregation": "unweighted mean across source-target-time/anatomy population groups",
            "note": "exact empirical W2 in the common gene panel; Sinkhorn remains a separately named diagnostic",
        },
    }
