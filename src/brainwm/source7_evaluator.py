"""Canonical population evaluator for the source-7 internal validation.

The evaluator is intentionally independent of the source-23 runners.  Those
runners produced historical results with a 50-iteration Sinkhorn helper,
whereas ``EVAL-S7-POPULATION-V1`` freezes 30 iterations.  Keeping the source-7
implementation separate prevents a retrospective metric change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .losses import barycentric_transport_target, sinkhorn_transport_plan


EVALUATOR_ID = "EVAL-S7-POPULATION-V1"
CONTRACT_ID = "INT-S7-ROW-DISJOINT-V1"
RECEIPT_FORMAT = "brainwm_source7_population_evaluation_v1"
# Verbose aliases make contract bindings self-documenting in route launchers.
SOURCE7_EVALUATOR_ID = EVALUATOR_ID
SOURCE7_CONTRACT_ID = CONTRACT_ID
PRIMARY_METRIC = "population_mse"
SECONDARY_METRICS = ("sinkhorn_cost", "ot_aligned_cosine_distance")
SINKHORN_EPSILON = 0.05
SINKHORN_ITERATIONS = 30


@dataclass(frozen=True)
class Source7PopulationMetrics:
    """JSON-compatible scalar receipt for one source-7 prediction route."""

    evaluator_id: str
    population_mse: float
    sinkhorn_cost: float
    ot_aligned_cosine_distance: float
    reward: float
    sinkhorn_epsilon: float
    sinkhorn_iterations: int
    source_rows: int
    target_rows: int
    feature_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_source7_receipt(
    metrics: Source7PopulationMetrics,
    *,
    method_id: str,
    seed: int,
    checkpoint_step: int,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one unregistered, contract-bound metric receipt.

    The receipt is deliberately separate from the official benchmark JSONL:
    source7 remains an in-domain internal-validation contract until a later
    review freezes a registry entry.  ``diagnostics`` is copied verbatim and
    cannot change any metric or access field.
    """

    if not isinstance(method_id, str) or not method_id:
        raise ValueError("method_id must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(checkpoint_step, int)
        or isinstance(checkpoint_step, bool)
        or checkpoint_step < 0
    ):
        raise ValueError("checkpoint_step must be a non-negative integer")
    if metrics.evaluator_id != EVALUATOR_ID:
        raise ValueError("metrics are bound to a different evaluator")
    if (
        metrics.sinkhorn_epsilon != SINKHORN_EPSILON
        or metrics.sinkhorn_iterations != SINKHORN_ITERATIONS
    ):
        raise ValueError("metrics use a different source-7 Sinkhorn configuration")
    dimensions = (
        metrics.source_rows,
        metrics.target_rows,
        metrics.feature_count,
    )
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        for value in dimensions
    ):
        raise ValueError("source-7 population dimensions must be positive integers")
    scalar_metrics = (
        metrics.population_mse,
        metrics.sinkhorn_cost,
        metrics.ot_aligned_cosine_distance,
        metrics.reward,
    )
    if not all(np.isfinite(float(value)) for value in scalar_metrics):
        raise ValueError("source-7 receipt metrics must be finite")
    if any(float(value) < 0.0 for value in scalar_metrics[:3]):
        raise ValueError("source-7 distance metrics must be non-negative")
    if not np.isclose(
        float(metrics.reward),
        -float(metrics.population_mse),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("source-7 reward must equal negative population_mse")
    payload: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "contract_id": CONTRACT_ID,
        "evaluator_id": EVALUATOR_ID,
        "claim_scope": "in_domain_internal_validation",
        "method_id": method_id,
        "seed": seed,
        "checkpoint_step": checkpoint_step,
        "metrics": {
            "population_mse": metrics.population_mse,
            "sinkhorn_cost": metrics.sinkhorn_cost,
            "ot_aligned_cosine_distance": metrics.ot_aligned_cosine_distance,
            "reward": metrics.reward,
        },
        "evaluator": {
            "balanced": True,
            "epsilon": SINKHORN_EPSILON,
            "iterations": SINKHORN_ITERATIONS,
            "target_sample_size": 256,
        },
        "population": {
            "source_rows": metrics.source_rows,
            "target_rows": metrics.target_rows,
            "feature_count": metrics.feature_count,
        },
        "access": {
            "validation_only": True,
            "query_opened": False,
            "test_opened": False,
            "target_used_for_fit": False,
        },
        "diagnostics": dict(diagnostics or {}),
    }
    return payload


def write_source7_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Write one receipt exclusively and refuse to overwrite existing files."""

    if receipt.get("format") != RECEIPT_FORMAT:
        raise ValueError("receipt format is not the source-7 canonical format")
    if receipt.get("contract_id") != CONTRACT_ID:
        raise ValueError("receipt is bound to a different source-7 contract")
    if receipt.get("evaluator_id") != EVALUATOR_ID:
        raise ValueError("receipt is bound to a different source-7 evaluator")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(dict(receipt), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target


def _finite_matrix(name: str, value: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty shape [cells, features]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor.contiguous()


def indexed_batch_to_log_expression(
    indexed: Mapping[str, Any],
    panel_gene_ids: np.ndarray | torch.Tensor,
    *,
    scale_factor: float = 1.0e4,
) -> np.ndarray:
    """Project one ``IndexedH5ADBatchReader`` batch to a dense log panel.

    ``indexed`` is the reader's raw-count dictionary (``row_ptr``,
    ``gene_index``, ``count`` and ``library_size``).  The conversion matches
    the frozen source7 normalization: panel counts are divided by each cell's
    full measured library size, multiplied by ``1e4``, and transformed with
    ``log1p``.  Gene IDs need not be sorted and duplicate sparse tokens are
    coalesced by accumulation.  No expression outside the supplied batch is
    read.
    """

    if not isinstance(indexed, Mapping):
        raise TypeError("indexed batch must be a mapping")
    required = ("row_ptr", "gene_index", "count", "library_size")
    missing = [name for name in required if name not in indexed]
    if missing:
        raise ValueError("indexed batch is missing: " + ", ".join(missing))
    panel = np.asarray(panel_gene_ids, dtype=np.int64)
    if panel.ndim != 1 or panel.size < 1 or np.unique(panel).size != panel.size:
        raise ValueError("panel_gene_ids must be a non-empty unique vector")
    if not np.isfinite(float(scale_factor)) or float(scale_factor) <= 0.0:
        raise ValueError("scale_factor must be finite and positive")

    row_ptr = np.asarray(indexed["row_ptr"], dtype=np.int64)
    genes = np.asarray(indexed["gene_index"], dtype=np.int64)
    counts = np.asarray(indexed["count"], dtype=np.float32)
    library = np.asarray(indexed["library_size"], dtype=np.float32)
    if row_ptr.ndim != 1 or row_ptr.size < 2:
        raise ValueError("indexed row_ptr must have shape [cells + 1]")
    if genes.ndim != 1 or counts.ndim != 1 or genes.size != counts.size:
        raise ValueError("indexed gene_index/count must be equal-length vectors")
    if (
        int(row_ptr[0]) != 0
        or int(row_ptr[-1]) != genes.size
        or np.any(row_ptr[1:] < row_ptr[:-1])
    ):
        raise ValueError("indexed row_ptr is not a valid sparse partition")
    if library.shape != (row_ptr.size - 1,):
        raise ValueError("indexed library_size does not match row_ptr")
    if np.any(genes < 0) or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("indexed genes/counts are invalid")
    if not np.isfinite(library).all() or np.any(library <= 0.0):
        raise ValueError("indexed library_size must be finite and positive")

    # Search against a sorted copy so the panel's declared gene order is
    # preserved in the returned matrix without allocating a vocabulary-sized
    # lookup array.
    order = np.argsort(panel, kind="stable")
    sorted_panel = panel[order]
    locations = np.searchsorted(sorted_panel, genes, side="left")
    valid = locations < sorted_panel.size
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        valid &= sorted_panel[np.minimum(locations, sorted_panel.size - 1)] == genes
    rows = np.repeat(np.arange(row_ptr.size - 1, dtype=np.int64), np.diff(row_ptr))
    dense = np.zeros((row_ptr.size - 1, panel.size), dtype=np.float32)
    if valid_indices.size:
        columns = order[locations[valid]]
        np.add.at(dense, (rows[valid], columns), counts[valid])
    return np.log1p(dense * (float(scale_factor) / library[:, None])).astype(
        np.float32, copy=False
    )


def evaluate_source7_indexed_population(
    prediction: np.ndarray | torch.Tensor,
    source_indexed: Mapping[str, Any],
    target_indexed: Mapping[str, Any],
    panel_gene_ids: np.ndarray | torch.Tensor,
    *,
    metric_mean: np.ndarray | torch.Tensor | None = None,
    metric_scale: np.ndarray | torch.Tensor | None = None,
    scale_factor: float = 1.0e4,
) -> Source7PopulationMetrics:
    """Evaluate a prediction against two raw indexed batches.

    The prediction must already be in the panel's log-expression space (the
    same space returned by :func:`indexed_batch_to_log_expression`).  This
    keeps model decoding explicit and prevents silently treating a latent
    vector as a gene-expression vector.  Source and target are projected from
    their indexed raw counts using only the supplied panel and batch rows.
    """

    source = indexed_batch_to_log_expression(
        source_indexed, panel_gene_ids, scale_factor=scale_factor
    )
    target = indexed_batch_to_log_expression(
        target_indexed, panel_gene_ids, scale_factor=scale_factor
    )
    prediction_tensor = _finite_matrix("prediction", prediction)
    if prediction_tensor.shape != (source.shape[0], source.shape[1]):
        raise ValueError(
            "prediction must have shape [source_cells, panel_genes]"
        )
    return evaluate_source7_population(
        prediction_tensor,
        source,
        target,
        metric_mean=metric_mean,
        metric_scale=metric_scale,
    )


def _metric_assets(
    feature_count: int,
    mean: np.ndarray | torch.Tensor | None,
    scale: np.ndarray | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (mean is None) != (scale is None):
        raise ValueError("metric mean and scale must be provided together")
    if mean is None:
        return torch.zeros(feature_count), torch.ones(feature_count)
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device="cpu")
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device="cpu")
    expected = (feature_count,)
    if mean_tensor.shape != expected or scale_tensor.shape != expected:
        raise ValueError(
            f"metric mean and scale must each have shape [{feature_count}]"
        )
    if not bool(torch.isfinite(mean_tensor).all()) or not bool(
        torch.isfinite(scale_tensor).all()
    ):
        raise ValueError("metric mean and scale must be finite")
    if not bool(torch.all(scale_tensor > 0.0)):
        raise ValueError("metric scale must be strictly positive")
    return mean_tensor, scale_tensor


def _balanced_plan(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Return uniform balanced OT, including rectangular populations.

    The project-level Sinkhorn helper requires equal row counts because all
    frozen source-7 cohorts contain 256 cells per side.  The small-array API is
    slightly more general so it remains useful for unit tests and bounded
    diagnostics without changing the equal-size source-7 contract.
    """

    if source.shape[0] == target.shape[0]:
        return sinkhorn_transport_plan(
            source, target, epsilon=epsilon, iterations=iterations
        )

    first = F.normalize(source.float(), dim=-1)
    second = F.normalize(target.float(), dim=-1)
    cost = torch.cdist(first, second).square()
    log_kernel = -cost / epsilon
    source_mass = -torch.log(torch.tensor(float(source.shape[0])))
    target_mass = -torch.log(torch.tensor(float(target.shape[0])))
    log_u = torch.zeros(source.shape[0], dtype=torch.float32)
    log_v = torch.zeros(target.shape[0], dtype=torch.float32)
    for _ in range(iterations):
        log_u = source_mass - torch.logsumexp(
            log_kernel + log_v[None, :], dim=1
        )
        log_v = target_mass - torch.logsumexp(
            log_kernel + log_u[:, None], dim=0
        )
    return torch.exp(log_kernel + log_u[:, None] + log_v[None, :])


@torch.inference_mode()
def evaluate_source7_population(
    prediction: np.ndarray | torch.Tensor,
    source: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    metric_mean: np.ndarray | torch.Tensor | None = None,
    metric_scale: np.ndarray | torch.Tensor | None = None,
) -> Source7PopulationMetrics:
    """Evaluate one cross-sectional source-7 population prediction.

    ``prediction`` and ``source`` must have the same row identity and shape.
    ``target`` may have a different number of cells but must share the feature
    axis.  If train-fitted ``metric_mean`` and ``metric_scale`` are supplied,
    the OT plan, MSE, and Sinkhorn cost are computed in that standardized
    space.  The cosine diagnostic remains in the unstandardized feature space,
    matching the established Stage-1 evaluator semantics.

    The primary target is the source-indexed barycenter induced by a balanced
    source-to-target OT plan.  ``sinkhorn_cost`` instead measures the predicted
    population against the observed target population with its own balanced
    plan.  ``ot_aligned_cosine_distance`` compares each prediction row to the
    source-to-target barycentric row.  None of these quantities implies cell
    lineage correspondence.
    """

    predicted = _finite_matrix("prediction", prediction)
    source_tensor = _finite_matrix("source", source)
    target_tensor = _finite_matrix("target", target)
    if predicted.shape != source_tensor.shape:
        raise ValueError("prediction and source must share shape [cells, features]")
    if target_tensor.shape[1] != source_tensor.shape[1]:
        raise ValueError("source and target must share the feature axis")

    mean, scale = _metric_assets(
        int(source_tensor.shape[1]), metric_mean, metric_scale
    )
    source_metric = (source_tensor - mean) / scale
    target_metric = (target_tensor - mean) / scale
    predicted_metric = (predicted - mean) / scale

    source_target_plan = _balanced_plan(
        source_metric,
        target_metric,
        epsilon=SINKHORN_EPSILON,
        iterations=SINKHORN_ITERATIONS,
    )
    barycentric_metric = barycentric_transport_target(
        source_target_plan, target_metric
    )
    barycentric_features = barycentric_transport_target(
        source_target_plan, target_tensor
    )

    predicted_target_plan = _balanced_plan(
        predicted_metric,
        target_metric,
        epsilon=SINKHORN_EPSILON,
        iterations=SINKHORN_ITERATIONS,
    )
    predicted_unit = F.normalize(predicted_metric, dim=-1)
    target_unit = F.normalize(target_metric, dim=-1)
    predicted_target_cost = torch.cdist(predicted_unit, target_unit).square()

    population_mse = float(F.mse_loss(predicted_metric, barycentric_metric))
    sinkhorn_cost = float((predicted_target_plan * predicted_target_cost).sum())
    ot_cosine = float(
        1.0
        - F.cosine_similarity(predicted, barycentric_features, dim=-1).mean()
    )
    scalars = (population_mse, sinkhorn_cost, ot_cosine)
    if not all(np.isfinite(value) for value in scalars):
        raise FloatingPointError("source-7 evaluator produced a non-finite metric")

    return Source7PopulationMetrics(
        evaluator_id=EVALUATOR_ID,
        population_mse=population_mse,
        sinkhorn_cost=sinkhorn_cost,
        ot_aligned_cosine_distance=ot_cosine,
        reward=-population_mse,
        sinkhorn_epsilon=SINKHORN_EPSILON,
        sinkhorn_iterations=SINKHORN_ITERATIONS,
        source_rows=int(source_tensor.shape[0]),
        target_rows=int(target_tensor.shape[0]),
        feature_count=int(source_tensor.shape[1]),
    )


def metric_selection_key(
    metrics: Source7PopulationMetrics | dict[str, Any],
    *,
    step: int = 0,
) -> tuple[float, float, float, int]:
    """Return the contract's deterministic checkpoint ordering key."""

    values = metrics.as_dict() if isinstance(metrics, Source7PopulationMetrics) else metrics
    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    key = tuple(
        float(values[name])
        for name in (PRIMARY_METRIC, *SECONDARY_METRICS)
    )
    if not all(np.isfinite(value) for value in key):
        raise ValueError("selection metrics must be finite")
    return key[0], key[1], key[2], step
