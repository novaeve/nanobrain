"""Project-native spatial feature-completion utilities.

The model in this module predicts globally held-out gene features from visible
features and a transparent section-local neighbor mean.  It is an auxiliary
baseline: the neighbor mean is not a learned cell graph, niche state, tissue
hierarchy, or masked-cell reconstruction model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn

SpatialContextMode = Literal["real", "permuted", "zero"]


def _validate_spatial_arrays(
    coordinates: np.ndarray, section_ids: Sequence[object]
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    sections = np.asarray(section_ids, dtype=object)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [cells, 2]")
    if sections.ndim != 1 or len(sections) != len(coordinates):
        raise ValueError("section_ids must have shape [cells]")
    if not len(coordinates):
        raise ValueError("spatial arrays must contain at least one cell")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    if any(value is None or not str(value).strip() for value in sections):
        raise ValueError("section_ids must be non-empty")
    return coordinates, sections.astype(str)


def section_knn_indices(
    coordinates: np.ndarray,
    section_ids: Sequence[object],
    *,
    k_neighbors: int,
    mode: SpatialContextMode = "real",
    seed: int = 0,
) -> np.ndarray:
    """Return directed kNN source rows without crossing section boundaries.

    ``permuted`` shuffles coordinate assignments within each section before
    rebuilding kNN.  It preserves section size and coordinate geometry while
    breaking the expression-location association.  ``zero`` returns an empty
    neighbor axis, allowing a capacity-matched model to receive zero context.
    """

    coordinates, sections = _validate_spatial_arrays(coordinates, section_ids)
    if not isinstance(k_neighbors, int) or isinstance(k_neighbors, bool):
        raise TypeError("k_neighbors must be an integer")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    if mode not in {"real", "permuted", "zero"}:
        raise ValueError(f"unsupported spatial context mode: {mode!r}")
    if mode == "zero":
        return np.empty((len(coordinates), 0), dtype=np.int64)

    output = np.empty((len(coordinates), k_neighbors), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for section in sorted(np.unique(sections).tolist()):
        rows = np.flatnonzero(sections == section)
        if len(rows) <= k_neighbors:
            raise ValueError(
                f"section {section!r} has {len(rows)} cells, but k={k_neighbors}"
            )
        local_coordinates = coordinates[rows].copy()
        if mode == "permuted":
            local_coordinates = local_coordinates[rng.permutation(len(rows))]
        tree = cKDTree(local_coordinates)
        # Query one extra point because every row must remove its self edge.
        _, local_neighbors = tree.query(
            local_coordinates, k=min(len(rows), k_neighbors + 1), workers=1
        )
        local_neighbors = np.asarray(local_neighbors, dtype=np.int64)
        if local_neighbors.ndim == 1:
            local_neighbors = local_neighbors[:, None]
        for local_row, candidates in enumerate(local_neighbors):
            candidates = candidates[candidates != local_row]
            if len(candidates) < k_neighbors:
                # Duplicate coordinates can make the self row absent from a
                # tied k+1 result.  Querying every row is the bounded fallback.
                _, expanded = tree.query(
                    local_coordinates[local_row], k=len(rows), workers=1
                )
                candidates = np.asarray(expanded, dtype=np.int64)
                candidates = candidates[candidates != local_row]
            output[rows[local_row]] = rows[candidates[:k_neighbors]]
    return output


def neighbor_mean_context(
    visible_values: np.ndarray, neighbor_indices: np.ndarray
) -> np.ndarray:
    """Average visible features over precomputed directed neighbors."""

    values = np.asarray(visible_values, dtype=np.float32)
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("visible_values must have shape [cells, features]")
    if neighbors.ndim != 2 or neighbors.shape[0] != values.shape[0]:
        raise ValueError("neighbor_indices must have shape [cells, k]")
    if neighbors.shape[1] == 0:
        return np.zeros_like(values)
    if np.any(neighbors < 0) or np.any(neighbors >= len(values)):
        raise ValueError("neighbor_indices contains an out-of-range row")
    return values[neighbors].mean(axis=1, dtype=np.float32)


def normalize_with_observed_library(
    raw_counts: np.ndarray,
    observed_gene_mask: np.ndarray,
    *,
    scale_factor: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize visible input and hidden truth without target-library leakage.

    The denominator contains observed genes only.  The returned target space
    uses that same denominator, so no hidden target count or target total is
    required at inference time.
    """

    raw = np.asarray(raw_counts, dtype=np.float32)
    observed = np.asarray(observed_gene_mask, dtype=bool)
    if raw.ndim != 2 or observed.shape != (raw.shape[1],):
        raise ValueError("raw_counts/observed_gene_mask shapes are incompatible")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("raw_counts must be finite and nonnegative")
    if not observed.any() or observed.all():
        raise ValueError("observed_gene_mask must keep and hide at least one gene")
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")

    observed_library = raw[:, observed].sum(axis=1, dtype=np.float64)
    denominator = np.maximum(observed_library, 1.0).astype(np.float32)
    normalized = np.log1p(raw * (scale_factor / denominator[:, None])).astype(
        np.float32
    )
    visible = normalized.copy()
    visible[:, ~observed] = 0.0
    return visible, normalized, observed_library.astype(np.float32)


class SpatialFeatureCompletionAdapter(nn.Module):
    """Fuse a cell-internal completer with a transparent neighbor mean."""

    def __init__(
        self,
        feature_backbone: nn.Module,
        n_features: int,
        *,
        context_hidden_dim: int = 128,
        context_gate_init: float = -2.0,
        n_context_scales: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(n_features, int) or n_features < 2:
            raise ValueError("n_features must be an integer greater than one")
        if not isinstance(context_hidden_dim, int) or context_hidden_dim < 1:
            raise ValueError("context_hidden_dim must be positive")
        if not math.isfinite(context_gate_init):
            raise ValueError("context_gate_init must be finite")
        if (
            not isinstance(n_context_scales, int)
            or isinstance(n_context_scales, bool)
            or n_context_scales < 1
        ):
            raise ValueError("n_context_scales must be a positive integer")
        self.feature_backbone = feature_backbone
        self.n_features = n_features
        self.n_context_scales = n_context_scales
        # Bias-free layers make the zero-context arm exactly cell-internal
        # while retaining identical trainable capacity across all five arms.
        self.context_projection = nn.Sequential(
            nn.Linear(n_features, context_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(context_hidden_dim, n_features, bias=False),
        )
        self.context_gate_logit = nn.Parameter(
            torch.full((n_features,), float(context_gate_init))
        )
        self.context_scale_logits = (
            nn.Parameter(torch.zeros(n_context_scales))
            if n_context_scales > 1
            else None
        )

    def normalized_context_scale_weights(self) -> Tensor:
        """Return normalized global weights over configured neighbor scales."""

        if self.context_scale_logits is None:
            return self.context_gate_logit.new_ones(1)
        return torch.softmax(self.context_scale_logits.float(), dim=0)

    def fuse_neighbor_context(
        self, neighbor_context: Tensor, observed: Tensor
    ) -> Tensor:
        """Mask hidden genes and fuse one or more context scales."""

        if neighbor_context.ndim == 2:
            if self.n_context_scales != 1 or neighbor_context.shape != observed.shape:
                raise ValueError(
                    "single-scale neighbor_context must match observed"
                )
            return neighbor_context.masked_fill(~observed, 0.0)
        expected = (observed.shape[0], self.n_context_scales, self.n_features)
        if neighbor_context.ndim != 3 or tuple(neighbor_context.shape) != expected:
            raise ValueError(
                "multi-scale neighbor_context must have shape "
                f"[B,{self.n_context_scales},{self.n_features}]"
            )
        safe = neighbor_context.masked_fill(~observed.unsqueeze(1), 0.0)
        weights = self.normalized_context_scale_weights().to(safe)
        return (safe * weights.view(1, -1, 1)).sum(dim=1)

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        neighbor_context: Tensor,
        order: Tensor | None = None,
        *,
        mode: str = "biological",
        seed: int | None = None,
    ) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(
                f"values must have shape [B,{self.n_features}]"
            )
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("observed must be boolean and match values")
        safe_context = self.fuse_neighbor_context(neighbor_context, observed)
        cell_prediction = self.feature_backbone(
            values,
            observed,
            order,
            mode=mode,
            seed=seed,
            preserve_unobserved_values=False,
        )
        context_delta = self.context_projection(safe_context)
        gate = torch.sigmoid(self.context_gate_logit).unsqueeze(0)
        return cell_prediction + gate * context_delta


def _pearson_loss(prediction: Tensor, target: Tensor, dim: int) -> Tensor:
    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(dim=dim, keepdim=True)
    target = target - target.mean(dim=dim, keepdim=True)
    numerator = (prediction * target).sum(dim=dim)
    denominator_squared = (
        prediction.square().sum(dim=dim) * target.square().sum(dim=dim)
    )
    valid = denominator_squared > 1.0e-16
    if not torch.any(valid):
        return prediction.sum() * 0.0
    denominator = torch.sqrt(denominator_squared.clamp_min(1.0e-16))
    correlation = numerator[valid] / denominator[valid]
    return (1.0 - correlation).mean()


def spatial_completion_loss(
    prediction: Tensor,
    target: Tensor,
    target_gene_index: Tensor,
    *,
    pearson_global_weight: float = 0.1,
    pearson_cell_weight: float = 0.1,
    sparsity_aware_weight: float = 0.0,
    nonzero_threshold: float = 0.0,
) -> dict[str, Tensor]:
    """Task-head loss over globally hidden target genes only."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [B,G]")
    if target_gene_index.ndim != 1 or not target_gene_index.numel():
        raise ValueError("target_gene_index must be a non-empty vector")
    if torch.any(target_gene_index < 0) or torch.any(
        target_gene_index >= prediction.shape[1]
    ):
        raise ValueError("target_gene_index contains an out-of-range gene")
    if pearson_global_weight < 0 or pearson_cell_weight < 0:
        raise ValueError("Pearson loss weights must be nonnegative")
    if not 0.0 <= sparsity_aware_weight <= 1.0:
        raise ValueError("sparsity_aware_weight must lie in [0,1]")
    if not math.isfinite(nonzero_threshold) or nonzero_threshold < 0.0:
        raise ValueError("nonzero_threshold must be finite and nonnegative")

    predicted = prediction.index_select(1, target_gene_index)
    truth = target.index_select(1, target_gene_index)
    squared_error = (predicted - truth).square()
    mse = torch.mean(squared_error)
    nonzero = truth > nonzero_threshold
    zero = ~nonzero
    nonzero_mse = squared_error[nonzero].mean() if torch.any(nonzero) else mse
    zero_mse = squared_error[zero].mean() if torch.any(zero) else mse
    balanced_mse = 0.5 * (nonzero_mse + zero_mse)
    reconstruction = (
        mse
        if sparsity_aware_weight == 0.0
        else (
            (1.0 - sparsity_aware_weight) * mse
            + sparsity_aware_weight * balanced_mse
        )
    )
    pearson_global = _pearson_loss(
        predicted.reshape(1, -1), truth.reshape(1, -1), dim=1
    )
    pearson_cell = _pearson_loss(predicted, truth, dim=1)
    total = (
        reconstruction
        + pearson_global_weight * pearson_global
        + pearson_cell_weight * pearson_cell
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "mse": mse,
        "balanced_mse": balanced_mse,
        "nonzero_mse": nonzero_mse,
        "zero_mse": zero_mse,
        "pearson_global": pearson_global,
        "pearson_cell": pearson_cell,
    }


def _pearson_numpy(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0


def completion_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_gene_index: np.ndarray,
) -> dict[str, float | int]:
    """Return fixed aggregate metrics in the observed-library logCP10K space."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    genes = np.asarray(target_gene_index, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [cells, genes]")
    if genes.ndim != 1 or not len(genes):
        raise ValueError("target_gene_index must be non-empty")
    predicted = prediction[:, genes]
    truth = target[:, genes]
    error = predicted - truth
    cell_pearson = [
        _pearson_numpy(predicted[row], truth[row])
        for row in range(len(predicted))
    ]
    gene_pearson = [
        _pearson_numpy(predicted[:, gene], truth[:, gene])
        for gene in range(predicted.shape[1])
    ]
    nonzero = truth > 0
    zero = ~nonzero
    return {
        "cells": len(predicted),
        "target_genes": len(genes),
        "entries": int(predicted.size),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "pearson_global": _pearson_numpy(predicted, truth),
        "pearson_cell_mean": float(np.mean(cell_pearson)),
        "pearson_gene_mean": float(np.mean(gene_pearson)),
        "target_nonzero_fraction": float(nonzero.mean()),
        "nonzero_rmse": (
            float(np.sqrt(np.mean(np.square(error[nonzero]))))
            if np.any(nonzero)
            else 0.0
        ),
        "zero_rmse": (
            float(np.sqrt(np.mean(np.square(error[zero]))))
            if np.any(zero)
            else 0.0
        ),
    }


__all__ = [
    "SpatialContextMode",
    "SpatialFeatureCompletionAdapter",
    "completion_metrics",
    "neighbor_mean_context",
    "normalize_with_observed_library",
    "section_knn_indices",
    "spatial_completion_loss",
]
