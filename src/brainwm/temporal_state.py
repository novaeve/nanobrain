"""Train-only population-state operators for developmental forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def center_population(value: np.ndarray) -> np.ndarray:
    """Center a finite cell-by-feature population without changing its shape."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("population must have shape [cells, features]")
    if not np.isfinite(array).all():
        raise ValueError("population must be finite")
    return (array - array.mean(axis=0, keepdims=True)).astype(
        np.float32, copy=False
    )


def all_state_shrinkage_forecast(
    history: list[tuple[float, np.ndarray]],
    source_time: float,
    source: np.ndarray,
    target_time: float,
    *,
    window: int,
    anchor_weight: float,
    slope_scale: float,
) -> np.ndarray:
    """Forecast a population mean from train-observed state replicates.

    ``history`` may contain source and target populations from completed
    training intervals. The current source is added as the only observation
    supplied by the forecast interval. Replicates at a time point are averaged
    before fitting the temporal line.
    """
    if window < 2:
        raise ValueError("window must contain at least two time points")
    if not 0.0 <= float(anchor_weight) <= 1.0:
        raise ValueError("anchor_weight must lie in [0, 1]")
    if not 0.0 <= float(slope_scale) <= 2.0:
        raise ValueError("slope_scale must lie in [0, 2]")
    current = np.asarray(source, dtype=np.float32)
    if current.ndim != 2 or not np.isfinite(current).all():
        raise ValueError("source must be a finite [cells, genes] matrix")
    states = [(float(time), np.asarray(value, dtype=np.float32)) for time, value in history]
    states.append((float(source_time), current))
    if any(value.ndim != 2 or value.shape[1] != current.shape[1] for _, value in states):
        raise ValueError("all state populations must share the source gene width")
    times = sorted({time for time, _ in states})[-int(window) :]
    means = np.stack(
        [
            np.mean(
                [value.mean(axis=0) for time, value in states if time == selected],
                axis=0,
            )
            for selected in times
        ]
    )
    current_mean = current.mean(axis=0)
    if len(times) == 1:
        curve_at_source = current_mean
        slope = np.zeros_like(current_mean)
    else:
        time_array = np.asarray(times, dtype=np.float64)
        centered_time = time_array - time_array.mean()
        denominator = float(np.square(centered_time).sum())
        slope = (
            centered_time[:, None] * (means - means.mean(axis=0))
        ).sum(axis=0) / max(denominator, 1.0e-12)
        curve_at_source = means.mean(axis=0) + (
            float(source_time) - float(time_array.mean())
        ) * slope
    forecast_mean = (
        float(anchor_weight) * current_mean
        + (1.0 - float(anchor_weight)) * curve_at_source
        + (float(target_time) - float(source_time)) * float(slope_scale) * slope
    )
    return np.repeat(forecast_mean[None, :], current.shape[0], axis=0).astype(
        np.float32, copy=False
    )


def fit_reduced_rank_ridge(
    features: Tensor,
    targets: Tensor,
    *,
    l2: float,
    rank: int,
) -> dict[str, Any]:
    """Fit ridge regression through a train-fitted target SVD bottleneck."""
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must have shape [N, D]")
    if features.shape[0] != targets.shape[0] or features.shape[0] < 2:
        raise ValueError("features and targets need matching nontrivial rows")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("features and targets must be finite")
    if float(l2) <= 0.0:
        raise ValueError("l2 must be positive")
    maximum_rank = min(targets.shape)
    if not 1 <= int(rank) <= maximum_rank:
        raise ValueError(f"rank must lie in [1, {maximum_rank}]")

    x = features.float()
    y = targets.float()
    feature_mean = x.mean(dim=0)
    feature_scale = x.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    normalized = (x - feature_mean) / feature_scale
    target_mean = y.mean(dim=0)
    centered_target = y - target_mean
    _, _, vh = torch.linalg.svd(centered_target, full_matrices=False)
    basis = vh[: int(rank)]
    scores = centered_target @ basis.transpose(0, 1)
    if normalized.shape[1] > normalized.shape[0]:
        identity = torch.eye(
            normalized.shape[0], device=normalized.device, dtype=normalized.dtype
        )
        dual = torch.linalg.solve(
            normalized @ normalized.transpose(0, 1) + float(l2) * identity,
            scores,
        )
        weight = normalized.transpose(0, 1) @ dual
    else:
        identity = torch.eye(
            normalized.shape[1], device=normalized.device, dtype=normalized.dtype
        )
        weight = torch.linalg.solve(
            normalized.transpose(0, 1) @ normalized + float(l2) * identity,
            normalized.transpose(0, 1) @ scores,
        )
    return {
        "feature_mean": feature_mean.detach().cpu(),
        "feature_scale": feature_scale.detach().cpu(),
        "target_mean": target_mean.detach().cpu(),
        "basis": basis.detach().cpu(),
        "weight": weight.detach().cpu(),
        "l2": float(l2),
        "rank": int(rank),
    }


@torch.no_grad()
def apply_reduced_rank_ridge(head: dict[str, Any], features: Tensor) -> Tensor:
    """Apply a reduced-rank ridge head on the feature tensor's device."""
    required = {
        "feature_mean",
        "feature_scale",
        "target_mean",
        "basis",
        "weight",
        "l2",
        "rank",
    }
    if not required.issubset(head):
        raise ValueError("reduced-rank head is incomplete")
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    device = features.device
    mean = torch.as_tensor(head["feature_mean"], device=device).float()
    scale = torch.as_tensor(head["feature_scale"], device=device).float()
    target_mean = torch.as_tensor(head["target_mean"], device=device).float()
    basis = torch.as_tensor(head["basis"], device=device).float()
    weight = torch.as_tensor(head["weight"], device=device).float()
    if features.shape[1] != mean.numel() or weight.shape[0] != mean.numel():
        raise ValueError("feature width does not match reduced-rank head")
    scores = ((features.float() - mean) / scale) @ weight
    prediction = target_mean + scores @ basis
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("reduced-rank head produced non-finite values")
    return prediction


def compose_population_forecast(
    mean_forecast: np.ndarray,
    source: np.ndarray,
    learned_residual: np.ndarray | None,
    *,
    raw_scale: float,
    learned_scale: float,
    lower: float = 0.0,
    upper: float = 12.0,
) -> np.ndarray:
    """Combine population state, raw skip, and a learned centered residual."""
    mean = np.asarray(mean_forecast, dtype=np.float32)
    raw = np.asarray(source, dtype=np.float32)
    if mean.shape != raw.shape or mean.ndim != 2:
        raise ValueError("mean_forecast and source must share shape [cells, genes]")
    value = mean + float(raw_scale) * center_population(raw)
    if learned_residual is not None:
        residual = np.asarray(learned_residual, dtype=np.float32)
        if residual.shape != value.shape:
            raise ValueError("learned residual must match the population shape")
        value = value + float(learned_scale) * center_population(residual)
    if not np.isfinite(value).all():
        raise FloatingPointError("population forecast is non-finite")
    return np.clip(value, float(lower), float(upper)).astype(np.float32, copy=False)


@torch.no_grad()
def hybrid_transport_barycenter(
    source_expression: Tensor,
    target_expression: Tensor,
    source_latent: Tensor,
    target_latent: Tensor,
    *,
    expression_mean: Tensor,
    expression_scale: Tensor,
    latent_mean: Tensor,
    latent_scale: Tensor,
    latent_weight: float,
    epsilon: float = 0.05,
    iterations: int = 50,
) -> Tensor:
    """Build a target-expression barycenter from expression and BrainWM cost."""
    if source_expression.ndim != 2 or source_expression.shape != target_expression.shape:
        raise ValueError("expression populations must have equal [cells, genes] shape")
    if source_latent.ndim != 2 or source_latent.shape != target_latent.shape:
        raise ValueError("latent populations must have equal [cells, features] shape")
    if source_latent.shape[0] != source_expression.shape[0]:
        raise ValueError("expression and latent populations must share cell count")
    if not 0.0 <= float(latent_weight) <= 1.0:
        raise ValueError("latent_weight must lie in [0, 1]")
    if float(epsilon) <= 0.0 or int(iterations) < 1:
        raise ValueError("epsilon and iterations must be positive")
    tensors = (
        source_expression,
        target_expression,
        source_latent,
        target_latent,
        expression_mean,
        expression_scale,
        latent_mean,
        latent_scale,
    )
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("hybrid transport inputs must be finite")
    expression_mean = expression_mean.to(source_expression.device).float()
    expression_scale = expression_scale.to(source_expression.device).float().clamp_min(1.0e-5)
    latent_mean = latent_mean.to(source_expression.device).float()
    latent_scale = latent_scale.to(source_expression.device).float().clamp_min(1.0e-5)
    source_expression_metric = F.normalize(
        (source_expression.float() - expression_mean) / expression_scale, dim=-1
    )
    target_expression_metric = F.normalize(
        (target_expression.float() - expression_mean) / expression_scale, dim=-1
    )
    source_latent_metric = F.normalize(
        (source_latent.float() - latent_mean) / latent_scale, dim=-1
    )
    target_latent_metric = F.normalize(
        (target_latent.float() - latent_mean) / latent_scale, dim=-1
    )
    expression_cost = torch.cdist(
        source_expression_metric, target_expression_metric
    ).square()
    latent_cost = torch.cdist(source_latent_metric, target_latent_metric).square()
    cost = (
        (1.0 - float(latent_weight)) * expression_cost
        + float(latent_weight) * latent_cost
    )
    log_kernel = -cost / float(epsilon)
    cells = source_expression.shape[0]
    log_mass = -torch.log(
        source_expression.new_tensor(float(cells), dtype=torch.float32)
    )
    log_u = torch.zeros(cells, device=source_expression.device, dtype=torch.float32)
    log_v = torch.zeros_like(log_u)
    for _ in range(int(iterations)):
        log_u = log_mass - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_mass - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
    mass = plan.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (plan @ target_expression.float()) / mass


__all__ = [
    "all_state_shrinkage_forecast",
    "apply_reduced_rank_ridge",
    "center_population",
    "compose_population_forecast",
    "fit_reduced_rank_ridge",
    "hybrid_transport_barycenter",
]
