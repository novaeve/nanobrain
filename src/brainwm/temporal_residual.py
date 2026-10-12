"""Small train-only heads for population-baseline temporal residuals."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def fit_ridge_head(features: Tensor, targets: Tensor, l2: float) -> dict[str, Any]:
    """Fit a multi-output ridge head with train-fitted feature scaling."""
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must have shape [N, D]")
    if features.shape[0] != targets.shape[0] or features.shape[0] < 2:
        raise ValueError("features and targets need the same nontrivial row count")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("ridge inputs must be finite")
    if not float(l2) > 0.0:
        raise ValueError("l2 must be positive")

    x = features.float()
    y = targets.float()
    feature_mean = x.mean(dim=0)
    feature_scale = x.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    normalized = (x - feature_mean) / feature_scale
    target_mean = y.mean(dim=0)
    centered_target = y - target_mean
    gram = normalized.transpose(0, 1) @ normalized
    gram.diagonal().add_(float(l2))
    weight = torch.linalg.solve(
        gram, normalized.transpose(0, 1) @ centered_target
    )
    return {
        "feature_mean": feature_mean.detach().cpu(),
        "feature_scale": feature_scale.detach().cpu(),
        "target_mean": target_mean.detach().cpu(),
        "weight": weight.detach().cpu(),
        "l2": float(l2),
    }


def apply_ridge_head(head: dict[str, Any], features: Tensor) -> Tensor:
    """Apply a frozen ridge head on the feature tensor's device."""
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    required = {"feature_mean", "feature_scale", "target_mean", "weight", "l2"}
    if not required.issubset(head):
        raise ValueError("ridge head is incomplete")
    mean = torch.as_tensor(head["feature_mean"], device=features.device).float()
    scale = torch.as_tensor(head["feature_scale"], device=features.device).float()
    target_mean = torch.as_tensor(head["target_mean"], device=features.device).float()
    weight = torch.as_tensor(head["weight"], device=features.device).float()
    if features.shape[1] != mean.numel() or weight.shape[0] != mean.numel():
        raise ValueError("feature width does not match ridge head")
    normalized = (features.float() - mean) / scale
    prediction = target_mean + normalized @ weight
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("ridge head produced non-finite predictions")
    return prediction


def add_temporal_residual(
    population_baseline: Tensor,
    residual: Tensor,
    scale: float,
    *,
    lower: float = 0.0,
    upper: float = 12.0,
) -> Tensor:
    """Add a scaled residual to a population forecast in expression space."""
    if population_baseline.shape != residual.shape or population_baseline.ndim != 2:
        raise ValueError("baseline and residual must share shape [N, G]")
    if not float(scale) >= 0.0:
        raise ValueError("scale must be nonnegative")
    if not float(upper) > float(lower):
        raise ValueError("upper must exceed lower")
    prediction = population_baseline.float() + float(scale) * residual.float()
    return prediction.clamp(min=float(lower), max=float(upper))


__all__ = ["add_temporal_residual", "apply_ridge_head", "fit_ridge_head"]
