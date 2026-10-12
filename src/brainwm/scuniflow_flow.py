"""Endpoint-flow utilities for the controlled scUniFlow objective study.

These functions model masked features within one cell. They do not introduce
cell-cell spatial edges, biological-time transitions, or Stage2 dynamics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .feature_completion import masked_huber_loss

DEFAULT_FLOW_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_FLOW_BIN_PROBABILITIES = (0.40, 0.20, 0.15, 0.15, 0.10)


@dataclass(frozen=True)
class EmpiricalGaussianPrior:
    """Scalar Gaussian fitted to finite nonzero training values only."""

    mean: float
    std: float

    @classmethod
    def fit(
        cls, training_values: Tensor, *, minimum_std: float = 1.0e-6
    ) -> EmpiricalGaussianPrior:
        if not training_values.is_floating_point():
            raise TypeError("training_values must use a floating-point dtype")
        if minimum_std <= 0.0:
            raise ValueError("minimum_std must be positive")
        active = training_values[torch.isfinite(training_values) & training_values.ne(0)]
        if active.numel() < 2:
            raise ValueError("at least two finite nonzero training values are required")
        mean = float(active.double().mean().item())
        std = float(active.double().std(unbiased=False).item())
        if not torch.isfinite(torch.tensor([mean, std])).all():
            raise FloatingPointError("empirical prior statistics are non-finite")
        return cls(mean=mean, std=max(std, float(minimum_std)))

    def sample_like(self, reference: Tensor, *, generator: torch.Generator) -> Tensor:
        if not reference.is_floating_point():
            raise TypeError("reference must use a floating-point dtype")
        noise = torch.randn(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            generator=generator,
        )
        return noise.mul(self.std).add(self.mean)


def sample_stratified_flow_time(
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    bin_edges: Sequence[float] = DEFAULT_FLOW_BIN_EDGES,
    probabilities: Sequence[float] = DEFAULT_FLOW_BIN_PROBABILITIES,
) -> tuple[Tensor, Tensor]:
    """Sample one uniform flow time within a categorical time stratum per row."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    edges = torch.as_tensor(bin_edges, dtype=dtype, device=device)
    weights = torch.as_tensor(probabilities, dtype=torch.float64, device=device)
    if edges.ndim != 1 or weights.ndim != 1 or edges.numel() != weights.numel() + 1:
        raise ValueError("bin_edges must contain exactly one more value than probabilities")
    if not torch.all(edges[1:] > edges[:-1]) or float(edges[0]) < 0.0 or float(edges[-1]) > 1.0:
        raise ValueError("flow bins must be strictly increasing within [0,1]")
    if torch.any(weights < 0) or not torch.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError("flow-bin probabilities must be finite, nonnegative, and nonzero")
    weights = weights / weights.sum()
    bins = torch.multinomial(weights, batch_size, replacement=True, generator=generator)
    lower = edges.index_select(0, bins)
    upper = edges.index_select(0, bins + 1)
    uniform = torch.rand(batch_size, dtype=dtype, device=device, generator=generator)
    flow_time = lower + (upper - lower) * uniform
    return flow_time, bins


def interpolate_flow_state(x0: Tensor, endpoint: Tensor, flow_time: Tensor) -> Tensor:
    if x0.shape != endpoint.shape:
        raise ValueError("x0 and endpoint must have the same shape")
    if flow_time.ndim != 1 or flow_time.shape[0] != endpoint.shape[0]:
        raise ValueError("flow_time must have shape [batch]")
    if torch.any((flow_time < 0.0) | (flow_time > 1.0)):
        raise ValueError("flow_time must lie in [0,1]")
    tau = flow_time.to(endpoint).unsqueeze(-1)
    return (1.0 - tau) * x0 + tau * endpoint


def build_endpoint_flow_context(
    x0: Tensor,
    target: Tensor,
    masked: Tensor,
    flow_time: Tensor,
) -> tuple[Tensor, Tensor]:
    """Keep observed targets fixed and expose x_t only at masked positions."""

    if target.shape != x0.shape or masked.shape != target.shape:
        raise ValueError("x0, target, and masked must have the same shape")
    if masked.dtype != torch.bool:
        raise TypeError("masked must be boolean")
    x_t = interpolate_flow_state(x0, target, flow_time)
    return torch.where(masked, x_t, target), x_t


def endpoint_velocity(
    current: Tensor,
    predicted_endpoint: Tensor,
    flow_time: Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> Tensor:
    if current.shape != predicted_endpoint.shape:
        raise ValueError("current and predicted_endpoint must have the same shape")
    if flow_time.ndim != 1 or flow_time.shape[0] != current.shape[0]:
        raise ValueError("flow_time must have shape [batch]")
    if epsilon <= 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must lie in (0,1]")
    remaining = (1.0 - flow_time.to(current)).clamp_min(epsilon).unsqueeze(-1)
    velocity = (predicted_endpoint - current) / remaining
    if not torch.isfinite(velocity).all():
        raise FloatingPointError("endpoint-derived velocity is non-finite")
    return velocity


def _pearson_correlation(x: Tensor, y: Tensor) -> Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt(
        x_centered.square().sum() * y_centered.square().sum()
    ).clamp_min(1.0e-8)
    return (x_centered * y_centered).sum() / denominator


def masked_pearson_penalties(
    prediction: Tensor,
    target: Tensor,
    masked: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return global and mean-cell Pearson penalties on masked entries."""

    if prediction.shape != target.shape or masked.shape != target.shape:
        raise ValueError("prediction, target, and masked must have the same shape")
    if masked.dtype != torch.bool:
        raise TypeError("masked must be boolean")
    if int(masked.sum()) < 2:
        zero = prediction.sum() * 0.0
        return zero, zero
    global_penalty = 1.0 - _pearson_correlation(prediction[masked], target[masked])
    cell_penalties = []
    for row in range(target.shape[0]):
        active = masked[row]
        if int(active.sum()) >= 2:
            cell_penalties.append(
                1.0 - _pearson_correlation(prediction[row, active], target[row, active])
            )
    cell_penalty = (
        torch.stack(cell_penalties).mean()
        if cell_penalties
        else prediction.sum() * 0.0
    )
    return global_penalty, cell_penalty


def endpoint_flow_objective(
    predicted_endpoint: Tensor,
    target: Tensor,
    masked: Tensor,
    *,
    pearson_global_weight: float = 0.0,
    pearson_cell_weight: float = 0.0,
) -> dict[str, Tensor]:
    """Weighted endpoint Huber plus optional, separately attributable Pearson terms."""

    if pearson_global_weight < 0.0 or pearson_cell_weight < 0.0:
        raise ValueError("Pearson weights must be nonnegative")
    reconstruction = masked_huber_loss(
        predicted_endpoint,
        target,
        masked,
        delta=0.5,
        top_weight=3.0,
        top_k=20,
    )
    if pearson_global_weight == 0.0 and pearson_cell_weight == 0.0:
        global_penalty = predicted_endpoint.sum() * 0.0
        cell_penalty = predicted_endpoint.sum() * 0.0
    else:
        global_penalty, cell_penalty = masked_pearson_penalties(
            predicted_endpoint, target, masked
        )
    total = (
        reconstruction
        + float(pearson_global_weight) * global_penalty
        + float(pearson_cell_weight) * cell_penalty
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "pearson_global": global_penalty,
        "pearson_cell": cell_penalty,
    }


def euler_endpoint_rollout(
    predict_endpoint: Callable[[Tensor, Tensor], Tensor],
    target_context: Tensor,
    masked: Tensor,
    x0: Tensor,
    *,
    steps: int = 10,
    epsilon: float = 1.0e-3,
) -> Tensor:
    """Integrate an endpoint-predicted vector field while clamping observations."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if target_context.shape != masked.shape or x0.shape != target_context.shape:
        raise ValueError("target_context, masked, and x0 must have the same shape")
    if masked.dtype != torch.bool:
        raise TypeError("masked must be boolean")
    current = torch.where(masked, x0, target_context)
    observed = ~masked
    step_size = 1.0 / steps
    for step in range(steps):
        tau = current.new_full((current.shape[0],), step / steps)
        endpoint = predict_endpoint(current, observed)
        if endpoint.shape != current.shape or not torch.isfinite(endpoint).all():
            raise FloatingPointError("endpoint rollout prediction is invalid")
        velocity = endpoint_velocity(current, endpoint, tau, epsilon=epsilon)
        current = torch.where(masked, current + step_size * velocity, target_context)
    if not torch.isfinite(current).all():
        raise FloatingPointError("Euler rollout produced non-finite values")
    return current


__all__ = [
    "DEFAULT_FLOW_BIN_EDGES",
    "DEFAULT_FLOW_BIN_PROBABILITIES",
    "EmpiricalGaussianPrior",
    "build_endpoint_flow_context",
    "endpoint_flow_objective",
    "endpoint_velocity",
    "euler_endpoint_rollout",
    "interpolate_flow_state",
    "masked_pearson_penalties",
    "sample_stratified_flow_time",
]
