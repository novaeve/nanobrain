r"""A small scFiT-style endpoint adapter for BrainWM latent dynamics.

This module is intentionally independent of :class:`BrainWorldModel` and the
existing Stage-1 trainer.  It provides a candidate parameterization that can be
added behind the current latent interface after a controlled ablation.

The adapter consumes an intermediate flow state ``z_tau`` with shape ``[B, D]``
and three time conditions:

* ``flow_time`` (``tau``), normalized to ``[0, 1]``;
* ``source_time`` and ``target_time``, in the caller's biological time unit
  (BrainWM uses days by default).

It always returns a predicted target endpoint with shape ``[B, D]``.  In
``endpoint`` mode the network predicts the endpoint directly and the velocity
is derived as

.. math::

   \hat v = (\hat z_t - z_\tau) / \max(1 - \tau, \epsilon).

The paper writes this stabilization as ``1 - tau + epsilon``; the maintained
scFiT code uses ``clamp_min(epsilon)``, which is the convention implemented
here.

In ``direct_velocity`` mode the network predicts ``\hat v`` and the endpoint is
recovered as ``z_tau + d_tau * \hat v`` with
``d_tau = clamp(1 - tau, min=epsilon)``.  The stabilized denominator mirrors
the maintained scFiT implementation and avoids an endpoint singularity.

The current BrainWM Stage-1 contract uses ``z_initial``/``target_latent`` with
shape ``[B, 256]``.  Instantiating ``LatentEndpointAdapter(latent_dim=256)``
therefore maps directly to that contract; no gene-token reinterpretation is
performed here.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import MLP


BRAINWM_STAGE1_LATENT_DIM: Final[int] = 256
_PARAMETERIZATIONS = frozenset({"endpoint", "direct_velocity"})


class _SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding that accepts arbitrary scalar biological units."""

    def __init__(self, dim: int, scale: float) -> None:
        super().__init__()
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("time embedding dimension must be a positive integer")
        if not math.isfinite(float(scale)) or scale <= 0.0:
            raise ValueError("time embedding scale must be finite and positive")
        self.dim = int(dim)
        self.scale = float(scale)

    def forward(self, values: Tensor) -> Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=values.device, dtype=values.dtype)
            / max(half - 1, 1)
        )
        angles = (values / self.scale).unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2:
            # ``embedding[:, :1]`` is empty when dim=1 (half=0), so construct
            # the odd-coordinate padding explicitly.
            padding = torch.zeros(
                (embedding.shape[0], 1), device=embedding.device, dtype=embedding.dtype
            )
            embedding = torch.cat((embedding, padding), dim=-1)
        return embedding


def _validate_scalar_hyperparameter(name: str, value: float, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _as_batch_scalar(value: Tensor | float, name: str, batch_size: int, device: torch.device,
                     dtype: torch.dtype) -> Tensor:
    """Normalize a scalar time condition to a finite ``[B]`` tensor."""

    if value is None:
        raise ValueError(f"{name} is required")
    if not isinstance(value, Tensor):
        try:
            value = torch.as_tensor(value, device=device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    else:
        value = value.to(device=device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.ndim == 1:
        if value.shape[0] != batch_size:
            raise ValueError(
                f"{name} must be a scalar, [B], or [B, 1] tensor; got {tuple(value.shape)} "
                f"for B={batch_size}"
            )
    elif value.ndim == 2 and value.shape == (batch_size, 1):
        value = value[:, 0]
    else:
        raise ValueError(
            f"{name} must be a scalar, [B], or [B, 1] tensor; got {tuple(value.shape)} "
            f"for B={batch_size}"
        )
    value = value.to(dtype=dtype)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_latent_pair(first: Tensor, second: Tensor, first_name: str, second_name: str) -> None:
    if not isinstance(first, Tensor) or not isinstance(second, Tensor):
        raise ValueError(f"{first_name} and {second_name} must be tensors")
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(f"{first_name} and {second_name} must have shape [B, D]")
    if first.shape != second.shape or first.shape[0] < 1 or first.shape[1] < 1:
        raise ValueError(
            f"{first_name} and {second_name} must have the same non-empty shape [B, D]"
        )
    if first.device != second.device:
        raise ValueError(f"{first_name} and {second_name} must be on the same device")
    if not torch.is_floating_point(first) or not torch.is_floating_point(second):
        raise ValueError(f"{first_name} and {second_name} must use a floating dtype")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError(f"{first_name} and {second_name} must contain only finite values")


def endpoint_velocity_from_endpoint(
    z_tau: Tensor,
    endpoint: Tensor,
    flow_time: Tensor | float,
    flow_eps: float = 1.0e-3,
) -> Tensor:
    """Derive the scFiT target-state velocity from an endpoint prediction.

    ``flow_time`` is normalized to ``[0, 1]``.  The denominator is clamped to
    ``flow_eps`` near the endpoint to avoid division by zero.
    """

    _validate_latent_pair(z_tau, endpoint, "z_tau", "endpoint")
    eps = _validate_scalar_hyperparameter("flow_eps", flow_eps, positive=True)
    if eps > 1.0:
        raise ValueError("flow_eps must not exceed 1")
    tau = _as_batch_scalar(
        flow_time, "flow_time", z_tau.shape[0], z_tau.device, z_tau.dtype
    )
    if torch.any((tau < 0.0) | (tau > 1.0)):
        raise ValueError("flow_time must lie in [0, 1]")
    denominator = (1.0 - tau).clamp_min(eps).unsqueeze(-1)
    velocity = (endpoint - z_tau) / denominator
    if not torch.isfinite(velocity).all():
        raise FloatingPointError("derived endpoint velocity is non-finite")
    return velocity


class LatentEndpointAdapter(nn.Module):
    """Predict a target latent endpoint from an intermediate flow state.

    Parameters are deliberately small and explicit so this class can be used as
    a Stage-1 ablation without changing the existing ``BrainWorldModel``.
    ``source_time`` and ``target_time`` may be reversed; this mirrors scFiT's
    bidirectional interpolation protocol.  A caller enforcing causal Stage-1
    forecasting should validate ``target_time > source_time`` at the data layer.
    """

    def __init__(
        self,
        latent_dim: int = BRAINWM_STAGE1_LATENT_DIM,
        hidden_dim: int = 256,
        time_dim: int = 32,
        layers: int = 2,
        dropout: float = 0.0,
        parameterization: str = "endpoint",
        flow_eps: float = 1.0e-3,
        biological_time_scale: float = 365.2425,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("hidden_dim", hidden_dim),
            ("time_dim", time_dim),
            ("layers", layers),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        dropout = _validate_scalar_hyperparameter("dropout", dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        flow_eps = _validate_scalar_hyperparameter("flow_eps", flow_eps, positive=True)
        if flow_eps > 1.0:
            raise ValueError("flow_eps must not exceed 1")
        biological_time_scale = _validate_scalar_hyperparameter(
            "biological_time_scale", biological_time_scale, positive=True
        )
        parameterization = str(parameterization).strip().lower()
        if parameterization not in _PARAMETERIZATIONS:
            raise ValueError(
                "parameterization must be one of {'endpoint', 'direct_velocity'}"
            )

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.layers = int(layers)
        self.parameterization = parameterization
        self.flow_eps = float(flow_eps)
        self.biological_time_scale = float(biological_time_scale)

        self.flow_time_embedding = _SinusoidalTimeEmbedding(self.time_dim, scale=1.0)
        self.source_time_embedding = _SinusoidalTimeEmbedding(
            self.time_dim, scale=self.biological_time_scale
        )
        self.target_time_embedding = _SinusoidalTimeEmbedding(
            self.time_dim, scale=self.biological_time_scale
        )
        self.network = MLP(
            self.latent_dim + 3 * self.time_dim,
            self.hidden_dim,
            self.latent_dim,
            layers=self.layers,
            dropout=dropout,
        )

    def _validate_inputs(
        self,
        z_tau: Tensor,
        flow_time: Tensor | float,
        source_time: Tensor | float,
        target_time: Tensor | float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(z_tau, Tensor) or z_tau.ndim != 2:
            raise ValueError("z_tau must have shape [B, D]")
        if z_tau.shape[0] < 1 or z_tau.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_tau must have shape [B, {self.latent_dim}] with B >= 1; "
                f"got {tuple(z_tau.shape)}"
            )
        if not torch.is_floating_point(z_tau):
            raise ValueError("z_tau must use a floating dtype")
        if not torch.isfinite(z_tau).all():
            raise ValueError("z_tau must contain only finite values")

        # Match the module's parameter dtype while preserving gradients through
        # caller-provided time tensors.  Device mismatches are reported clearly
        # instead of being silently copied for every forward call.
        parameter = next(self.parameters())
        if z_tau.device != parameter.device:
            raise ValueError(
                "z_tau and adapter parameters must be on the same device; "
                f"got {z_tau.device} and {parameter.device}"
            )
        working_z = z_tau.to(dtype=parameter.dtype)
        flow = _as_batch_scalar(
            flow_time, "flow_time", working_z.shape[0], working_z.device, working_z.dtype
        )
        source = _as_batch_scalar(
            source_time, "source_time", working_z.shape[0], working_z.device, working_z.dtype
        )
        target = _as_batch_scalar(
            target_time, "target_time", working_z.shape[0], working_z.device, working_z.dtype
        )
        if torch.any((flow < 0.0) | (flow > 1.0)):
            raise ValueError("flow_time must lie in [0, 1]")
        return working_z, flow, source, target

    def _raw_prediction(
        self,
        z_tau: Tensor,
        flow_time: Tensor,
        source_time: Tensor,
        target_time: Tensor,
    ) -> Tensor:
        conditioning = torch.cat(
            (
                z_tau,
                self.flow_time_embedding(flow_time),
                self.source_time_embedding(source_time),
                self.target_time_embedding(target_time),
            ),
            dim=-1,
        )
        raw = self.network(conditioning)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("endpoint adapter produced a non-finite prediction")
        return raw

    def predict_endpoint_and_velocity(
        self,
        z_tau: Tensor,
        flow_time: Tensor | float,
        source_time: Tensor | float,
        target_time: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(endpoint, velocity)`` for either parameterization."""

        z_work, flow, source, target = self._validate_inputs(
            z_tau, flow_time, source_time, target_time
        )
        raw = self._raw_prediction(z_work, flow, source, target)
        if self.parameterization == "endpoint":
            endpoint = raw
            velocity = endpoint_velocity_from_endpoint(
                z_work, endpoint, flow, flow_eps=self.flow_eps
            )
        else:
            velocity = raw
            # scFiT uses the same epsilon-stabilized remaining interval for
            # direct-velocity endpoint reconstruction and target reparameterization.
            remaining = (1.0 - flow).clamp_min(self.flow_eps).unsqueeze(-1)
            endpoint = z_work + remaining * velocity
        if not torch.isfinite(endpoint).all() or not torch.isfinite(velocity).all():
            raise FloatingPointError("endpoint adapter output is non-finite")
        return endpoint, velocity

    def forward(
        self,
        z_tau: Tensor,
        flow_time: Tensor | float,
        source_time: Tensor | float,
        target_time: Tensor | float,
    ) -> Tensor:
        """Return the predicted target endpoint with shape ``[B, D]``."""

        endpoint, _ = self.predict_endpoint_and_velocity(
            z_tau, flow_time, source_time, target_time
        )
        return endpoint


def latent_endpoint_loss(
    predicted_endpoint: Tensor,
    target_endpoint: Tensor,
    *,
    predicted_velocity: Tensor | None = None,
    target_velocity: Tensor | None = None,
    endpoint_weight: float = 1.0,
    velocity_weight: float = 0.0,
) -> Tensor:
    r"""Compute endpoint MSE with an optional conditional-flow-matching term.

    The endpoint term is the latent analogue of scFiT's target-state
    supervision.  When both velocity tensors are supplied, the optional second
    term provides a direct flow-matching objective:

    .. math::

       L = \lambda_e \operatorname{MSE}(\hat z_t,z_t)
           + \lambda_v \operatorname{MSE}(\hat v,v^*).

    A positive ``velocity_weight`` requires both velocity arguments.  No
    population Sinkhorn term is included here; that term needs a batch-level
    target population and belongs in the Stage-1 trainer/evaluator.
    """

    _validate_latent_pair(
        predicted_endpoint, target_endpoint, "predicted_endpoint", "target_endpoint"
    )
    endpoint_weight = _validate_scalar_hyperparameter(
        "endpoint_weight", endpoint_weight
    )
    velocity_weight = _validate_scalar_hyperparameter(
        "velocity_weight", velocity_weight
    )
    if endpoint_weight < 0.0 or velocity_weight < 0.0:
        raise ValueError("loss weights must be nonnegative")
    if endpoint_weight == 0.0 and velocity_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")

    loss = predicted_endpoint.new_zeros(())
    if endpoint_weight:
        loss = loss + endpoint_weight * F.mse_loss(
            predicted_endpoint, target_endpoint
        )
    if velocity_weight:
        if predicted_velocity is None or target_velocity is None:
            raise ValueError(
                "predicted_velocity and target_velocity are required when "
                "velocity_weight is positive"
            )
        _validate_latent_pair(
            predicted_velocity, target_velocity, "predicted_velocity", "target_velocity"
        )
        if (
            predicted_velocity.device != predicted_endpoint.device
            or target_velocity.device != predicted_endpoint.device
        ):
            raise ValueError("velocity tensors must be on the endpoint tensor device")
        loss = loss + velocity_weight * F.mse_loss(
            predicted_velocity, target_velocity
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("latent endpoint loss is non-finite")
    return loss


# Descriptive alias for callers that prefer the scFiT naming.
scfit_endpoint_loss = latent_endpoint_loss


__all__ = [
    "BRAINWM_STAGE1_LATENT_DIM",
    "LatentEndpointAdapter",
    "endpoint_velocity_from_endpoint",
    "latent_endpoint_loss",
    "scfit_endpoint_loss",
]
