"""Optional continuous residual links between BrainWM training stages.

The module keeps the semantic stages distinct:

* Stage0 produces a molecular cell state;
* Stage1 contributes a temporal population update;
* Stage2 contributes a spatial microenvironment update.

It does not turn scBrain's label hierarchy into BrainWM stages.  The adapter is
kept outside :class:`BrainWorldModel` so existing constructors and checkpoints
retain an identical state-dict contract until a controlled experiment selects
this candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_state(value: Tensor, name: str, width: int | None = None) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D]")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")
    if width is not None and value.shape[1] != width:
        raise ValueError(f"{name} must have shape [B, {width}]")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must use a floating dtype")


@dataclass
class StageResidualOutput:
    """One continuous stage transition and its auditable components."""

    state: Tensor
    projected_delta: Tensor
    gated_delta: Tensor
    gate: Tensor


class GatedStageResidual(nn.Module):
    """Add a bounded learned residual update to a continuous latent state.

    ``delta`` is a continuous state increment, never an absolute state or hard
    class label. Existing BrainWM Stage1/Stage2 outputs are absolute states and
    must be converted with :func:`absolute_state_delta` before this call.
    A zero gate makes the adapter an exact identity, which is useful for
    checkpoint-compatible warm starts.  ``gate_init`` is the effective gate
    value before the inverse-tanh parameterization and must lie in ``(-1, 1)``.

    At an exactly zero gate, gradients reach the gate but not the delta branch
    on the first optimization step. Use a small nonzero ``gate_init`` when the
    delta producer is also initialized at zero, or warm up the gate first.
    """

    def __init__(
        self,
        latent_dim: int,
        delta_dim: int | None = None,
        *,
        gate: str = "scalar",
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            not isinstance(latent_dim, int)
            or isinstance(latent_dim, bool)
            or latent_dim < 1
        ):
            raise ValueError("latent_dim must be a positive integer")
        if delta_dim is None:
            delta_dim = latent_dim
        if (
            not isinstance(delta_dim, int)
            or isinstance(delta_dim, bool)
            or delta_dim < 1
        ):
            raise ValueError("delta_dim must be a positive integer")
        if gate not in {"scalar", "feature"}:
            raise ValueError("gate must be 'scalar' or 'feature'")
        if not math.isfinite(float(gate_init)) or not -1.0 < float(gate_init) < 1.0:
            raise ValueError("gate_init must be finite and lie in (-1, 1)")
        self.latent_dim = latent_dim
        self.delta_dim = delta_dim
        self.gate_kind = gate
        self.projection = (
            nn.Identity()
            if delta_dim == latent_dim
            else nn.Linear(delta_dim, latent_dim, bias=False)
        )
        gate_width = 1 if gate == "scalar" else latent_dim
        raw_init = math.atanh(float(gate_init))
        self.raw_gate = nn.Parameter(torch.full((gate_width,), raw_init))

    def effective_gate(self) -> Tensor:
        """Return the bounded gate used by the residual path."""
        return torch.tanh(self.raw_gate)

    def forward(self, base: Tensor, delta: Tensor) -> StageResidualOutput:
        _validate_state(base, "base", self.latent_dim)
        _validate_state(delta, "delta", self.delta_dim)
        if delta.shape[0] != base.shape[0]:
            raise ValueError("base and delta must have the same batch size")
        if delta.device != base.device:
            raise ValueError("base and delta must be on the same device")
        if delta.dtype != base.dtype:
            raise ValueError("base and delta must use the same dtype")

        projected = self.projection(delta)
        gate = self.effective_gate().to(dtype=projected.dtype)
        gated = projected * gate
        state = base + gated.to(dtype=base.dtype)
        return StageResidualOutput(
            state=state,
            projected_delta=projected,
            gated_delta=gated,
            gate=gate,
        )


def absolute_state_delta(base_state: Tensor, updated_state: Tensor) -> Tensor:
    """Convert two compatible absolute states into one explicit delta."""

    _validate_state(base_state, "base_state")
    _validate_state(updated_state, "updated_state")
    if base_state.shape != updated_state.shape:
        raise ValueError("base_state and updated_state must have equal shape")
    if base_state.device != updated_state.device:
        raise ValueError("base_state and updated_state must be on the same device")
    if base_state.dtype != updated_state.dtype:
        raise ValueError("base_state and updated_state must use the same dtype")
    return updated_state - base_state


@dataclass
class MultiStageResidualOutput:
    """Molecular, temporal, and spatial states from a serial residual stack."""

    molecular: Tensor
    temporal: StageResidualOutput
    spatial: StageResidualOutput


class MultiStageResidualStack(nn.Module):
    """Compose temporal and spatial continuous updates without hard routing."""

    def __init__(
        self,
        latent_dim: int,
        *,
        temporal_delta_dim: int | None = None,
        spatial_delta_dim: int | None = None,
        gate: str = "scalar",
        temporal_gate_init: float = 0.0,
        spatial_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.temporal_link = GatedStageResidual(
            latent_dim,
            temporal_delta_dim,
            gate=gate,
            gate_init=temporal_gate_init,
        )
        self.spatial_link = GatedStageResidual(
            latent_dim,
            spatial_delta_dim,
            gate=gate,
            gate_init=spatial_gate_init,
        )

    def forward(
        self,
        molecular_state: Tensor,
        temporal_delta: Tensor,
        spatial_delta: Tensor,
    ) -> MultiStageResidualOutput:
        temporal = self.temporal_link(molecular_state, temporal_delta)
        spatial = self.spatial_link(temporal.state, spatial_delta)
        return MultiStageResidualOutput(
            molecular=molecular_state,
            temporal=temporal,
            spatial=spatial,
        )

    def forward_from_absolute_states(
        self,
        molecular_state: Tensor,
        temporal_state: Tensor,
        spatial_state: Tensor,
    ) -> MultiStageResidualOutput:
        """Gate existing BrainWM absolute states without double-adding bases."""

        temporal_delta = absolute_state_delta(molecular_state, temporal_state)
        spatial_delta = absolute_state_delta(temporal_state, spatial_state)
        return self(molecular_state, temporal_delta, spatial_delta)

    def set_trainable_links(self, *, temporal: bool, spatial: bool) -> dict[str, int]:
        """Set adapter-link freeze policy; upstream producers remain caller-owned."""

        counts: dict[str, int] = {}
        for name, link, trainable in (
            ("temporal", self.temporal_link, temporal),
            ("spatial", self.spatial_link, spatial),
        ):
            count = 0
            for parameter in link.parameters():
                parameter.requires_grad_(trainable)
                count += parameter.numel()
            counts[name] = count
        return counts


def representation_anchor_loss(
    current_stage0: Tensor,
    frozen_stage0: Tensor,
    *,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.0,
) -> Tensor:
    """Anchor Stage0 during low-rate joint tuning without suppressing updates.

    The reference is detached deliberately.  Callers should apply this loss to
    Stage0 representations of the same replay cells before temporal or spatial
    residuals are added.  Applying it directly to Stage1/Stage2 would erase the
    biological changes those stages are intended to learn.
    """

    _validate_state(current_stage0, "current_stage0")
    _validate_state(frozen_stage0, "frozen_stage0")
    if current_stage0.shape != frozen_stage0.shape:
        raise ValueError("current_stage0 and frozen_stage0 must have equal shape")
    if current_stage0.device != frozen_stage0.device:
        raise ValueError("current_stage0 and frozen_stage0 must be on the same device")
    for name, value in (("mse_weight", mse_weight), ("cosine_weight", cosine_weight)):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if mse_weight == 0 and cosine_weight == 0:
        raise ValueError("at least one anchor weight must be positive")

    current = current_stage0.float()
    reference = frozen_stage0.detach().float()
    loss = current.new_zeros(())
    if mse_weight:
        loss = loss + float(mse_weight) * F.mse_loss(current, reference)
    if cosine_weight:
        cosine = F.cosine_similarity(current, reference, dim=-1, eps=1.0e-8)
        both_zero = (current.norm(dim=-1) <= 1.0e-8) & (
            reference.norm(dim=-1) <= 1.0e-8
        )
        cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
        loss = loss + float(cosine_weight) * (1.0 - cosine).mean()
    return loss


__all__ = [
    "GatedStageResidual",
    "MultiStageResidualOutput",
    "MultiStageResidualStack",
    "StageResidualOutput",
    "absolute_state_delta",
    "representation_anchor_loss",
]
