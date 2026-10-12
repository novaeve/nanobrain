"""Controlled continuous-time shared/private latent dynamics."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .blocks import FourierTimeEmbedding, MLP
from .config import CrossSpeciesConfig, DynamicsConfig, InterventionConfig
from .types import StructuralContext


class SpeciesLowRankAdapter(nn.Module):
    """A_s F_private(z) in the paper's cross-species factorization."""

    def __init__(self, latent_dim: int, config: CrossSpeciesConfig):
        super().__init__()
        self.config = config
        self.down = nn.Parameter(torch.empty(
            config.n_species, latent_dim, config.adapter_rank
        ))
        self.up = nn.Parameter(torch.zeros(
            config.n_species, config.adapter_rank, latent_dim
        ))
        nn.init.normal_(self.down, std=0.02)

    def forward(self, value: Tensor, species_id: Tensor) -> Tensor:
        down = self.down[species_id]
        up = self.up[species_id]
        low_rank = torch.einsum("bd,bdr->br", value, down)
        return torch.einsum("br,brd->bd", low_rank, up)


class ControlledLatentDynamics(nn.Module):
    def __init__(self, latent_dim: int, dynamics: DynamicsConfig,
                 species: CrossSpeciesConfig,
                 intervention: InterventionConfig):
        super().__init__()
        self.config = dynamics
        self.species_config = species
        self.intervention_config = intervention
        self.time = FourierTimeEmbedding(dynamics.time_dim)
        self.species_embedding = nn.Embedding(
            species.n_species, species.species_embedding_dim
        )
        control_inputs = intervention.control_dim * (2 if intervention.use_delta_control else 1)
        self.control_encoder = (
            MLP(max(control_inputs, 1), intervention.embedding_dim,
                intervention.embedding_dim, layers=2)
            if intervention.control_dim else None
        )
        context_dim = (
            latent_dim * 4
            + dynamics.time_dim
            + species.species_embedding_dim
            + (intervention.embedding_dim if intervention.control_dim else 0)
        )
        self.shared_drift = MLP(context_dim, dynamics.hidden_dim, latent_dim, layers=3)
        self.private_drift = MLP(context_dim, dynamics.hidden_dim, latent_dim, layers=3)
        self.species_adapter = SpeciesLowRankAdapter(latent_dim, species)
        self.diffusion = MLP(context_dim, dynamics.hidden_dim, latent_dim, layers=2)

    def _features(self, z: Tensor, structure: StructuralContext,
                  control: Tensor | None, delta_control: Tensor | None,
                  species_id: Tensor, time_days: Tensor) -> Tensor:
        parts = [
            z,
            structure.message,
            structure.niche_context,
            structure.tissue_context,
            self.time(time_days),
            self.species_embedding(species_id),
        ]
        if self.intervention_config.control_dim:
            if control is None:
                raise ValueError("control is required for intervention-aware dynamics")
            control_parts = [control]
            if self.intervention_config.use_delta_control:
                control_parts.append(
                    torch.zeros_like(control) if delta_control is None else delta_control
                )
            parts.append(self.control_encoder(torch.cat(control_parts, dim=-1)))
        return torch.cat(parts, dim=-1)

    def forward(self, z: Tensor, structure: StructuralContext,
                control: Tensor | None, delta_control: Tensor | None,
                species_id: Tensor, time_days: Tensor) -> tuple[Tensor, Tensor]:
        features = self._features(
            z, structure, control, delta_control, species_id, time_days
        )
        shared = self.shared_drift(features)
        private = self.species_adapter(self.private_drift(features), species_id)
        drift = shared + self.species_config.private_scale * private
        sigma = torch.nn.functional.softplus(self.diffusion(features)) \
            + self.config.diffusion_floor
        return drift, sigma


RHS = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


class FixedStepIntegrator(nn.Module):
    """Differentiable fixed-step ODE or Euler-Maruyama SDE solver."""

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

    def forward(self, rhs: RHS, z0: Tensor, start_time: Tensor,
                end_time: Tensor) -> tuple[Tensor, list[Tensor], Tensor]:
        if start_time.shape != end_time.shape or start_time.shape != (z0.shape[0],):
            raise ValueError("start_time/end_time must have shape [B]")
        z = z0
        trajectory = [z0]
        step_days = (end_time - start_time) / self.config.steps
        step = step_days / self.config.integration_time_scale_days
        sigma = torch.zeros_like(z0)
        for index in range(self.config.steps):
            time = start_time + index * step_days
            if self.config.mode == "sde":
                drift, sigma = rhs(z, time)
                noise = torch.randn_like(z)
                z = z + step[:, None] * drift + torch.sqrt(step.abs())[:, None] * sigma * noise
            elif self.config.solver == "euler":
                drift, sigma = rhs(z, time)
                z = z + step[:, None] * drift
            else:
                k1, sigma = rhs(z, time)
                k2, _ = rhs(
                    z + 0.5 * step[:, None] * k1,
                    time + 0.5 * step_days,
                )
                k3, _ = rhs(
                    z + 0.5 * step[:, None] * k2,
                    time + 0.5 * step_days,
                )
                k4, _ = rhs(z + step[:, None] * k3, time + step_days)
                z = z + (step[:, None] / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(z)
        return z, trajectory, sigma
