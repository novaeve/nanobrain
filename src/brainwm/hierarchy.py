"""Soft cell-niche-tissue hierarchy from Section 3.3."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import MLP, segment_sum
from .config import HierarchyConfig


class CellNicheTissueHierarchy(nn.Module):
    def __init__(self, latent_dim: int, coordinate_dim: int, config: HierarchyConfig):
        super().__init__()
        self.config = config
        self.coordinate_dim = coordinate_dim
        self.assignment = MLP(
            latent_dim + coordinate_dim, config.hidden_dim, config.n_niches, layers=3
        )
        self.niche_value = nn.Linear(latent_dim, latent_dim)
        self.tissue_score = MLP(latent_dim, config.hidden_dim, 1, layers=2)

    def forward(
        self,
        z: Tensor,
        spatial_coords: Tensor | None,
        batch_index: Tensor,
        source_mask: Tensor | None = None,
        neutral_query_mask: Tensor | None = None,
        spatial_presence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, latent_dim = z.shape
        if batch_index.shape != (batch_size,):
            raise ValueError("batch_index must have shape [B]")
        batch_index = batch_index.to(device=z.device)
        for name, mask in (
            ("source_mask", source_mask),
            ("neutral_query_mask", neutral_query_mask),
        ):
            if mask is not None and (
                mask.shape != (batch_size,) or mask.dtype != torch.bool
            ):
                raise ValueError(f"{name} must be boolean with shape [B]")
        if spatial_presence is not None and (
            not isinstance(spatial_presence, Tensor)
            or
            spatial_presence.shape != (batch_size,)
            or spatial_presence.dtype != torch.bool
        ):
            raise ValueError("spatial_presence must be boolean with shape [B]")
        if source_mask is not None:
            source_mask = source_mask.to(device=z.device)
        if neutral_query_mask is not None:
            neutral_query_mask = neutral_query_mask.to(device=z.device)

        # Spatial coordinates are the availability signal for this branch.
        # A molecular-only (scRNA/snRNA) batch must not be interpreted as a
        # real section at the origin: doing so would create a learned niche
        # assignment and tissue context from fabricated coordinates.  Return
        # the same exact residual no-op contract as ``BrainWorldModel``'s
        # empty structural context instead.
        if spatial_coords is None:
            return (
                torch.zeros_like(z),
                torch.zeros_like(z),
                z.new_empty((batch_size, 0)),
                z.new_empty((0, 0, latent_dim)),
                z.new_empty((0, latent_dim)),
            )

        if spatial_coords.ndim != 2 or spatial_coords.shape != (
            batch_size,
            self.coordinate_dim,
        ):
            raise ValueError(
                f"spatial_coords must have shape [{batch_size}, {self.coordinate_dim}]"
            )
        spatial_coords = spatial_coords.to(device=z.device)
        if spatial_presence is not None:
            spatial_presence = spatial_presence.to(device=z.device)
            if not torch.any(spatial_presence):
                return (
                    torch.zeros_like(z),
                    torch.zeros_like(z),
                    z.new_empty((batch_size, 0)),
                    z.new_empty((0, 0, latent_dim)),
                    z.new_empty((0, latent_dim)),
                )
            # Permit arbitrary/NaN placeholders for unavailable rows while
            # ensuring they can never enter the learned spatial assignment.
            spatial_coords = torch.where(
                spatial_presence[:, None],
                spatial_coords,
                torch.zeros_like(spatial_coords),
            )
            if not torch.isfinite(spatial_coords[spatial_presence]).all():
                raise ValueError("spatial_coords must be finite for present cells")

        n_tissues = int(batch_index.max().item()) + 1 if batch_size else 0
        if not self.config.enabled:
            probability = z.new_ones((batch_size, 1))
            niche_state = z.new_zeros((n_tissues, 1, latent_dim))
            query = z
            if spatial_presence is not None:
                query = query * spatial_presence[:, None]
            if neutral_query_mask is not None:
                query = query.masked_fill(neutral_query_mask[:, None], 0.0)
            source = z
            source_counts = z.new_ones(batch_size)
            if spatial_presence is not None:
                source = source * spatial_presence[:, None]
                source_counts = source_counts * spatial_presence.to(z.dtype)
            if source_mask is not None:
                source = source * source_mask[:, None]
                source_counts = source_counts * source_mask.to(z.dtype)
            tissue_state = segment_sum(source, batch_index, n_tissues)
            counts = segment_sum(source_counts, batch_index, n_tissues).clamp_min(1)
            tissue_state = tissue_state / counts[:, None]
            tissue_context = tissue_state[batch_index]
            if spatial_presence is not None:
                probability = probability * spatial_presence[:, None]
                tissue_context = tissue_context * spatial_presence[:, None]
            return (
                query,
                tissue_context,
                probability,
                niche_state,
                tissue_state,
            )

        query = z
        if spatial_presence is not None:
            query = query.masked_fill(~spatial_presence[:, None], 0.0)
        if neutral_query_mask is not None:
            query = query.masked_fill(neutral_query_mask[:, None], 0.0)
        probability = torch.softmax(
            self.assignment(torch.cat([query, spatial_coords], dim=-1)), dim=-1
        )
        n_niches = self.config.n_niches
        flat_index = (
            batch_index[:, None] * n_niches
            + torch.arange(n_niches, device=z.device)[None, :]
        ).reshape(-1)
        source_weight = probability
        if spatial_presence is not None:
            source_weight = source_weight * spatial_presence[:, None]
        if source_mask is not None:
            source_weight = source_weight * source_mask[:, None]
        weighted = (source_weight[:, :, None] * z[:, None, :]).reshape(-1, latent_dim)
        numerator = segment_sum(weighted, flat_index, n_tissues * n_niches)
        denominator = segment_sum(
            source_weight.reshape(-1, 1), flat_index, n_tissues * n_niches
        )
        niche_state = (numerator / denominator.clamp_min(self.config.epsilon)).reshape(
            n_tissues, n_niches, latent_dim
        )
        niche_state = self.niche_value(niche_state)

        niche_attention = torch.softmax(
            self.tissue_score(niche_state).squeeze(-1), dim=1
        )
        tissue_state = torch.sum(niche_attention[:, :, None] * niche_state, dim=1)
        cell_niches = niche_state[batch_index]
        niche_context = torch.sum(probability[:, :, None] * cell_niches, dim=1)
        tissue_context = tissue_state[batch_index]
        if spatial_presence is not None:
            probability = probability * spatial_presence[:, None]
            niche_context = niche_context * spatial_presence[:, None]
            tissue_context = tissue_context * spatial_presence[:, None]
        return niche_context, tissue_context, probability, niche_state, tissue_state
