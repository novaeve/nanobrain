"""Train-only calibration primitives for strict spatial reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def section_source_mean(
    latent: Tensor,
    batch_index: Tensor,
    source_mask: Tensor,
    query_mask: Tensor,
) -> Tensor:
    """Return each query's mean latent over sources from the same section."""
    if latent.ndim != 2:
        raise ValueError("latent must have shape [B, D]")
    batch_size = latent.shape[0]
    if batch_index.shape != (batch_size,):
        raise ValueError("batch_index must have shape [B]")
    if source_mask.shape != (batch_size,) or query_mask.shape != (batch_size,):
        raise ValueError("source_mask and query_mask must have shape [B]")
    if source_mask.dtype != torch.bool or query_mask.dtype != torch.bool:
        raise ValueError("source_mask and query_mask must be boolean")
    if torch.any(source_mask & query_mask):
        raise ValueError("source and query masks must be disjoint")

    query_indices = torch.nonzero(query_mask, as_tuple=False).flatten()
    result = latent.new_empty((query_indices.numel(), latent.shape[1]))
    for section in torch.unique(batch_index[query_mask], sorted=True):
        output_positions = torch.nonzero(
            batch_index[query_indices] == section, as_tuple=False
        ).flatten()
        sources = source_mask & (batch_index == section)
        if not torch.any(sources):
            raise ValueError(f"section {int(section)} has no source cells")
        result[output_positions] = latent[sources].mean(dim=0)
    return result


class CenteredSpatialResidualCalibrator(nn.Module):
    """Bias-free low-rank map from centered neighbour signal to latent delta."""

    def __init__(self, latent_dim: int, rank: int) -> None:
        super().__init__()
        if latent_dim < 1 or rank < 1 or rank > latent_dim:
            raise ValueError("calibrator dimensions must satisfy 1 <= rank <= latent_dim")
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        self.down = nn.Linear(latent_dim, rank, bias=False)
        self.up = nn.Linear(rank, latent_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        neutral_latent: Tensor,
        centered_neighbor_residual: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if neutral_latent.ndim != 2 or neutral_latent.shape[1] != self.latent_dim:
            raise ValueError("neutral_latent has incompatible shape")
        if centered_neighbor_residual.shape != neutral_latent.shape:
            raise ValueError("centered residual must match neutral_latent")
        normalized = F.layer_norm(
            centered_neighbor_residual.float(),
            (self.latent_dim,),
            weight=None,
            bias=None,
        ).to(centered_neighbor_residual.dtype)
        delta = self.up(F.silu(self.down(normalized)))
        return neutral_latent + delta, delta


class QueryConditionedSpatialCalibrator(nn.Module):
    """Low-rank spatial residual conditioned on query-neighbour agreement.

    A real niche should be informative not only through its average state but
    through how that state relates to the masked query's neutral latent.  The
    elementwise interaction makes that relation explicit while zero
    initialization keeps the exact no-structure starting point.
    """

    def __init__(self, latent_dim: int, rank: int) -> None:
        super().__init__()
        if latent_dim < 1 or rank < 1 or rank > latent_dim:
            raise ValueError("calibrator dimensions must satisfy 1 <= rank <= latent_dim")
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        self.down = nn.Linear(3 * latent_dim, rank, bias=False)
        self.up = nn.Linear(rank, latent_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        neutral_latent: Tensor,
        query_residual: Tensor,
        centered_neighbor_residual: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected = (neutral_latent.shape[0], self.latent_dim)
        if neutral_latent.ndim != 2 or tuple(neutral_latent.shape) != expected:
            raise ValueError("neutral_latent has incompatible shape")
        if query_residual.shape != neutral_latent.shape:
            raise ValueError("query residual must match neutral_latent")
        if centered_neighbor_residual.shape != neutral_latent.shape:
            raise ValueError("centered neighbour residual must match neutral_latent")
        query = F.layer_norm(
            query_residual.float(), (self.latent_dim,), weight=None, bias=None
        )
        neighbor = F.layer_norm(
            centered_neighbor_residual.float(),
            (self.latent_dim,),
            weight=None,
            bias=None,
        )
        feature = torch.cat((query, neighbor, query * neighbor), dim=-1).to(
            neutral_latent.dtype
        )
        delta = self.up(F.silu(self.down(feature)))
        return neutral_latent + delta, delta


def spatial_margin_objective(
    real_loss: Tensor,
    permutation_loss: Tensor,
    zero_loss: Tensor,
    real_delta: Tensor,
    permutation_delta: Tensor,
    zero_delta: Tensor,
    *,
    margin: float,
    ranking_weight: float,
    real_residual_weight: float,
    control_residual_weight: float,
) -> dict[str, Tensor]:
    """Favor real-coordinate reconstruction without rewarding control drift."""
    scalar_losses = (real_loss, permutation_loss, zero_loss)
    if any(value.ndim != 0 for value in scalar_losses):
        raise ValueError("route losses must be scalars")
    if not all(value.shape == real_delta.shape for value in (permutation_delta, zero_delta)):
        raise ValueError("all route deltas must share shape")
    for name, value in (
        ("margin", margin),
        ("ranking_weight", ranking_weight),
        ("real_residual_weight", real_residual_weight),
        ("control_residual_weight", control_residual_weight),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    rank_permutation = F.relu(real_loss - permutation_loss + float(margin))
    rank_zero = F.relu(real_loss - zero_loss + float(margin))
    real_penalty = real_delta.float().square().sum(dim=1).mean()
    control_penalty = 0.5 * (
        permutation_delta.float().square().sum(dim=1).mean()
        + zero_delta.float().square().sum(dim=1).mean()
    )
    objective = (
        real_loss
        + float(ranking_weight) * (rank_permutation + rank_zero)
        + float(real_residual_weight) * real_penalty
        + float(control_residual_weight) * control_penalty
    )
    return {
        "objective": objective,
        "real_nb": real_loss,
        "permutation_nb": permutation_loss,
        "zero_nb": zero_loss,
        "rank_permutation": rank_permutation,
        "rank_zero": rank_zero,
        "real_residual_penalty": real_penalty,
        "control_residual_penalty": control_penalty,
    }
