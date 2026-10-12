"""Source-only expression-space primitives for strict masked-cell reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor


def source_only_neighbor_profiles(
    count: Tensor,
    coordinates: Tensor,
    batch_index: Tensor,
    source_mask: Tensor,
    query_mask: Tensor,
    *,
    k_neighbors: int,
    pseudocount_strength: float,
) -> tuple[Tensor, Tensor]:
    """Return local and section expression profiles for masked queries.

    Only rows selected by ``source_mask`` contribute expression.  The local
    profile pools the k nearest source cells in coordinate space and shrinks
    their aggregate counts toward the matching source-only section profile.
    Results follow the original row order of ``query_mask``.
    """
    if count.ndim != 2:
        raise ValueError("count must have shape [B, G]")
    batch_size, n_genes = count.shape
    if coordinates.ndim != 2 or coordinates.shape[0] != batch_size:
        raise ValueError("coordinates must have shape [B, coordinate_dim]")
    if batch_index.shape != (batch_size,):
        raise ValueError("batch_index must have shape [B]")
    if source_mask.shape != (batch_size,) or query_mask.shape != (batch_size,):
        raise ValueError("source_mask and query_mask must have shape [B]")
    if source_mask.dtype != torch.bool or query_mask.dtype != torch.bool:
        raise ValueError("source_mask and query_mask must be boolean")
    if torch.any(source_mask & query_mask):
        raise ValueError("source and query masks must be disjoint")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    if pseudocount_strength < 0:
        raise ValueError("pseudocount_strength must be nonnegative")

    query_indices = torch.nonzero(query_mask, as_tuple=False).flatten()
    local_profiles = count.new_empty((query_indices.numel(), n_genes)).float()
    section_profiles = count.new_empty((query_indices.numel(), n_genes)).float()
    eps = torch.finfo(torch.float32).eps

    for section in torch.unique(batch_index[query_mask], sorted=True):
        output_positions = torch.nonzero(
            batch_index[query_indices] == section, as_tuple=False
        ).flatten()
        section_queries = query_indices[output_positions]
        section_sources = torch.nonzero(
            source_mask & (batch_index == section), as_tuple=False
        ).flatten()
        if not section_sources.numel():
            raise ValueError(f"section {int(section)} has no source cells")

        source_count = count[section_sources].float()
        section_count = source_count.sum(dim=0)
        section_profile = section_count / section_count.sum().clamp_min(eps)
        section_profiles[output_positions] = section_profile

        neighbor_count = min(k_neighbors, int(section_sources.numel()))
        distances = torch.cdist(
            coordinates[section_queries].float(),
            coordinates[section_sources].float(),
        )
        nearest = torch.argsort(distances, dim=1, stable=True)[:, :neighbor_count]
        pooling = source_count.new_zeros(
            (section_queries.numel(), section_sources.numel())
        )
        pooling.scatter_(1, nearest, 1.0)
        local_count = pooling @ source_count
        if pseudocount_strength:
            local_count = (
                local_count
                + float(pseudocount_strength) * section_profile.unsqueeze(0)
            )
        local_profiles[output_positions] = local_count / local_count.sum(
            dim=1, keepdim=True
        ).clamp_min(eps)

    return local_profiles, section_profiles


def fuse_spatial_probabilities(
    base_logits: Tensor,
    local_profile: Tensor,
    section_profile: Tensor,
    *,
    method: str,
    alpha: float,
) -> Tensor:
    """Fuse a frozen BrainWM distribution with source-only niche expression."""
    if base_logits.ndim != 2:
        raise ValueError("base_logits must have shape [Q, G]")
    if local_profile.shape != base_logits.shape:
        raise ValueError("local_profile must match base_logits")
    if section_profile.shape != base_logits.shape:
        raise ValueError("section_profile must match base_logits")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if method not in {"arithmetic", "log_ratio"}:
        raise ValueError("method must be 'arithmetic' or 'log_ratio'")

    base = torch.softmax(base_logits.float(), dim=-1)
    local = local_profile.float()
    section = section_profile.float()
    if method == "arithmetic":
        if alpha > 1:
            raise ValueError("arithmetic alpha must not exceed one")
        fused = (1.0 - float(alpha)) * base + float(alpha) * local
        return fused / fused.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    eps = 1.0e-8
    spatial_log_ratio = torch.log(local.clamp_min(eps)) - torch.log(
        section.clamp_min(eps)
    )
    return torch.softmax(base_logits.float() + float(alpha) * spatial_log_ratio, dim=-1)


def hierarchical_spatial_probabilities(
    base_logits: Tensor,
    local_profile: Tensor,
    section_profile: Tensor,
    *,
    global_weight: float,
    local_weight: float,
) -> Tensor:
    """Add a bounded coordinate-local residual to a global section mixture.

    The three nonnegative mixture weights are ``1-global_weight`` for the
    frozen BrainWM prior, ``global_weight-local_weight`` for the source-only
    section profile, and ``local_weight`` for the coordinate-local profile.
    """
    if base_logits.ndim != 2:
        raise ValueError("base_logits must have shape [Q, G]")
    if local_profile.shape != base_logits.shape:
        raise ValueError("local_profile must match base_logits")
    if section_profile.shape != base_logits.shape:
        raise ValueError("section_profile must match base_logits")
    if not 0.0 <= local_weight <= global_weight <= 1.0:
        raise ValueError("weights must satisfy 0 <= local <= global <= 1")
    base = torch.softmax(base_logits.float(), dim=-1)
    fused = (
        (1.0 - float(global_weight)) * base
        + (float(global_weight) - float(local_weight)) * section_profile.float()
        + float(local_weight) * local_profile.float()
    )
    return fused / fused.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


__all__ = [
    "fuse_spatial_probabilities",
    "hierarchical_spatial_probabilities",
    "source_only_neighbor_profiles",
]
