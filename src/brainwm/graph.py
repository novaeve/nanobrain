"""Dynamic spatial graph G(t) and learned molecular/spatial edge attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import MLP, segment_softmax, segment_sum
from .config import GraphConfig


class DynamicSpatialGraph(nn.Module):
    def __init__(self, latent_dim: int, config: GraphConfig):
        super().__init__()
        self.config = config
        edge_input = (
            2 * latent_dim + config.coordinate_dim + 1 + config.ligand_receptor_dim
        )
        self.edge_score = MLP(edge_input, config.hidden_dim, 1, layers=3)
        self.value = nn.Linear(latent_dim, config.message_dim)
        self.output = MLP(config.message_dim, config.hidden_dim, latent_dim, layers=2)

    def build_knn(
        self,
        coords: Tensor,
        batch_index: Tensor,
        source_mask: Tensor | None = None,
        spatial_presence: Tensor | None = None,
    ) -> Tensor:
        if batch_index.ndim != 1:
            raise ValueError("batch_index must have shape [B]")
        if coords.ndim != 2 or coords.shape[0] != batch_index.shape[0]:
            raise ValueError("coords must have shape [B, D]")
        edges: list[Tensor] = []
        if source_mask is not None and (
            source_mask.shape != batch_index.shape or source_mask.dtype != torch.bool
        ):
            raise ValueError("source_mask must be boolean with shape [B]")
        if spatial_presence is not None and (
            not isinstance(spatial_presence, Tensor)
            or
            spatial_presence.shape != batch_index.shape
            or spatial_presence.dtype != torch.bool
        ):
            raise ValueError("spatial_presence must be boolean with shape [B]")
        if source_mask is not None:
            source_mask = source_mask.to(device=batch_index.device)
        if spatial_presence is not None:
            spatial_presence = spatial_presence.to(device=batch_index.device)
            coords = coords.to(device=batch_index.device)
            if torch.any(spatial_presence):
                coords = torch.where(
                    spatial_presence[:, None], coords, torch.zeros_like(coords)
                )
                if not torch.isfinite(coords[spatial_presence]).all():
                    raise ValueError("coords must be finite for present cells")

        # Keep the historical source-only topology unchanged when no
        # availability mask is supplied.  With a partial spatial batch, use
        # separate source and destination sets so an unobserved cell can
        # neither contribute to nor receive a spatial message.
        if spatial_presence is not None:
            for group in torch.unique(batch_index, sorted=True):
                cells = torch.nonzero(batch_index == group, as_tuple=False).flatten()
                if cells.numel() < 2:
                    continue
                present = spatial_presence[cells]
                destinations = cells[present]
                source_allowed = present
                if source_mask is not None:
                    source_allowed = source_allowed & source_mask[cells]
                sources = cells[source_allowed]
                if destinations.numel() == 0 or sources.numel() == 0:
                    continue

                distance = torch.cdist(coords[destinations], coords[sources])
                # A cell must not be its own neighbour, even when it is in
                # both the source and destination sets.
                distance = distance.masked_fill(
                    destinations[:, None] == sources[None, :], torch.inf
                )
                k = min(self.config.k_neighbors, int(sources.numel()))
                if k <= 0:
                    continue
                values, neighbor = torch.topk(
                    distance, k=k, dim=1, largest=False
                )
                dst = destinations[:, None].expand_as(neighbor).reshape(-1)
                src = sources[neighbor.reshape(-1)]
                finite = torch.isfinite(values).reshape(-1)
                if torch.any(finite):
                    edges.append(torch.stack([src[finite], dst[finite]], dim=0))
            if not edges:
                return torch.empty((2, 0), dtype=torch.long, device=coords.device)
            return torch.cat(edges, dim=1)

        for group in torch.unique(batch_index, sorted=True):
            cells = torch.nonzero(batch_index == group, as_tuple=False).flatten()
            if cells.numel() < 2:
                continue
            allowed = (
                torch.ones(cells.numel(), dtype=torch.bool, device=cells.device)
                if source_mask is None
                else source_mask[cells]
            )
            if not torch.any(allowed):
                continue
            k = min(self.config.k_neighbors, int(allowed.sum()))
            distance = torch.cdist(coords[cells], coords[cells])
            distance[:, ~allowed] = torch.inf
            distance.fill_diagonal_(torch.inf)
            values, neighbor = torch.topk(distance, k=k, dim=1, largest=False)
            dst = cells[:, None].expand_as(neighbor).reshape(-1)
            src = cells[neighbor.reshape(-1)]
            finite = torch.isfinite(values).reshape(-1)
            if torch.any(finite):
                edges.append(torch.stack([src[finite], dst[finite]], dim=0))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=coords.device)
        return torch.cat(edges, dim=1)

    def forward(
        self,
        z: Tensor,
        spatial_coords: Tensor | None,
        batch_index: Tensor,
        edge_index: Tensor | None = None,
        ligand_receptor: Tensor | None = None,
        source_mask: Tensor | None = None,
        neutral_query_mask: Tensor | None = None,
        spatial_presence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = z.shape[0]
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
        if not self.config.enabled or spatial_coords is None:
            empty_edge = torch.empty((2, 0), dtype=torch.long, device=z.device)
            return torch.zeros_like(z), empty_edge, z.new_empty((0,))
        if spatial_coords.shape[0] != batch_size:
            raise ValueError("spatial_coords must have shape [B, D]")
        if spatial_coords.ndim != 2 or spatial_coords.shape[1] != self.config.coordinate_dim:
            raise ValueError(
                f"spatial_coords must have shape [{batch_size}, {self.config.coordinate_dim}]"
            )
        spatial_coords = spatial_coords.to(device=z.device)
        if source_mask is not None:
            source_mask = source_mask.to(device=z.device)
        if neutral_query_mask is not None:
            neutral_query_mask = neutral_query_mask.to(device=z.device)
        if spatial_presence is not None:
            spatial_presence = spatial_presence.to(device=z.device)
            if not torch.any(spatial_presence):
                empty_edge = torch.empty((2, 0), dtype=torch.long, device=z.device)
                return torch.zeros_like(z), empty_edge, z.new_empty((0,))
            # Missing rows may carry NaN/placeholder coordinates from a
            # partially observed table.  Sanitize only those rows; observed
            # coordinates remain subject to the usual finite-value contract.
            spatial_coords = torch.where(
                spatial_presence[:, None],
                spatial_coords,
                torch.zeros_like(spatial_coords),
            )
            if not torch.isfinite(spatial_coords[spatial_presence]).all():
                raise ValueError("spatial_coords must be finite for present cells")
        if edge_index is None or self.config.recompute_topology:
            edge_index = self.build_knn(
                spatial_coords,
                batch_index,
                source_mask=source_mask,
                spatial_presence=spatial_presence,
            )
        else:
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape [2, E]")
            edge_index = edge_index.to(device=z.device, dtype=torch.long)
            if edge_index.numel():
                src, dst = edge_index
                if (
                    torch.any(src < 0)
                    or torch.any(src >= batch_size)
                    or torch.any(dst < 0)
                    or torch.any(dst >= batch_size)
                ):
                    raise ValueError("edge_index contains an out-of-range node")
                keep = torch.ones(
                    edge_index.shape[1], dtype=torch.bool, device=z.device
                )
                if source_mask is not None:
                    keep = keep & source_mask.to(device=z.device)[src]
                if spatial_presence is not None:
                    keep = keep & spatial_presence[src] & spatial_presence[dst]
                edge_index = edge_index[:, keep]
                if ligand_receptor is not None:
                    ligand_receptor = ligand_receptor[keep]
        if edge_index.shape[1] == 0:
            return torch.zeros_like(z), edge_index, z.new_empty((0,))

        src, dst = edge_index
        delta = (spatial_coords[src] - spatial_coords[dst]) / self.config.distance_scale
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        query = z[dst]
        if neutral_query_mask is not None:
            query = query.masked_fill(neutral_query_mask[dst, None], 0.0)
        parts = [query, z[src], delta, distance]
        if self.config.ligand_receptor_dim:
            if ligand_receptor is None:
                ligand_receptor = z.new_zeros(
                    (edge_index.shape[1], self.config.ligand_receptor_dim)
                )
            if ligand_receptor.shape != (
                edge_index.shape[1],
                self.config.ligand_receptor_dim,
            ):
                raise ValueError(
                    "ligand_receptor shape does not match graph configuration"
                )
            if neutral_query_mask is not None:
                ligand_receptor = ligand_receptor.masked_fill(
                    neutral_query_mask[dst, None], 0.0
                )
            parts.append(ligand_receptor)
        score = self.edge_score(torch.cat(parts, dim=-1)).squeeze(-1)
        weight = segment_softmax(score, dst, batch_size)
        raw_message = self.value(z[src]) * weight[:, None]
        message = self.output(segment_sum(raw_message, dst, batch_size))
        has_message = torch.bincount(dst, minlength=batch_size) > 0
        message = message.masked_fill(~has_message[:, None], 0.0)
        return message, edge_index, weight
