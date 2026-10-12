"""Reference-method-derived capability adapters for the BrainWM backbone.

The reference papers contribute mechanisms, not new BrainWM stages or
reproduction targets. This module keeps those capability branches behind small,
explicit adapters so a caller can turn them on for a controlled project-native
experiment without changing the default model state-dict contract:

* :class:`ScFiTTemporalAdapter` adds a Stage-1 population endpoint (or direct
  velocity) parameterization to the existing BrainWM temporal transition.
* :class:`ScUniFlowSparseGeneAdapter` applies an ordered row/column mixer to
  sparse within-cell gene tokens before the existing Stage-0 pooling path.
* :class:`ScUniFlowSpatialAdapter` adds a section-local, gated spatial context
  residual to a latent state. It strengthens the Stage-2 context path; it is
  not a replacement for BrainWM's graph/niche hierarchy.

All adapters are opt-in and use zero gates by default.  A zero gate is an
exact identity, which is useful when initializing from an existing Stage-0,
Stage-1, or Stage-2 checkpoint and warming up only the new branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .endpoint import LatentEndpointAdapter
from .ordered_mixer import (
    OrderedGeneMixer,
    OrderMode,
    _canonical_order_mode,
    _validate_grid_shape,
    inverse_permutation,
)


def _bounded_gate_parameter(init: float) -> nn.Parameter:
    """Create an inverse-tanh gate parameter with a bounded effective value."""
    value = float(init)
    if not math.isfinite(value) or not -1.0 < value < 1.0:
        raise ValueError("gate_init must be finite and lie in (-1, 1)")
    return nn.Parameter(torch.tensor(math.atanh(value), dtype=torch.float32))


def _validate_2d_state(value: Tensor, name: str, width: int | None = None) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, D]")
    if value.shape[0] < 1 or (width is not None and value.shape[1] != width):
        suffix = f" with width {width}" if width is not None else ""
        raise ValueError(f"{name} must be non-empty{suffix}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating-point data")


@dataclass
class ScFiTTemporalOutput:
    """Endpoint and velocity returned by :class:`ScFiTTemporalAdapter`."""

    state: Tensor
    endpoint: Tensor
    velocity: Tensor
    flow_time: Tensor


class ScFiTTemporalAdapter(nn.Module):
    """A BrainWM Stage-1 endpoint/direct-velocity adapter.

    ``z_initial`` is a cell-population sample, not a tracked cell lineage.
    At ``flow_time=0`` the adapter predicts a target population state.  In
    ``residual`` mode the prediction is blended with the input through a
    bounded gate; ``endpoint`` mode returns the predicted target directly.
    The latter is useful when this branch is selected as the Stage-1 model,
    while the former gives a checkpoint-compatible warm start.
    """

    _MODES = frozenset({"endpoint", "residual"})

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden_dim: int = 256,
        time_dim: int = 32,
        layers: int = 2,
        dropout: float = 0.0,
        parameterization: Literal["endpoint", "direct_velocity"] = "endpoint",
        flow_eps: float = 1.0e-3,
        biological_time_scale: float = 365.2425,
        mode: Literal["endpoint", "residual"] = "residual",
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if mode not in self._MODES:
            raise ValueError("mode must be 'endpoint' or 'residual'")
        self.latent_dim = latent_dim
        self.mode = mode
        self.predictor = LatentEndpointAdapter(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            layers=layers,
            dropout=dropout,
            parameterization=parameterization,
            flow_eps=flow_eps,
            biological_time_scale=biological_time_scale,
        )
        self.raw_gate = _bounded_gate_parameter(gate_init)

    @property
    def parameterization(self) -> str:
        return self.predictor.parameterization

    def effective_gate(self) -> Tensor:
        return torch.tanh(self.raw_gate)

    def forward(
        self,
        z_initial: Tensor,
        source_time: Tensor | float,
        target_time: Tensor | float,
        flow_time: Tensor | float = 0.0,
    ) -> ScFiTTemporalOutput:
        _validate_2d_state(z_initial, "z_initial", self.latent_dim)
        endpoint, velocity = self.predictor.predict_endpoint_and_velocity(
            z_initial, flow_time, source_time, target_time
        )
        if self.mode == "residual":
            gate = self.effective_gate().to(dtype=endpoint.dtype, device=endpoint.device)
            state = z_initial + gate * (endpoint - z_initial)
        else:
            state = endpoint
        if not torch.isfinite(state).all():
            raise FloatingPointError("scFiT temporal adapter produced a non-finite state")
        flow = torch.as_tensor(flow_time, device=z_initial.device, dtype=z_initial.dtype)
        if flow.ndim == 0:
            flow = flow.expand(z_initial.shape[0])
        elif flow.ndim == 2 and flow.shape == (z_initial.shape[0], 1):
            flow = flow[:, 0]
        return ScFiTTemporalOutput(
            state=state,
            endpoint=endpoint,
            velocity=velocity,
            flow_time=flow,
        )


class ScUniFlowSparseGeneAdapter(nn.Module):
    """Apply an ordered mixer to variable-length sparse gene-token rows.

    ``gene_order`` follows the project convention used by
    :func:`ordered_mixer.make_order`: it is an ordered-position-to-global-gene
    permutation.  Cells contain only a subset of that global vocabulary, so a
    local permutation is built for each CSR row before applying the shared
    mixer.  The output is restored to the original flattened token order.
    """

    def __init__(
        self,
        token_dim: int,
        *,
        order_mode: OrderMode | str = "biological",
        random_seed: int | None = None,
        grid_shape: tuple[int, int] | None = None,
        ffn_multiplier: float = 1.0,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        gene_order: Tensor | None = None,
        max_tokens_per_row: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(token_dim, int) or isinstance(token_dim, bool) or token_dim < 1:
            raise ValueError("token_dim must be a positive integer")
        self.token_dim = token_dim
        if (
            isinstance(max_tokens_per_row, bool)
            or not isinstance(max_tokens_per_row, int)
            or max_tokens_per_row < 0
        ):
            raise ValueError("max_tokens_per_row must be a nonnegative integer")
        self.max_tokens_per_row = max_tokens_per_row
        self.mixer = OrderedGeneMixer(
            token_dim,
            order_mode=order_mode,
            random_seed=random_seed,
            grid_shape=grid_shape,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.raw_gate = _bounded_gate_parameter(gate_init)
        if gene_order is not None:
            if (
                gene_order.ndim != 1
                or gene_order.dtype == torch.bool
                or gene_order.is_floating_point()
                or gene_order.is_complex()
            ):
                raise ValueError("gene_order must be a one-dimensional integer permutation")
            checked_order = gene_order.to(dtype=torch.long).clone()
            if (
                torch.any(checked_order < 0)
                or torch.any(checked_order >= checked_order.numel())
                or torch.unique(checked_order).numel() != checked_order.numel()
            ):
                raise ValueError("gene_order must be a permutation of [0, V)")
            self.register_buffer("gene_order", checked_order, persistent=False)
        else:
            self.register_buffer("gene_order", None, persistent=False)

    def effective_gate(self) -> Tensor:
        return torch.tanh(self.raw_gate)

    def _has_active_dropout(self) -> bool:
        """Return whether batching could change the mixer's RNG behavior."""
        return any(
            isinstance(module, nn.Dropout)
            and module.p > 0.0
            and module.training
            for module in self.mixer.modules()
        )

    def _mix_row(
        self,
        tokens: Tensor,
        gene_index: Tensor,
        start: int,
        end: int,
        rank: Tensor | None,
        selected_mode: OrderMode | str,
        biological_mode: bool,
        gate: Tensor,
        seed: int | None,
    ) -> Tensor:
        """Mix one CSR row, retaining the historical reference path."""
        local_genes = gene_index[start:end]
        if biological_mode:
            if rank is None:
                local_order = torch.argsort(local_genes, stable=True)
            else:
                local_order = torch.argsort(rank[local_genes], stable=True)
        else:
            local_order = torch.arange(
                end - start, device=tokens.device, dtype=torch.long
            )
        local = tokens[start:end].unsqueeze(0)
        # The local order is a permutation of positions in this cell;
        # ``mode='biological'`` would reinterpret it as a global order, so
        # pass it explicitly and use the identity mode after gathering.
        ordered = local.index_select(1, local_order)
        scan_mode = "none" if biological_mode else selected_mode
        mixed = self.mixer(ordered, mode=scan_mode, seed=seed).squeeze(0)
        restored = mixed.index_select(0, inverse_permutation(local_order))
        original = local.squeeze(0)
        return original + gate * (restored - original)

    @staticmethod
    def _validate_sparse(tokens: Tensor, row_ptr: Tensor, gene_index: Tensor) -> None:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [T, D]")
        if not tokens.is_floating_point() or not torch.isfinite(tokens).all():
            raise ValueError("tokens must be finite floating-point data")
        if row_ptr.ndim != 1 or row_ptr.numel() < 2:
            raise ValueError("row_ptr must have shape [B+1]")
        if gene_index.ndim != 1 or gene_index.numel() != tokens.shape[0]:
            raise ValueError("gene_index must have shape [T]")
        for value, name in ((row_ptr, "row_ptr"), (gene_index, "gene_index")):
            if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
                raise ValueError(f"{name} must use an integer dtype")
        if int(row_ptr[0]) != 0 or int(row_ptr[-1]) != tokens.shape[0]:
            raise ValueError("row_ptr does not bound tokens")
        if torch.any(row_ptr[1:] < row_ptr[:-1]):
            raise ValueError("row_ptr must be nondecreasing")

    def forward(
        self,
        tokens: Tensor,
        row_ptr: Tensor,
        gene_index: Tensor,
        gene_order: Tensor | None = None,
        *,
        mode: OrderMode | str | None = None,
        seed: int | None = None,
    ) -> Tensor:
        self._validate_sparse(tokens, row_ptr, gene_index)
        if tokens.shape[1] != self.token_dim:
            raise ValueError(f"tokens have width {tokens.shape[1]}, expected {self.token_dim}")
        if row_ptr.device != tokens.device or gene_index.device != tokens.device:
            raise ValueError("tokens, row_ptr, and gene_index must share a device")
        if gene_order is None:
            gene_order = self.gene_order
        if gene_order is not None:
            if (
                gene_order.ndim != 1
                or gene_order.dtype == torch.bool
                or gene_order.is_floating_point()
                or gene_order.is_complex()
            ):
                raise ValueError("gene_order must be a one-dimensional integer permutation")
            gene_order = gene_order.to(device=tokens.device, dtype=torch.long)
            if (
                torch.any(gene_order < 0)
                or torch.any(gene_order >= gene_order.numel())
                or torch.unique(gene_order).numel() != gene_order.numel()
            ):
                raise ValueError("gene_order must be a permutation of [0, V)")
            if gene_index.numel() and int(gene_index.max()) >= gene_order.numel():
                raise ValueError("gene_index is outside gene_order vocabulary")
            rank = torch.empty_like(gene_order)
            rank[gene_order] = torch.arange(gene_order.numel(), device=tokens.device)
        else:
            rank = None

        selected_mode = (
            self.mixer.order_mode
            if mode is None
            else _canonical_order_mode(mode)
        )
        selected_mode_key = selected_mode
        biological_mode = selected_mode_key == "biological"

        # An optional cap keeps the ordered branch bounded on dense/high-depth
        # CSR rows.  Selection is deterministic and source-local: biological
        # mode keeps the earliest configured gene-order ranks, while the
        # no-order/random controls keep the row prefix.  Unselected tokens
        # bypass the branch and are copied back unchanged.
        source_positions = torch.arange(
            tokens.shape[0], device=tokens.device, dtype=torch.long
        )
        work_tokens = tokens
        work_gene_index = gene_index
        work_row_ptr = row_ptr
        if self.max_tokens_per_row > 0:
            cap = self.max_tokens_per_row
            selected_parts: list[Tensor] = []
            selected_lengths: list[int] = []
            for row in range(row_ptr.numel() - 1):
                start = int(row_ptr[row].item())
                end = int(row_ptr[row + 1].item())
                length = end - start
                if length <= 0:
                    selected_lengths.append(0)
                    continue
                if length <= cap:
                    local = torch.arange(start, end, device=tokens.device)
                elif biological_mode:
                    local_key = (
                        gene_index[start:end]
                        if rank is None
                        else rank[gene_index[start:end]]
                    )
                    local = start + torch.argsort(local_key, stable=True)[:cap]
                else:
                    local = torch.arange(start, start + cap, device=tokens.device)
                selected_parts.append(local)
                selected_lengths.append(int(local.numel()))
            source_positions = (
                torch.cat(selected_parts)
                if selected_parts
                else torch.empty(0, device=tokens.device, dtype=torch.long)
            )
            work_tokens = tokens.index_select(0, source_positions)
            work_gene_index = gene_index.index_select(0, source_positions)
            work_row_ptr = torch.as_tensor(
                [0, *torch.cumsum(torch.as_tensor(selected_lengths, device=tokens.device), 0).tolist()],
                device=tokens.device,
                dtype=row_ptr.dtype,
            )

        mixed_result = work_tokens.clone()
        gate = self.effective_gate().to(dtype=tokens.dtype, device=tokens.device)

        lengths = work_row_ptr[1:] - work_row_ptr[:-1]
        nonempty = lengths > 0
        if not bool(nonempty.any().item()):
            return tokens.clone()

        # Dropout masks are sampled per input element.  Batching equal-length
        # rows would change both masks and RNG consumption, so retain the
        # historical row loop whenever dropout is active in train mode.
        if self._has_active_dropout():
            for row in range(work_row_ptr.numel() - 1):
                start = int(work_row_ptr[row].item())
                end = int(work_row_ptr[row + 1].item())
                if end > start:
                    mixed_result[start:end] = self._mix_row(
                        work_tokens,
                        work_gene_index,
                        start,
                        end,
                        rank,
                        selected_mode,
                        biological_mode,
                        gate,
                        seed,
                    )
            result = tokens.clone()
            result.index_copy_(0, source_positions, mixed_result)
            return result

        # Group rows by the underlying grid shape rather than exact token
        # count.  For the default near-square grid this collapses many nearby
        # CSR lengths into one bucket.  The mixer receives a per-sample valid
        # mask, so tail padding never contributes to another row or to a scan.
        # Random mode remains on the exact-length path above because its
        # seeded randperm is length-dependent and must retain historical
        # per-row semantics.
        if selected_mode_key in {"random"}:
            for length_value in torch.unique(lengths[nonempty], sorted=True):
                length = int(length_value.item())
                row_ids = torch.nonzero(lengths == length_value, as_tuple=False).flatten()
                row_starts = work_row_ptr.index_select(0, row_ids).to(dtype=torch.long)
                offsets = torch.arange(length, device=tokens.device, dtype=torch.long)
                flat_positions = (row_starts[:, None] + offsets[None, :]).reshape(-1)
                local_tokens = work_tokens.index_select(0, flat_positions).reshape(
                    row_ids.numel(), length, self.token_dim
                )
                local_genes = work_gene_index.index_select(0, flat_positions).reshape(
                    row_ids.numel(), length
                )
                identity = offsets.unsqueeze(0).expand(row_ids.numel(), -1)
                ordered = local_tokens.gather(
                    1, identity.unsqueeze(-1).expand(-1, -1, self.token_dim)
                )
                mixed = self.mixer(ordered, mode=selected_mode, seed=seed)
                inverse = torch.empty_like(identity)
                inverse.scatter_(1, identity, identity)
                restored = mixed.gather(
                    1, inverse.unsqueeze(-1).expand(-1, -1, self.token_dim)
                )
                updated = local_tokens + gate * (restored - local_tokens)
                mixed_result.index_copy_(0, flat_positions, updated.reshape(-1, self.token_dim))
            result = tokens.clone()
            result.index_copy_(0, source_positions, mixed_result)
            return result

        shape_groups: dict[tuple[int, int], list[int]] = {}
        for row, length_value in enumerate(lengths.tolist()):
            if length_value > 0:
                shape = _validate_grid_shape(self.mixer.grid_shape, int(length_value))
                shape_groups.setdefault(shape, []).append(row)

        for (height, width), row_list in shape_groups.items():
            row_ids = torch.tensor(row_list, device=tokens.device, dtype=torch.long)
            row_lengths = lengths.index_select(0, row_ids).to(dtype=torch.long)
            batch_size = row_ids.numel()
            slots = height * width
            row_starts = work_row_ptr.index_select(0, row_ids).to(dtype=torch.long)
            offsets = torch.arange(slots, device=tokens.device, dtype=torch.long)
            valid = offsets.unsqueeze(0) < row_lengths.unsqueeze(1)
            # Tail offsets repeat the final real token solely to keep gather
            # indices in-bounds; the per-row valid mask zeroes them before the
            # scan and excludes them from inverse scatter below.
            safe_offsets = torch.minimum(
                offsets.unsqueeze(0), (row_lengths - 1).unsqueeze(1)
            )
            flat_positions = (row_starts.unsqueeze(1) + safe_offsets).reshape(-1)
            local_tokens = work_tokens.index_select(0, flat_positions).reshape(
                batch_size, slots, self.token_dim
            )
            local_genes = work_gene_index.index_select(0, flat_positions).reshape(
                batch_size, slots
            )
            if biological_mode:
                if rank is None:
                    order_key = local_genes
                else:
                    order_key = rank[local_genes]
                sentinel = int(gene_order.numel()) if gene_order is not None else int(
                    gene_index.max().item() + 1
                )
                order_key = order_key.masked_fill(~valid, sentinel)
                local_order = torch.argsort(order_key, dim=1, stable=True)
            else:
                local_order = offsets.unsqueeze(0).expand(batch_size, -1)

            ordered_valid = valid.gather(1, local_order)
            ordered = local_tokens.gather(
                1, local_order.unsqueeze(-1).expand(-1, -1, self.token_dim)
            ).masked_fill(~ordered_valid.unsqueeze(-1), 0.0)
            grid = ordered.reshape(batch_size, height, width, self.token_dim)
            mixed = self.mixer(
                grid.reshape(batch_size, slots, self.token_dim),
                mode="none",
                valid_tokens=ordered_valid,
            )
            mixed = mixed.reshape(batch_size, slots, self.token_dim)
            inverse = torch.empty_like(local_order)
            inverse.scatter_(
                1, local_order, offsets.unsqueeze(0).expand(batch_size, -1)
            )
            restored = mixed.gather(
                1, inverse.unsqueeze(-1).expand(-1, -1, self.token_dim)
            )
            updated = local_tokens + gate * (restored - local_tokens)
            valid_flat = valid.reshape(-1)
            mixed_result.index_copy_(
                0,
                flat_positions[valid_flat],
                updated.reshape(-1, self.token_dim)[valid_flat],
            )
        result = tokens.clone()
        result.index_copy_(0, source_positions, mixed_result)
        return result


class ScUniFlowSpatialAdapter(nn.Module):
    """Section-local gated spatial context branch for Stage2.

    Neighbor context is computed from ``edge_index`` when supplied.  Edges
    connecting different ``batch_index`` sections are rejected instead of
    silently leaking information across slices.  The branch returns a latent
    residual and leaves the existing graph/niche hierarchy available to the
    caller.
    """

    def __init__(
        self,
        latent_dim: int,
        coordinate_dim: int = 2,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if not isinstance(coordinate_dim, int) or coordinate_dim < 1:
            raise ValueError("coordinate_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.latent_dim = latent_dim
        self.coordinate_dim = coordinate_dim
        self.context_net = nn.Sequential(
            nn.LayerNorm(2 * latent_dim + coordinate_dim),
            nn.Linear(2 * latent_dim + coordinate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.raw_gate = _bounded_gate_parameter(gate_init)

    def effective_gate(self) -> Tensor:
        return torch.tanh(self.raw_gate)

    def _neighbor_mean(
        self, z: Tensor, batch_index: Tensor, edge_index: Tensor | None
    ) -> Tensor:
        context = torch.zeros_like(z)
        if edge_index is None or edge_index.numel() == 0:
            return context
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        src, dst = edge_index.to(device=z.device, dtype=torch.long)
        if torch.any(src < 0) or torch.any(src >= z.shape[0]) or torch.any(dst < 0) or torch.any(dst >= z.shape[0]):
            raise ValueError("edge_index contains an out-of-range node")
        if torch.any(batch_index[src] != batch_index[dst]):
            raise ValueError("spatial adapter received an edge crossing section boundaries")
        context.index_add_(0, dst, z[src])
        counts = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        counts.index_add_(0, dst, torch.ones_like(dst, dtype=z.dtype))
        return context / counts.clamp_min(1.0).unsqueeze(-1)

    def forward(
        self,
        z: Tensor,
        spatial_coords: Tensor | None,
        batch_index: Tensor,
        edge_index: Tensor | None = None,
        *,
        return_context: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        _validate_2d_state(z, "z", self.latent_dim)
        if batch_index.shape != (z.shape[0],):
            raise ValueError("batch_index must have shape [B]")
        if spatial_coords is None:
            # No spatial observation means this Stage-2 branch is undefined;
            # fail closed to an exact identity instead of learning a hidden
            # metadata-only shortcut during molecular/temporal training.
            if return_context:
                return z, torch.zeros_like(z)
            return z
        else:
            if spatial_coords.shape != (z.shape[0], self.coordinate_dim):
                raise ValueError(
                    f"spatial_coords must have shape [{z.shape[0]}, {self.coordinate_dim}]"
                )
            coords = spatial_coords.to(device=z.device, dtype=z.dtype)
            if not torch.isfinite(coords).all():
                raise ValueError("spatial_coords must be finite")
        neighbor = self._neighbor_mean(z, batch_index.to(device=z.device), edge_index)
        delta = self.context_net(torch.cat((z, neighbor, coords), dim=-1))
        gate = self.effective_gate().to(dtype=z.dtype, device=z.device)
        state = z + gate * delta
        if return_context:
            return state, neighbor
        return state


__all__ = [
    "ScFiTTemporalAdapter",
    "ScFiTTemporalOutput",
    "ScUniFlowSparseGeneAdapter",
    "ScUniFlowSpatialAdapter",
]
