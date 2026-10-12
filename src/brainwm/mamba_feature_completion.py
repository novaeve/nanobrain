"""Optional real-Mamba2 adapter for ordered masked feature completion.

This module mirrors the row/column Mamba2 topology used by scUniFlow while
keeping BrainWM's existing feature-completion input/output contract.  The
``mamba_ssm`` dependency is imported only when a real adapter is constructed,
so the rest of BrainWM remains importable in environments without the CUDA
extension.

The adapter models genes within one cell.  It has no cell-cell graph, spatial
coordinates, temporal transition, niche state, or counterfactual rollout and
must not be reported as a Stage2 spatial-dynamics model.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .ordered_mixer import OrderMode, inverse_permutation, make_order

MambaFactory = Callable[..., nn.Module]


def _load_mamba2() -> MambaFactory:
    try:
        from mamba_ssm import Mamba2
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "real Mamba2 requires a working mamba-ssm/causal-conv1d CUDA "
            "installation in the active BrainWM environment"
        ) from exc
    return Mamba2


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_mode(mode: OrderMode | str) -> str:
    aliases = {
        "biological": "biological",
        "bio": "biological",
        "random": "random",
        "none": "none",
        "identity": "none",
        "no_order": "none",
        "no-order": "none",
    }
    if not isinstance(mode, str):
        raise TypeError("order mode must be a string")
    try:
        return aliases[mode.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported order mode {mode!r}") from exc


class _BidirectionalMamba2(nn.Module):
    """Independent forward/backward Mamba2 scans with nonlinear fusion."""

    def __init__(
        self,
        dim: int,
        *,
        state_dim: int,
        sequence_length: int,
        conv_width: int,
        head_dim: int,
        mamba_factory: MambaFactory,
    ) -> None:
        super().__init__()
        kwargs = {
            "d_model": dim,
            "d_state": state_dim,
            "d_conv": conv_width,
            "chunk_size": sequence_length,
            "headdim": head_dim,
        }
        self.forward_scan = mamba_factory(**kwargs)
        self.backward_scan = mamba_factory(**kwargs)
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("Mamba sequence must have shape [N,L,D]")
        forward = self.forward_scan(sequence)
        backward = self.backward_scan(sequence.flip(1)).flip(1)
        return self.fusion(torch.cat((forward, backward), dim=-1))


class Mamba2GridBlock(nn.Module):
    """Pre-norm residual row/column Mamba2 block over a square grid."""

    def __init__(
        self,
        dim: int,
        *,
        grid_size: int,
        row_state_dim: int,
        column_state_dim: int,
        conv_width: int,
        head_dim: int,
        ffn_multiplier: float,
        dropout: float,
        mamba_factory: MambaFactory,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(ffn_multiplier)) or ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be finite and positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.dim = dim
        self.grid_size = grid_size
        self.row_norm = nn.LayerNorm(dim)
        self.column_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.row_mamba = _BidirectionalMamba2(
            dim,
            state_dim=row_state_dim,
            sequence_length=grid_size,
            conv_width=conv_width,
            head_dim=head_dim,
            mamba_factory=mamba_factory,
        )
        self.column_mamba = _BidirectionalMamba2(
            dim,
            state_dim=column_state_dim,
            sequence_length=grid_size,
            conv_width=conv_width,
            head_dim=head_dim,
            mamba_factory=mamba_factory,
        )
        hidden_dim = max(1, round(dim * ffn_multiplier))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, grid: Tensor) -> Tensor:
        expected = (self.grid_size, self.grid_size, self.dim)
        if grid.ndim != 4 or tuple(grid.shape[1:]) != expected:
            raise ValueError(
                f"grid must have shape [B,{self.grid_size},{self.grid_size},{self.dim}]"
            )
        batch = grid.shape[0]
        row_input = self.row_norm(grid).reshape(
            batch * self.grid_size, self.grid_size, self.dim
        )
        row_update = self.row_mamba(row_input).reshape(
            batch, self.grid_size, self.grid_size, self.dim
        )
        grid = grid + row_update

        column_input = (
            self.column_norm(grid)
            .permute(0, 2, 1, 3)
            .reshape(batch * self.grid_size, self.grid_size, self.dim)
        )
        column_update = (
            self.column_mamba(column_input)
            .reshape(batch, self.grid_size, self.grid_size, self.dim)
            .permute(0, 2, 1, 3)
        )
        grid = grid + column_update
        return grid + self.ffn(self.ffn_norm(grid))


class MambaFeatureCompletionAdapter(nn.Module):
    """Predict masked feature values with real row/column Mamba2 blocks.

    ``order[j]`` is the original feature placed at ordered position ``j``.
    The returned prediction is always scattered back to original panel order.
    Tail slots in the square grid are zero placeholders, matching the public
    scUniFlow topology; they are not additional genes.
    """

    def __init__(
        self,
        n_features: int,
        *,
        token_dim: int = 64,
        grid_size: int | None = None,
        n_blocks: int = 2,
        row_state_dim: int = 64,
        column_state_dim: int = 16,
        conv_width: int = 4,
        head_dim: int = 64,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.0,
        order_mode: OrderMode | str = "biological",
        random_seed: int | None = None,
        mamba_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        self.n_features = _positive_integer(n_features, "n_features")
        self.token_dim = _positive_integer(token_dim, "token_dim")
        self.grid_size = (
            math.ceil(math.sqrt(self.n_features))
            if grid_size is None
            else _positive_integer(grid_size, "grid_size")
        )
        if self.grid_size * self.grid_size < self.n_features:
            raise ValueError("grid_size^2 must be at least n_features")
        self.n_blocks = _positive_integer(n_blocks, "n_blocks")
        _positive_integer(row_state_dim, "row_state_dim")
        _positive_integer(column_state_dim, "column_state_dim")
        _positive_integer(conv_width, "conv_width")
        _positive_integer(head_dim, "head_dim")
        if self.token_dim % head_dim != 0:
            raise ValueError("token_dim must be divisible by head_dim")
        if random_seed is not None and (
            not isinstance(random_seed, int) or isinstance(random_seed, bool)
        ):
            raise ValueError("random_seed must be an integer or None")
        self.order_mode = _canonical_mode(order_mode)
        self.random_seed = random_seed

        factory = _load_mamba2() if mamba_factory is None else mamba_factory
        self.feature_embedding = nn.Embedding(self.n_features, self.token_dim)
        self.value_embedding = nn.Sequential(
            nn.Linear(1, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.mask_embedding = nn.Linear(1, self.token_dim)
        self.blocks = nn.ModuleList(
            Mamba2GridBlock(
                self.token_dim,
                grid_size=self.grid_size,
                row_state_dim=row_state_dim,
                column_state_dim=column_state_dim,
                conv_width=conv_width,
                head_dim=head_dim,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                mamba_factory=factory,
            )
            for _ in range(self.n_blocks)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
        )

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        order: Tensor | None = None,
        *,
        mode: OrderMode | str | None = None,
        seed: int | None = None,
        preserve_unobserved_values: bool = False,
    ) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(
                f"values must have shape [B,{self.n_features}], got {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError("values must use a floating-point dtype")
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("observed must be boolean and match values shape")
        if not isinstance(preserve_unobserved_values, bool):
            raise TypeError("preserve_unobserved_values must be boolean")

        selected_mode = self.order_mode if mode is None else _canonical_mode(mode)
        active_order = make_order(
            self.n_features,
            mode=selected_mode,
            biological_order=order,
            seed=self.random_seed if seed is None else seed,
            device=values.device,
        )
        inverse = inverse_permutation(active_order)
        safe_values = (
            values
            if preserve_unobserved_values
            else values.masked_fill(~observed, 0.0)
        )
        feature_ids = torch.arange(
            self.n_features, dtype=torch.long, device=values.device
        )
        tokens = self.feature_embedding(feature_ids).unsqueeze(0)
        tokens = tokens.expand(values.shape[0], -1, -1)
        tokens = tokens + self.value_embedding(safe_values.unsqueeze(-1))
        tokens = tokens + self.mask_embedding(observed.to(values.dtype).unsqueeze(-1))
        ordered = tokens.index_select(1, active_order)

        slots = self.grid_size * self.grid_size
        padded = tokens.new_zeros((values.shape[0], slots, self.token_dim))
        padded[:, : self.n_features] = ordered
        grid = padded.view(
            values.shape[0], self.grid_size, self.grid_size, self.token_dim
        )
        for block in self.blocks:
            grid = block(grid)
        mixed_ordered = grid.reshape(values.shape[0], slots, self.token_dim)[
            :, : self.n_features
        ]
        mixed = mixed_ordered.index_select(1, inverse)
        return self.output(mixed).squeeze(-1)


class HilbertRoutedMambaFeatureCompletionAdapter(MambaFeatureCompletionAdapter):
    """Mamba2 adapter that preserves the hard router's occupied grid slots.

    The base adapter packs ordered genes into the first ``G`` row-major slots.
    SU-05 instead assigns the ordered genes to the first ``G`` slots visited by
    the fixed Hilbert traversal, including its non-contiguous tail holes.
    """

    def __init__(self, *args, occupied_slots: Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        slots = torch.as_tensor(occupied_slots, dtype=torch.long)
        if slots.ndim != 1 or slots.numel() != self.n_features:
            raise ValueError(
                f"occupied_slots must have shape [{self.n_features}]"
            )
        if torch.any(slots < 0) or torch.any(slots >= self.grid_size**2):
            raise ValueError("occupied_slots contains an index outside the grid")
        if torch.unique(slots).numel() != self.n_features:
            raise ValueError("occupied_slots must not contain duplicates")
        self.register_buffer("occupied_slots", slots.clone())

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        order: Tensor | None = None,
        *,
        mode: OrderMode | str | None = None,
        seed: int | None = None,
        preserve_unobserved_values: bool = False,
    ) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(
                f"values must have shape [B,{self.n_features}], got {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError("values must use a floating-point dtype")
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("observed must be boolean and match values shape")
        if not isinstance(preserve_unobserved_values, bool):
            raise TypeError("preserve_unobserved_values must be boolean")

        selected_mode = self.order_mode if mode is None else _canonical_mode(mode)
        active_order = make_order(
            self.n_features,
            mode=selected_mode,
            biological_order=order,
            seed=self.random_seed if seed is None else seed,
            device=values.device,
        )
        inverse = inverse_permutation(active_order)
        safe_values = (
            values
            if preserve_unobserved_values
            else values.masked_fill(~observed, 0.0)
        )
        feature_ids = torch.arange(
            self.n_features, dtype=torch.long, device=values.device
        )
        tokens = self.feature_embedding(feature_ids).unsqueeze(0)
        tokens = tokens.expand(values.shape[0], -1, -1)
        tokens = tokens + self.value_embedding(safe_values.unsqueeze(-1))
        tokens = tokens + self.mask_embedding(observed.to(values.dtype).unsqueeze(-1))
        ordered = tokens.index_select(1, active_order)

        slots = self.grid_size * self.grid_size
        padded = tokens.new_zeros((values.shape[0], slots, self.token_dim))
        padded.index_copy_(1, self.occupied_slots, ordered)
        grid = padded.view(
            values.shape[0], self.grid_size, self.grid_size, self.token_dim
        )
        for block in self.blocks:
            grid = block(grid)
        mixed_ordered = grid.reshape(
            values.shape[0], slots, self.token_dim
        ).index_select(1, self.occupied_slots)
        mixed = mixed_ordered.index_select(1, inverse)
        return self.output(mixed).squeeze(-1)


__all__ = [
    "HilbertRoutedMambaFeatureCompletionAdapter",
    "Mamba2GridBlock",
    "MambaFeatureCompletionAdapter",
]
