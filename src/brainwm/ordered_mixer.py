"""Candidate ordered gene mixer for BrainWM experiments.

This module is deliberately independent of :mod:`mamba_ssm` and CUDA
extensions.  It provides the *layout* part of an scUniFlow/scBrain-inspired
candidate while keeping the existing BrainWM encoder unchanged:

``original tokens -> permutation gather -> row/column scans -> inverse scatter``

The scans are stable first-order exponential recurrences, used here as a
small SSM-like mixer.  They are not an implementation of Mamba and should not
be described as one.  The module is intended for controlled ablations and
smoke tests; callers decide whether the permutation is a biological order, a
random control, or the identity (no-order control).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

OrderMode = Literal["biological", "random", "none", "no_order"]

_ORDER_MODE_ALIASES = {
    "biological": "biological",
    "bio": "biological",
    "random": "random",
    "none": "none",
    "identity": "none",
    "no_order": "none",
    "no-order": "none",
}


def _canonical_order_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("order mode must be a string")
    canonical = _ORDER_MODE_ALIASES.get(mode.lower())
    if canonical is None:
        choices = ", ".join(sorted(_ORDER_MODE_ALIASES))
        raise ValueError(f"unsupported order mode {mode!r}; choose one of {choices}")
    return canonical


def validate_permutation(order: Tensor, length: int, *, name: str = "order") -> Tensor:
    """Validate and return a one-dimensional permutation as ``torch.long``.

    ``order[j]`` is the original-token index placed at ordered position
    ``j``.  The returned tensor is a view/cast only; it is not sorted or
    otherwise repaired.  Strict validation is important because a duplicate
    index would make inverse scattering ambiguous and silently drop a token.
    """

    if not isinstance(order, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("permutation length must be a positive integer")
    if order.ndim != 1:
        raise ValueError(f"{name} must have shape [T], got {tuple(order.shape)}")
    if order.numel() != length:
        raise ValueError(
            f"{name} must contain exactly {length} entries, got {order.numel()}"
        )
    if order.dtype == torch.bool or order.is_floating_point() or order.is_complex():
        raise TypeError(f"{name} must use an integer dtype, got {order.dtype}")
    # Convert before checking so int8/int16 values are handled consistently.
    checked = order.to(dtype=torch.long)
    if torch.any(checked < 0) or torch.any(checked >= length):
        raise ValueError(f"{name} contains an index outside [0, {length})")
    if torch.unique(checked).numel() != length:
        raise ValueError(f"{name} must be a permutation without duplicate indices")
    return checked


def inverse_permutation(order: Tensor) -> Tensor:
    """Return ``inverse`` such that ``inverse[order] == arange(T)``."""

    if not isinstance(order, Tensor):
        raise TypeError("order must be a torch.Tensor")
    checked = validate_permutation(order, int(order.numel()))
    inverse = torch.empty_like(checked)
    inverse[checked] = torch.arange(
        checked.numel(), dtype=torch.long, device=checked.device
    )
    return inverse


def make_order(
    length: int,
    *,
    mode: OrderMode | str = "biological",
    biological_order: Tensor | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build a deterministic order for one of the mixer controls.

    Parameters
    ----------
    length:
        Number of input tokens.  It must be positive.
    mode:
        ``"biological"`` uses ``biological_order``; ``"none"`` uses the
        identity order; ``"random"`` applies a seeded permutation to the
        supplied biological order (or to identity when none is supplied).
    biological_order:
        Ordered-position-to-original-index permutation for the biological
        condition.
    seed:
        Required for the random condition.  A local ``torch.Generator`` is
        used, so constructing an order never advances global RNG state.
    device:
        Optional output device.
    """

    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    canonical = _canonical_order_mode(mode)
    target_device = torch.device(device) if device is not None else None

    if biological_order is None:
        base = torch.arange(length, dtype=torch.long, device=target_device)
    else:
        base = validate_permutation(biological_order, length)
        if target_device is not None:
            base = base.to(target_device)

    if canonical == "biological":
        if biological_order is None:
            raise ValueError("biological mode requires biological_order")
        return base
    if canonical == "none":
        return torch.arange(length, dtype=torch.long, device=base.device)

    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("random mode requires an explicit integer seed")
    generator = torch.Generator(device=base.device)
    generator.manual_seed(seed)
    shuffle = torch.randperm(length, generator=generator, device=base.device)
    return base.index_select(0, shuffle)


def _balanced_grid_shape(length: int) -> tuple[int, int]:
    """Return a near-square rectangular grid with enough slots for ``length``."""

    height = max(1, math.ceil(math.sqrt(length)))
    width = math.ceil(length / height)
    return height, width


def _validate_grid_shape(
    grid_shape: tuple[int, int] | None, length: int
) -> tuple[int, int]:
    if grid_shape is None:
        return _balanced_grid_shape(length)
    if (
        not isinstance(grid_shape, tuple)
        or len(grid_shape) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in grid_shape
        )
    ):
        raise ValueError("grid_shape must be a tuple of two positive integers")
    height, width = grid_shape
    if height * width < length:
        raise ValueError(
            f"grid_shape {grid_shape} has {height * width} slots for {length} tokens"
        )
    return height, width


class _BidirectionalExponentialScan(nn.Module):
    """A depthwise first-order bidirectional state-space-like scan."""

    def __init__(self, dim: int, initial_decay: float = 0.8):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not 0.0 < initial_decay < 1.0:
            raise ValueError("initial_decay must lie strictly between zero and one")
        logit = math.log(initial_decay / (1.0 - initial_decay))
        self.decay_logit = nn.Parameter(torch.full((dim,), logit))
        self.fuse = nn.Linear(2 * dim, dim)

    @staticmethod
    def _scan(sequence: Tensor, decay: Tensor, valid: Tensor | None = None) -> Tensor:
        # sequence: [N, L, D].  Evaluate the first-order recurrence in short
        # vectorized blocks.  The old implementation performed one Python/GPU
        # dispatch per token, which made large sparse CSR rows impractical;
        # block prefix products preserve the same recurrence while reducing
        # dispatches by roughly the block length.
        if sequence.shape[1] == 0:
            return sequence
        if valid is not None and (
            valid.shape != sequence.shape[:2] or valid.dtype != torch.bool
        ):
            raise ValueError("scan valid mask must have shape [N,L] and bool dtype")
        # Prefix products are numerically well behaved when bounded to a
        # modest block.  The sigmoid parameterization normally keeps decay in
        # the interior, but explicit bounds avoid overflow in exp(-log_prefix)
        # during early optimization or mixed-precision execution.
        block_size = 64
        work = sequence.float()
        retain = decay.to(device=sequence.device, dtype=torch.float32).view(1, -1)
        retain = retain.clamp(1.0e-4, 1.0 - 1.0e-4)
        state = torch.zeros_like(work[:, 0])
        outputs: list[Tensor] = []
        for start in range(0, sequence.shape[1], block_size):
            stop = min(start + block_size, sequence.shape[1])
            block = work[:, start:stop]
            if valid is None:
                block_valid = None
            else:
                block_valid = valid[:, start:stop]
            if block_valid is None:
                effective_retain = retain.expand(block.shape[0], block.shape[1], -1)
                effective_inject = 1.0 - effective_retain
            else:
                active = block_valid[:, :, None]
                effective_retain = torch.where(
                    active, retain, torch.ones_like(retain)
                )
                effective_inject = torch.where(
                    active, 1.0 - retain, torch.zeros_like(retain)
                )
            log_prefix = torch.cumsum(torch.log(effective_retain), dim=1)
            prefix = torch.exp(log_prefix)
            inverse_prefix = torch.exp(-log_prefix)
            raw_local = prefix * (
                state[:, None, :]
                + torch.cumsum(effective_inject * block * inverse_prefix, dim=1)
            )
            local = raw_local
            if block_valid is not None:
                local = local.masked_fill(~block_valid[:, :, None], 0.0)
                # Invalid positions use retain=1/inject=0, so the final raw
                # state already carries the last valid state through any
                # padding or interleaved mask entries.
                state = raw_local[:, -1]
            else:
                state = raw_local[:, -1]
            outputs.append(local)
        return torch.cat(outputs, dim=1).to(dtype=sequence.dtype)

    def forward(self, sequence: Tensor, valid: Tensor | None = None) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("scan input must have shape [N, L, D]")
        decay = torch.sigmoid(self.decay_logit)
        forward = self._scan(sequence, decay, valid)
        reverse_valid = valid.flip(1) if valid is not None else None
        backward = self._scan(sequence.flip(1), decay, reverse_valid).flip(1)
        return self.fuse(torch.cat((forward, backward), dim=-1))


class RowColumnSSMMixer(nn.Module):
    """Lightweight factorized row/column mixer over a 2-D token grid.

    The implementation follows the *shape* of a row-scan/column-scan block,
    but uses local exponential recurrences instead of Mamba.  The input and
    output are ``[B,H,W,D]``.  ``valid_tokens`` may be a shared flat
    ``[H*W]`` mask or a per-sample ``[B,H*W]`` mask.  The latter is used for
    bucketed sparse rows whose lengths differ but share one rectangular grid.
    """

    def __init__(
        self,
        dim: int,
        *,
        ffn_multiplier: float = 2.0,
        dropout: float = 0.0,
        initial_decay_row: float = 0.8,
        initial_decay_col: float = 0.8,
    ):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not math.isfinite(float(ffn_multiplier)) or ffn_multiplier < 0:
            raise ValueError("ffn_multiplier must be finite and nonnegative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.dim = dim
        self.norm_row = nn.LayerNorm(dim)
        self.norm_col = nn.LayerNorm(dim)
        self.row_scan = _BidirectionalExponentialScan(dim, initial_decay_row)
        self.col_scan = _BidirectionalExponentialScan(dim, initial_decay_col)
        self.ffn_enabled = ffn_multiplier > 0
        hidden = max(1, round(dim * ffn_multiplier))
        self.ffn = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
                nn.Dropout(dropout),
            )
            if ffn_multiplier > 0
            else nn.Identity()
        )

    def forward(self, grid: Tensor, valid_tokens: Tensor | None = None) -> Tensor:
        if grid.ndim != 4 or grid.shape[-1] != self.dim:
            raise ValueError(
                f"grid must have shape [B,H,W,{self.dim}], got {tuple(grid.shape)}"
            )
        batch, height, width, _ = grid.shape
        if height < 1 or width < 1:
            raise ValueError("grid height and width must be positive")
        if valid_tokens is None:
            valid = torch.ones(height * width, dtype=torch.bool, device=grid.device)
            valid_grid = valid.reshape(height, width)
            sample_valid_grid = valid_grid.unsqueeze(0).expand(batch, -1, -1)
        else:
            if valid_tokens.ndim == 1:
                if valid_tokens.numel() != height * width:
                    raise ValueError("valid_tokens must have shape [H*W] or [B,H*W]")
                valid = valid_tokens.to(device=grid.device, dtype=torch.bool)
                valid_grid = valid.reshape(height, width)
                sample_valid_grid = valid_grid.unsqueeze(0).expand(batch, -1, -1)
            elif valid_tokens.ndim == 2:
                if valid_tokens.shape != (batch, height * width):
                    raise ValueError("valid_tokens must have shape [H*W] or [B,H*W]")
                sample_valid_grid = valid_tokens.to(device=grid.device, dtype=torch.bool).reshape(
                    batch, height, width
                )
                # Keep a shared view for compatibility with callers that inspect
                # the flattened layout. Per-sample masks are used below.
                valid_grid = sample_valid_grid[0]
            else:
                raise ValueError("valid_tokens must have shape [H*W] or [B,H*W]")
        # Padding occupies the tail of the flattened grid.  Masking before and
        # after each residual branch keeps padded slots from contributing to
        # later branches and guarantees zero output for those slots.
        x = grid.masked_fill(~sample_valid_grid[:, :, :, None], 0.0)
        row_input = (
            self.norm_row(x)
            .masked_fill(~sample_valid_grid[:, :, :, None], 0.0)
            .reshape(batch * height, width, self.dim)
        )
        row_valid = sample_valid_grid.reshape(batch * height, width)
        row_update = self.row_scan(row_input, row_valid).reshape(
            batch, height, width, self.dim
        )
        x = (x + row_update).masked_fill(~sample_valid_grid[:, :, :, None], 0.0)

        col_input = (
            self.norm_col(x)
            .masked_fill(~sample_valid_grid[:, :, :, None], 0.0)
            .permute(0, 2, 1, 3)
            .reshape(batch * width, height, self.dim)
        )
        col_valid = sample_valid_grid.transpose(1, 2).reshape(batch * width, height)
        col_update = (
            self.col_scan(col_input, col_valid)
            .reshape(batch, width, height, self.dim)
            .permute(0, 2, 1, 3)
        )
        x = (x + col_update).masked_fill(~sample_valid_grid[:, :, :, None], 0.0)
        if self.ffn_enabled:
            x = x + self.ffn(x)
        x = x.masked_fill(~sample_valid_grid[:, :, :, None], 0.0)
        return x


class OrderedGeneMixer(nn.Module):
    """Gather, mix in ordered space, and scatter token features back.

    Parameters
    ----------
    dim:
        Feature width ``D``.
    order_mode:
        Default control.  ``biological`` requires a permutation at ``forward``
        time; ``none`` uses identity; ``random`` uses ``random_seed``.
    random_seed:
        Explicit seed used by the random control.  A per-call ``seed`` can
        override it.  No global RNG state is touched.
    grid_shape:
        Optional ``(H, W)``.  If omitted, a near-square rectangular grid is
        selected and tail slots are masked.
    """

    def __init__(
        self,
        dim: int,
        *,
        order_mode: OrderMode | str = "biological",
        random_seed: int | None = None,
        grid_shape: tuple[int, int] | None = None,
        ffn_multiplier: float = 2.0,
        dropout: float = 0.0,
        initial_decay_row: float = 0.8,
        initial_decay_col: float = 0.8,
    ):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.order_mode = _canonical_order_mode(order_mode)
        if random_seed is not None and (
            isinstance(random_seed, bool) or not isinstance(random_seed, int)
        ):
            raise ValueError("random_seed must be an integer or None")
        self.random_seed = random_seed
        if grid_shape is not None:
            # Validate the form at construction time; the capacity check is
            # repeated at forward time because T is supplied by the caller.
            _validate_grid_shape(grid_shape, 1)
        self.grid_shape = grid_shape
        self.mixer = RowColumnSSMMixer(
            dim,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            initial_decay_row=initial_decay_row,
            initial_decay_col=initial_decay_col,
        )

    def _resolve_order(
        self,
        length: int,
        order: Tensor | None,
        mode: str,
        seed: int | None,
        device: torch.device,
    ) -> Tensor:
        if mode == "biological" and order is None:
            raise ValueError("biological mode requires order with shape [T]")
        return make_order(
            length,
            mode=mode,
            biological_order=order,
            seed=self.random_seed if seed is None else seed,
            device=device,
        )

    def forward(
        self,
        x: Tensor,
        order: Tensor | None = None,
        *,
        mode: OrderMode | str | None = None,
        seed: int | None = None,
        valid_tokens: Tensor | None = None,
    ) -> Tensor:
        """Mix ``x`` while restoring its original token positions.

        ``order`` follows ordered-position-to-original-token semantics.  For
        example, ``order=[2, 0, 1]`` means original token 2 is processed first,
        then token 0, then token 1.  The result always has the same shape and
        token indexing as ``x``.
        """

        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B,T,D], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"x must use a floating-point dtype, got {x.dtype}")
        batch, length, dim = x.shape
        if length < 1:
            raise ValueError("x must contain at least one token")
        if dim != self.dim:
            raise ValueError(f"x has feature width {dim}, expected {self.dim}")
        selected_mode = self.order_mode if mode is None else _canonical_order_mode(mode)
        active_order = self._resolve_order(length, order, selected_mode, seed, x.device)
        # ``inverse_permutation`` performs the final strict validation after
        # any device conversion performed by ``make_order``.
        inverse = inverse_permutation(active_order)

        ordered = x.index_select(1, active_order)
        if valid_tokens is None:
            valid_ordered = None
        else:
            if valid_tokens.ndim == 1:
                if valid_tokens.numel() != length:
                    raise ValueError("valid_tokens must have shape [T] or [B,T]")
                valid_input = valid_tokens.to(device=x.device, dtype=torch.bool)
                valid_ordered = valid_input.index_select(0, active_order)
            elif valid_tokens.ndim == 2:
                if valid_tokens.shape != (batch, length):
                    raise ValueError("valid_tokens must have shape [T] or [B,T]")
                valid_input = valid_tokens.to(device=x.device, dtype=torch.bool)
                valid_ordered = valid_input.index_select(1, active_order)
            else:
                raise ValueError("valid_tokens must have shape [T] or [B,T]")
        height, width = _validate_grid_shape(self.grid_shape, length)
        slots = height * width
        grid = x.new_zeros((batch, slots, dim))
        grid[:, :length] = ordered
        if valid_ordered is None:
            valid = torch.zeros(slots, dtype=torch.bool, device=x.device)
            valid[:length] = True
        elif valid_ordered.ndim == 1:
            valid = torch.zeros(slots, dtype=torch.bool, device=x.device)
            valid[:length] = valid_ordered
        else:
            valid = torch.zeros(batch, slots, dtype=torch.bool, device=x.device)
            valid[:, :length] = valid_ordered
        mixed = self.mixer(grid.view(batch, height, width, dim), valid)
        mixed_ordered = mixed.reshape(batch, slots, dim)[:, :length]
        return mixed_ordered.index_select(1, inverse)


# A shorter alias is useful in experiment configs while preserving the more
# explicit class name for readers.
OrderedMixer = OrderedGeneMixer


__all__ = [
    "OrderMode",
    "OrderedGeneMixer",
    "OrderedMixer",
    "RowColumnSSMMixer",
    "inverse_permutation",
    "make_order",
    "validate_permutation",
]
