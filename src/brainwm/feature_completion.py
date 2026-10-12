"""Small ordered feature-completion adapter used by the scUniFlow study.

The adapter intentionally lives next to the BrainWM modules but does not alter
``BrainWorldModel``'s default constructor.  It is a controlled candidate for
cell-internal masked feature completion: a fixed feature identity/value/mask
embedding is mixed by :class:`OrderedGeneMixer`, then decoded independently
at each feature.  The order can be switched between a supplied biological
permutation, a seeded random permutation, and identity without changing the
parameter count or the input data.

This is not a cell-cell spatial graph and it does not implement a spatial or
temporal rollout.  Keeping the adapter explicit makes that boundary visible in
experiments and prevents a feature-imputation result from being reported as a
Stage-2 dynamics result.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .ordered_mixer import OrderedGeneMixer, OrderMode


class OrderedFeatureCompletionAdapter(nn.Module):
    """Predict masked continuous feature values from an ordered token grid.

    Parameters
    ----------
    n_features:
        Number of features in the fixed panel.  The feature index is local to
        that panel; callers must record the canonical-to-local mapping.
    token_dim:
        Width of the feature token and ordered mixer.
    grid_shape:
        Rectangular grid capacity.  It must have at least ``n_features`` slots.
    order_mode:
        Default order control.  ``forward`` can override it for matched A/B
        experiments.
    """

    def __init__(
        self,
        n_features: int,
        *,
        token_dim: int = 64,
        grid_shape: tuple[int, int] | None = None,
        order_mode: OrderMode | str = "biological",
        random_seed: int | None = None,
        ffn_multiplier: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(n_features, int) or isinstance(n_features, bool):
            raise TypeError("n_features must be an integer")
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if not isinstance(token_dim, int) or isinstance(token_dim, bool):
            raise TypeError("token_dim must be an integer")
        if token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.n_features = n_features
        self.token_dim = token_dim
        self.feature_embedding = nn.Embedding(n_features, token_dim)
        self.value_embedding = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.mask_embedding = nn.Linear(1, token_dim)
        self.mixer = OrderedGeneMixer(
            token_dim,
            order_mode=order_mode,
            random_seed=random_seed,
            grid_shape=grid_shape,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )

    def forward(
        self,
        values: Tensor,
        observed: Tensor,
        order: Tensor | None = None,
        *,
        mode: OrderMode | str | None = None,
        seed: int | None = None,
    ) -> Tensor:
        """Return one prediction per feature, preserving panel order.

        ``values`` contains the target panel values.  The adapter zeroes the
        unobserved/masked positions before embedding them; ``observed`` is a
        boolean tensor with ``True`` at visible positions.  The returned
        tensor has shape ``[batch, n_features]``.
        """

        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(
                f"values must have shape [B,{self.n_features}], got {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError("values must use a floating-point dtype")
        if observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError("observed must be boolean and match values shape")
        safe_values = values.masked_fill(~observed, 0.0)
        feature_ids = torch.arange(
            self.n_features, dtype=torch.long, device=values.device
        )
        identity = self.feature_embedding(feature_ids).unsqueeze(0)
        identity = identity.expand(values.shape[0], -1, -1)
        tokens = identity + self.value_embedding(safe_values.unsqueeze(-1))
        tokens = tokens + self.mask_embedding(observed.to(values.dtype).unsqueeze(-1))
        mixed = self.mixer(tokens, order, mode=mode, seed=seed)
        return self.output(mixed).squeeze(-1)


def masked_huber_loss(
    prediction: Tensor,
    target: Tensor,
    masked: Tensor,
    *,
    delta: float = 0.5,
    top_weight: float = 1.0,
    top_k: int = 20,
) -> Tensor:
    """Huber loss on masked entries with optional high-expression weighting.

    The top-expression weighting follows the scUniFlow training convention,
    but is intentionally optional and computed from the target only.  The
    function returns a finite zero scalar when a caller supplies an empty mask.
    """

    if prediction.shape != target.shape or prediction.shape != masked.shape:
        raise ValueError("prediction, target, and masked must have the same shape")
    if masked.dtype != torch.bool:
        raise ValueError("masked must be boolean")
    if delta <= 0:
        raise ValueError("delta must be positive")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if not torch.any(masked):
        return prediction.sum() * 0.0
    error = torch.nn.functional.smooth_l1_loss(
        prediction, target, beta=delta, reduction="none"
    )
    weights = torch.ones_like(error)
    if top_weight != 1.0 and top_k > 0:
        for row in range(target.shape[0]):
            positions = torch.nonzero(masked[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            count = min(int(top_k), int(positions.numel()))
            selected = positions[torch.topk(target[row, positions], count).indices]
            weights[row, selected] = float(top_weight)
    return (error[masked] * weights[masked]).sum() / weights[masked].sum().clamp_min(1.0)


__all__ = ["OrderedFeatureCompletionAdapter", "masked_huber_loss"]
