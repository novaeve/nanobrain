"""Cross-modal alignment for zero-shot cell-type annotation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from operator import index as _index

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CrossModalAnnotationAdapter(nn.Module):
    """Map molecular and spatial frozen features into one annotation space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(2)
        ])
        self.classifier = nn.Linear(output_dim, n_classes)

    def forward(self, features: Tensor, domain_id: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [B, {self.input_dim}]")
        if domain_id.shape != (features.shape[0],):
            raise ValueError("domain_id must have shape [B]")
        if torch.any((domain_id < 0) | (domain_id > 1)):
            raise ValueError("domain_id values must be 0 (molecular) or 1 (spatial)")
        embedding = features.new_empty((features.shape[0], self.output_dim))
        for domain, stem in enumerate(self.stems):
            selected = domain_id == domain
            if torch.any(selected):
                embedding[selected] = stem(features[selected])
        embedding = F.normalize(embedding.float(), dim=-1)
        return embedding, self.classifier(embedding)


def cross_modal_supervised_contrastive_loss(
    embedding: Tensor,
    label: Tensor,
    domain_id: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """InfoNCE whose positives share a label but come from another modality."""
    if (
        embedding.ndim != 2
        or label.shape != (embedding.shape[0],)
        or domain_id.shape != (embedding.shape[0],)
    ):
        raise ValueError("embedding/label/domain shapes are incompatible")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    embedding = F.normalize(embedding.float(), dim=-1)
    logits = embedding @ embedding.transpose(0, 1) / temperature
    identity = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = (
        (label[:, None] == label[None, :])
        & (domain_id[:, None] != domain_id[None, :])
        & ~identity
    )
    valid = positive.any(dim=1)
    if not torch.any(valid):
        return logits.new_zeros(())
    denominator = torch.logsumexp(logits.masked_fill(identity, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    return (denominator[valid] - numerator[valid]).mean()


def cross_modal_prototype_loss(embedding: Tensor, label: Tensor,
                               domain_id: Tensor) -> Tensor:
    """Align class means between molecular and spatial training cells."""
    losses = []
    for value in torch.unique(label, sorted=True):
        first = embedding[(label == value) & (domain_id == 0)]
        second = embedding[(label == value) & (domain_id == 1)]
        if len(first) and len(second):
            first_mean = F.normalize(first.mean(dim=0), dim=0)
            second_mean = F.normalize(second.mean(dim=0), dim=0)
            losses.append(1.0 - torch.sum(first_mean * second_mean))
    return torch.stack(losses).mean() if losses else embedding.new_zeros(())


class _AnnotationResidualBlock(nn.Module):
    """Small decoder block used by :class:`HierarchicalResidualAnnotationHead`.

    The block itself does not add a residual connection.  Residuals between
    annotation levels are added by the parent module so that ``feature`` and
    ``none`` have an unambiguous meaning.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(self.norm(x))
        x = F.gelu(x)
        x = self.dropout(x)
        return self.output(x)


def _coerce_positive_int_tuple(value: object, name: str,
                               *, allow_empty: bool = False) -> tuple[int, ...]:
    """Convert a sequence of positive integer dimensions with clear errors."""
    if isinstance(value, (str, bytes, Tensor)):
        raise TypeError(f"{name} must be a sequence of positive integers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of positive integers") from exc
    if not values and not allow_empty:
        raise ValueError(f"{name} must contain at least one level")
    result: list[int] = []
    for position, item in enumerate(values):
        try:
            converted = _index(item)
        except TypeError as exc:
            raise TypeError(f"{name}[{position}] must be an integer") from exc
        if converted <= 0:
            raise ValueError(f"{name}[{position}] must be positive")
        result.append(int(converted))
    return tuple(result)


class HierarchicalResidualAnnotationHead(nn.Module):
    """N-level annotation decoder with optional cross-level residuals.

    This is the annotation-side counterpart of the residual hierarchy used by
    scBrain.  It consumes a Stage-0 latent ``z`` and, optionally, intermediate
    encoder states.  Every level has its own projection, decoder block, and
    classifier.  The levels are deliberately independent of the biological
    label names: ``output_dims[i]`` is simply the number of classes at level
    ``i``.

    Parameters
    ----------
    input_dim:
        Width of ``z`` and, by default, every intermediate state.
    output_dims / output_num:
        Number of classes at each hierarchy level.  ``output_num`` is a
        compatibility alias used by the reference scBrain implementation.
        One of the two must be provided explicitly from the project's taxonomy
        manifest or resolved configuration.
    hidden_dim:
        Shared decoder feature width.
    residual_mode:
        ``"feature"`` adds the previous continuous decoder feature,
        ``"logit"`` projects and adds the previous level's logits, and
        ``"none"`` keeps levels independent.
    intermediate_dims:
        Optional state widths.  Supply one width per level, or one per
        explicit intermediate state (``n_levels - 1``); in the latter case
        ``input_dim`` is appended for the final ``z`` state.

    Notes
    -----
    Calling without ``intermediate_states`` reuses ``z`` as the input state at
    every level, while retaining separate learned projections and blocks.
    This makes the head usable directly on a Stage-0 checkpoint.  Supplying
    states does not detach them, so feature residuals remain differentiable.
    """

    _RESIDUAL_MODES = frozenset(("none", "logit", "feature"))

    def __init__(
        self,
        input_dim: int = 256,
        output_dims: Sequence[int] | None = None,
        hidden_dim: int = 256,
        residual_mode: str = "feature",
        dropout: float = 0.0,
        intermediate_dims: Sequence[int] | None = None,
        output_num: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim)):
            try:
                value = _index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be a positive integer") from exc
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if name == "input_dim":
                input_dim = int(value)
            else:
                hidden_dim = int(value)

        if output_dims is not None and output_num is not None:
            canonical_dims = _coerce_positive_int_tuple(output_dims, "output_dims")
            alias_dims = _coerce_positive_int_tuple(output_num, "output_num")
            if canonical_dims != alias_dims:
                raise ValueError("output_dims and output_num must agree")
            output_dims_tuple = canonical_dims
        elif output_dims is not None:
            output_dims_tuple = _coerce_positive_int_tuple(output_dims, "output_dims")
        elif output_num is not None:
            output_dims_tuple = _coerce_positive_int_tuple(output_num, "output_num")
        else:
            raise ValueError(
                "output_dims (or compatibility alias output_num) must be "
                "provided explicitly"
            )

        if residual_mode not in self._RESIDUAL_MODES:
            allowed = " | ".join(sorted(self._RESIDUAL_MODES))
            raise ValueError(f"residual_mode must be one of: {allowed}")
        if not isinstance(dropout, (int, float)) or not math.isfinite(float(dropout)):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        n_levels = len(output_dims_tuple)
        if intermediate_dims is None:
            state_dims = (int(input_dim),) * n_levels
        else:
            supplied_dims = _coerce_positive_int_tuple(
                intermediate_dims, "intermediate_dims", allow_empty=True
            )
            if len(supplied_dims) == n_levels:
                state_dims = supplied_dims
            elif len(supplied_dims) == n_levels - 1:
                state_dims = supplied_dims + (int(input_dim),)
            else:
                raise ValueError(
                    "intermediate_dims must have one entry per level or "
                    "one entry per level except the final z"
                )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dims = output_dims_tuple
        # ``output_num`` is the name used by scBrain and is useful when loading
        # a configuration without rewriting its hierarchy metadata.
        self.output_num = output_dims_tuple
        self.n_levels = n_levels
        self.residual_mode = residual_mode
        self.state_dims = state_dims

        # Keep a projection per level even when dimensions already match.  The
        # independent parameters let repeated Stage-0 latents specialize to
        # coarse and fine annotation levels.
        self.state_projections = nn.ModuleList([
            nn.Linear(state_dim, self.hidden_dim)
            for state_dim in state_dims
        ])
        self.blocks = nn.ModuleList([
            _AnnotationResidualBlock(self.hidden_dim, float(dropout))
            for _ in range(self.n_levels)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(self.hidden_dim, n_classes)
            for n_classes in self.output_dims
        ])
        if residual_mode == "logit":
            self.residual_projections = nn.ModuleList([
                nn.Linear(self.output_dims[level], self.hidden_dim)
                for level in range(self.n_levels - 1)
            ])
        else:
            self.residual_projections = nn.ModuleList()

    def _validate_z(self, z: Tensor) -> None:
        if not isinstance(z, Tensor) or z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f"z must have shape [B, {self.input_dim}]")

    def _resolve_states(
        self, z: Tensor, intermediate_states: Sequence[Tensor] | None
    ) -> list[Tensor]:
        if intermediate_states is None:
            states = [z for _ in range(self.n_levels)]
            if any(dim != self.input_dim for dim in self.state_dims):
                raise ValueError(
                    "intermediate_dims with non-input widths requires "
                    "intermediate_states"
                )
        else:
            if isinstance(intermediate_states, (str, bytes, Tensor)):
                raise TypeError("intermediate_states must be a sequence of tensors")
            try:
                states = list(intermediate_states)
            except TypeError as exc:
                raise TypeError("intermediate_states must be a sequence of tensors") from exc
            if len(states) == self.n_levels - 1:
                states.append(z)
            elif len(states) != self.n_levels:
                raise ValueError(
                    "intermediate_states must contain n_levels or n_levels - 1 tensors"
                )

        batch_size = z.shape[0]
        for level, (state, expected_dim) in enumerate(zip(states, self.state_dims)):
            if not isinstance(state, Tensor) or state.ndim != 2:
                raise ValueError(f"intermediate state {level} must have shape [B, D]")
            if state.shape[0] != batch_size or state.shape[1] != expected_dim:
                raise ValueError(
                    f"intermediate state {level} must have shape "
                    f"[{batch_size}, {expected_dim}]"
                )
        return states

    def forward(
        self,
        z: Tensor,
        intermediate_states: Sequence[Tensor] | None = None,
        *,
        return_features: bool = False,
    ) -> list[Tensor] | tuple[list[Tensor], list[Tensor]]:
        """Return one logits tensor per hierarchy level.

        ``return_features=True`` additionally returns the continuous feature
        supplied to each classifier.  These are the tensors used for feature
        residual propagation; no hard argmax or detached prediction is used.
        """
        self._validate_z(z)
        states = self._resolve_states(z, intermediate_states)

        logits_list: list[Tensor] = []
        feature_list: list[Tensor] = []
        previous_feature: Tensor | None = None
        previous_logits: Tensor | None = None

        for level, (state, projection, block, head) in enumerate(
            zip(states, self.state_projections, self.blocks, self.heads)
        ):
            base = projection(state)
            delta = block(base)
            if level == 0 or self.residual_mode == "none":
                current = delta
            elif self.residual_mode == "feature":
                # Preserve the continuous path; do not replace it with an
                # argmax-derived one-hot representation.
                assert previous_feature is not None
                current = delta + previous_feature
            else:  # self.residual_mode == "logit"
                assert previous_logits is not None
                current = delta + self.residual_projections[level - 1](previous_logits)

            logits = head(current)
            feature_list.append(current)
            logits_list.append(logits)
            previous_feature = current
            previous_logits = logits

        if return_features:
            return logits_list, feature_list
        return logits_list

    def forward_with_features(
        self,
        z: Tensor,
        intermediate_states: Sequence[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Explicit alias for ``forward(..., return_features=True)``."""
        result = self.forward(z, intermediate_states, return_features=True)
        assert isinstance(result, tuple)
        return result


def hierarchical_annotation_loss(
    logits: Sequence[Tensor] | Tensor,
    labels: Sequence[Tensor | None] | Tensor | None,
    level_weights: Sequence[float] | None = None,
    ignore_index: int = -100,
    return_level_losses: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor | None]]:
    """Compute normalized cross-entropy over available hierarchy labels.

    A missing label level (``None``, an empty tensor, or a tensor containing
    only negative/``ignore_index`` values) is explicitly skipped.  The total
    is normalized by the weights of levels that actually contribute, so a
    missing annotation does not dilute the remaining supervision.  With
    ``return_level_losses=True`` the second return value has one entry per
    level, using ``None`` for skipped levels.
    """
    if isinstance(logits, Tensor):
        logits_list = [logits]
    else:
        if isinstance(logits, (str, bytes)):
            raise TypeError("logits must be a tensor or a sequence of tensors")
        try:
            logits_list = list(logits)
        except TypeError as exc:
            raise TypeError("logits must be a tensor or a sequence of tensors") from exc
    if not logits_list:
        raise ValueError("logits must contain at least one level")

    batch_size: int | None = None
    for level, level_logits in enumerate(logits_list):
        if not isinstance(level_logits, Tensor) or level_logits.ndim != 2:
            raise ValueError(f"logits[{level}] must have shape [B, C]")
        if level_logits.shape[1] <= 0:
            raise ValueError(f"logits[{level}] must have at least one class")
        if batch_size is None:
            batch_size = int(level_logits.shape[0])
        elif level_logits.shape[0] != batch_size:
            raise ValueError("all logits levels must have the same batch size")

    n_levels = len(logits_list)
    if labels is None:
        labels_list: list[Tensor | None] = [None] * n_levels
    elif isinstance(labels, Tensor):
        if n_levels != 1:
            raise ValueError("a tensor label is valid only for a single logits level")
        labels_list = [labels]
    else:
        if isinstance(labels, (str, bytes)):
            raise TypeError("labels must be a tensor, sequence, or None")
        try:
            labels_list = list(labels)
        except TypeError as exc:
            raise TypeError("labels must be a tensor, sequence, or None") from exc
        if len(labels_list) != n_levels:
            raise ValueError("labels and logits must contain the same number of levels")

    if level_weights is None:
        weights = [1.0] * n_levels
    else:
        if isinstance(level_weights, (str, bytes, Tensor)):
            raise TypeError("level_weights must be a sequence of finite nonnegative values")
        try:
            raw_weights = list(level_weights)
        except TypeError as exc:
            raise TypeError("level_weights must be a sequence of finite nonnegative values") from exc
        if len(raw_weights) != n_levels:
            raise ValueError("level_weights and logits must contain the same number of levels")
        weights = []
        for level, weight in enumerate(raw_weights):
            try:
                weight_float = float(weight)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"level_weights[{level}] must be numeric") from exc
            if not math.isfinite(weight_float) or weight_float < 0:
                raise ValueError(
                    f"level_weights[{level}] must be finite and nonnegative"
                )
            weights.append(weight_float)

    try:
        ignore_index = _index(ignore_index)
    except TypeError as exc:
        raise TypeError("ignore_index must be an integer") from exc

    per_level: list[Tensor | None] = [None] * n_levels
    weighted_terms: list[Tensor] = []
    active_weights: list[float] = []
    for level, (level_logits, target, weight) in enumerate(
        zip(logits_list, labels_list, weights)
    ):
        if target is None:
            continue
        if not isinstance(target, Tensor) or target.ndim != 1:
            raise ValueError(f"labels[{level}] must have shape [B] or be None")
        # An empty label vector is an explicit missing-label marker and does
        # not need to be padded to the current batch size.
        if target.numel() == 0:
            continue
        if target.shape[0] != batch_size:
            raise ValueError(f"labels[{level}] must have shape [{batch_size}]")
        if target.is_floating_point() and not torch.isfinite(target).all():
            raise ValueError(f"labels[{level}] contains non-finite values")

        target = target.to(device=level_logits.device, dtype=torch.long)
        valid = (target >= 0) & (target != int(ignore_index))
        if not torch.any(valid):
            continue
        valid_target = target[valid]
        n_classes = level_logits.shape[1]
        if torch.any(valid_target >= n_classes):
            max_target = int(valid_target.max().item())
            raise ValueError(
                f"labels[{level}] contains class index {max_target}, "
                f"but logits[{level}] has {n_classes} classes"
            )
        level_loss = F.cross_entropy(level_logits[valid].float(), valid_target)
        per_level[level] = level_loss
        if weight > 0.0:
            weighted_terms.append(level_loss * weight)
            active_weights.append(weight)

    if weighted_terms:
        total = torch.stack(weighted_terms).sum() / sum(active_weights)
    else:
        # Keep the zero connected to the logits graph so callers can call
        # backward even for a batch with no available annotation labels.
        total = sum((level_logits.sum() for level_logits in logits_list)) * 0.0

    if return_level_losses:
        return total, per_level
    return total


# Descriptive aliases used by a few downstream experiment scripts.
hierarchical_cross_entropy_loss = hierarchical_annotation_loss
scbrain_hierarchical_loss = hierarchical_annotation_loss
