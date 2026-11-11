"""Stable tensor contracts shared by all nanoBrain modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor


TASKS = ("integration", "annotation", "niche", "development")
AUXILIARY_TASKS = ("completion",)


def _check_first_dim(name: str, value: Tensor | None, size: int) -> None:
    if value is None:
        return
    if value.ndim == 0 or int(value.shape[0]) != int(size):
        raise ValueError(f"{name} must have first dimension {size}, got {tuple(value.shape)}")


@dataclass
class BrainBatch:
    """Dense task view consumed by a candidate model.

    The formal V3 loader remains sparse and CSR based.  A nanoBrain adapter
    converts the selected rows into this compact view once per batch.  Keeping
    the model contract independent of the source representation lets the same
    candidate run on synthetic smoke data, a 100k-row subset, or V3 shards.
    """

    features: Tensor
    labels: dict[str, Tensor] = field(default_factory=dict)
    label_masks: dict[str, Tensor] = field(default_factory=dict)
    coordinates: Tensor | None = None
    neighbors: Tensor | None = None
    neighbor_mask: Tensor | None = None
    development_time: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional extensions are appended to preserve the original positional
    # constructor order used by early nanoBrain experiments.
    feature_observed: Tensor | None = None
    development_condition: Tensor | None = None
    development_condition_observed: Tensor | None = None
    completion_target: Tensor | None = None
    completion_mask: Tensor | None = None
    neighbor_features: Tensor | None = None
    neighbor_completion_mask: Tensor | None = None

    def validate(self) -> "BrainBatch":
        if not isinstance(self.features, Tensor) or self.features.ndim != 2:
            raise ValueError("features must be a rank-2 tensor [batch, input_dim]")
        size = int(self.features.shape[0])
        if size < 1:
            raise ValueError("batch must contain at least one row")
        if self.feature_observed is not None:
            if self.feature_observed.shape != self.features.shape:
                raise ValueError("feature_observed must match features shape")
            if self.feature_observed.dtype != torch.bool:
                raise TypeError("feature_observed must be bool")
        for name, value in self.labels.items():
            _check_first_dim(f"labels[{name!r}]", value, size)
        for name, value in self.label_masks.items():
            _check_first_dim(f"label_masks[{name!r}]", value, size)
            if value.dtype != torch.bool:
                raise TypeError(f"label_masks[{name!r}] must be bool")
        _check_first_dim("coordinates", self.coordinates, size)
        _check_first_dim("development_time", self.development_time, size)
        _check_first_dim("development_condition", self.development_condition, size)
        _check_first_dim(
            "development_condition_observed", self.development_condition_observed, size
        )
        if self.development_condition_observed is not None and self.development_condition_observed.dtype != torch.bool:
            raise TypeError("development_condition_observed must be bool")
        if self.completion_target is not None:
            if self.completion_target.shape != self.features.shape:
                raise ValueError("completion_target must match features shape")
        if self.completion_mask is not None:
            if self.completion_mask.shape != self.features.shape:
                raise ValueError("completion_mask must match features shape")
            if self.completion_mask.dtype != torch.bool:
                raise TypeError("completion_mask must be bool")
        if self.coordinates is not None and self.coordinates.shape[-1] < 2:
            raise ValueError("coordinates must provide at least x and y")
        if self.neighbors is not None:
            if self.neighbors.ndim != 2 or int(self.neighbors.shape[0]) != size:
                raise ValueError("neighbors must have shape [batch, bounded_neighbors]")
            if self.neighbor_mask is None or self.neighbor_mask.shape != self.neighbors.shape:
                raise ValueError("neighbor_mask must match neighbors")
            if self.neighbor_mask.dtype != torch.bool:
                raise TypeError("neighbor_mask must be bool")
        if self.neighbor_features is not None:
            if self.neighbor_features.ndim != 2 or self.neighbor_features.shape[1] != self.input_dim:
                raise ValueError("neighbor_features must be [context_cells,input_dim]")
            if self.neighbor_completion_mask is not None and self.neighbor_completion_mask.shape != self.neighbor_features.shape:
                raise ValueError("neighbor completion mask shape mismatch")
        return self

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[-1])

    def to(self, device: torch.device | str) -> "BrainBatch":
        def move(value: Any) -> Any:
            return value.to(device) if isinstance(value, Tensor) else value

        return BrainBatch(
            features=move(self.features),
            feature_observed=move(self.feature_observed),
            labels={key: move(value) for key, value in self.labels.items()},
            label_masks={key: move(value) for key, value in self.label_masks.items()},
            coordinates=move(self.coordinates),
            neighbors=move(self.neighbors),
            neighbor_mask=move(self.neighbor_mask),
            development_time=move(self.development_time),
            development_condition=move(self.development_condition),
            development_condition_observed=move(self.development_condition_observed),
            completion_target=move(self.completion_target),
            completion_mask=move(self.completion_mask),
            neighbor_features=move(self.neighbor_features),
            neighbor_completion_mask=move(self.neighbor_completion_mask),
            metadata=dict(self.metadata),
        )


@dataclass
class PredictionBundle:
    """Named outputs; absent tasks are represented by ``None``."""

    embedding: Tensor
    annotation_logits: Tensor | None = None
    niche_logits: Tensor | None = None
    development_prediction: Tensor | None = None
    integration_prediction: Tensor | None = None
    completion_prediction: Tensor | None = None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {"embedding": self.embedding}
        for name in (
            "annotation_logits",
            "niche_logits",
            "development_prediction",
            "integration_prediction",
            "completion_prediction",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        values.update(self.auxiliary)
        return values


@dataclass
class LossBundle:
    """Per-task losses and detached diagnostics for a receipt."""

    terms: dict[str, Tensor]
    weights: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> Tensor:
        if not self.terms:
            raise ValueError("cannot aggregate an empty loss bundle")
        first = next(iter(self.terms.values()))
        total = first.new_zeros(())
        for name, value in self.terms.items():
            total = total + float(self.weights.get(name, 1.0)) * value
        return total

    def detached(self) -> dict[str, float]:
        values = {name: float(value.detach().cpu()) for name, value in self.terms.items()}
        values["total"] = float(self.total.detach().cpu())
        values.update({str(key): float(value) for key, value in self.diagnostics.items()})
        return values


@dataclass(frozen=True)
class ModuleContract:
    """Metadata used to validate composition before building a candidate."""

    kind: str
    name: str
    input_rank: int | None = None
    output_rank: int | None = None
    complexity: str = "unknown"
    supports_missing: bool = True
    source: str = "nanobrain"
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "input_rank": self.input_rank,
            "output_rank": self.output_rank,
            "complexity": self.complexity,
            "supports_missing": self.supports_missing,
            "source": self.source,
            "notes": self.notes,
        }
