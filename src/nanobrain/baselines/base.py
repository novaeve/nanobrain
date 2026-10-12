"""Stable API shared by nanoBrain third-party baseline adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nanobrain.data import NanoDataset
from nanobrain.development import DevelopmentPairDataset


@dataclass(frozen=True)
class AdapterContext:
    """Inputs visible to one adapter.

    Adapters receive train and validation only.  The sealed test split is not
    part of this contract, which makes accidental test-label access impossible
    inside a method implementation.
    """

    train: NanoDataset
    validation: NanoDataset
    data_root: Path
    output_root: Path
    seed: int
    device: str
    threads: int
    development_train: DevelopmentPairDataset | None = None
    development_validation: DevelopmentPairDataset | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPrediction:
    """One evaluator-ready prediction produced by an adapter."""

    task: str
    arrays: Mapping[str, np.ndarray]
    fit_rows: int
    eval_rows: int
    notes: str
    training_regime: str


@dataclass(frozen=True)
class Availability:
    status: str
    reason: str = ""

    @property
    def runnable(self) -> bool:
        return self.status == "ready"


class BaselineAdapter(ABC):
    """Replaceable method boundary used by the benchmark runner."""

    method_id: str
    display_name: str
    family: str = "external"
    tasks: tuple[str, ...]

    def availability(self, context: AdapterContext) -> Availability:
        return Availability("ready")

    def metadata(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "family": self.family,
            "tasks": list(self.tasks),
        }

    @abstractmethod
    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        """Fit the method and return evaluator-ready validation predictions."""


__all__ = [
    "AdapterContext",
    "Availability",
    "BaselineAdapter",
    "TaskPrediction",
]
