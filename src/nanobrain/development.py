"""Split-local population endpoint batches for developmental forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .contracts import BrainBatch


@dataclass(frozen=True)
class DevelopmentPopulationBatch:
    """Two unpaired cross-sectional endpoint populations from one interval."""

    source: BrainBatch
    target: BrainBatch
    source_time_days: float
    target_time_days: float
    interval_id: str
    anatomy: str

    def __post_init__(self) -> None:
        if self.source.batch_size != self.target.batch_size:
            raise ValueError("development endpoint populations must have equal sizes")
        if self.source.batch_size < 2:
            raise ValueError("development population batches require at least two cells")
        if not np.isfinite(self.source_time_days) or not np.isfinite(self.target_time_days):
            raise ValueError("development endpoint times must be finite")
        if self.target_time_days <= self.source_time_days:
            raise ValueError("development target time must be later than source time")

    @property
    def cells(self) -> int:
        return self.source.batch_size

    def to(self, device: torch.device | str) -> "DevelopmentPopulationBatch":
        return DevelopmentPopulationBatch(
            source=self.source.to(device),
            target=self.target.to(device),
            source_time_days=self.source_time_days,
            target_time_days=self.target_time_days,
            interval_id=self.interval_id,
            anatomy=self.anatomy,
        )


class DevelopmentPairDataset:
    """Random-access view over materialized formal development intervals."""

    FORMAT = "nanobrain_population_pairs_v1"

    def __init__(
        self,
        dataset: Any,
        source_index: np.ndarray,
        target_index: np.ndarray,
        pair_ptr: np.ndarray,
        groups: list[dict[str, Any]],
        *,
        split: str,
    ) -> None:
        self.dataset = dataset
        self.source_index = np.asarray(source_index, dtype=np.int64)
        self.target_index = np.asarray(target_index, dtype=np.int64)
        self.pair_ptr = np.asarray(pair_ptr, dtype=np.int64)
        self.groups = list(groups)
        self.split = str(split)
        self._validate()

    def _validate(self) -> None:
        if self.source_index.ndim != 1 or self.target_index.ndim != 1:
            raise ValueError("development pair indices must be rank-1")
        if self.source_index.shape != self.target_index.shape:
            raise ValueError("development source and target index counts differ")
        if self.pair_ptr.ndim != 1 or len(self.pair_ptr) != len(self.groups) + 1:
            raise ValueError("development pair_ptr does not match group metadata")
        if len(self.pair_ptr) < 2 or self.pair_ptr[0] != 0:
            raise ValueError("development pair_ptr must start at zero")
        if np.any(np.diff(self.pair_ptr) < 2):
            raise ValueError("every development interval must contain at least two cells")
        if int(self.pair_ptr[-1]) != len(self.source_index):
            raise ValueError("development pair_ptr endpoint differs from index count")
        size = int(self.dataset.size)
        for name, value in (("source", self.source_index), ("target", self.target_index)):
            if np.any(value < 0) or np.any(value >= size):
                raise ValueError(f"development {name} index is outside the {self.split} split")
        for group in self.groups:
            if float(group["target_time_days"]) <= float(group["source_time_days"]):
                raise ValueError("development group has a non-positive time interval")

    def __len__(self) -> int:
        return len(self.groups)

    @classmethod
    def load(cls, root: str | Path, split: str, dataset: Any) -> "DevelopmentPairDataset | None":
        directory = Path(root) / str(split)
        receipt_path = directory / "development_pairs.json"
        paths = {
            "source": directory / "development_source_index.npy",
            "target": directory / "development_target_index.npy",
            "ptr": directory / "development_pair_ptr.npy",
        }
        if not receipt_path.is_file() and not any(path.is_file() for path in paths.values()):
            return None
        if not receipt_path.is_file() or not all(path.is_file() for path in paths.values()):
            raise ValueError(f"incomplete development population sidecars in {directory}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("format") != cls.FORMAT or receipt.get("split") != split:
            raise ValueError(f"invalid development population receipt: {receipt_path}")
        return cls(
            dataset,
            np.load(paths["source"], mmap_mode="r", allow_pickle=False),
            np.load(paths["target"], mmap_mode="r", allow_pickle=False),
            np.load(paths["ptr"], mmap_mode="r", allow_pickle=False),
            receipt.get("groups_metadata", []),
            split=split,
        )

    def batch(
        self,
        group_index: int,
        *,
        max_cells: int | None = None,
        seed: int = 0,
    ) -> DevelopmentPopulationBatch:
        group_index = int(group_index)
        if group_index < 0 or group_index >= len(self):
            raise IndexError(group_index)
        lo, hi = int(self.pair_ptr[group_index]), int(self.pair_ptr[group_index + 1])
        source = np.asarray(self.source_index[lo:hi], dtype=np.int64)
        target = np.asarray(self.target_index[lo:hi], dtype=np.int64)
        if max_cells is not None and len(source) > int(max_cells):
            if int(max_cells) < 2:
                raise ValueError("max_cells must be at least two")
            rng = np.random.default_rng(int(seed) + group_index * 104729)
            chosen = np.sort(rng.choice(len(source), int(max_cells), replace=False))
            source, target = source[chosen], target[chosen]
        group = self.groups[group_index]
        return DevelopmentPopulationBatch(
            source=self.dataset.batch(source),
            target=self.dataset.batch(target),
            source_time_days=float(group["source_time_days"]),
            target_time_days=float(group["target_time_days"]),
            interval_id=str(group["interval_id"]),
            anatomy=str(group["uberon_id"]),
        )

    def iter_batches(
        self,
        *,
        shuffle: bool = False,
        seed: int = 0,
        max_cells: int | None = None,
    ) -> Iterator[DevelopmentPopulationBatch]:
        order = np.arange(len(self), dtype=np.int64)
        if shuffle:
            np.random.default_rng(int(seed)).shuffle(order)
        for index in order:
            yield self.batch(int(index), max_cells=max_cells, seed=seed)


__all__ = ["DevelopmentPairDataset", "DevelopmentPopulationBatch"]
