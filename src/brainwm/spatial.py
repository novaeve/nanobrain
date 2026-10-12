"""Reusable spatial locality plans for spatial transcriptomics batches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .training import BatchDescriptor


def _interleave16(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interleave two uint16 coordinates into a uint32 Morton code."""
    code = np.zeros_like(x, dtype=np.uint32)
    for bit in range(16):
        code |= ((x >> bit) & 1).astype(np.uint32) << (2 * bit)
        code |= ((y >> bit) & 1).astype(np.uint32) << (2 * bit + 1)
    return code


def morton_order(frame: pd.DataFrame) -> np.ndarray:
    """Return row positions grouped by sample and locally ordered by Morton code."""
    required = {"sample_key", "spatial_x", "spatial_y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"spatial frame is missing columns: {sorted(missing)}")
    output: list[np.ndarray] = []
    for _, group in frame.groupby("sample_key", sort=True):
        positions = group.index.to_numpy()
        coordinates = group[["spatial_x", "spatial_y"]].to_numpy(np.float64)
        lower = coordinates.min(axis=0)
        span = np.maximum(coordinates.max(axis=0) - lower, 1.0e-12)
        scaled = np.clip(
            np.rint((coordinates - lower) / span * 65535.0), 0, 65535
        ).astype(np.uint16)
        codes = _interleave16(scaled[:, 0], scaled[:, 1])
        output.append(positions[np.argsort(codes, kind="stable")])
    return np.concatenate(output) if output else np.empty(0, dtype=np.int64)


@dataclass
class MortonSpatialBatchPlan:
    """Batch descriptors backed by one-time source-local Morton row orders."""

    reader: object
    root: Path
    split: str
    batch_size: int
    sample_keys: set[str] | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._memory_maps: list[np.memmap] = []
        metadata_path = self.root / f"{self.split}_batches.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata = pd.read_parquet(metadata_path)
        cells = self.reader.cells
        if self.sample_keys is not None:
            cells = cells[cells["sample_key"].isin(self.sample_keys)]
        cells = cells[
            (cells["split"] == self.split)
            & cells["spatial_x"].notna()
            & cells["spatial_y"].notna()
        ]
        self.frames: dict[int, pd.DataFrame] = {}
        self.orders: dict[int, np.ndarray] = {}
        descriptors: list[BatchDescriptor] = []
        for source_id in sorted(metadata["source_id"].unique()):
            source_id = int(source_id)
            order_path = self.root / f"{self.split}_source_{source_id}.npy"
            if not order_path.exists():
                raise FileNotFoundError(order_path)
            frame = cells[cells["source_id"] == source_id].copy()
            if frame.empty:
                continue
            mapped_order = np.load(order_path, mmap_mode="r")
            if isinstance(mapped_order, np.memmap):
                self._memory_maps.append(mapped_order)
            available = frame["row_index"].to_numpy(np.int64)
            ordered = np.asarray(mapped_order, dtype=np.int64)
            # Descriptor offsets refer to the complete persisted order.  Keep
            # that order intact when sample_keys selects only some sections;
            # eligible descriptors are filtered below by row membership.
            if not len(ordered) or not len(available):
                continue
            self.frames[source_id] = frame.set_index("row_index", drop=False)
            self.orders[source_id] = ordered
            if self.sample_keys is None:
                if len(ordered) != len(available):
                    raise ValueError(
                        f"{self.split} source {source_id} plan/index row count mismatch"
                    )
                available_in_order = np.ones(len(ordered), dtype=bool)
            else:
                # Compute membership once per source.  Calling np.isin for
                # every descriptor would repeatedly sort millions of row IDs.
                # Row indices are non-negative source-local integers.  NumPy's
                # table lookup is linear here; the default sorting path takes
                # minutes and large temporary buffers on multi-million-cell
                # spatial sources.
                if np.any(ordered < 0) or np.any(available < 0):
                    raise ValueError("spatial row indices must be non-negative")
                available_in_order = np.isin(
                    ordered, available, assume_unique=True, kind="table"
                )
            source_metadata = metadata[metadata["source_id"].astype(int) == source_id]
            if source_metadata.empty:
                continue
            # Persisted plans may skip per-sample remainders so a fixed batch
            # never crosses a section boundary.  Older plans contain the same
            # contiguous descriptors as the fallback below.
            for row in source_metadata.itertuples(index=False):
                start = int(row.start)
                stop = int(row.stop)
                if start < 0 or stop > len(ordered) or stop - start != self.batch_size:
                    continue
                if not available_in_order[start:stop].all():
                    continue
                descriptors.append(BatchDescriptor(source_id, start, stop))
        self.descriptors = descriptors

    def close(self) -> None:
        """Close persisted Morton-order memory maps owned by this plan."""
        self.orders.clear()
        for order in self._memory_maps:
            mmap = getattr(order, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()
        self._memory_maps.clear()

    def __enter__(self) -> "MortonSpatialBatchPlan":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def rank_descriptors(self, rank: int, world_size: int, seed: int,
                         shuffle: bool, maximum_per_rank: int = 0,
                         balance_by_cost: bool = True) -> list[BatchDescriptor]:
        descriptors = list(self.descriptors)
        usable = len(descriptors) - (len(descriptors) % world_size)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(descriptors)
        local = descriptors[:usable][rank:usable:world_size]
        return local[:maximum_per_rank] if maximum_per_rank else local

    def frame(self, descriptor: BatchDescriptor) -> pd.DataFrame:
        rows = self.orders[descriptor.source_id][descriptor.start:descriptor.stop]
        return self.frames[descriptor.source_id].loc[rows].reset_index(drop=True)


def build_morton_plan(artifact_dir: str | Path, output_dir: str | Path,
                      batch_size: int = 256,
                      preserve_sample_boundaries: bool = False) -> dict[str, object]:
    """Build split/source row orders once, without changing the corpus artifact."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(
        artifact_dir / "cells.parquet",
        columns=["source_id", "row_index", "sample_key", "split",
                 "spatial_x", "spatial_y"],
    )
    manifest: dict[str, object] = {
        "format": "brainwm_spatial_morton_plan_v1",
        "artifact_dir": str(artifact_dir.resolve()),
        "batch_size": batch_size,
        "preserve_sample_boundaries": preserve_sample_boundaries,
        "splits": {},
    }
    for split in sorted(cells["split"].dropna().unique()):
        split_meta: list[dict[str, int]] = []
        subset = cells[
            (cells["split"] == split)
            & cells["spatial_x"].notna()
            & cells["spatial_y"].notna()
        ]
        for source_id, frame in subset.groupby("source_id", sort=True):
            order_positions = morton_order(frame.reset_index(drop=True))
            row_values = frame.reset_index(drop=True).iloc[order_positions]["row_index"].to_numpy(np.int64)
            np.save(output_dir / f"{split}_source_{int(source_id)}.npy", row_values)
            if preserve_sample_boundaries:
                offset = 0
                for _, sample in frame.groupby("sample_key", sort=True):
                    sample_size = len(sample)
                    usable = sample_size - (sample_size % batch_size)
                    for local_start in range(0, usable, batch_size):
                        split_meta.append({
                            "source_id": int(source_id),
                            "start": offset + local_start,
                            "stop": offset + local_start + batch_size,
                        })
                    offset += sample_size
            else:
                usable = len(row_values) - (len(row_values) % batch_size)
                for start in range(0, usable, batch_size):
                    split_meta.append({
                        "source_id": int(source_id),
                        "start": start,
                        "stop": start + batch_size,
                    })
        pd.DataFrame(split_meta).to_parquet(
            output_dir / f"{split}_batches.parquet", index=False
        )
        manifest["splits"][split] = {
            "sources": int(subset["source_id"].nunique()),
            "cells": len(subset),
            "batches": len(split_meta),
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
