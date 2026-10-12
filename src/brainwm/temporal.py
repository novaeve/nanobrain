"""Balanced adjacent-time plans for cross-sectional Stage-1 training."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .data import MetadataTensorizer


@dataclass(frozen=True)
class TemporalBatchDescriptor:
    index: int
    source_id: int


class TemporalPairPlan:
    """Memory-mapped cell indices plus compact metadata for temporal batches."""

    def __init__(self, plan_dir: str | Path, split: str, source_cost: dict[int, int]):
        root = Path(plan_dir)
        self.metadata = pd.read_parquet(root / f"{split}_batches.parquet")
        self.source_indices = np.load(
            root / f"{split}_source_indices.npy", mmap_mode="r"
        )
        self.target_indices = np.load(
            root / f"{split}_target_indices.npy", mmap_mode="r"
        )
        if len(self.metadata) != len(self.source_indices) or self.source_indices.shape != self.target_indices.shape:
            raise ValueError(f"invalid temporal pair plan for split={split}")
        self.source_cost = source_cost

    @property
    def batch_size(self) -> int:
        return int(self.source_indices.shape[1])

    def rank_descriptors(self, rank: int, world_size: int, seed: int,
                         shuffle: bool = True,
                         maximum_per_rank: int = 0) -> list[TemporalBatchDescriptor]:
        descriptors = [
            TemporalBatchDescriptor(index, int(row.source_id))
            for index, row in enumerate(self.metadata.itertuples(index=False))
        ]
        usable = len(descriptors) - len(descriptors) % world_size
        descriptors.sort(
            key=lambda item: self.source_cost[item.source_id], reverse=True
        )
        descriptors = descriptors[:usable]
        rounds = [
            descriptors[start:start + world_size]
            for start in range(0, usable, world_size)
        ]
        if shuffle:
            rng = random.Random(seed)
            for group in rounds:
                rng.shuffle(group)
            rng.shuffle(rounds)
        local = [group[rank] for group in rounds]
        return local[:maximum_per_rank] if maximum_per_rank else local

    def row(self, descriptor: TemporalBatchDescriptor):
        return self.metadata.iloc[descriptor.index]


_CELL_METADATA_REQUIRED_COLUMNS = (
    "source_id",
    "row_index",
    "sample_key",
    "spatial_x",
    "spatial_y",
    "split",
)

# Keep ontology-backed anatomy in the bounded temporal prefetch.  These are
# optional at the file-schema level because some legacy cohorts have no
# anatomical annotation; when absent, the loader adds an explicit all-missing
# column so tensorization has the same shape and semantics.
_CELL_METADATA_ANATOMY_COLUMNS = (
    "uberon_id",
    "uberon_term",
    "original_anatomical_region",
    "anatomical_region",
    "anatomical_layer",
)
_CELL_METADATA_COLUMNS = (
    *_CELL_METADATA_REQUIRED_COLUMNS,
    *_CELL_METADATA_ANATOMY_COLUMNS,
)

_SAMPLE_CONTEXT_COLUMNS = (
    "assay_type", "technology", "dataset", "species", "species_id",
    "time_point_days", "time_lower_days", "time_upper_days", "time_basis",
    "time_status", "life_stage", "age_raw", "timepoint_raw",
    "developmental_stage_raw", *_CELL_METADATA_ANATOMY_COLUMNS,
)


def merge_sample_metadata(
    cell_metadata: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Join sample context while preserving explicit cell annotations.

    Temporal and spatial fast paths build a compact cell frame before reading
    expression.  A sample table can still carry anatomy (for example, one
    UBERON ID for a section), so suffix columns from the join must be filled
    back into the canonical cell columns before tensorization.  Cell-level
    values always take precedence over sample-level fallbacks.
    """
    metadata = cell_metadata.join(
        sample_metadata, on="sample_key", how="left", rsuffix="_sample"
    )
    for column in _SAMPLE_CONTEXT_COLUMNS:
        sample_column = f"{column}_sample"
        if sample_column not in metadata:
            continue
        if column not in metadata:
            metadata[column] = metadata[sample_column]
        else:
            metadata[column] = metadata[column].where(
                metadata[column].notna(), metadata[sample_column]
            )
    return metadata


def read_temporal_cell_metadata(
    artifact_dir: str | Path,
    positions: np.ndarray,
) -> pd.DataFrame:
    """Read only cell-index rows needed by a bounded temporal schedule.

    Temporal plans store physical row positions in ``cells.parquet``.  Reading
    that complete table materializes tens of millions of rows before the first
    optimizer step.  This loader visits each intersecting Parquet row group at
    most once, keeps only requested rows, and indexes the result by the original
    physical position for constant-time descriptor lookup.
    """

    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("temporal cell positions must be a non-empty vector")
    if np.any(positions < 0):
        raise ValueError("temporal cell positions must be nonnegative")

    path = Path(artifact_dir) / "cells.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"cell metadata not found: {path}")
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    missing = sorted(set(_CELL_METADATA_REQUIRED_COLUMNS) - available)
    if missing:
        raise ValueError(
            f"{path} is missing temporal metadata columns: {', '.join(missing)}"
        )
    selected_columns = [
        column for column in _CELL_METADATA_COLUMNS if column in available
    ]

    unique = np.unique(positions)
    pieces: list[pd.DataFrame] = []
    row_groups_read = 0
    offset = 0
    for row_group in range(parquet.num_row_groups):
        length = int(parquet.metadata.row_group(row_group).num_rows)
        left = int(np.searchsorted(unique, offset, side="left"))
        right = int(np.searchsorted(unique, offset + length, side="left"))
        if right > left:
            frame = parquet.read_row_group(
                row_group, columns=selected_columns
            ).to_pandas()
            selected_positions = unique[left:right]
            selected = frame.iloc[selected_positions - offset].copy()
            selected.index = pd.Index(
                selected_positions, name="_global_position"
            )
            pieces.append(selected)
            row_groups_read += 1
        offset += length
        if offset > int(unique[-1]):
            break

    if not pieces:
        raise IndexError(
            f"temporal cell positions exceed cells.parquet: {int(unique[-1])}"
        )
    result = pd.concat(pieces, axis=0, copy=False).sort_index()
    # Preserve a stable metadata contract for legacy artifacts without anatomy.
    # ``MetadataTensorizer`` maps these missing values to the vocabulary's
    # dedicated <missing> ID rather than silently using a real region.
    for column in _CELL_METADATA_ANATOMY_COLUMNS:
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
    missing_positions = unique[~np.isin(unique, result.index.to_numpy(np.int64))]
    if missing_positions.size:
        raise IndexError(
            "cell metadata is missing temporal positions: "
            f"{missing_positions[:5].tolist()}"
        )
    result.attrs["row_groups_read"] = row_groups_read
    result.attrs["rows_requested"] = int(unique.size)
    return result


def build_temporal_batch(reader, plan: TemporalPairPlan,
                         descriptor: TemporalBatchDescriptor,
                         sample_metadata: pd.DataFrame,
                         tensorizer: MetadataTensorizer,
                         device: torch.device,
                         cell_arrays: dict[str, np.ndarray] | None = None,
                         cell_metadata: pd.DataFrame | None = None):
    if cell_arrays is not None and cell_metadata is not None:
        raise ValueError("provide either cell_arrays or cell_metadata, not both")
    batches = []
    indexed_batches = []
    for positions in (
        plan.source_indices[descriptor.index],
        plan.target_indices[descriptor.index],
    ):
        positions = np.asarray(positions, dtype=np.int64)
        if cell_metadata is not None:
            try:
                frame = cell_metadata.loc[positions].copy()
            except KeyError as exc:
                raise IndexError(
                    "prefetched cell metadata does not cover the temporal descriptor"
                ) from exc
            frame = frame.sort_values("row_index").reset_index(drop=True)
        elif cell_arrays is None:
            frame = reader.cells.iloc[positions] \
                .sort_values("row_index").reset_index(drop=True)
        else:
            source_ids = cell_arrays["source_id"][positions]
            if len(set(source_ids.tolist())) != 1 or int(source_ids[0]) != descriptor.source_id:
                raise ValueError("temporal batch crossed H5AD sources")
            order = np.argsort(cell_arrays["row_index"][positions], kind="stable")
            ordered = positions[order]
            frame_values = {}
            for name in (
                "source_id", "row_index", "sample_key", "spatial_x",
                "spatial_y", *_CELL_METADATA_ANATOMY_COLUMNS,
            ):
                if name in cell_arrays:
                    frame_values[name] = cell_arrays[name][ordered]
            frame = pd.DataFrame(frame_values)
        source_ids = frame["source_id"].unique()
        if len(source_ids) != 1 or int(source_ids[0]) != descriptor.source_id:
            raise ValueError("temporal batch crossed H5AD sources")
        indexed = reader.read_batch(
            descriptor.source_id,
            frame["row_index"].to_numpy(np.int64),
            frame,
        )
        metadata = merge_sample_metadata(indexed["cells"], sample_metadata)
        batches.append(tensorizer.tensorize(indexed, metadata, device))
        indexed_batches.append(indexed)
    return batches[0], batches[1], indexed_batches


def _stable_seed(seed: int, values: tuple[object, ...]) -> int:
    payload = "|".join(str(value) for value in (seed, *values)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_temporal_pair_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path,
    assay_types: list[str],
    allowed_time_statuses: list[str],
    batch_size: int,
    min_cells_per_time: int,
    max_batches_per_interval: int,
    validation_max_batches_per_interval: int,
    seed: int,
) -> dict[str, object]:
    """Build balanced adjacent-time batches within one study/anatomy stratum."""
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(artifact_dir / "samples.parquet")
    samples = samples[
        (samples["species"] == "human")
        & samples["assay_type"].isin(assay_types)
        & samples["time_status"].isin(allowed_time_statuses)
        & samples["time_point_days"].notna()
        & (samples["life_stage"] != "unknown")
    ].copy()
    cells = pd.read_parquet(
        artifact_dir / "cells.parquet",
        columns=["source_id", "row_index", "sample_key", "uberon_id", "split"],
    )
    eligible_keys = set(samples["sample_key"].astype(str))
    cells = cells[
        cells["sample_key"].isin(eligible_keys)
        & cells["uberon_id"].notna()
        & cells["split"].isin(["train", "validation"])
    ].copy()
    cells["cell_position"] = cells.index.to_numpy(np.int64)
    sample_lookup = samples.set_index("sample_key")
    for field in ("dataset", "assay_type", "life_stage", "time_point_days"):
        cells[field] = cells["sample_key"].map(sample_lookup[field])

    group_fields = [
        "split", "source_id", "dataset", "assay_type", "life_stage", "uberon_id"
    ]
    summary: dict[str, object] = {}
    for split in ("train", "validation"):
        source_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        records: list[dict[str, object]] = []
        split_cells = cells[cells["split"] == split]
        cap = (
            max_batches_per_interval
            if split == "train" else validation_max_batches_per_interval
        )
        for group_key, group in split_cells.groupby(group_fields, sort=True):
            time_groups = {
                float(time): part["cell_position"].to_numpy(np.int64)
                for time, part in group.groupby("time_point_days", sort=True)
            }
            times = sorted(time_groups)
            for source_time, target_time in zip(times[:-1], times[1:]):
                source_cells = time_groups[source_time]
                target_cells = time_groups[target_time]
                if min(len(source_cells), len(target_cells)) < min_cells_per_time:
                    continue
                n_batches = min(
                    len(source_cells) // batch_size,
                    len(target_cells) // batch_size,
                    cap,
                )
                if n_batches < 1:
                    continue
                rng = np.random.default_rng(_stable_seed(
                    seed, (*group_key, source_time, target_time)
                ))
                # Keep each read local in the H5AD row order. Random individual
                # cells create hundreds of NAS reads per pair and make Stage 1
                # I/O-bound; randomized local blocks retain population sampling
                # while allowing the reader to coalesce adjacent rows.
                source_chunks = np.asarray(source_cells[:
                    len(source_cells) // batch_size * batch_size
                ]).reshape(-1, batch_size)
                target_chunks = np.asarray(target_cells[:
                    len(target_cells) // batch_size * batch_size
                ]).reshape(-1, batch_size)
                source_choice = rng.permutation(len(source_chunks))[:n_batches]
                target_choice = rng.permutation(len(target_chunks))[:n_batches]
                source_selected = source_chunks[source_choice]
                target_selected = target_chunks[target_choice]
                interval_id = hashlib.sha1(
                    "|".join(str(value) for value in (
                        *group_key, source_time, target_time
                    )).encode()
                ).hexdigest()[:16]
                for batch_index in range(n_batches):
                    source_batches.append(source_selected[batch_index])
                    target_batches.append(target_selected[batch_index])
                    records.append({
                        "interval_id": interval_id,
                        "interval_batch": batch_index,
                        "source_id": int(group_key[1]),
                        "dataset": str(group_key[2]),
                        "assay_type": str(group_key[3]),
                        "life_stage": str(group_key[4]),
                        "uberon_id": str(group_key[5]),
                        "source_time_days": source_time,
                        "target_time_days": target_time,
                        "delta_time_days": target_time - source_time,
                        "source_interval_cells": len(source_cells),
                        "target_interval_cells": len(target_cells),
                    })
        if not records:
            raise ValueError(f"no temporal batches were built for split={split}")
        source_array = np.stack(source_batches).astype(np.int64, copy=False)
        target_array = np.stack(target_batches).astype(np.int64, copy=False)
        np.save(output_dir / f"{split}_source_indices.npy", source_array)
        np.save(output_dir / f"{split}_target_indices.npy", target_array)
        pd.DataFrame(records).to_parquet(
            output_dir / f"{split}_batches.parquet", index=False
        )
        summary[split] = {
            "batches": len(records),
            "paired_cells_per_side": int(len(records) * batch_size),
            "intervals": int(pd.DataFrame(records)["interval_id"].nunique()),
            "datasets": int(pd.DataFrame(records)["dataset"].nunique()),
        }

    manifest = {
        "artifact_dir": str(artifact_dir.resolve()),
        "batch_size": batch_size,
        "assay_types": assay_types,
        "allowed_time_statuses": allowed_time_statuses,
        "min_cells_per_time": min_cells_per_time,
        "max_batches_per_interval": max_batches_per_interval,
        "validation_max_batches_per_interval": validation_max_batches_per_interval,
        "seed": seed,
        "splits": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
