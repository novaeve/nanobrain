# Trusted CSR helpers adapted from the original materializer. Sampling/panel fitting
# and anatomy-proxy target creation are NOT used by the scale builder.
#!/usr/bin/env python3
"""Materialize a small, leakage-safe BrainWM V3 subset for nanoBrain.

The formal V3 corpus is already materialized as NumPy CSR shards.  This
command never opens H5AD files: it selects rows from the shard metadata,
derives a train-only gene panel and normalization, and writes compact dense
panel features plus task sidecars.  The global V3 split remains authoritative;
the subset is sampled independently *inside* each frozen split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))



SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}
# The formal split row references are local to each split.  Reserve a wide,
# deterministic namespace so exported row IDs remain globally disjoint.
ROW_ID_STRIDE = 100_000_000
META_FIELDS = (
    "row_source_id",
    "assay",
    "dataset",
    "technology",
    "sample",
    "donor",
    "section",
    "split_unit",
    "split_group",
    "cell_type",
    "anatomy",
    "integration_batch",
    "integration_biology",
    "life_stage",
    "sex",
    "time_status",
    "time_point_days",
    "time_lower_days",
    "time_upper_days",
    "chronological_time_proxy_days",
    "spatial",
    "annotation_eligible",
    "development_eligible",
    "spatial_eligible",
    "integration_eligible",
)
FLAG_FIELDS = {
    "annotation_eligible",
    "development_eligible",
    "spatial_eligible",
    "integration_eligible",
}


@dataclass(frozen=True)
class Shard:
    split: str
    index: int
    row_ptr: Path
    stem: Path
    offset: int
    rows: int


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _discover(root: Path, split: str) -> list[Shard]:
    paths = sorted(root.glob(f"{split}-shard-*-row_ptr.npy"))
    if not paths:
        raise FileNotFoundError(f"no {split} row_ptr shards under {root}")
    result: list[Shard] = []
    offset = 0
    for index, row_ptr in enumerate(paths):
        stem = Path(str(row_ptr)[: -len("-row_ptr.npy")])
        for suffix in ("gene_index", "values", "row_source_id"):
            if not Path(f"{stem}-{suffix}.npy").is_file():
                raise FileNotFoundError(f"missing {stem}-{suffix}.npy")
        rows = len(np.load(row_ptr, mmap_mode="r")) - 1
        if rows < 1:
            continue
        result.append(Shard(split, index, row_ptr, stem, offset, rows))
        offset += rows
    return result


def _load(stem: Path, name: str) -> np.ndarray | None:
    path = Path(f"{stem}-{name}.npy")
    return np.load(path, mmap_mode="r") if path.is_file() else None


def _metadata(stem: Path, name: str, rows: int) -> np.ndarray:
    value = _load(stem, name)
    if value is not None:
        if len(value) != rows:
            raise ValueError(f"{stem}-{name}.npy has {len(value)} rows, expected {rows}")
        return np.asarray(value)
    if name == "spatial":
        return np.full((rows, 3), np.nan, dtype=np.float32)
    if name in {"time_point_days", "time_lower_days", "time_upper_days", "chronological_time_proxy_days"}:
        return np.full(rows, np.nan, dtype=np.float32)
    if name in FLAG_FIELDS:
        return np.zeros(rows, dtype=bool)
    return np.full(rows, -1, dtype=np.int32)


def _stratum_code(stem: Path, rows: int) -> np.ndarray:
    assay = _metadata(stem, "assay", rows).astype(np.int64, copy=False)
    life = _metadata(stem, "life_stage", rows).astype(np.int64, copy=False)
    assay = np.where(assay >= 0, assay + 1, 0)
    life = np.where(life >= 0, life + 1, 0)
    code = assay * 16 + life
    for bit, name in enumerate(("annotation_eligible", "development_eligible", "spatial_eligible")):
        code = code * 2 + _metadata(stem, name, rows).astype(np.int64, copy=False)
    return code


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Stable vectorized row scores; lower values are selected first."""
    x = np.asarray(values, dtype=np.uint64) + np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _count_strata(shards: Iterable[Shard]) -> tuple[Counter[int], dict[int, int]]:
    counts: Counter[int] = Counter()
    first_group_row: dict[int, int] = {}
    for shard in shards:
        code = _stratum_code(shard.stem, shard.rows)
        values, frequencies = np.unique(code, return_counts=True)
        counts.update({int(k): int(v) for k, v in zip(values, frequencies)})
        if shard.split == "train":
            groups = _metadata(shard.stem, "split_group", shard.rows).astype(np.int64, copy=False)
            for group in np.unique(groups[groups >= 0]):
                group = int(group)
                if group not in first_group_row:
                    first = int(np.flatnonzero(groups == group)[0])
                    first_group_row[group] = shard.offset + first
    return counts, first_group_row


def _quotas(counts: Counter[int], target: int) -> dict[int, int]:
    if target < 1 or not counts:
        return {}
    keys = sorted(counts)
    total = float(sum(counts.values()))
    raw = {key: counts[key] * target / total for key in keys}
    result = {key: min(counts[key], int(np.floor(raw[key]))) for key in keys}
    # Keep every non-empty stratum represented when the budget permits it.
    if target >= len(keys):
        for key in keys:
            if result[key] == 0:
                result[key] = 1
    assigned = sum(result.values())
    if assigned < target:
        order = sorted(keys, key=lambda key: (raw[key] - np.floor(raw[key]), counts[key], -key), reverse=True)
        while assigned < target:
            changed = False
            for key in order:
                if result[key] < counts[key]:
                    result[key] += 1
                    assigned += 1
                    changed = True
                    if assigned == target:
                        break
            if not changed:
                break
    elif assigned > target:
        order = sorted(keys, key=lambda key: (result[key] > 1, raw[key], counts[key]), reverse=True)
        while assigned > target:
            changed = False
            for key in order:
                minimum = 1 if target >= len(keys) and counts[key] else 0
                if result[key] > minimum:
                    result[key] -= 1
                    assigned -= 1
                    changed = True
                    if assigned == target:
                        break
            if not changed:
                break
    return {key: value for key, value in result.items() if value > 0}


def _select_refs(shards: list[Shard], target: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    counts, first_group_row = _count_strata(shards)
    quotas = _quotas(counts, target)
    best_scores: dict[int, np.ndarray] = {key: np.empty(0, dtype=np.uint64) for key in quotas}
    best_refs: dict[int, np.ndarray] = {key: np.empty(0, dtype=np.int64) for key in quotas}
    for shard in shards:
        codes = _stratum_code(shard.stem, shard.rows)
        refs = shard.offset + np.arange(shard.rows, dtype=np.int64)
        scores = _splitmix64(refs, seed)
        for code, quota in quotas.items():
            positions = np.flatnonzero(codes == code)
            if not len(positions):
                continue
            keep = min(quota, len(positions))
            if len(positions) > keep:
                positions = positions[np.argpartition(scores[positions], keep - 1)[:keep]]
            merged_scores = np.concatenate((best_scores[code], scores[positions]))
            merged_refs = np.concatenate((best_refs[code], refs[positions]))
            if len(merged_refs) > quota:
                chosen = np.argpartition(merged_scores, quota - 1)[:quota]
                merged_scores, merged_refs = merged_scores[chosen], merged_refs[chosen]
            best_scores[code], best_refs[code] = merged_scores, merged_refs
    selected = np.concatenate(list(best_refs.values())) if best_refs else np.empty(0, dtype=np.int64)
    selected = np.unique(selected)
    # Ensure every train split group has at least one representative whenever
    # the requested budget allows it. The global split already prevents a
    # group from crossing train/validation/test; this only improves coverage.
    if shards and shards[0].split == "train" and len(selected) >= len(first_group_row):
        present = set()
        for shard in shards:
            groups = _metadata(shard.stem, "split_group", shard.rows).astype(np.int64, copy=False)
            local = selected[(selected >= shard.offset) & (selected < shard.offset + shard.rows)] - shard.offset
            if len(local):
                present.update(int(v) for v in np.unique(groups[local]) if v >= 0)
        missing = [row for group, row in sorted(first_group_row.items()) if group not in present]
        if missing:
            selected_scores = _splitmix64(selected, seed)
            replace = np.argsort(selected_scores)[-len(missing):]
            selected[replace] = np.asarray(missing, dtype=np.int64)
            selected = np.unique(selected)
    selected.sort()
    receipt = {
        "target_rows": int(target),
        "actual_rows": int(len(selected)),
        "strata": {str(key): {"available": int(counts[key]), "selected": int(quotas.get(key, 0))} for key in sorted(counts)},
        "train_group_count": int(len(first_group_row)) if shards and shards[0].split == "train" else None,
    }
    return selected, receipt


def _locate(shards: list[Shard], refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray([shard.offset for shard in shards], dtype=np.int64)
    indices = np.searchsorted(offsets, refs, side="right") - 1
    indices = np.clip(indices, 0, len(shards) - 1)
    local = refs - offsets[indices]
    for index, shard in enumerate(shards):
        mask = indices == index
        if np.any(local[mask] < 0) or np.any(local[mask] >= shard.rows):
            raise ValueError("row reference falls outside its shard")
    return indices.astype(np.int64, copy=False), local.astype(np.int64, copy=False)


def _gather(shards: list[Shard], refs: np.ndarray, name: str, dtype: Any | None = None) -> np.ndarray:
    shard_indices, locals_ = _locate(shards, refs)
    sample = _metadata(shards[0].stem, name, shards[0].rows)
    shape = (len(refs), *sample.shape[1:])
    result = np.empty(shape, dtype=dtype or sample.dtype)
    for index, shard in enumerate(shards):
        mask = shard_indices == index
        if np.any(mask):
            values = _metadata(shard.stem, name, shard.rows)
            result[mask] = np.asarray(values[locals_[mask]], dtype=result.dtype)
    return result


def _panel_catalog(root: Path) -> dict[int, np.ndarray]:
    source_path, ptr_path, gene_path = (root / f"panel_{name}.npy" for name in ("source_id", "ptr", "gene_index"))
    if not all(path.is_file() for path in (source_path, ptr_path, gene_path)):
        raise FileNotFoundError("panel_source_id.npy, panel_ptr.npy and panel_gene_index.npy are required")
    sources = np.load(source_path, mmap_mode="r")
    ptr = np.load(ptr_path, mmap_mode="r")
    genes = np.load(gene_path, mmap_mode="r")
    return {int(source): np.asarray(genes[int(ptr[i]): int(ptr[i + 1])]) for i, source in enumerate(sources)}


def _compute_panel(shards: list[Shard], refs: np.ndarray, panel_size: int) -> tuple[np.ndarray, int]:
    shard_indices, locals_ = _locate(shards, refs)
    max_gene = 0
    prevalence: np.ndarray | None = None
    for index, shard in enumerate(shards):
        mask = shard_indices == index
        if not np.any(mask):
            continue
        genes = np.load(Path(f"{shard.stem}-gene_index.npy"), mmap_mode="r")
        ptr = np.load(shard.row_ptr, mmap_mode="r")
        chunks: list[np.ndarray] = []
        for row in locals_[mask]:
            lo, hi = int(ptr[row]), int(ptr[row + 1])
            if hi > lo:
                chunk = np.asarray(genes[lo:hi], dtype=np.int64)
                chunks.append(chunk)
                if len(chunk):
                    max_gene = max(max_gene, int(chunk.max()))
        if chunks:
            if prevalence is None:
                prevalence = np.zeros(max_gene + 1, dtype=np.int64)
            if len(prevalence) <= max_gene:
                prevalence = np.pad(prevalence, (0, max_gene + 1 - len(prevalence)))
            tokens = np.concatenate(chunks)
            prevalence += np.bincount(tokens, minlength=len(prevalence))
    if prevalence is None or not np.any(prevalence):
        raise ValueError("selected train rows contain no expression tokens")
    panel_size = min(int(panel_size), int(np.count_nonzero(prevalence)))
    order = np.argsort(-prevalence, kind="stable")
    panel = np.sort(order[:panel_size].astype(np.int64, copy=False))
    return panel, int(len(prevalence))


def _dense_features(
    shards: list[Shard], refs: np.ndarray, panel: np.ndarray, panels: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shard_indices, locals_ = _locate(shards, refs)
    features = np.zeros((len(refs), len(panel)), dtype=np.float32)
    observed = np.zeros((len(refs), len(panel)), dtype=bool)
    library = np.zeros(len(refs), dtype=np.float32)
    for index, shard in enumerate(shards):
        mask = shard_indices == index
        if not np.any(mask):
            continue
        genes = np.load(Path(f"{shard.stem}-gene_index.npy"), mmap_mode="r")
        values = np.load(Path(f"{shard.stem}-values.npy"), mmap_mode="r")
        ptr = np.load(shard.row_ptr, mmap_mode="r")
        source_ids = _metadata(shard.stem, "row_source_id", shard.rows).astype(np.int64, copy=False)
        selected_positions = np.flatnonzero(mask)
        for out_index in selected_positions:
            row = int(locals_[out_index])
            lo, hi = int(ptr[row]), int(ptr[row + 1])
            row_genes = np.asarray(genes[lo:hi], dtype=np.int64)
            row_values = np.asarray(values[lo:hi], dtype=np.float32)
            total = float(row_values.sum(dtype=np.float64))
            library[out_index] = total
            if len(row_genes):
                positions = np.searchsorted(panel, row_genes)
                valid = (positions < len(panel)) & (panel[np.minimum(positions, len(panel) - 1)] == row_genes)
                if np.any(valid):
                    np.add.at(features[out_index], positions[valid], row_values[valid])
            source_panel = panels.get(int(source_ids[row]), np.empty(0, dtype=np.int64))
            if len(source_panel):
                positions = np.searchsorted(source_panel, panel)
                valid = (positions < len(source_panel)) & (source_panel[np.minimum(positions, len(source_panel) - 1)] == panel)
                observed[out_index] = valid
    scale = np.maximum(library[:, None], 1.0)
    features = np.log1p(features / scale * 10_000.0)
    return features, observed, library


def _neighbors(coordinates: np.ndarray, source: np.ndarray, section: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    neighbors = np.arange(len(coordinates), dtype=np.int32)[:, None]
    neighbors = np.broadcast_to(neighbors, (len(coordinates), max(1, int(k)))).copy()
    mask = np.zeros_like(neighbors, dtype=bool)
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return neighbors, mask
    groups: dict[tuple[int, int], list[int]] = {}
    finite = np.isfinite(coordinates[:, :2]).all(axis=1)
    for index in np.flatnonzero(finite):
        # A missing section is deliberately treated as no spatial information.
        if int(section[index]) < 0:
            continue
        groups.setdefault((int(source[index]), int(section[index])), []).append(int(index))
    for members in groups.values():
        if len(members) < 2:
            continue
        points = coordinates[members, :2]
        query_k = min(len(members), max(1, int(k)) + 1)
        _, local = cKDTree(points).query(points, k=query_k)
        local = np.asarray(local).reshape(len(members), query_k)
        for row, member in enumerate(members):
            chosen = [members[int(value)] for value in local[row] if members[int(value)] != member][: int(k)]
            if chosen:
                neighbors[member, : len(chosen)] = np.asarray(chosen, dtype=np.int32)
                mask[member, : len(chosen)] = True
    return neighbors, mask


