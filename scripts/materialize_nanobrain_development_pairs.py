#!/usr/bin/env python3
"""Materialize formal source/target development intervals inside nanoBrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_v3_development_pairs import stable_id  # noqa: E402
from scripts.materialize_nanobrain_subset import _discover, _gather  # noqa: E402
from nanobrain.paths import locate_v3_shards  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _parse_keys(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = list(value)
    return [str(item) for item in parsed]


def _materialize_split(
    *,
    split: str,
    compact_root: Path,
    shard_root: Path,
    contract_root: Path,
    vocab: dict[str, Any],
    cells_per_side: int,
    min_cells: int,
    seed: int,
) -> dict[str, Any]:
    directory = compact_root / split
    source_refs = np.load(directory / "source_row_ref.npy", mmap_mode="r")
    anatomy_ids = np.load(directory / "anatomy.npy", mmap_mode="r")
    if split == "train":
        sample_ids = np.load(directory / "sample.npy", mmap_mode="r")
    else:
        sample_ids = _gather(
            _discover(shard_root, split), source_refs, "evaluation_sample", np.int64
        )

    row_lookup: dict[tuple[int, int], list[int]] = {}
    for row, (sample_id, anatomy_id) in enumerate(zip(sample_ids, anatomy_ids)):
        if int(sample_id) >= 0 and int(anatomy_id) >= 0:
            row_lookup.setdefault((int(sample_id), int(anatomy_id)), []).append(row)

    intervals = pd.read_parquet(contract_root / "task_development_labels.parquet")
    intervals = intervals[intervals.split.eq(split)].set_index("interval_id")
    anatomy = pd.read_parquet(contract_root / "development_interval_anatomy.parquet")
    anatomy = anatomy[anatomy.split.eq(split)]
    anatomy_vocab = vocab["anatomy"]
    sample_vocab = vocab["sample"]
    rng = np.random.default_rng(int(seed) + {"train": 0, "validation": 1009, "test": 2018}[split])

    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    pointer = [0]
    metadata: list[dict[str, Any]] = []
    for row in anatomy.itertuples(index=False):
        if row.interval_id not in intervals.index:
            continue
        interval = intervals.loc[row.interval_id]
        anatomy_id = anatomy_vocab.get(str(row.uberon_id))
        if anatomy_id is None:
            continue
        source_samples = _parse_keys(interval.source_sample_keys)
        target_samples = _parse_keys(interval.target_sample_keys)
        if split == "train":
            source_sample_ids = [sample_vocab.get(value, -1) for value in source_samples]
            target_sample_ids = [sample_vocab.get(value, -1) for value in target_samples]
        else:
            source_sample_ids = [stable_id(value) for value in source_samples]
            target_sample_ids = [stable_id(value) for value in target_samples]
        source_rows = np.asarray(
            [
                index
                for sample_id in source_sample_ids
                for index in row_lookup.get((int(sample_id), int(anatomy_id)), ())
            ],
            dtype=np.int64,
        )
        target_rows = np.asarray(
            [
                index
                for sample_id in target_sample_ids
                for index in row_lookup.get((int(sample_id), int(anatomy_id)), ())
            ],
            dtype=np.int64,
        )
        count = min(int(cells_per_side), len(source_rows), len(target_rows))
        if count < int(min_cells):
            continue
        source_selected = rng.choice(source_rows, count, replace=False)
        target_selected = rng.choice(target_rows, count, replace=False)
        source.append(np.asarray(source_selected, dtype=np.int32))
        target.append(np.asarray(target_selected, dtype=np.int32))
        pointer.append(pointer[-1] + count)
        metadata.append(
            {
                "interval_id": str(row.interval_id),
                "uberon_id": str(row.uberon_id),
                "source_time_days": float(interval.source_time_days),
                "target_time_days": float(interval.target_time_days),
                "cells": int(count),
                "available_source_cells": int(len(source_rows)),
                "available_target_cells": int(len(target_rows)),
            }
        )
    if not metadata:
        raise RuntimeError(f"no formal development intervals survived in {split}")

    arrays = {
        "development_source_index": np.concatenate(source),
        "development_target_index": np.concatenate(target),
        "development_pair_ptr": np.asarray(pointer, dtype=np.int64),
    }
    files: dict[str, Any] = {}
    for name, value in arrays.items():
        path = directory / f"{name}.npy"
        _atomic_npy(path, value)
        files[name] = {
            "path": str(path.relative_to(compact_root)),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256(path),
        }
    receipt = {
        "format": "nanobrain_population_pairs_v1",
        "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
        "split": split,
        "groups": len(metadata),
        "pairs": int(pointer[-1]),
        "cells_per_side_max": int(cells_per_side),
        "cells_per_side_min": int(min_cells),
        "pairing_runtime": "train-time balanced Sinkhorn barycentric matching",
        "cell_correspondence": "unpaired_cross_sectional",
        "groups_metadata": metadata,
    }
    receipt_path = directory / "development_pairs.json"
    _atomic_json(receipt_path, receipt)
    files["development_pairs"] = {
        "path": str(receipt_path.relative_to(compact_root)),
        "sha256": _sha256(receipt_path),
    }
    return {"receipt": receipt, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "assets/nanobrain_v3_100k")
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "contracts/healthy_global_benchmark_v3",
    )
    parser.add_argument("--source-shards", type=Path, default=None)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--cells-per-side", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required when --splits includes test; use only after protocol freeze",
    )
    args = parser.parse_args()
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not splits or set(splits) - {"train", "validation", "test"}:
        parser.error("--splits must contain train, validation, or test")
    if "test" in splits and not args.allow_test:
        parser.error("materializing test pairs requires --allow-test after protocol freeze")
    if args.cells_per_side < args.min_cells or args.min_cells < 2:
        parser.error("require cells-per-side >= min-cells >= 2")

    compact_root = args.data.resolve()
    contract_root = args.contract.resolve()
    shard_root = locate_v3_shards(args.source_shards, project_root=ROOT).root
    manifest_path = compact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vocab = json.loads((shard_root / "joint_metadata_vocab.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {}
    for split in splits:
        result = _materialize_split(
            split=split,
            compact_root=compact_root,
            shard_root=shard_root,
            contract_root=contract_root,
            vocab=vocab,
            cells_per_side=args.cells_per_side,
            min_cells=args.min_cells,
            seed=args.seed,
        )
        manifest["splits"][split].setdefault("files", {}).update(result["files"])
        manifest["splits"][split]["population_development"] = {
            "groups": result["receipt"]["groups"],
            "pairs": result["receipt"]["pairs"],
        }
        summary[split] = {
            "groups": result["receipt"]["groups"],
            "pairs": result["receipt"]["pairs"],
        }
    manifest["development_policy"] = (
        "formal unpaired cross-sectional source/target populations; "
        "balanced Sinkhorn matching and conditional Flow Matching"
    )
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"data": str(compact_root), "splits": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
