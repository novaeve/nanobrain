#!/usr/bin/env python3
"""Build split-local NumPy row descriptors for developmental OT batches."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MISSING = {"", "nan", "none", "null", "<na>"}


def text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return None if value.lower() in MISSING else value


@functools.lru_cache(maxsize=None)
def stable_id(value):
    value = text(value)
    if value is None:
        return -1
    digest = hashlib.blake2b(
        value.encode(), digest_size=8, person=b"brainwm3"
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def output_prefix(split):
    return "development" if split == "train" else f"development_{split}"


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--cells-per-side", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    vocab = json.loads((args.shard_dir / "joint_metadata_vocab.json").read_text())
    anatomy_vocab = vocab["anatomy"]
    intervals = pd.read_parquet(args.benchmark_dir / "task_development_labels.parquet")
    intervals = intervals[intervals.split.eq(args.split)].set_index("interval_id")
    anatomy = pd.read_parquet(args.benchmark_dir / "development_interval_anatomy.parquet")
    anatomy = anatomy[anatomy.split.eq(args.split)]

    requested_samples = set()
    for row in intervals.itertuples():
        requested_samples.update(map(str, json.loads(row.source_sample_keys)))
        requested_samples.update(map(str, json.loads(row.target_sample_keys)))
    if args.split == "train":
        sample_to_id = {sample: vocab["sample"].get(sample, -1) for sample in requested_samples}
        sample_field = "sample"
    else:
        sample_to_id = {sample: stable_id(sample) for sample in requested_samples}
        sample_field = "evaluation_sample"
    valid_sample_ids = np.asarray(
        sorted({value for value in sample_to_id.values() if value >= 0}), dtype=np.int64
    )
    if not len(valid_sample_ids):
        raise RuntimeError(f"no known developmental samples for {args.split}")
    dense_by_id = {int(value): index for index, value in enumerate(valid_sample_ids)}

    keys = []
    references = []
    sample_files = sorted(args.shard_dir.glob(f"{args.split}-shard-*-{sample_field}.npy"))
    if not sample_files:
        raise FileNotFoundError(f"no {sample_field} sidecars for {args.split}")
    for path in sample_files:
        shard = int(path.name.split("-")[2])
        sample = np.load(path, mmap_mode="r")
        anatomy_path = Path(str(path).replace(f"-{sample_field}.npy", "-anatomy.npy"))
        anatomy_id = np.load(anatomy_path, mmap_mode="r")
        positions = np.searchsorted(valid_sample_ids, sample)
        bounded = np.minimum(positions, len(valid_sample_ids) - 1)
        known = (positions < len(valid_sample_ids)) & (valid_sample_ids[bounded] == sample)
        valid = known & (anatomy_id >= 0)
        rows = np.nonzero(valid)[0].astype(np.int64)
        if not len(rows):
            continue
        key = positions[valid].astype(np.int64) * len(anatomy_vocab) + anatomy_id[valid]
        keys.append(key)
        references.append((np.int64(shard) << 32) | rows)
    if not keys:
        raise RuntimeError(f"no developmental rows found for {args.split}")
    keys = np.concatenate(keys)
    references = np.concatenate(references)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    references = references[order]

    def rows_for(samples, anatomy_label):
        chunks = []
        anatomy_id = anatomy_vocab.get(str(anatomy_label))
        if anatomy_id is None:
            return np.empty(0, dtype=np.int64)
        for sample in samples:
            sample_id = sample_to_id.get(str(sample), -1)
            dense_id = dense_by_id.get(int(sample_id))
            if dense_id is None:
                continue
            key = dense_id * len(anatomy_vocab) + anatomy_id
            lower = np.searchsorted(keys, key, "left")
            upper = np.searchsorted(keys, key, "right")
            if upper > lower:
                chunks.append(references[lower:upper])
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    split_offset = {"train": 0, "validation": 1, "test": 2}[args.split]
    rng = np.random.default_rng(args.seed + split_offset)
    source = []
    target = []
    pointer = [0]
    metadata = []
    for row in anatomy.itertuples(index=False):
        interval = intervals.loc[row.interval_id]
        source_rows = rows_for(json.loads(interval.source_sample_keys), row.uberon_id)
        target_rows = rows_for(json.loads(interval.target_sample_keys), row.uberon_id)
        count = min(args.cells_per_side, len(source_rows), len(target_rows))
        if count < 2:
            continue
        source.append(rng.choice(source_rows, count, replace=False))
        target.append(rng.choice(target_rows, count, replace=False))
        pointer.append(pointer[-1] + count)
        metadata.append({
            "interval_id": row.interval_id,
            "uberon_id": row.uberon_id,
            "source_time_days": float(interval.source_time_days),
            "target_time_days": float(interval.target_time_days),
            "cells": count,
        })
    if not metadata:
        raise RuntimeError(f"no valid developmental endpoint pairs for {args.split}")

    prefix = output_prefix(args.split)
    atomic_save(args.shard_dir / f"{prefix}_source_ref.npy", np.concatenate(source))
    atomic_save(args.shard_dir / f"{prefix}_target_ref.npy", np.concatenate(target))
    atomic_save(args.shard_dir / f"{prefix}_pair_ptr.npy", np.asarray(pointer, dtype=np.int64))
    receipt = {
        "format": "brainwm_v3_development_pairs_v2",
        "split": args.split,
        "groups": len(metadata),
        "pairs": pointer[-1],
        "cells_per_side_max": args.cells_per_side,
        "pairing_runtime": "entropic_OT_barycentric",
        "groups_metadata": metadata,
    }
    (args.shard_dir / f"{prefix}_pairs.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({key: receipt[key] for key in ("split", "groups", "pairs")}))


if __name__ == "__main__":
    main()
