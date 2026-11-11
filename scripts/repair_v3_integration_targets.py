#!/usr/bin/env python3
"""Replace mistakenly encoded integration schema strings with row targets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = Path(f"{path}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    args = parser.parse_args()

    vocab_path = args.shard_dir / "joint_metadata_vocab.json"
    vocab = json.loads(vocab_path.read_text())
    vocab["integration_batch"] = dict(vocab["dataset"])
    vocab["integration_biology"] = dict(vocab["cell_type"])
    vocab_path.write_text(json.dumps(vocab, indent=2, ensure_ascii=False) + "\n")

    dataset_paths = sorted(args.shard_dir.glob(f"{args.split}-shard-*-dataset.npy"))
    for dataset_path in dataset_paths:
        stem = str(dataset_path)[: -len("-dataset.npy")]
        dataset = np.asarray(np.load(dataset_path, mmap_mode="r"), dtype=np.int32)
        cell_type = np.asarray(np.load(f"{stem}-cell_type.npy", mmap_mode="r"), dtype=np.int32)
        atomic_save(Path(f"{stem}-integration_batch.npy"), dataset)
        atomic_save(Path(f"{stem}-integration_biology.npy"), cell_type)

    manifest_path = args.shard_dir / f"{args.split}-joint-metadata-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["vocab_sizes"] = {name: len(values) for name, values in vocab.items()}
    manifest["integration_targets"] = {
        "primary_batch": "dataset",
        "primary_biology": "cell_type",
        "batch_factors": ["assay", "technology", "dataset", "donor", "sample"],
        "biology_factors": ["anatomy", "development", "sex", "cell_type"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"split": args.split, "shards": len(dataset_paths), "rows": manifest["rows"]}))


if __name__ == "__main__":
    main()
