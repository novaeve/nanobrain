#!/usr/bin/env python3
"""Audit every V3 metadata sidecar against its expression shard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CATEGORICAL = (
    "dataset", "assay", "technology", "donor", "sample", "section",
    "split_group", "split_unit", "sex", "disease_status", "life_stage",
    "time_status", "cl_id", "cl_term", "original_cell_type",
    "original_subclass", "cell_type", "uberon_id", "uberon_term",
    "original_anatomical_region", "anatomy", "integration_batch",
    "integration_biology",
)
NUMERIC = (
    "time_point_days", "time_lower_days", "time_upper_days",
    "chronological_time_proxy_days",
)
BOOLEAN = (
    "annotation_eligible", "development_eligible", "spatial_eligible",
    "integration_eligible",
)
SPATIAL = "spatial"
EVALUATION_IDS = (
    "evaluation_dataset", "evaluation_donor", "evaluation_sample",
    "evaluation_batch", "evaluation_cell_type", "evaluation_anatomy",
)
SEALED_TEST_LABELS = (
    "cell_type", "cl_id", "cl_term", "original_cell_type",
    "original_subclass",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    vocab = json.loads((args.shard_dir / "joint_metadata_vocab.json").read_text())
    ptrs = sorted(args.shard_dir.glob(f"{args.split}-shard-*-row_ptr.npy"))
    manifest = json.loads((args.shard_dir / f"{args.split}-manifest.json").read_text())
    failures: list[str] = []
    stats = {
        name: {"observed": 0, "missing": 0, "minimum": None, "maximum": None}
        for name in (*CATEGORICAL, *NUMERIC, *BOOLEAN, *EVALUATION_IDS, SPATIAL)
    }
    rows = 0

    for shard_number, ptr_path in enumerate(ptrs):
        expected = len(np.load(ptr_path, mmap_mode="r")) - 1
        rows += expected
        stem = str(ptr_path)[: -len("-row_ptr.npy")]
        for name in (*CATEGORICAL, *NUMERIC, *BOOLEAN, *EVALUATION_IDS, SPATIAL):
            path = Path(f"{stem}-{name}.npy")
            if not path.exists():
                failures.append(f"shard {shard_number}: missing {name}")
                continue
            value = np.load(path, mmap_mode="r")
            wanted_shape = (expected, 3) if name == SPATIAL else (expected,)
            if value.shape != wanted_shape:
                failures.append(
                    f"shard {shard_number}: {name} shape {value.shape} != {wanted_shape}"
                )
                continue
            if name in CATEGORICAL:
                missing = int(np.count_nonzero(value == -1))
                valid = value[value != -1]
                vocab_size = len(vocab[name])
                if valid.size and (int(valid.min()) < 0 or int(valid.max()) >= vocab_size):
                    failures.append(
                        f"shard {shard_number}: {name} outside [-1, {vocab_size - 1}]"
                    )
            elif name in NUMERIC or name == SPATIAL:
                missing = int(np.count_nonzero(~np.isfinite(value)))
                valid = value[np.isfinite(value)]
            elif name in EVALUATION_IDS:
                missing = int(np.count_nonzero(value == -1))
                valid = value[value != -1]
                if valid.size and int(valid.min()) < 0:
                    failures.append(f"shard {shard_number}: {name} has invalid negative ID")
            else:
                missing = 0
                valid = value
                if value.dtype != np.bool_:
                    failures.append(f"shard {shard_number}: {name} dtype {value.dtype} != bool")
            stat = stats[name]
            stat["missing"] += missing
            stat["observed"] += int(value.size) - missing
            if valid.size:
                lo, hi = valid.min().item(), valid.max().item()
                stat["minimum"] = lo if stat["minimum"] is None else min(stat["minimum"], lo)
                stat["maximum"] = hi if stat["maximum"] is None else max(stat["maximum"], hi)
            if args.split == "test" and name in (*SEALED_TEST_LABELS, "evaluation_cell_type") and np.any(value != -1):
                failures.append(f"shard {shard_number}: sealed test label {name} is visible")

    expected_shards = int(manifest["shards"])
    expected_rows = int(manifest["rows"])
    if len(ptrs) != expected_shards:
        failures.append(f"found {len(ptrs)} expression shards, manifest declares {expected_shards}")
    if rows != expected_rows:
        failures.append(f"found {rows} expression rows, manifest declares {expected_rows}")
    receipt = {
        "format": "brainwm_v3_joint_metadata_audit_v1",
        "split_id": manifest["split_id"],
        "split": args.split,
        "passed": not failures,
        "rows": rows,
        "shards": len(ptrs),
        "required_sidecars": len(CATEGORICAL) + len(NUMERIC) + len(BOOLEAN) + len(EVALUATION_IDS) + 1,
        "statistics": stats,
        "failures": failures,
    }
    output = args.output or args.shard_dir / f"{args.split}-joint-metadata-audit.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
