#!/usr/bin/env python3
"""Apply the predeclared compact-train-known annotation policy to test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.open_nanobrain_test_labels import (  # noqa: E402
    _atomic_save,
    _filter_to_compact_train_classes,
    _sha256,
    _write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "assets" / "nanobrain_v3_100k"
    )
    parser.add_argument(
        "--open-receipt",
        type=Path,
        default=(
            ROOT
            / "results/nanobrain/final-100k-20260916/test_label_open_receipt.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/nanobrain/final-100k-20260916/compact_train_label_filter_receipt.json"
        ),
    )
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.output.resolve()
    labels_path = data / "test" / "label_annotation.npy"
    mask_path = data / "test" / "mask_annotation.npy"
    manifest_path = data / "manifest.json"

    if output.is_file():
        receipt = json.loads(output.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError("existing compact-train filter receipt is incomplete")
        for key, path in (("labels", labels_path), ("mask", mask_path), ("manifest", manifest_path)):
            if _sha256(path) != receipt["after"][key]["sha256"]:
                raise ValueError(f"current {key} does not match compact-train filter receipt")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    open_path = args.open_receipt.resolve()
    opened = json.loads(open_path.read_text(encoding="utf-8"))
    if opened.get("status") != "complete":
        raise ValueError("test-label open receipt is incomplete")
    for key, path in (("labels", labels_path), ("mask", mask_path), ("manifest", manifest_path)):
        if _sha256(path) != opened["outputs"][key]["sha256"]:
            raise ValueError(f"pre-filter {key} does not match test-label open receipt")

    labels = np.load(labels_path)
    eligible = np.load(data / "test" / "annotation_eligible.npy")
    train_labels = np.load(data / "train" / "label_annotation.npy", mmap_mode="r")
    train_mask = np.load(data / "train" / "mask_annotation.npy", mmap_mode="r")
    corrected_labels, corrected_mask, stats = _filter_to_compact_train_classes(
        labels, eligible, train_labels, train_mask
    )
    before = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for key, path in (("labels", labels_path), ("mask", mask_path), ("manifest", manifest_path))
    }
    if stats["compact_train_unseen_rows"] < 1:
        raise ValueError("no compact-train-unseen test labels require filtering")
    _atomic_save(labels_path, corrected_labels)
    _atomic_save(mask_path, corrected_mask)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest["splits"]["test"]
    for key, path, value in (
        ("label_annotation", labels_path, corrected_labels),
        ("mask_annotation", mask_path, corrected_mask),
    ):
        split["files"][key].update(
            {
                "path": str(path.relative_to(data)),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _sha256(path),
            }
        )
    split["task_rows"]["annotation"] = int(corrected_mask.sum())
    manifest["test_label_access"]["compact_train_filter"] = {
        "state": "complete",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    _write_json(manifest_path, manifest)
    after = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for key, path in (("labels", labels_path), ("mask", mask_path), ("manifest", manifest_path))
    }
    receipt = {
        "format": "nanobrain_compact_train_label_filter_v1",
        "status": "complete",
        "protocol_id": opened["protocol_id"],
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "policy": "test classes absent from compact train are unknown and excluded",
        "test_label_open_receipt": str(open_path.relative_to(ROOT)),
        "test_label_open_receipt_sha256": _sha256(open_path),
        "before": before,
        "after": after,
        "coverage": stats,
        "post_filter_policy": "rerun every annotation- or biology-label-dependent test evaluator",
    }
    _write_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
