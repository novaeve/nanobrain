#!/usr/bin/env python3
"""Verify that two BrainWM checkpoints encode the same training state."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainwm.training_observability import atomic_json, utc_now


STATE_FIELDS = ("model", "optimizer", "loss_balancer")
PROGRESS_FIELDS = ("step", "full_pass", "rows_in_pass", "rows_by_rank")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_value(
    reference: Any,
    candidate: Any,
    path: str,
    report: dict[str, Any],
) -> None:
    if isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor):
        report["tensor_count"] += 1
        if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            report["mismatches"].append({"path": path, "reason": "tensor_type"})
            return
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            report["mismatches"].append({
                "path": path,
                "reason": "tensor_contract",
                "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape),
                "reference_dtype": str(reference.dtype),
                "candidate_dtype": str(candidate.dtype),
            })
            return
        if not torch.equal(reference, candidate):
            delta = (reference.to(torch.float64) - candidate.to(torch.float64)).abs()
            report["mismatches"].append({
                "path": path,
                "reason": "tensor_value",
                "maximum_absolute_difference": float(delta.max()),
                "mean_absolute_difference": float(delta.mean()),
            })
        return
    if isinstance(reference, Mapping) or isinstance(candidate, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            report["mismatches"].append({"path": path, "reason": "mapping_type"})
            return
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        if reference_keys != candidate_keys:
            report["mismatches"].append({
                "path": path,
                "reason": "mapping_keys",
                "reference_only": sorted(map(str, reference_keys - candidate_keys)),
                "candidate_only": sorted(map(str, candidate_keys - reference_keys)),
            })
        for key in reference_keys & candidate_keys:
            compare_value(reference[key], candidate[key], f"{path}.{key}", report)
        return
    sequence_types = (list, tuple)
    if isinstance(reference, sequence_types) or isinstance(candidate, sequence_types):
        if not isinstance(reference, sequence_types) or not isinstance(candidate, sequence_types):
            report["mismatches"].append({"path": path, "reason": "sequence_type"})
            return
        if len(reference) != len(candidate):
            report["mismatches"].append({
                "path": path,
                "reason": "sequence_length",
                "reference": len(reference),
                "candidate": len(candidate),
            })
        for index, (left, right) in enumerate(zip(reference, candidate)):
            compare_value(left, right, f"{path}[{index}]", report)
        return
    report["scalar_count"] += 1
    if reference != candidate:
        report["mismatches"].append({
            "path": path,
            "reason": "value",
            "reference": reference,
            "candidate": candidate,
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    report: dict[str, Any] = {
        "format": "brainwm_checkpoint_equivalence_v1",
        "timestamp_utc": utc_now(),
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "reference_sha256": file_sha256(args.reference),
        "candidate_sha256": file_sha256(args.candidate),
        "compared_state_fields": list(STATE_FIELDS),
        "compared_progress_fields": list(PROGRESS_FIELDS),
        "tensor_count": 0,
        "scalar_count": 0,
        "mismatches": [],
    }
    for field in (*PROGRESS_FIELDS, *STATE_FIELDS):
        compare_value(reference.get(field), candidate.get(field), field, report)
    report["equivalent"] = not report["mismatches"]
    report["mismatch_count"] = len(report["mismatches"])

    if args.output is not None:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))
    if not report["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
