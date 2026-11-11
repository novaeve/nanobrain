#!/usr/bin/env python3
"""Fail only on corrupt core assets; report execution gates separately."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.data import verify_subset_manifest
from nanobrain.paths import locate_v3_shards


def _formal_roots() -> list[Path]:
    """Return likely formal checkouts without assuming logs are copied here."""
    candidates = [
        ROOT,
        Path(os.environ["BRAINWM_FORMAL_ROOT"]).expanduser()
        if os.environ.get("BRAINWM_FORMAL_ROOT") else None,
        Path("/data/brainwm/BrainWM-Training-Benchmark"),
    ]
    result = []
    seen = set()
    for value in candidates:
        if value is None:
            continue
        path = value.resolve()
        if path not in seen and path.is_dir():
            seen.add(path)
            result.append(path)
    return result


def _find_joint_smoke() -> tuple[Path | None, dict | None]:
    """Find a passed eight-loss smoke receipt in either checkout."""
    names = (
        "results/smoke/joint_all_losses_easy.json",
        "results/diagnostics/joint-objective-v3-corrected/smoke.json",
        "results/smoke/joint_all_losses.json",
    )
    required = {"mgm", "zero", "cell", "anatomy", "spatial", "flow", "completion", "expert"}
    for base in _formal_roots():
        for relative in names:
            path = base / relative
            if not path.is_file():
                continue
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if (value.get("passed") is True
                    and value.get("moe_backend") == "easy_deepspeed"
                    and set(value.get("losses", {})) == required
                    and value.get("parameters_with_finite_gradients", 0) > 0
                    and value.get("adamw_changed_tracked_parameter") is True):
                return path, value
    return None, None


def main():
    contract = ROOT / "contracts/healthy_global_benchmark_v3"
    frozen = json.loads((contract / "FROZEN_SHA256.json").read_text())
    mismatches = [name for name, digest in frozen.items()
                  if hashlib.sha256((contract / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise SystemExit("frozen contract checksum mismatch: " + ", ".join(mismatches))
    manifest = json.loads((contract / "manifest.json").read_text())
    if manifest["split_id"] != "BRAINWM-HUMAN-GLOBAL-6-2-2-V3":
        raise SystemExit("wrong split id")
    methods = list(csv.DictReader((ROOT / "registry/benchmark_methods.csv").open()))
    metrics = list(csv.DictReader((ROOT / "registry/benchmark_metrics.csv").open()))
    try:
        data_location = locate_v3_shards(project_root=ROOT)
        shard_root = data_location.root
    except FileNotFoundError:
        data_location = None
        shard_root = ROOT / "assets/v3_numpy_shards"
    shards = sorted(shard_root.glob("train-shard-*-gene_index.npy"))
    split_receipts={}
    metadata_receipts={}
    metadata_audits={}
    for split in ("train","validation","test"):
        path=shard_root/f"{split}-manifest.json"
        progress=shard_root/f"{split}-progress.json"
        split_receipts[split]=json.loads(path.read_text()) if path.exists() else (json.loads(progress.read_text()) if progress.exists() else None)
        metadata_path=shard_root/f"{split}-joint-metadata-manifest.json"
        audit_path=shard_root/f"{split}-joint-metadata-audit.json"
        metadata_receipts[split]=json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        metadata_audits[split]=json.loads(audit_path.read_text()) if audit_path.exists() else None
    all_complete=all(value and value.get("complete") for value in split_receipts.values())
    labels_complete=all(
        metadata_receipts[split]
        and metadata_receipts[split].get("rows") == split_receipts[split].get("rows")
        and metadata_receipts[split].get("shards") == split_receipts[split].get("shards")
        and metadata_audits[split]
        and metadata_audits[split].get("passed") is True
        and metadata_audits[split].get("required_sidecars") == 38
        for split in ("train", "validation", "test")
    )
    test_sealed=bool(
        metadata_receipts["test"]
        and metadata_receipts["test"].get("test_cell_type_sealed") is True
        and metadata_receipts["test"].get("test_labels_opened") is False
    )
    development_path=shard_root/"development_pairs.json"
    development=json.loads(development_path.read_text()) if development_path.exists() else None
    development_ready=bool(
        development and development.get("split") == "train"
        and development.get("groups",0)>0 and development.get("pairs",0)>0
    )
    smoke_path, smoke = _find_joint_smoke()
    smoke_ready=bool(
        smoke and smoke.get("passed") is True
        and smoke.get("moe_backend") == "easy_deepspeed"
        and set(smoke.get("losses",{})) == {
            "mgm","zero","cell","anatomy","spatial","flow","completion","expert"
        }
        and smoke.get("parameters_with_finite_gradients",0)>0
        and smoke.get("adamw_changed_tracked_parameter") is True
    )
    nano_subset_path = Path(
        os.environ.get("NANOBRAIN_DATA", str(ROOT / "assets" / "nanobrain_v3_100k"))
    ).expanduser().resolve()
    try:
        nano_manifest = verify_subset_manifest(
            nano_subset_path,
            verify_hashes=False,
            data_splits=("train", "validation"),
        )
        nano_subset_ready = all(
            int(nano_manifest["splits"][split]["rows"]) == expected
            for split, expected in (("train", 60000), ("validation", 20000), ("test", 20000))
        )
    except (FileNotFoundError, ValueError, KeyError):
        nano_manifest = None
        nano_subset_ready = False
    nano_smoke_receipts = sorted(
        (ROOT / "results" / "nanobrain").glob("smoke*/receipt-*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    nano_smoke = json.loads(nano_smoke_receipts[-1].read_text()) if nano_smoke_receipts else None
    nano_smoke_ready = bool(
        nano_smoke
        and nano_smoke.get("status") == "complete"
        and nano_smoke.get("gradient_finite_parameters", 0) > 0
        and nano_smoke.get("gradient_nonzero_parameters", 0) > 0
        and nano_smoke.get("optimizer_changed_parameter") is True
        and nano_smoke.get("data_manifest", {}).get("h5ad_in_training_loop") is False
        and nano_smoke.get("test_opened") is False
        and nano_smoke.get("data_manifest", {}).get("test_opened") is False
    )
    latest_path = ROOT / "results" / "nanobrain" / "LATEST_PILOT.json"
    latest = json.loads(latest_path.read_text()) if latest_path.is_file() else None
    benchmark_root = Path(latest["path"]) if latest and latest.get("path") else None
    benchmark_manifest_path = benchmark_root / "run_manifest.json" if benchmark_root else None
    benchmark_coverage_path = benchmark_root / "coverage.json" if benchmark_root else None
    benchmark_manifest = (
        json.loads(benchmark_manifest_path.read_text())
        if benchmark_manifest_path and benchmark_manifest_path.is_file() else None
    )
    benchmark_coverage = (
        json.loads(benchmark_coverage_path.read_text())
        if benchmark_coverage_path and benchmark_coverage_path.is_file() else None
    )
    expected_tasks = {"annotation", "integration", "spatial_niche", "spatial_completion", "development"}
    nano_benchmark_ready = bool(
        benchmark_manifest
        and benchmark_manifest.get("test_opened") is False
        and benchmark_coverage
        and benchmark_coverage.get("test_opened") is False
        and benchmark_coverage.get("error_rows") == 0
        and not benchmark_coverage.get("status_errors")
        and set(benchmark_coverage.get("evaluator_tasks", [])) == expected_tasks
        and all(
            benchmark_coverage.get("task_summary", {}).get(task, {}).get("complete_methods")
            for task in expected_tasks
        )
    )
    payload = {
        "split_id": manifest["split_id"], "contract_checksums": "passed",
        "methods": len(methods), "metrics": len(metrics),
        "model_import": False, "numpy_train_shards": len(shards),
        "dataset_location": data_location.as_dict() if data_location is not None else None,
        "split_materialization": {key: (value or {"complete":False,"rows":0}) for key,value in split_receipts.items()},
        "joint_metadata": {key: (value or {"rows":0,"shards":0}) for key,value in metadata_receipts.items()},
        "joint_metadata_audits": {key: (value or {"passed":False}) for key,value in metadata_audits.items()},
        "test_labels_sealed": test_sealed,
        "development_pairs": development or {"groups":0,"pairs":0},
        "joint_gradient_smoke": smoke or {"passed":False},
        "joint_gradient_smoke_receipt": str(smoke_path) if smoke_path else None,
        "nano_subset": {
            "path": str(nano_subset_path),
            "format": nano_manifest.get("format") if nano_manifest else None,
            "split_id": nano_manifest.get("split_id") if nano_manifest else None,
            "rows": {
                split: int(nano_manifest["splits"][split]["rows"])
                for split in ("train", "validation", "test")
            } if nano_manifest else None,
            "panel_size": int(nano_manifest.get("panel", {}).get("size", 0)) if nano_manifest else 0,
            "manifest_sha256": hashlib.sha256((nano_subset_path / "manifest.json").read_bytes()).hexdigest()
            if nano_manifest else None,
            "ready": nano_subset_ready,
            "smoke_receipt": str(nano_smoke_receipts[-1]) if nano_smoke_receipts else None,
            "smoke_ready": nano_smoke_ready,
        },
        "nano_benchmark": {
            "path": str(benchmark_root) if benchmark_root else None,
            "manifest": str(benchmark_manifest_path) if benchmark_manifest_path else None,
            "coverage": str(benchmark_coverage_path) if benchmark_coverage_path else None,
            "ready": nano_benchmark_ready,
            "metric_rows": benchmark_manifest.get("metric_rows", 0) if benchmark_manifest else 0,
            "methods": len(benchmark_manifest.get("methods", [])) if benchmark_manifest else 0,
            "external_status_counts": benchmark_coverage.get("external_status_counts", {}) if benchmark_coverage else {},
        },
        "full_training_ready": bool(data_location and shards) and all_complete and labels_complete and test_sealed and development_ready and smoke_ready,
        "readiness_gates": {
            "formal_numpy_data": bool(data_location and shards and all_complete),
            "formal_metadata_and_labels": labels_complete,
            "formal_test_sealed": test_sealed,
            "formal_development_pairs": development_ready,
            "formal_joint_gradient_smoke": smoke_ready,
            "nano_subset": nano_subset_ready,
            "nano_smoke": nano_smoke_ready,
            "nano_benchmark": nano_benchmark_ready,
        },
        "dependencies": {name: bool(importlib.util.find_spec(name)) for name in
                         ("torch", "numpy", "scipy", "sklearn", "scib", "scib_metrics", "rpy2")},
        "baseline_dependencies": {name: bool(importlib.util.find_spec(name)) for name in (
            "bbknn", "celltypist", "GraphST", "harmonypy", "scanorama", "scvi",
            "SpaGCN", "transformers", "xgboost",
        )},
    }
    from brainwm_foundation import BrainWMFoundationMoE
    payload["model_import"] = BrainWMFoundationMoE is not None
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
