#!/usr/bin/env python3
"""Run API-backed third-party baselines on the frozen nanoBrain validation set."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.baselines import AdapterContext, build, names  # noqa: E402
from nanobrain.data import DatasetSplits, load_v3_subset, verify_subset_manifest  # noqa: E402
from scripts.run_nanobrain_baselines import (  # noqa: E402
    _append_group_classification_rows,
    _metric_rows,
    _write_npz,
    _write_rows,
)


def _parse_options(values: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"adapter option must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            output[key] = json.loads(raw)
        except json.JSONDecodeError:
            output[key] = raw
    return output


def _status_fields() -> list[str]:
    return [
        "method", "display_name", "tasks", "status", "reason", "started_at",
        "finished_at", "runtime_seconds", "library", "library_version",
        "prediction_artifacts", "metric_rows", "error_log",
    ]


def _write_status(output: Path, records: list[dict[str, Any]]) -> None:
    path = output / "method_status.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_status_fields(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output / "method_status.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "assets" / "nanobrain_v3_100k")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--methods", default="all", help=f"comma separated; available: {','.join(names())}")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--option", action="append", default=[], help="adapter option as KEY=JSON_VALUE")
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required for the single frozen-protocol test evaluation",
    )
    args = parser.parse_args()
    if args.evaluation_split == "test" and not args.allow_test:
        parser.error("test evaluation requires --allow-test after protocol freeze")
    if args.allow_test and args.evaluation_split != "test":
        parser.error("--allow-test is only valid with --evaluation-split test")

    selected = list(names()) if args.methods == "all" else [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(names()))
    if unknown:
        raise ValueError(f"unknown adapters: {unknown}; available={list(names())}")
    output = args.output or ROOT / "results" / "nanobrain" / f"external-baselines-{datetime.now():%Y%m%d-%H%M%S}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    (output / "errors").mkdir(exist_ok=True)

    threads = max(1, int(args.threads))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except (ImportError, RuntimeError):
        pass

    data_root = args.data.expanduser().resolve()
    source_manifest = verify_subset_manifest(
        data_root,
        verify_hashes=False,
        data_splits=("train", args.evaluation_split),
    )
    source_splits = load_v3_subset(
        data_root,
        verify=False,
        include_test=args.evaluation_split == "test",
    )
    if args.evaluation_split == "test":
        if source_splits.test is None:
            parser.error("test split is unavailable")
        splits = DatasetSplits(
            train=source_splits.train,
            validation=source_splits.test,
            test=None,
            development_train=source_splits.development_train,
            development_validation=source_splits.development_test,
            development_test=None,
        )
    else:
        splits = source_splits
    context = AdapterContext(
        train=splits.train,
        validation=splits.validation,
        data_root=data_root,
        output_root=output,
        seed=int(args.seed),
        device=str(args.device),
        threads=threads,
        development_train=splits.development_train,
        development_validation=splits.development_validation,
        options=_parse_options(args.option),
    )

    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for method_id in selected:
        adapter = build(method_id)
        adapter_metadata = adapter.metadata()
        availability = adapter.availability(context)
        start_wall = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        record: dict[str, Any] = {
            "method": method_id,
            "display_name": adapter.display_name,
            "tasks": ";".join(adapter.tasks),
            "status": availability.status,
            "reason": availability.reason,
            "started_at": start_wall,
            "library": adapter_metadata.get("library", ""),
            "library_version": adapter_metadata.get("library_version", ""),
            "prediction_artifacts": "",
            "metric_rows": 0,
            "error_log": "",
        }
        artifacts: list[str] = []
        method_rows: list[dict[str, Any]] = []
        if availability.runnable:
            try:
                predictions = adapter.run(context)
                if {prediction.task for prediction in predictions} != set(adapter.tasks):
                    raise RuntimeError(
                        f"adapter returned tasks {sorted(p.task for p in predictions)}, expected {sorted(adapter.tasks)}"
                    )
                for prediction in predictions:
                    artifact = output / "predictions" / f"{method_id}__{prediction.task}.npz"
                    _write_npz(artifact, **{key: np.asarray(value) for key, value in prediction.arrays.items()})
                    artifacts.append(str(artifact.relative_to(output)))
                    task_rows = _metric_rows(
                        task=prediction.task,
                        artifact=artifact,
                        method=method_id,
                        family=adapter.family,
                        regime=prediction.training_regime,
                        fit_rows=prediction.fit_rows,
                        eval_rows=prediction.eval_rows,
                        runtime=time.perf_counter() - started,
                        notes=prediction.notes,
                    )
                    if any(row.get("status") != "complete" for row in task_rows):
                        raise RuntimeError(task_rows[0].get("error", "metric evaluation failed"))
                    if prediction.task == "annotation":
                        _append_group_classification_rows(
                            task_rows,
                            artifact=artifact,
                            method=method_id,
                            family=adapter.family,
                            regime=prediction.training_regime,
                            fit_rows=prediction.fit_rows,
                            runtime=time.perf_counter() - started,
                            notes=prediction.notes,
                            dataset=splits.validation,
                        )
                    method_rows.extend(task_rows)
                rows.extend(method_rows)
                record["status"] = "complete"
                record["reason"] = ""
            except Exception as exc:  # optional libraries are isolated by design
                error_path = output / "errors" / f"{method_id}.log"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                record["status"] = "failed"
                record["reason"] = f"{type(exc).__name__}: {exc}"
                record["error_log"] = str(error_path.relative_to(output))
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["runtime_seconds"] = time.perf_counter() - started
        record["prediction_artifacts"] = ";".join(artifacts)
        record["metric_rows"] = len(method_rows)
        statuses.append(record)
        _write_rows(output / "benchmark_table.csv", rows)
        _write_status(output, statuses)
        print(json.dumps(record, sort_keys=True), flush=True)

    table_path = output / "benchmark_table.csv"
    _write_rows(table_path, rows)
    _write_status(output, statuses)
    manifest = {
        "format": "nanobrain_external_baseline_run_v1",
        "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
        "subset": "nanobrain_v3_100k",
        "fit_split": "train",
        "evaluation_split": args.evaluation_split,
        "test_opened": args.evaluation_split == "test",
        "methods_requested": selected,
        "method_status": {record["method"]: record["status"] for record in statuses},
        "metric_rows": len(rows),
        "prediction_artifacts": [artifact for record in statuses for artifact in record["prediction_artifacts"].split(";") if artifact],
        "data_manifest_sha256": hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest(),
        "data_manifest_format": source_manifest.get("format"),
        "seed": int(args.seed),
        "device": str(args.device),
        "threads": threads,
        "options": dict(context.options),
        "python": platform.python_version(),
        "runtime_seconds": time.perf_counter() - run_started,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **manifest}, indent=2, sort_keys=True))
    if any(record["status"] == "failed" for record in statuses):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
