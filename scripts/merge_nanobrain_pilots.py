#!/usr/bin/env python3
"""Merge independent nanoBrain pilot runs into one auditable table.

The source runs are intentionally immutable.  This command selects the
non-overlapping rows (the fast pilot for all methods, the logistic-only
extension, and fully evaluated gen2 candidates), hard-links their prediction
artifacts into one directory, and rewrites artifact paths in the merged CSV.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "split_id", "subset", "task", "method", "method_family", "training_regime",
    "fit_rows", "eval_rows", "status", "runtime_seconds", "prediction_artifact",
    "metric", "value", "direction", "stratum_type", "stratum_value", "notes", "error",
]
STATUS_FIELDS = [
    "method", "display_name", "tasks", "status", "reason", "started_at",
    "finished_at", "runtime_seconds", "library", "library_version",
    "prediction_artifacts", "metric_rows", "error_log", "source_run",
]
MINIMIZE_METRICS = {
    "rmse", "mae", "nonzero_rmse", "heldout_rmse", "heldout_mae",
    "row_mse", "row_mae", "population_mse", "mean_error", "variance_error",
    "mmd_rbf", "gearys_c", "ece", "nll", "brier_multiclass",
    "zero_calibration_error",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _copy_artifacts(source: Path, target: Path, prefix: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    source_predictions = source / "predictions"
    target_predictions = target / "predictions"
    target_predictions.mkdir(parents=True, exist_ok=True)
    for artifact in sorted(source_predictions.glob("*.npz")):
        destination_name = f"{prefix}__{artifact.name}"
        destination = target_predictions / destination_name
        # Hard links preserve a single physical copy on the ESSD while making
        # the consolidated directory independently addressable.
        try:
            destination.unlink(missing_ok=True)
            destination.hardlink_to(artifact)
        except OSError:
            shutil.copy2(artifact, destination)
        mapping[str(artifact)] = str(destination)
        receipt = artifact.with_suffix(".metrics.json")
        if receipt.is_file():
            receipt_destination = target_predictions / f"{prefix}__{receipt.name}"
            try:
                receipt_destination.unlink(missing_ok=True)
                receipt_destination.hardlink_to(receipt)
            except OSError:
                shutil.copy2(receipt, receipt_destination)
    return mapping


def _collect_method_status(
    sources: list[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect one auditable status per external method.

    A measured ``complete`` record always wins over an unavailable record for
    the same method.  Other duplicate statuses must agree, which prevents a
    stale failed run from silently replacing a current resource diagnosis.
    """
    by_method: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    for source_name, source in sources:
        status_path = source / "method_status.json"
        manifest_path = source / "run_manifest.json"
        if not status_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete status run: {source}")
        records = json.loads(status_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("test_opened") is not False:
            raise ValueError(f"status run does not attest sealed test: {source}")
        source_records.append({
            "name": source_name,
            "path": str(source),
            "status_records": len(records),
            "status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        })
        for raw in records:
            record = dict(raw)
            method = str(record.get("method", ""))
            if not method:
                raise ValueError(f"method status without method id: {status_path}")
            record["source_run"] = source_name
            previous = by_method.get(method)
            if previous is None or (record.get("status") == "complete" and previous.get("status") != "complete"):
                by_method[method] = record
            elif previous.get("status") != record.get("status"):
                raise ValueError(
                    f"conflicting non-complete statuses for {method}: "
                    f"{previous.get('status')} vs {record.get('status')}"
                )
    return [by_method[key] for key in sorted(by_method)], source_records


def _write_method_status(output: Path, records: list[dict[str, Any]]) -> None:
    with (output / "method_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output / "method_status.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "nanobrain" / "benchmark-pilot-20260915-final")
    parser.add_argument(
        "--external-run", action="append", type=Path, default=[],
        help="additional adapter-run directory containing benchmark_table.csv and predictions/",
    )
    parser.add_argument(
        "--status-run", action="append", type=Path, default=[],
        help="adapter-run directory contributing statuses but no measured rows",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Keep the source runs explicit and deterministic.  The v2 run contains
    # all fast controls plus gen1; the logistic run contributes only its new
    # method; generation candidates, deep-control and annotation-extension
    # rows are selected from their independent evaluator passes.
    sources: list[tuple[str, Path, str]] = [
        ("fast_controls_and_gen1", ROOT / "results/nanobrain/benchmark-pilot-20260915-v2", "v2"),
        ("logistic_extension", ROOT / "results/nanobrain/benchmark-pilot-20260915-logistic", "logistic"),
        ("gen2_candidate01", ROOT / "results/nanobrain/benchmark-pilot-gen2-candidate01", "gen2c01"),
        ("gen2_candidate00", ROOT / "results/nanobrain/benchmark-pilot-gen2-candidate00", "gen2c00"),
        ("gen3_candidate00", ROOT / "results/nanobrain/benchmark-pilot-gen3-candidate00", "gen3c00"),
        ("gen3_candidate01", ROOT / "results/nanobrain/benchmark-pilot-gen3-candidate01", "gen3c01"),
        ("gen3_candidate02", ROOT / "results/nanobrain/benchmark-pilot-gen3-candidate02", "gen3c02"),
        ("gen4_candidate00", ROOT / "results/nanobrain/benchmark-pilot-gen4-candidate00", "gen4c00"),
        ("gen4_candidate01", ROOT / "results/nanobrain/benchmark-pilot-gen4-candidate01", "gen4c01"),
        ("gen4_candidate02", ROOT / "results/nanobrain/benchmark-pilot-gen4-candidate02", "gen4c02"),
        ("parent_seed20260980", ROOT / "results/nanobrain/benchmark-pilot-parent-seed20260980", "parentseed80"),
        ("deep_mlp_shared", ROOT / "results/nanobrain/deep-baseline-20260915", "deepmlp"),
        ("annotation_extra", ROOT / "results/nanobrain/benchmark-pilot-20260915-annotation-extra-v2", "annx"),
    ]
    for index, source in enumerate(args.external_run):
        sources.append((f"external_{index}_{source.name}", source.resolve(), f"external{index}"))
    status_sources = [
        (source_name, source)
        for source_name, source, _ in sources
        if (source / "method_status.json").is_file()
    ]
    status_sources.extend(
        (f"status_{index}_{source.name}", source.resolve())
        for index, source in enumerate(args.status_run)
    )
    selected: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    for source_name, source, prefix in sources:
        table = source / "benchmark_table.csv"
        if not table.is_file():
            raise FileNotFoundError(table)
        mapping = _copy_artifacts(source, output, prefix)
        rows = _read(table)
        source_records.append({
            "name": source_name,
            "path": str(source),
            "table": str(table),
            "rows": len(rows),
            "artifact_count": len(mapping),
        })
        for row in rows:
            # The fast v2 table already has the methods we want.  The logistic
            # source is selected only for that method; gen2 and deep sources
            # are selected wholesale because their method IDs are unique.
            method = row.get("method", "")
            if row.get("metric") in MINIMIZE_METRICS:
                row["direction"] = "min"
            if source_name == "logistic_extension" and method != "logistic_regression":
                continue
            if source_name == "fast_controls_and_gen1" and method == "logistic_regression":
                continue
            key = (
                row.get("task", ""), method, row.get("metric", ""),
                row.get("stratum_type", ""), row.get("stratum_value", ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            original = row.get("prediction_artifact", "")
            if original in mapping:
                row["prediction_artifact"] = str(Path(mapping[original]).relative_to(output))
            row["source_run"] = source_name
            selected.append(row)
    table_path = output / "benchmark_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + ["source_run"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    method_status, status_source_records = _collect_method_status(status_sources)
    _write_method_status(output, method_status)
    status_counts: dict[str, int] = {}
    for record in method_status:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "format": "nanobrain_pilot_benchmark_merged_v1",
        "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
        "subset": "nanobrain_v3_100k",
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_opened": False,
        "sources": source_records,
        "status_sources": status_source_records,
        "external_method_status": {
            str(record["method"]): str(record.get("status", "unknown"))
            for record in method_status
        },
        "external_status_counts": status_counts,
        "metric_rows": len(selected),
        "tasks": sorted({row.get("task", "") for row in selected}),
        "methods": sorted({row.get("method", "") for row in selected}),
        "table": str(table_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "sources.json").write_text(json.dumps(source_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pointer = ROOT / "results/nanobrain/LATEST_PILOT.json"
    pointer.write_text(json.dumps({"path": str(output), **manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metric_rows": len(selected), "tasks": manifest["tasks"], "methods": manifest["methods"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
