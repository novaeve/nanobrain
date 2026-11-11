#!/usr/bin/env python3
"""Audit method/task coverage in a merged nanoBrain benchmark table.

The audit is intentionally derived from prediction artifacts and receipts, so
registry entries marked ``planned`` cannot be mistaken for measured results.
It emits a compact JSON ledger suitable for reports and CI checks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HIGH_LEVEL_TASK = {
    "annotation": "annotation",
    "integration": "integration",
    "spatial_niche": "spatial",
    "spatial_completion": "spatial",
    "development": "development",
}


def _resolve_artifact(table: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else table.parent / candidate


def audit(table: Path) -> dict[str, Any]:
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        pairs.setdefault((row.get("method", ""), row.get("task", "")), []).append(row)

    entries: list[dict[str, Any]] = []
    for (method, task), group in sorted(pairs.items()):
        artifacts = sorted({row.get("prediction_artifact", "") for row in group if row.get("prediction_artifact")})
        artifact_checks = []
        for raw in artifacts:
            path = _resolve_artifact(table, raw)
            receipt = path.with_suffix(".metrics.json")
            receipt_status = None
            if receipt.is_file():
                try:
                    receipt_status = json.loads(receipt.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    receipt_status = {"parse_error": True}
            artifact_checks.append({
                "path": str(path),
                "exists": path.is_file(),
                "receipt": str(receipt),
                "receipt_exists": receipt.is_file(),
                "receipt_status": receipt_status.get("status") if isinstance(receipt_status, dict) else None,
            })
        complete_rows = [row for row in group if row.get("status") == "complete" and not row.get("error")]
        overall_metrics = sorted({row.get("metric", "") for row in group if row.get("stratum_type") == "overall"})
        complete = bool(complete_rows) and bool(artifact_checks) and all(
            check["exists"] and check["receipt_exists"] for check in artifact_checks
        )
        entries.append({
            "method": method,
            "task": task,
            "high_level_task": HIGH_LEVEL_TASK.get(task, task),
            "status": "complete" if complete else "incomplete",
            "metric_rows": len(group),
            "overall_metrics": overall_metrics,
            "artifacts": artifact_checks,
            "fit_rows": sorted({int(row["fit_rows"]) for row in group if row.get("fit_rows", "").isdigit()}),
            "eval_rows": sorted({int(row["eval_rows"]) for row in group if row.get("eval_rows", "").isdigit()}),
            "source_runs": sorted({row.get("source_run", "") for row in group}),
        })

    task_summary: dict[str, dict[str, Any]] = {}
    for entry in entries:
        task = entry["task"]
        summary = task_summary.setdefault(task, {"high_level_task": entry["high_level_task"], "methods": [], "complete_methods": []})
        summary["methods"].append(entry["method"])
        if entry["status"] == "complete":
            summary["complete_methods"].append(entry["method"])
    for summary in task_summary.values():
        summary["methods"] = sorted(set(summary["methods"]))
        summary["complete_methods"] = sorted(set(summary["complete_methods"]))

    method_status_path = table.with_name("method_status.json")
    method_status: list[dict[str, Any]] = []
    if method_status_path.is_file():
        payload = json.loads(method_status_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"method status ledger must be a list: {method_status_path}")
        method_status = [dict(record) for record in payload]
    table_methods = {row.get("method", "") for row in rows}
    status_counts: dict[str, int] = {}
    status_errors: list[str] = []
    for record in method_status:
        method = str(record.get("method", ""))
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "complete" and method not in table_methods:
            status_errors.append(f"{method}: complete status has no metric rows")
        if status != "complete" and method in table_methods:
            status_errors.append(f"{method}: {status} status conflicts with metric rows")

    return {
        "format": "nanobrain_benchmark_coverage_v1",
        "table": str(table.resolve()),
        "rows": len(rows),
        "methods": sorted({row.get("method", "") for row in rows}),
        "evaluator_tasks": sorted(task_summary),
        "high_level_tasks": sorted(set(HIGH_LEVEL_TASK.values())),
        "test_opened": any(row.get("split_id") and row.get("test_opened") == "true" for row in rows),
        "error_rows": sum(1 for row in rows if row.get("status") != "complete" or row.get("error")),
        "method_status_path": str(method_status_path) if method_status_path.is_file() else None,
        "method_status": method_status,
        "external_status_counts": status_counts,
        "status_errors": status_errors,
        "unmeasured_external_methods": sorted(
            str(record.get("method", ""))
            for record in method_status
            if record.get("status") != "complete"
        ),
        "task_summary": task_summary,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("results/nanobrain/benchmark-pilot-20260915-final/benchmark_table.csv"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on metric errors, status inconsistencies, or an opened test split",
    )
    args = parser.parse_args()
    table = args.table.resolve()
    if not table.is_file():
        raise FileNotFoundError(table)
    output = (args.output or table.with_name("coverage.json")).resolve()
    report = audit(table)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "rows": report["rows"],
        "methods": len(report["methods"]),
        "evaluator_tasks": report["evaluator_tasks"],
        "error_rows": report["error_rows"],
        "external_status_counts": report["external_status_counts"],
        "status_errors": report["status_errors"],
    }, indent=2, sort_keys=True))
    if args.strict and (report["error_rows"] or report["status_errors"] or report["test_opened"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
