#!/usr/bin/env python3
"""Aggregate independent seeds of one nanoBrain architecture."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    (20260960, "230773410176717e", ROOT / "results/nanobrain/benchmark-pilot-gen3-candidate00"),
    (20260970, "fa1bc0b431658e35", ROOT / "results/nanobrain/benchmark-pilot-gen4-candidate00"),
    (20260980, "77f861004bbc4027", ROOT / "results/nanobrain/benchmark-pilot-parent-seed20260980"),
)
KEY_METRICS = {
    ("annotation", "accuracy"),
    ("annotation", "f1_macro"),
    ("integration", "integration_composite"),
    ("spatial_niche", "ari"),
    ("spatial_niche", "nmi"),
    ("spatial_completion", "heldout_rmse"),
    ("development", "population_mse"),
}
MINIMIZE_METRICS = {
    "rmse", "mae", "nonzero_rmse", "heldout_rmse", "heldout_mae",
    "row_mse", "row_mae", "population_mse", "mean_error", "variance_error",
    "mmd_rbf", "gearys_c", "ece", "nll", "brier_multiclass",
    "zero_calibration_error",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_run(value: str) -> tuple[int, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("run must be SEED:CANDIDATE_ID:TABLE_DIRECTORY")
    return int(parts[0]), parts[1], Path(parts[2]).expanduser().resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty aggregate: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_receipt(directory: Path, candidate_id: str) -> tuple[Path, dict[str, Any]] | None:
    search_roots = (directory, directory.parent)
    for root in search_roots:
        for path in sorted(root.glob("receipt-*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("candidate_id") == candidate_id:
                return path, payload
    return None


def _best_validation_step(receipt: dict[str, Any]) -> int | None:
    records = [
        record
        for record in receipt.get("validation_history", [])
        if isinstance(record, dict)
        and isinstance(record.get("quality"), (int, float))
        and math.isfinite(float(record["quality"]))
    ]
    if not records:
        return None
    return int(max(records, key=lambda record: float(record["quality"]))["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/nanobrain/parent-three-seed-20260916",
    )
    args = parser.parse_args()
    runs = list(args.run) or list(DEFAULT_RUNS)
    if len(runs) < 3:
        raise ValueError("at least three independent seeds are required")
    if len({seed for seed, _, _ in runs}) != len(runs):
        raise ValueError("replicate seeds must be unique")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    training_receipts: list[dict[str, Any]] = []
    for seed, candidate_id, directory in runs:
        table = directory / "benchmark_table.csv"
        manifest_path = directory / "run_manifest.json"
        if not table.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete replicate directory: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("test_opened") is not False:
            raise ValueError(f"replicate does not attest sealed test: {directory}")
        rows = _read(table)
        methods = {row["method"] for row in rows}
        expected_method = f"nanobrain_candidate-{candidate_id}"
        if methods != {expected_method}:
            raise ValueError(f"unexpected methods in {table}: {methods}")
        complete = [row for row in rows if row.get("status") == "complete" and row.get("value", "") != ""]
        if not complete:
            raise ValueError(f"no complete metrics in {table}")
        for row in complete:
            per_seed.append(
                {
                    "seed": seed,
                    "candidate_id": candidate_id,
                    "task": row["task"],
                    "metric": row["metric"],
                    "stratum_type": row.get("stratum_type", "overall"),
                    "stratum_value": row.get("stratum_value", "overall"),
                    "direction": "min" if row["metric"] in MINIMIZE_METRICS else row["direction"],
                    "value": float(row["value"]),
                    "eval_rows": int(row["eval_rows"]),
                    "source_table": str(table),
                }
            )
        source_record = {
                "seed": seed,
                "candidate_id": candidate_id,
                "directory": str(directory),
                "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                "metric_rows": len(complete),
                "checkpoint": manifest.get("checkpoint"),
            }
        training_source = _training_receipt(directory, candidate_id)
        if training_source is not None:
            receipt_path, training_receipt = training_source
            if training_receipt.get("test_opened") is not False:
                raise ValueError(f"training receipt does not attest sealed test: {receipt_path}")
            training_receipts.append(training_receipt)
            source_record.update(
                {
                    "training_receipt": str(receipt_path),
                    "training_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    "max_steps": int(training_receipt["steps"]),
                    "best_validation_step": _best_validation_step(training_receipt),
                }
            )
        source_records.append(source_record)

    if training_receipts and len(training_receipts) != len(runs):
        raise ValueError("training receipt coverage is incomplete across replicate seeds")
    if training_receipts:
        architectures = [
            {key: value for key, value in receipt["genome"].items() if key != "seed"}
            for receipt in training_receipts
        ]
        training_protocols = [
            {key: value for key, value in receipt["train"].items() if key != "seed"}
            for receipt in training_receipts
        ]
        if any(value != architectures[0] for value in architectures[1:]):
            raise ValueError("replicate architecture mismatch after excluding the seed")
        if any(value != training_protocols[0] for value in training_protocols[1:]):
            raise ValueError("replicate training protocol mismatch after excluding the seed")
        architecture: dict[str, Any] = architectures[0]
        training_protocol: dict[str, Any] = {
            **training_protocols[0],
            "checkpoint_selection": "maximum joint validation quality within the fixed budget",
            "best_validation_steps": [
                _best_validation_step(receipt) for receipt in training_receipts
            ],
        }
    else:
        architecture = {"status": "training_receipt_unavailable"}
        training_protocol = {"status": "training_receipt_unavailable"}

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in per_seed:
        key = (row["task"], row["metric"], row["stratum_type"], row["stratum_value"], row["direction"])
        grouped.setdefault(key, []).append(row)
    aggregate: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        task, metric, stratum_type, stratum_value, direction = key
        numbers = [float(row["value"]) for row in values]
        aggregate.append(
            {
                "task": task,
                "metric": metric,
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "direction": direction,
                "n_seeds": len(numbers),
                "mean": statistics.fmean(numbers),
                "std": statistics.stdev(numbers) if len(numbers) > 1 else math.nan,
                "min": min(numbers),
                "max": max(numbers),
                "seeds": ";".join(str(row["seed"]) for row in values),
                "values": ";".join(f"{number:.12g}" for number in numbers),
            }
        )
    key_rows = [
        row for row in aggregate
        if (row["task"], row["metric"]) in KEY_METRICS
        and row["stratum_type"] == "overall"
        and row["stratum_value"] == "overall"
    ]
    missing = sorted(KEY_METRICS - {(row["task"], row["metric"]) for row in key_rows})
    incomplete = [(row["task"], row["metric"], row["n_seeds"]) for row in key_rows if row["n_seeds"] != len(runs)]
    if missing or incomplete:
        raise ValueError(f"replicate coverage failed: missing={missing}, incomplete={incomplete}")

    _write_csv(output / "per_seed_metrics.csv", per_seed)
    _write_csv(output / "aggregate_metrics.csv", aggregate)
    key_payload = {
        f"{row['task']}.{row['metric']}": {
            "mean": row["mean"], "std": row["std"], "min": row["min"], "max": row["max"],
            "n_seeds": row["n_seeds"], "direction": row["direction"], "values": row["values"],
        }
        for row in key_rows
    }
    receipt = {
        "format": "nanobrain_three_seed_replicate_v2",
        "architecture": architecture,
        "training_protocol": training_protocol,
        "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
        "subset": "nanobrain_v3_100k",
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_opened": False,
        "independent_seeds": [seed for seed, _, _ in runs],
        "sources": source_records,
        "key_metrics": key_payload,
        "all_key_metrics_have_three_seeds": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# nanoBrain parent: three-seed validation",
        "",
        "The retained architecture was trained independently with a fixed maximum budget on seeds "
        + ", ".join(str(seed) for seed, _, _ in runs)
        + ". All values use the fixed validation split; the test split remained sealed.",
        "",
        "| Task | Metric | Mean | SD | Per-seed values |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in key_rows:
        report.append(
            f"| {row['task']} | {row['metric']} | {row['mean']:.6g} | {row['std']:.6g} | {row['values']} |"
        )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "seeds": receipt["independent_seeds"], "key_metrics": key_payload}, indent=2))


if __name__ == "__main__":
    main()
