#!/usr/bin/env python3
"""Audit and finalize the frozen nanoBrain 100k test campaign.

This is a reporting-only command. It reads existing prediction metric tables
and manifests, verifies the frozen campaign contract, and writes aggregate
tables and receipts. It never loads a model or invokes an evaluator.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "results/nanobrain/final-100k-20260916/test-v3"
SPLIT_ID = "BRAINWM-HUMAN-GLOBAL-6-2-2-V3"
SUBSET = "nanobrain_v3_100k"
EXPECTED_EXTERNAL_METHODS = {
    "bbknn",
    "celltypist",
    "geneformer_v1_10m",
    "graphst",
    "harmony",
    "scanorama",
    "scfit",
    "scgpt_brain",
    "scgpt_human",
    "scnode",
    "scvi_scanvi",
    "trajectorynet",
    "xgboost",
}
KEY_METRICS = (
    ("annotation", "accuracy"),
    ("annotation", "f1_macro"),
    ("integration", "integration_composite"),
    ("spatial_niche", "ari"),
    ("spatial_niche", "nmi"),
    ("spatial_completion", "heldout_rmse"),
    ("development", "population_mse"),
    ("development", "mmd_rbf"),
)
MINIMIZE_METRICS = {
    "rmse",
    "mae",
    "nonzero_rmse",
    "heldout_rmse",
    "heldout_mae",
    "row_mse",
    "row_mae",
    "population_mse",
    "mean_error",
    "variance_error",
    "mmd_rbf",
    "gearys_c",
    "ece",
    "nll",
    "brier_multiclass",
    "zero_calibration_error",
}
DISPLAY_NAMES = {
    "nanobrain_fmt": "nanoBrain FMT",
    "batch_centered_pca": "Batch-centered PCA",
    "bbknn": "BBKNN",
    "celltypist": "CellTypist",
    "development_global_delta": "Global delta",
    "development_identity": "Identity",
    "development_ridge_delta": "Ridge delta",
    "gene_mean": "Gene mean",
    "geneformer_v1_10m": "Geneformer V1-10M",
    "graphst": "GraphST",
    "harmony": "Harmony",
    "knn": "kNN",
    "linear_sgd": "Linear SGD",
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic regression",
    "majority": "Majority",
    "multinomial_nb": "Multinomial NB",
    "nearest_centroid": "Nearest centroid",
    "raw_pca": "Raw PCA",
    "row_mean": "Row mean",
    "scanorama": "Scanorama",
    "scfit": "scFiT",
    "scgpt_brain": "scGPT-brain",
    "scgpt_human": "scGPT-human",
    "scnode": "scNODE",
    "scvi_scanvi": "scVI/scANVI",
    "spatial_coordinate_kmeans": "Coordinate K-means",
    "spatial_neighbor_mean": "Spatial neighbor mean",
    "spatial_no_structure": "No spatial structure",
    "trajectorynet": "TrajectoryNet",
    "xgboost": "XGBoost",
    "zero_fill": "Zero fill",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _direction(row: dict[str, str]) -> str:
    return "min" if row.get("metric") in MINIMIZE_METRICS else row.get("direction", "max")


def _numeric(value: str) -> float | None:
    if value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _checkpoint_audit(
    checkpoint: Path, training_receipt: dict[str, Any]
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"unsupported nanoBrain checkpoint payload: {checkpoint}")
    history = [
        record
        for record in training_receipt.get("validation_history", [])
        if isinstance(record, dict)
        and isinstance(record.get("quality"), (int, float))
        and math.isfinite(float(record["quality"]))
    ]
    if not history:
        raise ValueError(f"training receipt has no finite validation history: {checkpoint}")
    best_record = max(history, key=lambda record: float(record["quality"]))
    embedded_quality = payload.get(
        "validation_quality", payload.get("best_validation_quality")
    )
    if not isinstance(embedded_quality, (int, float)) or not math.isclose(
        float(embedded_quality), float(best_record["quality"]), rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise ValueError(f"checkpoint best-quality metadata mismatch: {checkpoint}")
    final_value = training_receipt.get("checkpoint")
    final_checkpoint = Path(str(final_value)) if final_value else None
    if final_checkpoint is not None and not final_checkpoint.is_absolute():
        final_checkpoint = ROOT / final_checkpoint
    weights_equal_to_final: bool | None = None
    if final_checkpoint is not None and final_checkpoint.is_file():
        final_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        final_state = final_payload.get("model") if isinstance(final_payload, dict) else None
        best_state = payload["model"]
        if not isinstance(final_state, dict) or set(final_state) != set(best_state):
            raise ValueError(f"best/final checkpoint state schema mismatch: {checkpoint}")
        weights_equal_to_final = all(torch.equal(best_state[key], final_state[key]) for key in best_state)
        if int(best_record["step"]) != int(training_receipt["steps"]) and weights_equal_to_final:
            raise ValueError(f"historical best unexpectedly equals final-step weights: {checkpoint}")
    return {
        "historical_best_step": int(best_record["step"]),
        "historical_best_quality": float(best_record["quality"]),
        "embedded_step": int(payload.get("step", -1)),
        "embedded_best_quality": float(embedded_quality),
        "step_metadata_matches_history": int(payload.get("step", -1)) == int(best_record["step"]),
        "weights_equal_to_final_checkpoint": weights_equal_to_final,
        "parameter_count": sum(
            int(value.numel()) for value in payload["model"].values() if hasattr(value, "numel")
        ),
    }


def _validate_metric_table(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"empty metric table: {path}")
    required = {
        "split_id",
        "subset",
        "task",
        "method",
        "training_regime",
        "eval_rows",
        "status",
        "metric",
        "value",
        "direction",
        "stratum_type",
        "stratum_value",
        "error",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    bad = [
        row
        for row in rows
        if row.get("status") != "complete" or row.get("error", "").strip()
    ]
    if bad:
        raise ValueError(f"metric table contains {len(bad)} failed/error rows: {path}")
    for row in rows:
        if row["split_id"] != SPLIT_ID or row["subset"] != SUBSET:
            raise ValueError(f"split/subset mismatch in {path}")
        if row["value"]:
            number = float(row["value"])
            if not math.isfinite(number):
                raise ValueError(f"non-finite metric in {path}: {row}")
        if _direction(row) not in {"min", "max"}:
            raise ValueError(f"invalid metric direction in {path}: {row}")
    return rows


def _validate_run_manifest(path: Path, data_manifest_sha256: str) -> dict[str, Any]:
    manifest = _read_json(path)
    expected = {
        "split_id": SPLIT_ID,
        "fit_split": "train",
        "evaluation_split": "test",
        "test_opened": True,
        "data_manifest_sha256": data_manifest_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"run manifest mismatch in {path}: {mismatches}")
    return manifest


def _aggregate_nanobrain(
    sources: list[tuple[int, str, Path, list[dict[str, str]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for seed, candidate_id, table, rows in sources:
        seen: set[tuple[str, str, str, str, str]] = set()
        for row in rows:
            key = (
                row["task"],
                row["metric"],
                row["stratum_type"],
                row["stratum_value"],
                _direction(row),
            )
            if key in seen:
                raise ValueError(f"duplicate nanoBrain metric key in {table}: {key}")
            seen.add(key)
            record = {
                "seed": seed,
                "candidate_id": candidate_id,
                "task": row["task"],
                "metric": row["metric"],
                "stratum_type": row["stratum_type"],
                "stratum_value": row["stratum_value"],
                "direction": key[-1],
                "value": row["value"],
                "eval_rows": int(row["eval_rows"]),
                "status": row["status"],
                "source_table": str(table),
            }
            per_seed.append(record)
            grouped.setdefault(key, []).append(record)

    expected_seeds = len(sources)
    aggregate: list[dict[str, Any]] = []
    for key, records in sorted(grouped.items()):
        if len(records) != expected_seeds:
            raise ValueError(f"incomplete seed coverage for {key}: {len(records)}/{expected_seeds}")
        eval_rows = {int(record["eval_rows"]) for record in records}
        if len(eval_rows) != 1:
            raise ValueError(f"eval-row mismatch across seeds for {key}: {sorted(eval_rows)}")
        numbers = [number for record in records if (number := _numeric(str(record["value"]))) is not None]
        if numbers and len(numbers) != expected_seeds:
            raise ValueError(f"partial numeric seed coverage for {key}: {len(numbers)}/{expected_seeds}")
        aggregate.append(
            {
                "task": key[0],
                "metric": key[1],
                "stratum_type": key[2],
                "stratum_value": key[3],
                "direction": key[4],
                "n_seeds": expected_seeds,
                "n_numeric": len(numbers),
                "mean": statistics.fmean(numbers) if numbers else "",
                "std": statistics.stdev(numbers) if len(numbers) > 1 else (0.0 if numbers else ""),
                "min": min(numbers) if numbers else "",
                "max": max(numbers) if numbers else "",
                "eval_rows": next(iter(eval_rows)),
                "seeds": ";".join(str(record["seed"]) for record in records),
                "values": ";".join(str(record["value"]) for record in records),
                "status": "complete" if numbers else "not_defined",
            }
        )
    return per_seed, aggregate


def _overall(
    rows: Iterable[dict[str, Any]], task: str, metric: str, *, method: str | None = None
) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("metric") == metric
        and row.get("stratum_type") == "overall"
        and row.get("stratum_value") == "overall"
        and (method is None or row.get("method") == method)
        and row.get("mean", row.get("value", "")) != ""
    ]
    if len(matches) > 1:
        regimes = {row.get("training_regime", "") for row in matches}
        if len(regimes) > 1:
            raise ValueError(f"ambiguous overall metric for {method}: {task}.{metric}")
    return matches[0] if matches else None


def _format_value(value: Any) -> str:
    if value in {None, ""}:
        return "N/A"
    number = float(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    return f"{number:.6f}"


def _make_long_table(
    aggregate: list[dict[str, Any]],
    baseline_sources: list[tuple[str, Path, list[dict[str, str]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in aggregate:
        rows.append(
            {
                "split_id": SPLIT_ID,
                "subset": SUBSET,
                "evaluation_split": "test",
                "source_group": "nanobrain_three_seed",
                "method": "nanobrain_fmt",
                "display_name": DISPLAY_NAMES["nanobrain_fmt"],
                "method_family": "ours",
                "training_regime": "joint_from_scratch_2000_steps",
                "task": row["task"],
                "metric": row["metric"],
                "direction": row["direction"],
                "stratum_type": row["stratum_type"],
                "stratum_value": row["stratum_value"],
                "mean": row["mean"],
                "std": row["std"],
                "n_runs": row["n_seeds"],
                "n_numeric": row["n_numeric"],
                "eval_rows": row["eval_rows"],
                "status": row["status"],
                "notes": "three frozen validation-selected checkpoints; independent seeds",
                "source_table": "aggregate_metrics.csv",
            }
        )
    for source_group, table, source_rows in baseline_sources:
        for row in source_rows:
            value = _numeric(row["value"])
            rows.append(
                {
                    "split_id": row["split_id"],
                    "subset": row["subset"],
                    "evaluation_split": "test",
                    "source_group": source_group,
                    "method": row["method"],
                    "display_name": DISPLAY_NAMES.get(row["method"], row["method"]),
                    "method_family": row.get("method_family", ""),
                    "training_regime": row["training_regime"],
                    "task": row["task"],
                    "metric": row["metric"],
                    "direction": _direction(row),
                    "stratum_type": row["stratum_type"],
                    "stratum_value": row["stratum_value"],
                    "mean": value if value is not None else "",
                    "std": "",
                    "n_runs": 1,
                    "n_numeric": 1 if value is not None else 0,
                    "eval_rows": int(row["eval_rows"]),
                    "status": row["status"] if value is not None else "not_defined",
                    "notes": row.get("notes", ""),
                    "source_table": str(table),
                }
            )
    return rows


def _report(
    aggregate: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    seeds: list[int],
    audit: dict[str, Any],
) -> str:
    lines = [
        "# nanoBrain 100k final test report",
        "",
        f"Protocol: `{audit['protocol_id']}`. The fixed split is 60,000 train / "
        "20,000 validation / 20,000 test rows. Test labels were opened once after "
        "the protocol and checkpoints were frozen.",
        "",
        "## nanoBrain FMT: three-seed result",
        "",
        f"The same architecture was trained from scratch for 2,000 optimizer steps with seeds "
        f"{', '.join(str(seed) for seed in seeds)}. Values below are mean +/- sample SD.",
        "",
        "| Task | Metric | Test mean | SD | Direction |",
        "| --- | --- | ---: | ---: | :---: |",
    ]
    labels = {
        ("annotation", "accuracy"): ("Annotation", "Accuracy"),
        ("annotation", "f1_macro"): ("Annotation", "Macro-F1"),
        ("integration", "integration_composite"): ("Integration", "Composite"),
        ("spatial_niche", "ari"): ("Spatial niche", "ARI"),
        ("spatial_niche", "nmi"): ("Spatial niche", "NMI"),
        ("spatial_completion", "heldout_rmse"): ("Spatial completion", "Held-out RMSE"),
        ("development", "population_mse"): ("Development", "Population MSE"),
        ("development", "mmd_rbf"): ("Development", "MMD-RBF"),
    }
    for task, metric in KEY_METRICS:
        row = _overall(aggregate, task, metric)
        if row is None or int(row["n_numeric"]) != len(seeds):
            raise ValueError(f"missing three-seed key test metric: {task}.{metric}")
        task_label, metric_label = labels[(task, metric)]
        lines.append(
            f"| {task_label} | {metric_label} | {_format_value(row['mean'])} | "
            f"{_format_value(row['std'])} | {row['direction']} |"
        )

    task_specs = [
        ("annotation", "Annotation", ("accuracy", "f1_macro"), ("Accuracy", "Macro-F1")),
        ("integration", "Integration", ("integration_composite",), ("Composite",)),
        ("spatial_niche", "Spatial niche", ("ari", "nmi"), ("ARI", "NMI")),
        ("spatial_completion", "Spatial completion", ("heldout_rmse",), ("Held-out RMSE",)),
        ("development", "Development", ("population_mse", "mmd_rbf"), ("Population MSE", "MMD-RBF")),
    ]
    lines.extend(["", "## Method comparison"])
    for task, title, metrics, metric_labels in task_specs:
        methods = sorted(
            {str(row["method"]) for row in long_rows if row["task"] == task},
            key=lambda method: (method != "nanobrain_fmt", DISPLAY_NAMES.get(method, method).lower()),
        )
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                "| Method | Training/evaluation mode | " + " | ".join(metric_labels) + " |",
                "| --- | --- | " + " | ".join("---:" for _ in metrics) + " |",
            ]
        )
        for method in methods:
            method_rows = [row for row in long_rows if row["task"] == task and row["method"] == method]
            regimes = sorted({str(row["training_regime"]) for row in method_rows})
            values = []
            for metric in metrics:
                hit = _overall(method_rows, task, metric, method=method)
                if hit is None:
                    values.append("N/A")
                elif method == "nanobrain_fmt":
                    values.append(f"{_format_value(hit['mean'])} +/- {_format_value(hit['std'])}")
                else:
                    values.append(_format_value(hit["mean"]))
            lines.append(
                f"| {DISPLAY_NAMES.get(method, method)} | {', '.join(regimes)} | "
                + " | ".join(values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- nanoBrain source rows: {audit['nanobrain_source_rows']} "
            f"({audit['nanobrain_numeric_rows']} numeric; {audit['nanobrain_undefined_rows']} mathematically undefined).",
            f"- Internal baseline rows: {audit['internal_rows']}; external baseline rows: {audit['external_rows']}.",
            f"- External adapters complete: {audit['external_methods_complete']}/13.",
            f"- Failed or error rows: {audit['error_rows']}.",
            f"- Final data manifest SHA-256: `{audit['data_manifest_sha256']}`.",
            "- Annotation evaluation uses 18,547 test rows whose labels occur in the compact 60k train split.",
            f"- Validation-selected checkpoint steps: {audit['selected_checkpoint_steps']}; "
            f"parameter count: {audit['parameter_count']:,}.",
            "- The frozen best-checkpoint weights and best quality are verified against each training receipt. "
            "Their legacy embedded step field remains 2000 and is recorded as a metadata mismatch.",
            "- Undefined metrics remain blank. They are not converted to zero and are counted separately in the receipt.",
            "",
            "`final_benchmark_long.csv` contains every metric and stratum for nanoBrain, internal controls, and external baselines.",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize(campaign: Path, *, data_manifest: Path) -> dict[str, Any]:
    campaign = campaign.resolve()
    data_manifest = data_manifest.resolve()
    parent = campaign.parent
    intent_path = campaign / "campaign_intent.json"
    receipt_path = campaign / "campaign_receipt.json"
    intent = _read_json(intent_path)
    if intent.get("status") not in {"running", "complete"}:
        raise ValueError(f"campaign is not finalizable: status={intent.get('status')}")
    if intent.get("fit_split") != "train" or intent.get("evaluation_split") != "test":
        raise ValueError("campaign split contract mismatch")
    intent_snapshot_path = campaign / "campaign_intent.pre_finalize.json"
    if not intent_snapshot_path.is_file():
        intent_snapshot = dict(intent)
        for key in ("completed_at", "final_receipt", "final_outputs"):
            intent_snapshot.pop(key, None)
        intent_snapshot["status"] = "running"
        _write_json(intent_snapshot_path, intent_snapshot)
    previous_receipt = (
        _read_json(receipt_path)
        if receipt_path.is_file()
        else None
    )

    freeze_path = parent / "protocol_freeze.json"
    open_path = parent / "test_label_open_receipt.json"
    filter_path = parent / "compact_train_label_filter_receipt.json"
    freeze = _read_json(freeze_path)
    opened = _read_json(open_path)
    filtered = _read_json(filter_path)
    protocol_id = str(intent.get("protocol_id", ""))
    for name, payload in (("freeze", freeze), ("test open", opened), ("label filter", filtered)):
        if payload.get("protocol_id") != protocol_id:
            raise ValueError(f"{name} receipt protocol mismatch")
    if freeze.get("test_opened") is not False:
        raise ValueError("protocol freeze does not attest sealed test")
    if opened.get("status") != "complete" or filtered.get("status") != "complete":
        raise ValueError("test label receipt chain is incomplete")

    expected_manifest_sha256 = str(filtered["after"]["manifest"]["sha256"])
    actual_manifest_sha256 = _sha256(data_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"data manifest hash mismatch: {actual_manifest_sha256} != {expected_manifest_sha256}"
        )

    seeds = [int(seed) for seed in intent.get("runs", {}).get("nanobrain", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(f"campaign must contain three unique nanoBrain seeds: {seeds}")

    nanobrain_sources: list[tuple[int, str, Path, list[dict[str, str]]]] = []
    source_records: list[dict[str, Any]] = []
    genomes: list[dict[str, Any]] = []
    train_protocols: list[dict[str, Any]] = []
    for seed in seeds:
        run = campaign / f"nanobrain-seed-{seed}"
        table = run / "benchmark_table.csv"
        manifest_path = run / "run_manifest.json"
        rows = _validate_metric_table(table)
        manifest = _validate_run_manifest(manifest_path, expected_manifest_sha256)
        methods = {row["method"] for row in rows}
        if len(methods) != 1 or not next(iter(methods)).startswith("nanobrain_candidate-"):
            raise ValueError(f"unexpected nanoBrain methods in {table}: {methods}")
        method = next(iter(methods))
        candidate_id = method.removeprefix("nanobrain_candidate-")
        checkpoint = Path(str(manifest.get("checkpoint", ""))).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint}")
        receipts = sorted(checkpoint.parent.glob(f"receipt-{candidate_id}-steps*.json"))
        if len(receipts) != 1:
            raise ValueError(f"expected one training receipt for {candidate_id}: {receipts}")
        training_receipt = _read_json(receipts[0])
        if (
            training_receipt.get("status") != "complete"
            or training_receipt.get("fit_split") != "train"
            or training_receipt.get("evaluation_split") != "validation"
            or training_receipt.get("test_opened") is not False
            or int(training_receipt.get("steps", 0)) != 2000
        ):
            raise ValueError(f"invalid frozen training receipt: {receipts[0]}")
        checkpoint_audit = _checkpoint_audit(checkpoint, training_receipt)
        genome = {key: value for key, value in training_receipt["genome"].items() if key != "seed"}
        train_protocol = {key: value for key, value in training_receipt["train"].items() if key != "seed"}
        genomes.append(genome)
        train_protocols.append(train_protocol)
        nanobrain_sources.append((seed, candidate_id, table, rows))
        source_records.append(
            {
                "kind": "nanobrain",
                "seed": seed,
                "candidate_id": candidate_id,
                "table": str(table),
                "table_sha256": _sha256(table),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "metric_rows": len(rows),
                "numeric_rows": sum(_numeric(row["value"]) is not None for row in rows),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "training_receipt": str(receipts[0]),
                "training_receipt_sha256": _sha256(receipts[0]),
                "checkpoint_audit": checkpoint_audit,
            }
        )
    if any(genome != genomes[0] for genome in genomes[1:]):
        raise ValueError("nanoBrain replicate genome mismatch after excluding seed")
    if any(protocol != train_protocols[0] for protocol in train_protocols[1:]):
        raise ValueError("nanoBrain training protocol mismatch after excluding seed")

    baseline_sources: list[tuple[str, Path, list[dict[str, str]]]] = []
    for source_group, directory in (
        ("internal_baseline", campaign / "internal-baselines"),
        ("external_baseline", campaign / "external-baselines"),
    ):
        table = directory / "benchmark_table.csv"
        manifest_path = directory / "run_manifest.json"
        rows = _validate_metric_table(table)
        _validate_run_manifest(manifest_path, expected_manifest_sha256)
        baseline_sources.append((source_group, table, rows))
        source_records.append(
            {
                "kind": source_group,
                "table": str(table),
                "table_sha256": _sha256(table),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "metric_rows": len(rows),
                "numeric_rows": sum(_numeric(row["value"]) is not None for row in rows),
            }
        )

    internal_rows = baseline_sources[0][2]
    external_rows = baseline_sources[1][2]
    external_manifest = _read_json(campaign / "external-baselines/run_manifest.json")
    requested = set(external_manifest.get("methods_requested", []))
    statuses = external_manifest.get("method_status", {})
    external_table_methods = {row["method"] for row in external_rows}
    if requested != EXPECTED_EXTERNAL_METHODS:
        raise ValueError(f"external method set mismatch: {sorted(requested)}")
    if set(statuses) != EXPECTED_EXTERNAL_METHODS or set(statuses.values()) != {"complete"}:
        raise ValueError(f"external method completion failed: {statuses}")
    if external_table_methods != EXPECTED_EXTERNAL_METHODS:
        raise ValueError(
            "external methods marked complete must all have metric rows: "
            f"{sorted(external_table_methods)}"
        )

    per_seed, aggregate = _aggregate_nanobrain(nanobrain_sources)
    for task, metric in KEY_METRICS:
        row = _overall(aggregate, task, metric)
        if row is None or int(row["n_numeric"]) != len(seeds):
            raise ValueError(f"missing key metric: {task}.{metric}")
    long_rows = _make_long_table(aggregate, baseline_sources)

    per_seed_path = campaign / "per_seed_metrics.csv"
    aggregate_path = campaign / "aggregate_metrics.csv"
    long_path = campaign / "final_benchmark_long.csv"
    report_path = campaign / "REPORT.md"
    _write_csv(per_seed_path, per_seed, list(per_seed[0]))
    _write_csv(aggregate_path, aggregate, list(aggregate[0]))
    _write_csv(long_path, long_rows, list(long_rows[0]))

    audit = {
        "protocol_id": protocol_id,
        "data_manifest_sha256": expected_manifest_sha256,
        "nanobrain_source_rows": len(per_seed),
        "nanobrain_numeric_rows": sum(_numeric(str(row["value"])) is not None for row in per_seed),
        "nanobrain_undefined_rows": sum(_numeric(str(row["value"])) is None for row in per_seed),
        "internal_rows": len(internal_rows),
        "external_rows": len(external_rows),
        "external_methods_complete": len(EXPECTED_EXTERNAL_METHODS),
        "error_rows": 0,
        "selected_checkpoint_steps": [
            record["checkpoint_audit"]["historical_best_step"]
            for record in source_records
            if record["kind"] == "nanobrain"
        ],
        "parameter_count": source_records[0]["checkpoint_audit"]["parameter_count"],
    }
    report_path.write_text(_report(aggregate, long_rows, seeds, audit), encoding="utf-8")

    generated_at = (
        str(previous_receipt["generated_at"])
        if previous_receipt is not None
        and previous_receipt.get("status") == "complete"
        and previous_receipt.get("protocol_id") == protocol_id
        and previous_receipt.get("generated_at")
        else datetime.now(timezone.utc).isoformat()
    )
    outputs = {
        path.name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (per_seed_path, aggregate_path, long_path, report_path)
    }
    receipt = {
        "format": "nanobrain_final_test_campaign_receipt_v1",
        "status": "complete",
        "protocol_id": protocol_id,
        "generated_at": generated_at,
        "split_id": SPLIT_ID,
        "subset": SUBSET,
        "fit_split": "train",
        "selection_split": "validation",
        "evaluation_split": "test",
        "test_opened": True,
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": expected_manifest_sha256,
        "independent_seeds": seeds,
        "architecture": genomes[0],
        "training_protocol": train_protocols[0],
        "checkpoint_metadata_audit": {
            "status": "weights_and_best_quality_verified",
            "legacy_step_metadata_mismatch": any(
                not record["checkpoint_audit"]["step_metadata_matches_history"]
                for record in source_records
                if record["kind"] == "nanobrain"
            ),
            "explanation": (
                "Frozen *-best.pt files contain validation-selected weights and the historical "
                "best quality, but their legacy embedded step field may retain the full 2000-step "
                "budget. The authoritative selected steps are in each training receipt."
            ),
        },
        "sources": source_records,
        "audit": audit,
        "row_counts": {
            "per_seed_metrics": len(per_seed),
            "aggregate_metrics": len(aggregate),
            "final_benchmark_long": len(long_rows),
        },
        "undefined_numeric_rows": [
            {
                "source_group": row["source_group"],
                "method": row["method"],
                "task": row["task"],
                "metric": row["metric"],
                "stratum_type": row["stratum_type"],
                "stratum_value": row["stratum_value"],
            }
            for row in long_rows
            if int(row["n_numeric"]) == 0
        ],
        "receipt_chain": {
            "protocol_freeze": {"path": str(freeze_path), "sha256": _sha256(freeze_path)},
            "test_label_open": {"path": str(open_path), "sha256": _sha256(open_path)},
            "compact_train_label_filter": {"path": str(filter_path), "sha256": _sha256(filter_path)},
            "campaign_intent_pre_finalize": {
                "path": str(intent_snapshot_path),
                "sha256": _sha256(intent_snapshot_path),
            },
        },
        "prior_attempts": intent.get("prior_attempts", []),
        "post_open_actions": [
            "materialized annotation labels through the predeclared test-label API",
            "excluded 19 labels from seven classes absent from compact train according to the predeclared unknown-label policy",
            "re-ran the full campaign after label filtering",
            "fixed BBKNN singleton-batch k=1 exact-neighbor dispatch without changing neighbors or metrics",
            "audited the legacy best-checkpoint step metadata against training receipts and final-step weights",
            "aggregated existing frozen predictions; no model, hyperparameter, task, or metric selection used test results",
        ],
        "outputs": outputs,
    }
    _write_json(receipt_path, receipt)
    receipt_sha256 = _sha256(receipt_path)

    completed_intent = dict(intent)
    completed_intent.update(
        {
            "status": "complete",
            "completed_at": generated_at,
            "final_receipt": {
                "path": "campaign_receipt.json",
                "sha256": receipt_sha256,
            },
            "final_outputs": {
                name: {"path": name, "sha256": record["sha256"]}
                for name, record in outputs.items()
            },
        }
    )
    _write_json(intent_path, completed_intent)
    return {
        "campaign": str(campaign),
        "status": "complete",
        "seeds": seeds,
        "external_methods_complete": len(EXPECTED_EXTERNAL_METHODS),
        "error_rows": 0,
        "row_counts": receipt["row_counts"],
        "data_manifest_sha256": expected_manifest_sha256,
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=ROOT / "assets/nanobrain_v3_100k/manifest.json",
    )
    args = parser.parse_args()
    print(json.dumps(finalize(args.campaign, data_manifest=args.data_manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
