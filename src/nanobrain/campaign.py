"""Generation-scoped benchmark records and baseline promotion gates.

The small benchmark is deliberately validation-first: every evolutionary
generation can attach the strongest baseline observed in the same run, while
the sealed test split remains outside the search loop.  The JSON ledger is
append-only at the generation level so a long BrainWM campaign can resume and
still explain which baseline was available when a candidate was promoted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PRIMARY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("annotation", "f1_macro", "max"),
    ("integration", "integration_composite", "max"),
    ("spatial_niche", "ari", "max"),
    ("development", "population_mse", "min"),
)


def _row_value(row: Mapping[str, Any]) -> float | None:
    try:
        value = float(row["value"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def primary_metric_record(metrics: Mapping[tuple[str, str], float]) -> dict[str, float]:
    """Convert tuple-keyed evaluator output to stable ``task.metric`` keys."""
    return {
        f"{task}.{metric}": float(metrics[(task, metric)])
        for task, metric, _ in PRIMARY_METRICS
        if (task, metric) in metrics
    }


def strongest_baselines(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the best non-nanoBrain method for every primary metric.

    Only complete overall rows are eligible.  This makes the result safe to
    compare with a candidate even when a baseline does not implement every
    task.
    """
    rows = list(rows)
    result: dict[str, dict[str, Any]] = {}
    for task, metric, direction in PRIMARY_METRICS:
        candidates: list[tuple[float, str]] = []
        for row in rows:
            method = str(row.get("method", ""))
            if method.startswith("nanobrain_candidate-"):
                continue
            if row.get("task") != task or row.get("metric") != metric:
                continue
            if row.get("status") not in {None, "", "complete"}:
                continue
            if row.get("stratum_type", "overall") != "overall":
                continue
            if row.get("stratum_value", "overall") != "overall":
                continue
            value = _row_value(row)
            if value is not None:
                candidates.append((value, method))
        if candidates:
            value, method = (max(candidates) if direction == "max" else min(candidates))
            result[f"{task}.{metric}"] = {
                "task": task,
                "metric": metric,
                "direction": direction,
                "method": method,
                "value": float(value),
            }
    return result


def baseline_regression_gate(
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    *,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Check a candidate against the strongest baseline available this generation."""
    tolerance = max(0.0, float(relative_tolerance))
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    for task, metric, direction in PRIMARY_METRICS:
        key = f"{task}.{metric}"
        baseline = baseline_metrics.get(key)
        candidate = candidate_metrics.get(key)
        if baseline is None:
            reasons.append(f"baseline_missing:{key}")
            continue
        if candidate is None or not math.isfinite(float(candidate)):
            reasons.append(f"candidate_metric_missing:{key}")
            continue
        reference = float(baseline["value"])
        allowance = tolerance * max(abs(reference), 1.0e-12)
        passed = (
            float(candidate) + allowance >= reference
            if direction == "max"
            else float(candidate) - allowance <= reference
        )
        checks.append({
            "metric": key,
            "direction": direction,
            "candidate": float(candidate),
            "baseline": reference,
            "baseline_method": baseline.get("method"),
            "allowance": allowance,
            "passed": bool(passed),
        })
        if not passed:
            reasons.append(f"baseline_regression:{key}")
    return {"passed": not reasons, "reasons": reasons, "checks": checks}


def generation_baseline_record(
    rows: Iterable[Mapping[str, Any]],
    *,
    generation: int,
    evaluation_split: str = "validation",
    split_id: str = "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
) -> dict[str, Any]:
    """Build the baseline snapshot that an evolution generation consumes."""
    rows = list(rows)
    best = strongest_baselines(rows)
    methods = sorted({str(row.get("method")) for row in rows if row.get("method")})
    candidate_methods = sorted({
        str(row.get("method"))
        for row in rows
        if str(row.get("method", "")).startswith("nanobrain_candidate-")
    })
    return {
        "format": "nanobrain_generation_baseline_v1",
        "generation": int(generation),
        "split_id": split_id,
        "fit_split": "train",
        "evaluation_split": str(evaluation_split),
        "test_used_for_selection": str(evaluation_split) == "test",
        "status": "complete" if best else "incomplete",
        "baseline_metrics": best,
        "methods": methods,
        "candidate_methods": candidate_methods,
    }


def append_generation_baseline(path: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append or replace one generation in a resumable campaign ledger."""
    target = Path(path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("format") != "nanobrain_generation_ledger_v1":
            raise ValueError(f"unsupported generation ledger: {target}")
        generations = list(payload.get("generations", []))
    else:
        payload = {"format": "nanobrain_generation_ledger_v1", "generations": []}
        generations = []
    generation = int(record["generation"])
    generations = [item for item in generations if int(item.get("generation", -1)) != generation]
    generations.append(dict(record))
    generations.sort(key=lambda item: int(item["generation"]))
    payload["generations"] = generations
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_generation_baseline(path: str | Path, generation: int) -> dict[str, Any]:
    """Read one generation's baseline snapshot from a ledger."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "nanobrain_generation_ledger_v1":
        raise ValueError("unsupported generation ledger format")
    for record in payload.get("generations", []):
        if int(record.get("generation", -1)) == int(generation):
            return record
    raise KeyError(f"generation baseline not found: {generation}")


__all__ = [
    "PRIMARY_METRICS",
    "append_generation_baseline",
    "baseline_regression_gate",
    "generation_baseline_record",
    "load_generation_baseline",
    "primary_metric_record",
    "strongest_baselines",
]
