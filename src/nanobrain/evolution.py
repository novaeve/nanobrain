"""Small, deterministic promotion helpers shared by nanoBrain and BrainWM."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from typing import Any

from .campaign import PRIMARY_METRICS, baseline_regression_gate
from .genome import CandidateGenome

EVOLUTION_TASKS: tuple[str, ...] = tuple(task for task, _, _ in PRIMARY_METRICS)


def architecture_id(genome: Mapping[str, Any] | CandidateGenome) -> str:
    """Identify an architecture independently of its random training seed."""
    payload = genome.to_dict() if isinstance(genome, CandidateGenome) else CandidateGenome.from_dict(dict(genome)).to_dict()
    payload.pop("seed", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def joint_score(metrics: Mapping[str, float], tasks: list[str] | tuple[str, ...] | None = None) -> float:
    """Map the four primary metrics to bounded utilities where higher is better."""
    selected = set(tasks) if tasks is not None else None
    values = []
    for task, metric, direction in PRIMARY_METRICS:
        if selected is not None and task not in selected:
            continue
        key = f"{task}.{metric}"
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite primary metric: {key}")
        if direction == "min":
            if value < 0:
                raise ValueError(f"negative error: {key}")
            value = 1.0 / (1.0 + value)
        elif metric == "ari":
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"ARI outside [-1, 1]: {value}")
            value = (value + 1.0) / 2.0
        elif not 0.0 <= value <= 1.0:
            raise ValueError(f"metric outside [0, 1]: {key}")
        values.append(value)
    if not values:
        raise ValueError("no primary metrics selected")
    return statistics.mean(values)


def improvement_gate(candidate: Mapping[str, float], reference: Mapping[str, float], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Require a joint gain while bounding per-task regressions."""
    reasons: list[str] = []
    checks: list[dict[str, Any]] = []
    tasks = policy.get("tasks")
    try:
        delta = joint_score(candidate, tasks) - joint_score(reference, tasks)
        if delta < float(policy.get("minimum_joint_gain", 0.0)):
            reasons.append("joint_gain_below_threshold")
        selected = set(tasks) if tasks is not None else None
        for task, metric, direction in PRIMARY_METRICS:
            if selected is not None and task not in selected:
                continue
            key = f"{task}.{metric}"
            observed, base = float(candidate[key]), float(reference[key])
            allowance = (float(policy.get("max_score_drop", 0.0)) if direction == "max"
                         else float(policy.get("max_error_relative_increase", 0.0)) * max(abs(base), 1e-12))
            deterioration = base - observed if direction == "max" else observed - base
            passed = deterioration <= allowance + 1e-12
            checks.append({"metric": key, "candidate": observed, "reference": base,
                          "deterioration": deterioration, "allowance": allowance, "passed": passed})
            if not passed:
                reasons.append(f"task_regression:{key}")
    except (KeyError, TypeError, ValueError) as exc:
        delta = None
        reasons.append(f"invalid_metrics:{exc}")
    return {"passed": not reasons, "joint_gain": delta, "checks": checks, "reasons": reasons}


__all__ = [
    "EVOLUTION_TASKS",
    "PRIMARY_METRICS",
    "architecture_id",
    "baseline_regression_gate",
    "improvement_gate",
    "joint_score",
]
