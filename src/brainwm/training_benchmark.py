"""Contract and run-manifest helpers for the BrainWM training benchmark.

This is the training layer above the global task split.  It makes a run
comparable by fixing the data contract, rung budget, seed policy, validation
selection rule, and resource accounting.  It intentionally does not launch
training or read expression/test payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FORMAT = "brainwm_training_benchmark_v1"
RUN_FORMAT = "brainwm_training_benchmark_run_v1"
Rungs = ("smoke", "short", "standard", "full")
TASKS = ("integration", "annotation", "development", "spatial_niche")
METRIC_DIRECTIONS = {"maximize", "minimize"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{name} must be a positive integer")
    return value


def validate_training_benchmark(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable training benchmark contract."""
    _require(isinstance(spec, Mapping), "training benchmark must be a mapping")
    _require(spec.get("format") == FORMAT, "unsupported training benchmark format")
    _require(bool(spec.get("benchmark_id")), "benchmark_id is required")
    _require(spec.get("status") == "frozen", "training benchmark must be frozen")
    data = spec.get("data")
    _require(isinstance(data, Mapping), "data contract is required")
    _require(data.get("split_id") == "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
             "training benchmark must bind the active V3 global human split")
    _require(data.get("fractions") == {"train": 0.6, "validation": 0.2, "test": 0.2},
             "training benchmark must bind the 6:2:2 split")
    _require(data.get("test_state") == "sealed", "test must remain sealed")

    budgets = spec.get("budgets")
    _require(isinstance(budgets, Mapping) and tuple(budgets) == Rungs,
             "budgets must define smoke, short, standard in order")
    previous = 0
    for rung in Rungs:
        row = budgets[rung]
        _require(isinstance(row, Mapping), f"budget {rung} must be a mapping")
        steps = _positive_int(row.get("optimizer_steps"), f"{rung}.optimizer_steps")
        rows = _positive_int(row.get("max_train_rows"), f"{rung}.max_train_rows")
        _positive_int(row.get("validation_rows"), f"{rung}.validation_rows")
        _positive_int(row.get("validate_every"), f"{rung}.validate_every")
        _require(steps > previous, "rung optimizer steps must increase")
        previous = steps
        _require(row.get("precision") in {"bf16", "fp32"}, f"invalid precision in {rung}")
        _require(rows >= row["validation_rows"], f"{rung} validation rows exceed train rows")
        if rung == "full":
            _positive_int(row.get("minimum_train_passes"), "full.minimum_train_passes")
            _positive_int(row.get("maximum_train_passes"), "full.maximum_train_passes")
            _positive_int(row.get("validation_patience"), "full.validation_patience")
            _require(
                row["maximum_train_passes"] >= row["minimum_train_passes"],
                "full maximum_train_passes must cover minimum_train_passes",
            )

    policy = spec.get("policy")
    _require(isinstance(policy, Mapping), "policy is required")
    _require(policy.get("seed_count") == 3, "learned routes require exactly three seeds")
    _require(policy.get("seed_set") == [20260901, 20260902, 20260903],
             "seed set drifted")
    _require(policy.get("checkpoint_selection") == "validation_primary_metric_only",
             "checkpoint selection must use validation primary metric")
    _require(policy.get("effective_batch_size") == 4096,
             "effective batch size must be fixed at 4096")
    _require(policy.get("final_model_requires_full_rung") is True,
             "final models must use the full rung")
    _require(policy.get("full_stop_rule") == "min_passes_then_validation_plateau",
             "full stop rule drifted")
    _require(policy.get("cross_task_reward_aggregation") is False,
             "cross-task reward aggregation is forbidden")
    _require(policy.get("test_access") == "after_prediction_freeze_once",
             "test access policy drifted")

    tasks = spec.get("tasks")
    _require(isinstance(tasks, Mapping) and tuple(tasks) == TASKS,
             "training benchmark must define the four active task lanes")
    for task in TASKS:
        lane = tasks[task]
        _require(isinstance(lane, Mapping), f"task lane {task} must be a mapping")
        _require(bool(lane.get("evaluator_id")), f"{task}.evaluator_id is required")
        _require(bool(lane.get("primary_metric")), f"{task}.primary_metric is required")
        _require(lane.get("direction") in METRIC_DIRECTIONS, f"{task}.direction is invalid")
        _require(lane.get("split_id") == data["split_id"], f"{task} split binding drifted")
        _require(isinstance(lane.get("required_controls"), list)
                 and len(lane["required_controls"]) >= 2,
                 f"{task} needs matched controls")
    _require(spec.get("blocked_tasks") == {"perturbation": "missing_intervention_outcomes"},
             "blocked perturbation lane must be explicit")
    return {"valid": True, "benchmark_id": spec["benchmark_id"],
            "rungs": list(Rungs), "tasks": list(TASKS), "test_opened": False}


def canonical_json(value: Any) -> str:
    """Return stable JSON for config and manifest hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_run_manifest(
    spec: Mapping[str, Any], *, task: str, method_id: str, seed: int, rung: str,
    config: Mapping[str, Any], checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Create a run receipt without touching data or a checkpoint."""
    validate_training_benchmark(spec)
    _require(task in TASKS, f"unknown task: {task}")
    _require(rung in Rungs, f"unknown rung: {rung}")
    _require(seed in spec["policy"]["seed_set"], "seed is outside frozen seed set")
    _require(method_id.strip() != "", "method_id is required")
    budget = spec["budgets"][rung]
    now = datetime.now(timezone.utc).isoformat()
    return {
        "format": RUN_FORMAT,
        "status": "prepared",
        "benchmark_id": spec["benchmark_id"],
        "task": task,
        "method_id": method_id,
        "seed": seed,
        "rung": rung,
        "budget": dict(budget),
        "data_contract": dict(spec["data"]),
        "evaluator_id": spec["tasks"][task]["evaluator_id"],
        "primary_metric": spec["tasks"][task]["primary_metric"],
        "metric_direction": spec["tasks"][task]["direction"],
        "config_sha256": sha256_json(config),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "test_opened": False,
        "created_at": now,
    }


__all__ = ["FORMAT", "RUN_FORMAT", "Rungs", "TASKS", "make_run_manifest",
           "validate_training_benchmark", "canonical_json", "sha256_json"]
