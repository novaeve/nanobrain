"""Loader and fail-closed validator for the multimodal encoder screen.

The production benchmark registry intentionally remains unchanged until a
paired, independently sealed multimodal cohort exists.  This small contract
loader makes the proposed condition matrix machine-checkable without reading
large expression matrices or starting a training job.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONTRACT_FORMAT = "brainwm_multimodal_encoder_benchmark_contract_v1"
ALLOWED_STATUS = {"protocol_pending", "blocked_data", "engineering_only"}
REQUIRED_CONTEXT_MODALITIES = {"spatial_coordinates", "spatial_niche"}
EXTERNAL_MODALITIES = {"atac", "protein", "morphology"}
REQUIRED_TASKS = {"annotation", "integration", "development"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_path(value: str | Path, project_root: Path) -> Path:
    raw = Path(value)
    return raw if raw.is_absolute() else project_root / raw


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML contract without touching data arrays."""

    contract_path = Path(path)
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    project_root = contract_path.resolve().parents[2]
    validate_contract(payload, project_root=project_root, check_paths=False)
    return payload


def validate_contract(
    contract: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    check_paths: bool = False,
) -> None:
    """Validate schema, fairness invariants, and optionally small path checks.

    ``check_paths`` only checks path metadata (``is_file``/``is_dir``); it
    never opens H5AD, NPY, Parquet, or checkpoint payloads.
    """

    _require(isinstance(contract, Mapping), "multimodal contract must be a mapping")
    _require(contract.get("format") == CONTRACT_FORMAT, "unsupported multimodal contract format")
    _require(
        isinstance(contract.get("contract_id"), str) and contract["contract_id"],
        "contract_id is required",
    )
    _require(contract.get("project_native_only") is True, "contract must be project-native")
    _require(contract.get("status") in ALLOWED_STATUS, "invalid multimodal contract status")

    parent = contract.get("parent_benchmark")
    _require(isinstance(parent, Mapping), "parent_benchmark is required")
    for field in ("benchmark_id", "split_id", "evaluator_contract_id"):
        _require(bool(parent.get(field)), f"parent_benchmark.{field} is required")
    _require(parent.get("immutable") is True, "parent benchmark binding must be immutable")

    data = contract.get("data")
    _require(isinstance(data, Mapping), "data section is required")
    for field in ("artifact_dir", "feature_root", "row_manifest", "row_key", "phases"):
        _require(data.get(field) not in (None, "", []), f"data.{field} is required")
    _require(data.get("row_key") == ["source_id", "row_index"], "row identity must be source_id,row_index")
    phases = data.get("phases")
    _require(phases == ["train", "validation"], "only train and validation may be opened")
    _require(data.get("test_access") == "forbidden", "test access must remain forbidden")
    manifest_requirements = data.get("paired_manifest_requirements")
    _require(isinstance(manifest_requirements, Mapping), "paired manifest requirements are required")
    _require(manifest_requirements.get("one_row_per_key") is True,
             "paired modality manifest must be one-row-per-key")
    _require(manifest_requirements.get("feature_transform_fit_scope") == "train_rows_only",
             "paired modality transforms must fit on train rows only")
    _require(manifest_requirements.get("missingness_field") == "modality_presence",
             "paired modality manifest must declare modality_presence")
    _require(manifest_requirements.get("provenance_required") is True,
             "paired modality provenance is required")
    _require(
        set(manifest_requirements.get("key", []))
        == {"species", "source_id", "row_index", "cell_id", "modality"},
        "paired modality manifest key is incomplete",
    )
    _require(
        {"feature_path", "split", "paper_id"}
        <= set(manifest_requirements.get("required_columns", [])),
        "paired modality manifest provenance columns are incomplete",
    )
    rows = data.get("expected_rows")
    _require(
        isinstance(rows, Mapping)
        and all(isinstance(rows.get(phase), int) and rows[phase] > 0 for phase in phases),
        "expected_rows must contain positive train/validation counts",
    )
    _require(int(data.get("panel_size", 0)) == 477, "mouse panel size must remain 477")

    modalities = data.get("modalities")
    _require(isinstance(modalities, Mapping), "data.modalities is required")
    _require("expression" in modalities, "canonical expression modality is required")
    expression = modalities["expression"]
    _require(expression.get("kind") == "canonical_sparse_expression", "expression route must be canonical sparse input")
    names = set(modalities)
    _require(REQUIRED_CONTEXT_MODALITIES <= names, "spatial context modalities are missing")
    _require(EXTERNAL_MODALITIES <= names, "blocked external modality declarations are missing")
    for name, spec in modalities.items():
        _require(isinstance(spec, Mapping), f"modality {name!r} must be a mapping")
        _require(spec.get("name") == name, f"modality name mismatch for {name!r}")
        _require(spec.get("status") in {"available_for_engineering", "blocked_missing_paired_artifact"},
                 f"invalid readiness status for modality {name!r}")
        if name in EXTERNAL_MODALITIES:
            _require(spec.get("status") == "blocked_missing_paired_artifact",
                     f"external modality {name!r} cannot be marked available")
            _require(bool(spec.get("required_manifest")), f"blocked modality {name!r} needs a required_manifest")
        elif name == "expression":
            _require(spec.get("status") == "available_for_engineering",
                     "canonical expression modality is not marked available")
            _require(spec.get("kind") == "canonical_sparse_expression",
                     "canonical expression modality kind drifted")
        else:
            _require(spec.get("status") == "available_for_engineering",
                     f"current context modality {name!r} is not marked available")
            _require(isinstance(spec.get("dimension"), int) and spec["dimension"] > 0,
                     f"modality {name!r} dimension must be positive")
            _require(bool(spec.get("source")), f"modality {name!r} source is required")

    conditions = contract.get("conditions")
    _require(isinstance(conditions, list) and conditions, "conditions must be a non-empty list")
    condition_ids: set[str] = set()
    for condition in conditions:
        _require(isinstance(condition, Mapping), "each condition must be a mapping")
        condition_id = condition.get("id")
        _require(isinstance(condition_id, str) and condition_id, "condition id is required")
        _require(condition_id not in condition_ids, f"duplicate condition id: {condition_id}")
        condition_ids.add(condition_id)
        _require(condition.get("status") in {"engineering_only", "blocked_data"},
                 f"invalid status for condition {condition_id}")
        inputs = condition.get("inputs")
        _require(isinstance(inputs, list) and "expression" in inputs, f"{condition_id} must include expression")
        _require(len(inputs) == len(set(inputs)), f"{condition_id} repeats an input")
        _require(set(inputs) <= names, f"{condition_id} references an undeclared modality")
        expected_adapters = set(inputs) - {"expression"}
        _require(set(condition.get("trainable_adapters", [])) == expected_adapters,
                 f"{condition_id} trainable adapter set does not match inputs")
        has_blocked_input = bool(set(inputs) & EXTERNAL_MODALITIES)
        if condition.get("status") == "blocked_data":
            _require(has_blocked_input, f"blocked condition {condition_id} has no blocked modality")
        else:
            _require(not has_blocked_input, f"engineering condition {condition_id} uses blocked modality")
        router = condition.get("router")
        _require(router in {"disabled", "enabled"}, f"{condition_id} has invalid router mode")
        if inputs == ["expression"]:
            _require(router == "disabled", f"expression-only condition {condition_id} must disable router")
        else:
            _require(router == "enabled", f"modality condition {condition_id} must enable router")
        dropout = condition.get("dropout")
        if dropout is not None:
            _require(isinstance(dropout, Mapping), f"{condition_id}.dropout must be a mapping")
            _require(dropout.get("unit") == "cell", f"{condition_id} dropout must be cell-wise")
            _require(isinstance(dropout.get("seed"), int), f"{condition_id} dropout seed is required")
            probabilities = dropout.get("probability_by_modality")
            _require(isinstance(probabilities, Mapping), f"{condition_id} dropout probabilities are required")
            _require(set(probabilities) == set(inputs) - {"expression"},
                     f"{condition_id} dropout probabilities must cover named inputs")
            for modality, probability in probabilities.items():
                _require(0.0 <= float(probability) < 1.0,
                         f"{condition_id} dropout probability for {modality} must lie in [0,1)")
            _require(dropout.get("mask_expression") is False, f"{condition_id} may not drop canonical expression")
        permutation = condition.get("permutation")
        if permutation is not None:
            _require(isinstance(permutation, Mapping), f"{condition_id}.permutation must be a mapping")
            _require(permutation.get("unit") == "section", f"{condition_id} permutation must stay within section")
            _require(isinstance(permutation.get("seed"), int), f"{condition_id} permutation seed is required")
        _require(condition.get("split_id") == parent.get("split_id"), f"{condition_id} split drifted")
        _require(condition.get("evaluator_contract_id") == parent.get("evaluator_contract_id"),
                 f"{condition_id} evaluator binding drifted")

    tasks = contract.get("task_lanes")
    _require(isinstance(tasks, Mapping) and REQUIRED_TASKS <= set(tasks), "annotation/integration/development task lanes are required")
    for task_id in REQUIRED_TASKS:
        lane = tasks[task_id]
        _require(isinstance(lane, Mapping), f"task lane {task_id} must be a mapping")
        _require(lane.get("primary_metric"), f"task lane {task_id} needs a primary metric")
        _require(lane.get("direction") in {"maximize", "minimize"}, f"task lane {task_id} has invalid direction")
        _require(lane.get("cross_task_aggregation") == "forbidden", f"task lane {task_id} may not be cross-aggregated")

    training = contract.get("adapter_training")
    _require(isinstance(training, Mapping), "adapter_training is required")
    _require(training.get("status") == "protocol_only_no_runner",
             "adapter training must remain protocol-only until a runner exists")
    _require(training.get("base_checkpoint"), "adapter_training.base_checkpoint is required")
    _require(training.get("base_checkpoint_role") == "exploratory_non_promotable_human_mouse_stage0",
             "base checkpoint role must remain exploratory/non-promotable")
    _require(isinstance(training.get("requires_before_execution"), list)
             and training["requires_before_execution"],
             "adapter_training.requires_before_execution is required")
    _require(training.get("trainable_only") == ["modality_adapters"], "only modality adapters may be trainable")
    _require(training.get("objective") == "stage0_self_supervised_reconstruction", "adapter objective must be self-supervised reconstruction")
    seeds = training.get("seeds")
    _require(isinstance(seeds, list) and len(seeds) == 3 and all(isinstance(seed, int) for seed in seeds),
             "adapter training must declare three integer seeds")

    promotion = contract.get("promotion")
    _require(isinstance(promotion, Mapping), "promotion policy is required")
    _require(promotion.get("production_registry_append") is False, "pending screen may not append production results")
    _require(promotion.get("requires_independent_paired_cohort") is True,
             "promotion must require an independent paired cohort")
    _require(promotion.get("primary_metric_only") is True, "promotion must use the declared primary metric")

    if check_paths:
        root = Path(project_root or ".").resolve()
        for key in ("artifact_dir", "feature_root"):
            _require(_as_path(str(data[key]), root).is_dir(), f"missing data path: {data[key]}")
        manifest = _as_path(str(data["row_manifest"]), root)
        _require(manifest.is_file(), f"missing row manifest: {manifest}")
        checkpoint = _as_path(str(training["base_checkpoint"]), root)
        _require(checkpoint.is_file(), f"missing base checkpoint: {checkpoint}")
        for phase in phases:
            source = _as_path(str(modalities["spatial_niche"]["source"][phase]), root)
            _require(source.is_file(), f"missing spatial niche source: {source}")


def available_conditions(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Return conditions that can run with currently declared data."""

    validate_contract(contract)
    blocked = set(EXTERNAL_MODALITIES)
    result = []
    for condition in contract["conditions"]:
        if not (set(condition["inputs"]) & blocked):
            result.append(str(condition["id"]))
    return tuple(result)
