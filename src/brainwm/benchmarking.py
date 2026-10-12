"""Strict benchmark registry and Experimental Results Table rendering."""

from __future__ import annotations

import csv
import fcntl
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from html import escape
from pathlib import Path
from statistics import median
from typing import Any

import yaml

REGISTRY_FORMAT = "brainwm_benchmark_registry_v1"
RESULT_FORMAT = "brainwm_benchmark_result_v1"
CONTRACT_ID_FIELDS = ("data_contract_id", "evaluator_contract_id")
LEGACY_CONTRACT_STATUS = "legacy_unverified"
VERIFIED_CONTRACT_STATUS = "verified"
ANN_LABEL_BLIND_V2_BENCHMARK_ID = "ANN-S64-LABEL-BLIND-DH-V2"
ANN_LABEL_BLIND_V2_SCORE_FORMAT = "brainwm_ann_s64_label_blind_score_v2"
ANN_LABEL_BLIND_V2_AGGREGATE_FORMAT = (
    "brainwm_ann_s64_label_blind_score_aggregate_v2"
)
MOUSE_STAGE2_SELECTION_FORMAT = "brainwm_mouse_stage2_selection_v1"
MOUSE_SPATIAL_ANNOTATION_VALIDATION_FORMAT = (
    "brainwm_mouse_spatial_annotation_validation_results_v1"
)
MOUSE_SPATIAL_ANNOTATION_GATE_FORMAT = (
    "brainwm_mouse_spatial_annotation_validation_gate_v1"
)
MOUSE_SPATIAL_RAW_SU07_TEST_FORMAT = (
    "brainwm_mouse_spatial_annotation_test_result_v1"
)
MOUSE_SPATIAL_RAW_SU07_GATE_FORMAT = (
    "brainwm_mouse_spatial_raw_su07_confirmation_gate_v1"
)
TRAIN_INTERNAL_ARCHITECTURE_SCOPE = "train_internal_architecture_gate_only"
TASK_IDS = ("integration", "annotation", "development", "perturbation")
AUXILIARY_TASK_ID = "auxiliary"
METHOD_CLASSES = {"baseline", "foundation", "candidate", "ablation", "hparam"}
PROMOTION_METHOD_CLASSES = {"foundation", "candidate"}
VARIATION_AXES = {"reference", "data", "architecture", "algorithm"}
MUTABLE_VARIATION_AXES = {"data", "architecture", "algorithm"}
RANKABLE_STATUSES = {"active_frozen", "completed_frozen"}
REWARD_TRANSFORMS = {"identity", "negate"}
_PRODUCTION_REGISTRY_ID = "brainwm_experimental_table_v1"
_PRODUCTION_REGISTRY_ROLES = frozenset(
    {"production", "production_frozen", "frozen", "frozen_production"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_production_registry(registry: dict[str, Any], project_root: Path) -> bool:
    """Return whether append-time freeze policy must be enforced.

    A caller can validate a copied registry and temporary evidence tree in
    tests or in an isolated staging run.  The completed-frozen append guard is
    therefore scoped to an explicit production role, or to the canonical
    project registry rooted at this package.  This keeps the guard meaningful
    for the real table without making ordinary fixture registries unusable.
    """

    role = registry.get("registry_role")
    if isinstance(role, str) and role.lower() in _PRODUCTION_REGISTRY_ROLES:
        return True
    if registry.get("production_registry") is True or registry.get("production") is True:
        return True
    canonical_root = Path(__file__).resolve().parents[1]
    return (
        registry.get("registry_id") == _PRODUCTION_REGISTRY_ID
        and project_root.resolve() == canonical_root
    )


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    validate_registry(payload)
    manifest_ref = payload.get("contract_manifest_path")
    if manifest_ref:
        # Resolve project-relative references as well as references relative to
        # the config directory.  The former keeps the path stable when a
        # caller invokes the loader from another working directory.
        candidates = [
            registry_path.parent / str(manifest_ref),
            registry_path.parent.parent / str(manifest_ref),
        ]
        manifest_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        _require(manifest_path is not None, f"missing contract manifest: {manifest_ref}")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        _validate_contract_manifest(payload, manifest)
    return payload


def load_results(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result JSON on line {line_number}: {exc}") from exc
        _require(isinstance(row, dict), f"result line {line_number} must be an object")
        # Results written before the immutable data/evaluator contract fields
        # were introduced remain readable.  Mark them in-memory so they can
        # never be mistaken for newly verified evidence; the append path below
        # still requires explicit contract IDs for every new row.
        missing_contracts = [field for field in CONTRACT_ID_FIELDS if not row.get(field)]
        if len(missing_contracts) == len(CONTRACT_ID_FIELDS):
            diagnostics = row.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
                row["diagnostics"] = diagnostics
            diagnostics.setdefault("contract_status", LEGACY_CONTRACT_STATUS)
            diagnostics.setdefault("legacy_contract_fields_missing", missing_contracts)
            row.setdefault("contract_status", LEGACY_CONTRACT_STATUS)
        elif not missing_contracts:
            diagnostics = row.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
                row["diagnostics"] = diagnostics
            _require(
                row.get("contract_status") in (None, VERIFIED_CONTRACT_STATUS),
                f"result line {line_number} has conflicting contract_status",
            )
            _require(
                diagnostics.get("contract_status") in (None, VERIFIED_CONTRACT_STATUS),
                f"result line {line_number} has conflicting diagnostics.contract_status",
            )
            row["contract_status"] = VERIFIED_CONTRACT_STATUS
            diagnostics["contract_status"] = VERIFIED_CONTRACT_STATUS
        rows.append(row)
    return rows


def append_result(
    path: str | Path,
    registry: dict[str, Any],
    row: dict[str, Any],
    *,
    project_root: str | Path | None = None,
    allow_completed_frozen: bool = False,
) -> dict[str, Any]:
    """Validate and append one result while holding an exclusive file lock.

    The file is an append-only JSONL store.  Re-reading under the lock closes
    the small race where two evaluators finish at the same time and both try
    to register the same result id.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    root = Path(project_root) if project_root is not None else None
    if root is None:
        raise ValueError(
            "project_root is required when appending a verified result receipt"
        )
    if not isinstance(allow_completed_frozen, bool):
        raise TypeError("allow_completed_frozen must be a boolean")
    benchmark_id = row.get("benchmark_id")
    benchmark = benchmark_index(registry).get(benchmark_id)
    completed_override = (
        allow_completed_frozen
        and benchmark is not None
        and benchmark.get("status") == "completed_frozen"
    )
    if (
        benchmark is not None
        and benchmark.get("status") != "active_frozen"
        and _is_production_registry(registry, root)
        and not completed_override
    ):
        raise ValueError(
            f"new results can only be appended to active_frozen benchmarks: {benchmark_id}"
        )
    # A result appended after the contract migration must carry both immutable
    # bindings.  Historical rows are accepted by ``load_results`` and marked
    # legacy, but a new row cannot silently enter that compatibility path.
    if not all(row.get(field) for field in CONTRACT_ID_FIELDS):
        raise ValueError(
            "new benchmark results must bind data_contract_id and evaluator_contract_id"
        )
    with target.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = load_results(target)
            if any(item.get("result_id") == row.get("result_id") for item in existing):
                raise ValueError(f"duplicate result_id: {row.get('result_id')}")
            checked = validate_results(registry, [*existing, row], project_root=root)
            persisted = checked[-1]
            handle.seek(0, 2)
            if existing and handle.tell() > 0:
                # Text streams do not support non-zero end-relative seeks;
                # inspect one byte through a separate descriptor instead.
                with target.open("rb") as tail_handle:
                    tail_handle.seek(-1, 2)
                    trailing = tail_handle.read(1)
                if trailing != b"\n":
                    handle.write("\n")
            handle.write(json.dumps(persisted, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return deepcopy(persisted)


def downstream_benchmark_index(
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return only the four declared downstream-task benchmarks."""

    return {
        benchmark["benchmark_id"]: benchmark
        for task_id in TASK_IDS
        for benchmark in registry["tasks"][task_id].get("benchmarks", [])
    }


def auxiliary_benchmark_index(
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return architecture/evaluator benchmarks outside downstream tasks."""

    return {
        benchmark["benchmark_id"]: benchmark
        for benchmark in registry.get("auxiliary_benchmarks", [])
    }


def benchmark_index(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    task_benchmarks = downstream_benchmark_index(registry)
    auxiliary = auxiliary_benchmark_index(registry)
    overlap = set(task_benchmarks) & set(auxiliary)
    _require(not overlap, f"duplicate benchmark_id: {sorted(overlap)}")
    return {**task_benchmarks, **auxiliary}


def validate_registry(registry: dict[str, Any]) -> None:
    _require(isinstance(registry, dict), "registry must be an object")
    _require(registry.get("format") == REGISTRY_FORMAT, "unsupported registry format")
    _require(registry.get("task_order") == list(TASK_IDS), "task_order must freeze all four tasks")
    tasks = registry.get("tasks")
    _require(isinstance(tasks, dict) and set(tasks) == set(TASK_IDS), "registry must contain exactly four tasks")
    auxiliary_benchmarks = registry.get("auxiliary_benchmarks", [])
    _require(
        isinstance(auxiliary_benchmarks, list),
        "auxiliary_benchmarks must be a list",
    )
    policy = registry.get("ranking_policy", {})
    _require(policy.get("scope") == "within_benchmark_only", "ranking must be within one benchmark")
    _require(policy.get("clipping") == "diagnostic_only", "clipping must remain diagnostic only")
    _require(
        policy.get("cross_benchmark_reward_comparison") == "forbidden",
        "cross-benchmark reward comparison must remain forbidden",
    )

    seen_benchmarks: set[str] = set()

    def validate_benchmark(
        benchmark: dict[str, Any], *, owner_id: str, auxiliary: bool
    ) -> None:
        benchmark_id = benchmark.get("benchmark_id")
        _require(
            isinstance(benchmark_id, str) and benchmark_id,
            f"{owner_id} has an invalid benchmark_id",
        )
        _require(
            benchmark_id not in seen_benchmarks,
            f"duplicate benchmark_id: {benchmark_id}",
        )
        seen_benchmarks.add(benchmark_id)
        _require(
            benchmark.get("task_id") == owner_id,
            f"benchmark {benchmark_id} task mismatch",
        )
        if auxiliary:
            _require(
                isinstance(benchmark.get("auxiliary_group"), str)
                and bool(benchmark["auxiliary_group"]),
                f"auxiliary benchmark {benchmark_id} needs an auxiliary_group",
            )
            _require(
                isinstance(benchmark.get("benchmark_kind"), str)
                and bool(benchmark["benchmark_kind"]),
                f"auxiliary benchmark {benchmark_id} needs a benchmark_kind",
            )
        _require(
            "training_stage" in benchmark,
            f"benchmark {benchmark_id} must bind training_stage separately",
        )
        _require(
            benchmark.get("status")
            in RANKABLE_STATUSES | {"protocol_pending", "blocked_data"},
            f"invalid benchmark status: {benchmark_id}",
        )
        baselines = benchmark.get("baselines", [])
        _require(
            isinstance(baselines, list),
            f"benchmark {benchmark_id} baselines must be a list",
        )
        baseline_ids = [row.get("method_id") for row in baselines]
        _require(
            all(isinstance(value, str) and value for value in baseline_ids),
            f"benchmark {benchmark_id} has an invalid baseline method_id",
        )
        _require(
            len(baseline_ids) == len(set(baseline_ids)),
            f"benchmark {benchmark_id} repeats a baseline",
        )
        for baseline in baselines:
            if "training_stage" in baseline:
                _require(
                    isinstance(baseline["training_stage"], str)
                    and bool(baseline["training_stage"]),
                    f"benchmark {benchmark_id} baseline training_stage is invalid",
                )
        if benchmark["status"] in RANKABLE_STATUSES:
            _validate_rankable_contract(benchmark)

    for task_id in TASK_IDS:
        task = tasks[task_id]
        _require(task.get("task_id") == task_id, f"task id mismatch for {task_id}")
        _require(isinstance(task.get("status"), str), f"task {task_id} needs a status")
        benchmarks = task.get("benchmarks")
        _require(isinstance(benchmarks, list) and benchmarks, f"task {task_id} needs a benchmark contract")
        for benchmark in benchmarks:
            validate_benchmark(benchmark, owner_id=task_id, auxiliary=False)

    for benchmark in auxiliary_benchmarks:
        validate_benchmark(
            benchmark, owner_id=AUXILIARY_TASK_ID, auxiliary=True
        )


def _validate_rankable_contract(benchmark: dict[str, Any]) -> None:
    benchmark_id = benchmark["benchmark_id"]
    split = benchmark.get("split")
    evaluator = benchmark.get("evaluator")
    metric = benchmark.get("primary_metric")
    _require(isinstance(split, dict) and split.get("fixed") is True and split.get("split_id"),
             f"rankable benchmark {benchmark_id} needs a fixed split")
    _require(isinstance(evaluator, dict) and evaluator.get("fixed") is True and evaluator.get("evaluator_id"),
             f"rankable benchmark {benchmark_id} needs a fixed evaluator")
    _require(isinstance(metric, dict) and metric.get("name"),
             f"rankable benchmark {benchmark_id} needs a primary metric")
    _require(metric.get("direction") in {"maximize", "minimize"},
             f"benchmark {benchmark_id} metric direction is invalid")
    _require(metric.get("reward_transform") in REWARD_TRANSFORMS,
             f"benchmark {benchmark_id} reward transform is invalid")
    expected = "maximize" if metric["reward_transform"] == "identity" else "minimize"
    _require(metric["direction"] == expected,
             f"benchmark {benchmark_id} metric direction and reward transform disagree")
    _validate_immutable_contracts(benchmark)
    _validate_io_contracts(benchmark)
    _validate_variation_policy(benchmark)
    if benchmark.get("claim_scope") == TRAIN_INTERNAL_ARCHITECTURE_SCOPE:
        _require(
            benchmark.get("benchmark_kind") == "architecture_gate",
            f"train-internal benchmark {benchmark_id} must be an architecture gate",
        )
        _require(
            benchmark.get("promotion_eligible") is False,
            f"train-internal benchmark {benchmark_id} must be non-promotion",
        )
        _require(
            benchmark.get("leaderboard_role") == "internal_diagnostic_only",
            f"train-internal benchmark {benchmark_id} must be diagnostic-only",
        )
        _require(
            benchmark.get("heldout_generalization_evidence") is False,
            f"train-internal benchmark {benchmark_id} cannot claim held-out evidence",
        )
        _require(
            split.get("phase") == "train_internal"
            and split.get("validation_opened") is False
            and split.get("test_opened") is False,
            f"train-internal benchmark {benchmark_id} opened held-out data",
        )
        _require(
            evaluator.get("claim_scope") == TRAIN_INTERNAL_ARCHITECTURE_SCOPE
            and evaluator.get("promotion_eligible") is False
            and evaluator.get("heldout_generalization_evidence") is False,
            f"train-internal benchmark {benchmark_id} evaluator scope is not frozen",
        )


def _contract_id(benchmark: dict[str, Any], kind: str) -> str | None:
    """Resolve one contract ID while accepting the explicit nested spelling."""
    direct = benchmark.get(f"{kind}_id")
    nested = benchmark.get(kind)
    nested_id = nested.get("contract_id") if isinstance(nested, dict) else None
    if direct is not None and nested_id is not None:
        _require(direct == nested_id, f"{benchmark.get('benchmark_id')} {kind} ID aliases disagree")
    value = direct or nested_id
    return str(value) if value else None


def benchmark_contract_ids(benchmark: dict[str, Any]) -> dict[str, str]:
    """Return the immutable IDs that every newly registered result must bind."""
    data_id = _contract_id(benchmark, "data_contract")
    evaluator_id = _contract_id(benchmark, "evaluator_contract")
    _require(data_id is not None and evaluator_id is not None, "benchmark contract IDs are not defined")
    return {
        "data_contract_id": data_id,
        "evaluator_contract_id": evaluator_id,
    }


def bind_result_contract(row: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    """Attach benchmark contract IDs to a result row without changing its axes."""
    bound = dict(row)
    bound.update(benchmark_contract_ids(benchmark))
    return bound


def _validate_immutable_contracts(benchmark: dict[str, Any]) -> None:
    """Validate the frozen data/evaluator boundary for a rankable benchmark.

    The IDs are versioned names under git; the structured payload makes the
    cohort, panel, assay, normalization fit scope and query policy explicit.
    This is deliberately a small schema so changing a model or loss does not
    require changing the benchmark split.
    """
    benchmark_id = benchmark["benchmark_id"]
    split = benchmark["split"]
    metric = benchmark["primary_metric"]
    for kind, required in (
        (
            "data_contract",
            ("cohort_id", "split_id", "panel_id", "assay", "normalization", "query_policy"),
        ),
        (
            "evaluator_contract",
            ("split_id", "evaluator_id", "primary_metric", "query_policy"),
        ),
    ):
        contract_id = _contract_id(benchmark, kind)
        _require(contract_id is not None and contract_id, f"rankable benchmark {benchmark_id} needs {kind}_id")
        contract = benchmark.get(kind)
        _require(isinstance(contract, dict), f"benchmark {benchmark_id} needs {kind} details")
        _require(contract.get("frozen") is True, f"benchmark {benchmark_id} {kind} must be frozen")
        for field in required:
            value = contract.get(field)
            _require(value not in (None, "", []), f"benchmark {benchmark_id} {kind}.{field} is required")
        _require(
            contract.get("split_id") == split.get("split_id"),
            f"benchmark {benchmark_id} {kind}.split_id must match split.split_id",
        )
        if kind == "data_contract":
            normalization = contract.get("normalization")
            _require(
                isinstance(normalization, dict)
                and bool(normalization.get("fit_scope")),
                f"benchmark {benchmark_id} data_contract.normalization.fit_scope is required",
            )
        else:
            _require(
                contract.get("evaluator_id") == benchmark["evaluator"].get("evaluator_id"),
                f"benchmark {benchmark_id} evaluator_contract.evaluator_id must match evaluator.evaluator_id",
            )
            _require(
                contract.get("primary_metric") == metric.get("name"),
                f"benchmark {benchmark_id} evaluator primary_metric must match benchmark metric",
            )


def _validate_io_contracts(benchmark: dict[str, Any]) -> None:
    """Require an explicit model-facing boundary for every rankable task."""

    benchmark_id = benchmark["benchmark_id"]
    input_contract = benchmark.get("input_contract")
    output_contract = benchmark.get("output_contract")
    _require(isinstance(input_contract, dict), f"benchmark {benchmark_id} needs input_contract")
    _require(isinstance(output_contract, dict), f"benchmark {benchmark_id} needs output_contract")
    _require(input_contract.get("frozen") is True, f"benchmark {benchmark_id} input_contract must be frozen")
    _require(output_contract.get("frozen") is True, f"benchmark {benchmark_id} output_contract must be frozen")
    for field in ("unit", "expression", "required_metadata", "label_access", "row_identity"):
        _require(
            input_contract.get(field) not in (None, "", []),
            f"benchmark {benchmark_id} input_contract.{field} is required",
        )
    corpus = input_contract.get("pretraining_corpus")
    _require(isinstance(corpus, dict), f"benchmark {benchmark_id} needs a pretraining corpus")
    for field in ("corpus_id", "artifact_dir", "vocabulary_path", "stats_path", "split", "assay_types"):
        _require(
            corpus.get(field) not in (None, "", []),
            f"benchmark {benchmark_id} pretraining_corpus.{field} is required",
        )
    _require(corpus.get("frozen") is True, f"benchmark {benchmark_id} pretraining corpus must be frozen")
    _require(corpus.get("split") == "train", f"benchmark {benchmark_id} pretraining split must be train")
    by_assay = corpus.get("expected_cells_by_assay")
    _require(
        isinstance(by_assay, dict)
        and set(by_assay) == set(corpus["assay_types"])
        and all(isinstance(value, int) and value > 0 for value in by_assay.values())
        and sum(by_assay.values()) == corpus.get("expected_cells"),
        f"benchmark {benchmark_id} pretraining cell inventory is invalid",
    )
    for field in ("artifact", "schema", "row_identity", "gene_or_label_order"):
        _require(
            output_contract.get(field) not in (None, "", []),
            f"benchmark {benchmark_id} output_contract.{field} is required",
        )


def _validate_variation_policy(benchmark: dict[str, Any]) -> None:
    """Freeze the benchmark while permitting only preprocessing, model, and loss search."""

    benchmark_id = benchmark["benchmark_id"]
    policy = benchmark.get("variation_policy")
    _require(isinstance(policy, dict), f"benchmark {benchmark_id} needs variation_policy")
    _require(policy.get("frozen") is True, f"benchmark {benchmark_id} variation_policy must be frozen")
    mutable_axes = policy.get("mutable_axes")
    _require(
        isinstance(mutable_axes, dict) and set(mutable_axes) == MUTABLE_VARIATION_AXES,
        f"benchmark {benchmark_id} mutable axes must be exactly data/architecture/algorithm",
    )
    data_axis = mutable_axes["data"]
    architecture_axis = mutable_axes["architecture"]
    algorithm_axis = mutable_axes["algorithm"]
    _require(data_axis.get("scope") == "normalization_only", f"benchmark {benchmark_id} data axis must be normalization-only")
    allowed_normalizations = benchmark["data_contract"]["normalization"].get("allowed_variants")
    _require(
        isinstance(allowed_normalizations, list)
        and allowed_normalizations
        and data_axis.get("allowed_values") == allowed_normalizations,
        f"benchmark {benchmark_id} normalization variants must match the data contract",
    )
    foundation_values = data_axis.get("foundation_values")
    baseline_values = data_axis.get("representation_free_baseline_values")
    if foundation_values is not None or baseline_values is not None:
        _require(
            isinstance(foundation_values, list)
            and isinstance(baseline_values, list)
            and set(foundation_values).isdisjoint(baseline_values)
            and set(foundation_values) | set(baseline_values) == set(allowed_normalizations),
            f"benchmark {benchmark_id} normalization roles must partition allowed values",
        )
    _require(
        architecture_axis.get("scope") == "model_architecture_and_capacity_only",
        f"benchmark {benchmark_id} architecture axis scope is invalid",
    )
    _require(
        algorithm_axis.get("scope") == "loss_objective_and_optimization_only",
        f"benchmark {benchmark_id} algorithm axis scope is invalid",
    )
    immutable = policy.get("immutable_fields")
    required_immutable = {
        "data_contract_id",
        "evaluator_contract_id",
        "input_contract",
        "output_contract",
        "split",
        "primary_metric",
    }
    _require(
        isinstance(immutable, list) and required_immutable <= set(immutable),
        f"benchmark {benchmark_id} variation policy does not freeze the benchmark boundary",
    )
    _require(
        policy.get("reference_role") == "control_only_not_a_mutable_axis",
        f"benchmark {benchmark_id} reference role is invalid",
    )


def _validate_contract_manifest(registry: dict[str, Any], manifest: Any) -> None:
    """Check the external, git-tracked contract index against registry entries."""
    _require(isinstance(manifest, dict), "contract manifest must be an object")
    _require(
        manifest.get("format") == "brainwm_immutable_benchmark_contracts_v1",
        "unsupported contract manifest format",
    )
    groups = manifest.get("contracts")
    _require(isinstance(groups, dict), "contract manifest needs data/evaluator groups")
    data_group = groups.get("data", {})
    evaluator_group = groups.get("evaluator", {})
    _require(isinstance(data_group, dict) and isinstance(evaluator_group, dict), "contract manifest groups must be objects")
    snapshot_fields = manifest.get("snapshot_fields")
    expected_snapshot_fields = [
        "data_contract",
        "evaluator_contract",
        "split",
        "evaluator",
        "primary_metric",
        "input_contract",
        "output_contract",
        "variation_policy",
    ]
    _require(
        snapshot_fields == expected_snapshot_fields,
        "contract manifest snapshot_fields are incomplete",
    )
    snapshots = manifest.get("benchmark_snapshots")
    _require(isinstance(snapshots, dict), "contract manifest needs benchmark_snapshots")
    benchmarks = benchmark_index(registry)
    rankable_ids = {
        benchmark_id
        for benchmark_id, benchmark in benchmarks.items()
        if benchmark.get("status") in RANKABLE_STATUSES
    }
    _require(set(snapshots) == rankable_ids, "contract manifest benchmark snapshot set mismatch")
    for benchmark_id, benchmark in benchmarks.items():
        if benchmark.get("status") not in RANKABLE_STATUSES:
            continue

        encoded_snapshot = snapshots.get(benchmark_id)
        _require(isinstance(encoded_snapshot, str), f"invalid benchmark snapshot: {benchmark_id}")
        try:
            frozen_snapshot = json.loads(encoded_snapshot)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid benchmark snapshot JSON: {benchmark_id}") from exc
        live_snapshot = {field: benchmark[field] for field in snapshot_fields}
        _require(
            frozen_snapshot == live_snapshot,
            f"full benchmark contract drift: {benchmark_id}",
        )
        data_id = _contract_id(benchmark, "data_contract")
        evaluator_id = _contract_id(benchmark, "evaluator_contract")
        data = data_group.get(data_id, {})
        evaluator = evaluator_group.get(evaluator_id, {})
        _require(isinstance(data, dict), f"contract manifest missing data contract: {data_id}")
        _require(isinstance(evaluator, dict), f"contract manifest missing evaluator contract: {evaluator_id}")
        _require(data.get("benchmark_id") == benchmark_id, f"data contract benchmark mismatch: {benchmark_id}")
        _require(evaluator.get("benchmark_id") == benchmark_id, f"evaluator contract benchmark mismatch: {benchmark_id}")
        registry_data = benchmark["data_contract"]
        registry_eval = benchmark["evaluator_contract"]
        for field in ("cohort_id", "split_id", "panel_id", "assay"):
            _require(data.get(field) == registry_data.get(field), f"data contract manifest mismatch for {benchmark_id}: {field}")
        _require(
            data.get("normalization_fit_scope") == registry_data.get("normalization", {}).get("fit_scope"),
            f"data contract manifest mismatch for {benchmark_id}: normalization_fit_scope",
        )
        _require(
            data.get("query_policy") == registry_data.get("query_policy"),
            f"data contract manifest mismatch for {benchmark_id}: query_policy",
        )
        for field in ("evaluator_id", "split_id", "primary_metric", "query_policy"):
            _require(evaluator.get(field) == registry_eval.get(field), f"evaluator contract manifest mismatch for {benchmark_id}: {field}")


def validate_results(
    registry: dict[str, Any],
    results: Iterable[dict[str, Any]],
    *,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    validate_registry(registry)
    benchmarks = benchmark_index(registry)
    checked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    root = Path(project_root) if project_root is not None else None
    receipt_cache: dict[Path, dict[str, Any]] = {}

    required = {
        "format",
        "result_id",
        "benchmark_id",
        "task_id",
        "training_stage",
        "method_id",
        "method_name",
        "method_class",
        "axis",
        "seed",
        "budget",
        "iteration",
        "split_id",
        "evaluator_id",
        "primary_metric",
        "direction",
        "raw_metric",
        "reward",
        "evidence_path",
        "diagnostics",
    }
    for row in results:
        _require(isinstance(row, dict), "every result must be an object")
        missing = required - set(row)
        _require(not missing, f"result is missing required fields: {sorted(missing)}")
        _require(row["format"] == RESULT_FORMAT, f"unsupported result format: {row.get('result_id')}")
        result_id = row["result_id"]
        _require(isinstance(result_id, str) and result_id and result_id not in seen_ids,
                 f"invalid or duplicate result_id: {result_id}")
        seen_ids.add(result_id)

        benchmark_id = row["benchmark_id"]
        _require(benchmark_id in benchmarks, f"unknown benchmark_id: {benchmark_id}")
        benchmark = benchmarks[benchmark_id]
        _require(benchmark["status"] in RANKABLE_STATUSES,
                 f"results cannot be attached to {benchmark['status']} benchmark {benchmark_id}")
        _require(row["task_id"] == benchmark["task_id"], f"task mismatch in {result_id}")
        _require(row["method_class"] in METHOD_CLASSES, f"invalid method_class in {result_id}")
        baseline_specs = {
            item["method_id"]: item for item in benchmark.get("baselines", [])
        }
        baseline_spec = baseline_specs.get(row["method_id"])
        expected_stage = benchmark["training_stage"]
        if baseline_spec is not None:
            _require(
                row["method_class"] == "baseline",
                f"registered baseline method must use method_class=baseline in {result_id}",
            )
            expected_stage = baseline_spec.get("training_stage", expected_stage)
        _require(
            row["training_stage"] == expected_stage,
            f"training_stage mismatch in {result_id}",
        )
        _validate_axis(row["axis"], result_id)
        _validate_seed(row["seed"], result_id)
        _validate_budget(row["budget"], result_id)
        _require(isinstance(row["iteration"], int) and row["iteration"] >= 0,
                 f"iteration must be a non-negative integer in {result_id}")

        metric = benchmark["primary_metric"]
        _require(row["split_id"] == benchmark["split"]["split_id"], f"split mismatch in {result_id}")
        _require(row["evaluator_id"] == benchmark["evaluator"]["evaluator_id"],
                 f"evaluator mismatch in {result_id}")
        expected_data_contract_id = _contract_id(benchmark, "data_contract")
        expected_evaluator_contract_id = _contract_id(benchmark, "evaluator_contract")
        data_contract_id = row.get("data_contract_id")
        evaluator_contract_id = row.get("evaluator_contract_id")
        # A row lacking both fields is an old table entry.  It remains
        # readable for historical reporting, but ``append_result`` rejects it
        # so every newly persisted result is explicitly bound.  A row missing
        # only one field is never considered legacy: partial bindings are an
        # error because they are ambiguous.
        legacy_contract = (
            data_contract_id in (None, "")
            and evaluator_contract_id in (None, "")
            and row.get("contract_status") == LEGACY_CONTRACT_STATUS
            and isinstance(row.get("diagnostics"), dict)
            and row["diagnostics"].get("contract_status") == LEGACY_CONTRACT_STATUS
        )
        _require(
            legacy_contract or (data_contract_id not in (None, "") and evaluator_contract_id not in (None, "")),
            f"result must bind both data_contract_id and evaluator_contract_id in {result_id}",
        )
        if not legacy_contract:
            _require(
                data_contract_id == expected_data_contract_id,
                f"data contract mismatch in {result_id}",
            )
            _require(
                evaluator_contract_id == expected_evaluator_contract_id,
                f"evaluator contract mismatch in {result_id}",
            )
            allowed_normalizations = benchmark["variation_policy"]["mutable_axes"]["data"]["allowed_values"]
            _require(
                row["axis"]["data"] in allowed_normalizations,
                f"axis.data must be a declared normalization variant in {result_id}",
            )
            data_axis_policy = benchmark["variation_policy"]["mutable_axes"]["data"]
            if row["method_class"] == "baseline" and data_axis_policy.get("representation_free_baseline_values"):
                _require(
                    row["axis"]["data"] in data_axis_policy["representation_free_baseline_values"],
                    f"baseline axis.data must declare an evaluation normalization view in {result_id}",
                )
            elif row["method_class"] != "baseline" and data_axis_policy.get("foundation_values"):
                _require(
                    row["axis"]["data"] in data_axis_policy["foundation_values"],
                    f"foundation axis.data must declare its pretraining normalization in {result_id}",
                )
            fixed_head_seed = benchmark["evaluator"].get("head_seed")
            if fixed_head_seed is not None:
                _require(
                    row["seed"].get("evaluator") == [fixed_head_seed],
                    f"evaluator seed mismatch in {result_id}",
                )
        _require(row["primary_metric"] == metric["name"], f"primary metric mismatch in {result_id}")
        _require(row["direction"] == metric["direction"], f"metric direction mismatch in {result_id}")
        _require(_finite_number(row["raw_metric"]), f"raw_metric must be finite in {result_id}")
        _require(_finite_number(row["reward"]), f"reward must be finite in {result_id}")
        expected_reward = row["raw_metric"] if metric["reward_transform"] == "identity" else -row["raw_metric"]
        _require(math.isclose(row["reward"], expected_reward, rel_tol=1e-12, abs_tol=1e-12),
                 f"reward does not match the frozen transform in {result_id}")
        _require(isinstance(row["diagnostics"], dict), f"diagnostics must be an object in {result_id}")
        if benchmark.get("claim_scope") == TRAIN_INTERNAL_ARCHITECTURE_SCOPE:
            _require(
                row["diagnostics"].get("evidence_scope")
                == TRAIN_INTERNAL_ARCHITECTURE_SCOPE,
                f"train-internal evidence scope mismatch in {result_id}",
            )
            _require(
                row["diagnostics"].get("promotion_eligible") is False,
                f"train-internal result permits promotion in {result_id}",
            )
            _require(
                row["diagnostics"].get("validation_opened") is False
                and row["diagnostics"].get("test_opened") is False,
                f"train-internal result opened held-out data in {result_id}",
            )
        if baseline_spec is not None and "representation_role" in baseline_spec:
            _require(
                row["diagnostics"].get("representation_role")
                == baseline_spec["representation_role"],
                f"representation_role mismatch in {result_id}",
            )
        if baseline_spec is not None and "stage0_encoder_used" in baseline_spec:
            _require(
                row["diagnostics"].get("stage0_encoder_used")
                is baseline_spec["stage0_encoder_used"],
                f"stage0_encoder_used mismatch in {result_id}",
            )

        evidence = row["evidence_path"]
        _require(isinstance(evidence, str) and evidence and not Path(evidence).is_absolute(),
                 f"evidence_path must be project-relative in {result_id}")
        if root is not None:
            evidence_path = _resolve_project_file(
                evidence,
                project_root=root,
                result_id=result_id,
                label="evidence_path",
            )
            _require(evidence_path.is_file(), f"missing evidence_path for {result_id}: {evidence}")
            if not legacy_contract:
                receipt = _load_receipt_json(
                    evidence_path,
                    project_root=root,
                    cache=receipt_cache,
                    result_id=result_id,
                    label="evidence receipt",
                )
                _validate_verified_receipt(
                    benchmark,
                    row,
                    receipt,
                    receipt_path=evidence_path,
                    project_root=root,
                    receipt_cache=receipt_cache,
                )
        checked_row = deepcopy(row)
        checked_row.setdefault("diagnostics", {})
        if legacy_contract:
            checked_row["contract_status"] = LEGACY_CONTRACT_STATUS
            checked_row["diagnostics"]["contract_status"] = LEGACY_CONTRACT_STATUS
            checked_row["diagnostics"].setdefault(
                "legacy_contract_fields_missing", list(CONTRACT_ID_FIELDS)
            )
        else:
            checked_row["contract_status"] = VERIFIED_CONTRACT_STATUS
            checked_row["diagnostics"]["contract_status"] = VERIFIED_CONTRACT_STATUS
        checked.append(checked_row)
    return checked


def validate_scoped_results(
    registry: dict[str, Any],
    results: Iterable[dict[str, Any]],
    *,
    scope: str,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Validate receipts and enforce downstream/auxiliary file separation."""

    checked = validate_results(registry, results, project_root=project_root)
    if scope == "downstream":
        allowed = set(downstream_benchmark_index(registry))
    elif scope == "auxiliary":
        allowed = set(auxiliary_benchmark_index(registry))
    else:
        raise ValueError("result scope must be downstream or auxiliary")
    misplaced = sorted(
        {row["benchmark_id"] for row in checked if row["benchmark_id"] not in allowed}
    )
    _require(
        not misplaced,
        f"{scope} result table contains out-of-scope benchmarks: {misplaced}",
    )
    return checked


def _validate_axis(axis: Any, result_id: str) -> None:
    _require(isinstance(axis, dict), f"axis must be an object in {result_id}")
    _require(set(axis) == {"data", "architecture", "algorithm", "variation"},
             f"axis must bind Data/Architecture/Algorithm and variation in {result_id}")
    for key in ("data", "architecture", "algorithm"):
        _require(isinstance(axis[key], str) and axis[key], f"axis.{key} is invalid in {result_id}")
    _require(axis["variation"] in VARIATION_AXES, f"invalid variation axis in {result_id}")


def _verified_rows(results: Iterable[dict[str, Any]], benchmark_id: str) -> list[dict[str, Any]]:
    """Select rows eligible for the formal table.

    Rows written before immutable contracts were introduced remain available
    to historical reports, but cannot affect promotion, rank, or trajectories.
    """

    return [
        row
        for row in results
        if row.get("benchmark_id") == benchmark_id
        and row.get("contract_status") == VERIFIED_CONTRACT_STATUS
        and all(row.get(field) for field in CONTRACT_ID_FIELDS)
    ]


def _validate_seed(seed: Any, result_id: str) -> None:
    _require(isinstance(seed, dict) and isinstance(seed.get("aggregation"), str),
             f"seed must bind values and aggregation in {result_id}")
    groups = [value for key, value in seed.items() if key != "aggregation"]
    _require(groups and all(isinstance(value, list) for value in groups),
             f"seed groups must be lists in {result_id}")
    flattened = [item for values in groups for item in values]
    _require(flattened and all(isinstance(item, (int, str)) and not isinstance(item, bool) for item in flattened),
             f"seed values are invalid in {result_id}")


def _validate_budget(budget: Any, result_id: str) -> None:
    _require(isinstance(budget, dict) and budget, f"budget must be a non-empty object in {result_id}")
    for key, value in budget.items():
        if isinstance(value, list):
            _require(all(_finite_number(item) and item >= 0 for item in value),
                     f"budget list {key} is invalid in {result_id}")
        else:
            _require(_finite_number(value) and value >= 0, f"budget {key} is invalid in {result_id}")


def _resolve_project_file(
    value: str | Path,
    *,
    project_root: Path,
    result_id: str,
    label: str,
) -> Path:
    root = project_root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside project root in {result_id}: {value}") from exc
    return resolved


def _resolve_workspace_or_project_file(
    value: str | Path,
    *,
    project_root: Path,
    result_id: str,
    label: str,
) -> Path:
    """Resolve declarations rooted at either the package or its workspace."""

    candidate = Path(value)
    if (
        not candidate.is_absolute()
        and candidate.parts
        and candidate.parts[0] == project_root.name
    ):
        candidate = Path(*candidate.parts[1:])
    return _resolve_project_file(
        candidate,
        project_root=project_root,
        result_id=result_id,
        label=label,
    )


def _load_receipt_json(
    value: str | Path,
    *,
    project_root: Path,
    cache: dict[Path, dict[str, Any]],
    result_id: str,
    label: str,
) -> dict[str, Any]:
    path = _resolve_project_file(
        value,
        project_root=project_root,
        result_id=result_id,
        label=label,
    )
    _require(path.is_file(), f"missing {label} for {result_id}: {value}")
    if path in cache:
        return cache[path]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON {label} in {result_id}: {value}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object in {result_id}")
    cache[path] = payload
    return payload


def _require_same_number(actual: Any, expected: Any, message: str) -> None:
    _require(
        _finite_number(actual)
        and _finite_number(expected)
        and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
        message,
    )


def _require_closed_access(receipt: dict[str, Any], result_id: str) -> None:
    _require(receipt.get("query_access") == {"opened": False},
             f"receipt query access is not closed in {result_id}")
    _require(receipt.get("test_access") == {"opened": False},
             f"receipt test access is not closed in {result_id}")


def _resolve_receipt_file(
    value: str | Path,
    *,
    base_dir: Path,
    project_root: Path,
    result_id: str,
    label: str,
) -> Path:
    """Resolve a receipt-local artifact while keeping it inside the project.

    V2 score receipts intentionally use content/Git provenance rather than a
    hash of the large prediction files.  Receipt-local references still need a
    strict boundary so a caller cannot substitute an arbitrary file outside
    the checked-out project.
    """

    _require(isinstance(value, (str, Path)) and str(value),
             f"{label} is missing in {result_id}")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} is outside project root in {result_id}: {value}"
        ) from exc
    return resolved


def _v2_scope_is_internal(
    benchmark: dict[str, Any],
    receipt: dict[str, Any],
    result_id: str,
    *,
    score_receipt: bool = True,
) -> None:
    """Enforce the explicit non-promotion scope on every V2 artifact."""

    scope = receipt.get("benchmark_scope")
    _require(isinstance(scope, dict), f"V2 benchmark scope is missing in {result_id}")
    _require(scope.get("kind") == "label_blind_internal_retest",
             f"V2 benchmark scope mismatch in {result_id}")
    _require(scope.get("independent_target") is False,
             f"V2 independent-target flag is invalid in {result_id}")
    _require(scope.get("promotion_eligible") is False,
             f"V2 scope permits promotion in {result_id}")
    _require(scope.get("new_benchmark_id_does_not_reset_target_history") is True,
             f"V2 target-history boundary is missing in {result_id}")
    _require(scope.get("required_claim") == "internal label-blind resample only",
             f"V2 claim scope is invalid in {result_id}")
    _require(receipt.get("promotion_eligible") is False,
             f"V2 receipt permits promotion in {result_id}")
    if score_receipt:
        _require(receipt.get("scientific_claim") == "label_blind_internal_retest_only",
                 f"V2 scientific claim is invalid in {result_id}")
    _require(receipt.get("historical_target_adaptation") is True,
             f"V2 historical-target flag is missing in {result_id}")
    if score_receipt:
        _require(receipt.get("same_run_post_score_refit") == "forbidden",
                 f"V2 post-score refit policy is invalid in {result_id}")
        _require(receipt.get("cross_experiment_search_feedback")
                 == "allowed_internal_validation_only",
                 f"V2 cross-experiment feedback policy is invalid in {result_id}")
    # Keep the registry-level declarations in sync when this validator is used
    # with a synthetic row in tests or an independently rendered snapshot.
    if benchmark.get("claim_scope") is not None:
        _require(benchmark.get("claim_scope") == "internal_validation_only",
                 f"V2 registry claim scope drifted in {result_id}")
    if benchmark.get("promotion_eligible") is not None:
        _require(benchmark.get("promotion_eligible") is False,
                 f"V2 registry promotion flag drifted in {result_id}")


def _validate_v2_selection(
    benchmark: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    result_id: str,
) -> dict[str, Any]:
    """Check the immutable label-blind selection receipt and manifest."""

    split = benchmark["split"]
    selection = receipt.get("selection")
    _require(isinstance(selection, dict), f"V2 selection receipt is missing in {result_id}")
    expected_rows = int(split.get("train_rows", 0)) + int(split.get("validation_rows", 0))
    _require(selection.get("rows") == expected_rows,
             f"V2 selection row count mismatch in {result_id}")
    _require(selection.get("partition_counts") == {
        "train": int(split.get("train_rows")),
        "validation": int(split.get("validation_rows")),
    }, f"V2 selection partition counts mismatch in {result_id}")
    _require(selection.get("rows_per_donor") == int(split.get("rows_per_donor")),
             f"V2 selection donor quota mismatch in {result_id}")
    _require(selection.get("label_columns_present") == [],
             f"V2 selection contains label columns in {result_id}")
    _require(selection.get("target_label_values_read", False) is False,
             f"V2 selection read target labels in {result_id}")

    selection_manifest = _resolve_receipt_file(
        selection.get("manifest_path", ""), base_dir=receipt_path.parent,
        project_root=project_root, result_id=result_id, label="V2 selection manifest",
    )
    selection_receipt_path = _resolve_receipt_file(
        selection.get("receipt_path", ""), base_dir=receipt_path.parent,
        project_root=project_root, result_id=result_id, label="V2 selection receipt",
    )
    _require(selection_manifest.is_file(),
             f"missing V2 selection manifest in {result_id}")
    selection_payload = _load_receipt_json(
        selection_receipt_path, project_root=project_root, cache=receipt_cache,
        result_id=result_id, label="V2 selection receipt",
    )
    _require(selection_payload.get("format") == "brainwm_ann_s64_label_blind_selection_v2",
             f"V2 selection receipt format mismatch in {result_id}")
    _require(selection_payload.get("status") == "materialized_labels_closed_not_rankable",
             f"V2 selection receipt status mismatch in {result_id}")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("data_contract_id", _contract_id(benchmark, "data_contract")),
        ("split_id", split["split_id"]),
        ("evaluator_contract_id", _contract_id(benchmark, "evaluator_contract")),
        ("evaluator_id", benchmark["evaluator"]["evaluator_id"]),
    ):
        _require(selection_payload.get(field) == expected,
                 f"V2 selection {field} binding mismatch in {result_id}")
    selection_block = selection_payload.get("selection")
    _require(isinstance(selection_block, dict),
             f"V2 selection details are missing in {result_id}")
    selection_manifest_ref = selection_payload.get("selection_manifest")
    _require(selection_manifest_ref in {
        str(selection_manifest), selection.get("manifest_path"),
        selection_manifest.relative_to(project_root.resolve()).as_posix(),
    }, f"V2 selection manifest binding mismatch in {result_id}")
    _require(selection_payload.get("label_access", {}).get("target_labels_remain_closed") is True,
             f"V2 selection target-label boundary is invalid in {result_id}")

    # The selection parquet is a content boundary, not a hash dependency.  A
    # lightweight schema/row-count check catches accidental replacement while
    # keeping the registry validator independent of a particular parquet
    # engine.  The evaluator itself additionally records exact row equality.
    try:
        import pandas as pd  # local optional dependency used by the evaluator
    except ImportError:  # pragma: no cover - project environments provide pandas
        pd = None
    if pd is not None:
        try:
            frame = pd.read_parquet(selection_manifest)
        except Exception as exc:  # pragma: no cover - engine-specific error text
            raise ValueError(f"invalid V2 selection manifest in {result_id}") from exc
        expected_columns = {
            "source_id", "row_index", "sample_key", "split", "donor_id",
            "partition", "selection_rank",
        }
        _require(set(frame.columns) == expected_columns,
                 f"V2 selection manifest columns mismatch in {result_id}")
        _require(len(frame) == expected_rows,
                 f"V2 selection manifest rows mismatch in {result_id}")
        _require(not bool(frame.duplicated(["source_id", "row_index"]).any()),
                 f"V2 selection manifest row identity is not unique in {result_id}")

    return {
        "manifest_path": selection_manifest,
        "receipt_path": selection_receipt_path,
        "payload": selection_payload,
    }


def _validate_v2_prediction_manifest(
    benchmark: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    result_id: str,
    expected_method_id: str,
    expected_model_seed: int | None,
    expected_training_budget: int,
) -> dict[str, Any]:
    """Validate the prediction freeze and one-time target reveal marker."""

    manifest_path = _resolve_receipt_file(
        receipt.get("prediction_manifest", ""), base_dir=receipt_path.parent,
        project_root=project_root, result_id=result_id, label="V2 prediction manifest",
    )
    manifest = _load_receipt_json(
        manifest_path, project_root=project_root, cache=receipt_cache,
        result_id=result_id, label="V2 prediction manifest",
    )
    _require(manifest.get("format") == "brainwm_ann_s64_label_blind_prediction_manifest_v2",
             f"V2 prediction manifest format mismatch in {result_id}")
    _require(manifest.get("status") == "predictions_frozen_target_labels_closed",
             f"V2 prediction manifest is not frozen in {result_id}")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("data_contract_id", _contract_id(benchmark, "data_contract")),
        ("split_id", benchmark["split"]["split_id"]),
        ("evaluator_contract_id", _contract_id(benchmark, "evaluator_contract")),
        ("evaluator_id", benchmark["evaluator"]["evaluator_id"]),
        ("method_id", expected_method_id),
        ("model_seed", expected_model_seed),
        ("training_budget_steps", expected_training_budget),
        ("head_final_step", benchmark["evaluator"]["max_steps"]),
    ):
        _require(manifest.get(field) == expected,
                 f"V2 prediction manifest {field} mismatch in {result_id}")
    _v2_scope_is_internal(benchmark, manifest, result_id, score_receipt=False)
    _require(manifest.get("target_label_values_read") is False,
             f"V2 prediction manifest exposes target labels in {result_id}")
    _require(manifest.get("target_metric_evaluations") == 0,
             f"V2 prediction manifest scored target during prepare in {result_id}")
    _require(manifest.get("content_binding") == "prediction_file_must_equal_embedded_records",
             f"V2 prediction content binding is invalid in {result_id}")
    selection = manifest.get("selection")
    split = benchmark["split"]
    _require(isinstance(selection, dict)
             and selection.get("rows") == int(split["train_rows"]) + int(split["validation_rows"])
             and selection.get("partition_counts") == {
                 "train": int(split["train_rows"]),
                 "validation": int(split["validation_rows"]),
             }
             and selection.get("label_columns_present") == [],
             f"V2 prediction selection binding mismatch in {result_id}")
    _require(manifest.get("prediction_rows") == int(split["validation_rows"]),
             f"V2 prediction row count mismatch in {result_id}")
    columns = manifest.get("prediction_columns")
    records = manifest.get("prediction_records")
    _require(isinstance(columns, list) and isinstance(records, list)
             and len(records) == manifest.get("prediction_rows"),
             f"V2 prediction records are incomplete in {result_id}")
    required_columns = {
        "source_id", "row_index", "sample_key", "split", "donor_id",
        "partition", "selection_rank", "prediction_cl_term",
        "confidence_cl_term", "prediction_original_cell_type",
        "confidence_original_cell_type", "prediction_original_subclass",
        "confidence_original_subclass",
    }
    _require(set(columns) == required_columns,
             f"V2 prediction columns mismatch in {result_id}")
    forbidden = {
        "cl_term", "original_cell_type", "original_subclass",
        "label", "target_label", "truth",
    }
    _require(not (forbidden & set(columns)),
             f"V2 prediction manifest contains target labels in {result_id}")
    for record in records:
        _require(isinstance(record, dict) and set(record) == required_columns,
                 f"V2 prediction record schema mismatch in {result_id}")
        for key in ("source_id", "row_index", "selection_rank"):
            _require(isinstance(record[key], int) and not isinstance(record[key], bool),
                     f"V2 prediction identity field is invalid in {result_id}")
        for key in ("confidence_cl_term", "confidence_original_cell_type",
                    "confidence_original_subclass"):
            _require(_finite_number(record[key]) and 0.0 <= float(record[key]) <= 1.0,
                     f"V2 prediction confidence is invalid in {result_id}")

    prediction_path = _resolve_receipt_file(
        manifest.get("prediction_file", ""), base_dir=manifest_path.parent,
        project_root=project_root, result_id=result_id, label="V2 frozen prediction file",
    )
    _require(prediction_path.is_file(),
             f"missing V2 frozen prediction file in {result_id}")
    scored_path = _resolve_receipt_file(
        receipt.get("scored_predictions", ""), base_dir=receipt_path.parent,
        project_root=project_root, result_id=result_id, label="V2 scored prediction file",
    )
    _require(scored_path.is_file(),
             f"missing V2 scored prediction file in {result_id}")

    marker_name = receipt.get("integrity", {}).get("exclusive_reveal_marker")
    marker_path = _resolve_receipt_file(
        marker_name or "", base_dir=receipt_path.parent, project_root=project_root,
        result_id=result_id, label="V2 reveal marker",
    )
    marker = _load_receipt_json(
        marker_path, project_root=project_root, cache=receipt_cache,
        result_id=result_id, label="V2 reveal marker",
    )
    _require(marker.get("format") == "brainwm_ann_s64_label_blind_score_v2_reveal_marker"
             and marker.get("status") == "target_label_reveal_started_no_retry",
             f"V2 reveal marker is invalid in {result_id}")
    _require(marker.get("benchmark_id") == benchmark["benchmark_id"]
             and marker.get("prediction_manifest") == manifest_path.name
             and marker.get("target_rows") == int(split["validation_rows"])
             and marker.get("label_read_count_before_marker") == 0
             and marker.get("exclusive_create") is True,
             f"V2 reveal marker binding mismatch in {result_id}")
    return {
        "path": manifest_path,
        "manifest": manifest,
        "prediction_path": prediction_path,
        "scored_path": scored_path,
        "marker_path": marker_path,
    }


def _validate_v2_metrics(
    benchmark: dict[str, Any], row: dict[str, Any], receipt: dict[str, Any]
) -> None:
    result_id = row["result_id"]
    split = benchmark["split"]
    total = int(split["validation_rows"])
    metrics = receipt.get("metrics")
    _require(isinstance(metrics, dict), f"V2 metrics are missing in {result_id}")
    expected_levels = [f"level_{index}" for index in range(3)]
    _require(set(metrics) == set(expected_levels),
             f"V2 metric levels mismatch in {result_id}")
    for level_name in expected_levels:
        level = metrics[level_name]
        _require(isinstance(level, dict), f"V2 metric level is invalid in {result_id}")
        for field in (
            "valid", "unknown", "coverage", "unknown_fraction", "accuracy",
            "macro_f1", "balanced_accuracy", "coverage_adjusted_macro_f1",
            "coverage_adjusted_accuracy",
        ):
            _require(field in level and _finite_number(level[field]),
                     f"V2 metric field {field} is invalid in {result_id}")
        _require(int(level["valid"]) + int(level["unknown"]) == total,
                 f"V2 metric population mismatch in {result_id}")
        _require_same_number(level["coverage"], float(level["valid"]) / total,
                             f"V2 coverage mismatch in {result_id}")
        _require_same_number(level["unknown_fraction"], float(level["unknown"]) / total,
                             f"V2 unknown fraction mismatch in {result_id}")
        _require_same_number(
            level["coverage_adjusted_macro_f1"],
            float(level["coverage"]) * float(level["macro_f1"]),
            f"V2 coverage-adjusted macro-F1 mismatch in {result_id}",
        )
        _require_same_number(
            level["coverage_adjusted_accuracy"],
            float(level["coverage"]) * float(level["accuracy"]),
            f"V2 coverage-adjusted accuracy mismatch in {result_id}",
        )
    metric_name = benchmark["primary_metric"]["name"]
    primary_level = metrics["level_2"]
    _require_same_number(primary_level.get("coverage_adjusted_macro_f1"), row["raw_metric"],
                         f"V2 primary metric mismatch in {result_id}")
    _require_same_number(receipt.get("raw_metric"), row["raw_metric"],
                         f"V2 receipt raw metric mismatch in {result_id}")
    _require_same_number(receipt.get("reward"), row["reward"],
                         f"V2 receipt reward mismatch in {result_id}")
    _require(receipt.get("primary_metric") == metric_name,
             f"V2 primary metric name mismatch in {result_id}")


def _validate_ann_label_blind_v2_score_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    expected_method_id: str | None = None,
    expected_model_seed: int | None = None,
    expected_training_budget: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Fail-closed validator for one V2 score receipt."""

    result_id = row["result_id"]
    _require(receipt.get("format") == ANN_LABEL_BLIND_V2_SCORE_FORMAT,
             f"V2 score receipt format mismatch in {result_id}")
    _require(receipt.get("status") == "complete_internal_retest_only",
             f"V2 score receipt is not complete in {result_id}")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("data_contract_id", _contract_id(benchmark, "data_contract")),
        ("split_id", benchmark["split"]["split_id"]),
        ("evaluator_contract_id", _contract_id(benchmark, "evaluator_contract")),
        ("evaluator_id", benchmark["evaluator"]["evaluator_id"]),
        ("method_id", expected_method_id or row["method_id"]),
    ):
        _require(receipt.get(field) == expected,
                 f"V2 score receipt {field} binding mismatch in {result_id}")
    row_model_seeds = row.get("seed", {}).get("model", [])
    row_model = row_model_seeds[0] if row_model_seeds else None
    model_seed = expected_model_seed if expected_model_seed is not None else receipt.get("model_seed")
    if expected_model_seed is None:
        _require(receipt.get("model_seed") in (None, row_model),
                 f"V2 score receipt model seed mismatch in {result_id}")
    training_budget = (
        int(expected_training_budget)
        if expected_training_budget is not None
        else int(receipt.get("training_budget_steps", -1))
    )
    _require(receipt.get("training_budget_steps") == training_budget,
             f"V2 score receipt training budget mismatch in {result_id}")
    _v2_scope_is_internal(benchmark, receipt, result_id)
    _require(receipt.get("target_label_read_count") == 1
             and receipt.get("target_label_read_purpose") == "final_metric_only",
             f"V2 target-label access boundary mismatch in {result_id}")
    _require(receipt.get("integrity", {}).get("sha256_used") is False,
             f"V2 receipt must use content/Git integrity in {result_id}")
    selection_info = _validate_v2_selection(
        benchmark, receipt, receipt_path=receipt_path, project_root=project_root,
        receipt_cache=receipt_cache, result_id=result_id,
    )
    prediction_info = _validate_v2_prediction_manifest(
        benchmark, receipt, receipt_path=receipt_path, project_root=project_root,
        receipt_cache=receipt_cache, result_id=result_id,
        expected_method_id=expected_method_id or row["method_id"],
        expected_model_seed=model_seed,
        expected_training_budget=training_budget,
    )
    _require(receipt.get("head_final_step") == benchmark["evaluator"]["max_steps"] == 400,
             f"V2 fixed head step mismatch in {result_id}")
    seed = row.get("seed", {})
    if model_seed is None:
        _require(seed == {
            "model": [],
            "evaluator": [benchmark["evaluator"]["head_seed"]],
            "aggregation": "single",
        }, f"V2 representation-free seed schema mismatch in {result_id}")
        _require(row.get("budget") == {
            "stage0_training_steps": 0,
            "annotation_head_max_steps": 400,
        }, f"V2 representation-free budget schema mismatch in {result_id}")
    else:
        _require(seed == {
            "model": [model_seed],
            "evaluator": [benchmark["evaluator"]["head_seed"]],
            "aggregation": "single",
        }, f"V2 Stage0 seed schema mismatch in {result_id}")
        _require(row.get("budget") == {
            "stage0_training_steps": training_budget,
            "annotation_head_max_steps": 400,
        }, f"V2 Stage0 budget schema mismatch in {result_id}")
    head = prediction_info["manifest"].get("head_training")
    _require(isinstance(head, dict)
             and head.get("head_seed") == benchmark["evaluator"]["head_seed"]
             and head.get("completed_steps") == 400
             and head.get("selected_checkpoint_step") == 400
             and head.get("checkpoint_policy") == "fixed_final_step_400"
             and head.get("target_metric_evaluations_during_training") == 0
             and head.get("target_label_reads_during_training") == 0
             and head.get("training_rows") == benchmark["split"]["train_rows"],
             f"V2 head-training provenance mismatch in {result_id}")
    provenance = prediction_info["manifest"].get("method_provenance")
    _require(isinstance(provenance, dict)
             and provenance.get("format") == "brainwm_ann_s64_label_blind_method_provenance_v2"
             and provenance.get("status") == "frozen_before_v2_prepare"
             and provenance.get("method_id") == (expected_method_id or row["method_id"])
             and provenance.get("historical_target_adaptation") is True
             and provenance.get("promotion_eligible") is False,
             f"V2 method provenance mismatch in {result_id}")
    if model_seed is None:
        _require(provenance.get("model_seed") is None and training_budget == 0,
                 f"V2 representation-free seed/budget mismatch in {result_id}")
    else:
        _require(isinstance(model_seed, int) and not isinstance(model_seed, bool)
                 and provenance.get("model_seed") == model_seed and training_budget > 0,
                 f"V2 Stage0 seed/budget mismatch in {result_id}")
        state = prediction_info["manifest"].get("state_provenance", {})
        checkpoint_step = state.get("checkpoint_global_step")
        if checkpoint_step is None:
            checkpoint_step = state.get("stage0", {}).get("checkpoint_global_step")
        _require(checkpoint_step == training_budget,
                 f"V2 Stage0 checkpoint step mismatch in {result_id}")
    _validate_ann_label_blind_v2_metrics(benchmark, row, receipt)
    return {
        "model_seed": model_seed,
        "training_budget_steps": training_budget,
        "raw_metric": float(row["raw_metric"]),
        "selection": selection_info,
        "prediction": prediction_info,
    }


def _validate_ann_label_blind_v2_metrics(
    benchmark: dict[str, Any], row: dict[str, Any], receipt: dict[str, Any]
) -> None:
    # Kept as a named shim so aggregate validation and external tests can call
    # the same metric checks without depending on the row-level dispatcher.
    _validate_v2_metrics(benchmark, row, receipt)


def _validate_ann_label_blind_v2_aggregate_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    **_: Any,
) -> None:
    """Validate a deterministic three-seed median V2 aggregate."""

    result_id = row["result_id"]
    _require(receipt.get("format") in {
        ANN_LABEL_BLIND_V2_AGGREGATE_FORMAT,
        "brainwm_ann_s64_label_blind_aggregate_v2",
        "brainwm_ann_s64_label_blind_stage0_aggregate_v2",
        "brainwm_ann_s64_label_blind_fusion_aggregate_v2",
    }, f"unsupported V2 aggregate format in {result_id}")
    _require(receipt.get("status") == "complete_internal_retest_only",
             f"V2 aggregate receipt is not complete in {result_id}")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("data_contract_id", _contract_id(benchmark, "data_contract")),
        ("split_id", benchmark["split"]["split_id"]),
        ("evaluator_contract_id", _contract_id(benchmark, "evaluator_contract")),
        ("evaluator_id", benchmark["evaluator"]["evaluator_id"]),
        ("method_id", row["method_id"]),
        ("primary_metric", row["primary_metric"]),
    ):
        _require(receipt.get(field) == expected,
                 f"V2 aggregate {field} binding mismatch in {result_id}")
    _v2_scope_is_internal(benchmark, receipt, result_id)
    _require(receipt.get("aggregation") == "median",
             f"V2 aggregate must use median in {result_id}")
    _require(receipt.get("head_final_step") == 400,
             f"V2 aggregate fixed head step mismatch in {result_id}")
    model_seeds = receipt.get("model_seeds")
    evaluator_seeds = receipt.get("evaluator_seeds")
    _require(isinstance(model_seeds, list) and len(model_seeds) == 3
             and len(set(model_seeds)) == 3
             and all(isinstance(value, int) and not isinstance(value, bool) for value in model_seeds),
             f"V2 aggregate model seeds are invalid in {result_id}")
    _require(evaluator_seeds == [benchmark["evaluator"]["head_seed"]],
             f"V2 aggregate evaluator seed mismatch in {result_id}")
    seed = row["seed"]
    _require(seed.get("model") == model_seeds
             and seed.get("evaluator") == evaluator_seeds
             and seed.get("aggregation") == "median",
             f"V2 aggregate row seed mismatch in {result_id}")
    budget = row["budget"]
    _require(set(budget) == {"stage0_training_steps", "annotation_head_max_steps", "seed_count"}
             and budget.get("seed_count") == 3
             and budget.get("annotation_head_max_steps") == 400,
             f"V2 aggregate budget schema mismatch in {result_id}")
    _require(receipt.get("stage0_training_steps") == budget.get("stage0_training_steps")
             and budget.get("stage0_training_steps") == 6000,
             f"V2 aggregate fixed Stage0 budget mismatch in {result_id}")
    values = receipt.get("raw_metric_values")
    _require(isinstance(values, list) and len(values) == 3
             and all(_finite_number(value) for value in values),
             f"V2 aggregate metric values are invalid in {result_id}")
    aggregate = float(median(float(value) for value in values))
    _require_same_number(receipt.get("raw_metric_median"), aggregate,
                         f"V2 aggregate median is invalid in {result_id}")
    _require_same_number(row["raw_metric"], aggregate,
                         f"V2 aggregate row metric mismatch in {result_id}")
    _require_same_number(receipt.get("reward"), row["reward"],
                         f"V2 aggregate reward mismatch in {result_id}")
    _require(receipt.get("target_label_read_count") == 3,
             f"V2 aggregate target-read count mismatch in {result_id}")

    child_refs = receipt.get("validation_receipts")
    _require(isinstance(child_refs, list) and len(child_refs) == 3,
             f"V2 aggregate child receipt count mismatch in {result_id}")
    _require(len(set(map(str, child_refs))) == 3,
             f"V2 aggregate child receipts must be distinct in {result_id}")
    child_metrics: list[float] = []
    child_seeds: list[int] = []
    child_budgets: list[int] = []
    for index, child_ref in enumerate(child_refs):
        child_path = _resolve_receipt_file(
            child_ref, base_dir=receipt_path.parent, project_root=project_root,
            result_id=result_id, label=f"V2 aggregate child receipt {index}",
        )
        child = _load_receipt_json(
            child_path, project_root=project_root, cache=receipt_cache,
            result_id=result_id, label=f"V2 aggregate child receipt {index}",
        )
        child_method = child.get("method_id")
        _require(isinstance(child_method, str),
                 f"V2 aggregate child method is missing in {result_id}")
        if row["method_id"].startswith("stage0_v4_parent_contrastive"):
            _require(child_method.startswith("stage0_v4_parent_contrastive_s")
                     and child_method.endswith("_v2"),
                     f"V2 Stage0 child method mismatch in {result_id}")
        elif row["method_id"].startswith("stage0_v4_clr_fusion"):
            _require(child_method.startswith("stage0_v4_clr_fusion_all_depth_alpha4_s")
                     and child_method.endswith("_v2"),
                     f"V2 fusion child method mismatch in {result_id}")
        else:
            raise ValueError(f"unsupported V2 aggregate method in {result_id}")
        child_seed = child.get("model_seed")
        _require(isinstance(child_seed, int) and child_seed in model_seeds,
                 f"V2 aggregate child seed mismatch in {result_id}")
        child_row = {
            "result_id": f"{result_id}::child{index}",
            "benchmark_id": benchmark["benchmark_id"],
            "method_id": child_method,
            "seed": {
                "model": [child_seed],
                "evaluator": [benchmark["evaluator"]["head_seed"]],
                "aggregation": "single",
            },
            "budget": {
                "stage0_training_steps": 6000,
                "annotation_head_max_steps": 400,
            },
            "primary_metric": benchmark["primary_metric"]["name"],
            "raw_metric": child.get("raw_metric"),
            "reward": child.get("reward"),
        }
        _validate_ann_label_blind_v2_score_receipt(
            benchmark, child_row, child, receipt_path=child_path,
            project_root=project_root, receipt_cache=receipt_cache,
            expected_method_id=child_method, expected_model_seed=child_seed,
            expected_training_budget=6000,
        )
        child_metrics.append(float(child["raw_metric"]))
        child_seeds.append(child_seed)
        child_budgets.append(int(child["training_budget_steps"]))
    _require(child_seeds == model_seeds,
             f"V2 aggregate child seed order mismatch in {result_id}")
    _require(child_budgets == [6000, 6000, 6000],
             f"V2 aggregate child budgets mismatch in {result_id}")
    _require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
                 for a, b in zip(child_metrics, values, strict=True)),
             f"V2 aggregate child metrics mismatch in {result_id}")
    if row["method_id"].startswith("stage0_v4_clr_fusion"):
        _require(receipt.get("fusion_internal_diagnostic") is True,
                 f"V2 fusion aggregate must be internal diagnostic in {result_id}")
        _require(receipt.get("route") == {
            "mode": "all_depth", "alpha": 4.0,
        }, f"V2 fusion route mismatch in {result_id}")
        _require(row["diagnostics"].get("fusion_internal_diagnostic") is True,
                 f"V2 fusion row diagnostic flag is missing in {result_id}")
    else:
        _require(receipt.get("fusion_internal_diagnostic") is False,
                 f"V2 non-fusion aggregate diagnostic flag is invalid in {result_id}")
        _require(row["diagnostics"].get("fusion_internal_diagnostic") is False,
                 f"V2 non-fusion row diagnostic flag is invalid in {result_id}")


def _validate_ann_autoresearch_receipt(
    benchmark: dict[str, Any], row: dict[str, Any], receipt: dict[str, Any], **_: Any
) -> None:
    result_id = row["result_id"]
    metric_name = row["primary_metric"]
    evaluator = benchmark["evaluator"]
    config_binding = receipt.get("config_binding")
    protocol = receipt.get("protocol")
    _require(receipt.get("status") == "complete", f"receipt is incomplete in {result_id}")
    _require(receipt.get("stage") == "stage0_annotation", f"receipt stage mismatch in {result_id}")
    _require(receipt.get("candidate_id") == row["method_id"],
             f"receipt method binding mismatch in {result_id}")
    _require(receipt.get("unit_id")
             and result_id == f"{benchmark['benchmark_id']}__{receipt['unit_id']}",
             f"receipt unit binding mismatch in {result_id}")
    _require(receipt.get("selection_scope") == evaluator.get("selection_scope"),
             f"receipt selection scope mismatch in {result_id}")
    _require(receipt.get("project_native_only") is True,
             f"receipt is not project-native in {result_id}")
    _require_closed_access(receipt, result_id)
    receipt_metrics = receipt.get("metrics")
    _require(isinstance(receipt_metrics, dict), f"receipt metrics are missing in {result_id}")
    _require_same_number(receipt_metrics.get(metric_name), row["raw_metric"],
                         f"receipt metric mismatch in {result_id}")

    seed = row["seed"]
    _require(set(seed) == {"model", "evaluator", "aggregation"},
             f"annotation seed schema mismatch in {result_id}")
    _require(seed["aggregation"] == "single" and len(seed["model"]) == 1,
             f"annotation single-seed aggregation mismatch in {result_id}")
    _require(seed["evaluator"] == [evaluator.get("head_seed")],
             f"annotation evaluator seed mismatch in {result_id}")
    model_seed = seed["model"][0]
    _require(receipt.get("seed") == model_seed, f"receipt model seed mismatch in {result_id}")
    _require(isinstance(config_binding, dict), f"receipt config binding is missing in {result_id}")
    _require(config_binding.get("candidate_id") == row["method_id"],
             f"receipt config method binding mismatch in {result_id}")
    _require(config_binding.get("training_seed") == model_seed,
             f"receipt config seed mismatch in {result_id}")

    budget = row["budget"]
    _require(set(budget) == {"stage0_training_steps", "annotation_head_max_steps"},
             f"annotation budget schema mismatch in {result_id}")
    _require(config_binding.get("maximum_steps") == budget["stage0_training_steps"],
             f"receipt training budget mismatch in {result_id}")
    _require(budget["annotation_head_max_steps"] == evaluator.get("max_steps"),
             f"annotation evaluator budget mismatch in {result_id}")
    _require(isinstance(protocol, dict), f"receipt protocol is missing in {result_id}")
    _require(protocol.get("fixed_head_seed") == evaluator.get("head_seed"),
             f"receipt evaluator seed mismatch in {result_id}")
    _require(protocol.get("fixed_head_budget") == evaluator.get("max_steps"),
             f"receipt evaluator budget mismatch in {result_id}")
    _require(protocol.get("source_id") == benchmark["split"].get("source_id")
             and protocol.get("donor_disjoint") is True,
             f"receipt source isolation mismatch in {result_id}")
    _require(receipt.get("diagnostics", {}).get("stage0_checkpoint_step")
             == budget["stage0_training_steps"],
             f"receipt checkpoint budget mismatch in {result_id}")


def _validate_ann_aggregate_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    **_: Any,
) -> None:
    result_id = row["result_id"]
    evaluator = benchmark["evaluator"]
    metric_name = row["primary_metric"]
    _require(receipt.get("status") == "complete", f"receipt is incomplete in {result_id}")
    _require(receipt.get("benchmark_id") == benchmark["benchmark_id"],
             f"receipt benchmark binding mismatch in {result_id}")
    _require(receipt.get("stage") == "stage0_annotation", f"receipt stage mismatch in {result_id}")
    _require(receipt.get("candidate_id") == row["method_id"],
             f"receipt method binding mismatch in {result_id}")
    _require(receipt.get("primary_metric") == metric_name,
             f"receipt primary metric binding mismatch in {result_id}")
    _require_closed_access(receipt, result_id)

    seed = row["seed"]
    model_seeds = receipt.get("model_seeds")
    _require(set(seed) == {"model", "evaluator", "aggregation"},
             f"annotation aggregate seed schema mismatch in {result_id}")
    _require(seed["aggregation"] == receipt.get("aggregation") == "median",
             f"receipt aggregation mismatch in {result_id}")
    _require(model_seeds == seed["model"] and len(model_seeds) == 3,
             f"receipt model seeds mismatch in {result_id}")
    _require(len(set(model_seeds)) == 3, f"aggregate model seeds must be unique in {result_id}")
    _require(receipt.get("evaluator_seeds") == seed["evaluator"]
             == [evaluator.get("head_seed")],
             f"receipt evaluator seeds mismatch in {result_id}")

    budget = row["budget"]
    _require(set(budget) == {
        "stage0_training_steps", "annotation_head_max_steps", "seed_count"
    }, f"annotation aggregate budget schema mismatch in {result_id}")
    _require(budget["seed_count"] == len(model_seeds),
             f"aggregate seed_count mismatch in {result_id}")
    _require(budget["annotation_head_max_steps"] == evaluator.get("max_steps"),
             f"annotation evaluator budget mismatch in {result_id}")
    _require(receipt.get("budgets") == [budget["stage0_training_steps"]] * len(model_seeds),
             f"receipt training budgets mismatch in {result_id}")

    raw_values = receipt.get("raw_metric_values")
    _require(isinstance(raw_values, list) and len(raw_values) == len(model_seeds)
             and all(_finite_number(value) for value in raw_values),
             f"receipt per-seed metrics are invalid in {result_id}")
    aggregate_value = median(float(value) for value in raw_values)
    _require_same_number(receipt.get("raw_metric_median"), aggregate_value,
                         f"receipt median metric is invalid in {result_id}")
    _require_same_number(row["raw_metric"], aggregate_value,
                         f"receipt aggregate metric mismatch in {result_id}")

    child_paths = receipt.get("validation_evidence")
    _require(isinstance(child_paths, list) and len(child_paths) == len(model_seeds),
             f"aggregate child receipt count mismatch in {result_id}")
    for index, (child_path, model_seed, raw_value) in enumerate(
        zip(child_paths, model_seeds, raw_values, strict=True)
    ):
        child = _load_receipt_json(
            child_path, project_root=project_root, cache=receipt_cache,
            result_id=result_id, label=f"aggregate child receipt {index}",
        )
        child_config = child.get("config_binding")
        child_protocol = child.get("protocol")
        _require(child.get("format") == "brainwm_autoresearch_validation_v1"
                 and child.get("status") == "complete",
                 f"aggregate child receipt is invalid in {result_id}")
        _require(child.get("candidate_id") == row["method_id"]
                 and child.get("seed") == model_seed,
                 f"aggregate child binding mismatch in {result_id}")
        _require(isinstance(child_config, dict)
                 and child_config.get("training_seed") == model_seed
                 and child_config.get("maximum_steps") == budget["stage0_training_steps"],
                 f"aggregate child seed/budget mismatch in {result_id}")
        _require_same_number(child.get("metrics", {}).get(metric_name), raw_value,
                             f"aggregate child metric mismatch in {result_id}")
        _require(isinstance(child_protocol, dict)
                 and child_protocol.get("fixed_head_seed") == evaluator.get("head_seed")
                 and child_protocol.get("fixed_head_budget") == evaluator.get("max_steps"),
                 f"aggregate child evaluator contract mismatch in {result_id}")


def _validate_ann_baseline_receipt(
    benchmark: dict[str, Any], row: dict[str, Any], receipt: dict[str, Any], **_: Any
) -> None:
    result_id = row["result_id"]
    evaluator = benchmark["evaluator"]
    _require(receipt.get("status") == "complete", f"receipt is incomplete in {result_id}")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("method_id", row["method_id"]), ("method_class", "baseline"),
        ("split_id", row["split_id"]), ("evaluator_id", row["evaluator_id"]),
        ("primary_metric", row["primary_metric"]), ("direction", row["direction"]),
    ):
        _require(receipt.get(field) == expected, f"receipt {field} binding mismatch in {result_id}")
    _require(receipt.get("task_id") in (None, row["task_id"]),
             f"receipt task_id binding mismatch in {result_id}")
    _require(receipt.get("training_stage") == benchmark["training_stage"],
             f"receipt training stage mismatch in {result_id}")
    receipt_axis = receipt.get("axis")
    _require(isinstance(receipt_axis, dict)
             and all(receipt_axis.get(key) == row["axis"].get(key)
                     for key in ("architecture", "algorithm", "variation")),
             f"receipt axis binding mismatch in {result_id}")
    _require(receipt.get("seed") == row["seed"], f"receipt seed binding mismatch in {result_id}")
    _require(receipt.get("budget") == row["budget"], f"receipt budget binding mismatch in {result_id}")
    _require_same_number(receipt.get("raw_metric"), row["raw_metric"],
                         f"receipt metric mismatch in {result_id}")
    _require_same_number(receipt.get("reward"), row["reward"],
                         f"receipt reward mismatch in {result_id}")
    _require_closed_access(receipt, result_id)
    _require(row["seed"].get("model", []) == []
             and row["seed"].get("evaluator") == [evaluator.get("head_seed")]
             and row["seed"].get("aggregation") == "single",
             f"annotation baseline seed contract mismatch in {result_id}")
    budget = row["budget"]
    if row["method_id"] in {"ann_raw_counts_head", "ann_clr_head"}:
        head_contract = receipt.get("protocol", {}).get("head_contract", {})
        _require(head_contract.get("seed") == evaluator.get("head_seed")
                 and head_contract.get("max_steps") == evaluator.get("max_steps"),
                 f"annotation baseline evaluator budget mismatch in {result_id}")
        if "annotation_head_max_steps" in budget:
            _require(set(budget) == {
                "annotation_head_max_steps", "annotation_head_completed_steps", "projection_fit_rows"
            }, f"annotation baseline budget schema mismatch in {result_id}")
            _require(budget["annotation_head_max_steps"] == evaluator.get("max_steps")
                     and budget["annotation_head_completed_steps"] <= evaluator.get("max_steps"),
                     f"annotation baseline evaluator budget mismatch in {result_id}")
            _require(budget["projection_fit_rows"] == benchmark["split"].get("train_rows"),
                     f"annotation baseline fit-row budget mismatch in {result_id}")
        else:
            _require(budget == {"training_steps": 0},
                     f"annotation baseline budget schema mismatch in {result_id}")
    else:
        _require(row["method_id"] == "ann_source_only_mnb"
                 and budget.get("training_steps") == 0
                 and set(budget) <= {"training_steps", "source_cv_folds"}
                 and budget.get("source_cv_folds", 3) == 3,
                 f"annotation MNB budget mismatch in {result_id}")


def _validate_ann_fusion_aggregate_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    **_: Any,
) -> None:
    result_id = row["result_id"]
    evaluator = benchmark["evaluator"]
    _require(receipt.get("status") == "complete", f"receipt is incomplete in {result_id}")
    _require(receipt.get("benchmark_id") == benchmark["benchmark_id"],
             f"receipt benchmark binding mismatch in {result_id}")
    _require(receipt.get("method_id") == row["method_id"],
             f"receipt method binding mismatch in {result_id}")
    _require(receipt.get("primary_metric") == row["primary_metric"],
             f"receipt primary metric binding mismatch in {result_id}")
    _require_closed_access(receipt, result_id)
    _require(receipt.get("stage0_frozen") is True
             and receipt.get("source_train_only_statistics") is True,
             f"fusion isolation receipt mismatch in {result_id}")

    model_seeds = receipt.get("model_seeds")
    seed = row["seed"]
    _require(set(seed) == {"model", "evaluator", "aggregation"},
             f"fusion aggregate seed schema mismatch in {result_id}")
    _require(seed.get("aggregation") == receipt.get("aggregation") == "median",
             f"receipt aggregation mismatch in {result_id}")
    _require(model_seeds == seed.get("model") and isinstance(model_seeds, list)
             and len(model_seeds) == 3 and len(set(model_seeds)) == 3,
             f"receipt model seeds mismatch in {result_id}")
    _require(seed.get("evaluator") == [evaluator.get("head_seed")]
             and receipt.get("evaluator_head_seed") == evaluator.get("head_seed"),
             f"receipt evaluator seed mismatch in {result_id}")

    values = receipt.get("primary_metric_values")
    _require(isinstance(values, list) and len(values) == len(model_seeds)
             and all(_finite_number(value) for value in values),
             f"receipt per-seed metrics are invalid in {result_id}")
    aggregate_value = median(float(value) for value in values)
    _require_same_number(receipt.get("primary_metric_median"), aggregate_value,
                         f"receipt median metric is invalid in {result_id}")
    _require_same_number(row["raw_metric"], aggregate_value,
                         f"receipt aggregate metric mismatch in {result_id}")
    _require(row["diagnostics"].get("per_seed_rewards") == values,
             f"receipt per-seed diagnostics mismatch in {result_id}")

    budget = row["budget"]
    _require(set(budget) == {
        "stage0_training_steps", "annotation_head_max_steps", "seed_count"
    }, f"fusion aggregate budget schema mismatch in {result_id}")
    _require(budget["seed_count"] == len(model_seeds),
             f"fusion aggregate seed_count mismatch in {result_id}")
    _require(budget["stage0_training_steps"] == receipt.get("stage0_training_steps"),
             f"fusion training budget mismatch in {result_id}")
    _require(budget["annotation_head_max_steps"] == evaluator.get("max_steps"),
             f"fusion evaluator budget mismatch in {result_id}")

    route = receipt.get("route")
    _require(route == {"mode": "all_depth", "alpha": 4.0},
             f"fusion route mismatch in {result_id}")
    child_refs = receipt.get("validation_receipts")
    checkpoint_refs = receipt.get("stage0_checkpoints")
    _require(isinstance(child_refs, list) and len(child_refs) == 3,
             f"fusion child receipt list mismatch in {result_id}")
    _require(isinstance(checkpoint_refs, list) and len(checkpoint_refs) == 3,
             f"fusion checkpoint list mismatch in {result_id}")

    child_seeds: list[int] = []
    child_metrics: list[float] = []
    child_checkpoints: list[str] = []
    child_clr_values: list[float] = []
    for index, child_ref in enumerate(child_refs):
        child = _load_receipt_json(
            child_ref,
            project_root=project_root,
            cache=receipt_cache,
            result_id=result_id,
            label=f"fusion child receipt {index}",
        )
        _require(
            child.get("format") == "brainwm_ann_s64_stage0_clr_skip_fusion_v1"
            and child.get("status") == "complete",
            f"fusion child receipt format/status mismatch in {result_id}",
        )
        _require(
            child.get("benchmark_id") == benchmark["benchmark_id"]
            and child.get("split_id") == benchmark["split"]["split_id"]
            and child.get("evaluator_id") == evaluator["evaluator_id"],
            f"fusion child benchmark binding mismatch in {result_id}",
        )
        child_route = child.get("fusion", {})
        best = child.get("best_candidate", {})
        _require(
            child_route.get("modes") == ["all_depth"]
            and child_route.get("alphas") == [4.0]
            and best.get("key") == "all_depth_alpha_4"
            and best.get("mode") == "all_depth"
            and best.get("alpha") == 4.0,
            f"fusion child route mismatch in {result_id}",
        )
        isolation = child.get("isolation", {})
        _require(
            isolation.get("stage0_frozen") is True
            and isolation.get("clr_statistics_fit_on_source_train_only") is True
            and isolation.get("fusion_pca_fit_on_source_train_only") is True
            and isolation.get("head_training_labels_source_only") is True
            and isolation.get("query_accessed") is False
            and isolation.get("test_accessed") is False
            and isolation.get("official_table_updated") is False
            and isolation.get("autoresearch_ledger_updated") is False,
            f"fusion child isolation mismatch in {result_id}",
        )
        arguments = child.get("arguments", {})
        _require(
            arguments.get("head_seed") == evaluator.get("head_seed")
            and arguments.get("head_steps") == evaluator.get("max_steps"),
            f"fusion child evaluator mismatch in {result_id}",
        )
        stage0 = child.get("stage0", {})
        _require(stage0.get("global_step") == budget["stage0_training_steps"],
                 f"fusion child training budget mismatch in {result_id}")
        checkpoint_value = stage0.get("checkpoint")
        checkpoint = _resolve_project_file(
            checkpoint_value,
            project_root=project_root,
            result_id=result_id,
            label=f"fusion child checkpoint {index}",
        )
        _require(checkpoint.is_file(), f"missing fusion child checkpoint in {result_id}")
        resolved_config = _load_receipt_json(
            checkpoint.parent / "config.resolved.json",
            project_root=project_root,
            cache=receipt_cache,
            result_id=result_id,
            label=f"fusion child resolved config {index}",
        )
        model_seed = resolved_config.get("training", {}).get("seed")
        _require(isinstance(model_seed, int) and not isinstance(model_seed, bool),
                 f"fusion child model seed mismatch in {result_id}")
        metric_value = best.get("metrics", {}).get(row["primary_metric"])
        _require(_finite_number(metric_value),
                 f"fusion child metric mismatch in {result_id}")
        clr_value = child.get("transparent_baselines", {}).get("ann_clr_head", {}).get(
            row["primary_metric"]
        )
        _require(_finite_number(clr_value),
                 f"fusion child CLR baseline mismatch in {result_id}")
        child_seeds.append(model_seed)
        child_metrics.append(float(metric_value))
        child_clr_values.append(float(clr_value))
        child_checkpoints.append(
            checkpoint.resolve().relative_to(project_root.resolve()).as_posix()
        )

    _require(child_seeds == model_seeds,
             f"fusion child model seeds mismatch in {result_id}")
    _require(child_checkpoints == checkpoint_refs,
             f"fusion child checkpoints mismatch in {result_id}")
    _require(
        all(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(child_metrics, values, strict=True)),
        f"fusion child metrics mismatch in {result_id}",
    )
    _require(
        all(math.isclose(value, child_clr_values[0], rel_tol=0.0, abs_tol=1e-12)
            for value in child_clr_values),
        f"fusion child CLR baselines mismatch in {result_id}",
    )
    _require_same_number(receipt.get("clr_baseline"), child_clr_values[0],
                         f"fusion CLR baseline mismatch in {result_id}")
    _require_same_number(
        receipt.get("delta_vs_clr"), aggregate_value - child_clr_values[0],
        f"fusion CLR delta mismatch in {result_id}",
    )


def _validate_integration_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    **_: Any,
) -> None:
    result_id = row["result_id"]
    evaluator = benchmark["evaluator"]
    expected_receipt_methods = {
        "int_fresh_raw_pca": "raw_pca",
        "int_fresh_harmony": "harmony",
        "int_fresh_matched_raw_cluster": "matched_raw_cluster_ablation",
        "int_fresh_brainwm_raw_preserving": "brainwm_raw_preserving",
    }
    _require(receipt.get("status") == "complete" and receipt.get("rankable") is True,
             f"receipt is not rankable and complete in {result_id}")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]),
        ("split_id", row["split_id"]),
        ("evaluator_id", row["evaluator_id"]),
        ("primary_metric", row["primary_metric"]),
        ("direction", row["direction"]),
        ("backend", evaluator.get("backend")),
        ("method_id", expected_receipt_methods.get(row["method_id"])),
    ):
        _require(receipt.get(field) == expected, f"receipt {field} binding mismatch in {result_id}")
    metrics = receipt.get("metrics")
    _require(isinstance(metrics, dict), f"receipt metrics are missing in {result_id}")
    _require_same_number(metrics.get(row["primary_metric"]), row["raw_metric"],
                         f"receipt metric mismatch in {result_id}")
    _require_same_number(receipt.get("raw_metric"), row["raw_metric"],
                         f"receipt raw metric mismatch in {result_id}")
    _require_same_number(receipt.get("reward"), row["reward"],
                         f"receipt reward mismatch in {result_id}")
    _require_same_number(
        metrics.get("scib_total_project_v1"),
        0.6 * float(metrics.get("biology", math.nan)) + 0.4 * float(metrics.get("batch", math.nan)),
        f"receipt integration formula mismatch in {result_id}",
    )

    split_seed = benchmark["split"].get("seed")
    seed = row["seed"]
    _require(set(seed) == {"model", "evaluator", "aggregation"}
             and len(seed["model"]) == 1 and seed["aggregation"] == "single",
             f"integration seed schema mismatch in {result_id}")
    _require(seed["evaluator"] == [split_seed] and receipt.get("seed") == split_seed,
             f"integration evaluator seed mismatch in {result_id}")

    protocol = _load_receipt_json(
        receipt.get("protocol_path", ""), project_root=project_root, cache=receipt_cache,
        result_id=result_id, label="integration protocol receipt",
    )
    _require(protocol.get("format") == "brainwm_integration_fresh_protocol_v1"
             and protocol.get("status") == "frozen_labels_closed",
             f"integration protocol receipt is invalid in {result_id}")
    for field, expected in (
        ("benchmark_id", benchmark["benchmark_id"]), ("split_id", row["split_id"]),
        ("evaluator_id", row["evaluator_id"]), ("selection_seed", split_seed),
    ):
        _require(protocol.get(field) == expected,
                 f"integration protocol {field} mismatch in {result_id}")
    protocol_evaluator = protocol.get("evaluator")
    _require(isinstance(protocol_evaluator, dict),
             f"integration evaluator protocol is missing in {result_id}")
    for field, expected in (
        ("backend", evaluator.get("backend")), ("primary_metric", row["primary_metric"]),
        ("direction", row["direction"]), ("formula", evaluator.get("formula")),
        ("seed", split_seed), ("label_support_policy_id", evaluator.get("label_support_policy_id")),
        ("support_threshold", evaluator.get("label_support_threshold")),
    ):
        _require(protocol_evaluator.get(field) == expected,
                 f"integration evaluator protocol {field} mismatch in {result_id}")
    _require(receipt.get("support", {}).get("policy_id") == evaluator.get("label_support_policy_id")
             and receipt.get("support", {}).get("ready") is True,
             f"integration label-support receipt mismatch in {result_id}")

    budget = row["budget"]
    _require(set(budget) == {
        "representation_training_steps", "source_selection_rows", "representation_rows",
        "latent_dimension", "evaluation_rows",
    }, f"integration budget schema mismatch in {result_id}")
    rows_by_role = protocol.get("rows_by_role")
    _require(isinstance(rows_by_role, dict)
             and all(isinstance(value, int) and value > 0 for value in rows_by_role.values()),
             f"integration representation rows are invalid in {result_id}")
    _require(budget["representation_training_steps"] == 0,
             f"integration representation training budget mismatch in {result_id}")
    _require(budget["source_selection_rows"] == protocol.get("algorithm", {}).get("selection_rows"),
             f"integration source-selection budget mismatch in {result_id}")
    _require(budget["representation_rows"] == sum(rows_by_role.values()),
             f"integration representation-row budget mismatch in {result_id}")
    _require(budget["latent_dimension"] == receipt.get("latent_dim") == 256,
             f"integration latent budget mismatch in {result_id}")
    _require(budget["evaluation_rows"] == receipt.get("rows")
             == receipt.get("subsample") == protocol_evaluator.get("subsample_rows"),
             f"integration evaluator row budget mismatch in {result_id}")


def _validate_development_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
    **_: Any,
) -> None:
    result_id = row["result_id"]
    split = benchmark["split"]
    metric_name = row["primary_metric"]
    control_metric_keys = {
        "dev23_identity": "identity", "dev23_source_mean": "source_mean",
        "dev23_source_linear_mean": "source_linear_mean", "dev23_source_only_ot": "source_only_ot",
    }
    learned_arms = {
        "dev23_scfit_target_endpoint": "target_endpoint",
        "dev23_scfit_direct_velocity": "direct_velocity",
    }
    _require(receipt.get("format") == "brainwm_stage1_dev_source23_v1"
             and receipt.get("status") == "complete"
             and receipt.get("phase") == "single_aggregate_query_score",
             f"development receipt is invalid in {result_id}")
    query_access = receipt.get("query_access")
    _require(isinstance(query_access, dict) and query_access.get("opened") is True
             and query_access.get("descriptor_indices") == split.get("query_descriptors")
             and query_access.get("target_rows_read") == split.get("query_target_rows")
             and query_access.get("target_labels_read") is False
             and query_access.get("updates_after_reveal") is False,
             f"development query receipt mismatch in {result_id}")
    frozen = receipt.get("frozen_prediction_manifest")
    _require(isinstance(frozen, dict) and frozen.get("status") == "complete"
             and frozen.get("predictions_constructed_before_reveal") is True
             and frozen.get("query_access") == {"opened": False},
             f"development frozen-prediction receipt mismatch in {result_id}")

    contract = _load_receipt_json(
        receipt_path.with_name("contract.json"), project_root=project_root, cache=receipt_cache,
        result_id=result_id, label="development run contract",
    )
    _require(contract.get("format") == "brainwm_stage1_dev_source23_v1"
             and contract.get("metadata_only") is True
             and contract.get("query_expression_opened") is False,
             f"development run contract is invalid in {result_id}")
    _require(contract.get("source_id") == split.get("source_id")
             and contract.get("assay_type") == split.get("assay_type")
             and contract.get("uberon_id") == split.get("region")
             and contract.get("data_seed") == split.get("data_seed"),
             f"development data contract mismatch in {result_id}")
    _require(contract.get("query", {}).get("descriptor_indices") == split.get("query_descriptors")
             and contract.get("query", {}).get("cells_per_side") == split.get("cells_per_side"),
             f"development query split mismatch in {result_id}")

    metrics = receipt.get("metrics")
    _require(isinstance(metrics, dict), f"development metrics are missing in {result_id}")
    data_seed = split.get("data_seed")
    if row["method_id"] in control_metric_keys:
        _require(row["seed"] == {"data": [data_seed], "aggregation": "deterministic"},
                 f"development deterministic seed contract mismatch in {result_id}")
        _require(row["budget"] == {"training_steps": 0},
                 f"development deterministic budget mismatch in {result_id}")
        value = metrics.get(control_metric_keys[row["method_id"]], {}).get(metric_name)
        _require_same_number(value, row["raw_metric"],
                             f"development receipt metric mismatch in {result_id}")
        return

    _require(row["method_id"] in learned_arms,
             f"unsupported development receipt method in {result_id}")
    model_seeds = contract.get("model_seeds")
    _require(isinstance(model_seeds, list) and len(model_seeds) == 3
             and len(set(model_seeds)) == 3,
             f"development run model seeds are invalid in {result_id}")
    _require(row["seed"] == {
        "data": [data_seed], "model": model_seeds, "aggregation": "median"
    }, f"development learned seed/aggregation mismatch in {result_id}")
    budget = row["budget"]
    _require(set(budget) == {"maximum_steps_per_seed", "selected_checkpoint_steps"},
             f"development learned budget schema mismatch in {result_id}")
    maximum_steps = contract.get("algorithm", {}).get("steps_maximum")
    _require(isinstance(maximum_steps, int) and maximum_steps > 0,
             f"development maximum-step contract is invalid in {result_id}")
    _require(budget["maximum_steps_per_seed"] == maximum_steps,
             f"development maximum-step budget mismatch in {result_id}")
    arm = learned_arms[row["method_id"]]
    checkpoint_metadata = receipt.get("checkpoint_metadata_loaded_before_reveal")
    _require(isinstance(checkpoint_metadata, dict),
             f"development checkpoint receipt is missing in {result_id}")
    selected_steps = [checkpoint_metadata.get(str(seed), {}).get(arm, {}).get("best_step")
                      for seed in model_seeds]
    _require(budget["selected_checkpoint_steps"] == selected_steps
             and all(isinstance(step, int) and 0 <= step <= maximum_steps for step in selected_steps),
             f"development selected-checkpoint budget mismatch in {result_id}")
    per_seed_values = [metrics.get(f"seed_{seed}_{arm}", {}).get(metric_name)
                       for seed in model_seeds]
    _require(all(_finite_number(value) for value in per_seed_values),
             f"development per-seed metrics are invalid in {result_id}")
    _require_same_number(median(float(value) for value in per_seed_values), row["raw_metric"],
                         f"development median metric mismatch in {result_id}")
    _require(row["diagnostics"].get("per_seed_population_mse") == per_seed_values,
             f"development per-seed diagnostics mismatch in {result_id}")


def _validate_mouse_stage2_mcr_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    project_root: Path,
    **_: Any,
) -> None:
    """Bind the Stage2 table rows to one sealed train-internal MCR receipt."""

    result_id = row["result_id"]
    split = benchmark["split"]
    evaluator = benchmark["evaluator"]
    _require(
        receipt.get("format") == MOUSE_STAGE2_SELECTION_FORMAT
        and receipt.get("contract_id")
        == evaluator.get("source_experiment_contract_id")
        == benchmark.get("source_experiment_contract_id")
        and receipt.get("status") == "train_internal_gate_failed",
        f"mouse Stage2 receipt status/contract mismatch in {result_id}",
    )
    data = receipt.get("data")
    _require(isinstance(data, dict), f"mouse Stage2 data receipt is missing in {result_id}")
    for field, expected in (
        ("split", "train"),
        ("species", split.get("species")),
        ("assay_type", split.get("assay_type")),
        ("section_keys", split.get("section_keys")),
        ("sections", split.get("sections")),
        ("descriptors", split.get("descriptors")),
        ("masked_cells", split.get("masked_cells")),
        ("cell_type_or_outcome_labels_opened", False),
        ("validation_opened", False),
        ("test_opened", False),
    ):
        _require(
            data.get(field) == expected,
            f"mouse Stage2 data {field} mismatch in {result_id}",
        )

    architecture = receipt.get("architecture")
    _require(
        isinstance(architecture, dict)
        and architecture.get("brainwm_frozen") is True
        and architecture.get("trainable_parameters") == 0
        and architecture.get("selected") is None,
        f"mouse Stage2 frozen architecture mismatch in {result_id}",
    )
    brainwm = architecture.get("brainwm")
    _require(
        isinstance(brainwm, dict)
        and brainwm.get("global_step") == evaluator.get("brainwm_checkpoint_step")
        and brainwm.get("mutated") is False,
        f"mouse Stage2 checkpoint receipt mismatch in {result_id}",
    )
    checkpoint = _resolve_project_file(
        evaluator.get("brainwm_checkpoint", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Stage2 frozen checkpoint",
    )
    _require(
        checkpoint.is_file()
        and Path(str(brainwm.get("path", ""))).resolve() == checkpoint,
        f"mouse Stage2 checkpoint path mismatch in {result_id}",
    )

    algorithm = receipt.get("algorithm")
    _require(
        isinstance(algorithm, dict)
        and algorithm.get("primary_metric") == row["primary_metric"]
        and algorithm.get("direction") == row["direction"]
        and algorithm.get("controls") == evaluator.get("required_controls"),
        f"mouse Stage2 evaluator binding mismatch in {result_id}",
    )
    decision = receipt.get("decision")
    _require(
        isinstance(decision, dict)
        and decision.get("split") == "train"
        and decision.get("primary_metric") == row["primary_metric"]
        and decision.get("passing_candidates") == 0
        and decision.get("selected_candidate") is None
        and decision.get("promotion_gate") is False
        and decision.get("validation_access_granted") is False
        and decision.get("test_access_granted") is False,
        f"mouse Stage2 gate decision mismatch in {result_id}",
    )

    metrics = receipt.get("metrics")
    _require(isinstance(metrics, dict), f"mouse Stage2 metrics are missing in {result_id}")
    candidates = metrics.get("top_candidates")
    _require(
        isinstance(candidates, list) and candidates,
        f"mouse Stage2 candidate metrics are missing in {result_id}",
    )
    candidate = candidates[0]
    neighbor = metrics.get("strongest_neighbor_baseline")
    global_prior = metrics.get("strongest_global_baseline")
    _require(
        isinstance(candidate, dict)
        and candidate.get("key")
        == "hierarchical|k=32|pseudo=25|global=0.95|local=0.4"
        and candidate.get("full_gate") is False,
        f"mouse Stage2 candidate selection mismatch in {result_id}",
    )
    _require(
        isinstance(neighbor, dict)
        and neighbor.get("key") == "neighbor_only|k=64|pseudo=400",
        f"mouse Stage2 neighbor selection mismatch in {result_id}",
    )
    _require(
        isinstance(global_prior, dict)
        and global_prior.get("key") == "global_prior|weight=1",
        f"mouse Stage2 global selection mismatch in {result_id}",
    )
    control_values = {
        "candidate": candidate.get("real"),
        "neighbor": neighbor.get("real"),
        "permutation": candidate.get("permutation"),
        "zero": candidate.get("zero"),
        "complete_global": global_prior.get("loss"),
        "matched_global": candidate.get("matched_global"),
        "no_structure": metrics.get("no_structure"),
    }
    _require(
        all(_finite_number(value) for value in control_values.values()),
        f"mouse Stage2 control metrics are invalid in {result_id}",
    )
    candidate_nll = float(control_values["candidate"])
    _require(
        candidate_nll > float(control_values["neighbor"])
        and all(
            candidate_nll < float(control_values[name])
            for name in (
                "permutation",
                "zero",
                "complete_global",
                "matched_global",
                "no_structure",
            )
        ),
        f"mouse Stage2 negative gate metrics are inconsistent in {result_id}",
    )
    method_metrics = {
        "stg2_mcr_brainwm_hierarchical": candidate.get("real"),
        "stg2_mcr_neighbor_only": neighbor.get("real"),
    }
    _require(
        row["method_id"] in method_metrics,
        f"unsupported mouse Stage2 method in {result_id}",
    )
    _require_same_number(
        method_metrics[row["method_id"]],
        row["raw_metric"],
        f"mouse Stage2 receipt metric mismatch in {result_id}",
    )
    budget = row["budget"]
    all_routes = metrics.get("all_routes")
    _require(
        isinstance(all_routes, dict)
        and budget
        == {
            "sections": data["sections"],
            "descriptors": data["descriptors"],
            "masked_cells": data["masked_cells"],
            "evaluated_routes": len(all_routes),
        },
        f"mouse Stage2 budget mismatch in {result_id}",
    )
    _require(
        row["seed"] == {"mask": [split["mask_seed"]], "aggregation": "single"},
        f"mouse Stage2 seed mismatch in {result_id}",
    )


def _validate_mouse_spatial_annotation_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
) -> None:
    """Bind five matched mouse annotation rows to frozen validation evidence."""

    result_id = row["result_id"]
    split = benchmark["split"]
    evaluator = benchmark["evaluator"]
    expected_methods = {
        "raw_cell": "baseline",
        "raw_neighbor": "baseline",
        "stage0_cell": "foundation",
        "stage0_neighbor": "ablation",
        "stage0_su07": "candidate",
    }
    _require(
        receipt.get("format") == MOUSE_SPATIAL_ANNOTATION_VALIDATION_FORMAT
        and receipt.get("contract_id")
        == benchmark.get("source_experiment_contract_id")
        and receipt.get("status") == "complete"
        and receipt.get("primary_metric") == row["primary_metric"]
        and receipt.get("direction") == row["direction"],
        f"mouse spatial annotation receipt status/contract mismatch in {result_id}",
    )
    methods = receipt.get("methods")
    _require(
        isinstance(methods, dict) and set(methods) == set(expected_methods),
        f"mouse spatial annotation method inventory mismatch in {result_id}",
    )
    expected_seeds = evaluator.get("head_seeds")
    maximum_steps = evaluator.get("head_max_steps")
    trainable_parameters = evaluator.get("head_trainable_parameters")
    _require(
        isinstance(expected_seeds, list)
        and len(expected_seeds) == 3
        and all(isinstance(seed, int) for seed in expected_seeds)
        and isinstance(maximum_steps, int)
        and maximum_steps > 0
        and isinstance(trainable_parameters, int)
        and trainable_parameters > 0,
        f"mouse spatial annotation evaluator budget is invalid in {result_id}",
    )

    run_root = receipt_path.parent.parent
    summaries: dict[str, dict[str, Any]] = {}
    for method_id in expected_methods:
        summary = methods.get(method_id)
        _require(
            isinstance(summary, dict),
            f"mouse spatial annotation summary is missing for {method_id} in {result_id}",
        )
        rewards = summary.get("validation_rewards")
        best_steps = summary.get("best_steps")
        _require(
            isinstance(rewards, list)
            and len(rewards) == len(expected_seeds)
            and all(_finite_number(value) for value in rewards),
            f"mouse spatial annotation seed metrics mismatch for {method_id} in {result_id}",
        )
        _require(
            isinstance(best_steps, list)
            and len(best_steps) == len(expected_seeds)
            and all(
                isinstance(step, int) and 0 <= step <= maximum_steps
                for step in best_steps
            ),
            f"mouse spatial annotation checkpoint steps mismatch for {method_id} in {result_id}",
        )
        _require_same_number(
            median(float(value) for value in rewards),
            summary.get("median_validation_reward"),
            f"mouse spatial annotation median mismatch for {method_id} in {result_id}",
        )
        reward_range = summary.get("reward_range")
        _require(
            isinstance(reward_range, list)
            and len(reward_range) == 2,
            f"mouse spatial annotation reward range mismatch for {method_id} in {result_id}",
        )
        _require_same_number(
            min(float(value) for value in rewards),
            reward_range[0],
            f"mouse spatial annotation reward minimum mismatch for {method_id} in {result_id}",
        )
        _require_same_number(
            max(float(value) for value in rewards),
            reward_range[1],
            f"mouse spatial annotation reward maximum mismatch for {method_id} in {result_id}",
        )
        _require_same_number(
            median(best_steps),
            summary.get("median_best_step"),
            f"mouse spatial annotation checkpoint median mismatch for {method_id} in {result_id}",
        )
        _require(
            summary.get("trainable_parameters") == trainable_parameters
            and all(
                _finite_number(summary.get(field))
                for field in (
                    "median_initial_train_loss",
                    "median_selected_train_loss",
                    "median_selected_validation_loss",
                    "median_gradient_clip_fraction",
                )
            ),
            f"mouse spatial annotation head diagnostics mismatch for {method_id} in {result_id}",
        )
        for seed, reward, step in zip(expected_seeds, rewards, best_steps, strict=True):
            head_root = run_root / "heads" / method_id / f"seed_{seed}"
            child = _load_receipt_json(
                head_root / "result.json",
                project_root=project_root,
                cache=receipt_cache,
                result_id=result_id,
                label="mouse spatial annotation head receipt",
            )
            _require(
                child.get("format")
                == "brainwm_mouse_spatial_annotation_head_result_v1"
                and child.get("contract_id") == receipt.get("contract_id")
                and child.get("status") == "validation_complete"
                and child.get("representation") == method_id
                and child.get("seed") == seed
                and child.get("primary_metric") == row["primary_metric"]
                and child.get("direction") == row["direction"]
                and child.get("best_step") == step
                and child.get("test_access")
                == {"expression_opened": False, "labels_opened": False},
                f"mouse spatial annotation head receipt mismatch for {method_id}/{seed} in {result_id}",
            )
            _require_same_number(
                child.get("reward"),
                reward,
                f"mouse spatial annotation child metric mismatch for {method_id}/{seed} in {result_id}",
            )
            architecture = child.get("architecture")
            gradients = child.get("gradient_norms")
            _require(
                isinstance(architecture, dict)
                and architecture.get("trainable_parameters") == trainable_parameters
                and isinstance(gradients, dict)
                and gradients.get("steps") == maximum_steps
                and (head_root / "head_best.pt").is_file(),
                f"mouse spatial annotation frozen head mismatch for {method_id}/{seed} in {result_id}",
            )
        summaries[method_id] = summary

    candidate = float(summaries["stage0_su07"]["median_validation_reward"])
    raw_values = {
        method_id: float(summaries[method_id]["median_validation_reward"])
        for method_id in ("raw_cell", "raw_neighbor")
    }
    strongest_raw_id = max(raw_values, key=raw_values.get)
    strongest_raw = raw_values[strongest_raw_id]
    stage0_neighbor = float(
        summaries["stage0_neighbor"]["median_validation_reward"]
    )
    expected_conditions = {
        "stage0_su07_strictly_beats_best_raw_baseline": candidate > strongest_raw,
        "stage0_su07_strictly_beats_stage0_neighbor": candidate > stage0_neighbor,
    }
    _require(
        receipt.get("candidate") == evaluator.get("candidate") == "stage0_su07"
        and receipt.get("strongest_raw_baseline")
        == evaluator.get("strongest_raw_baseline")
        == strongest_raw_id
        and receipt.get("gate_conditions") == expected_conditions
        and receipt.get("gate_passed") is False
        and not all(expected_conditions.values())
        and receipt.get("test_expression_opened") is False
        and receipt.get("test_labels_opened") is False
        and receipt.get("test_label_reads") == 0,
        f"mouse spatial annotation validation gate mismatch in {result_id}",
    )

    gate_path = receipt_path.with_name("validation_gate.json")
    gate = _load_receipt_json(
        gate_path,
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse spatial annotation validation gate",
    )
    _require(
        gate.get("format") == MOUSE_SPATIAL_ANNOTATION_GATE_FORMAT
        and gate.get("contract_id") == receipt.get("contract_id")
        and gate.get("status") == "validation_gate_failed"
        and gate.get("primary_metric") == row["primary_metric"]
        and gate.get("conditions") == expected_conditions
        and gate.get("frozen_head_checkpoints") == 15
        and gate.get("expected_head_checkpoints") == 15
        and gate.get("test_expression_access_granted") is False
        and gate.get("test_label_access_granted_after_predictions") is False
        and gate.get("test_label_reads") == 0
        and gate.get("post_test_updates_allowed") is False,
        f"mouse spatial annotation gate receipt mismatch in {result_id}",
    )
    gate_candidate = gate.get("candidate")
    gate_raw = gate.get("strongest_raw_baseline")
    gate_neighbor = gate.get("stage0_neighbor")
    _require(
        isinstance(gate_candidate, dict)
        and gate_candidate.get("method") == "stage0_su07"
        and isinstance(gate_raw, dict)
        and gate_raw.get("method") == strongest_raw_id
        and isinstance(gate_neighbor, dict),
        f"mouse spatial annotation gate methods mismatch in {result_id}",
    )
    _require_same_number(
        gate_candidate.get("median"),
        candidate,
        f"mouse spatial annotation gate candidate metric mismatch in {result_id}",
    )
    _require_same_number(
        gate_raw.get("median"),
        strongest_raw,
        f"mouse spatial annotation gate raw metric mismatch in {result_id}",
    )
    _require_same_number(
        gate_neighbor.get("median"),
        stage0_neighbor,
        f"mouse spatial annotation gate neighbor metric mismatch in {result_id}",
    )

    frozen_heads = gate.get("frozen_heads")
    _require(
        isinstance(frozen_heads, list) and len(frozen_heads) == 15,
        f"mouse spatial annotation frozen head inventory mismatch in {result_id}",
    )
    seen_heads: set[tuple[str, int]] = set()
    for frozen_head in frozen_heads:
        _require(
            isinstance(frozen_head, dict),
            f"mouse spatial annotation frozen head entry is invalid in {result_id}",
        )
        method_id = frozen_head.get("representation")
        seed = frozen_head.get("seed")
        key = (method_id, seed)
        _require(
            method_id in expected_methods
            and seed in expected_seeds
            and key not in seen_heads,
            f"mouse spatial annotation frozen head identity mismatch in {result_id}",
        )
        seen_heads.add(key)
        seed_index = expected_seeds.index(seed)
        expected_step = methods[method_id]["best_steps"][seed_index]
        expected_checkpoint = (
            run_root / "heads" / method_id / f"seed_{seed}" / "head_best.pt"
        ).resolve()
        declared_checkpoint = _resolve_project_file(
            frozen_head.get("path", ""),
            project_root=project_root,
            result_id=result_id,
            label="mouse spatial annotation frozen checkpoint",
        )
        _require(
            frozen_head.get("selected_step") == expected_step
            and declared_checkpoint == expected_checkpoint
            and declared_checkpoint.is_file(),
            f"mouse spatial annotation frozen head path/step mismatch in {result_id}",
        )
    _require(
        seen_heads
        == {
            (method_id, seed)
            for method_id in expected_methods
            for seed in expected_seeds
        },
        f"mouse spatial annotation frozen head coverage mismatch in {result_id}",
    )

    provenance = gate.get("provenance")
    _require(
        isinstance(provenance, dict)
        and provenance.get("representations") == list(expected_methods)
        and provenance.get("head_seeds") == expected_seeds
        and provenance.get("query_folds_per_descriptor")
        == split.get("query_folds_per_descriptor"),
        f"mouse spatial annotation provenance mismatch in {result_id}",
    )
    config_path = _resolve_project_file(
        provenance.get("config", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse spatial annotation config",
    )
    prepared_path = _resolve_project_file(
        provenance.get("prepared", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse spatial annotation prepared directory",
    )
    _require(
        config_path.is_file()
        and prepared_path == (run_root / "prepared").resolve()
        and prepared_path.is_dir(),
        f"mouse spatial annotation config/prepared provenance mismatch in {result_id}",
    )
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"invalid mouse spatial annotation config in {result_id}"
        ) from exc
    _require(
        isinstance(config, dict)
        and config.get("format") == "brainwm_mouse_spatial_annotation_config_v1"
        and config.get("experiment_id") == receipt.get("contract_id")
        and config.get("representations") == list(expected_methods)
        and config.get("head", {}).get("seeds") == expected_seeds
        and config.get("head", {}).get("max_steps") == maximum_steps
        and config.get("data", {}).get("sources") == split.get("source_ids")
        and config.get("data", {}).get("common_genes") == split.get("panel_size")
        and config.get("data", {}).get("query_folds_per_descriptor")
        == split.get("query_folds_per_descriptor")
        and config.get("data", {}).get("sections", {}).get("train")
        == split.get("train_sections")
        and config.get("data", {}).get("sections", {}).get("validation")
        == split.get("validation_sections")
        and config.get("data", {}).get("sections", {}).get("test")
        == split.get("sealed_test_sections"),
        f"mouse spatial annotation frozen config mismatch in {result_id}",
    )

    frozen_models = provenance.get("frozen_models")
    config_models = config.get("frozen_models", {})
    _require(
        isinstance(frozen_models, dict)
        and frozen_models.get("neighbor_k") == evaluator.get("neighbor_k")
        and frozen_models.get("neighbor_pseudocount")
        == evaluator.get("neighbor_pseudocount"),
        f"mouse spatial annotation frozen model settings mismatch in {result_id}",
    )
    for provenance_key, evaluator_key in (
        ("model_config", "brainwm_model_config"),
        ("stage0_checkpoint", "brainwm_checkpoint"),
        ("su07_head", "su07_head"),
    ):
        declared = _resolve_project_file(
            frozen_models.get(provenance_key, ""),
            project_root=project_root,
            result_id=result_id,
            label=f"mouse spatial annotation {provenance_key}",
        )
        frozen = _resolve_project_file(
            evaluator.get(evaluator_key, ""),
            project_root=project_root,
            result_id=result_id,
            label=f"frozen evaluator {evaluator_key}",
        )
        configured = _resolve_workspace_or_project_file(
            config_models.get(provenance_key, ""),
            project_root=project_root,
            result_id=result_id,
            label=f"configured mouse annotation {provenance_key}",
        )
        _require(
            declared == frozen == configured and declared.is_file(),
            f"mouse spatial annotation {provenance_key} provenance mismatch in {result_id}",
        )

    preparation = _load_receipt_json(
        prepared_path / "preparation_receipt.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse spatial annotation preparation receipt",
    )
    preparation_data = preparation.get("data", {})
    _require(
        preparation.get("format")
        == "brainwm_mouse_spatial_annotation_preparation_v1"
        and preparation.get("contract_id") == receipt.get("contract_id")
        and preparation.get("status") == "prepared_test_labels_closed"
        and preparation_data.get("sources") == split.get("source_ids")
        and preparation_data.get("common_genes") == split.get("panel_size")
        and preparation_data.get("rows_by_phase")
        == {
            "train": split.get("train_rows"),
            "validation": split.get("validation_rows"),
            "test": split.get("validation_rows"),
        }
        and preparation_data.get("descriptors_by_phase")
        == {"train": 64, "validation": 64, "test": 64}
        and set(preparation_data.get("descriptor_inventory", {}).get("train", {}))
        == set(split.get("train_sections", []))
        and set(
            preparation_data.get("descriptor_inventory", {}).get("validation", {})
        )
        == set(split.get("validation_sections", []))
        and set(preparation_data.get("descriptor_inventory", {}).get("test", {}))
        == set(split.get("sealed_test_sections", []))
        and preparation.get("label_access")
        == {
            "train_opened": True,
            "validation_opened": True,
            "test_opened": False,
            "test_label_reads": 0,
        },
        f"mouse spatial annotation preparation provenance mismatch in {result_id}",
    )

    transform = _load_receipt_json(
        run_root / "transformed" / "transform_receipt.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse spatial annotation transform receipt",
    )
    expected_shape = [split["validation_rows"], 256]
    transform_phases = transform.get("phases", {})
    _require(
        transform.get("format") == "brainwm_mouse_spatial_annotation_transform_v1"
        and transform.get("contract_id") == receipt.get("contract_id")
        and transform.get("status") == "train_validation_transformed"
        and transform.get("transform_fit") == "training raw expression only"
        and transform.get("transform")
        == "row-wise log1p(1e4 * composition), centering, then exact covariance PCA"
        and transform.get("input_genes") == split.get("panel_size")
        and transform.get("output_dim") == 256
        and transform.get("representations") == list(expected_methods)
        and transform.get("test_expression_opened") is False
        and transform.get("test_labels_opened") is False
        and all(
            transform_phases.get(phase, {}).get(view) == expected_shape
            for phase in ("train", "validation")
            for view in ("raw", "neighbor", "su07", "stage0")
        ),
        f"mouse spatial annotation transform provenance mismatch in {result_id}",
    )

    method_id = row["method_id"]
    _require(
        method_id in expected_methods
        and row["method_class"] == expected_methods[method_id],
        f"unsupported mouse spatial annotation method in {result_id}",
    )
    expected_data_axis = (
        "eval_train_fit_log1p_cpm_pca"
        if method_id.startswith("raw_")
        else "pretrain_log1p_library"
    )
    _require(
        row["axis"]["data"] == expected_data_axis,
        f"mouse spatial annotation data axis mismatch in {result_id}",
    )
    method_summary = summaries[method_id]
    _require_same_number(
        method_summary["median_validation_reward"],
        row["raw_metric"],
        f"mouse spatial annotation receipt metric mismatch in {result_id}",
    )
    _require(
        row["seed"] == {"head": expected_seeds, "aggregation": "median"},
        f"mouse spatial annotation seed mismatch in {result_id}",
    )
    _require(
        row["budget"]
        == {
            "head_max_steps_per_seed": maximum_steps,
            "selected_checkpoint_steps": method_summary["best_steps"],
            "head_trainable_parameters": trainable_parameters,
            "train_rows": split["train_rows"],
            "validation_rows": split["validation_rows"],
        },
        f"mouse spatial annotation budget mismatch in {result_id}",
    )
    diagnostics = row["diagnostics"]
    _require(
        diagnostics.get("evidence_scope") == "section_heldout_validation_only"
        and diagnostics.get("promotion_eligible") is False
        and diagnostics.get("test_generalization_evidence") is False
        and diagnostics.get("zero_shot_evidence") is False
        and diagnostics.get("validation_gate_status") == "validation_gate_failed"
        and diagnostics.get("validation_gate_passed") is False
        and diagnostics.get("test_expression_opened") is False
        and diagnostics.get("test_labels_opened") is False
        and diagnostics.get("test_label_reads") == 0
        and diagnostics.get("per_seed_validation_rewards")
        == method_summary["validation_rewards"]
        and diagnostics.get("stage2_mcr_auxiliary_metric_included") is False
        and diagnostics.get("clipping_affects_reward") is False,
        f"mouse spatial annotation test/access diagnostics mismatch in {result_id}",
    )
    _require_same_number(
        diagnostics.get("candidate_margin_vs_strongest_raw"),
        candidate - strongest_raw,
        f"mouse spatial annotation raw gate margin mismatch in {result_id}",
    )
    _require_same_number(
        diagnostics.get("candidate_margin_vs_stage0_neighbor"),
        candidate - stage0_neighbor,
        f"mouse spatial annotation neighbor gate margin mismatch in {result_id}",
    )


def _validate_mouse_spatial_raw_su07_test_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
) -> None:
    """Validate the sealed Raw-skip SU-07 fresh-test confirmation."""

    result_id = row["result_id"]
    split = benchmark["split"]
    evaluator = benchmark["evaluator"]
    expected_methods = {
        "rawdup_su07": "candidate",
        "rawdup_neighbor": "baseline",
    }
    expected_seeds = evaluator.get("head_seeds")
    test_rows = evaluator.get("test_rows")
    _require(
        receipt.get("format") == MOUSE_SPATIAL_RAW_SU07_TEST_FORMAT
        and receipt.get("contract_id")
        == benchmark.get("source_experiment_contract_id")
        and receipt.get("status") == "test_complete_no_further_updates"
        and receipt.get("primary_metric") == row["primary_metric"]
        and receipt.get("candidate") == evaluator.get("candidate")
        and receipt.get("control") == evaluator.get("matched_control")
        and receipt.get("test_label_reads") == 1
        and receipt.get("post_test_updates_allowed") is False,
        f"mouse Raw-SU07 test receipt status/contract mismatch in {result_id}",
    )
    _require(
        isinstance(expected_seeds, list)
        and len(expected_seeds) == 3
        and all(isinstance(seed, int) for seed in expected_seeds)
        and test_rows == split.get("test_rows") == 16384,
        f"mouse Raw-SU07 evaluator seed/row contract mismatch in {result_id}",
    )

    def validate_metric(
        metric: Any, *, expected_rows: int, label: str
    ) -> None:
        _require(
            isinstance(metric, dict)
            and metric.get("rows") == expected_rows
            and isinstance(metric.get("known_rows"), int)
            and 0 <= metric["known_rows"] <= expected_rows
            and all(
                _finite_number(metric.get(key))
                and 0.0 <= float(metric[key]) <= 1.0
                for key in (
                    "coverage",
                    "macro_f1",
                    "balanced_accuracy",
                    "coverage_adjusted_macro_f1",
                )
            ),
            f"mouse Raw-SU07 {label} metric schema mismatch in {result_id}",
        )
        _require_same_number(
            metric["known_rows"] / expected_rows,
            metric["coverage"],
            f"mouse Raw-SU07 {label} coverage mismatch in {result_id}",
        )
        _require_same_number(
            metric["coverage"] * metric["macro_f1"],
            metric["coverage_adjusted_macro_f1"],
            f"mouse Raw-SU07 {label} adjusted metric mismatch in {result_id}",
        )

    methods = receipt.get("methods")
    _require(
        isinstance(methods, dict) and set(methods) == set(expected_methods),
        f"mouse Raw-SU07 test method inventory mismatch in {result_id}",
    )
    method_rewards: dict[str, list[float]] = {}
    for method_id in expected_methods:
        method = methods.get(method_id)
        seed_records = method.get("seeds") if isinstance(method, dict) else None
        rewards = method.get("rewards") if isinstance(method, dict) else None
        _require(
            isinstance(seed_records, list)
            and len(seed_records) == len(expected_seeds)
            and isinstance(rewards, list)
            and len(rewards) == len(expected_seeds)
            and all(_finite_number(value) for value in rewards),
            f"mouse Raw-SU07 per-seed results mismatch for {method_id} in {result_id}",
        )
        observed_rewards: list[float] = []
        for expected_seed, seed_record in zip(
            expected_seeds, seed_records, strict=True
        ):
            _require(
                isinstance(seed_record, dict)
                and seed_record.get("seed") == expected_seed,
                f"mouse Raw-SU07 seed identity mismatch for {method_id} in {result_id}",
            )
            metrics = seed_record.get("metrics")
            overall = metrics.get("overall") if isinstance(metrics, dict) else None
            by_source = metrics.get("by_source") if isinstance(metrics, dict) else None
            _require(
                isinstance(overall, dict)
                and set(overall) == set(split["label_levels"])
                and isinstance(by_source, dict)
                and set(by_source)
                == {str(source_id) for source_id in split["source_ids"]},
                f"mouse Raw-SU07 metric inventory mismatch for {method_id} in {result_id}",
            )
            for label_name in split["label_levels"]:
                validate_metric(
                    overall[label_name],
                    expected_rows=test_rows,
                    label=f"{method_id}/{expected_seed}/{label_name}",
                )
            rows_per_source = test_rows // len(split["source_ids"])
            for source_id, source_metrics in by_source.items():
                _require(
                    isinstance(source_metrics, dict)
                    and set(source_metrics) == set(split["label_levels"]),
                    f"mouse Raw-SU07 source metrics mismatch for {method_id}/{source_id} in {result_id}",
                )
                for label_name in split["label_levels"]:
                    validate_metric(
                        source_metrics[label_name],
                        expected_rows=rows_per_source,
                        label=(
                            f"{method_id}/{expected_seed}/source-{source_id}/"
                            f"{label_name}"
                        ),
                    )
            reward = seed_record.get("reward")
            _require_same_number(
                overall["original_cell_type"]["coverage_adjusted_macro_f1"],
                reward,
                f"mouse Raw-SU07 seed reward mismatch for {method_id}/{expected_seed} in {result_id}",
            )
            observed_rewards.append(float(reward))
        _require(
            all(
                math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(observed_rewards, rewards, strict=True)
            ),
            f"mouse Raw-SU07 reward vector mismatch for {method_id} in {result_id}",
        )
        _require_same_number(
            median(observed_rewards),
            method.get("median_reward"),
            f"mouse Raw-SU07 median reward mismatch for {method_id} in {result_id}",
        )
        method_rewards[method_id] = observed_rewards

    candidate_median = float(methods["rawdup_su07"]["median_reward"])
    control_median = float(methods["rawdup_neighbor"]["median_reward"])
    test_delta = candidate_median - control_median
    _require_same_number(
        receipt.get("candidate_minus_control"),
        test_delta,
        f"mouse Raw-SU07 test delta mismatch in {result_id}",
    )
    _require(
        receipt.get("candidate_beats_control") is True and test_delta > 0.0,
        f"mouse Raw-SU07 positive test comparison mismatch in {result_id}",
    )

    gate_path = receipt_path.with_name("validation_gate.json")
    gate = _load_receipt_json(
        gate_path,
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 validation gate",
    )
    _require(
        gate.get("format") == MOUSE_SPATIAL_RAW_SU07_GATE_FORMAT
        and gate.get("contract_id") == receipt.get("contract_id")
        and gate.get("status") == "validation_gate_passed"
        and gate.get("primary_metric") == row["primary_metric"]
        and gate.get("frozen_head_checkpoints") == 6
        and gate.get("test_expression_access_granted") is True
        and gate.get("test_label_access_granted_after_predictions") is True
        and gate.get("test_label_reads") == 0
        and gate.get("post_test_updates_allowed") is False,
        f"mouse Raw-SU07 validation gate mismatch in {result_id}",
    )
    gate_candidate = gate.get("candidate")
    gate_control = gate.get("matched_control")
    _require(
        isinstance(gate_candidate, dict)
        and gate_candidate.get("method") == "rawdup_su07"
        and isinstance(gate_control, dict)
        and gate_control.get("method") == "rawdup_neighbor",
        f"mouse Raw-SU07 gate method binding mismatch in {result_id}",
    )
    _require_same_number(
        gate_candidate.get("median"),
        evaluator.get("validation_candidate_median"),
        f"mouse Raw-SU07 validation candidate metric mismatch in {result_id}",
    )
    _require_same_number(
        gate_control.get("median"),
        evaluator.get("validation_control_median"),
        f"mouse Raw-SU07 validation control metric mismatch in {result_id}",
    )
    _require_same_number(
        gate.get("candidate_minus_control"),
        gate_candidate["median"] - gate_control["median"],
        f"mouse Raw-SU07 validation delta mismatch in {result_id}",
    )

    provenance = gate.get("provenance")
    _require(
        isinstance(provenance, dict)
        and provenance.get("representations")
        == ["rawdup_su07", "rawdup_neighbor"]
        and provenance.get("head_seeds") == expected_seeds
        and provenance.get("query_folds_per_descriptor")
        == split.get("query_folds_per_descriptor"),
        f"mouse Raw-SU07 gate provenance mismatch in {result_id}",
    )
    data_config_path = _resolve_project_file(
        provenance.get("data_config", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Raw-SU07 data config",
    )
    factorial_config_path = _resolve_project_file(
        provenance.get("factorial_config", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Raw-SU07 factorial config",
    )
    factorial_validation_path = _resolve_project_file(
        provenance.get("factorial_validation_results", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Raw-SU07 factorial validation",
    )
    prepared_path = _resolve_project_file(
        provenance.get("prepared", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Raw-SU07 prepared directory",
    )
    feature_run = _resolve_project_file(
        provenance.get("feature_run", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Raw-SU07 feature run",
    )
    _require(
        data_config_path.is_file()
        and factorial_config_path.is_file()
        and factorial_validation_path.is_file()
        and prepared_path == feature_run / "prepared"
        and prepared_path.is_dir(),
        f"mouse Raw-SU07 source paths mismatch in {result_id}",
    )
    try:
        data_config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
        factorial_config = yaml.safe_load(
            factorial_config_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid mouse Raw-SU07 config in {result_id}") from exc
    _require(
        isinstance(data_config, dict)
        and data_config.get("format")
        == "brainwm_mouse_spatial_annotation_config_v1"
        and data_config.get("data", {}).get("sources") == split["source_ids"]
        and data_config.get("data", {}).get("common_genes")
        == split["panel_size"]
        and data_config.get("data", {}).get("sections", {}).get("train")
        == split["train_sections"]
        and data_config.get("data", {}).get("sections", {}).get("validation")
        == split["validation_sections"]
        and data_config.get("data", {}).get("sections", {}).get("test")
        == split["test_sections"]
        and data_config.get("head", {}).get("seeds") == expected_seeds
        and data_config.get("head", {}).get("max_steps")
        == evaluator["head_max_steps"],
        f"mouse Raw-SU07 data config mismatch in {result_id}",
    )
    _require(
        isinstance(factorial_config, dict)
        and factorial_config.get("format")
        == "brainwm_mouse_spatial_raw_skip_factorial_config_v1"
        and factorial_config.get("experiment_id")
        == benchmark.get("selection_experiment_contract_id")
        and set(factorial_config.get("representations", []))
        >= set(expected_methods)
        and factorial_config.get("head", {}).get("seeds") == expected_seeds
        and factorial_config.get("head", {}).get("max_steps")
        == evaluator["head_max_steps"],
        f"mouse Raw-SU07 factorial config mismatch in {result_id}",
    )

    factorial = _load_receipt_json(
        factorial_validation_path,
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 factorial validation receipt",
    )
    factorial_methods = factorial.get("methods")
    _require(
        factorial.get("format")
        == "brainwm_mouse_spatial_raw_skip_factorial_result_v1"
        and factorial.get("contract_id")
        == benchmark.get("selection_experiment_contract_id")
        and factorial.get("status") == "validation_complete"
        and factorial.get("primary_metric") == row["primary_metric"]
        and isinstance(factorial_methods, dict)
        and set(factorial_methods) >= set(expected_methods)
        and factorial.get("test_expression_opened") is False
        and factorial.get("test_labels_opened") is False
        and factorial.get("test_label_reads") == 0,
        f"mouse Raw-SU07 factorial validation mismatch in {result_id}",
    )
    for method_id, expected_median in (
        ("rawdup_su07", evaluator["validation_candidate_median"]),
        ("rawdup_neighbor", evaluator["validation_control_median"]),
    ):
        validation_method = factorial_methods[method_id]
        validation_rewards = validation_method.get("validation_rewards")
        _require(
            isinstance(validation_rewards, list)
            and len(validation_rewards) == len(expected_seeds)
            and validation_method.get("best_steps")
            == evaluator["selected_checkpoint_steps"][method_id]
            and validation_method.get("trainable_parameters")
            == evaluator["head_trainable_parameters"],
            f"mouse Raw-SU07 validation selection mismatch for {method_id} in {result_id}",
        )
        _require_same_number(
            median(validation_rewards),
            expected_median,
            f"mouse Raw-SU07 validation median mismatch for {method_id} in {result_id}",
        )

    frozen_heads = gate.get("frozen_heads")
    _require(
        isinstance(frozen_heads, list) and len(frozen_heads) == 6,
        f"mouse Raw-SU07 frozen head inventory mismatch in {result_id}",
    )
    factorial_run = factorial_validation_path.parent.parent
    frozen_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for frozen_head in frozen_heads:
        _require(
            isinstance(frozen_head, dict),
            f"mouse Raw-SU07 frozen head entry mismatch in {result_id}",
        )
        pair = (frozen_head.get("representation"), frozen_head.get("seed"))
        _require(
            pair[0] in expected_methods
            and pair[1] in expected_seeds
            and pair not in frozen_by_pair,
            f"mouse Raw-SU07 frozen head identity mismatch in {result_id}",
        )
        seed_index = expected_seeds.index(pair[1])
        selected_step = evaluator["selected_checkpoint_steps"][pair[0]][seed_index]
        head_root = factorial_run / "heads" / pair[0] / f"seed_{pair[1]}"
        checkpoint = _resolve_project_file(
            frozen_head.get("path", ""),
            project_root=project_root,
            result_id=result_id,
            label="mouse Raw-SU07 frozen head",
        )
        head_receipt = _load_receipt_json(
            head_root / "result.json",
            project_root=project_root,
            cache=receipt_cache,
            result_id=result_id,
            label="mouse Raw-SU07 head receipt",
        )
        _require(
            frozen_head.get("selected_step") == selected_step
            and checkpoint == (head_root / "head_best.pt").resolve()
            and checkpoint.is_file()
            and head_receipt.get("format")
            == "brainwm_mouse_spatial_annotation_head_result_v1"
            and head_receipt.get("contract_id")
            == benchmark.get("selection_experiment_contract_id")
            and head_receipt.get("status") == "validation_complete"
            and head_receipt.get("representation") == pair[0]
            and head_receipt.get("seed") == pair[1]
            and head_receipt.get("best_step") == selected_step
            and head_receipt.get("architecture", {}).get("trainable_parameters")
            == evaluator["head_trainable_parameters"]
            and head_receipt.get("gradient_norms", {}).get("steps")
            == evaluator["head_max_steps"]
            and head_receipt.get("test_access")
            == {"expression_opened": False, "labels_opened": False},
            f"mouse Raw-SU07 frozen head evidence mismatch for {pair} in {result_id}",
        )
        frozen_by_pair[pair] = frozen_head
    expected_pairs = {
        (method_id, seed)
        for method_id in expected_methods
        for seed in expected_seeds
    }
    _require(
        set(frozen_by_pair) == expected_pairs,
        f"mouse Raw-SU07 frozen head coverage mismatch in {result_id}",
    )

    frozen_models = provenance.get("frozen_models")
    _require(
        isinstance(frozen_models, dict)
        and frozen_models.get("neighbor_k") == evaluator.get("neighbor_k")
        and frozen_models.get("neighbor_pseudocount")
        == evaluator.get("neighbor_pseudocount")
        and evaluator.get("stage0_cell_latent_used") is False
        and evaluator.get("su07_frozen") is True,
        f"mouse Raw-SU07 frozen model settings mismatch in {result_id}",
    )
    for provenance_key, evaluator_key in (
        ("model_config", "brainwm_model_config"),
        ("stage0_checkpoint", "brainwm_checkpoint"),
        ("su07_head", "su07_head"),
    ):
        declared = _resolve_project_file(
            frozen_models.get(provenance_key, ""),
            project_root=project_root,
            result_id=result_id,
            label=f"mouse Raw-SU07 {provenance_key}",
        )
        frozen = _resolve_project_file(
            evaluator.get(evaluator_key, ""),
            project_root=project_root,
            result_id=result_id,
            label=f"mouse Raw-SU07 evaluator {evaluator_key}",
        )
        _require(
            declared == frozen and declared.is_file(),
            f"mouse Raw-SU07 {provenance_key} mismatch in {result_id}",
        )

    standardization_run = _resolve_workspace_or_project_file(
        factorial_config.get("standardization_run", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Raw-SU07 standardization run",
    )
    standardizer = _load_receipt_json(
        standardization_run / "standardizer_receipt.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 standardizer receipt",
    )
    standardizer_checkpoint = _resolve_project_file(
        standardizer.get("checkpoint", ""),
        project_root=project_root,
        result_id=result_id,
        label="mouse Raw-SU07 standardizer checkpoint",
    )
    _require(
        standardizer.get("format")
        == "brainwm_mouse_spatial_feature_standardizer_v1"
        and standardizer.get("status") == "train_statistics_frozen"
        and _resolve_project_file(
            standardizer.get("parent_run", ""),
            project_root=project_root,
            result_id=result_id,
            label="mouse Raw-SU07 standardizer parent",
        )
        == feature_run
        and standardizer.get("test_expression_opened") is False
        and standardizer.get("test_labels_opened") is False
        and standardizer_checkpoint.is_file()
        and all(
            audit.get("rows") == split["train_rows"]
            and audit.get("features") == 256
            for audit in standardizer.get("audit", {}).values()
        ),
        f"mouse Raw-SU07 train-fit standardizer mismatch in {result_id}",
    )

    feature_receipt = _load_receipt_json(
        feature_run / "features_test" / "feature_receipt.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 test feature receipt",
    )
    feature_access = feature_receipt.get("test_access", {})
    _require(
        feature_receipt.get("format")
        == "brainwm_mouse_spatial_annotation_features_v1"
        and feature_receipt.get("status") == "test_features_complete"
        and feature_receipt.get("phase") == "test"
        and feature_receipt.get("rows") == test_rows
        and feature_receipt.get("common_genes") == split["panel_size"]
        and feature_receipt.get("descriptors")
        == split["sections_per_phase"] * split["descriptors_per_section"]
        and feature_receipt.get("brainwm", {}).get("global_step")
        == evaluator["brainwm_checkpoint_step"]
        and feature_receipt.get("brainwm", {}).get("mutated") is False
        and feature_access.get("test_expression_opened") is True
        and feature_access.get("test_labels_opened") is False
        and feature_access.get("test_label_reads") == 0
        and _resolve_project_file(
            feature_access.get("validation_gate", ""),
            project_root=project_root,
            result_id=result_id,
            label="mouse Raw-SU07 feature gate",
        )
        == gate_path.resolve(),
        f"mouse Raw-SU07 test feature boundary mismatch in {result_id}",
    )
    transform_receipt = _load_receipt_json(
        feature_run / "transformed" / "test_transform_receipt.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 test transform receipt",
    )
    _require(
        transform_receipt.get("format")
        == "brainwm_mouse_spatial_annotation_transform_v1"
        and transform_receipt.get("status") == "test_transformed"
        and transform_receipt.get("phase") == "test"
        and transform_receipt.get("test_expression_opened") is True
        and transform_receipt.get("test_labels_opened") is False
        and all(
            shape == [test_rows, 256]
            for shape in transform_receipt.get("shapes", {}).values()
        ),
        f"mouse Raw-SU07 test transform boundary mismatch in {result_id}",
    )

    run_root = receipt_path.parent
    seal = _load_receipt_json(
        run_root / "predictions_frozen.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 prediction seal",
    )
    seal_records = seal.get("predictions")
    _require(
        seal.get("format") == "brainwm_mouse_spatial_raw_su07_prediction_seal_v1"
        and seal.get("contract_id") == receipt.get("contract_id")
        and seal.get("status")
        == "all_test_predictions_frozen_before_label_access"
        and seal.get("test_prediction_count") == 6
        and seal.get("test_label_reads") == 0
        and seal.get("post_seal_model_updates_allowed") is False
        and isinstance(seal_records, list)
        and len(seal_records) == 6,
        f"mouse Raw-SU07 prediction seal mismatch in {result_id}",
    )
    sealed_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    child_by_method: dict[str, list[dict[str, Any]]] = {
        method_id: [] for method_id in expected_methods
    }
    for record in seal_records:
        _require(
            isinstance(record, dict),
            f"mouse Raw-SU07 sealed prediction entry mismatch in {result_id}",
        )
        pair = (record.get("method"), record.get("seed"))
        _require(
            pair in expected_pairs and pair not in sealed_by_pair,
            f"mouse Raw-SU07 sealed prediction identity mismatch in {result_id}",
        )
        seed_index = expected_seeds.index(pair[1])
        selected_step = evaluator["selected_checkpoint_steps"][pair[0]][seed_index]
        checkpoint = _resolve_project_file(
            record.get("checkpoint", ""),
            project_root=project_root,
            result_id=result_id,
            label="mouse Raw-SU07 prediction checkpoint",
        )
        prediction = _resolve_project_file(
            record.get("prediction", ""),
            project_root=project_root,
            result_id=result_id,
            label="mouse Raw-SU07 frozen prediction",
        )
        child_path = run_root / "predictions" / pair[0] / f"seed_{pair[1]}.json"
        child = _load_receipt_json(
            child_path,
            project_root=project_root,
            cache=receipt_cache,
            result_id=result_id,
            label="mouse Raw-SU07 prediction receipt",
        )
        _require(
            record.get("selected_step") == selected_step
            and checkpoint
            == _resolve_project_file(
                frozen_by_pair[pair]["path"],
                project_root=project_root,
                result_id=result_id,
                label="mouse Raw-SU07 gate checkpoint",
            )
            and checkpoint.is_file()
            and prediction.is_file()
            and record.get("rows") == test_rows
            and record.get("bytes") == prediction.stat().st_size
            and child.get("format")
            == "brainwm_mouse_spatial_annotation_test_prediction_v1"
            and child.get("contract_id") == receipt.get("contract_id")
            and child.get("status") == "test_prediction_frozen"
            and child.get("method") == pair[0]
            and child.get("seed") == pair[1]
            and child.get("selected_step") == selected_step
            and _resolve_project_file(
                child.get("checkpoint", ""),
                project_root=project_root,
                result_id=result_id,
                label="mouse Raw-SU07 child checkpoint",
            )
            == checkpoint
            and _resolve_project_file(
                child.get("prediction", ""),
                project_root=project_root,
                result_id=result_id,
                label="mouse Raw-SU07 child prediction",
            )
            == prediction
            and child.get("rows") == test_rows
            and child.get("test_expression_opened") is True
            and child.get("test_labels_opened") is False,
            f"mouse Raw-SU07 sealed prediction evidence mismatch for {pair} in {result_id}",
        )
        sealed_by_pair[pair] = record
        child_by_method[pair[0]].append(child)
    _require(
        set(sealed_by_pair) == expected_pairs,
        f"mouse Raw-SU07 sealed prediction coverage mismatch in {result_id}",
    )
    for method_id in expected_methods:
        summary = _load_receipt_json(
            run_root / "predictions" / method_id / "summary.json",
            project_root=project_root,
            cache=receipt_cache,
            result_id=result_id,
            label="mouse Raw-SU07 prediction summary",
        )
        children = sorted(child_by_method[method_id], key=lambda item: item["seed"])
        _require(
            summary.get("format")
            == "brainwm_mouse_spatial_test_prediction_method_summary_v1"
            and summary.get("contract_id") == receipt.get("contract_id")
            and summary.get("status") == "method_predictions_frozen"
            and summary.get("method") == method_id
            and sorted(summary.get("predictions", []), key=lambda item: item["seed"])
            == children
            and summary.get("test_labels_opened") is False,
            f"mouse Raw-SU07 prediction summary mismatch for {method_id} in {result_id}",
        )

    preflight = _load_receipt_json(
        run_root / "test_preflight.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 test preflight",
    )
    access_intent = _load_receipt_json(
        run_root / "test_label_access_started.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 label access intent",
    )
    access_receipt = _load_receipt_json(
        run_root / "test_label_access_receipt.json",
        project_root=project_root,
        cache=receipt_cache,
        result_id=result_id,
        label="mouse Raw-SU07 label access receipt",
    )
    _require(
        preflight
        == {
            "format": "brainwm_mouse_spatial_raw_su07_test_preflight_v1",
            "contract_id": receipt.get("contract_id"),
            "status": "ready_for_single_test_label_read",
            "prediction_count": split["test_prediction_count"],
            "prediction_rows": test_rows,
            "test_label_reads": 0,
        }
        and access_intent.get("format")
        == "brainwm_mouse_spatial_test_label_access_intent_v1"
        and access_intent.get("contract_id") == receipt.get("contract_id")
        and access_intent.get("status") == "single_read_started"
        and _resolve_project_file(
            access_intent.get("authorized_by", ""),
            project_root=project_root,
            result_id=result_id,
            label="mouse Raw-SU07 access authorization",
        )
        == gate_path.resolve()
        and access_receipt.get("format")
        == "brainwm_mouse_spatial_test_label_access_receipt_v1"
        and access_receipt.get("contract_id") == receipt.get("contract_id")
        and access_receipt.get("status") == "single_read_complete"
        and access_receipt.get("rows") == test_rows
        and access_receipt.get("sections") == split["test_sections"]
        and access_receipt.get("test_label_reads") == 1
        and access_receipt.get("post_test_updates_allowed") is False,
        f"mouse Raw-SU07 single label-access boundary mismatch in {result_id}",
    )

    method_id = row["method_id"]
    _require(
        method_id in expected_methods
        and row["method_class"] == expected_methods[method_id],
        f"unsupported mouse Raw-SU07 result method in {result_id}",
    )
    method_median = float(methods[method_id]["median_reward"])
    _require_same_number(
        method_median,
        row["raw_metric"],
        f"mouse Raw-SU07 receipt metric mismatch in {result_id}",
    )
    _require(
        row["axis"]["data"] == "eval_train_fit_log1p_cpm_pca_zscore",
        f"mouse Raw-SU07 data axis mismatch in {result_id}",
    )
    _require(
        row["seed"] == {"head": expected_seeds, "aggregation": "median"},
        f"mouse Raw-SU07 result seed mismatch in {result_id}",
    )
    _require(
        row["budget"]
        == {
            "head_max_steps_per_seed": evaluator["head_max_steps"],
            "selected_checkpoint_steps": evaluator["selected_checkpoint_steps"][
                method_id
            ],
            "head_trainable_parameters": evaluator["head_trainable_parameters"],
            "test_rows": test_rows,
            "test_prediction_count": evaluator["test_prediction_count"],
        },
        f"mouse Raw-SU07 result budget mismatch in {result_id}",
    )
    diagnostics = row["diagnostics"]
    _require(
        diagnostics.get("evidence_scope")
        == "independent_section_heldout_fresh_test_only"
        and diagnostics.get("promotion_eligible") is True
        and diagnostics.get("heldout_generalization_evidence") is True
        and diagnostics.get("zero_shot_evidence") is False
        and diagnostics.get("cross_species_transfer_evidence") is False
        and diagnostics.get("stage0_cell_latent_used") is False
        and diagnostics.get("test_predictions_frozen_before_label_access") is True
        and diagnostics.get("test_label_reads") == 1
        and diagnostics.get("post_test_updates_allowed") is False
        and diagnostics.get("per_seed_test_rewards") == method_rewards[method_id]
        and diagnostics.get("stage2_mcr_auxiliary_metric_included") is False
        and diagnostics.get("clipping_affects_reward") is False,
        f"mouse Raw-SU07 result claim/access diagnostics mismatch in {result_id}",
    )
    _require_same_number(
        diagnostics.get("candidate_minus_control"),
        test_delta,
        f"mouse Raw-SU07 result delta mismatch in {result_id}",
    )


_VERIFIED_RECEIPT_VALIDATORS = {
    ANN_LABEL_BLIND_V2_SCORE_FORMAT: (
        "annotation", _validate_ann_label_blind_v2_score_receipt
    ),
    ANN_LABEL_BLIND_V2_AGGREGATE_FORMAT: (
        "annotation", _validate_ann_label_blind_v2_aggregate_receipt
    ),
    "brainwm_ann_s64_label_blind_aggregate_v2": (
        "annotation", _validate_ann_label_blind_v2_aggregate_receipt
    ),
    "brainwm_ann_s64_label_blind_stage0_aggregate_v2": (
        "annotation", _validate_ann_label_blind_v2_aggregate_receipt
    ),
    "brainwm_ann_s64_label_blind_fusion_aggregate_v2": (
        "annotation", _validate_ann_label_blind_v2_aggregate_receipt
    ),
    "brainwm_autoresearch_validation_v1": ("annotation", _validate_ann_autoresearch_receipt),
    "brainwm_autoresearch_annotation_aggregate_v1": ("annotation", _validate_ann_aggregate_receipt),
    "brainwm_ann_s64_matched_baselines_v1": ("annotation", _validate_ann_baseline_receipt),
    "brainwm_ann_s64_stage0_clr_skip_fusion_confirmation_v1": (
        "annotation", _validate_ann_fusion_aggregate_receipt
    ),
    "brainwm_integration_fresh_evaluation_v1": ("integration", _validate_integration_receipt),
    MOUSE_STAGE2_SELECTION_FORMAT: (
        AUXILIARY_TASK_ID,
        _validate_mouse_stage2_mcr_receipt,
    ),
    MOUSE_SPATIAL_ANNOTATION_VALIDATION_FORMAT: (
        "annotation",
        _validate_mouse_spatial_annotation_receipt,
    ),
    MOUSE_SPATIAL_RAW_SU07_TEST_FORMAT: (
        "annotation",
        _validate_mouse_spatial_raw_su07_test_receipt,
    ),
    "brainwm_stage1_dev_source23_v1": ("development", _validate_development_receipt),
}


def _validate_verified_receipt(
    benchmark: dict[str, Any],
    row: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    project_root: Path,
    receipt_cache: dict[Path, dict[str, Any]],
) -> None:
    result_id = row["result_id"]
    entry = _VERIFIED_RECEIPT_VALIDATORS.get(receipt.get("format"))
    _require(entry is not None,
             f"unsupported verified receipt format in {result_id}: {receipt.get('format')}")
    expected_task, validator = entry
    _require(row["task_id"] == expected_task, f"receipt format/task mismatch in {result_id}")
    validator(
        benchmark, row, receipt, receipt_path=receipt_path, project_root=project_root,
        receipt_cache=receipt_cache,
    )


def leaderboard(
    registry: dict[str, Any],
    results: Iterable[dict[str, Any]],
    benchmark_id: str,
) -> list[dict[str, Any]]:
    """Return one leaderboard; a benchmark id is mandatory by design."""

    benchmarks = benchmark_index(registry)
    _require(benchmark_id in benchmarks, f"unknown benchmark_id: {benchmark_id}")
    rows = _verified_rows(results, benchmark_id)
    return sorted(rows, key=lambda row: (-float(row["reward"]), row["result_id"]))


def trajectory(results: Iterable[dict[str, Any]], benchmark_id: str) -> list[dict[str, Any]]:
    """Aggregate reward progress by iteration within exactly one benchmark."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _verified_rows(results, benchmark_id):
        grouped[int(row["iteration"])].append(row)
    best = -math.inf
    points: list[dict[str, Any]] = []
    for iteration in sorted(grouped):
        rows = sorted(grouped[iteration], key=lambda row: row["result_id"])
        iteration_best = max(float(row["reward"]) for row in rows)
        best = max(best, iteration_best)
        points.append(
            {
                "iteration": iteration,
                "method_ids": [row["method_id"] for row in rows],
                "rewards": [float(row["reward"]) for row in rows],
                "iteration_best": iteration_best,
                "best_so_far": best,
            }
        )
    return points


def baseline_coverage(benchmark: dict[str, Any], results: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    completed = {
        row["method_id"]
        for row in _verified_rows(results, benchmark["benchmark_id"])
    }
    legacy = {
        row["method_id"]
        for row in results
        if row.get("benchmark_id") == benchmark["benchmark_id"]
        and row.get("contract_status") == LEGACY_CONTRACT_STATUS
    }
    coverage = []
    for baseline in benchmark.get("baselines", []):
        row = dict(baseline)
        if baseline["method_id"] in completed:
            row["coverage"] = "verified"
        elif baseline["method_id"] in legacy:
            row["coverage"] = LEGACY_CONTRACT_STATUS
        else:
            row["coverage"] = baseline.get("status", "planned")
        coverage.append(row)
    return coverage


def baseline_gate(benchmark: dict[str, Any], results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare the best BrainWM route with the strongest declared baseline.

    Registration preserves all valid experiments.  This separate gate controls
    scientific promotion, so a negative experiment remains visible without
    becoming a positive project claim.
    """

    benchmark_rows = _verified_rows(results, benchmark["benchmark_id"])
    declared_baseline_ids = {
        baseline["method_id"] for baseline in benchmark.get("baselines", [])
    }
    completed_baseline_ids = {
        row["method_id"]
        for row in benchmark_rows
        if row["method_class"] == "baseline"
        and row["method_id"] in declared_baseline_ids
    }
    missing_baseline_ids = sorted(declared_baseline_ids - completed_baseline_ids)
    baseline_rows = [
        row
        for row in benchmark_rows
        if row["method_class"] == "baseline"
        and row["method_id"] in declared_baseline_ids
    ]
    candidate_rows = [
        row
        for row in benchmark_rows
        if row["method_class"] in PROMOTION_METHOD_CLASSES
    ]

    outcome: dict[str, Any] = {
        "status": "baseline_incomplete",
        "passed": False,
        "metric_gate_passed": False,
        "promotion_eligible": benchmark.get("promotion_eligible", True),
        "missing_baseline_ids": missing_baseline_ids,
        "best_candidate": None,
        "strongest_baseline": None,
        "reward_margin": None,
    }
    if not baseline_rows:
        return outcome
    if not candidate_rows:
        outcome["status"] = "candidate_missing"
        return outcome

    best_candidate = max(candidate_rows, key=lambda row: (float(row["reward"]), row["result_id"]))
    strongest_baseline = max(baseline_rows, key=lambda row: (float(row["reward"]), row["result_id"]))
    reward_margin = float(best_candidate["reward"]) - float(strongest_baseline["reward"])
    if reward_margin <= 0.0:
        status = "failed"
    elif missing_baseline_ids:
        status = "baseline_incomplete"
    else:
        status = "passed"
    outcome.update(
        {
            "status": status,
            "passed": status == "passed",
            "metric_gate_passed": status == "passed",
            "best_candidate": best_candidate,
            "strongest_baseline": strongest_baseline,
            "reward_margin": reward_margin,
        }
    )
    return outcome


def render_markdown(
    registry: dict[str, Any],
    results: Iterable[dict[str, Any]],
    *,
    scope: str = "downstream",
) -> str:
    rows = list(results)
    task_labels = {
        "integration": "整合 Integration",
        "annotation": "注释 Annotation",
        "development": "发育 Development",
        "perturbation": "扰动 Perturbation",
    }
    status_labels = {
        "active_frozen": "协议已冻结，持续迭代",
        "completed_frozen": "本轮结果已完成并冻结",
        "protocol_pending": "协议待冻结",
        "blocked_data": "受数据阻塞",
    }
    method_class_labels = {
        "baseline": "Baseline",
        "foundation": "Foundation",
        "candidate": "Candidate",
        "ablation": "Ablation",
        "hparam": "Hparam",
    }
    axis_labels = {
        "reference": "参照",
        "data": "数据",
        "architecture": "架构",
        "algorithm": "算法",
    }
    if scope == "downstream":
        title = "# BrainWM Experimental Results Table"
        introduction = (
            "本表固定 BrainWM 的整合、注释、发育和扰动四个下游任务，训练 Stage "
            "仅作为方法元数据。Stage2 MCR 等训练阶段辅助基准另表登记，不计作下游任务性能。"
        )
        report_groups = [
            (task_labels[task_id], registry["tasks"][task_id]["benchmarks"])
            for task_id in TASK_IDS
        ]
    elif scope == "auxiliary":
        title = "# BrainWM Auxiliary Experimental Results Table"
        introduction = (
            "本表只登记训练阶段的辅助架构与评估基准，不属于 Integration、Annotation、"
            "Development 或 Perturbation 下游任务，也不与下游 reward 横向比较。"
        )
        report_groups = [
            (
                "辅助架构基准 Auxiliary Architecture Benchmarks",
                registry.get("auxiliary_benchmarks", []),
            )
        ]
    else:
        raise ValueError("render scope must be downstream or auxiliary")

    lines = [
        title,
        "",
        introduction
        + " scBrain、scFiT、scUniFlow 等名称只表示候选组件的方法来源，用于赋能 BrainWM，不表示复现其论文任务、数据集或论文结果。正式结果必须绑定冻结的 Data/Evaluator 合同；历史结果仅保留供追溯，不参与排名、轨迹或 baseline gate。所有名次只在同一个 `benchmark_id` 内计算；主指标经固定变换得到 reward。Clipping 仅保留在 diagnostics 中，不参与 reward、筛选或排名。",
        "",
        "## 任务总览",
        "",
        "| 任务 | 状态 | Benchmark | Verified | Legacy | Baseline gate | 当前最优（仅限本 Benchmark） |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for group_label, benchmarks in report_groups:
        for benchmark in benchmarks:
            benchmark_id = benchmark["benchmark_id"]
            ranked = leaderboard(registry, rows, benchmark_id)
            historical = [
                row for row in rows
                if row["benchmark_id"] == benchmark_id
                and row.get("contract_status") == LEGACY_CONTRACT_STATUS
            ]
            gate = baseline_gate(benchmark, rows)
            best = "-"
            if ranked:
                best = f"{ranked[0]['method_name']} / {_format_number(ranked[0]['raw_metric'])}"
            gate_label = {
                "passed": "通过",
                "failed": "未通过",
                "baseline_incomplete": "Baseline 未齐",
                "candidate_missing": "BrainWM 结果未齐",
            }[gate["status"]]
            lines.append(
                f"| {group_label} | {status_labels[benchmark['status']]} | `{benchmark_id}` | "
                f"{len(ranked)} | {len(historical)} | {gate_label} | {best} |"
            )

    for group_label, benchmarks in report_groups:
        lines.extend(["", f"## {group_label}"])
        for benchmark in benchmarks:
            benchmark_id = benchmark["benchmark_id"]
            lines.extend(["", f"### `{benchmark_id}`", "", benchmark["scope"]])
            if benchmark.get("claim_scope") == TRAIN_INTERNAL_ARCHITECTURE_SCOPE:
                lines.extend(
                    [
                        "",
                        (
                            "证据范围：`train_internal_architecture_gate_only`；"
                            "`promotion_eligible=false`。本节只比较该 benchmark 内的匹配路线，"
                            "不与其他 benchmark 横向排名，也不构成 held-out 或 zero-shot 结论。"
                        ),
                    ]
                )
            if benchmark["status"] in RANKABLE_STATUSES:
                metric = benchmark["primary_metric"]
                mutable = benchmark["variation_policy"]["mutable_axes"]
                lines.extend(
                    [
                        "",
                        (
                            f"固定 split：`{benchmark['split']['split_id']}`；固定 evaluator："
                            f"`{benchmark['evaluator']['evaluator_id']}`；主指标：`{metric['name']}`"
                            f"（{metric['direction']}）；reward：`{metric['reward_formula']}`。"
                        ),
                        (
                            f"固定输入：{benchmark['input_contract']['expression']}；固定输出："
                            f"{benchmark['output_contract']['artifact']}（{benchmark['output_contract']['schema']}）。"
                        ),
                        (
                            "允许变化：Data="
                            f"`{mutable['data']['scope']}`、Architecture=`{mutable['architecture']['scope']}`、"
                            f"Algorithm=`{mutable['algorithm']['scope']}`；cohort、split、输入输出和 metric 不允许变化。"
                        ),
                    ]
                )
            else:
                lines.extend(["", f"当前原因：{benchmark['blocker']}"])
                draft = benchmark.get("metric_contract") or benchmark.get("proposed_evaluator")
                if isinstance(draft, dict) and draft.get("primary_metric"):
                    direction = draft.get("direction", "待定")
                    lines.extend(
                        [
                            "",
                            (
                                f"指标合同草案：主指标 `{draft['primary_metric']}`（{direction}）；"
                                f"reward：`{draft.get('reward_formula', '待定义')}`。"
                            ),
                        ]
                    )

            ranked = leaderboard(registry, rows, benchmark_id)
            gate = baseline_gate(benchmark, rows)
            gate_heading = (
                "Internal architecture gate"
                if benchmark.get("claim_scope") == TRAIN_INTERNAL_ARCHITECTURE_SCOPE
                else "Baseline gate"
            )
            ranking_heading = (
                "匹配路线 Matched Routes（仅限本内部门控）"
                if benchmark.get("claim_scope") == TRAIN_INTERNAL_ARCHITECTURE_SCOPE
                else "排行榜 Leaderboard"
            )
            if gate["status"] in {"passed", "failed"}:
                candidate = gate["best_candidate"]
                baseline = gate["strongest_baseline"]
                decision = "通过" if gate["passed"] else "未通过"
                missing_suffix = ""
                if gate["missing_baseline_ids"]:
                    missing = "、".join(
                        f"`{value}`" for value in gate["missing_baseline_ids"]
                    )
                    missing_suffix = f" 其余待补 baseline：{missing}。"
                lines.extend(
                    [
                        "",
                        f"#### {gate_heading}",
                        "",
                        (
                            f"**{decision}**：当前最佳 BrainWM 为 {candidate['method_name']} "
                            f"（{_format_number(candidate['raw_metric'])}），最强公平 baseline 为 "
                            f"{baseline['method_name']}（{_format_number(baseline['raw_metric'])}）；"
                            f"reward margin = `{_format_number(gate['reward_margin'])}`。"
                            f"{missing_suffix}"
                        ),
                    ]
                )
            elif gate["status"] == "baseline_incomplete":
                missing = "、".join(f"`{value}`" for value in gate["missing_baseline_ids"])
                lines.extend(
                    [
                        "",
                        f"#### {gate_heading}",
                        "",
                        f"尚不可判断：公平 baseline 未齐。缺少 {missing or '已登记 baseline 结果'}。",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        f"#### {gate_heading}",
                        "",
                        "尚不可判断：没有可晋级的 BrainWM candidate/foundation 结果。",
                    ]
                )
            if ranked:
                lines.extend(
                    [
                        "",
                        f"#### {ranking_heading}",
                        "",
                        "| Rank | 方法 | 类别 | 合同 | Stage | Data | Architecture | Algorithm | 变化轴 | Seed / 聚合 | Budget | Raw metric | Reward | 证据 |",
                        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
                    ]
                )
                for rank, row in enumerate(ranked, 1):
                    evidence = f"[{row['evidence_path']}](../../{row['evidence_path']})"
                    lines.append(
                        f"| {rank} | {row['method_name']} | {method_class_labels[row['method_class']]} | "
                        f"`{row.get('contract_status', LEGACY_CONTRACT_STATUS)}` | "
                        f"`{row['training_stage']}` | `{row['axis']['data']}` | `{row['axis']['architecture']}` | "
                        f"`{row['axis']['algorithm']}` | {axis_labels[row['axis']['variation']]} | {_format_seed(row['seed'])} | "
                        f"{_format_budget(row['budget'])} | {_format_number(row['raw_metric'])} | "
                        f"{_format_number(row['reward'])} | {evidence} |"
                    )

                points = trajectory(rows, benchmark_id)
                lines.extend(
                    [
                        "",
                        "#### 迭代轨迹 Iteration -> Reward -> Best-so-far",
                        "",
                        "| Iteration | 本轮方法 | 本轮 rewards | 本轮最优 | Best-so-far |",
                        "| ---: | --- | --- | ---: | ---: |",
                    ]
                )
                for point in points:
                    methods = ", ".join(f"`{value}`" for value in point["method_ids"])
                    rewards = ", ".join(_format_number(value) for value in point["rewards"])
                    lines.append(
                        f"| {point['iteration']} | {methods} | {rewards} | "
                        f"{_format_number(point['iteration_best'])} | {_format_number(point['best_so_far'])} |"
                    )
            else:
                historical_count = sum(
                    row["benchmark_id"] == benchmark_id
                    and row.get("contract_status") == LEGACY_CONTRACT_STATUS
                    for row in rows
                )
                lines.extend(
                    [
                        "",
                        (
                            "尚无合同绑定的可排名结果。"
                            f"历史参考结果 `{historical_count}` 条，仅保留在 append-only JSONL 中。"
                        ),
                    ]
                )

            coverage = baseline_coverage(benchmark, rows)
            if coverage:
                lines.extend(
                    [
                        "",
                        "#### 基线覆盖 Baseline Coverage",
                        "",
                        "| Baseline | 覆盖状态 | 公平性边界 |",
                        "| --- | --- | --- |",
                    ]
                )
                for baseline in coverage:
                    lines.append(
                        f"| {baseline['method_name']} | `{baseline['coverage']}` | {baseline['fairness']} |"
                    )
            planned_metrics = benchmark.get("planned_metrics", [])
            if planned_metrics:
                lines.extend(["", "候选指标：" + "、".join(f"`{value}`" for value in planned_metrics) + "。"])

    lines.extend(
        [
            "",
            "## 使用方式",
            "",
            "```bash",
            ".runtime/venv_gpu_20260901/bin/python scripts/render_experimental_results.py",
            "```",
            "",
            "新增结果前必须先冻结 benchmark 合同，并确保 result 中的 pretraining corpus、input/output、split、evaluator、primary metric 与 direction 完全匹配。新实验只允许改变标准化、模型架构或 loss/优化算法。旧结果不能补写 contract ID；经新协议重新运行后才能进入 leaderboard。",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_csvs(
    registry: dict[str, Any], results: Iterable[dict[str, Any]], output_dir: str | Path
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    written: list[Path] = []
    fields = [
        "rank",
        "benchmark_id",
        "task_id",
        "training_stage",
        "iteration",
        "method_id",
        "method_name",
        "method_class",
        "contract_status",
        "data_contract_id",
        "evaluator_contract_id",
        "data_axis",
        "architecture_axis",
        "algorithm_axis",
        "variation_axis",
        "primary_metric",
        "direction",
        "raw_metric",
        "reward",
        "evidence_path",
    ]
    for benchmark_id, benchmark in benchmark_index(registry).items():
        if benchmark["status"] not in RANKABLE_STATUSES:
            continue
        ranked = leaderboard(registry, rows, benchmark_id)
        path = output / f"{benchmark_id}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for rank, row in enumerate(ranked, 1):
                writer.writerow(
                    {
                        "rank": rank,
                        "benchmark_id": benchmark_id,
                        "task_id": row["task_id"],
                        "training_stage": row["training_stage"],
                        "iteration": row["iteration"],
                        "method_id": row["method_id"],
                        "method_name": row["method_name"],
                        "method_class": row["method_class"],
                        "contract_status": row.get("contract_status", LEGACY_CONTRACT_STATUS),
                        "data_contract_id": row.get("data_contract_id", ""),
                        "evaluator_contract_id": row.get("evaluator_contract_id", ""),
                        "data_axis": row["axis"]["data"],
                        "architecture_axis": row["axis"]["architecture"],
                        "algorithm_axis": row["axis"]["algorithm"],
                        "variation_axis": row["axis"]["variation"],
                        "primary_metric": row["primary_metric"],
                        "direction": row["direction"],
                        "raw_metric": row["raw_metric"],
                        "reward": row["reward"],
                        "evidence_path": row["evidence_path"],
                    }
                )
        written.append(path)
    return written


def write_reward_svgs(
    registry: dict[str, Any], results: Iterable[dict[str, Any]], output_dir: str | Path
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    written: list[Path] = []
    for benchmark_id, benchmark in benchmark_index(registry).items():
        if benchmark["status"] not in RANKABLE_STATUSES:
            continue
        points = trajectory(rows, benchmark_id)
        path = output / f"{benchmark_id}.svg"
        if points:
            payload = _trajectory_svg(benchmark_id, points)
        else:
            payload = _empty_trajectory_svg(benchmark_id)
        path.write_text(payload, encoding="utf-8")
        written.append(path)
    return written


def _empty_trajectory_svg(benchmark_id: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="280" '
        'viewBox="0 0 800 280">'
        '<rect width="800" height="280" fill="#ffffff"/>'
        f'<text x="400" y="120" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18" fill="#111827">{escape(benchmark_id)}</text>'
        '<text x="400" y="158" text-anchor="middle" font-family="sans-serif" '
        'font-size="15" fill="#6b7280">No verified contract-bound results</text>'
        '</svg>\n'
    )


def _trajectory_svg(benchmark_id: str, points: list[dict[str, Any]]) -> str:
    width, height = 800, 280
    left, right, top, bottom = 76, 24, 46, 54
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [value for point in points for value in (point["iteration_best"], point["best_so_far"])]
    low, high = min(values), max(values)
    pad = max((high - low) * 0.12, 1e-6)
    low, high = low - pad, high + pad

    def x(index: int) -> float:
        return left + (plot_width * index / max(len(points) - 1, 1))

    def y(value: float) -> float:
        return top + (high - value) * plot_height / (high - low)

    current = " ".join(f"{x(i):.1f},{y(point['iteration_best']):.1f}" for i, point in enumerate(points))
    best = " ".join(f"{x(i):.1f},{y(point['best_so_far']):.1f}" for i, point in enumerate(points))
    labels = []
    points_svg = []
    for i, point in enumerate(points):
        px = x(i)
        labels.append(
            f'<text x="{px:.1f}" y="{height - 22}" text-anchor="middle" class="tick">{point["iteration"]}</text>'
        )
        points_svg.append(f'<circle cx="{px:.1f}" cy="{y(point["iteration_best"]):.1f}" r="4" class="current"/>')
        points_svg.append(f'<circle cx="{px:.1f}" cy="{y(point["best_so_far"]):.1f}" r="4" class="best"/>')
    y_ticks = []
    for fraction in (0.0, 0.5, 1.0):
        value = low + fraction * (high - low)
        py = y(value)
        y_ticks.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" class="grid"/>')
        y_ticks.append(f'<text x="{left-10}" y="{py+4:.1f}" text-anchor="end" class="tick">{value:.4g}</text>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="280" viewBox="0 0 800 280">\n'
        '<style>.bg{fill:#fff}.axis{stroke:#24292f;stroke-width:1}.grid{stroke:#d8dee4;stroke-width:1}.current{fill:#cf6b2c}.best{fill:#226b4d}.line1{fill:none;stroke:#cf6b2c;stroke-width:2}.line2{fill:none;stroke:#226b4d;stroke-width:2}.title{font:600 16px sans-serif;fill:#24292f}.tick{font:12px sans-serif;fill:#57606a}.legend{font:12px sans-serif;fill:#24292f}</style>\n'
        f'<rect class="bg" width="{width}" height="{height}"/>\n'
        f'<text x="{left}" y="25" class="title">{escape(benchmark_id)} reward trajectory</text>\n'
        + "\n".join(y_ticks)
        + f'\n<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>'
        + f'\n<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>'
        + f'\n<polyline points="{current}" class="line1"/><polyline points="{best}" class="line2"/>\n'
        + "\n".join(points_svg + labels)
        + f'\n<text x="{width-250}" y="25" class="legend" fill="#cf6b2c">iteration best</text>'
        + f'<text x="{width-130}" y="25" class="legend" fill="#226b4d">best-so-far</text>\n'
        + f'<text x="{width/2}" y="{height-5}" text-anchor="middle" class="tick">iteration</text>\n</svg>\n'
    )


def _format_number(value: float) -> str:
    return f"{float(value):.6g}"


def _format_seed(seed: dict[str, Any]) -> str:
    groups = []
    for key, values in seed.items():
        if key == "aggregation" or not values:
            continue
        groups.append(f"{key}=" + ",".join(str(value) for value in values))
    return "; ".join(groups) + f" / {seed['aggregation']}"


def _format_budget(budget: dict[str, Any]) -> str:
    chunks = []
    for key, value in budget.items():
        if isinstance(value, list):
            text = ",".join(str(item) for item in value)
        else:
            text = str(value)
        chunks.append(f"{key}={text}")
    return "; ".join(chunks)
