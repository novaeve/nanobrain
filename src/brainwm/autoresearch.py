"""Validation-only, ledger-backed search control for BrainWM experiments."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

from .config import load_config

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

SPEC_FORMAT = "brainwm_autoresearch_spec_v1"
RESULT_FORMAT = "brainwm_autoresearch_validation_v1"
LEDGER_FORMAT = "brainwm_autoresearch_ledger_v1"
SNAPSHOT_FORMAT = "brainwm_autoresearch_snapshot_v1"
ALLOWED_STAGES = frozenset(
    {
        "stage0_annotation",
        "stage1_forecasting",
        "stage2_spatial_mcr",
        "stage2_aux_imputation",
    }
)
ALLOWED_PHYSICAL_GPUS = frozenset(range(3, 8))
DEFAULT_MAX_GPU_JOBS = 2
MAX_GPU_JOBS_LIMIT = 3
FORBIDDEN_SCOPE_MARKERS = ("sea-ad", "sea_ad", "seaad")
EXPECTED_REWARD_CONTRACTS = {
    "stage0_annotation": ("coverage_adjusted_macro_f1", "maximize"),
    "stage1_forecasting": ("candidate.mse", "minimize"),
    "stage2_spatial_mcr": ("real.joint_mcr_objective", "minimize"),
    "stage2_aux_imputation": ("candidate.rmse", "minimize"),
}
STAGE_TASKS = {
    "stage0_annotation": "annotation",
    "stage1_forecasting": "development",
    "stage2_spatial_mcr": "integration",
    "stage2_aux_imputation": "integration",
}
STAGE0_NORMALIZATION_AXES = {
    "log1p_library": "pretrain_log1p_library",
    "clr_nonzero": "pretrain_clr_nonzero",
    "log1p_cpm": "pretrain_log1p_cpm",
}
STAGE0_AXIS_PATH_DIMENSIONS = {
    "data.normalization": "data",
    "model.encoder.variational": "architecture",
    "model.encoder.token_context_layers": "architecture",
    "losses.kl": "algorithm",
    "losses.contrastive": "algorithm",
    "losses.assay_adversarial": "algorithm",
    "losses.flow_matching": "algorithm",
    "losses.partition_independence": "algorithm",
    "model.scuniflow_gene_adapter.enabled": "architecture",
    "model.scuniflow_gene_adapter.order_mode": "architecture",
    "model.scuniflow_gene_adapter.gene_order_path": "architecture",
    "model.scuniflow_gene_adapter.ffn_multiplier": "architecture",
    "model.scuniflow_gene_adapter.dropout": "architecture",
    "model.scuniflow_gene_adapter.gate_init": "architecture",
}
# V3/V4 jobs materialized before the scUniFlow adapter paths were added to the
# Stage0 genome.  They are retained as historical evidence and must remain
# readable, but only when the complete old key set is present.  In particular,
# accepting an arbitrary subset here would let a newly-produced receipt omit a
# searched field without being noticed.
LEGACY_STAGE0_AXIS_PATHS = frozenset(
    {
        "data.normalization",
        "model.encoder.variational",
        "model.encoder.token_context_layers",
        "losses.kl",
        "losses.contrastive",
        "losses.assay_adversarial",
        "losses.flow_matching",
        "losses.partition_independence",
    }
)
STRICT_STAGE0_AXIS_SEARCH_IDS = frozenset(
    {"brainwm_autoresearch_v3", "brainwm_autoresearch_v4"}
)


class SearchContractError(RuntimeError):
    """Raised when a search, genome, or validation boundary is invalid."""


@dataclass(frozen=True)
class RewardOutcome:
    feasible: bool
    reward: float | None
    tie_break: tuple[float, ...]
    primary_metric: str
    comparator: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tie_break"] = list(self.tie_break)
        return payload


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _assert_project_native(value: Any, *, context: str) -> None:
    lowered = _canonical(value).lower()
    marker = next(
        (candidate for candidate in FORBIDDEN_SCOPE_MARKERS if candidate in lowered),
        None,
    )
    if marker is not None:
        raise SearchContractError(
            f"{context} contains forbidden non-project scope marker {marker!r}"
        )


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_workspace_path(value: str | Path) -> Path:
    """Resolve paths declared relative to the outer Brain-Word-Model workspace."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WORKSPACE / path
    return path.resolve()


def _resolve_command_path(value: str | Path) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    if not path.is_file():
        raise SearchContractError(f"command executable is missing: {path}")
    return path


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _deep_get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for token in dotted.split("."):
        if not isinstance(current, dict) or token not in current:
            raise SearchContractError(f"unsupported genome path: {dotted}")
        current = current[token]
    return current


def _deep_set(payload: dict[str, Any], dotted: str, value: Any) -> None:
    tokens = dotted.split(".")
    current: Any = payload
    for token in tokens[:-1]:
        if not isinstance(current, dict) or token not in current:
            raise SearchContractError(f"unsupported genome path: {dotted}")
        current = current[token]
    if not isinstance(current, dict) or tokens[-1] not in current:
        raise SearchContractError(f"unsupported genome path: {dotted}")
    current[tokens[-1]] = copy.deepcopy(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SearchContractError(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SearchContractError(f"JSON root must be a mapping: {path}")
    return payload


def _git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "tracked_worktree_dirty": (
            bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        ),
    }


def _safe_name(value: str, *, field: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(value)):
        raise SearchContractError(f"{field} must use lowercase letters/digits/_/-")
    return str(value)


def _space_entry(
    spec: dict[str, Any], stage: str, dimension: str, path: str
) -> dict[str, Any]:
    try:
        entry = spec["stages"][stage]["search_space"][dimension][path]
    except KeyError as exc:
        raise SearchContractError(
            f"unsupported {stage}/{dimension} genome path: {path}"
        ) from exc
    if not isinstance(entry, dict) or not isinstance(entry.get("values"), list):
        raise SearchContractError(f"search-space entry is malformed: {stage}/{path}")
    return entry


def _value_is_allowed(value: Any, allowed: list[Any]) -> bool:
    encoded = _canonical(value)
    return any(encoded == _canonical(candidate) for candidate in allowed)


def _validate_consumers(entry: dict[str, Any], genome_path: str) -> None:
    values = entry.get("values")
    if not isinstance(values, list) or not values:
        raise SearchContractError(f"{genome_path} has no allowed values")
    consumers = entry.get("consumers")
    if not isinstance(consumers, list) or not consumers:
        raise SearchContractError(f"{genome_path} has no verified code consumer")
    for consumer in consumers:
        if not isinstance(consumer, dict):
            raise SearchContractError(f"invalid consumer evidence for {genome_path}")
        source_path = _resolve_project_path(str(consumer.get("path", "")))
        token = str(consumer.get("token", ""))
        if not source_path.is_file() or not token:
            raise SearchContractError(f"consumer evidence is incomplete for {genome_path}")
        if token not in source_path.read_text(encoding="utf-8"):
            raise SearchContractError(
                f"consumer token {token!r} is absent from {source_path}"
            )


def _validate_stage0_benchmark_data(
    *, registry: dict[str, Any], benchmark: dict[str, Any], definition: dict[str, Any], base: dict[str, Any]
) -> None:
    corpus_id = benchmark.get("input_contract", {}).get("pretraining_corpus", {}).get(
        "corpus_id"
    )
    catalog = registry.get("pretraining_corpus_catalog", {})
    corpus = catalog.get(corpus_id)
    if not isinstance(corpus, dict) or corpus.get("frozen") is not True:
        raise SearchContractError("Stage0 benchmark lacks a frozen pretraining corpus")

    data = base.get("data")
    if not isinstance(data, dict):
        raise SearchContractError("Stage0 base config lacks data settings")
    expected_artifact = _resolve_workspace_path(str(corpus.get("artifact_dir", "")))
    expected_vocabulary = _resolve_workspace_path(str(corpus.get("vocabulary_path", "")))
    if Path(str(data.get("artifact_dir", ""))).resolve() != expected_artifact:
        raise SearchContractError("Stage0 artifact differs from the benchmark corpus")
    if Path(str(data.get("vocabulary_path", ""))).resolve() != expected_vocabulary:
        raise SearchContractError("Stage0 vocabulary differs from the benchmark corpus")
    if data.get("split") != corpus.get("split"):
        raise SearchContractError("Stage0 split differs from the benchmark corpus")
    if data.get("assay_types") != corpus.get("assay_types"):
        raise SearchContractError("Stage0 assay cohort differs from the benchmark corpus")

    canonical_normalization = STAGE0_NORMALIZATION_AXES.get(data.get("normalization"))
    foundation_values = (
        benchmark.get("variation_policy", {})
        .get("mutable_axes", {})
        .get("data", {})
        .get("foundation_values", [])
    )
    if canonical_normalization not in foundation_values:
        raise SearchContractError(
            "Stage0 normalization is outside the benchmark Data-axis contract"
        )

    stats_path = corpus.get("stats_path")
    expected_by_assay = corpus.get("expected_cells_by_assay")
    expected_cells = corpus.get("expected_cells")
    if stats_path is not None or expected_by_assay is not None or expected_cells is not None:
        if not isinstance(stats_path, str) or not isinstance(expected_by_assay, dict):
            raise SearchContractError("Stage0 corpus count contract is incomplete")
        stats = _load_json(_resolve_workspace_path(stats_path))
        observed = {
            str(row["assay_type"]): int(row["cells"])
            for row in stats.get("by_split_assay", [])
            if row.get("split") == corpus.get("split")
            and row.get("assay_type") in corpus.get("assay_types", [])
        }
        normalized_expected = {
            str(key): int(value) for key, value in expected_by_assay.items()
        }
        if observed != normalized_expected or sum(observed.values()) != int(expected_cells):
            raise SearchContractError("Stage0 corpus row counts differ from the frozen contract")

    evaluator = definition.get("evaluator")
    if not isinstance(evaluator, dict):
        raise SearchContractError("Stage0 benchmark-bound search lacks evaluator settings")
    project_contract = _load_json(
        _resolve_project_path(str(evaluator.get("base_project_contract", "")))
    )
    project_data = project_contract.get("data", {})
    split = benchmark.get("split", {})
    comparisons = {
        "source_id": (project_data.get("source_id"), split.get("source_id")),
        "train_donors": (project_data.get("train_donors"), split.get("train_donors")),
        "validation_donors": (
            project_data.get("validation_donors"),
            split.get("validation_donors"),
        ),
        "train_rows": (project_data.get("train_rows"), split.get("train_rows")),
        "validation_rows": (
            project_data.get("validation_rows"),
            split.get("validation_rows"),
        ),
    }
    mismatches = [name for name, values in comparisons.items() if values[0] != values[1]]
    if mismatches:
        raise SearchContractError(
            "Stage0 evaluator cohort differs from benchmark split: " + ", ".join(mismatches)
        )
    head = evaluator.get("head", {})
    frozen_evaluator = benchmark.get("evaluator", {})
    if (
        head.get("seed") != frozen_evaluator.get("head_seed")
        or head.get("max_steps") != frozen_evaluator.get("max_steps")
    ):
        raise SearchContractError("Stage0 head seed/budget differs from frozen evaluator")


def _validate_benchmark_bindings(
    spec: dict[str, Any], stages: dict[str, Any], bases: dict[str, dict[str, Any]]
) -> None:
    """Bind new searches to one immutable benchmark instead of labels in prose."""

    bound = spec.get("benchmark_contract_bound", False)
    if not isinstance(bound, bool):
        raise SearchContractError("benchmark_contract_bound must be boolean")
    bindings = [definition.get("benchmark") for definition in stages.values()]
    if not bound:
        if any(binding is not None for binding in bindings):
            raise SearchContractError(
                "stage benchmark bindings require benchmark_contract_bound=true"
            )
        return

    from .benchmarking import benchmark_contract_ids, benchmark_index, load_registry

    for stage, definition in stages.items():
        binding = definition.get("benchmark")
        required = {
            "registry",
            "benchmark_id",
            "data_contract_id",
            "evaluator_contract_id",
        }
        if not isinstance(binding, dict) or set(binding) != required:
            raise SearchContractError(
                f"{stage} benchmark binding must contain exactly {sorted(required)}"
            )
        registry = load_registry(_resolve_project_path(str(binding["registry"])))
        benchmarks = benchmark_index(registry)
        benchmark = benchmarks.get(str(binding["benchmark_id"]))
        if not isinstance(benchmark, dict) or benchmark.get("status") not in {
            "active_frozen",
            "completed_frozen",
        }:
            raise SearchContractError(f"{stage} benchmark must be active_frozen")
        if benchmark.get("task_id") != STAGE_TASKS[stage]:
            raise SearchContractError(f"{stage} is bound to the wrong scientific task")
        contract_ids = benchmark_contract_ids(benchmark)
        if any(binding[key] != contract_ids[key] for key in contract_ids):
            raise SearchContractError(f"{stage} immutable contract ID mismatch")
        reward = definition["reward"]
        metric = benchmark["primary_metric"]
        if reward.get("metric") != metric.get("name") or reward.get("direction") != metric.get("direction"):
            raise SearchContractError(f"{stage} reward differs from the benchmark metric")
        if stage == "stage0_annotation":
            _validate_stage0_benchmark_data(
                registry=registry,
                benchmark=benchmark,
                definition=definition,
                base=bases[stage],
            )


def load_search_spec(path: str | Path) -> dict[str, Any]:
    """Load and fail-closed validate an autoresearch specification."""

    spec_path = _resolve_project_path(path)
    spec = _load_yaml(spec_path)
    if spec.get("format") != SPEC_FORMAT:
        raise SearchContractError(f"unexpected search format in {spec_path}")
    _assert_project_native(spec, context="search spec")
    if spec.get("project_native_only") is not True:
        raise SearchContractError("search must declare project_native_only=true")
    _safe_name(str(spec.get("search_id", "")), field="search_id")

    controller = spec.get("controller")
    if not isinstance(controller, dict):
        raise SearchContractError("controller settings are required")
    configured_max_gpu_jobs = int(controller.get("max_gpu_jobs", -1))
    # Historical searches are fixed at two concurrent jobs.  A new search may
    # explicitly opt into the user's three-way planner, but never exceed the
    # hard project limit.
    if configured_max_gpu_jobs == 3:
        if controller.get("allow_three_gpu_jobs") is not True:
            raise SearchContractError(
                "controller.max_gpu_jobs must be exactly 2 unless "
                "allow_three_gpu_jobs=true"
            )
    elif configured_max_gpu_jobs != DEFAULT_MAX_GPU_JOBS:
        raise SearchContractError("controller.max_gpu_jobs must be exactly 2")
    physical = tuple(int(value) for value in controller.get("physical_gpus", []))
    if not physical or len(set(physical)) != len(physical):
        raise SearchContractError("physical_gpus must be a non-empty unique list")
    if not set(physical).issubset(ALLOWED_PHYSICAL_GPUS):
        raise SearchContractError("physical_gpus must be a subset of CUDA 3-7")
    if int(controller.get("halving_eta", 0)) < 2:
        raise SearchContractError("halving_eta must be at least 2")

    run_root = _resolve_project_path(controller.get("run_root", ""))
    if not _under(run_root, ROOT / "runs"):
        raise SearchContractError("controller.run_root must be under project runs/")
    ledger_path = _resolve_project_path(controller.get("ledger_path", ""))
    if not _under(ledger_path, ROOT / "docs" / "autoresearch"):
        raise SearchContractError(
            "controller.ledger_path must be Git-trackable under docs/autoresearch/"
        )

    stages = spec.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise SearchContractError("at least one stage definition is required")
    unknown_stages = set(stages) - ALLOWED_STAGES
    if unknown_stages:
        raise SearchContractError(f"unsupported stages: {sorted(unknown_stages)}")
    stage_bases: dict[str, dict[str, Any]] = {}
    for stage, definition in stages.items():
        if not isinstance(definition, dict):
            raise SearchContractError(f"stage definition must be a mapping: {stage}")
        base_config = _resolve_project_path(definition.get("base_config", ""))
        runner = _resolve_project_path(definition.get("train_entrypoint", ""))
        if not base_config.is_file() or not runner.is_file():
            raise SearchContractError(f"{stage} base config or runner is missing")
        base = _load_yaml(base_config)
        stage_bases[stage] = base
        _assert_project_native(base, context=f"{stage} base config")
        reward = definition.get("reward")
        expected_reward = EXPECTED_REWARD_CONTRACTS[stage]
        if (
            not isinstance(reward, dict)
            or reward.get("metric") != expected_reward[0]
            or reward.get("direction") != expected_reward[1]
            or reward.get("controls_affect_reward") is not False
            or reward.get("clipping_affects_reward") is not False
        ):
            raise SearchContractError(f"{stage} reward contract is invalid")
        fidelities = definition.get("fidelities")
        if not isinstance(fidelities, list) or not fidelities:
            raise SearchContractError(f"{stage} needs one or more fidelities")
        previous_budget = -1
        for rung, fidelity in enumerate(fidelities):
            if not isinstance(fidelity, dict):
                raise SearchContractError(f"{stage} fidelity {rung} is malformed")
            budget = int(fidelity.get("max_steps", 0))
            seeds = fidelity.get("seeds")
            if budget <= previous_budget or not isinstance(seeds, list) or not seeds:
                raise SearchContractError(
                    f"{stage} fidelities need increasing budgets and seeds"
                )
            if len(set(map(int, seeds))) != len(seeds):
                raise SearchContractError(f"{stage} fidelity seeds must be unique")
            previous_budget = budget
        search_space = definition.get("search_space")
        if not isinstance(search_space, dict):
            raise SearchContractError(f"{stage} search_space is required")
        for dimension in ("data", "architecture", "algorithm"):
            entries = search_space.get(dimension, {})
            if not isinstance(entries, dict):
                raise SearchContractError(f"{stage}/{dimension} must be a mapping")
            for genome_path, entry in entries.items():
                default = _deep_get(base, str(genome_path))
                _validate_consumers(entry, str(genome_path))
                if not _value_is_allowed(default, entry["values"]):
                    raise SearchContractError(
                        f"base value is outside search space: "
                        f"{genome_path}={default!r}"
                    )

        # Optional per-search whitelist for the mutable axes.  This lets a
        # benchmark explicitly freeze cohort/sampling fields while retaining
        # compatibility with older exploratory specifications that predate the
        # whitelist.
        mutable_paths = definition.get("mutable_paths")
        if mutable_paths is not None:
            if not isinstance(mutable_paths, dict):
                raise SearchContractError(f"{stage} mutable_paths must be a mapping")
            for dimension, paths in mutable_paths.items():
                if dimension not in {"data", "architecture", "algorithm"}:
                    raise SearchContractError(
                        f"{stage} mutable_paths has an invalid dimension: {dimension}"
                    )
                if not isinstance(paths, list) or not all(
                    isinstance(value, str) and value for value in paths
                ):
                    raise SearchContractError(
                        f"{stage} mutable_paths.{dimension} must be a list of paths"
                    )
                declared = set(paths)
                searched = set(search_space.get(dimension, {}))
                if searched - declared:
                    raise SearchContractError(
                        f"{stage} search space exposes paths outside mutable_paths: "
                        f"{sorted(searched - declared)}"
                    )

        if definition.get("execution_ready") is True:
            evaluator = _resolve_project_path(
                definition.get("validation_entrypoint", "")
            )
            if not evaluator.is_file():
                raise SearchContractError(
                    f"execution-ready stage {stage} lacks validation entrypoint"
                )
            extra_train_args = definition.get("extra_train_args", [])
            if not isinstance(extra_train_args, list) or not all(
                isinstance(value, str) and value for value in extra_train_args
            ):
                raise SearchContractError(
                    f"execution-ready stage {stage} has invalid extra_train_args"
                )

    _validate_benchmark_bindings(spec, stages, stage_bases)

    candidates = spec.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SearchContractError("candidate genomes are required")
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise SearchContractError("candidate must be a mapping")
        unsupported_candidate_fields = set(candidate) - {"id", "stage", "changes"}
        if unsupported_candidate_fields:
            raise SearchContractError(
                "candidate has unsupported fields: "
                f"{sorted(unsupported_candidate_fields)}"
            )
        candidate_id = _safe_name(str(candidate.get("id", "")), field="candidate id")
        if candidate_id in seen:
            raise SearchContractError(f"duplicate candidate id: {candidate_id}")
        seen.add(candidate_id)
        stage = str(candidate.get("stage", ""))
        if stage not in stages:
            raise SearchContractError(f"candidate {candidate_id} has unknown stage")
        changes = candidate.get("changes", [])
        if not isinstance(changes, list):
            raise SearchContractError(f"candidate {candidate_id} changes must be a list")
        touched: set[str] = set()
        base = _load_yaml(_resolve_project_path(stages[stage]["base_config"]))
        for change in changes:
            if not isinstance(change, dict):
                raise SearchContractError(f"candidate {candidate_id} change is malformed")
            unsupported_change_fields = set(change) - {"dimension", "path", "value"}
            if unsupported_change_fields:
                raise SearchContractError(
                    f"candidate {candidate_id} change has unsupported fields: "
                    f"{sorted(unsupported_change_fields)}"
                )
            dimension = str(change.get("dimension", ""))
            if dimension not in {"data", "architecture", "algorithm"}:
                raise SearchContractError(f"candidate {candidate_id} has bad dimension")
            genome_path = str(change.get("path", ""))
            mutable_paths = stages[stage].get("mutable_paths")
            if mutable_paths is not None and genome_path not in set(
                mutable_paths.get(dimension, [])
            ):
                raise SearchContractError(
                    f"candidate {candidate_id} changes a frozen {dimension} path: "
                    f"{genome_path}"
                )
            if genome_path in touched:
                raise SearchContractError(
                    f"candidate {candidate_id} changes {genome_path} twice"
                )
            touched.add(genome_path)
            entry = _space_entry(spec, stage, dimension, genome_path)
            if not _value_is_allowed(change.get("value"), entry["values"]):
                raise SearchContractError(
                    f"candidate {candidate_id} value is outside search space: "
                    f"{genome_path}={change.get('value')!r}"
                )
            _deep_get(base, genome_path)

    # Optional matched-pair declarations are a small admission boundary for
    # controlled A/B experiments.  They are deliberately validated against
    # the complete searched genome here, before any GPU job is materialized,
    # so a pair cannot silently differ in an unreported field.
    matched_pairs = spec.get("matched_pairs", [])
    if not isinstance(matched_pairs, list):
        raise SearchContractError("matched_pairs must be a list")
    candidate_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    seen_pair_ids: set[str] = set()
    for pair in matched_pairs:
        if not isinstance(pair, dict):
            raise SearchContractError("matched pair must be a mapping")
        required_pair_fields = {"pair_id", "stage", "candidate_ids", "factor_path"}
        if set(pair) != required_pair_fields:
            raise SearchContractError(
                "matched pair fields must be exactly "
                f"{sorted(required_pair_fields)}"
            )
        pair_id = _safe_name(str(pair["pair_id"]), field="matched pair id")
        if pair_id in seen_pair_ids:
            raise SearchContractError(f"duplicate matched pair id: {pair_id}")
        seen_pair_ids.add(pair_id)
        stage = str(pair["stage"])
        candidate_ids = pair["candidate_ids"]
        if (
            stage not in stages
            or not isinstance(candidate_ids, list)
            or len(candidate_ids) != 2
            or len(set(map(str, candidate_ids))) != 2
        ):
            raise SearchContractError("matched pair needs two distinct candidates in one stage")
        candidate_ids = [str(value) for value in candidate_ids]
        if any(
            candidate_id not in candidate_by_id
            or candidate_by_id[candidate_id].get("stage") != stage
            for candidate_id in candidate_ids
        ):
            raise SearchContractError("matched pair candidates are unknown or cross-stage")
        factor_path = str(pair["factor_path"])
        factor_dimension = next(
            (
                dimension
                for dimension in ("data", "architecture", "algorithm")
                if factor_path in stages[stage].get("search_space", {}).get(dimension, {})
            ),
            None,
        )
        if factor_dimension is None:
            raise SearchContractError("matched pair factor_path is outside the search space")

        def candidate_genome(candidate_id: str) -> dict[str, Any]:
            definition = stages[stage]
            base = _load_yaml(_resolve_project_path(definition["base_config"]))
            candidate = candidate_by_id[candidate_id]
            for change in candidate.get("changes", []):
                _deep_set(base, str(change["path"]), change.get("value"))
            return {
                path: _deep_get(base, path)
                for dimension in ("data", "architecture", "algorithm")
                for path in definition["search_space"].get(dimension, {})
            }

        left = candidate_genome(candidate_ids[0])
        right = candidate_genome(candidate_ids[1])
        differing_paths = [
            path for path in sorted(left)
            if _canonical(left[path]) != _canonical(right[path])
        ]
        if differing_paths != [factor_path]:
            raise SearchContractError(
                "matched pair must differ at exactly factor_path; "
                f"observed {differing_paths!r}"
            )
    spec["matched_pairs"] = matched_pairs
    spec["_path"] = str(spec_path)
    return spec


def _require_finite(values: dict[str, Any], fields: tuple[str, ...]) -> list[float]:
    result = []
    for field in fields:
        value = values.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SearchContractError(f"required metric {field!r} is missing")
        number = float(value)
        if not math.isfinite(number):
            raise FloatingPointError(f"metric {field!r} is non-finite")
        result.append(number)
    return result


def _require_nonnegative(values: list[float], *, context: str) -> None:
    if any(value < 0 for value in values):
        raise SearchContractError(f"{context} metrics must be nonnegative")


def _require_closed_access(payload: dict[str, Any], field: str) -> None:
    access = payload.get(field)
    if not isinstance(access, dict) or access.get("opened") is not False:
        raise SearchContractError(f"{field} must explicitly remain closed")


def compact_clipping_diagnostics(value: Any) -> dict[str, Any]:
    """Keep aggregate clipping telemetry while leaving step values in trial evidence."""

    if not isinstance(value, dict):
        return {"role": "diagnostic_only", "affects_reward": False}
    fields = (
        "optimized_batches",
        "gradient_clip_threshold",
        "clipping_fraction",
        "maximum_preclip_gradient_norm",
        "preclip_gradient_distribution",
    )
    compact = {key: copy.deepcopy(value[key]) for key in fields if key in value}
    # Older receipts use ``fraction``; preserve it without copying raw vectors.
    if "fraction" in value and "clipping_fraction" not in compact:
        compact["clipping_fraction"] = value["fraction"]
    compact["role"] = "diagnostic_only"
    compact["affects_reward"] = False
    return compact


def _axis_number(value: float) -> str:
    return format(float(value), ".12g")


def _axis_weight(
    materialized_config: dict[str, Any], path: str, *, nonnegative: bool = True
) -> float:
    value = _deep_get(materialized_config, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"materialized config has non-numeric axis value: {path}")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"materialized config has invalid axis value: {path}")
    return result


def _weighted_axis_token(name: str, value: float, default: float) -> str:
    if value == default:
        return name
    return f"{name}_{_axis_number(value)}"


def canonical_stage0_axes(materialized_config: dict[str, Any]) -> dict[str, str]:
    """Derive readable Stage0 axes from the resolved configuration itself."""

    normalization = _deep_get(materialized_config, "data.normalization")
    data_axis = STAGE0_NORMALIZATION_AXES.get(normalization)
    if data_axis is None:
        raise ValueError(f"unsupported Stage0 normalization axis: {normalization!r}")
    encoder_normalization = _deep_get(
        materialized_config, "model.encoder.expression_normalization"
    )
    if encoder_normalization != normalization:
        raise ValueError(
            "data.normalization and model.encoder.expression_normalization differ"
        )

    variational = _deep_get(materialized_config, "model.encoder.variational")
    context_layers = _deep_get(
        materialized_config, "model.encoder.token_context_layers"
    )
    if not isinstance(variational, bool):
        raise TypeError("model.encoder.variational must be boolean")
    if (
        isinstance(context_layers, bool)
        or not isinstance(context_layers, int)
        or context_layers < 0
    ):
        raise ValueError("model.encoder.token_context_layers must be nonnegative")
    architecture_axis = (
        f"{'variational' if variational else 'deterministic'}_context{context_layers}"
    )

    reconstruction = _axis_weight(materialized_config, "losses.reconstruction")
    kl = _axis_weight(materialized_config, "losses.kl")
    mgm = _axis_weight(materialized_config, "losses.masked_gene_modeling")
    contrastive = _axis_weight(materialized_config, "losses.contrastive")
    assay_adversarial = _axis_weight(
        materialized_config, "losses.assay_adversarial"
    )
    flow_matching = _axis_weight(materialized_config, "losses.flow_matching")
    partition_independence = _axis_weight(
        materialized_config, "losses.partition_independence"
    )
    algorithm_tokens: list[str] = []
    if reconstruction > 0:
        algorithm_tokens.append(_weighted_axis_token("NB", reconstruction, 1.0))
    else:
        algorithm_tokens.append("no_NB")
    if kl > 0:
        algorithm_tokens.append(_weighted_axis_token("KL", kl, 0.01))
    if mgm > 0:
        algorithm_tokens.append(_weighted_axis_token("MGM", mgm, 0.5))
    else:
        algorithm_tokens.append("no_MGM")
    if contrastive > 0:
        objective = str(
            _deep_get(materialized_config, "training.contrastive_objective")
        )
        objective_name = {
            "infonce": "InfoNCE",
            "barlow_twins": "BarlowTwins",
        }.get(objective.lower())
        if objective_name is None:
            objective_name = re.sub(r"[^A-Za-z0-9]+", "_", objective).strip("_")
        if not objective_name:
            raise ValueError("training.contrastive_objective is empty")
        algorithm_tokens.append(f"{objective_name}_{_axis_number(contrastive)}")
    if assay_adversarial > 0:
        algorithm_tokens.append(
            f"assay_adversarial_{_axis_number(assay_adversarial)}"
        )
    if flow_matching > 0:
        algorithm_tokens.append(
            f"flow_matching_{_axis_number(flow_matching)}"
        )
    if partition_independence > 0:
        algorithm_tokens.append(
            f"partition_independence_{_axis_number(partition_independence)}"
        )
    if kl == 0:
        algorithm_tokens.append("no_KL")
    return {
        "data": data_axis,
        "architecture": architecture_axis,
        "algorithm": "_".join(algorithm_tokens),
    }


def _infer_searched_dimensions(
    binding: dict[str, Any], searched_values: dict[str, Any]
) -> dict[str, str]:
    declared = binding.get("searched_dimensions")
    if declared is not None:
        if not isinstance(declared, dict) or set(declared) != set(searched_values):
            raise ValueError(
                "config binding searched_dimensions does not cover searched_values"
            )
        dimensions = {str(path): str(value) for path, value in declared.items()}
    else:
        dimensions = {}
        for path in searched_values:
            if path in STAGE0_AXIS_PATH_DIMENSIONS:
                dimensions[path] = STAGE0_AXIS_PATH_DIMENSIONS[path]
            elif path.startswith("data."):
                dimensions[path] = "data"
            elif path.startswith("model."):
                dimensions[path] = "architecture"
            elif path.startswith(("losses.", "optimizer.", "training.")):
                dimensions[path] = "algorithm"
            else:
                raise ValueError(f"cannot infer search dimension for path: {path}")
    if any(
        value not in {"data", "architecture", "algorithm"}
        for value in dimensions.values()
    ):
        raise ValueError("config binding contains an invalid searched dimension")
    for path, expected in STAGE0_AXIS_PATH_DIMENSIONS.items():
        if path in dimensions and dimensions[path] != expected:
            raise ValueError(f"searched dimension is wrong for path: {path}")
    return dimensions


def _comparison_diagnostics(
    *,
    binding: dict[str, Any],
    searched_values: dict[str, Any],
    searched_dimensions: dict[str, str],
    variation_axis: str,
) -> dict[str, Any]:
    comparison = binding.get("comparison_parent")
    if comparison is None:
        return {
            "status": "unavailable_missing_parent_metadata",
            "parent_id": None,
            "changed_paths": None,
            "changed_dimensions": None,
            "single_factor_status": "not_provable",
        }
    if not isinstance(comparison, dict):
        raise TypeError("comparison_parent metadata must be a mapping")
    if comparison.get("status") == "unavailable_missing_parent_metadata":
        if set(comparison) != {"status"}:
            raise ValueError("unavailable comparison_parent contains unsupported fields")
        return {
            "status": "unavailable_missing_parent_metadata",
            "parent_id": None,
            "changed_paths": None,
            "changed_dimensions": None,
            "single_factor_status": "not_provable",
        }
    if comparison.get("status") != "verified":
        raise ValueError("comparison_parent status is unsupported")
    required = {
        "status",
        "candidate_id",
        "searched_values",
        "searched_dimensions",
        "changed_paths",
        "changed_dimensions",
    }
    if set(comparison) != required:
        raise ValueError("verified comparison_parent metadata has the wrong schema")
    parent_id = comparison.get("candidate_id")
    parent_values = comparison.get("searched_values")
    parent_dimensions = comparison.get("searched_dimensions")
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("verified comparison_parent lacks candidate_id")
    if not isinstance(parent_values, dict) or set(parent_values) != set(searched_values):
        raise ValueError("comparison parent genome does not match candidate paths")
    if parent_dimensions != searched_dimensions:
        raise ValueError("comparison parent search dimensions differ from candidate")
    expected_changes = [
        {
            "dimension": searched_dimensions[path],
            "path": path,
            "parent_value": copy.deepcopy(parent_values[path]),
            "candidate_value": copy.deepcopy(searched_values[path]),
        }
        for path in sorted(searched_values)
        if _canonical(parent_values[path]) != _canonical(searched_values[path])
    ]
    expected_dimensions = sorted(
        {str(change["dimension"]) for change in expected_changes}
    )
    if comparison.get("changed_paths") != expected_changes:
        raise ValueError("comparison_parent changed_paths does not match genome diff")
    if comparison.get("changed_dimensions") != expected_dimensions:
        raise ValueError(
            "comparison_parent changed_dimensions does not match genome diff"
        )
    if not expected_dimensions:
        expected_variation = "reference"
    elif len(expected_dimensions) == 1:
        expected_variation = expected_dimensions[0]
    else:
        raise ValueError(
            "comparison changes multiple dimensions and cannot claim a single variation axis"
        )
    if variation_axis != expected_variation:
        raise ValueError(
            "axis.variation does not match comparison-parent diff: "
            f"expected {expected_variation}, got {variation_axis}"
        )
    return {
        "status": "verified",
        "parent_id": parent_id,
        "changed_paths": expected_changes,
        "changed_dimensions": expected_dimensions,
        "single_factor_status": "verified",
    }


def validate_stage0_axis_provenance(
    *,
    validation: dict[str, Any],
    job: dict[str, Any],
    axis_data: str,
    axis_architecture: str,
    axis_algorithm: str,
    variation_axis: str,
    evaluator_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind result-table axes to searched values and the resolved job config."""

    validation_binding = validation.get("config_binding")
    job_binding = job.get("config_binding")
    if not isinstance(validation_binding, dict) or not isinstance(job_binding, dict):
        raise TypeError("validation/job lacks config_binding")
    if validation_binding != job_binding:
        raise ValueError("validation and job config_binding differ")
    searched_values = validation_binding.get("searched_values")
    if not isinstance(searched_values, dict) or not searched_values:
        raise ValueError("validation result lacks searched_values")
    search_id = validation.get("search_id")
    if search_id in STRICT_STAGE0_AXIS_SEARCH_IDS and set(searched_values) != set(
        STAGE0_AXIS_PATH_DIMENSIONS
    ):
        # Preserve compatibility with the historical V3/V4 materializations,
        # whose frozen genome predates the adapter dimensions above.  The
        # compatibility path is deliberately exact: a missing or extra key is
        # still rejected, and non-legacy search receipts remain fail-closed.
        legacy_shape = (
            validation.get("format") == RESULT_FORMAT
            and set(searched_values) == LEGACY_STAGE0_AXIS_PATHS
        )
        if not legacy_shape:
            raise ValueError(
                f"{search_id} searched_values does not match its frozen Stage0 genome"
            )
    materialized_config = job.get("materialized_config")
    if not isinstance(materialized_config, dict):
        raise TypeError("job lacks materialized_config")
    for path, expected in searched_values.items():
        try:
            observed = _deep_get(materialized_config, str(path))
        except SearchContractError as exc:
            raise ValueError(f"materialized config lacks searched path: {path}") from exc
        if _canonical(observed) != _canonical(expected):
            raise ValueError(
                f"searched_values differs from materialized config at: {path}"
            )

    snapshot = job.get("evaluator_contract_snapshot")
    if snapshot is not None:
        if not isinstance(snapshot, dict):
            raise ValueError("job evaluator_contract_snapshot is malformed")
        if snapshot.get("config_binding") != validation_binding:
            raise ValueError("job evaluator snapshot config_binding differs")
        if snapshot.get("materialized_config") != materialized_config:
            raise ValueError("job evaluator snapshot materialized_config differs")
        if snapshot.get("search_id") not in (None, search_id):
            raise ValueError("job evaluator snapshot search_id differs")
    if evaluator_contract is not None:
        if snapshot is not None and evaluator_contract != snapshot:
            raise ValueError("evaluator contract differs from job snapshot")
        if evaluator_contract.get("config_binding") not in (None, validation_binding):
            raise ValueError("evaluator contract config_binding differs")
        if evaluator_contract.get("materialized_config") not in (
            None,
            materialized_config,
        ):
            raise ValueError("evaluator contract materialized_config differs")

    derived_axes = canonical_stage0_axes(materialized_config)
    provided_axes = {
        "data": axis_data,
        "architecture": axis_architecture,
        "algorithm": axis_algorithm,
    }
    for dimension, expected in derived_axes.items():
        if provided_axes[dimension] != expected:
            raise ValueError(
                f"axis.{dimension} does not match materialized config: "
                f"expected {expected}, got {provided_axes[dimension]}"
            )
    searched_dimensions = _infer_searched_dimensions(
        validation_binding, searched_values
    )
    comparison = _comparison_diagnostics(
        binding=validation_binding,
        searched_values=searched_values,
        searched_dimensions=searched_dimensions,
        variation_axis=variation_axis,
    )
    return {
        "provenance_status": (
            "axis_and_comparison_parent_verified"
            if comparison["status"] == "verified"
            else "axis_verified_parent_unavailable"
        ),
        "derived_axis": derived_axes,
        "searched_values": copy.deepcopy(searched_values),
        "searched_dimensions": searched_dimensions,
        "comparison_parent": comparison,
    }


def validation_reward(payload: dict[str, Any], expected_stage: str) -> RewardOutcome:
    """Compute one metric-first reward without consulting clipping telemetry."""

    try:
        _assert_project_native(payload, context="validation result")
        if payload.get("format") != RESULT_FORMAT:
            raise SearchContractError("validation result format mismatch")
        if payload.get("stage") != expected_stage:
            raise SearchContractError("validation result stage mismatch")
        if not isinstance(payload.get("search_id"), str) or not payload["search_id"]:
            raise SearchContractError("validation result search_id is missing")
        if payload.get("project_native_only") is not True:
            raise SearchContractError("validation result is not project-native")
        if payload.get("status") != "complete":
            return RewardOutcome(
                False, None, (), "feasibility", None, "trial crashed or incomplete"
            )
        if payload.get("split_role") != "validation":
            raise SearchContractError("selection split must be validation")
        if payload.get("selection_scope") != "validation_only":
            raise SearchContractError("selection scope must be validation_only")
        _require_closed_access(payload, "query_access")
        _require_closed_access(payload, "test_access")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise SearchContractError("validation result metrics are missing")

        if expected_stage == "stage0_annotation":
            if payload.get("training_labels_used_for_fit") is not True:
                raise SearchContractError("Stage0 training-label fit must be explicit")
            if payload.get("validation_labels_used_for_fit") is not False:
                raise SearchContractError("Stage0 validation labels entered fitting")
            macro_f1, balanced_accuracy, coverage, adjusted_macro_f1 = _require_finite(
                metrics,
                (
                    "macro_f1_covered",
                    "balanced_accuracy_covered",
                    "coverage",
                    "coverage_adjusted_macro_f1",
                ),
            )
            if not all(0.0 <= value <= 1.0 for value in (macro_f1, balanced_accuracy, coverage)):
                raise SearchContractError("annotation metrics must lie in [0, 1]")
            if not 0.0 <= adjusted_macro_f1 <= 1.0 or not math.isclose(
                adjusted_macro_f1,
                coverage * macro_f1,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                raise SearchContractError(
                    "coverage_adjusted_macro_f1 is inconsistent with coverage"
                )
            return RewardOutcome(
                True,
                adjusted_macro_f1,
                (coverage * balanced_accuracy,),
                "coverage_adjusted_validation_macro_f1",
                None,
            )

        if expected_stage == "stage1_forecasting":
            candidate = metrics.get("candidate")
            controls = metrics.get("controls")
            if not isinstance(candidate, dict) or not isinstance(controls, dict):
                raise SearchContractError("Stage1 candidate/controls are missing")
            candidate_values = _require_finite(
                candidate, ("mse", "sinkhorn_cost", "cosine_distance")
            )
            _require_nonnegative(candidate_values, context="Stage1 candidate")
            required_controls = ("identity", "online_balanced_sinkhorn_ot_only")
            if set(controls) != set(required_controls):
                raise SearchContractError("Stage1 requires the two strict controls")
            control_values = {
                name: _require_finite(
                    controls[name], ("mse", "sinkhorn_cost", "cosine_distance")
                )
                for name in required_controls
            }
            for name, values in control_values.items():
                _require_nonnegative(values, context=f"Stage1 control {name}")
            tie_break = (
                -candidate_values[1],
                -candidate_values[2],
            )
            return RewardOutcome(
                True,
                -candidate_values[0],
                tie_break,
                "negative_validation_population_mse",
                None,
            )

        if expected_stage == "stage2_spatial_mcr":
            real = metrics.get("real")
            controls = metrics.get("controls")
            if not isinstance(real, dict) or not isinstance(controls, dict):
                raise SearchContractError("Stage2 real/controls are missing")
            (real_objective,) = _require_finite(real, ("joint_mcr_objective",))
            _require_nonnegative([real_objective], context="Stage2 real")
            names = ("permutation", "zero", "no_structure")
            if set(controls) != set(names):
                raise SearchContractError("Stage2 requires permutation/zero/no_structure")
            control_values = {
                name: _require_finite(controls[name], ("joint_mcr_objective",))[0]
                for name in names
            }
            for name, value in control_values.items():
                _require_nonnegative([value], context=f"Stage2 control {name}")
            return RewardOutcome(
                True,
                -real_objective,
                (),
                "negative_validation_real_joint_mcr_objective",
                None,
            )

        if expected_stage == "stage2_aux_imputation":
            candidate = metrics.get("candidate")
            controls = metrics.get("controls")
            if not isinstance(candidate, dict) or not isinstance(controls, dict):
                raise SearchContractError("auxiliary imputation metrics are missing")
            candidate_rmse, candidate_mae = _require_finite(candidate, ("rmse", "mae"))
            _require_nonnegative(
                [candidate_rmse, candidate_mae], context="imputation candidate"
            )
            names = ("matched_transparent", "no_context")
            if set(controls) != set(names):
                raise SearchContractError("auxiliary imputation controls are incomplete")
            control_rmse = {
                name: _require_finite(controls[name], ("rmse",))[0] for name in names
            }
            for name, value in control_rmse.items():
                _require_nonnegative([value], context=f"imputation control {name}")
            return RewardOutcome(
                True,
                -candidate_rmse,
                (-candidate_mae,),
                "negative_validation_rmse",
                None,
            )
        raise SearchContractError(f"unsupported reward stage: {expected_stage}")
    except (FloatingPointError, SearchContractError) as exc:
        return RewardOutcome(False, None, (), "feasibility", None, str(exc))


def unit_id(candidate_id: str, rung: int, seed: int) -> str:
    return f"{candidate_id}__r{int(rung)}__s{int(seed)}"


def _parse_ledger(text: str, *, path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise SearchContractError(f"bad ledger object at line {line_number}")
        if event.get("format") != LEDGER_FORMAT:
            raise SearchContractError(f"bad ledger event at line {line_number}")
        events.append(event)
    return events


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _parse_ledger(path.read_text(encoding="utf-8"), path=path)


def _write_ledger_event(handle: TextIO, event: dict[str, Any]) -> None:
    handle.seek(0, 2)
    handle.write(_canonical({"format": LEDGER_FORMAT, **event}) + "\n")
    handle.flush()


@contextmanager
def _ledger_transaction(path: Path) -> Iterator[tuple[list[dict[str, Any]], TextIO]]:
    if not path.is_file():
        raise SearchContractError("search ledger is not initialized")
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        events = _parse_ledger(handle.read(), path=path)
        try:
            yield events, handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_ledger(path: Path, event: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "a"
    with path.open(mode, encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _write_ledger_event(handle, event)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def dry_run_job(job_path: str | Path) -> dict[str, Any]:
    """Validate generated commands and config without starting a training job."""

    path = Path(job_path).resolve()
    job = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(job, dict):
        raise SearchContractError("dry-run job manifest must be a mapping")
    _assert_project_native(job, context="dry-run job")
    if job.get("state") != "prepared_not_launched":
        raise SearchContractError("dry-run accepts only prepared job manifests")
    if job.get("launch_authorized_by_controller") is not False:
        raise SearchContractError("v1 controller must not authorize launches")
    if int(job.get("gpu_processes_started", -1)) != 0:
        raise SearchContractError("job manifest claims a GPU process was started")
    physical_gpu = int(job.get("physical_gpu", -1))
    if physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise SearchContractError("job physical GPU is outside CUDA 3-7")
    if job.get("environment") != {"CUDA_VISIBLE_DEVICES": str(physical_gpu)}:
        raise SearchContractError("job GPU environment is not an exact mapping")
    for output_field in ("train_output_dir", "evaluation_output_dir"):
        if Path(job[output_field]).exists():
            raise SearchContractError(f"fresh {output_field} already exists")
    config = load_config(job["config_path"])
    if Path(config.training.output_dir).resolve() != Path(
        job["train_output_dir"]
    ).resolve():
        raise SearchContractError("materialized config output directory drifted")
    if config.to_dict() != job.get("materialized_config"):
        raise SearchContractError("materialized resolved config snapshot drifted")
    contract = json.loads(Path(job["evaluator_contract"]).read_text(encoding="utf-8"))
    if contract != job.get("evaluator_contract_snapshot"):
        raise SearchContractError("evaluator contract differs from job snapshot")
    _require_closed_access(contract, "query_access")
    _require_closed_access(contract, "test_access")
    if contract.get("materialized_config") != job.get("materialized_config"):
        raise SearchContractError("evaluator materialized config binding drifted")
    if contract.get("config_binding") != job.get("config_binding"):
        raise SearchContractError("evaluator searched-genome binding drifted")
    command_reports = []
    for command_field in ("train_command", "validation_command"):
        command = [str(value) for value in job[command_field]]
        result = subprocess.run(
            [command[0], command[1], "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode != 0:
            raise SearchContractError(
                f"{command_field} help failed: {result.stderr.strip()}"
            )
        declared_options = {token for token in command[2:] if token.startswith("--")}
        missing = sorted(option for option in declared_options if option not in result.stdout)
        if missing:
            raise SearchContractError(
                f"{command_field} contains unsupported options: {missing}"
            )
        command_reports.append(
            {
                "command_field": command_field,
                "entrypoint": command[1],
                "declared_options": sorted(declared_options),
                "help_exit_code": result.returncode,
            }
        )
    return {
        "unit_id": job["unit_id"],
        "config_validated": True,
        "fresh_output_dirs": True,
        "query_access_opened": False,
        "test_access_opened": False,
        "gpu_processes_started": 0,
        "launch_authorized_by_controller": False,
        "commands": command_reports,
    }


class SearchController:
    """Plan and resume validation-only successive-halving experiments."""

    def __init__(self, spec_path: str | Path):
        self.spec = load_search_spec(spec_path)
        self.spec_path = Path(self.spec.pop("_path"))
        controller = self.spec["controller"]
        self.ledger_path = _resolve_project_path(controller["ledger_path"])
        self.run_root = _resolve_project_path(controller["run_root"])
        self.candidates = {
            candidate["id"]: {
                **copy.deepcopy(candidate),
                "generation": 0,
            }
            for candidate in self.spec["candidates"]
        }
        if self.ledger_path.exists():
            self.candidates = self._replay_candidates(self.events())

    def _live_snapshot(self) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        for stage, definition in self.spec["stages"].items():
            base_path = _resolve_project_path(definition["base_config"])
            stage_snapshot: dict[str, Any] = {
                "base_config_path": str(base_path),
                "base_config": _load_yaml(base_path),
                "train_entrypoint": str(
                    _resolve_project_path(definition["train_entrypoint"])
                ),
                "validation_entrypoint": (
                    str(_resolve_project_path(definition["validation_entrypoint"]))
                    if definition.get("validation_entrypoint")
                    else None
                ),
            }
            evaluator = definition.get("evaluator")
            if isinstance(evaluator, dict) and evaluator.get("base_project_contract"):
                contract_path = _resolve_project_path(
                    evaluator["base_project_contract"]
                )
                stage_snapshot["base_project_contract_path"] = str(contract_path)
                stage_snapshot["base_project_contract"] = _load_json(contract_path)
            benchmark_binding = definition.get("benchmark")
            if isinstance(benchmark_binding, dict):
                from .benchmarking import (
                    benchmark_contract_ids,
                    benchmark_index,
                    load_registry,
                )

                registry_path = _resolve_project_path(benchmark_binding["registry"])
                benchmark = benchmark_index(load_registry(registry_path))[
                    benchmark_binding["benchmark_id"]
                ]
                immutable_fields = (
                    "data_contract",
                    "evaluator_contract",
                    "split",
                    "evaluator",
                    "primary_metric",
                    "input_contract",
                    "output_contract",
                    "variation_policy",
                )
                stage_snapshot["benchmark_registry_path"] = str(registry_path)
                stage_snapshot["benchmark_contract"] = {
                    "benchmark_id": benchmark["benchmark_id"],
                    **benchmark_contract_ids(benchmark),
                    **{
                        field: copy.deepcopy(benchmark[field])
                        for field in immutable_fields
                    },
                }
            stages[stage] = stage_snapshot
        snapshot = {
            "format": SNAPSHOT_FORMAT,
            "search_id": self.spec["search_id"],
            "spec_path": str(self.spec_path),
            "spec": copy.deepcopy(self.spec),
            "stages": stages,
        }
        _assert_project_native(snapshot, context="autoresearch snapshot")
        return snapshot

    def _replay_candidates(
        self, events: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        candidates = {
            candidate["id"]: {
                **copy.deepcopy(candidate),
                "generation": 0,
            }
            for candidate in self.spec["candidates"]
        }
        genomes = {
            self._candidate_genome(candidate): candidate_id
            for candidate_id, candidate in candidates.items()
        }
        for event in events:
            if event.get("event") != "candidate_registered":
                continue
            candidate = event.get("candidate")
            if not isinstance(candidate, dict):
                raise SearchContractError(
                    "candidate_registered ledger event is malformed"
                )
            required = {"id", "stage", "parent_id", "generation", "changes"}
            if set(candidate) != required:
                raise SearchContractError(
                    "registered candidate fields do not match the immutable schema"
                )
            candidate_id = _safe_name(
                str(candidate.get("id", "")), field="registered candidate id"
            )
            parent_id = str(candidate.get("parent_id", ""))
            if candidate_id in candidates or parent_id not in candidates:
                raise SearchContractError(
                    f"registered candidate has duplicate id or unknown parent: {candidate_id}"
                )
            parent = candidates[parent_id]
            if (
                event.get("candidate_id") != candidate_id
                or event.get("parent_id") != parent_id
                or event.get("stage") != candidate.get("stage")
                or int(event.get("start_rung", -1)) != 0
                or candidate.get("stage") != parent.get("stage")
                or int(candidate.get("generation", -1))
                != int(parent.get("generation", 0)) + 1
            ):
                raise SearchContractError(
                    f"registered candidate lineage is inconsistent: {candidate_id}"
                )
            changes = candidate.get("changes")
            if not isinstance(changes, list):
                raise SearchContractError(
                    f"registered candidate changes are malformed: {candidate_id}"
                )
            touched: set[str] = set()
            for change in changes:
                if not isinstance(change, dict) or set(change) != {
                    "dimension",
                    "path",
                    "value",
                }:
                    raise SearchContractError(
                        f"registered candidate change is malformed: {candidate_id}"
                    )
                dimension = str(change["dimension"])
                path = str(change["path"])
                if path in touched:
                    raise SearchContractError(
                        f"registered candidate changes {path} twice"
                    )
                touched.add(path)
                entry = _space_entry(
                    self.spec, str(candidate["stage"]), dimension, path
                )
                if not _value_is_allowed(change["value"], entry["values"]):
                    raise SearchContractError(
                        f"registered candidate value is outside search space: {path}"
                    )
            genome = self._candidate_genome(candidate)
            if genome in genomes:
                raise SearchContractError(
                    f"registered candidate duplicates genome of {genomes[genome]}"
                )
            genomes[genome] = candidate_id
            candidates[candidate_id] = copy.deepcopy(candidate)
        return candidates

    def initialize(self) -> dict[str, Any]:
        if self.ledger_path.exists():
            raise FileExistsError(self.ledger_path)
        snapshot = self._live_snapshot()
        event = {
            "event": "search_initialized",
            "search_id": self.spec["search_id"],
            "spec_path": str(self.spec_path),
            "resolved_snapshot": snapshot,
            "git_state": _git_state(),
            "candidate_ids": list(self.candidates),
            "gpu_processes_started": 0,
        }
        append_ledger(self.ledger_path, event, exclusive=True)
        return event

    def _validate_events(self, events: list[dict[str, Any]]) -> None:
        if not events or events[0].get("event") != "search_initialized":
            raise SearchContractError("search ledger is not initialized")
        if events[0].get("search_id") != self.spec["search_id"]:
            raise SearchContractError("ledger search_id does not match spec")
        if any(event.get("search_id") != self.spec["search_id"] for event in events):
            raise SearchContractError("ledger contains a foreign search_id")
        frozen = events[0].get("resolved_snapshot")
        if not isinstance(frozen, dict) or frozen.get("format") != SNAPSHOT_FORMAT:
            raise SearchContractError("frozen autoresearch snapshot format mismatch")
        if frozen != self._live_snapshot():
            raise SearchContractError(
                "live search spec/base/evaluator drifted from frozen snapshot"
            )

    def events(self) -> list[dict[str, Any]]:
        events = read_ledger(self.ledger_path)
        self._validate_events(events)
        return events

    def _unit_states_from(
        self, events: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for event in events[1:]:
            identifier = event.get("unit_id")
            if identifier:
                states[str(identifier)] = event
        return states

    def _unit_states(self) -> dict[str, dict[str, Any]]:
        return self._unit_states_from(self.events())

    def _promoted(
        self,
        stage: str,
        rung: int,
        generation: int = 0,
        *,
        events: list[dict[str, Any]] | None = None,
        candidates: dict[str, dict[str, Any]] | None = None,
    ) -> set[str]:
        candidate_map = self.candidates if candidates is None else candidates
        if rung == 0:
            return {
                candidate_id
                for candidate_id, candidate in candidate_map.items()
                if candidate["stage"] == stage
                and int(candidate.get("generation", 0)) == generation
            }
        return {
            str(event["candidate_id"])
            for event in (self.events() if events is None else events)
            if event.get("event") == "candidate_promoted"
            and event.get("stage") == stage
            and int(event.get("generation", -1)) == generation
            and int(event.get("to_rung", -1)) == rung
        }

    def _eligible_units_from(
        self,
        events: list[dict[str, Any]],
        stage_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        states = self._unit_states_from(events)
        result = []
        for stage, definition in self.spec["stages"].items():
            if stage_filter is not None and stage != stage_filter:
                continue
            if definition.get("execution_ready") is not True:
                continue
            generations = sorted(
                {
                    int(candidate.get("generation", 0))
                    for candidate in self.candidates.values()
                    if candidate["stage"] == stage
                }
            )
            for generation in generations:
                for rung, fidelity in enumerate(definition["fidelities"]):
                    promoted = self._promoted(
                        stage, rung, generation, events=events
                    )
                    for candidate_id in self.candidates:
                        if candidate_id not in promoted:
                            continue
                        for seed in map(int, fidelity["seeds"]):
                            identifier = unit_id(candidate_id, rung, seed)
                            if identifier not in states:
                                result.append(
                                    {
                                        "unit_id": identifier,
                                        "candidate_id": candidate_id,
                                        "stage": stage,
                                        "generation": generation,
                                        "rung": rung,
                                        "seed": seed,
                                        "fidelity": fidelity,
                                    }
                                )
        return result

    def _eligible_units(self, stage_filter: str | None = None) -> list[dict[str, Any]]:
        events = self.events()
        return self._eligible_units_from(events, stage_filter)

    def _materialized_config(self, unit: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        candidate = self.candidates[unit["candidate_id"]]
        definition = self.spec["stages"][unit["stage"]]
        config = _load_yaml(_resolve_project_path(definition["base_config"]))
        for change in candidate.get("changes", []):
            _deep_set(config, change["path"], change["value"])
        config["experiment_name"] = unit["unit_id"]
        config["training"]["seed"] = int(unit["seed"])
        config["training"]["max_steps"] = int(unit["fidelity"]["max_steps"])
        if "validation_batches" in unit["fidelity"]:
            config["training"]["validation_batches"] = int(
                unit["fidelity"]["validation_batches"]
            )
        config["training"]["output_dir"] = str(output_dir)
        _assert_project_native(config, context=f"materialized {unit['unit_id']} config")
        return config

    def _candidate_values(self, candidate: dict[str, Any]) -> dict[str, Any]:
        stage = str(candidate["stage"])
        definition = self.spec["stages"][stage]
        config = _load_yaml(_resolve_project_path(definition["base_config"]))
        for change in candidate.get("changes", []):
            _deep_set(config, change["path"], change["value"])
        return {
            path: _deep_get(config, path)
            for dimension in ("data", "architecture", "algorithm")
            for path in definition["search_space"].get(dimension, {})
        }

    def _candidate_genome(self, candidate: dict[str, Any]) -> str:
        stage = str(candidate["stage"])
        values = self._candidate_values(candidate)
        return _canonical({"stage": stage, "values": values})

    def _config_binding(
        self, unit: dict[str, Any], resolved_config: dict[str, Any]
    ) -> dict[str, Any]:
        candidate = self.candidates[unit["candidate_id"]]
        searched_values = self._candidate_values(candidate)
        searched_dimensions = {
            path: dimension
            for dimension in ("data", "architecture", "algorithm")
            for path in self.spec["stages"][unit["stage"]]["search_space"].get(
                dimension, {}
            )
        }
        parent_id = candidate.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            if parent_id not in self.candidates:
                raise SearchContractError(
                    f"candidate comparison parent is unknown: {parent_id}"
                )
            parent_values = self._candidate_values(self.candidates[parent_id])
            changed_paths = [
                {
                    "dimension": searched_dimensions[path],
                    "path": path,
                    "parent_value": copy.deepcopy(parent_values[path]),
                    "candidate_value": copy.deepcopy(searched_values[path]),
                }
                for path in sorted(searched_values)
                if _canonical(parent_values[path]) != _canonical(searched_values[path])
            ]
            comparison_parent = {
                "status": "verified",
                "candidate_id": parent_id,
                "searched_values": parent_values,
                "searched_dimensions": searched_dimensions,
                "changed_paths": changed_paths,
                "changed_dimensions": sorted(
                    {str(change["dimension"]) for change in changed_paths}
                ),
            }
        else:
            comparison_parent = {
                "status": "unavailable_missing_parent_metadata"
            }
        binding = {
            "experiment_name": resolved_config["experiment_name"],
            "candidate_id": unit["candidate_id"],
            "generation": int(unit["generation"]),
            "rung": int(unit["rung"]),
            "training_seed": int(resolved_config["training"]["seed"]),
            "maximum_steps": int(resolved_config["training"]["max_steps"]),
            "training_output_dir": str(resolved_config["training"]["output_dir"]),
            "searched_values": searched_values,
            "searched_dimensions": searched_dimensions,
            "comparison_parent": comparison_parent,
        }
        # Preserve the controller-level matched-pair admission in every job
        # and evaluator contract.  This is metadata only; it cannot mutate the
        # materialized model configuration or reward.
        matched_pairs = self.spec.get("matched_pairs", [])
        for pair in matched_pairs:
            if unit["candidate_id"] in pair.get("candidate_ids", []):
                binding["matched_pair_id"] = str(pair["pair_id"])
                binding["matched_factor_path"] = str(pair["factor_path"])
                binding["matched_role"] = (
                    "control"
                    if unit["candidate_id"] == pair["candidate_ids"][0]
                    else "treatment"
                )
                break
        return binding

    def mutate(
        self,
        parent_id: str,
        *,
        dimension: str,
        path: str,
        value: Any,
        child_id: str | None = None,
    ) -> dict[str, Any]:
        """Register one deterministic, deduplicated mutation at rung zero."""

        with _ledger_transaction(self.ledger_path) as (events, handle):
            self._validate_events(events)
            candidates = self._replay_candidates(events)
            if parent_id not in candidates:
                raise SearchContractError(f"unknown parent candidate: {parent_id}")
            parent = candidates[parent_id]
            stage = str(parent["stage"])
            source_seals = [
                event
                for event in events
                if event.get("event") == "rung_sealed"
                and event.get("stage") == stage
                and parent_id in event.get("selected_candidate_ids", [])
            ]
            if not source_seals:
                raise SearchContractError(
                    "evolution parent must be selected by a completed sealed rung"
                )
            source_seal = source_seals[-1]
            target_generation = int(parent.get("generation", 0)) + 1
            if any(
                event.get("event") == "rung_sealed"
                and event.get("stage") == stage
                and int(event.get("generation", -1)) == target_generation
                and int(event.get("rung", -1)) == 0
                for event in events
            ):
                raise SearchContractError(
                    "target mutation generation already has a sealed rung-zero cohort"
                )
            if dimension not in {"data", "architecture", "algorithm"}:
                raise SearchContractError("mutation dimension is unsupported")
            entry = _space_entry(self.spec, stage, dimension, path)
            if not _value_is_allowed(value, entry["values"]):
                raise SearchContractError(
                    "mutation value is outside the allowed space"
                )
            base = _load_yaml(
                _resolve_project_path(self.spec["stages"][stage]["base_config"])
            )
            for change in parent.get("changes", []):
                _deep_set(base, change["path"], change["value"])
            if _canonical(_deep_get(base, path)) == _canonical(value):
                raise SearchContractError("mutation does not change the parent genome")

            child_changes = [
                copy.deepcopy(change) for change in parent.get("changes", [])
            ]
            replacement = {
                "dimension": dimension,
                "path": path,
                "value": copy.deepcopy(value),
            }
            for index, change in enumerate(child_changes):
                if change["path"] == path:
                    child_changes[index] = replacement
                    break
            else:
                child_changes.append(replacement)
            child_changes.sort(
                key=lambda change: (change["dimension"], change["path"])
            )
            if child_id is None:
                existing_children = sum(
                    event.get("event") == "candidate_registered"
                    and event.get("parent_id") == parent_id
                    for event in events
                )
                child_id = f"{parent_id}_m{existing_children + 1:03d}"
            child_id = _safe_name(child_id, field="child candidate id")
            if child_id in candidates:
                raise SearchContractError(f"candidate id already exists: {child_id}")
            child = {
                "id": child_id,
                "stage": stage,
                "parent_id": parent_id,
                "generation": target_generation,
                "changes": child_changes,
            }
            genome = self._candidate_genome(child)
            duplicates = [
                candidate_id
                for candidate_id, candidate in candidates.items()
                if self._candidate_genome(candidate) == genome
            ]
            if duplicates:
                raise SearchContractError(
                    f"mutation duplicates existing genome: {sorted(duplicates)}"
                )
            event = {
                "event": "candidate_registered",
                "search_id": self.spec["search_id"],
                "candidate_id": child_id,
                "parent_id": parent_id,
                "stage": stage,
                "start_rung": 0,
                "source_selection": {
                    "generation": int(source_seal["generation"]),
                    "rung": int(source_seal["rung"]),
                },
                "mutation": replacement,
                "candidate": child,
            }
            _write_ledger_event(handle, event)
            candidates[child_id] = child
            self.candidates = candidates
            return event

    def _stage0_evaluator_contract(
        self,
        unit: dict[str, Any],
        config_path: Path,
        checkpoint_path: Path,
        evaluation_dir: Path,
        physical_gpu: int,
        resolved_config: dict[str, Any],
        config_binding: dict[str, Any],
    ) -> dict[str, Any]:
        definition = self.spec["stages"][unit["stage"]]
        evaluator = definition["evaluator"]
        return {
            "format": "brainwm_autoresearch_stage0_annotation_contract_v1",
            "search_id": self.spec["search_id"],
            "unit_id": unit["unit_id"],
            "candidate_id": unit["candidate_id"],
            "generation": int(unit["generation"]),
            "rung": int(unit["rung"]),
            "training_seed": int(unit["seed"]),
            "maximum_candidate_step": int(unit["fidelity"]["max_steps"]),
            "stage0_config": str(config_path),
            "stage0_checkpoint": str(checkpoint_path),
            "output_dir": str(evaluation_dir),
            "base_project_contract": str(
                _resolve_project_path(evaluator["base_project_contract"])
            ),
            "physical_gpu": int(physical_gpu),
            "device": "cuda:0",
            "head": copy.deepcopy(evaluator["head"]),
            "materialized_config": copy.deepcopy(resolved_config),
            "config_binding": copy.deepcopy(config_binding),
            "project_native_only": True,
            "selection_scope": "validation_only",
            "fixed_arm": "aligned_feature",
            "candidate_artifact_pair_is_only_variable": True,
            "training_labels_used_for_fit": True,
            "validation_labels_used_for_fit": False,
            "query_access": {"opened": False},
            "test_access": {"opened": False},
        }

    def _recover_materialized_job(
        self,
        unit: dict[str, Any],
        physical_gpu: int,
        job_dir: Path,
    ) -> dict[str, Any]:
        job_path = job_dir / "job.json"
        if not job_path.is_file():
            raise SearchContractError(
                f"orphan job directory is incomplete and cannot be recovered: {job_dir}"
            )
        job = _load_json(job_path)
        expected = {
            "unit_id": unit["unit_id"],
            "candidate_id": unit["candidate_id"],
            "stage": unit["stage"],
            "generation": unit["generation"],
            "rung": unit["rung"],
            "seed": unit["seed"],
            "physical_gpu": int(physical_gpu),
            "state": "prepared_not_launched",
            "launch_authorized_by_controller": False,
        }
        mismatches = [key for key, value in expected.items() if job.get(key) != value]
        if mismatches:
            raise SearchContractError(
                "recoverable job manifest differs for: " + ", ".join(mismatches)
            )
        for field in ("train_output_dir", "evaluation_output_dir"):
            if Path(job[field]).exists():
                raise SearchContractError(
                    f"orphan job has already materialized {field}; manual review required"
                )
        config = load_config(job["config_path"])
        if config.to_dict() != job.get("materialized_config"):
            raise SearchContractError("recoverable job config snapshot drifted")
        contract = _load_json(Path(job["evaluator_contract"]))
        if contract != job.get("evaluator_contract_snapshot"):
            raise SearchContractError("recoverable evaluator contract drifted")
        return job

    def _materialize(self, unit: dict[str, Any], physical_gpu: int) -> dict[str, Any]:
        definition = self.spec["stages"][unit["stage"]]
        if definition.get("execution_ready") is not True:
            raise SearchContractError(
                f"stage {unit['stage']} is not bound to an executable validator"
            )
        job_dir = self.run_root / "jobs" / unit["unit_id"]
        train_dir = self.run_root / "trials" / unit["unit_id"] / "train"
        evaluation_dir = self.run_root / "trials" / unit["unit_id"] / "evaluation"
        for path in (train_dir, evaluation_dir):
            if path.exists():
                raise FileExistsError(f"fresh output path already exists: {path}")
        if job_dir.exists():
            return self._recover_materialized_job(unit, physical_gpu, job_dir)

        jobs_root = job_dir.parent
        jobs_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{unit['unit_id']}.", dir=jobs_root)
        )
        try:
            config_path = job_dir / "config.yaml"
            temporary_config_path = temporary_dir / "config.yaml"
            config = self._materialized_config(unit, train_dir)
            temporary_config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            resolved_config = load_config(temporary_config_path).to_dict()
            config_binding = self._config_binding(unit, resolved_config)
            python = _resolve_command_path(self.spec["controller"]["python"])
            runner = _resolve_project_path(definition["train_entrypoint"])
            train_command = [
                str(python),
                str(runner),
                "--config",
                str(config_path),
                "--max-steps",
                str(unit["fidelity"]["max_steps"]),
                "--seed",
                str(unit["seed"]),
                "--output-dir",
                str(train_dir),
            ]
            train_command.extend(
                str(value) for value in definition.get("extra_train_args", [])
            )
            evaluator_contract_path = job_dir / "evaluator_contract.json"
            evaluator_contract = self._stage0_evaluator_contract(
                unit,
                config_path,
                train_dir / "best.pt",
                evaluation_dir,
                physical_gpu,
                resolved_config,
                config_binding,
            )
            (temporary_dir / "evaluator_contract.json").write_text(
                json.dumps(evaluator_contract, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            validation_entrypoint = _resolve_project_path(
                definition["validation_entrypoint"]
            )
            validation_command = [
                str(python),
                str(validation_entrypoint),
                "--contract",
                str(evaluator_contract_path),
            ]
            job_path = job_dir / "job.json"
            job = {
                **unit,
                "physical_gpu": int(physical_gpu),
                "environment": {"CUDA_VISIBLE_DEVICES": str(physical_gpu)},
                "train_command": train_command,
                "validation_command": validation_command,
                "config_path": str(config_path),
                "materialized_config": resolved_config,
                "config_binding": config_binding,
                "evaluator_contract": str(evaluator_contract_path),
                "evaluator_contract_snapshot": evaluator_contract,
                "train_output_dir": str(train_dir),
                "evaluation_output_dir": str(evaluation_dir),
                "expected_validation_result": str(
                    evaluation_dir / "validation_result.json"
                ),
                "job_path": str(job_path),
                "gpu_processes_started": 0,
                "state": "prepared_not_launched",
                "launch_authorized_by_controller": False,
            }
            (temporary_dir / "job.json").write_text(
                json.dumps(job, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_dir.rename(job_dir)
            return job
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

    def prepare(
        self,
        *,
        stage: str | None = None,
        limit: int | None = None,
        physical_gpus: Sequence[int] | None = None,
    ) -> list[dict[str, Any]]:
        max_gpu_jobs = int(
            self.spec["controller"].get("max_gpu_jobs", DEFAULT_MAX_GPU_JOBS)
        )
        if limit is None:
            limit = max_gpu_jobs
        if limit < 1 or limit > max_gpu_jobs:
            raise SearchContractError(
                f"prepare limit must be between 1 and {max_gpu_jobs}"
            )
        if stage is not None and stage not in self.spec["stages"]:
            raise SearchContractError(f"unknown stage filter: {stage}")
        if (
            stage is not None
            and self.spec["stages"][stage].get("execution_ready") is not True
        ):
            raise SearchContractError(f"stage {stage} is not execution-ready")
        configured_gpus = tuple(
            int(value) for value in self.spec["controller"]["physical_gpus"]
        )
        if physical_gpus is None:
            gpu_order = configured_gpus
        else:
            requested_gpus = tuple(int(value) for value in physical_gpus)
            if not requested_gpus:
                raise SearchContractError("physical_gpus override cannot be empty")
            if len(set(requested_gpus)) != len(requested_gpus):
                raise SearchContractError(
                    "physical_gpus override must contain unique devices"
                )
            if not set(requested_gpus).issubset(set(configured_gpus)):
                raise SearchContractError(
                    "physical_gpus override must be a subset of controller physical_gpus"
                )
            # This is a scheduler-only preference; it does not alter the
            # frozen search snapshot or any model/data setting.
            gpu_order = requested_gpus + tuple(
                value for value in configured_gpus if value not in requested_gpus
            )
        with _ledger_transaction(self.ledger_path) as (events, handle):
            self._validate_events(events)
            self.candidates = self._replay_candidates(events)
            states = self._unit_states_from(events)
            active = [
                value
                for value in states.values()
                if value.get("event") == "unit_prepared"
            ]
            capacity = min(limit, max_gpu_jobs - len(active))
            if capacity <= 0:
                return []
            used_gpus = {int(value["physical_gpu"]) for value in active}
            available = [
                int(value)
                for value in gpu_order
                if int(value) not in used_gpus
            ]
            jobs = []
            eligible = self._eligible_units_from(events, stage)
            for unit, physical_gpu in zip(
                eligible[:capacity], available, strict=False
            ):
                job = self._materialize(unit, physical_gpu)
                prepared_event = {
                    "event": "unit_prepared",
                    "search_id": self.spec["search_id"],
                    "unit_id": unit["unit_id"],
                    "candidate_id": unit["candidate_id"],
                    "stage": unit["stage"],
                    "generation": unit["generation"],
                    "rung": unit["rung"],
                    "seed": unit["seed"],
                    "physical_gpu": physical_gpu,
                    "job_path": job["job_path"],
                    "job_snapshot": copy.deepcopy(job),
                    "gpu_processes_started": 0,
                    "launch_authorized_by_controller": False,
                }
                _write_ledger_event(handle, prepared_event)
                events.append({"format": LEDGER_FORMAT, **prepared_event})
                jobs.append(job)
            return jobs

    def ingest(self, unit_identifier: str, result_path: str | Path) -> dict[str, Any]:
        with _ledger_transaction(self.ledger_path) as (events, handle):
            self._validate_events(events)
            states = self._unit_states_from(events)
            state = states.get(unit_identifier)
            if state is None or state.get("event") != "unit_prepared":
                raise SearchContractError(
                    "unit is not awaiting validation ingestion"
                )
            job = _load_json(Path(state["job_path"]))
            if job != state.get("job_snapshot"):
                raise SearchContractError("job manifest differs from ledger snapshot")
            if job.get("launch_authorized_by_controller") is not False:
                raise SearchContractError("job launch authorization drifted")
            config = load_config(job["config_path"])
            if config.to_dict() != job.get("materialized_config"):
                raise SearchContractError("job materialized config drifted")
            evaluator_contract = _load_json(Path(job["evaluator_contract"]))
            if evaluator_contract != job.get("evaluator_contract_snapshot"):
                raise SearchContractError("job evaluator contract drifted")
            actual = Path(result_path).resolve()
            expected = Path(job["expected_validation_result"]).resolve()
            if actual != expected or not actual.is_file():
                raise SearchContractError(
                    "validation result path differs from job contract"
                )
            payload = _load_json(actual)
            expected_fields = {
                "search_id": self.spec["search_id"],
                "unit_id": job["unit_id"],
                "candidate_id": job["candidate_id"],
                "stage": job["stage"],
                "generation": job["generation"],
                "rung": job["rung"],
                "seed": job["seed"],
                "config_binding": job["config_binding"],
            }
            for field, expected_value in expected_fields.items():
                if payload.get(field) != expected_value:
                    raise SearchContractError(
                        f"validation result {field} mismatch"
                    )
            protocol = payload.get("protocol")
            if (
                not isinstance(protocol, dict)
                or protocol.get("checkpoint_config_matches_materialized_config")
                is not True
                or protocol.get(
                    "checkpoint_resume_identity_matches_materialized_config"
                )
                is not True
            ):
                raise SearchContractError(
                    "validation result lacks checkpoint/config provenance proof"
                )
            outcome = validation_reward(payload, str(job["stage"]))
            event = {
                "event": "unit_evaluated" if outcome.feasible else "unit_failed",
                "search_id": self.spec["search_id"],
                "unit_id": unit_identifier,
                "candidate_id": job["candidate_id"],
                "stage": job["stage"],
                "generation": job["generation"],
                "rung": job["rung"],
                "seed": job["seed"],
                "physical_gpu": job["physical_gpu"],
                "result_path": str(actual),
                "outcome": outcome.to_dict(),
                "clipping_diagnostics": compact_clipping_diagnostics(
                    payload.get("diagnostics", {}).get("clipping")
                ),
                "clipping_affects_reward": False,
            }
            _write_ledger_event(handle, event)
            return event

    def record_failure(self, unit_identifier: str, reason: str) -> dict[str, Any]:
        with _ledger_transaction(self.ledger_path) as (events, handle):
            self._validate_events(events)
            states = self._unit_states_from(events)
            state = states.get(unit_identifier)
            if state is None or state.get("event") != "unit_prepared":
                raise SearchContractError("unit is not active")
            event = {
                "event": "unit_failed",
                "search_id": self.spec["search_id"],
                "unit_id": unit_identifier,
                "candidate_id": state["candidate_id"],
                "stage": state["stage"],
                "generation": state["generation"],
                "rung": state["rung"],
                "seed": state["seed"],
                "physical_gpu": state["physical_gpu"],
                "outcome": RewardOutcome(
                    False, None, (), "feasibility", None, str(reason)
                ).to_dict(),
            }
            _write_ledger_event(handle, event)
            return event

    def promote(self, stage: str, rung: int, generation: int = 0) -> dict[str, Any]:
        definition = self.spec["stages"].get(stage)
        if definition is None:
            raise SearchContractError(f"unknown stage: {stage}")
        if rung < 0 or rung + 1 >= len(definition["fidelities"]):
            raise SearchContractError("rung has no successor")
        with _ledger_transaction(self.ledger_path) as (events, handle):
            self._validate_events(events)
            if any(
                event.get("event") == "rung_sealed"
                and event.get("stage") == stage
                and int(event.get("generation", -1)) == generation
                and int(event.get("rung", -1)) == rung
                for event in events
            ):
                raise SearchContractError("generation/rung promotion is already sealed")
            candidates_by_id = self._replay_candidates(events)
            states = self._unit_states_from(events)
            seeds = list(map(int, definition["fidelities"][rung]["seeds"]))
            cohort = sorted(
                self._promoted(
                    stage,
                    rung,
                    generation,
                    events=events,
                    candidates=candidates_by_id,
                )
            )
            if not cohort:
                raise SearchContractError("generation/rung cohort is empty")
            rows = []
            for candidate_id in cohort:
                outcomes = []
                for seed in seeds:
                    state = states.get(unit_id(candidate_id, rung, seed))
                    if state is None or state.get("event") not in {
                        "unit_evaluated",
                        "unit_failed",
                    }:
                        raise SearchContractError(
                            "cannot promote before the sealed cohort is complete"
                        )
                    outcome = state.get("outcome", {})
                    if not outcome.get("feasible"):
                        outcomes = []
                        break
                    outcomes.append(outcome)
                if outcomes:
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "reward": float(
                                np.median([row["reward"] for row in outcomes])
                            ),
                            "tie_break": tuple(
                                float(
                                    np.median(
                                        [
                                            row["tie_break"][index]
                                            for row in outcomes
                                        ]
                                    )
                                )
                                for index in range(
                                    len(outcomes[0]["tie_break"])
                                )
                            ),
                        }
                    )
            rows.sort(key=lambda row: row["candidate_id"])
            rows.sort(
                key=lambda row: (row["reward"], row["tie_break"]), reverse=True
            )
            keep = max(
                1,
                math.ceil(
                    len(cohort) / int(self.spec["controller"]["halving_eta"])
                ),
            )
            selected = rows[:keep]
            selected_ids = [row["candidate_id"] for row in selected]
            seal = {
                "event": "rung_sealed",
                "search_id": self.spec["search_id"],
                "stage": stage,
                "generation": generation,
                "rung": rung,
                "candidate_ids": cohort,
                "denominator": len(cohort),
                "required_seeds": seeds,
                "selected_candidate_ids": selected_ids,
                "infeasible_or_not_selected": sorted(
                    set(cohort) - set(selected_ids)
                ),
            }
            _write_ledger_event(handle, seal)
            for row in selected:
                _write_ledger_event(
                    handle,
                    {
                        "event": "candidate_promoted",
                        "search_id": self.spec["search_id"],
                        "candidate_id": row["candidate_id"],
                        "stage": stage,
                        "generation": generation,
                        "from_rung": rung,
                        "to_rung": rung + 1,
                        "aggregate_reward": row["reward"],
                        "aggregate_tie_break": list(row["tie_break"]),
                    },
                )
            return {
                "stage": stage,
                "generation": generation,
                "from_rung": rung,
                "to_rung": rung + 1,
                "denominator": len(cohort),
                "selected": selected,
                "infeasible_or_not_selected": seal[
                    "infeasible_or_not_selected"
                ],
            }

    def seal_terminal(self, stage: str, rung: int, generation: int = 0) -> dict[str, Any]:
        """Seal the final fidelity so a completed search can spawn mutations.

        ``promote`` intentionally requires a successor rung.  A terminal rung
        still needs an immutable selection event, however, otherwise
        ``mutate`` has no auditable parent after a confirmation run.  This
        method writes only ``rung_sealed`` (no synthetic successor units).
        """

        definition = self.spec["stages"].get(stage)
        if definition is None:
            raise SearchContractError(f"unknown stage: {stage}")
        fidelities = definition.get("fidelities", [])
        if rung < 0 or rung >= len(fidelities):
            raise SearchContractError("rung is outside the stage fidelities")
        if rung + 1 < len(fidelities):
            raise SearchContractError("seal_terminal accepts only the final rung")
        with _ledger_transaction(self.ledger_path) as (events, handle):
            self._validate_events(events)
            if any(
                event.get("event") == "rung_sealed"
                and event.get("stage") == stage
                and int(event.get("generation", -1)) == generation
                and int(event.get("rung", -1)) == rung
                for event in events
            ):
                raise SearchContractError("generation/rung is already sealed")
            candidates_by_id = self._replay_candidates(events)
            states = self._unit_states_from(events)
            seeds = list(map(int, fidelities[rung]["seeds"]))
            cohort = sorted(
                self._promoted(
                    stage,
                    rung,
                    generation,
                    events=events,
                    candidates=candidates_by_id,
                )
            )
            if not cohort:
                raise SearchContractError("generation/rung cohort is empty")

            rows: list[dict[str, Any]] = []
            for candidate_id in cohort:
                outcomes: list[dict[str, Any]] = []
                for seed in seeds:
                    state = states.get(unit_id(candidate_id, rung, seed))
                    if state is None or state.get("event") not in {
                        "unit_evaluated",
                        "unit_failed",
                    }:
                        raise SearchContractError(
                            "cannot seal before the terminal cohort is complete: "
                            f"{unit_id(candidate_id, rung, seed)}"
                        )
                    outcome = state.get("outcome", {})
                    if not outcome.get("feasible"):
                        outcomes = []
                        break
                    outcomes.append(outcome)
                if outcomes:
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "reward": float(
                                np.median([row["reward"] for row in outcomes])
                            ),
                            "tie_break": tuple(
                                float(
                                    np.median(
                                        [
                                            row["tie_break"][index]
                                            for row in outcomes
                                        ]
                                    )
                                )
                                for index in range(len(outcomes[0]["tie_break"]))
                            ),
                        }
                    )
            rows.sort(key=lambda row: row["candidate_id"])
            rows.sort(
                key=lambda row: (row["reward"], row["tie_break"]), reverse=True
            )
            keep = max(
                1,
                math.ceil(
                    len(cohort) / int(self.spec["controller"]["halving_eta"])
                ),
            )
            selected = rows[:keep]
            selected_ids = [row["candidate_id"] for row in selected]
            seal = {
                "event": "rung_sealed",
                "search_id": self.spec["search_id"],
                "stage": stage,
                "generation": generation,
                "rung": rung,
                "terminal": True,
                "candidate_ids": cohort,
                "denominator": len(cohort),
                "required_seeds": seeds,
                "selected_candidate_ids": selected_ids,
                "aggregate_rewards": selected,
                "infeasible_or_not_selected": sorted(
                    set(cohort) - set(selected_ids)
                ),
            }
            _write_ledger_event(handle, seal)
            return {
                "stage": stage,
                "generation": generation,
                "rung": rung,
                "terminal": True,
                "denominator": len(cohort),
                "selected": selected,
                "infeasible_or_not_selected": seal[
                    "infeasible_or_not_selected"
                ],
            }

    def status(self) -> dict[str, Any]:
        states = self._unit_states()
        counts: dict[str, int] = {}
        for state in states.values():
            counts[state["event"]] = counts.get(state["event"], 0) + 1
        return {
            "search_id": self.spec["search_id"],
            "ledger_path": str(self.ledger_path),
            "events": len(self.events()),
            "unit_state_counts": counts,
            "pending_units": self._eligible_units(),
            "gpu_processes_started_by_controller": 0,
        }
