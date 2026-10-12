"""Fail-closed contract for the versioned BrainWM global benchmark.

The global split artifact is a metadata index, while this module defines how
that index may be consumed by training and evaluators.  Validation deliberately
does not open expression matrices, label columns, or sealed test manifests.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


CONTRACT_FORMAT = "brainwm_global_multispecies_benchmark_contract_v1"
GLOBAL_SPLIT_FORMAT = "brainwm_global_benchmark_split_v1"
GLOBAL_ACCESS_FORMAT = "brainwm_global_test_access_policy_v1"
UNIFIED_VOCAB_FORMATS = {
    "brainwm_unified_gene_vocabulary_v1",
    "brainwm_unified_gene_vocabulary_v2",
}
TASKS = ("integration", "annotation", "development", "spatial", "perturbation")
ACTIVE_TASKS = TASKS[:-1]
PARTITIONS = ("train", "validation", "test")
METRICS = {
    "integration": ("scib_total_project_v1", "maximize"),
    "annotation": ("coverage_adjusted_macro_f1", "maximize"),
    "development": ("population_mse", "minimize"),
    "spatial": ("masked_cell_nb_nll", "minimize"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path(value: str | Path, project_root: Path) -> Path:
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path} must contain a JSON object")
    return payload


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and schema-check a tracked contract without opening data arrays."""

    contract_path = Path(path).resolve()
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    project_root = contract_path.parents[2]
    validate_contract(payload, project_root=project_root, check_paths=False)
    return payload


def validate_contract(
    contract: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    check_paths: bool = False,
) -> dict[str, Any]:
    """Validate schema and, optionally, small metadata artifacts.

    ``check_paths=True`` opens only JSON manifests and access policies.  It
    never opens H5AD files, expression arrays, labels, or the sealed test row
    manifests themselves.
    """

    root = Path(project_root or ".").resolve()
    _require(isinstance(contract, Mapping), "global benchmark contract must be a mapping")
    _require(contract.get("format") == CONTRACT_FORMAT, "unsupported global benchmark contract format")
    _require(bool(contract.get("contract_id")), "contract_id is required")
    _require(contract.get("status") == "protocol_ready_data_blocked", "current contract must remain data-blocked")
    _require(contract.get("rankable") is False, "current global contract may not be rankable")
    _require(contract.get("zero_shot_claim_allowed") is False, "zero-shot claim must remain disabled")

    data = contract.get("data")
    _require(isinstance(data, Mapping), "data section is required")
    for field in (
        "global_artifact",
        "global_split_id",
        "split_seed",
        "unified_gene_vocabulary",
        "unified_gene_vocabulary_id",
        "row_identity",
        "corpora",
    ):
        _require(data.get(field) not in (None, "", []), f"data.{field} is required")
    _require(
        data.get("row_identity") == ["corpus_id", "local_source_id", "row_index"],
        "global row identity must be corpus_id,local_source_id,row_index",
    )
    _require(data.get("expression_copied") is False, "global benchmark must remain zero-copy")
    _require(data.get("labels_copied") is False, "global benchmark may not copy labels")
    _require(int(data.get("n_genes", 0)) > 0, "data.n_genes must be positive")

    split = data.get("split")
    _require(isinstance(split, Mapping), "data.split is required")
    _require(list(split) == list(PARTITIONS), "split must list train, validation, test in order")
    _require(
        all(abs(float(split[name]) - expected) < 1.0e-12 for name, expected in zip(PARTITIONS, (0.6, 0.2, 0.2))),
        "global split fractions must remain 0.6/0.2/0.2",
    )
    corpora = data.get("corpora")
    _require(isinstance(corpora, Mapping) and corpora, "data.corpora must be non-empty")
    _require(set(corpora) == {"human_molecular", "mouse_molecular", "mouse_spatial"}, "unexpected corpus set")
    for corpus_id, spec in corpora.items():
        _require(isinstance(spec, Mapping), f"corpus {corpus_id} must be a mapping")
        for field in ("artifact_dir", "species", "reader_kind", "use_status"):
            _require(spec.get(field) not in (None, ""), f"corpus {corpus_id}.{field} is required")
        _require(spec.get("species") in {"human", "mouse"}, f"corpus {corpus_id} has invalid species")
        _require(spec.get("use_status") in {"formal_human_only", "engineering_only"}, f"corpus {corpus_id} has invalid use_status")
        if corpus_id.startswith("mouse"):
            _require(spec.get("formal_cross_species_training_allowed") is False, f"{corpus_id} must remain non-formal")
            _require(bool(spec.get("blockers")), f"{corpus_id} must declare blockers")

    reader = contract.get("reader")
    _require(isinstance(reader, Mapping), "reader section is required")
    _require(
        reader.get("class") == "brainwm_data.GlobalMultiCorpusReader",
        "global reader class is not fixed",
    )
    _require(reader.get("require_explicit_unified_vocabulary") is True, "unified vocabulary must be explicit")
    _require(reader.get("allow_unmapped_unified_genes") is False, "unmapped genes may not be silently dropped")
    _require(reader.get("source_local_batches") is True, "expression reads must remain source-local")

    roles = contract.get("access_policy")
    _require(isinstance(roles, Mapping) and set(roles) == set(PARTITIONS), "access policy must define all partitions")
    _require(roles["train"].get("model_fit") is True, "train partition must be the only fit partition")
    _require(roles["validation"].get("model_fit") is False, "validation may not fit the model")
    _require(roles["test"].get("model_fit") is False, "test may not fit the model")
    _require(roles["test"].get("state") == "sealed", "test partition must remain sealed")
    _require(roles["test"].get("expression") == "evaluator_only_after_prediction_freeze", "test expression access is unsafe")
    _require(roles["test"].get("labels") == "evaluator_only_after_prediction_freeze", "test label access is unsafe")

    task_lanes = contract.get("task_lanes")
    _require(isinstance(task_lanes, Mapping) and set(task_lanes) == set(TASKS), "all five task lanes are required")
    for task in ACTIVE_TASKS:
        lane = task_lanes[task]
        _require(isinstance(lane, Mapping), f"task lane {task} must be a mapping")
        expected_metric, expected_direction = METRICS[task]
        _require(lane.get("primary_metric") == expected_metric, f"{task} primary metric drifted")
        _require(lane.get("direction") == expected_direction, f"{task} metric direction drifted")
        _require(lane.get("split_id") == data.get("global_split_id"), f"{task} split binding drifted")
        _require(lane.get("test_access") == "sealed", f"{task} test access must remain sealed")
        _require(lane.get("target_labels_used_for_fit") is False, f"{task} may not fit target labels")
        _require(bool(lane.get("input_contract")), f"{task} input contract is required")
        _require(bool(lane.get("output_contract")), f"{task} output contract is required")
        baselines = lane.get("baselines")
        _require(isinstance(baselines, list) and len(baselines) >= 2, f"{task} needs matched baselines")
        _require(len(baselines) == len(set(baselines)), f"{task} repeats a baseline")
    perturbation = task_lanes["perturbation"]
    _require(perturbation.get("status") == "blocked_data", "perturbation must remain data-blocked")
    _require(perturbation.get("primary_metric") is None, "perturbation cannot declare an unsupported metric")

    evaluator = contract.get("evaluator")
    _require(isinstance(evaluator, Mapping), "evaluator section is required")
    _require(evaluator.get("fixed") is True, "evaluator must be fixed")
    _require(evaluator.get("split_id") == data.get("global_split_id"), "evaluator split binding drifted")
    _require(evaluator.get("current_phase") == "validation_only", "current evaluator phase must remain validation-only")
    _require(evaluator.get("test_opened") is False, "test must remain unopened")
    _require(evaluator.get("cross_task_reward_aggregation") == "forbidden", "cross-task reward aggregation is forbidden")
    _require(evaluator.get("clipping_affects_reward") is False, "clipping may not affect reward")

    search = contract.get("search_space")
    _require(isinstance(search, Mapping), "search_space is required")
    _require(set(search) == {"data", "architecture", "algorithm"}, "search space must contain only data/architecture/algorithm")
    _require(all(bool(search[axis]) for axis in search), "each search axis must be non-empty")

    execution = contract.get("execution")
    _require(isinstance(execution, Mapping), "execution section is required")
    _require(execution.get("training_enabled") is False, "training must remain disabled while data blockers exist")
    _require(execution.get("test_evaluation_enabled") is False, "test evaluation must remain disabled")
    _require(bool(execution.get("blockers")), "execution blockers must be explicit")
    _require(execution.get("refuse_overwrite") is True, "new runs must refuse overwrite")

    checked: dict[str, Any] = {
        "valid": True,
        "contract_id": contract["contract_id"],
        "global_split_id": data["global_split_id"],
        "rankable": False,
        "training_enabled": False,
        "test_opened": False,
        "expression_opened": False,
        "labels_opened": False,
        "sealed_manifest_opened": False,
        "task_count": len(task_lanes),
        "blockers": list(execution["blockers"]),
    }
    if not check_paths:
        return checked

    global_root = _path(data["global_artifact"], root)
    vocabulary_root = _path(data["unified_gene_vocabulary"], root)
    for required in ("manifest.json", "split_receipt.json", "access_policy.json", "cell_manifest.json"):
        _require((global_root / required).is_file(), f"missing global artifact file: {required}")
    # Existence is checked, but sealed_test_samples/groups are never opened.
    for sealed_name in ("sealed_test_samples.parquet", "sealed_test_groups.parquet"):
        _require((global_root / sealed_name).is_file(), f"missing sealed boundary file: {sealed_name}")
    manifest = _json(global_root / "manifest.json")
    receipt = _json(global_root / "split_receipt.json")
    access = _json(global_root / "access_policy.json")
    _require(manifest.get("format") == GLOBAL_SPLIT_FORMAT, "global artifact format mismatch")
    _require(manifest.get("multi_corpus") is True, "global artifact is not multi-corpus")
    _require(manifest.get("seed") == data["split_seed"], "global split seed drifted")
    _require(receipt.get("split_id") == data["global_split_id"], "global split id drifted")
    _require(receipt.get("checkpoint_modified") is False, "global split modified a checkpoint")
    _require(receipt.get("expression_copied") is False and receipt.get("labels_copied") is False, "global split copied payload data")
    _require(manifest.get("source_cell_audit", {}).get("status") == "passed", "global source row audit did not pass")
    _require(access.get("format") == GLOBAL_ACCESS_FORMAT and access.get("sealed") is True, "global test access policy is not sealed")
    _require(access.get("same_split_for_all_tasks") is True, "task splits are not shared")
    _require(access.get("release_condition") == "open target labels/expression once all representations and predictions are frozen", "test release condition drifted")

    manifest_corpora = {item["corpus_id"]: item for item in manifest.get("corpora", [])}
    _require(set(manifest_corpora) == set(corpora), "contract corpus set differs from artifact")
    for corpus_id, spec in corpora.items():
        artifact = _path(spec["artifact_dir"], root)
        _require(artifact.is_dir(), f"missing corpus artifact: {corpus_id}")
        for required in ("sources.parquet", "samples.parquet", "cells.parquet", "genes.parquet", "source_gene_map.parquet"):
            _require((artifact / required).is_file(), f"{corpus_id} missing {required}")
        _require(Path(manifest_corpora[corpus_id]["artifact_dir"]).resolve() == artifact, f"{corpus_id} artifact binding drifted")
        local_manifest = artifact / "manifest.json"
        if local_manifest.is_file():
            local = _json(local_manifest)
            if corpus_id == "mouse_molecular":
                _require(local.get("candidate_only") is True, "mouse molecular candidate status changed")
                _require(local.get("formal_training_allowed") is False, "mouse molecular formal status changed")

    _require((vocabulary_root / "manifest.json").is_file(), "unified vocabulary manifest is missing")
    _require((vocabulary_root / "genes.parquet").is_file(), "unified vocabulary genes are missing")
    vocabulary = _json(vocabulary_root / "manifest.json")
    _require(vocabulary.get("format") in UNIFIED_VOCAB_FORMATS, "unified vocabulary format mismatch")
    total_genes = vocabulary.get("append", {}).get("total_genes") or vocabulary.get("stats", {}).get("total_genes")
    _require(int(total_genes or 0) == int(data["n_genes"]), "unified vocabulary size drifted")
    _require(vocabulary.get("policy", {}).get("human_prefix_unchanged") is True, "human gene prefix is not preserved")

    expected_counts = data.get("expected_counts", {})
    observed_counts = manifest.get("counts", {})
    for key in ("source_samples", "selected_groups", "selected_cells"):
        _require(int(observed_counts.get(key, -1)) == int(expected_counts.get(key, -2)), f"global count drifted: {key}")
    checked["global_artifact"] = str(global_root)
    checked["unified_gene_vocabulary"] = str(vocabulary_root)
    checked["selected_cells"] = int(observed_counts["selected_cells"])
    checked["source_audit_status"] = "passed"
    return checked


def materialize_contract(
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a small immutable snapshot and validation receipt."""

    source = Path(contract_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract(source)
    project_root = source.parents[2]
    validation = validate_contract(contract, project_root=project_root, check_paths=True)
    generated = datetime.now(timezone.utc).isoformat()
    snapshot = dict(contract)
    snapshot["materialized_at_utc"] = generated
    snapshot["source_contract"] = str(source)
    snapshot["git_commit"] = _git_commit(project_root)
    (output / "contract.snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "format": "brainwm_global_multispecies_benchmark_validation_receipt_v1",
        "generated_at_utc": generated,
        "source_contract": str(source),
        "output_dir": str(output),
        "git_commit": snapshot["git_commit"],
        "validation": validation,
        "checks": {
            "metadata_only": True,
            "expression_opened": False,
            "labels_opened": False,
            "sealed_manifest_opened": False,
            "test_opened": False,
            "checkpoint_modified": False,
            "large_file_hashes_computed": False,
        },
    }
    (output / "validation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


__all__ = [
    "ACTIVE_TASKS",
    "CONTRACT_FORMAT",
    "METRICS",
    "PARTITIONS",
    "TASKS",
    "load_contract",
    "materialize_contract",
    "validate_contract",
]
