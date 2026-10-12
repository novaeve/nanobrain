"""Build and validate the BrainWM cross-task split contract.

This module deliberately works on *metadata and existing manifests only*.  It
does not open H5AD matrices, expression columns, labels, checkpoints, or any
sealed target.  The resulting directory is a small, append-only contract that
lets later benchmark runners share one role ledger without silently changing
an existing benchmark split.

The contract is an index of the current contracts, not a new random split.
Every physical row keeps a stable ``global_row_id``; a role-specific
``ledger_entry_id`` allows the same physical row to be reported when it is
exposed by more than one historical benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # PyYAML is part of the project environment; keep import errors clear.
    import yaml
except Exception as exc:  # pragma: no cover - exercised only in a broken env
    yaml = None
    _YAML_IMPORT_ERROR = exc


GLOBAL_SPLIT_CONTRACT_FORMAT = "brainwm_global_split_contract_v1"
GLOBAL_SPLIT_SCHEMA_VERSION = "1.0.0"
BUILDER_VERSION = "1.0.0"
PARTITIONS = ("train", "validation", "query", "test")
TASKS = ("integration", "annotation", "development", "spatial", "auxiliary", "pretraining")

# Keep this list explicit.  It is also used to guarantee that the sealed JSON
# never accidentally contains an expression/label column added in the future.
LEDGER_COLUMNS = (
    "ledger_entry_id",
    "global_row_id",
    "identity_scope",
    "unit_type",
    "source_id",
    "dataset_id",
    "sample_id",
    "sample_key",
    "donor_id",
    "subject_id",
    "cell_or_spot_id",
    "row_index",
    "assay_type",
    "species",
    "platform",
    "section_id",
    "region",
    "time",
    "partition",
    "task_id",
    "benchmark_id",
    "task_role",
    "role_side",
    "panel_id",
    "gene_vocab_id",
    "feature_order_id",
    "hidden_observed_mask_id",
    "normalization_id",
    "normalization_fit_scope",
    "temporal_descriptor_id",
    "source_time_days",
    "target_time_days",
    "delta_time_days",
    "coordinate_registration_id",
    "label_vocab_id",
    "source_fresh",
    "row_position_fresh",
    "sample_fresh",
    "donor_fresh",
    "historical_exposure_ref",
    "sealed_reveal_event",
    "receipt_id",
    "expression_access",
    "label_access",
    "metadata_access",
    "coordinate_access",
    "access_state",
    "declared_count",
)

SAFE_SEALED_COLUMNS = (
    "global_row_id",
    "identity_scope",
    "unit_type",
    "source_id",
    "dataset_id",
    "sample_id",
    "sample_key",
    "donor_id",
    "subject_id",
    "cell_or_spot_id",
    "row_index",
    "assay_type",
    "species",
    "platform",
    "section_id",
    "region",
    "time",
    "partition",
    "task_id",
    "benchmark_id",
    "task_role",
    "role_side",
    "panel_id",
    "gene_vocab_id",
    "feature_order_id",
    "hidden_observed_mask_id",
    "normalization_id",
    "normalization_fit_scope",
    "temporal_descriptor_id",
    "source_time_days",
    "target_time_days",
    "delta_time_days",
    "coordinate_registration_id",
    "label_vocab_id",
    "source_fresh",
    "row_position_fresh",
    "sample_fresh",
    "donor_fresh",
    "historical_exposure_ref",
    "sealed_reveal_event",
    "receipt_id",
    "access_state",
    "declared_count",
)

# Repeated panel/normalization strings belong in the full ledger and manifest;
# repeating them for every sealed row needlessly inflates the reveal artifact.
# The sealed manifest therefore carries this compact, sufficient metadata view.
SEALED_MINIMAL_COLUMNS = (
    "global_row_id",
    "identity_scope",
    "unit_type",
    "source_id",
    "dataset_id",
    "sample_id",
    "sample_key",
    "donor_id",
    "subject_id",
    "cell_or_spot_id",
    "row_index",
    "assay_type",
    "species",
    "platform",
    "section_id",
    "region",
    "time",
    "partition",
    "task_id",
    "benchmark_id",
    "task_role",
    "role_side",
    "temporal_descriptor_id",
    "source_time_days",
    "target_time_days",
    "delta_time_days",
    "coordinate_registration_id",
    "source_fresh",
    "row_position_fresh",
    "sample_fresh",
    "donor_fresh",
    "historical_exposure_ref",
    "sealed_reveal_event",
    "receipt_id",
    "access_state",
    "declared_count",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")
    value = str(value).strip()
    return None if value == "" or value.lower() in {"nan", "none", "null", "<na>", "na"} else value


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return isinstance(missing, (bool, np.bool_)) and bool(missing)
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> int | float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return int(result) if result.is_integer() else result


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value)
    if text is None:
        return None
    if text.lower() in {"true", "yes", "1", "y"}:
        return True
    if text.lower() in {"false", "no", "0", "n"}:
        return False
    return None


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-safe values."""
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _stable_id(prefix: str, *parts: Any) -> str:
    """Create a deterministic identifier from small metadata keys only."""
    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    # UUID5 is deterministic and readable in receipts; no data-file hashing is
    # performed by this builder.
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, payload)}"


def _repo_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit or None


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError(f"PyYAML is unavailable: {_YAML_IMPORT_ERROR}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _snapshot_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for benchmark_id, raw in (registry.get("benchmark_snapshots") or {}).items():
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
        else:
            parsed = raw
        if isinstance(parsed, dict):
            snapshots[str(benchmark_id)] = parsed
    return snapshots


def _task_lookup(task_registry: Mapping[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for task_name, task in (task_registry.get("tasks") or {}).items():
        for benchmark in (task or {}).get("benchmarks", []) or []:
            if isinstance(benchmark, dict) and benchmark.get("benchmark_id"):
                lookup[str(benchmark["benchmark_id"])] = str(task_name)
    for auxiliary in task_registry.get("auxiliary_benchmarks", []) or []:
        if isinstance(auxiliary, dict) and auxiliary.get("benchmark_id"):
            lookup[str(auxiliary["benchmark_id"])] = str(auxiliary.get("task_id", "auxiliary"))
    return lookup


def _slug(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)[:240]


def _path_value(value: Any, repo_root: Path) -> str | None:
    text = _text(value)
    if text is None:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = repo_root / text
    # Keep paths relative where possible so the contract can move with the repo.
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _partition_for(value: Any, default: str | None = None) -> str | None:
    text = _text(value)
    if text is None:
        return default
    lower = text.lower()
    if lower in {"train", "fit", "training", "source", "source_fit"}:
        return "train"
    if lower in {"validation", "valid", "val", "select", "selection"}:
        return "validation"
    if lower in {"query", "sealed_query", "holdout", "heldout"}:
        return "query"
    if lower in {"test", "sealed_test", "evaluation"}:
        return "test"
    return default


def _partition_from_key(key: str) -> str | None:
    lower = key.lower()
    if "validation" in lower or lower.startswith("val"):
        return "validation"
    if "query" in lower:
        return "query"
    if "test" in lower or "sealed" in lower:
        return "test"
    if "train" in lower or "source_fit" in lower or "training" in lower:
        return "train"
    return None


def _task_role(partition: str, task: str, *, internal: bool = False) -> str:
    if internal and partition == "train":
        return "diagnostic_fit"
    if partition == "train":
        return "fit"
    if partition == "validation":
        return "select"
    if partition in {"query", "test"}:
        return "score"
    return "unknown"


def _access_for(partition: str, task: str, policy: str | None, *, sealed: bool = False) -> dict[str, str]:
    policy_text = (policy or "").lower()
    if partition == "train":
        expression = "fit"
        labels = "fit" if task == "annotation" and "label" in policy_text else "never"
        coordinates = "fit" if task == "spatial" else "never"
        state = "open"
    elif partition == "validation":
        expression = "select"
        labels = "select" if task == "annotation" else "never"
        coordinates = "select" if task == "spatial" else "never"
        state = "open"
    else:
        expression = "score"
        labels = "score" if task == "annotation" and "label" in policy_text else "never"
        coordinates = "score" if task == "spatial" else "never"
        state = "sealed" if sealed or partition in {"query", "test"} else "open"
    return {
        "expression_access": expression,
        "label_access": labels,
        "metadata_access": "fit" if partition == "train" else "select" if partition == "validation" else "score",
        "coordinate_access": coordinates,
        "access_state": state,
    }


def _source_side_access(partition: str, task: str) -> dict[str, str]:
    """Access for an observed source/input side of a query or test pair.

    Temporal pair manifests contain both the observed source and the future
    target under one logical ``query`` partition.  Only the target side is
    sealed; the source side is required to make a prediction and is therefore
    open for scoring input.  Keeping this distinction in the ledger prevents a
    later runner from either leaking the target or unnecessarily hiding its
    allowed source input.
    """
    return {
        "expression_access": "score",
        "label_access": "never",
        "metadata_access": "score",
        "coordinate_access": "score" if task == "spatial" else "never",
        "access_state": "open",
    }


def _normalization_id(snapshot: Mapping[str, Any], repo_root: Path) -> str | None:
    data = snapshot.get("data_contract") or {}
    norm = data.get("normalization") or {}
    if not norm:
        return None
    variants = norm.get("allowed_variants") or []
    fit_scope = norm.get("fit_scope")
    return _slug("normalization:" + ",".join(map(str, variants)) + ":" + str(fit_scope))


def _snapshot_context(
    benchmark_id: str,
    snapshot: Mapping[str, Any],
    task: str,
    repo_root: Path,
) -> dict[str, Any]:
    data = snapshot.get("data_contract") or {}
    evaluator = snapshot.get("evaluator_contract") or {}
    split = snapshot.get("split") or {}
    input_contract = snapshot.get("input_contract") or {}
    corpus = input_contract.get("pretraining_corpus") or {}
    split_id = _text(split.get("split_id") or data.get("split_id"))
    panel_id = _text(data.get("panel_id"))
    vocab = _path_value(corpus.get("vocabulary_path"), repo_root)
    label_order = _text((snapshot.get("output_contract") or {}).get("gene_or_label_order"))
    # Explicit historical flags are preferable to inferring freshness from a
    # path.  A missing value remains unknown rather than being overclaimed.
    historical = bool(
        snapshot.get("historical_target_reuse")
        or snapshot.get("independent_target") is False
        or "historical" in str(snapshot.get("scope", "")).lower()
        or "historical" in str(data.get("query_policy", "")).lower()
    )
    source_fresh: bool | None = False if historical else None
    row_fresh: bool | None = True if "fresh" in str(snapshot.get("scope", "")).lower() else None
    return {
        "benchmark_id": benchmark_id,
        "task_id": task,
        "data_contract_id": _text(data.get("contract_id")),
        "evaluator_contract_id": _text(evaluator.get("contract_id")),
        "split_id": split_id,
        "panel_id": panel_id,
        "gene_vocab_id": vocab,
        "feature_order_id": _text((snapshot.get("output_contract") or {}).get("gene_or_label_order")),
        "label_vocab_id": label_order if "label" in (label_order or "").lower() else None,
        "normalization_id": _normalization_id(snapshot, repo_root),
        "normalization_fit_scope": _text(((data.get("normalization") or {}).get("fit_scope"))),
        "query_policy": _text(data.get("query_policy") or evaluator.get("query_policy")),
        "split": split,
        "species": _text(split.get("species") or corpus.get("species")),
        "assay_type": _text((data.get("assay") or [None])[0]),
        "platform": _text(split.get("platform")),
        "dataset_id": _text(
            split.get("dataset")
            or split.get("target_dataset")
            or ((split.get("source_datasets") or [None])[0])
        ),
        "region": _text(split.get("region") or split.get("uberon_id")),
        "source_fresh": source_fresh,
        "row_position_fresh": row_fresh,
        "sample_fresh": False if historical else None,
        "donor_fresh": False if historical else None,
        "historical_exposure_ref": "declared_historical_reuse" if historical else None,
    }


def _descriptor_time_map(split: Mapping[str, Any]) -> dict[str, tuple[float | None, float | None]]:
    """Map descriptor identifiers to declared source/target times when present."""
    result: dict[str, tuple[float | None, float | None]] = {}

    def add(values: Any, times: Any) -> None:
        if not isinstance(values, list) or not isinstance(times, (list, tuple)) or len(times) < 2:
            return
        source = _number(times[0])
        target = _number(times[1])
        for value in values:
            if isinstance(value, list):
                add(value, times)
            else:
                result[str(value)] = (source, target)

    # Source23 stores one descriptor group for each adjacent training interval.
    groups = split.get("training_descriptor_groups")
    times = split.get("training_times_days")
    if isinstance(groups, list) and isinstance(times, list):
        for index, group in enumerate(groups):
            if index + 1 < len(times):
                add(group, [times[index], times[index + 1]])
    add(split.get("validation_descriptors"), split.get("validation_pair_days"))
    add(
        split.get("query_descriptors"),
        split.get("query_adjacent_artifact_pair_days") or split.get("query_pair_days"),
    )
    # Some contracts use explicit interval objects instead of descriptor lists.
    for interval in split.get("intervals", []) or []:
        if not isinstance(interval, Mapping):
            continue
        interval_times = [interval.get("source_time_days"), interval.get("target_time_days")]
        add(interval.get("descriptor_ids"), interval_times)
    return result


def _empty_row() -> dict[str, Any]:
    return {column: None for column in LEDGER_COLUMNS}


def _entry_from_identity(
    *,
    context: Mapping[str, Any],
    partition: str,
    unit_type: str,
    identity_scope: str,
    physical_parts: Sequence[Any],
    role_side: str | None = None,
    source_id: Any = None,
    dataset_id: Any = None,
    sample_id: Any = None,
    sample_key: Any = None,
    donor_id: Any = None,
    subject_id: Any = None,
    cell_or_spot_id: Any = None,
    row_index: Any = None,
    section_id: Any = None,
    region: Any = None,
    time: Any = None,
    temporal_descriptor_id: Any = None,
    source_time_days: Any = None,
    target_time_days: Any = None,
    delta_time_days: Any = None,
    declared_count: Any = None,
    internal: bool = False,
    sealed: bool | None = None,
) -> dict[str, Any]:
    if partition not in PARTITIONS:
        raise ValueError(f"invalid partition {partition!r}")
    task = str(context.get("task_id") or "auxiliary")
    policy = context.get("query_policy")
    # ``sealed`` is the default for query/test partitions, but a temporal
    # source side is an observed input and must remain open.
    sealed_state = partition in {"query", "test"} if sealed is None else bool(sealed)
    if role_side in {"source", "input", "observed"} and partition in {"query", "test"}:
        sealed_state = False
    access = (
        _source_side_access(partition, task)
        if role_side in {"source", "input", "observed"} and partition in {"query", "test"}
        else _access_for(partition, task, policy, sealed=sealed_state)
    )
    global_row_id = _stable_id("row", *physical_parts)
    entry = _empty_row()
    entry.update(
        {
            "global_row_id": global_row_id,
            "identity_scope": identity_scope,
            "unit_type": unit_type,
            "source_id": _number(source_id),
            "dataset_id": _text(dataset_id),
            "sample_id": _text(sample_id),
            "sample_key": _text(sample_key),
            "donor_id": _text(donor_id),
            "subject_id": _text(subject_id),
            "cell_or_spot_id": _text(cell_or_spot_id),
            "row_index": _number(row_index),
            "assay_type": _text(context.get("assay_type")),
            "species": _text(context.get("species")),
            "platform": _text(context.get("platform")),
            "section_id": _text(section_id),
            "region": _text(region or context.get("region")),
            "time": _text(time),
            "partition": partition,
            "task_id": task,
            "benchmark_id": _text(context.get("benchmark_id")),
            "task_role": _task_role(partition, task, internal=internal),
            "role_side": _text(role_side),
            "panel_id": _text(context.get("panel_id")),
            "gene_vocab_id": _text(context.get("gene_vocab_id")),
            "feature_order_id": _text(context.get("feature_order_id")),
            "hidden_observed_mask_id": _text(context.get("hidden_observed_mask_id")),
            "normalization_id": _text(context.get("normalization_id")),
            "normalization_fit_scope": _text(context.get("normalization_fit_scope")),
            "temporal_descriptor_id": _text(temporal_descriptor_id),
            "source_time_days": _number(source_time_days),
            "target_time_days": _number(target_time_days),
            "delta_time_days": _number(delta_time_days),
            "coordinate_registration_id": _text(
                context.get("coordinate_registration_id")
                or (f"section:{section_id}" if task == "spatial" and section_id is not None else None)
            ),
            "label_vocab_id": _text(context.get("label_vocab_id")),
            "source_fresh": context.get("source_fresh"),
            "row_position_fresh": context.get("row_position_fresh"),
            "sample_fresh": context.get("sample_fresh"),
            "donor_fresh": context.get("donor_fresh"),
            "historical_exposure_ref": _text(context.get("historical_exposure_ref")),
            "sealed_reveal_event": _stable_id("reveal", context.get("benchmark_id"), context.get("split_id")) if sealed_state else None,
            "receipt_id": _stable_id("receipt", context.get("benchmark_id"), partition),
            "declared_count": _number(declared_count),
            **access,
        }
    )
    # Include the role in the entry key but keep the physical ID independent of
    # task, allowing the exposure audit to detect cross-task reuse.
    entry["ledger_entry_id"] = _stable_id(
        "entry", global_row_id, context.get("benchmark_id"), partition, role_side, unit_type, row_index
    )
    return entry


def _add_row(rows: list[dict[str, Any]], row: dict[str, Any], seen: set[str]) -> None:
    entry_id = str(row["ledger_entry_id"])
    if entry_id in seen:
        return
    seen.add(entry_id)
    rows.append(row)


def _source_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    if "source_id" not in frame.columns:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        result[str(int(record["source_id"]))] = record
    return result


def _corpus_context(corpus_id: str, artifact_dir: Path, split: str, task: str = "pretraining") -> dict[str, Any]:
    context = {
        "benchmark_id": corpus_id,
        "task_id": task,
        "split_id": f"{corpus_id}:{split}",
        "panel_id": None,
        "gene_vocab_id": None,
        "feature_order_id": None,
        "label_vocab_id": None,
        "normalization_id": None,
        "normalization_fit_scope": "corpus_declared_fit_scope",
        "query_policy": "test_labels_and_expression_sealed",
        "source_fresh": False,
        "row_position_fresh": None,
        "sample_fresh": False,
        "donor_fresh": False,
        "historical_exposure_ref": str(artifact_dir),
    }
    manifest = artifact_dir / "manifest.json"
    if manifest.is_file():
        try:
            data = _load_json(manifest)
            context["gene_vocab_id"] = _path_value(
                data.get("vocabulary_path") or data.get("feature_vocab_path"), artifact_dir.parents[2]
            )
        except (OSError, ValueError):
            pass
    return context


def _append_corpus_samples(
    rows: list[dict[str, Any]],
    seen: set[str],
    repo_root: Path,
    source_cache: dict[str, dict[str, Any]],
    artifact_specs: Sequence[tuple[str, Path]],
) -> list[str]:
    refs: list[str] = []
    for corpus_id, artifact_dir in artifact_specs:
        path = artifact_dir / "samples.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        refs.append(_path_value(path, repo_root) or str(path))
        for record in frame.to_dict(orient="records"):
            split = _partition_for(record.get("split"), "train")
            if split is None:
                continue
            context = _corpus_context(corpus_id, artifact_dir, split)
            context["species"] = _text(record.get("species"))
            context["assay_type"] = _text(record.get("assay_type") or record.get("assay"))
            context["platform"] = _text(record.get("technology") or record.get("platform"))
            context["region"] = _text(record.get("uberon_id") or record.get("region"))
            source_id = None
            h5ad_path = record.get("h5ad_path")
            if h5ad_path is not None:
                for sid, source in source_cache.items():
                    if str(source.get("original_h5ad_path")) == str(h5ad_path):
                        source_id = sid
                        break
            if source_id is None and record.get("source_ids_json"):
                try:
                    source_id = json.loads(str(record["source_ids_json"]))[0]
                except (ValueError, IndexError, TypeError):
                    source_id = None
            sample_key = record.get("sample_key") or record.get("sample_id")
            donor = record.get("donor_id") or record.get("animal_id")
            dataset = record.get("dataset")
            if dataset is None and record.get("datasets_json"):
                try:
                    dataset = json.loads(str(record["datasets_json"]))[0]
                except (ValueError, IndexError, TypeError):
                    dataset = None
            section = record.get("section_id")
            if section is None:
                section = record.get("sample_key")
            row = _entry_from_identity(
                context=context,
                partition=split,
                unit_type="sample",
                identity_scope="sample",
                physical_parts=("sample", corpus_id, sample_key),
                source_id=source_id,
                dataset_id=dataset,
                sample_id=record.get("sample_id"),
                sample_key=sample_key,
                donor_id=donor,
                subject_id=record.get("subject_id") or record.get("animal_id"),
                section_id=section,
                region=record.get("uberon_id") or record.get("region") or record.get("regions_json"),
                time=record.get("time_point_days") or record.get("timepoint_raw") or record.get("age_raw"),
                declared_count=record.get("selected_cells") or record.get("n_observations"),
                sealed=split in {"query", "test"},
            )
            _add_row(rows, row, seen)
    return refs


def _records_from_snapshot(
    rows: list[dict[str, Any]],
    seen: set[str],
    benchmark_id: str,
    snapshot: Mapping[str, Any],
    task: str,
    repo_root: Path,
) -> dict[str, Any]:
    context = _snapshot_context(benchmark_id, snapshot, task, repo_root)
    split = context["split"]
    context["split_id"] = context.get("split_id") or benchmark_id
    descriptor_times = _descriptor_time_map(split)
    source_id = split.get("source_id")
    source_dataset = split.get("dataset")
    source_datasets = split.get("source_datasets") or []
    if source_dataset is None and source_datasets:
        source_dataset = source_datasets[0]
    # Explicit list-valued role fields (donors, sections, descriptors) are the
    # most useful metadata.  Numeric time arrays and unrelated lists are not
    # materialized as fake cells.
    handled_keys: set[str] = set()
    for key, value in split.items():
        if not isinstance(value, list) or not value:
            continue
        partition = _partition_from_key(str(key))
        lower = str(key).lower()
        if partition is None or not any(token in lower for token in ("donor", "section", "descriptor", "row")):
            continue
        handled_keys.add(str(key))
        flat_values: list[Any] = []
        for item in value:
            if isinstance(item, list):
                flat_values.extend(item)
            else:
                flat_values.append(item)
        for ordinal, item in enumerate(flat_values):
            if "donor" in lower:
                unit_type, scope = "donor", "donor"
                donor, sample_key, section, descriptor = item, None, None, None
            elif "section" in lower:
                unit_type, scope = "section", "section"
                donor, sample_key, section, descriptor = None, item, item, None
            elif "descriptor" in lower:
                unit_type, scope = "descriptor", "descriptor"
                donor, sample_key, section, descriptor = None, None, None, item
            else:
                unit_type, scope = "aggregate", "aggregate"
                donor, sample_key, section, descriptor = None, None, None, None
            row = _entry_from_identity(
                context=context,
                partition=partition,
                unit_type=unit_type,
                identity_scope=scope,
                physical_parts=("contract", benchmark_id, key, item),
                source_id=source_id,
                dataset_id=source_dataset,
                sample_key=sample_key,
                donor_id=donor,
                section_id=section,
                temporal_descriptor_id=descriptor,
                source_time_days=descriptor_times.get(str(descriptor), (None, None))[0],
                target_time_days=descriptor_times.get(str(descriptor), (None, None))[1],
                delta_time_days=(
                    descriptor_times.get(str(descriptor), (None, None))[1]
                    - descriptor_times.get(str(descriptor), (None, None))[0]
                    if descriptor_times.get(str(descriptor), (None, None))[0] is not None
                    and descriptor_times.get(str(descriptor), (None, None))[1] is not None
                    else None
                ),
                region=split.get("region") or split.get("uberon_id"),
                time=split.get("time") or split.get("life_stage"),
                declared_count=None,
                internal="internal" in str(snapshot.get("status", "")).lower()
                or "internal" in str(snapshot.get("scope", "")).lower(),
                sealed=partition in {"query", "test"},
            )
            _add_row(rows, row, seen)

    # Count-only fields remain useful as aggregate records, but never pretend
    # that an aggregate is an individual cell or donor.
    for key, value in split.items():
        if key in handled_keys or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        partition = _partition_from_key(str(key))
        lower = str(key).lower()
        if partition is None or not any(token in lower for token in ("donor", "row", "cell", "spot", "descriptor", "section")):
            continue
        row = _entry_from_identity(
            context=context,
            partition=partition,
            unit_type="aggregate",
            identity_scope="aggregate",
            physical_parts=("aggregate", benchmark_id, key),
            source_id=source_id,
            dataset_id=source_dataset,
            region=split.get("region") or split.get("uberon_id"),
            declared_count=value,
            internal="internal" in str(snapshot.get("status", "")).lower(),
            sealed=partition in {"query", "test"},
        )
        _add_row(rows, row, seen)

    # Preserve contract metadata even when a split has no materializable unit.
    return {
        "benchmark_id": benchmark_id,
        "task_id": task,
        "data_contract_id": context.get("data_contract_id"),
        "evaluator_contract_id": context.get("evaluator_contract_id"),
        "split_id": context.get("split_id"),
        "panel_id": context.get("panel_id"),
        "normalization_fit_scope": context.get("normalization_fit_scope"),
        "query_policy": context.get("query_policy"),
        "status": snapshot.get("status"),
        "source_path": None,
    }


def _append_row_manifest(
    rows: list[dict[str, Any]],
    seen: set[str],
    repo_root: Path,
    path: Path,
    context: Mapping[str, Any],
    default_partition: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        return {"path": _path_value(path, repo_root) or str(path), "exists": False, "rows": 0}
    # Read only metadata columns.  In particular, never request expression or
    # label columns even if a manifest grows those columns later.
    allowed = {
        "source_id", "row_index", "sample_key", "cell_id", "donor_id", "partition", "split",
        "phase", "descriptor_index", "descriptor_id", "within_descriptor_index", "section_id",
        "region", "uberon_id", "time_days", "source_time_days", "target_time_days", "delta_time_days",
    }
    # Inspect the schema first so an accidentally enlarged manifest cannot
    # cause expression or label columns to be loaded from disk.
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    requested = [column for column in available if column in allowed]
    frame = pd.read_parquet(path, columns=sorted(requested)) if requested else pd.DataFrame()
    added = 0
    for ordinal, record in enumerate(frame.to_dict(orient="records")):
        partition = _partition_for(
            record.get("partition") or record.get("split") or record.get("phase"),
            default_partition,
        )
        if partition is None:
            continue
        source_id = record.get("source_id")
        row_index = record.get("row_index")
        sample_key = record.get("sample_key")
        physical_parts = ("physical", source_id, row_index, sample_key)
        if row_index is None:
            physical_parts = ("manifest", _path_value(path, repo_root), sample_key, record.get("descriptor_index"), ordinal)
        descriptor = record.get("descriptor_id")
        if descriptor is None and record.get("descriptor_index") is not None:
            descriptor = f"{sample_key}::descriptor-{record['descriptor_index']}"
        local_context = dict(context)
        local_context["source_fresh"] = context.get("source_fresh")
        local_context["row_position_fresh"] = context.get("row_position_fresh")
        local_context["sample_fresh"] = context.get("sample_fresh")
        local_context["donor_fresh"] = context.get("donor_fresh")
        row = _entry_from_identity(
            context=local_context,
            partition=partition,
            unit_type="cell_or_spot",
            identity_scope="physical_row" if row_index is not None else "manifest_row",
            physical_parts=physical_parts,
            source_id=source_id,
            dataset_id=context.get("dataset_id"),
            sample_id=sample_key,
            sample_key=sample_key,
            donor_id=record.get("donor_id"),
            cell_or_spot_id=record.get("cell_id") or row_index,
            row_index=row_index,
            section_id=record.get("section_id") or sample_key,
            region=record.get("region") or record.get("uberon_id"),
            time=record.get("time_days"),
            temporal_descriptor_id=descriptor,
            source_time_days=record.get("source_time_days"),
            target_time_days=record.get("target_time_days"),
            delta_time_days=record.get("delta_time_days"),
            role_side="row",
            sealed=partition in {"query", "test"},
        )
        _add_row(rows, row, seen)
        added += 1
    return {
        "path": _path_value(path, repo_root) or str(path),
        "exists": True,
        "rows": int(added),
        "source_rows": int(len(frame)),
    }


def _append_row_selection(
    rows: list[dict[str, Any]],
    seen: set[str],
    repo_root: Path,
    selection_path: Path,
    contract_path: Path,
    benchmark_id: str,
    task: str = "development",
) -> dict[str, Any]:
    if not selection_path.is_file() or not contract_path.is_file():
        return {"selection": _path_value(selection_path, repo_root), "exists": False, "rows": 0}
    selection = _load_json(selection_path)
    contract = _load_json(contract_path)
    source = contract.get("source") or {}
    feature = contract.get("feature_contract") or {}
    evaluator = contract.get("evaluator") or {}
    access_policy = "target expression scoring only; labels never read"
    context = {
        "benchmark_id": benchmark_id,
        "task_id": task,
        "split_id": contract.get("split", {}).get("split_id") or benchmark_id,
        "panel_id": feature.get("panel_id"),
        "gene_vocab_id": None,
        "feature_order_id": feature.get("panel_id"),
        "label_vocab_id": None,
        "normalization_id": _slug(feature.get("normalization")),
        "normalization_fit_scope": feature.get("normalization_fit_scope") or (contract.get("preprocessing") or {}).get("fit_scope"),
        "query_policy": access_policy,
        "species": source.get("species"),
        "assay_type": source.get("assay_type"),
        "platform": (contract.get("split") or {}).get("platform"),
        "region": source.get("uberon_id") or (source.get("region")),
        "source_fresh": (contract.get("freshness") or {}).get("source_fresh"),
        "row_position_fresh": (contract.get("freshness") or {}).get("row_position_fresh"),
        "sample_fresh": (contract.get("freshness") or {}).get("sample_fresh"),
        "donor_fresh": (contract.get("freshness") or {}).get("donor_fresh"),
        "historical_exposure_ref": (contract.get("historical_exposure") or {}).get("definition"),
    }
    added = 0
    for role in ("train", "validation", "sealed_query", "query", "test"):
        for descriptor in selection.get(role, []) or []:
            partition = _partition_for(descriptor.get("role") or role)
            if partition is None:
                continue
            descriptor_id = descriptor.get("descriptor_id")
            interval = descriptor.get("interval_id")
            source_time = descriptor.get("source_time_days")
            target_time = descriptor.get("target_time_days")
            delta = descriptor.get("delta_time_days")
            for side, positions_key, side_time in (
                ("source", "source_positions", source_time),
                ("target", "target_positions", target_time),
            ):
                for position in descriptor.get(positions_key, []) or []:
                    local = dict(context)
                    local["dataset_id"] = source.get("dataset")
                    local["sample_key"] = f"source:{source.get('source_id')}"
                    row = _entry_from_identity(
                        context=local,
                        partition=partition,
                        unit_type="cell_or_spot",
                        identity_scope="physical_row",
                        physical_parts=("physical", source.get("source_id"), position),
                        source_id=source.get("source_id"),
                        dataset_id=source.get("dataset"),
                        sample_key=f"source:{source.get('source_id')}",
                        cell_or_spot_id=position,
                        row_index=position,
                        region=source.get("uberon_id") or source.get("region"),
                        time=side_time,
                        temporal_descriptor_id=descriptor_id,
                        source_time_days=source_time,
                        target_time_days=target_time,
                        delta_time_days=delta,
                        role_side=side,
                        sealed=partition in {"query", "test"},
                    )
                    _add_row(rows, row, seen)
                    added += 1
    return {
        "selection": _path_value(selection_path, repo_root),
        "contract": _path_value(contract_path, repo_root),
        "exists": True,
        "rows": int(added),
        "evaluator_id": evaluator.get("evaluator_id"),
    }


def _contract_paths(repo_root: Path, extra: Sequence[Path] | None) -> list[Path]:
    if extra is not None:
        return [Path(p) if Path(p).is_absolute() else repo_root / p for p in extra]
    return [
        repo_root / "brain_world_model/configs/dev_source6_scfit_source_only_internal_v1.yaml",
        repo_root / "brain_world_model/configs/source7_row_disjoint_internal_validation_v1.yaml",
    ]


def _default_manifest_specs(repo_root: Path) -> list[tuple[str, Path, str | None]]:
    return [
        (
            "ANN-S64-LABEL-BLIND-DH-V2",
            repo_root / "brain_world_model/runs/ann_s64_label_blind_v2_manifest_20260903/selected_cells.parquet",
            None,
        ),
        (
            "ANN-MOUSE-SPATIAL-SECTION-FRESH-V1",
            repo_root / "brain_world_model/runs/ann_mouse_spatial_section_fresh_v1_20260903/features_train/row_manifest.parquet",
            "train",
        ),
        (
            "ANN-MOUSE-SPATIAL-SECTION-FRESH-V1",
            repo_root / "brain_world_model/runs/ann_mouse_spatial_section_fresh_v1_20260903/features_validation/row_manifest.parquet",
            "validation",
        ),
        (
            "ANN-MOUSE-SPATIAL-SECTION-FRESH-V1",
            repo_root / "brain_world_model/runs/ann_mouse_spatial_section_fresh_v1_20260903/features_test/row_manifest.parquet",
            "test",
        ),
    ]


def _default_corpus_specs(repo_root: Path) -> list[tuple[str, Path]]:
    return [
        (
            "HEALTHY-HUMAN-SCRNA-SNRNA-TRAIN-V1",
            repo_root / "pretrain_dataset/artifacts/healthy_pretrain_v1",
        ),
        (
            "BRAINWM-HUMAN-MOUSE-MOLECULAR-STAGE0-SOURCE-POOL-V1",
            repo_root / "pretrain_dataset/artifacts/mouse_molecular_cell_index_candidate_full_20260827",
        ),
    ]


def _audit_ledger(frame: pd.DataFrame, *, contract_count: int, sealed: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if frame.empty:
        blockers.append("partition_ledger_is_empty")
    if frame["ledger_entry_id"].duplicated().any():
        blockers.append("ledger_entry_id_not_unique")
    if set(frame["partition"].dropna().astype(str)) - set(PARTITIONS):
        blockers.append("unknown_partition_value")
    # Exact physical-row overlap across roles is the strongest signal.  Sample
    # and donor overlap are reported separately because some tasks intentionally
    # reuse a source cohort but still keep row-level roles disjoint.
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for identity_column, label in (
        ("global_row_id", "row"),
        ("sample_key", "sample"),
        ("donor_id", "donor"),
        ("temporal_descriptor_id", "descriptor"),
    ):
        subset = frame[frame[identity_column].notna()].copy()
        examples: list[dict[str, Any]] = []
        if not subset.empty:
            for identity, part in subset.groupby(identity_column, sort=False):
                partitions = sorted(set(part["partition"].astype(str)))
                roles = sorted(set(part["task_role"].astype(str)))
                benchmarks = sorted(set(part["benchmark_id"].astype(str)))
                if len(partitions) > 1 or len(roles) > 1:
                    examples.append(
                        {
                            "identity": str(identity),
                            "partitions": partitions,
                            "roles": roles,
                            "benchmarks": benchmarks,
                        }
                    )
                    if len(examples) >= 50:
                        break
        conflicts[label] = examples
        if examples:
            # A shared fit role is historical reuse; a cross-role overlap can
            # leak selection or score information and blocks new training.
            blockers.append(f"cross_task_{label}_role_overlap")
    sealed_frame = pd.DataFrame(sealed)
    safe_ok = set(sealed_frame.columns).issubset(set(SAFE_SEALED_COLUMNS)) if not sealed_frame.empty else True
    if not safe_ok:
        blockers.append("sealed_manifest_contains_non_metadata_field")
    source_fresh_counts = {
        str(value): int(count)
        for value, count in frame["source_fresh"].value_counts(dropna=False).items()
    }
    row_fresh_counts = {
        str(value): int(count)
        for value, count in frame["row_position_fresh"].value_counts(dropna=False).items()
    }
    if source_fresh_counts.get("False", 0) > 0:
        warnings.append("historically_exposed_sources_present")
    # A declared fit scope containing query/test is always suspicious.  It is
    # retained as an audit finding rather than silently normalized.
    fit_scope_bad = sorted(
        set(
            str(value)
            for value in frame["normalization_fit_scope"].dropna()
            if any(token in str(value).lower() for token in ("query", "test"))
        )
    )
    if fit_scope_bad:
        blockers.append("fit_scope_mentions_query_or_test")

    interval_checks = _audit_development_intervals(frame)
    blockers.extend(interval_checks["blockers"])
    status = "blocked_conflicts" if blockers else "pass"
    return {
        "format": "brainwm_global_split_exposure_audit_v1",
        "status": status,
        "contract_count": int(contract_count),
        "ledger_rows": int(len(frame)),
        "unique_global_row_ids": int(frame["global_row_id"].nunique(dropna=True)),
        "unique_ledger_entry_ids": int(frame["ledger_entry_id"].nunique(dropna=True)),
        "partition_counts": {str(k): int(v) for k, v in frame["partition"].value_counts(dropna=False).items()},
        "task_counts": {str(k): int(v) for k, v in frame["task_id"].value_counts(dropna=False).items()},
        "identity_conflicts": {key: {"count": len(value), "examples": value} for key, value in conflicts.items()},
        "freshness": {
            "source_fresh": source_fresh_counts,
            "row_position_fresh": row_fresh_counts,
            "sample_fresh": {str(k): int(v) for k, v in frame["sample_fresh"].value_counts(dropna=False).items()},
            "donor_fresh": {str(k): int(v) for k, v in frame["donor_fresh"].value_counts(dropna=False).items()},
        },
        "fit_scope_findings": fit_scope_bad,
        "development_interval_checks": interval_checks,
        "sealed_rows": int(len(sealed)),
        "sealed_metadata_only": bool(safe_ok),
        "blockers": blockers,
        "warnings": warnings,
        "ready_for_new_benchmark": not blockers,
    }


def _build_development_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a compact descriptor/pair index from the physical ledger.

    The table contains counts and time metadata only.  It is intentionally
    separate from the row ledger so a temporal runner can validate pair
    boundaries without loading expression values.
    """
    columns = [
        "benchmark_id", "task_id", "temporal_descriptor_id", "partition",
        "source_time_days", "target_time_days", "delta_time_days",
        "source_rows", "target_rows", "unique_rows", "source_row_ids", "target_row_ids",
    ]
    if frame.empty or "temporal_descriptor_id" not in frame:
        return pd.DataFrame(columns=columns)
    subset = frame[frame["temporal_descriptor_id"].notna()].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    result: list[dict[str, Any]] = []
    for (benchmark, descriptor), part in subset.groupby(
        ["benchmark_id", "temporal_descriptor_id"], sort=True, dropna=False
    ):
        partitions = sorted(set(part["partition"].astype(str)))
        source = part[part["role_side"] == "source"]
        target = part[part["role_side"] == "target"]
        # Row manifests use role_side=row; they are still useful descriptor
        # counts, but cannot be labelled source/target without extra metadata.
        if source.empty and target.empty:
            source = part.iloc[0:0]
            target = part.iloc[0:0]
        first = part.iloc[0]
        result.append(
            {
                "benchmark_id": benchmark,
                "task_id": first.get("task_id"),
                "temporal_descriptor_id": descriptor,
                "partition": partitions[0] if len(partitions) == 1 else "+".join(partitions),
                "source_time_days": _number(first.get("source_time_days")),
                "target_time_days": _number(first.get("target_time_days")),
                "delta_time_days": _number(first.get("delta_time_days")),
                "source_rows": int(len(source)),
                "target_rows": int(len(target)),
                "unique_rows": int(part["global_row_id"].nunique()),
                "source_row_ids": json.dumps(sorted(set(source["global_row_id"].astype(str))), separators=(",", ":")),
                "target_row_ids": json.dumps(sorted(set(target["global_row_id"].astype(str))), separators=(",", ":")),
            }
        )
    return pd.DataFrame(result, columns=columns)


def _audit_development_intervals(frame: pd.DataFrame) -> dict[str, Any]:
    intervals = _build_development_intervals(frame)
    blockers: list[str] = []
    invalid_time: list[str] = []
    for record in intervals.to_dict(orient="records"):
        source_time = record.get("source_time_days")
        target_time = record.get("target_time_days")
        if source_time is None or target_time is None:
            continue
        try:
            source_value = float(source_time)
            target_value = float(target_time)
            # Pandas nullable columns may expose missing values as NaN.  A
            # descriptor without declared times is audited separately as
            # metadata-incomplete; it is not a false chronological violation.
            if not (np.isfinite(source_value) and np.isfinite(target_value)):
                continue
            if not (source_value < target_value):
                invalid_time.append(str(record.get("temporal_descriptor_id")))
        except (TypeError, ValueError):
            invalid_time.append(str(record.get("temporal_descriptor_id")))
    if invalid_time:
        blockers.append("development_source_time_not_before_target_time")
    # The row IDs in source/target sides must be disjoint within a descriptor.
    side_overlap: list[str] = []
    cross_partition: list[str] = []
    for record in intervals.to_dict(orient="records"):
        if "+" in str(record.get("partition", "")):
            cross_partition.append(str(record.get("temporal_descriptor_id")))
        source = set(json.loads(record.get("source_row_ids") or "[]"))
        target = set(json.loads(record.get("target_row_ids") or "[]"))
        if source & target:
            side_overlap.append(str(record.get("temporal_descriptor_id")))
    if side_overlap:
        blockers.append("development_source_target_row_overlap")
    if cross_partition:
        blockers.append("development_descriptor_cross_partition")
    # A physical row must not be assigned to two temporal descriptors within
    # one benchmark.  This check uses only stable IDs from the ledger.
    duplicate_rows: list[str] = []
    temporal = frame[frame["temporal_descriptor_id"].notna()].copy()
    if not temporal.empty:
        for (benchmark, row_id), part in temporal.groupby(["benchmark_id", "global_row_id"], sort=False):
            if part["temporal_descriptor_id"].nunique() > 1:
                duplicate_rows.append(f"{benchmark}:{row_id}")
                if len(duplicate_rows) >= 50:
                    break
    if duplicate_rows:
        blockers.append("development_row_reused_across_descriptors")
    return {
        "interval_count": int(len(intervals)),
        "invalid_time_descriptors": invalid_time,
        "source_target_overlap_descriptors": side_overlap,
        "cross_partition_descriptors": cross_partition,
        "row_reused_across_descriptors": duplicate_rows,
        "blockers": blockers,
        "valid": not blockers,
    }


def _policy_document() -> dict[str, Any]:
    return {
        "format": "brainwm_global_access_policy_v1",
        "fail_closed": True,
        "operation_vocabulary": ["fit", "select", "score", "never"],
        "partition_policy": {
            "train": {
                "expression": "fit",
                "labels": "fit_only_when_task_contract_allows",
                "metadata": "fit",
                "coordinate": "fit_only_for_spatial_tasks",
                "before_reveal": "open",
            },
            "validation": {
                "expression": "select",
                "labels": "select_only_when_task_contract_allows",
                "metadata": "select",
                "coordinate": "select_only_for_spatial_tasks",
                "before_reveal": "open",
            },
            "query": {
                "expression": "score",
                "labels": "score_only_when_contract_explicitly_allows",
                "metadata": "score",
                "coordinate": "score_only_for_spatial_tasks",
                "before_reveal": "sealed",
            },
            "test": {
                "expression": "score",
                "labels": "score_only_when_contract_explicitly_allows",
                "metadata": "score",
                "coordinate": "score_only_for_spatial_tasks",
                "before_reveal": "sealed",
            },
        },
        "forbidden_pre_reveal_fields": [
            "expression",
            "raw_counts",
            "normalized_expression",
            "labels",
            "target_labels",
            "gene_values",
            "checkpoint_selection_feedback",
        ],
        "fit_assets": {
            "panel": "train-only unless a benchmark contract explicitly declares a source-fit subset",
            "normalization": "train/source-fit only",
            "pca": "train/source-fit only",
            "nb_dispersion": "train/source-fit only",
            "label_vocabulary": "train labels only",
        },
        "cross_task_rule": "a physical row with a score/select role cannot be used as fit data for a new benchmark",
        "reveal_rule": "one receipt before query/test expression or labels; no post-reveal method selection",
    }


def build_global_split_contract(
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    registry_path: str | Path | None = None,
    task_registry_path: str | Path | None = None,
    extra_contract_paths: Sequence[str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize the global contract and return its split receipt.

    Existing output is refused by default.  ``overwrite=True`` is intentionally
    explicit because the artifact is append-only from the benchmark's point of
    view and should not be replaced accidentally.
    """
    repo = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    registry = Path(registry_path) if registry_path else repo / "brain_world_model/configs/brainwm_benchmark_contracts_v1.yaml"
    task_registry = Path(task_registry_path) if task_registry_path else repo / "brain_world_model/configs/brainwm_benchmark_tasks_v1.yaml"
    if not registry.is_absolute():
        registry = repo / registry
    if not task_registry.is_absolute():
        task_registry = repo / task_registry
    registry_data = _load_yaml(registry)
    task_data = _load_yaml(task_registry)
    snapshots = _snapshot_map(registry_data)
    task_lookup = _task_lookup(task_data)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    contract_records: list[dict[str, Any]] = []
    corpus_specs = _default_corpus_specs(repo)
    source_cache: dict[str, dict[str, Any]] = {}
    for _, artifact in corpus_specs:
        source_cache.update(_source_lookup(artifact / "sources.parquet"))
    corpus_refs = _append_corpus_samples(rows, seen, repo, source_cache, corpus_specs)

    for benchmark_id, snapshot in snapshots.items():
        task = task_lookup.get(benchmark_id, "auxiliary")
        contract_records.append(_records_from_snapshot(rows, seen, benchmark_id, snapshot, task, repo))

    # Add the small, explicit row manifests used by the current contracts.
    manifest_receipts: list[dict[str, Any]] = []
    context_by_benchmark = {
        record["benchmark_id"]: _snapshot_context(record["benchmark_id"], snapshots[record["benchmark_id"]], record["task_id"], repo)
        for record in contract_records
        if record["benchmark_id"] in snapshots
    }
    for benchmark_id, path, default_partition in _default_manifest_specs(repo):
        context = dict(context_by_benchmark.get(benchmark_id, {"benchmark_id": benchmark_id, "task_id": "auxiliary"}))
        # Preserve a dataset key when a manifest's sample_key is the only
        # available identifier; it is metadata, not expression.
        context.setdefault("dataset_id", None)
        manifest_receipts.append(
            _append_row_manifest(rows, seen, repo, path, context, default_partition)
        )

    selection_receipts: list[dict[str, Any]] = []
    selection_specs = [
        (
            "INT-S6-ROW-DISJOINT-V1",
            repo / "brain_world_model/runs/source6_row_disjoint_internal_validation_20260904_v1/row_selection.json",
            repo / "brain_world_model/runs/source6_row_disjoint_internal_validation_20260904_v1/contract.json",
        ),
        (
            "INT-S7-ROW-DISJOINT-V1",
            repo / "brain_world_model/runs/source7_row_disjoint_internal_validation_20260904_v1/row_selection.json",
            repo / "brain_world_model/runs/source7_row_disjoint_internal_validation_20260904_v1/contract.json",
        ),
    ]
    for benchmark_id, selection_path, contract_path in selection_specs:
        selection_receipts.append(
            _append_row_selection(rows, seen, repo, selection_path, contract_path, benchmark_id)
        )

    frame = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    if not frame.empty:
        # Stable ordering is important for diffable receipts and downstream
        # streaming.  Missing IDs sort last without changing values.
        frame = frame.sort_values(
            ["global_row_id", "benchmark_id", "partition", "role_side", "ledger_entry_id"],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    ledger_path = output / "partition_ledger.parquet"
    frame.to_parquet(ledger_path, index=False, compression="zstd")
    interval_frame = _build_development_intervals(frame)
    interval_frame.to_parquet(output / "development_interval_ledger.parquet", index=False, compression="zstd")

    policy = _policy_document()
    (output / "access_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # A query/test partition may contain an observed source side.  The reveal
    # manifest contains only rows whose current access state is actually
    # sealed, never the source inputs needed to produce a prediction.
    sealed_frame = frame[
        frame["partition"].isin(["query", "test"]) & frame["access_state"].eq("sealed")
    ].copy() if not frame.empty else frame.copy()
    # The reveal boundary only needs stable identity, role, timing, freshness,
    # and access metadata.  Keep the compact allow-list here instead of the
    # full ledger projection: this artifact may be consumed before any target
    # payload is revealed, so unnecessary repeated metadata is a liability.
    sealed_columns = [column for column in SEALED_MINIMAL_COLUMNS if column in sealed_frame.columns]
    # Keep the complete physical IDs but omit null fields.  This preserves a
    # metadata-only reveal boundary while avoiding a multi-megabyte repetition
    # of nulls and pretty-print whitespace.
    sealed_records = []
    for record in sealed_frame[sealed_columns].to_dict(orient="records"):
        compact = {
            column: _json_value(value)
            for column, value in record.items()
            if not _is_missing_scalar(value)
        }
        sealed_records.append(compact)
    sealed_document = {
        "format": "brainwm_sealed_test_manifest_v1",
        "metadata_only": True,
        "expression_included": False,
        "labels_included": False,
        "reveal_required": True,
        "row_count": len(sealed_records),
        "columns": sealed_columns,
        "rows": sealed_records,
    }
    (output / "sealed_test_manifest.json").write_text(
        json.dumps(sealed_document, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    exposure = _audit_ledger(frame, contract_count=len(contract_records), sealed=sealed_records)
    exposure["source_manifest_receipts"] = manifest_receipts
    exposure["row_selection_receipts"] = selection_receipts
    exposure["large_file_hashes"] = False
    exposure["historical_exposure_method"] = "declared contract flags and metadata manifests; no H5AD scan"
    (output / "exposure_audit.json").write_text(
        json.dumps(_json_value(exposure), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    counts = {
        "ledger_rows": int(len(frame)),
        "unique_global_row_ids": int(frame["global_row_id"].nunique(dropna=True)) if not frame.empty else 0,
        "sealed_rows": int(len(sealed_records)),
        "by_partition": {str(k): int(v) for k, v in frame["partition"].value_counts(dropna=False).items()} if not frame.empty else {},
        "by_task": {str(k): int(v) for k, v in frame["task_id"].value_counts(dropna=False).items()} if not frame.empty else {},
    }
    global_manifest = {
        "format": GLOBAL_SPLIT_CONTRACT_FORMAT,
        "schema_version": GLOBAL_SPLIT_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "contract_id": "GLOBAL-BRAINWM-SPLIT-V1",
        "generated_at_utc": _now(),
        "git_commit": _repo_commit(repo),
        "purpose": "unified metadata-only role ledger; does not redefine existing benchmark splits",
        "zero_copy": True,
        "large_file_hashes": False,
        "partition_definitions": {
            "train": "fit data allowed by the bound task contract",
            "validation": "selection/evaluation data before query reveal",
            "query": "single-use score partition, sealed before reveal",
            "test": "sealed score partition, opened only by its receipt",
        },
        "fraction_policy": {
            "target_ratio": "6:2:2",
            "status": "not_applied_to_existing_frozen_contracts",
            "rule": "a future cohort may use 60/20/20 at the declared group unit; this artifact preserves each bound benchmark split and never reassigns rows",
        },
        "role_precedence": ["test", "query", "validation", "train"],
        "ledger_identity_rule": "physical source_id + row_index when available; otherwise qualified sample/descriptor key; never parquet row number alone",
        "schema": {
            "ledger_columns": list(LEDGER_COLUMNS),
            "sealed_metadata_columns": sealed_columns,
            "development_interval_columns": list(interval_frame.columns),
        },
        "inputs": {
            "benchmark_contract": _path_value(registry, repo),
            "task_registry": _path_value(task_registry, repo),
            "corpus_sample_manifests": corpus_refs,
            "row_manifests": manifest_receipts,
            "row_selections": selection_receipts,
        },
        "counts": counts,
        "files": {
            "partition_ledger": "partition_ledger.parquet",
            "development_interval_ledger": "development_interval_ledger.parquet",
            "access_policy": "access_policy.json",
            "sealed_test_manifest": "sealed_test_manifest.json",
            "exposure_audit": "exposure_audit.json",
            "split_receipt": "split_receipt.json",
        },
        "status": exposure["status"],
        "ready_for_new_benchmark": exposure["ready_for_new_benchmark"],
    }
    (output / "global_manifest.json").write_text(
        json.dumps(_json_value(global_manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    receipt = {
        "format": "brainwm_global_split_receipt_v1",
        "receipt_id": _stable_id("split-receipt", global_manifest["contract_id"], _repo_commit(repo)),
        "generated_at_utc": global_manifest["generated_at_utc"],
        "contract_id": global_manifest["contract_id"],
        "schema_version": GLOBAL_SPLIT_SCHEMA_VERSION,
        "git_commit": global_manifest["git_commit"],
        "input_contract_count": len(contract_records),
        "counts": counts,
        "checks": {
            "metadata_only_build": True,
            "expression_matrix_opened": False,
            "labels_opened": False,
            "sealed_test_opened": False,
            "large_file_hashes": False,
            "ledger_entry_ids_unique": bool(frame["ledger_entry_id"].is_unique),
            "sealed_manifest_metadata_only": exposure["sealed_metadata_only"],
            "cross_task_role_audit_complete": True,
        },
        "status": exposure["status"],
        "ready_for_new_benchmark": exposure["ready_for_new_benchmark"],
        "blockers": exposure["blockers"],
        "warnings": exposure["warnings"],
        "next_step": "freeze a new cohort/evaluator after blockers are resolved" if exposure["blockers"] else "bind the artifact in the next benchmark contract",
    }
    (output / "split_receipt.json").write_text(
        json.dumps(_json_value(receipt), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def validate_global_split_contract(root: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Validate the generated files without opening any source expression."""
    path = Path(root).resolve()
    required = [
        "global_manifest.json",
        "partition_ledger.parquet",
        "development_interval_ledger.parquet",
        "access_policy.json",
        "sealed_test_manifest.json",
        "exposure_audit.json",
        "split_receipt.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    manifest = _load_json(path / "global_manifest.json")
    ledger = pd.read_parquet(path / "partition_ledger.parquet")
    intervals = pd.read_parquet(path / "development_interval_ledger.parquet")
    sealed = _load_json(path / "sealed_test_manifest.json")
    exposure = _load_json(path / "exposure_audit.json")
    receipt = _load_json(path / "split_receipt.json")
    errors: list[str] = []
    missing_columns = sorted(set(LEDGER_COLUMNS) - set(ledger.columns))
    if missing_columns:
        errors.append("ledger_missing_columns:" + ",".join(missing_columns))
    if "ledger_entry_id" in ledger and not ledger["ledger_entry_id"].is_unique:
        errors.append("ledger_entry_id_not_unique")
    if "partition" in ledger and set(ledger["partition"].dropna().astype(str)) - set(PARTITIONS):
        errors.append("unknown_partition")
    sealed_rows = sealed.get("rows", [])
    sealed_columns = set(sealed.get("columns", []))
    if not sealed_columns.issubset(set(SAFE_SEALED_COLUMNS)):
        errors.append("sealed_columns_not_metadata_only")
    if sealed.get("expression_included") or sealed.get("labels_included"):
        errors.append("sealed_manifest_claims_expression_or_labels")
    if int(sealed.get("row_count", -1)) != len(sealed_rows):
        errors.append("sealed_row_count_mismatch")
    if exposure.get("sealed_rows") != len(sealed_rows):
        errors.append("exposure_sealed_count_mismatch")
    if receipt.get("contract_id") != manifest.get("contract_id"):
        errors.append("receipt_manifest_contract_mismatch")
    if "temporal_descriptor_id" in ledger and "temporal_descriptor_id" in intervals:
        ledger_descriptors = set(ledger["temporal_descriptor_id"].dropna().astype(str))
        interval_descriptors = set(intervals["temporal_descriptor_id"].dropna().astype(str))
        if not interval_descriptors.issubset(ledger_descriptors):
            errors.append("interval_ledger_references_unknown_descriptor")
    result = {
        "format": GLOBAL_SPLIT_CONTRACT_FORMAT,
        "root": str(path),
        "manifest_status": manifest.get("status"),
        "receipt_status": receipt.get("status"),
        "ledger_rows": int(len(ledger)),
        "interval_rows": int(len(intervals)),
        "sealed_rows": len(sealed_rows),
        "errors": errors,
        "valid": not errors,
        "ready_for_new_benchmark": bool(receipt.get("ready_for_new_benchmark", False)) and not errors,
    }
    if strict and errors:
        raise ValueError("global split contract validation failed: " + "; ".join(errors))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--task-registry", type=Path)
    parser.add_argument("--extra-contract", action="append", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_global_split_contract(
        args.repo_root,
        args.output,
        registry_path=args.registry,
        task_registry_path=args.task_registry,
        extra_contract_paths=args.extra_contract,
        overwrite=args.overwrite,
    )
    print(json.dumps(_json_value(receipt), ensure_ascii=False, indent=2))
    if args.validate:
        print(json.dumps(validate_global_split_contract(args.output, strict=args.strict), ensure_ascii=False, indent=2))


__all__ = [
    "GLOBAL_SPLIT_CONTRACT_FORMAT",
    "LEDGER_COLUMNS",
    "SEALED_MINIMAL_COLUMNS",
    "build_global_split_contract",
    "validate_global_split_contract",
]


if __name__ == "__main__":  # pragma: no cover
    main()
