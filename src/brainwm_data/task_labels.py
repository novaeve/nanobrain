"""Versioned task labels for the shared BrainWM benchmark.

The global benchmark is deliberately zero-copy.  Cell-level labels and
coordinates remain in the source ``cells.parquet``; this module materializes
only compact task manifests/vocabularies and provides a split-aware lazy view.
Test annotation labels are fail-closed unless an evaluator explicitly opens
them after prediction freeze.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from .global_split import GlobalBenchmarkIndex, SPLITS

TASK_LABEL_FORMAT = "brainwm_global_task_labels_v1"
TASK_LABEL_BUILDER_VERSION = "1.0.0"
TASKS: tuple[str, ...] = (
    "integration",
    "annotation",
    "development",
    "pseudotime",
    "spatial_niche",
    "perturbation",
)
ANNOTATION_SOURCE_COLUMNS: tuple[str, ...] = (
    "cl_id",
    "cl_term",
    "original_cell_type",
    "original_subclass",
)
ANNOTATION_PRIORITY: tuple[str, ...] = ANNOTATION_SOURCE_COLUMNS
ANATOMY_SOURCE_COLUMNS: tuple[str, ...] = (
    "uberon_id",
    "uberon_term",
    "original_anatomical_region",
)
SPATIAL_SOURCE_COLUMNS: tuple[str, ...] = (
    "spatial_x",
    "spatial_y",
    "spatial_z",
)
_MISSING = {
    "", "nan", "none", "null", "<na>", "na", "n/a",
    "unknown", "unk", "not_available", "not available",
}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    result = str(value).strip()
    return None if result.casefold() in _MISSING else result


def canonicalize_annotation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical annotation values without inventing missing labels.

    ``cl_id`` is preferred because it is ontology-backed.  The remaining
    source fields are fallbacks for corpora that do not provide a CL ID; the
    chosen source is retained so heterogeneous fallback labels cannot be
    mistaken for one ontology level.
    """

    result = pd.DataFrame(index=frame.index)
    label = pd.Series(pd.NA, index=frame.index, dtype="string")
    source = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in ANNOTATION_PRIORITY:
        if column not in frame:
            continue
        values = frame[column].map(_text).astype("string")
        take = label.isna() & values.notna()
        label.loc[take] = values.loc[take]
        source.loc[take] = column
    level = source.map(
        {
            "cl_id": "cell_ontology_id",
            "cl_term": "cell_ontology_term",
            "original_cell_type": "source_cell_type",
            "original_subclass": "source_subclass",
        }
    ).astype("string")
    result["annotation_label"] = label
    result["annotation_label_source"] = source
    result["annotation_label_level"] = level
    result["annotation_available"] = label.notna()
    return result


def _safe_columns(path: Path, requested: Sequence[str]) -> list[str]:
    available = set(pq.ParquetFile(path).schema.names)
    return [column for column in requested if column in available]


def _json_list(value: Any) -> list[str]:
    result = json.loads(str(value))
    if not isinstance(result, list):
        raise ValueError("development interval sample keys must be a JSON list")
    return [str(item) for item in result]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _annotation_vocab_frame(
    counts: Mapping[tuple[str, str, str], int],
    samples: Mapping[tuple[str, str, str], set[str]],
) -> pd.DataFrame:
    rows = [
        {
            "annotation_label": label,
            "annotation_label_source": source,
            "annotation_label_level": level,
            "cells": int(count),
            "samples": int(len(samples.get(key, set()))),
        }
        for key, count in counts.items()
        for label, source, level in [key]
    ]
    columns = [
        "annotation_label", "annotation_label_source",
        "annotation_label_level", "cells", "samples",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["annotation_label_source", "annotation_label"], kind="stable"
    ).reset_index(drop=True)


def _anatomy_vocab_frame(counts: Mapping[tuple[str, str], int]) -> pd.DataFrame:
    rows = [
        {"uberon_id": key[0], "uberon_term": key[1], "cells": int(value)}
        for key, value in counts.items()
    ]
    return pd.DataFrame(
        rows, columns=["uberon_id", "uberon_term", "cells"]
    ).sort_values(["uberon_id", "uberon_term"], kind="stable").reset_index(drop=True)


def _development_labels(
    intervals: pd.DataFrame,
    anatomy_counts: Mapping[tuple[str, str], int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = intervals.sort_values(
        ["split", "dataset", "assay_type", "source_time_days", "target_time_days"],
        kind="stable",
    ).reset_index(drop=True)
    ordered.insert(0, "interval_id", [f"DEV-{index:05d}" for index in range(len(ordered))])
    ordered["label_definition"] = "population_endpoint_expression"
    ordered["cell_correspondence"] = "unpaired_cross_sectional"
    ordered["runtime_target"] = "OT_barycentric_target_within_shared_anatomy"

    by_sample: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for (sample_key, uberon_id), count in anatomy_counts.items():
        by_sample[str(sample_key)][str(uberon_id)] = int(count)
    rows: list[dict[str, Any]] = []
    for row in ordered.itertuples(index=False):
        source_keys = _json_list(row.source_sample_keys)
        target_keys = _json_list(row.target_sample_keys)
        # A sample can be eligible for development while lacking a usable
        # anatomy annotation. Treat that side as an empty intersection rather
        # than failing the whole label build or inventing a region.
        source_ids = set().union(*(set(by_sample.get(key, {})) for key in source_keys)) if source_keys else set()
        target_ids = set().union(*(set(by_sample.get(key, {})) for key in target_keys)) if target_keys else set()
        for uberon_id in sorted(source_ids & target_ids):
            source_cells = sum(by_sample.get(key, {}).get(uberon_id, 0) for key in source_keys)
            target_cells = sum(by_sample.get(key, {}).get(uberon_id, 0) for key in target_keys)
            if source_cells and target_cells:
                rows.append(
                    {
                        "interval_id": row.interval_id,
                        "split": row.split,
                        "uberon_id": uberon_id,
                        "source_cells": int(source_cells),
                        "target_cells": int(target_cells),
                        "pairing": "population_level_OT_not_cell_lineage",
                    }
                )
    anatomy = pd.DataFrame(
        rows,
        columns=[
            "interval_id", "split", "uberon_id", "source_cells",
            "target_cells", "pairing",
        ],
    )
    return ordered, anatomy


def build_global_task_labels(
    benchmark_dir: str | Path,
    *,
    overwrite: bool = False,
    row_group_batch_size: int = 1,
) -> dict[str, Any]:
    """Build compact labels for every supportable task in one bounded pass."""

    root = Path(benchmark_dir).resolve()
    if row_group_batch_size < 1:
        raise ValueError("row_group_batch_size must be positive")
    global_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if global_manifest.get("multi_corpus"):
        raise ValueError("task-label builder currently requires the human single-corpus benchmark")
    source_root = Path(global_manifest["source_artifact"]).resolve()
    source_cells = source_root / "cells.parquet"
    if not source_cells.is_file():
        raise FileNotFoundError(source_cells)

    files = {
        "receipt": root / "task_labels_build_receipt.json",
        "manifest": root / "task_labels_manifest.json",
        "context": root / "task_context_labels.parquet",
        "annotation_train": root / "annotation_vocab_train.parquet",
        "annotation_validation": root / "annotation_vocab_validation.parquet",
        "annotation_coverage": root / "annotation_label_coverage.parquet",
        "anatomy_train": root / "anatomy_vocab_train.parquet",
        "development": root / "task_development_labels.parquet",
        "development_anatomy": root / "development_interval_anatomy.parquet",
        "pseudotime": root / "task_pseudotime.parquet",
        "niche": root / "task_spatial_niche.parquet",
        "cell_view": root / "cell_label_view.json",
        "readme": root / "TASK_LABELS.md",
    }
    occupied = [path for path in files.values() if path.exists()]
    if occupied and not overwrite:
        raise FileExistsError(
            "task-label outputs already exist; pass overwrite=True for an explicit rebuild: "
            + ", ".join(path.name for path in occupied)
        )

    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        files["receipt"],
        {
            "format": "brainwm_task_labels_build_receipt_v1",
            "status": "running",
            "pid": int(os.getpid()),
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "input": {
                "benchmark_dir": str(root),
                "global_manifest": str(root / "manifest.json"),
                "source_cells": str(source_cells),
                "row_group_batch_size": int(row_group_batch_size),
            },
            "outputs": [path.name for key, path in files.items() if key not in {"receipt", "manifest", "readme"}],
        },
    )

    samples = pd.read_parquet(root / global_manifest["files"]["samples"])
    sample_lookup = samples.set_index("sample_key")
    sample_split = sample_lookup["split"].astype(str).to_dict()
    sample_assay = sample_lookup["assay_type"].astype(str).to_dict()

    context_columns = [
        "sample_key", "split", "split_group", "split_unit", "source_id",
        "dataset", "assay_type", "technology", "donor_id", "section_id",
        "disease_status", "sex", "age_raw", "time_point_days",
        "time_lower_days", "time_upper_days", "time_status", "life_stage",
        "selected_cells", "annotation_eligible", "development_eligible",
        "spatial_eligible", "integration_eligible",
    ]
    context = samples[[column for column in context_columns if column in samples]].copy()
    context["integration_batch_labels"] = "assay_type|technology|dataset|donor_id|sample_key"
    context["integration_biology_labels"] = "anatomy|age|sex|cell_type_when_available"
    context.to_parquet(files["context"], index=False, compression="zstd")

    annotation_counts: dict[str, Counter[tuple[str, str, str]]] = {
        "train": Counter(), "validation": Counter(),
    }
    annotation_samples: dict[str, defaultdict[tuple[str, str, str], set[str]]] = {
        "train": defaultdict(set), "validation": defaultdict(set),
    }
    coverage: Counter[tuple[str, str, str]] = Counter()
    anatomy_train: Counter[tuple[str, str]] = Counter()
    anatomy_counts: Counter[tuple[str, str]] = Counter()

    requested = [
        "source_id", "row_index", "sample_key",
        *ANNOTATION_SOURCE_COLUMNS, *ANATOMY_SOURCE_COLUMNS, *SPATIAL_SOURCE_COLUMNS,
    ]
    read_columns = _safe_columns(source_cells, requested)
    required = {"source_id", "row_index", "sample_key"}
    if not required <= set(read_columns):
        raise ValueError("source cell index lacks required row identity columns")
    parquet = pq.ParquetFile(source_cells)
    for start in range(0, parquet.num_row_groups, row_group_batch_size):
        end = min(start + row_group_batch_size, parquet.num_row_groups)
        parts = [parquet.read_row_group(index, columns=read_columns).to_pandas() for index in range(start, end)]
        frame = pd.concat(parts, ignore_index=True)
        split = frame["sample_key"].astype(str).map(sample_split)
        assay = frame["sample_key"].astype(str).map(sample_assay)
        if split.isna().any():
            raise ValueError("source cell index contains a sample outside the global split")
        canonical = canonicalize_annotation_frame(frame)
        for split_name in SPLITS:
            split_mask = split.eq(split_name)
            for assay_name in sorted(set(assay.loc[split_mask].dropna().astype(str))):
                assay_mask = split_mask & assay.eq(assay_name)
                available = canonical["annotation_available"].loc[assay_mask]
                coverage[(split_name, assay_name, "total_cells")] += int(assay_mask.sum())
                coverage[(split_name, assay_name, "labeled_cells")] += int(available.sum())
            if split_name not in annotation_counts:
                continue
            selected = canonical.loc[split_mask & canonical["annotation_available"]]
            sample_values = frame.loc[selected.index, "sample_key"].astype(str)
            keys = list(
                zip(
                    selected["annotation_label"].astype(str),
                    selected["annotation_label_source"].astype(str),
                    selected["annotation_label_level"].astype(str),
                )
            )
            annotation_counts[split_name].update(keys)
            for key, sample_key in zip(keys, sample_values):
                annotation_samples[split_name][key].add(sample_key)

        if "uberon_id" in frame:
            uberon = frame["uberon_id"].map(_text)
            term = (
                frame["uberon_term"].map(_text)
                if "uberon_term" in frame
                else pd.Series(None, index=frame.index)
            )
            valid = uberon.notna()
            for (sample_key, uberon_id), count in frame.loc[valid].assign(
                _uberon=uberon.loc[valid].astype(str)
            ).groupby(["sample_key", "_uberon"], sort=False).size().items():
                anatomy_counts[(str(sample_key), str(uberon_id))] += int(count)
            train_valid = valid & split.eq("train")
            pairs = zip(uberon.loc[train_valid].astype(str), term.loc[train_valid].fillna("<missing>").astype(str))
            anatomy_train.update(pairs)

    _annotation_vocab_frame(
        annotation_counts["train"], annotation_samples["train"]
    ).to_parquet(files["annotation_train"], index=False, compression="zstd")
    _annotation_vocab_frame(
        annotation_counts["validation"], annotation_samples["validation"]
    ).to_parquet(files["annotation_validation"], index=False, compression="zstd")
    coverage_rows = []
    for split_name in SPLITS:
        for assay_name in sorted(set(sample_assay.values())):
            total = int(coverage[(split_name, assay_name, "total_cells")])
            labeled = int(coverage[(split_name, assay_name, "labeled_cells")])
            coverage_rows.append(
                {
                    "split": split_name, "assay_type": assay_name,
                    "total_cells": total, "labeled_cells": labeled,
                    "unlabeled_cells": total - labeled,
                    "coverage": (labeled / total if total else 0.0),
                    "test_label_distribution_exposed": False,
                }
            )
    pd.DataFrame(coverage_rows).to_parquet(
        files["annotation_coverage"], index=False, compression="zstd"
    )
    _anatomy_vocab_frame(anatomy_train).to_parquet(
        files["anatomy_train"], index=False, compression="zstd"
    )

    intervals = pd.read_parquet(root / global_manifest["files"]["development_intervals"])
    development, development_anatomy = _development_labels(intervals, anatomy_counts)
    development.to_parquet(files["development"], index=False, compression="zstd")
    development_anatomy.to_parquet(
        files["development_anatomy"], index=False, compression="zstd"
    )

    pseudotime = samples.loc[
        samples["development_eligible"].astype(bool),
        [
            "sample_key", "split", "split_group", "dataset", "assay_type",
            "time_point_days", "time_lower_days", "time_upper_days",
            "time_status", "life_stage", "selected_cells",
        ],
    ].copy()
    pseudotime["chronological_time_proxy_days"] = pseudotime["time_point_days"]
    pseudotime["pseudotime_ground_truth_available"] = False
    pseudotime["label_status"] = "chronological_proxy_only_not_lineage_pseudotime"
    pseudotime.to_parquet(files["pseudotime"], index=False, compression="zstd")

    niche_columns = [
        "sample_key", "split", "split_group", "dataset", "assay_type",
        "technology", "section_id", "selected_cells",
    ]
    niche = samples.loc[
        samples["spatial_eligible"].astype(bool),
        [column for column in niche_columns if column in samples],
    ].copy()
    niche["coordinate_columns"] = "spatial_x|spatial_y|spatial_z_when_available"
    niche["neighbor_scope"] = "section_local_query_excluded"
    niche["niche_supervision"] = "self_supervised_coordinate_graph"
    niche["niche_category_label_available"] = False
    niche["nonspatial_context"] = "null_identity_context"
    niche.to_parquet(files["niche"], index=False, compression="zstd")

    cell_view = {
        "format": "brainwm_lazy_cell_label_view_v1",
        "source_cells_path": str(source_cells),
        "join": {"cell": "sample_key", "sample": "samples.parquet.sample_key"},
        "row_identity": ["source_id", "row_index", "sample_key"],
        "annotation_source_columns": list(ANNOTATION_SOURCE_COLUMNS),
        "annotation_canonical_priority": list(ANNOTATION_PRIORITY),
        "anatomy_columns": list(ANATOMY_SOURCE_COLUMNS),
        "spatial_columns": list(SPATIAL_SOURCE_COLUMNS),
        "test_annotation_access": "explicit_evaluator_open_after_prediction_freeze",
        "expression_copied": False,
        "labels_copied": False,
    }
    _write_json(files["cell_view"], cell_view)

    manifest = {
        "format": TASK_LABEL_FORMAT,
        "builder_version": TASK_LABEL_BUILDER_VERSION,
        "label_contract_id": "LABELS-BRAINWM-HUMAN-GLOBAL-6-2-2-V1",
        "split_id": global_manifest.get("split_id"),
        "data_contract_id": global_manifest.get("data_contract_id"),
        "species": "human",
        "status": "ready_with_declared_unavailable_supervision",
        "zero_copy_cell_labels": True,
        "input": {
            "global_split_manifest": str(root / "manifest.json"),
            "source_cells": str(source_cells),
            "source_columns_read": read_columns,
        },
        "tasks": {
            "integration": {
                "status": "ready",
                "unit": "cell_with_sample_context",
                "labels": ["assay_type", "technology", "dataset", "donor_id", "sample_key", "anatomy"],
                "role": "batch_mixing_and_biology_preservation_covariates",
            },
            "annotation": {
                "status": "ready_where_source_labels_exist",
                "unit": "cell",
                "canonical_priority": list(ANNOTATION_PRIORITY),
                "test_labels": "sealed",
            },
            "development": {
                "status": "ready_population_level",
                "unit": "same_split_adjacent_time_population",
                "input": "X_source,t_source,t_target,context",
                "target": "target_population_expression_and_OT_barycentric_latent",
                "cell_lineage_claim": False,
            },
            "pseudotime": {
                "status": "proxy_only_not_rankable",
                "ground_truth_available": False,
                "proxy": "chronological_time_proxy_days",
            },
            "spatial_niche": {
                "status": "ready_for_self_supervised_spatial_tasks",
                "coordinate_context_available": True,
                "niche_category_ground_truth_available": False,
            },
            "perturbation": {
                "status": "blocked_data",
                "label_available": False,
                "reason": "healthy_observational_data_have_no_intervention_control_outcome",
            },
        },
        "files": {key: path.name for key, path in files.items() if key not in {"manifest", "readme"}},
        "outputs": [path.name for key, path in files.items() if key not in {"manifest", "readme"}],
        "counts": {
            "cells": int(sum(global_manifest["counts"]["by_split"][split]["cells"] for split in SPLITS)),
            "samples": int(len(samples)),
            "development_intervals": int(len(development)),
            "development_interval_anatomies": int(len(development_anatomy)),
            "spatial_samples": int(len(niche)),
            "pseudotime_proxy_samples": int(len(pseudotime)),
            "perturbation_rows": 0,
        },
        "coverage": {
            "annotation_by_split_assay": coverage_rows,
            "annotation_train_vocab_labels": int(len(annotation_counts["train"])),
            "annotation_validation_vocab_labels": int(len(annotation_counts["validation"])),
            "anatomy_train_vocab_labels": int(len(anatomy_train)),
        },
        "access_policy": {
            "train": "labels_allowed_for_fit",
            "validation": "labels_allowed_for_selection_only",
            "test": "labels_evaluator_only_after_prediction_freeze",
            "test_annotation_vocabulary_materialized": False,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_pid": int(os.getpid()),
    }
    _write_json(files["manifest"], manifest)
    # Keep the parent split discoverable without duplicating any cell labels.
    # A future explicit label rebuild therefore cannot silently detach the
    # label contract from the global benchmark artifact.
    global_manifest["task_labels_manifest"] = files["manifest"].name
    global_manifest["label_contract_id"] = manifest["label_contract_id"]
    global_manifest.setdefault("files", {})["task_labels_manifest"] = files["manifest"].name
    _write_json(root / "manifest.json", global_manifest)
    files["readme"].write_text(
        "# BrainWM task labels\n\n"
        "This directory keeps cell expression and cell labels zero-copy. "
        "`task_labels_manifest.json` is the label contract; cell labels are "
        "streamed from the source `cells.parquet` through `GlobalTaskLabelIndex`.\n\n"
        "Integration, annotation and population-level development labels are ready. "
        "Pseudotime is only a chronological proxy, spatial niche supervision is "
        "self-supervised from real coordinates, and perturbation is blocked because "
        "the healthy observational corpus has no intervention outcomes. Test labels "
        "remain sealed until predictions are frozen.\n",
        encoding="utf-8",
    )
    validate_global_task_labels(root, require_complete_receipt=False)
    _write_json(
        files["receipt"],
        {
            "format": "brainwm_task_labels_build_receipt_v1",
            "status": "complete",
            "pid": int(os.getpid()),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "benchmark_dir": str(root),
                "global_manifest": str(root / "manifest.json"),
                "source_cells": str(source_cells),
                "source_columns_read": read_columns,
                "row_group_batch_size": int(row_group_batch_size),
            },
            "outputs": [path.name for key, path in files.items() if key not in {"receipt", "manifest", "readme"}],
            "error": None,
        },
    )
    validate_global_task_labels(root)
    return manifest


class GlobalTaskLabelIndex:
    """Lazy, split-aware access to task labels under the global contract."""

    def __init__(
        self, benchmark_dir: str | Path, *, index: GlobalBenchmarkIndex | None = None
    ):
        self.root = Path(benchmark_dir).resolve()
        self.index = index or GlobalBenchmarkIndex(self.root)
        path = self.root / "task_labels_manifest.json"
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != TASK_LABEL_FORMAT:
            raise ValueError("unexpected BrainWM task-label format")

    def sample_labels(self, split: str, task: str) -> pd.DataFrame:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        if task not in TASKS:
            raise ValueError(f"unknown task: {task}")
        key = {
            "integration": "context",
            "pseudotime": "pseudotime",
            "spatial_niche": "niche",
            "development": "development",
        }.get(task)
        if key is None:
            if task == "perturbation":
                return pd.DataFrame()
            raise ValueError("annotation is cell-level; use iter_annotation_labels")
        frame = pd.read_parquet(self.root / self.manifest["files"][key])
        return frame.loc[frame["split"].astype(str).eq(split)].reset_index(drop=True)

    def iter_annotation_labels(
        self,
        split: str,
        *,
        allow_sealed_test_labels: bool = False,
        row_group_batch_size: int = 1,
    ) -> Iterator[pd.DataFrame]:
        if split == "test" and not allow_sealed_test_labels:
            raise PermissionError(
                "test annotation labels are sealed until predictions are frozen"
            )
        columns = ["source_id", "row_index", "sample_key", *ANNOTATION_SOURCE_COLUMNS]
        for frame in self.index.iter_cell_index(
            split, task="annotation", columns=columns,
            row_group_batch_size=row_group_batch_size,
        ):
            canonical = canonicalize_annotation_frame(frame)
            yield pd.concat([frame.reset_index(drop=True), canonical.reset_index(drop=True)], axis=1)

    def iter_spatial_context(
        self, split: str, *, row_group_batch_size: int = 1
    ) -> Iterator[pd.DataFrame]:
        columns = [
            "source_id", "row_index", "sample_key",
            *SPATIAL_SOURCE_COLUMNS, *ANATOMY_SOURCE_COLUMNS,
        ]
        yield from self.index.iter_cell_index(
            split, task="spatial", columns=columns,
            row_group_batch_size=row_group_batch_size,
        )

    def iter_cell_labels(
        self,
        split: str,
        *,
        task: str | None = None,
        allow_sealed_test_labels: bool = False,
        row_group_batch_size: int = 1,
    ) -> Iterator[pd.DataFrame]:
        """Stream canonical labels and context from the source cell index.

        This is the common trainer/evaluator view.  ``task`` may restrict the
        rows to annotation or spatial eligibility; otherwise all v2-split
        cells are streamed.  Test annotation fields remain evaluator-gated.
        """

        if split == "test" and not allow_sealed_test_labels and task in (None, "annotation"):
            raise PermissionError(
                "test annotation labels are sealed until predictions are frozen"
            )
        include_annotation = not (
            split == "test" and not allow_sealed_test_labels
        )
        source_columns = [
            "source_id", "row_index", "sample_key",
            *ANATOMY_SOURCE_COLUMNS, *SPATIAL_SOURCE_COLUMNS,
        ]
        if include_annotation:
            source_columns[3:3] = list(ANNOTATION_SOURCE_COLUMNS)
        for frame in self.index.iter_cell_index(
            split,
            task=task,
            columns=source_columns,
            row_group_batch_size=row_group_batch_size,
        ):
            canonical = canonicalize_annotation_frame(frame)
            output = pd.concat(
                [frame.reset_index(drop=True), canonical.reset_index(drop=True)],
                axis=1,
            )
            output["anatomy_available"] = output.get(
                "uberon_id", pd.Series(pd.NA, index=output.index)
            ).map(_text).notna()
            output["spatial_coordinates_available"] = (
                output.get("spatial_x", pd.Series(pd.NA, index=output.index)).map(_text).notna()
                & output.get("spatial_y", pd.Series(pd.NA, index=output.index)).map(_text).notna()
            )
            yield output


def validate_global_task_labels(
    benchmark_dir: str | Path,
    *,
    require_complete_receipt: bool = True,
) -> dict[str, Any]:
    """Validate compact label artifacts without reading expression arrays."""

    root = Path(benchmark_dir).resolve()
    task_manifest = json.loads((root / "task_labels_manifest.json").read_text(encoding="utf-8"))
    global_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if task_manifest.get("format") != TASK_LABEL_FORMAT:
        raise ValueError("unexpected task-label manifest format")
    if task_manifest.get("split_id") != global_manifest.get("split_id"):
        raise ValueError("task labels and global benchmark use different split IDs")
    if set(task_manifest.get("tasks", {})) != set(TASKS):
        raise ValueError("task-label manifest does not define all tasks")
    for filename in task_manifest.get("files", {}).values():
        if not (root / str(filename)).is_file():
            raise FileNotFoundError(root / str(filename))
    if task_manifest["tasks"]["pseudotime"].get("ground_truth_available") is not False:
        raise ValueError("chronological time may not be declared pseudotime ground truth")
    if task_manifest["tasks"]["spatial_niche"].get("niche_category_ground_truth_available") is not False:
        raise ValueError("coordinate context may not be declared a niche category label")
    if task_manifest["tasks"]["perturbation"].get("label_available") is not False:
        raise ValueError("healthy observational data may not expose perturbation labels")
    if task_manifest["access_policy"].get("test_annotation_vocabulary_materialized") is not False:
        raise ValueError("test annotation vocabulary must remain sealed")
    if (root / "annotation_vocab_test.parquet").exists():
        raise ValueError("test annotation vocabulary must not be materialized")

    samples = pd.read_parquet(root / global_manifest["files"]["samples"], columns=["sample_key", "split"])
    sample_splits = samples.set_index("sample_key")["split"].astype(str)
    context = pd.read_parquet(root / task_manifest["files"]["context"], columns=["sample_key", "split"])
    if set(context["sample_key"].astype(str)) != set(samples["sample_key"].astype(str)):
        raise ValueError("context labels do not cover the global sample manifest")
    expected = context["sample_key"].astype(str).map(sample_splits)
    if not expected.equals(context["split"].astype(str)):
        raise ValueError("context labels cross the global split")

    development = pd.read_parquet(root / task_manifest["files"]["development"])
    if development["interval_id"].duplicated().any():
        raise ValueError("development interval IDs are not unique")
    for row in development.itertuples(index=False):
        for column in ("source_sample_keys", "target_sample_keys"):
            for sample_key in _json_list(getattr(row, column)):
                if sample_key not in sample_splits.index or sample_splits.loc[sample_key] != row.split:
                    raise ValueError("development interval crosses the global split")

    coverage = pd.read_parquet(root / task_manifest["files"]["annotation_coverage"])
    if not (coverage["labeled_cells"] <= coverage["total_cells"]).all():
        raise ValueError("annotation coverage exceeds total cells")
    view = json.loads((root / task_manifest["files"]["cell_view"]).read_text(encoding="utf-8"))
    if view.get("labels_copied") is not False or view.get("expression_copied") is not False:
        raise ValueError("task labels must preserve the zero-copy contract")
    source = Path(view["source_cells_path"])
    available = set(pq.ParquetFile(source).schema.names)
    if not {"source_id", "row_index", "sample_key"} <= available:
        raise ValueError("source cell label view lacks row identity")
    receipt_path = root / "task_labels_build_receipt.json"
    if require_complete_receipt and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError("task-label build receipt is not complete")
    return {
        "format": TASK_LABEL_FORMAT,
        "status": "passed",
        "samples": int(len(samples)),
        "development_intervals": int(len(development)),
        "spatial_samples": int(task_manifest["counts"]["spatial_samples"]),
        "test_labels_sealed": True,
    }
