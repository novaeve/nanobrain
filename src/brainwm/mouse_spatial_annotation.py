"""Contracts shared by the fresh mouse spatial annotation benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from .training import BatchDescriptor, row_index_from_ptr
from .types import WorldModelBatch
from .modalities import mask_named_modalities

CONFIG_FORMAT = "brainwm_mouse_spatial_annotation_config_v1"
CONTRACT_ID = "ANN-MOUSE-SPATIAL-SECTION-FRESH-V1"
RAW_SU07_CONFIRM_CONTRACT_ID = "ANN-MOUSE-SPATIAL-RAW-SU07-CONFIRM-V1"
FEATURE_FORMAT = "brainwm_mouse_spatial_annotation_features_v1"
TRANSFORM_FORMAT = "brainwm_mouse_spatial_annotation_transform_v1"
PREPARATION_FORMAT = "brainwm_mouse_spatial_annotation_preparation_v1"
HEAD_RESULT_FORMAT = "brainwm_mouse_spatial_annotation_head_result_v1"
VALIDATION_GATE_FORMAT = "brainwm_mouse_spatial_annotation_validation_gate_v1"
RAW_SU07_CONFIRM_GATE_FORMAT = "brainwm_mouse_spatial_raw_su07_confirmation_gate_v1"
TEST_PREDICTION_FORMAT = "brainwm_mouse_spatial_annotation_test_prediction_v1"
TEST_RESULT_FORMAT = "brainwm_mouse_spatial_annotation_test_result_v1"

REPRESENTATIONS = (
    "raw_cell",
    "raw_neighbor",
    "stage0_cell",
    "stage0_neighbor",
    "stage0_su07",
)
PHASES = ("train", "validation", "test")
LABEL_COLUMNS = ("cl_term", "original_cell_type")
ROW_ID_COLUMNS = (
    "phase",
    "source_id",
    "sample_key",
    "descriptor_index",
    "descriptor_start",
    "descriptor_stop",
    "within_descriptor_index",
    "row_index",
)

FROZEN_DATA_PATHS = {
    "artifact_dir": "pretrain_dataset/artifacts/mouse_spatial_cell_index_v1_full_20260825",
    "morton_plan": "pretrain_dataset/artifacts/mouse_spatial_morton_plan_v2_section_local_20260825",
    "unified_vocabulary": "pretrain_dataset/artifacts/brainbeacon_unified_gene_vocabulary_v2_orthology_r116_20260827",
}
FROZEN_SECTIONS = {
    "train": (
        "mouse::MERFISH-C57BL6J-638850::C57BL6J-638850.35",
        "mouse::Zhuang-ABCA-1::Zhuang-ABCA-1.068",
        "mouse::Zhuang-ABCA-2::Zhuang-ABCA-2.039",
        "mouse::Zhuang-ABCA-3::Zhuang-ABCA-3.013",
    ),
    "validation": (
        "mouse::MERFISH-C57BL6J-638850::C57BL6J-638850.54",
        "mouse::Zhuang-ABCA-1::Zhuang-ABCA-1.108",
        "mouse::Zhuang-ABCA-2::Zhuang-ABCA-2.019",
        "mouse::Zhuang-ABCA-3::Zhuang-ABCA-3.008",
    ),
    "test": (
        "mouse::MERFISH-C57BL6J-638850::C57BL6J-638850.32",
        "mouse::Zhuang-ABCA-1::Zhuang-ABCA-1.067",
        "mouse::Zhuang-ABCA-2::Zhuang-ABCA-2.035",
        "mouse::Zhuang-ABCA-3::Zhuang-ABCA-3.012",
    ),
}
FROZEN_MODELS = {
    "model_config": "brain_world_model/runs/exploratory_multispecies_stage0_joint_full_homology_species_rr_1200_v1/config.resolved.json",
    "stage0_checkpoint": "brain_world_model/runs/exploratory_multispecies_stage0_joint_full_homology_species_rr_1200_v1/checkpoint_step_1000.pt",
    "su07_head": "brain_world_model/runs/su07_mouse_mcr_gene_task_head_20260903_gpu5/head_final.pt",
    "neighbor_k": 64,
    "neighbor_pseudocount": 400.0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_workspace_path(workspace: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workspace / path).resolve()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "experiment config must be a mapping")
    require(payload.get("format") == CONFIG_FORMAT, "unexpected config format")
    require(payload.get("experiment_id") == CONTRACT_ID, "unexpected contract ID")
    data = payload.get("data")
    require(isinstance(data, dict), "config.data must be a mapping")
    for key, expected in FROZEN_DATA_PATHS.items():
        require(data.get(key) == expected, f"data.{key} drifted")
    require(data.get("sources") == [0, 1, 2, 3], "source set drifted")
    require(data.get("labels") == list(LABEL_COLUMNS), "label hierarchy drifted")
    require(int(data.get("batch_size", 0)) == 256, "batch size drifted")
    require(
        int(data.get("descriptors_per_section", 0)) == 16,
        "descriptor count drifted",
    )
    require(int(data.get("common_genes", 0)) == 477, "common panel drifted")
    require(
        int(data.get("cells_per_phase", 0)) == 16_384,
        "phase population drifted",
    )
    require(
        int(data.get("query_folds_per_descriptor", 0)) == 8,
        "query-fold count drifted",
    )
    sections = data.get("sections")
    require(isinstance(sections, dict), "section map is required")
    phase_sections: dict[str, set[str]] = {}
    for phase in PHASES:
        values = sections.get(phase)
        require(
            isinstance(values, list)
            and len(values) == 4
            and len(set(map(str, values))) == 4,
            f"{phase} must contain four unique sections",
        )
        phase_sections[phase] = set(map(str, values))
    require(
        sum(map(len, phase_sections.values()))
        == len(set().union(*phase_sections.values())),
        "train, validation, and test sections must be disjoint",
    )
    for phase in PHASES:
        require(
            tuple(map(str, sections[phase])) == FROZEN_SECTIONS[phase],
            f"{phase} section assignment drifted",
        )
    require(
        tuple(payload.get("representations", ())) == REPRESENTATIONS,
        "representation set or order drifted",
    )
    frozen_models = payload.get("frozen_models")
    require(isinstance(frozen_models, dict), "frozen_models must be a mapping")
    require(frozen_models == FROZEN_MODELS, "frozen model or neighborhood settings drifted")
    head = payload.get("head")
    require(isinstance(head, dict), "head settings must be a mapping")
    seeds = head.get("seeds")
    require(seeds == [20260931, 20260932, 20260933], "head seeds drifted")
    require(
        {
            key: head.get(key)
            for key in (
                "hidden_dim",
                "residual_mode",
                "dropout",
                "max_steps",
                "validation_interval",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "gradient_clip_norm",
            )
        }
        == {
            "hidden_dim": 256,
            "residual_mode": "feature",
            "dropout": 0.1,
            "max_steps": 800,
            "validation_interval": 40,
            "batch_size": 512,
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "gradient_clip_norm": 5.0,
        },
        "annotation head contract drifted",
    )
    evaluation = payload.get("evaluation")
    require(isinstance(evaluation, dict), "evaluation settings must be a mapping")
    require(
        evaluation
        == {
            "primary_metric": "coverage_adjusted_macro_f1",
            "direction": "maximize",
            "selection_tiebreak": ["balanced_accuracy", "earlier_step"],
            "validation_gate": [
                "stage0_su07_strictly_beats_best_raw_baseline",
                "stage0_su07_strictly_beats_stage0_neighbor",
            ],
            "test_label_reads": 1,
            "post_test_updates": False,
        },
        "evaluation contract drifted",
    )
    return payload


def lower_median_fresh_test_sections(
    samples: pd.DataFrame,
    sources: Sequence[int],
    consumed_sections: Iterable[str],
) -> tuple[dict[int, str], dict[int, int]]:
    """Select test sections from metadata only, with no label-dependent choice."""

    required = {"source_id", "sample_key", "split"}
    require(required <= set(samples), "sample inventory lacks section fields")
    consumed = set(map(str, consumed_sections))
    selected: dict[int, str] = {}
    eligible_counts: dict[int, int] = {}
    for source_id in map(int, sources):
        values = sorted(
            set(
                samples.loc[
                    samples["source_id"].astype(int).eq(source_id)
                    & samples["split"].astype(str).eq("test"),
                    "sample_key",
                ].astype(str)
            )
            - consumed
        )
        eligible_counts[source_id] = len(values)
        if values:
            selected[source_id] = values[(len(values) - 1) // 2]
    return selected, eligible_counts


def common_species_gene_panel(
    source_gene_map: pd.DataFrame,
    unified_genes: pd.DataFrame,
    sources: Sequence[int],
) -> pd.DataFrame:
    """Return the exact one-to-one species-gene intersection for all sources."""

    map_required = {
        "source_id",
        "species_gene_key",
        "gene_symbol",
        "mapping_status",
    }
    gene_required = {
        "gene_index",
        "species_gene_key",
        "species",
        "gene_symbol",
        "gene_type",
    }
    require(map_required <= set(source_gene_map), "source gene map schema drifted")
    require(gene_required <= set(unified_genes), "unified gene schema drifted")
    source_ids = tuple(map(int, sources))
    keys: list[set[str]] = []
    for source_id in source_ids:
        frame = source_gene_map[
            source_gene_map["source_id"].astype(int).eq(source_id)
        ]
        require(not frame.empty, f"source {source_id} has no genes")
        # MERFISH source 0 contains blank-control probes. They have no species
        # gene identity and are excluded before taking the biological panel
        # intersection; they must never be converted to the string "nan".
        mapped = frame[
            frame["species_gene_key"].notna()
            & frame["mapping_status"].astype(str).eq("mapped_gencode")
        ]
        require(not mapped.empty, f"source {source_id} has no mapped genes")
        values = set(mapped["species_gene_key"].astype(str))
        require(len(values) == len(mapped), f"source {source_id} repeats a gene key")
        keys.append(values)
    shared = set.intersection(*keys)
    genes = unified_genes[
        unified_genes["species_gene_key"].astype(str).isin(shared)
    ].copy()
    require(len(genes) == len(shared), "shared genes do not map one-to-one to vocabulary")
    require(set(genes["species"].astype(str)) == {"mouse"}, "panel is not mouse-only")
    genes = genes[
        ["gene_index", "species_gene_key", "gene_symbol", "gene_type"]
    ].sort_values("gene_index", kind="stable")
    genes.insert(0, "panel_index", np.arange(len(genes), dtype=np.int64))
    return genes.reset_index(drop=True)


def descriptor_rows(
    phase: str,
    selected: Sequence[Any],
    plan: Any,
) -> pd.DataFrame:
    """Materialize label-free row identity for selected Morton descriptors."""

    rows: list[pd.DataFrame] = []
    for descriptor_index, selected_descriptor in enumerate(selected):
        descriptor: BatchDescriptor = selected_descriptor.descriptor
        frame = plan.frame(descriptor)
        require(len(frame) == descriptor.stop - descriptor.start, "descriptor size drifted")
        require(
            set(frame["sample_key"].astype(str)) == {selected_descriptor.sample_key},
            "descriptor crossed a section boundary",
        )
        row = pd.DataFrame(
            {
                "phase": phase,
                "source_id": int(descriptor.source_id),
                "sample_key": str(selected_descriptor.sample_key),
                "descriptor_index": int(descriptor_index),
                "descriptor_start": int(descriptor.start),
                "descriptor_stop": int(descriptor.stop),
                "within_descriptor_index": np.arange(len(frame), dtype=np.int16),
                "row_index": frame["row_index"].to_numpy(np.int64),
            }
        )
        rows.append(row)
    require(bool(rows), f"{phase} descriptor selection is empty")
    result = pd.concat(rows, ignore_index=True)
    require(not result.duplicated(["source_id", "row_index"]).any(), "rows repeat")
    return result.loc[:, ROW_ID_COLUMNS]


def attach_open_labels(rows: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join already-authorized train/validation labels and fail on ambiguity."""

    join_columns = ["phase", "source_id", "sample_key", "row_index"]
    require(set(join_columns) <= set(rows), "row identity schema drifted")
    required = {*join_columns, *LABEL_COLUMNS}
    require(required <= set(labels), "label frame schema drifted")
    require(
        not labels.duplicated(join_columns).any(),
        "label frame repeats row identity",
    )
    joined = rows.merge(
        labels.loc[:, [*join_columns, *LABEL_COLUMNS]],
        on=join_columns,
        how="left",
        validate="one_to_one",
    )
    require(joined.loc[:, LABEL_COLUMNS].notna().all(axis=None), "labels are missing")
    return joined


def validate_split_manifest(
    rows: pd.DataFrame,
    sections: Mapping[str, Sequence[str]],
    sources: Sequence[int],
) -> None:
    """Validate exact section ownership and row isolation across all phases."""

    require(tuple(rows.columns) == ROW_ID_COLUMNS, "row manifest schema drifted")
    require(set(rows["phase"].astype(str)) == set(PHASES), "manifest phases drifted")
    require(
        not rows.duplicated(["source_id", "row_index"]).any(),
        "row identity crosses phases",
    )
    source_ids = tuple(map(int, sources))
    for phase in PHASES:
        expected = set(
            zip(source_ids, map(str, sections[phase]), strict=True)
        )
        phase_rows = rows[rows["phase"].astype(str).eq(phase)]
        actual = set(
            phase_rows.loc[:, ["source_id", "sample_key"]]
            .astype({"source_id": int, "sample_key": str})
            .itertuples(index=False, name=None)
        )
        require(actual == expected, f"{phase} section/source assignment drifted")


def mask_exact_cells(batch: WorldModelBatch, cell_mask: Tensor) -> WorldModelBatch:
    """Zero every token for an explicit set of query cells."""

    if cell_mask.dtype != torch.bool or cell_mask.shape != (
        batch.expression.batch_size,
    ):
        raise ValueError("cell_mask must be boolean with shape [B]")
    if not torch.any(cell_mask) or torch.all(cell_mask):
        raise ValueError("cell_mask must retain query and source cells")
    row_index = row_index_from_ptr(batch.expression.row_ptr)
    token_mask = cell_mask[row_index]
    if batch.expression.token_mask is not None:
        existing = batch.expression.token_mask
        if existing.dtype != torch.bool or existing.shape != token_mask.shape:
            raise ValueError("existing token_mask must be boolean and match token count")
        token_mask = token_mask | existing
    counts = batch.expression.count.clone()
    counts.masked_fill_(token_mask, 0.0)
    expression = replace(batch.expression, count=counts, token_mask=token_mask)
    modality_inputs, modality_presence = mask_named_modalities(
        batch.modality_inputs,
        batch.modality_presence,
        cell_mask,
    )
    return replace(
        batch,
        expression=expression,
        modality_inputs=modality_inputs,
        modality_presence=modality_presence,
    )


def query_fold_masks(
    batch_size: int, folds: int, *, device: torch.device | str = "cpu"
) -> tuple[Tensor, ...]:
    """Partition descriptor rows into deterministic, exhaustive query folds."""

    if batch_size < 2 or folds < 2 or folds > batch_size:
        raise ValueError("query folds require 2 <= folds <= batch_size")
    indices = torch.arange(batch_size, device=device)
    masks = tuple(indices.remainder(folds).eq(fold) for fold in range(folds))
    coverage = torch.stack(masks).sum(dim=0)
    require(bool(torch.all(coverage == 1)), "query folds are not a partition")
    require(all(bool(mask.any()) for mask in masks), "query fold is empty")
    return masks


def manifest_reorder_indices(
    actual_row_ids: Sequence[int], expected_row_ids: Sequence[int]
) -> np.ndarray:
    """Return positions that restore reader-sorted rows to manifest order."""

    actual = np.asarray(actual_row_ids, dtype=np.int64)
    expected = np.asarray(expected_row_ids, dtype=np.int64)
    require(actual.ndim == expected.ndim == 1, "row IDs must be vectors")
    require(len(actual) == len(expected), "reader and manifest row counts differ")
    require(len(np.unique(actual)) == len(actual), "reader row IDs repeat")
    require(len(np.unique(expected)) == len(expected), "manifest row IDs repeat")
    positions = pd.Index(actual).get_indexer(expected)
    require(bool(np.all(positions >= 0)), "reader and manifest row sets differ")
    require(
        bool(np.array_equal(actual[positions], expected)),
        "reader rows cannot be restored to manifest order",
    )
    return positions.astype(np.int64, copy=False)


def clr_from_counts_or_probability(value: Tensor, scale: float = 1.0e4) -> Tensor:
    """Apply the project's centered log-library transform to compositions."""

    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError("value must be a floating matrix")
    if not torch.isfinite(value).all() or torch.any(value < 0):
        raise ValueError("value must be finite and nonnegative")
    if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    probability = value / value.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    logged = torch.log1p(probability * float(scale))
    return logged - logged.mean(dim=1, keepdim=True)


@torch.no_grad()
def fit_compositional_pca(
    train_value: Tensor, dimension: int = 256
) -> tuple[Tensor, Tensor, float]:
    """Fit an exact PCA basis on training rows only, on the current device."""

    clr = clr_from_counts_or_probability(train_value).float()
    maximum = min(clr.shape[0] - 1, clr.shape[1])
    if not 1 <= int(dimension) <= maximum:
        raise ValueError(f"dimension must be in [1, {maximum}]")
    feature_mean = clr.mean(dim=0)
    centered = clr - feature_mean
    covariance = centered.T @ centered / float(max(len(centered) - 1, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[: int(dimension)]
    selected_values = eigenvalues.index_select(0, order).clamp_min(0.0)
    components = eigenvectors.index_select(1, order)
    anchors = components.abs().argmax(dim=0)
    positions = torch.arange(len(anchors), device=components.device)
    signs = components[anchors, positions].sign().masked_fill_(
        components[anchors, positions].eq(0), 1.0
    )
    components = components * signs
    denominator = eigenvalues.clamp_min(0.0).sum().clamp_min(1.0e-12)
    explained = float((selected_values.sum() / denominator).item())
    require(bool(torch.isfinite(components).all()), "PCA components are non-finite")
    return feature_mean, components, explained


@torch.no_grad()
def apply_compositional_pca(
    value: Tensor, feature_mean: Tensor, components: Tensor
) -> Tensor:
    """Apply a frozen train-fitted compositional PCA transform."""

    if feature_mean.ndim != 1 or components.ndim != 2:
        raise ValueError("PCA mean and components have invalid rank")
    if value.shape[1] != len(feature_mean) or components.shape[0] != len(feature_mean):
        raise ValueError("PCA feature dimensions do not match")
    if value.device != feature_mean.device or value.device != components.device:
        raise ValueError("PCA tensors must share one device")
    result = (clr_from_counts_or_probability(value).float() - feature_mean) @ components
    require(bool(torch.isfinite(result).all()), "PCA output is non-finite")
    return result


def representation_states(
    representation: str,
    raw: Tensor,
    stage0: Tensor,
    neighbor: Tensor,
    su07: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build the matched coarse/fine state pair for one declared method."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    values = (raw, stage0, neighbor, su07)
    if any(value.ndim != 2 or value.shape[1] != 256 for value in values):
        raise ValueError("all transformed views must have shape [N, 256]")
    if len({value.shape[0] for value in values}) != 1:
        raise ValueError("transformed views have different row counts")
    zeros = torch.zeros_like(raw)
    if representation == "raw_cell":
        cell, spatial = raw, zeros
    elif representation == "raw_neighbor":
        cell, spatial = raw, neighbor
    elif representation == "stage0_cell":
        cell, spatial = stage0, zeros
    elif representation == "stage0_neighbor":
        cell, spatial = stage0, neighbor
    else:
        cell, spatial = stage0, su07
    return cell, torch.cat((cell, spatial), dim=1)


def stable_label_vocabulary(values: Sequence[object]) -> tuple[str, ...]:
    vocabulary = tuple(sorted(set(map(str, values))))
    require(bool(vocabulary), "label vocabulary is empty")
    return vocabulary


def encode_labels(values: Sequence[object], vocabulary: Sequence[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(vocabulary)}
    return np.asarray([lookup.get(str(value), -1) for value in values], dtype=np.int64)


def classification_metrics(
    target: Sequence[object],
    prediction: Sequence[object],
    train_vocabulary: Sequence[str],
) -> dict[str, float | int]:
    """Compute the frozen coverage-adjusted fine-label metric."""

    import warnings

    from sklearn.metrics import balanced_accuracy_score, f1_score

    truth = np.asarray(list(map(str, target)), dtype=object)
    predicted = np.asarray(list(map(str, prediction)), dtype=object)
    require(truth.shape == predicted.shape and truth.ndim == 1, "labels misalign")
    known = np.isin(truth, np.asarray(tuple(train_vocabulary), dtype=object))
    coverage = float(known.mean()) if len(known) else 0.0
    if not np.any(known):
        return {
            "rows": len(truth),
            "known_rows": 0,
            "coverage": coverage,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "coverage_adjusted_macro_f1": 0.0,
        }
    labels = list(map(str, train_vocabulary))
    macro_f1 = float(
        f1_score(truth[known], predicted[known], labels=labels, average="macro", zero_division=0)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true", category=UserWarning
        )
        balanced = float(balanced_accuracy_score(truth[known], predicted[known]))
    return {
        "rows": len(truth),
        "known_rows": int(known.sum()),
        "coverage": coverage,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced,
        "coverage_adjusted_macro_f1": coverage * macro_f1,
    }


def load_json(path: str | Path) -> Mapping[str, Any]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload


__all__ = [
    "CONFIG_FORMAT",
    "CONTRACT_ID",
    "FEATURE_FORMAT",
    "FROZEN_DATA_PATHS",
    "FROZEN_MODELS",
    "FROZEN_SECTIONS",
    "HEAD_RESULT_FORMAT",
    "LABEL_COLUMNS",
    "PHASES",
    "PREPARATION_FORMAT",
    "RAW_SU07_CONFIRM_CONTRACT_ID",
    "RAW_SU07_CONFIRM_GATE_FORMAT",
    "REPRESENTATIONS",
    "ROW_ID_COLUMNS",
    "TEST_PREDICTION_FORMAT",
    "TEST_RESULT_FORMAT",
    "TRANSFORM_FORMAT",
    "VALIDATION_GATE_FORMAT",
    "apply_compositional_pca",
    "attach_open_labels",
    "classification_metrics",
    "clr_from_counts_or_probability",
    "common_species_gene_panel",
    "descriptor_rows",
    "encode_labels",
    "fit_compositional_pca",
    "load_experiment_config",
    "load_json",
    "lower_median_fresh_test_sections",
    "manifest_reorder_indices",
    "mask_exact_cells",
    "query_fold_masks",
    "representation_states",
    "require",
    "resolve_workspace_path",
    "stable_label_vocabulary",
    "validate_split_manifest",
]
