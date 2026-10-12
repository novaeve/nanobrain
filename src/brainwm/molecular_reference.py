"""Deterministic source-only cohorts for molecular reference mapping.

The selector consumes an already projected in-memory cell table and owns no
Parquet or H5AD I/O.  This implementation is appropriate for the current
roughly 15-million-cell molecular snapshot.  A 100M-scale corpus needs a
separate two-pass adapter: first count strata, then retain bounded hash top-k
rows per allocated stratum without materializing the complete cell table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .diversity_planner import largest_remainder_quotas

MOLECULAR_REFERENCE_FORMAT = "brainwm_molecular_source_cohort_v1"
MOLECULAR_ASSAYS = ("scRNA", "snRNA")
DEFAULT_CELLS_PER_LABEL = 448
DEFAULT_SELECTION_SEED = 20260827
_MISSING_TOKENS = frozenset(
    {"", "<na>", "<nat>", "nan", "none", "null", "unknown", "unk"}
)
_IDENTITY_COLUMNS = ("source_id", "row_index", "sample_key")
_HASH_COLUMNS = (
    "source_id",
    "row_index",
    "sample_key",
    "split",
    "assay_type",
    "dataset",
    "donor_id",
    "label",
    "source_group",
)


def _normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def _valid_text(values: pd.Series) -> pd.Series:
    normalized = _normalized_text(values)
    return ~normalized.str.lower().isin(_MISSING_TOKENS)


def _stable_rank(seed: int, *values: object) -> int:
    payload = json.dumps(
        [MOLECULAR_REFERENCE_FORMAT, int(seed), *(str(value) for value in values)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _normalize_assays(assays: Sequence[str]) -> tuple[str, ...]:
    requested = (assays,) if isinstance(assays, str) else tuple(assays)
    requested = tuple(str(value).strip() for value in requested)
    if not requested:
        raise ValueError("assays must contain scRNA, snRNA, or both")
    if len(set(requested)) != len(requested):
        raise ValueError("assays must not contain duplicates")
    unsupported = sorted(set(requested) - set(MOLECULAR_ASSAYS))
    if unsupported:
        raise ValueError(f"unsupported molecular assays: {unsupported}")
    return tuple(assay for assay in MOLECULAR_ASSAYS if assay in requested)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _normalize_source_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("source_id", "row_index"):
        numeric = pd.to_numeric(result[column], errors="raise")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain non-null integers")
        result[column] = numeric.astype("int64")
    result["sample_key"] = _normalized_text(result["sample_key"])
    if not _valid_text(result["sample_key"]).all():
        raise ValueError("sample_key must contain non-missing values")
    if result.duplicated(["source_id", "row_index"]).any():
        raise ValueError("projected cells contain duplicate source_id/row_index keys")
    return result


def _validate_projected_sample_metadata(frame: pd.DataFrame) -> None:
    columns = ["sample_key", "split", "assay_type", "dataset", "donor_id"]
    metadata = frame[columns].copy()
    for column in columns:
        normalized = _normalized_text(metadata[column])
        metadata[column] = normalized.mask(~_valid_text(normalized), "__missing__")
    metadata = metadata.drop_duplicates()
    inconsistent = (
        metadata.groupby("sample_key", sort=False, dropna=False)
        .size()
        .loc[lambda values: values > 1]
    )
    if not inconsistent.empty:
        examples = inconsistent.index.astype(str).tolist()[:5]
        raise ValueError(
            "projected sample metadata is inconsistent for sample_key values: "
            f"{examples}"
        )


def _encode_source_group(kind: str, values: Sequence[str]) -> str:
    return (
        f"{kind}:{json.dumps(list(values), ensure_ascii=False, separators=(',', ':'))}"
    )


def decode_source_group_id(value: object) -> dict[str, str | None]:
    """Decode a tagged source-group ID into its unambiguous components."""
    text = str(value)
    try:
        kind, payload = text.split(":", 1)
        values = json.loads(payload)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid tagged source-group ID: {text!r}") from error
    if kind == "donor" and isinstance(values, list) and len(values) == 2:
        dataset, donor_id = (str(item) for item in values)
        return {
            "kind": "donor",
            "dataset": dataset,
            "donor_id": donor_id,
            "sample_key": None,
        }
    if kind == "sample" and isinstance(values, list) and len(values) == 1:
        return {
            "kind": "sample",
            "dataset": None,
            "donor_id": None,
            "sample_key": str(values[0]),
        }
    raise ValueError(f"invalid tagged source-group ID: {text!r}")


def qualified_source_group_ids(frame: pd.DataFrame) -> pd.Series:
    """Return reversible tagged donor IDs, falling back to tagged sample IDs."""
    required = {"sample_key", "dataset", "donor_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"projected cells missing group columns: {missing}")
    sample_key = _normalized_text(frame["sample_key"])
    if not _valid_text(sample_key).all():
        raise ValueError("sample_key must contain non-missing values")
    dataset = _normalized_text(frame["dataset"])
    donor = _normalized_text(frame["donor_id"])
    metadata = pd.DataFrame(
        {"sample_key": sample_key, "dataset": dataset, "donor_id": donor}
    ).drop_duplicates()
    conflicts = metadata.groupby("sample_key", sort=False).size()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index.astype(str).tolist()[:5]
        raise ValueError(
            "projected group metadata is inconsistent for sample_key values: "
            f"{examples}"
        )
    mapping: dict[str, str] = {}
    for row in metadata.itertuples(index=False):
        if bool(_valid_text(pd.Series([row.dataset])).iloc[0]) and bool(
            _valid_text(pd.Series([row.donor_id])).iloc[0]
        ):
            group = _encode_source_group("donor", (row.dataset, row.donor_id))
        else:
            group = _encode_source_group("sample", (row.sample_key,))
        mapping[str(row.sample_key)] = group
    groups = sample_key.map(mapping)
    if groups.isna().any():
        raise RuntimeError("failed to encode every projected source group")
    return groups.astype("string")


def _assay_quotas(
    capacities: dict[str, int],
    assays: tuple[str, ...],
    label_budget: int,
    cells_per_label: int,
) -> dict[str, int]:
    """Split a label budget by assay, then refill any unavailable half."""
    if label_budget < 0 or label_budget > sum(capacities.values()):
        raise ValueError("label budget must fit assay capacities")
    if len(assays) == 1:
        return {assays[0]: label_budget}

    initial_targets = {
        assays[0]: cells_per_label // 2,
        assays[1]: cells_per_label - cells_per_label // 2,
    }
    quotas = {
        assay: min(capacities.get(assay, 0), initial_targets[assay]) for assay in assays
    }
    remaining = label_budget - sum(quotas.values())
    for assay in assays:
        if not remaining:
            break
        available = capacities.get(assay, 0) - quotas[assay]
        add = min(remaining, available)
        quotas[assay] += add
        remaining -= add
    if remaining:
        raise RuntimeError("assay quota allocation did not exhaust the label budget")
    return quotas


def _quota_table(
    candidates: pd.DataFrame,
    assays: tuple[str, ...],
    cells_per_label: int,
    seed: int,
) -> pd.DataFrame:
    capacities = (
        candidates.groupby(
            ["label", "assay_type", "source_group"],
            sort=True,
            observed=True,
        )
        .size()
        .rename("available_cells")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for label, label_frame in capacities.groupby("label", sort=True):
        assay_capacities = {
            assay: int(
                label_frame.loc[
                    label_frame["assay_type"].eq(assay), "available_cells"
                ].sum()
            )
            for assay in assays
        }
        label_budget = min(cells_per_label, sum(assay_capacities.values()))
        assay_quotas = _assay_quotas(
            assay_capacities, assays, label_budget, cells_per_label
        )
        for assay in assays:
            assay_frame = label_frame[label_frame["assay_type"].eq(assay)]
            if assay_frame.empty:
                continue
            group_capacities = {
                str(row.source_group): int(row.available_cells)
                for row in assay_frame.itertuples(index=False)
            }
            group_order = sorted(
                group_capacities,
                key=lambda group: (_stable_rank(seed, label, assay, group), group),
            )
            group_quotas = largest_remainder_quotas(
                group_capacities, assay_quotas[assay], group_order
            )
            for row in assay_frame.itertuples(index=False):
                rows.append(
                    {
                        "label": str(label),
                        "assay_type": str(assay),
                        "source_group": str(row.source_group),
                        "available_cells": int(row.available_cells),
                        "selected_quota": int(group_quotas[str(row.source_group)]),
                        "label_cap": cells_per_label,
                        "label_budget": label_budget,
                        "assay_quota": int(assay_quotas[assay]),
                        "fairness_unit": "label_x_assay_within_source_group",
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty or int(result["selected_quota"].sum()) <= 0:
        raise ValueError("molecular quota allocation selected no cells")
    return result.sort_values(
        ["label", "assay_type", "source_group"], kind="mergesort"
    ).reset_index(drop=True)


def _select_from_quotas(
    candidates: pd.DataFrame,
    quotas: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    quota_lookup = {
        (str(row.label), str(row.assay_type), str(row.source_group)): int(
            row.selected_quota
        )
        for row in quotas.itertuples(index=False)
    }
    selected_indices: list[int] = []
    grouped = candidates.groupby(
        ["label", "assay_type", "source_group"],
        sort=True,
        observed=True,
    )
    for key, group in grouped:
        canonical_key = tuple(str(value) for value in key)
        quota = quota_lookup[canonical_key]
        if quota == 0:
            continue
        ordered = group.sort_values(
            ["source_id", "row_index", "sample_key"], kind="mergesort"
        )
        rng = np.random.default_rng(_stable_rank(seed, *canonical_key))
        positions = np.sort(rng.choice(len(ordered), size=quota, replace=False))
        selected_indices.extend(ordered.index.to_numpy()[positions].tolist())
    result = candidates.loc[selected_indices].copy()
    expected = int(quotas["selected_quota"].sum())
    if len(result) != expected:
        raise RuntimeError(
            f"molecular selector returned {len(result)} cells, expected {expected}"
        )
    return result.sort_values(
        ["source_id", "row_index", "sample_key"], kind="mergesort"
    ).reset_index(drop=True)


def molecular_source_selection_sha256(frame: pd.DataFrame) -> str:
    """Hash canonical selected identities independently of DataFrame order."""
    missing = sorted(set(_HASH_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"selected cohort missing hash columns: {missing}")
    canonical = _normalize_source_identity(frame[list(_HASH_COLUMNS)])
    for column in ("split", "assay_type", "label", "source_group"):
        canonical[column] = _normalized_text(canonical[column])
        if not _valid_text(canonical[column]).all():
            raise ValueError(f"selected cohort has missing {column} values")
    for column in ("dataset", "donor_id"):
        normalized = _normalized_text(canonical[column])
        canonical[column] = normalized.mask(~_valid_text(normalized), "__missing__")
    for group in canonical["source_group"].unique():
        decode_source_group_id(group)
    canonical = canonical.sort_values(
        ["source_id", "row_index", "sample_key"], kind="mergesort"
    )
    digest = hashlib.sha256()
    digest.update(f"{MOLECULAR_REFERENCE_FORMAT}\n".encode())
    for row in canonical.itertuples(index=False, name=None):
        digest.update(
            (
                json.dumps(
                    [str(value) for value in row],
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def molecular_quota_sha256(quotas: pd.DataFrame) -> str:
    """Hash the complete quota contract independently of row order."""
    columns = (
        "label",
        "assay_type",
        "source_group",
        "available_cells",
        "selected_quota",
        "label_cap",
        "label_budget",
        "assay_quota",
    )
    missing = sorted(set(columns) - set(quotas.columns))
    if missing:
        raise ValueError(f"quota audit missing hash columns: {missing}")
    canonical = quotas[list(columns)].sort_values(
        ["label", "assay_type", "source_group"], kind="mergesort"
    )
    digest = hashlib.sha256()
    digest.update(f"{MOLECULAR_REFERENCE_FORMAT}:quotas\n".encode())
    for row in canonical.itertuples(index=False, name=None):
        digest.update(
            (
                json.dumps(list(row), ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _fairness_audit(quotas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (label, assay), part in quotas.groupby(
        ["label", "assay_type"], sort=True, observed=True
    ):
        selected = part[part["selected_quota"].gt(0)]
        unsaturated = part[part["selected_quota"].lt(part["available_cells"])]
        rows.append(
            {
                "label": str(label),
                "assay_type": str(assay),
                "available_cells": int(part["available_cells"].sum()),
                "selected_cells": int(part["selected_quota"].sum()),
                "available_source_groups": len(part),
                "selected_source_groups": len(selected),
                "saturated_source_groups": int(
                    part["selected_quota"].eq(part["available_cells"]).sum()
                ),
                "unsaturated_quota_min": (
                    int(unsaturated["selected_quota"].min())
                    if not unsaturated.empty
                    else None
                ),
                "unsaturated_quota_max": (
                    int(unsaturated["selected_quota"].max())
                    if not unsaturated.empty
                    else None
                ),
                "fairness_unit": "label_x_assay_within_source_group",
                "combined_donor_global_fairness_claimed": False,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["label", "assay_type"], kind="mergesort")
        .reset_index(drop=True)
    )


def _label_support_audit(
    candidates: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    available = candidates.groupby("label", sort=True, observed=True).agg(
        available_cells=("label", "size"),
        available_groups=("source_group", "nunique"),
        available_assays=("assay_type", "nunique"),
    )
    chosen = selected.groupby("label", sort=True, observed=True).agg(
        selected_cells=("label", "size"),
        selected_groups=("source_group", "nunique"),
        selected_assays=("assay_type", "nunique"),
    )
    result = available.join(chosen, how="left").fillna(0).reset_index()
    for column in (
        "available_cells",
        "available_groups",
        "available_assays",
        "selected_cells",
        "selected_groups",
        "selected_assays",
    ):
        result[column] = result[column].astype(int)
    result["singleton_available_group"] = result["available_groups"].eq(1)
    result["singleton_selected_group"] = result["selected_groups"].eq(1)
    result["requires_forced_train_in_group_cv"] = result["singleton_selected_group"]
    return result.sort_values("label", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class MolecularSourceCohort:
    """Selected source cells plus quota and group-support provenance."""

    selected_cells: pd.DataFrame
    quota_audit: pd.DataFrame
    fairness_audit: pd.DataFrame
    label_support_audit: pd.DataFrame
    assays: tuple[str, ...]
    split: str
    label_column: str
    cells_per_label: int
    seed: int
    selection_sha256: str
    quota_sha256: str

    @property
    def format(self) -> str:
        return MOLECULAR_REFERENCE_FORMAT

    @property
    def labels_requiring_forced_train(self) -> tuple[str, ...]:
        rows = self.label_support_audit.loc[
            self.label_support_audit["singleton_selected_group"], "label"
        ]
        return tuple(rows.astype(str))

    def manifest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "source_cohort_mode": "molecular_balanced",
            "source_context": "not_applicable_dissociated_molecular_cells",
            "assays": list(self.assays),
            "split": self.split,
            "label_column": self.label_column,
            "cells_per_label": self.cells_per_label,
            "selected_cells": len(self.selected_cells),
            "selected_labels": int(self.label_support_audit["label"].nunique()),
            "selected_groups": int(self.selected_cells["source_group"].nunique()),
            "labels_requiring_forced_train": list(self.labels_requiring_forced_train),
            "cv_split_applied": False,
            "cv_requirement": (
                "downstream grouped CV must keep each listed label's only "
                "source group in train"
            ),
            "group_definition": (
                "tagged donor:[dataset,donor_id] or sample:[sample_key] fallback"
            ),
            "combined_assay_allocation": "half_then_capacity_refill",
            "quota_fairness_unit": "label_x_assay_within_source_group",
            "combined_donor_global_fairness_claimed": False,
            "fairness_audit_units": len(self.fairness_audit),
            "selection_engine": "in_memory_projected_dataframe",
            "scale_scope": "current_approximately_15m_molecular_snapshot",
            "streaming_implementation": False,
            "scale_100m_requirement": (
                "two_pass_stratum_counts_then_bounded_hash_top_k"
            ),
            "seed": self.seed,
            "selection_sha256": self.selection_sha256,
            "quota_sha256": self.quota_sha256,
            "source_only": True,
            "target_expression_used": False,
            "target_labels_used": False,
        }


def select_molecular_source_cohort(
    projected_cells: pd.DataFrame,
    *,
    assays: Sequence[str],
    split: str = "train",
    label_column: str = "cl_id",
    cells_per_label: int = DEFAULT_CELLS_PER_LABEL,
    seed: int = DEFAULT_SELECTION_SEED,
) -> MolecularSourceCohort:
    """Select a label- and group-balanced scRNA/snRNA source cohort.

    ``projected_cells`` must already contain cell identity, split, assay,
    dataset, donor and label columns.  Target data, spatial coordinates and
    Morton descriptors are neither required nor inspected.  The API is
    intentionally in-memory for the current roughly 15M-cell snapshot; it is
    not the 100M-scale two-pass bounded top-k implementation.
    """
    normalized_assays = _normalize_assays(assays)
    cells_per_label = _positive_integer(cells_per_label, "cells_per_label")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if not isinstance(label_column, str) or not label_column:
        raise ValueError("label_column must be a non-empty string")
    required = {
        *_IDENTITY_COLUMNS,
        "split",
        "assay_type",
        "dataset",
        "donor_id",
        label_column,
    }
    missing = sorted(required - set(projected_cells.columns))
    if missing:
        raise ValueError(f"projected cells missing required columns: {missing}")

    # Validate the complete projection before filtering so a duplicate source
    # identity cannot be hidden by assigning its copies to different splits or
    # assays.  Sample metadata conflicts outside the requested arm also fail.
    projected_all = _normalize_source_identity(projected_cells)
    _validate_projected_sample_metadata(projected_all)
    projected_split = _normalized_text(projected_all["split"])
    projected_assay = _normalized_text(projected_all["assay_type"])
    projected = projected_all.loc[
        projected_split.eq(split) & projected_assay.isin(normalized_assays)
    ].copy()
    if projected.empty:
        raise ValueError(
            f"no projected cells found for split={split!r}, assays={normalized_assays}"
        )
    projected["split"] = _normalized_text(projected["split"])
    projected["assay_type"] = _normalized_text(projected["assay_type"])
    projected["dataset"] = _normalized_text(projected["dataset"])
    projected["donor_id"] = _normalized_text(projected["donor_id"])
    labels = _normalized_text(projected[label_column])
    projected = projected.loc[_valid_text(labels)].copy()
    projected["label"] = labels.loc[projected.index].astype("string")
    if projected.empty:
        raise ValueError(f"no usable source labels in column {label_column!r}")
    projected["source_group"] = qualified_source_group_ids(projected)
    projected = projected.sort_values(
        ["source_id", "row_index", "sample_key"], kind="mergesort"
    ).reset_index(drop=True)

    quotas = _quota_table(projected, normalized_assays, cells_per_label, int(seed))
    selected = _select_from_quotas(projected, quotas, int(seed))
    fairness = _fairness_audit(quotas)
    support = _label_support_audit(projected, selected)
    selection_sha256 = molecular_source_selection_sha256(selected)
    quota_sha256 = molecular_quota_sha256(quotas)
    return MolecularSourceCohort(
        selected_cells=selected,
        quota_audit=quotas,
        fairness_audit=fairness,
        label_support_audit=support,
        assays=normalized_assays,
        split=split,
        label_column=label_column,
        cells_per_label=cells_per_label,
        seed=int(seed),
        selection_sha256=selection_sha256,
        quota_sha256=quota_sha256,
    )


__all__ = [
    "DEFAULT_CELLS_PER_LABEL",
    "DEFAULT_SELECTION_SEED",
    "MOLECULAR_ASSAYS",
    "MOLECULAR_REFERENCE_FORMAT",
    "MolecularSourceCohort",
    "decode_source_group_id",
    "molecular_quota_sha256",
    "molecular_source_selection_sha256",
    "qualified_source_group_ids",
    "select_molecular_source_cohort",
]
