"""Source-only diversity-stratified cohort planning.

This module is an opt-in data-layer candidate.  It does not change the
historical :class:`~brainwm.training.IndexedBatchPlan` schedule or any model
configuration.  The planner consumes only indexed cell provenance and source
metadata, then returns a fixed-size cell cohort that can be passed through the
existing source-local batch plan.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PLAN_FORMAT = "brainwm_diversity_stratified_cohort_v1"
STRATA = ("assay_type", "source_id", "donor_id", "region")
_MISSING_TOKENS = frozenset({"", "<na>", "<nat>", "nan", "none", "null", "unknown", "unk"})


def _is_missing(value: Any) -> bool:
    """Treat null-like metadata as missing without interpreting labels."""
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _MISSING_TOKENS


def _text(value: Any, fallback: str) -> str:
    if _is_missing(value):
        return fallback
    return str(value)


def _normalized_series(values: pd.Series, fallback: Any) -> pd.Series:
    """Normalize null-like values without materializing Python string lists."""
    result = values.astype("string")
    missing = result.isna() | result.str.strip().str.lower().isin(_MISSING_TOKENS)
    if isinstance(fallback, pd.Series):
        replacement = fallback.astype("string")
        result = result.where(~missing, replacement)
    else:
        result = result.mask(missing, str(fallback))
    return result


def _source_local_ordered(frame: pd.DataFrame) -> bool:
    """Check source/row order using compact numeric views."""
    if len(frame) < 2:
        return True
    source = frame["source_id"].to_numpy(copy=False)
    row = frame["row_index"].to_numpy(copy=False)
    return bool(
        np.logical_or(
            source[1:] > source[:-1],
            np.logical_and(source[1:] == source[:-1], row[1:] >= row[:-1]),
        ).all()
    )


def _stable_rank(value: Any, seed: int) -> int:
    """Return a process-independent tie-break rank for ``value``."""
    payload = f"brainwm-diversity-v1\0{int(seed)}\0{value!r}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _metadata_index(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    """Normalize sample metadata to a unique ``sample_key`` index."""
    metadata = sample_metadata.copy()
    if "sample_key" in metadata.columns:
        metadata = metadata.set_index("sample_key", drop=False)
    if metadata.index.name is None:
        raise ValueError("sample_metadata must have a sample_key column or index")
    index = metadata.index.astype("string")
    if index.isna().any() or index.duplicated().any():
        raise ValueError("sample_metadata sample_key index must be non-null and unique")
    metadata.index = index
    return metadata


def _mapped_column(
    cells: pd.DataFrame,
    metadata: pd.DataFrame,
    name: str,
) -> pd.Series:
    """Prefer explicit cell provenance, filling missing values from samples."""
    values = cells[name] if name in cells.columns else pd.Series(pd.NA, index=cells.index)
    if "sample_key" not in cells.columns or name not in metadata.columns:
        return values
    mapped = cells["sample_key"].astype("string").map(metadata[name])
    return _normalized_series(values, mapped)


def _region_values(cells: pd.DataFrame, metadata: pd.DataFrame) -> pd.Series:
    """Resolve a source region without requiring a target annotation."""
    candidates: list[pd.Series] = []
    for name in ("region", "uberon_term", "original_anatomical_region", "uberon_id"):
        if name in cells.columns:
            candidates.append(cells[name])
        elif name in metadata.columns and "sample_key" in cells.columns:
            candidates.append(cells["sample_key"].astype("string").map(metadata[name]))
    if not candidates:
        return pd.Series(pd.NA, index=cells.index)
    result = candidates[0].copy()
    for candidate in candidates[1:]:
        result = _normalized_series(result, candidate)
    return result


def _largest_remainder_quotas(
    capacities: Mapping[Any, int],
    budget: int,
    order: Sequence[Any],
) -> dict[Any, int]:
    """Allocate an exact uniform quota with capacity-aware Hamilton rounding.

    The declared ``order`` is the deterministic fractional-remainder tie
    breaker.  Uniform weights deliberately make this a diversity allocation:
    each selected composite stratum receives the same ideal share before
    leftovers are distributed.  Capacities are handled by redistributing the
    quota of exhausted strata.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    keys = list(order)
    if len(set(keys)) != len(keys):
        raise ValueError("order must contain unique quota keys")
    if set(keys) != set(capacities):
        raise ValueError("order and capacities must contain the same keys")
    normalized = {key: int(value) for key, value in capacities.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("quota capacities must be nonnegative")
    if budget > sum(normalized.values()):
        raise ValueError("budget exceeds total quota capacity")

    quota = {key: 0 for key in keys}
    remaining = int(budget)
    active = [key for key in keys if normalized[key] > 0]
    while remaining and active:
        # Uniform ideal shares are the largest-remainder allocation.  Re-run
        # after a group reaches capacity so its unused share is redistributed.
        base, remainder = divmod(remaining, len(active))
        if base:
            allocated = 0
            for key in active:
                add = min(base, normalized[key] - quota[key])
                quota[key] += add
                allocated += add
            remaining -= allocated
            active = [key for key in active if quota[key] < normalized[key]]
            if allocated:
                continue
        # ``remaining < len(active)`` (or every residual capacity is below
        # ``base``): assign one unit to the first fractional remainders.  All
        # fractions tie under uniform weights, so ``order`` is authoritative.
        if remainder or not base:
            for key in list(active):
                if not remaining:
                    break
                if quota[key] < normalized[key]:
                    quota[key] += 1
                    remaining -= 1
            active = [key for key in active if quota[key] < normalized[key]]
    if remaining:
        raise RuntimeError("largest-remainder allocation failed to exhaust budget")
    return quota


def _coverage_order(
    records: Sequence[dict[str, Any]],
    seed: int,
    dimensions: Sequence[str],
    budget: int,
) -> list[dict[str, Any]]:
    """Order composite strata by deterministic marginal coverage.

    A max-coverage pass gives assay/source/region values a chance to appear
    before the larger donor cardinality dominates.  Once all marginal values
    are covered, stable seeded ranks provide the remaining order.
    """
    remaining = list(records)
    result: list[dict[str, Any]] = []
    covered = {dimension: set() for dimension in dimensions}
    for _ in range(min(len(remaining), budget)):
        best_index = 0
        best_score: tuple[int, int] | None = None
        for index, record in enumerate(remaining):
            new_values = sum(
                int(record["values"][dimension] not in covered[dimension])
                for dimension in dimensions
            )
            tie = -_stable_rank(record["key"], seed)
            score = (new_values, tie)
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        record = remaining.pop(best_index)
        result.append(record)
        for dimension in dimensions:
            covered[dimension].add(record["values"][dimension])
    # ``budget`` can exceed the number of composite strata.  Include every
    # stratum in seeded order before applying quotas.
    remaining.sort(key=lambda item: _stable_rank(item["key"], seed))
    result.extend(remaining)
    return result


def _json_key(key: Iterable[Any]) -> str:
    return json.dumps([str(value) for value in key], ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class DiversityStratifiedCohort:
    """A fixed-budget, provenance-only cell cohort and its audit metadata."""

    selected_cells: pd.DataFrame
    split: str
    cell_budget: int
    seed: int
    strata: tuple[str, ...]
    donor_fallback: str
    quota_by_composite: dict[str, int]
    strata_counts: dict[str, dict[str, int]]

    @property
    def format(self) -> str:
        return PLAN_FORMAT

    def to_indexed_batch_plan(
        self,
        reader: Any,
        batch_size: int,
        drop_last: bool = False,
    ) -> Any:
        """Build the existing source-local plan over this selected cohort.

        A tiny reader view keeps the original reader and artifact untouched;
        ``IndexedBatchPlan`` and ``build_batch`` remain the sole batch APIs.
        """
        from .training import IndexedBatchPlan

        view = SimpleNamespace(cells=self.selected_cells, sources=reader.sources)
        return IndexedBatchPlan(view, self.split, batch_size, drop_last)

    def manifest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "split": self.split,
            "cell_budget": self.cell_budget,
            "selected_cells": int(len(self.selected_cells)),
            "seed": self.seed,
            "strata": list(self.strata),
            "donor_fallback": self.donor_fallback,
            "quota_by_composite": self.quota_by_composite,
            "strata_counts": self.strata_counts,
            "source_only": True,
            "target_labels_used": False,
        }


def plan_diversity_stratified_cells(
    cells: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    split: str = "train",
    cell_budget: int,
    seed: int = 0,
    strata: Sequence[str] = STRATA,
) -> DiversityStratifiedCohort:
    """Plan an exact cell budget using source-only diversity strata.

    ``cells`` must contain ``source_id``, ``row_index``, ``sample_key`` and
    ``split``.  ``sample_metadata`` supplies assay and donor fields;
    extra columns (including labels) are ignored.  Missing donors fall back to
    ``sample_key`` so unrelated samples are never silently merged into one
    donor stratum.
    """
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if int(cell_budget) != cell_budget or cell_budget <= 0:
        raise ValueError("cell_budget must be a positive integer")
    cell_budget = int(cell_budget)
    requested_strata = tuple(str(value) for value in strata)
    if requested_strata != STRATA:
        raise ValueError(f"strata must be exactly {STRATA!r} for this planner")
    required_cells = {"source_id", "row_index", "sample_key", "split"}
    missing = sorted(required_cells - set(cells.columns))
    if missing:
        raise ValueError(f"cells missing required columns: {missing}")

    metadata = _metadata_index(sample_metadata)
    if "split" not in metadata.columns:
        raise ValueError("sample_metadata must include split for leakage checking")
    # Keep only columns needed to build a provenance stratum.  The indexed
    # artifact also stores cell annotations and other metadata, but carrying
    # those through a full-corpus planning pass needlessly multiplies memory.
    split_mask = cells["split"].eq(split)
    needed_columns = [
        column for column in (
            "source_id", "row_index", "sample_key", "split", "region",
            "uberon_term", "original_anatomical_region",
        ) if column in cells.columns
    ]
    frame = cells.loc[split_mask, needed_columns].copy()
    if frame.empty:
        raise ValueError(f"no cells found for split={split!r}")
    # Make the plan independent of the caller's DataFrame row order.  The
    # backed reader contract already treats ``(source_id, row_index)`` as the
    # stable source-local identity.
    if not _source_local_ordered(frame):
        frame = frame.sort_values(
            ["source_id", "row_index"], kind="mergesort"
        ).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    if cell_budget > len(frame):
        raise ValueError(
            f"cell_budget={cell_budget} exceeds available {split!r} cells={len(frame)}"
        )
    sample_keys = frame["sample_key"].astype("string")
    unknown = sorted(set(sample_keys.dropna()) - set(metadata.index))
    if unknown:
        raise ValueError(f"sample_metadata missing sample keys: {unknown[:5]}")
    metadata_split = sample_keys.map(metadata["split"])
    missing_metadata_split = metadata_split.isna()
    if bool(missing_metadata_split.any()):
        raise ValueError("sample_metadata has missing split for selected samples")
    mismatch = metadata_split.astype("string").ne(split)
    if bool(mismatch.any()):
        raise ValueError("sample_metadata split mismatch would leak another split")

    # The planner reads only these provenance dimensions.  In particular, no
    # cell type, disease, perturbation, or target evaluation column is touched.
    assay = _mapped_column(frame, metadata, "assay_type")
    donor_raw = _mapped_column(frame, metadata, "donor_id")
    region_raw = _region_values(frame, metadata)
    source = frame["source_id"]
    canonical = pd.DataFrame(index=frame.index)
    canonical["assay_type"] = _normalized_series(assay, "__missing_assay__")
    canonical["source_id"] = _normalized_series(source, "__missing_source__")
    canonical["region"] = _normalized_series(region_raw, "__missing_region__")
    sample_fallback = sample_keys.astype("string").fillna(canonical["source_id"])
    canonical["donor_id"] = _normalized_series(donor_raw, sample_fallback)
    canonical["__row_pos"] = range(len(frame))

    group_columns = list(requested_strata)
    # Categorical codes keep the provenance-only working frame bounded for a
    # full indexed corpus while retaining the original string values in the
    # returned cell frame.
    for column in group_columns:
        canonical[column] = canonical[column].astype("category")
    grouped = canonical.groupby(
        group_columns, sort=False, dropna=False, observed=True
    )
    records: list[dict[str, Any]] = []
    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = dict(zip(group_columns, key_tuple))
        positions = group["__row_pos"].astype("int64").tolist()
        records.append({
            "key": key_tuple,
            "values": values,
            "positions": positions,
            "capacity": len(positions),
        })
    if not records:
        raise ValueError("no non-empty diversity strata found")

    ordered = _coverage_order(records, seed, requested_strata, cell_budget)
    chosen = ordered[: min(len(ordered), cell_budget)]
    capacities = {record["key"]: record["capacity"] for record in chosen}
    keys = [record["key"] for record in chosen]
    quotas = _largest_remainder_quotas(capacities, cell_budget, keys)

    selected_positions: list[int] = []
    quota_manifest: dict[str, int] = {}
    for record in chosen:
        quota = quotas[record["key"]]
        if not quota:
            continue
        positions = sorted(record["positions"])
        offset = _stable_rank(record["key"], seed) % len(positions)
        rotated = positions[offset:] + positions[:offset]
        selected_positions.extend(rotated[:quota])
        quota_manifest[_json_key(record["key"])] = int(quota)
    if len(selected_positions) != cell_budget:
        raise RuntimeError("diversity planner did not produce the requested budget")

    selected_work = canonical.iloc[selected_positions]
    # Source-local ordering is required by the backed H5AD reader and is also
    # what makes the output directly reusable by IndexedBatchPlan.
    selected = frame.iloc[selected_positions].copy()
    selected["__planner_order"] = range(len(selected))
    selected = selected.sort_values(["source_id", "row_index"], kind="mergesort")
    selected = selected.drop(columns=["__planner_order"])
    counts: dict[str, dict[str, int]] = {}
    for dimension in requested_strata:
        values = selected_work[dimension].astype("string")
        counts[dimension] = {
            str(key): int(value)
            for key, value in values.value_counts(sort=True).items()
        }
    return DiversityStratifiedCohort(
        selected_cells=selected.reset_index(drop=True),
        split=split,
        cell_budget=cell_budget,
        seed=int(seed),
        strata=requested_strata,
        donor_fallback="sample_key_then_source_id",
        quota_by_composite=quota_manifest,
        strata_counts=counts,
    )


class DiversityStratifiedPlanner:
    """Small stateful adapter convenient for an opt-in training caller."""

    def __init__(self, cell_budget: int, *, seed: int = 0):
        self.cell_budget = int(cell_budget)
        self.seed = int(seed)

    def plan(
        self,
        cells: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        *,
        split: str = "train",
    ) -> DiversityStratifiedCohort:
        return plan_diversity_stratified_cells(
            cells,
            sample_metadata,
            split=split,
            cell_budget=self.cell_budget,
            seed=self.seed,
        )


# Public spelling for callers that want to test the allocator independently.
largest_remainder_quotas = _largest_remainder_quotas


__all__ = [
    "DiversityStratifiedCohort",
    "DiversityStratifiedPlanner",
    "PLAN_FORMAT",
    "STRATA",
    "largest_remainder_quotas",
    "plan_diversity_stratified_cells",
]
