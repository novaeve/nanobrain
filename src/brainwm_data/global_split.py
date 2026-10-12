"""Global benchmark split and task manifests for BrainWM.

The source corpus is a zero-copy index.  This module only materializes a
small sample/group manifest and task eligibility tables; expression counts and
the large ``cells.parquet`` table remain in the source artifact.  A single
group assignment is shared by integration, annotation, development and
spatial tasks so that benchmark numbers are comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GLOBAL_SPLIT_FORMAT = "brainwm_global_benchmark_split_v1"
BUILDER_VERSION = "1.0.0"
SPLITS: tuple[str, ...] = ("train", "validation", "test")
TASKS: tuple[str, ...] = ("integration", "annotation", "development", "perturbation")
AUXILIARY_TASKS: tuple[str, ...] = ("spatial",)
DEFAULT_FRACTIONS: dict[str, float] = {
    "train": 0.6,
    "validation": 0.2,
    "test": 0.2,
}
DEFAULT_TIME_STATUSES = frozenset({"exact", "approximate", "range_midpoint"})
DEFAULT_ASSAYS = ("scRNA", "snRNA", "spatialRNA")


@dataclass(frozen=True)
class BenchmarkCorpusSpec:
    """Metadata-only description of one indexed corpus.

    The split builder intentionally accepts a small, reader-independent spec
    instead of opening H5AD files.  ``artifact_dir`` must contain the usual
    ``samples.parquet`` and ``sources.parquet`` tables (and may contain a
    ``cells.parquet`` table for lazy lookup).  ``species`` is an explicit
    annotation for corpora whose tables do not carry one; it is never inferred
    from expression values.

    ``name`` is provided as a compatibility alias for callers that already use
    :class:`brainwm_data.mixed_indexed_h5ad.CorpusSpec`.
    """

    corpus_id: str
    artifact_dir: str | Path
    species: str | None = None
    species_id: int | None = None
    sample_key_prefix: str | None = None
    species_filter: str | None = None
    # Optional explicit reader dispatch.  Kept at the end for compatibility
    # with callers that construct this dataclass positionally.
    reader_kind: str | None = None

    @property
    def name(self) -> str:
        return self.corpus_id

    def __post_init__(self) -> None:
        corpus = _canonical_corpus_id(self.corpus_id)
        object.__setattr__(self, "corpus_id", corpus)
        if self.species is not None and not _text(self.species):
            raise ValueError("corpus species must be non-empty when provided")
        if self.species_id is not None and int(self.species_id) < 0:
            raise ValueError("species_id must be nonnegative")
        if self.sample_key_prefix is not None and not _text(self.sample_key_prefix):
            raise ValueError("sample_key_prefix must be non-empty when provided")
        if self.reader_kind is not None and not _text(self.reader_kind):
            raise ValueError("reader_kind must be non-empty when provided")


# ``GlobalCorpusSpec`` is a descriptive alias used by external callers.  Keep
# the longer name above as the canonical public type to avoid colliding with
# the existing reader-level ``CorpusSpec`` export.
GlobalCorpusSpec = BenchmarkCorpusSpec
# Local alias for callers that import the split-layer spec directly.  The
# package-level ``brainwm_data.CorpusSpec`` remains the reader-layer type.
CorpusSpec = BenchmarkCorpusSpec


def _canonical_corpus_id(value: Any) -> str:
    """Return a stable, path-independent identifier safe for composite keys."""
    text = _text(value)
    if text is None:
        raise ValueError("corpus_id must be non-empty")
    # Do not silently let ``::`` create ambiguous nested namespaces.  A small
    # slug transformation keeps IDs deterministic while retaining readable
    # names from existing artifact directories.
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    if not slug:
        raise ValueError(f"corpus_id has no usable characters: {value!r}")
    return slug


def qualify_sample_key(corpus_id: Any, local_sample_key: Any) -> str:
    """Return the deterministic global key for one corpus-local sample."""
    corpus = _canonical_corpus_id(corpus_id)
    local = _text(local_sample_key)
    if local is None:
        raise ValueError("local_sample_key must be non-empty")
    return local if local.startswith(corpus + "::") else f"{corpus}::{local}"


def qualify_source_id(corpus_id: Any, local_source_id: Any) -> str:
    """Return the deterministic global source identifier."""
    corpus = _canonical_corpus_id(corpus_id)
    try:
        number = float(local_source_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("local_source_id must be an integer") from exc
    if not math.isfinite(number) or number != math.floor(number) or number < 0:
        raise ValueError("local_source_id must be a nonnegative integer")
    return f"{corpus}::source::{int(number)}"


def qualify_row_id(corpus_id: Any, local_source_id: Any, row_index: Any) -> str:
    """Return a stable row key independent of parquet ordering."""
    source = qualify_source_id(corpus_id, local_source_id)
    try:
        number = float(row_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_index must be an integer") from exc
    if not math.isfinite(number) or number != math.floor(number) or number < 0:
        raise ValueError("row_index must be a nonnegative integer")
    return f"{source}::row::{int(number)}"


def _infer_species_from_corpus(corpus_id: str, artifact: Path) -> str | None:
    """Infer only a conservative species hint from explicit metadata/path."""
    lowered = f"{corpus_id} {artifact.name}".lower()
    if "mouse" in lowered or "mice" in lowered:
        return "mouse"
    if "human" in lowered:
        return "human"
    return None


def _coerce_benchmark_corpus_specs(
    artifact_dir: str | Path | Mapping[str, Any] | Sequence[Any] | None,
    corpus_specs: Sequence[Any] | Mapping[str, Any] | None,
    corpora: Sequence[Any] | Mapping[str, Any] | None,
    *,
    default_species: str | Sequence[str] | None,
) -> tuple[list[BenchmarkCorpusSpec], bool]:
    """Normalize legacy and multi-corpus inputs.

    Returns ``(specs, explicit_multi)``.  ``explicit_multi`` lets the caller
    route a one-item ``corpus_specs`` call through alias normalization while
    retaining byte-for-byte-compatible behavior for the historical positional
    single-artifact API.
    """
    provided = [value is not None for value in (corpus_specs, corpora)]
    if sum(provided) > 1:
        raise ValueError("pass only one of corpus_specs or corpora")
    raw: Any = corpus_specs if corpus_specs is not None else corpora
    explicit_multi = raw is not None
    if raw is None:
        raw = artifact_dir
    elif artifact_dir is not None:
        raise ValueError("artifact_dir cannot be combined with corpus_specs/corpora")
    if raw is None:
        raise ValueError("an artifact_dir or corpus_specs value is required")

    # A mapping is the convenient ``{"human": "/...", "mouse": "/..."``
    # form.  A mapping that already describes one spec is accepted as well.
    entries: list[tuple[str | None, Any]] = []
    if isinstance(raw, Mapping):
        spec_keys = {
            "corpus_id", "name", "id", "artifact_dir", "artifact", "path",
            "species", "species_id", "sample_key_prefix", "species_filter",
            "reader_kind",
        }
        if spec_keys.intersection(raw):
            entries.append((None, raw))
        else:
            entries.extend((str(key), value) for key, value in raw.items())
    elif isinstance(raw, (str, Path, BenchmarkCorpusSpec)) or (
        hasattr(raw, "artifact_dir") and (hasattr(raw, "name") or hasattr(raw, "corpus_id"))
    ):
        entries.append((None, raw))
    else:
        try:
            entries = [(None, value) for value in list(raw)]
        except TypeError as exc:
            raise TypeError("corpus specs must be a path, mapping, or sequence") from exc
    if not entries:
        raise ValueError("at least one corpus spec is required")

    specs: list[BenchmarkCorpusSpec] = []
    for position, (mapping_name, value) in enumerate(entries):
        if isinstance(value, BenchmarkCorpusSpec):
            spec = value
            if mapping_name is not None and _canonical_corpus_id(mapping_name) != spec.corpus_id:
                raise ValueError(
                    f"mapping key {mapping_name!r} disagrees with corpus_id={spec.corpus_id!r}"
                )
            specs.append(spec)
            continue
        # Accept the reader-layer CorpusSpec without importing that module (it
        # would eagerly import H5AD readers).  Only its small metadata surface
        # is needed by the split builder.
        if hasattr(value, "artifact_dir") and (
            hasattr(value, "name") or hasattr(value, "corpus_id")
        ):
            corpus_value = getattr(value, "corpus_id", getattr(value, "name", None))
            specs.append(
                BenchmarkCorpusSpec(
                    corpus_id=str(corpus_value),
                    artifact_dir=getattr(value, "artifact_dir"),
                    species=_text(getattr(value, "species", None)),
                    species_id=getattr(value, "species_id", None),
                    sample_key_prefix=_text(getattr(value, "sample_key_prefix", None)),
                    reader_kind=_text(getattr(value, "reader_kind", None)),
                )
            )
            continue
        if isinstance(value, Mapping):
            path_value = value.get("artifact_dir", value.get("artifact", value.get("path")))
            if path_value is None:
                raise ValueError("corpus spec mapping requires artifact_dir/path")
            corpus_value = value.get(
                "corpus_id", value.get("name", value.get("id", mapping_name))
            )
            if corpus_value is None:
                corpus_value = Path(path_value).name or f"corpus_{position}"
            species_value = value.get("species")
            species_id = value.get("species_id")
            prefix = value.get("sample_key_prefix")
            species_filter = value.get("species_filter")
            reader_kind = value.get("reader_kind")
        else:
            path_value = value
            corpus_value = mapping_name or Path(path_value).name or f"corpus_{position}"
            species_value = None
            species_id = None
            prefix = None
            species_filter = None
            reader_kind = None
        artifact = Path(path_value).resolve()
        if species_value is None:
            species_value = _infer_species_from_corpus(str(corpus_value), artifact)
        if species_value is None:
            if isinstance(default_species, (list, tuple)):
                species_value = (
                    default_species[position]
                    if position < len(default_species) else None
                )
            elif len(entries) == 1:
                species_value = default_species
        specs.append(
            BenchmarkCorpusSpec(
                corpus_id=str(corpus_value),
                artifact_dir=artifact,
                species=_text(species_value),
                species_id=(int(species_id) if species_id is not None else None),
                sample_key_prefix=_text(prefix),
                species_filter=_text(species_filter),
                reader_kind=_text(reader_kind),
            )
        )
    names = [spec.corpus_id.casefold() for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("corpus IDs must be unique (case-insensitive)")
    return specs, explicit_multi or len(specs) > 1


def _nullish(value: Any) -> bool:
    """Return whether a metadata scalar should be treated as unavailable."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {
        "nan", "none", "null", "<na>", "na", "n/a", "unknown", "unk",
        "not_available", "not available",
    }


def _text(value: Any) -> str | None:
    if _nullish(value):
        return None
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


def _yes(value: Any) -> bool:
    text = _text(value)
    return text is not None and text.lower() in {"yes", "true", "1", "y"}


def _stable_rank(seed: str, key: str) -> bytes:
    # This hashes only small group identifiers, never expression or metadata
    # files.  It gives deterministic ordering independent of Python hash salt.
    return hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=16).digest()


def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    values = {name: float(fractions[name]) for name in SPLITS}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("split fractions must be finite and positive")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("split fractions must sum to one")
    return values


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _canonical_species(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    aliases = {
        "homo sapiens": "human",
        "human": "human",
        "mus musculus": "mouse",
        "mouse": "mouse",
        "mice": "mouse",
        "macaca mulatta": "macaque",
        "macaque": "macaque",
        "callithrix jacchus": "marmoset",
        "marmoset": "marmoset",
    }
    lowered = text.lower()
    return aliases.get(lowered, lowered)


def _canonical_assay(value: Any) -> str:
    text = _text(value)
    if text is None:
        return "unknown"
    aliases = {
        "scrna": "scRNA",
        "sc-rna": "scRNA",
        "single-cell rna": "scRNA",
        "single-cell rna-seq": "scRNA",
        "snrna": "snRNA",
        "sn-rna": "snRNA",
        "single-nucleus rna": "snRNA",
        "single-nucleus rna-seq": "snRNA",
        "spatial": "spatialRNA",
        "spatialrna": "spatialRNA",
        "spatial transcriptomics": "spatialRNA",
        "spatial-rna": "spatialRNA",
    }
    return aliases.get(text.lower(), text)


def _valid_identifier(value: Any) -> str | None:
    text = _text(value)
    return text


def _validate_donor_scope(value: str) -> str:
    scope = str(value).strip().lower()
    if scope not in {"dataset", "corpus", "global"}:
        raise ValueError("donor_scope must be one of: dataset, corpus, global")
    return scope


def derive_split_groups(
    samples: pd.DataFrame,
    *,
    donor_scope: str = "dataset",
) -> pd.DataFrame:
    """Add a leakage-safe group key and task eligibility to sample rows.

    The grouping priority is donor, matched sample, spatial section, then
    sample.  ``donor_scope='dataset'`` (the historical behavior) qualifies a
    donor by dataset; ``donor_scope='corpus'`` additionally qualifies it by
    corpus; ``donor_scope='global'`` treats the donor identifier as a global
    identity so the same subject cannot cross train/validation/test.
    """
    donor_scope = _validate_donor_scope(donor_scope)
    required = ("sample_key", "dataset", "assay_type")
    _require_columns(samples, required, "samples")
    frame = samples.copy()
    if "selected_cells" not in frame:
        if "n_observations" not in frame:
            raise ValueError("samples needs selected_cells or n_observations")
        frame["selected_cells"] = frame["n_observations"]
    frame["selected_cells"] = pd.to_numeric(frame["selected_cells"], errors="raise")
    if (frame["selected_cells"] <= 0).any():
        raise ValueError("selected_cells must be positive")
    frame["assay_type"] = frame["assay_type"].map(_canonical_assay)
    frame["dataset"] = frame["dataset"].map(lambda value: _text(value) or "unknown_dataset")
    frame["sample_key"] = frame["sample_key"].map(
        lambda value: _text(value) or ""
    )
    if frame["sample_key"].eq("").any() or frame["sample_key"].duplicated().any():
        raise ValueError("sample_key must be present and globally unique")

    group_keys: list[str] = []
    group_units: list[str] = []
    has_corpus = "corpus_id" in frame.columns
    for row in frame.itertuples(index=False):
        dataset = str(row.dataset)
        corpus = _valid_identifier(getattr(row, "corpus_id", None)) if has_corpus else None
        # A dataset name is not globally unique once human and mouse (or two
        # independent copies of one study) are combined.  Keep the historical
        # key shape for legacy single-artifact callers, while adding an
        # explicit corpus namespace for composite manifests.
        dataset_key = f"{corpus}::{dataset}" if corpus is not None else dataset
        donor = _valid_identifier(getattr(row, "donor_id", None))
        matched = _valid_identifier(getattr(row, "matched_sample_id", None))
        section = _valid_identifier(getattr(row, "section_id", None))
        assay = str(row.assay_type)
        if donor is not None:
            unit = "donor"
            if donor_scope == "global":
                # Subject IDs are identifiers, not natural language.  A
                # case-insensitive key closes the common ``jhu95``/``JHU95``
                # collision without touching the retained source metadata.
                value = donor.casefold()
                group_key = f"global::{unit}::{value}"
            else:
                value = donor
                group_key = f"{dataset_key}::{unit}::{value}"
        elif matched is not None:
            unit, value = "matched_sample", matched
            group_key = f"{dataset_key}::{unit}::{value}"
        elif assay == "spatialRNA" and section is not None:
            unit, value = "section", section
            group_key = f"{dataset_key}::{unit}::{value}"
        else:
            unit, value = "sample", str(row.sample_key)
            group_key = f"{dataset_key}::{unit}::{value}"
        group_keys.append(group_key)
        group_units.append(unit)
    frame["split_group"] = group_keys
    frame["split_unit"] = group_units
    # Preserve an eligibility decision made by a corpus adapter (for example,
    # a labels column declared in the indexed cell schema).  Older callers
    # have no such column and continue to derive it from the sample metadata.
    metadata_annotation = frame.get(
        "with_cell_annotation", pd.Series(False, index=frame.index)
    ).map(_yes)
    if "annotation_eligible" in frame:
        frame["annotation_eligible"] = frame["annotation_eligible"].map(_yes) | metadata_annotation
    else:
        frame["annotation_eligible"] = metadata_annotation
    statuses = frame.get(
        "time_status", pd.Series(pd.NA, index=frame.index)
    ).map(lambda value: (_text(value) or "").lower())
    times = pd.to_numeric(
        frame.get("time_point_days", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    frame["development_eligible"] = (
        statuses.isin(DEFAULT_TIME_STATUSES) & times.notna() & np.isfinite(times)
    )
    frame["time_point_days"] = times
    frame["spatial_eligible"] = frame["assay_type"].eq("spatialRNA")
    frame["integration_eligible"] = True
    frame["time_stratum"] = frame.get(
        "life_stage", pd.Series("unknown", index=frame.index)
    ).map(lambda value: _text(value) or "unknown")
    frame["legacy_split"] = frame.get(
        "split", pd.Series(pd.NA, index=frame.index)
    )
    return frame


def _group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    assays = sorted(set(frame["assay_type"].astype(str)))
    groups: list[dict[str, Any]] = []
    for group_key, part in frame.groupby("split_group", sort=True, dropna=False):
        row: dict[str, Any] = {
            "split_group": str(group_key),
            "split_unit": str(part["split_unit"].iloc[0]),
            "dataset": str(part["dataset"].iloc[0]),
            "n_samples": int(len(part)),
            "selected_cells": int(part["selected_cells"].sum()),
            "annotation_cells": int(
                part.loc[part["annotation_eligible"], "selected_cells"].sum()
            ),
            "development_cells": int(
                part.loc[part["development_eligible"], "selected_cells"].sum()
            ),
            "spatial_cells": int(
                part.loc[part["spatial_eligible"], "selected_cells"].sum()
            ),
            "annotation_samples": int(part["annotation_eligible"].sum()),
            "development_samples": int(part["development_eligible"].sum()),
            "spatial_samples": int(part["spatial_eligible"].sum()),
            "time_strata": "+".join(sorted(set(part["time_stratum"].astype(str)))),
        }
        if "corpus_id" in part.columns:
            row["corpus_id"] = str(part["corpus_id"].iloc[0])
        for assay in assays:
            assay_part = part[part["assay_type"] == assay]
            row[f"cells_{assay}"] = int(assay_part["selected_cells"].sum())
            row[f"groups_{assay}"] = 1 if len(assay_part) else 0
            row[f"annotation_cells_{assay}"] = int(
                assay_part.loc[assay_part["annotation_eligible"], "selected_cells"].sum()
            )
            row[f"development_cells_{assay}"] = int(
                assay_part.loc[assay_part["development_eligible"], "selected_cells"].sum()
            )
            row[f"spatial_cells_{assay}"] = int(
                assay_part.loc[assay_part["spatial_eligible"], "selected_cells"].sum()
            )
        groups.append(row)
    if not groups:
        raise ValueError("no split groups available")
    return pd.DataFrame(groups)


def _assignment_dimensions(groups: pd.DataFrame, assays: Sequence[str]) -> list[tuple[str, str]]:
    dimensions: list[tuple[str, str]] = []
    for assay in assays:
        dimensions.append((f"cells_{assay}", "modality_cells"))
    # These auxiliary dimensions prevent all annotated or all developmental
    # samples from landing in one partition while keeping modality cell count
    # as the dominant objective.
    for name in ("annotation_cells", "development_cells", "spatial_cells"):
        dimensions.append((name, "task_cells"))
    for assay in assays:
        for name in ("annotation_cells", "development_cells", "spatial_cells"):
            dimensions.append((f"{name}_{assay}", "task_cells"))
    return [(column, kind) for column, kind in dimensions if column in groups]


def _objective(
    actual: np.ndarray,
    targets: np.ndarray,
    scales: np.ndarray,
    kinds: Sequence[str],
) -> float:
    delta = (actual - targets) / scales
    weights = np.asarray(
        [1.0 if kind == "modality_cells" else 0.18 for kind in kinds], dtype=np.float64
    )
    return float(np.sum(weights * delta * delta))


def _repair_coverage(
    groups: pd.DataFrame,
    assignment: dict[str, str],
    assays: Sequence[str],
    seed: str,
) -> None:
    """Move the least disruptive group when a modality is absent in a split."""
    required = [
        (f"cells_{assay}", assay)
        for assay in assays
        if f"cells_{assay}" in groups
        and int((groups[f"cells_{assay}"] > 0).sum()) >= len(SPLITS)
    ]
    required += [
        ("annotation_cells", "annotation")
        for _ in [0]
        if "annotation_cells" in groups
        and int((groups["annotation_cells"] > 0).sum()) >= len(SPLITS)
    ]
    required += [
        ("development_cells", "development")
        for _ in [0]
        if "development_cells" in groups
        and int((groups["development_cells"] > 0).sum()) >= len(SPLITS)
    ]
    required += [
        ("spatial_cells", "spatial")
        for _ in [0]
        if "spatial_cells" in groups
        and int((groups["spatial_cells"] > 0).sum()) >= len(SPLITS)
    ]
    for column, _label in required:
        present = {
            split
            for split in SPLITS
            if any(
                assignment[str(key)] == split and float(value) > 0
                for key, value in zip(groups["split_group"], groups[column])
            )
        }
        missing = [split for split in SPLITS if split not in present]
        for target_split in missing:
            candidates = groups[groups[column] > 0].copy()
            candidates["rank"] = candidates["split_group"].map(
                lambda key: _stable_rank(seed, str(key))
            )
            candidates = candidates[
                candidates["split_group"].map(assignment.__getitem__) != target_split
            ]
            # Prefer a small group and only move from a source split retaining
            # another group for this dimension.
            valid: list[tuple[float, bytes, str]] = []
            for row in candidates.itertuples(index=False):
                key = str(row.split_group)
                source = assignment[key]
                source_values = groups[
                    (groups[column] > 0)
                    & groups["split_group"].map(assignment.__getitem__).eq(source)
                ]
                if len(source_values) <= 1:
                    continue
                valid.append((float(getattr(row, column)), bytes(row.rank), key))
            if not valid:
                continue
            valid.sort(key=lambda item: (item[0], item[1], item[2]))
            assignment[valid[0][2]] = target_split


def assign_group_splits(
    groups: pd.DataFrame,
    *,
    seed: str = "brainwm-global-20260904",
    fractions: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Assign each group once using deterministic weighted 60/20/20 balancing."""
    fractions = _validate_fractions(fractions or DEFAULT_FRACTIONS)
    _require_columns(groups, ("split_group", "selected_cells", "assay_type"), "groups")
    if groups["split_group"].duplicated().any():
        raise ValueError("groups must contain one row per split_group")
    assays = sorted(set(groups["assay_type"].astype(str)))
    # ``_group_frame`` has per-assay columns; synthetic callers may provide one
    # row per assay and still use this function through the normal build path.
    if not any(f"cells_{assay}" in groups for assay in assays):
        groups = groups.copy()
        for assay in assays:
            groups[f"cells_{assay}"] = np.where(
                groups["assay_type"].astype(str).eq(assay),
                pd.to_numeric(groups["selected_cells"], errors="raise"),
                0,
            )
        # Compact public callers need not provide optional task dimensions.
        # Zero-valued columns keep the objective and coverage code uniform.
        for name in ("annotation_cells", "development_cells", "spatial_cells"):
            if name not in groups:
                groups[name] = 0
    # Derive modalities from the aggregate columns too; a group may contain
    # more than one assay and its representative ``assay_type`` is only a
    # convenience field for older callers.
    aggregate_assays = {
        column.removeprefix("cells_")
        for column in groups.columns
        if column.startswith("cells_") and column != "cells_"
    }
    assays = sorted(set(assays) | aggregate_assays)
    dimensions = _assignment_dimensions(groups, assays)
    columns = [item[0] for item in dimensions]
    kinds = [item[1] for item in dimensions]
    values = groups[columns].to_numpy(dtype=np.float64)
    totals = values.sum(axis=0)
    targets = np.asarray(
        [[totals[index] * fractions[split] for index in range(len(columns))] for split in SPLITS],
        dtype=np.float64,
    )
    # A modality with a tiny total still contributes; the floor only prevents
    # division by zero for optional task dimensions.
    scales = np.maximum(totals, 1.0)

    order = sorted(
        range(len(groups)),
        key=lambda index: (-float(groups.iloc[index]["selected_cells"]),
                           _stable_rank(seed, str(groups.iloc[index]["split_group"])),
                           str(groups.iloc[index]["split_group"])),
    )
    actual = np.zeros((len(SPLITS), len(columns)), dtype=np.float64)
    assignment: dict[str, str] = {}
    for index in order:
        key = str(groups.iloc[index]["split_group"])
        vector = values[index]
        choices: list[tuple[float, bytes, str]] = []
        for split_index, split in enumerate(SPLITS):
            candidate = actual.copy()
            candidate[split_index] += vector
            score = _objective(candidate, targets, scales, kinds)
            # Stable tie-breaking preserves seeded randomization without a
            # process-global RNG state.
            choices.append((score, _stable_rank(seed + "|choice", key + "|" + split), split))
        choices.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = choices[0][2]
        assignment[key] = chosen
        actual[SPLITS.index(chosen)] += vector

    _repair_coverage(groups, assignment, assays, seed)

    # A bounded local search repairs obvious one-group moves after coverage
    # repair.  Statistics are kept in arrays; this avoids repeatedly scanning
    # the full pandas frame (which made large real-world manifests needlessly
    # expensive).
    positive = values > 0
    required_coverage = positive.sum(axis=0) >= len(SPLITS)
    for _ in range(min(2, len(groups))):
        current = np.zeros_like(actual)
        split_indices = np.asarray(
            [SPLITS.index(assignment[str(key)]) for key in groups["split_group"]],
            dtype=np.int8,
        )
        coverage_counts = np.zeros((len(SPLITS), len(columns)), dtype=np.int32)
        for split_index in range(len(SPLITS)):
            current[split_index] = values[split_indices == split_index].sum(axis=0)
            coverage_counts[split_index] = positive[split_indices == split_index].sum(axis=0)
        current_score = _objective(current, targets, scales, kinds)
        improved = False
        for index, key_value in enumerate(groups["split_group"].astype(str)):
            source_index = int(split_indices[index])
            for target_index, target in enumerate(SPLITS):
                if target_index == source_index:
                    continue
                candidate = current.copy()
                candidate[source_index] -= values[index]
                candidate[target_index] += values[index]
                score = _objective(candidate, targets, scales, kinds)
                if score + 1e-12 >= current_score:
                    continue
                # Keep every dimension that has enough positive groups covered
                # in all three partitions.  Counts make this O(dimensions),
                # rather than scanning all groups for every candidate move.
                if positive[index].any():
                    candidate_coverage = coverage_counts.copy()
                    candidate_coverage[source_index] -= positive[index].astype(np.int32)
                    candidate_coverage[target_index] += positive[index].astype(np.int32)
                    valid = bool(np.all(candidate_coverage[:, required_coverage] > 0))
                else:
                    valid = True
                if not valid:
                    continue
                assignment[key_value] = target
                current = candidate
                current_score = score
                split_indices[index] = target_index
                coverage_counts = candidate_coverage if positive[index].any() else coverage_counts
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return assignment


def _source_lookup(artifact: Path) -> pd.DataFrame:
    path = artifact / "sources.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    source = pd.read_parquet(path)
    _require_columns(source, ("source_id", "original_h5ad_path", "local_path"), "sources")
    return source


def _attach_source_ids(samples: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    frame = samples.copy()
    mapping = {
        str(path): int(source_id)
        for path, source_id in zip(source["original_h5ad_path"], source["source_id"])
    }
    frame["source_id"] = frame["h5ad_path"].map(mapping) if "h5ad_path" in frame else pd.NA
    # Some externally assembled manifests retain only the local basename.  A
    # basename fallback is safe when unique and avoids opening H5AD files.
    if frame["source_id"].isna().any() and "h5ad_path" in frame:
        by_name: dict[str, list[int]] = defaultdict(list)
        for path, source_id in zip(source["original_h5ad_path"], source["source_id"]):
            by_name[Path(str(path)).name].append(int(source_id))
        for index, value in frame.loc[frame["source_id"].isna(), "h5ad_path"].items():
            candidates = by_name.get(Path(str(value)).name, [])
            if len(candidates) == 1:
                frame.at[index, "source_id"] = candidates[0]
    frame["source_id"] = pd.to_numeric(frame["source_id"], errors="coerce").astype("Int64")
    return frame


def _validate_identity_boundaries(
    frame: pd.DataFrame,
    *,
    donor_scope: str = "dataset",
) -> None:
    """Fail closed if an identity key would occur in multiple partitions."""
    donor_scope = _validate_donor_scope(donor_scope)
    for column, subset in (
        ("donor_id", frame),
        ("matched_sample_id", frame),
        (
            "section_id",
            frame[frame["assay_type"].eq("spatialRNA")]
            if "assay_type" in frame else frame.iloc[0:0],
        ),
    ):
        if column not in subset:
            continue
        valid = subset[column].map(lambda value: not _nullish(value))
        if not valid.any():
            continue
        # In the strict human benchmark, donor IDs are the subject boundary
        # and must be disjoint even when the same person contributes to
        # several datasets/brain regions.  Composite corpora keep donor
        # identities corpus-qualified to avoid conflating different species.
        audit_frame = subset.loc[valid].copy()
        if donor_scope == "global" and column in {"donor_id", "matched_sample_id"}:
            # Match the case-insensitive normalization used by
            # ``derive_split_groups`` so validation cannot miss aliases such
            # as ``Subject-7`` and ``subject-7``.
            audit_frame["_identity_key"] = audit_frame[column].map(
                lambda value: (_text(value) or "").casefold()
            )
            group_columns = ["_identity_key"]
        else:
            group_columns = ["dataset", column]
            if "corpus_id" in audit_frame.columns:
                group_columns.insert(0, "corpus_id")
        grouped = audit_frame.groupby(group_columns)["split"].nunique()
        if (grouped > 1).any():
            examples = grouped[grouped > 1].head(3).index.tolist()
            raise ValueError(f"identity key {column} crosses global splits: {examples}")


def _task_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "integration":
        eligible = frame["integration_eligible"]
    elif task == "annotation":
        eligible = frame["annotation_eligible"]
    elif task == "development":
        eligible = frame["development_eligible"]
    elif task == "spatial":
        eligible = frame["spatial_eligible"]
    else:
        raise ValueError(f"unknown task: {task}")
    columns = [
        "sample_key", "local_sample_key", "global_sample_key", "split",
        "split_group", "split_unit", "corpus_id", "source_id",
        "local_source_id", "global_source_id", "global_row_id",
        "dataset", "assay_type", "selected_cells", "time_point_days",
        "time_stratum", "annotation_eligible", "development_eligible",
        "spatial_eligible",
    ]
    columns = [column for column in columns if column in frame.columns]
    result = frame.loc[eligible, columns].copy()
    result.insert(0, "task", task)
    if task == "annotation":
        result["label_access"] = result["split"].map(
            {
                "train": "train_labels_allowed",
                "validation": "validation_labels_for_selection",
                "test": "sealed_until_final_score",
            }
        )
    else:
        result["label_access"] = "not_applicable"
    return result.reset_index(drop=True)


def _empty_perturbation_frame() -> pd.DataFrame:
    """Return an explicit, schema-bearing placeholder for the blocked task.

    Keeping an empty table in the artifact makes the four-task benchmark
    contract discoverable without pretending that healthy observational data
    contain intervention labels or counterfactual outcomes.
    """
    return pd.DataFrame(
        {
            "task": pd.Series([], dtype="string"),
            "sample_key": pd.Series([], dtype="string"),
            "split": pd.Series([], dtype="string"),
            "split_group": pd.Series([], dtype="string"),
            "split_unit": pd.Series([], dtype="string"),
            "source_id": pd.Series([], dtype="Int64"),
            "dataset": pd.Series([], dtype="string"),
            "assay_type": pd.Series([], dtype="string"),
            "selected_cells": pd.Series([], dtype="int64"),
            "status": pd.Series([], dtype="string"),
            "reason": pd.Series([], dtype="string"),
        }
    )


def _source_cell_audit(
    artifact: Path,
    selected_samples: pd.DataFrame,
    *,
    batch_size: int = 1_000_000,
) -> dict[str, Any]:
    """Check the lightweight split against the source cell index in one pass.

    Only three metadata columns are read.  No expression matrix or label
    column is opened, and the bounded dictionaries contain at most one entry
    per sample/source rather than one entry per cell.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    path = artifact / "cells.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = {
        str(key): int(value)
        for key, value in zip(
            selected_samples["sample_key"], selected_samples["selected_cells"]
        )
    }
    observed: defaultdict[str, int] = defaultdict(int)
    source_rows: defaultdict[int, int] = defaultdict(int)
    unknown_sample_rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["source_id", "row_index", "sample_key"],
        batch_size=int(batch_size),
    ):
        frame = batch.to_pandas()
        keys = frame["sample_key"].astype(str)
        counts = keys.value_counts()
        for key, count in counts.items():
            key = str(key)
            if key not in expected:
                unknown_sample_rows += int(count)
            else:
                observed[key] += int(count)
        for source_id, count in frame["source_id"].value_counts().items():
            source_rows[int(source_id)] += int(count)
    mismatches = {
        key: {"expected": expected[key], "observed": int(observed.get(key, 0))}
        for key in expected
        if int(observed.get(key, 0)) != expected[key]
    }
    return {
        "source_rows": int(sum(source_rows.values())),
        "selected_rows": int(sum(observed.values())),
        "unknown_sample_rows": int(unknown_sample_rows),
        "sample_count_checked": int(len(expected)),
        "sample_count_mismatched": int(len(mismatches)),
        "mismatches": mismatches,
        "source_count": int(len(source_rows)),
        "status": "passed"
        if not unknown_sample_rows and not mismatches
        else "failed",
    }


def _build_development_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Build compact adjacent-time population candidates, not cell pairs."""
    eligible = frame[frame["development_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=[
            "split", "dataset", "assay_type", "time_stratum", "source_time_days",
            "target_time_days", "delta_time_days", "source_sample_keys",
            "target_sample_keys", "source_samples", "target_samples",
            "pair_scope",
        ])
    rows: list[dict[str, Any]] = []
    by = ["split", "dataset", "assay_type", "time_stratum"]
    if "corpus_id" in eligible.columns:
        by.insert(1, "corpus_id")
    for key, part in eligible.groupby(by, sort=True, dropna=False):
        times = sorted(float(value) for value in part["time_point_days"].dropna().unique())
        for source_time, target_time in zip(times[:-1], times[1:]):
            source = part[np.isclose(part["time_point_days"], source_time)]
            target = part[np.isclose(part["time_point_days"], target_time)]
            if source.empty or target.empty:
                continue
            rows.append({
                "split": str(key[0]),
                "dataset": str(key[2] if "corpus_id" in eligible.columns else key[1]),
                "assay_type": str(key[3] if "corpus_id" in eligible.columns else key[2]),
                "time_stratum": str(key[4] if "corpus_id" in eligible.columns else key[3]),
                "source_time_days": source_time,
                "target_time_days": target_time,
                "delta_time_days": target_time - source_time,
                "source_sample_keys": json.dumps(sorted(source["sample_key"].astype(str)), ensure_ascii=False),
                "target_sample_keys": json.dumps(sorted(target["sample_key"].astype(str)), ensure_ascii=False),
                "source_samples": int(len(source)),
                "target_samples": int(len(target)),
                "pair_scope": "population_level_same_split; cell_level_anatomy_intersection_required",
            })
            if "corpus_id" in eligible.columns:
                rows[-1]["corpus_id"] = str(key[1])
    return pd.DataFrame(rows)


def _validate_sealed_boundary(
    root: Path,
    manifest: Mapping[str, Any],
    samples: pd.DataFrame,
    groups: pd.DataFrame,
) -> None:
    """Validate the compact test boundary without touching source cells.

    The sealed tables are deliberately small.  Comparing their identities to
    the primary manifests prevents a stale or hand-edited test file from being
    mistaken for the global test set while keeping validation metadata-only.
    """
    files = manifest.get("files", {})
    sample_path = root / str(files.get("sealed_test_samples", ""))
    group_path = root / str(files.get("sealed_test_groups", ""))
    policy_path = root / str(files.get("access_policy", ""))
    if not sample_path.is_file() or not group_path.is_file() or not policy_path.is_file():
        raise ValueError("sealed test boundary files are incomplete")
    sealed_samples = pd.read_parquet(sample_path)
    sealed_groups = pd.read_parquet(group_path)
    _require_columns(sealed_samples, ("sample_key",), "sealed test samples")
    _require_columns(sealed_groups, ("split_group", "split"), "sealed test groups")
    expected_samples = set(
        samples.loc[samples["split"].astype(str).eq("test"), "sample_key"].astype(str)
    )
    actual_samples = set(sealed_samples["sample_key"].astype(str))
    if actual_samples != expected_samples or len(sealed_samples) != len(actual_samples):
        raise ValueError("sealed test samples do not match the global test split")
    expected_groups = set(
        groups.loc[groups["split"].astype(str).eq("test"), "split_group"].astype(str)
    )
    actual_groups = set(sealed_groups["split_group"].astype(str))
    if actual_groups != expected_groups or len(sealed_groups) != len(actual_groups):
        raise ValueError("sealed test groups do not match the global test split")
    if set(sealed_groups["split"].astype(str)) != {"test"}:
        raise ValueError("sealed test groups contain a non-test split")
    forbidden_fragments = {
        "expression", "raw_counts", "normalized_expression", "gene_values",
        "target_label", "cell_type_label", "perturbation_outcome",
    }
    leaked = {
        column
        for column in set(sealed_samples.columns) | set(sealed_groups.columns)
        if any(fragment in str(column).lower() for fragment in forbidden_fragments)
    }
    if leaked:
        raise ValueError(f"sealed test metadata contains target-like columns: {sorted(leaked)}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not policy.get("sealed") or policy.get("split") != "test":
        raise ValueError("access policy does not seal the test split")
    if policy.get("same_split_for_all_tasks") is not True:
        raise ValueError("access policy must require one test split for all tasks")
    if not policy.get("independent_unit"):
        raise ValueError("access policy is missing independent unit")


def _validate_manifest_counts(
    manifest: Mapping[str, Any],
    samples: pd.DataFrame,
    groups: pd.DataFrame,
) -> None:
    """Check declared sample/group/cell totals against the compact tables."""
    counts = manifest.get("counts", {})
    expected = {
        "selected_samples": len(samples),
        "selected_groups": len(groups),
        "selected_cells": int(pd.to_numeric(samples["selected_cells"], errors="raise").sum()),
    }
    for key, value in expected.items():
        if key in counts and int(counts[key]) != int(value):
            raise ValueError(f"manifest count drifted for {key}")
    declared_by_split = counts.get("by_split", {})
    for split in SPLITS:
        declared = declared_by_split.get(split)
        if not isinstance(declared, Mapping):
            continue
        actual = {
            "samples": int((samples["split"].astype(str) == split).sum()),
            "groups": int((groups["split"].astype(str) == split).sum()),
            "cells": int(
                samples.loc[samples["split"].astype(str) == split, "selected_cells"].sum()
            ),
        }
        for key, value in actual.items():
            if key in declared and int(declared[key]) != value:
                raise ValueError(f"manifest split count drifted for {split}.{key}")


# ---------------------------------------------------------------------------
# Multi-corpus metadata normalization
# ---------------------------------------------------------------------------

_SAMPLE_KEY_ALIASES = (
    "sample_key", "sample_id", "library_label", "library_id", "sample",
    "section_label", "section_id",
)
_ASSAY_ALIASES = ("assay_type", "assay", "modality")
_DATASET_ALIASES = ("dataset", "dataset_id", "study", "collection")
_SOURCE_ID_ALIASES = ("source_id", "local_source_id", "source", "source_index")
_SOURCE_PATH_ALIASES = (
    "h5ad_path", "original_h5ad_path", "local_path", "source_path", "path",
)


def _first_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    return next((column for column in aliases if column in frame.columns), None)


def _series_from_aliases(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    default: Any = None,
) -> pd.Series:
    column = _first_column(frame, aliases)
    if column is None:
        return pd.Series(default, index=frame.index)
    return frame[column].copy()


def _scalar_text(value: Any) -> str | None:
    """Coerce scalar/list-like metadata to one deterministic text value."""
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        values = [_text(item) for item in value]
        values = sorted({item for item in values if item is not None})
        if len(values) == 1:
            return values[0]
        return None
    return _text(value)


def _assay_value(value: Any) -> str:
    """Normalize scalar and array assay aliases without guessing mixtures."""
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        values = [_canonical_assay(item) for item in value if not _nullish(item)]
        values = sorted(set(values))
        return values[0] if len(values) == 1 else "unknown"
    return _canonical_assay(value)


def _parse_identifier_list(value: Any) -> list[int]:
    """Parse source-id list fields such as ``source_ids_json`` safely."""
    if _nullish(value):
        return []
    candidate: Any = value
    if isinstance(value, (bytes, np.bytes_)):
        candidate = value.decode("utf-8", errors="replace")
    if isinstance(candidate, str):
        text = candidate.strip()
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            candidate = [part.strip() for part in text.split(",") if part.strip()]
    if not isinstance(candidate, (list, tuple, set, np.ndarray, pd.Series)):
        candidate = [candidate]
    result: list[int] = []
    for item in candidate:
        if _nullish(item):
            continue
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source identifier is not numeric: {item!r}") from exc
        if not math.isfinite(number) or number != math.floor(number):
            raise ValueError(f"source identifier is not an integer: {item!r}")
        result.append(int(number))
    return sorted(set(result))


def _source_ids_for_sample(
    row: pd.Series,
    *,
    source_ids: set[int],
    path_to_ids: Mapping[str, Sequence[int]],
    dataset_to_ids: Mapping[str, Sequence[int]] | None = None,
) -> list[int]:
    values: list[int] = []
    for column in ("source_ids_json", "source_ids", "source_id", "local_source_id"):
        if column in row.index:
            values.extend(_parse_identifier_list(row[column]))
            if values:
                break
    if not values:
        dataset_to_ids = dataset_to_ids or {}
        for column in _DATASET_ALIASES:
            if column not in row.index or _nullish(row[column]):
                continue
            dataset_value = str(row[column])
            candidates = dataset_to_ids.get(dataset_value, ())
            if len(candidates) == 1:
                values = [int(candidates[0])]
                break
    if not values:
        for column in _SOURCE_PATH_ALIASES:
            if column not in row.index or _nullish(row[column]):
                continue
            path = str(row[column])
            candidates = path_to_ids.get(path, ())
            if not candidates:
                candidates = path_to_ids.get(Path(path).name, ())
            if len(candidates) == 1:
                values = [int(candidates[0])]
                break
    if not values and len(source_ids) == 1:
        values = [next(iter(source_ids))]
    unknown = sorted(set(values) - source_ids)
    if unknown:
        raise ValueError(f"sample references unknown source IDs: {unknown[:5]}")
    return sorted(set(values))


def _count_source_cells_by_sample(artifact: Path) -> dict[str, int]:
    """Count source rows by local sample key using bounded Parquet batches."""
    path = artifact / "cells.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    sample_column = _first_column(
        pd.DataFrame(columns=parquet.schema.names), _SAMPLE_KEY_ALIASES
    )
    if sample_column is None:
        raise ValueError(f"{path} has no sample-key column")
    counts: defaultdict[str, int] = defaultdict(int)
    for batch in parquet.iter_batches(columns=[sample_column], batch_size=1_000_000):
        values = batch.to_pandas()[sample_column].dropna().astype(str).value_counts()
        for key, count in values.items():
            counts[str(key)] += int(count)
    return dict(counts)


def _map_local_sample_key(
    value: Any,
    mapping: Mapping[str, str],
    *,
    prefix: str,
) -> str | None:
    """Map either a local or already-prefixed cell key to its global key."""
    text = _text(value)
    if text is None:
        return None
    if text in mapping:
        return str(mapping[text])
    marker = prefix + "::"
    if text.startswith(marker):
        local = text[len(marker):]
        if local in mapping:
            return str(mapping[local])
    return None


def _normalise_corpus_tables(
    spec: BenchmarkCorpusSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Read and normalize one corpus's compact metadata tables.

    The returned sample frame uses global ``sample_key`` values.  The mapping
    in the third return value maps local sample keys to those global keys and
    is used by the lazy cell-index reader.
    """
    artifact = Path(spec.artifact_dir).resolve()
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    samples_path = artifact / "samples.parquet"
    sources_path = artifact / "sources.parquet"
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)
    if not sources_path.is_file():
        raise FileNotFoundError(sources_path)
    raw_sources = pd.read_parquet(sources_path)
    raw_samples = pd.read_parquet(samples_path)
    if raw_sources.empty:
        raise ValueError(f"{spec.corpus_id}.sources.parquet is empty")
    if raw_samples.empty:
        raise ValueError(f"{spec.corpus_id}.samples.parquet is empty")

    source_id_column = _first_column(raw_sources, _SOURCE_ID_ALIASES)
    if source_id_column is None:
        raise ValueError(f"{spec.corpus_id}.sources is missing source_id")
    local_source = pd.to_numeric(raw_sources[source_id_column], errors="raise")
    if local_source.isna().any() or (local_source % 1 != 0).any() or (local_source < 0).any():
        raise ValueError(f"{spec.corpus_id}.sources source IDs must be nonnegative integers")
    local_source = local_source.astype("int64")
    if local_source.duplicated().any():
        raise ValueError(f"{spec.corpus_id}.sources source IDs are not unique")
    source = raw_sources.copy()
    source["source_id"] = local_source
    source["local_source_id"] = local_source
    source["corpus_id"] = spec.corpus_id
    source["global_source_id"] = source["local_source_id"].map(
        lambda value: qualify_source_id(spec.corpus_id, value)
    )
    source["source_key"] = source["global_source_id"]
    source["artifact_dir"] = str(artifact)
    species_value = _canonical_species(spec.species)
    if species_value is None:
        species_column = _first_column(source, ("species", "species_name", "organism"))
        species_value = (
            _canonical_species(_scalar_text(source[species_column].iloc[0]))
            if species_column else None
        )
    species_value = species_value or "unknown"
    canonical_species_id = spec.species_id
    if canonical_species_id is None:
        species_id_map = {"human": 0, "mouse": 1, "macaque": 2, "marmoset": 3}
        canonical_species_id = species_id_map.get(species_value.lower())
    source["species"] = species_value
    if "species_id" not in source:
        source["species_id"] = canonical_species_id
    elif spec.species_id is not None:
        source["species_id"] = spec.species_id
    elif canonical_species_id is not None:
        source["species_id"] = pd.to_numeric(
            source["species_id"], errors="coerce"
        ).fillna(canonical_species_id).astype("int16")
    assay_column = _first_column(source, _ASSAY_ALIASES)
    if assay_column is None and "assay_types" in source:
        source["assay_type"] = source["assay_types"].map(_assay_value)
    elif assay_column is not None:
        source["assay_type"] = source[assay_column].map(_assay_value)
    else:
        source["assay_type"] = "unknown"
    dataset_column = _first_column(source, _DATASET_ALIASES)
    source["dataset"] = (
        source[dataset_column].map(lambda value: _text(value) or spec.corpus_id)
        if dataset_column else spec.corpus_id
    )
    if "has_spatial" in source:
        source["has_spatial"] = source["has_spatial"].map(_yes)
    else:
        source["has_spatial"] = source["assay_type"].eq("spatialRNA")
    # Some indexed spatial corpora expose only ``has_spatial`` and platform
    # metadata in ``sources.parquet``.  This is an explicit source property,
    # so it is safe to normalize an otherwise unknown assay to spatialRNA;
    # do not infer molecular assay types from a platform name alone.
    source.loc[
        source["assay_type"].eq("unknown") & source["has_spatial"],
        "assay_type",
    ] = "spatialRNA"

    source_ids = set(int(value) for value in local_source.tolist())
    path_to_ids: defaultdict[str, list[int]] = defaultdict(list)
    for row in source.itertuples(index=False):
        local_id = int(row.local_source_id)
        for column in _SOURCE_PATH_ALIASES:
            if not hasattr(row, column):
                continue
            value = getattr(row, column)
            if _nullish(value):
                continue
            text = str(value)
            path_to_ids[text].append(local_id)
            path_to_ids[Path(text).name].append(local_id)
    path_to_ids = defaultdict(
        list,
        {key: sorted(set(values)) for key, values in path_to_ids.items()},
    )
    dataset_to_ids: defaultdict[str, list[int]] = defaultdict(list)
    for row in source.itertuples(index=False):
        local_id = int(row.local_source_id)
        for column in _DATASET_ALIASES + ("datasets", "datasets_json"):
            if not hasattr(row, column):
                continue
            value = getattr(row, column)
            if _nullish(value):
                continue
            candidates = value
            if isinstance(value, str):
                try:
                    candidates = json.loads(value)
                except json.JSONDecodeError:
                    candidates = [value]
            if not isinstance(candidates, (list, tuple, set, np.ndarray, pd.Series)):
                candidates = [candidates]
            for candidate in candidates:
                text = _text(candidate)
                if text is not None:
                    dataset_to_ids[text].append(local_id)
    dataset_to_ids = defaultdict(
        list,
        {key: sorted(set(values)) for key, values in dataset_to_ids.items()},
    )

    sample = raw_samples.copy()
    explicit_annotation_column = _first_column(
        sample,
        ("with_cell_annotation", "has_cell_annotation", "labels_available"),
    )
    species_column = _first_column(sample, ("species", "species_name", "organism"))
    if spec.species_filter is not None:
        if species_column is None:
            raise ValueError(
                f"{spec.corpus_id}.samples has no species column for species_filter"
            )
        wanted = _canonical_species(spec.species_filter)
        sample = sample[sample[species_column].map(_canonical_species).eq(wanted)].copy()
    if sample.empty:
        raise ValueError(f"{spec.corpus_id} has no samples after species filtering")
    local_key_column = _first_column(sample, _SAMPLE_KEY_ALIASES)
    if local_key_column is None:
        raise ValueError(f"{spec.corpus_id}.samples is missing a sample key alias")
    local_keys = sample[local_key_column].map(_text)
    if local_keys.isna().any() or local_keys.eq("").any():
        raise ValueError(f"{spec.corpus_id}.samples contains null sample keys")
    if local_keys.duplicated().any():
        raise ValueError(f"{spec.corpus_id}.samples sample keys are not unique")
    sample["local_sample_key"] = local_keys.astype(str)
    prefix = _text(spec.sample_key_prefix) or spec.corpus_id
    sample["global_sample_key"] = sample["local_sample_key"].map(
        lambda value: qualify_sample_key(prefix, value)
    )
    sample["sample_key"] = sample["global_sample_key"]
    sample["corpus_id"] = spec.corpus_id
    sample["species"] = species_value
    if "species_id" not in sample or spec.species_id is not None:
        sample["species_id"] = canonical_species_id
    elif canonical_species_id is not None:
        sample["species_id"] = pd.to_numeric(
            sample["species_id"], errors="coerce"
        ).fillna(canonical_species_id).astype("int16")
    assay_column = _first_column(sample, _ASSAY_ALIASES)
    sample["assay_type"] = (
        sample[assay_column].map(_assay_value) if assay_column else "unknown"
    )
    dataset_column = _first_column(sample, _DATASET_ALIASES)
    sample["dataset"] = (
        sample[dataset_column].map(lambda value: _text(value) or spec.corpus_id)
        if dataset_column else spec.corpus_id
    )
    # Keep source-local dataset for provenance while making the split group
    # namespace explicit through ``corpus_id`` in derive_split_groups.
    sample["dataset_local"] = sample["dataset"]
    selected_column = _first_column(
        sample, ("selected_cells", "n_observations", "n_obs", "cell_count", "n_cells")
    )
    if selected_column is None:
        counts = _count_source_cells_by_sample(artifact)
        selected = sample["local_sample_key"].map(counts)
    else:
        selected = pd.to_numeric(sample[selected_column], errors="coerce")
        if selected.isna().any():
            counts = _count_source_cells_by_sample(artifact)
            selected = selected.fillna(sample["local_sample_key"].map(counts))
    if selected.isna().any() or (selected <= 0).any():
        raise ValueError(f"{spec.corpus_id}.samples has invalid selected cell counts")
    sample["selected_cells"] = selected.astype("int64")
    sample["n_observations"] = sample["selected_cells"]

    # Canonical identity aliases used by the grouping policy.
    for target, aliases in (
        ("donor_id", ("donor_id", "donor", "animal_id", "animal")),
        ("matched_sample_id", ("matched_sample_id", "matched_id")),
        ("section_id", ("section_id", "section_label", "section", "spatial_section")),
        ("h5ad_path", _SOURCE_PATH_ALIASES),
        ("time_point_days", ("time_point_days", "time_days", "timepoint_days", "timepoint_raw")),
        ("time_status", ("time_status", "timepoint_status")),
        ("life_stage", ("life_stage", "developmental_stage", "developmental_stage_raw", "stage")),
        ("with_cell_annotation", ("with_cell_annotation", "has_cell_annotation", "labels_available")),
    ):
        if target not in sample:
            sample[target] = _series_from_aliases(sample, aliases)
    sample["source_ids"] = sample.apply(
        lambda row: _source_ids_for_sample(
            row,
            source_ids=source_ids,
            path_to_ids=path_to_ids,
            dataset_to_ids=dataset_to_ids,
        ),
        axis=1,
    )
    sample["source_ids_json"] = sample["source_ids"].map(
        lambda values: json.dumps(values, separators=(",", ":"))
    )
    # Fill a missing sample-level assay from its explicitly linked source.
    # This covers section indexes whose sample table has only platform/source
    # fields (for example mouse MERFISH) without guessing from free text.
    source_assays = source.set_index("local_source_id")["assay_type"].to_dict()
    for index, value in sample.loc[sample["assay_type"].eq("unknown"), "source_ids"].items():
        inferred = sorted(
            {
                str(source_assays[int(source_id)])
                for source_id in value
                if int(source_id) in source_assays
                and str(source_assays[int(source_id)]) != "unknown"
            }
        )
        if len(inferred) == 1:
            sample.at[index, "assay_type"] = inferred[0]
    sample["local_source_id"] = sample["source_ids"].map(
        lambda values: values[0] if len(values) == 1 else pd.NA
    ).astype("Int64")
    sample["source_id"] = sample["local_source_id"]
    sample["global_source_id"] = sample.apply(
        lambda row: (
            qualify_source_id(spec.corpus_id, row.local_source_id)
            if pd.notna(row.local_source_id) else pd.NA
        ),
        axis=1,
    )
    sample["global_source_ids_json"] = sample["source_ids"].map(
        lambda values: json.dumps(
            [qualify_source_id(spec.corpus_id, value) for value in values],
            separators=(",", ":"),
        )
    )
    sample["legacy_split"] = sample["split"] if "split" in sample else pd.NA
    # Do not infer labels from expression.  Existing eligibility flags are
    # honored; otherwise use the explicit metadata availability field.
    if "annotation_eligible" in sample:
        sample["annotation_eligible"] = sample["annotation_eligible"].map(_yes)
    else:
        sample["annotation_eligible"] = sample["with_cell_annotation"].map(_yes)
    # Label columns are part of the indexed cell metadata schema, not label
    # values.  Their presence is enough to expose an annotation lane while
    # preserving the evaluator's per-label coverage checks.
    cells_schema_path = artifact / "cells.parquet"
    if cells_schema_path.is_file():
        cell_columns = set(pq.read_schema(cells_schema_path).names)
        annotation_columns = {
            "cell_label", "cell_type", "original_cell_type", "original_subclass",
            "cl_term", "cl_id", "celltype", "label",
        }
        # A sample-level availability flag is authoritative when present.  A
        # schema-only fallback is used only for artifacts that omit that
        # field entirely (the mouse index tables), otherwise samples with no
        # labels would be over-counted as annotation-eligible.
        if explicit_annotation_column is None and cell_columns.intersection(annotation_columns):
            sample["annotation_eligible"] = True
    sample["integration_eligible"] = True
    sample["spatial_eligible"] = sample["assay_type"].eq("spatialRNA")
    times = pd.to_numeric(sample["time_point_days"], errors="coerce")
    statuses = sample["time_status"].map(lambda value: (_text(value) or "").lower())
    sample["time_point_days"] = times
    sample["development_eligible"] = (
        statuses.isin(DEFAULT_TIME_STATUSES) & times.notna() & np.isfinite(times)
    )
    sample["time_stratum"] = sample["life_stage"].map(
        lambda value: _text(value) or "unknown"
    )
    # A stable row-independent key is useful even when a source contains no
    # explicit cell ID.  It is materialized lazily by GlobalBenchmarkIndex.
    sample["global_sample_index"] = np.arange(len(sample), dtype=np.int64)
    mapping = dict(zip(sample["local_sample_key"].astype(str), sample["sample_key"].astype(str)))
    return source.reset_index(drop=True), sample.reset_index(drop=True), mapping


def _source_cell_audit_multi(
    specs: Sequence[BenchmarkCorpusSpec],
    selected: pd.DataFrame,
    mappings: Mapping[str, Mapping[str, str]],
    *,
    batch_size: int = 1_000_000,
) -> dict[str, Any]:
    """Audit multiple source cell indexes without loading expression values."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    expected = {
        (str(row.corpus_id), str(row.local_sample_key)): int(row.selected_cells)
        for row in selected.itertuples(index=False)
    }
    observed: defaultdict[tuple[str, str], int] = defaultdict(int)
    unknown_rows = 0
    source_rows = 0
    # Keep a set across row-groups.  A source commonly spans many row-groups;
    # summing per-batch ``nunique`` would over-count it.
    source_ids_seen: set[tuple[str, int]] = set()
    for spec in specs:
        path = Path(spec.artifact_dir).resolve() / "cells.parquet"
        if not path.is_file():
            return {
                "status": "failed",
                "reason": f"missing cells.parquet for corpus {spec.corpus_id}",
                "source_rows": source_rows,
            }
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema.names)
        sample_column = next((name for name in _SAMPLE_KEY_ALIASES if name in names), None)
        source_column = next((name for name in _SOURCE_ID_ALIASES if name in names), None)
        if sample_column is None:
            return {
                "status": "failed",
                "reason": f"{path} has no sample-key alias",
                "source_rows": source_rows,
            }
        columns = [sample_column]
        if source_column is not None:
            columns.append(source_column)
        for batch in parquet.iter_batches(columns=columns, batch_size=int(batch_size)):
            frame = batch.to_pandas()
            source_rows += len(frame)
            if source_column is not None:
                values = pd.to_numeric(frame[source_column], errors="raise")
                source_ids_seen.update(
                    (spec.corpus_id, int(value))
                    for value in values.dropna().unique()
                )
            local_keys = frame[sample_column].map(_text)
            for key, count in local_keys.value_counts(dropna=True).items():
                local_key = str(key)
                prefix = _text(spec.sample_key_prefix) or spec.corpus_id
                if local_key not in mappings.get(spec.corpus_id, {}) and local_key.startswith(prefix + "::"):
                    local_key = local_key[len(prefix) + 2 :]
                lookup_key = (spec.corpus_id, local_key)
                if lookup_key not in expected:
                    # A declared species_filter intentionally selects a
                    # subset from a mixed artifact; rows belonging to the
                    # excluded species are not audit failures.  For an
                    # unfiltered corpus, unknown rows remain fail-closed.
                    if spec.species_filter is None:
                        unknown_rows += int(count)
                else:
                    observed[lookup_key] += int(count)
    mismatches = {
        f"{corpus}::{key}": {
            "expected": expected[(corpus, key)],
            "observed": int(observed.get((corpus, key), 0)),
        }
        for corpus, key in expected
        if int(observed.get((corpus, key), 0)) != expected[(corpus, key)]
    }
    return {
        "source_rows": int(source_rows),
        "selected_rows": int(sum(observed.values())),
        "unknown_sample_rows": int(unknown_rows),
        "sample_count_checked": int(len(expected)),
        "sample_count_mismatched": int(len(mismatches)),
        "mismatches": mismatches,
        "source_count": int(len(source_ids_seen)),
        "status": "passed" if not unknown_rows and not mismatches else "failed",
    }


def _multi_task_paths(output: Path, selected: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    task_paths: dict[str, str] = {}
    for task in ("integration", "annotation", "development", "spatial"):
        path = output / f"task_{task}.parquet"
        _task_frame(selected, task).to_parquet(path, index=False, compression="zstd")
        task_paths[task] = path.name
    perturbation = _empty_perturbation_frame()
    perturbation.to_parquet(output / "task_perturbation.parquet", index=False, compression="zstd")
    return task_paths, perturbation


def _build_multi_global_benchmark_split(
    specs: Sequence[BenchmarkCorpusSpec],
    output_dir: str | Path,
    *,
    seed: str,
    fractions: Mapping[str, float],
    overwrite: bool,
    audit_source: bool,
    source_batch_size: int,
) -> dict[str, Any]:
    """Build a shared split over two or more metadata-only corpora."""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output is non-empty: {output}; pass overwrite=True for an explicit rebuild"
        )
    output.mkdir(parents=True, exist_ok=True)
    normalized_sources: list[pd.DataFrame] = []
    normalized_samples: list[pd.DataFrame] = []
    mappings: dict[str, dict[str, str]] = {}
    for spec in specs:
        source, sample, mapping = _normalise_corpus_tables(spec)
        normalized_sources.append(source)
        normalized_samples.append(sample)
        mappings[spec.corpus_id] = mapping
    sources = pd.concat(normalized_sources, ignore_index=True, sort=False)
    selected = pd.concat(normalized_samples, ignore_index=True, sort=False)
    if selected["sample_key"].duplicated().any():
        duplicate = selected.loc[selected["sample_key"].duplicated(), "sample_key"].head(5).tolist()
        raise ValueError(f"global sample key collision: {duplicate}")
    if sources["global_source_id"].duplicated().any():
        raise ValueError("global source IDs are not unique")

    selected = derive_split_groups(selected, donor_scope="corpus")
    groups = _group_frame(selected)
    group_assay = selected.groupby("split_group", sort=True)["assay_type"].first().rename("assay_type")
    groups = groups.merge(group_assay, left_on="split_group", right_index=True, how="left")
    assignment = assign_group_splits(groups, seed=seed, fractions=fractions)
    selected["split"] = selected["split_group"].map(assignment)
    groups["split"] = groups["split_group"].map(assignment)
    groups["assignment_rank"] = groups["split_group"].map(
        lambda key: int.from_bytes(_stable_rank(seed, str(key))[:8], "big")
    )
    # Stable row identity is defined independently of output row ordering.
    selected["global_sample_index"] = np.arange(len(selected), dtype=np.int64)
    _validate_identity_boundaries(selected, donor_scope="corpus")

    selected.to_parquet(output / "samples.parquet", index=False, compression="zstd")
    groups.to_parquet(output / "split_groups.parquet", index=False, compression="zstd")
    source_columns = list(sources.columns)
    sources[source_columns].to_parquet(output / "source_lookup.parquet", index=False, compression="zstd")
    task_paths, _perturbation = _multi_task_paths(output, selected)
    intervals = _build_development_intervals(selected)
    intervals.to_parquet(output / "development_intervals.parquet", index=False, compression="zstd")

    source_audit = (
        _source_cell_audit_multi(
            specs, selected, mappings, batch_size=int(source_batch_size)
        )
        if audit_source
        else {"status": "not_run", "reason": "audit_source_false"}
    )
    species_ids_by_corpus: dict[str, int | None] = {}
    species_by_corpus: dict[str, str] = {}
    for spec in specs:
        species_values = sources.loc[
            sources["corpus_id"].eq(spec.corpus_id), "species"
        ].dropna().astype(str).unique()
        species_by_corpus[spec.corpus_id] = (
            (_canonical_species(spec.species) or "unknown") if spec.species is not None else
            (str(species_values[0]) if len(species_values) else "unknown")
        )
        values = pd.to_numeric(
            sources.loc[sources["corpus_id"].eq(spec.corpus_id), "species_id"],
            errors="coerce",
        ).dropna()
        species_ids_by_corpus[spec.corpus_id] = int(values.iloc[0]) if not values.empty else None
    corpora_manifest = [
        {
            "corpus_id": spec.corpus_id,
            "artifact_dir": str(Path(spec.artifact_dir).resolve()),
            "species": species_by_corpus[spec.corpus_id],
            "species_id": species_ids_by_corpus[spec.corpus_id],
            "reader_kind": _text(spec.reader_kind),
            "sample_key_prefix": _text(spec.sample_key_prefix) or spec.corpus_id,
            "sample_key_column": "sample_key",
            "source_id_column": "source_id",
            "row_identity": ["corpus_id", "local_source_id", "row_index"],
        }
        for spec in specs
    ]
    by_split = {
        split: {
            "samples": int((selected["split"] == split).sum()),
            "cells": int(selected.loc[selected["split"] == split, "selected_cells"].sum()),
            "groups": int((groups["split"] == split).sum()),
        }
        for split in SPLITS
    }
    assays = sorted(set(selected["assay_type"].astype(str)))
    by_split_assay = {
        split: {
            assay: int(selected.loc[
                (selected["split"] == split) & (selected["assay_type"] == assay),
                "selected_cells",
            ].sum())
            for assay in assays
        }
        for split in SPLITS
    }
    coverage: dict[str, dict[str, int]] = {}
    for task in ("integration", "annotation", "development", "spatial"):
        task_frame = _task_frame(selected, task)
        coverage[task] = {
            split: int(task_frame.loc[task_frame["split"] == split, "selected_cells"].sum())
            for split in SPLITS
        }
    cell_manifest = {
        "format": "brainwm_global_cell_manifest_view_v1",
        "source_cells": [str(Path(spec.artifact_dir).resolve() / "cells.parquet") for spec in specs],
        "join_key": ["corpus_id", "local_sample_key"],
        "row_identity": ["corpus_id", "local_source_id", "row_index", "global_row_id"],
        "split_column": "split in samples.parquet",
        "task_columns": {**task_paths, "perturbation": "task_perturbation.parquet"},
        "materialized_rows": 0,
        "expression_copied": False,
        "labels_copied": False,
    }
    (output / "cell_manifest.json").write_text(
        json.dumps(cell_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    test_samples = selected.loc[
        selected["split"].eq("test"),
        [
            "sample_key", "local_sample_key", "global_sample_key", "split_group",
            "split_unit", "corpus_id", "source_id", "local_source_id",
            "global_source_id", "dataset", "assay_type", "selected_cells",
            "annotation_eligible", "development_eligible", "spatial_eligible",
        ],
    ].copy()
    test_samples.to_parquet(output / "sealed_test_samples.parquet", index=False, compression="zstd")
    groups.loc[groups["split"].eq("test")].to_parquet(
        output / "sealed_test_groups.parquet", index=False, compression="zstd"
    )
    access_policy = {
        "format": "brainwm_global_test_access_policy_v1",
        "split": "test",
        "sealed": True,
        "allowed_before_freeze": [
            "corpus_id", "global_source_id", "global_row_id", "sample_key",
            "assay_type", "spatial coordinates when required by the declared task",
        ],
        "blocked_before_freeze": [
            "cell-type labels", "target expression used for model selection",
            "development target expression used for model selection",
            "perturbation outcomes (not present in this corpus)",
        ],
        "release_condition": "open target labels/expression once all representations and predictions are frozen",
        "same_split_for_all_tasks": True,
        "independent_unit": "corpus-qualified split_group",
    }
    (output / "access_policy.json").write_text(
        json.dumps(access_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "format": GLOBAL_SPLIT_FORMAT,
        "builder_version": BUILDER_VERSION,
        "multi_corpus": True,
        "corpora": corpora_manifest,
        # Keep a scalar source_artifact only for old readers that inspect it;
        # new readers use ``corpora``.
        "source_artifact": str(Path(specs[0].artifact_dir).resolve()),
        "source_artifacts": [str(Path(spec.artifact_dir).resolve()) for spec in specs],
        "species": sorted({str(spec.species or "unknown").lower() for spec in specs}),
        "seed": seed,
        "fractions": fractions,
        "donor_scope": "corpus",
        "split_unit_policy": [
            "corpus-qualified dataset donor when available",
            "matched sample when donor is missing",
            "spatial section when donor and matched sample are missing",
            "corpus-qualified sample key fallback",
        ],
        "id_policy": {
            "sample_key": "corpus_id::sample_key (global sample_key; local_sample_key retained)",
            "source_id": "corpus_id::source::local_source_id",
            "row_id": "corpus_id::source::local_source_id::row::row_index",
            "collision_scope": "case-sensitive corpus IDs are canonicalized and unique case-insensitively",
        },
        "schema_aliases": {
            "assay_type": list(_ASSAY_ALIASES),
            "selected_cells": ["selected_cells", "n_observations", "n_obs", "cell_count", "n_cells"],
            "sample_key": list(_SAMPLE_KEY_ALIASES),
            "source_id": list(_SOURCE_ID_ALIASES),
            "gene_index": ["gene_index", "namespace_gene_index"],
        },
        "zero_copy": True,
        "large_file_hashes": False,
        "counts": {
            "source_samples": int(sum(len(frame) for frame in normalized_samples)),
            "selected_samples": int(len(selected)),
            "selected_groups": int(len(groups)),
            "selected_cells": int(selected["selected_cells"].sum()),
            "by_split": by_split,
            "by_split_assay_cells": by_split_assay,
            "task_cells_by_split": coverage,
            "development_intervals": int(len(intervals)),
            "perturbation_rows": 0,
        },
        "files": {
            "samples": "samples.parquet", "split_groups": "split_groups.parquet",
            "source_lookup": "source_lookup.parquet",
            "development_intervals": "development_intervals.parquet",
            "perturbation": "task_perturbation.parquet", "cell_manifest": "cell_manifest.json",
            "sealed_test_samples": "sealed_test_samples.parquet",
            "sealed_test_groups": "sealed_test_groups.parquet",
            "access_policy": "access_policy.json", "tasks": task_paths,
        },
        "task_status": {
            "integration": "available", "annotation": "available_where_labels_exist",
            "development": "available_for_reliable_time_statuses",
            "spatial": "auxiliary_available_for_spatialRNA", "perturbation": "blocked_data",
        },
        "source_cell_audit": source_audit,
        "lazy_cell_lookup": {
            "source_cells_paths": [str(Path(spec.artifact_dir).resolve() / "cells.parquet") for spec in specs],
            "key_columns": ["corpus_id", "local_source_id", "row_index"],
            "row_identity_columns": ["corpus_id", "local_source_id", "row_index"],
            "api": "brainwm_data.global_split.GlobalBenchmarkIndex.iter_cell_index",
            "expression_copied": False,
        },
        "leakage_guards": [
            "one split per corpus-qualified split_group",
            "global sample/source/row IDs are corpus-qualified",
            "spatial sections remain within one donor/group split",
            "development intervals require source and target samples in one split",
            "annotation labels are not copied into task manifests",
            "test labels and target expression remain evaluator-gated",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_data_card(output, manifest)
    _write_readme(output)
    split_receipt = {
        "format": "brainwm_global_split_receipt_v1",
        "split_id": "BRAINWM-MULTI-CORPUS-GLOBAL-6-2-2-V1",
        "seed": seed,
        "corpora": [spec.corpus_id for spec in specs],
        "selected_samples": int(len(selected)),
        "selected_groups": int(len(groups)),
        "selected_cells": int(selected["selected_cells"].sum()),
        "fractions": fractions,
        "counts_by_split": by_split,
        "source_cell_audit_status": source_audit.get("status"),
        "checkpoint_modified": False,
        "expression_copied": False,
        "labels_copied": False,
    }
    (output / "split_receipt.json").write_text(
        json.dumps(split_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_global_benchmark_split(
    artifact_dir: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    output_dir: str | Path | None = None,
    *,
    species: str = "human",
    seed: str = "brainwm-global-20260904",
    fractions: Mapping[str, float] | None = None,
    overwrite: bool = False,
    audit_source: bool = True,
    source_batch_size: int = 1_000_000,
    donor_scope: str = "dataset",
    split_id: str | None = None,
    corpus_specs: Sequence[Any] | Mapping[str, Any] | None = None,
    corpora: Sequence[Any] | Mapping[str, Any] | None = None,
    group_assignment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a versioned sample/group split without copying expression data.

    The historical positional form is unchanged::

        build_global_benchmark_split(artifact_dir, output_dir, species="human")

    For a shared human/mouse (or multi-study) boundary, pass ``corpus_specs``
    or its ``corpora`` alias.  It accepts mappings such as
    ``{"human": human_dir, "mouse": mouse_dir}``, sequences of
    :class:`BenchmarkCorpusSpec`, or dictionaries with ``corpus_id``,
    ``artifact_dir`` and optional species/alias fields.  Multi-corpus output
    retains local identifiers and adds corpus-qualified stable IDs.
    """
    if output_dir is None:
        raise TypeError("output_dir is required")
    donor_scope = _validate_donor_scope(donor_scope)
    specs, explicit_multi = _coerce_benchmark_corpus_specs(
        artifact_dir,
        corpus_specs,
        corpora,
        default_species=species,
    )
    # A plain positional human v1 artifact keeps its historical schema.  If a
    # caller supplies a mouse artifact (or a one-item explicit spec), route it
    # through alias normalization even though there is only one corpus.
    if not explicit_multi and len(specs) == 1:
        spec = specs[0]
        sample_path = Path(spec.artifact_dir).resolve() / "samples.parquet"
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        sample_columns = set(pd.read_parquet(sample_path, columns=None).columns)
        if (
            spec.species not in (None, "human")
            or "assay_type" not in sample_columns
            or "dataset" not in sample_columns
            or "sample_key" not in sample_columns
        ):
            explicit_multi = True
    if explicit_multi:
        if group_assignment is not None:
            raise ValueError('group_assignment currently supports single human corpus only')
        fractions = _validate_fractions(fractions or DEFAULT_FRACTIONS)
        return _build_multi_global_benchmark_split(
            specs,
            output_dir,
            seed=seed,
            fractions=fractions,
            overwrite=overwrite,
            audit_source=audit_source,
            source_batch_size=source_batch_size,
        )
    # The branch below is the original single-artifact implementation.  It is
    # intentionally kept intact so existing v1 manifests and callers retain
    # their historical key and source-id semantics.
    artifact = Path(artifact_dir).resolve()
    output = Path(output_dir).resolve()
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output is non-empty: {output}; pass overwrite=True for an explicit rebuild"
        )
    output.mkdir(parents=True, exist_ok=True)
    fractions = _validate_fractions(fractions or DEFAULT_FRACTIONS)
    if split_id is None:
        split_id = (
            "BRAINWM-HUMAN-GLOBAL-6-2-2-V2"
            if donor_scope == "global"
            else "BRAINWM-HUMAN-GLOBAL-6-2-2-V1"
        )
    source = _source_lookup(artifact)
    source_samples = pd.read_parquet(artifact / "samples.parquet")
    _require_columns(source_samples, ("sample_key", "dataset", "assay_type"), "samples")
    frame = source_samples.copy()
    species_name = str(species).strip().lower()
    if "species" in frame:
        selected = frame[frame["species"].map(_canonical_species).eq(species_name)].copy()
    else:
        selected = frame.copy()
    if selected.empty:
        raise ValueError(f"no samples for species={species_name!r}")
    selected = derive_split_groups(selected, donor_scope=donor_scope)
    selected = _attach_source_ids(selected, source)
    if selected["source_id"].isna().any():
        missing = selected.loc[selected["source_id"].isna(), "sample_key"].head(5).tolist()
        raise ValueError(f"samples could not be mapped to source_id: {missing}")
    source_spatial = source.set_index("source_id")["has_spatial"].to_dict() if "has_spatial" in source else {}
    if source_spatial:
        selected["spatial_eligible"] = selected["spatial_eligible"] & selected["source_id"].map(
            lambda value: _yes(source_spatial.get(int(value), False))
            if not _nullish(value) else False
        )
    groups = _group_frame(selected)
    # assign_group_splits consumes a compact group frame but needs assay names
    # to infer the per-assay columns.  Attach one representative assay column
    # for its public API while preserving all modality aggregates.
    group_assay = (
        selected.groupby("split_group", sort=True)["assay_type"].first().rename("assay_type")
    )
    groups = groups.merge(group_assay, left_on="split_group", right_index=True, how="left")
    if group_assignment is None:
        assignment = assign_group_splits(groups, seed=seed, fractions=fractions)
    else:
        assignment = dict(group_assignment)
        if set(assignment) != set(groups['split_group']):
            raise ValueError('provided assignment must cover exactly the generated split groups')
        if set(assignment.values()) != set(SPLITS):
            raise ValueError('provided assignment must use train, validation and test')
    selected["split"] = selected["split_group"].map(assignment)
    groups["split"] = groups["split_group"].map(assignment)
    groups["assignment_rank"] = groups["split_group"].map(
        lambda key: int.from_bytes(_stable_rank(seed, str(key))[:8], "big")
    )
    selected["global_sample_index"] = np.arange(len(selected), dtype=np.int64)
    _validate_identity_boundaries(selected, donor_scope=donor_scope)

    # Keep the source's old split only as an audit field.  Consumers must use
    # this new ``split`` column and the task tables below.
    sample_columns = list(selected.columns)
    selected[sample_columns].to_parquet(output / "samples.parquet", index=False, compression="zstd")
    groups.to_parquet(output / "split_groups.parquet", index=False, compression="zstd")
    source_columns = [
        column for column in (
            "source_id", "original_h5ad_path", "local_path", "file_name",
            "expression_element", "matrix_encoding", "n_obs", "n_vars", "has_spatial",
        ) if column in source.columns
    ]
    source[source_columns].to_parquet(output / "source_lookup.parquet", index=False, compression="zstd")

    task_paths: dict[str, str] = {}
    for task in ("integration", "annotation", "development", "spatial"):
        path = output / f"task_{task}.parquet"
        _task_frame(selected, task).to_parquet(path, index=False, compression="zstd")
        task_paths[task] = path.name
    intervals = _build_development_intervals(selected)
    intervals.to_parquet(output / "development_intervals.parquet", index=False, compression="zstd")
    perturbation = _empty_perturbation_frame()
    perturbation.to_parquet(
        output / "task_perturbation.parquet", index=False, compression="zstd"
    )

    source_audit = (
        _source_cell_audit(
            artifact,
            selected,
            batch_size=int(source_batch_size),
        )
        if audit_source
        else {
            "status": "not_run",
            "reason": "audit_source_false",
        }
    )

    # These small JSON files make the test boundary and the zero-copy cell
    # join explicit to evaluators and future loaders.  They intentionally do
    # not include expression values or target labels.
    cell_manifest = {
        "format": "brainwm_global_cell_manifest_view_v1",
        "source_cells": str(artifact / "cells.parquet"),
        "join_key": "sample_key",
        "row_identity": ["source_id", "row_index", "sample_key"],
        "split_column": "split in samples.parquet",
        "task_columns": {
            "integration": "task_integration.parquet",
            "annotation": "task_annotation.parquet",
            "development": "task_development.parquet",
            "spatial": "task_spatial.parquet",
            "perturbation": "task_perturbation.parquet",
        },
        "materialized_rows": 0,
        "expression_copied": False,
        "labels_copied": False,
    }
    (output / "cell_manifest.json").write_text(
        json.dumps(cell_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    test_samples = selected.loc[
        selected["split"].eq("test"),
        [
            "sample_key",
            "split_group",
            "split_unit",
            "source_id",
            "dataset",
            "assay_type",
            "selected_cells",
            "annotation_eligible",
            "development_eligible",
            "spatial_eligible",
        ],
    ].copy()
    test_samples.to_parquet(
        output / "sealed_test_samples.parquet", index=False, compression="zstd"
    )
    test_groups = groups.loc[groups["split"].eq("test")].copy()
    test_groups.to_parquet(
        output / "sealed_test_groups.parquet", index=False, compression="zstd"
    )
    access_policy = {
        "format": "brainwm_global_test_access_policy_v1",
        "split": "test",
        "sealed": True,
        "allowed_before_freeze": [
            "source_id",
            "row_index",
            "sample_key",
            "assay_type",
            "spatial coordinates when required by the declared task",
        ],
        "blocked_before_freeze": [
            "cell-type labels",
            "target expression used for model selection",
            "development target expression used for model selection",
            "perturbation outcomes (not present in this corpus)",
        ],
        "release_condition": "open target labels/expression once all representations and predictions are frozen",
        "same_split_for_all_tasks": True,
        "independent_unit": "split_group",
    }
    (output / "access_policy.json").write_text(
        json.dumps(access_policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    by_split = {
        split: {
            "samples": int((selected["split"] == split).sum()),
            "cells": int(selected.loc[selected["split"] == split, "selected_cells"].sum()),
            "groups": int((groups["split"] == split).sum()),
        }
        for split in SPLITS
    }
    by_split_assay: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        by_split_assay[split] = {
            assay: int(selected.loc[
                (selected["split"] == split) & (selected["assay_type"] == assay),
                "selected_cells",
            ].sum())
            for assay in sorted(set(selected["assay_type"].astype(str)))
        }
    total_cells = int(selected["selected_cells"].sum())
    actual_fractions = {
        split: float(by_split[split]["cells"] / total_cells)
        for split in SPLITS
    }
    coverage: dict[str, dict[str, int]] = {}
    for task in (*TASKS[:3], *AUXILIARY_TASKS):
        task_frame = _task_frame(selected, task)
        coverage[task] = {
            split: int(task_frame.loc[task_frame["split"] == split, "selected_cells"].sum())
            for split in SPLITS
        }

    manifest: dict[str, Any] = {
        "format": GLOBAL_SPLIT_FORMAT,
        "builder_version": BUILDER_VERSION,
        "split_id": split_id,
        "data_contract_id": f"DATA-{split_id}",
        "source_artifact": str(artifact),
        "species": species_name,
        "seed": seed,
        "fractions": fractions,
        "actual_cell_fractions": actual_fractions,
        "fraction_absolute_error": {
            split: float(abs(actual_fractions[split] - fractions[split]))
            for split in SPLITS
        },
        "donor_scope": donor_scope,
        "split_unit_policy": [
            "global donor when donor_scope=global",
            "dataset-qualified donor when donor_scope=dataset",
            "matched sample when donor is missing",
            "spatial section when donor and matched sample are missing",
            "sample key fallback",
        ],
        "zero_copy": True,
        "large_file_hashes": False,
        "counts": {
            "source_samples": int(len(source_samples)),
            "selected_samples": int(len(selected)),
            "selected_groups": int(len(groups)),
            "selected_cells": int(selected["selected_cells"].sum()),
            "by_split": by_split,
            "by_split_assay_cells": by_split_assay,
            "task_cells_by_split": coverage,
            "development_intervals": int(len(intervals)),
            "perturbation_rows": 0,
        },
        "files": {
            "samples": "samples.parquet",
            "split_groups": "split_groups.parquet",
            "source_lookup": "source_lookup.parquet",
            "development_intervals": "development_intervals.parquet",
            "perturbation": "task_perturbation.parquet",
            "cell_manifest": "cell_manifest.json",
            "sealed_test_samples": "sealed_test_samples.parquet",
            "sealed_test_groups": "sealed_test_groups.parquet",
            "access_policy": "access_policy.json",
            "tasks": task_paths,
        },
        "task_status": {
            "integration": "available",
            "annotation": "available_where_labels_exist",
            "development": "available_for_reliable_time_statuses",
            "spatial": "auxiliary_available_for_spatialRNA",
            "perturbation": "blocked_data",
        },
        "task_contract": {
            "integration": {
                "primary_unit": "sample",
                "eligible_assays": list(DEFAULT_ASSAYS),
                "test_access": "representation_frozen_then_score",
            },
            "annotation": {
                "primary_unit": "cell",
                "label_source": "source cell annotations when available",
                "test_access": "labels_open_once_after_prediction_freeze",
            },
            "development": {
                "primary_unit": "same-split population interval",
                "allowed_time_statuses": sorted(DEFAULT_TIME_STATUSES),
                "identity_claim": "population_level_only; no cell lineage",
            },
            "spatial": {
                "primary_unit": "section_or_donor_group",
                "query_isolation": "within-section neighbors exclude query cells",
            },
            "perturbation": {
                "status": "blocked_data",
                "reason": "healthy observational corpus contains no intervention/control outcomes",
            },
        },
        "source_cell_audit": source_audit,
        "lazy_cell_lookup": {
            "source_cells_path": str(artifact / "cells.parquet"),
            "key_column": "sample_key",
            "row_identity_columns": ["source_id", "row_index"],
            "api": "brainwm_data.global_split.GlobalBenchmarkIndex.iter_cell_index",
            "expression_copied": False,
        },
        "leakage_guards": [
            "one split per split_group",
            "global donor IDs do not cross splits when donor_scope=global",
            "spatial sections remain within one donor/group split",
            "development intervals require source and target samples in one split",
            "annotation labels are not copied into task manifests",
            "global test is sealed by sample/group assignment and can serve every task",
            "test labels and target expression remain evaluator-gated",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_data_card(output, manifest)
    _write_readme(output)
    split_receipt = {
        "format": "brainwm_global_split_receipt_v1",
        "split_id": split_id,
        "seed": seed,
        "source_artifact": str(artifact),
        "selected_samples": int(len(selected)),
        "selected_groups": int(len(groups)),
        "selected_cells": int(selected["selected_cells"].sum()),
        "fractions": fractions,
        "counts_by_split": by_split,
        "source_cell_audit_status": source_audit.get("status"),
        "checkpoint_modified": False,
        "expression_copied": False,
        "labels_copied": False,
    }
    (output / "split_receipt.json").write_text(
        json.dumps(split_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_data_card(output: Path, manifest: Mapping[str, Any]) -> None:
    counts = manifest["counts"]
    donor_scope = str(manifest.get("donor_scope", "dataset"))
    if donor_scope == "global":
        group_policy = "The primary split unit is a global donor ID. This keeps one donor in exactly one partition even when the donor appears in several datasets or brain regions."
    else:
        group_policy = "The primary split unit is a dataset-qualified donor. Missing donors use matched sample, spatial section, or sample key in that order."
    lines = [
        "# BrainWM global benchmark split",
        "",
        "This artifact is a lightweight, zero-copy 6:2:2 split for the global BrainWM benchmark.",
        "Expression and the source cell index remain in the source artifact; only sample/group and task indexes are written here.",
        "",
        f"- Source: `{manifest['source_artifact']}`",
        f"- Species: `{manifest['species']}`",
        f"- Seed: `{manifest['seed']}`",
        "- Fractions: train 0.60, validation 0.20, test 0.20 (weighted by selected cells; group boundaries take precedence)",
        f"- Samples / groups / cells: {counts['selected_samples']:,} / {counts['selected_groups']:,} / {counts['selected_cells']:,}",
        "",
        "## Split policy",
        "",
        group_policy + " A group is assigned exactly once, and all task manifests reuse that assignment.",
        "",
        "## Task manifests",
        "",
        "`task_integration.parquet`, `task_annotation.parquet`, `task_development.parquet`, and `task_spatial.parquet` contain sample-level eligibility and split references. `task_perturbation.parquet` is an explicit empty, blocked-data placeholder because this healthy observational corpus has no intervention outcomes. `development_intervals.parquet` contains adjacent-time population candidates; actual cell-level anatomy matching is performed lazily from the source cell index.",
        "",
        "## Test boundary",
        "",
        "The `test` rows are independent at the split-group level. `sealed_test_samples.parquet`, `sealed_test_groups.parquet`, and `access_policy.json` define the global test boundary. Target labels are not added to task tables; evaluators should open them only after representation/prediction artifacts are frozen.",
        "",
        "## Reading source rows",
        "",
        "Use `GlobalBenchmarkIndex.iter_cell_index(split, task=...)` to stream source row identifiers without opening expression matrices or copying `cells.parquet`.",
        "",
    ]
    (output / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def _write_readme(output: Path) -> None:
    lines = [
        "# Global benchmark split",
        "",
        "This directory contains the shared BrainWM benchmark boundary. It is an index, not a second expression corpus.",
        "",
        "```python",
        "from brainwm_data.global_split import GlobalBenchmarkIndex",
        f"index = GlobalBenchmarkIndex({str(output)!r})",
        "for cell_rows in index.iter_cell_index('test', task='annotation'):",
        "    # row identifiers only; expression is read later from the source H5AD",
        "    consume(cell_rows)",
        "```",
        "",
        "The task tables use the same `split` and `split_group`: integration, annotation, development, and the auxiliary spatial lane. The perturbation table is an explicit blocked placeholder. Development intervals are population-level candidates; the training loader must still intersect cell rows by split, dataset, assay, life stage, and anatomy before constructing an OT/flow batch.",
        "",
        "Do not edit `samples.parquet` or `split_groups.parquet` in place. Rebuild a new version with a new seed/version when the cohort changes. No checkpoint or source expression file is modified by this builder.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


class GlobalBenchmarkReader:
    """Source-expression reader whose batches obey the global split.

    ``IndexedH5ADBatchReader`` predates this artifact and interprets the
    source table's historical ``split`` column.  This adapter supplies an
    explicit cell frame from :meth:`GlobalBenchmarkIndex.iter_cell_index`, so
    the historical split can never leak into a global benchmark run.  Cell
    metadata is streamed row-group by row-group; expression remains lazy.
    """

    def __init__(
        self,
        index: "GlobalBenchmarkIndex",
        split: str,
        task: str | None,
        *,
        cell_columns: Sequence[str] | None = None,
        cell_source_ids: Sequence[int] | None = None,
        row_group_batch_size: int = 1,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        if task is not None and task not in (*TASKS, *AUXILIARY_TASKS):
            raise ValueError(f"unknown task: {task}")
        if row_group_batch_size < 1:
            raise ValueError("row_group_batch_size must be positive")
        if cell_source_ids is not None:
            self.cell_source_ids = tuple(sorted({int(value) for value in cell_source_ids}))
            if any(value < 0 for value in self.cell_source_ids):
                raise ValueError("cell_source_ids must be nonnegative")
        else:
            self.cell_source_ids = None
        self.index = index
        self.split = split
        self.task = task
        self.cell_columns = tuple(cell_columns) if cell_columns is not None else None
        self.row_group_batch_size = int(row_group_batch_size)
        from .indexed_h5ad import IndexedH5ADBatchReader

        # An explicit empty frame prevents the legacy reader from applying its
        # source ``split`` filter.  ``read_batch`` only needs sources and gene
        # maps, which it loads from the source artifact as usual.
        empty = pd.DataFrame(
            {
                "source_id": pd.Series([], dtype="int32"),
                "row_index": pd.Series([], dtype="int64"),
                "sample_key": pd.Series([], dtype="string"),
                "split": pd.Series([], dtype="string"),
            }
        )
        self._reader = IndexedH5ADBatchReader(
            index.source_artifact,
            cells=empty,
            allow_empty_cells=True,
        )

    @property
    def sources(self) -> pd.DataFrame:
        return self._reader.sources

    @property
    def gene_map(self) -> pd.DataFrame:
        return self._reader.gene_map

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> "GlobalBenchmarkReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def read_batch(
        self,
        source_id: int,
        row_indices: np.ndarray,
        cell_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """Read counts for an explicit global-split metadata frame."""
        if cell_frame is None:
            raise ValueError("global reader requires cell_frame for split-aware reads")
        frame = cell_frame.copy()
        if "global_split" in frame and set(frame["global_split"].astype(str)) != {self.split}:
            raise ValueError("cell_frame contains a different global split")
        if "sample_key" not in frame:
            raise ValueError("cell_frame must include sample_key")
        sample_keys = frame["sample_key"].map(_text)
        allowed_samples = self.index.sample_keys_for(self.split, self.task)
        if sample_keys.isna().any() or not sample_keys.astype(str).isin(allowed_samples).all():
            raise ValueError("cell_frame contains a sample outside the bound split/task")
        if "source_id" in frame:
            values = pd.to_numeric(frame["source_id"], errors="raise").to_numpy(np.int64)
            if len(values) != len(frame) or np.any(values != int(source_id)):
                raise ValueError("cell_frame source_id does not match source_id")
        rows = np.asarray(row_indices, dtype=np.int64)
        if "row_index" not in frame:
            raise ValueError("cell_frame must include row_index")
        listed = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(np.int64)
        if not np.array_equal(listed, rows):
            raise ValueError("cell_frame row_index does not match row_indices")
        if "split" not in frame:
            frame["split"] = self.split
        return self._reader.read_batch(int(source_id), rows, frame)

    def iter_local_batches(
        self,
        batch_size: int = 256,
        *,
        shuffle_sources: bool = True,
        seed: int = 0,
        metadata_columns: Sequence[str] | None = None,
        start_row: int = 0,
    ) -> Iterator[dict[str, Any]]:
        """Yield source-local expression batches under the global split."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if start_row < 0:
            raise ValueError("start_row must be nonnegative")
        rows_to_skip = int(start_row)
        columns = metadata_columns or self.cell_columns
        if columns is None:
            # Match the historical reader's useful metadata surface while
            # avoiding unrelated provenance columns that are never consumed
            # by model batches.
            columns = (
                "source_id", "row_index", "sample_key", "cell_id",
                "spatial_x", "spatial_y", "spatial_z", "uberon_id",
                "uberon_term", "cl_id", "cl_term", "original_cell_type",
                "original_subclass", "original_anatomical_region",
            )
        else:
            columns = tuple(columns)
        available_columns = set(
            pq.read_schema(self.index.source_artifact / "cells.parquet").names
        )
        columns = tuple(column for column in columns if column in available_columns)
        chunks = self.index.iter_cell_index(
            self.split,
            task=self.task,
            columns=columns,
            row_group_batch_size=self.row_group_batch_size,
        )
        # A source can span multiple row groups.  Processing each row-group
        # chunk independently bounds memory and still guarantees increasing
        # row indices within every expression read.
        for frame in chunks:
            if self.cell_source_ids is not None:
                frame = frame[frame["source_id"].isin(self.cell_source_ids)]
            if frame.empty:
                continue
            for source_id, part in frame.groupby("source_id", sort=False):
                source_id = int(source_id)
                part = part.sort_values("row_index", kind="stable")
                if rows_to_skip:
                    skipped = min(rows_to_skip, len(part))
                    rows_to_skip -= skipped
                    part = part.iloc[skipped:]
                if part.empty:
                    continue
                for start in range(0, len(part), batch_size):
                    batch = part.iloc[start : start + batch_size].copy()
                    yield self.read_batch(
                        source_id,
                        batch["row_index"].to_numpy(np.int64),
                        batch,
                    )
        if rows_to_skip:
            raise ValueError(f"start_row exceeds selected rows by {rows_to_skip}")

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            pass


class GlobalMultiCorpusReader:
    """Lazy expression reader for a multi-corpus global split.

    The global index stores corpus-qualified string IDs while the existing
    H5AD readers require source-local integer IDs. This adapter performs that
    translation per batch and keeps all expression handles source-local. It
    never materializes the combined cell table, which is important for the
    62M-row v7 artifact.
    """

    def __init__(
        self,
        index: "GlobalBenchmarkIndex",
        split: str,
        task: str | None,
        *,
        unified_vocabulary: str | Path,
        allow_unmapped_unified_genes: bool = False,
        cell_columns: Sequence[str] | None = None,
        cell_source_ids: Sequence[str] | None = None,
        row_group_batch_size: int = 1,
    ) -> None:
        if not index.multi_corpus:
            raise ValueError("GlobalMultiCorpusReader requires a multi-corpus index")
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        if task is not None and task not in (*TASKS, *AUXILIARY_TASKS):
            raise ValueError(f"unknown task: {task}")
        if row_group_batch_size < 1:
            raise ValueError("row_group_batch_size must be positive")
        if unified_vocabulary in (None, ""):
            raise ValueError("unified_vocabulary is required for multi-corpus reads")
        self.index = index
        self.split = split
        self.task = task
        self.unified_vocabulary = Path(unified_vocabulary).resolve()
        self.allow_unmapped_unified_genes = bool(allow_unmapped_unified_genes)
        self.cell_columns = tuple(cell_columns) if cell_columns is not None else None
        self.row_group_batch_size = int(row_group_batch_size)
        self._readers: dict[str, Any] = {}
        self._gene_adapters: dict[str, Any] = {}
        self._corpus_metadata: dict[str, Mapping[str, Any]] = {
            str(corpus["corpus_id"]): corpus for corpus in index.corpora
        }
        self._source_lookup: dict[str, tuple[str, int]] = {}
        # Keep a separate inventory so an explicitly requested source can be
        # distinguished from a source that exists in a corpus but was not
        # selected for this reader instance.
        all_source_ids: set[str] = set()
        self._closed = False

        from .indexed_h5ad import IndexedH5ADBatchReader
        from .mouse_molecular_index import MouseMolecularBatchReader
        from .mouse_spatial_index import MouseSpatialBatchReader
        from .unified_gene_index import UnifiedGeneTableAdapter

        requested_sources = None
        if cell_source_ids is not None:
            if isinstance(cell_source_ids, (str, bytes)):
                raise TypeError("cell_source_ids must be a collection of global source IDs")
            requested_sources = {str(value) for value in cell_source_ids}
            if not requested_sources:
                raise ValueError("cell_source_ids must not be empty")

        empty = pd.DataFrame(
            {
                "source_id": pd.Series([], dtype="int32"),
                "row_index": pd.Series([], dtype="int64"),
                "sample_key": pd.Series([], dtype="string"),
                "split": pd.Series([], dtype="string"),
            }
        )
        for corpus in index.corpora:
            corpus_id = str(corpus["corpus_id"])
            artifact = Path(corpus["artifact_dir"]).resolve()
            species = _canonical_species(corpus.get("species")) or _infer_species_from_corpus(
                corpus_id, artifact
            )
            reader_kind = str(corpus.get("reader_kind") or "").strip().lower()
            artifact_format: str | None = None
            artifact_manifest = artifact / "manifest.json"
            if artifact_manifest.is_file():
                try:
                    payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
                    value = payload.get("format")
                    artifact_format = str(value).strip() if value is not None else None
                except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                    # Older generic artifacts may have no readable manifest;
                    # retain the explicit reader_kind/name fallback below.
                    artifact_format = None
            source_path = artifact / "sources.parquet"
            source_frame = pd.read_parquet(source_path, columns=["source_id"])
            local_ids = pd.to_numeric(source_frame["source_id"], errors="raise").astype("int64")
            selected_local: list[int] = []
            for local_id in local_ids.tolist():
                global_id = qualify_source_id(corpus_id, int(local_id))
                all_source_ids.add(global_id)
                if requested_sources is None or global_id in requested_sources:
                    selected_local.append(int(local_id))
            if requested_sources is not None and not selected_local:
                continue
            for local_id in selected_local:
                global_id = qualify_source_id(corpus_id, int(local_id))
                self._source_lookup[global_id] = (corpus_id, int(local_id))

            lowered = corpus_id.lower()
            is_spatial = (
                reader_kind == "mouse_spatial"
                or artifact_format == "brainwm_mouse_spatial_cell_index_v1"
                or (not reader_kind and "spatial" in lowered)
            )
            is_mouse = (
                species == "mouse"
                or artifact_format in {
                    "brainwm_mouse_molecular_cell_index_v1",
                    "brainwm_mouse_spatial_cell_index_v1",
                }
                or "mouse" in lowered
            )
            if is_spatial:
                reader = MouseSpatialBatchReader(
                    artifact,
                    cells=empty,
                    cell_source_ids=selected_local,
                    allow_empty_cells=True,
                    unified_vocabulary=self.unified_vocabulary,
                    allow_unmapped_unified_genes=self.allow_unmapped_unified_genes,
                )
            elif reader_kind == "mouse_molecular" or is_mouse:
                reader = MouseMolecularBatchReader(
                    artifact,
                    cells=empty,
                    cell_source_ids=selected_local,
                    allow_empty_cells=True,
                    unified_vocabulary=self.unified_vocabulary,
                    allow_unmapped_unified_genes=self.allow_unmapped_unified_genes,
                )
            else:
                reader = IndexedH5ADBatchReader(
                    artifact,
                    cells=empty,
                    cell_source_ids=selected_local,
                    allow_empty_cells=True,
                )
            self._readers[corpus_id] = reader

            # The historical human artifact stores a compact, local
            # ``gene_index`` table.  Do not assume that its positions happen
            # to equal the unified vocabulary; construct an explicit
            # Ensembl-keyed projection and apply it at read time.  Specialized
            # mouse readers already perform the equivalent namespace adapter.
            if not is_mouse:
                if species is None:
                    self.close()
                    raise ValueError(
                        f"multi-corpus reader requires explicit species for {corpus_id!r}"
                    )
                genes_path = artifact / "genes.parquet"
                if not genes_path.is_file():
                    self.close()
                    raise FileNotFoundError(genes_path)
                try:
                    namespace_genes = pd.read_parquet(genes_path)
                    self._gene_adapters[corpus_id] = UnifiedGeneTableAdapter(
                        namespace_genes,
                        self.unified_vocabulary,
                        species=species,
                        allow_unmapped=self.allow_unmapped_unified_genes,
                    )
                except Exception:
                    self.close()
                    raise

        if requested_sources is not None:
            unknown = sorted(requested_sources - all_source_ids)
            if unknown:
                self.close()
                raise KeyError(f"unknown global source IDs: {unknown[:5]}")
        vocab_path = self.unified_vocabulary / "genes.parquet"
        if not vocab_path.is_file():
            self.close()
            raise FileNotFoundError(vocab_path)
        vocab = pd.read_parquet(vocab_path, columns=["gene_index"])
        indices = pd.to_numeric(vocab["gene_index"], errors="raise").to_numpy(np.int64)
        if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
            self.close()
            raise ValueError("unified vocabulary gene_index must be contiguous")
        self.n_genes = int(len(indices))
        self.cell_source_ids = (
            tuple(sorted(requested_sources)) if requested_sources is not None else None
        )

    @property
    def sources(self) -> pd.DataFrame:
        """Return a small source table with corpus-qualified identifiers."""
        parts: list[pd.DataFrame] = []
        for corpus_id, reader in self._readers.items():
            # ``IndexedH5ADBatchReader.sources`` is indexed by the local
            # ``source_id`` and does not retain that value as a data column.
            # Preserve the index when flattening the per-corpus view; using
            # ``drop=True`` silently discarded it and made this property fail
            # with ``KeyError('source_id')`` for every non-empty corpus.
            frame = reader.sources.reset_index().copy()
            frame["local_source_id"] = pd.to_numeric(frame["source_id"], errors="raise").astype("int64")
            frame["global_source_id"] = frame["local_source_id"].map(
                lambda value, corpus=corpus_id: qualify_source_id(corpus, int(value))
            )
            frame["corpus_id"] = corpus_id
            parts.append(frame)
        if not parts:
            return pd.DataFrame(columns=["global_source_id", "corpus_id", "local_source_id"])
        return pd.concat(parts, ignore_index=True, sort=False)

    def _resolve(self, global_source_id: str, frame: pd.DataFrame | None = None) -> tuple[str, int, str]:
        key = str(global_source_id)
        if frame is not None and "global_source_id" in frame:
            values = frame["global_source_id"].dropna().astype(str).unique()
            if len(values) != 1 or values[0] != key:
                raise ValueError("cell_frame global_source_id does not match source_id")
        try:
            corpus, local = self._source_lookup[key]
        except KeyError as exc:
            raise KeyError(f"unknown global source_id={key}") from exc
        return corpus, local, key

    def read_batch(
        self,
        global_source_id: str,
        row_indices: np.ndarray,
        cell_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """Read one source-local batch while preserving global row metadata."""
        if cell_frame is None:
            raise ValueError("multi-corpus reader requires cell_frame")
        frame = cell_frame.copy()
        if "global_split" in frame and set(frame["global_split"].astype(str)) != {self.split}:
            raise ValueError("cell_frame contains a different global split")
        corpus, local_source, key = self._resolve(str(global_source_id), frame)
        if "corpus_id" in frame:
            corpora = frame["corpus_id"].dropna().astype(str).unique()
            if len(corpora) != 1 or corpora[0] != corpus:
                raise ValueError("cell_frame corpus_id does not match global_source_id")
        for column in ("local_source_id", "source_id"):
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(np.int64)
            if len(values) != len(frame) or np.any(values != local_source):
                raise ValueError(
                    f"cell_frame {column} does not match global_source_id"
                )
        if "sample_key" not in frame:
            raise ValueError("cell_frame must include sample_key")
        sample_keys = frame["sample_key"].map(_text)
        allowed_samples = self.index.sample_keys_for(self.split, self.task)
        if sample_keys.isna().any() or not sample_keys.astype(str).isin(allowed_samples).all():
            raise ValueError("cell_frame contains a sample outside the bound split/task")
        rows = np.asarray(row_indices, dtype=np.int64)
        if "row_index" not in frame:
            raise ValueError("cell_frame must include row_index")
        listed = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(np.int64)
        if not np.array_equal(listed, rows):
            raise ValueError("cell_frame row_index does not match row_indices")
        local_frame = frame.copy()
        local_frame["source_id"] = local_source
        local_frame["split"] = self.split
        batch = self._readers[corpus].read_batch(local_source, rows, local_frame)
        adapter = self._gene_adapters.get(corpus)
        if adapter is not None:
            batch = adapter.remap_batch(batch)
        corpus_meta = self._corpus_metadata.get(corpus, {})
        species = _canonical_species(corpus_meta.get("species"))
        if species is not None:
            batch["species"] = species
        species_id = corpus_meta.get("species_id")
        if species_id is not None and not _nullish(species_id):
            batch["species_id"] = int(species_id)
        # Keep source_id compatible with the underlying model batch contract;
        # the unambiguous corpus-qualified ID is carried alongside it.
        batch["source_id"] = int(local_source)
        batch["local_source_id"] = int(local_source)
        batch["global_source_id"] = key
        batch["corpus_id"] = corpus
        batch["cells"] = frame.reset_index(drop=True)
        if "assay_type" not in batch:
            assay_values: list[Any] = []
            for column in ("assay_type", "assay", "modality"):
                if column in frame:
                    assay_values.extend(frame[column].dropna().tolist())
            if not assay_values:
                source = self._readers[corpus].sources.loc[local_source]
                for column in ("assay_type", "assay", "modality", "assay_types"):
                    if column in source.index and not _nullish(source[column]):
                        assay_values.append(source[column])
                        break
                if not assay_values and bool(source.get("has_spatial", False)):
                    assay_values.append("spatialRNA")
            if assay_values:
                normalized = {_assay_value(value) for value in assay_values}
                normalized.discard("unknown")
                if len(normalized) == 1:
                    batch["assay_type"] = next(iter(normalized))
        return batch

    def iter_local_batches(
        self,
        batch_size: int = 256,
        *,
        shuffle_sources: bool = True,
        seed: int = 0,
        metadata_columns: Sequence[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream source-local batches from the fixed global split."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        columns = metadata_columns or self.cell_columns
        if columns is None:
            columns = (
                "source_id", "row_index", "sample_key", "cell_id",
                "spatial_x", "spatial_y", "spatial_z", "uberon_id",
                "uberon_term", "cl_id", "cl_term", "original_cell_type",
                "original_subclass", "original_anatomical_region",
                "assay_type", "species", "species_id",
            )
        source_order = list(self._source_lookup)
        if self.cell_source_ids is not None:
            allowed = set(self.cell_source_ids)
            source_order = [value for value in source_order if value in allowed]
        if shuffle_sources:
            np.random.default_rng(seed).shuffle(source_order)
        source_order_map = {value: pos for pos, value in enumerate(source_order)}
        for frame in self.index.iter_cell_index(
            self.split,
            task=self.task,
            columns=columns,
            row_group_batch_size=self.row_group_batch_size,
        ):
            if frame.empty:
                continue
            if self.cell_source_ids is not None:
                frame = frame[frame["global_source_id"].astype(str).isin(set(self.cell_source_ids))]
            if frame.empty:
                continue
            groups = sorted(
                frame.groupby("global_source_id", sort=False),
                key=lambda item: source_order_map.get(str(item[0]), len(source_order_map)),
            )
            for global_source_id, part in groups:
                part = part.sort_values("row_index", kind="stable")
                for start in range(0, len(part), batch_size):
                    batch_frame = part.iloc[start : start + batch_size].copy()
                    yield self.read_batch(
                        str(global_source_id),
                        batch_frame["row_index"].to_numpy(np.int64),
                        batch_frame,
                    )

    def close(self) -> None:
        if self._closed:
            return
        for reader in self._readers.values():
            reader.close()
        self._closed = True

    def __enter__(self) -> "GlobalMultiCorpusReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            pass


class GlobalBenchmarkIndex:
    """Lazy access to source cell rows under the global split contract."""

    def __init__(self, manifest_dir: str | Path):
        self.root = Path(manifest_dir).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != GLOBAL_SPLIT_FORMAT:
            raise ValueError("unexpected global benchmark split format")
        corpus_entries = self.manifest.get("corpora")
        self.multi_corpus = bool(
            self.manifest.get("multi_corpus")
            or (corpus_entries is not None and len(corpus_entries) > 1)
        )
        if corpus_entries:
            self.corpora = tuple(
                {
                    "corpus_id": _canonical_corpus_id(entry["corpus_id"]),
                    "artifact_dir": str(
                        (self.root / entry["artifact_dir"]).resolve()
                        if not Path(entry["artifact_dir"]).is_absolute()
                        else Path(entry["artifact_dir"]).resolve()
                    ),
                    # Preserve explicit corpus metadata from the split
                    # manifest.  Readers must not infer species from a path
                    # when the builder already recorded it.
                    "species": entry.get("species"),
                    "species_id": entry.get("species_id"),
                    "reader_kind": entry.get("reader_kind"),
                    "sample_key_prefix": entry.get("sample_key_prefix"),
                }
                for entry in corpus_entries
            )
        else:
            self.corpora = ({
                "corpus_id": str(self.manifest.get("species", "human")),
                "artifact_dir": str(Path(self.manifest["source_artifact"]).resolve()),
                "sample_key_prefix": None,
            },)
        # Keep this attribute for legacy clients.  Multi-corpus callers should
        # use ``corpora`` or ``iter_cell_index`` instead.
        self.source_artifact = Path(self.corpora[0]["artifact_dir"])
        self.samples = pd.read_parquet(self.root / self.manifest["files"]["samples"])
        self._task_cache: dict[str, set[str]] = {}
        self._split_cache: dict[str, set[str]] = {}
        self._corpus_sample_maps: dict[str, dict[str, str]] = {}
        if self.multi_corpus:
            if "corpus_id" not in self.samples or "local_sample_key" not in self.samples:
                raise ValueError(
                    "multi-corpus samples must contain corpus_id and local_sample_key"
                )
            for corpus, part in self.samples.groupby("corpus_id", sort=False):
                if "global_sample_key" in part:
                    values = part["global_sample_key"]
                else:
                    values = part["sample_key"]
                self._corpus_sample_maps[str(corpus)] = dict(
                    zip(part["local_sample_key"].astype(str), values.astype(str))
                )

    def sample_keys_for(self, split: str, task: str | None = None) -> set[str]:
        """Return sample keys for a split/task without reading expression."""
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        cache_key = f"{split}|{task or '*'}"
        if cache_key in self._split_cache:
            return set(self._split_cache[cache_key])
        frame = self.samples[self.samples["split"] == split]
        if task is not None:
            if task not in (*TASKS, *AUXILIARY_TASKS):
                raise ValueError(f"unknown task: {task}")
            files = self.manifest.get("files", {})
            filename = files.get("tasks", {}).get(task) or files.get(task)
            task_path = self.root / str(filename) if filename else None
            if task_path is None or not task_path.is_file():
                # The perturbation table is intentionally empty but still has
                # a file entry; this branch gives a clear error if an old
                # artifact predates that entry.
                raise ValueError(f"task manifest is missing: {task}")
            task_frame = pd.read_parquet(task_path, columns=["sample_key"])
            keys = set(task_frame["sample_key"].astype(str))
            frame = frame[frame["sample_key"].astype(str).isin(keys)]
        result = set(frame["sample_key"].astype(str))
        self._split_cache[cache_key] = result
        return set(result)

    def samples_for(self, split: str, task: str | None = None) -> pd.DataFrame:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        keys = self.sample_keys_for(split, task)
        return self.samples[self.samples["sample_key"].astype(str).isin(keys)].copy()

    def _task_keys(self, task: str | None) -> set[str] | None:
        if task is None:
            return None
        if task not in (*TASKS, *AUXILIARY_TASKS):
            raise ValueError(f"unknown task: {task}")
        if task not in self._task_cache:
            self._task_cache[task] = self.sample_keys_for("train", task) | self.sample_keys_for("validation", task) | self.sample_keys_for("test", task)
        return self._task_cache[task]

    def reader(self, split: str, *, task: str | None = None, **kwargs: Any):
        """Construct a split-aware lazy expression reader.

        ``IndexedH5ADBatchReader``'s historical split column is intentionally
        bypassed.  Supported keyword arguments are ``cell_columns``,
        ``cell_source_ids`` and ``row_group_batch_size``.  Multi-corpus
        manifests additionally require an explicit ``unified_vocabulary``;
        without it we fail closed rather than concatenating species-local gene
        indices.
        """
        forbidden = {"cell_sample_keys", "cell_splits", "cells", "allow_empty_cells"}
        unexpected = forbidden.intersection(kwargs)
        if unexpected:
            raise ValueError(
                "global reader owns these arguments: " + ", ".join(sorted(unexpected))
            )
        allowed = {
            "cell_columns",
            "cell_source_ids",
            "row_group_batch_size",
            "unified_vocabulary",
            "allow_unmapped_unified_genes",
        }
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError("unsupported global reader arguments: " + ", ".join(sorted(unexpected)))
        if self.multi_corpus:
            unified = kwargs.get("unified_vocabulary")
            if unified in (None, ""):
                raise ValueError(
                    "multi-corpus reader requires explicit unified_vocabulary"
                )
            return GlobalMultiCorpusReader(
                self,
                split,
                task,
                cell_columns=kwargs.get("cell_columns"),
                cell_source_ids=kwargs.get("cell_source_ids"),
                row_group_batch_size=kwargs.get("row_group_batch_size", 1),
                unified_vocabulary=unified,
                allow_unmapped_unified_genes=kwargs.get(
                    "allow_unmapped_unified_genes", False
                ),
            )
        return GlobalBenchmarkReader(self, split, task, **kwargs)

    def _iter_multi_cell_index(
        self,
        split: str,
        task: str | None,
        columns: Sequence[str] | None,
        row_group_batch_size: int,
    ) -> Iterator[pd.DataFrame]:
        """Stream and namespace rows from each corpus independently."""
        split_keys = self.sample_keys_for(split)
        task_keys = self._task_keys(task)
        requested = list(columns or ())
        # Derived identity columns are produced below and must not be sent to
        # Parquet.  Every source still needs local source/row/sample columns.
        derived = {
            "corpus_id", "local_source_id", "global_source_id", "source_key",
            "local_sample_key", "global_sample_key", "global_row_id", "global_split",
            "task",
        }
        for corpus in self.corpora:
            corpus_id = str(corpus["corpus_id"])
            path = Path(corpus["artifact_dir"]) / "cells.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            parquet = pq.ParquetFile(path)
            available = set(parquet.schema.names)
            sample_column = next((name for name in _SAMPLE_KEY_ALIASES if name in available), None)
            source_column = next((name for name in _SOURCE_ID_ALIASES if name in available), None)
            row_column = "row_index" if "row_index" in available else "obs_index" if "obs_index" in available else None
            if sample_column is None or source_column is None or row_column is None:
                raise ValueError(
                    f"{path} must contain sample/source/row identity aliases"
                )
            read_columns: list[str] = []
            for column in requested:
                if column in derived:
                    continue
                if column in available and column not in read_columns:
                    read_columns.append(column)
            for column in (sample_column, source_column, row_column):
                if column not in read_columns:
                    read_columns.append(column)
            sample_map = self._corpus_sample_maps.get(corpus_id, {})
            sample_prefix = str(corpus.get("sample_key_prefix") or corpus_id)
            for start in range(0, parquet.num_row_groups, row_group_batch_size):
                end = min(start + row_group_batch_size, parquet.num_row_groups)
                pieces = [
                    parquet.read_row_group(index, columns=read_columns).to_pandas()
                    for index in range(start, end)
                ]
                frame = pd.concat(pieces, ignore_index=True)
                local_sample = frame[sample_column].map(_text)
                global_sample = local_sample.map(
                    lambda value: _map_local_sample_key(
                        value, sample_map, prefix=sample_prefix
                    )
                )
                mask = global_sample.isin(split_keys)
                if task_keys is not None:
                    mask &= global_sample.isin(task_keys)
                selected = frame.loc[mask].copy()
                if selected.empty:
                    continue
                selected["corpus_id"] = corpus_id
                selected["local_sample_key"] = local_sample.loc[mask].astype(str).to_numpy()
                selected["global_sample_key"] = global_sample.loc[mask].astype(str).to_numpy()
                selected["sample_key"] = selected["global_sample_key"]
                local_source = pd.to_numeric(
                    selected[source_column], errors="raise"
                ).astype("int64")
                selected["local_source_id"] = local_source.to_numpy()
                selected["source_id"] = selected["local_source_id"]
                selected["global_source_id"] = selected["local_source_id"].map(
                    lambda value: qualify_source_id(corpus_id, value)
                )
                selected["source_key"] = selected["global_source_id"]
                local_row = pd.to_numeric(selected[row_column], errors="raise").astype("int64")
                selected["row_index"] = local_row.to_numpy()
                selected["global_row_id"] = [
                    qualify_row_id(corpus_id, source_id, row_index)
                    for source_id, row_index in zip(local_source, local_row)
                ]
                selected["global_split"] = split
                if task is not None:
                    selected["task"] = task
                if columns is not None:
                    # Keep requested source columns first, then the mandatory
                    # identity fields.  Missing derived names are intentional.
                    keep = [column for column in requested if column in selected.columns]
                    for column in (
                        "sample_key", "local_sample_key", "global_sample_key",
                        "source_id", "local_source_id", "global_source_id",
                        "row_index", "global_row_id", "corpus_id", "global_split",
                    ):
                        if column not in keep and column in selected.columns:
                            keep.append(column)
                    if task is not None and "task" not in keep:
                        keep.append("task")
                    selected = selected[keep]
                yield selected.reset_index(drop=True)

    def iter_cell_index(
        self,
        split: str,
        *,
        task: str | None = None,
        columns: Sequence[str] | None = None,
        row_group_batch_size: int = 1,
    ) -> Iterator[pd.DataFrame]:
        """Yield source cell-index rows lazily, without reading expression."""
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        if row_group_batch_size < 1:
            raise ValueError("row_group_batch_size must be positive")
        if self.multi_corpus:
            yield from self._iter_multi_cell_index(
                split, task, columns, row_group_batch_size
            )
            return
        split_keys = self.sample_keys_for(split)
        task_keys = self._task_keys(task)
        path = self.source_artifact / "cells.parquet"
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)
        requested = list(columns or ("source_id", "row_index", "sample_key"))
        for required in ("source_id", "row_index", "sample_key"):
            if required not in requested:
                requested.append(required)
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"source cells are missing columns: {', '.join(missing)}")
        # Reading one row group at a time bounds memory and avoids materializing
        # the 40M-row source table.  ``row_group_batch_size`` permits callers
        # to combine a few neighboring groups when desired.
        for start in range(0, parquet.num_row_groups, row_group_batch_size):
            end = min(start + row_group_batch_size, parquet.num_row_groups)
            pieces = [
                parquet.read_row_group(index, columns=requested).to_pandas()
                for index in range(start, end)
            ]
            frame = pd.concat(pieces, ignore_index=True)
            mask = frame["sample_key"].astype(str).isin(split_keys)
            if task_keys is not None:
                mask &= frame["sample_key"].astype(str).isin(task_keys)
            selected = frame.loc[mask].copy()
            if selected.empty:
                continue
            if "global_row_id" not in selected:
                selected["global_row_id"] = [
                    f"source::{int(source_id)}::row::{int(row_index)}"
                    for source_id, row_index in zip(
                        pd.to_numeric(selected["source_id"], errors="raise"),
                        pd.to_numeric(selected["row_index"], errors="raise"),
                    )
                ]
            selected["global_split"] = split
            if task is not None:
                selected["task"] = task
            yield selected.reset_index(drop=True)


def _validate_multi_global_benchmark_split(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    check_source: bool,
    source_batch_size: int,
) -> dict[str, Any]:
    """Validate the composite metadata contract without opening H5AD files."""
    files = manifest.get("files", {})
    samples = pd.read_parquet(root / files["samples"])
    groups = pd.read_parquet(root / files["split_groups"])
    _require_columns(
        samples,
        (
            "sample_key", "global_sample_key", "local_sample_key", "corpus_id",
            "split", "split_group", "assay_type", "selected_cells",
        ),
        "multi-corpus samples",
    )
    _require_columns(groups, ("split_group", "split"), "multi-corpus split groups")
    if samples["sample_key"].duplicated().any():
        raise ValueError("duplicate global sample keys")
    if samples["global_sample_key"].astype(str).ne(samples["sample_key"].astype(str)).any():
        raise ValueError("sample_key differs from global_sample_key")
    if samples[["corpus_id", "local_sample_key"]].duplicated().any():
        raise ValueError("duplicate corpus/local sample keys")
    if groups["split_group"].duplicated().any():
        raise ValueError("duplicate split groups")
    if set(samples["split"].astype(str)) - set(SPLITS) or set(groups["split"].astype(str)) - set(SPLITS):
        raise ValueError("unknown split name")
    mapping = groups.set_index("split_group")["split"]
    expected = samples["split_group"].map(mapping)
    if expected.isna().any() or not expected.astype(str).equals(samples["split"].astype(str)):
        raise ValueError("sample split does not match group assignment")
    if samples.groupby("split_group")["split"].nunique().max() != 1:
        raise ValueError("a split group crosses partitions")
    # Reuse the identity scope declared by the artifact.  The historical
    # default is dataset-qualified, but strict human benchmarks use a global
    # donor scope; silently falling back to the default would make validation
    # weaker than the build-time contract.
    _validate_identity_boundaries(
        samples,
        donor_scope=_validate_donor_scope(str(manifest.get("donor_scope", "dataset"))),
    )
    _validate_manifest_counts(manifest, samples, groups)
    _validate_sealed_boundary(root, manifest, samples, groups)

    source_lookup_path = root / files["source_lookup"]
    sources = pd.read_parquet(source_lookup_path)
    _require_columns(sources, ("corpus_id", "local_source_id", "global_source_id"), "multi-corpus source lookup")
    if sources["global_source_id"].duplicated().any():
        raise ValueError("duplicate global source IDs")
    if sources[["corpus_id", "local_source_id"]].duplicated().any():
        raise ValueError("duplicate corpus/local source IDs")

    available_assays = sorted(set(samples["assay_type"].astype(str)))
    modality_coverage: dict[str, dict[str, int]] = {}
    for assay in available_assays:
        by = samples[samples["assay_type"].astype(str) == assay].groupby("split")["selected_cells"].sum()
        modality_coverage[assay] = {split: int(by.get(split, 0)) for split in SPLITS}

    task_stats: dict[str, dict[str, int]] = {}
    task_files = files.get("tasks", {})
    sample_lookup_frame = samples.set_index("sample_key")
    sample_lookup = sample_lookup_frame["split"]
    for task in ("integration", "annotation", "development", "spatial"):
        filename = task_files.get(task)
        if not filename:
            raise ValueError(f"task manifest is missing: {task}")
        task_frame = pd.read_parquet(root / filename)
        _require_columns(task_frame, ("sample_key", "split", "split_group"), f"task {task}")
        if set(task_frame["sample_key"].astype(str)) - set(sample_lookup.index.astype(str)):
            raise ValueError(f"task {task} has unknown samples")
        expected_task = task_frame["sample_key"].astype(str).map(sample_lookup.astype(str))
        if not expected_task.equals(task_frame["split"].astype(str)):
            raise ValueError(f"task {task} split drifted from global sample manifest")
        expected_groups = task_frame["sample_key"].astype(str).map(
            sample_lookup_frame["split_group"].astype(str)
        )
        if not expected_groups.equals(task_frame["split_group"].astype(str)):
            raise ValueError(f"task {task} group drifted from global sample manifest")
        task_stats[task] = {
            split: int(task_frame.loc[task_frame["split"].astype(str) == split, "selected_cells"].sum())
            for split in SPLITS
        }
    perturbation_path = root / files.get("perturbation", "task_perturbation.parquet")
    if not perturbation_path.is_file() or not pd.read_parquet(perturbation_path).empty:
        raise ValueError("multi-corpus perturbation task must be an empty placeholder")

    _validate_manifest_counts(manifest, samples, groups)
    _validate_sealed_boundary(root, manifest, samples, groups)

    intervals_path = root / files["development_intervals"]
    intervals = pd.read_parquet(intervals_path)
    if not intervals.empty:
        for row in intervals.itertuples(index=False):
            split = str(row.split)
            for column in ("source_sample_keys", "target_sample_keys"):
                keys = json.loads(str(getattr(row, column)))
                if not set(keys) <= set(sample_lookup.index):
                    raise ValueError("development interval references unknown sample")
                if any(str(sample_lookup.loc[key]) != split for key in keys):
                    raise ValueError("development interval crosses global split")

    source_audit = manifest.get("source_cell_audit", {"status": "not_recorded"})
    if check_source:
        specs = [
            BenchmarkCorpusSpec(
                corpus_id=entry["corpus_id"],
                artifact_dir=entry["artifact_dir"],
                species=entry.get("species"),
                species_id=entry.get("species_id"),
                sample_key_prefix=entry.get("sample_key_prefix"),
            )
            for entry in manifest.get("corpora", [])
        ]
        mappings = {
            str(corpus): dict(
                zip(part["local_sample_key"].astype(str), part["sample_key"].astype(str))
            )
            for corpus, part in samples.groupby("corpus_id", sort=False)
        }
        source_audit = _source_cell_audit_multi(
            specs, samples, mappings, batch_size=int(source_batch_size)
        )
        if source_audit.get("status") != "passed":
            raise ValueError("source cell audit failed")
    return {
        "format": GLOBAL_SPLIT_FORMAT,
        "multi_corpus": True,
        "corpora": [str(entry["corpus_id"]) for entry in manifest.get("corpora", [])],
        "samples": int(len(samples)),
        "groups": int(len(groups)),
        "cells": int(samples["selected_cells"].sum()),
        "by_split": {
            split: {
                "samples": int((samples["split"].astype(str) == split).sum()),
                "groups": int((groups["split"].astype(str) == split).sum()),
                "cells": int(samples.loc[samples["split"].astype(str) == split, "selected_cells"].sum()),
            }
            for split in SPLITS
        },
        "modality_cells": modality_coverage,
        "task_cells": task_stats,
        "source_cell_audit": source_audit,
        "perturbation_status": "blocked_data_empty",
    }


def validate_global_benchmark_split(
    root: str | Path,
    *,
    check_source: bool = False,
    source_batch_size: int = 1_000_000,
) -> dict[str, Any]:
    """Validate split, task, and lazy-source invariants.

    The default checks only the compact manifests.  ``check_source=True``
    performs one bounded pass over the source cell index to verify that the
    sample counts in the split are still backed by the source rows.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != GLOBAL_SPLIT_FORMAT:
        raise ValueError("unexpected global benchmark split format")
    if manifest.get("multi_corpus"):
        return _validate_multi_global_benchmark_split(
            root,
            manifest,
            check_source=check_source,
            source_batch_size=source_batch_size,
        )
    samples = pd.read_parquet(root / manifest["files"]["samples"])
    groups = pd.read_parquet(root / manifest["files"]["split_groups"])
    _require_columns(samples, ("sample_key", "split", "split_group", "assay_type", "selected_cells"), "global samples")
    _require_columns(groups, ("split_group", "split"), "split groups")
    if samples["sample_key"].duplicated().any():
        raise ValueError("duplicate sample keys")
    if groups["split_group"].duplicated().any():
        raise ValueError("duplicate split groups")
    if set(samples["split"]) - set(SPLITS) or set(groups["split"]) - set(SPLITS):
        raise ValueError("unknown split name")
    mapping = groups.set_index("split_group")["split"]
    expected = samples["split_group"].map(mapping)
    if expected.isna().any() or not expected.equals(samples["split"]):
        raise ValueError("sample split does not match group assignment")
    if samples.groupby("split_group")["split"].nunique().max() != 1:
        raise ValueError("a split group crosses partitions")
    # Validate against the scope recorded at build time.  In particular, a
    # strict global-donor artifact must not be downgraded to the historical
    # dataset-qualified check during metadata validation.
    _validate_identity_boundaries(
        samples,
        donor_scope=_validate_donor_scope(str(manifest.get("donor_scope", "dataset"))),
    )

    _validate_manifest_counts(manifest, samples, groups)
    _validate_sealed_boundary(root, manifest, samples, groups)

    available_assays = sorted(set(samples["assay_type"].astype(str)))
    modality_coverage: dict[str, dict[str, int]] = {}
    for assay in available_assays:
        by = samples[samples["assay_type"] == assay].groupby("split")["selected_cells"].sum()
        modality_coverage[assay] = {split: int(by.get(split, 0)) for split in SPLITS}
        group_count = samples.loc[samples["assay_type"] == assay, "split_group"].nunique()
        if group_count >= len(SPLITS) and any(modality_coverage[assay][split] <= 0 for split in SPLITS):
            raise ValueError(f"assay {assay} is absent from one split")

    task_stats: dict[str, dict[str, int]] = {}
    task_files = manifest.get("files", {}).get("tasks", {})
    for task in (*TASKS[:3], *AUXILIARY_TASKS):
        path = root / task_files[task]
        task_frame = pd.read_parquet(path)
        _require_columns(task_frame, ("sample_key", "split", "split_group"), f"task {task}")
        if set(task_frame["sample_key"]) - set(samples["sample_key"]):
            raise ValueError(f"task {task} has unknown samples")
        sample_lookup_frame = samples.set_index("sample_key")
        if not task_frame["split"].equals(task_frame["sample_key"].map(sample_lookup_frame["split"])):
            raise ValueError(f"task {task} split drifted from global sample manifest")
        if not task_frame["split_group"].equals(
            task_frame["sample_key"].map(sample_lookup_frame["split_group"])
        ):
            raise ValueError(f"task {task} group drifted from global sample manifest")
        task_stats[task] = {
            split: int(task_frame.loc[task_frame["split"] == split, "selected_cells"].sum())
            for split in SPLITS
        }
        if task in TASKS[:3] and not (task_frame["split"] == "test").any():
            raise ValueError(f"global test has no eligible rows for required task {task}")
    perturbation_path = root / manifest.get("files", {}).get(
        "perturbation", "task_perturbation.parquet"
    )
    if not perturbation_path.is_file():
        raise ValueError("missing perturbation task placeholder")
    perturbation = pd.read_parquet(perturbation_path)
    if not perturbation.empty:
        raise ValueError("healthy global split must not contain perturbation rows")
    intervals = pd.read_parquet(root / manifest["files"]["development_intervals"])
    if not intervals.empty:
        if set(intervals["split"]) - set(SPLITS):
            raise ValueError("development interval has unknown split")
        sample_lookup = samples.set_index("sample_key")["split"]
        for row in intervals.itertuples(index=False):
            split = str(row.split)
            for column in ("source_sample_keys", "target_sample_keys"):
                encoded = getattr(row, column)
                keys = json.loads(str(encoded))
                if not set(keys) <= set(sample_lookup.index):
                    raise ValueError("development interval references unknown sample")
                if any(str(sample_lookup.loc[key]) != split for key in keys):
                    raise ValueError("development interval crosses global split")

    source_audit = manifest.get("source_cell_audit", {"status": "not_recorded"})
    if check_source:
        source_audit = _source_cell_audit(
            Path(manifest["source_artifact"]),
            samples,
            batch_size=int(source_batch_size),
        )
        if source_audit.get("status") != "passed":
            raise ValueError("source cell audit failed")
    result = {
        "format": GLOBAL_SPLIT_FORMAT,
        "samples": int(len(samples)),
        "groups": int(len(groups)),
        "cells": int(samples["selected_cells"].sum()),
        "by_split": {
            split: {
                "samples": int((samples["split"] == split).sum()),
                "groups": int((groups["split"] == split).sum()),
                "cells": int(samples.loc[samples["split"] == split, "selected_cells"].sum()),
            }
            for split in SPLITS
        },
        "modality_cells": modality_coverage,
        "task_cells": task_stats,
        "source_cell_audit": source_audit,
        "perturbation_status": "blocked_data_empty",
    }
    return result


def _parse_fraction(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="legacy single artifact path")
    parser.add_argument(
        "--corpus-spec",
        action="append",
        default=None,
        metavar="JSON",
        help=(
            "repeatable JSON corpus spec, e.g. "
            '{"corpus_id":"mouse","artifact_dir":"/data/mouse","species":"mouse"}'
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--species", default="human")
    parser.add_argument("--seed", default="brainwm-global-20260904")
    parser.add_argument(
        "--donor-scope",
        choices=("dataset", "corpus", "global"),
        default="dataset",
        help="identity scope for donor grouping; use global for donor-disjoint human benchmarks",
    )
    parser.add_argument(
        "--split-id",
        default=None,
        help="optional explicit split contract ID (otherwise selected from donor scope)",
    )
    parser.add_argument("--train-fraction", type=_parse_fraction, default=0.6)
    parser.add_argument("--validation-fraction", type=_parse_fraction, default=0.2)
    parser.add_argument("--test-fraction", type=_parse_fraction, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-source-audit",
        action="store_true",
        help="skip the bounded source cells.parquet count audit",
    )
    parser.add_argument("--source-batch-size", type=int, default=1_000_000)
    parser.add_argument("--validate", action="store_true", help="validate an existing output after building")
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="when validating, re-scan the source cells.parquet index",
    )
    args = parser.parse_args(argv)
    if args.artifact is None and not args.corpus_spec:
        parser.error("one of --artifact or --corpus-spec is required")
    parsed_specs = None
    if args.corpus_spec:
        try:
            parsed_specs = [json.loads(value) for value in args.corpus_spec]
        except json.JSONDecodeError as exc:
            parser.error(f"--corpus-spec must be valid JSON: {exc}")
    fractions = {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "test": args.test_fraction,
    }
    manifest = build_global_benchmark_split(
        args.artifact,
        args.output,
        species=args.species,
        seed=args.seed,
        fractions=fractions,
        overwrite=args.overwrite,
        audit_source=not args.skip_source_audit,
        source_batch_size=args.source_batch_size,
        donor_scope=args.donor_scope,
        split_id=args.split_id,
        corpus_specs=parsed_specs,
    )
    if args.validate:
        report = validate_global_benchmark_split(
            args.output,
            check_source=args.check_source,
            source_batch_size=args.source_batch_size,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "BenchmarkCorpusSpec",
    "CorpusSpec",
    "GlobalCorpusSpec",
    "GLOBAL_SPLIT_FORMAT",
    "GlobalBenchmarkIndex",
    "GlobalBenchmarkReader",
    "GlobalMultiCorpusReader",
    "assign_group_splits",
    "build_global_benchmark_split",
    "derive_split_groups",
    "qualify_row_id",
    "qualify_sample_key",
    "qualify_source_id",
    "validate_global_benchmark_split",
]
