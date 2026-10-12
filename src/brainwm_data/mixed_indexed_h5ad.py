"""Composite, zero-copy readers for multiple indexed H5AD corpora.

The historical :class:`IndexedH5ADBatchReader` deliberately assumes that one
artifact owns one source-id namespace.  This module adds the small amount of
bookkeeping needed to train over more than one artifact without changing that
reader (or copying any expression matrix).  Every corpus keeps its local row
indices and source metadata; only the in-memory index views receive global
source/sample identifiers.

``MixedIndexedH5ADBatchReader`` is intentionally a data-layer component.  It
does not decide how corpora are sampled, how metadata is encoded, or which
losses are enabled.  A caller can therefore pass its ``sources`` and ``cells``
views to the existing ``IndexedBatchPlan`` and use ``read_batch`` unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .indexed_h5ad import IndexedH5ADBatchReader
from .mouse_molecular_index import MouseMolecularBatchReader
from .mouse_spatial_index import MouseSpatialBatchReader

MIXED_READER_FORMAT = "brainwm_mixed_indexed_h5ad_reader_v1"


@dataclass(frozen=True)
class CorpusSpec:
    """Description of one indexed corpus.

    ``species`` is explicit rather than inferred from a path.  This keeps a
    typo in an artifact name from silently changing the model's species ID.
    ``reader_kind`` may be omitted for the built-in ``mouse_molecular``
    artifact; all other values use the existing human/general reader.
    """

    name: str
    artifact_dir: str | Path
    species: str
    reader_kind: str = "indexed_h5ad"
    source_id_offset: int | None = None
    sample_key_prefix: str | None = None
    source_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        species = str(self.species).strip()
        if not name:
            raise ValueError("corpus name must be non-empty")
        if not species:
            raise ValueError("corpus species must be non-empty")
        if self.source_id_offset is not None and int(self.source_id_offset) < 0:
            raise ValueError("source_id_offset must be nonnegative")
        if self.sample_key_prefix is not None and not str(self.sample_key_prefix).strip():
            raise ValueError("sample_key_prefix must be non-empty when provided")
        normalized = IndexedH5ADBatchReader._normalize_source_ids(self.source_ids)
        object.__setattr__(self, "source_ids", normalized)


def _coerce_specs(
    corpora: Mapping[str, str | Path | CorpusSpec] | Sequence[CorpusSpec],
) -> list[CorpusSpec]:
    if isinstance(corpora, Mapping):
        values: list[CorpusSpec] = []
        for name, value in corpora.items():
            if isinstance(value, CorpusSpec):
                if value.name != str(name):
                    raise ValueError(
                        f"mapping key {name!r} disagrees with CorpusSpec.name={value.name!r}"
                    )
                values.append(value)
            else:
                # The two standard names are enough to select the specialized
                # mouse reader.  Callers with another name can use CorpusSpec
                # and choose reader_kind explicitly.
                lowered = str(name).lower()
                species = "mouse" if lowered.startswith("mouse") else "human"
                kind = (
                    "mouse_spatial" if "spatial" in lowered
                    else "mouse_molecular" if species == "mouse"
                    else "indexed_h5ad"
                )
                values.append(CorpusSpec(str(name), value, species, kind))
        return values
    values = list(corpora)
    if not values:
        raise ValueError("at least one corpus is required")
    if not all(isinstance(value, CorpusSpec) for value in values):
        raise TypeError("sequence corpora entries must be CorpusSpec values")
    return values


def _global_sample_key(corpus: CorpusSpec, value: Any) -> str:
    """Namespace one local sample key while preserving existing mouse keys."""
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if value is None or missing:
        raise ValueError(f"{corpus.name} contains a null sample_key")
    local = str(value)
    prefix = str(corpus.sample_key_prefix or corpus.name).strip()
    # Mouse molecular artifacts already use ``mouse::...`` keys.  Keeping the
    # key stable avoids a needless second prefix and makes artifact joins
    # deterministic.  A key from any other corpus is always namespaced.
    if local.startswith(prefix + "::"):
        return local
    return f"{prefix}::{local}"


def _as_int_source_ids(frame: pd.DataFrame, *, name: str) -> np.ndarray:
    if "source_id" not in frame:
        raise ValueError(f"{name} is missing source_id")
    values = pd.to_numeric(frame["source_id"], errors="raise").to_numpy()
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name}.source_id must contain integer values")
    return values.astype(np.int64, copy=False)


def _validate_local_index(frame: pd.DataFrame, *, name: str) -> None:
    ids = _as_int_source_ids(frame, name=name)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}.source_id must be unique")
    if np.any(ids < 0):
        raise ValueError(f"{name}.source_id must be nonnegative")


def _is_source_row_sorted(frame: pd.DataFrame) -> bool:
    """Check source/row order without allocating a sorted copy."""
    if len(frame) < 2:
        return True
    source = pd.to_numeric(frame["source_id"], errors="raise").to_numpy(
        np.int64, copy=False
    )
    rows = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(
        np.int64, copy=False
    )
    source_delta = source[1:] - source[:-1]
    return bool(
        np.all((source_delta > 0) | ((source_delta == 0) & (rows[1:] >= rows[:-1])))
    )


def _concat_cell_parts(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate source-local cell frames, sorting only when necessary.

    Indexed artifacts are written in ``source_id,row_index`` order.  The
    historical unconditional sort duplicated the complete mixed table and
    dominated startup memory for tens of millions of cells.  We retain the
    old sorted result for irregular/custom readers, but take the zero-copy
    ordering path when each part and the part boundaries are already ordered.
    """
    if not parts:
        return pd.DataFrame()
    ordered = True
    previous_last: tuple[int, int] | None = None
    for part in parts:
        if not _is_source_row_sorted(part):
            ordered = False
        if len(part):
            first = (
                int(part["source_id"].iloc[0]),
                int(part["row_index"].iloc[0]),
            )
            last = (
                int(part["source_id"].iloc[-1]),
                int(part["row_index"].iloc[-1]),
            )
            if previous_last is not None and first < previous_last:
                ordered = False
            previous_last = last
    cells = pd.concat(parts, ignore_index=True, sort=False)
    if ordered:
        return cells
    return cells.sort_values(
        ["source_id", "row_index"], kind="mergesort"
    ).reset_index(drop=True)


class MultiCorpus:
    """Composite reader over source-local indexed H5AD artifacts.

    The class owns one existing reader per corpus.  Its combined metadata
    frames are views for planning and joins; expression remains in the source
    H5AD files and is opened lazily by those readers.
    """

    def __init__(
        self,
        corpora: Mapping[str, str | Path | CorpusSpec] | Sequence[CorpusSpec],
        *,
        unified_vocabulary: str | Path | None = None,
        allow_unmapped_unified_genes: bool = False,
        cell_columns: Sequence[str] | None = None,
        assay_types: Sequence[str] | None = None,
        sample_splits: Sequence[str] | None = None,
    ) -> None:
        specs = _coerce_specs(corpora)
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("corpus names must be unique")
        self.specs = tuple(specs)
        self.unified_vocabulary = (
            Path(unified_vocabulary).resolve()
            if unified_vocabulary is not None else None
        )
        self.allow_unmapped_unified_genes = bool(allow_unmapped_unified_genes)
        # Keep the projection optional for backwards compatibility.  A
        # shared projection is particularly useful for mixed human/mouse
        # Stage-0 runs because the two artifacts expose different optional
        # annotation columns.
        self.cell_columns = IndexedH5ADBatchReader._normalize_cell_columns(
            cell_columns
        )
        self.assay_types = IndexedH5ADBatchReader._normalize_filter_values(
            assay_types, name="assay_types"
        )
        self.sample_splits = IndexedH5ADBatchReader._normalize_filter_values(
            sample_splits, name="sample_splits"
        )
        self.readers: dict[str, Any] = {}
        self._sample_tables: dict[str, pd.DataFrame] = {}
        self._source_lookup: dict[int, tuple[str, int]] = {}
        self._closed = False

        # Construct readers before combining metadata so malformed artifacts
        # fail before any composite state is exposed to a caller.
        for spec in self.specs:
            artifact = Path(spec.artifact_dir).resolve()
            cell_sample_keys = None
            if self.assay_types is not None or self.sample_splits is not None:
                samples_path = artifact / "samples.parquet"
                if not samples_path.is_file():
                    raise FileNotFoundError(
                        f"sample predicate pushdown requires {samples_path}"
                    )
                samples = pd.read_parquet(samples_path)
                self._sample_tables[spec.name] = samples
                selected = pd.Series(True, index=samples.index)
                if self.assay_types is not None:
                    assay_column = (
                        "assay_type" if "assay_type" in samples
                        else "assay" if "assay" in samples
                        else None
                    )
                    if assay_column is None:
                        raise ValueError(
                            f"{spec.name}.samples lacks assay_type/assay for filtering"
                        )
                    selected &= samples[assay_column].astype("string").isin(
                        self.assay_types
                    )
                if self.sample_splits is not None:
                    if "split" not in samples:
                        raise ValueError(
                            f"{spec.name}.samples lacks split for filtering"
                        )
                    selected &= samples["split"].astype("string").isin(
                        self.sample_splits
                    )
                if "sample_key" not in samples:
                    raise ValueError(f"{spec.name}.samples is missing sample_key")
                cell_sample_keys = tuple(
                    samples.loc[selected, "sample_key"].dropna().astype(str).unique()
                )
                if not cell_sample_keys:
                    raise ValueError(
                        f"{spec.name} has no samples matching assay/split filters"
                    )
            if spec.reader_kind == "mouse_molecular":
                reader = MouseMolecularBatchReader(
                    artifact,
                    cell_columns=self.cell_columns,
                    cell_sample_keys=cell_sample_keys,
                    cell_splits=self.sample_splits,
                    cell_source_ids=spec.source_ids,
                    unified_vocabulary=self.unified_vocabulary,
                    allow_unmapped_unified_genes=self.allow_unmapped_unified_genes,
                )
            elif spec.reader_kind == "mouse_spatial":
                reader = MouseSpatialBatchReader(
                    artifact,
                    cell_columns=self.cell_columns,
                    cell_sample_keys=cell_sample_keys,
                    cell_splits=self.sample_splits,
                    cell_source_ids=spec.source_ids,
                    unified_vocabulary=self.unified_vocabulary,
                    allow_unmapped_unified_genes=self.allow_unmapped_unified_genes,
                )
            elif spec.reader_kind == "indexed_h5ad":
                reader = IndexedH5ADBatchReader(
                    artifact,
                    cell_columns=self.cell_columns,
                    cell_sample_keys=cell_sample_keys,
                    cell_splits=self.sample_splits,
                    cell_source_ids=spec.source_ids,
                )
            else:
                raise ValueError(
                    f"unsupported reader_kind={spec.reader_kind!r} for corpus {spec.name!r}"
                )
            self.readers[spec.name] = reader

        try:
            self.sources, self.cells, self.samples = self._combine_metadata()
            self.n_genes = self._infer_gene_count()
        except Exception:
            self.close()
            raise

    @classmethod
    def human_mouse(
        cls,
        human_artifact: str | Path,
        mouse_artifact: str | Path,
        *,
        unified_vocabulary: str | Path,
        allow_unmapped_unified_genes: bool = False,
        cell_columns: Sequence[str] | None = None,
        assay_types: Sequence[str] | None = None,
        sample_splits: Sequence[str] | None = None,
    ) -> MultiCorpus:
        """Construct the standard human + mouse molecular composite."""
        return cls(
            (
                CorpusSpec("human", human_artifact, "human", "indexed_h5ad"),
                CorpusSpec("mouse", mouse_artifact, "mouse", "mouse_molecular"),
            ),
            unified_vocabulary=unified_vocabulary,
            allow_unmapped_unified_genes=allow_unmapped_unified_genes,
            cell_columns=cell_columns,
            assay_types=assay_types,
            sample_splits=sample_splits,
        )

    def _species_id(self, spec: CorpusSpec, source_frame: pd.DataFrame) -> int:
        values = source_frame.get("species_id")
        if values is not None:
            unique = pd.to_numeric(values, errors="raise").dropna().unique()
            if len(unique) > 1:
                raise ValueError(f"{spec.name} has multiple species_id values")
            if len(unique) == 1:
                value = int(unique[0])
                if value < 0:
                    raise ValueError(f"{spec.name} species_id must be nonnegative")
                return value
        # Keep the canonical IDs stable for the built-in human/mouse pair.
        if spec.species == "human":
            return 0
        if spec.species == "mouse":
            return 1
        raise ValueError(
            f"corpus {spec.name!r} lacks explicit species_id for species {spec.species!r}"
        )

    def _source_offsets(self) -> dict[str, int]:
        offsets: dict[str, int] = {}
        used: set[int] = set()
        next_offset = 0
        # Explicit offsets are honored first, then implicit offsets are
        # assigned in declaration order.  This keeps human source IDs exactly
        # equal to their historical IDs when it is listed first with offset 0.
        for spec in self.specs:
            frame = self.readers[spec.name].sources.reset_index()
            local_ids = _as_int_source_ids(frame, name=f"{spec.name}.sources")
            requested = spec.source_id_offset
            if requested is None:
                requested = next_offset
                while any((int(value) + requested) in used for value in local_ids):
                    requested += 1
            requested = int(requested)
            global_ids = {int(value) + requested for value in local_ids}
            if global_ids & used:
                overlap = sorted(global_ids & used)[:5]
                raise ValueError(
                    f"source ID collision in corpus {spec.name!r}: {overlap}"
                )
            offsets[spec.name] = requested
            used.update(global_ids)
            next_offset = max(used, default=-1) + 1
        return offsets

    @staticmethod
    def _ensure_column(frame: pd.DataFrame, column: str, value: Any) -> None:
        if column not in frame:
            frame[column] = value

    def _combine_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        offsets = self._source_offsets()
        source_parts: list[pd.DataFrame] = []
        cell_parts: list[pd.DataFrame] = []
        sample_parts: list[pd.DataFrame] = []
        used_sample_keys: set[str] = set()

        for spec in self.specs:
            reader = self.readers[spec.name]
            source = reader.sources.reset_index().copy()
            cells = reader.cells.copy()
            samples_path = Path(reader.root) / "samples.parquet"
            cached_samples = self._sample_tables.get(spec.name)
            samples = (
                cached_samples.copy()
                if cached_samples is not None
                else pd.read_parquet(samples_path).copy()
                if samples_path.is_file()
                else pd.DataFrame()
            )
            _validate_local_index(source, name=f"{spec.name}.sources")
            required_cells = {"source_id", "row_index", "sample_key", "split"}
            missing = sorted(required_cells - set(cells.columns))
            if missing:
                raise ValueError(f"{spec.name}.cells is missing columns: {missing}")
            if not samples.empty:
                observed_sample_keys = set(cells["sample_key"].dropna().astype(str))
                samples = samples[
                    samples["sample_key"].astype(str).isin(observed_sample_keys)
                ].copy()
            local_source_ids = _as_int_source_ids(cells, name=f"{spec.name}.cells")
            if np.any(local_source_ids < 0):
                raise ValueError(f"{spec.name}.cells.source_id must be nonnegative")
            if cells.duplicated(["source_id", "row_index"]).any():
                raise ValueError(
                    f"{spec.name}.cells contains duplicate source-local rows"
                )

            # Sample tables carry the authoritative context for human and
            # mixed-source artifacts.  Build local-key lookups before adding
            # source fallbacks so a missing cell column can inherit values
            # such as assay, platform, dataset, or time without materializing
            # another copy of the expression matrix.
            sample_context: dict[str, pd.Series] = {}
            if not samples.empty and "sample_key" in samples:
                raw_sample_keys = samples["sample_key"].astype("string")
                sample_context["assay_type"] = (
                    samples["assay_type"]
                    if "assay_type" in samples
                    else samples["assay"]
                    if "assay" in samples
                    else pd.Series(pd.NA, index=samples.index)
                )
                for field in (
                    "technology", "dataset", "time_point_days",
                    "time_lower_days", "time_upper_days", "time_basis",
                    "time_status", "life_stage", "age_raw", "timepoint_raw",
                    "developmental_stage_raw", "uberon_id", "uberon_term",
                    "original_anatomical_region", "anatomical_region",
                    "anatomical_layer", "brain_region", "brain_area", "region",
                    "location", "layer", "cortical_layer",
                ):
                    if field in samples:
                        sample_context[field] = samples[field]
                sample_context = {
                    field: pd.Series(values.to_numpy(copy=False), index=raw_sample_keys)
                    for field, values in sample_context.items()
                }
            source_id_map = {
                int(local): int(local) + offsets[spec.name]
                for local in source["source_id"].to_numpy(np.int64)
            }
            unknown_sources = sorted(set(local_source_ids) - set(source_id_map))
            if unknown_sources:
                raise ValueError(
                    f"{spec.name}.cells references unknown source IDs: {unknown_sources[:5]}"
                )

            species_id = self._species_id(spec, source)
            global_source = cells["source_id"].map(source_id_map).astype(np.int64)
            local_sample = cells["sample_key"].astype("string")
            local_sample_values = local_sample.dropna().astype(str).unique()
            sample_key_map = {
                value: _global_sample_key(spec, value)
                for value in local_sample_values
            }
            global_sample = local_sample.map(sample_key_map)
            if global_sample.isna().any():
                raise ValueError(f"{spec.name}.cells contains a null sample_key")
            corpus_sample_keys = set(global_sample.dropna().astype(str).unique())
            if corpus_sample_keys & used_sample_keys:
                overlap = sorted(corpus_sample_keys & used_sample_keys)[:5]
                raise ValueError(f"global sample key collision: {overlap}")

            source["local_source_id"] = source["source_id"].astype(np.int64)
            source["source_id"] = source["source_id"].map(source_id_map).astype(np.int64)
            source["corpus"] = spec.name
            source["species"] = spec.species
            source["species_id"] = species_id
            self._ensure_column(source, "assay_type", None)
            if "assay" in source and source["assay_type"].isna().all():
                source["assay_type"] = source["assay"]
            elif "assay_types" in source and source["assay_type"].isna().all():
                # The human index stores one or more assays as an array per
                # source.  Expose a scalar only when the source is unambiguous;
                # mixed sources retain an explicit missing value and can be
                # resolved at cell level below.
                def _single_assay(value: Any) -> str | None:
                    if value is None:
                        return None
                    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                        values = [str(item) for item in value if not pd.isna(item)]
                    else:
                        values = [str(value)] if not pd.isna(value) else []
                    return values[0] if len(set(values)) == 1 else None

                source["assay_type"] = source["assay_types"].map(_single_assay)
            self._ensure_column(source, "technology", None)
            if source["technology"].isna().all() and "platform" in source:
                source["technology"] = source["platform"]
            self._ensure_column(source, "dataset", spec.name)
            source["dataset"] = source["dataset"].fillna(spec.name).astype(str)
            source_parts.append(source)

            cells["local_source_id"] = cells["source_id"].astype(np.int64)
            cells["source_id"] = global_source
            cells["local_sample_key"] = local_sample.astype(object)
            cells["sample_key"] = global_sample.astype(object)
            cells["corpus"] = spec.name
            self._ensure_column(cells, "species", spec.species)
            cells["species"] = cells["species"].fillna(spec.species).astype(str)
            self._ensure_column(cells, "species_id", species_id)
            cells["species_id"] = pd.to_numeric(
                cells["species_id"], errors="coerce"
            ).fillna(species_id).astype(np.int16)
            self._ensure_column(cells, "assay_type", None)
            if "assay" in cells and cells["assay_type"].isna().all():
                cells["assay_type"] = cells["assay"]
            elif (
                spec.reader_kind == "mouse_spatial"
                and cells["assay_type"].isna().all()
            ):
                cells["assay_type"] = "spatialRNA"
            # Human cells do not carry sample-level context.  Coalesce every
            # available sample field into the cell view while retaining any
            # explicit cell annotation as the higher-priority value.
            for field, values in sample_context.items():
                mapped = local_sample.map(values)
                self._ensure_column(cells, field, None)
                cells[field] = cells[field].where(cells[field].notna(), mapped)
            # Normalize source/sample context names used by MetadataTensorizer.
            self._ensure_column(cells, "technology", None)
            source_technology = {
                str(local): value
                for local, value in zip(
                    source["local_source_id"], source["technology"]
                )
                if pd.notna(value)
            }
            cells["technology"] = cells["technology"].fillna(
                cells["local_source_id"].astype(str).map(source_technology)
            )
            self._ensure_column(cells, "dataset", None)
            source_dataset = {
                str(local): value
                for local, value in zip(
                    source["local_source_id"], source["dataset"]
                )
            }
            cells["dataset"] = cells["dataset"].fillna(
                cells["local_source_id"].astype(str).map(source_dataset)
            ).fillna(spec.name).astype(str)
            # Keep a stable set of spatial and temporal columns across
            # molecular, spatial, and human artifacts. Missing values remain
            # missing; age/time is never inferred from a filename.
            for column in ("spatial_x", "spatial_y", "spatial_z"):
                self._ensure_column(cells, column, np.nan)
            for column in ("time_point_days", "time_lower_days", "time_upper_days"):
                self._ensure_column(cells, column, np.nan)
            for column in ("time_basis", "time_status", "life_stage"):
                self._ensure_column(cells, column, None)
            cell_parts.append(cells)
            used_sample_keys.update(corpus_sample_keys)

            if samples.empty:
                # A samples table is useful but not required by the low-level
                # reader.  Synthesize one row per observed key so downstream
                # joins remain one-to-one and explicit missing fields survive.
                samples = pd.DataFrame({"sample_key": sorted(set(local_sample))})
            if "sample_key" not in samples:
                raise ValueError(f"{spec.name}.samples is missing sample_key")
            samples["local_sample_key"] = samples["sample_key"].astype(object)
            samples["sample_key"] = samples["sample_key"].map(
                lambda value, corpus=spec: _global_sample_key(corpus, value)
            )
            samples["corpus"] = spec.name
            samples["species"] = spec.species
            samples["species_id"] = species_id
            if "assay" in samples:
                self._ensure_column(samples, "assay_type", samples["assay"])
            else:
                self._ensure_column(samples, "assay_type", None)
            if (
                spec.reader_kind == "mouse_spatial"
                and samples["assay_type"].isna().all()
            ):
                samples["assay_type"] = "spatialRNA"
            self._ensure_column(samples, "technology", None)
            if samples["technology"].isna().all() and "platform" in samples:
                samples["technology"] = samples["platform"]
            self._ensure_column(samples, "dataset", spec.name)
            samples["dataset"] = samples["dataset"].fillna(spec.name).astype(str)
            for column in ("time_point_days", "time_lower_days", "time_upper_days"):
                self._ensure_column(samples, column, np.nan)
            for column in ("time_basis", "time_status", "life_stage"):
                self._ensure_column(samples, column, None)
            sample_parts.append(samples)

            for local, global_id in source_id_map.items():
                self._source_lookup[global_id] = (spec.name, local)
            # ``global_sample`` is already the vectorized local->global map
            # used by the cell frame above.  Do not build a redundant
            # per-cell Python dictionary here; it was never consumed by the
            # reader and made full-corpus construction quadratic in memory.

        sources = pd.concat(source_parts, ignore_index=True, sort=False)
        sources = sources.sort_values("source_id", kind="mergesort").reset_index(drop=True)
        if sources["source_id"].duplicated().any():
            raise ValueError("combined source IDs are not unique")
        sources = sources.set_index("source_id", drop=False)
        cells = _concat_cell_parts(cell_parts)
        samples = pd.concat(sample_parts, ignore_index=True, sort=False)
        samples = samples.sort_values("sample_key", kind="mergesort").reset_index(drop=True)
        if samples["sample_key"].duplicated().any():
            raise ValueError("combined sample keys are not unique")
        return sources, cells, samples

    def _infer_gene_count(self) -> int:
        if self.unified_vocabulary is not None:
            path = self.unified_vocabulary / "genes.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            genes = pd.read_parquet(path, columns=["gene_index"])
            indices = pd.to_numeric(genes["gene_index"], errors="raise").to_numpy(np.int64)
            if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
                raise ValueError("unified gene_index must be contiguous")
            return len(indices)
        maximum = -1
        for reader in self.readers.values():
            mapping = reader.gene_map
            column = "gene_index" if "gene_index" in mapping else "namespace_gene_index"
            values = pd.to_numeric(mapping[column], errors="coerce")
            if values.notna().any():
                maximum = max(maximum, int(values.max()))
        if maximum < 0:
            raise ValueError("could not infer a non-empty gene vocabulary")
        return maximum + 1

    @property
    def source_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.sources["source_id"].to_numpy())

    def cells_for_split(self, split: str) -> pd.DataFrame:
        return self.cells[self.cells["split"].astype(str) == str(split)].copy()

    def _resolve_source(self, source_id: int) -> tuple[str, int]:
        try:
            return self._source_lookup[int(source_id)]
        except KeyError as exc:
            raise KeyError(f"unknown global source_id={source_id}") from exc

    def read_batch(
        self,
        source_id: int,
        row_indices: np.ndarray,
        cell_frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Route a global source read to its source-local reader.

        ``row_indices`` remain H5AD-local by design.  The optional ``cell_frame``
        is expected to use the composite global source/sample IDs; it is
        passed through unchanged after the underlying reader has read counts.
        """
        corpus, local_source = self._resolve_source(int(source_id))
        reader = self.readers[corpus]
        frame = cell_frame
        if frame is not None:
            if "source_id" in frame:
                source_values = pd.to_numeric(frame["source_id"], errors="raise")
                if not np.all(source_values.astype(np.int64) == int(source_id)):
                    raise ValueError("cell_frame contains a different global source_id")
            if "row_index" in frame:
                requested = np.asarray(row_indices, dtype=np.int64)
                listed = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(np.int64)
                if not np.array_equal(listed, requested):
                    raise ValueError("cell_frame row_index does not match row_indices")
        batch = reader.read_batch(
            local_source,
            np.asarray(row_indices, dtype=np.int64),
            frame,
        )
        source = self.sources.loc[int(source_id)]
        batch["source_id"] = int(source_id)
        batch["local_source_id"] = int(local_source)
        batch["corpus"] = corpus
        batch["species"] = str(source["species"])
        batch["species_id"] = int(source["species_id"])
        if "assay_type" not in batch:
            assay = source.get("assay_type")
            if pd.notna(assay) and not isinstance(assay, (list, tuple, np.ndarray)):
                batch["assay_type"] = str(assay)
        # A source may contain multiple assays.  In that case the exact
        # per-cell value is available in the composite frame; only expose it
        # when this batch is homogeneous.
        if (
            "assay_type" not in batch
            and frame is not None
            and "assay_type" in frame
        ):
            values = frame["assay_type"].dropna().astype(str).unique()
            if len(values) == 1:
                batch["assay_type"] = str(values[0])
        # Human readers already return global gene indices.  Mouse readers with
        # a unified adapter expose both the mapped IDs and local namespace IDs.
        batch["unified_gene_index"] = np.asarray(
            batch["gene_index"], dtype=np.int32
        )
        return batch

    def iter_local_batches(
        self,
        split: str,
        batch_size: int = 256,
        shuffle_sources: bool = True,
        seed: int = 0,
    ):
        """Yield the same batch shape as ``IndexedH5ADBatchReader``."""
        frame = self.cells_for_split(split)
        source_ids = frame["source_id"].drop_duplicates().to_numpy(np.int64)
        rng = np.random.default_rng(seed)
        if shuffle_sources:
            rng.shuffle(source_ids)
        for source_id in source_ids:
            part = frame[frame["source_id"] == source_id].sort_values("row_index")
            for start in range(0, len(part), batch_size):
                selected = part.iloc[start:start + batch_size]
                yield self.read_batch(
                    int(source_id),
                    selected["row_index"].to_numpy(np.int64),
                    selected,
                )

    def close(self) -> None:
        if self._closed:
            return
        for reader in self.readers.values():
            reader.close()
        self._closed = True

    def __enter__(self) -> MultiCorpus:  # noqa: PYI034
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110
            pass


class MixedIndexedH5ADBatchReader(MultiCorpus):
    """Descriptive alias for the human/mouse-capable composite reader."""


# Keep a discoverable name for callers that prefer the longer form.
MultiCorpusIndexedH5ADBatchReader = MixedIndexedH5ADBatchReader


__all__ = [
    "MIXED_READER_FORMAT",
    "CorpusSpec",
    "MixedIndexedH5ADBatchReader",
    "MultiCorpus",
    "MultiCorpusIndexedH5ADBatchReader",
]
