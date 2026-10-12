"""Species-aware zero-copy cell index for the audited mouse spatial corpus.

The index stores observation metadata and source row numbers only.  Expression
counts remain in the original H5AD files and are read through the existing
CSR/dense backed reader.  ``namespace_gene_index`` is deliberately local to
the mouse namespace artifact; it is never treated as the human model's gene
index.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .h5ad_io import (
    dataframe_index_name,
    file_fingerprint,
    matrix_encoding,
    matrix_shape,
    read_column,
)
from .indexed_h5ad import IndexedH5ADBatchReader
from .unified_gene_index import UnifiedGeneIndexAdapter

MOUSE_SPATIAL_INDEX_FORMAT = "brainwm_mouse_spatial_cell_index_v1"
SPLIT_UNIT = "section"
CELL_COLUMNS = [
    "source_id",
    "row_index",
    "species",
    "species_id",
    "dataset",
    "sample_key",
    "sample_id",
    "section_label",
    "cell_id",
    "split",
    "split_unit",
    "spatial_x",
    "spatial_y",
    "spatial_z",
    "cl_id",
    "cl_term",
    "original_cell_type",
    "original_subclass",
    "original_anatomical_region",
    "uberon_id",
    "uberon_term",
    "cell_label",
]
LABEL_COLUMNS = [
    "cl_id",
    "cl_term",
    "original_cell_type",
    "original_subclass",
    "original_anatomical_region",
    "uberon_id",
    "uberon_term",
    "cell_label",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}|{value}".encode()).digest()


def _string_or_none(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _string_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([_string_or_none(value) for value in values], dtype=object)


def _read_optional(obs: h5py.Group, name: str, indices: np.ndarray) -> np.ndarray:
    if name not in obs:
        return np.asarray([None] * len(indices), dtype=object)
    return _string_array(read_column(obs, name, indices))


def assign_section_splits(
    section_keys: Iterable[str],
    *,
    seed: str = "20260825",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign deterministic section-held-out splits within one source.

    The split unit is intentionally a section, not a donor: the audit found no
    donor/animal identifier.  Small sources receive the strongest split that
    their section count supports (three sections yield one train/validation/
    test section each; one or two sections remain explicitly limited).
    """
    keys = sorted({str(key) for key in section_keys})
    if not keys:
        raise ValueError("cannot split an empty section set")
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must lie in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must leave train sections")
    n = len(keys)
    if n == 1:
        return {keys[0]: "train"}
    n_test = max(1, round(n * test_fraction))
    n_validation = max(1, round(n * validation_fraction))
    if n_test + n_validation >= n:
        n_test = 1
        n_validation = 1 if n >= 3 else 0
    ranked = sorted(keys, key=lambda key: _stable_rank(seed, key))
    assignment = {key: "train" for key in keys}
    for key in ranked[:n_test]:
        assignment[key] = "test"
    for key in ranked[n_test : n_test + n_validation]:
        assignment[key] = "validation"
    return assignment


def _select_sections(all_sections: Iterable[str], limit: int | None) -> list[str]:
    sections = sorted({str(value) for value in all_sections})
    if limit is not None:
        if limit < 1:
            raise ValueError("max_sections_per_source must be positive")
        sections = sections[:limit]
    if not sections:
        raise ValueError("source has no non-empty brain sections")
    return sections


def _write_parquet_chunks(path: Path, chunks: Iterable[pd.DataFrame]) -> int:
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for frame in chunks:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame[CELL_COLUMNS], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
            rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("no cell rows were written")
    return rows


def _source_inventory(
    path: Path,
    source_id: int,
    namespace_source: pd.Series,
    *,
    max_sections: int | None,
    split_seed: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read only obs section labels to establish the section inventory."""
    with h5py.File(path, "r") as handle:
        expression = handle[str(namespace_source["expression_element"])]
        n_obs, n_vars = matrix_shape(expression)
        if "obs" not in handle or "brain_section_label" not in handle["obs"]:
            raise KeyError(f"{path} lacks required obs.brain_section_label")
        sections = _string_array(read_column(handle["obs"], "brain_section_label"))
        if len(sections) != n_obs:
            raise ValueError(f"{path} section labels do not match matrix rows")
        valid = np.asarray([value is not None and value != "" for value in sections])
        if not valid.all():
            raise ValueError(f"{path} has missing brain_section_label values")
        selected = _select_sections(sections, max_sections)
        selected_mask = np.isin(sections, np.asarray(selected, dtype=object))
        counts = Counter(sections[selected_mask].tolist())
        split = assign_section_splits(
            [f"{namespace_source['dataset']}::{section}" for section in selected],
            seed=split_seed,
        )
        section_rows = []
        for section in selected:
            key = f"{namespace_source['dataset']}::{section}"
            section_rows.append(
                {
                    "source_id": int(source_id),
                    "species": "mouse",
                    "species_id": 1,
                    "dataset": str(namespace_source["dataset"]),
                    "sample_key": f"mouse::{key}",
                    "section_label": section,
                    "split": split[key],
                    "split_unit": SPLIT_UNIT,
                    "n_observations": int(counts[section]),
                    "selected_cells": int(counts[section]),
                    "donor_id": None,
                    "donor_status": "missing_external_donor_field",
                    "platform": _string_or_none(namespace_source.get("platform")),
                    "platform_status": _string_or_none(
                        namespace_source.get("platform_status")
                    ),
                    "paper_provenance_status": "pending_external_sidecar_doi",
                    "provenance_status": "pending_external_sidecar",
                }
            )
        source = namespace_source.to_dict()
        source.update(
            {
                "source_id": int(source_id),
                "species": "mouse",
                "species_id": 1,
                "selected_cells": int(selected_mask.sum()),
                "selected_sections": len(selected),
                "split_unit": SPLIT_UNIT,
                "donor_status": "missing_external_donor_field",
                "provenance_status": "pending_external_sidecar_doi",
            }
        )
        inventory = {
            "path": path,
            "source_id": int(source_id),
            "sections": sections,
            "selected_sections": selected,
            "selected_mask": selected_mask,
            "split": {
                section: split[f"{namespace_source['dataset']}::{section}"]
                for section in selected
            },
            "n_obs": n_obs,
            "n_vars": n_vars,
            "matrix_encoding": matrix_encoding(expression),
            "has_spatial": "obsm" in handle and "spatial" in handle["obsm"],
        }
    if not inventory["has_spatial"]:
        raise ValueError(f"{path} lacks obsm/spatial")
    return source, section_rows, inventory


def _cell_chunks(
    inventory: dict[str, Any],
    source: pd.Series,
    section_rows: list[dict[str, Any]],
    *,
    chunk_size: int,
):
    path = inventory["path"]
    selected_mask = inventory["selected_mask"]
    selected_rows = np.flatnonzero(selected_mask).astype(np.int64)
    section_by_name = {row["section_label"]: row for row in section_rows}
    dataset = str(source["dataset"])
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_name = dataframe_index_name(obs)
        # ABCA-4 omits the dataframe ``_index`` attribute although its
        # supplied cell_label column is present and unique.  Use that actual
        # source field rather than inventing row-based identifiers.
        if index_name not in obs:
            if "cell_label" in obs:
                index_name = "cell_label"
            else:
                raise KeyError(f"{path} has no readable observation index column")
        for start in range(0, len(selected_rows), chunk_size):
            rows = selected_rows[start : start + chunk_size]
            section = _string_array(read_column(obs, "brain_section_label", rows))
            sample_id = _read_optional(obs, "sample_id", rows)
            # Sample identifiers are retained as supplied.  The section is
            # the split unit because the audit found no donor identifier.
            split = np.asarray(
                [section_by_name[value]["split"] for value in section], dtype=object
            )
            sample_key = np.asarray(
                [f"mouse::{dataset}::{value}" for value in section], dtype=object
            )
            coords = np.asarray(handle["obsm/spatial"][rows])
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(f"{path} obsm/spatial is not a 2D coordinate matrix")
            if not np.isfinite(coords[:, :2]).all():
                raise ValueError(f"{path} has non-finite spatial coordinates")
            frame: dict[str, Any] = {
                "source_id": np.full(
                    len(rows), int(inventory["source_id"]), dtype=np.int32
                ),
                "row_index": rows,
                "species": np.full(len(rows), "mouse", dtype=object),
                "species_id": np.full(len(rows), 1, dtype=np.int16),
                "dataset": np.full(len(rows), dataset, dtype=object),
                "sample_key": sample_key,
                "sample_id": sample_id,
                "section_label": section,
                "cell_id": _string_array(read_column(obs, index_name, rows)),
                "split": split,
                "split_unit": np.full(len(rows), SPLIT_UNIT, dtype=object),
                "spatial_x": coords[:, 0].astype(np.float32),
                "spatial_y": coords[:, 1].astype(np.float32),
                "spatial_z": (
                    coords[:, 2].astype(np.float32)
                    if coords.shape[1] > 2
                    else np.full(len(rows), np.nan, dtype=np.float32)
                ),
            }
            frame.update(
                {name: _read_optional(obs, name, rows) for name in LABEL_COLUMNS}
            )
            yield pd.DataFrame(frame, columns=CELL_COLUMNS), rows, section


def _schema() -> dict[str, Any]:
    return {
        "format": MOUSE_SPATIAL_INDEX_FORMAT,
        "scope": "mouse spatial metadata and zero-copy source row index",
        "species": {"name": "mouse", "species_id": 1, "unknown_policy": "error"},
        "split": {
            "unit": SPLIT_UNIT,
            "meaning": "section-held-out; donor is unavailable and never inferred",
            "donor_status": "missing_external_donor_field",
        },
        "expression": {
            "storage": "original H5AD raw/X only; no expression copy",
            "reader": "brainwm_data.mouse_spatial_index.MouseSpatialBatchReader",
            "encodings": ["csr_matrix", "array"],
            "token_index": "namespace_gene_index (mouse-local only)",
        },
        "files": {
            "cells.parquet": "source row numbers, section split, coordinates and labels",
            "samples.parquet": "one row per brain section with split/provenance status",
            "sources.parquet": "source H5AD identity and encoding metadata",
            "genes.parquet": "copied mouse namespace rows; local indices only",
            "source_gene_map.parquet": "source column to namespace_gene_index mapping",
            "species_vocabulary.json": "explicit mouse species identity",
            "manifest.json": "inputs, counts, hashes and validation status",
            "reflection.md": "build and smoke validation reflection",
        },
        "missing_features": {
            "homology_group": "pending ortholog source; all namespace values remain null",
            "esm2_embedding_index": "pending embedding artifact; all namespace values remain null",
            "paper_doi": "pending provenance sidecar",
            "donor_id": "pending external donor metadata",
        },
        "forbidden": [
            "treat mouse namespace_gene_index as human gene_index",
            "map mouse genes by symbol or case folding",
            "claim donor-held-out, temporal or perturbation supervision",
        ],
    }


def build_mouse_spatial_index(
    namespace_artifact: str | Path,
    output_dir: str | Path,
    *,
    max_sections_per_source: int | None = None,
    chunk_size: int = 100_000,
    split_seed: str = "20260825",
) -> dict[str, Any]:
    """Build an independent metadata index without copying expression values."""
    started = time.perf_counter()
    namespace = Path(namespace_artifact)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mouse spatial index: {output}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    namespace_manifest = json.loads((namespace / "manifest.json").read_text())
    if namespace_manifest.get("format") != "brainwm_species_gene_namespace_v1":
        raise ValueError("namespace artifact has an unexpected format")
    if (
        namespace_manifest.get("species") != "mouse"
        or int(namespace_manifest.get("species_id")) != 1
    ):
        raise ValueError("mouse index requires namespace species=mouse, species_id=1")
    namespace_sources = pd.read_parquet(namespace / "sources.parquet")
    if namespace_sources.empty:
        raise ValueError("namespace artifact has no sources")
    if not (namespace_sources["species"] == "mouse").all():
        raise ValueError("namespace sources contain a non-mouse species")
    source_gene_map = pd.read_parquet(namespace / "source_gene_map.parquet")
    if "namespace_gene_index" not in source_gene_map:
        raise ValueError("mouse namespace map lacks namespace_gene_index")

    inventories: list[tuple[pd.Series, list[dict[str, Any]], dict[str, Any]]] = []
    for source_id, namespace_source in namespace_sources.sort_values(
        "source_id"
    ).iterrows():
        source_id = int(namespace_source["source_id"])
        path = Path(str(namespace_source["local_path"])).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        actual_fingerprint = file_fingerprint(path)
        expected_fingerprint = str(namespace_source["file_fingerprint"])
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(f"source fingerprint changed since namespace scan: {path}")
        source, section_rows, inventory = _source_inventory(
            path,
            source_id,
            namespace_source,
            max_sections=max_sections_per_source,
            split_seed=split_seed,
        )
        inventories.append((pd.Series(source), section_rows, inventory))

    output.mkdir(parents=True, exist_ok=False)
    # Copy namespace tables into this independent artifact.  They are small
    # metadata tables; expression remains referenced by absolute source path.
    for name in ("genes.parquet", "source_gene_map.parquet", "species_vocabulary.json"):
        source_path = namespace / name
        target = output / name
        if source_path.suffix == ".parquet":
            pd.read_parquet(source_path).to_parquet(
                target, index=False, compression="zstd"
            )
        else:
            target.write_bytes(source_path.read_bytes())
    cells_path = output / "cells.parquet"
    source_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    selected_cells = 0
    cells_by_split: Counter[str] = Counter()
    cells_by_encoding: Counter[str] = Counter()

    def chunks():
        nonlocal selected_cells
        for source, sections, inventory in inventories:
            source_rows.append(source.to_dict())
            sample_rows.extend(sections)
            for frame, rows, section in _cell_chunks(
                inventory, source, sections, chunk_size=chunk_size
            ):
                selected_cells += len(frame)
                cells_by_split.update(frame["split"].astype(str).tolist())
                cells_by_encoding.update(
                    [str(inventory["matrix_encoding"])] * len(frame)
                )
                yield frame

    _write_parquet_chunks(cells_path, chunks())
    indexed_regions = int(
        pd.read_parquet(cells_path, columns=["original_anatomical_region"])[
            "original_anatomical_region"
        ]
        .dropna()
        .astype(str)
        .nunique()
    )
    samples = pd.DataFrame(sample_rows)
    sources = pd.DataFrame(source_rows)
    # Keep source IDs and section counts explicit for backed readers and plans.
    samples.to_parquet(output / "samples.parquet", index=False, compression="zstd")
    sources.to_parquet(output / "sources.parquet", index=False, compression="zstd")
    schema = _schema()
    (output / "SCHEMA.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    namespace_hashes = {
        name: _sha256(namespace / name)
        for name in (
            "genes.parquet",
            "source_gene_map.parquet",
            "species_vocabulary.json",
        )
    }
    index_files = [
        output / "cells.parquet",
        output / "samples.parquet",
        output / "sources.parquet",
        output / "genes.parquet",
        output / "source_gene_map.parquet",
        output / "species_vocabulary.json",
        output / "SCHEMA.json",
    ]
    artifact_hashes = {path.name: _sha256(path) for path in index_files}
    index_digest = hashlib.sha256(
        "\n".join(
            f"{name}:{artifact_hashes[name]}" for name in sorted(artifact_hashes)
        ).encode()
    ).hexdigest()
    elapsed = time.perf_counter() - started
    manifest = {
        "format": MOUSE_SPATIAL_INDEX_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "namespace_artifact": str(namespace.resolve()),
        "namespace_manifest_sha256": _sha256(namespace / "manifest.json"),
        "species": "mouse",
        "species_id": 1,
        "human_gene_vocabulary_modified": False,
        "expression_copy_bytes": 0,
        "scan": {
            "max_sections_per_source": max_sections_per_source,
            "chunk_size": chunk_size,
            "split_seed": split_seed,
            "split_unit": SPLIT_UNIT,
            "source_rows_read": int(selected_cells),
        },
        "stats": {
            "sources": len(sources),
            "cells": int(selected_cells),
            "sections": len(samples),
            "genes": len(pd.read_parquet(output / "genes.parquet")),
            "anatomical_regions": indexed_regions,
            "by_split": {str(key): int(value) for key, value in cells_by_split.items()},
            "by_matrix_encoding": {
                str(key): int(value) for key, value in cells_by_encoding.items()
            },
            "has_spatial": bool(sources["has_spatial"].all()),
            "stage2_ready": bool(sources["has_spatial"].all() and len(samples) >= 3),
            "stage1_ready": False,
            "perturbation_ready": False,
            "donor_status": "missing_external_donor_field",
            "provenance_status": "pending_external_sidecar_doi",
            "orthology_status": "pending_missing_ortholog_source",
            "esm2_status": "pending_missing_embedding_artifact",
        },
        "source_hashes": {
            str(int(row["source_id"])): {
                "path": str(row["local_path"]),
                "file_fingerprint": str(row["file_fingerprint"]),
                "source_md5": _string_or_none(row.get("source_md5")),
            }
            for _, row in sources.iterrows()
        },
        "namespace_hashes": namespace_hashes,
        "artifact_hashes": artifact_hashes,
        "index_sha256": index_digest,
        "output": {
            "path": str(output.resolve()),
            "files": {path.name: path.stat().st_size for path in index_files},
            "bytes": int(sum(path.stat().st_size for path in index_files)),
        },
        "validation": {"status": "pending", "report": None},
        "missing_required_external_fields": [
            "paper DOI/PMID and canonical dataset URL",
            "donor/animal identifier",
            "BioMart/Ensembl orthology groups",
            "ESM2 embedding index/artifact",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "reflection.md").write_text(
        "# Mouse Spatial Cell Index Reflection\n\n"
        f"- Format: `{MOUSE_SPATIAL_INDEX_FORMAT}`\n"
        f"- Cells: `{selected_cells:,}`; sections: `{len(samples):,}`; sources: `{len(sources)}`\n"
        f"- Build elapsed: `{elapsed:.2f}s`; expression copy bytes: `0`\n"
        "- Split unit: section-held-out. Donor identifiers were unavailable and were not inferred.\n"
        "- Provenance DOI, orthology groups, and ESM2 embeddings remain pending; this artifact is not a cross-species checkpoint input.\n"
        "- Validation status: pending backed CSR/dense smoke read.\n",
        encoding="utf-8",
    )
    return manifest


class MouseSpatialBatchReader(IndexedH5ADBatchReader):
    """Read mouse-local token batches from a mouse spatial cell index."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        cells: pd.DataFrame | None = None,
        cell_columns: Sequence[str] | None = None,
        cell_sample_keys: Collection[str] | None = None,
        cell_splits: Collection[str] | None = None,
        cell_source_ids: Collection[int] | None = None,
        allow_empty_cells: bool = False,
        unified_vocabulary: str | Path | None = None,
        allow_unmapped_unified_genes: bool = False,
    ):
        self.mouse_root = Path(artifact_dir)
        manifest_path = self.mouse_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != MOUSE_SPATIAL_INDEX_FORMAT:
            raise ValueError("artifact is not a mouse spatial cell index")
        if manifest.get("species") != "mouse" or int(manifest.get("species_id")) != 1:
            raise ValueError(
                "mouse spatial reader requires species=mouse, species_id=1"
            )
        super().__init__(
            self.mouse_root,
            cells=cells,
            cell_columns=cell_columns,
            cell_sample_keys=cell_sample_keys,
            cell_splits=cell_splits,
            cell_source_ids=cell_source_ids,
            allow_empty_cells=allow_empty_cells,
        )
        if not (self.sources["species"].astype(str) == "mouse").all():
            raise ValueError("mouse spatial sources contain a non-mouse species")
        self.species = "mouse"
        self.species_id = 1
        self.unified_gene_adapter = (
            UnifiedGeneIndexAdapter(
                self.mouse_root,
                unified_vocabulary,
                allow_unmapped=allow_unmapped_unified_genes,
            )
            if unified_vocabulary is not None
            else None
        )

    def _gene_lookup(self, source_id: int) -> np.ndarray:
        source_id = int(source_id)
        if source_id not in self._column_to_gene:
            source = self.sources.loc[source_id]
            mapping = self.gene_map[self.gene_map["source_id"] == source_id]
            lookup = np.full(int(source["n_vars"]), -1, dtype=np.int32)
            valid = mapping["namespace_gene_index"] >= 0
            lookup[mapping.loc[valid, "source_col"].to_numpy(np.int64)] = mapping.loc[
                valid, "namespace_gene_index"
            ].to_numpy(np.int32)
            self._column_to_gene[source_id] = lookup
        return self._column_to_gene[source_id]

    def read_batch(
        self,
        source_id: int,
        row_indices: np.ndarray,
        cell_frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        batch = super().read_batch(source_id, row_indices, cell_frame)
        batch["namespace_gene_index"] = batch["gene_index"]
        batch["species"] = self.species
        batch["species_id"] = self.species_id
        if self.unified_gene_adapter is not None:
            batch = self.unified_gene_adapter.remap_batch(batch)
        return batch


__all__ = [
    "MOUSE_SPATIAL_INDEX_FORMAT",
    "MouseSpatialBatchReader",
    "assign_section_splits",
    "build_mouse_spatial_index",
]
