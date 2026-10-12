"""Zero-copy cell index for the candidate mouse scRNA/snRNA corpus.

Only observation metadata and source row numbers are materialized. Raw counts
remain in the source H5AD files and are projected into the mouse-local
``namespace_gene_index`` through :class:`IndexedH5ADBatchReader`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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

from .h5ad_io import file_fingerprint, matrix_encoding, matrix_shape, read_column
from .indexed_h5ad import IndexedH5ADBatchReader
from .unified_gene_index import UnifiedGeneIndexAdapter

MOUSE_MOLECULAR_INDEX_FORMAT = "brainwm_mouse_molecular_cell_index_v1"
SPLIT_UNIT = "sample_or_library_section"
ALLOWED_ASSAYS = {"scRNA", "snRNA"}
MOUSE_MOLECULAR_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"

LABEL_COLUMNS = [
    "cl_id",
    "cl_term",
    "original_cell_type",
    "original_subclass",
    "original_anatomical_region",
    "anatomical_division_label",
    "uberon_id",
    "uberon_term",
]
CONTEXT_COLUMNS = {
    "age_raw": ("age",),
    "timepoint_raw": ("timepoint", "time"),
    "developmental_stage_raw": ("developmental_stage", "stage"),
}
SAMPLE_ONLY_COLUMNS = {
    "donor_id": ("donor_id",),
    "animal_id": ("animal_id",),
}
CELL_COLUMNS = [
    "source_id",
    "row_index",
    "species",
    "species_id",
    "assay",
    "sample_key",
    "split",
    "split_unit",
    *LABEL_COLUMNS,
    *CONTEXT_COLUMNS,
]
CELL_SCHEMA = pa.schema([
    pa.field("source_id", pa.int32(), nullable=False),
    pa.field("row_index", pa.int64(), nullable=False),
    pa.field("species", pa.string(), nullable=False),
    pa.field("species_id", pa.int16(), nullable=False),
    pa.field("assay", pa.string(), nullable=False),
    pa.field("sample_key", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("split_unit", pa.string(), nullable=False),
    *[pa.field(name, pa.string()) for name in LABEL_COLUMNS],
    *[pa.field(name, pa.string()) for name in CONTEXT_COLUMNS],
])


def resolve_mouse_molecular_artifact_path(value: str | Path) -> Path:
    """Resolve one direct child of the project artifact root, fail-closed."""
    artifact_root = MOUSE_MOLECULAR_ARTIFACT_ROOT.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        if len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
            raise ValueError(
                "relative mouse molecular output must be a single artifact name"
            )
        candidate = artifact_root / candidate.name
    resolved = candidate.resolve()
    if resolved == artifact_root or resolved.parent != artifact_root:
        raise ValueError(
            f"mouse molecular artifact must be a direct child of {artifact_root}"
        )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(seed: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}|{value}".encode()).digest()


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")
    value = str(value).strip()
    return value or None


def _string_array(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(values)
    # ``read_column`` has already decoded H5AD string/categorical payloads.
    # Preserve that vectorized result; per-cell Python normalization becomes
    # material at the full 9.5M-row scale.
    if array.dtype.kind in {"O", "U"}:
        return array.astype(object, copy=False)
    missing = pd.isna(array)
    result = array.astype(str).astype(object)
    result[missing] = None
    return result


def _read_optional(obs: h5py.Group, name: str | None,
                   indices: np.ndarray) -> np.ndarray:
    if name is None or name not in obs:
        return np.asarray([None] * len(indices), dtype=object)
    return _string_array(read_column(obs, name, indices))


def _resolve_column(obs: h5py.Group, candidates: Iterable[str]) -> str | None:
    return next((name for name in candidates if name in obs), None)


def _infer_assay(path: Path) -> str:
    matches = ALLOWED_ASSAYS.intersection(path.parts)
    if len(matches) != 1:
        raise ValueError(
            f"cannot resolve exactly one assay from curated scRNA/snRNA path: {path}"
        )
    return next(iter(matches))


def _effective_sample_keys(
    sample_ids: np.ndarray,
    library_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sample_ids) != len(library_labels):
        raise ValueError("sample_id and library_label lengths differ")
    keys = np.empty(len(sample_ids), dtype=object)
    fields = np.empty(len(sample_ids), dtype=object)
    for index, (sample_id, library_label) in enumerate(zip(sample_ids, library_labels)):
        if sample_id is not None:
            keys[index] = f"mouse::sample_id::{sample_id}"
            fields[index] = "sample_id"
        elif library_label is not None:
            keys[index] = f"mouse::library_label::{library_label}"
            fields[index] = "library_label"
        else:
            raise ValueError("a molecular observation lacks both sample_id and library_label")
    return keys, fields


def assign_sample_splits(
    sample_keys: Iterable[str],
    *,
    seed: str = "20260827",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign one deterministic split to every global sample/library key."""
    keys = sorted({str(key) for key in sample_keys})
    if not keys:
        raise ValueError("cannot split an empty sample set")
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must lie in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("split fractions must leave train samples")
    if len(keys) == 1:
        return {keys[0]: "train"}
    n_test = max(1, round(len(keys) * test_fraction))
    n_validation = max(1, round(len(keys) * validation_fraction))
    if n_test + n_validation >= len(keys):
        n_test = 1
        n_validation = 1 if len(keys) >= 3 else 0
    ranked = sorted(keys, key=lambda key: _stable_rank(seed, key))
    result = {key: "train" for key in keys}
    for key in ranked[:n_test]:
        result[key] = "test"
    for key in ranked[n_test:n_test + n_validation]:
        result[key] = "validation"
    return result


def _source_inventory(path: Path, namespace_source: pd.Series) -> dict[str, Any]:
    assay = _infer_assay(path)
    with h5py.File(path, "r") as handle:
        if "obs" not in handle:
            raise KeyError(f"{path} lacks obs")
        expression_element = str(namespace_source["expression_element"])
        expression = handle[expression_element]
        n_obs, n_vars = matrix_shape(expression)
        if n_obs != int(namespace_source["n_obs"]):
            raise ValueError(f"{path} observation shape changed since namespace scan")
        if n_vars != int(namespace_source["n_vars"]):
            raise ValueError(f"{path} feature shape changed since namespace scan")
        obs = handle["obs"]
        sample_id = (
            _string_array(read_column(obs, "sample_id"))
            if "sample_id" in obs else np.asarray([None] * n_obs, dtype=object)
        )
        library_label = (
            _string_array(read_column(obs, "library_label"))
            if "library_label" in obs else np.asarray([None] * n_obs, dtype=object)
        )
        if len(sample_id) != n_obs or len(library_label) != n_obs:
            raise ValueError(f"{path} sample metadata length differs from raw/X rows")
        keys, fields = _effective_sample_keys(sample_id, library_label)
        counts = pd.Series(keys, dtype="string").value_counts(sort=False)
        column_map = {
            output: _resolve_column(obs, candidates)
            for output, candidates in {**CONTEXT_COLUMNS, **SAMPLE_ONLY_COLUMNS}.items()
        }
        available_columns = sorted(str(name) for name in obs)
        encoding = matrix_encoding(expression)
    return {
        "path": path,
        "source_id": int(namespace_source["source_id"]),
        "assay": assay,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "matrix_encoding": encoding,
        "sample_counts": {str(key): int(value) for key, value in counts.items()},
        "sample_key_fields": sorted({str(value) for value in fields}),
        "column_map": column_map,
        "available_obs_columns": available_columns,
    }


def _cell_chunks(
    inventory: dict[str, Any],
    source: pd.Series,
    split_by_sample: dict[str, str],
    *,
    chunk_size: int,
):
    path = inventory["path"]
    n_obs = int(inventory["n_obs"])
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        for start in range(0, n_obs, chunk_size):
            rows = np.arange(start, min(start + chunk_size, n_obs), dtype=np.int64)
            sample_ids = _read_optional(obs, "sample_id", rows)
            library_labels = _read_optional(obs, "library_label", rows)
            sample_keys, sample_fields = _effective_sample_keys(sample_ids, library_labels)
            context = {
                output: _read_optional(obs, inventory["column_map"][output], rows)
                for output in CONTEXT_COLUMNS
            }
            labels = {name: _read_optional(obs, name, rows) for name in LABEL_COLUMNS}
            frame = pd.DataFrame({
                "source_id": np.full(len(rows), inventory["source_id"], dtype=np.int32),
                "row_index": rows,
                "species": np.full(len(rows), "mouse", dtype=object),
                "species_id": np.full(len(rows), 1, dtype=np.int16),
                "assay": np.full(len(rows), inventory["assay"], dtype=object),
                "sample_key": sample_keys,
                "split": np.asarray([split_by_sample[key] for key in sample_keys], dtype=object),
                "split_unit": np.full(len(rows), SPLIT_UNIT, dtype=object),
                **labels,
                **context,
            }, columns=CELL_COLUMNS)
            sample_meta = pd.DataFrame({
                "sample_key": sample_keys,
                "sample_key_field": sample_fields,
                "sample_id": sample_ids,
                "library_label": library_labels,
                "donor_id": _read_optional(
                    obs, inventory["column_map"]["donor_id"], rows
                ),
                "animal_id": _read_optional(
                    obs, inventory["column_map"]["animal_id"], rows
                ),
                **context,
                "original_anatomical_region": labels["original_anatomical_region"],
                "uberon_id": labels["uberon_id"],
            })
            yield frame, sample_meta


def _write_parquet_chunks(path: Path, chunks: Iterable[pd.DataFrame]) -> int:
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for frame in chunks:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(
                frame[CELL_COLUMNS], schema=CELL_SCHEMA, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    path,
                    CELL_SCHEMA,
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


def _new_sample_state(split: str) -> dict[str, Any]:
    return {
        "split": split,
        "selected_cells": 0,
        "sample_key_fields": set(),
        "sample_ids": set(),
        "library_labels": set(),
        "assays": set(),
        "datasets": set(),
        "source_ids": set(),
        "donor_ids": set(),
        "animal_ids": set(),
        "age_raw": set(),
        "timepoint_raw": set(),
        "developmental_stage_raw": set(),
        "regions": set(),
        "uberon_ids": set(),
    }


def _add_non_null(target: set[str], values: pd.Series) -> None:
    # Sample metadata are low-cardinality. Deduplicate with pandas before the
    # small Python set update instead of normalizing every cell repeatedly.
    target.update(
        value
        for value in (_string_or_none(item) for item in values.dropna().unique())
        if value is not None
    )


def _update_sample_states(
    states: dict[str, dict[str, Any]],
    sample_meta: pd.DataFrame,
    *,
    inventory: dict[str, Any],
    source: pd.Series,
    split_by_sample: dict[str, str],
) -> None:
    for sample_key, part in sample_meta.groupby("sample_key", sort=False, dropna=False):
        key = str(sample_key)
        state = states.setdefault(key, _new_sample_state(split_by_sample[key]))
        if state["split"] != split_by_sample[key]:
            raise ValueError(f"sample split changed while indexing: {key}")
        state["selected_cells"] += len(part)
        state["assays"].add(str(inventory["assay"]))
        state["datasets"].add(str(source["dataset"]))
        state["source_ids"].add(int(inventory["source_id"]))
        for field, target in (
            ("sample_key_field", "sample_key_fields"),
            ("sample_id", "sample_ids"),
            ("library_label", "library_labels"),
            ("donor_id", "donor_ids"),
            ("animal_id", "animal_ids"),
            ("age_raw", "age_raw"),
            ("timepoint_raw", "timepoint_raw"),
            ("developmental_stage_raw", "developmental_stage_raw"),
            ("original_anatomical_region", "regions"),
            ("uberon_id", "uberon_ids"),
        ):
            _add_non_null(state[target], part[field])


def _single_value(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _value_status(values: set[str], missing: str) -> str:
    if not values:
        return missing
    if len(values) == 1:
        return "single_raw_value"
    return "conflicting_multiple_raw_values"


def _json_values(values: set[Any]) -> str:
    return json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))


def _sample_table(states: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for sample_key in sorted(states):
        state = states[sample_key]
        donor_values = state["donor_ids"] or state["animal_ids"]
        rows.append({
            "sample_key": sample_key,
            "species": "mouse",
            "species_id": 1,
            "sample_key_field": _single_value(state["sample_key_fields"]),
            "sample_id": _single_value(state["sample_ids"]),
            "library_label": _single_value(state["library_labels"]),
            "assay": _single_value(state["assays"]),
            "datasets_json": _json_values(state["datasets"]),
            "source_ids_json": _json_values(state["source_ids"]),
            "split": state["split"],
            "split_unit": SPLIT_UNIT,
            "n_observations": int(state["selected_cells"]),
            "selected_cells": int(state["selected_cells"]),
            "regions_json": _json_values(state["regions"]),
            "uberon_ids_json": _json_values(state["uberon_ids"]),
            "age_raw": _single_value(state["age_raw"]),
            "age_status": _value_status(state["age_raw"], "missing_source_field"),
            "timepoint_raw": _single_value(state["timepoint_raw"]),
            "timepoint_status": _value_status(
                state["timepoint_raw"], "missing_source_field"
            ),
            "developmental_stage_raw": _single_value(
                state["developmental_stage_raw"]
            ),
            "developmental_stage_status": _value_status(
                state["developmental_stage_raw"], "missing_source_field"
            ),
            "donor_id": _single_value(donor_values),
            "animal_id": _single_value(state["animal_ids"]),
            "donor_status": _value_status(
                donor_values, "missing_external_donor_or_animal_field"
            ),
            "paper_provenance_status": "unverified_missing_file_to_paper_mapping",
        })
    return pd.DataFrame(rows)


def _schema() -> dict[str, Any]:
    return {
        "format": MOUSE_MOLECULAR_INDEX_FORMAT,
        "scope": "candidate mouse scRNA/snRNA metadata and zero-copy source row index",
        "species": {"name": "mouse", "species_id": 1, "unknown_policy": "error"},
        "assays": sorted(ALLOWED_ASSAYS),
        "split": {
            "unit": SPLIT_UNIT,
            "priority": ["sample_id", "library_label"],
            "meaning": (
                "global sample/library-held-out across source files; donor is unavailable "
                "and never inferred"
            ),
        },
        "expression": {
            "storage": "original H5AD raw/X only; no expression copy",
            "reader": "brainwm_data.mouse_molecular_index.MouseMolecularBatchReader",
            "token_index": "namespace_gene_index (mouse-local only)",
            "target": "non-negative raw counts in the measured source gene panel",
        },
        "cells_columns": {
            "identity": ["source_id", "row_index"],
            "model_context": [
                "species", "species_id", "assay", "sample_key", "split",
                "original_anatomical_region", "anatomical_division_label",
                "uberon_id", "uberon_term", "age_raw", "timepoint_raw",
                "developmental_stage_raw",
            ],
            "annotation_targets": [
                "cl_id", "cl_term", "original_cell_type", "original_subclass"
            ],
        },
        "files": {
            "cells.parquet": "compressed training metadata and source row pointers only",
            "samples.parquet": "one row per global sample/library split unit",
            "sources.parquet": "source identity, assay and metadata coverage",
            "genes.parquet": "mouse-local namespace with GENCODE gene_type",
            "source_gene_map.parquet": "source columns to mouse-local namespace indices",
            "species_vocabulary.json": "explicit human=0/mouse=1 identity contract",
            "manifest.json": "candidate gates, counts, hashes and storage boundary",
            "validation.json": "streamed table checks and all-source backed-read smoke",
        },
        "candidate_gate": {
            "candidate_only": True,
            "formal_training_allowed": False,
            "allowed": ["candidate Stage0 technical/pretraining reader experiments"],
            "blocked": [
                "donor-held-out evaluation",
                "Stage1 temporal supervision",
                "formal cross-species corpus promotion",
            ],
        },
        "missing_features": {
            "donor_or_animal_id": "absent in all audited obs tables",
            "age_time_developmental_stage": "absent in all audited obs tables",
            "paper_provenance": "file-to-paper mapping not verified",
            "homology_group": "pending ortholog source",
            "esm2_embedding_index": "pending embedding artifact",
        },
        "forbidden": [
            "treat mouse namespace_gene_index as human gene_index",
            "infer donor or age from file/dataset names (including Zeng-Aging)",
            "claim donor-held-out or temporal supervision",
            "claim requested-origin provenance from candidate-copy checksums",
        ],
    }


def build_mouse_molecular_index(
    namespace_artifact: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int = 100_000,
    split_seed: str = "20260827",
) -> dict[str, Any]:
    """Build a new candidate index without materializing expression values."""
    started = time.perf_counter()
    namespace = Path(namespace_artifact).resolve()
    output = resolve_mouse_molecular_artifact_path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mouse molecular index: {output}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    namespace_manifest = json.loads((namespace / "manifest.json").read_text())
    if namespace_manifest.get("format") != "brainwm_species_gene_namespace_v1":
        raise ValueError("namespace artifact has an unexpected format")
    if namespace_manifest.get("species") != "mouse" or int(
        namespace_manifest.get("species_id")
    ) != 1:
        raise ValueError("mouse molecular index requires species=mouse, species_id=1")
    namespace_sources = pd.read_parquet(namespace / "sources.parquet")
    if namespace_sources.empty or not namespace_sources["species"].astype(str).eq("mouse").all():
        raise ValueError("namespace sources must be non-empty and mouse-only")
    source_gene_map = pd.read_parquet(namespace / "source_gene_map.parquet")
    if "namespace_gene_index" not in source_gene_map:
        raise ValueError("mouse namespace map lacks namespace_gene_index")

    inventories = []
    all_sample_keys: set[str] = set()
    source_rows = []
    for _, namespace_source in namespace_sources.sort_values("source_id").iterrows():
        path = Path(str(namespace_source["local_path"])).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if file_fingerprint(path) != str(namespace_source["file_fingerprint"]):
            raise ValueError(f"source fingerprint changed since namespace scan: {path}")
        inventory = _source_inventory(path, namespace_source)
        inventories.append((namespace_source, inventory))
        all_sample_keys.update(inventory["sample_counts"])
        source = namespace_source.to_dict()
        source.update({
            "source_id": inventory["source_id"],
            "assay": inventory["assay"],
            "selected_cells": inventory["n_obs"],
            "selected_samples": len(inventory["sample_counts"]),
            "split_unit": SPLIT_UNIT,
            "sample_key_fields": json.dumps(inventory["sample_key_fields"]),
            "available_obs_columns_json": json.dumps(
                inventory["available_obs_columns"], separators=(",", ":")
            ),
            "age_column": inventory["column_map"]["age_raw"],
            "timepoint_column": inventory["column_map"]["timepoint_raw"],
            "developmental_stage_column": inventory["column_map"][
                "developmental_stage_raw"
            ],
            "donor_column": inventory["column_map"]["donor_id"],
            "animal_column": inventory["column_map"]["animal_id"],
            "temporal_status": "pending_value_scan",
            "donor_status": "pending_value_scan",
            "provenance_status": "candidate_copy_requested_origin_unverified",
        })
        source_rows.append(source)
    split_by_sample = assign_sample_splits(all_sample_keys, seed=split_seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name in ("genes.parquet", "source_gene_map.parquet", "species_vocabulary.json"):
            shutil.copyfile(namespace / name, staging / name)

        samples_state: dict[str, dict[str, Any]] = {}
        cells_by_split: Counter[str] = Counter()
        cells_by_assay: Counter[str] = Counter()
        source_coverage: dict[int, Counter[str]] = {
            int(row["source_id"]): Counter() for row in source_rows
        }

        def chunks():
            for (namespace_source, inventory), source_row in zip(inventories, source_rows):
                source = pd.Series(source_row)
                coverage = source_coverage[inventory["source_id"]]
                for frame, sample_meta in _cell_chunks(
                    inventory, source, split_by_sample, chunk_size=chunk_size
                ):
                    cells_by_split.update(frame["split"].astype(str).tolist())
                    cells_by_assay.update(frame["assay"].astype(str).tolist())
                    for name in [*LABEL_COLUMNS, *CONTEXT_COLUMNS]:
                        coverage[name] += int(frame[name].notna().sum())
                    coverage["donor_id"] += int(sample_meta["donor_id"].notna().sum())
                    coverage["animal_id"] += int(sample_meta["animal_id"].notna().sum())
                    _update_sample_states(
                        samples_state,
                        sample_meta,
                        inventory=inventory,
                        source=source,
                        split_by_sample=split_by_sample,
                    )
                    yield frame

        written_cells = _write_parquet_chunks(staging / "cells.parquet", chunks())
        if written_cells != int(namespace_sources["n_obs"].sum()):
            raise ValueError("indexed cell count differs from namespace source shapes")
        samples = _sample_table(samples_state)
        if set(samples["sample_key"]) != all_sample_keys:
            raise ValueError("sample table does not cover the global split keys")

        sources = pd.DataFrame(source_rows)
        for index, row in sources.iterrows():
            coverage = source_coverage[int(row["source_id"])]
            sources.at[index, "temporal_status"] = (
                "available_raw_values"
                if sum(coverage[name] for name in CONTEXT_COLUMNS) > 0
                else "missing_source_fields"
            )
            sources.at[index, "donor_status"] = (
                "available_raw_values"
                if coverage["donor_id"] + coverage["animal_id"] > 0
                else "missing_external_donor_or_animal_field"
            )
            sources.at[index, "metadata_non_null_counts_json"] = json.dumps(
                {name: int(value) for name, value in sorted(coverage.items())},
                separators=(",", ":"),
            )
        samples.to_parquet(
            staging / "samples.parquet", index=False, compression="zstd"
        )
        sources.to_parquet(
            staging / "sources.parquet", index=False, compression="zstd"
        )
        (staging / "SCHEMA.json").write_text(
            json.dumps(_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        index_files = [
            staging / name for name in (
                "cells.parquet", "samples.parquet", "sources.parquet", "genes.parquet",
                "source_gene_map.parquet", "species_vocabulary.json", "SCHEMA.json",
            )
        ]
        artifact_hashes = {path.name: _sha256(path) for path in index_files}
        index_digest = hashlib.sha256(
            "\n".join(
                f"{name}:{artifact_hashes[name]}" for name in sorted(artifact_hashes)
            ).encode()
        ).hexdigest()
        source_md5_rows = [
            f"{int(row['source_id'])}:{_string_or_none(row.get('source_md5'))}"
            for _, row in sources.sort_values("source_id").iterrows()
        ]
        source_md5_set_sha256 = hashlib.sha256(
            "\n".join(source_md5_rows).encode()
        ).hexdigest()
        namespace_bytes = sum(
            (namespace / name).stat().st_size
            for name in ("genes.parquet", "source_gene_map.parquet", "species_vocabulary.json")
        )
        estimated_cells_bytes = int(
            namespace_manifest.get("disk_estimate", {}).get(
                "estimated_future_cells_parquet_bytes", 170_000_000
            )
        )
        actual_bytes = int(sum(path.stat().st_size for path in index_files))
        by_assay_sources = sources.groupby("assay").size().to_dict()
        by_assay_samples = samples.groupby("assay").size().to_dict()
        by_split_samples = samples.groupby("split").size().to_dict()
        temporal_values = int(
            samples[[*CONTEXT_COLUMNS]].notna().any(axis=1).sum()
        )
        donor_values = int(samples[["donor_id", "animal_id"]].notna().any(axis=1).sum())
        manifest = {
            "format": MOUSE_MOLECULAR_INDEX_FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "namespace_artifact": str(namespace),
            "namespace_manifest_sha256": _sha256(namespace / "manifest.json"),
            "species": "mouse",
            "species_id": 1,
            "candidate_only": True,
            "formal_training_allowed": False,
            "human_gene_vocabulary_modified": False,
            "expression_copy_bytes": 0,
            "scan": {
                "scope": "obs metadata plus raw/X shape/encoding; expression values not read",
                "chunk_size": chunk_size,
                "split_seed": split_seed,
                "split_unit": SPLIT_UNIT,
                "source_rows_indexed": written_cells,
            },
            "stats": {
                "sources": len(sources),
                "cells": int(written_cells),
                "samples": len(samples),
                "source_sample_memberships": int(sources["selected_samples"].sum()),
                "genes": len(pd.read_parquet(staging / "genes.parquet")),
                "by_assay_sources": {
                    str(key): int(value) for key, value in by_assay_sources.items()
                },
                "by_assay_cells": {
                    str(key): int(value) for key, value in cells_by_assay.items()
                },
                "by_assay_samples": {
                    str(key): int(value) for key, value in by_assay_samples.items()
                },
                "by_split_cells": {
                    str(key): int(value) for key, value in cells_by_split.items()
                },
                "by_split_samples": {
                    str(key): int(value) for key, value in by_split_samples.items()
                },
                "temporal_samples_with_raw_value": temporal_values,
                "donor_or_animal_samples_with_raw_value": donor_values,
                "stage0_candidate_reader_ready": True,
                "temporal_metadata_present": temporal_values > 0,
                "stage1_temporal_supervision_ready": False,
                "donor_held_out_ready": donor_values > 0,
                "formal_cross_species_ready": False,
            },
            "training_gate": {
                "allowed": "candidate-only Stage0 technical/pretraining reader experiments",
                "formal_training": "blocked_unverified_requested_origin_and_paper_provenance",
                "donor_held_out": "blocked_missing_donor_or_animal_identifier",
                "stage1_temporal": "blocked_missing_age_time_developmental_stage",
                "cross_species": "blocked_missing_provenance_orthology_and_esm2",
            },
            "source_integrity": {
                "file_fingerprint_rechecked": True,
                "source_md5_entries": int(sources["source_md5"].notna().sum()),
                "candidate_source_md5_set_sha256": source_md5_set_sha256,
                "source_md5_recomputed_during_index": False,
                "requested_origin_verified": False,
                "status": "candidate_copy_checksums_not_requested_origin_provenance",
            },
            "source_hashes": {
                str(int(row["source_id"])): {
                    "path": str(row["local_path"]),
                    "file_fingerprint": str(row["file_fingerprint"]),
                    "source_md5": _string_or_none(row.get("source_md5")),
                }
                for _, row in sources.iterrows()
            },
            "artifact_hashes": artifact_hashes,
            "index_sha256": index_digest,
            "storage": {
                "prebuild_estimated_cells_parquet_bytes": estimated_cells_bytes,
                "prebuild_estimated_artifact_bytes": int(
                    estimated_cells_bytes + namespace_bytes + 8_000_000
                ),
                "actual_artifact_bytes_before_manifest": actual_bytes,
                "expression_copy_bytes": 0,
                "morton_or_order_array_bytes": 0,
            },
            "output": {
                "path": str(output),
                "files": {path.name: path.stat().st_size for path in index_files},
                "bytes_before_manifest": actual_bytes,
            },
            "validation": {
                "status": "pending",
                "backed_read": "pending_all_source_csr_smoke",
                "report": None,
            },
            "missing_required_external_fields": [
                "verified requested-source identity and file-to-paper DOI/PMID mapping",
                "donor/animal identifier",
                "age/time/developmental-stage metadata",
                "BioMart/Ensembl orthology groups",
                "ESM2 embedding index/artifact",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "reflection.md").write_text(
            "# Mouse Molecular Cell Index Reflection\n\n"
            f"- Candidate cells: `{written_cells:,}`; global samples: `{len(samples):,}`; "
            f"sources: `{len(sources)}`.\n"
            f"- Assays: `{dict(cells_by_assay)}`; split unit: `{SPLIT_UNIT}`.\n"
            f"- Build elapsed: `{time.perf_counter() - started:.2f}s`; expression copy: `0` bytes.\n"
            "- Cell metadata is ZSTD Parquet; no Morton/order array was generated.\n"
            "- Formal training remains blocked by requested-origin/paper provenance.\n"
            "- Donor-held-out and Stage1 temporal supervision remain blocked by missing metadata.\n"
            "- Validation: pending all-source backed CSR smoke.\n",
            encoding="utf-8",
        )
        staging.rename(output)
        return manifest
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise


class MouseMolecularBatchReader(IndexedH5ADBatchReader):
    """Read mouse-local raw-count tokens from the molecular cell index."""

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
        if manifest.get("format") != MOUSE_MOLECULAR_INDEX_FORMAT:
            raise ValueError("artifact is not a mouse molecular cell index")
        if manifest.get("species") != "mouse" or int(manifest.get("species_id")) != 1:
            raise ValueError("mouse molecular reader requires species=mouse, species_id=1")
        super().__init__(
            self.mouse_root,
            cells=cells,
            cell_columns=cell_columns,
            cell_sample_keys=cell_sample_keys,
            cell_splits=cell_splits,
            cell_source_ids=cell_source_ids,
            allow_empty_cells=allow_empty_cells,
        )
        if not self.sources["species"].astype(str).eq("mouse").all():
            raise ValueError("mouse molecular sources contain a non-mouse species")
        if not self.sources["assay"].astype(str).isin(ALLOWED_ASSAYS).all():
            raise ValueError("mouse molecular sources contain an unsupported assay")
        self.species = "mouse"
        self.species_id = 1
        self.unified_gene_adapter = (
            UnifiedGeneIndexAdapter(
                self.mouse_root,
                unified_vocabulary,
                allow_unmapped=allow_unmapped_unified_genes,
            )
            if unified_vocabulary is not None else None
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
        batch["assay"] = str(self.sources.loc[int(source_id), "assay"])
        if self.unified_gene_adapter is not None:
            batch = self.unified_gene_adapter.remap_batch(batch)
        return batch


__all__ = [
    "ALLOWED_ASSAYS",
    "CELL_COLUMNS",
    "MOUSE_MOLECULAR_ARTIFACT_ROOT",
    "MOUSE_MOLECULAR_INDEX_FORMAT",
    "SPLIT_UNIT",
    "MouseMolecularBatchReader",
    "assign_sample_splits",
    "build_mouse_molecular_index",
    "resolve_mouse_molecular_artifact_path",
]
