"""Append-only persistent CSR counts for native-panel evaluations."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

CACHE_FORMAT = "brainwm_native_count_csr_cache_v1"
STORAGE_FORMAT = "csr_npy_v1"
EXTRACTION_CONTRACT = "raw_projected_counts_v1"
ROW_COLUMNS = ("source_id", "row_index", "sample_key")
GENE_COLUMNS = (
    "gene_index",
    "species",
    "species_id",
    "ensembl_id",
    "species_gene_key",
    "homology_group",
)


def _normalize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    missing = set(ROW_COLUMNS) - set(rows)
    if missing:
        raise ValueError(f"count-cache rows lack columns: {sorted(missing)}")
    unexpected = set(rows) - set(ROW_COLUMNS)
    if unexpected:
        raise ValueError(
            "count-cache rows must contain identity fields only: "
            f"{sorted(unexpected)}"
        )
    result = rows.loc[:, list(ROW_COLUMNS)].copy()
    result["source_id"] = pd.to_numeric(
        result["source_id"], errors="raise"
    ).astype(np.int64)
    result["row_index"] = pd.to_numeric(
        result["row_index"], errors="raise"
    ).astype(np.int64)
    if result["sample_key"].isna().any():
        raise ValueError("count-cache sample keys must be non-null")
    result["sample_key"] = result["sample_key"].astype(str)
    if result.empty:
        raise ValueError("count-cache rows must be non-empty")
    if result.duplicated(["source_id", "row_index"]).any():
        raise ValueError("count-cache expression row keys must be unique")
    return result.reset_index(drop=True)


def _normalize_genes(genes: pd.DataFrame) -> pd.DataFrame:
    missing = set(GENE_COLUMNS) - set(genes)
    if missing:
        raise ValueError(f"count-cache genes lack columns: {sorted(missing)}")
    unexpected = set(genes) - set(GENE_COLUMNS)
    if unexpected:
        raise ValueError(
            "count-cache genes contain fields outside the semantic contract: "
            f"{sorted(unexpected)}"
        )
    result = genes.loc[:, list(GENE_COLUMNS)].copy()
    for column in ("gene_index", "species_id"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(
            np.int64
        )
    for column in ("species", "ensembl_id", "species_gene_key"):
        if result[column].isna().any():
            raise ValueError(f"count-cache gene field must be non-null: {column}")
        result[column] = result[column].astype(str)
    result["homology_group"] = (
        result["homology_group"].astype("string").fillna("").astype(str)
    )
    if result.empty:
        raise ValueError("count-cache genes must be non-empty")
    if result["gene_index"].duplicated().any():
        raise ValueError("count-cache gene indices must be unique")
    if result["species_gene_key"].duplicated().any():
        raise ValueError("count-cache species gene keys must be unique")
    return result.reset_index(drop=True)


def _normalize_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    def reject_sha_fields(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if "sha256" in str(key).lower():
                    raise ValueError("count-cache provenance must not contain SHA256")
                reject_sha_fields(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                reject_sha_fields(child)

    reject_sha_fields(provenance)
    try:
        payload = json.dumps(
            dict(provenance), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("count-cache provenance must be finite JSON data") from exc
    normalized = json.loads(payload)
    if not isinstance(normalized, dict):
        raise TypeError("count-cache provenance must be a mapping")
    return normalized


def _safe_component(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."}:
        raise ValueError(f"count-cache {name} must be non-empty")
    if any(not (char.isalnum() or char in "._-") for char in text):
        raise ValueError(f"count-cache {name} contains unsafe characters: {text!r}")
    return text


def _semantic_gene_positions(
    cached: pd.DataFrame, requested: pd.DataFrame
) -> np.ndarray | None:
    lookup = {
        int(gene_index): position
        for position, gene_index in enumerate(cached["gene_index"].to_numpy())
    }
    try:
        positions = np.asarray(
            [lookup[int(value)] for value in requested["gene_index"].to_numpy()],
            dtype=np.int64,
        )
    except KeyError:
        return None
    selected = cached.iloc[positions].reset_index(drop=True)
    if not selected.equals(requested.reset_index(drop=True)):
        return None
    return positions


def _row_positions(cached: pd.DataFrame, requested: pd.DataFrame) -> np.ndarray:
    lookup = {
        (int(row.source_id), int(row.row_index), str(row.sample_key)): position
        for position, row in enumerate(cached.itertuples(index=False))
    }
    keys = [
        (int(row.source_id), int(row.row_index), str(row.sample_key))
        for row in requested.itertuples(index=False)
    ]
    try:
        positions = np.asarray([lookup[key] for key in keys], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("requested count-cache rows are not contained") from exc
    selected = cached.iloc[positions].reset_index(drop=True)
    if not selected.equals(requested.reset_index(drop=True)):
        raise AssertionError("count-cache row order changed during reindexing")
    return positions


def _validated_csr(
    matrix: sp.spmatrix, rows: int, columns: int
) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.float32, copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.eliminate_zeros()
    if result.shape != (rows, columns):
        raise ValueError(
            f"count-cache matrix shape changed: {result.shape} != {(rows, columns)}"
        )
    if not np.isfinite(result.data).all() or np.any(result.data < 0):
        raise ValueError("count-cache values must be finite non-negative counts")
    return result


@dataclass(frozen=True)
class NativeCountCacheEntry:
    path: Path
    rows: pd.DataFrame
    genes: pd.DataFrame
    matrix: sp.csr_matrix
    manifest: Mapping[str, Any]

    def view(self, rows: pd.DataFrame, genes: pd.DataFrame) -> sp.csr_matrix:
        """Return a row- and gene-ordered semantic subset of this entry."""
        requested_rows = _normalize_rows(rows)
        requested_genes = _normalize_genes(genes)
        row_positions = _row_positions(self.rows, requested_rows)
        gene_positions = _semantic_gene_positions(self.genes, requested_genes)
        if gene_positions is None:
            raise ValueError("requested panel genes are not semantically contained")
        return self.matrix[row_positions][:, gene_positions].tocsr()


@dataclass(frozen=True)
class NativeCountCacheResolution:
    entry: NativeCountCacheEntry
    cache_hit: bool


class PersistentNativeCountCache:
    """Small append-only store keyed by exact rows and semantic panel genes."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _scope_root(self, domain: str, scope: str) -> Path:
        return self.root / _safe_component(domain, name="domain") / _safe_component(
            scope, name="scope"
        )

    def find(
        self,
        *,
        domain: str,
        scope: str,
        rows: pd.DataFrame,
        genes: pd.DataFrame,
        provenance: Mapping[str, Any],
    ) -> NativeCountCacheEntry | None:
        domain = _safe_component(domain, name="domain")
        scope = _safe_component(scope, name="scope")
        expected_rows = _normalize_rows(rows)
        expected_genes = _normalize_genes(genes)
        expected_provenance = _normalize_provenance(provenance)
        scope_root = self._scope_root(domain, scope)
        if not scope_root.is_dir():
            return None
        candidates = sorted(
            (
                path
                for path in scope_root.iterdir()
                if path.is_dir() and path.name.startswith("entry-")
            ),
            reverse=True,
        )
        for path in candidates:
            manifest_path = path / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                manifest.get("format") != CACHE_FORMAT
                or manifest.get("status") != "completed"
                or manifest.get("storage_format") != STORAGE_FORMAT
                or manifest.get("extraction_contract") != EXTRACTION_CONTRACT
                or manifest.get("domain") != domain
                or manifest.get("scope") != scope
                or manifest.get("provenance") != expected_provenance
                or int(manifest.get("rows", -1)) != len(expected_rows)
                or int(manifest.get("genes", -1)) < len(expected_genes)
            ):
                continue
            cached_rows = _normalize_rows(pd.read_parquet(path / "rows.parquet"))
            if not cached_rows.equals(expected_rows):
                continue
            cached_genes = _normalize_genes(pd.read_parquet(path / "genes.parquet"))
            if _semantic_gene_positions(cached_genes, expected_genes) is None:
                continue
            return self._load_entry(path, manifest, cached_rows, cached_genes)
        return None

    def get_or_build(
        self,
        *,
        domain: str,
        scope: str,
        rows: pd.DataFrame,
        genes: pd.DataFrame,
        provenance: Mapping[str, Any],
        builder: Callable[[], sp.spmatrix],
    ) -> NativeCountCacheResolution:
        cached = self.find(
            domain=domain,
            scope=scope,
            rows=rows,
            genes=genes,
            provenance=provenance,
        )
        if cached is not None:
            return NativeCountCacheResolution(cached, True)
        matrix = builder()
        entry = self.append(
            domain=domain,
            scope=scope,
            rows=rows,
            genes=genes,
            provenance=provenance,
            matrix=matrix,
        )
        return NativeCountCacheResolution(entry, False)

    def append(
        self,
        *,
        domain: str,
        scope: str,
        rows: pd.DataFrame,
        genes: pd.DataFrame,
        provenance: Mapping[str, Any],
        matrix: sp.spmatrix,
    ) -> NativeCountCacheEntry:
        domain = _safe_component(domain, name="domain")
        scope = _safe_component(scope, name="scope")
        normalized_rows = _normalize_rows(rows)
        normalized_genes = _normalize_genes(genes)
        normalized_provenance = _normalize_provenance(provenance)
        normalized_matrix = _validated_csr(
            matrix, len(normalized_rows), len(normalized_genes)
        )
        scope_root = self._scope_root(domain, scope)
        scope_root.mkdir(parents=True, exist_ok=True)
        entry_path: Path | None = None
        for attempt in range(100):
            candidate = scope_root / (
                f"entry-{time.time_ns()}-{os.getpid()}-{attempt:02d}"
            )
            try:
                candidate.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            entry_path = candidate
            break
        if entry_path is None:
            raise FileExistsError("could not allocate an append-only cache entry")

        normalized_rows.to_parquet(entry_path / "rows.parquet", index=False)
        normalized_genes.to_parquet(entry_path / "genes.parquet", index=False)
        np.save(entry_path / "data.npy", normalized_matrix.data, allow_pickle=False)
        np.save(
            entry_path / "indices.npy", normalized_matrix.indices, allow_pickle=False
        )
        np.save(
            entry_path / "indptr.npy", normalized_matrix.indptr, allow_pickle=False
        )
        manifest = {
            "format": CACHE_FORMAT,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "domain": domain,
            "scope": scope,
            "storage_format": STORAGE_FORMAT,
            "extraction_contract": EXTRACTION_CONTRACT,
            "rows": int(normalized_matrix.shape[0]),
            "genes": int(normalized_matrix.shape[1]),
            "nnz": int(normalized_matrix.nnz),
            "data_dtype": str(normalized_matrix.data.dtype),
            "indices_dtype": str(normalized_matrix.indices.dtype),
            "indptr_dtype": str(normalized_matrix.indptr.dtype),
            "row_columns": list(ROW_COLUMNS),
            "gene_columns": list(GENE_COLUMNS),
            "provenance": normalized_provenance,
            "integrity_mode": "git_and_semantic_contract",
            "append_only": True,
        }
        with (entry_path / "manifest.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return NativeCountCacheEntry(
            entry_path,
            normalized_rows,
            normalized_genes,
            normalized_matrix,
            manifest,
        )

    @staticmethod
    def _load_entry(
        path: Path,
        manifest: Mapping[str, Any],
        rows: pd.DataFrame,
        genes: pd.DataFrame,
    ) -> NativeCountCacheEntry:
        data = np.load(path / "data.npy", mmap_mode="r", allow_pickle=False)
        indices = np.load(path / "indices.npy", mmap_mode="r", allow_pickle=False)
        indptr = np.load(path / "indptr.npy", mmap_mode="r", allow_pickle=False)
        expected_rows = int(manifest["rows"])
        expected_genes = int(manifest["genes"])
        expected_nnz = int(manifest["nnz"])
        if len(rows) != expected_rows or len(genes) != expected_genes:
            raise ValueError("count-cache semantic table dimensions changed")
        if tuple(manifest.get("row_columns", ())) != ROW_COLUMNS:
            raise ValueError("count-cache row semantic contract changed")
        if tuple(manifest.get("gene_columns", ())) != GENE_COLUMNS:
            raise ValueError("count-cache gene semantic contract changed")
        if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
            raise ValueError("count-cache CSR arrays must be vectors")
        if len(data) != expected_nnz or len(indices) != expected_nnz:
            raise ValueError("count-cache CSR nnz metadata changed")
        if len(indptr) != expected_rows + 1:
            raise ValueError("count-cache CSR row pointer length changed")
        if int(indptr[0]) != 0 or int(indptr[-1]) != expected_nnz:
            raise ValueError("count-cache CSR row pointer bounds changed")
        if np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("count-cache CSR row pointers are not monotonic")
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= expected_genes):
            raise ValueError("count-cache CSR column index is outside the matrix")
        if str(data.dtype) != str(manifest["data_dtype"]):
            raise ValueError("count-cache data dtype changed")
        if str(indices.dtype) != str(manifest["indices_dtype"]):
            raise ValueError("count-cache indices dtype changed")
        if str(indptr.dtype) != str(manifest["indptr_dtype"]):
            raise ValueError("count-cache row-pointer dtype changed")
        matrix = sp.csr_matrix(
            (data, indices, indptr),
            shape=(expected_rows, expected_genes),
            copy=False,
        )
        return NativeCountCacheEntry(path, rows, genes, matrix, manifest)


__all__ = [
    "CACHE_FORMAT",
    "EXTRACTION_CONTRACT",
    "GENE_COLUMNS",
    "ROW_COLUMNS",
    "NativeCountCacheEntry",
    "NativeCountCacheResolution",
    "PersistentNativeCountCache",
]
