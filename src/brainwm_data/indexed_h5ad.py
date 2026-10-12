"""Batch access to raw counts from a zero-copy indexed H5AD corpus."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .h5ad_io import matrix_encoding, matrix_shape


class IndexedH5ADBatchReader:
    """Read sparse count-token batches described by a built artifact.

    Batches must contain a single ``source_id``. The returned gene indices are
    global indices from ``genes.parquet``. This reader intentionally returns
    raw counts; normalization/masking belongs in the model pipeline.
    """

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
    ):
        self.root = Path(artifact_dir)
        self.cell_columns = self._normalize_cell_columns(cell_columns)
        self.cell_sample_keys = self._normalize_filter_values(
            cell_sample_keys, name="cell_sample_keys"
        )
        self.cell_splits = self._normalize_filter_values(
            cell_splits, name="cell_splits"
        )
        self.cell_source_ids = self._normalize_source_ids(cell_source_ids)
        if not isinstance(allow_empty_cells, bool):
            raise TypeError("allow_empty_cells must be boolean")
        self.allow_empty_cells = allow_empty_cells
        # An explicitly empty frame is used by split-aware lazy readers as a
        # placeholder: it suppresses loading the potentially huge cells table
        # while ``cell_source_ids`` still narrows the source/gene-map metadata.
        # Keep rejecting filters alongside a populated frame (or sample/split
        # filters in all explicit-frame cases), where the semantics would be
        # ambiguous.
        if cells is not None and (
            self.cell_sample_keys is not None
            or self.cell_splits is not None
            or (
                self.cell_source_ids is not None
                and not cells.empty
            )
        ):
            raise ValueError(
                "cell filters cannot be combined with an explicit cells frame"
            )
        sources = pd.read_parquet(self.root / "sources.parquet")
        if self.cell_source_ids is not None:
            available = set(pd.to_numeric(sources["source_id"], errors="raise"))
            missing = sorted(set(self.cell_source_ids) - available)
            if missing:
                raise ValueError(f"requested source IDs are absent: {missing}")
            sources = sources[sources["source_id"].isin(self.cell_source_ids)].copy()
        self.sources = sources.set_index("source_id")
        gene_filters = (
            [("source_id", "in", list(self.cell_source_ids))]
            if self.cell_source_ids is not None
            else None
        )
        self.gene_map = pd.read_parquet(
            self.root / "source_gene_map.parquet", filters=gene_filters
        )
        # Read-only cohort audits can provide an exact, already filtered cell
        # frame.  The default remains the historical full index for training
        # and callers that plan over complete splits.
        self.cells = (
            self._read_cells(
                self.root / "cells.parquet",
                self.cell_columns,
                self.cell_sample_keys,
                self.cell_splits,
                self.cell_source_ids,
            )
            if cells is None else cells
        )
        if self.cells.empty and not self.allow_empty_cells:
            raise ValueError("cell filters selected no observations")
        self._handles: dict[int, h5py.File] = {}
        self._column_to_gene: dict[int, np.ndarray] = {}

    @staticmethod
    def _normalize_cell_columns(
        cell_columns: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        """Validate an optional projection of the cell metadata table.

        The four key columns are needed by every indexed reader.  Optional
        columns are resolved against each artifact's schema at read time so a
        shared projection can be used for human and mouse artifacts with
        different metadata columns.
        """
        if cell_columns is None:
            return None
        if isinstance(cell_columns, (str, bytes)):
            raise TypeError("cell_columns must be a sequence of column names")
        required = ("source_id", "row_index", "sample_key", "split")
        if any(not isinstance(value, str) for value in cell_columns):
            raise TypeError("cell_columns entries must be strings")
        values = tuple(value.strip() for value in cell_columns)
        if any(not value for value in values):
            raise ValueError("cell_columns entries must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError("cell_columns entries must be unique")
        missing = sorted(set(required) - set(values))
        if missing:
            raise ValueError(
                "cell_columns must include required columns: "
                + ", ".join(missing)
            )
        return values

    @staticmethod
    def _normalize_filter_values(
        values: Collection[str] | None, *, name: str
    ) -> tuple[str, ...] | None:
        if values is None:
            return None
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must be a collection of strings")
        if any(not isinstance(value, str) for value in values):
            raise TypeError(f"{name} entries must be strings")
        normalized = tuple(sorted({str(value).strip() for value in values}))
        if not normalized or any(not value for value in normalized):
            raise ValueError(f"{name} must contain non-empty strings")
        return normalized

    @staticmethod
    def _normalize_source_ids(
        values: Collection[int] | None,
    ) -> tuple[int, ...] | None:
        if values is None:
            return None
        if isinstance(values, (str, bytes)):
            raise TypeError("cell_source_ids must be a collection of integers")
        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("cell_source_ids entries must be integers")
            integer = int(value)
            if integer < 0:
                raise ValueError("cell_source_ids entries must be nonnegative")
            normalized.append(integer)
        if not normalized:
            raise ValueError("cell_source_ids must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("cell_source_ids entries must be unique")
        return tuple(sorted(normalized))

    @staticmethod
    def _read_cells(
        path: Path,
        cell_columns: tuple[str, ...] | None,
        cell_sample_keys: tuple[str, ...] | None = None,
        cell_splits: tuple[str, ...] | None = None,
        cell_source_ids: tuple[int, ...] | None = None,
    ) -> pd.DataFrame:
        filters = []
        if cell_sample_keys is not None:
            filters.append(("sample_key", "in", list(cell_sample_keys)))
        if cell_splits is not None:
            filters.append(("split", "in", list(cell_splits)))
        if cell_source_ids is not None:
            filters.append(("source_id", "in", list(cell_source_ids)))
        if cell_columns is None:
            return pd.read_parquet(path, filters=filters or None)
        # Artifacts intentionally do not share every annotation column.  Read
        # the intersection while preserving the requested order; the mixed
        # reader fills absent optional columns from source/sample context.
        available = set(pq.read_schema(path).names)
        missing_required = {
            name for name in ("source_id", "row_index", "sample_key", "split")
            if name not in available
        }
        if missing_required:
            raise ValueError(
                f"{path} is missing required cell columns: "
                + ", ".join(sorted(missing_required))
            )
        selected = [name for name in cell_columns if name in available]
        return pd.read_parquet(
            path,
            columns=selected,
            filters=filters or None,
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _gene_lookup(self, source_id: int) -> np.ndarray:
        source_id = int(source_id)
        if source_id not in self._column_to_gene:
            source = self.sources.loc[source_id]
            mapping = self.gene_map[self.gene_map["source_id"] == source_id]
            lookup = np.full(int(source["n_vars"]), -1, dtype=np.int32)
            valid = mapping["gene_index"] >= 0
            lookup[mapping.loc[valid, "source_col"].to_numpy(np.int64)] = \
                mapping.loc[valid, "gene_index"].to_numpy(np.int32)
            self._column_to_gene[source_id] = lookup
        return self._column_to_gene[source_id]

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            pass

    def _handle(self, source_id: int) -> h5py.File:
        source_id = int(source_id)
        if source_id not in self._handles:
            self._handles[source_id] = h5py.File(
                self.sources.loc[source_id, "local_path"], "r"
            )
        return self._handles[source_id]

    def cells_for_split(self, split: str) -> pd.DataFrame:
        return self.cells[self.cells["split"] == split].copy()

    def iter_local_batches(self, split: str, batch_size: int = 256,
                           shuffle_sources: bool = True,
                           seed: int = 0) -> Iterator[dict[str, Any]]:
        """Yield source-local batches; row locality avoids pathological HDF5 I/O."""
        frame = self.cells_for_split(split)
        source_ids = frame["source_id"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(seed)
        if shuffle_sources:
            rng.shuffle(source_ids)
        for source_id in source_ids:
            part = frame[frame["source_id"] == source_id].sort_values("row_index")
            for start in range(0, len(part), batch_size):
                batch = part.iloc[start:start + batch_size]
                yield self.read_batch(source_id, batch["row_index"].to_numpy(np.int64), batch)

    def read_batch(self, source_id: int, row_indices: np.ndarray,
                   cell_frame: pd.DataFrame | None = None) -> dict[str, Any]:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if len(row_indices) == 0:
            raise ValueError("empty batch")
        if np.any(row_indices[1:] <= row_indices[:-1]):
            raise ValueError("row_indices must be strictly increasing")
        source = self.sources.loc[int(source_id)]
        col_to_gene = self._gene_lookup(source_id)

        handle = self._handle(source_id)
        node = handle[source["expression_element"]]
        encoding = matrix_encoding(node)
        if encoding == "csr_matrix":
            token_gene, token_count, row_ptr = self._read_csr(node, row_indices, col_to_gene)
        elif encoding == "csc_matrix":
            token_gene, token_count, row_ptr = self._read_csc(node, row_indices, col_to_gene)
        elif encoding == "array":
            token_gene, token_count, row_ptr = self._read_dense(node, row_indices, col_to_gene)
        else:
            raise NotImplementedError(f"unsupported matrix encoding: {encoding}")

        # The model space G contains only valid global Ensembl tokens, so its
        # library size must be sum_g x_g over that same projected space.
        library_size = np.asarray([
            token_count[row_ptr[i]:row_ptr[i + 1]].sum(dtype=np.float64)
            for i in range(len(row_indices))
        ], dtype=np.float32)
        return {
            "source_id": int(source_id),
            "row_index": row_indices,
            "row_ptr": row_ptr,
            "gene_index": token_gene,
            "count": token_count,
            "library_size": library_size,
            "measured_gene_index": np.unique(col_to_gene[col_to_gene >= 0]),
            "cells": cell_frame.reset_index(drop=True) if cell_frame is not None else None,
        }

    @staticmethod
    def _coalesce(rows: np.ndarray, genes: np.ndarray, values: np.ndarray,
                  n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = genes >= 0
        rows, genes, values = rows[keep], genes[keep], values[keep]
        nonzero = values != 0
        rows, genes, values = rows[nonzero], genes[nonzero], values[nonzero]
        if len(values):
            order = np.lexsort((genes, rows))
            rows, genes, values = rows[order], genes[order], values[order]
            new = np.r_[True, (rows[1:] != rows[:-1]) | (genes[1:] != genes[:-1])]
            starts = np.flatnonzero(new)
            values = np.add.reduceat(values.astype(np.float32), starts)
            rows, genes = rows[starts], genes[starts]
        counts = np.bincount(rows, minlength=n_rows) if len(rows) else np.zeros(n_rows, dtype=np.int64)
        row_ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        return genes.astype(np.int32), values.astype(np.float32), row_ptr

    @classmethod
    def _read_csr(cls, node: h5py.Group, selected: np.ndarray,
                  col_to_gene: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = np.asarray(node["indptr"][selected], dtype=np.int64)
        ends = np.asarray(node["indptr"][selected + 1], dtype=np.int64)
        # HDF5 reads are expensive on the shared NAS. Coalesce nearby selected
        # rows while bounding how many intervening tokens are read. Temporal
        # population batches are local in row order but may contain small gaps
        # inherited from corpus downsampling.
        max_gap_tokens = 250_000
        max_span_tokens = 2_000_000
        block_starts = [0]
        current_start = 0
        for index in range(1, len(selected)):
            gap = int(starts[index] - ends[index - 1])
            span = int(ends[index] - starts[current_start])
            if gap > max_gap_tokens or span > max_span_tokens:
                block_starts.append(index)
                current_start = index
        block_starts = np.asarray(block_starts, dtype=np.int64)
        block_ends = np.r_[block_starts[1:], len(selected)]
        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []
        for left, right in zip(block_starts, block_ends):
            start = int(starts[left])
            end = int(ends[right - 1])
            block_cols = np.asarray(node["indices"][start:end], dtype=np.int64)
            block_values = np.asarray(node["data"][start:end], dtype=np.float32)
            local_starts = starts[left:right] - start
            local_ends = ends[left:right] - start
            col_chunks.extend([
                block_cols[int(item_start):int(item_end)]
                for item_start, item_end in zip(local_starts, local_ends)
            ])
            value_chunks.extend([
                block_values[int(item_start):int(item_end)]
                for item_start, item_end in zip(local_starts, local_ends)
            ])
            row_chunks.append(np.repeat(
                np.arange(left, right, dtype=np.int32),
                ends[left:right] - starts[left:right],
            ))
        rows = np.concatenate(row_chunks) if row_chunks else np.empty(0, np.int32)
        cols = np.concatenate(col_chunks) if col_chunks else np.empty(0, np.int64)
        vals = np.concatenate(value_chunks) if value_chunks else np.empty(0, np.float32)
        return cls._coalesce(rows, col_to_gene[cols], vals, len(selected))

    @classmethod
    def _read_csc(cls, node: h5py.Group, selected: np.ndarray,
                  col_to_gene: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The cohort has one small CSC source. Scan it once for a requested
        # batch; callers should keep batches source-local and reasonably large.
        source_rows = np.asarray(node["indices"][:], dtype=np.int64)
        data = np.asarray(node["data"][:], dtype=np.float32)
        indptr = np.asarray(node["indptr"][:], dtype=np.int64)
        row_lookup = np.full(matrix_shape(node)[0], -1, dtype=np.int32)
        row_lookup[selected] = np.arange(len(selected), dtype=np.int32)
        hit = row_lookup[source_rows] >= 0
        cols = np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr))[hit]
        rows = row_lookup[source_rows[hit]]
        return cls._coalesce(rows, col_to_gene[cols], data[hit], len(selected))

    @classmethod
    def _read_dense(cls, node: h5py.Dataset, selected: np.ndarray,
                    col_to_gene: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = np.asarray(node[selected])
        rows, cols = np.nonzero(array)
        return cls._coalesce(rows, col_to_gene[cols], array[rows, cols], len(selected))
