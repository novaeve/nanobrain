"""Explicit mapping from a species-local gene namespace to a unified index.

The mouse spatial index deliberately keeps ``namespace_gene_index`` local to
its artifact.  This module is the narrow bridge to an append-only unified
vocabulary; it keys rows by the authoritative ``species_gene_key`` and never
falls back to gene symbols or positional coincidence.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

UNIFIED_VOCABULARY_FORMAT = "brainwm_unified_gene_vocabulary_v1"
UNIFIED_VOCABULARY_FORMATS = {
    UNIFIED_VOCABULARY_FORMAT,
    "brainwm_unified_gene_vocabulary_v2",
}
NAMESPACE_FORMATS = {
    "brainwm_species_gene_namespace_v1",
    "brainwm_mouse_molecular_cell_index_v1",
    "brainwm_mouse_spatial_cell_index_v1",
}
_SPECIES_GENE_PATTERNS = {
    "human": re.compile(r"^ENSG\d{11}$"),
    "mouse": re.compile(r"^ENSMUSG\d{11}$"),
}


def _remap_sparse_batch(
    batch: dict[str, Any],
    map_indices: Callable[[Any], np.ndarray],
) -> dict[str, Any]:
    """Project a sparse count batch through a local-to-unified map.

    Keeping this operation in one place is important because the human
    indexed corpus and the mouse namespace artifacts expose different local
    gene-index column names, while the model batch contract is identical.
    """
    required = {"row_ptr", "gene_index", "count", "measured_gene_index"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"batch is missing fields: {sorted(missing)}")
    local_gene = np.asarray(batch["gene_index"], dtype=np.int64)
    mapped_gene = np.asarray(map_indices(local_gene), dtype=np.int32)
    if mapped_gene.shape != local_gene.shape:
        raise ValueError("mapped gene index shape differs from local gene index")
    row_ptr = np.asarray(batch["row_ptr"], dtype=np.int64)
    counts = np.asarray(batch["count"], dtype=np.float32)
    if (
        row_ptr.ndim != 1
        or len(row_ptr) == 0
        or row_ptr[0] != 0
        or np.any(row_ptr[1:] < row_ptr[:-1])
        or row_ptr[-1] != len(local_gene)
    ):
        raise ValueError("batch row_ptr does not match sparse token length")
    if len(counts) != len(local_gene):
        raise ValueError("batch count length does not match gene_index")
    keep = mapped_gene >= 0
    if np.all(keep):
        new_row_ptr = row_ptr.copy()
        new_count = counts.copy()
        new_local = local_gene.astype(np.int32, copy=True)
        new_gene = mapped_gene
    else:
        row_indices = np.repeat(np.arange(len(row_ptr) - 1), np.diff(row_ptr))
        row_lengths = np.bincount(
            row_indices[keep], minlength=len(row_ptr) - 1
        )
        new_row_ptr = np.r_[0, np.cumsum(row_lengths)].astype(np.int64)
        new_count = counts[keep]
        new_local = local_gene[keep].astype(np.int32, copy=False)
        new_gene = mapped_gene[keep]

    remapped = dict(batch)
    remapped["row_ptr"] = new_row_ptr
    remapped["count"] = new_count
    remapped["gene_index"] = new_gene
    remapped["namespace_gene_index"] = new_local
    for mask_name in ("token_mask", "missing_token_mask", "measured_zero_mask"):
        if mask_name in batch and batch[mask_name] is not None:
            mask = np.asarray(batch[mask_name], dtype=bool)
            if mask.shape != local_gene.shape:
                raise ValueError(
                    f"batch {mask_name} length does not match gene_index"
                )
            remapped[mask_name] = mask.copy() if np.all(keep) else mask[keep]
    measured = np.asarray(map_indices(batch["measured_gene_index"]), dtype=np.int32)
    remapped["measured_gene_index"] = np.unique(measured[measured >= 0]).astype(
        np.int32
    )
    remapped["unified_gene_index"] = remapped["gene_index"]
    remapped["library_size"] = np.asarray(
        [
            new_count[start:end].sum(dtype=np.float64)
            for start, end in pairwise(new_row_ptr)
        ],
        dtype=np.float32,
    )
    return remapped


def _validate_key_rows(genes: pd.DataFrame, *, index_column: str) -> None:
    """Check key derivation and species-specific Ensembl syntax."""
    species = genes["species"].astype(str)
    ensembl = genes["ensembl_id"].astype(str).str.split(".").str[0]
    expected = species + ":" + ensembl
    if genes["species_gene_key"].astype(str).tolist() != expected.tolist():
        raise ValueError(
            f"{index_column} species_gene_key is inconsistent with species/Ensembl ID"
    )
    for species_name, pattern in _SPECIES_GENE_PATTERNS.items():
        values = ensembl[species.eq(species_name)]
        if not values.str.fullmatch(pattern.pattern).all():
            raise ValueError(
                f"{index_column} contains malformed {species_name} Ensembl IDs"
            )


class UnifiedGeneIndexAdapter:
    """Map local namespace indices into an append-only unified vocabulary.

    ``allow_unmapped=False`` is the safe default: constructing the adapter
    fails when a local namespace gene is absent from the unified artifact.
    With ``allow_unmapped=True``, those local rows map to ``-1`` and are
    filtered by :meth:`remap_batch`; this is useful for audited panel gaps but
    must be an explicit caller decision.
    """

    def __init__(
        self,
        namespace_artifact: str | Path,
        unified_vocabulary: str | Path,
        *,
        allow_unmapped: bool = False,
    ) -> None:
        self.namespace_root = Path(namespace_artifact)
        self.unified_root = Path(unified_vocabulary)
        self.allow_unmapped = bool(allow_unmapped)
        self._validate_artifact_paths()

        self.namespace_genes = pd.read_parquet(self.namespace_root / "genes.parquet")
        self.unified_genes = pd.read_parquet(self.unified_root / "genes.parquet")
        self._validate_namespace_genes(self.namespace_genes)
        self._validate_unified_genes(self.unified_genes)

        global_by_key = self.unified_genes.set_index("species_gene_key")["gene_index"]
        mapped = self.namespace_genes["species_gene_key"].map(global_by_key)
        missing = self.namespace_genes.loc[mapped.isna(), "species_gene_key"].astype(str)
        self.missing_namespace_keys = tuple(missing.tolist())
        if self.missing_namespace_keys and not self.allow_unmapped:
            preview = ", ".join(self.missing_namespace_keys[:5])
            suffix = "..." if len(self.missing_namespace_keys) > 5 else ""
            raise ValueError(
                "namespace genes are absent from the unified vocabulary: "
                f"{preview}{suffix}; pass allow_unmapped=True to drop them explicitly"
            )

        local_indices = self.namespace_genes["namespace_gene_index"].to_numpy(np.int64)
        self.local_to_unified = np.full(len(self.namespace_genes), -1, dtype=np.int32)
        valid = mapped.notna().to_numpy()
        self.local_to_unified[local_indices[valid]] = mapped.loc[valid].to_numpy(np.int32)
        self.n_unified_genes = len(self.unified_genes)

    @property
    def namespace_size(self) -> int:
        return int(self.local_to_unified.size)

    @property
    def mapped_namespace_genes(self) -> int:
        return int(np.count_nonzero(self.local_to_unified >= 0))

    def _validate_artifact_paths(self) -> None:
        for root in (self.namespace_root, self.unified_root):
            if not root.is_dir():
                raise FileNotFoundError(root)
            for name in ("manifest.json", "genes.parquet"):
                if not (root / name).is_file():
                    raise FileNotFoundError(root / name)
        namespace_manifest = json.loads(
            (self.namespace_root / "manifest.json").read_text(encoding="utf-8")
        )
        if namespace_manifest.get("format") not in NAMESPACE_FORMATS:
            raise ValueError(
                "unsupported namespace artifact format: "
                f"{namespace_manifest.get('format')!r}"
            )
        unified_manifest = json.loads(
            (self.unified_root / "manifest.json").read_text(encoding="utf-8")
        )
        if unified_manifest.get("format") not in UNIFIED_VOCABULARY_FORMATS:
            raise ValueError(
                "unsupported unified vocabulary format: "
                f"{unified_manifest.get('format')!r}"
            )

    @staticmethod
    def _validate_namespace_genes(genes: pd.DataFrame) -> None:
        required = {
            "namespace_gene_index", "species", "species_id", "ensembl_id",
            "species_gene_key",
        }
        missing = required - set(genes.columns)
        if missing:
            raise ValueError(f"namespace genes are missing columns: {sorted(missing)}")
        indices = genes["namespace_gene_index"].to_numpy(np.int64)
        if not np.array_equal(indices, np.arange(len(genes), dtype=np.int64)):
            raise ValueError("namespace_gene_index must be contiguous and ordered")
        if set(genes["species"].astype(str)) != {"mouse"}:
            raise ValueError("unified adapter currently accepts only mouse namespace rows")
        if set(genes["species_id"].astype(int)) != {1}:
            raise ValueError("mouse namespace species_id must be exactly 1")
        if genes["ensembl_id"].isna().any():
            raise ValueError("namespace Ensembl IDs must be non-null")
        if genes["species_gene_key"].isna().any() or genes["species_gene_key"].duplicated().any():
            raise ValueError("namespace species_gene_key must be non-null and unique")
        _validate_key_rows(genes, index_column="namespace")

    @staticmethod
    def _validate_unified_genes(genes: pd.DataFrame) -> None:
        required = {
            "gene_index", "species", "species_id", "ensembl_id", "species_gene_key",
        }
        missing = required - set(genes.columns)
        if missing:
            raise ValueError(f"unified genes are missing columns: {sorted(missing)}")
        indices = genes["gene_index"].to_numpy(np.int64)
        if not np.array_equal(indices, np.arange(len(genes), dtype=np.int64)):
            raise ValueError("unified gene_index must be contiguous and ordered")
        if genes["species_gene_key"].isna().any() or genes["species_gene_key"].duplicated().any():
            raise ValueError("unified species_gene_key must be non-null and unique")
        if genes["ensembl_id"].isna().any() or genes["species"].isna().any():
            raise ValueError("unified species and Ensembl IDs must be non-null")
        _validate_key_rows(genes, index_column="unified")
        human = genes["species"].astype(str).eq("human")
        if human.any():
            human_ids = genes.loc[human, "species_id"].astype(int)
            if not (human_ids == 0).all():
                raise ValueError("unified human rows must use species_id=0")
            human_indices = genes.loc[human, "gene_index"].to_numpy(np.int64)
            if not np.array_equal(human_indices, np.arange(len(human_indices), dtype=np.int64)):
                raise ValueError("unified vocabulary must preserve the human prefix")
            nonhuman = ~human
            if nonhuman.any() and int(genes.loc[nonhuman, "gene_index"].min()) < len(human_indices):
                raise ValueError("non-human unified rows must be append-only after the human prefix")
        mouse = genes["species"].astype(str).eq("mouse")
        if not mouse.any():
            raise ValueError("unified vocabulary contains no mouse rows")
        mouse_ids = genes.loc[mouse, "species_id"].astype(int)
        if not (mouse_ids == 1).all():
            raise ValueError("unified mouse rows must use species_id=1")

    def map_indices(self, namespace_indices: Any) -> np.ndarray:
        """Map an array of local indices; ``-1`` remains the invalid sentinel."""
        values = np.asarray(namespace_indices)
        if not np.issubdtype(values.dtype, np.integer):
            if not np.all(np.equal(values, np.asarray(values, dtype=np.int64))):
                raise ValueError("namespace gene indices must be integer-valued")
            values = values.astype(np.int64)
        else:
            values = values.astype(np.int64, copy=False)
        if np.any(values < -1) or np.any(values >= self.namespace_size):
            raise ValueError("namespace gene index is outside the local vocabulary")
        result = np.full(values.shape, -1, dtype=np.int32)
        valid = values >= 0
        result[valid] = self.local_to_unified[values[valid]]
        if not self.allow_unmapped and np.any(result[valid] < 0):
            raise ValueError("namespace gene index resolves to an unmapped unified gene")
        return result

    def remap_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of a sparse reader batch in unified gene-index space.

        Unmapped tokens are removed together with their counts, and row
        pointers/library sizes are rebuilt so the sparse batch remains valid.
        The local token IDs are retained as ``namespace_gene_index`` for audit.
        """
        return _remap_sparse_batch(batch, self.map_indices)


class UnifiedGeneTableAdapter:
    """Map a generic artifact-local gene table into a unified vocabulary.

    ``UnifiedGeneIndexAdapter`` is intentionally strict about the mouse
    namespace artifact format.  Human indexed corpora predate that format and
    expose ``gene_index`` plus ``ensembl_id`` in ``genes.parquet`` instead.
    This table-level adapter handles that legacy surface without requiring a
    manifest or assuming that local and unified positions happen to match.
    """

    def __init__(
        self,
        namespace_genes: pd.DataFrame,
        unified_vocabulary: str | Path,
        *,
        species: str,
        allow_unmapped: bool = False,
        local_index_column: str | None = None,
    ) -> None:
        if not isinstance(namespace_genes, pd.DataFrame):
            raise TypeError("namespace_genes must be a pandas DataFrame")
        self.namespace_genes = namespace_genes.copy()
        self.species = str(species).strip().lower()
        if not self.species:
            raise ValueError("species must be non-empty")
        self.allow_unmapped = bool(allow_unmapped)
        if local_index_column is None:
            local_index_column = (
                "namespace_gene_index"
                if "namespace_gene_index" in self.namespace_genes.columns
                else "gene_index"
            )
        if local_index_column not in self.namespace_genes.columns:
            raise ValueError(
                f"namespace genes are missing local index column {local_index_column!r}"
            )
        self.local_index_column = local_index_column
        required = {local_index_column, "ensembl_id"}
        missing = required - set(self.namespace_genes.columns)
        if missing:
            raise ValueError(f"namespace genes are missing columns: {sorted(missing)}")
        local_indices = pd.to_numeric(
            self.namespace_genes[local_index_column], errors="raise"
        ).to_numpy(np.int64)
        if not np.array_equal(local_indices, np.arange(len(local_indices), dtype=np.int64)):
            raise ValueError(f"{local_index_column} must be contiguous and ordered")
        if "species" in self.namespace_genes:
            observed = self.namespace_genes["species"].dropna().astype(str).str.lower()
            if not observed.empty and set(observed) != {self.species}:
                raise ValueError(
                    f"namespace genes contain species other than {self.species!r}"
                )
        ids = self.namespace_genes["ensembl_id"].astype("string").str.split(".").str[0]
        if ids.isna().any() or ids.str.strip().eq("").any():
            raise ValueError("namespace Ensembl IDs must be non-null and non-empty")
        keys = self.species + ":" + ids.astype(str)
        if keys.duplicated().any():
            raise ValueError("namespace species_gene_key must be unique")

        unified_root = Path(unified_vocabulary).resolve()
        vocab_path = unified_root / "genes.parquet"
        if not vocab_path.is_file():
            raise FileNotFoundError(vocab_path)
        self.unified_genes = pd.read_parquet(vocab_path)
        UnifiedGeneIndexAdapter._validate_unified_genes(self.unified_genes)
        global_by_key = self.unified_genes.set_index("species_gene_key")["gene_index"]
        mapped = keys.map(global_by_key)
        missing_keys = keys.loc[mapped.isna()].astype(str)
        self.missing_namespace_keys = tuple(missing_keys.tolist())
        if self.missing_namespace_keys and not self.allow_unmapped:
            preview = ", ".join(self.missing_namespace_keys[:5])
            suffix = "..." if len(self.missing_namespace_keys) > 5 else ""
            raise ValueError(
                "namespace genes are absent from the unified vocabulary: "
                f"{preview}{suffix}; pass allow_unmapped=True to drop them explicitly"
            )
        self.local_to_unified = np.full(len(local_indices), -1, dtype=np.int32)
        valid = mapped.notna().to_numpy()
        self.local_to_unified[valid] = mapped.loc[valid].to_numpy(np.int32)
        self.n_unified_genes = int(len(self.unified_genes))

    @property
    def namespace_size(self) -> int:
        return int(self.local_to_unified.size)

    @property
    def mapped_namespace_genes(self) -> int:
        return int(np.count_nonzero(self.local_to_unified >= 0))

    def map_indices(self, namespace_indices: Any) -> np.ndarray:
        values = np.asarray(namespace_indices)
        if not np.issubdtype(values.dtype, np.integer):
            if not np.all(np.equal(values, np.asarray(values, dtype=np.int64))):
                raise ValueError("namespace gene indices must be integer-valued")
            values = values.astype(np.int64)
        else:
            values = values.astype(np.int64, copy=False)
        if np.any(values < -1) or np.any(values >= self.namespace_size):
            raise ValueError("namespace gene index is outside the local vocabulary")
        result = np.full(values.shape, -1, dtype=np.int32)
        valid = values >= 0
        result[valid] = self.local_to_unified[values[valid]]
        if not self.allow_unmapped and np.any(result[valid] < 0):
            raise ValueError("namespace gene index resolves to an unmapped unified gene")
        return result

    def remap_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return _remap_sparse_batch(batch, self.map_indices)


__all__ = [
    "UNIFIED_VOCABULARY_FORMAT",
    "UNIFIED_VOCABULARY_FORMATS",
    "UnifiedGeneIndexAdapter",
    "UnifiedGeneTableAdapter",
]
