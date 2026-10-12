"""Species-aware gene namespace planning without cell/expression conversion."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

from .h5ad_io import (
    file_fingerprint,
    matrix_encoding,
    matrix_shape,
    read_column,
)


NAMESPACE_FORMAT = "brainwm_species_gene_namespace_v1"
SPECIES_PATTERNS = {
    "human": re.compile(r"^ENSG\d{11}$"),
    "mouse": re.compile(r"^ENSMUSG\d{11}$"),
}
BLANK_CONTROL_RE = re.compile(r"^Blank-\d+$", re.IGNORECASE)
MISSING_FEATURE_FIELDS = ("homology_group", "esm2_embedding_index")
# The candidate mouse exports use ``gene_identifier`` consistently, but two
# 10x files omit ``gene_symbol`` and retain the equivalent cleaned-name field.
# Keep identity anchored to an Ensembl column; these aliases only supply the
# human-readable annotation carried in the namespace tables.
GENE_ID_COLUMNS = (
    "gene_identifier", "ensembl_id_clean", "ensembl_id", "gene_id", "_index",
)
GENE_SYMBOL_COLUMNS = (
    "gene_symbol", "gene_name_clean", "gene_name", "symbol",
)


def normalize_species_gene_id(value: object, species: str) -> str | None:
    """Normalize a versioned Ensembl ID under an explicit species namespace."""
    if species not in SPECIES_PATTERNS:
        raise ValueError(f"unsupported species namespace: {species}")
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().split(".", 1)[0]
    return text if SPECIES_PATTERNS[species].fullmatch(text) else None


def species_gene_key(species: str, ensembl_id: str) -> str:
    normalized = normalize_species_gene_id(ensembl_id, species)
    if normalized is None:
        raise ValueError(f"invalid {species} Ensembl gene ID: {ensembl_id!r}")
    return f"{species}:{normalized}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gencode(path: Path, species: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype="string")
    required = {"gene_name", "ensembl_id", "gene_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing GENCODE columns: {sorted(missing)}")
    frame = frame.copy()
    frame["ensembl_id"] = frame["ensembl_id"].map(
        lambda value: normalize_species_gene_id(value, species)
    )
    frame = frame.dropna(subset=["ensembl_id"])
    if frame["ensembl_id"].duplicated().any():
        duplicate = frame.loc[frame["ensembl_id"].duplicated(), "ensembl_id"].iloc[0]
        raise ValueError(f"GENCODE contains duplicate normalized ID: {duplicate}")
    return frame[["ensembl_id", "gene_name", "gene_type"]].reset_index(drop=True)


def _read_checksums(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        checksum, name = parts
        name = name.lstrip("*")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", checksum):
            continue
        basename = Path(name).name
        if basename in result and result[basename] != checksum.lower():
            raise ValueError(f"conflicting MD5 entries for {basename}")
        result[basename] = checksum.lower()
    return result


def _matrix_dtype(node: h5py.Group | h5py.Dataset) -> str:
    return str(node["data"].dtype if isinstance(node, h5py.Group) else node.dtype)


def _resolve_var_column(
    var: h5py.Group,
    candidates: tuple[str, ...],
    *,
    required: bool,
    path: Path,
    role: str,
) -> str | None:
    for name in candidates:
        if name in var:
            return name
    if required:
        raise KeyError(
            f"{path} raw/var lacks a usable {role} column; "
            f"tried {list(candidates)}"
        )
    return None


def _platform(dataset: str) -> tuple[str | None, str]:
    if dataset.startswith("MERFISH-"):
        return "MERFISH", "declared_in_collection_name"
    if dataset.startswith("Zhuang-ABCA-"):
        return "MERFISH", "catalog_inferred_unverified"
    return None, "not_available"


def _mapping_status(raw_id: str, normalized: str | None,
                    gencode_ids: set[str]) -> str:
    if normalized is not None:
        return "mapped_gencode" if normalized in gencode_ids else "valid_ensembl_unannotated"
    if BLANK_CONTROL_RE.fullmatch(raw_id):
        return "excluded_blank_control"
    return "rejected_non_species_gene_id"


def _load_species_vocabulary(path: Path | None, species: str,
                             species_id: int) -> dict[str, int]:
    vocabulary = {"human": 0}
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocabulary = {str(key): int(value) for key, value in payload["species"].items()}
    if species in vocabulary and vocabulary[species] != species_id:
        raise ValueError(
            f"species {species!r} already has ID {vocabulary[species]}, not {species_id}"
        )
    owners = {value: name for name, value in vocabulary.items()}
    if species_id in owners and owners[species_id] != species:
        raise ValueError(
            f"species ID {species_id} is already assigned to {owners[species_id]!r}"
        )
    vocabulary[species] = species_id
    if vocabulary.get("human") != 0:
        raise ValueError("existing human species ID must remain 0 for checkpoint compatibility")
    return vocabulary


def plan_species_gene_namespace(
    h5ad_paths: Iterable[str | Path],
    gencode_path: str | Path,
    *,
    species: str,
    species_id: int,
    expression_element: str = "raw/X",
    checksum_manifest: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Plan a local species namespace by reading only source shape and ``var``.

    The returned ``namespace_gene_index`` is local to this artifact. It is not
    a replacement for, or extension of, the current human model vocabulary.
    """
    paths = sorted(Path(path).resolve() for path in h5ad_paths)
    if not paths:
        raise FileNotFoundError("no H5AD paths supplied")
    if species != "mouse":
        raise ValueError("the current dry-run contract only permits species='mouse'")
    if species_id < 0:
        raise ValueError("species_id must be non-negative")
    gencode_path = Path(gencode_path)
    gencode = _read_gencode(gencode_path, species)
    gencode_lookup = gencode.set_index("ensembl_id")[["gene_name", "gene_type"]].to_dict("index")
    gencode_ids = set(gencode_lookup)
    checksums = _read_checksums(Path(checksum_manifest) if checksum_manifest else None)

    source_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    symbols: dict[str, Counter[str]] = defaultdict(Counter)
    presence: Counter[str] = Counter()
    for source_id, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            if expression_element not in handle:
                raise KeyError(f"{path} does not contain {expression_element}")
            expression = handle[expression_element]
            n_obs, n_vars = matrix_shape(expression)
            if "raw" not in handle or "var" not in handle["raw"]:
                raise KeyError(f"{path} does not contain raw/var")
            var = handle["raw/var"]
            id_column = _resolve_var_column(
                var, GENE_ID_COLUMNS, required=True, path=path, role="gene identifier"
            )
            symbol_column = _resolve_var_column(
                var, GENE_SYMBOL_COLUMNS, required=False, path=path, role="gene symbol"
            )
            raw_ids = [str(value) for value in read_column(var, id_column)]
            # A missing symbol is allowed: the Ensembl ID remains the sole
            # identity key, and an empty annotation is preferable to an
            # invented symbol.  Current WMB Isocortex files use gene_name_clean.
            gene_symbols = (
                [str(value) for value in read_column(var, symbol_column)]
                if symbol_column is not None
                else [""] * len(raw_ids)
            )
            if len(raw_ids) != n_vars or len(gene_symbols) != n_vars:
                raise ValueError(f"{path} raw/var length does not match {expression_element}")
            source_genes: set[str] = set()
            for source_col, (raw_id, symbol) in enumerate(zip(raw_ids, gene_symbols)):
                normalized = normalize_species_gene_id(raw_id, species)
                status = _mapping_status(raw_id, normalized, gencode_ids)
                if normalized is not None:
                    source_genes.add(normalized)
                    symbols[normalized][symbol] += 1
                annotation = gencode_lookup.get(normalized or "", {})
                map_rows.append({
                    "source_id": np.int32(source_id),
                    "source_col": np.int32(source_col),
                    "species": species,
                    "species_id": np.int16(species_id),
                    "source_gene_id": raw_id,
                    "ensembl_id": normalized,
                    "species_gene_key": (
                        species_gene_key(species, normalized) if normalized else None
                    ),
                    "namespace_gene_index": np.int32(-1),
                    "gene_symbol": symbol,
                    "gene_type": annotation.get("gene_type"),
                    "mapping_status": status,
                })
            presence.update(source_genes)
            dataset = path.parent.name
            platform, platform_status = _platform(dataset)
            source_rows.append({
                "source_id": np.int32(source_id),
                "species": species,
                "species_id": np.int16(species_id),
                "dataset": dataset,
                "local_path": str(path),
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "file_fingerprint": file_fingerprint(path),
                "source_md5": checksums.get(path.name),
                "expression_element": expression_element,
                "gene_identifier_column": id_column,
                "gene_symbol_column": symbol_column,
                "matrix_encoding": matrix_encoding(expression),
                "matrix_dtype": _matrix_dtype(expression),
                "n_obs": n_obs,
                "n_vars": n_vars,
                "has_spatial": "obsm" in handle and "spatial" in handle["obsm"],
                "platform": platform,
                "platform_status": platform_status,
                "paper_provenance_status": "unverified_missing_h5ad_identifier",
            })

    ordered_ids = sorted(symbols)
    local_index = {ensembl_id: index for index, ensembl_id in enumerate(ordered_ids)}
    gene_rows = []
    for ensembl_id in ordered_ids:
        annotation = gencode_lookup.get(ensembl_id, {})
        symbol_counts = symbols[ensembl_id]
        preferred_symbol = sorted(symbol_counts, key=lambda value: (-symbol_counts[value], value))[0]
        if not preferred_symbol.strip():
            preferred_symbol = str(annotation.get("gene_name") or ensembl_id)
        gene_rows.append({
            "namespace_gene_index": np.int32(local_index[ensembl_id]),
            "species": species,
            "species_id": np.int16(species_id),
            "ensembl_id": ensembl_id,
            "species_gene_key": species_gene_key(species, ensembl_id),
            "gene_symbol": preferred_symbol,
            "gene_type": annotation.get("gene_type"),
            "annotation_status": (
                "mapped_gencode" if ensembl_id in gencode_ids else "valid_ensembl_unannotated"
            ),
            "homology_group": None,
            "esm2_embedding_index": None,
            "source_count": np.int16(presence[ensembl_id]),
        })
    genes = pd.DataFrame(gene_rows)
    source_gene_map = pd.DataFrame(map_rows)
    valid = source_gene_map["ensembl_id"].notna()
    source_gene_map.loc[valid, "namespace_gene_index"] = (
        source_gene_map.loc[valid, "ensembl_id"].map(local_index).astype(np.int32)
    )
    sources = pd.DataFrame(source_rows)
    for frame in (genes, source_gene_map, sources):
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].astype("string")
    genes["esm2_embedding_index"] = pd.array(
        [pd.NA] * len(genes), dtype="Int64"
    )
    status_counts = source_gene_map["mapping_status"].value_counts().to_dict()
    manifest = {
        "format": NAMESPACE_FORMAT,
        "dry_run": True,
        "species": species,
        "species_id": species_id,
        "namespace_scope": "mouse_only_local_indices",
        "human_gene_vocabulary_modified": False,
        "expression_or_observation_rows_read": False,
            "scan_scope": [
            f"{expression_element} shape/encoding/dtype",
            "raw/var gene identifier and symbol aliases",
            "obsm/spatial presence",
            "file stat and supplied checksum sidecar",
        ],
        "normalization": {
            "primary_key": "species_gene_key",
            "mouse_pattern": SPECIES_PATTERNS["mouse"].pattern,
            "ensembl_version_policy": "strip suffix after first dot",
            "blank_control_policy": "exclude from namespace; preserve in source map",
        },
        "inputs": {
            "gencode": str(gencode_path.resolve()),
            "gencode_sha256": _sha256(gencode_path),
            "checksum_manifest": (
                str(Path(checksum_manifest).resolve()) if checksum_manifest else None
            ),
            "sources": [str(path) for path in paths],
        },
        "stats": {
            "sources": len(sources),
            "cells_declared_by_shape": int(sources["n_obs"].sum()),
            "source_feature_rows": len(source_gene_map),
            "namespace_genes": len(genes),
            "gencode_annotated_namespace_genes": int(
                (genes["annotation_status"] == "mapped_gencode").sum()
            ),
            "gencode_unannotated_namespace_genes": int(
                (genes["annotation_status"] == "valid_ensembl_unannotated").sum()
            ),
            "source_mapping_status": {
                str(key): int(value) for key, value in status_counts.items()
            },
        },
        "missing_features": {
            "homology_group": "not supplied; all values null",
            "esm2_embedding_index": "not supplied; all values null",
        },
    }
    return genes, source_gene_map, sources, manifest


def _schema() -> dict[str, Any]:
    return {
        "format": NAMESPACE_FORMAT,
        "scope": "gene namespace dry-run only; no cells or expression payload",
        "primary_key": ["species", "ensembl_id"],
        "species_gene_key": "<species>:<version-stripped Ensembl gene ID>",
        "files": {
            "genes.parquet": "local mouse namespace with GENCODE gene_type",
            "source_gene_map.parquet": "source column to local namespace index; excluded rows retained with status",
            "sources.parquet": "explicit species and H5AD shape/encoding identity",
            "species_vocabulary.json": "proposed append-only species IDs with unknown_policy=error",
            "manifest.json": "inputs, scan boundary, counts, hashes, and disk estimate",
        },
        "missing_but_reserved": {
            "homology_group": "nullable string",
            "esm2_embedding_index": "nullable integer",
        },
        "forbidden": [
            "treat namespace_gene_index as the current human model gene_index",
            "map mouse genes by case-folded symbols",
            "encode unknown species as human",
        ],
    }


def write_species_gene_namespace(
    h5ad_paths: Iterable[str | Path],
    gencode_path: str | Path,
    output_dir: str | Path,
    *,
    species: str = "mouse",
    species_id: int = 1,
    existing_vocabulary_path: str | Path | None = None,
    expression_element: str = "raw/X",
    checksum_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Write a new namespace artifact and refuse to overwrite any path."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite namespace artifact: {output}")
    vocabulary_path = Path(existing_vocabulary_path) if existing_vocabulary_path else None
    species_vocabulary = _load_species_vocabulary(vocabulary_path, species, species_id)
    genes, source_gene_map, sources, manifest = plan_species_gene_namespace(
        h5ad_paths,
        gencode_path,
        species=species,
        species_id=species_id,
        expression_element=expression_element,
        checksum_manifest=checksum_manifest,
    )
    output.mkdir(parents=True, exist_ok=False)
    genes.to_parquet(output / "genes.parquet", index=False)
    source_gene_map.to_parquet(output / "source_gene_map.parquet", index=False)
    sources.to_parquet(output / "sources.parquet", index=False)
    (output / "species_vocabulary.json").write_text(
        json.dumps({
            "format": "brainwm_species_vocabulary_v1",
            "species": species_vocabulary,
            "required_species": species,
            "required_species_id": species_id,
            "unknown_policy": "error",
            "source_vocabulary": (
                str(vocabulary_path.resolve()) if vocabulary_path else None
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "SCHEMA.json").write_text(
        json.dumps(_schema(), indent=2) + "\n", encoding="utf-8"
    )
    artifact_files = [
        output / "genes.parquet",
        output / "source_gene_map.parquet",
        output / "sources.parquet",
        output / "species_vocabulary.json",
        output / "SCHEMA.json",
    ]
    cells = int(manifest["stats"]["cells_declared_by_shape"])
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output"] = {
        "path": str(output.resolve()),
        "files": {path.name: path.stat().st_size for path in artifact_files},
        "artifact_bytes_before_manifest": sum(path.stat().st_size for path in artifact_files),
    }
    manifest["disk_estimate"] = {
        "existing_source_h5ad_bytes": int(sources["file_size"].sum()),
        "namespace_artifact_bytes_before_manifest": manifest["output"]["artifact_bytes_before_manifest"],
        "estimated_future_cells_parquet_bytes": int(cells * (763_069_225 / 42_837_963)),
        "estimated_future_morton_order_bytes": int(cells * np.dtype(np.int64).itemsize),
        "recommended_future_zero_copy_reserve_bytes": 600_000_000,
        "expression_copy_bytes": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_species_gene_namespace(root: str | Path) -> dict[str, Any]:
    """Validate namespace identity and source-column coverage without H5AD reads."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    vocabulary = json.loads(
        (root / "species_vocabulary.json").read_text(encoding="utf-8")
    )
    genes = pd.read_parquet(root / "genes.parquet")
    mapping = pd.read_parquet(root / "source_gene_map.parquet")
    sources = pd.read_parquet(root / "sources.parquet")
    if manifest.get("format") != NAMESPACE_FORMAT or manifest.get("dry_run") is not True:
        raise ValueError("invalid namespace format or dry_run declaration")
    species = str(manifest["species"])
    species_id = int(manifest["species_id"])
    if vocabulary.get("unknown_policy") != "error":
        raise ValueError("species vocabulary must reject unknown values")
    if int(vocabulary["species"].get(species, -1)) != species_id:
        raise ValueError("species vocabulary ID does not match manifest")
    if vocabulary["species"].get("human") != 0:
        raise ValueError("human species ID must remain 0")
    if species == "mouse" and species_id == 0:
        raise ValueError("mouse must not share the human species ID")
    expected_indices = np.arange(len(genes), dtype=np.int64)
    if not np.array_equal(genes["namespace_gene_index"].to_numpy(np.int64), expected_indices):
        raise ValueError("namespace gene indices must be contiguous and sorted")
    if genes["species_gene_key"].duplicated().any():
        raise ValueError("species_gene_key must be unique")
    if not (genes["species"] == species).all() or not (
        genes["species_id"].astype(int) == species_id
    ).all():
        raise ValueError("genes contain a species identity mismatch")
    expected_keys = [species_gene_key(species, value) for value in genes["ensembl_id"]]
    if genes["species_gene_key"].tolist() != expected_keys:
        raise ValueError("species_gene_key does not match species and Ensembl ID")
    if mapping[["source_id", "source_col"]].duplicated().any():
        raise ValueError("source mapping contains duplicate source columns")
    expected_source_rows = sources.set_index("source_id")["n_vars"].astype(int).to_dict()
    actual_source_rows = mapping.groupby("source_id").size().astype(int).to_dict()
    if actual_source_rows != expected_source_rows:
        raise ValueError("source mapping does not cover every source column exactly once")
    if not (mapping["species"] == species).all() or not (
        mapping["species_id"].astype(int) == species_id
    ).all():
        raise ValueError("source mapping contains a species identity mismatch")
    indexed = mapping["namespace_gene_index"] >= 0
    if not mapping.loc[~indexed, "mapping_status"].isin({
        "excluded_blank_control", "rejected_non_species_gene_id"
    }).all():
        raise ValueError("an eligible Ensembl gene is missing a namespace index")
    gene_by_index = genes.set_index("namespace_gene_index")["species_gene_key"]
    mapped_keys = mapping.loc[indexed, "namespace_gene_index"].map(gene_by_index)
    if mapped_keys.tolist() != mapping.loc[indexed, "species_gene_key"].tolist():
        raise ValueError("source mapping index resolves to the wrong species gene key")
    if genes["homology_group"].notna().any():
        raise ValueError("dry-run homology_group must remain null until an orthology source exists")
    if genes["esm2_embedding_index"].notna().any():
        raise ValueError("dry-run ESM2 indices must remain null until embeddings exist")
    if (root / "cells.parquet").exists():
        raise ValueError("gene namespace dry-run must not contain cells.parquet")
    return {
        "format": NAMESPACE_FORMAT,
        "species": species,
        "species_id": species_id,
        "sources": len(sources),
        "source_feature_rows": len(mapping),
        "namespace_genes": len(genes),
        "gencode_annotated_namespace_genes": int(
            (genes["annotation_status"] == "mapped_gencode").sum()
        ),
        "excluded_source_features": int((~indexed).sum()),
        "valid": True,
    }
