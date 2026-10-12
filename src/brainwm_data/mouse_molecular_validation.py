"""Streaming validation for a candidate mouse molecular cell index."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .h5ad_io import file_fingerprint
from .mouse_molecular_index import (
    ALLOWED_ASSAYS,
    CELL_COLUMNS,
    MOUSE_MOLECULAR_INDEX_FORMAT,
    SPLIT_UNIT,
    MouseMolecularBatchReader,
    resolve_mouse_molecular_artifact_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def validate_mouse_molecular_index(
    artifact_dir: str | Path,
    *,
    smoke_rows_per_source: int = 4,
    parquet_batch_rows: int = 100_000,
) -> dict[str, Any]:
    """Validate metadata in streaming batches and backed-read every source."""
    root = resolve_mouse_molecular_artifact_path(artifact_dir)
    if smoke_rows_per_source < 1 or parquet_batch_rows < 1:
        raise ValueError("smoke/parquet batch row counts must be positive")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != MOUSE_MOLECULAR_INDEX_FORMAT:
        raise ValueError("unexpected mouse molecular index format")
    if manifest.get("candidate_only") is not True:
        raise ValueError("molecular index must remain candidate-only")
    if manifest.get("formal_training_allowed") is not False:
        raise ValueError("formal training must remain fail-closed")
    if int(manifest.get("expression_copy_bytes", -1)) != 0:
        raise ValueError("manifest reports copied expression bytes")

    required = {
        "cells.parquet", "samples.parquet", "sources.parquet", "genes.parquet",
        "source_gene_map.parquet", "species_vocabulary.json", "SCHEMA.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"mouse molecular index missing files: {missing}")
    expression_suffixes = {".h5", ".h5ad", ".hdf5", ".npy", ".npz", ".mtx"}
    copied_expression = sorted(
        path.name for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in expression_suffixes
    )
    if copied_expression:
        raise ValueError(f"artifact contains expression-like payloads: {copied_expression}")
    for name, expected in manifest.get("artifact_hashes", {}).items():
        actual = _sha256(root / name)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {name}")

    schema = json.loads((root / "SCHEMA.json").read_text())
    if schema.get("format") != MOUSE_MOLECULAR_INDEX_FORMAT:
        raise ValueError("SCHEMA format does not match the manifest")
    vocabulary = json.loads((root / "species_vocabulary.json").read_text())
    if vocabulary.get("species") != {"human": 0, "mouse": 1}:
        raise ValueError("species vocabulary must preserve human=0 and mouse=1")
    if vocabulary.get("unknown_policy") != "error":
        raise ValueError("unknown species policy must be error")

    samples = pd.read_parquet(root / "samples.parquet")
    sources = pd.read_parquet(root / "sources.parquet")
    genes = pd.read_parquet(root / "genes.parquet")
    mapping = pd.read_parquet(root / "source_gene_map.parquet")
    if samples.empty or sources.empty or genes.empty:
        raise ValueError("samples, sources and genes must be non-empty")
    if samples["sample_key"].duplicated().any():
        raise ValueError("samples.parquet contains duplicate global sample keys")
    if sources["source_id"].duplicated().any():
        raise ValueError("sources.parquet contains duplicate source IDs")
    if not sources["species"].astype(str).eq("mouse").all():
        raise ValueError("sources contain a non-mouse species")
    if not sources["assay"].astype(str).isin(ALLOWED_ASSAYS).all():
        raise ValueError("sources contain an unsupported assay")
    if not sources["matrix_encoding"].astype(str).isin(
        {"csr_matrix", "csc_matrix", "array"}
    ).all():
        raise ValueError("sources contain an unsupported matrix encoding")
    if not samples["split_unit"].astype(str).eq(SPLIT_UNIT).all():
        raise ValueError("samples use an unexpected split unit")
    if not samples["split"].astype(str).isin({"train", "validation", "test"}).all():
        raise ValueError("samples contain an unexpected split")
    expected_gene_index = np.arange(len(genes), dtype=np.int64)
    if not np.array_equal(
        genes["namespace_gene_index"].to_numpy(np.int64), expected_gene_index
    ):
        raise ValueError("namespace_gene_index must be contiguous and ordered")
    valid_mapping = mapping["namespace_gene_index"] >= 0
    if valid_mapping.any():
        mapped = mapping.loc[valid_mapping]
        if not mapped["species"].astype(str).eq("mouse").all():
            raise ValueError("mapped genes contain a non-mouse species")
        if int(mapped["namespace_gene_index"].max()) >= len(genes):
            raise ValueError("source map references an out-of-range namespace gene")

    source_by_id = sources.set_index("source_id", drop=False)
    source_ids = set(source_by_id.index.astype(int))
    for source_id, source in source_by_id.iterrows():
        path = Path(str(source["local_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_fingerprint(path) != str(source["file_fingerprint"]):
            raise ValueError(f"source fingerprint changed after indexing: {path}")

    sample_split = samples.set_index("sample_key")["split"].astype(str).to_dict()
    sample_counts: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    assay_counts: Counter[str] = Counter()
    last_row_by_source: dict[int, int] = {}
    smoke_rows: dict[int, list[int]] = defaultdict(list)
    parquet = pq.ParquetFile(root / "cells.parquet")
    if parquet.schema_arrow.names != CELL_COLUMNS:
        raise ValueError("cells.parquet columns do not match the compact schema")
    scan_columns = [
        "source_id", "row_index", "species", "species_id", "assay",
        "sample_key", "split", "split_unit",
    ]
    scanned_rows = 0
    for record_batch in parquet.iter_batches(
        batch_size=parquet_batch_rows, columns=scan_columns
    ):
        frame = record_batch.to_pandas()
        scanned_rows += len(frame)
        if not frame["species"].astype(str).eq("mouse").all():
            raise ValueError("cells contain a non-mouse species")
        if not frame["species_id"].astype(int).eq(1).all():
            raise ValueError("cells contain an invalid species ID")
        if not frame["assay"].astype(str).isin(ALLOWED_ASSAYS).all():
            raise ValueError("cells contain an unsupported assay")
        if not frame["split_unit"].astype(str).eq(SPLIT_UNIT).all():
            raise ValueError("cells use an unexpected split unit")
        if not frame["source_id"].astype(int).isin(source_ids).all():
            raise ValueError("cells reference an unknown source")
        expected_splits = frame["sample_key"].map(sample_split)
        if expected_splits.isna().any():
            raise ValueError("cells reference a sample absent from samples.parquet")
        if not expected_splits.astype(str).eq(frame["split"].astype(str)).all():
            raise ValueError("global sample split is inconsistent in cells.parquet")
        for source_id, part in frame.groupby("source_id", sort=False):
            sid = int(source_id)
            rows = part["row_index"].to_numpy(np.int64)
            if np.any(rows < 0) or np.any(np.diff(rows) <= 0):
                raise ValueError(f"source {sid} row indices are not strictly increasing")
            if sid in last_row_by_source and int(rows[0]) <= last_row_by_source[sid]:
                raise ValueError(f"source {sid} row order regressed across parquet batches")
            if int(rows[-1]) >= int(source_by_id.loc[sid, "n_obs"]):
                raise ValueError(f"source {sid} row index exceeds raw/X shape")
            last_row_by_source[sid] = int(rows[-1])
            source_counts[sid] += len(rows)
            remaining = smoke_rows_per_source - len(smoke_rows[sid])
            if remaining > 0:
                smoke_rows[sid].extend(int(value) for value in rows[:remaining])
        sample_counts.update(
            {str(key): int(value) for key, value in frame["sample_key"].value_counts().items()}
        )
        split_counts.update(
            {str(key): int(value) for key, value in frame["split"].value_counts().items()}
        )
        assay_counts.update(
            {str(key): int(value) for key, value in frame["assay"].value_counts().items()}
        )

    if scanned_rows != int(parquet.metadata.num_rows):
        raise ValueError("streamed row count differs from Parquet metadata")
    if scanned_rows != int(manifest["stats"]["cells"]):
        raise ValueError("cell count differs from manifest")
    declared_samples = samples.set_index("sample_key")["selected_cells"].astype(int).to_dict()
    if sample_counts != Counter(declared_samples):
        raise ValueError("cells/samples selected cell counts differ")
    declared_sources = sources.set_index("source_id")["selected_cells"].astype(int).to_dict()
    if source_counts != Counter(declared_sources):
        raise ValueError("cells/sources selected cell counts differ")
    if set(smoke_rows) != source_ids:
        raise ValueError("not every source supplied backed-read smoke rows")

    smoke_cells = pd.DataFrame([
        {
            "source_id": source_id,
            "row_index": row_index,
            "split": "smoke",
        }
        for source_id in sorted(smoke_rows)
        for row_index in smoke_rows[source_id]
    ])
    reader = MouseMolecularBatchReader(root, cells=smoke_cells)
    backed_results = []
    try:
        for source_id in sorted(source_ids):
            selected = np.asarray(smoke_rows[source_id], dtype=np.int64)
            cell_frame = smoke_cells[smoke_cells["source_id"] == source_id]
            started = time.perf_counter()
            batch = reader.read_batch(source_id, selected, cell_frame)
            elapsed = time.perf_counter() - started
            row_ptr = np.asarray(batch["row_ptr"], dtype=np.int64)
            counts = np.asarray(batch["count"], dtype=np.float32)
            tokens = np.asarray(batch["namespace_gene_index"], dtype=np.int64)
            if len(row_ptr) != len(selected) + 1 or row_ptr[0] != 0:
                raise ValueError(f"source {source_id} returned an invalid row_ptr")
            if np.any(np.diff(row_ptr) < 0) or row_ptr[-1] != len(counts):
                raise ValueError(f"source {source_id} returned inconsistent CSR offsets")
            if np.any(counts < 0) or not np.isfinite(counts).all():
                raise ValueError(f"source {source_id} returned invalid raw counts")
            if len(counts) and not np.allclose(counts, np.rint(counts), atol=1e-6):
                raise ValueError(f"source {source_id} raw count target is non-integral")
            if np.any(tokens < 0) or np.any(tokens >= len(genes)):
                raise ValueError(f"source {source_id} returned an invalid namespace token")
            if not np.array_equal(batch["gene_index"], batch["namespace_gene_index"]):
                raise ValueError(f"source {source_id} namespace alias is inconsistent")
            if np.any(np.asarray(batch["library_size"]) < 0):
                raise ValueError(f"source {source_id} returned a negative library size")
            source = source_by_id.loc[source_id]
            backed_results.append({
                "source_id": source_id,
                "assay": str(source["assay"]),
                "matrix_encoding": str(source["matrix_encoding"]),
                "rows": len(selected),
                "tokens": len(counts),
                "seconds": float(elapsed),
                "rows_per_second": float(len(selected) / max(elapsed, 1e-12)),
                "status": "passed",
            })
    finally:
        reader.close()

    report = {
        "format": MOUSE_MOLECULAR_INDEX_FORMAT,
        "artifact": str(root.resolve()),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "candidate_only": True,
        "formal_training_allowed": False,
        "cells": int(scanned_rows),
        "samples": len(samples),
        "sources": len(sources),
        "genes": len(genes),
        "split_cells": _counter_dict(split_counts),
        "assay_cells": _counter_dict(assay_counts),
        "backed_read": {
            "status": "passed_all_sources",
            "sources_checked": len(backed_results),
            "assays_checked": sorted({row["assay"] for row in backed_results}),
            "expression_element": "raw/X",
            "results": backed_results,
        },
        "expression_copy_bytes": 0,
        "morton_or_order_array_bytes": 0,
        "index_sha256": manifest.get("index_sha256"),
        "candidate_source_md5_set_sha256": manifest.get("source_integrity", {}).get(
            "candidate_source_md5_set_sha256"
        ),
        "blocking_gates": manifest.get("training_gate", {}),
    }
    (root / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["validation"] = {
        "status": "passed",
        "backed_read": "passed_all_sources",
        "sources_checked": len(backed_results),
        "assays_checked": report["backed_read"]["assays_checked"],
        "report": str((root / "validation.json").resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    throughput = [
        f"- source `{row['source_id']}` ({row['assay']}): `{row['rows']}` rows, "
        f"`{row['tokens']}` tokens, `{row['rows_per_second']:.1f}` rows/s."
        for row in backed_results
    ]
    (root / "reflection.md").write_text(
        "# Mouse Molecular Cell Index Reflection\n\n"
        f"- Validation passed for `{scanned_rows:,}` cells, `{len(samples):,}` global "
        f"samples and `{len(sources)}` sources.\n"
        f"- Assay coverage: `{_counter_dict(assay_counts)}`.\n"
        f"- Backed raw-count smoke passed on all `{len(backed_results)}` sources and both "
        "scRNA/snRNA assays.\n"
        "- Expression copy bytes: `0`; Morton/order array bytes: `0`.\n"
        "- Formal training remains blocked by requested-origin and paper provenance.\n"
        "- Donor-held-out and Stage1 temporal supervision remain blocked by missing metadata.\n\n"
        "## Backed-read smoke\n\n"
        + "\n".join(throughput)
        + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["validate_mouse_molecular_index"]
