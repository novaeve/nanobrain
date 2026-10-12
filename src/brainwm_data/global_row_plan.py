"""Metadata-only compact row plans for the global BrainWM benchmark.

The global benchmark index is intentionally a small sample/group manifest while
the source ``cells.parquet`` tables remain in place.  A training epoch should
not have to rediscover the same source/sample boundaries by scanning those
tables.  This module records contiguous source-row intervals instead::

    (corpus_id, local_source_id, global_sample_key, split,
     row_start, row_end)

``row_end`` is exclusive.  Only the three identity columns needed to build the
plan are read from a source table; expression, labels, and sealed targets are
never opened.  The plan is an index, not a copy of cell data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .global_split import qualify_source_id

ROW_PLAN_FORMAT = "brainwm_global_compact_row_plan_v1"
ROW_PLAN_SCHEMA_VERSION = "1.0.0"
ROW_PLAN_FILE = "row_plan.parquet"
ROW_PLAN_MANIFEST = "manifest.json"
ROW_PLAN_RECEIPT = "receipt.json"
ROW_PLAN_COLUMNS = (
    "corpus_id",
    "local_source_id",
    "global_source_id",
    "local_sample_key",
    "global_sample_key",
    "split",
    "row_start",
    "row_end",
    "n_rows",
)
SPLITS = ("train", "validation", "test")
# A model-fit reader is deliberately narrower than the benchmark index.  The
# sealed test partition is never opened by this adapter; evaluators should use
# ``GlobalBenchmarkIndex`` directly after predictions have been frozen.
TRAINING_SPLITS = ("train", "validation")
ROW_SELECTION_FORMAT = "brainwm_global_row_plan_selection_v1"

_SOURCE_ALIASES = ("source_id", "source", "source_idx", "source_index")
_ROW_ALIASES = ("row_index", "obs_index", "cell_index")
_SAMPLE_ALIASES = ("sample_key", "sample_id", "sample")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "na"}:
        return None
    return text


def _integer(value: Any, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or number != math.floor(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _column(names: Iterable[str], aliases: Sequence[str], *, kind: str) -> str:
    available = set(names)
    for alias in aliases:
        if alias in available:
            return alias
    raise ValueError(f"source cells table is missing {kind} identity column")


def _path(value: str | os.PathLike[str], *, base: Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_files(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("global benchmark manifest has no files mapping")
    return files


def _corpus_entries(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("corpora")
    if entries:
        result: list[dict[str, Any]] = []
        for item in entries:
            if not isinstance(item, Mapping):
                raise ValueError("global manifest corpus entries must be mappings")
            corpus_id = _text(item.get("corpus_id"))
            artifact_dir = item.get("artifact_dir")
            if corpus_id is None or artifact_dir is None:
                raise ValueError("global manifest corpus entry lacks id/artifact_dir")
            result.append(
                {
                    "corpus_id": corpus_id,
                    "artifact_dir": str(_path(str(artifact_dir), base=root)),
                    "species": item.get("species"),
                }
            )
        return result
    artifact = manifest.get("source_artifact")
    if artifact is None:
        raise ValueError("global manifest has no source artifact")
    return [
        {
            "corpus_id": _text(manifest.get("species")) or "human",
            "artifact_dir": str(_path(str(artifact), base=root)),
            "species": manifest.get("species"),
        }
    ]


def _sample_index(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    files = _manifest_files(manifest)
    filename = files.get("samples")
    if not filename:
        raise ValueError("global manifest does not name samples.parquet")
    path = _path(str(filename), base=root)
    # This is the small global metadata table, never a source expression table.
    frame = pd.read_parquet(path)
    required = {"split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"global samples table is missing: {', '.join(sorted(missing))}")
    corpus_col = "corpus_id" if "corpus_id" in frame.columns else None
    local_col = "local_sample_key" if "local_sample_key" in frame.columns else "sample_key"
    global_col = "global_sample_key" if "global_sample_key" in frame.columns else "sample_key"
    if local_col not in frame.columns or global_col not in frame.columns:
        raise ValueError("global samples table needs local/global sample keys")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    expected_by_split: dict[str, int] = defaultdict(int)
    for row in frame.to_dict("records"):
        split = _text(row.get("split"))
        if split not in SPLITS:
            raise ValueError(f"unexpected sample split: {split!r}")
        corpus = _text(row.get(corpus_col)) if corpus_col else (_text(manifest.get("species")) or "human")
        local = _text(row.get(local_col))
        global_key = _text(row.get(global_col))
        if corpus is None or local is None or global_key is None:
            raise ValueError("global samples table has null corpus/sample identity")
        key = (corpus, local)
        if key in result:
            raise ValueError(f"duplicate global sample identity: {key!r}")
        selected = row.get("selected_cells")
        expected = None
        if selected is not None:
            try:
                number = float(selected)
                if math.isfinite(number) and number >= 0 and number == math.floor(number):
                    expected = int(number)
            except (TypeError, ValueError):
                pass
        result[key] = {
            "corpus_id": corpus,
            "local_sample_key": local,
            "global_sample_key": global_key,
            "split": split,
            "expected_rows": expected,
        }
        if expected is not None:
            expected_by_split[split] += expected
    return result, dict(expected_by_split)


def _source_artifact_map(root: Path, manifest: Mapping[str, Any]) -> dict[tuple[str, int], str]:
    """Return ``(corpus, local_source_id) -> global source id`` deterministically."""
    files = _manifest_files(manifest)
    filename = files.get("source_lookup")
    if filename:
        path = _path(str(filename), base=root)
        if path.is_file():
            frame = pd.read_parquet(path)
            corpus_col = "corpus_id" if "corpus_id" in frame.columns else None
            local_col = "local_source_id" if "local_source_id" in frame.columns else "source_id"
            global_col = "global_source_id" if "global_source_id" in frame.columns else None
            if corpus_col and local_col:
                result: dict[tuple[str, int], str] = {}
                for row in frame.to_dict("records"):
                    corpus = _text(row.get(corpus_col))
                    local = _integer(row.get(local_col), "local_source_id")
                    global_id = _text(row.get(global_col)) if global_col else None
                    if corpus is not None:
                        result[(corpus, local)] = global_id or qualify_source_id(corpus, local)
                if result:
                    return result
    result = {}
    for corpus_entry in _corpus_entries(root, manifest):
        corpus = corpus_entry["corpus_id"]
        source_path = Path(corpus_entry["artifact_dir"]) / "sources.parquet"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_frame = pd.read_parquet(source_path, columns=["source_id"])
        for value in source_frame["source_id"].tolist():
            local = _integer(value, "source_id")
            result[(corpus, local)] = qualify_source_id(corpus, local)
    return result


@dataclass
class _ActiveRange:
    corpus_id: str
    local_source_id: int
    local_sample_key: str
    global_sample_key: str
    split: str
    row_start: int
    row_end: int

    def extend(self, row_index: int) -> None:
        self.row_end = row_index + 1

    def record(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "local_source_id": self.local_source_id,
            "global_source_id": qualify_source_id(self.corpus_id, self.local_source_id),
            "local_sample_key": self.local_sample_key,
            "global_sample_key": self.global_sample_key,
            "split": self.split,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "n_rows": self.row_end - self.row_start,
        }


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("corpus_id", pa.string()),
            ("local_source_id", pa.int64()),
            ("global_source_id", pa.string()),
            ("local_sample_key", pa.string()),
            ("global_sample_key", pa.string()),
            ("split", pa.string()),
            ("row_start", pa.int64()),
            ("row_end", pa.int64()),
            ("n_rows", pa.int64()),
        ]
    )


def _iter_source_rows(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    source_col = _column(parquet.schema.names, _SOURCE_ALIASES, kind="source")
    row_col = _column(parquet.schema.names, _ROW_ALIASES, kind="row")
    sample_col = _column(parquet.schema.names, _SAMPLE_ALIASES, kind="sample")
    for batch in parquet.iter_batches(
        columns=[source_col, row_col, sample_col],
        batch_size=int(batch_size),
        use_threads=False,
    ):
        frame = batch.to_pandas()
        frame = frame.rename(
            columns={source_col: "_source_id", row_col: "_row_index", sample_col: "_sample_key"}
        )
        yield frame


def build_global_row_plan(
    benchmark_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    splits: Sequence[str] = SPLITS,
    batch_size: int = 262_144,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a compact row interval index from metadata identity columns only."""
    root = Path(benchmark_dir).resolve()
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected_splits = tuple(str(value) for value in splits)
    if not selected_splits or any(value not in SPLITS for value in selected_splits):
        raise ValueError(f"splits must be a non-empty subset of {SPLITS}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    global_manifest = _load_json(root / "manifest.json")
    sample_map, expected_by_split = _sample_index(root, global_manifest)
    source_global_ids = _source_artifact_map(root, global_manifest)
    corpus_entries = _corpus_entries(root, global_manifest)
    allowed = set(selected_splits)

    row_plan_path = output / ROW_PLAN_FILE
    schema = _schema()
    # The interval index is metadata-only but can still contain millions of
    # short ranges for sparsely selected source rows.  Zstandard keeps this
    # compact artifact small without touching the source payloads.
    writer = pq.ParquetWriter(
        row_plan_path,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
    )
    buffered: list[dict[str, Any]] = []
    active: _ActiveRange | None = None
    observed_by_sample: dict[tuple[str, str], int] = defaultdict(int)
    rows_by_split: dict[str, int] = defaultdict(int)
    rows_by_corpus_split: dict[tuple[str, str], int] = defaultdict(int)
    intervals_by_split: dict[str, int] = defaultdict(int)
    source_rows_scanned = 0
    unknown_sample_rows = 0
    source_order_violations = 0
    interval_count = 0

    def flush() -> None:
        nonlocal active, interval_count
        if active is None:
            return
        if active.split in allowed:
            record = active.record()
            buffered.append(record)
            interval_count += 1
            intervals_by_split[active.split] += 1
            if len(buffered) >= 50_000:
                writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
                buffered.clear()
        active = None

    try:
        for corpus_entry in corpus_entries:
            corpus = str(corpus_entry["corpus_id"])
            cells_path = Path(corpus_entry["artifact_dir"]) / "cells.parquet"
            if not cells_path.is_file():
                raise FileNotFoundError(cells_path)
            previous_by_source: dict[int, int] = {}
            for frame in _iter_source_rows(cells_path, batch_size):
                for source_value, row_value, sample_value in zip(
                    frame["_source_id"].tolist(),
                    frame["_row_index"].tolist(),
                    frame["_sample_key"].tolist(),
                ):
                    source_id = _integer(source_value, "source_id")
                    row_index = _integer(row_value, "row_index")
                    if source_id in previous_by_source and row_index <= previous_by_source[source_id]:
                        source_order_violations += 1
                    previous_by_source[source_id] = row_index
                    source_rows_scanned += 1
                    local_sample = _text(sample_value)
                    metadata = sample_map.get((corpus, local_sample or ""))
                    if metadata is None:
                        unknown_sample_rows += 1
                        flush()
                        continue
                    split = str(metadata["split"])
                    if split not in allowed:
                        # Still count no rows for an omitted split, but close an
                        # interval so it cannot bridge into an included split.
                        flush()
                        continue
                    key_matches = (
                        active is not None
                        and active.corpus_id == corpus
                        and active.local_source_id == source_id
                        and active.local_sample_key == metadata["local_sample_key"]
                        and active.global_sample_key == metadata["global_sample_key"]
                        and active.split == split
                        and row_index == active.row_end
                    )
                    if not key_matches:
                        flush()
                        active = _ActiveRange(
                            corpus_id=corpus,
                            local_source_id=source_id,
                            local_sample_key=str(metadata["local_sample_key"]),
                            global_sample_key=str(metadata["global_sample_key"]),
                            split=split,
                            row_start=row_index,
                            row_end=row_index + 1,
                        )
                    else:
                        active.extend(row_index)
                    observed_by_sample[(corpus, str(metadata["local_sample_key"]))] += 1
                    rows_by_split[split] += 1
                    rows_by_corpus_split[(corpus, split)] += 1
            flush()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
            buffered.clear()
    finally:
        writer.close()

    mismatches: dict[str, dict[str, int | None]] = {}
    for key, metadata in sample_map.items():
        if metadata["split"] not in allowed:
            continue
        expected = metadata.get("expected_rows")
        observed = int(observed_by_sample.get(key, 0))
        if expected is not None and expected != observed:
            mismatches[f"{key[0]}::{key[1]}"] = {"expected": expected, "observed": observed}

    split_receipt_path = root / "split_receipt.json"
    split_receipt = _load_json(split_receipt_path) if split_receipt_path.is_file() else {}
    global_split_id = split_receipt.get("split_id") or global_manifest.get("split_id")
    manifest = {
        "format": ROW_PLAN_FORMAT,
        "schema_version": ROW_PLAN_SCHEMA_VERSION,
        "generated_at_utc": _now(),
        "benchmark_manifest": str(root / "manifest.json"),
        "benchmark_format": global_manifest.get("format"),
        "global_split_id": global_split_id,
        "splits": list(selected_splits),
        "row_end_semantics": "exclusive",
        "columns": list(ROW_PLAN_COLUMNS),
        "files": {"row_plan": ROW_PLAN_FILE, "receipt": ROW_PLAN_RECEIPT},
        "corpora": corpus_entries,
        "counts": {
            "source_rows_scanned": source_rows_scanned,
            "intervals": interval_count,
            "rows_by_split": dict(rows_by_split),
            "rows_by_corpus_split": {
                f"{corpus}::{split}": count
                for (corpus, split), count in sorted(rows_by_corpus_split.items())
            },
            "intervals_by_split": dict(intervals_by_split),
            "unknown_sample_rows": unknown_sample_rows,
            "source_order_violations": source_order_violations,
            "sample_count_mismatches": len(mismatches),
        },
        "zero_copy": True,
        "expression_copied": False,
        "labels_copied": False,
        "sealed_test_payload_opened": False,
        "large_file_hashes_computed": False,
    }
    receipt = {
        "format": "brainwm_global_compact_row_plan_receipt_v1",
        "generated_at_utc": manifest["generated_at_utc"],
        "output_dir": str(output),
        "benchmark_manifest": str(root / "manifest.json"),
        "status": "passed" if not unknown_sample_rows and not mismatches else "failed",
        "validation": {
            "source_rows_scanned": source_rows_scanned,
            "intervals": interval_count,
            "unknown_sample_rows": unknown_sample_rows,
            "sample_count_mismatches": mismatches,
            "source_order_violations": source_order_violations,
            "expected_rows_by_split": expected_by_split,
            "observed_rows_by_split": dict(rows_by_split),
        },
        "checks": {
            "metadata_only": True,
            "identity_columns_only": True,
            "expression_opened": False,
            "labels_opened": False,
            "sealed_test_payload_opened": False,
            "checkpoint_modified": False,
            "large_file_hashes_computed": False,
        },
    }
    (output / ROW_PLAN_MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / ROW_PLAN_RECEIPT).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "\n".join(
            [
                "# BrainWM compact row plan",
                "",
                "This is a metadata-only, zero-copy interval index for the global benchmark.",
                "Each row is `(corpus_id, local_source_id, global_sample_key, split, row_start, row_end)`; `row_end` is exclusive.",
                "The source `cells.parquet` files and all expression/label payloads remain in place.",
                "",
                "The plan is not a training result. It only removes repeated identity-table scans from later validation runs.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return receipt


def validate_global_row_plan(
    output_dir: str | os.PathLike[str],
    *,
    benchmark_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate a compact plan from its small metadata files only."""
    output = Path(output_dir).resolve()
    manifest = _load_json(output / ROW_PLAN_MANIFEST)
    receipt = _load_json(output / ROW_PLAN_RECEIPT)
    if manifest.get("format") != ROW_PLAN_FORMAT:
        raise ValueError("unexpected compact row-plan format")
    if manifest.get("row_end_semantics") != "exclusive":
        raise ValueError("row_end semantics must be exclusive")
    if tuple(manifest.get("columns", ())) != ROW_PLAN_COLUMNS:
        raise ValueError("row-plan columns drifted")
    if receipt.get("checks", {}).get("sealed_test_payload_opened") is not False:
        raise ValueError("sealed test payload was opened")
    if receipt.get("checks", {}).get("expression_opened") is not False:
        raise ValueError("expression payload was opened")
    path = output / str(manifest.get("files", {}).get("row_plan", ROW_PLAN_FILE))
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = set(ROW_PLAN_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"row plan is missing columns: {', '.join(sorted(missing))}")
    if not frame.empty:
        for column in ("local_source_id", "row_start", "row_end", "n_rows"):
            values = pd.to_numeric(frame[column], errors="raise")
            if (values < 0).any():
                raise ValueError(f"row plan contains negative {column}")
        if not (frame["row_end"] > frame["row_start"]).all():
            raise ValueError("row plan contains empty intervals")
        if not (frame["n_rows"] == frame["row_end"] - frame["row_start"]).all():
            raise ValueError("row plan n_rows does not match interval bounds")
        ordered = frame.sort_values(
            ["corpus_id", "local_source_id", "row_start", "row_end"], kind="stable"
        )
        previous: dict[tuple[str, int], int] = {}
        for row in ordered.itertuples(index=False):
            key = (str(row.corpus_id), int(row.local_source_id))
            start = int(row.row_start)
            if key in previous and start < previous[key]:
                raise ValueError(f"row plan intervals are not ordered for {key!r}")
            previous[key] = int(row.row_end)
    counts = {
        "rows": int(frame["n_rows"].sum()) if not frame.empty else 0,
        "intervals": int(len(frame)),
        "by_split": {
            str(split): int(part["n_rows"].sum())
            for split, part in frame.groupby("split", sort=True)
        },
    }
    if benchmark_dir is not None:
        benchmark_root = Path(benchmark_dir).resolve()
        benchmark_manifest = _load_json(benchmark_root / "manifest.json")
        benchmark_receipt_path = benchmark_root / "split_receipt.json"
        benchmark_receipt = (
            _load_json(benchmark_receipt_path)
            if benchmark_receipt_path.is_file()
            else {}
        )
        benchmark_split_id = benchmark_receipt.get("split_id") or benchmark_manifest.get("split_id")
        if manifest.get("global_split_id") != benchmark_split_id:
            raise ValueError("row plan is bound to a different global split")
    return {
        "valid": True,
        "format": ROW_PLAN_FORMAT,
        "status": receipt.get("status"),
        "counts": counts,
        "metadata_only": True,
        "expression_opened": False,
        "labels_opened": False,
        "sealed_test_payload_opened": False,
    }


def _validate_training_row_plan(
    row_plan_dir: str | os.PathLike[str],
    *,
    benchmark_dir: str | os.PathLike[str] | None,
    split: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Validate only compact metadata before a training slice is opened."""
    if split not in TRAINING_SPLITS:
        raise ValueError(
            f"training row-plan selection supports only {TRAINING_SPLITS}; "
            f"got {split!r}"
        )
    output = Path(row_plan_dir).resolve()
    manifest = _load_json(output / ROW_PLAN_MANIFEST)
    receipt = _load_json(output / ROW_PLAN_RECEIPT)
    if manifest.get("format") != ROW_PLAN_FORMAT:
        raise ValueError("unexpected compact row-plan format")
    if receipt.get("status") != "passed":
        raise ValueError("cannot train from a failed compact row plan")
    if split not in set(manifest.get("splits", ())):
        raise ValueError(f"row plan does not contain split={split!r}")
    path = output / str(manifest.get("files", {}).get("row_plan", ROW_PLAN_FILE))
    if not path.is_file():
        raise FileNotFoundError(path)
    if benchmark_dir is None:
        return path, manifest, {}
    benchmark_root = Path(benchmark_dir).resolve()
    benchmark_manifest = _load_json(benchmark_root / "manifest.json")
    split_receipt_path = benchmark_root / "split_receipt.json"
    split_receipt = (
        _load_json(split_receipt_path) if split_receipt_path.is_file() else {}
    )
    benchmark_split_id = split_receipt.get("split_id") or benchmark_manifest.get(
        "split_id"
    )
    if manifest.get("global_split_id") != benchmark_split_id:
        raise ValueError("row plan is bound to a different global split")
    return path, manifest, benchmark_manifest


def iter_global_row_plan_selection(
    row_plan_dir: str | os.PathLike[str],
    split: str,
    *,
    max_rows: int,
    benchmark_dir: str | os.PathLike[str] | None = None,
    task: str | None = None,
    row_group_batch_size: int = 1,
) -> Iterator[pd.DataFrame]:
    """Yield a bounded identity-only selection from a compact row plan.

    The source ``cells.parquet`` table is never opened.  At most ``max_rows``
    source row identities are expanded, making this suitable for a validation
    vertical slice while preserving the global split and optional task
    eligibility.  Selection is a deterministic prefix in row-plan order; a
    caller that needs assay balancing can request separate slices and merge
    them before constructing a training reader.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if isinstance(row_group_batch_size, bool) or row_group_batch_size < 1:
        raise ValueError("row_group_batch_size must be a positive integer")
    path, _manifest, _benchmark_manifest = _validate_training_row_plan(
        row_plan_dir, benchmark_dir=benchmark_dir, split=split
    )
    task_keys: set[str] | None = None
    if task is not None:
        if benchmark_dir is None:
            raise ValueError("benchmark_dir is required for task-filtered selection")
        # Import lazily so metadata-only row-plan tooling remains independent
        # of the heavier global reader module at import time.
        from .global_split import GlobalBenchmarkIndex

        index = GlobalBenchmarkIndex(benchmark_dir)
        task_keys = index.sample_keys_for(split, task)

    parquet = pq.ParquetFile(path)
    columns = list(ROW_PLAN_COLUMNS)
    remaining = int(max_rows)
    for batch in parquet.iter_batches(
        columns=columns,
        batch_size=max(1, int(row_group_batch_size) * 65_536),
        use_threads=False,
    ):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        selected = frame[frame["split"].astype(str).eq(split)]
        if task_keys is not None:
            selected = selected[
                selected["global_sample_key"].astype(str).isin(task_keys)
            ]
        if selected.empty:
            continue
        records: list[pd.DataFrame] = []
        for row in selected.itertuples(index=False):
            if remaining <= 0:
                break
            start = _integer(row.row_start, "row_start")
            end = _integer(row.row_end, "row_end")
            if end <= start:
                raise ValueError("row plan contains an empty interval")
            take = min(end - start, remaining)
            local_rows = np.arange(start, start + take, dtype=np.int64)
            records.append(
                pd.DataFrame(
                    {
                        "corpus_id": str(row.corpus_id),
                        "local_source_id": int(row.local_source_id),
                        "global_source_id": str(row.global_source_id),
                        "local_sample_key": str(row.local_sample_key),
                        "global_sample_key": str(row.global_sample_key),
                        "sample_key": str(row.global_sample_key),
                        "split": split,
                        "global_split": split,
                        "row_index": local_rows,
                    }
                )
            )
            remaining -= int(take)
        if records:
            yield pd.concat(records, ignore_index=True)
        if remaining <= 0:
            break


def select_global_row_plan_rows(
    row_plan_dir: str | os.PathLike[str],
    split: str,
    *,
    max_rows: int,
    benchmark_dir: str | os.PathLike[str] | None = None,
    task: str | None = None,
    row_group_batch_size: int = 1,
) -> pd.DataFrame:
    """Collect at most ``max_rows`` identity rows into one compact frame."""
    parts = list(
        iter_global_row_plan_selection(
            row_plan_dir,
            split,
            max_rows=max_rows,
            benchmark_dir=benchmark_dir,
            task=task,
            row_group_batch_size=row_group_batch_size,
        )
    )
    if not parts:
        raise ValueError(f"row plan contains no eligible rows for split={split!r}")
    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates(
        ["corpus_id", "local_source_id", "row_index"], keep="first"
    )
    result = result.sort_values(
        ["corpus_id", "local_source_id", "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    if len(result) > max_rows:
        result = result.iloc[:max_rows].copy()
    return result


class GlobalRowPlanTrainingReader:
    """Bounded adapter exposing a global row plan to the existing trainer.

    The historical trainer expects ``reader.cells`` and integer source IDs.
    This wrapper supplies only a bounded identity frame expanded from the
    compact row plan, maps corpus-qualified source IDs to deterministic local
    aliases, and routes each expression read through
    ``GlobalBenchmarkReader``/``GlobalMultiCorpusReader``.  No legacy source
    ``split`` value is consulted for partitioning.
    """

    def __init__(
        self,
        benchmark_dir: str | os.PathLike[str],
        row_plan_dir: str | os.PathLike[str],
        *,
        max_rows_per_split: int,
        task: str | None = None,
        unified_vocabulary: str | os.PathLike[str] | None = None,
        cell_columns: Sequence[str] | None = None,
        row_group_batch_size: int = 1,
    ) -> None:
        if isinstance(max_rows_per_split, bool) or not isinstance(
            max_rows_per_split, int
        ) or max_rows_per_split < 1:
            raise ValueError("max_rows_per_split must be a positive integer")
        self.root = Path(benchmark_dir).resolve()
        self.row_plan_dir = Path(row_plan_dir).resolve()
        self.max_rows_per_split = int(max_rows_per_split)
        self.task = task
        self.row_group_batch_size = int(row_group_batch_size)
        self.index = __import__(
            "brainwm_data.global_split", fromlist=["GlobalBenchmarkIndex"]
        ).GlobalBenchmarkIndex(self.root)
        self._readers: dict[str, Any] = {}
        self._closed = False

        # Construct split-specific lazy expression readers.  They open source
        # H5AD handles only when ``read_batch`` is called.
        for split in TRAINING_SPLITS:
            kwargs: dict[str, Any] = {
                "cell_columns": cell_columns,
                "row_group_batch_size": self.row_group_batch_size,
            }
            if self.index.multi_corpus:
                if unified_vocabulary in (None, ""):
                    raise ValueError(
                        "unified_vocabulary is required for a multi-corpus row-plan reader"
                    )
                kwargs["unified_vocabulary"] = unified_vocabulary
                kwargs["allow_unmapped_unified_genes"] = False
            self._readers[split] = self.index.reader(
                split, task=task, **kwargs
            )

        self.sources, self._source_alias, self._alias_to_global = (
            self._build_source_view()
        )
        self._global_to_local = {
            str(global_id): int(local_id)
            for global_id, local_id in zip(
                self.sources["global_source_id"], self.sources["local_source_id"]
            )
        }
        self._global_to_corpus = {
            str(global_id): str(corpus)
            for global_id, corpus in zip(
                self.sources["global_source_id"], self.sources["corpus_id"]
            )
        }
        self.samples = self._build_sample_view()
        cell_parts: list[pd.DataFrame] = []
        for split in TRAINING_SPLITS:
            selected = select_global_row_plan_rows(
                self.row_plan_dir,
                split,
                max_rows=self.max_rows_per_split,
                benchmark_dir=self.root,
                task=task,
                row_group_batch_size=self.row_group_batch_size,
            )
            selected["source_id"] = selected["global_source_id"].map(
                self._source_alias
            )
            if selected["source_id"].isna().any():
                raise ValueError("row plan references a source absent from source metadata")
            selected["source_id"] = selected["source_id"].astype("int64")
            cell_parts.append(selected)
        self.cells = pd.concat(cell_parts, ignore_index=True, sort=False)
        self.cells = self.cells.sort_values(
            ["source_id", "row_index", "split"], kind="mergesort"
        ).reset_index(drop=True)
        first_reader = next(iter(self._readers.values()))
        if hasattr(first_reader, "n_genes"):
            self.n_genes = int(first_reader.n_genes)
        else:
            # ``GlobalBenchmarkReader`` intentionally keeps the historical
            # reader surface; derive its vocabulary width from the compact
            # source gene map without opening expression.
            gene_map = first_reader.gene_map
            gene_column = (
                "gene_index" if "gene_index" in gene_map
                else "namespace_gene_index"
            )
            values = pd.to_numeric(gene_map[gene_column], errors="coerce")
            if not values.notna().any():
                raise ValueError("source gene map has no valid gene indices")
            self.n_genes = int(values.max()) + 1
        # Kept for provenance emitters that inspect corpus specs.
        self.specs = tuple()

    def _build_source_view(self) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
        parts: list[pd.DataFrame] = []
        # Both split readers expose the same source inventory.  The first one
        # is sufficient and avoids another metadata pass over the NAS.
        reader = next(iter(self._readers.values()))
        raw = reader.sources.copy()
        # The single-corpus reader stores the local source ID in the index,
        # whereas the multi-corpus reader already exposes it as a column.
        if "source_id" not in raw.columns:
            raw = raw.reset_index()
            if "source_id" not in raw.columns and "index" in raw.columns:
                raw = raw.rename(columns={"index": "source_id"})
        else:
            raw = raw.reset_index(drop=True)
        if "global_source_id" not in raw:
            corpus = str(self.index.corpora[0].get("corpus_id", "human"))
            if "source_id" not in raw:
                raw["source_id"] = raw.index.astype("int64")
            raw["local_source_id"] = pd.to_numeric(
                raw["source_id"], errors="raise"
            ).astype("int64")
            raw["global_source_id"] = raw["local_source_id"].map(
                lambda value: qualify_source_id(corpus, int(value))
            )
            raw["corpus_id"] = corpus
        else:
            raw["global_source_id"] = raw["global_source_id"].astype(str)
            if "local_source_id" not in raw:
                raw["local_source_id"] = pd.to_numeric(
                    raw["source_id"], errors="raise"
                ).astype("int64")
            if "corpus_id" not in raw:
                raw["corpus_id"] = raw["global_source_id"].str.split(
                    "::source::", n=1
                ).str[0]
        parts.append(raw)
        source = pd.concat(parts, ignore_index=True, sort=False)
        source = source.sort_values("global_source_id", kind="mergesort").reset_index(
            drop=True
        )
        aliases = {
            str(value): int(index)
            for index, value in enumerate(source["global_source_id"].astype(str))
        }
        source["source_id"] = source["global_source_id"].map(aliases).astype("int64")
        source = source.set_index("source_id", drop=False)
        return source, aliases, {value: key for key, value in aliases.items()}

    def _build_sample_view(self) -> pd.DataFrame:
        samples = self.index.samples.copy()
        if "global_sample_key" not in samples:
            samples["global_sample_key"] = samples["sample_key"].astype(str)
        samples["sample_key"] = samples["global_sample_key"].astype(str)
        if "corpus" not in samples and "corpus_id" in samples:
            samples["corpus"] = samples["corpus_id"]
        elif "corpus" not in samples:
            corpus = str(self.index.corpora[0].get("corpus_id", "human"))
            samples["corpus"] = corpus
        samples["split"] = samples["split"].astype(str)
        return samples

    def read_batch(
        self,
        source_id: int,
        row_indices: np.ndarray,
        cell_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        if cell_frame is None or cell_frame.empty:
            raise ValueError("global row-plan reader requires a non-empty cell frame")
        alias = int(source_id)
        try:
            global_source = self._alias_to_global[alias]
        except KeyError as exc:
            raise KeyError(f"unknown row-plan source alias={alias}") from exc
        frame = cell_frame.copy()
        splits = set(frame.get("split", pd.Series(dtype="string")).astype(str))
        if "global_split" in frame:
            splits |= set(frame["global_split"].astype(str))
        if len(splits) != 1 or not splits.intersection(TRAINING_SPLITS):
            raise ValueError("cell frame must contain exactly one train/validation split")
        split = next(iter(splits))
        if "source_id" in frame:
            values = pd.to_numeric(frame["source_id"], errors="raise").to_numpy()
            if not np.all(values.astype(np.int64) == alias):
                raise ValueError("cell frame source_id does not match alias")
        local_source = self._global_to_local[global_source]
        delegate = frame.copy()
        delegate["source_id"] = local_source
        delegate["local_source_id"] = local_source
        delegate["global_source_id"] = global_source
        delegate["corpus_id"] = self._global_to_corpus[global_source]
        delegate["global_split"] = split
        delegate["split"] = split
        reader = self._readers[split]
        delegate_id: int | str = global_source if self.index.multi_corpus else local_source
        batch = reader.read_batch(
            delegate_id,
            np.asarray(row_indices, dtype=np.int64),
            delegate,
        )
        batch["source_id"] = alias
        batch["local_source_id"] = local_source
        batch["global_source_id"] = global_source
        batch["cells"] = frame.reset_index(drop=True)
        return batch

    def iter_local_batches(
        self,
        split: str = "train",
        batch_size: int = 256,
        *,
        shuffle_sources: bool = True,
        seed: int = 0,
    ) -> Iterator[dict[str, Any]]:
        if split not in TRAINING_SPLITS:
            raise ValueError(f"unknown training split={split!r}")
        frame = self.cells[self.cells["split"].astype(str).eq(split)]
        source_ids = frame["source_id"].drop_duplicates().to_numpy(np.int64)
        if shuffle_sources:
            np.random.default_rng(seed).shuffle(source_ids)
        for source_id in source_ids:
            part = frame[frame["source_id"].eq(int(source_id))].sort_values(
                "row_index", kind="mergesort"
            )
            for start in range(0, len(part), int(batch_size)):
                selected = part.iloc[start : start + int(batch_size)]
                yield self.read_batch(
                    int(source_id),
                    selected["row_index"].to_numpy(np.int64),
                    selected,
                )

    def close(self) -> None:
        if self._closed:
            return
        for reader in self._readers.values():
            reader.close()
        self._closed = True

    def __enter__(self) -> "GlobalRowPlanTrainingReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001,S110
            pass


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        result = validate_global_row_plan(args.output_dir, benchmark_dir=args.benchmark_dir)
    else:
        result = build_global_row_plan(
            args.benchmark_dir,
            args.output_dir,
            splits=args.splits,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


__all__ = [
    "ROW_PLAN_COLUMNS",
    "ROW_PLAN_FORMAT",
    "TRAINING_SPLITS",
    "ROW_SELECTION_FORMAT",
    "build_global_row_plan",
    "validate_global_row_plan",
    "iter_global_row_plan_selection",
    "select_global_row_plan_rows",
    "GlobalRowPlanTrainingReader",
]


if __name__ == "__main__":  # pragma: no cover
    main()
