"""Small, project-local provenance manifests for BrainWM training runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str], workspace: Path, *, text: bool = True):
    return subprocess.run(
        ["git", *command],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def git_provenance(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    commit = _git(["rev-parse", "HEAD"], root).strip()
    branch = _git(["branch", "--show-current"], root).strip()
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    diff = _git(["diff", "--binary", "HEAD"], root, text=False)
    return {
        "commit": commit,
        "branch": branch,
        "tracked_worktree_dirty": bool(status.strip()),
        "tracked_status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def _require_sha256(value: object, *, source: str) -> str:
    checksum = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(checksum):
        raise ValueError(f"invalid SHA-256 in {source}: {value!r}")
    return checksum


def _artifact_file(root: Path, raw_name: object, *, source: str) -> tuple[str, Path]:
    name = str(raw_name)
    relative = Path(name)
    if not name or relative.is_absolute():
        raise ValueError(f"artifact file path in {source} must be relative: {name!r}")
    resolved = (root / relative).resolve()
    try:
        normalized = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"artifact file path in {source} escapes artifact root: {name!r}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"artifact file declared by {source} is missing: {resolved}"
        )
    return normalized, resolved


def _parse_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        match = re.fullmatch(r"([0-9A-Fa-f]{64})[ \t]+[* ]?(.+)", raw_line)
        if match is None:
            raise ValueError(
                f"malformed SHA256SUMS entry at {path}:{line_number}"
            )
        name = match.group(2)
        if name in checksums:
            raise ValueError(f"duplicate SHA256SUMS entry: {name!r}")
        checksums[name] = match.group(1).lower()
    if not checksums:
        raise ValueError(f"SHA256SUMS is empty: {path}")
    return checksums


def _load_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid artifact manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"artifact manifest must be a JSON object: {path}")
    return payload


def _checksum_value(value: object, *, source: str) -> str | None:
    if isinstance(value, str):
        return _require_sha256(value, source=source)
    if isinstance(value, Mapping) and "sha256" in value:
        return _require_sha256(value["sha256"], source=source)
    return None


def _verified_artifact_files(
    root: Path, manifest: Mapping[str, Any] | None
) -> dict[str, Any]:
    declarations: dict[str, dict[str, Any]] = {}

    def add(name: object, checksum: object, source: str) -> None:
        normalized, _ = _artifact_file(root, name, source=source)
        value = _require_sha256(checksum, source=f"{source}:{normalized}")
        existing = declarations.get(normalized)
        if existing is not None and existing["sha256"] != value:
            raise ValueError(
                "conflicting declared SHA-256 values for "
                f"{normalized}: {existing['sha256']} != {value}"
            )
        if existing is None:
            declarations[normalized] = {
                "sha256": value,
                "checksum_sources": [source],
            }
        elif source not in existing["checksum_sources"]:
            existing["checksum_sources"].append(source)

    checksum_manifest = root / "SHA256SUMS"
    if checksum_manifest.is_file():
        for name, checksum in _parse_sha256sums(checksum_manifest).items():
            add(name, checksum, "SHA256SUMS")

    if manifest is not None:
        for field_name in ("artifact_hashes", "files"):
            raw_mapping = manifest.get(field_name)
            if raw_mapping is None:
                continue
            if not isinstance(raw_mapping, Mapping):
                raise TypeError(f"manifest.json:{field_name} must be an object")
            for name, raw_value in raw_mapping.items():
                checksum = _checksum_value(
                    raw_value, source=f"manifest.json:{field_name}.{name}"
                )
                if checksum is None:
                    raise ValueError(
                        f"manifest.json:{field_name}.{name} lacks a valid SHA-256"
                    )
                add(name, checksum, f"manifest.json:{field_name}")

    if not declarations:
        raise ValueError(
            f"artifact has no verifiable SHA-256 file inventory: {root}"
        )

    declared_sizes: dict[str, int] = {}
    if manifest is not None:
        output = manifest.get("output")
        if output is not None and not isinstance(output, Mapping):
            raise TypeError("manifest.json:output must be an object")
        raw_sizes = output.get("files") if isinstance(output, Mapping) else None
        if raw_sizes is not None:
            if not isinstance(raw_sizes, Mapping):
                raise TypeError("manifest.json:output.files must be an object")
            for name, raw_size in raw_sizes.items():
                normalized, _ = _artifact_file(
                    root, name, source="manifest.json:output.files"
                )
                if isinstance(raw_size, bool) or not isinstance(raw_size, int):
                    raise TypeError(
                        "manifest.json:output.files sizes must be integers: "
                        f"{name!r}"
                    )
                if raw_size < 0:
                    raise ValueError(
                        f"negative size in manifest.json:output.files: {name!r}"
                    )
                if normalized not in declarations:
                    raise ValueError(
                        "manifest output file lacks a declared SHA-256: "
                        f"{normalized}"
                    )
                declared_sizes[normalized] = raw_size

    verified: dict[str, dict[str, Any]] = {}
    for name, declaration in sorted(declarations.items()):
        _, candidate = _artifact_file(root, name, source="checksum inventory")
        stat = candidate.stat()
        expected_size = declared_sizes.get(name)
        if expected_size is not None and stat.st_size != expected_size:
            raise ValueError(
                f"artifact size mismatch for {name}: "
                f"expected {expected_size}, found {stat.st_size}"
            )
        actual = sha256_file(candidate)
        if actual != declaration["sha256"]:
            raise ValueError(
                f"artifact SHA-256 mismatch for {name}: "
                f"expected {declaration['sha256']}, found {actual}"
            )
        verified[name] = {
            "path": str(candidate),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": actual,
            "checksum_sources": sorted(declaration["checksum_sources"]),
        }
    return {
        "status": "verified",
        "algorithm": "sha256",
        "files": verified,
        "file_count": len(verified),
        "total_bytes": sum(item["bytes"] for item in verified.values()),
    }


def _cheap_file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()


def _source_rows_from_parquet(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "sources.parquet"
    if not path.is_file():
        return {}
    schema_names = set(pq.read_schema(path).names)
    fingerprint_column = next(
        (name for name in ("file_fingerprint", "fingerprint") if name in schema_names),
        None,
    )
    required = {"local_path", "file_size"}
    if fingerprint_column is None or not required.issubset(schema_names):
        raise ValueError(
            "zero-copy sources.parquet must contain local_path, file_size, "
            "and fingerprint/file_fingerprint"
        )
    columns = ["local_path", "file_size", fingerprint_column]
    for optional in ("source_id", "source_md5"):
        if optional in schema_names:
            columns.append(optional)
    rows = pq.read_table(path, columns=columns).to_pylist()
    if not rows:
        raise ValueError(f"zero-copy sources.parquet is empty: {path}")
    inventory: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        source_id = str(row.get("source_id", row_index))
        if source_id in inventory:
            raise ValueError(f"duplicate source_id in sources.parquet: {source_id}")
        inventory[source_id] = {
            "path": row["local_path"],
            "bytes": row["file_size"],
            "fingerprints": {
                f"sources.parquet:{fingerprint_column}": row[fingerprint_column]
            },
            "content_checksums": (
                {"sources.parquet:source_md5": row["source_md5"]}
                if row.get("source_md5") else {}
            ),
        }
    return inventory


def _source_rows_from_manifest(
    manifest: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if manifest is None or "source_hashes" not in manifest:
        return {}
    raw_sources = manifest["source_hashes"]
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ValueError("manifest.json:source_hashes must be a non-empty object")
    inventory: dict[str, dict[str, Any]] = {}
    for raw_source_id, raw_record in raw_sources.items():
        source_id = str(raw_source_id)
        if not isinstance(raw_record, Mapping):
            raise TypeError(
                f"manifest source_hashes.{source_id} must be an object"
            )
        if not raw_record.get("path") or not raw_record.get("file_fingerprint"):
            raise ValueError(
                f"manifest source_hashes.{source_id} lacks path/file_fingerprint"
            )
        inventory[source_id] = {
            "path": raw_record["path"],
            "bytes": raw_record.get("file_size"),
            "fingerprints": {
                "manifest.json:source_hashes.file_fingerprint": raw_record[
                    "file_fingerprint"
                ]
            },
            "content_checksums": (
                {
                    "manifest.json:source_hashes.source_md5": raw_record[
                        "source_md5"
                    ]
                }
                if raw_record.get("source_md5") else {}
            ),
        }
    return inventory


def _merge_source_rows(
    parquet_rows: dict[str, dict[str, Any]],
    manifest_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if parquet_rows and manifest_rows and set(parquet_rows) != set(manifest_rows):
        raise ValueError(
            "sources.parquet and manifest.json:source_hashes list different "
            "source IDs"
        )
    merged = parquet_rows or manifest_rows
    if not parquet_rows or not manifest_rows:
        return merged
    for source_id, manifest_record in manifest_rows.items():
        record = parquet_rows[source_id]
        parquet_path = Path(str(record["path"])).resolve()
        manifest_path = Path(str(manifest_record["path"])).resolve()
        if parquet_path != manifest_path:
            raise ValueError(
                f"source path disagreement for source_id {source_id}: "
                f"{parquet_path} != {manifest_path}"
            )
        manifest_size = manifest_record.get("bytes")
        if manifest_size is not None and int(record["bytes"]) != int(manifest_size):
            raise ValueError(f"source size disagreement for source_id {source_id}")
        record["fingerprints"].update(manifest_record["fingerprints"])
        record["content_checksums"].update(
            manifest_record["content_checksums"]
        )
    return parquet_rows


def _source_h5ad_inventory(
    root: Path,
    manifest: Mapping[str, Any] | None,
    verified_artifact_files: Mapping[str, Any],
) -> dict[str, Any] | None:
    parquet_path = root / "sources.parquet"
    if parquet_path.is_file() and "sources.parquet" not in verified_artifact_files:
        raise ValueError(
            "sources.parquet exists but is not covered by a verified SHA-256"
        )
    rows = _merge_source_rows(
        _source_rows_from_parquet(root),
        _source_rows_from_manifest(manifest),
    )
    if not rows:
        return None
    files: list[dict[str, Any]] = []
    for source_id, record in sorted(rows.items(), key=lambda item: item[0]):
        raw_path = Path(str(record["path"])).expanduser()
        source_path = (
            raw_path.resolve() if raw_path.is_absolute()
            else (root / raw_path).resolve()
        )
        if not source_path.is_file():
            raise FileNotFoundError(
                f"zero-copy source H5AD is missing for source_id {source_id}: "
                f"{source_path}"
            )
        stat = source_path.stat()
        expected_size = record.get("bytes")
        if expected_size is not None:
            try:
                normalized_size = int(expected_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid source size for source_id {source_id}: {expected_size!r}"
                ) from exc
            if normalized_size != stat.st_size:
                raise ValueError(
                    f"zero-copy source size mismatch for source_id {source_id}: "
                    f"expected {normalized_size}, found {stat.st_size}"
                )
        actual_fingerprint = _cheap_file_fingerprint(source_path)
        expected_fingerprints = {
            source: _require_sha256(value, source=source)
            for source, value in record["fingerprints"].items()
        }
        if len(set(expected_fingerprints.values())) != 1:
            raise ValueError(
                f"source fingerprint declarations disagree for source_id {source_id}"
            )
        expected_fingerprint = next(iter(expected_fingerprints.values()))
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                f"zero-copy source fingerprint mismatch for source_id {source_id}: "
                f"{source_path}"
            )
        checksum_values = {}
        for checksum_source, raw_value in record["content_checksums"].items():
            checksum = str(raw_value).strip().lower()
            if not _MD5_PATTERN.fullmatch(checksum):
                raise ValueError(
                    f"invalid source MD5 in {checksum_source}: {raw_value!r}"
                )
            checksum_values[checksum_source] = checksum
        if checksum_values and len(set(checksum_values.values())) != 1:
            raise ValueError(
                f"source content checksum declarations disagree for source_id {source_id}"
            )
        file_record: dict[str, Any] = {
            "source_id": source_id,
            "path": str(source_path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "identity": {
                "algorithm": "sha256(resolved_path|bytes|mtime_ns)",
                "sha256": actual_fingerprint,
                "verified": True,
                "declaration_sources": sorted(expected_fingerprints),
            },
        }
        if checksum_values:
            file_record["content_checksum"] = {
                "algorithm": "md5",
                "value": next(iter(checksum_values.values())),
                "declaration_sources": sorted(checksum_values),
                "verified_during_binding": False,
                "reason": "full payload hashing intentionally avoided",
            }
        files.append(file_record)
    return {
        "status": "identity_verified",
        "identity_scope": "resolved path, file size, and nanosecond mtime",
        "content_checksum_policy": "bind existing declarations without rehashing large H5AD payloads",
        "source_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def artifact_binding(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory is missing: {root}")
    manifest = _load_manifest(root)
    integrity = _verified_artifact_files(root, manifest)
    binding: dict[str, Any] = {
        "path": str(root),
        "integrity": integrity,
    }
    for name in (
        "manifest.json",
        "validation.json",
        "SCHEMA.json",
        "SHA256SUMS",
        "stats.json",
    ):
        candidate = root / name
        if candidate.is_file():
            binding[name] = _file_binding(candidate)
    source_inventory = _source_h5ad_inventory(
        root,
        manifest,
        integrity["files"],
    )
    if source_inventory is not None:
        binding["source_h5ad_inventory"] = source_inventory
    return binding


def _training_data_binding(
    *,
    corpus_paths: list[str | Path],
    unified_vocabulary_path: str | Path | None,
    metadata_vocabulary_path: str | Path | None,
) -> dict[str, Any]:
    return {
        "corpora": [artifact_binding(path) for path in corpus_paths],
        "unified_vocabulary": (
            artifact_binding(unified_vocabulary_path)
            if unified_vocabulary_path else None
        ),
        "metadata_vocabulary": (
            _file_binding(Path(metadata_vocabulary_path))
            if metadata_vocabulary_path else None
        ),
    }


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return payload


def _canonical_json_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _required_file_binding(
    path: str | Path,
    *,
    description: str,
) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{description} is missing: {candidate}")
    return _file_binding(candidate)


def _protocol_bindings(
    protocol_paths: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[dict[str, Any]]:
    if isinstance(protocol_paths, (str, Path)):
        raise TypeError("protocol_paths must be a sequence of paths")
    bindings = []
    seen: set[str] = set()
    for index, protocol_path in enumerate(protocol_paths or ()):
        binding = _required_file_binding(
            protocol_path,
            description=f"protocol {index}",
        )
        if binding["path"] in seen:
            raise ValueError(f"duplicate protocol path: {binding['path']}")
        seen.add(binding["path"])
        bindings.append(binding)
    return bindings


def _verify_file_binding(binding: object, *, description: str) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise TypeError(f"{description} binding must be an object")
    path = binding.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{description} binding lacks a path")
    actual = _required_file_binding(path, description=description)
    if dict(binding) != actual:
        raise ValueError(f"{description} binding drifted: {actual['path']}")
    return actual


def _data_binding_from_record(record: object) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("run manifest data binding must be an object")
    raw_corpora = record.get("corpora")
    if not isinstance(raw_corpora, list) or not raw_corpora:
        raise ValueError("run manifest data.corpora must be a non-empty list")

    def artifact_path(binding: object, *, description: str) -> str:
        if not isinstance(binding, Mapping):
            raise TypeError(f"{description} binding must be an object")
        path = binding.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{description} binding lacks a path")
        return path

    corpus_paths = [
        artifact_path(binding, description=f"corpus {index}")
        for index, binding in enumerate(raw_corpora)
    ]
    raw_unified = record.get("unified_vocabulary")
    unified_path = (
        artifact_path(raw_unified, description="unified vocabulary")
        if raw_unified is not None else None
    )
    raw_metadata = record.get("metadata_vocabulary")
    metadata_path = (
        artifact_path(raw_metadata, description="metadata vocabulary")
        if raw_metadata is not None else None
    )
    return _training_data_binding(
        corpus_paths=corpus_paths,
        unified_vocabulary_path=unified_path,
        metadata_vocabulary_path=metadata_path,
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite an existing resume manifest: {path}"
        ) from exc


def write_start_manifest(
    output_dir: str | Path,
    *,
    workspace: str | Path,
    script_path: str | Path,
    source_config_path: str | Path,
    resolved_config_path: str | Path,
    corpus_paths: list[str | Path],
    unified_vocabulary_path: str | Path | None,
    metadata_vocabulary_path: str | Path | None,
    protocol_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path:
    output = Path(output_dir)
    script = Path(script_path)
    source_config = Path(source_config_path)
    resolved_config = Path(resolved_config_path)
    manifest = {
        "format": "brainwm_training_run_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_provenance(workspace),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
        },
        "code": {"training_script": _file_binding(script)},
        "configuration": {
            "source": _file_binding(source_config),
            "resolved": _file_binding(resolved_config),
        },
        "protocols": _protocol_bindings(protocol_paths),
        "data": _training_data_binding(
            corpus_paths=corpus_paths,
            unified_vocabulary_path=unified_vocabulary_path,
            metadata_vocabulary_path=metadata_vocabulary_path,
        ),
    }
    path = output / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_resume_manifest(
    output_dir: str | Path,
    *,
    step: int,
    checkpoint_path: str | Path,
    source_config_path: str | Path,
    resolved_config_path: str | Path,
    corpus_paths: list[str | Path],
    unified_vocabulary_path: str | Path | None,
    metadata_vocabulary_path: str | Path | None,
    protocol_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path:
    """Write an immutable resume event after exact input-data verification."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("resume step must be a non-negative integer")
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"training output directory is missing: {output}")
    path = output / f"resume_manifest.step_{step:012d}.json"
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing resume manifest: {path}"
        )

    start_path = output / "run_manifest.json"
    start_binding_before = _required_file_binding(
        start_path,
        description="original run manifest",
    )
    start_payload = _json_object(
        start_path,
        description="original run manifest",
    )
    if start_payload.get("format") != "brainwm_training_run_manifest_v1":
        raise ValueError("unsupported original run manifest format")
    original_data = start_payload.get("data")
    current_data = _training_data_binding(
        corpus_paths=corpus_paths,
        unified_vocabulary_path=unified_vocabulary_path,
        metadata_vocabulary_path=metadata_vocabulary_path,
    )
    original_data_sha256 = _canonical_json_sha256(original_data)
    current_data_sha256 = _canonical_json_sha256(current_data)
    if current_data != original_data:
        raise ValueError(
            "current data binding differs from the original run manifest: "
            f"original={original_data_sha256}, current={current_data_sha256}"
        )

    original_configuration = start_payload.get("configuration")
    if not isinstance(original_configuration, Mapping):
        raise TypeError("original run manifest configuration must be an object")
    original_source = _verify_file_binding(
        original_configuration.get("source"),
        description="original source config",
    )
    original_resolved = _verify_file_binding(
        original_configuration.get("resolved"),
        description="original resolved config",
    )

    source_config_binding = _required_file_binding(
        source_config_path,
        description="resume source config",
    )
    resolved_config_binding = _required_file_binding(
        resolved_config_path,
        description="resolved resume config",
    )
    if source_config_binding != original_source:
        raise ValueError("resume source config differs from the original run")
    if resolved_config_binding["sha256"] != original_resolved["sha256"]:
        raise ValueError("resolved resume config differs from the original run")

    original_protocols = start_payload.get("protocols", [])
    if not isinstance(original_protocols, list):
        raise TypeError("original run manifest protocols must be a list")
    verified_original_protocols = [
        _verify_file_binding(binding, description=f"original protocol {index}")
        for index, binding in enumerate(original_protocols)
    ]
    protocol_bindings = _protocol_bindings(protocol_paths)
    if protocol_bindings != verified_original_protocols:
        raise ValueError("resume protocol bindings differ from the original run")

    checkpoint_binding = _required_file_binding(
        checkpoint_path,
        description="resume checkpoint",
    )
    start_binding_after = _required_file_binding(
        start_path,
        description="original run manifest",
    )
    if start_binding_after != start_binding_before:
        raise RuntimeError("original run manifest changed during resume verification")
    manifest = {
        "format": "brainwm_training_resume_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "original_run_manifest": start_binding_after,
        "checkpoint": checkpoint_binding,
        "configuration": {
            "source": source_config_binding,
            "resolved": resolved_config_binding,
        },
        "protocols": protocol_bindings,
        "data": current_data,
        "data_verification": {
            "status": "exact_match",
            "original_sha256": original_data_sha256,
            "current_sha256": current_data_sha256,
        },
    }
    _write_json_exclusive(path, manifest)
    return path


def _validate_resume_manifest(
    path: Path,
    *,
    start_binding: Mapping[str, Any],
    original_data: object,
    original_configuration: object,
    original_protocols: object,
) -> dict[str, Any]:
    payload = _json_object(path, description="resume manifest")
    if payload.get("format") != "brainwm_training_resume_manifest_v1":
        raise ValueError(f"unsupported resume manifest format: {path}")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"invalid resume step in {path}")
    expected_name = f"resume_manifest.step_{step:012d}.json"
    if path.name != expected_name:
        raise ValueError(
            f"resume manifest filename/step mismatch: {path.name} != {expected_name}"
        )
    bound_start = _verify_file_binding(
        payload.get("original_run_manifest"),
        description="original run manifest",
    )
    if bound_start != dict(start_binding):
        raise ValueError(f"resume manifest binds a different original run: {path}")
    if payload.get("data") != original_data:
        raise ValueError(f"resume manifest data differs from the original run: {path}")
    data_sha256 = _canonical_json_sha256(original_data)
    verification = payload.get("data_verification")
    expected_verification = {
        "status": "exact_match",
        "original_sha256": data_sha256,
        "current_sha256": data_sha256,
    }
    if verification != expected_verification:
        raise ValueError(f"resume data verification record is invalid: {path}")

    _verify_file_binding(
        payload.get("checkpoint"),
        description="resume checkpoint",
    )
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError(f"resume configuration binding must be an object: {path}")
    resume_source = _verify_file_binding(
        configuration.get("source"),
        description="resume source config",
    )
    resume_resolved = _verify_file_binding(
        configuration.get("resolved"),
        description="resolved resume config",
    )
    if not isinstance(original_configuration, Mapping):
        raise TypeError("original run configuration binding must be an object")
    original_source = _verify_file_binding(
        original_configuration.get("source"),
        description="original source config",
    )
    original_resolved = _verify_file_binding(
        original_configuration.get("resolved"),
        description="original resolved config",
    )
    if resume_source != original_source:
        raise ValueError(f"resume source config differs from the original run: {path}")
    if resume_resolved["sha256"] != original_resolved["sha256"]:
        raise ValueError(f"resolved resume config differs from the original run: {path}")
    protocols = payload.get("protocols")
    if not isinstance(protocols, list):
        raise TypeError(f"resume protocols binding must be a list: {path}")
    protocol_paths: set[str] = set()
    for index, binding in enumerate(protocols):
        verified = _verify_file_binding(
            binding,
            description=f"resume protocol {index}",
        )
        if verified["path"] in protocol_paths:
            raise ValueError(f"duplicate resume protocol binding: {verified['path']}")
        protocol_paths.add(verified["path"])
    if protocols != original_protocols:
        raise ValueError(f"resume protocols differ from the original run: {path}")
    return payload


def write_completion_manifest(
    output_dir: str | Path,
    *,
    step: int,
    best_validation: float,
    best_validation_metric: str,
) -> Path:
    output = Path(output_dir)
    start = output / "run_manifest.json"
    metrics = output / "metrics.jsonl"
    start_binding = _required_file_binding(
        start,
        description="original run manifest",
    )
    start_payload = _json_object(start, description="original run manifest")
    if start_payload.get("format") != "brainwm_training_run_manifest_v1":
        raise ValueError("unsupported original run manifest format")
    original_protocols = start_payload.get("protocols", [])
    if not isinstance(original_protocols, list):
        raise TypeError("original run manifest protocols must be a list")
    for index, binding in enumerate(original_protocols):
        _verify_file_binding(binding, description=f"original protocol {index}")
    resume_paths = sorted(output.glob("resume_manifest.*.json"))
    resume_manifests = {}
    if resume_paths:
        original_data = start_payload.get("data")
        original_configuration = start_payload.get("configuration")
        for resume_path in resume_paths:
            _validate_resume_manifest(
                resume_path,
                start_binding=start_binding,
                original_data=original_data,
                original_configuration=original_configuration,
                original_protocols=original_protocols,
            )
            resume_manifests[resume_path.name] = _file_binding(resume_path)
        current_data = _data_binding_from_record(original_data)
        if current_data != original_data:
            raise ValueError(
                "training data drifted after resume: "
                f"original={_canonical_json_sha256(original_data)}, "
                f"current={_canonical_json_sha256(current_data)}"
            )
    checkpoints = {}
    for name in ("best.pt", "final.pt"):
        path = output / name
        if path.is_file():
            checkpoints[name] = _file_binding(path)
    manifest = {
        "format": "brainwm_training_completion_manifest_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": int(step),
        "best_validation": float(best_validation),
        "best_validation_metric": best_validation_metric,
        "run_manifest": start_binding,
        "protocols": original_protocols,
        "resume_manifests": resume_manifests,
        "metrics": _file_binding(metrics),
        "checkpoints": checkpoints,
    }
    path = output / "completion_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
