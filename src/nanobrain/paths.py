"""Canonical locations for the BrainWM human NumPy corpus.

The nanoBrain checkout is intentionally small.  The 42.8M-row expression
corpus lives in one shared location and is consumed through NumPy memmaps;
this module keeps path discovery explicit and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "validation", "test")
SPLIT_ID = "BRAINWM-HUMAN-GLOBAL-6-2-2-V3"
DEFAULT_EXTERNAL_ROOT = Path("/data/brainwm/BrainWM-Training-Benchmark/assets/v3_numpy_shards")
DEFAULT_NAS_ROOT = Path("/mnt/zim_rider/Brain-Word-Model/brainwm_training_benchmark/assets/v3_numpy_shards")


@dataclass(frozen=True)
class DatasetLocation:
    """A validated corpus location and the receipts used to validate it."""

    root: Path
    source: str
    split_manifests: dict[str, dict[str, Any]]

    @property
    def rows(self) -> dict[str, int]:
        return {split: int(self.split_manifests[split].get("rows", 0)) for split in SPLITS}

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    @property
    def complete(self) -> bool:
        return all(bool(self.split_manifests[split].get("complete")) for split in SPLITS)

    def as_dict(self) -> dict[str, Any]:
        split_bytes = {split: 0 for split in SPLITS}
        split_file_count = {split: 0 for split in SPLITS}
        total_bytes = 0
        total_file_count = 0
        try:
            for path in self.root.iterdir():
                if not path.is_file():
                    continue
                size = path.stat().st_size
                total_bytes += size
                total_file_count += 1
                name = path.name
                for split in SPLITS:
                    if name.startswith(f"{split}-shard-"):
                        split_bytes[split] += size
                        split_file_count[split] += 1
                        break
        except OSError:
            total_bytes = 0
            total_file_count = 0
            split_bytes = {split: 0 for split in SPLITS}
            split_file_count = {split: 0 for split in SPLITS}

        manifest_bytes = {
            split: self.split_manifests[split].get("bytes")
            for split in SPLITS
        }
        payload = {
            "root": str(self.root),
            "source": self.source,
            "split_id": SPLIT_ID,
            "complete": self.complete,
            "rows": self.rows,
            "total_rows": self.total_rows,
            "shards": {
                split: int(self.split_manifests[split].get("shards", 0))
                for split in SPLITS
            },
            "bytes": {
                split: int(manifest_bytes[split])
                if manifest_bytes[split] is not None
                else split_bytes[split]
                for split in SPLITS
            },
            "bytes_gib": {
                split: round((int(manifest_bytes[split])
                              if manifest_bytes[split] is not None
                              else split_bytes[split]) / 2**30, 3)
                for split in SPLITS
            },
            "split_file_count": split_file_count,
            "load": "numpy_memmap_then_torch",
            "h5ad_in_training_loop": False,
        }
        payload["file_count"] = total_file_count or None
        payload["bytes_on_disk"] = total_bytes or None
        payload["bytes_on_disk_gib"] = round(total_bytes / 2**30, 3) if total_bytes else None
        return payload


def _normalise_candidate(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    # Accept either the shard directory or the benchmark directory in the
    # environment/CLI.  This makes copied checkouts portable.
    if path.name == "BrainWM-Training-Benchmark":
        path = path / "assets" / "v3_numpy_shards"
    elif path.name == "assets":
        path = path / "v3_numpy_shards"
    return path.resolve()


def _manifest(path: Path, split: str) -> dict[str, Any] | None:
    manifest_path = path / f"{split}-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("split_id") != SPLIT_ID or value.get("split") != split:
        return None
    return value


def _validate(path: Path, *, require_complete: bool) -> DatasetLocation | None:
    if not path.is_dir():
        return None
    manifests = {split: _manifest(path, split) for split in SPLITS}
    if any(value is None for value in manifests.values()):
        return None
    typed = {split: value for split, value in manifests.items() if value is not None}
    if require_complete and not all(bool(value.get("complete")) for value in typed.values()):
        return None
    # A cheap structural check catches a stale/partially mounted directory
    # without walking all 55k files.
    for split in SPLITS:
        sample = next(path.glob(f"{split}-shard-*-row_ptr.npy"), None)
        if sample is None:
            return None
        stem = str(sample)[: -len("-row_ptr.npy")]
        if not all(Path(f"{stem}-{suffix}.npy").is_file() for suffix in ("gene_index", "values")):
            return None
    source = "local_project" if str(path).startswith(str(Path.cwd())) else "external"
    if path == DEFAULT_EXTERNAL_ROOT.resolve():
        source = "local_essd"
    elif path == DEFAULT_NAS_ROOT.resolve() or str(path).startswith("/mnt/"):
        source = "nas_fallback"
    return DatasetLocation(path, source, typed)


def candidate_roots(
    explicit: str | os.PathLike[str] | None = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> Iterable[tuple[Path, str]]:
    """Yield path candidates in hot-storage-first order."""

    seen: set[Path] = set()

    def add(value: str | os.PathLike[str] | None, source: str) -> Iterable[tuple[Path, str]]:
        if value is None:
            return ()
        path = _normalise_candidate(value)
        if path in seen:
            return ()
        seen.add(path)
        return ((path, source),)

    for value, source in add(explicit, "explicit"):
        yield value, source
    env_value = os.environ.get("BRAINWM_V3_SHARD_DIR")
    for value, source in add(env_value, "environment"):
        yield value, source
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    for value, source in add(root / "assets" / "v3_numpy_shards", "local_project"):
        yield value, source
    for value, source in add(DEFAULT_EXTERNAL_ROOT, "local_essd"):
        yield value, source
    for value, source in add(DEFAULT_NAS_ROOT, "nas_fallback"):
        yield value, source


def locate_v3_shards(
    explicit: str | os.PathLike[str] | None = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
    require_complete: bool = True,
) -> DatasetLocation:
    """Resolve and validate the formal human V3 NumPy corpus."""

    for path, requested_source in candidate_roots(explicit, project_root=project_root):
        location = _validate(path, require_complete=require_complete)
        if location is not None:
            # Preserve whether the caller selected an explicit/environment
            # path while retaining the more precise storage classification.
            if requested_source in {"explicit", "environment"}:
                return DatasetLocation(location.root, requested_source, location.split_manifests)
            return location
    candidates = [str(path) for path, _ in candidate_roots(explicit, project_root=project_root)]
    raise FileNotFoundError(
        "no complete BrainWM V3 NumPy corpus found; checked: " + ", ".join(candidates)
    )


def describe_v3_shards(
    explicit: str | os.PathLike[str] | None = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe inventory for logs and experiment receipts."""

    location = locate_v3_shards(explicit, project_root=project_root)
    payload = location.as_dict()
    alias_root = (Path(project_root).resolve()
                  if project_root is not None
                  else Path(__file__).resolve().parents[2]) / "assets" / "v3_numpy_shards"
    try:
        if alias_root.exists() and alias_root.resolve() == location.root:
            payload["project_alias"] = str(alias_root)
            payload["project_alias_is_symlink"] = alias_root.is_symlink()
    except OSError:
        pass
    try:
        stat = os.statvfs(location.root)
        payload["filesystem"] = {
            "mount_free_bytes": int(stat.f_bavail * stat.f_frsize),
            "mount_total_bytes": int(stat.f_blocks * stat.f_frsize),
        }
    except OSError:
        payload["filesystem"] = None
    return payload


__all__ = [
    "DEFAULT_EXTERNAL_ROOT",
    "DEFAULT_NAS_ROOT",
    "DatasetLocation",
    "SPLIT_ID",
    "SPLITS",
    "candidate_roots",
    "describe_v3_shards",
    "locate_v3_shards",
]
