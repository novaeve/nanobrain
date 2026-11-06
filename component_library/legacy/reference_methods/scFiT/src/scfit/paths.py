"""Path helpers for repository-relative configs and scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, os.PathLike]


def project_root() -> Path:
    """Return the repository root, overridable with SCFIT_ROOT."""
    override = os.environ.get("SCFIT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Optional[PathLike], base_dir: Optional[PathLike] = None) -> Optional[Path]:
    """Resolve a path from absolute, base-relative, or repository-relative input."""
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if expanded.is_absolute():
        return expanded
    if base_dir is not None:
        candidate = Path(base_dir) / expanded
        if candidate.exists():
            return candidate.resolve()
    return (project_root() / expanded).resolve()


def as_posix_path(path: Optional[PathLike], base_dir: Optional[PathLike] = None) -> Optional[str]:
    resolved = resolve_path(path, base_dir=base_dir)
    return None if resolved is None else resolved.as_posix()

