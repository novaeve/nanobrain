"""Path helpers for local, non-versioned data and result locations."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def env_path(name: str, default: str | Path) -> Path:
    """Return an environment-configurable path without hard-coding workstation paths."""
    return Path(os.environ.get(name, str(default))).expanduser()


DATA_ROOT = env_path("SCUNIFLOW_DATA", PROJECT_ROOT / "data")
SEA_AD_ROOT = env_path("SEA_AD_ROOT", DATA_ROOT / "SEA-AD")
RESULTS_ROOT = env_path("SCUNIFLOW_RESULTS", PROJECT_ROOT / "results")
EXTERNAL_ROOT = env_path("SCUNIFLOW_EXTERNAL", PROJECT_ROOT / "external")
TMP_ROOT = env_path("SCUNIFLOW_TMPDIR", PROJECT_ROOT / ".tmp")


def ensure_tmp_root() -> str:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    return str(TMP_ROOT)
