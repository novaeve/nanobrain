"""Run records: what produced a result (config, code, data, environment), written next to it.

Stdlib only, so both the training env and the evaluation env can write records.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], check=False, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def data_fingerprint(root: str) -> dict:
    """The data directory's manifest hash (rows and sampling are in it) and its row count."""
    manifest = Path(root) / "manifest.json"
    if not manifest.exists():
        return {"root": root, "manifest_sha256": None}
    text = manifest.read_bytes()
    return {"root": root, "manifest_sha256": hashlib.sha256(text).hexdigest(), "rows": json.loads(text).get("rows")}


def fingerprint() -> dict:
    """Code, package versions and (when torch is already loaded) the GPU. Versions come from package
    metadata, never by importing: torch, JAX and cuML loaded together in one process crash."""
    env = {"python": platform.python_version(), "host": socket.gethostname()}
    for dist in ("torch", "numpy", "scikit-learn", "xgboost", "scib-metrics", "jax", "cuml-cu12", "scvi-tools"):
        try:
            env[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    torch = sys.modules.get("torch")
    if torch is not None:
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name()
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "env": env,
    }


def write(out_dir: str, kind: str, **fields) -> dict:
    """Create or update out_dir/record.json; `kind` is "train" or "eval"."""
    path = Path(out_dir) / "record.json"
    rec = json.loads(path.read_text()) if path.exists() else {"kind": kind, "created": time.strftime("%FT%T%z")}
    rec.update(fields, updated=time.strftime("%FT%T%z"))
    rec.setdefault("fingerprint", fingerprint())
    os.makedirs(out_dir, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    tmp.replace(path)
    return rec
