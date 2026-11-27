"""Record the project-scoped component packages without changing the shared venv."""

import hashlib
import importlib.metadata
import json
import platform

from .contracts import ROOT, sha


def snapshot(*, allow_missing_components=False):
    scoped = []
    for directory in sorted((ROOT / ".component_deps").glob("*.dist-info")):
        distribution = importlib.metadata.PathDistribution(directory)
        scoped.append(
            {
                "name": distribution.metadata["Name"],
                "version": distribution.version,
                "metadata_sha256": sha(directory / "METADATA"),
                "record_sha256": sha(directory / "RECORD"),
            }
        )
    shared = {}
    for name in ("torch", "numpy", "scvi-tools", "einops"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            if not allow_missing_components or name in ("torch", "numpy"):
                raise
            shared[name] = None
            continue
        shared[name] = {
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256(distribution.read_text("METADATA").encode()).hexdigest(),
        }
    return {"python": platform.python_version(), "shared_environment": shared, "project_scoped": scoped}


def verify(device="cuda"):
    if device == "cpu":
        # CPU wheels have different metadata. The run contract pins the installed environment on resume.
        return snapshot(allow_missing_components=True)
    expected = json.loads((ROOT / "component_library/dependencies.json").read_text())
    current = snapshot()
    if current != expected["environment"]:
        raise ValueError("component dependency metadata changed since preparation; recheck before running")
    return current
