"""Record the project-scoped component packages without changing the shared venv."""

import hashlib
import importlib.metadata
import json
import platform

from .contracts import ROOT, sha


def snapshot():
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
        distribution = importlib.metadata.distribution(name)
        shared[name] = {
            "version": distribution.version,
            "metadata_sha256": hashlib.sha256(distribution.read_text("METADATA").encode()).hexdigest(),
        }
    return {"python": platform.python_version(), "shared_environment": shared, "project_scoped": scoped}


def verify():
    expected = json.loads((ROOT / "component_library/dependencies.json").read_text())
    current = snapshot()
    if current != expected["environment"]:
        raise ValueError("component dependency metadata changed since preparation; recheck before running")
    return current
