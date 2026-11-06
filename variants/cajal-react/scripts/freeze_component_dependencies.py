#!/usr/bin/env python3
"""Preparation only: record installed component dependencies, without importing models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cajal_research.contracts import ROOT, save
from cajal_research.dependencies import snapshot

environment = snapshot()
save(
    ROOT / "component_library/dependencies.json",
    {
        "environment": environment,
        "install_target": ".component_deps",
        "shared_venv_modified": False,
        "torch_extension_wheel_index": "https://data.pyg.org/whl/torch-2.9.0+cu128.html",
        "note": "Exact versions and metadata fingerprints. CPU probes verify imports and tensor contracts; GPU execution has not been tested.",
    },
)
(ROOT / "component_library/requirements.txt").write_text(
    "# Install into .component_deps with --no-deps; use Cajal's existing .venv.\n"
    "--find-links https://data.pyg.org/whl/torch-2.9.0+cu128.html\n"
    + "".join(f"{row['name']}=={row['version']}\n" for row in environment["project_scoped"])
)
