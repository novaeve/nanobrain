#!/usr/bin/env python3
"""Print the registered BrainWM backend catalog as JSON.

This is intentionally a tiny inspection command. It imports the model module
only when needed, optionally loads application plugins, and emits the exact
capability metadata that the trainer stores in run manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainwm.backends import (  # noqa: E402
    backend_catalog,
    load_backend_entry_points,
    load_backend_plugins,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-plugin", action="append", default=[])
    parser.add_argument("--load-backend-entry-points", action="store_true")
    args = parser.parse_args()
    modules = list(args.backend_plugin)
    modules.extend(
        item.strip()
        for item in os.environ.get("BRAINWM_BACKEND_PLUGINS", "").split(",")
        if item.strip()
    )
    loaded = list(load_backend_plugins(modules))
    entry_points = []
    if args.load_backend_entry_points or os.environ.get(
        "BRAINWM_LOAD_BACKEND_ENTRY_POINTS", "0"
    ) == "1":
        entry_points = list(load_backend_entry_points())
    print(json.dumps({
        "plugins_loaded": loaded,
        "entry_points_loaded": entry_points,
        "backends": backend_catalog(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
