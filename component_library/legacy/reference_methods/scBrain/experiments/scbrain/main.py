#!/usr/bin/env python3
"""Public SEA-AD scBrain entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    if "--model_name" not in sys.argv:
        sys.argv.extend(["--model_name", "scBrain"])
    runpy.run_path(str(ROOT / "experiments" / "mjm" / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
