#!/usr/bin/env python3
"""Compatibility wrapper for the release-wide metric parser."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = ROOT / "experiments" / "extract_all_metrics.py"
    runpy.run_path(str(parser), run_name="__main__")


if __name__ == "__main__":
    main()
