#!/usr/bin/env python3
"""Standalone entry; shared venv need not be installed again."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cajal_research.cli import main

if __name__ == "__main__":
    main()
