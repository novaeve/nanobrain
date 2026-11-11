#!/usr/bin/env python3
"""Create a separate composition dataset without changing the parent subset."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanobrain.niche_data import build_niche_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "source", "output", "cache"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--geometry-proof", type=Path)
    p.add_argument("--evaluation-label-contract", type=Path)
    args = p.parse_args()
    result = build_niche_dataset(args.parent, args.source, args.output, args.cache, args.geometry_proof,
                                args.evaluation_label_contract)
    print(json.dumps({k: v for k, v in result.items() if k != "full_sections"}, indent=2))
