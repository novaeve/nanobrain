#!/usr/bin/env python3
"""Evaluate one standard NPZ prediction artifact and emit a JSON receipt."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brainwm_benchmark import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["integration", "annotation", "spatial_niche", "spatial_completion", "development"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as payload:
        metrics = evaluate(args.task, payload)
    receipt = {"format": "brainwm_v3_metric_receipt_v1", "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3", "task": args.task, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=True) + "\n")
    print(json.dumps(receipt, indent=2, allow_nan=True))


if __name__ == "__main__": main()
