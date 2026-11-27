"""Score the training-only global neighbor-type distribution on the spatial evaluation set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_niche import METRIC, N_TYPES, composition, read_spatial_labels


def build(train, spatial):
    _, train_labels, _ = read_spatial_labels(train)
    _, eval_labels, _ = read_spatial_labels(spatial)
    proportions = np.bincount(train_labels.ravel(), minlength=N_TYPES) / train_labels.size
    targets = composition(eval_labels)
    value = float(np.abs(targets - proportions).mean())
    roots = {"pretrain": Path(train), "spatial": Path(spatial)}
    return {
        "method": "training_only_global_neighbor_composition",
        "data_manifest_sha256": {
            name: hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() for name, root in roots.items()
        },
        "training_centers": len(train_labels),
        "evaluation_centers": len(eval_labels),
        "proportions": proportions.tolist(),
        "metrics": {METRIC: value},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--spatial", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    value = build(args.train, args.spatial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{METRIC} = {value['metrics'][METRIC]:.8f}")


if __name__ == "__main__":
    main()
