#!/usr/bin/env python3
"""Extend the existing Qian-2025 960-gene acquisition proof to held-out donors.

Reads original sample/cell identities and coordinates, never expression or cell
types. The accepted acquisition rule matches the existing source-65 proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def read_column(group, name, start, end):
    obj = group[name]
    if isinstance(obj, h5py.Group):
        return np.asarray(obj["categories"][:])[obj["codes"][start:end]]
    return obj[start:end]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lookup = pd.read_parquet(args.contract / "source_lookup.parquet")
    source = lookup[lookup.source_id == 65].iloc[0]
    samples = pd.read_parquet(args.contract / "samples.parquet")
    samples = samples[samples.source_id == 65]
    expected = {str(row.sample_id): row for row in samples.itertuples()}
    if set(expected) != {"UMB1031-O1", "UMB1759-O1", "UMB1932-O1"}:
        raise ValueError("source-65 acquisition inventory drift")
    counts = {k: 0 for k in expected}
    digest = hashlib.sha256()
    with h5py.File(source.local_path) as file:
        obs = file["obs"]
        index = obs.attrs.get("_index", "_index")
        n = len(obs[index])
        for start in range(0, n, 100000):
            end = min(n, start + 100000)
            names = read_column(obs, "sample_id", start, end).astype(str)
            cells = read_column(obs, index, start, end).astype(str)
            donors = read_column(obs, "sample", start, end).astype(str)
            xy = file["obsm"]["spatial"][start:end, :2]
            if not np.isfinite(xy).all():
                raise ValueError("source-65 coordinates are incomplete")
            for name in np.unique(names):
                if name not in expected:
                    raise ValueError("unregistered spatial acquisition")
                take = names == name
                donor = str(expected[name].donor_id)
                if not np.all(donors[take] == donor) or not all(donor.lower() in c.lower().split(":")[0] for c in cells[take]):
                    raise ValueError("sample/donor/original-cell-ID acquisition mismatch")
                counts[name] += int(take.sum())
            digest.update("\n".join(names).encode()); digest.update("\n".join(cells).encode()); digest.update(xy.tobytes())
    for name, count in counts.items():
        if count != int(expected[name].n_observations):
            raise ValueError("spatial acquisition row count mismatch")
    result = dict(status="complete", source_id=65, original_file=source.local_path,
        evidence="Qian2025 named O1 acquisition; all original cell IDs and donor IDs match source sample IDs; finite 2D coordinates; declared sample counts matched",
        metadata_sha256=digest.hexdigest(), expression_read=False, cell_type_labels_read=False,
        verified_sample_keys=[str(r.sample_key) for r in samples.itertuples()], counts=counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
