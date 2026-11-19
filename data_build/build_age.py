"""Per-cell age labels in days post fertilization (dpf) for data roots built from the V4 shards.

    .venv/bin/python data_build/build_age.py /data/brainwm/unified_1m /data/brainwm/eval_10k \
        /data/brainwm/eval_integration_10k /data/brainwm/eval_spatial

Each root needs source_row.npy ((shard, row) into the V4 shards, in build_unified_1m.shard_prefixes order). It
gets age_lower_dpf.npy and age_upper_dpf.npy (float32; NaN = open end; both NaN = no numeric age), from the
shards' time_lower_days / time_upper_days / time_point_days / time_status / time_basis / life_stage (the
bioinformatics group's V4 contract, days relative to birth) through brainwm_unified.age.label_interval.
Nothing else in the root is touched, and existing age files are refused unless --overwrite.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_unified_1m import shard_prefixes  # same directory

from brainwm_unified.age import label_interval

SHARDS = "/data/brainwm/BrainWM-Training-Benchmark/assets/v4_numpy_shards"


def gather(source_row: np.ndarray, prefixes: list[str], names: list[str]) -> dict:
    out = {}
    for s in np.unique(source_row[:, 0]):
        m = source_row[:, 0] == s
        for name in names:
            col = np.load(prefixes[s] + name + ".npy")[source_row[m, 1]]
            if name not in out:
                out[name] = np.empty(len(source_row), col.dtype)
            out[name][m] = col
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--shards", default=SHARDS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    prefixes = shard_prefixes(args.shards)
    with open(os.path.join(args.shards, "joint_metadata_vocab.json")) as fh:
        vocab = json.load(fh)
    names = {k: {v: n for n, v in vocab[k].items()} for k in ("time_status", "time_basis", "life_stage")}
    for root in args.roots:
        targets = [os.path.join(root, f"age_{b}_dpf.npy") for b in ("lower", "upper")]
        if not args.overwrite and any(os.path.exists(t) for t in targets):
            raise SystemExit(f"{root}: age files exist (--overwrite to rebuild)")
        sr = np.load(os.path.join(root, "source_row.npy"))
        cols = ("time_lower_days", "time_upper_days", "time_point_days", "time_status", "time_basis", "life_stage")
        g = gather(sr, prefixes, list(cols))
        status = np.array([names["time_status"].get(int(c), "unknown") for c in g["time_status"]])
        basis = np.array([names["time_basis"].get(int(c), "") for c in g["time_basis"]])
        stage = np.array([names["life_stage"].get(int(c), "unknown") for c in g["life_stage"]])
        lo, hi = label_interval(g["time_lower_days"], g["time_upper_days"], g["time_point_days"], status, basis, stage)
        np.save(targets[0], lo)
        np.save(targets[1], hi)
        has = np.isfinite(lo) | np.isfinite(hi)
        closed = np.isfinite(lo) & np.isfinite(hi)
        print(
            f"{root}: {len(lo)} cells, {has.mean():.3f} with an age, {closed.mean():.3f} closed intervals, "
            f"dpf {np.nanmin(lo):.0f}..{np.nanmax(np.where(np.isfinite(hi), hi, lo)):.0f}"
        )


if __name__ == "__main__":
    main()
