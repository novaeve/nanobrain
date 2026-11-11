"""Merge per-prediction score folders into one table and a training-trajectory table (stdlib only).

    python scripts/collect_scores.py runs/eval/integ_scores --out runs/eval/integ_table_latest

Each folder under the input holds a summary.csv from eval_zero_shot.py for one prediction named
"<method>@<step>" (plus the baseline rows, which are identical across folders and kept once).
Writes summary.csv (every row) and trajectory.csv (method, step, metrics) sorted by method and step.
"""

import argparse
import csv
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    header, rows = None, {}
    for summary in sorted(args.scores.glob("*/summary.csv")):
        with open(summary) as fh:
            table = list(csv.reader(fh))
        header = header or table[0]
        for row in table[1:]:
            rows.setdefault(row[0], row)  # baselines repeat in every folder; keep the first
    if header is None:
        raise SystemExit(f"no summary.csv under {args.scores}")

    def key(name):
        method, _, step = name.partition("@")
        return (step == "", method, int(step) if step.isdigit() else 0)

    os.makedirs(args.out, exist_ok=True)
    ordered = sorted(rows, key=key)
    with open(args.out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows[n] for n in ordered)
    with open(args.out / "trajectory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "step", *header[1:]])
        for n in ordered:
            method, _, step = n.partition("@")
            if step.isdigit():
                w.writerow([method, int(step), *rows[n][1:]])
    print(f"{len(rows)} rows from {args.scores} -> {args.out}")


if __name__ == "__main__":
    main()
