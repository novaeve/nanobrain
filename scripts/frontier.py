"""Where each method stands on the joint frontier: its margin over the best opponent on every task (stdlib only).

    python scripts/frontier.py runs/eval/five_tasks.csv [--methods Cajal@30000 balance_grad_equal@10000 ...]

A task's opponents are the external baselines that do it (OPPONENTS below; scVI fitted on the evaluation
set itself is a reference, not an opponent, since it sees those cells). For each method and task the margin is
(method - best opponent) / best opponent, all metrics being higher-is-better. The frontier is pushed by raising
the smallest margin: `won` counts tasks above the best opponent, `worst` is the smallest margin and its task.
"""

import argparse
import csv

TASKS = {
    "cell type": ("Cell type fine macro-F1", ["XGBoost", "LogReg", "scANVI"]),
    "region": ("Region fine macro-F1", ["XGBoost", "LogReg", "scANVI"]),
    "integration bio": ("Integration bio", ["scVI", "scANVI", "PCA (unintegrated)"]),
    "integration batch": ("Integration batch", ["scVI", "scANVI", "PCA (unintegrated)"]),
    "gene imputation": ("Gene imputation Pearson (cell median)", ["kNN (k=15, same panel)", "gene mean"]),
    "spatial imputation": (
        "Spatial imputation Pearson (cell median)",
        ["neighbour mean (log space)", "neighbour mean (CP10K space)"],
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", help="five_tasks.csv from merge_tables.py")
    ap.add_argument("--methods", nargs="*", help="rows to report (default: every <method>@<step> row)")
    args = ap.parse_args()
    with open(args.table) as fh:
        rows = {r["method"]: r for r in csv.DictReader(fh)}

    def value(row, col):
        v = rows.get(row, {}).get(col, "")
        return float(v) if v not in ("", None) else None

    best = {}
    for task, (col, opponents) in TASKS.items():
        scored = [(value(o, col), o) for o in opponents if value(o, col) is not None]
        best[task] = max(scored) if scored else (None, None)
    methods = args.methods or [m for m in rows if "@" in m]

    print("best opponent: " + "; ".join(f"{t} {v:.3f} ({o})" for t, (v, o) in best.items() if v is not None))
    print(f"{'method':28s} " + " ".join(f"{t[:12]:>12s}" for t in TASKS) + "   won  worst")
    for m in methods:
        margins = {}
        for task, (col, _) in TASKS.items():
            v, (b, _) = value(m, col), best[task]
            if v is not None and b:
                margins[task] = (v - b) / b
        if not margins:
            continue
        cells = " ".join(f"{margins[t]:+12.1%}" if t in margins else f"{'-':>12s}" for t in TASKS)
        worst = min(margins, key=margins.get)
        won = sum(x > 0 for x in margins.values())
        print(f"{m:28s} {cells}   {won}/{len(margins)}  {margins[worst]:+.1%} {worst}")


if __name__ == "__main__":
    main()
