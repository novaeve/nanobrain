"""One table across tasks from per-task summary tables (stdlib only).

    python scripts/merge_tables.py --out runs/eval/five_tasks.csv \\
        "runs/eval/annot_table_latest/summary.csv:Cell type fine macro-F1,Region fine macro-F1" \\
        "runs/eval/integ_table_latest/summary.csv:Integration bio,Integration batch" \\
        "runs/eval/genes_table_latest/summary.csv:Gene imputation Pearson (cell median)" \\
        "runs/eval/spatial_table_latest/summary.csv:Spatial imputation Pearson (cell median)"

Each argument is SUMMARY.csv:COLUMN,COLUMN,... (collect_scores.py / eval_*.py output, first column the
method). Rows are joined on the method name ("<method>@<step>" or a baseline's name); a method missing
from a table gets an empty cell. Rows keep the order in which methods first appear.
"""

import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+", help="SUMMARY.csv:COLUMN,COLUMN,...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    columns, rows = [], {}
    for spec in args.tables:
        path, _, wanted = spec.rpartition(":")
        with open(path) as fh:
            table = list(csv.reader(fh))
        header = table[0]
        picked = wanted.split(",")
        missing = [c for c in picked if c not in header]
        if missing:
            raise SystemExit(f"{path} has no column {missing}; it has {header[1:]}")
        columns += picked
        for row in table[1:]:
            got = rows.setdefault(row[0], {})
            got.update({c: row[header.index(c)] for c in picked})
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", *columns])
        for name, got in rows.items():
            w.writerow([name, *(got.get(c, "") for c in columns)])
    print(f"{len(rows)} methods x {len(columns)} columns -> {args.out}")


if __name__ == "__main__":
    main()
