#!/usr/bin/env python3
"""Build a four-task public leaderboard from the frozen long benchmark table."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainwm_benchmark.protocol import (  # noqa: E402
    INTEGRATION_REPORT_METRICS, METRIC_PROTOCOL_ID, REPORTING_REVISION, metric_role,
)

TASKS = {
    "annotation": (("f1_macro", "Macro F1", "max"),),
    "integration": INTEGRATION_REPORT_METRICS,
    "development": (("wasserstein2", "Wasserstein-2", "min"),),
    "spatial_niche": (("composition_mae", "Neighborhood composition MAE", "min"),),
}
OUTPUT_FIELDS = (
    "metric_protocol_id",
    "reporting_revision",
    "metric_role",
    "split_id",
    "subset",
    "evaluation_split",
    "source_group",
    "method",
    "display_name",
    "method_family",
    "training_regime",
    "task",
    "metric",
    "direction",
    "mean",
    "std",
    "n_runs",
    "n_numeric",
    "eval_rows",
    "status",
    "notes",
)
DEFAULT_INPUT = Path("benchmarks/nanobrain_v3_100k/frozen_test_metrics.csv")


def build(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("benchmark source table is empty")
    if any(row.get("metric_protocol_id") != METRIC_PROTOCOL_ID for row in rows):
        raise ValueError("metric protocol mismatch: re-evaluate under the current protocol before publishing")
    if any(row.get("evaluation_split") != "test" for row in rows):
        raise ValueError("leaderboard source must contain only the frozen test evaluation")
    if any(row.get("split_id") != "BRAINWM-HUMAN-GLOBAL-6-2-2-V3" for row in rows):
        raise ValueError("benchmark source uses an unexpected split contract")

    selected = [
        row
        for row in rows
        if row.get("task") in TASKS
        and metric_role(row.get("task", ""), row.get("metric", "")) == "benchmark"
        and row.get("stratum_type") == "overall"
        and row.get("status") == "complete"
        and row.get("mean") not in {"", "NA", "nan", None}
    ]
    if not selected:
        raise ValueError("benchmark source has no overall results for the four current tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "overall_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        safe_rows = []
        for row in selected:
            safe = dict(row)
            safe.update(reporting_revision=REPORTING_REVISION, metric_role="benchmark")
            safe["notes"] = re.sub(
                r"(?:^|;\s*)checkpoint=[^;]*",
                "; checkpoint path omitted",
                safe.get("notes", ""),
            ).strip("; ")
            safe_rows.append(safe)
        writer.writerows(
            sorted(safe_rows, key=lambda row: (row["task"], row["metric"], row["method"]))
        )

    markdown = [
        "# nanoBrain 100k Leaderboard",
        "",
        "> [!IMPORTANT]",
        "> These are descriptive results from the existing frozen test evaluation. The test split has been opened; it is not a new blind holdout. Methods use different published training regimes and run counts.",
        "",
        "Dataset: `nanobrain_v3_100k` · split: `BRAINWM-HUMAN-GLOBAL-6-2-2-V3` · rows: 60k/20k/20k · source: frozen benchmark long table.",
        "",
        "Each table reports an established individual task metric. Integration has separate biology and batch-removal metrics; there is no ranking by a project composite. The standard overall metrics and provenance fields are in [`overall_metrics.csv`](overall_metrics.csv).",
        "",
    ]
    for task, metrics in TASKS.items():
        markdown.extend([f"## {task.replace('_', ' ').title()}", ""])
        for primary_metric, label, direction in metrics:
            task_rows = [row for row in selected if row["task"] == task and row["metric"] == primary_metric]
            reverse = direction == "max"
            task_rows.sort(key=lambda row: float(row["mean"]), reverse=reverse)
            markdown.extend(
                [f"### {label}", "", f"{'Higher' if reverse else 'Lower'} is better.", "",
                 "| Rank | Method | Score | SD | Runs | Evaluation cells |",
                 "|---:|---|---:|---:|---:|---:|"]
            )
            for rank, row in enumerate(task_rows, start=1):
                sd = row.get("std") or ""
                sd_text = f"{float(sd):.4f}" if sd not in {"", "NA", "nan"} else "—"
                markdown.append(
                    f"| {rank} | {row['display_name']} (`{row['method']}`) | "
                    f"{float(row['mean']):.4f} | {sd_text} | {row['n_runs']} | {row['eval_rows']} |"
                )
            if not task_rows:
                markdown.append("| — | No result in the source table | — | — | — | — |")
            markdown.append("")
    markdown.extend(
        [
            "## Protocol",
            "",
            "Scores are grouped over the `overall` stratum only; anatomy-specific rows are not mixed into task rankings. The table preserves source means and standard deviations without recomputing a composite across unrelated metrics. Undefined metrics and the legacy spatial-completion task are excluded from the four-task ranking.",
            "",
            "Regenerate with `uv run python scripts/build_leaderboard.py --input PATH/TO/final_benchmark_long.csv`.",
        ]
    )
    markdown_path = output_dir / "LEADERBOARD.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="four-task frozen metrics source table"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/nanobrain_v3_100k"))
    args = parser.parse_args()
    csv_path, markdown_path = build(args.input, args.output_dir)
    print(f"Wrote {csv_path} and {markdown_path}")


if __name__ == "__main__":
    main()
