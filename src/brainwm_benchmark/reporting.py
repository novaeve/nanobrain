"""Keep established task metrics separate from project search utilities."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from .protocol import (
    BENCHMARK_METRICS, METRIC_PROTOCOL_ID, METRIC_REFERENCES, REPORTING_REVISION,
    metric_role,
)


def write_standard_report(output: Path, rows: list[dict], *, evaluation_split: str) -> dict:
    """Export overall task results without ranking by a project aggregate."""
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("standard report requires an explicit validation or test split")
    if any(row.get("metric_protocol_id") != METRIC_PROTOCOL_ID for row in rows):
        raise ValueError("standard report metric protocol mismatch")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["metric_protocol_id", "reporting_revision", "evaluation_split", "task", "method",
              "metric", "metric_role", "value", "direction", "fit_rows", "eval_rows"]
    exports = {"benchmark": [], "selection_only": []}
    for row in rows:
        role = metric_role(row.get("task", ""), row.get("metric", ""))
        if (role not in exports or row.get("status") != "complete"
                or row.get("stratum_type") != "overall" or row.get("stratum_value") != "overall"):
            continue
        try:
            if not math.isfinite(float(row["value"])):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        exports[role].append({**row, "metric_role": role, "evaluation_split": evaluation_split,
                              "reporting_revision": REPORTING_REVISION})
    for role, filename in [("benchmark", "standard_metrics.csv"), ("selection_only", "selection_scores.csv")]:
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(exports[role], key=lambda r: (r["task"], r["metric"], r["method"])))
    lines = ["# Standard task metrics", "", f"Evaluation split: **{evaluation_split}**.", "",
             "These are individual task metrics. Project checkpoint/search aggregates are kept in "
             "`selection_scores.csv` and are not official benchmark totals.", ""]
    for task in dict.fromkeys(t for t, _, _ in BENCHMARK_METRICS):
        lines.extend([f"## {task}", "", f"[Metric/task reference]({METRIC_REFERENCES[task]}). "
                      "Dataset-specific sampling and target construction follow METRICS_V3.md; "
                      "this is not a complete reproduction of that paper.", "",
                      "| Method | Metric | Value | Direction | Evaluation cells |",
                      "|---|---|---:|---|---:|"])
        for row in exports["benchmark"]:
            if row["task"] == task:
                lines.append(f"| {row['method']} | {row['metric']} | {float(row['value']):.6g} | "
                             f"{row['direction']} | {row.get('eval_rows', '')} |")
        lines.append("")
    (output / "STANDARD_METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"revision": REPORTING_REVISION, "evaluation_split": evaluation_split,
            "standard_metrics": "standard_metrics.csv", "standard_metric_rows": len(exports["benchmark"]),
            "selection_scores": "selection_scores.csv", "selection_score_rows": len(exports["selection_only"])}
