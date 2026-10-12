from __future__ import annotations

import csv

from scripts.build_leaderboard import build


def test_leaderboard_keeps_only_overall_rows_for_four_tasks(tmp_path) -> None:
    source = tmp_path / "source.csv"
    fields = [
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
        "stratum_type",
        "stratum_value",
        "mean",
        "std",
        "n_runs",
        "n_numeric",
        "eval_rows",
        "status",
        "notes",
    ]
    rows = [
        {
            "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
            "subset": "nanobrain_v3_100k",
            "evaluation_split": "test",
            "source_group": "baseline",
            "method": "nanobrain_fmt",
            "display_name": "nanoBrain FMT",
            "method_family": "ours",
            "training_regime": "3 seeds",
            "task": "annotation",
            "metric": "f1_macro",
            "direction": "max",
            "stratum_type": "overall",
            "stratum_value": "overall",
            "mean": "0.5",
            "std": "0.1",
            "n_runs": "3",
            "n_numeric": "3",
            "eval_rows": "20",
            "status": "complete",
            "notes": "",
        },
        {
            "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
            "subset": "nanobrain_v3_100k",
            "evaluation_split": "test",
            "source_group": "baseline",
            "method": "too_specific",
            "display_name": "Anatomy row",
            "method_family": "baseline",
            "training_regime": "1 seed",
            "task": "annotation",
            "metric": "f1_macro",
            "direction": "max",
            "stratum_type": "anatomy",
            "stratum_value": "1",
            "mean": "0.9",
            "std": "0",
            "n_runs": "1",
            "n_numeric": "1",
            "eval_rows": "3",
            "status": "complete",
            "notes": "",
        },
        {
            "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
            "subset": "nanobrain_v3_100k",
            "evaluation_split": "test",
            "source_group": "baseline",
            "method": "legacy",
            "display_name": "Legacy completion",
            "method_family": "baseline",
            "training_regime": "1 seed",
            "task": "spatial_completion",
            "metric": "heldout_rmse",
            "direction": "min",
            "stratum_type": "overall",
            "stratum_value": "overall",
            "mean": "0.2",
            "std": "0",
            "n_runs": "1",
            "n_numeric": "1",
            "eval_rows": "3",
            "status": "complete",
            "notes": "",
        },
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    csv_path, markdown_path = build(source, tmp_path / "leaderboard")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        output_rows = list(csv.DictReader(handle))
    assert [(row["task"], row["method"]) for row in output_rows] == [
        ("annotation", "nanobrain_fmt")
    ]
    assert "test split has been opened" in markdown_path.read_text(encoding="utf-8")
