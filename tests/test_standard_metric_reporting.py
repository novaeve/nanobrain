import csv

import pytest

from brainwm_benchmark.protocol import (
    BENCHMARK_METRICS, METRIC_PROTOCOL_ID, REPORTING_REVISION, SELECTION_METRICS,
    metric_protocol, metric_role,
)
from brainwm_benchmark.reporting import write_standard_report


def rows_fixture():
    rows = [dict(task=task, metric=metric, value=-0.1 if metric == "scib_ari" else 0.4,
                 direction=direction) for task, metric, direction in BENCHMARK_METRICS]
    rows.extend(dict(task="integration", metric=m, value=0.9, direction="max", metric_role="benchmark")
                for m in ["scib_composite", "biology_conservation", "batch_removal"])
    return [dict(metric_protocol_id=METRIC_PROTOCOL_ID, split_id="BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
                 evaluation_split="test", method="fixture", display_name="Fixture", mean=row["value"],
                 n_runs=1, fit_rows=20, eval_rows=10, status="complete",
                 stratum_type="overall", stratum_value="overall", **row) for row in rows]


def test_standard_exports_exclude_search_utilities_and_keep_raw_ari(tmp_path):
    rows = rows_fixture()
    # A falsely supplied metric_role must not let a project utility into results.
    rows.append({**rows[0], "stratum_type": "anatomy", "value": 0.99})
    summary = write_standard_report(tmp_path, rows, evaluation_split="validation")
    with (tmp_path / "standard_metrics.csv").open() as handle:
        standard = list(csv.DictReader(handle))
    with (tmp_path / "selection_scores.csv").open() as handle:
        selection = list(csv.DictReader(handle))
    assert len(standard) == summary["standard_metric_rows"] == 10
    assert len(selection) == 3
    assert all(r["metric_role"] == "benchmark" and r["evaluation_split"] == "validation" for r in standard)
    assert all(r["metric_role"] == "selection_only" for r in selection)
    assert float(next(r for r in standard if r["metric"] == "scib_ari")["value"]) == -0.1
    assert "| scib_composite |" not in (tmp_path / "STANDARD_METRICS.md").read_text()


def test_formal_leaderboard_has_separate_integration_metrics(tmp_path):
    from scripts.build_leaderboard import build
    rows = rows_fixture()
    source = tmp_path / "source.csv"
    fields = sorted({key for row in rows for key in row})
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table, markdown = build(source, tmp_path / "public")
    with table.open() as handle:
        public = list(csv.DictReader(handle))
    assert len(public) == 10
    assert all(r["metric_role"] == "benchmark" for r in public)
    assert "scib_composite" not in table.read_text()
    assert "### iLISI" in markdown.read_text()
    assert "### KMeans ARI" in markdown.read_text()


def test_reporting_rejects_foreign_protocol_results(tmp_path):
    from scripts.build_leaderboard import build
    rows = [{**rows_fixture()[0], "metric_protocol_id": "legacy"}]
    with pytest.raises(ValueError, match="protocol mismatch"):
        write_standard_report(tmp_path, rows, evaluation_split="validation")
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="protocol mismatch"):
        build(path, tmp_path / "public")


def test_reporting_protocol_distinguishes_scientific_metrics_from_selection():
    protocol = metric_protocol()
    assert protocol["reporting_revision"] == REPORTING_REVISION
    assert ("integration", "scib_composite", "max") in SELECTION_METRICS
    assert ["integration", "scib_composite", "max"] not in protocol["primary_metrics"]
    assert len([m for m in protocol["primary_metrics"] if m[0] == "integration"]) == 7
    assert metric_role("integration", "scib_composite") == "selection_only"


def test_final_evolution_report_keeps_project_scores_out_of_test_results(tmp_path):
    from scripts.run_nanobrain_baselines import _write_rows
    from scripts.run_nanobrain_evolution import Experiment

    source = tmp_path / "benchmark_table.csv"
    rows = [{**row, "method": "nanobrain_candidate-fixture"} for row in rows_fixture()]
    _write_rows(source, rows)
    experiment = Experiment.__new__(Experiment)
    experiment.output = tmp_path
    experiment.protocol = dict(protocol_id="fixture", screen_steps=2, confirmation_steps=2, gate={})
    experiment.state = dict(status="complete", generations=[], test_reevaluation={"selected": [
        dict(evaluation=str(source), candidate_id="fixture", genome={"seed": seed}) for seed in (19, 20, 21)]})
    experiment.report()
    with (tmp_path / "final_standard_metrics_long.csv").open() as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 30
    assert "scib_composite" not in (tmp_path / "final_standard_metrics_long.csv").read_text()
    assert "scib_composite" in (tmp_path / "final_metrics_long.csv").read_text()
    assert "scib_composite" not in (tmp_path / "REPORT.md").read_text()
