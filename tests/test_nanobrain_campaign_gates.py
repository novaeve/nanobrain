from nanobrain.campaign import (
    PRIMARY_METRICS,
    baseline_regression_gate,
    metric_objectives,
    overall_metrics,
    strongest_baselines,
)
from nanobrain.search import CandidateScore, dominates, pareto_front


def _metrics(offset=0.0):
    values = {}
    for index, (task, metric, direction) in enumerate(PRIMARY_METRICS):
        base = float(index + 1)
        values[(task, metric)] = base + offset if direction == "max" else base - offset
    return values


def test_baseline_gate_is_fail_closed_and_direction_aware():
    rows = []
    for (task, metric, direction), value in zip(PRIMARY_METRICS, _metrics().values()):
        rows.append({
            "task": task,
            "metric": metric,
            "method": f"baseline_{task}",
            "status": "complete",
            "stratum_type": "overall",
            "stratum_value": "overall",
            "value": str(value),
            "direction": direction,
        })
    baselines = strongest_baselines(rows)

    assert baseline_regression_gate(_metrics(offset=0.1), baselines)["passed"] is True
    assert baseline_regression_gate(_metrics(offset=-0.1), baselines)["passed"] is False
    incomplete = dict(baselines)
    incomplete.pop(("development", "population_mse"))
    gate = baseline_regression_gate(_metrics(offset=0.1), incomplete)
    assert gate["passed"] is False
    assert "baseline_missing:development.population_mse" in gate["reasons"]


def test_task_metric_vector_drives_pareto_selection():
    left = CandidateScore("left", 0.0, 0.0, 2.0, 20, {}, (0.8, 0.5))
    right = CandidateScore("right", 0.0, 0.0, 3.0, 30, {}, (0.7, 0.4))

    assert dominates(left, right)
    assert [score.candidate_id for score in pareto_front([left, right])] == ["left"]


def test_overall_metrics_ignore_strata_and_extract_complete_rows():
    rows = [
        {
            "method": "candidate",
            "task": "annotation",
            "metric": "accuracy",
            "value": "0.75",
            "status": "complete",
            "stratum_type": "overall",
            "stratum_value": "overall",
        },
        {
            "method": "candidate",
            "task": "annotation",
            "metric": "accuracy",
            "value": "0.5",
            "status": "complete",
            "stratum_type": "anatomy",
            "stratum_value": "1",
        },
    ]

    assert overall_metrics(rows, "candidate") == {("annotation", "accuracy"): 0.75}
