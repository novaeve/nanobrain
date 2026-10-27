import json

from nanobrain.campaign import (
    append_generation_baseline,
    baseline_regression_gate,
    generation_baseline_record,
    load_generation_baseline,
    strongest_baselines,
)
from nanobrain.evolution import improvement_gate, joint_score


def _rows():
    return [
        {"method": "majority", "task": "annotation", "metric": "f1_macro", "value": "0.20", "status": "complete", "stratum_type": "overall", "stratum_value": "overall"},
        {"method": "linear_sgd", "task": "annotation", "metric": "f1_macro", "value": "0.30", "status": "complete", "stratum_type": "overall", "stratum_value": "overall"},
        {"method": "raw_pca", "task": "integration", "metric": "integration_composite", "value": "0.40", "status": "complete", "stratum_type": "overall", "stratum_value": "overall"},
        {"method": "raw_pca", "task": "spatial_niche", "metric": "ari", "value": "0.05", "status": "complete", "stratum_type": "overall", "stratum_value": "overall"},
        {"method": "global_mean", "task": "development", "metric": "population_mse", "value": "0.60", "status": "complete", "stratum_type": "overall", "stratum_value": "overall"},
        {"method": "nanobrain_candidate-dead", "task": "annotation", "metric": "f1_macro", "value": "0.99", "status": "complete", "stratum_type": "overall", "stratum_value": "overall"},
    ]


def test_generation_baseline_chooses_best_non_candidate_and_round_trips(tmp_path):
    record = generation_baseline_record(_rows(), generation=3)
    assert record["generation"] == 3
    assert record["baseline_metrics"]["annotation.f1_macro"]["method"] == "linear_sgd"
    assert record["baseline_metrics"]["development.population_mse"]["value"] == 0.6
    assert "nanobrain_candidate-dead" in record["candidate_methods"]

    path = tmp_path / "campaign.json"
    append_generation_baseline(path, record)
    append_generation_baseline(path, generation_baseline_record(_rows(), generation=1))
    loaded = load_generation_baseline(path, 3)
    assert loaded["baseline_metrics"] == record["baseline_metrics"]
    assert len(json.loads(path.read_text())["generations"]) == 2


def test_baseline_gate_and_joint_score_use_task_directions():
    baseline = strongest_baselines(_rows())
    candidate = {
        "annotation.f1_macro": 0.31,
        "integration.integration_composite": 0.41,
        "spatial_niche.ari": 0.05,
        "development.population_mse": 0.59,
    }
    gate = baseline_regression_gate(candidate, baseline)
    assert gate["passed"]
    assert joint_score(candidate) > joint_score({
        "annotation.f1_macro": 0.30,
        "integration.integration_composite": 0.40,
        "spatial_niche.ari": 0.05,
        "development.population_mse": 0.60,
    })
    assert improvement_gate(candidate, candidate, {"minimum_joint_gain": 0.0})["passed"]
