import json

import numpy as np
import pytest
import torch

from brainwm_benchmark.metrics import annotation, development, integration, spatial_niche
from brainwm_benchmark.protocol import METRIC_PROTOCOL_ID, PRIMARY_METRICS
from nanobrain.campaign import overall_metrics, strongest_baselines
from nanobrain.training import _quality


def test_macro_f1_exposes_majority_class_collapse():
    scores = annotation({"y_true": np.r_[np.zeros(90), np.ones(10)], "y_pred": np.zeros(100)})
    assert scores["accuracy"] == pytest.approx(0.9)
    assert scores["f1_macro"] == pytest.approx((180 / 190) / 2)
    assert scores["balanced_accuracy"] == pytest.approx(0.5)


def test_macro_class_universe_depends_only_on_true_support():
    scores = annotation({"y_true": np.array([0, 0, 1, 1]), "y_pred": np.array([0, 7, 1, 7])})
    assert scores["f1_macro"] == pytest.approx(2 / 3)


def test_development_detects_mean_preserving_collapse_and_is_unpaired():
    truth = np.tile([[0.0, 0.0], [2.0, 2.0]], (8, 1))
    exact = development({"target": truth, "prediction": truth[::-1].copy()})
    collapsed = development({"target": truth, "prediction": np.ones_like(truth)})
    assert exact["sinkhorn_divergence"] == pytest.approx(0.0, abs=1e-8)
    assert collapsed["population_mse"] == 0
    assert collapsed["sinkhorn_divergence"] > 0.1
    assert collapsed["mmd_rbf"] > 0


def test_development_equally_weights_population_groups():
    a = np.zeros((4, 2))
    b = np.zeros((12, 2))
    group1 = development({"target": a, "prediction": a + 1})["sinkhorn_divergence"]
    group2 = development({"target": b, "prediction": b + 3})["sinkhorn_divergence"]
    scores = development({"target": np.r_[a, b], "prediction": np.r_[a + 1, b + 3],
                          "group_ptr": np.array([0, 4, 16])})
    assert scores["sinkhorn_divergence"] == pytest.approx((group1 + group2) / 2)


def test_spatial_cluster_permutation_never_becomes_classification_accuracy():
    y = np.repeat([0, 1], 10)
    data = {"embedding": np.c_[y, np.arange(20)], "labels": y, "clusters": 1 - y}
    clustering = spatial_niche(data)
    assert clustering["ari"] == 1
    assert "f1_macro" not in clustering
    classified = spatial_niche({**data, "y_pred": 1 - y})
    assert classified["ari"] == 1
    assert classified["f1_macro"] == 0


def test_spatial_reports_per_section_cluster_diagnostics():
    y = np.tile([0, 0, 1, 1], 2)
    scores = spatial_niche({"embedding": y[:, None], "labels": y, "clusters": 1 - y,
                            "section_id": np.repeat([0, 1], 4)})
    assert scores["section_mean_ari"] == 1
    assert scores["section_mean_nmi"] == 1


def test_scib_official_components_reward_biology_and_are_repeatable():
    rng = np.random.default_rng(11)
    labels = np.repeat([0, 1, 2], 40)
    batches = np.tile([0, 1], 60)
    x = rng.normal(0, 0.15, (120, 4)) + np.eye(4)[labels] * 4
    data = {"embedding": x, "biology_label": labels, "batch_label": batches}
    scores = integration(data)
    repeated = integration(data)
    collapsed = integration({**data, "embedding": rng.normal(size=x.shape)})
    assert scores == pytest.approx(repeated, abs=1e-6)
    assert scores["biology_conservation"] > collapsed["biology_conservation"]
    assert scores["scib_composite"] == pytest.approx(
        0.6 * scores["biology_conservation"] + 0.4 * scores["batch_removal"])
    assert "isolated_label_f1" not in scores
    assert "integration_composite" not in scores


def test_validation_quality_uses_task_primary_and_development_minimization():
    assert _quality({"annotation.f1_macro": 0.3, "annotation.accuracy": 0.99}) == 0.3
    assert _quality({"spatial_niche.composition_mae": 0.4, "spatial_niche.ari": 1.0}) == -0.4
    assert _quality({"development.wasserstein2": 0.1}) > _quality(
        {"development.wasserstein2": 0.2})


def test_legacy_tables_cannot_supply_new_protocol_results():
    row = dict(method="candidate", task="annotation", metric="f1_macro", value="0.9",
               status="complete", stratum_type="overall", stratum_value="overall")
    with pytest.raises(ValueError, match="protocol mismatch"):
        overall_metrics([row], "candidate")
    assert not strongest_baselines([row])
    tagged = {**row, "metric_protocol_id": METRIC_PROTOCOL_ID}
    assert overall_metrics([tagged], "candidate")[("annotation", "f1_macro")] == 0.9


def test_validation_best_weights_survive_a_worse_final_epoch(tmp_path, monkeypatch):
    from nanobrain.data import make_synthetic_splits
    from nanobrain.genome import CandidateGenome
    from nanobrain.model import NanoBrainModelConfig
    from nanobrain.training import TrainConfig, train_candidate

    splits = make_synthetic_splits(train_rows=24, validation_rows=12, test_rows=8,
                                  input_dim=8, annotation_classes=3, niche_classes=2,
                                  integration_classes=2, development_dim=8, seed=17)
    values = iter([0.8, 0.2])
    monkeypatch.setattr("nanobrain.training._evaluate", lambda *a, **k: {"annotation.f1_macro": next(values)})
    receipt = train_candidate(splits, NanoBrainModelConfig(8, 3, 2, 2, 8),
        CandidateGenome(hidden_dim=32, attention="identity", spatial="identity", seed=17),
        TrainConfig(steps=2, batch_size=8, eval_interval=1, task="annotation", precision="fp32",
                    device="cpu", checkpoint_policy="best_weights"), output_dir=tmp_path)
    best = torch.load(receipt["best_checkpoint"], weights_only=False, map_location="cpu")
    final = torch.load(receipt["checkpoint"], weights_only=False, map_location="cpu")
    assert best["step"] == 1 and final["step"] == 2
    assert best["validation"]["annotation.f1_macro"] == 0.8
    assert any(not torch.equal(value, final["model"][key]) for key, value in best["model"].items())
    assert receipt["metric_protocol_id"] == METRIC_PROTOCOL_ID
    assert not list(tmp_path.glob("*-step-*.pt"))


def test_frozen_config_declares_current_metrics_and_validation_best_weights():
    from pathlib import Path
    config = json.loads(Path("configs/nanobrain_graph_v2/metrics_v3_20261102.json").read_text())
    assert config["metric_protocol_id"] == METRIC_PROTOCOL_ID
    assert config["checkpoint_policy"] == "best_weights"
    assert config["checkpoint_selection"] == "validation_primary_metric"
    assert config["llm_reasoning_effort"] == "max"
    assert len([x for x in PRIMARY_METRICS if x[0] in config["tasks"]]) == 4


@pytest.mark.parametrize("patch,message", [
    ({"metric_protocol_id": "legacy"}, "legacy metric protocols"),
    ({"checkpoint_selection": "final_step_fixed_budget"}, "validation_primary_metric"),
    ({"evaluation_seed": 5}, "evaluation_seed=20260915"),
])
def test_driver_rejects_incompatible_protocol_before_loading_data(tmp_path, patch, message):
    from types import SimpleNamespace
    from scripts.run_nanobrain_evolution import Experiment

    protocol = dict(metric_protocol_id=METRIC_PROTOCOL_ID, training_mode="independent",
                    checkpoint_policy="best_weights", checkpoint_selection="validation_primary_metric",
                    evaluation_seed=20260915)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**protocol, **patch}))
    with pytest.raises(ValueError, match=message):
        Experiment(SimpleNamespace(output=tmp_path / "output", config=config))
