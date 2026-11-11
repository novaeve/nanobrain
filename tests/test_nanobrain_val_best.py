import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nanobrain.data import DatasetSplits, make_synthetic_splits
from nanobrain.development import DevelopmentPairDataset
from nanobrain.evolution import architecture_id
from nanobrain.genome import CandidateGenome
from nanobrain.model import NanoBrainModelConfig
from nanobrain.training import TrainConfig, train_candidate
from scripts.run_nanobrain_baselines import _run_nanobrain, _write_rows
from scripts.run_nanobrain_evolution import Experiment, evaluation_checkpoint, summarize_runs


TASKS = ["annotation", "integration", "spatial_niche", "development"]
METRICS = {"annotation.accuracy": 0.5, "integration.integration_composite": 0.5,
           "spatial_niche.ari": 0.1, "development.population_mse": 0.3}


def fixture_splits():
    splits = make_synthetic_splits(train_rows=24, validation_rows=12, test_rows=8,
        input_dim=8, annotation_classes=3, niche_classes=2, integration_classes=2,
        development_dim=8, seed=17)

    def pairs(dataset, split):
        n = dataset.size // 2
        return DevelopmentPairDataset(dataset, np.arange(n), np.arange(n, 2 * n),
            np.array([0, n]), [{"source_time_days": 10., "target_time_days": 20.,
                "interval_id": "DEV-FIXTURE", "uberon_id": "UBERON:fixture", "cells": n}], split=split)

    return DatasetSplits(splits.train, splits.validation, splits.test,
        pairs(splits.train, "train"), pairs(splits.validation, "validation"), pairs(splits.test, "test"))


@pytest.mark.parametrize("task,key,qualities", [
    ("annotation", "annotation_accuracy", [.9, .4, .6]),
    ("spatial_niche", "niche_accuracy", [.9, .4, .6]),
    ("integration", "integration_objective", [.1, .9, .7]),
    ("development", "development_population_mse", [.1, .9, .7]),
])
def test_earlier_validation_best_weights_are_selected_and_test_evaluated(
        tmp_path, monkeypatch, task, key, qualities):
    full = fixture_splits()
    training = DatasetSplits(full.train, full.validation, None,
                            full.development_train, full.development_validation)
    states = []

    def controlled_validation(model, splits, *args, **kwargs):
        assert splits.test is None and splits.development_test is None
        states.append(copy.deepcopy(model.state_dict()))
        return {key: qualities[len(states) - 1]}

    monkeypatch.setattr("nanobrain.training._evaluate", controlled_validation)
    genome = CandidateGenome(hidden_dim=32, encoder_depth=1, attention="identity",
        spatial="identity", residual="none", pairing="identity", flow="mlp", seed=17)
    receipt = train_candidate(training, NanoBrainModelConfig(8, 3, 2, 2, 8), genome,
        TrainConfig(steps=3, batch_size=8, validation_batch_size=12, eval_interval=1,
            validation_development_groups=1, development_cells=4, precision="fp32",
            device="cpu", task=task, checkpoint_policy="best_weights"), output_dir=tmp_path)
    selected = evaluation_checkpoint(receipt, "best_validation")
    assert receipt["best_step"] == 1 and receipt["completed_steps"] == 3
    assert receipt["test_opened"] is False
    assert selected != Path(receipt["checkpoint"])
    assert len(list(tmp_path.glob("*.pt"))) == 2
    payload = torch.load(selected, weights_only=False)
    assert payload["step"] == 1 and "optimizer" not in payload
    assert all(torch.equal(value, states[0][name]) for name, value in payload["model"].items())
    assert any(not torch.equal(value, states[-1][name]) for name, value in payload["model"].items())
    test_splits = DatasetSplits(full.train, full.test, None,
                              full.development_train, full.development_test)
    rows = _run_nanobrain(test_splits, selected, tmp_path / "test", "cpu", 12, evaluation_seed=5)
    assert rows and {row["task"] for row in rows} == {task}
    assert not any(row["status"] in {"error", "failed"} for row in rows)
    assert all(str(selected) in row["notes"] for row in rows)

    # Exercise the real driver handoff as well as the evaluator itself.
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    experiment = Experiment.__new__(Experiment)
    experiment.args = SimpleNamespace(gpus=[0])
    experiment.data = tmp_path
    experiment.state_path = tmp_path / "driver-campaign.json"
    experiment.state = {}
    experiment.protocol = dict(checkpoint_selection="best_validation", evaluation_batch_size=12,
                               evaluation_seed=5, selection_evaluation_split="test")

    def load_receipt(item, steps):
        item["receipt"] = str(receipt_path)
        return receipt

    def evaluate_candidates(*, candidates, evaluation_split, **kwargs):
        assert evaluation_split == "test"
        for item in candidates:
            assert item["training_audit"]["checkpoint"] == str(selected)
            table = Path(item["directory"]) / "test_selection_evaluation/benchmark_table.csv"
            table.parent.mkdir(parents=True)
            _write_rows(table, rows)
            item.update(evaluation=str(table), evaluation_returncode=0)
        return []

    monkeypatch.setattr("scripts.run_nanobrain_evolution._load_training_receipt", load_receipt)
    monkeypatch.setattr("scripts.run_nanobrain_evolution._launch_evaluations", evaluate_candidates)
    monkeypatch.setattr("scripts.run_nanobrain_evolution._finalize_evaluations", lambda *a, **k: None)
    results = experiment._run_single_batch([dict(genome=genome.to_dict())], tmp_path / "driver", 3, task)
    assert results[0]["checkpoint"] == str(selected) and results[0]["selected_step"] == 1
    assert results[0]["checkpoint_selection"] == "best_validation"
    Path(receipt["best_checkpoint"]).unlink()
    with pytest.raises(RuntimeError, match="best_checkpoint missing"):
        evaluation_checkpoint(receipt, "best_validation")


@pytest.mark.parametrize("failed", [False, True])
def test_missing_or_failed_replicate_invalidates_whole_architecture(failed):
    runs = [dict(genome={"seed": seed}, metrics=dict(METRICS), runtime_seconds=1.)
            for seed in [1, 2, 3]]
    runs[1]["metrics"] = {}
    runs[1]["failed"] = failed
    result = summarize_runs(runs, TASKS)
    assert result["failed"] and result["metrics"] == {} and result["score"] < 0
    assert result["failure_reasons"][0]["seed"] == 2


def test_failed_candidates_do_not_stop_generation_or_promote_partial_seed_results(tmp_path):
    experiment = Experiment.__new__(Experiment)
    experiment.args = SimpleNamespace()
    experiment.output = tmp_path
    experiment.state_path = tmp_path / "campaign.json"
    experiment.state = {"generations": []}
    reference = CandidateGenome(hidden_dim=32, seed=1).to_dict()
    good = CandidateGenome(hidden_dim=64, seed=1).to_dict()
    bad = CandidateGenome(hidden_dim=96, seed=1).to_dict()
    experiment.protocol = dict(protocol_id="VAL-BEST-REGRESSION", tasks=TASKS, reference_genome=reference,
        evolution_steps=3, confirmation_steps=3, evolution_seeds=[1, 2, 3],
        confirmation_seeds=[1, 2, 3], selection_evaluation_split="test", generations=2,
        checkpoint_selection="best_validation", gate=dict(tasks=TASKS,
            minimum_joint_gain=.002, max_score_drop=.02, max_error_relative_increase=.05))

    def propose(parents, history, generation, seen):
        genomes = [bad, good] if generation == 0 else [bad]
        return [dict(genome=genome, architecture_id=architecture_id(genome),
                     parent_ids=[parents[0]["architecture_id"]]) for genome in genomes]

    def run_batch(specs, directory, steps):
        runs = []
        for spec in specs:
            item = copy.deepcopy(spec)
            item.update(architecture_id=architecture_id(spec["genome"]), runtime_seconds=1.)
            item["metrics"] = dict(METRICS)
            if item["architecture_id"] == architecture_id(good):
                item["metrics"]["annotation.accuracy"] = .7
            if item["architecture_id"] == architecture_id(bad) and item["genome"]["seed"] == 2:
                item.update(metrics={}, failed=True, failure_reasons=["training_stability_failed"])
            runs.append(item)
        return runs

    experiment.propose = propose
    experiment.run_batch = run_batch
    experiment.run()
    assert experiment.state["status"] == "complete"
    assert len(experiment.state["generations"]) == 2
    assert experiment.state["selected"] == architecture_id(good)
    for generation in experiment.state["generations"]:
        failed = next(x for x in generation["candidates"] if x.get("failed"))
        assert not failed["eligible"] and not failed["reference_gate"]["passed"]
    assert experiment.state["generations"][1]["status"] == "retained_incumbent_continue"
    assert experiment.state["test_used_for_selection"] is True
    assert "best_validation" in (tmp_path / "REPORT.md").read_text()


def test_new_config_keeps_budget_seeds_and_test_feedback():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/nanobrain_graph_v2/test_driven_single_stage_50g_20261027.json").read_text())
    current = json.loads((root / "configs/nanobrain_graph_v2/test_driven_val_best_50g_20261029.json").read_text())
    for key in ("data", "tasks", "evolution_steps", "evolution_seeds",
                "generations", "candidates_per_generation", "selection_evaluation_split", "gate"):
        assert current[key] == original[key]
    assert current["checkpoint_selection"] == "best_validation"
    assert current["checkpoint_policy"] == "best_weights"
    old_genome = copy.deepcopy(original["reference_genome"])
    old_genome["architecture"]["version"] = 3
    assert current["reference_genome"] == old_genome
    assert current["llm_model"] == "gpt-6.1-sol" and current["llm_reasoning_effort"] == "max"


def test_llm_prompt_contains_all_17_new_components_and_sends_max_effort(tmp_path, monkeypatch):
    from test_component_library_v3 import ENTRIES
    from nanobrain.genome import SEARCH_SPACE

    root = Path(__file__).resolve().parents[1]
    experiment = Experiment.__new__(Experiment)
    experiment.output = tmp_path
    experiment.state_path = tmp_path / "campaign.json"
    experiment.protocol = json.loads((root / "configs/nanobrain_graph_v2/test_driven_val_best_50g_20261029.json").read_text())
    experiment.protocol.update(candidates_per_generation=1, minimum_module_change_candidates=0,
                               required_component_coverage=[], proposal_semantic_attempts=1)
    genome = copy.deepcopy(experiment.protocol["reference_genome"])
    genome["seed"] = experiment.protocol["screen_seed"]
    genome = CandidateGenome.from_dict(genome).to_dict()
    parent = dict(architecture_id=architecture_id(genome), genome=genome,
                  metrics=METRICS, score=.5, diagnostics={"annotation.f1_macro": .2})
    experiment.state = dict(reference=parent)
    experiment.api = SimpleNamespace(api_key="unused", model="gpt-6.1-sol")
    calls = []

    def complete(prompt, **kwargs):
        calls.append((prompt, kwargs))
        rate = next(x for x in SEARCH_SPACE["learning_rate"] if x != genome["learning_rate"])
        return json.dumps(dict(candidates=[dict(parent_ids=[parent["architecture_id"]],
            target_tasks=["annotation"], hypothesis="fixture", expected_tradeoff="fixture",
            patch=[dict(op="replace", path="/learning_rate", value=rate)])]))

    monkeypatch.setattr("scripts.run_nanobrain_evolution.OpenAICompatibleClient",
                        lambda api: SimpleNamespace(complete=complete))
    proposals = experiment.propose([parent], [], 0, {parent["architecture_id"]})
    assert len(proposals) == 1 and len(calls) == 1
    prompt, options = calls[0]
    assert all(name in prompt for kind, name in ENTRIES)
    assert options["reasoning_effort"] == "max"
    assert "batch_id[B,1]" in prompt and "nonnegative_features[B,G]" in prompt
    assert all(key in prompt for key in METRICS)
