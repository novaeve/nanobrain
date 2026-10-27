import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobrain.evolution import architecture_id, joint_score, improvement_gate, confirm_replicates, validate_proposals
from nanobrain.genome import CandidateGenome, SEARCH_SPACE
from scripts.run_nanobrain_campaign import _parse_gpus, _launch_evaluations
from scripts.run_nanobrain_evolution import Experiment
from scripts.run_nanobrain_evolution import proposal_history


POLICY = dict(minimum_joint_gain=0.002, max_score_drop=0.02, max_error_relative_increase=0.05)


def test_hundred_generation_prompt_history_is_bounded_without_changing_archive():
    history = [dict(generation=i, candidates=[dict(architecture_id=f"c{i}", score=i / 100,
        metrics={"annotation.accuracy": i / 100}, diagnostics={"large_unused_detail": list(range(500))})])
        for i in range(100)]
    old = copy.deepcopy(history)
    summary = proposal_history(history, recent_generations=5, best_candidates=12)
    assert summary["completed_generations"] == 100
    assert [x["generation"] for x in summary["recent_generations"]] == list(range(95, 100))
    assert len(summary["best_observed_candidates"]) == 12
    assert summary["best_observed_candidates"][0]["architecture_id"] == "c99"
    assert history == old and "large_unused_detail" not in json.dumps(summary)
REFERENCE = {"annotation.accuracy": 0.4, "integration.integration_composite": 0.5,
             "spatial_niche.ari": 0.1, "spatial_completion.heldout_rmse": 1.0,
             "development.population_mse": 0.3}


def test_gate_accepts_joint_gain_with_small_tradeoff_but_rejects_task_collapse():
    better = {**REFERENCE, "annotation.accuracy": 0.5, "integration.integration_composite": 0.49}
    assert improvement_gate(better, REFERENCE, POLICY)["passed"]
    assert not improvement_gate(REFERENCE, REFERENCE, POLICY)["passed"]
    worse = {**better, "development.population_mse": 0.6}
    assert not improvement_gate(worse, REFERENCE, POLICY)["passed"]
    assert not improvement_gate({**better, "spatial_niche.ari": float('nan')}, REFERENCE, POLICY)["passed"]
    missing = dict(better)
    del missing["development.population_mse"]
    assert not improvement_gate(missing, REFERENCE, POLICY)["passed"]


def test_fixed_score_is_independent_of_mapping_order_and_uses_correct_directions():
    assert joint_score(REFERENCE) == joint_score(dict(reversed(list(REFERENCE.items()))))
    assert joint_score({**REFERENCE, "development.population_mse": 0.1}) > joint_score(REFERENCE)


def test_confirmation_requires_matched_seeds_and_reproducible_gains():
    refs = [dict(genome=CandidateGenome(seed=s).to_dict(), metrics=REFERENCE) for s in (10, 11, 12)]
    candidates = [dict(genome=CandidateGenome(seed=s, hidden_dim=128).to_dict(),
                       metrics={**REFERENCE, "annotation.accuracy": 0.5}) for s in (12, 10, 11)]
    assert confirm_replicates(candidates, refs, POLICY)["passed"]
    candidates[0]["metrics"] = {**REFERENCE, "development.population_mse": 0.9}
    assert not confirm_replicates(candidates, refs, POLICY)["passed"]
    candidates[0]["genome"]["seed"] = 11
    with pytest.raises(ValueError, match="same independent seeds"):
        confirm_replicates(candidates, refs, POLICY)


def test_llm_proposals_require_real_lineage_full_genome_and_novel_architecture():
    parent = CandidateGenome(seed=7).to_dict()
    parent_id = architecture_id(parent)
    assert architecture_id({**parent, "seed": 8}) == parent_id
    genome = {key: parent[key] for key in SEARCH_SPACE}
    genome["hidden_dim"] = 128
    proposal = dict(parent_ids=[parent_id], target_tasks=["annotation"],
                    hypothesis="increase encoder capacity", expected_tradeoff="more parameters", genome=genome)
    parents = [dict(architecture_id=parent_id, genome=parent)]
    accepted = validate_proposals(dict(candidates=[proposal]), parents, count=1, seed=17, seen={parent_id})
    assert accepted[0]["changes"][parent_id]["hidden_dim"] == {"from": 64, "to": 128}
    assert accepted[0]["proposal_source"] == "llm"
    invalid = copy.deepcopy(proposal)
    del invalid["genome"]["pairing"]
    with pytest.raises(ValueError, match="exactly"):
        validate_proposals(dict(candidates=[invalid]), parents, count=1, seed=17, seen=set())
    invalid = {**proposal, "parent_ids": ["invented"]}
    with pytest.raises(ValueError, match="unknown"):
        validate_proposals(dict(candidates=[invalid]), parents, count=1, seed=17, seen=set())
    with pytest.raises(ValueError, match="duplicate architecture"):
        validate_proposals(dict(candidates=[proposal]), parents, count=1, seed=18,
                           seen={accepted[0]["architecture_id"]})


def test_evaluator_launch_freezes_population_seed_and_defaults_to_validation(tmp_path, monkeypatch):
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr("scripts.run_nanobrain_campaign.subprocess.Popen", popen)
    candidates = [dict(candidate_id="one", directory=str(tmp_path / "one"),
                       training_audit=dict(passed=True, checkpoint="one.pt"))]
    processes = _launch_evaluations(candidates=candidates, gpus=[7], data=tmp_path, batch_size=256,
                                    evaluation_seed=17)
    for item in processes:
        item["log_handle"].close()
    command = commands[0]
    assert command[command.index("--population-evaluation-seed") + 1] == "17"
    assert "--allow-test" not in command
    assert _parse_gpus("0,1,2,3,4,5,6,7") == list(range(8))


@pytest.mark.parametrize("initial_failure", [False, True])
def test_multigeneration_driver_propagates_confirmed_parents_and_freezes_before_test(tmp_path, monkeypatch, initial_failure):
    """Simulated metrics exercise orchestration; they are never research results."""
    experiment = Experiment.__new__(Experiment)
    experiment.output = tmp_path
    experiment.state_path = tmp_path / "campaign.json"
    experiment.protocol = json.loads(Path("configs/nanobrain_evolution_100k.json").read_text())
    experiment.protocol.update(generations=2, candidates_per_generation=1,
                               propagation_candidates=1, confirmation_candidates=1,
                               parent_count=1, no_improvement_patience=0)
    experiment.state = {"generations": [], "status": "initialized"}
    experiment.args = SimpleNamespace(gpus=[0])
    experiment.data = tmp_path
    experiment.fingerprint = {"fixture": "synthetic orchestration only"}
    (tmp_path / "protocol.json").write_text("{}")
    seen_parents = []

    def write_table(path, item):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "task", "metric", "value", "status", "stratum_type", "stratum_value"])
            w.writeheader()
            for key, value in item["metrics"].items():
                task, metric = key.split(".")
                w.writerow(dict(method="nanobrain_candidate-" + item["candidate_id"], task=task,
                                metric=metric, value=value, status="complete",
                                stratum_type="overall", stratum_value="overall"))

    def fake_batch(specs, directory, steps):
        rows = []
        for i, spec in enumerate(specs):
            genome = CandidateGenome.from_dict(spec["genome"])
            target = directory / f"candidate_{i:02d}"
            target.mkdir(parents=True, exist_ok=True)
            checkpoint = target / "checkpoint.pt"
            checkpoint.write_bytes(b"fixture")
            metrics = {**REFERENCE, "annotation.accuracy": 0.4 + (genome.hidden_dim - 64) / 800.0}
            row = dict(spec, genome=genome.to_dict(), architecture_id=architecture_id(genome.to_dict()),
                       candidate_id=genome.genome_id, metrics=metrics, score=joint_score(metrics),
                       runtime_seconds=1.0, parameter_count=100, steps=steps, index=i,
                       checkpoint=str(checkpoint), directory=str(target),
                       evaluation=str(target / "evaluation/benchmark_table.csv"),
                       training_audit=dict(passed=True, checkpoint=str(checkpoint)))
            write_table(Path(row["evaluation"]), row)
            rows.append(row)
        return rows

    def fake_propose(parents, history, generation, seen):
        seen_parents.append(parents[0]["genome"]["hidden_dim"])
        assert len(parents[0]["replicates"]) == 3
        assert len(history) == generation
        folder = tmp_path / f"generation_{generation:02d}"
        folder.mkdir(exist_ok=True)
        width = 32 if initial_failure and generation == 0 else 96 + 32 * generation
        genome = {**parents[0]["genome"], "hidden_dim": width}
        return [dict(genome=genome, parent_ids=[parents[0]["architecture_id"]],
                     proposal_source="llm", target_tasks=["annotation"], hypothesis="fixture")]

    test_calls = []
    def fake_test(**kwargs):
        assert kwargs["evaluation_split"] == "test"
        assert (tmp_path / "selection_freeze.json").is_file()
        assert experiment.state["selected"] is not None
        test_calls.append(True)
        for row in kwargs["candidates"]:
            test_row = {**row, "metrics": {**row["metrics"], "annotation.accuracy": 0.25}}
            write_table(Path(row["directory"]) / "test_reevaluation/benchmark_table.csv", test_row)
        return []

    experiment.run_batch = fake_batch
    experiment.propose = fake_propose
    monkeypatch.setattr("scripts.run_nanobrain_evolution._launch_evaluations", fake_test)
    monkeypatch.setattr("scripts.run_nanobrain_evolution._finalize_evaluations", lambda *a, **k: True)
    experiment.run()
    assert seen_parents == ([64, 64] if initial_failure else [64, 96])
    if initial_failure:
        assert experiment.state["generations"][0]["status"] == "retained_incumbent_continue"
    assert experiment.state["scientific_promotion_passed"] is True
    assert experiment.state["status"] == "complete"
    assert len(test_calls) == 6
    receipt = json.loads((tmp_path / "completion_receipt.json").read_text())
    assert receipt["generations_completed"] == 2
    assert receipt["llm_candidates"] == 2
    assert receipt["test_used_for_selection"] is False
    for runs in experiment.state["test_reevaluation"].values():
        for row in runs:
            assert row["evaluation_split"] == "test"
            assert row["evaluation_sha256"] == hashlib.sha256(Path(row["evaluation"]).read_bytes()).hexdigest()
            assert row["metrics"]["annotation.accuracy"] == 0.25
            assert row["diagnostics"]["annotation.accuracy"] == 0.25
            assert row["score"] == joint_score(row["metrics"])
    assert experiment.state["confirmation_reference"][0]["metrics"]["annotation.accuracy"] == 0.4


def test_four_task_score_does_not_require_completion_and_enforces_all_four_guards():
    tasks = ["annotation", "integration", "spatial_niche", "development"]
    ref = {k: v for k, v in REFERENCE.items() if not k.startswith("spatial_completion.")}
    better = {**ref, "annotation.accuracy": 0.5}
    assert joint_score(better, tasks) > joint_score(ref, tasks)
    assert improvement_gate(better, ref, {**POLICY, "tasks": tasks})["passed"]
    assert not improvement_gate({**better, "development.population_mse": 0.9}, ref, {**POLICY, "tasks": tasks})["passed"]


def test_real_proposal_builder_describes_independent_tasks_and_exposes_metric_feedback(tmp_path, monkeypatch):
    """A stub provider verifies the actual prompt/validation path without API calls."""
    experiment = Experiment.__new__(Experiment)
    experiment.output = tmp_path
    experiment.state_path = tmp_path / "campaign.json"
    experiment.protocol = json.loads(Path("configs/nanobrain_evolution_independent_100k.json").read_text())
    experiment.protocol.update(candidates_per_generation=2, minimum_module_change_candidates=1)
    genome = CandidateGenome(seed=experiment.protocol["screen_seed"]).to_dict()
    parent = dict(architecture_id=architecture_id(genome), genome=genome, metrics=REFERENCE,
                  diagnostics={"annotation.f1_macro": 0.2}, score=0.5, runtime_seconds=1, parameter_count=100)
    experiment.state = dict(reference=parent)
    experiment.api = SimpleNamespace(api_key="unused")
    prompts = []
    def complete(prompt, **kwargs):
        prompts.append(prompt)
        raw = {k: genome[k] for k in SEARCH_SPACE}
        candidates = []
        for edited in ({**raw, "encoder": "residual_mlp"}, {**raw, "hidden_dim": 96}):
            candidates.append(dict(parent_ids=[parent["architecture_id"]], target_tasks=["annotation"],
                                   hypothesis="test fixture", expected_tradeoff="test fixture", genome=edited))
        return json.dumps(dict(candidates=candidates))
    monkeypatch.setattr("scripts.run_nanobrain_evolution.OpenAICompatibleClient", lambda api: SimpleNamespace(complete=complete))
    proposals = experiment.propose([parent], [], 0, {parent["architecture_id"]})
    assert len(proposals) == 2 and len(prompts) == 1
    assert "INDEPENDENTLY" in prompts[0]
    assert "annotation.f1_macro" in prompts[0]
    assert "All five tasks train jointly" not in prompts[0]


def test_proposal_builder_keeps_valid_siblings_and_fills_remaining_slots(tmp_path, monkeypatch):
    experiment = Experiment.__new__(Experiment)
    experiment.output = tmp_path
    experiment.state_path = tmp_path / "campaign.json"
    experiment.protocol = json.loads(Path("configs/nanobrain_evolution_independent_100k.json").read_text())
    experiment.protocol.update(candidates_per_generation=2, minimum_module_change_candidates=1,
                               proposal_semantic_attempts=3)
    genome = CandidateGenome(seed=experiment.protocol["screen_seed"]).to_dict()
    parent_id = architecture_id(genome)
    parent = dict(architecture_id=parent_id, genome=genome, metrics=REFERENCE, score=0.5)
    experiment.state = dict(reference=parent)
    experiment.api = SimpleNamespace(api_key="unused")
    responses = []
    first = {**{key: genome[key] for key in SEARCH_SPACE}, "encoder": "residual_mlp"}
    second = {**{key: genome[key] for key in SEARCH_SPACE}, "hidden_dim": 96}
    duplicate = dict(first)

    def complete(prompt, **kwargs):
        responses.append(prompt)
        payload = [first, duplicate] if len(responses) == 1 else [second]
        return json.dumps({"candidates": [
            dict(parent_ids=[parent_id], target_tasks=["annotation"], hypothesis="fixture",
                 expected_tradeoff="fixture", genome=item) for item in payload
        ]})

    monkeypatch.setattr("scripts.run_nanobrain_evolution.OpenAICompatibleClient",
                        lambda api: SimpleNamespace(complete=complete))
    proposals = experiment.propose([parent], [], 0, {parent_id})
    assert len(proposals) == 2
    assert len(responses) == 2
    assert not (tmp_path / "generation_00" / "proposal_exhausted.json").exists()


def test_proposal_builder_bounds_semantic_retries_and_preserves_incumbent(tmp_path, monkeypatch):
    experiment = Experiment.__new__(Experiment)
    experiment.output = tmp_path
    experiment.state_path = tmp_path / "campaign.json"
    experiment.protocol = json.loads(Path("configs/nanobrain_evolution_independent_100k.json").read_text())
    experiment.protocol.update(candidates_per_generation=2, minimum_module_change_candidates=1,
                               proposal_semantic_attempts=2)
    genome = CandidateGenome(seed=experiment.protocol["screen_seed"]).to_dict()
    parent_id = architecture_id(genome)
    parent = dict(architecture_id=parent_id, genome=genome, metrics=REFERENCE, score=0.5)
    experiment.state = dict(reference=parent)
    experiment.api = SimpleNamespace(api_key="unused")

    def complete(prompt, **kwargs):
        return json.dumps({"candidates": []})

    monkeypatch.setattr("scripts.run_nanobrain_evolution.OpenAICompatibleClient",
                        lambda api: SimpleNamespace(complete=complete))
    assert experiment.propose([parent], [], 0, {parent_id}) is None
    receipt = json.loads((tmp_path / "generation_00" / "proposal_exhausted.json").read_text())
    assert receipt["incumbent_preserved"] is True and receipt["accepted"] == 0
