"""Synthetic orchestration tests; these are not research results."""
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from brainwm_benchmark.protocol import METRIC_PROTOCOL_ID
from nanobrain.evolution import architecture_id, joint_score
from nanobrain.genome import CandidateGenome
from scripts.nanobrain_scheduler import available_slots
from scripts.nanobrain_single_stage import compact_generation
from scripts.run_nanobrain_baselines import _write_rows
from scripts.run_nanobrain_evolution import Experiment, evaluation_checkpoint, summarize_runs


def test_shared_pool_uses_per_gpu_slots_instead_of_legacy_global_five_cap():
    snapshot = {gpu: dict(free_mb=65536, external_mb=24000, utilization=99) for gpu in range(8)}
    assert len(available_slots(snapshot, Counter(), per_gpu=8, allow_shared=True)) == 64
    assert len(available_slots(snapshot, Counter({0: 8}), per_gpu=8, allow_shared=True)) == 56
    assert len(available_slots(snapshot, Counter(), per_gpu=5)) == 5


def test_one_failed_seed_cannot_be_averaged_away():
    metrics = {"annotation.f1_macro": .4, "integration.scib_composite": .5,
               "spatial_niche.composition_mae": .02, "development.wasserstein2": .5}
    good = dict(genome={"seed": 1}, metrics=metrics, score=.5, runtime_seconds=1)
    bad = dict(genome={"seed": 2}, metrics={}, failed=True, failure_reasons=["fixture"])
    result = summarize_runs([good, bad], ["annotation", "integration", "spatial_niche", "development"])
    assert result["failed"] and result["metrics"] == {} and result["score"] < 0


def test_best_checkpoint_selector_uses_canonical_validation_selection(tmp_path):
    best = tmp_path / "best.pt"; best.write_bytes(b"fixture")
    assert evaluation_checkpoint(dict(best_checkpoint=str(best), best_step=500),
                                 "validation_primary_metric") == best


def test_campaign_ledger_keeps_summary_and_full_details_separately():
    item = dict(architecture_id="fixture", genome={"large": "x" * 20000},
                diagnostics={"large": "x" * 20000}, replicates=[{"large": "x" * 20000}],
                metrics={"annotation.f1_macro": .4}, score=.4, directory="full-record")
    record = dict(generation=0, candidates=[item], incumbent_id="fixture")
    compact = compact_generation(record)
    assert len(json.dumps(compact)) < len(json.dumps(record)) / 20
    assert compact["candidates"][0]["directory"] == "full-record"


def test_single_stage_uses_three_seeds_one_budget_test_feedback_and_keeps_incumbent(tmp_path):
    experiment = Experiment.__new__(Experiment)
    genome = CandidateGenome(hidden_dim=64, seed=1).to_dict()
    tasks = ["annotation", "integration", "spatial_niche", "development"]
    experiment.protocol = dict(protocol_id="fixture", single_stage_evolution=True, generations=3,
        candidates_per_generation=1, evolution_steps=2, evolution_seeds=[1, 2, 3],
        confirmation_steps=2, confirmation_seeds=[1, 2, 3], selection_evaluation_split="test",
        tasks=tasks, reference_genome=genome,
        gate=dict(minimum_joint_gain=.001, max_score_drop=.02, max_error_relative_increase=.05, tasks=tasks))
    experiment.output = tmp_path; experiment.state = dict(status="initialized", generations=[])
    experiment.save = lambda status=None: experiment.state.update(status=status or experiment.state["status"])
    calls, parents = [], []

    def batch(specs, directory, steps):
        calls.append((directory.name, len(specs), steps))
        result = []
        for index, spec in enumerate(specs):
            g = CandidateGenome.from_dict(spec["genome"])
            path = directory / f"candidate_{index:02d}"; path.mkdir(parents=True, exist_ok=True)
            metric = {64: .3, 96: .4, 32: .2, 128: .5}[g.hidden_dim]
            metrics = {"annotation.f1_macro": metric, "integration.scib_composite": .5,
                       "spatial_niche.composition_mae": .02, "development.wasserstein2": .5}
            table = path / "benchmark_table.csv"
            _write_rows(table, [dict(metric_protocol_id=METRIC_PROTOCOL_ID, evaluation_split="test",
                method="nanobrain_candidate-" + g.genome_id, task=k.split('.')[0], metric=k.split('.')[1],
                value=v, status="complete", stratum_type="overall", stratum_value="overall") for k,v in metrics.items()])
            result.append(dict(spec, genome=g.to_dict(), architecture_id=architecture_id(g.to_dict()),
                candidate_id=g.genome_id, metrics=metrics, diagnostics=metrics, score=joint_score(metrics, tasks),
                runtime_seconds=1, parameter_count=100, evaluation=str(table), directory=str(path)))
        return result

    def propose(current_parents, history, generation, seen):
        parent = current_parents[0]
        parents.append(parent["genome"]["hidden_dim"])
        assert len(parent["replicates"]) == 3
        assert len(history) == generation
        folder = tmp_path / f"generation_{generation:02d}"; folder.mkdir()
        child = {**copy.deepcopy(parent["genome"]), "hidden_dim": [96, 32, 128][generation]}
        return [dict(genome=child, architecture_id=architecture_id(child),
                     parent_ids=[parent["architecture_id"]], target_tasks=["annotation"], proposal_source="fixture")]

    experiment.run_batch = batch; experiment.propose = propose
    experiment.run()
    assert calls == [("reference_single_stage", 3, 2), ("generation_00", 3, 2),
                     ("generation_01", 3, 2), ("generation_02", 3, 2)]
    assert parents == [64, 96, 96]
    assert experiment.state["status"] == "complete" and experiment.state["test_used_for_selection"]
    assert len(experiment.state["generations"]) == 3
    assert experiment.state["generations"][1]["status"] == "retained_incumbent_continue"
    assert "untouched holdout" in (tmp_path / "REPORT.md").read_text()
