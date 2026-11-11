"""Prove allowed search works and both full/patch head edits are rejected."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanobrain.evolution import architecture_id, validate_proposals
from nanobrain.fixed_heads import SEARCH_SCOPE, build_prompt, fixed_heads, search_catalog, validate_search_scope
from nanobrain.genome import CandidateGenome
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
from nanobrain.modules import ClassificationHead, RegressionHead
from nanobrain.task_training import configure_task
from scripts.run_nanobrain_evolution import Experiment


CONFIG = Path("configs/nanobrain_graph_v2/backbone_training_fixed_heads_50g_20261030.json")


def protocol():
    return json.loads(CONFIG.read_text())


def parent():
    genome = CandidateGenome.from_dict({**protocol()["reference_genome"], "seed": 17}).to_dict()
    return dict(architecture_id=architecture_id(genome), genome=genome)


def proposal():
    p = parent()
    return dict(parent_ids=[p["architecture_id"]], target_tasks=["annotation"],
                hypothesis="test an allowed training change", expected_tradeoff="different convergence",
                patch=[dict(op="replace", path="/learning_rate", value=.0005)])


def accept(p):
    ancestor = parent()
    return validate_proposals({"candidates": [p]}, [ancestor], count=1, seed=17,
                              seen={ancestor["architecture_id"]}, search_scope=SEARCH_SCOPE)[0]


def full_proposal():
    p = proposal()
    del p["patch"]
    p["genome"] = copy.deepcopy(parent()["genome"])
    p["genome"].pop("seed")
    p["genome"]["learning_rate"] = .0005
    return p


def test_tutorial_heads_and_search_catalog():
    p = parent()
    validate_search_scope(p["genome"], SEARCH_SCOPE)
    assert p["genome"]["architecture"]["heads"] == fixed_heads()
    assert set(search_catalog()) == {"encoder", "attention", "spatial", "residual", "operation"}
    assert set(protocol()["required_component_coverage"]) == {"harmony", "scanorama", "bbknn"}
    prompt = build_prompt(protocol(), [p], p, {}, 12)
    payload = json.loads(prompt.split("\n", 1)[1])
    assert set(payload["objective_catalog"]["registry"]) == {
        "classification", "adversarial_reconstruction", "masked_regression", "conditional_flow_matching"}
    assert "head" not in payload["component_catalog"]
    assert "flow" not in payload["component_catalog"]
    assert "endpoint" not in payload["component_catalog"]
    assert "TASK ADAPTERS ARE FIXED" in prompt


@pytest.mark.parametrize("task", ["annotation", "niche", "integration", "integration_reconstruction", "development", "completion"])
def test_full_genomes_cannot_replace_any_task_head(task):
    p = full_proposal()
    p["genome"]["architecture"]["heads"][task] = {"name": "scfit", "params": {"dropout": .1}}
    with pytest.raises(ValueError, match="task heads are fixed"):
        accept(p)


def test_full_genome_cannot_add_head_adapter_pregraph():
    p = full_proposal()
    p["genome"]["architecture"]["heads"]["annotation"]["pre"] = {
        "nodes": [{"id": "adapter", "kind": "operation", "name": "layer_norm", "inputs": ["state"], "params": {}}],
        "output": "adapter"}
    with pytest.raises(ValueError, match="task heads are fixed"):
        accept(p)


@pytest.mark.parametrize("path,value", [
    ("/architecture/heads/annotation/name", "scfit"),
    ("/architecture/heads/development/params", {}),
    ("/architecture/flow/nodes/0/params/depth", 4),
    ("/architecture/flow/nodes/0/name", "mlp"),
    ("/architecture/pairing/name", "identity"),
    ("/architecture/objectives/development/name", "endpoint_transport"),
    ("/architecture/version", 2),
    ("/architecture/input_dim", 128),
    ("/architecture", {}),
])
def test_patch_paths_cannot_touch_fixed_fields(path, value):
    p = proposal()
    p["patch"] = [dict(op="replace", path=path, value=value)]
    with pytest.raises(ValueError, match="patch path"):
        accept(p)


@pytest.mark.parametrize("op", ["copy", "move"])
def test_patch_from_cannot_bypass_fixed_field_paths(op):
    p = proposal()
    p["patch"] = [{"op": op, "from": "/architecture/heads/annotation/params", "path": "/architecture/backbone"}]
    with pytest.raises(ValueError, match="forbids patch path"):
        accept(p)


@pytest.mark.parametrize("field,value", [("depth", 4), ("heads", 8)])
def test_full_genome_cannot_change_fixed_fmt(field, value):
    p = full_proposal()
    p["genome"]["architecture"]["flow"]["nodes"][0]["params"][field] = value
    with pytest.raises(ValueError, match="flow is fixed"):
        accept(p)


def test_full_genome_cannot_change_pairing_family():
    p = full_proposal()
    p["genome"]["architecture"]["pairing"] = {"name": "identity", "params": {}}
    with pytest.raises(ValueError, match="Sinkhorn pairing is fixed"):
        accept(p)


@pytest.mark.parametrize("patch", [
    dict(op="replace", path="/learning_rate", value=.0005),
    dict(op="replace", path="/architecture/objectives/annotation/params/label_smoothing", value=.1),
    dict(op="replace", path="/architecture/objectives/development/params/velocity_regression", value="smooth_l1"),
    dict(op="replace", path="/architecture/objectives/integration/params/adversarial_strength", value=.5),
    dict(op="replace", path="/architecture/pairing/params/epsilon", value=.06),
    dict(op="replace", path="/architecture/solver", value={"name": "euler", "steps": 24}),
    dict(op="replace", path="/architecture/backbone/nodes/2/params/dropout", value=.1),
])
def test_allowed_training_and_backbone_edits_remain_searchable(patch):
    p = proposal()
    p["patch"] = [patch]
    accepted = accept(p)
    assert accepted["genome"]["architecture"]["heads"] == fixed_heads()


def test_backbone_width_change_automatically_adapts_fixed_interfaces():
    p = proposal()
    p["patch"] = [dict(op="replace", path="/architecture/width", value=128),
                  dict(op="replace", path="/architecture/backbone", value={
                      "nodes": [{"id": "encode", "kind": "encoder", "name": "mlp", "inputs": ["features"],
                                 "params": {"hidden_dim": 128}}], "output": "encode"})]
    g = accept(p)["genome"]
    assert g["architecture"]["flow"]["nodes"][0]["params"] == {"hidden_dim": 128, "heads": 4, "depth": 2}
    assert g["architecture"]["heads"] == fixed_heads()
    model = NanoBrainModel(NanoBrainModelConfig(256, 3, 2, 2, 256), CandidateGenome.from_dict(g))
    assert model.annotation_head(torch.randn(4, 128)).shape == (4, 3)
    assert model.development_head(torch.randn(4, 128)).shape == (4, 256)


@pytest.mark.parametrize("task,head", [("annotation", "annotation"), ("spatial_niche", "niche"),
                                     ("integration", "integration"), ("development", "development")])
def test_fixed_head_weights_still_train(task, head):
    model = NanoBrainModel(NanoBrainModelConfig(256, 3, 2, 2, 256), CandidateGenome.from_dict(parent()["genome"]))
    configure_task(model, task)
    module = getattr(model, head + "_head")
    assert isinstance(module, ClassificationHead if head != "development" else RegressionHead)
    assert all(p.requires_grad for p in module.parameters())
    module(torch.randn(4, 96)).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_cached_proposal_and_training_entry_reject_head_edits(tmp_path):
    experiment = Experiment.__new__(Experiment)
    experiment.protocol = protocol()
    experiment.output = tmp_path
    bad = full_proposal()["genome"]
    bad["architecture"]["heads"]["annotation"] = {"name": "scfit", "params": {"dropout": .1}}
    folder = tmp_path / "generation_00"
    folder.mkdir()
    (folder / "accepted_proposals.json").write_text(json.dumps({"candidates": [{"genome": bad}]}))
    with pytest.raises(ValueError, match="task heads are fixed"):
        experiment.propose([parent()], [], 0, set())
    with pytest.raises(ValueError, match="task heads are fixed"):
        experiment.run_batch([{"genome": bad}], tmp_path, 2)


def test_driver_sends_restricted_prompt_and_rejects_invalid_llm_head_edit(tmp_path, monkeypatch):
    experiment = Experiment.__new__(Experiment)
    experiment.output = tmp_path
    experiment.state_path = tmp_path / "campaign.json"
    experiment.protocol = {**protocol(), "candidates_per_generation": 1, "required_component_coverage": [],
                           "minimum_module_change_candidates": 0}
    experiment.state = {"reference": parent()}
    experiment.api = SimpleNamespace(model="fixture", api_key="")
    calls = []
    invalid = full_proposal()
    invalid["genome"]["architecture"]["heads"]["annotation"] = {"name": "scfit", "params": {}}
    def complete(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({"candidates": [invalid, proposal()]})
    monkeypatch.setattr("scripts.run_nanobrain_evolution.OpenAICompatibleClient", lambda _: SimpleNamespace(complete=complete))
    accepted = experiment.propose([parent()], [], 0, {parent()["architecture_id"]})
    assert len(calls) == 1 and "TASK ADAPTERS ARE FIXED" in calls[0]
    assert len(accepted) == 1 and accepted[0]["genome"]["architecture"]["heads"] == fixed_heads()
