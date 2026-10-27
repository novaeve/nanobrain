"""Architecture semantics, real optimization, and search boundary regressions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nanobrain.architecture import ModuleGraph, Shape, canonical_architecture, validate_architecture
from nanobrain.architecture_migration import legacy_to_graph
from nanobrain.genome import CandidateGenome
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
from nanobrain.contracts import BrainBatch
from nanobrain.evolution import architecture_id, validate_proposals, has_module_change
from nanobrain.task_training import configure_task, row_task_loss, task_input, TASKS
from nanobrain.training import TrainConfig, train_candidate
from test_nanobrain_independent_tasks import fixture_data


def node(key, kind, name, inputs, **params):
    return dict(id=key, kind=kind, name=name, inputs=inputs, params=params)


def architecture():
    spec = legacy_to_graph(CandidateGenome(hidden_dim=32, dropout=0., flow="trajectorynet"), 8)
    spec["backbone"] = dict(nodes=[
        node("encode", "encoder", "identity", ["features"], hidden_dim=32),
        node("norm", "operation", "layer_norm", ["encode"]),
        node("expand", "operation", "linear", ["norm"], hidden_dim=64),
        node("activation", "operation", "gelu", ["expand"]),
        node("project", "operation", "linear", ["activation"], hidden_dim=32),
        node("skip", "operation", "add", ["encode", "project"]),
        node("spatial", "spatial", "neighbor_mean", ["skip"]),
    ], output="spatial")
    spec["flow"] = dict(nodes=[
        node("time_encoding", "operation", "scfit_time", ["time"], hidden_dim=32, dropout=0.),
        node("condition", "operation", "add", ["context", "time_encoding"]),
        node("tokens", "operation", "stack", ["state", "condition"]),
        node("norm", "operation", "layer_norm", ["tokens"]),
        node("attn", "operation", "scfit_attention", ["norm"], heads=4, dropout=0.),
        node("skip", "operation", "add", ["tokens", "attn"]),
        node("ffn", "operation", "scfit_ffn", ["skip"], expansion=2., dropout=0.),
        node("pooled", "operation", "first_pool", ["ffn"]),
        node("velocity", "operation", "linear", ["pooled"], hidden_dim=32),
    ], output="velocity")
    return spec


def genome(spec=None):
    return CandidateGenome.from_dict(dict(architecture=architecture() if spec is None else spec, learning_rate=.001, seed=17))


def model(spec=None):
    return NanoBrainModel(NanoBrainModelConfig(8, 3, 2, 2, 8), genome(spec))


def test_topology_branches_parameters_gradients_and_reorder():
    torch.manual_seed(5)
    net = model()
    batch = BrainBatch(torch.randn(6, 8), neighbors=torch.tensor([[1], [0], [3], [2], [5], [4]]),
                       neighbor_mask=torch.ones(6, 1, dtype=torch.bool))
    loss = net.annotation_head(net.encode(batch)).square().mean()
    loss.backward()
    grads = {n: any(p.grad is not None and p.grad.abs().sum() > 0 for p in layers.parameters())
             for n, layers in net.backbone.layers.items() if any(True for _ in layers.parameters())}
    assert all(grads.values()), grads
    # Moving spatial aggregation ahead of the nonlinear FFN changes computation.
    spec = architecture(); nodes = spec["backbone"]["nodes"]
    spatial = nodes.pop(); spatial["inputs"] = ["encode"]; nodes.insert(1, spatial)
    nodes[2]["inputs"] = ["spatial"]; spec["backbone"]["output"] = "skip"
    other = model(spec)
    assert architecture_id(net.genome.to_dict()) != architecture_id(other.genome.to_dict())
    assert net.encode(batch).shape == other.encode(batch).shape == (6, 32)


def test_flow_subcomponents_use_context_and_time_and_reload():
    torch.manual_seed(5); net = model().eval()
    state, context, time = torch.randn(6, 32), torch.randn(6, 32), torch.rand(6, 3)
    before = net.flow(state, context, time)
    before.square().mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layers.parameters())
               for layers in net.flow.layers.values() if any(True for _ in layers.parameters()))
    changed_context = context.clone(); changed_context[:, 0] += 2
    assert not torch.allclose(before, net.flow(state, changed_context, time))
    assert not torch.allclose(before, net.flow(state, context, time + 2))
    restored = NanoBrainModel(net.config, CandidateGenome.from_dict(net.genome.to_dict())).eval()
    restored.load_state_dict(net.state_dict())
    torch.testing.assert_close(before, restored.flow(state, context, time))


def test_repeat_has_independent_weights_and_explicit_token_pooling():
    spec = architecture()
    spec["backbone"] = dict(nodes=[
        node("patches", "encoder", "panel_patches", ["features"], hidden_dim=32),
        dict(node("ffn", "operation", "scbrain_ffn", ["patches"], expansion=2., dropout=0.), repeat=3),
        node("pool", "operation", "mean_pool", ["ffn"]),
    ], output="pool")
    net = model(spec)
    layers = net.backbone.layers["node_1"]
    assert len(layers) == 3
    assert len({next(layer.parameters()).data_ptr() for layer in layers}) == 3
    assert net.encode(BrainBatch(torch.randn(5, 8))).shape == (5, 32)


@pytest.mark.parametrize("change,match", [
    (lambda s: s["backbone"]["nodes"][0].update(inputs=["skip"]), "earlier"),
    (lambda s: s["backbone"]["nodes"][1].update(params={"typo": 2}), "unsupported params"),
    (lambda s: s["backbone"]["nodes"][4]["params"].update(hidden_dim=16), "equal shapes"),
    (lambda s: s["backbone"]["nodes"][5].update(repeat=2), "unary"),
    (lambda s: s["backbone"]["nodes"].append(node("dead", "operation", "gelu", ["features"])), "dead"),
    (lambda s: s["backbone"]["nodes"][1].update(name="mean_pool"), "pool requires"),
    (lambda s: s["objectives"]["development"]["params"].update(velocity_weight=0), "cannot be disabled"),
    (lambda s: s["flow"]["nodes"][4]["params"].update(heads=3), "divisible"),
    (lambda s: s["backbone"]["nodes"][3].update(name="silu", params={"inplace": True}), "in-place"),
])
def test_invalid_graph_fails_before_training(change, match):
    spec = architecture(); change(spec)
    with pytest.raises(ValueError, match=match):
        genome(spec)


def test_repeated_spatial_hops_reject_incomplete_context():
    spec = architecture(); spec["backbone"]["nodes"][-1]["repeat"] = 2
    with pytest.raises(ValueError, match="one-hop"):
        genome(spec)


def test_ids_defaults_do_not_create_fake_mutations_and_legacy_hashes_unchanged():
    g = genome().to_dict(); other = copy.deepcopy(g)
    for part in ("backbone", "flow"):
        graph = other["architecture"][part]; mapping = {n["id"]: "renamed_" + n["id"] for n in graph["nodes"]}
        for n in graph["nodes"]:
            n["id"] = mapping[n["id"]]; n["inputs"] = [mapping.get(x, x) for x in n["inputs"]]
        graph["output"] = mapping[graph["output"]]
    assert architecture_id(g) == architecture_id(other)
    assert "architecture" not in CandidateGenome().to_dict()
    assert architecture_id(CandidateGenome(seed=202609180).to_dict()) == "174675ab8d23e025"


@pytest.mark.parametrize("task", TASKS)
def test_graph_all_independent_tasks_train_reload_and_evaluate(tmp_path, task):
    from scripts.run_nanobrain_baselines import _run_nanobrain
    splits = fixture_data(); g = genome()
    receipt = train_candidate(splits, NanoBrainModelConfig(8, 3, 2, 2, 8), g,
        TrainConfig(steps=2, batch_size=8, validation_batch_size=12, development_cells=4,
                    validation_development_groups=1, precision="fp32", task=task), output_dir=tmp_path)
    assert receipt["optimizer_changed_parameter"] and receipt["gradient_nonzero_parameters"] > 0
    assert receipt["architecture_version"] == 2 and receipt["module_manifest"]
    assert not receipt["test_opened"]
    rows = _run_nanobrain(splits, Path(receipt["checkpoint"]), tmp_path / "eval", "cpu", 12, evaluation_seed=5)
    assert rows and {r["task"] for r in rows} == {task}
    assert all(r["status"] not in {"error", "failed"} for r in rows)


def test_graph_completion_mask_and_independent_labels():
    net = model().eval(); batch = fixture_data().train.batch([0, 1, 2, 3])
    first = net.encode(task_input(batch, "spatial_completion"))
    batch.features = batch.features.masked_fill(batch.completion_mask, 9999)
    torch.testing.assert_close(first, net.encode(task_input(batch, "spatial_completion")))
    initial, _ = row_task_loss(net, batch, "integration")
    batch.labels = {"integration": batch.labels["integration"]}
    batch.label_masks = {"integration": batch.label_masks["integration"]}
    reduced, _ = row_task_loss(net, batch, "integration")
    torch.testing.assert_close(initial.total, reduced.total)


def test_flow_edits_cannot_change_annotation_initialization_or_updates():
    one = architecture(); two = copy.deepcopy(one)
    two["flow"]["nodes"][-1]["repeat"] = 2
    a, b = model(one), model(two)
    configure_task(a, "annotation"); configure_task(b, "annotation")
    batch = fixture_data().train.batch([0, 1, 2, 3])
    for net in (a, b):
        optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=.001)
        loss, _ = row_task_loss(net, batch, "annotation"); loss.total.backward(); optimizer.step()
    for key, tensor in a.backbone.state_dict().items():
        torch.testing.assert_close(tensor, b.backbone.state_dict()[key], rtol=0, atol=0)
    for key, tensor in a.annotation_head.state_dict().items():
        torch.testing.assert_close(tensor, b.annotation_head.state_dict()[key], rtol=0, atol=0)


def test_graph_llm_validation_lineage_and_structural_quota():
    g = genome().to_dict(); pid = architecture_id(g); parent = dict(architecture_id=pid, genome=g)
    child = copy.deepcopy(g); child.pop("seed")
    child["architecture"]["backbone"]["nodes"][1]["repeat"] = 2
    proposal = dict(genome=child, parent_ids=[pid], target_tasks=["annotation"], hypothesis="test", expected_tradeoff="test")
    accepted = validate_proposals(dict(candidates=[proposal]), [parent], count=1, seed=17, seen={pid})
    assert has_module_change(accepted[0], [parent])
    child["architecture"] = copy.deepcopy(g["architecture"])
    child["architecture"]["objectives"]["annotation"]["params"]["label_smoothing"] = .1
    accepted = validate_proposals(dict(candidates=[proposal]), [parent], count=1, seed=17, seen={pid})
    assert not has_module_change(accepted[0], [parent])
    with pytest.raises(ValueError, match="unused legacy"):
        CandidateGenome.from_dict({**child, "encoder_depth": 2})


def test_graph_driver_sends_structural_schema_without_api_call(tmp_path, monkeypatch):
    from scripts.run_nanobrain_evolution import Experiment
    e = Experiment.__new__(Experiment); e.output = tmp_path; e.state_path = tmp_path / "campaign.json"
    e.protocol = json.loads(Path("configs/nanobrain_evolution_independent_100k.json").read_text())
    g = genome().to_dict(); pid = architecture_id(g)
    e.protocol.update(reference_genome=g, candidates_per_generation=1, minimum_module_change_candidates=1)
    parent = dict(architecture_id=pid, genome=g, metrics={}, score=.5)
    e.state = dict(reference=parent); e.api = SimpleNamespace(api_key="unused")
    child = copy.deepcopy(g); child.pop("seed"); child["architecture"]["backbone"]["nodes"][1]["repeat"] = 2
    payload = dict(candidates=[dict(genome=child,parent_ids=[pid],target_tasks=["annotation"],hypothesis="test",expected_tradeoff="test")])
    prompts = []
    def complete(prompt, **kwargs):
        prompts.append(prompt); return json.dumps(payload)
    monkeypatch.setattr("scripts.run_nanobrain_evolution.OpenAICompatibleClient", lambda api: SimpleNamespace(complete=complete))
    result = e.propose([parent], [], 0, {pid})
    assert len(result) == len(prompts) == 1
    assert "reorder" in prompts[0] and "INDEPENDENTLY" in prompts[0]


def test_objective_api_changes_training_without_editing_trainer():
    from nanobrain.objectives import OBJECTIVES, register_objective, masked_classification
    def temperature_objective(net, batch, embedding, params, task):
        logits = net.annotation_head(embedding) / params["temperature"]
        return masked_classification(logits, batch.labels["annotation"], batch.label_masks["annotation"]), {}
    register_objective("temperature_classification_test", tasks=("annotation",),
                       compute=temperature_objective, defaults={"temperature": 1.})
    try:
        spec = architecture()
        spec["objectives"]["annotation"] = dict(name="temperature_classification_test", params={"temperature": 2.})
        custom = model(spec).eval(); default = model().eval()
        batch = fixture_data().train.batch([0, 1, 2, 3])
        left, _ = row_task_loss(custom, batch, "annotation"); right, _ = row_task_loss(default, batch, "annotation")
        assert not torch.allclose(left.total, right.total)
        left.total.backward()
        assert next(custom.annotation_head.parameters()).grad.abs().sum() > 0
    finally:
        del OBJECTIVES["temperature_classification_test"]


def test_development_decoder_and_loss_selection_are_effective():
    spec = architecture(); spec["heads"]["development"]["name"] = "scfit"
    spec["heads"]["development"]["params"] = {"dropout": 0.}
    spec["objectives"]["development"]["params"]["expression_regression"] = "smooth_l1"
    net = model(spec); configure_task(net, "development")
    batch = fixture_data().development_train.batch(0, max_cells=4, seed=17)
    loss, _ = net.population_flow_losses(batch)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.development_head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in net.completion_head.parameters())
    assert net.predict_development_population(batch.source, 10, 20, steps=2)[0].shape == (4, 8)


def test_task_head_subgraph_has_own_width_and_gradients():
    spec = architecture()
    spec["heads"]["annotation"] = dict(name="classification", params={}, pre=dict(nodes=[
        node("project", "operation", "linear", ["state"], hidden_dim=16),
        dict(node("ffn", "operation", "scfit_ffn", ["project"], expansion=2., dropout=0.), repeat=2),
        node("skip", "operation", "add", ["project", "ffn"]),
    ], output="skip"))
    net = model(spec); configure_task(net, "annotation")
    batch = fixture_data().train.batch([0, 1, 2, 3]); loss, _ = row_task_loss(net, batch, "annotation")
    loss.total.backward()
    assert net.annotation_head.output.projection.in_features == 16
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.annotation_head.parameters())
    restored = NanoBrainModel(net.config, CandidateGenome.from_dict(net.genome.to_dict()))
    restored.load_state_dict(net.state_dict())
    assert has_module_change(dict(genome=net.genome.to_dict(), parent_ids=["parent"], changes={}),
                             [dict(architecture_id="parent", genome=genome().to_dict())])


def test_input_data_contract_cannot_be_changed_by_llm():
    g = genome().to_dict(); pid = architecture_id(g); child = copy.deepcopy(g); child.pop("seed")
    child["architecture"]["input_dim"] = 16
    proposal = dict(genome=child, parent_ids=[pid], target_tasks=["annotation"], hypothesis="test", expected_tradeoff="test")
    with pytest.raises(ValueError, match="fixed data"):
        validate_proposals(dict(candidates=[proposal]), [dict(architecture_id=pid, genome=g)], count=1, seed=17, seen={pid})


def test_standard_json_patch_edits_topology_without_mutating_parent():
    g = genome().to_dict(); pid = architecture_id(g); before = copy.deepcopy(g)
    parent = dict(architecture_id=pid, genome=g)
    proposal = dict(parent_ids=[pid],target_tasks=["annotation"],hypothesis="test",expected_tradeoff="test",
                    patch=[dict(op="replace",path="/architecture/backbone/nodes/1/repeat",value=2)])
    result = validate_proposals(dict(candidates=[proposal]), [parent], count=1, seed=17, seen={pid})[0]
    assert result["genome"]["architecture"]["backbone"]["nodes"][1]["repeat"] == 2
    assert has_module_change(result, [parent]) and g == before
    proposal["patch"][0]["path"] = "/seed"
    with pytest.raises(ValueError, match="restricted"):
        validate_proposals(dict(candidates=[proposal]), [parent], count=1, seed=17, seen={pid})
