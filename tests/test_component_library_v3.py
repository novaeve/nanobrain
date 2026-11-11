import copy

import numpy as np
import pytest
import torch

from nanobrain.registry import registry
from nanobrain.architecture import validate_architecture
from nanobrain.genome import CandidateGenome
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
from nanobrain.training import TrainConfig, train_candidate
from nanobrain.task_training import configure_task
from test_nanobrain_graph import architecture
from test_nanobrain_independent_tasks import fixture_data

ENTRIES = [
    ("encoder", "sctab_tabnet"), ("operation", "sctab_glu"), ("spatial", "stagate_gat"),
    ("encoder", "scpoli_conditional"), ("head", "scpoli_prototype"),
    ("flow", "mioflow_ode"), ("flow", "prescient_potential"),
    ("encoder", "scnym_mlp"), ("residual", "scnym_resblock"),
    ("encoder", "sedr_mlp"), ("spatial", "sedr_gcn"), ("spatial", "spagcn_gcn"),
    ("encoder", "scfoundation_autobin"), ("encoder", "scprint_value"),
    ("head", "scprint_classifier"), ("attention", "scbert_performer"),
    ("endpoint", "cellot_icnn_transport"),
]


def invoke(net, kind, name, x):
    if name == "scpoli_conditional":
        return net(x, torch.arange(len(x), device=x.device).remainder(2)[:, None])
    if kind in {"flow", "endpoint"}:
        return net(x, x * .4, torch.ones(len(x), 3, device=x.device))
    if kind == "spatial":
        ids = torch.arange(len(x), device=x.device)
        return net(x, neighbors=(ids + 1).remainder(len(x))[:, None],
                   neighbor_mask=torch.ones(len(x), 1, device=x.device, dtype=torch.bool))
    return net(x)


@pytest.mark.parametrize("kind,name", ENTRIES)
def test_all_sources_update_and_reload_with_no_grad(kind, name):
    torch.manual_seed(171)
    params = {"hidden_dim": 32}
    if kind == "encoder": params["input_dim"] = 8
    if kind == "head": params["output_dim"] = 3
    net = registry(kind).create(name, **params)
    x = torch.randn(12, 8 if kind == "encoder" else 32)
    if name == "scprint_value": x = x.abs()
    if kind == "attention": x = x[:, None].expand(-1, 4, -1).clone()
    if name == "scpoli_prototype":
        net.observe(x, torch.arange(12).remainder(3), torch.ones(12, dtype=torch.bool))
    before = {k: p.detach().clone() for k, p in net.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    for step in range(2):
        optimizer.zero_grad()
        output = invoke(net, kind, name, x)
        assert torch.isfinite(output).all()
        # Nonzero target is essential for the zero-initialized potential field.
        loss = (output.float() - .5).square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
    assert any(not torch.equal(before[k], p) for k, p in net.named_parameters() if k in before), name
    net.eval()
    state = copy.deepcopy(net.state_dict())
    fresh = registry(kind).create(name, **params).eval()
    fresh.load_state_dict(state)
    with torch.no_grad():
        expected = invoke(net, kind, name, x)
        actual = invoke(fresh, kind, name, x)
    torch.testing.assert_close(expected, actual)


def config_genome(kind, name):
    spec = architecture()
    spec["version"] = 3
    first = spec["backbone"]["nodes"][0]
    if kind == "encoder":
        first["name"] = name
        first["params"] = {"hidden_dim": 32}
        if name == "scpoli_conditional": first["inputs"] = ["features", "batch_id"]
        if name == "scprint_value": first["inputs"] = ["nonnegative_features"]
        if name in {"scfoundation_autobin", "scprint_value"}:
            for node in spec["backbone"]["nodes"][1:]:
                node["inputs"] = ["pool" if r == "encode" else r for r in node["inputs"]]
            spec["backbone"]["nodes"].insert(1, dict(id="pool", kind="operation", name="mean_pool", inputs=["encode"], params={}))
    elif kind == "head":
        spec["heads"]["annotation"] = {"name": name, "params": {}}
    elif kind == "spatial":
        spec["backbone"]["nodes"][-1]["name"] = name
    elif kind in {"operation", "residual", "attention"}:
        nodes = spec["backbone"]["nodes"]
        node = dict(id="extension", kind=kind, name=name, inputs=[nodes[-1]["id"]], params={})
        nodes.append(node); spec["backbone"]["output"] = "extension"
    elif kind == "flow":
        spec["flow"] = dict(nodes=[dict(id="field", kind="flow", name=name,
            inputs=["state", "context", "time"], params={})], output="field")
    elif kind == "endpoint":
        spec["endpoint"] = {"name": name, "params": {}}
        spec["objectives"]["development"] = dict(name="endpoint_transport", params={
            "endpoint_regression": "mse", "expression_regression": "mse", "endpoint_weight": 1., "expression_weight": 1.})
    return CandidateGenome.from_dict(dict(architecture=spec, learning_rate=.001, seed=17))


@pytest.mark.parametrize("kind,name", ENTRIES)
def test_all_registry_entries_work_in_existing_trainer_and_evaluator(tmp_path, kind, name):
    splits = fixture_data()
    for dataset in (splits.train, splits.validation):
        dataset.metadata["feature_normalization"] = {"mean": np.full(8, 4., dtype=np.float32),
                                                     "std": np.ones(8, dtype=np.float32)}
    genome = config_genome(kind, name)
    model_config = NanoBrainModelConfig(8, 3, 2, 2, 8)
    task = "development" if kind in {"flow", "endpoint"} else "annotation"
    receipt = train_candidate(splits, model_config, genome, TrainConfig(steps=2, batch_size=8,
        validation_batch_size=12, development_cells=4, validation_development_groups=1,
        task=task, precision="fp32"), output_dir=tmp_path)
    assert receipt["stability"]["passed"] and receipt["optimizer_changed_parameter"]
    payload = torch.load(receipt["checkpoint"], weights_only=False)
    fresh = NanoBrainModel(model_config, CandidateGenome.from_dict(payload["genome"]))
    fresh.load_state_dict(payload["model"])
    fresh.eval()
    batch = splits.validation.batch(np.arange(6))
    with torch.no_grad():
        if task == "development":
            predicted, _ = fresh.predict_development_population(batch, 10., 20., steps=2)
        else:
            predicted = fresh.annotation_head(fresh.encode(batch))
    assert torch.isfinite(predicted).all()


def test_prototype_is_fitted_on_training_only_and_checkpointed():
    net = registry("head").create("scpoli_prototype", hidden_dim=8, output_dim=3)
    x = torch.randn(12, 8)
    labels = torch.arange(12).remainder(3)
    net.observe(x, labels, torch.ones(12, dtype=torch.bool))
    before = copy.deepcopy(net.state_dict())
    net.eval()
    with pytest.raises(RuntimeError): net.observe(x, labels, None)
    net(x)
    for k, value in before.items(): torch.testing.assert_close(value, net.state_dict()[k])


def test_scprint_signed_features_rejected_and_endpoint_semantics_checked():
    spec = config_genome("encoder", "scprint_value").to_dict()["architecture"]
    spec["backbone"]["nodes"][0]["inputs"] = ["features"]
    with pytest.raises(ValueError): validate_architecture(spec)
    spec = config_genome("endpoint", "cellot_icnn_transport").to_dict()["architecture"]
    spec["endpoint"]["params"]["hidden_dim"] = 64
    with pytest.raises(ValueError): validate_architecture(spec)


def test_conditional_encoder_has_no_biology_label_dependency():
    g = config_genome("encoder", "scpoli_conditional")
    net = NanoBrainModel(NanoBrainModelConfig(8, 3, 2, 2, 8), g).eval()
    batch = fixture_data().validation.batch(np.arange(6))
    before = net.encode(batch)
    for label in ["annotation", "niche", "development"]:
        batch.labels[label] = torch.full_like(batch.labels[label], 999)
    torch.testing.assert_close(before, net.encode(batch))
    configure_task(net, "annotation")
    assert all(not p.requires_grad for n, p in net.named_parameters() if "log_var_encoder" in n)
