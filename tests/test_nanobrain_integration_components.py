import copy

import numpy as np
import pytest
import torch

from nanobrain.architecture import ModuleGraph, Shape, validate_architecture
from nanobrain.data import make_synthetic_splits
from nanobrain.integration_components import compute_component, prepare_component_views, PREFIX
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
from nanobrain.genome import CandidateGenome
from nanobrain.training import train_candidate, TrainConfig
from test_nanobrain_graph import architecture


def new_genome(method="harmony"):
    spec = architecture()
    spec["backbone"]["nodes"].insert(0, dict(id="integrated", kind="operation", name=method,
        inputs=["features"], params={"strength": .75}))
    spec["backbone"]["nodes"][1]["inputs"] = ["integrated"]
    spec["heads"]["annotation"] = dict(name="celltypist_linear", params={})
    return CandidateGenome.from_dict(dict(architecture=spec, learning_rate=.001, seed=17))


@pytest.mark.parametrize("method", ["harmony", "scanorama", "bbknn"])
def test_official_components_have_real_finite_outputs_and_keep_rows(method):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    batch = np.repeat(np.arange(3), 20)
    x[batch == 1] += 2
    values = compute_component(method, x, batch)
    assert values.shape == x.shape and np.isfinite(values).all()
    assert not np.allclose(values, x)
    if method != "harmony":
        assert np.allclose(values, compute_component(method, x, batch), atol=1e-5)


def test_cached_views_ignore_biology_keep_spatial_context_and_do_not_open_test(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOBRAIN_COMPONENT_CACHE", str(tmp_path))
    splits = make_synthetic_splits(train_rows=60, validation_rows=30, test_rows=20, input_dim=8)
    splits.train.metadata["subset_manifest"] = {"identity_repair": True}
    g = new_genome()
    receipts = prepare_component_views(splits, g)
    before = np.array(splits.train.metadata[PREFIX + "harmony"])
    assert PREFIX + "harmony" not in splits.test.metadata
    for d in (splits.train, splits.validation):
        d.labels["annotation"].fill_(999)
        d.labels["niche"].fill_(999)
    again = prepare_component_views(splits, g)
    assert [r["output_sha256"] for r in receipts] == [r["output_sha256"] for r in again]
    np.testing.assert_array_equal(before, splits.train.metadata[PREFIX + "harmony"])
    batch = splits.train.batch(np.array([5, 2, 9]))
    np.testing.assert_array_equal(batch.metadata[PREFIX + "harmony"][:3], before[[5, 2, 9]])
    net = NanoBrainModel(NanoBrainModelConfig(8, 5, 4, 3, 8), g)
    logits = net.annotation_head(net.encode(batch))
    logits.square().mean().backward()
    assert net.annotation_head.weight.grad.abs().sum() > 0
    encoder = next(n for n in net.backbone.plan if n["kind"] == "encoder")
    assert next(net.backbone.layers[encoder["id"]].parameters()).grad.abs().sum() > 0


def test_contract_rejects_embedding_flow_repeat_and_bad_strength():
    g = new_genome()
    for change in ("embedding", "repeat", "strength"):
        spec = copy.deepcopy(g.to_dict()["architecture"])
        n = spec["backbone"]["nodes"][0]
        if change == "embedding":
            n["inputs"] = ["encode"]
        elif change == "repeat":
            n["repeat"] = 2
        else:
            n["params"]["strength"] = 1.1
        with pytest.raises(ValueError):
            validate_architecture(spec)
    with pytest.raises(ValueError):
        ModuleGraph(dict(nodes=[dict(id="h", kind="operation", name="harmony", inputs=["state"])],
                         output="h"), {"state": Shape(8)})


def test_new_components_train_and_checkpoint_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOBRAIN_COMPONENT_CACHE", str(tmp_path / "cache"))
    splits = make_synthetic_splits(train_rows=60, validation_rows=30, test_rows=20, input_dim=8)
    g = new_genome("bbknn")
    config = NanoBrainModelConfig(8, 5, 4, 3, 8)
    result = train_candidate(splits, config, g, TrainConfig(steps=2, batch_size=16,
        validation_batch_size=30, task="annotation", precision="fp32"), output_dir=tmp_path / "run")
    assert result["optimizer_changed_parameter"] and result["stability"]["passed"]
    assert len(result["integration_component_receipts"]) == 2
    payload = torch.load(result["checkpoint"], weights_only=False)
    net = NanoBrainModel(config, CandidateGenome.from_dict(payload["genome"]))
    net.load_state_dict(payload["model"])
    assert torch.isfinite(net.encode(splits.validation.batch(np.arange(5)))).all()


def test_bbknn_accepts_singleton_batches():
    x = np.random.default_rng(18).normal(size=(31, 8)).astype(np.float32)
    values = compute_component("bbknn", x, np.array([0] + [1] * 15 + [2] * 15))
    assert values.shape == x.shape and np.isfinite(values).all()
