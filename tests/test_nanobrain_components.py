import json

import numpy as np
import pytest
import torch

from nanobrain.contracts import BrainBatch
from nanobrain.genome import CandidateGenome, SEARCH_SPACE
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
from nanobrain.component_catalog import component_catalog
from nanobrain.task_training import configure_task, row_task_loss, task_input


NEW_COMPONENTS = [("encoder", name) for name in ("scbrain_swiglu", "scnode_mlp", "scvi_fc", "panel_patches")]
NEW_COMPONENTS += [("attention", name) for name in ("scfit_block", "bert", "torch_transformer")]
NEW_COMPONENTS += [("spatial", "scbrain_gcn"), ("flow", "scfit_adaln"), ("flow", "scnode_time_mlp"),
                   ("spatial", "graphst_sparse"), ("flow", "trajectorynet"), ("flow", "scuniflow_field"),
                   ("classification_head", "scbrain"), ("classification_head", "scfit"), ("regression_head", "scfit")]


def model(**overrides):
    genome = CandidateGenome.from_dict({**CandidateGenome(hidden_dim=32, dropout=0.).to_dict(), **overrides})
    return NanoBrainModel(NanoBrainModelConfig(32, 3, 3, 3), genome)


@pytest.mark.parametrize("field,name", NEW_COMPONENTS)
def test_imported_components_compose_and_update(field, name):
    torch.manual_seed(9)
    net = model(**{field: name})
    batch = BrainBatch(torch.randn(6, 32), neighbors=torch.tensor([[1],[0],[1],[2],[3],[4]]),
                       neighbor_mask=torch.ones(6, 1, dtype=torch.bool))
    state = net.encode(batch)
    assert state.shape == (6, 32)
    velocity = net.flow(state, state, torch.rand(6, 3))
    loss = net.annotation_head(state).square().mean() + net.completion_head(velocity).square().mean()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    loss.backward()
    assert torch.isfinite(loss)
    optimizer.step()
    prefix = {"classification_head": "annotation_head", "regression_head": "completion_head"}.get(field, field)
    assert any(not torch.equal(p, before[n]) for n, p in net.named_parameters() if n.startswith(prefix + "."))
    restored = model(**net.genome.to_dict())
    restored.load_state_dict(net.state_dict())
    net.eval(); restored.eval()
    torch.testing.assert_close(net.encode(batch), restored.encode(batch))


def test_completion_targets_cannot_be_read_by_encoder():
    net = model(encoder="panel_patches", attention="bert", regression_head="scfit").eval()
    configure_task(net, "spatial_completion")
    x = torch.randn(6, 32)
    mask = torch.zeros_like(x, dtype=torch.bool); mask[:, :4] = True
    batch = BrainBatch(x, completion_target=x.clone(), completion_mask=mask)
    initial = net.encode(task_input(batch, "spatial_completion"))
    batch.features = x.masked_fill(mask, 123456.)
    torch.testing.assert_close(initial, net.encode(task_input(batch, "spatial_completion")))
    assert not task_input(batch, "spatial_completion").feature_observed[mask].any()
    loss, _ = row_task_loss(net, batch, "spatial_completion")
    loss.total.backward()
    assert torch.isfinite(loss.total)


def test_new_flow_uses_age_interval_and_exact_scfit_depth():
    for name in ("scfit_adaln", "scnode_time_mlp"):
        net = model(flow=name, flow_depth=3).eval()
        state = torch.randn(6, 32); time = torch.rand(6, 3)
        before = net.flow(state, state, time)
        # scFiT modulation starts at zero; the conditioning path becomes active
        # after training, so prove time dependence after a gradient update.
        optimizer = torch.optim.AdamW(net.flow.parameters(), lr=.01)
        before.square().mean().backward(); optimizer.step()
        changed = time.clone(); changed[:, 1:] += 2
        assert not torch.allclose(net.flow(state, state, time), net.flow(state, state, changed))
        if name == "scfit_adaln":
            assert len(net.flow.blocks) == 3


def test_component_catalog_matches_all_search_choices_and_rejects_unknown_fields():
    catalog = component_catalog()
    json.dumps(catalog)
    for field in ("encoder", "attention", "spatial", "residual", "pairing", "flow", "classification_head", "regression_head"):
        kind = "head" if field.endswith("_head") else field
        available = {entry["name"] for entry in catalog[kind]}
        assert set(SEARCH_SPACE[field]) <= available
    with pytest.raises(ValueError, match="unknown genome"):
        CandidateGenome.from_dict({"encoder_depht": 2})


def test_metadata_v2_preserves_outside_batch_neighbors_without_context_labels():
    from nanobrain.data import NanoDataset
    x = torch.randn(6, 32)
    dataset = NanoDataset(x, {"annotation": torch.arange(6)}, {},
        neighbors=torch.tensor([[1], [0], [3], [2], [5], [4]]),
        neighbor_mask=torch.ones(6, 1, dtype=torch.bool),
        metadata={"subset_manifest": {"identity_repair": {"version": 2}}},
        completion_mask=torch.zeros_like(x, dtype=torch.bool))
    batch = dataset.batch(np.array([0, 4]))
    assert batch.features.shape[0] == 2 and batch.labels["annotation"].shape[0] == 2
    assert batch.neighbor_features.shape[0] == 2
    assert batch.neighbor_mask.all()
    net = model(spatial="scbrain_gcn").eval()
    initial = net.encode(batch)
    batch.neighbor_features = batch.neighbor_features + 3
    changed = net.encode(batch)
    assert not torch.allclose(initial, changed)
    changed.square().mean().backward()
    assert net.spatial.block.linear.weight.grad.abs().sum() > 0
    # Every target gets the same one-hop context regardless of minibatch width.
    torch.testing.assert_close(net.encode(dataset.batch(np.array([0, 4]))), net.encode(dataset.batch(np.arange(6)))[[0,4]])
