import types

import pytest
import torch

from brainwm.foundation_moe import BrainWMFoundationMoE, ModalityAdapter


def _constant_velocity(self, context, state, tau, source_time, target_time):
    del state, tau, source_time, target_time
    return torch.full_like(context, 2.0)


def test_categorical_adapter_keeps_legacy_numeric_branch_frozen():
    adapter=ModalityAdapter(8,value_dim=4,categorical=True)
    assert not adapter.value.weight.requires_grad
    assert not adapter.value.bias.requires_grad
    assert adapter.category.weight.requires_grad
    assert "value.weight" in adapter.state_dict()


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_development_rollout_integrates_normalized_velocity(solver):
    model = BrainWMFoundationMoE(7, d_model=4, d_hidden=8, n_blocks=1,
                                 n_experts=2, moe_backend="mage_beta")
    model.velocity_at = types.MethodType(_constant_velocity, model)
    source = torch.zeros(3, 4)
    endpoint, trajectory = model.predict_development(
        source, source, 10.0, 30.0, steps=5, solver=solver,
        return_trajectory=True,
    )
    assert torch.allclose(endpoint, torch.full_like(source, 2.0))
    assert trajectory.shape == (6, 3, 4)


def test_development_rollout_missing_and_equal_time_are_identity():
    model = BrainWMFoundationMoE(7, d_model=4, d_hidden=8, n_blocks=1,
                                 n_experts=2, moe_backend="mage_beta")
    model.velocity_at = types.MethodType(_constant_velocity, model)
    source = torch.randn(3, 4)
    endpoint = model.predict_development(
        source, source, torch.tensor([1.0, float("nan"), 3.0]),
        torch.tensor([2.0, float("nan"), 3.0]), steps=2,
    )
    assert torch.allclose(endpoint[0], source[0] + 2.0)
    assert torch.equal(endpoint[1:], source[1:])


def test_development_rollout_rejects_backward_observed_time():
    model = BrainWMFoundationMoE(7, d_model=4, d_hidden=8, n_blocks=1,
                                 n_experts=2, moe_backend="mage_beta")
    state = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="target_time >= source_time"):
        model.predict_development(state, state, 2.0, 1.0)


def test_self_describing_checkpoint_round_trip(tmp_path):
    model = BrainWMFoundationMoE(
        7, d_model=4, d_hidden=8, n_blocks=1, n_experts=2,
        modality_dims={"life_stage": 3}, categorical_modalities={"life_stage"},
        annotation_classes=2, anatomy_classes=3, moe_backend="mage_beta",
    )
    path = tmp_path / "model.pt"
    torch.save({"model_config": model.model_config, "model": model.state_dict()}, path)
    restored, payload = BrainWMFoundationMoE.from_checkpoint(path)
    assert restored.model_config == model.model_config
    assert payload["model_config"]["n_experts"] == 2
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), restored.state_dict().values()))


def test_development_training_outputs_include_endpoint_supervision():
    model = BrainWMFoundationMoE(7, d_model=4, d_hidden=8, n_blocks=1,
                                 n_experts=2, moe_backend="mage_beta")
    batch = types.SimpleNamespace(
        developmental_observed=torch.tensor([True, False]),
        source_state=torch.zeros(2, 4), target_state=torch.ones(2, 4),
        source_time=torch.tensor([1.0, float("nan")]),
        target_time=torch.tensor([2.0, float("nan")]), flow_tau=torch.tensor([0.5, 0.5]),
    )
    cell = torch.randn(2, 4)
    velocity, target, endpoint, endpoint_target = model.development_outputs(cell, batch)
    assert torch.equal(target[0], torch.ones(4))
    assert torch.equal(endpoint_target[0], torch.ones(4))
    assert torch.equal(velocity[1], torch.zeros(4))
    assert torch.equal(endpoint[1], torch.zeros(4))


def test_development_training_outputs_scatter_has_finite_gradients():
    model = BrainWMFoundationMoE(7, d_model=4, d_hidden=8, n_blocks=1,
                                 n_experts=2, moe_backend="mage_beta")
    batch = types.SimpleNamespace(
        developmental_observed=torch.tensor([True, False, True]),
        source_state=torch.zeros(3, 4), target_state=torch.ones(3, 4),
        source_time=torch.tensor([1.0, float("nan"), 2.0]),
        target_time=torch.tensor([2.0, float("nan"), 4.0]),
        flow_tau=torch.tensor([0.5, 0.5, 0.25]),
    )
    cell = torch.randn(3, 4, requires_grad=True)
    outputs = model.development_outputs(cell, batch)
    loss = sum(value.square().mean() for value in outputs)
    loss.backward()
    assert torch.isfinite(cell.grad).all()
    assert torch.equal(outputs[0][1], torch.zeros(4))
