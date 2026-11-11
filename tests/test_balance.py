"""Loss balancers and dual weights on toy losses (CPU)."""

import math

import pytest
import torch

from brainwm_unified.balance import BALANCERS, FAMO, CapacityWeight, DualWeight, GradEqual, InitialLoss, build_balancer
from brainwm_unified.objectives import TASKS


class Toy(torch.nn.Module):
    """Shared encoder; each loss is a differently scaled function of its output."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = torch.nn.Linear(8, 8)
        self.x = torch.randn(32, 8)

    def losses(self, factors: dict) -> dict:
        h = self.encoder(self.x)
        return {n: f * h[:, i].pow(2).mean() for i, (n, f) in enumerate(factors.items())}


FACTORS = {"gene": 1.0, "integration": 3000.0, "cell_type": 0.5, "anatomy": 2.0, "spatial": 5.0, "extra": 0.01}


def test_initial_matches_the_frozen_normaliser():
    bal = InitialLoss(norm_steps=2)
    seq = [{t: torch.tensor(float(i + k)) for k, t in enumerate(TASKS)} for i in (1, 3, 9)]
    for losses in seq:
        scales = bal.scales(0, losses, None)
    assert scales == pytest.approx({t: 2 / ((1 + k) + (3 + k)) for k, t in enumerate(TASKS)})  # first two steps only
    assert set(scales) == set(TASKS)


def test_grad_equal_gives_every_loss_the_same_encoder_gradient():
    model = Toy()
    bal = GradEqual(norm_steps=2, every=1, beta=0.0)
    for step in range(3):
        losses = model.losses(FACTORS)
        scales = bal.scales(step, losses, model)
    assert set(scales) == set(FACTORS)  # the extra objective is balanced too
    scaled = {n: scales[n] * losses[n] for n in losses}
    params = list(model.encoder.parameters())
    norms = [
        torch.cat([g.flatten() for g in torch.autograd.grad(v, params, retain_graph=True)]).norm().item()
        for v in scaled.values()
    ]
    assert max(norms) / min(norms) == pytest.approx(1.0, rel=1e-4)


def test_famo_moves_weight_to_the_loss_that_stalls():
    bal = FAMO(w_lr=0.1)
    fast, slow = 1.0, 1.0
    for step in range(20):
        bal.scales(step, {"fast": torch.tensor(fast), "slow": torch.tensor(slow)}, None)
        fast *= 0.8
        slow *= 0.99
        bal.after_step({"fast": torch.tensor(fast), "slow": torch.tensor(slow)})
    w = bal.weights()
    assert w["slow"] > 0.6 > w["fast"]
    again = FAMO(w_lr=0.1)
    again.load_state_dict(bal.state_dict())
    assert again.weights() == w


def test_famo_scales_reproduce_the_log_objective_gradient():
    bal = FAMO()
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    losses = {"a": p[0] ** 2 * 10, "b": p[1] ** 2 * 0.1}
    scales = bal.scales(0, losses, None)
    (g_scaled,) = torch.autograd.grad(sum(scales[n] * losses[n] for n in losses), p, retain_graph=True)
    z = torch.full((2,), 0.5)
    L = torch.stack([losses["a"], losses["b"]])
    c = (z / L.detach()).sum()
    (g_log,) = torch.autograd.grad((z * L.log()).sum() / c, p)
    assert torch.allclose(g_scaled, g_log)


def test_dual_weight_follows_the_budget():
    d = DualWeight({"cell_type": 0.3}, init=1.0, lr=0.1, beta=0.0)
    for _ in range(10):
        d.update({"cell_type": 0.6})  # twice the budget
    assert d.weight == pytest.approx(math.exp(-1.0))
    for _ in range(10):
        d.update({"cell_type": 0.15})  # half the budget
    assert d.weight == pytest.approx(math.exp(-0.5))
    for _ in range(1000):
        d.update({"cell_type": 30.0})
    assert d.weight == pytest.approx(d.floor)
    restored = DualWeight({"cell_type": 0.3})
    restored.load_state_dict(d.state_dict())
    assert restored.weight == d.weight and restored.ema == d.ema


def test_worst_guard_decides():
    d = DualWeight({"cell_type": 0.3, "anatomy": 0.4}, init=1.0, lr=0.1, beta=0.0)
    d.update({"cell_type": 0.15, "anatomy": 0.8})  # one under, one over: shrinks
    assert d.weight < 1.0


def test_build_balancer_fills_norm_steps_and_rejects_unknown():
    assert build_balancer("initial", 7, {}).norm_steps == 7
    assert build_balancer("grad_equal", 7, {"norm_steps": 3}).norm_steps == 3
    assert isinstance(build_balancer("famo", 7, {}), FAMO)
    with pytest.raises(TypeError, match="valid"):
        build_balancer("famo", 7, {"every": 3})
    assert set(BALANCERS.names()) == {"initial", "grad_equal", "famo"}


def test_capacity_weight_holds_the_penalty_at_its_target():
    w = CapacityWeight(target=30.0, init=1.0, lr=0.1, beta=0.0)
    for _ in range(10):
        w.update(60.0)  # twice the target: the weight grows
    assert w.weight == pytest.approx(math.exp(1.0))
    for _ in range(20):
        w.update(15.0)  # half the target: it shrinks
    assert w.weight == pytest.approx(math.exp(0.0))
    restored = CapacityWeight(target=30.0)
    restored.load_state_dict(w.state_dict())
    assert restored.weight == w.weight and restored.ema == w.ema
