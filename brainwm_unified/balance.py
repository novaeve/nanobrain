"""How the task losses and extra objectives become one training loss.

    [train]
    balancer = "grad_equal"            # registered name
    balancer_params = { every = 50 }   # the balancer's own parameters

  initial     each task loss over its mean in the first norm_steps steps, then frozen   (Cajal v0)
  grad_equal  every loss gets the same gradient norm on the encoder (EMA of measured)  GradNorm-style
  famo        weights follow each loss's log-decrease rate, one extra forward a step   FAMO (NeurIPS 2023)

A balancer returns a multiplier per loss: total = sum(weight * scale * loss). Losses it does not
scale (extra objectives under `initial`) get 1.0. An objective given a `budget` gets a DualWeight
as its weight instead of a constant: a Lagrange multiplier that grows while the guarded losses stay
within budget and shrinks when one goes over.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .objectives import TASKS
from .registry import Registry

BALANCERS = Registry("balancer")


def encoder_grad_norms(model, losses: dict, names) -> dict[str, float]:
    """Gradient norm of each named loss on the encoder parameters (the graph is kept)."""
    params = [p for p in model.encoder.parameters() if p.requires_grad]
    out = {}
    for n in names:
        grads = torch.autograd.grad(losses[n], params, retain_graph=True, allow_unused=True)
        out[n] = math.sqrt(sum(float(g.float().pow(2).sum()) for g in grads if g is not None))
    return out


@BALANCERS.register("initial")
@dataclass
class InitialLoss:
    """Each task loss divided by its mean over the first `norm_steps` steps, then frozen; extras unscaled."""

    norm_steps: int = 200
    sums: dict = field(init=False, default_factory=lambda: {t: 0.0 for t in TASKS})
    n: int = field(init=False, default=0)
    wants_after_step = False

    def scales(self, step: int, losses: dict, model) -> dict[str, float]:
        if self.n < self.norm_steps:
            for t in TASKS:
                self.sums[t] += float(losses[t])
            self.n += 1
        n = max(self.n, 1)
        return {t: 1.0 / max(self.sums[t] / n, 1e-8) if self.n else 1.0 for t in TASKS}

    def state_dict(self):
        return {"sums": self.sums, "n": self.n}

    def load_state_dict(self, s):
        self.sums, self.n = s["sums"], s["n"]


@BALANCERS.register("grad_equal")
@dataclass
class GradEqual:
    """Every loss, extras included, gets the same gradient norm on the encoder.

    Norms are measured every `every` steps (one backward per loss) and smoothed with `beta`. The
    common target is the mean task norm that `initial` would give, so the total keeps its usual
    size against gradient clipping. The first `norm_steps` steps run as `initial`.
    """

    norm_steps: int = 200
    every: int = 50
    beta: float = 0.9
    initial: InitialLoss = field(init=False)
    ema: dict = field(init=False, default_factory=dict)

    wants_after_step = False

    def __post_init__(self):
        self.initial = InitialLoss(self.norm_steps)

    def scales(self, step: int, losses: dict, model) -> dict[str, float]:
        base = self.initial.scales(step, losses, model)
        if step % self.every == 0:
            for n, g in encoder_grad_norms(model, losses, list(losses)).items():
                self.ema[n] = g if n not in self.ema else self.beta * self.ema[n] + (1 - self.beta) * g
        if step < self.norm_steps or not all(n in self.ema for n in losses):
            return base
        target = sum(base[t] * self.ema[t] for t in TASKS) / len(TASKS)
        return {n: target / max(self.ema[n], 1e-12) for n in losses}

    def state_dict(self):
        return {"initial": self.initial.state_dict(), "ema": dict(self.ema)}

    def load_state_dict(self, s):
        self.initial.load_state_dict(s["initial"])
        self.ema = dict(s["ema"])


@BALANCERS.register("famo")
@dataclass
class FAMO:
    """Fast adaptive multitask optimisation (Liu et al., NeurIPS 2023), over every loss.

    The model follows sum(z_i log L_i) / c with z = softmax(w) and c = sum(z_i / L_i), written as
    scales z_i / (c L_i) on the losses. After the optimiser step the same batch is evaluated again
    and w moves so that losses decreasing slowly in log space gain weight.
    """

    w_lr: float = 0.025
    gamma: float = 0.01  # weight decay on w
    eps: float = 1e-8
    names: list = field(init=False, default_factory=list)
    w: torch.Tensor = field(init=False, default=None)
    opt: torch.optim.Optimizer = field(init=False, default=None)
    prev: torch.Tensor = field(init=False, default=None)

    wants_after_step = True

    def _setup(self, names):
        self.names = list(names)
        self.w = torch.zeros(len(self.names), requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=self.w_lr, weight_decay=self.gamma)

    def _vector(self, losses: dict) -> torch.Tensor:
        return torch.tensor([float(losses[n]) for n in self.names]).clamp(min=self.eps)

    def scales(self, step: int, losses: dict, model) -> dict[str, float]:
        if self.w is None:
            self._setup(losses)
        self.prev = self._vector(losses)
        z = F.softmax(self.w.detach(), -1)
        c = (z / self.prev).sum()
        return {n: float(z[i] / (c * self.prev[i])) for i, n in enumerate(self.names)}

    def after_step(self, losses: dict):
        delta = self.prev.log() - self._vector(losses).log()
        with torch.enable_grad():
            (d,) = torch.autograd.grad(F.softmax(self.w, -1), self.w, grad_outputs=delta)
        self.opt.zero_grad()
        self.w.grad = d
        self.opt.step()

    def weights(self) -> dict[str, float]:
        z = F.softmax(self.w.detach(), -1)
        return {n: float(z[i]) for i, n in enumerate(self.names)}

    def state_dict(self):
        return {"names": self.names, "w": self.w.detach().clone(), "opt": self.opt.state_dict()}

    def load_state_dict(self, s):
        self._setup(s["names"])
        with torch.no_grad():
            self.w.copy_(s["w"])
        self.opt.load_state_dict(s["opt"])


def build_balancer(name: str, norm_steps: int, params: dict):
    """The balancer `name`; `norm_steps` (train.norm_steps) fills in when it takes one and none is given."""
    if "norm_steps" in BALANCERS.params(name):
        params = {"norm_steps": norm_steps, **params}
    return BALANCERS.build(name, **params)


@dataclass
class CapacityWeight:
    """Weight of a penalty that holds the penalised quantity near `target` (e.g. the KL in nats).

    Dual ascent on log(weight) by lr * (ema / target - 1): above target the weight grows, below it
    shrinks, within [floor, ceiling]. The beta-VAE capacity target (Burgess et al. 2018), in GECO's
    form (Rezende & Viola 2018).
    """

    target: float
    init: float = 1.0
    lr: float = 0.01
    beta: float = 0.98
    floor: float = 1e-3
    ceiling: float = 100.0
    log_weight: float = field(init=False)
    ema: float | None = field(init=False, default=None)

    def __post_init__(self):
        self.log_weight = math.log(self.init)

    @property
    def weight(self) -> float:
        return math.exp(self.log_weight)

    def update(self, value) -> float:
        x = float(value)
        self.ema = x if self.ema is None else self.beta * self.ema + (1 - self.beta) * x
        step = self.lr * (self.ema / self.target - 1)
        self.log_weight = min(max(self.log_weight + step, math.log(self.floor)), math.log(self.ceiling))
        return self.weight

    def state_dict(self):
        return {"log_weight": self.log_weight, "ema": self.ema}

    def load_state_dict(self, s):
        self.log_weight, self.ema = s["log_weight"], s["ema"]


@dataclass
class DualWeight:
    """Weight of an objective under loss budgets, e.g. budget = {cell_type = 0.28}.

    Dual ascent on log(weight): each step it moves by -lr * v, where v is the largest relative
    excess ema/budget - 1 over the guarded losses (their EMAs, smoothed with `beta`). Within budget
    v < 0 and the weight grows; over budget it shrinks, down to `floor`. GECO-style (Rezende & Viola 2018).
    """

    budget: dict
    init: float = 0.1
    lr: float = 0.01
    beta: float = 0.98
    floor: float = 1e-4
    ceiling: float = 100.0
    log_weight: float = field(init=False)
    ema: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.log_weight = math.log(self.init)

    @property
    def weight(self) -> float:
        return math.exp(self.log_weight)

    def update(self, losses: dict) -> float:
        for k in self.budget:
            x = float(losses[k])
            self.ema[k] = x if k not in self.ema else self.beta * self.ema[k] + (1 - self.beta) * x
        excess = max(self.ema[k] / self.budget[k] - 1 for k in self.budget)
        self.log_weight = min(max(self.log_weight - self.lr * excess, math.log(self.floor)), math.log(self.ceiling))
        return self.weight

    def state_dict(self):
        return {"log_weight": self.log_weight, "ema": dict(self.ema)}

    def load_state_dict(self, s):
        self.log_weight, self.ema = s["log_weight"], dict(s["ema"])
