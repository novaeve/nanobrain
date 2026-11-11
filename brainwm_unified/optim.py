"""Muon with same-shape matrices orthogonalised together (2-D weights and stacked 3-D expert weights).

Numerically the update of torch.optim.Muon (PyTorch 2.9): Nesterov momentum,
quintic Newton-Schulz in bf16 with the same coefficients, the same learning-rate
adjustment and decoupled weight decay. The difference is execution: momentum uses
foreach kernels, and every group of equal-shape matrices (e.g. the 32 experts of a
layer) runs Newton-Schulz as one batched matmul instead of one small matmul each.
"""

import math
from collections import defaultdict

import torch


def batched_newton_schulz(g: torch.Tensor, coefficients, steps: int, eps: float) -> torch.Tensor:
    """Orthogonalise each matrix of a (S, m, n) stack; mirrors torch.optim._muon."""
    a, b, c = coefficients
    x = g.bfloat16()
    tall = g.shape[-2] > g.shape[-1]
    if tall:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.mT
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, gram_update, x, beta=a)
    return x.mT if tall else x


def adjusted_lr(lr: float, adjust_lr_fn: str | None, shape) -> float:
    a, b = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1, a / b))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(a, b))
    return lr


class GroupedMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn=None,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            for p in params:
                if p.ndim not in (2, 3):
                    raise ValueError("GroupedMuon handles 2-D matrices and 3-D stacks of matrices")
            grads = [p.grad for p in params]
            bufs = []
            for p in params:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                bufs.append(state["momentum_buffer"])
            m = group["momentum"]
            torch._foreach_lerp_(bufs, grads, 1 - m)
            updates = torch._foreach_lerp(grads, bufs, m) if group["nesterov"] else bufs

            # Every parameter becomes a stack of matrices: (1, m, n) for 2-D, (E, m, n) for stacked
            # experts, (E * s, m / s, n) when `muon_split` = s packs s matrices per expert.
            def as_matrices(t, p):
                if p.ndim == 2:
                    return t.unsqueeze(0)
                split = getattr(p, "muon_split", 1)
                return t.reshape(p.shape[0] * split, p.shape[1] // split, p.shape[2])

            by_shape = defaultdict(list)
            for i, p in enumerate(params):
                by_shape[tuple(as_matrices(p, p).shape[1:])].append(i)
            torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            for shape, idx in by_shape.items():
                stacks = [as_matrices(updates[i], params[i]) for i in idx]
                ortho = batched_newton_schulz(
                    torch.cat(stacks), group["ns_coefficients"], group["ns_steps"], group["eps"]
                )
                step = adjusted_lr(group["lr"], group["adjust_lr_fn"], shape)
                pieces = ortho.split([len(st) for st in stacks])
                torch._foreach_add_(
                    [params[i] for i in idx], [o.reshape(params[i].shape) for o, i in zip(pieces, idx)], alpha=-step
                )
        return loss
