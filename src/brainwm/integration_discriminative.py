"""Harmony-anchored discriminative residual adapter for Integration source CV."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(values: Tensor, strength: float = 1.0) -> Tensor:
    return _GradientReverse.apply(values, float(strength))


class HarmonyResidualAdapter(nn.Module):
    """Add a bounded, learned dual-view residual to a standardized Harmony trunk."""

    def __init__(
        self,
        *,
        raw_mean: Tensor,
        raw_std: Tensor,
        aux_mean: Tensor,
        aux_std: Tensor,
        trunk_mean: Tensor,
        trunk_std: Tensor,
        labels: int,
        domains: int,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        for name, value in (
            ("raw_mean", raw_mean),
            ("raw_std", raw_std),
            ("aux_mean", aux_mean),
            ("aux_std", aux_std),
            ("trunk_mean", trunk_mean),
            ("trunk_std", trunk_std),
        ):
            self.register_buffer(name, value.detach().float().clone())
        self.raw_encoder = nn.Sequential(
            nn.Linear(256, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.aux_encoder = nn.Sequential(
            nn.Linear(256, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.residual = nn.Linear(hidden, 256)
        self.label_head = nn.Linear(256, int(labels))
        self.domain_head = nn.Sequential(
            nn.Linear(256, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, int(domains))
        )
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def _standardize(values: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return (values - mean) / std.clamp_min(1.0e-6)

    def forward(
        self,
        raw: Tensor,
        auxiliary: Tensor,
        trunk: Tensor,
        *,
        beta: float,
        domain_strength: float = 1.0,
    ) -> dict[str, Tensor]:
        raw_values = self._standardize(raw, self.raw_mean, self.raw_std)
        aux_values = self._standardize(auxiliary, self.aux_mean, self.aux_std)
        trunk_features = self._standardize(trunk, self.trunk_mean, self.trunk_std)
        raw_view = F.normalize(self.raw_encoder(raw_values), dim=1)
        aux_view = F.normalize(self.aux_encoder(aux_values), dim=1)
        fused = self.fusion(torch.cat([raw_view, aux_view], dim=1))
        delta_unit = torch.tanh(self.residual(fused))
        delta = self.trunk_std * delta_unit
        embedding = trunk + float(beta) * delta
        training_embedding = trunk_features + float(beta) * delta_unit
        return {
            "embedding": embedding,
            "training_embedding": training_embedding,
            "anchor": trunk,
            "delta": delta,
            "delta_unit": delta_unit,
            "raw_view": raw_view,
            "aux_view": aux_view,
            "label_logits": self.label_head(training_embedding),
            "domain_logits": self.domain_head(
                gradient_reverse(training_embedding, float(domain_strength))
            ),
        }


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    *,
    temperature: float,
) -> Tensor:
    valid = labels.ge(0)
    values = F.normalize(embeddings[valid], dim=1)
    targets = labels[valid]
    if len(values) < 2:
        return embeddings.sum() * 0.0
    logits = values @ values.T / float(temperature)
    eye = torch.eye(len(values), dtype=torch.bool, device=values.device)
    positives = targets[:, None].eq(targets[None, :]) & ~eye
    eligible = positives.any(dim=1)
    if not bool(eligible.any()):
        return embeddings.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    mean_positive = (log_prob * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -mean_positive[eligible].mean()


def same_cell_view_loss(raw_view: Tensor, aux_view: Tensor, *, temperature: float) -> Tensor:
    logits = raw_view @ aux_view.T / float(temperature)
    targets = torch.arange(len(raw_view), device=raw_view.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    )


__all__ = [
    "HarmonyResidualAdapter",
    "gradient_reverse",
    "same_cell_view_loss",
    "supervised_contrastive_loss",
]
