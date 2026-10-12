"""Adapters that make the three historical methods first-class BrainWM heads.

This module records the integration boundary: scBrain contributes hierarchical
residual annotation, scFiT contributes population Flow Matching, and scUniFlow
contributes ordered sparse-feature completion. They share one encoder and one
input contract; these are not separately staged models.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn

from .annotation import HierarchicalResidualAnnotationHead
from .endpoint import LatentEndpointAdapter, latent_endpoint_loss
from .feature_completion import OrderedFeatureCompletionAdapter


@dataclass
class CapabilityOutput:
    annotation_logits: list[Tensor]
    endpoint: Tensor
    velocity: Tensor
    completed_expression: Tensor


class UnifiedCapabilityHeads(nn.Module):
    """Joint heads; unavailable modalities use identity/zero fallbacks."""
    def __init__(self, latent_dim: int, expression_dim: int,
                 annotation_dims=(3, 24, 137)):
        super().__init__()
        self.annotation = HierarchicalResidualAnnotationHead(latent_dim, annotation_dims)
        self.flow = LatentEndpointAdapter(latent_dim=latent_dim, hidden_dim=max(64, latent_dim // 2),
                                          time_dim=32, parameterization="endpoint")
        side = int(expression_dim ** 0.5) + 1
        self.completion = OrderedFeatureCompletionAdapter(
            expression_dim, token_dim=latent_dim, grid_shape=(side, side))

    def forward(self, state: Tensor, expression: Tensor, *, flow_time: Tensor | None = None,
                source_time: Tensor | None = None, target_time: Tensor | None = None,
                developmental_observed: Tensor | None = None,
                completion_observed: Tensor | None = None) -> CapabilityOutput:
        logits, _ = self.annotation.forward_with_features(state)
        b = state.shape[0]
        if flow_time is None or source_time is None or target_time is None:
            endpoint, velocity = state, torch.zeros_like(state)
        else:
            endpoint, velocity = self.flow.predict_endpoint_and_velocity(
                state, flow_time, source_time, target_time)
            if developmental_observed is not None:
                missing = ~developmental_observed.bool().view(-1, 1)
                endpoint = torch.where(missing, state, endpoint)
                velocity = torch.where(missing, torch.zeros_like(velocity), velocity)
        if completion_observed is None or not completion_observed.bool().any():
            completed = expression
        else:
            completed = self.completion(expression)
        return CapabilityOutput(logits, endpoint, velocity, completed)


__all__ = ["CapabilityOutput", "UnifiedCapabilityHeads"]
