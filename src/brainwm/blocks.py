"""Reusable neural blocks and segment operations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        modules: list[nn.Module] = []
        width = in_dim
        for _ in range(layers - 1):
            modules.extend([nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            width = hidden_dim
        modules.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = MLP(dim, hidden_dim, dim, layers=2, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim: int, scale_days: float = 365.2425):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim
        self.scale_days = scale_days

    def forward(self, time_days: Tensor) -> Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time_days.device, dtype=time_days.dtype)
            / max(half - 1, 1)
        )
        angle = (time_days / self.scale_days)[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def segment_sum(values: Tensor, index: Tensor, n_segments: int) -> Tensor:
    shape = (n_segments,) + values.shape[1:]
    result = values.new_zeros(shape)
    result.index_add_(0, index, values)
    return result


def segment_softmax(scores: Tensor, index: Tensor, n_segments: int) -> Tensor:
    if scores.ndim != 1:
        raise ValueError("segment_softmax currently expects one score per item")
    maxima = scores.new_full((n_segments,), -torch.inf)
    maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    stable = torch.exp(scores - maxima[index])
    denominator = segment_sum(stable, index, n_segments).clamp_min(1.0e-12)
    return stable / denominator[index]


def row_index_from_ptr(row_ptr: Tensor) -> Tensor:
    counts = row_ptr[1:] - row_ptr[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=row_ptr.device, dtype=torch.long), counts
    )
