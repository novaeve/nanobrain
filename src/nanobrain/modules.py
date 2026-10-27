"""Small composable modules used by the first nanoBrain search space."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .registry import registry


def _as_sequence(value: Tensor) -> tuple[Tensor, bool]:
    if value.ndim == 2:
        return value.unsqueeze(1), True
    if value.ndim == 3:
        return value, False
    raise ValueError(f"expected [batch, width] or [batch, tokens, width], got {tuple(value.shape)}")


class IdentityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, **_: object):
        super().__init__()
        self.projection = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 2, dropout: float = 0.0, **_: object):
        super().__init__()
        depth = max(1, int(depth))
        layers: list[nn.Module] = []
        width = int(input_dim)
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(float(dropout))))
            width = int(hidden_dim)
        layers.append(nn.Linear(width, hidden_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.net(x))


class ResidualMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 2, dropout: float = 0.0, **_: object):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(max(1, int(depth)))
        )

    def forward(self, x: Tensor) -> Tensor:
        value = self.input(x)
        for block in self.blocks:
            value = value + block(value)
        return value


class IdentityAttention(nn.Module):
    def __init__(self, hidden_dim: int, **_: object):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return x


class SDPAAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.0, **_: object):
        super().__init__()
        heads = max(1, min(int(heads), int(hidden_dim)))
        while hidden_dim % heads:
            heads -= 1
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=float(dropout), batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        sequence, squeezed = _as_sequence(x)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        output = self.norm(sequence + attended)
        return output[:, 0] if squeezed else output


class DifferentialLinearAttention(nn.Module):
    """Positive-minus-correction linear attention for a replaceable baseline."""

    def __init__(self, hidden_dim: int, heads: int = 4, correction_init: float = 0.2, **_: object):
        super().__init__()
        heads = max(1, min(int(heads), int(hidden_dim)))
        while hidden_dim % heads:
            heads -= 1
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 6)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.correction = nn.Parameter(torch.tensor(float(correction_init)))

    def _reshape(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _kernel(value: Tensor) -> Tensor:
        return F.elu(value.float()) + 1.0

    def _path(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        query = self._kernel(query)
        key = self._kernel(key)
        value = value.float()
        key_value = torch.einsum("bhnd,bhne->bhde", key, value)
        normalizer = torch.einsum("bhnd,bhd->bhn", query, key.sum(dim=2)).clamp_min(1e-5)
        return torch.einsum("bhnd,bhde,bhn->bhne", query, key_value, normalizer.reciprocal())

    def forward(self, x: Tensor) -> Tensor:
        sequence, squeezed = _as_sequence(x)
        q_pos, k_pos, v_pos, q_neg, k_neg, v_neg = self.qkv(sequence).chunk(6, dim=-1)
        positive = self._path(self._reshape(q_pos), self._reshape(k_pos), self._reshape(v_pos))
        correction = self._path(self._reshape(q_neg), self._reshape(k_neg), self._reshape(v_neg))
        merged = positive - self.correction.to(positive.dtype) * correction
        merged = merged.transpose(1, 2).reshape_as(sequence).to(sequence.dtype)
        output = self.norm(sequence + self.output(merged))
        return output[:, 0] if squeezed else output


class IdentitySpatial(nn.Module):
    def __init__(self, hidden_dim: int, **_: object):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: Tensor, neighbors: Tensor | None = None, neighbor_mask: Tensor | None = None, **_: object) -> Tensor:
        return x


class NeighborMeanSpatial(nn.Module):
    """Sparse neighbor residual; no coordinates/neighbors means exact identity."""

    def __init__(self, hidden_dim: int, **_: object):
        super().__init__()
        self.update = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: Tensor, neighbors: Tensor | None = None, neighbor_mask: Tensor | None = None, **_: object) -> Tensor:
        if neighbors is None or neighbor_mask is None:
            return x
        if x.ndim != 2 or neighbors.ndim != 2 or neighbors.shape[0] != x.shape[0]:
            raise ValueError("NeighborMeanSpatial expects x [cells,width] and neighbors [cells,k]")
        safe = neighbors.long().masked_fill(~neighbor_mask.bool(), 0)
        valid = neighbor_mask.bool() & (safe != torch.arange(len(x), device=x.device)[:, None])
        has = valid.any(dim=1)
        gathered = x[safe]
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (gathered * valid.unsqueeze(-1)).sum(dim=1) / denom
        correction = self.update(mean - x)
        return x + torch.where(has.unsqueeze(-1), correction, torch.zeros_like(correction))


class NoResidual(nn.Module):
    def __init__(self, hidden_dim: int, **_: object):
        super().__init__()

    def forward(self, x: Tensor, **_: object) -> Tensor:
        return x


class NBeatsResidual(nn.Module):
    """Compact backcast/forecast residual correction for cell states."""

    def __init__(self, hidden_dim: int, expansion: float = 2.0, **_: object):
        super().__init__()
        basis = max(4, round(hidden_dim * float(expansion)))
        self.norm = nn.LayerNorm(hidden_dim)
        self.backcast = nn.Sequential(nn.Linear(hidden_dim, basis), nn.GELU(), nn.Linear(basis, hidden_dim))
        self.forecast = nn.Sequential(nn.Linear(hidden_dim, basis), nn.GELU(), nn.Linear(basis, hidden_dim))
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor, **_: object) -> Tensor:
        normalized = self.norm(x)
        backcast = self.backcast(normalized)
        forecast = self.forecast(normalized - backcast)
        return x + torch.sigmoid(self.gate) * forecast


class NoFlow(nn.Module):
    def __init__(self, hidden_dim: int, **_: object):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, state: Tensor, context: Tensor, time: Tensor, **_: object) -> Tensor:
        del context, time
        return torch.zeros_like(state)


class FlowMLP(nn.Module):
    def __init__(self, hidden_dim: int, depth: int = 2, **_: object):
        super().__init__()
        width = max(4, int(hidden_dim))
        layers: list[nn.Module] = []
        layers.extend((nn.Linear(width * 2 + 1, width), nn.SiLU()))
        for _ in range(max(1, int(depth)) - 1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        layers.append(nn.Linear(width, width))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, context: Tensor, time: Tensor, **_: object) -> Tensor:
        if time.ndim == 1:
            time = time[:, None]
        if time.shape[-1] > 1:
            time = time[:, :1]
        return self.net(torch.cat((state, context, time.to(state.dtype)), dim=-1))


class FlowMatchingTransformerModule(nn.Module):
    """Adapter around the production AdaLN-Zero population FMT."""

    def __init__(self, hidden_dim: int, heads: int = 4, depth: int = 2, **_: object):
        super().__init__()
        from brainwm.flow_transformer import FlowMatchingTransformer

        requested = max(2, int(depth))
        if requested % 2:
            requested += 1
        self.transformer = FlowMatchingTransformer(
            int(hidden_dim), int(heads), depth=requested,
            expansion=2.67, attention_mode="auto", sdpa_backend="auto",
        )

    def forward(self, state: Tensor, context: Tensor, time: Tensor, **_: object) -> Tensor:
        if time.ndim == 1:
            time = time[:, None]
        if time.shape != (len(state), 3):
            features = time.new_zeros((len(state), 3))
            features[:, 0] = time[:, 0]
            time = features
        return self.transformer(context, state, time.to(state.dtype))


class IdentityPairing(nn.Module):
    def __init__(self, **_: object):
        super().__init__()

    def forward(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if source.shape != target.shape or source.ndim != 2:
            raise ValueError("population pairing expects equal [cells,width] tensors")
        count = len(source)
        plan = torch.eye(count, device=source.device, dtype=torch.float32) / count
        return target.detach(), plan


class SinkhornPairing(nn.Module):
    def __init__(self, epsilon: float = 0.05, iterations: int = 30, **_: object):
        super().__init__()
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)

    def forward(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        from brainwm.losses import barycentric_transport_target, sinkhorn_transport_plan

        plan = sinkhorn_transport_plan(
            source.detach().float(), target.detach().float(),
            epsilon=self.epsilon, iterations=self.iterations,
        )
        matched = barycentric_transport_target(plan, target.detach().float())
        return matched.to(source.dtype), plan


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, **_: object):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, int(output_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, **_: object):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, int(output_dim)))

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


def _register() -> None:
    enc = registry("encoder")
    enc.register("identity", IdentityEncoder, aliases=("linear",), complexity="O(BD)")
    enc.register("mlp", MLPEncoder, complexity="O(BD^2)")
    enc.register("residual_mlp", ResidualMLPEncoder, complexity="O(LBD^2)")

    attn = registry("attention")
    attn.register("identity", IdentityAttention, complexity="O(BD)")
    attn.register("sdpa", SDPAAttention, complexity="O(BN^2D)", notes="PyTorch MultiheadAttention")
    attn.register("differential_linear", DifferentialLinearAttention, complexity="O(BND^2)", notes="positive-minus-correction kernel attention")

    spatial = registry("spatial")
    spatial.register("identity", IdentitySpatial, complexity="O(BD)", notes="exact self-loop fallback")
    spatial.register("neighbor_mean", NeighborMeanSpatial, complexity="O(BKD)", notes="bounded sparse neighbor residual")

    residual = registry("residual")
    residual.register("none", NoResidual, complexity="O(BD)")
    residual.register("nbeats", NBeatsResidual, complexity="O(BD^2)", source="N-BEATS", notes="backcast/forecast residual correction")

    pairing = registry("pairing")
    pairing.register("identity", IdentityPairing, complexity="O(BD)", notes="index-aligned ablation")
    pairing.register("sinkhorn", SinkhornPairing, complexity="O(B^2D)", source="BrainWM/scFiT", notes="balanced entropic OT barycentric matching")

    flow = registry("flow")
    flow.register("none", NoFlow, complexity="O(BD)", notes="zero-velocity developmental fallback")
    flow.register("mlp", FlowMLP, complexity="O(BD^2)", notes="small conditional velocity field")
    flow.register("fmt", FlowMatchingTransformerModule, complexity="O(BND^2)", source="BrainWM/scBrain/scFiT", notes="AdaLN-Zero U-shaped population Flow Matching Transformer")

    head = registry("head")
    head.register("classification", ClassificationHead, complexity="O(BDC)")
    head.register("regression", RegressionHead, complexity="O(BDO)")


_register()

from .baseline_components import register_components
register_components()

from . import integration_components as _integration_components
