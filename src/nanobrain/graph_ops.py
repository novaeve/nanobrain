"""Fine-grained graph primitives, composed from PyTorch and upstream modules.

Residual connections and normalization are explicit graph nodes. Upstream
whole blocks remain available alongside their independently usable parts.
"""
from torch import nn
import torch

from .baseline_components import reference_module
from .registry import registry


class Linear(nn.Linear):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__(input_dim, hidden_dim, bias=bias)


class LayerNorm(nn.LayerNorm):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__(hidden_dim, eps=eps)


class RMSNorm(nn.RMSNorm):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__(hidden_dim, eps=eps)


class Dropout(nn.Dropout):
    def __init__(self, dropout=0.0):
        super().__init__(dropout)


class ScBrainFFN(nn.Module):
    def __init__(self, hidden_dim, expansion=2.67, dropout=0.):
        super().__init__()
        self.ffn = reference_module("scbrain").SwiGLU(
            hidden_dim, hidden_dim, expansion_factor=expansion, dropout=dropout)

    def forward(self, x):
        return self.ffn(x)


class ScFiTFFN(nn.Module):
    def __init__(self, hidden_dim, expansion=4., dropout=0.):
        super().__init__()
        self.ffn = reference_module("scfit").AdaLNTransformerBlock(
            hidden_dim, 1, round(hidden_dim * expansion), dropout=dropout,
            use_adaln=False).ffn

    def forward(self, x):
        return self.ffn(x)


class ScFiTSelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.):
        super().__init__()
        self.attention = reference_module("scfit").AdaLNTransformerBlock(
            hidden_dim, heads, hidden_dim, dropout=dropout, use_adaln=False).self_attn

    def forward(self, x):
        seq = x[:, None] if x.ndim == 2 else x
        value = self.attention(seq, seq, seq, need_weights=False)[0]
        return value[:, 0] if x.ndim == 2 else value


class ScFiTTime(nn.Module):
    def __init__(self, hidden_dim, dropout=0.):
        super().__init__()
        self.condition = reference_module("scfit").SequenceConditionEncoder(hidden_dim, dropout=dropout)

    def forward(self, time):
        return self.condition(time[:, :1], time[:, 1:2], time[:, 1:2] + time[:, 2:3])


class MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(1)


class FirstPool(nn.Module):
    def forward(self, x):
        return x[:, 0]


class Add(nn.Module):
    def forward(self, *values):
        return torch.stack(values).sum(0)


class Subtract(nn.Module):
    def forward(self, left, right):
        return left - right


class Concat(nn.Module):
    def forward(self, *values):
        return torch.cat(values, -1)


class Stack(nn.Module):
    def forward(self, *values):
        return torch.stack(values, 1)


class FiLM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, state, condition):
        scale, shift = self.modulation(condition).chunk(2, -1)
        if state.ndim == 3:
            scale, shift = scale[:, None], shift[:, None]
        return state * (1 + scale) + shift


for name, factory, source in (
    ("linear", Linear, "torch.nn.Linear"),
    ("layer_norm", LayerNorm, "torch.nn.LayerNorm"),
    ("rms_norm", RMSNorm, "torch.nn.RMSNorm"),
    ("gelu", nn.GELU, "torch.nn.GELU"),
    ("silu", nn.SiLU, "torch.nn.SiLU"),
    ("dropout", Dropout, "torch.nn.Dropout"),
    ("scbrain_ffn", ScBrainFFN, "scBrain SwiGLU (FFN residual and norm are separate nodes)"),
    ("scfit_ffn", ScFiTFFN, "scFiT AdaLNTransformerBlock.ffn"),
    ("scfit_attention", ScFiTSelfAttention, "scFiT AdaLNTransformerBlock.self_attn"),
    ("scfit_time", ScFiTTime, "scFiT SequenceConditionEncoder"),
    ("mean_pool", MeanPool, "torch.mean, token axis only"),
    ("first_pool", FirstPool, "token indexing"),
    ("add", Add, "torch.stack/sum"),
    ("subtract", Subtract, "torch.sub"),
    ("concat", Concat, "torch.cat, feature axis"),
    ("stack", Stack, "torch.stack, token axis"),
    ("film", FiLM, "torch.nn.Linear affine conditioning"),
):
    registry("operation").register(name, factory, source=source)
