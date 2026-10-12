"""Population-level Flow Matching Transformer with U-shaped skip fusion."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .profiling import profile_range
from .backends import register_flow_backend


class FlowSwiGLU(nn.Module):
    def __init__(self, width: int, expansion: float = 2.67):
        super().__init__()
        hidden = round(width * expansion)
        self.input = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        with profile_range("brainwm.flow_block/ffn_input"):
            gate, value = self.input(x).chunk(2, dim=-1)
        with profile_range("brainwm.flow_block/ffn_activation"):
            hidden = F.silu(gate) * value
        with profile_range("brainwm.flow_block/ffn_output"):
            return self.output(hidden)


class AdaLNZeroFlowBlock(nn.Module):
    """DiT-style AdaLN-Zero block using PyTorch fused SDPA."""
    def __init__(self, width: int, heads: int, expansion: float = 2.67,
                 attention_mode: str = "auto", linear_threshold: int = 256,
                 sdpa_backend: str = "flash"):
        super().__init__()
        if heads < 1 or width % heads:
            raise ValueError("flow width must be divisible by flow heads")
        self.heads = int(heads)
        self.head_dim = width // heads
        if attention_mode not in {"auto", "sdpa", "linear"}:
            raise ValueError("flow attention_mode must be auto, sdpa, or linear")
        if sdpa_backend not in {"auto","flash","efficient","cudnn","math"}:
            raise ValueError(
                "flow sdpa_backend must be auto, flash, efficient, cudnn, or math"
            )
        self.attention_mode=attention_mode;self.linear_threshold=int(linear_threshold)
        self.sdpa_backend=sdpa_backend
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.ffn = FlowSwiGLU(width, expansion)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        with profile_range("brainwm.flow_block/modulation"):
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = \
                self.modulation(condition).chunk(6, dim=-1)
        with profile_range("brainwm.flow_block/attention_norm_modulate"):
            attention_input = self.modulate(
                self.norm_attention(x), shift_a, scale_a
            )
        with profile_range("brainwm.flow_block/qkv_projection"):
            qkv = self.qkv(attention_input)
        batch, tokens, _ = qkv.shape
        with profile_range("brainwm.flow_block/qkv_layout"):
            qkv = qkv.view(batch, tokens, 3, self.heads, self.head_dim)
            query, key, value = qkv.unbind(dim=2)
            query, key, value = (
                item.transpose(1, 2) for item in (query, key, value)
            )
        use_linear=self.attention_mode=="linear" or (
            self.attention_mode=="auto" and tokens>self.linear_threshold)
        if use_linear:
            with profile_range("brainwm.flow_block/linear_attention"):
                query=F.elu(query)+1;key=F.elu(key)+1
                key_value=torch.einsum("bhnd,bhne->bhde",key,value)
                normalizer=torch.einsum(
                    "bhnd,bhd->bhn",query,key.sum(dim=2)
                ).clamp_min(1e-6).reciprocal()
                attended=torch.einsum(
                    "bhnd,bhde,bhn->bhne",query,key_value,normalizer
                )
        else:
            with profile_range("brainwm.flow_block/sdpa_attention"):
                backend_context=nullcontext()
                if query.is_cuda and self.sdpa_backend!="auto":
                    from torch.nn.attention import SDPBackend,sdpa_kernel
                    backend={
                        "flash":SDPBackend.FLASH_ATTENTION,
                        "efficient":SDPBackend.EFFICIENT_ATTENTION,
                        "cudnn":SDPBackend.CUDNN_ATTENTION,
                        "math":SDPBackend.MATH,
                    }[self.sdpa_backend]
                    backend_context=sdpa_kernel(backend)
                with backend_context:
                    attended=F.scaled_dot_product_attention(query,key,value)
        with profile_range("brainwm.flow_block/attention_output"):
            attended = attended.transpose(1, 2).reshape(batch, tokens, -1)
            x = x + gate_a * self.attention_output(attended)
        with profile_range("brainwm.flow_block/ffn_norm_modulate"):
            ffn_input = self.modulate(self.norm_ffn(x), shift_f, scale_f)
        with profile_range("brainwm.flow_block/ffn"):
            update = self.ffn(ffn_input)
        with profile_range("brainwm.flow_block/ffn_residual"):
            return x + gate_f * update


class FlowMatchingTransformer(nn.Module):
    """Permutation-equivariant U-shaped transformer over a cell population."""
    def __init__(self, width: int, heads: int, depth: int = 4,
                 expansion: float = 2.67, attention_mode: str = "auto",
                 sdpa_backend: str = "flash"):
        super().__init__()
        if depth < 2 or depth % 2:
            raise ValueError("flow transformer depth must be a positive even integer")
        half = depth // 2
        self.input = nn.Linear(2 * width, width)
        self.condition = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width),
        )
        self.encoder = nn.ModuleList(
            AdaLNZeroFlowBlock(
                width,heads,expansion,attention_mode,sdpa_backend=sdpa_backend,
            ) for _ in range(half)
        )
        self.middle = AdaLNZeroFlowBlock(
            width,heads,expansion,attention_mode,sdpa_backend=sdpa_backend,
        )
        self.skip_fusion = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(half))
        self.decoder = nn.ModuleList(
            AdaLNZeroFlowBlock(
                width,heads,expansion,attention_mode,sdpa_backend=sdpa_backend,
            ) for _ in range(half)
        )
        self.output_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.output = nn.Linear(width, width)

    def set_sdpa_backend(self,backend: str) -> None:
        if backend not in {"auto","flash","efficient","cudnn","math"}:
            raise ValueError(f"unknown flow SDPA backend: {backend}")
        for block in (*self.encoder,self.middle,*self.decoder):
            block.sdpa_backend=backend

    def forward(self, context: Tensor, state: Tensor, time_features: Tensor) -> Tensor:
        if context.shape != state.shape or context.ndim != 2:
            raise ValueError("flow context and state must have shape [cells, width]")
        if time_features.shape != (len(context), 3):
            raise ValueError("flow time features must have shape [cells, 3]")
        # A population is one sequence. No positional encoding makes the map
        # equivariant to arbitrary cell ordering.
        with profile_range("brainwm.flow/input_projection"):
            x = self.input(torch.cat((context, state), dim=-1)).unsqueeze(0)
        with profile_range("brainwm.flow/time_condition"):
            condition = self.condition(time_features).unsqueeze(0)
        skips = []
        for index, block in enumerate(self.encoder):
            with profile_range(f"brainwm.flow/encoder_{index:02d}"):
                x = block(x, condition)
            skips.append(x)
        with profile_range("brainwm.flow/middle"):
            x = self.middle(x, condition)
        for index, (fusion, block, skip) in enumerate(
            zip(self.skip_fusion, self.decoder, reversed(skips))
        ):
            with profile_range(f"brainwm.flow/skip_fusion_{index:02d}"):
                x = fusion(torch.cat((x, skip), dim=-1))
            with profile_range(f"brainwm.flow/decoder_{index:02d}"):
                x = block(x, condition)
        with profile_range("brainwm.flow/output_projection"):
            return self.output(self.output_norm(x)).squeeze(0)


# Keep the public flow interface replaceable.  The registry entry is installed
# in this module, after the class exists, so optional DiT/FMT implementations
# can register beside it without changing the foundation model constructor.
register_flow_backend(
    "fmt",
    FlowMatchingTransformer,
    aliases=("flow_matching_transformer", "dit_adaln_zero"),
    capabilities={
        "permutation_equivariant": True,
        "flow_matching": True,
        "adaln_zero": True,
        "u_skip": True,
        "sdpa_backend_selection": True,
        "bf16": True,
        "ddp": True,
        "state_dict_compatible": True,
    },
)
