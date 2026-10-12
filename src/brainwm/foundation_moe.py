"""Linear-complexity, modality-extensible BrainWM foundation model.

The public input contract never changes when a new modality is added: every
field carries values, an observed flag, and an optional training mask. Missing
fields use a modality-specific learnable null token; zero is a real value.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import sys
import warnings

import torch
import torch.distributed as dist
from torch import Tensor, nn
from .sparse_tokens import SparseTokenBatch, add_csr_rows, concat_detached_suffix
from .spatial_attention import SpatialCellAttention
from .flow_transformer import FlowMatchingTransformer
from .losses import SinkhornRuntime, sinkhorn_transport_plan
from .profiling import profile_range
from .backends import (
    backend_spec,
    compile_callable,
    create_backend,
    register_attention_backend,
    register_moe_backend,
)
from .router_acceleration import CompiledTopKGate, run_router_projection_logits


_FLA_LAYER_NORM_LINEAR = None
_FLA_LAYER_NORM = None
_RESIDUAL_NORM_EXECUTIONS = {"torch", "torch_bf16", "fla", "auto"}
DEFAULT_FLA_RESIDUAL_NORM_MIN_TOKENS = 131072
_FEATURE_MAP_EXECUTIONS = {"torch", "compile", "bucket", "auto"}
_DIFFERENTIAL_MERGE_EXECUTIONS = {"torch", "compile", "auto"}
_ROUTER_EXECUTIONS = {"eager", "compile", "auto"}
_TORCH_COMPILE_MODES = {
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs",
}
DEFAULT_COMPILED_FEATURE_MAP_MIN_TOKENS = 131072
DEFAULT_FEATURE_MAP_BUCKET_SIZE = 65536
DEFAULT_FEATURE_MAP_MAX_PADDING_RATIO = 1.10
DEFAULT_COMPILED_DIFFERENTIAL_MERGE_MIN_TOKENS = 131072
_COMPILED_FEATURE_MAPS: dict[tuple[object, ...], object] = {}
_COMPILED_FEATURE_MAP_CACHE_LIMIT = 16
_COMPILED_DIFFERENTIAL_MERGES: dict[tuple[object, ...], object] = {}
_COMPILED_DIFFERENTIAL_MERGE_CACHE_LIMIT = 16


def feature_map_bucket_for_tokens(token_count: int, bucket_size: int) -> int:
    """Return the fixed token bucket used by the static feature-map path."""
    token_count = int(token_count)
    bucket_size = int(bucket_size)
    if token_count < 1 or bucket_size < 1:
        raise ValueError("token_count and bucket_size must be positive")
    return ((token_count + bucket_size - 1) // bucket_size) * bucket_size


def feature_map_bucket_padding_ratio(token_count: int, bucket_size: int) -> float:
    """Return allocated/static tokens divided by real tokens."""
    bucket = feature_map_bucket_for_tokens(token_count, bucket_size)
    return float(bucket) / float(int(token_count))


def _cached_feature_map_compile(
    key: tuple[object, ...], function, *, mode: str, dynamic: bool,
):
    """Compile one tensor-only callable and retain a bounded process cache."""
    compiled = _COMPILED_FEATURE_MAPS.get(key)
    if compiled is None:
        if getattr(torch, "compile", None) is None:
            raise RuntimeError("compiled feature map requires PyTorch 2.x")
        compiled = compile_callable(
            "torch_compile", function, mode=mode, dynamic=bool(dynamic),
            fullgraph=True,
        )
        _COMPILED_FEATURE_MAPS[key] = compiled
        while len(_COMPILED_FEATURE_MAPS) > _COMPILED_FEATURE_MAP_CACHE_LIMIT:
            _COMPILED_FEATURE_MAPS.pop(next(iter(_COMPILED_FEATURE_MAPS)))
    return compiled


def _flash_bla_feature_map_eager(
    queries: Tensor, keys: Tensor, token_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply BrainWM's positive feature map before the varlen kernel.

    This function is deliberately a small, tensor-only boundary.  It is used
    both as the reference implementation and as the input to PyTorch
    Inductor, so the compiled path cannot change the attention equation.
    Flash-BLA performs the sequence-length normalization internally; scaling
    queries by each row length preserves BrainWM's original normalization.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        queries = queries.exp().clamp(max=1e4)
        # The packed model state pre-casts this immutable sidecar once per
        # input stream.  Keep the fallback cast for public/direct callers that
        # still pass integer row lengths (and for a backend that changes the
        # projection dtype between preparation and execution).
        scale = token_lengths if token_lengths.dtype == queries.dtype else token_lengths.to(queries.dtype)
        scale = scale.view(-1, 1, 1)
        queries = queries * scale
        keys = keys.exp().clamp(max=1e4)
    return queries, keys


def _compiled_flash_bla_feature_map(
    queries: Tensor, keys: Tensor, token_lengths: Tensor,
    *, mode: str = "max-autotune-no-cudagraphs", dynamic: bool = True,
) -> tuple[Tensor, Tensor]:
    """Run the feature map through one process-wide Inductor callable.

    Attention blocks share this callable so a model with many blocks pays one
    graph-construction cost per process.  The callable is intentionally kept
    outside every ``nn.Module`` and therefore never enters a checkpoint or an
    optimizer state.  The cache key only contains compiler policy; Inductor's
    own device/dtype guards handle the tensor specialization.
    """
    key = (
        "dynamic", mode, bool(dynamic), str(queries.device), str(queries.dtype),
        tuple(queries.shape[1:]), str(token_lengths.dtype),
    )
    compiled = _cached_feature_map_compile(
        key, _flash_bla_feature_map_eager, mode=mode, dynamic=dynamic,
    )
    return compiled(queries, keys, token_lengths)


def _compiled_flash_bla_feature_map_bucket(
    queries: Tensor, keys: Tensor, token_lengths: Tensor, *,
    bucket_size: int, mode: str, max_padding_ratio: float,
) -> tuple[Tensor, Tensor]:
    """Run a static-shape feature map on a bounded token bucket.

    The feature map is pointwise. Padding rows therefore cannot affect any
    real row and are trimmed before the varlen attention kernel. Autograd sees
    ordinary ``cat``/slice operations, so this path supports both forward and
    backward without a custom kernel. If a batch would exceed the configured
    padding ratio, the caller should keep the eager path instead.
    """
    token_count = int(queries.shape[0])
    bucket = feature_map_bucket_for_tokens(token_count, bucket_size)
    ratio = float(bucket) / float(token_count)
    if ratio > float(max_padding_ratio):
        raise ValueError(
            f"feature-map bucket ratio {ratio:.4f} exceeds limit "
            f"{float(max_padding_ratio):.4f}"
        )
    key = (
        "bucket", mode, bucket, str(queries.device), str(queries.dtype),
        tuple(queries.shape[1:]), str(token_lengths.dtype),
    )
    compiled = _cached_feature_map_compile(
        key, _flash_bla_feature_map_eager, mode=mode, dynamic=False,
    )
    if bucket == token_count:
        return compiled(queries, keys, token_lengths)
    padding = bucket - token_count
    # F.pad keeps the source in the autograd graph while issuing one compact
    # copy per input. Padded values are discarded before Flash-BLA sees them.
    pad_shape = (0, 0, 0, 0, 0, padding)
    padded_queries = torch.nn.functional.pad(queries, pad_shape)
    padded_keys = torch.nn.functional.pad(keys, pad_shape)
    padded_lengths = torch.nn.functional.pad(
        token_lengths, (0, padding), value=1,
    )
    mapped_queries, mapped_keys = compiled(
        padded_queries, padded_keys, padded_lengths,
    )
    return mapped_queries.narrow(0, 0, token_count), mapped_keys.narrow(
        0, 0, token_count,
    )


def _differential_merge_eager(
    positive: Tensor,
    correction: Tensor,
    lambda_a: Tensor,
    lambda_b: Tensor,
    lambda_c: Tensor,
    lambda_d: Tensor,
    omega: float,
) -> Tensor:
    """Reference positive-minus-correction attention merge."""
    lam = omega + torch.exp((lambda_a * lambda_b).sum()) \
        - torch.exp((lambda_c * lambda_d).sum())
    return positive - lam * correction


def _compiled_differential_merge(
    positive: Tensor,
    correction: Tensor,
    lambda_a: Tensor,
    lambda_b: Tensor,
    lambda_c: Tensor,
    lambda_d: Tensor,
    omega: float,
    *,
    mode: str = "max-autotune-no-cudagraphs",
    dynamic: bool = True,
) -> Tensor:
    """Compile the tensor-only differential merge through TorchInductor.

    The callable is shared by every compatible block and stays outside the
    module/state dict. Dynamic token lengths let one graph cover the variable
    CSR batches used by full-corpus training.
    """
    key = (
        mode,
        bool(dynamic),
        str(positive.device),
        str(positive.dtype),
        positive.ndim,
        tuple(positive.shape[1:]),
        str(lambda_a.dtype),
        tuple(lambda_a.shape),
        float(omega),
    )
    compiled = _COMPILED_DIFFERENTIAL_MERGES.get(key)
    if compiled is None:
        compiled = compile_callable(
            "torch_compile",
            _differential_merge_eager,
            mode=mode,
            dynamic=bool(dynamic),
            fullgraph=True,
        )
        _COMPILED_DIFFERENTIAL_MERGES[key] = compiled
        while (
            len(_COMPILED_DIFFERENTIAL_MERGES)
            > _COMPILED_DIFFERENTIAL_MERGE_CACHE_LIMIT
        ):
            _COMPILED_DIFFERENTIAL_MERGES.pop(
                next(iter(_COMPILED_DIFFERENTIAL_MERGES))
            )
    return compiled(
        positive, correction, lambda_a, lambda_b, lambda_c, lambda_d, omega,
    )


def _load_fla_layer_norm_linear():
    """Load the reviewed FLA fused op without making it a hard dependency.

    The benchmark keeps the upstream review checkout outside the tracked model
    package.  A normal environment may install it as ``fla``; during local
    development we also accept that checkout directly.  No FLA code is copied
    into BrainWM and the import is performed only when the opt-in backend is
    selected.
    """
    global _FLA_LAYER_NORM_LINEAR
    if _FLA_LAYER_NORM_LINEAR is not None:
        return _FLA_LAYER_NORM_LINEAR
    try:
        from fla.modules.layernorm import layer_norm_linear
    except ModuleNotFoundError as exc:
        if exc.name is None or not (exc.name == "fla" or exc.name.startswith("fla.")):
            raise
        review_root = Path(__file__).resolve().parents[2] / \
            "runtime/upstream-review/flash-linear-attention"
        if review_root.is_dir():
            sys.path.insert(0, str(review_root))
            from fla.modules.layernorm import layer_norm_linear
        else:
            # Keep the opt-in execution policy usable in minimal environments.
            # This is the exact PyTorch composition fused by FLA, so it keeps
            # model parameters, autograd, and checkpoint names unchanged.
            def layer_norm_linear(
                x, norm_weight, norm_bias, linear_weight, linear_bias, *, eps,
            ):
                normalized = nn.functional.layer_norm(
                    x, (x.shape[-1],), norm_weight, norm_bias, eps,
                )
                return nn.functional.linear(
                    normalized, linear_weight, linear_bias,
                )

            warnings.warn(
                "flash-linear-attention is unavailable; fla_layernorm_qkv "
                "is using the equivalent PyTorch layer_norm + linear fallback",
                RuntimeWarning,
                stacklevel=2,
            )
    _FLA_LAYER_NORM_LINEAR = layer_norm_linear
    return layer_norm_linear


def _load_fla_layer_norm():
    """Load FLA's fused residual LayerNorm implementation on demand.

    The operator is kept optional for CPU and minimal installations.  The
    review checkout is resolved in exactly the same way as the fused
    LayerNorm+linear path, so the model package does not copy or fork the
    upstream kernel.
    """
    global _FLA_LAYER_NORM
    if _FLA_LAYER_NORM is not None:
        return _FLA_LAYER_NORM
    try:
        from fla.modules.layernorm import layer_norm
    except ModuleNotFoundError as exc:
        if exc.name is None or not (exc.name == "fla" or exc.name.startswith("fla.")):
            raise
        review_root = Path(__file__).resolve().parents[2] / \
            "runtime/upstream-review/flash-linear-attention"
        if not review_root.is_dir():
            raise RuntimeError(
                "fla residual normalization requires flash-linear-attention; "
                "install the pinned upstream package or provide runtime/"
                "upstream-review/flash-linear-attention"
            ) from exc
        sys.path.insert(0, str(review_root))
        from fla.modules.layernorm import layer_norm
    _FLA_LAYER_NORM = layer_norm
    return layer_norm


@dataclass
class BrainWMInput:
    expression: Tensor
    expression_observed: Tensor
    modality: Mapping[str, Tensor] | None = None
    modality_observed: Mapping[str, Tensor] | None = None
    modality_masked: Mapping[str, Tensor] | None = None
    spatial_relation: Tensor | None = None
    spatial_observed: Tensor | None = None
    source_state: Tensor | None = None
    target_state: Tensor | None = None
    source_time: Tensor | None = None
    target_time: Tensor | None = None
    developmental_observed: Tensor | None = None
    flow_tau: Tensor | None = None
    neighbor_indices: Tensor | None = None
    neighbor_mask: Tensor | None = None


@dataclass(frozen=True)
class PackedAttentionLayout:
    """Reusable length-bucketed views over one CSR token batch."""

    groups: tuple[tuple[Tensor, Tensor], ...]
    restore_order: Tensor
    allocated_tokens: int

    def pack(self, value: Tensor) -> tuple[Tensor, ...]:
        return tuple(value[indices] for indices,_ in self.groups)

    def unpack(self, values: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        flattened=torch.cat([value[token_mask] for value,(_,token_mask)
                             in zip(values,self.groups)])
        return flattened[self.restore_order]


@dataclass
class _PackedSparseState:
    """Prepared CSR input and immutable attention-layout metadata."""

    batch: SparseTokenBatch
    x: Tensor
    row_ptr: Tensor
    lengths: Tensor
    token_lengths: Tensor
    row_ids: Tensor
    layout: PackedAttentionLayout | None
    cu_seqlens: Tensor | None
    max_seqlen: int | None
    attention_backend: str
    minimum_sequence_length: int
    maximum_sequence_length: int
    maximum_sequence_bucket: int


class ModalityAdapter(nn.Module):
    def __init__(self, d_model: int, value_dim: int = 1, categorical: bool = False):
        super().__init__()
        self.value = nn.Linear(value_dim, d_model)
        self.null = nn.Parameter(torch.randn(d_model) / d_model**0.5)
        self.mask = nn.Parameter(torch.randn(d_model) / d_model**0.5)
        self.category = nn.Embedding(value_dim, d_model) if categorical else None
        if categorical:
            # Retain legacy state_dict keys while excluding the unreachable
            # numeric branch from autograd, AdamW work, and DDP reduction.
            self.value.requires_grad_(False)

    def forward(self, value: Tensor | None, observed: Tensor | None,
                masked: Tensor | None = None) -> Tensor:
        if value is None:
            if observed is None:
                raise ValueError("observed mask is required for a missing modality")
            shape = (*observed.shape, self.null.numel())
            return self.null.expand(shape)
        if self.category is not None:
            ids=value.squeeze(-1).long() if value.ndim>1 else value.long()
            safe=ids.clamp(0,self.category.num_embeddings-1)
            out=self.category(safe)
            if observed is not None:out=torch.where(observed.bool().unsqueeze(-1),out,self.null)
            if masked is not None:
                effective=masked.bool() if observed is None else masked.bool()&observed.bool()
                out=torch.where(effective.unsqueeze(-1),self.mask,out)
            return out
        clean=value.float()
        # Packed expression rows contain measured values by construction.  In
        # that common path both masks are ``None``; avoid allocating a
        # token-sized boolean tensor and two elementwise kernels.
        if observed is None and masked is None:
            return self.value(clean)
        visible=torch.ones(value.shape[:-1],dtype=torch.bool,device=value.device)
        if observed is not None:
            visible=visible & observed.bool()
        if masked is not None:
            visible=visible & ~masked.bool()
        clean=torch.where(visible.unsqueeze(-1),clean,torch.zeros_like(clean))
        out = self.value(clean)
        if observed is not None:
            out = torch.where(observed.bool().unsqueeze(-1), out, self.null)
        if masked is not None:
            effective=masked.bool() if observed is None else masked.bool() & observed.bool()
            out = torch.where(effective.unsqueeze(-1), self.mask, out)
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expansion: float = 2.67):
        super().__init__()
        if expansion <= 0:
            raise ValueError('FFN expansion must be positive')
        self.hidden_dim = round(d_model * expansion)
        self.in_proj = nn.Linear(d_model, 2 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        with profile_range("brainwm.expert_swiglu/input_projection"):
            gate, value = self.in_proj(x).chunk(2, dim=-1)
        with profile_range("brainwm.expert_swiglu/activation"):
            hidden = nn.functional.silu(gate) * value
        with profile_range("brainwm.expert_swiglu/output_projection"):
            return self.out_proj(hidden)


def _fused_nbeats_pair(
    backcast: nn.Linear, forecast: nn.Linear, features: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project both N-BEATS heads with one GEMM and split the result.

    The two heads have independent parameters and remain separate in the
    state dict.  Concatenating their rows only changes the execution schedule:
    one ``[*, d] x [2d, d]`` matmul replaces two launches over the same input.
    Autograd still routes each half to its original Linear parameters.
    """
    weight = torch.cat((backcast.weight, forecast.weight), dim=0)
    bias = torch.cat((backcast.bias, forecast.bias), dim=0)
    projected = nn.functional.linear(features, weight, bias)
    width = backcast.out_features
    return projected[..., :width], projected[..., width:]


class BatchedSwiGLUExperts(nn.Module):
    """Execute DeepSpeed's fixed-capacity local experts with batched GEMMs.

    The legacy DeepSpeed router presents local expert input as
    ``[ep_size, experts, capacity, d_model]``. BrainWM uses ``ep_size=1``, so
    the expert dimension is already a regular strided batch and PyTorch's
    optimized ``bmm`` can execute all experts in two calls. Keeping the
    original ModuleList preserves every parameter key, shape, and ordering for
    old model and AdamW checkpoints.
    """

    def __init__(self, legacy_experts: nn.Module, execution: str = "batched"):
        super().__init__()
        if execution not in {"batched", "loop"}:
            raise ValueError(f"unknown expert execution backend: {execution}")
        self.deepspeed_experts = legacy_experts.deepspeed_experts
        self.num_local_experts = int(legacy_experts.num_local_experts)
        self.ep_size = int(getattr(legacy_experts, "ep_size", 1))
        if self.ep_size != 1:
            raise ValueError(
                "batched/grouped expert execution currently requires ep_size=1; "
                f"got {self.ep_size}"
            )
        self.execution = execution
        if len(self.deepspeed_experts) != self.num_local_experts:
            raise ValueError("DeepSpeed local expert count is inconsistent")
        if not all(isinstance(expert, SwiGLU) for expert in self.deepspeed_experts):
            raise TypeError("batched execution requires BrainWM SwiGLU experts")
        hidden_dims = {expert.hidden_dim for expert in self.deepspeed_experts}
        if len(hidden_dims) != 1:
            raise ValueError("all batched experts must use the same hidden width")
        self._cached_weights: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        # A joint training step encodes main/source/target populations with
        # the same parameters. Reuse dtype casts across those branches; the
        # parent model clears this cache around each optimizer step.
        self._cached_cast_weights: dict[tuple[torch.dtype, torch.device], tuple[Tensor, Tensor, Tensor, Tensor]] = {}

    def prepare_weights(self) -> None:
        if self.execution not in {"batched", "grouped"} or self._cached_weights is not None:
            return
        self._cached_weights = self._pack_weights()
        self._cached_cast_weights.clear()

    def clear_prepared_weights(self) -> None:
        self._cached_weights = None
        self._cached_cast_weights.clear()

    def _pack_weights(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with profile_range("brainwm.experts/weight_pack"):
            return (
                torch.stack(
                    [expert.in_proj.weight for expert in self.deepspeed_experts]
                ),
                torch.stack(
                    [expert.in_proj.bias for expert in self.deepspeed_experts]
                ),
                torch.stack(
                    [expert.out_proj.weight for expert in self.deepspeed_experts]
                ),
                torch.stack(
                    [expert.out_proj.bias for expert in self.deepspeed_experts]
                ),
            )

    def weights_for(self, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return stacked expert weights in the input compute dtype."""
        packed = self._cached_weights
        if packed is None:
            packed = self._pack_weights()
            # Without an explicit preparation scope the caller may update or
            # move parameters immediately after this call; do not retain a
            # cast tensor that could then reference stale weights.
            return tuple(value.to(dtype=dtype) for value in packed)
        key = (dtype, packed[0].device)
        cached = self._cached_cast_weights.get(key)
        if cached is None:
            cached = tuple(value.to(dtype=dtype) for value in packed)
            self._cached_cast_weights[key] = cached
        return cached

    def forward(self, inputs: Tensor) -> Tensor:
        if self.execution == "loop":
            chunks = inputs.chunk(self.num_local_experts, dim=1)
            return torch.cat(
                [expert(chunk) for chunk, expert in zip(chunks, self.deepspeed_experts)],
                dim=1,
            )
        if inputs.ndim != 4 or inputs.shape[0] != 1:
            raise ValueError(
                "batched expert execution requires ep_size=1 input "
                "[1, experts, capacity, d_model]"
            )
        if inputs.shape[1] != self.num_local_experts:
            raise ValueError("expert input count does not match the local experts")

        expert_input = inputs.squeeze(0)
        # Autocast otherwise converts the same expert weights on every stream
        # invocation.  Reuse the prepared BF16 tensors and run the two GEMMs
        # with autocast disabled; the original FP32 parameters remain the
        # gradient leaves through ``Tensor.to``.
        device_type = expert_input.device.type
        autocast_enabled = (
            device_type in {"cuda", "cpu"}
            and torch.is_autocast_enabled(device_type)
        )
        compute_dtype = (
            torch.get_autocast_dtype(device_type)
            if autocast_enabled else expert_input.dtype
        )
        input_weight, input_bias, output_weight, output_bias = self.weights_for(
            compute_dtype
        )
        expert_input = expert_input.to(dtype=compute_dtype)
        matmul_context = torch.autocast(
            device_type=device_type, enabled=False
        ) if device_type in {"cuda", "cpu"} else nullcontext()
        with matmul_context:
            with profile_range("brainwm.experts/batched_input_projection"):
                # ``baddbmm`` folds the per-expert bias into the GEMM
                # epilogue.  It is a standard PyTorch primitive and keeps
                # the legacy parameter layout/checkpoint contract intact.
                projected = torch.baddbmm(
                    input_bias.unsqueeze(1),
                    expert_input,
                    input_weight.transpose(1, 2),
                )
                gate, value = projected.chunk(2, dim=-1)
            with profile_range("brainwm.experts/batched_activation"):
                hidden = nn.functional.silu(gate) * value
            with profile_range("brainwm.experts/batched_output_projection"):
                output = torch.baddbmm(
                    output_bias.unsqueeze(1),
                    hidden,
                    output_weight.transpose(1, 2),
                )
        return output.unsqueeze(0)


class MAGEAttentionExpert(nn.Module):
    """MAGE-style differential linear attention expert (no explicit graph)."""
    def __init__(self, d_model: int, d_hidden: int, ffn_expansion: float = 2.67):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.q1, self.k1, self.v1 = (nn.Linear(d_model, d_hidden) for _ in range(3))
        self.q2, self.k2, self.v2 = (nn.Linear(d_model, d_hidden) for _ in range(3))
        self.lambda_a = nn.Parameter(torch.zeros(d_hidden))
        self.lambda_b = nn.Parameter(torch.zeros(d_hidden))
        self.lambda_c = nn.Parameter(torch.zeros(d_hidden))
        self.lambda_d = nn.Parameter(torch.zeros(d_hidden))
        self.omega = 0.5
        self.ff = SwiGLU(d_model, ffn_expansion)
        self.out = nn.Linear(d_hidden, d_model)

    @staticmethod
    def _kernel(q: Tensor, k: Tensor, v: Tensor, token_mask: Tensor | None = None) -> Tensor:
        q, k = q.exp().clamp(max=1e4), k.exp().clamp(max=1e4)
        if token_mask is not None:
            visible = (token_mask[:, None, :, None] if q.ndim == 4
                       else token_mask.unsqueeze(-1))
            k = k.masked_fill(~visible, 0)
            v = v.masked_fill(~visible, 0)
        if q.ndim == 4:
            kv = torch.einsum("bhld,bhle->bhde", k, v)
            z = 1.0 / (torch.einsum("bhld,bhd->bhl", q, k.sum(2)) + 1e-6)
            return torch.einsum("bhld,bhde,bhl->bhle", q, kv, z)
        kv = torch.einsum("bld,ble->bde", k, v)
        z = 1.0 / (torch.einsum("bld,bd->bl", q, k.sum(1)) + 1e-6)
        return torch.einsum("bld,bde,bl->ble", q, kv, z)

    def forward(self, x: Tensor, token_mask: Tensor | None = None) -> Tensor:
        with profile_range("brainwm.mage_attention/norm"):
            h = self.norm(x)
        with profile_range("brainwm.mage_attention/positive_projections"):
            positive_q, positive_k, positive_v = self.q1(h), self.k1(h), self.v1(h)
        with profile_range("brainwm.mage_attention/positive_kernel"):
            positive = self._kernel(
                positive_q, positive_k, positive_v, token_mask
            )
        with profile_range("brainwm.mage_attention/correction_projections"):
            correction_q, correction_k, correction_v = self.q2(h), self.k2(h), self.v2(h)
        with profile_range("brainwm.mage_attention/correction_kernel"):
            correction = self._kernel(
                correction_q, correction_k, correction_v, token_mask
            )
        with profile_range("brainwm.mage_attention/differential_output"):
            lam = self.omega + torch.exp((self.lambda_a * self.lambda_b).sum()) \
                - torch.exp((self.lambda_c * self.lambda_d).sum())
            return self.out(positive - lam * correction) + self.ff(h)


class TorchSDPAAttention(nn.Module):
    """Standard PyTorch multi-head attention compatibility backend.

    This is an intentionally small reference/swap implementation: PyTorch
    owns projection packing and scaled-dot-product attention dispatch (Flash,
    efficient, or math kernels where available).  The backbone only relies on
    the ``forward(x, token_mask)`` protocol, so a different transformer
    package can be registered without changing the block or checkpoint
    loader.  It is quadratic and is therefore a compatibility baseline rather
    than a production replacement for Flash-BLA.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        n_heads: int = 1,
        **_: object,
    ):
        super().__init__()
        del d_hidden  # Standard MHA keeps the d_model projection contract.
        if n_heads < 1 or d_model % n_heads:
            raise ValueError(
                "PyTorch SDPA attention requires d_model divisible by n_heads"
            )
        self.n_heads = int(n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            self.n_heads,
            dropout=0.0,
            batch_first=True,
            bias=True,
        )

    def prepare_projection(self) -> None:
        """Satisfy the optional preparation protocol used by the backbone."""

    def clear_prepared_projection(self) -> None:
        """Satisfy the optional preparation protocol used by the backbone."""

    def forward(
        self, x: Tensor, token_mask: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "PyTorch SDPA attention expects [batch, tokens, width]"
            )
        query_mask = None
        key_padding_mask = None
        if token_mask is not None:
            if token_mask.shape != x.shape[:2]:
                raise ValueError("token_mask must have shape [batch, tokens]")
            query_mask = token_mask.to(device=x.device, dtype=torch.bool)
            key_padding_mask = ~query_mask
        with profile_range("brainwm.sdpa_attention/norm"):
            normalized = self.norm(x)
        with profile_range("brainwm.sdpa_attention/mha"):
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        # A key padding mask does not suppress invalid query rows.  Keep
        # bucket padding inert for the residual/N-BEATS path explicitly.
        if query_mask is not None:
            attended = attended * query_mask.unsqueeze(-1).to(attended.dtype)
        return attended


class DifferentialLinearAttention(nn.Module):
    """Shared positive-minus-correction linear attention."""
    CUDA_GRID_Y_LIMIT = 65535

    def __init__(self, d_model: int, d_hidden: int, n_heads: int = 1,
                 projection_execution: str = "fused",
                 feature_map_execution: str = "torch",
                 feature_map_min_tokens: int = DEFAULT_COMPILED_FEATURE_MAP_MIN_TOKENS,
                 feature_map_compile_mode: str = "max-autotune-no-cudagraphs",
                 feature_map_bucket_size: int = DEFAULT_FEATURE_MAP_BUCKET_SIZE,
                 feature_map_max_padding_ratio: float = DEFAULT_FEATURE_MAP_MAX_PADDING_RATIO,
                 differential_merge_execution: str = "torch",
                 differential_merge_min_tokens: int = DEFAULT_COMPILED_DIFFERENTIAL_MERGE_MIN_TOKENS,
                 differential_merge_compile_mode: str = "max-autotune-no-cudagraphs"):
        super().__init__()
        if n_heads < 1 or d_hidden % n_heads:
            raise ValueError("attention hidden size must be divisible by attention heads")
        self.n_heads = int(n_heads)
        self.head_dim = d_hidden // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.q1, self.k1, self.v1 = (nn.Linear(d_model, d_hidden) for _ in range(3))
        self.q2, self.k2, self.v2 = (nn.Linear(d_model, d_hidden) for _ in range(3))
        self.lambda_a = nn.Parameter(torch.zeros(d_hidden))
        self.lambda_b = nn.Parameter(torch.zeros(d_hidden))
        self.lambda_c = nn.Parameter(torch.zeros(d_hidden))
        self.lambda_d = nn.Parameter(torch.zeros(d_hidden))
        self.omega = 0.5
        self.out = nn.Linear(d_hidden, d_model)
        if projection_execution not in {
            "fla_layernorm_qkv", "fused_triplet", "fused", "separate",
        }:
            raise ValueError(f"unknown attention projection execution: {projection_execution}")
        if feature_map_execution not in _FEATURE_MAP_EXECUTIONS:
            raise ValueError(
                "feature_map_execution must be torch, compile, bucket, or auto"
            )
        if feature_map_min_tokens < 1:
            raise ValueError("feature_map_min_tokens must be positive")
        if feature_map_compile_mode not in _TORCH_COMPILE_MODES:
            raise ValueError(
                f"unknown feature-map compile mode {feature_map_compile_mode!r}"
            )
        if feature_map_bucket_size < 1:
            raise ValueError("feature_map_bucket_size must be positive")
        if feature_map_max_padding_ratio < 1.0:
            raise ValueError("feature_map_max_padding_ratio must be at least 1")
        if differential_merge_execution not in _DIFFERENTIAL_MERGE_EXECUTIONS:
            raise ValueError(
                "differential_merge_execution must be torch, compile, or auto"
            )
        if differential_merge_min_tokens < 1:
            raise ValueError("differential_merge_min_tokens must be positive")
        if differential_merge_compile_mode not in _TORCH_COMPILE_MODES:
            raise ValueError(
                "unknown differential-merge compile mode "
                f"{differential_merge_compile_mode!r}"
            )
        self.projection_execution = projection_execution
        self.feature_map_execution = feature_map_execution
        self.feature_map_min_tokens = int(feature_map_min_tokens)
        self.feature_map_compile_mode = feature_map_compile_mode
        self.feature_map_bucket_size = int(feature_map_bucket_size)
        self.feature_map_max_padding_ratio = float(feature_map_max_padding_ratio)
        self._feature_map_compile_failed = False
        self._last_feature_map_backend = "torch"
        self._last_feature_map_bucket = None
        self.differential_merge_execution = differential_merge_execution
        self.differential_merge_min_tokens = int(differential_merge_min_tokens)
        self.differential_merge_compile_mode = differential_merge_compile_mode
        self._differential_merge_compile_failed = False
        self._last_differential_merge_backend = "torch"
        self._cached_projection: tuple | None = None
        # Prepared packed projections are cast once per forward stream.  The
        # original FP32 leaves remain the checkpoint/optimizer parameters; the
        # BF16 views are ordinary differentiable ``Tensor.to`` results and are
        # discarded when the preparation scope closes.
        self._cached_projection_cast: dict[tuple[torch.dtype, torch.device], tuple] = {}

    def _pack_projection(self) -> tuple[Tensor, Tensor]:
        # Q/K/V-major order lets the varlen path view one linear result as
        # [tokens, QKV, differential_heads, head_dim] without three full-size
        # concatenations. _project restores the public six-tensor order used by
        # the padded compatibility path.
        projections = (self.q1, self.q2, self.k1, self.k2, self.v1, self.v2)
        with profile_range("brainwm.linear_attention/projection_weight_pack"):
            return (
                torch.cat([projection.weight for projection in projections]),
                torch.cat([projection.bias for projection in projections]),
            )

    def _pack_triplet_projection(self) -> tuple[tuple[Tensor, Tensor], ...]:
        pairs = ((self.q1, self.q2), (self.k1, self.k2), (self.v1, self.v2))
        with profile_range("brainwm.linear_attention/projection_weight_pack"):
            return tuple(
                (
                    torch.cat([projection.weight for projection in pair]),
                    torch.cat([projection.bias for projection in pair]),
                )
                for pair in pairs
            )

    def prepare_projection(self) -> None:
        if self._cached_projection is not None:
            return
        if self.projection_execution in {"fla_layernorm_qkv", "fused"}:
            self._cached_projection = self._pack_projection()
        elif self.projection_execution == "fused_triplet":
            self._cached_projection = self._pack_triplet_projection()
        self._cached_projection_cast.clear()

    def clear_prepared_projection(self) -> None:
        self._cached_projection = None
        self._cached_projection_cast.clear()

    def _projection_for_autocast(
        self, weight_bias: tuple, hidden: Tensor,
    ) -> tuple:
        """Return one cached reduced-precision projection view for this stream."""
        if not (
            hidden.device.type in {"cuda", "cpu"}
            and torch.is_autocast_enabled(hidden.device.type)
            and self._cached_projection is weight_bias
        ):
            return weight_bias
        dtype = torch.get_autocast_dtype(hidden.device.type)
        key = (dtype, hidden.device)
        cached = self._cached_projection_cast.get(key)
        if cached is None:
            cached = tuple(
                tuple(value.to(dtype=dtype) for value in pair)
                if isinstance(pair, tuple) else pair.to(dtype=dtype)
                for pair in weight_bias
            )
            self._cached_projection_cast[key] = cached
        return cached

    def set_projection_execution(self, execution: str) -> None:
        if execution not in {
            "fla_layernorm_qkv", "fused_triplet", "fused", "separate",
        }:
            raise ValueError(f"unknown attention projection execution: {execution}")
        self.projection_execution = execution
        self._cached_projection = None

    def set_feature_map_execution(self, execution: str) -> None:
        if execution not in _FEATURE_MAP_EXECUTIONS:
            raise ValueError(
                "feature_map_execution must be torch, compile, bucket, or auto"
            )
        self.feature_map_execution = execution
        self._feature_map_compile_failed = False

    def set_feature_map_min_tokens(self, minimum: int) -> None:
        if minimum < 1:
            raise ValueError("feature_map_min_tokens must be positive")
        self.feature_map_min_tokens = int(minimum)

    def set_feature_map_compile_mode(self, mode: str) -> None:
        """Select the TorchInductor policy used by the feature-map path."""
        if mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown feature-map compile mode {mode!r}")
        self.feature_map_compile_mode = mode
        # A previous compile failure is policy-specific.  Let the next call
        # retry after the caller explicitly changes the mode.
        self._feature_map_compile_failed = False

    def set_feature_map_bucket_size(self, bucket_size: int) -> None:
        if bucket_size < 1:
            raise ValueError("feature_map_bucket_size must be positive")
        self.feature_map_bucket_size = int(bucket_size)

    def set_feature_map_max_padding_ratio(self, ratio: float) -> None:
        if ratio < 1.0:
            raise ValueError("feature_map_max_padding_ratio must be at least 1")
        self.feature_map_max_padding_ratio = float(ratio)

    def set_differential_merge_execution(self, execution: str) -> None:
        if execution not in _DIFFERENTIAL_MERGE_EXECUTIONS:
            raise ValueError(
                "differential_merge_execution must be torch, compile, or auto"
            )
        self.differential_merge_execution = execution
        self._differential_merge_compile_failed = False

    def set_differential_merge_min_tokens(self, minimum: int) -> None:
        if minimum < 1:
            raise ValueError("differential_merge_min_tokens must be positive")
        self.differential_merge_min_tokens = int(minimum)

    def set_differential_merge_compile_mode(self, mode: str) -> None:
        if mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown differential-merge compile mode {mode!r}")
        self.differential_merge_compile_mode = mode
        self._differential_merge_compile_failed = False

    def _merge_differential(
        self, positive: Tensor, correction: Tensor,
    ) -> Tensor:
        token_count = positive.numel() // (self.n_heads * self.head_dim)
        should_compile = (
            self.differential_merge_execution == "compile"
            or (
                self.differential_merge_execution == "auto"
                and positive.is_cuda
                and token_count >= self.differential_merge_min_tokens
            )
        )
        if should_compile and not self._differential_merge_compile_failed:
            try:
                merged = _compiled_differential_merge(
                    positive,
                    correction,
                    self.lambda_a,
                    self.lambda_b,
                    self.lambda_c,
                    self.lambda_d,
                    self.omega,
                    mode=self.differential_merge_compile_mode,
                )
                self._last_differential_merge_backend = "compile"
                return merged
            except Exception as exc:
                import os
                if os.environ.get("BRAINWM_COMPILE_STRICT", "0") == "1":
                    raise
                warnings.warn(
                    "compiled differential attention merge disabled after "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._differential_merge_compile_failed = True
                self._last_differential_merge_backend = "torch_fallback"
        else:
            self._last_differential_merge_backend = "torch"
        return _differential_merge_eager(
            positive,
            correction,
            self.lambda_a,
            self.lambda_b,
            self.lambda_c,
            self.lambda_d,
            self.omega,
        )

    def _project(self, h: Tensor) -> tuple[Tensor, ...]:
        if self.projection_execution == "separate":
            return tuple(projection(h) for projection in (
                self.q1, self.k1, self.v1, self.q2, self.k2, self.v2,
            ))
        if self.projection_execution == "fused_triplet":
            weight_bias = self._cached_projection
            if weight_bias is None:
                weight_bias = self._pack_triplet_projection()
            q, k, v = (
                nn.functional.linear(h, *pair) for pair in weight_bias
            )
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            v1, v2 = v.chunk(2, dim=-1)
            return q1, k1, v1, q2, k2, v2
        weight_bias = self._cached_projection
        if weight_bias is None:
            weight_bias = self._pack_projection()
        projected = nn.functional.linear(h, *weight_bias)
        q1,q2,k1,k2,v1,v2=projected.chunk(6,dim=-1)
        return q1,k1,v1,q2,k2,v2

    def _project_varlen(self, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return contiguous paired Q/K/V heads for the varlen kernel."""
        if self.projection_execution == "fused_triplet":
            weight_bias = self._cached_projection
            if weight_bias is None:
                weight_bias = self._pack_triplet_projection()
            weight_bias = self._projection_for_autocast(weight_bias, h)
            if self._cached_projection is not None and weight_bias is not self._cached_projection:
                compute_dtype = weight_bias[0][0].dtype
                with torch.autocast(device_type=h.device.type, enabled=False):
                    h = h.to(dtype=compute_dtype)
                    return tuple(
                        nn.functional.linear(h, *pair).view(
                            h.shape[0], 2 * self.n_heads, self.head_dim
                        )
                        for pair in weight_bias
                    )
            return tuple(
                nn.functional.linear(h, *pair).view(
                    h.shape[0], 2 * self.n_heads, self.head_dim
                )
                for pair in weight_bias
            )
        if self.projection_execution == "fused":
            weight_bias = self._cached_projection
            if weight_bias is None:
                weight_bias = self._pack_projection()
            weight_bias = self._projection_for_autocast(weight_bias, h)
            if self._cached_projection is not None and weight_bias is not self._cached_projection:
                compute_dtype = weight_bias[0].dtype
                with torch.autocast(device_type=h.device.type, enabled=False):
                    h = h.to(dtype=compute_dtype)
                    projected=nn.functional.linear(h,*weight_bias).view(
                        h.shape[0],3,2*self.n_heads,self.head_dim,
                    )
            else:
                projected=nn.functional.linear(h,*weight_bias).view(
                    h.shape[0],3,2*self.n_heads,self.head_dim,
                )
            queries,keys,values=projected.unbind(dim=1)
            return queries,keys,values

        def split(value: Tensor) -> Tensor:
            return value.view(value.shape[0], self.n_heads, self.head_dim)

        positive_q, positive_k, positive_v, correction_q, correction_k, correction_v = (
            split(value) for value in self._project(h)
        )
        return (
            torch.cat((positive_q, correction_q), dim=1),
            torch.cat((positive_k, correction_k), dim=1),
            torch.cat((positive_v, correction_v), dim=1),
        )

    def _fla_layernorm_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run FLA's fused LayerNorm + six-way projection on CUDA.

        The concatenated weights are differentiable views built from the
        existing six Linear parameters.  FLA's autograd function therefore
        sends gradients through ``torch.cat`` to the original parameter leaves;
        no state-dict or optimizer slot is changed.  CPU callers use the
        regular fused path because the Triton implementation is CUDA-only.
        """
        weight_bias = self._cached_projection
        if weight_bias is None:
            weight_bias = self._pack_projection()
        layer_norm_linear = _load_fla_layer_norm_linear()
        projected = layer_norm_linear(
            x,
            self.norm.weight,
            self.norm.bias,
            weight_bias[0],
            weight_bias[1],
            eps=self.norm.eps,
        )
        projected = projected.view(
            *x.shape[:-1], 3, 2 * self.n_heads, self.head_dim,
        )
        # The single GEMM lays Q/K/V in adjacent feature ranges.  Flash-BLA
        # consumes each stream with a contiguous token-major stride, so make
        # the three views contiguous once here rather than forcing hidden
        # copies inside every downstream kernel launch.
        return tuple(value.contiguous() for value in projected.unbind(dim=-3))

    def forward(self, x: Tensor, token_mask: Tensor | None = None) -> Tensor:
        if self.projection_execution == "fla_layernorm_qkv" and x.is_cuda:
            with profile_range("brainwm.linear_attention/fla_layernorm_qkv"):
                queries, keys, values = self._fla_layernorm_qkv(x)
            positive_q, correction_q = queries.split(self.n_heads, dim=-2)
            positive_k, correction_k = keys.split(self.n_heads, dim=-2)
            positive_v, correction_v = values.split(self.n_heads, dim=-2)
            positive_q, correction_q = positive_q.transpose(-3, -2).contiguous(), correction_q.transpose(-3, -2).contiguous()
            positive_k, correction_k = positive_k.transpose(-3, -2).contiguous(), correction_k.transpose(-3, -2).contiguous()
            positive_v, correction_v = positive_v.transpose(-3, -2).contiguous(), correction_v.transpose(-3, -2).contiguous()
        else:
            with profile_range("brainwm.linear_attention/norm"):
                h = self.norm(x)
            def split(value: Tensor) -> Tensor:
                return value.view(*value.shape[:-1], self.n_heads, self.head_dim).transpose(1, 2)
            with profile_range("brainwm.linear_attention/projections"):
                positive_q, positive_k, positive_v, correction_q, correction_k, correction_v = (
                    split(value) for value in self._project(h)
                )
        with profile_range("brainwm.linear_attention/positive_kernel"):
            positive = MAGEAttentionExpert._kernel(
                positive_q, positive_k, positive_v, token_mask,
            )
        with profile_range("brainwm.linear_attention/correction_kernel"):
            correction = MAGEAttentionExpert._kernel(
                correction_q, correction_k, correction_v, token_mask,
            )
        with profile_range("brainwm.linear_attention/differential_merge"):
            combined = self._merge_differential(
                positive, correction,
            ).transpose(1, 2).reshape(*x.shape[:-1], -1)
        with profile_range("brainwm.linear_attention/output_projection"):
            return self.out(combined)

    @staticmethod
    def build_packed_layout(row_ptr: Tensor, token_count: int,
                            group_cells: int = 32,
                            max_attention_tokens: int | None = None,
                            ) -> PackedAttentionLayout:
        """Group similar CSR lengths while preserving exact cell boundaries.

        ``group_cells`` defines a memory-safe base partition. When an absolute
        attention-slot budget is supplied, adjacent base groups are greedily
        coalesced while the resulting padded layout stays inside that budget.
        This removes small-kernel launches on sparse batches without changing
        cell boundaries or the attention equation.
        """
        if group_cells < 1:
            raise ValueError("packed attention group size must be positive")
        if token_count < 1:
            raise ValueError("packed attention requires at least one token")
        if max_attention_tokens is not None and max_attention_tokens < 1:
            raise ValueError("packed attention token budget must be positive")
        lengths = row_ptr[1:] - row_ptr[:-1]
        order = torch.argsort(lengths, stable=True)
        ordered_lengths = lengths[order]
        ends = list(range(min(group_cells, len(order)), len(order) + 1, group_cells))
        if not ends or ends[-1] != len(order):
            ends.append(len(order))
        widths = ordered_lengths[torch.as_tensor(ends, device=row_ptr.device) - 1].detach().cpu().tolist()
        begins = [0, *ends[:-1]]
        segments = [[begin, end, int(width)]
                    for begin, end, width in zip(begins, ends, widths)]
        allocated_tokens = sum(
            (end - begin) * width for begin, end, width in segments
        )
        if max_attention_tokens is not None:
            budget = max(int(max_attention_tokens), int(ordered_lengths[-1]))
            if allocated_tokens > budget:
                raise ValueError(
                    f"base packed layout needs {allocated_tokens} attention slots, "
                    f"above budget {budget}"
                )
            while len(segments) > 1:
                candidates = []
                for index, (left, right) in enumerate(
                    zip(segments[:-1], segments[1:])
                ):
                    extra = (left[1] - left[0]) * (right[2] - left[2])
                    candidates.append((extra, index))
                extra, index = min(candidates)
                if allocated_tokens + extra > budget:
                    break
                left, right = segments[index:index + 2]
                segments[index:index + 2] = [[left[0], right[1], right[2]]]
                allocated_tokens += extra
            begins = [segment[0] for segment in segments]
            ends = [segment[1] for segment in segments]
            widths = [segment[2] for segment in segments]
        groups=[]; target_indices=[]; allocated_tokens=0; begin=0
        for begin,end,width in zip(begins,ends,widths):
            cells=order[begin:end];width=int(width)
            if width == 0:
                continue
            positions=torch.arange(width,device=row_ptr.device)
            valid=positions.unsqueeze(0)<lengths[cells].unsqueeze(1)
            indices=(row_ptr[cells].unsqueeze(1)+positions).clamp_max(token_count-1)
            groups.append((indices,valid))
            target_indices.append(indices[valid])
            allocated_tokens += indices.numel()
        concatenated=torch.cat(target_indices)
        if concatenated.numel() != token_count:
            raise ValueError("packed attention layout does not cover every CSR token")
        return PackedAttentionLayout(tuple(groups),torch.argsort(concatenated),allocated_tokens)

    def forward_packed(self, x: Tensor, layout: PackedAttentionLayout) -> Tensor:
        with profile_range("brainwm.linear_attention/pack"):
            buckets=layout.pack(x)
        with profile_range("brainwm.linear_attention/packed_buckets"):
            outputs=tuple(self.forward(bucket,token_mask) for bucket,(_,token_mask)
                          in zip(buckets,layout.groups))
        with profile_range("brainwm.linear_attention/unpack"):
            return layout.unpack(outputs)

    def forward_varlen(
        self, x: Tensor, cu_seqlens: Tensor, token_lengths: Tensor,
        max_seqlen: int,
    ) -> Tensor:
        """Run packed non-causal attention with the upstream Flash BLA kernel."""
        if not x.is_cuda:
            raise ValueError("Flash BLA variable-length attention requires CUDA")
        sequence_count=cu_seqlens.numel()-1
        combined_heads=2*self.n_heads
        if sequence_count*combined_heads > self.CUDA_GRID_Y_LIMIT:
            maximum=self.CUDA_GRID_Y_LIMIT//combined_heads
            raise ValueError(
                f"Flash BLA supports at most {maximum} cells per call with "
                f"{self.n_heads} differential-attention heads; got {sequence_count}"
            )
        from flash_bla.ops.linear_attn import varlen_linear_attention

        if self.projection_execution == "fla_layernorm_qkv":
            with profile_range("brainwm.linear_attention/fla_layernorm_qkv"):
                queries, keys, values = self._fla_layernorm_qkv(x)
            h = x
        else:
            with profile_range("brainwm.linear_attention/norm"):
                h=self.norm(x)
            with profile_range("brainwm.linear_attention/projections"):
                queries,keys,values=self._project_varlen(h)
        token_count = int(queries.shape[0])
        eligible = token_count >= self.feature_map_min_tokens
        use_bucket = (
            not self._feature_map_compile_failed
            and self.feature_map_execution == "bucket"
        )
        use_dynamic_compile = (
            not self._feature_map_compile_failed
            and (
                self.feature_map_execution == "compile"
                or (
                    self.feature_map_execution == "auto"
                    and queries.is_cuda
                    and eligible
                )
            )
        )
        if use_bucket or use_dynamic_compile:
            try:
                with profile_range(
                    "brainwm.linear_attention/varlen_feature_map_"
                    + ("bucket" if use_bucket else "compiled")
                ):
                    if use_bucket:
                        queries, keys = _compiled_flash_bla_feature_map_bucket(
                            queries, keys, token_lengths,
                            bucket_size=self.feature_map_bucket_size,
                            mode=self.feature_map_compile_mode,
                            max_padding_ratio=self.feature_map_max_padding_ratio,
                        )
                        self._last_feature_map_bucket = feature_map_bucket_for_tokens(
                            token_count, self.feature_map_bucket_size,
                        )
                        self._last_feature_map_backend = "compile_bucket"
                    else:
                        queries, keys = _compiled_flash_bla_feature_map(
                            queries, keys, token_lengths,
                            mode=self.feature_map_compile_mode,
                        )
                        self._last_feature_map_bucket = None
                        self._last_feature_map_backend = "compile"
            except Exception as exc:
                # A compiler or Triton version mismatch must not make a
                # checkpoint unusable.  Disable this instance after the first
                # failure and retain the exact eager equation for the rest of
                # the run.  Strict infrastructure tests can request a hard
                # failure through the same convention as the FMT compiler.
                import os
                if os.environ.get("BRAINWM_COMPILE_STRICT", "0") == "1":
                    raise
                warnings.warn(
                    f"compiled Flash-BLA feature map disabled after "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._feature_map_compile_failed = True
                self._last_feature_map_backend = "torch_fallback"
                self._last_feature_map_bucket = None
                with profile_range("brainwm.linear_attention/varlen_feature_map"):
                    queries, keys = _flash_bla_feature_map_eager(
                        queries, keys, token_lengths,
                    )
        else:
            with profile_range("brainwm.linear_attention/varlen_feature_map"):
                # Flash BLA normalizes both K^T V and sum(K) by sequence length.
                # Scaling Q by that same length preserves the existing fixed-
                # epsilon BrainWM equation exactly while using the upstream
                # kernel unchanged.
                queries, keys = _flash_bla_feature_map_eager(
                    queries, keys, token_lengths,
                )
            self._last_feature_map_backend = "torch"
            self._last_feature_map_bucket = None
        with profile_range("brainwm.linear_attention/varlen_kernel"):
            attended=varlen_linear_attention(
                queries,keys,values,cu_seqlens,max_seqlen,eps=1e-6,
            )
        with profile_range("brainwm.linear_attention/differential_merge"):
            positive,correction=attended.split(self.n_heads,dim=1)
            combined=self._merge_differential(
                positive,correction,
            ).reshape(x.shape[0],-1)
        with profile_range("brainwm.linear_attention/output_projection"):
            return self.out(combined)


class DeepSpeedEasyMoEBlock(nn.Module):
    """Shared differential attention plus optimized top-1/top-2 sparse FFN MoE."""
    def __init__(self, d_model: int, d_hidden: int, n_experts: int,
                 ffn_expansion: float = 2.67, top_k: int = 2,
                 attention_heads: int = 1,
                 attention_backend: str = "flash_bla_differential",
                 expert_attention_backend: str = "mage_differential",
                 expert_execution: str = "batched",
                 attention_projection_execution: str = "fused",
                 fuse_nbeats_heads: bool = True,
                 residual_norm_execution: str = "torch",
                 residual_norm_min_tokens: int = DEFAULT_FLA_RESIDUAL_NORM_MIN_TOKENS,
                 feature_map_execution: str = "torch",
                 feature_map_min_tokens: int = DEFAULT_COMPILED_FEATURE_MAP_MIN_TOKENS,
                 feature_map_compile_mode: str = "max-autotune-no-cudagraphs",
                 feature_map_bucket_size: int = DEFAULT_FEATURE_MAP_BUCKET_SIZE,
                 feature_map_max_padding_ratio: float = DEFAULT_FEATURE_MAP_MAX_PADDING_RATIO,
                 differential_merge_execution: str = "torch",
                 differential_merge_min_tokens: int = DEFAULT_COMPILED_DIFFERENTIAL_MERGE_MIN_TOKENS,
                 differential_merge_compile_mode: str = "max-autotune-no-cudagraphs",
                 population_balance_mode: str = "exact",
                 router_execution: str = "eager",
                 router_min_tokens: int = 256,
                 router_compile_mode: str = "max-autotune-no-cudagraphs",
                 router_compile_dynamic: bool = False):
        super().__init__()
        try:
            from deepspeed.moe.layer import MoE
        except ImportError as exc:
            raise RuntimeError(
                "easy_deepspeed backend requires deepspeed==0.19.6"
            ) from exc
        if top_k not in {1, 2}:
            raise ValueError("DeepSpeed routing supports top_k=1 or top_k=2")
        self.attention_backend = backend_spec("attention", attention_backend).name
        # The easy block uses one shared attention stream; retain the common
        # field for factory symmetry and checkpoint/telemetry introspection.
        self.expert_attention_backend = backend_spec(
            "attention", expert_attention_backend,
        ).name
        self.attention = create_backend(
            "attention", self.attention_backend,
            d_model=d_model, d_hidden=d_hidden, n_heads=attention_heads,
            projection_execution=attention_projection_execution,
            feature_map_execution=feature_map_execution,
            feature_map_min_tokens=feature_map_min_tokens,
            feature_map_compile_mode=feature_map_compile_mode,
            feature_map_bucket_size=feature_map_bucket_size,
            feature_map_max_padding_ratio=feature_map_max_padding_ratio,
            differential_merge_execution=differential_merge_execution,
            differential_merge_min_tokens=differential_merge_min_tokens,
            differential_merge_compile_mode=differential_merge_compile_mode,
        )
        self.moe_norm = nn.LayerNorm(d_model)
        self.moe = MoE(
            hidden_size=d_model, expert=SwiGLU(d_model, ffn_expansion),
            num_experts=n_experts, ep_size=1, k=top_k,
            capacity_factor=1.0, eval_capacity_factor=1.0,
            # Source-homogeneous dynamic batches can legitimately contain one
            # cell at a shard boundary. DeepSpeed's default floor rejects it.
            min_capacity=0, drop_tokens=False, use_rts=False,
            use_tutel=False, top2_2nd_expert_sampling=False,
        )
        # Keep DeepSpeed's complete capacity/dispatch implementation while
        # exposing the dense gate projection to the optional fixed-shape
        # compiler. The wrapper owns the same wg parameter, so checkpoint keys
        # remain unchanged.
        self.moe.deepspeed_moe.gate = CompiledTopKGate(
            self.moe.deepspeed_moe.gate,
            execution=router_execution,
            min_tokens=router_min_tokens,
            mode=router_compile_mode,
            dynamic=router_compile_dynamic,
        )
        self.moe.deepspeed_moe.experts = BatchedSwiGLUExperts(
            self.moe.deepspeed_moe.experts,
            "batched" if expert_execution == "grouped" else expert_execution,
        )
        self.expert_execution = expert_execution
        if router_execution not in _ROUTER_EXECUTIONS:
            raise ValueError("router_execution must be eager, compile, or auto")
        if router_min_tokens < 1:
            raise ValueError("router_min_tokens must be positive")
        if router_compile_mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown router compile mode {router_compile_mode!r}")
        self.router_execution = router_execution
        self.router_min_tokens = int(router_min_tokens)
        self.router_compile_mode = router_compile_mode
        self.router_compile_dynamic = bool(router_compile_dynamic)
        self._last_router_backend = "eager"
        if population_balance_mode not in {"exact", "combined"}:
            raise ValueError(
                "population_balance_mode must be exact or combined"
            )
        self.population_balance_mode = population_balance_mode
        if residual_norm_execution not in _RESIDUAL_NORM_EXECUTIONS:
            raise ValueError(
                "residual_norm_execution must be torch, torch_bf16, fla, or auto"
            )
        self.residual_norm_execution = residual_norm_execution
        if residual_norm_min_tokens < 1:
            raise ValueError("residual_norm_min_tokens must be positive")
        self.residual_norm_min_tokens = int(residual_norm_min_tokens)
        self.backcast = nn.Linear(d_model, d_model)
        self.forecast = nn.Linear(d_model, d_model)
        self.fuse_nbeats_heads = bool(fuse_nbeats_heads)
        self._cached_nbeats_projection: tuple[Tensor, Tensor] | None = None
        self.routing_balance_loss = torch.tensor(0.0)

    def _residual_norm(self, x: Tensor, attended: Tensor) -> Tensor:
        """Normalize ``x + attended`` with an execution-only backend.

        ``torch`` preserves the historical autocast behavior exactly.  The
        ``torch_bf16`` path keeps PyTorch's numerically stable LayerNorm but
        casts its result back to the activation dtype, avoiding an accidental
        FP32 state promotion under BF16 autocast.  ``fla`` uses the reviewed
        upstream fused residual+LayerNorm kernel and naturally returns the
        autocast activation dtype on CUDA.
        """
        execution = self.residual_norm_execution
        token_count = self._token_count(x)
        use_fla = (
            x.is_cuda
            and (execution == "fla"
                 or (execution == "auto"
                     and token_count >= self.residual_norm_min_tokens))
        )
        if use_fla:
            norm_x, norm_residual = x, attended
            # The upstream Triton kernel is specialized for reduced-precision
            # activations.  Keep explicit ``fla`` usable outside an autocast
            # context while preserving the same dtype for both residual terms.
            if norm_x.dtype not in {torch.float16, torch.bfloat16}:
                target_dtype = (
                    attended.dtype
                    if attended.dtype in {torch.float16, torch.bfloat16}
                    else torch.bfloat16
                )
                norm_x = norm_x.to(dtype=target_dtype)
                norm_residual = norm_residual.to(dtype=target_dtype)
            with profile_range(
                f"{getattr(self, 'profile_prefix', 'brainwm.backbone_block')}/"
                "residual_norm_fla"
            ):
                return _load_fla_layer_norm()(
                    norm_x,
                    self.moe_norm.weight,
                    self.moe_norm.bias,
                    residual=norm_residual,
                    eps=self.moe_norm.eps,
                )
        with profile_range(
            f"{getattr(self, 'profile_prefix', 'brainwm.backbone_block')}/"
            "residual_norm_torch"
        ):
            normalized = self.moe_norm(x + attended)
        use_bf16_state = (
            execution == "torch_bf16"
            or (execution == "auto"
                and self._token_count(x) >= self.residual_norm_min_tokens)
        )
        if use_bf16_state:
            # The token embedding path can start in FP32 (Embedding is not an
            # autocast op) while the attention branch is BF16.  Use the
            # computed activation dtype as the state contract; casting to
            # ``x.dtype`` here would leave the accidental FP32 promotion in
            # place on the first block.
            activation_dtype = attended.dtype if attended.is_floating_point() else x.dtype
            if normalized.dtype != activation_dtype:
                normalized = normalized.to(dtype=activation_dtype)
        return normalized

    def _token_count(self, value: Tensor) -> int:
        if value.ndim == 0 or value.shape[-1] == 0:
            return 0
        return int(value.numel() // value.shape[-1])

    def _residual_subtract(self, x: Tensor, backcast: Tensor) -> Tensor:
        """Keep the residual stream in the selected activation dtype."""
        use_bf16_state = (
            self.residual_norm_execution in {"torch_bf16", "fla"}
            or (self.residual_norm_execution == "auto"
                and self._token_count(x) >= self.residual_norm_min_tokens)
        )
        if use_bf16_state:
            if (x.is_floating_point() and backcast.is_floating_point()
                    and x.dtype != backcast.dtype):
                x = x.to(dtype=backcast.dtype)
        return x - backcast

    def _nbeats_projection(
        self, features: Tensor, *, return_forecast: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Project correction and forecast heads with one wide GEMM."""
        if not return_forecast or not self.fuse_nbeats_heads:
            backcast = self.backcast(features)
            forecast = self.forecast(features) if return_forecast else None
            return backcast, forecast
        with profile_range("brainwm.nbeats/fused_projection"):
            packed = self._cached_nbeats_projection
            if packed is None:
                backcast, forecast = _fused_nbeats_pair(
                    self.backcast, self.forecast, features,
                )
            else:
                weight, bias = packed
                projected = nn.functional.linear(features, weight, bias)
                width = self.backcast.out_features
                backcast, forecast = projected[..., :width], projected[..., width:]
        return backcast, forecast

    def prepare_nbeats_heads(self) -> None:
        """Pack N-BEATS rows once for the current forward graph."""
        if not self.fuse_nbeats_heads or self._cached_nbeats_projection is not None:
            return
        with profile_range("brainwm.nbeats/weight_pack"):
            self._cached_nbeats_projection = (
                torch.cat((self.backcast.weight, self.forecast.weight), dim=0),
                torch.cat((self.backcast.bias, self.forecast.bias), dim=0),
            )

    def clear_nbeats_heads(self) -> None:
        self._cached_nbeats_projection = None

    def _grouped_moe(self, cell_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run dropless routed SwiGLU with the upstream grouped-GEMM kernel.

        DeepSpeed still owns the gate and all expert parameters. This adapter
        changes only dispatch and matrix multiplication; state-dict keys and
        AdamW parameter identities therefore stay compatible with old runs.
        ``grouped-gemm==0.3.0`` currently requires its batch-size descriptor on
        CPU, so one explicit descriptor copy is retained and measured.
        """
        try:
            from grouped_gemm import ops as grouped_ops
        except ImportError as exc:
            raise RuntimeError(
                "grouped expert execution requires grouped-gemm==0.3.0"
            ) from exc
        if not cell_input.is_cuda:
            raise RuntimeError("grouped expert execution requires CUDA")
        flat = cell_input.reshape(-1, cell_input.shape[-1])
        gate = self.moe.deepspeed_moe.gate
        experts = self.moe.deepspeed_moe.experts
        ep_size = int(getattr(self.moe.deepspeed_moe, "ep_size", 1))
        if ep_size != 1:
            raise RuntimeError(
                "grouped expert execution requires ep_size=1; "
                f"got {ep_size}"
            )
        expert_count = int(experts.num_local_experts)
        top_k = int(gate.k)
        with profile_range("brainwm.grouped_moe/gate"):
            logits, router_backend = run_router_projection_logits(
                flat, gate.wg.weight, execution=self.router_execution,
                mode=self.router_compile_mode, dynamic=self.router_compile_dynamic,
                min_tokens=self.router_min_tokens,
            )
            self._last_router_backend = router_backend
            probabilities = logits.softmax(dim=-1)
            route_weights, route_experts = torch.topk(
                probabilities, k=top_k, dim=-1, sorted=True,
            )
            first_counts = torch.bincount(
                route_experts[:, 0], minlength=expert_count,
            )
            importance = probabilities.mean(dim=0)
            first_load = first_counts.to(importance.dtype) / max(len(flat), 1)
            auxiliary_loss = (importance * first_load).sum() * expert_count
            if top_k > 1:
                route_weights = route_weights / route_weights.sum(
                    dim=-1, keepdim=True,
                ).clamp_min(torch.finfo(route_weights.dtype).eps)
        with profile_range("brainwm.grouped_moe/sort_dispatch"):
            token_ids = torch.arange(len(flat), device=flat.device).unsqueeze(1)
            token_ids = token_ids.expand(-1, top_k).reshape(-1)
            route_experts = route_experts.reshape(-1)
            route_weights = route_weights.reshape(-1)
            order = torch.argsort(route_experts, stable=True)
            sorted_experts = route_experts[order]
            sorted_tokens = token_ids[order]
            sorted_input = flat[sorted_tokens].contiguous()
            sorted_weights = route_weights[order]
            tokens_per_expert = torch.bincount(
                sorted_experts, minlength=expert_count,
            )
        with profile_range("brainwm.grouped_moe/counts_to_cpu"):
            batch_sizes = tokens_per_expert.to(device="cpu", dtype=torch.long)
        # grouped-gemm 0.3.0's CUDA kernel is BF16-only. Keep the adapter
        # usable outside an autocast region by doing the expert matmuls in
        # BF16 and restoring the caller's dtype at the boundary.
        compute_dtype = torch.bfloat16
        sorted_input = sorted_input.to(dtype=compute_dtype)
        input_weight, input_bias, output_weight, output_bias = experts.weights_for(
            compute_dtype,
        )
        with profile_range("brainwm.grouped_moe/input_projection"):
            projected = grouped_ops.gmm(
                sorted_input, input_weight, batch_sizes, trans_b=True,
            )
            projected = projected + input_bias[sorted_experts]
            gate_values, values = projected.chunk(2, dim=-1)
        with profile_range("brainwm.grouped_moe/activation"):
            hidden = nn.functional.silu(gate_values) * values
        with profile_range("brainwm.grouped_moe/output_projection"):
            expert_output = grouped_ops.gmm(
                hidden, output_weight, batch_sizes, trans_b=True,
            )
            expert_output = expert_output + output_bias[sorted_experts]
        with profile_range("brainwm.grouped_moe/combine"):
            combined = flat.to(dtype=compute_dtype).new_zeros(flat.shape)
            combined.index_add_(
                0, sorted_tokens,
                expert_output * sorted_weights.to(
                    expert_output.dtype,
                ).unsqueeze(-1),
            )
        return combined.to(dtype=flat.dtype).view_as(cell_input), auxiliary_loss, tokens_per_expert

    def _run_moe(self, cell_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.expert_execution == "grouped":
            return self._grouped_moe(cell_input)
        result = self.moe(cell_input)
        self._last_router_backend = getattr(
            self.moe.deepspeed_moe.gate, "last_backend", "eager"
        )
        return result

    def _routing_auxiliary_loss(self, cell_input: Tensor) -> Tensor:
        """Compute DeepSpeed's balance loss for one population.

        ``finish_varlen_pair`` can dispatch the main and source populations in
        one call, but their balance terms must still be population-local.  The
        gate loss is inexpensive compared with expert dispatch, so reproduce
        DeepSpeed's documented top-1/top-2 formula from the same gate weights
        while retaining the original objective and checkpoint contract.
        """
        flat = cell_input.reshape(-1, cell_input.shape[-1])
        if flat.shape[0] == 0:
            return flat.sum() * 0.0
        gate = self.moe.deepspeed_moe.gate
        logits = nn.functional.linear(
            flat.float(), gate.wg.weight.float(), bias=None,
        )
        probabilities = logits.softmax(dim=-1)
        first = probabilities.argmax(dim=-1)
        expert_count = int(probabilities.shape[-1])
        importance = probabilities.mean(dim=0)
        # Keep the reference dense route-load equation here.  On the actual
        # H20 batch shapes the one-hot reduction is faster than a separate
        # bincount kernel, so replacing it would regress the exact-balance
        # path despite using less temporary memory.
        first_mask = nn.functional.one_hot(
            first, num_classes=expert_count,
        ).to(probabilities.dtype)
        load = first_mask.mean(dim=0)
        if int(gate.k) == 1:
            return (importance * load).sum() * expert_count
        # DeepSpeed's top2gating balance term only uses the first route.
        return (importance * load).mean() * expert_count * expert_count

    def _routing_auxiliary_losses_many(
        self, populations: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Compute population-local balance losses with one gate projection.

        The DeepSpeed dispatcher still owns routing and expert execution.  Its
        combined call does not expose per-population gate probabilities, so the
        old fused path projected each population again just to build telemetry.
        Concatenating the cell rows is exact for a row-wise linear gate and
        reduces that bookkeeping to one GEMM/softmax/one-hot pass.
        """
        if not populations:
            raise ValueError("at least one MoE population is required")
        sizes = tuple(int(population.reshape(-1, population.shape[-1]).shape[0])
                      for population in populations)
        if any(size == 0 for size in sizes):
            return tuple(
                population.reshape(-1, population.shape[-1]).sum() * 0.0
                for population in populations
            )
        flat = torch.cat(
            [population.reshape(-1, population.shape[-1])
             for population in populations], dim=0,
        )
        gate = self.moe.deepspeed_moe.gate
        with profile_range("brainwm.experts/balance_gate"):
            logits = nn.functional.linear(
                flat.float(), gate.wg.weight.float(), bias=None,
            )
            probabilities = logits.softmax(dim=-1)
            first = probabilities.argmax(dim=-1)
            expert_count = int(probabilities.shape[-1])
            probability_parts = probabilities.split(sizes, dim=0)
            first_mask = nn.functional.one_hot(
                first, num_classes=expert_count,
            ).to(probabilities.dtype)
            mask_parts = first_mask.split(sizes, dim=0)
            losses = []
            for part, mask in zip(probability_parts, mask_parts, strict=True):
                importance = part.mean(dim=0)
                load = mask.mean(dim=0)
                if int(gate.k) == 1:
                    losses.append((importance * load).sum() * expert_count)
                else:
                    losses.append(
                        (importance * load).mean() * expert_count * expert_count
                    )
        return tuple(losses)

    def _run_moe_pair(
        self, main_cells: Tensor, source_cells: Tensor,
        *, main_only_balance: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Dispatch two populations together and retain separate gate losses.

        The cell rows remain concatenated only for the execution backend.  A
        single DeepSpeed dispatcher therefore amortizes routing, capacity
        construction, and expert GEMMs, while the returned tensors are split
        back at the original boundary.  The legacy loop backend is kept as a
        reference path because it is used for controlled numerical A/B tests.
        """
        outputs, balances = self._run_moe_many(
            (main_cells, source_cells), main_only_balance=main_only_balance,
        )
        return outputs[0], outputs[1], balances[0], balances[1]

    def _run_moe_many(
        self, populations: tuple[Tensor, ...],
        *, main_only_balance: bool = False,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Dispatch several cell populations in one exact top-1 call.

        DeepSpeed's ``drop_tokens=False`` top-1 path assigns each row exactly
        one expert and sets capacity to the largest observed expert count.
        Concatenating populations therefore changes neither a row's gate nor
        its expert output; it only amortizes dispatcher and GEMM launches. We
        deliberately fall back for top-2 or loop execution because their
        stochastic routing/capacity behavior must remain population-local.
        """
        if not populations:
            raise ValueError("at least one MoE population is required")
        gate = self.moe.deepspeed_moe.gate
        can_fuse = (
            self.expert_execution != "loop"
            and int(gate.k) == 1
            and all(population.shape[0] > 0 for population in populations)
        )
        if not can_fuse or len(populations) == 1:
            outputs = []
            balances = []
            for population in populations:
                output, balance, _ = self._run_moe(population)
                outputs.append(output)
                balances.append(balance)
            return tuple(outputs), tuple(balances)

        sizes = tuple(int(population.shape[0]) for population in populations)
        combined = torch.cat(populations, dim=0)
        combined_output, combined_balance, _combined_counts = self._run_moe(combined)
        outputs = tuple(combined_output.split(sizes, dim=0))
        # The first population is the trainable main stream; other populations
        # are detached by the caller and their balance values are telemetry
        # only. ``exact`` retains the historical main-local objective, while
        # ``combined`` reuses the joint DeepSpeed objective and avoids a second
        # gate projection. The selected mode is part of the model config.
        if main_only_balance and self.population_balance_mode == "combined":
            # DeepSpeed has already computed this objective while routing the
            # concatenated populations. Reusing it removes a second gate
            # projection/softmax per block; the mode is explicit because the
            # exact population-local statistic remains available for audits.
            balances = (combined_balance,) * len(populations)
        elif main_only_balance:
            with profile_range("brainwm.experts/main_balance_gate"):
                main_balance = self._routing_auxiliary_loss(populations[0])
            balances = (main_balance,) + tuple(
                combined_balance.detach() for _ in populations[1:]
            )
        else:
            balances = self._routing_auxiliary_losses_many(populations)
        balances = tuple(
            balance if index == 0 else balance.detach()
            for index, balance in enumerate(balances)
        )
        self.moe.deepspeed_moe.l_aux = balances[0]
        return outputs, balances

    def forward(self, x: Tensor, token_mask: Tensor | None = None,
                *, return_forecast: bool = True) -> tuple[Tensor, Tensor | None]:
        prefix=getattr(self,"profile_prefix","brainwm.backbone_block")
        with profile_range(f"{prefix}/linear_attention"):
            attended = self.attention(x, token_mask)
        with profile_range(f"{prefix}/moe_norm"):
            expert_input = self._residual_norm(x, attended)
        with profile_range(f"{prefix}/cell_pool"):
            if token_mask is None:
                cell_input = expert_input.mean(dim=1, keepdim=True)
            else:
                weights = token_mask.unsqueeze(-1).to(expert_input.dtype)
                cell_input = (expert_input * weights).sum(1, keepdim=True) \
                    / weights.sum(1, keepdim=True).clamp_min(1)
        with profile_range(f"{prefix}/moe_router_experts"):
            cell_output, auxiliary_loss, _ = self._run_moe(cell_input)
        with profile_range(f"{prefix}/expert_broadcast"):
            expert_output = cell_output.expand_as(expert_input)
        self.routing_balance_loss = auxiliary_loss
        with profile_range(f"{prefix}/feature_merge"):
            features = attended + expert_output
        with profile_range(f"{prefix}/nbeats_backcast"):
            backcast, forecast = self._nbeats_projection(
                features, return_forecast=return_forecast,
            )
            residual = self._residual_subtract(x, backcast)
        return residual, forecast

    def forward_packed(self,x:tuple[Tensor,...],layout:PackedAttentionLayout,
                       *, return_forecast: bool = True):
        prefix=getattr(self,"profile_prefix","brainwm.backbone_block")
        masks=tuple(token_mask for _,token_mask in layout.groups)
        with profile_range(f"{prefix}/linear_attention"):
            attended=tuple(
                self.attention(bucket,mask) for bucket,mask in zip(x,masks)
            )
        with profile_range(f"{prefix}/moe_norm"):
            expert_input=tuple(
                self._residual_norm(bucket, attention)
                for bucket, attention in zip(x, attended)
            )
        with profile_range(f"{prefix}/cell_pool"):
            cell_input=[]
            for bucket,mask in zip(expert_input,masks):
                weights=mask.unsqueeze(-1).to(bucket.dtype)
                cell_input.append(
                    (bucket*weights).sum(1)/weights.sum(1).clamp_min(1)
                )
            cell_input=torch.cat(cell_input).unsqueeze(1)
        with profile_range(f"{prefix}/moe_router_experts"):
            cell_output,auxiliary_loss,_=self._run_moe(cell_input)
        with profile_range(f"{prefix}/expert_split"):
            cell_output=cell_output.squeeze(1).split([len(bucket) for bucket in x])
        self.routing_balance_loss=auxiliary_loss
        with profile_range(f"{prefix}/feature_merge"):
            features=tuple(attention+output.unsqueeze(1).expand_as(bucket)
                           for attention,output,bucket in zip(attended,cell_output,expert_input))
        with profile_range(f"{prefix}/nbeats_backcast"):
            projections=tuple(
                self._nbeats_projection(feature, return_forecast=return_forecast)
                for feature in features
            )
            residual=tuple(
                self._residual_subtract(bucket, backcast)
                for bucket,(backcast,_) in zip(x,projections)
            )
        forecast=(tuple(item for _,item in projections)
                  if return_forecast else None)
        return residual,forecast

    def forward_varlen(
        self, x: Tensor, cu_seqlens: Tensor, row_ids: Tensor,
        token_lengths: Tensor, max_seqlen: int, *, return_forecast: bool = True,
    ):
        prefix=getattr(self,"profile_prefix","brainwm.backbone_block")
        with profile_range(f"{prefix}/linear_attention"):
            attended=self.attention.forward_varlen(
                x,cu_seqlens,token_lengths,max_seqlen,
            )
        return self.finish_varlen(
            x,attended,cu_seqlens,row_ids,return_forecast=return_forecast,
        )

    def finish_varlen(self,x:Tensor,attended:Tensor,cu_seqlens:Tensor,
                      row_ids:Tensor,*,return_forecast:bool=True):
        """Finish a block after an already computed varlen attention result."""
        prefix=getattr(self,"profile_prefix","brainwm.backbone_block")
        with profile_range(f"{prefix}/moe_norm"):
            expert_input=self._residual_norm(x, attended)
        with profile_range(f"{prefix}/cell_pool"):
            cell_input=torch.segment_reduce(
                expert_input,"mean",offsets=cu_seqlens,
            ).unsqueeze(1)
        with profile_range(f"{prefix}/moe_router_experts"):
            cell_output,auxiliary_loss,_=self._run_moe(cell_input)
        with profile_range(f"{prefix}/expert_broadcast"):
            expert_output=cell_output.squeeze(1)
        self.routing_balance_loss=auxiliary_loss
        with profile_range(f"{prefix}/feature_merge"):
            features=add_csr_rows(
                attended,expert_output,row_ids,cu_seqlens,
            )
        with profile_range(f"{prefix}/nbeats_backcast"):
            backcast, forecast = self._nbeats_projection(
                features, return_forecast=return_forecast,
            )
            residual=self._residual_subtract(x, backcast)
        return residual,forecast

    def finish_varlen_pair(
        self,x:Tensor,attended:Tensor,cu_seqlens:Tensor,row_ids:Tensor,
        main_token_count:int,main_cell_count:int,
    )->tuple[Tensor,Tensor,Tensor]:
        """Finish joined main/source streams with independent MoE routing.

        Token-wise operations stay joined so the next block can consume the
        same contiguous CSR stream without another full hidden-state copy.
        Cell-wise routing is split at ``main_cell_count`` and therefore has the
        same gate inputs and expert populations as two separate encoder calls.
        Only the main partition produces an N-BEATS forecast and balance loss.
        """
        prefix=getattr(self,"profile_prefix","brainwm.backbone_block")
        with profile_range(f"{prefix}/joint_finish/moe_norm"):
            expert_input=self._residual_norm(x, attended)
        with profile_range(f"{prefix}/joint_finish/cell_pool"):
            cell_input=torch.segment_reduce(
                expert_input,"mean",offsets=cu_seqlens,
            ).unsqueeze(1)
            main_cells,source_cells=cell_input.split(
                (main_cell_count,cell_input.shape[0]-main_cell_count),dim=0,
            )
        with profile_range(f"{prefix}/joint_finish/fused_router_experts"):
            main_output,source_output,main_balance,source_balance=(
                self._run_moe_pair(
                    main_cells,source_cells,main_only_balance=True,
                )
            )
        # Match the observable state of two sequential finish_varlen calls;
        # the caller retains main_balance for the training objective.
        self.routing_balance_loss=source_balance
        with profile_range(f"{prefix}/joint_finish/expert_broadcast"):
            cell_output=torch.cat((main_output,source_output),dim=0).squeeze(1)
        with profile_range(f"{prefix}/joint_finish/feature_merge"):
            features=add_csr_rows(attended,cell_output,row_ids,cu_seqlens)
        with profile_range(f"{prefix}/joint_finish/nbeats_backcast"):
            residual=self._residual_subtract(x,self.backcast(features))
        with profile_range(f"{prefix}/joint_finish/main_nbeats_forecast"):
            forecast=self.forecast(features[:main_token_count])
        return residual,forecast,main_balance

    def finish_varlen_triple(
        self, x: Tensor, attended: Tensor, cu_seqlens: Tensor,
        row_ids: Tensor, main_token_count: int, main_cell_count: int,
        source_cell_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Finish three streams with one top-1 dispatch when possible.

        ``x`` and ``attended`` contain main, source, and target rows in that
        order.  The caller detaches the target residual after this method; all
        token-wise work can consequently share the same kernels while the
        first population retains its original balance objective.
        """
        prefix = getattr(self, "profile_prefix", "brainwm.backbone_block")
        with profile_range(f"{prefix}/joint_finish/moe_norm"):
            expert_input = self._residual_norm(x, attended)
        with profile_range(f"{prefix}/joint_finish/cell_pool"):
            cell_input = torch.segment_reduce(
                expert_input, "mean", offsets=cu_seqlens,
            ).unsqueeze(1)
            main_cells, source_cells, target_cells = torch.split(
                cell_input,
                (main_cell_count, source_cell_count,
                 cell_input.shape[0] - main_cell_count - source_cell_count),
                dim=0,
            )
        with profile_range(f"{prefix}/joint_finish/fused_router_experts"):
            outputs, balances = self._run_moe_many(
                (main_cells, source_cells, target_cells),
                main_only_balance=True,
            )
        # Keep target balance telemetry out of the trainable objective; the
        # main balance value is returned to the caller exactly as before.
        self.routing_balance_loss = balances[-1]
        with profile_range(f"{prefix}/joint_finish/expert_broadcast"):
            cell_output = torch.cat(outputs, dim=0).squeeze(1)
        with profile_range(f"{prefix}/joint_finish/feature_merge"):
            features = add_csr_rows(attended, cell_output, row_ids, cu_seqlens)
        with profile_range(f"{prefix}/joint_finish/nbeats_backcast"):
            residual = self._residual_subtract(x, self.backcast(features))
        with profile_range(f"{prefix}/joint_finish/main_nbeats_forecast"):
            forecast = self.forecast(features[:main_token_count])
        return residual, forecast, balances[0]

    @torch.no_grad()
    def update_beta_balance(self, rate: float = 0.01) -> None:
        del rate

    def set_expert_execution(self, execution: str) -> None:
        if execution not in {"batched", "loop", "grouped"}:
            raise ValueError(f"unknown expert execution backend: {execution}")
        self.moe.deepspeed_moe.experts.execution = (
            "batched" if execution == "grouped" else execution
        )
        self.expert_execution = execution

    def set_router_execution(
        self, execution: str, *, min_tokens: int | None = None,
        mode: str | None = None, dynamic: bool | None = None,
    ) -> None:
        if execution not in _ROUTER_EXECUTIONS:
            raise ValueError("router_execution must be eager, compile, or auto")
        if min_tokens is not None and min_tokens < 1:
            raise ValueError("router_min_tokens must be positive")
        if mode is not None and mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown router compile mode {mode!r}")
        self.router_execution = execution
        if min_tokens is not None:
            self.router_min_tokens = int(min_tokens)
        if mode is not None:
            self.router_compile_mode = mode
        if dynamic is not None:
            self.router_compile_dynamic = bool(dynamic)
        gate = self.moe.deepspeed_moe.gate
        if isinstance(gate, CompiledTopKGate):
            gate.execution = self.router_execution
            gate.min_tokens = self.router_min_tokens
            gate.mode = self.router_compile_mode
            gate.dynamic = self.router_compile_dynamic

    def set_attention_projection_execution(self, execution: str) -> None:
        self.attention.set_projection_execution(execution)

    def set_fuse_nbeats_heads(self, enabled: bool) -> None:
        self.fuse_nbeats_heads = bool(enabled)
        self.clear_nbeats_heads()

    def set_feature_map_execution(self, execution: str) -> None:
        self.attention.set_feature_map_execution(execution)

    def set_feature_map_min_tokens(self, minimum: int) -> None:
        self.attention.set_feature_map_min_tokens(minimum)

    def set_feature_map_compile_mode(self, mode: str) -> None:
        self.attention.set_feature_map_compile_mode(mode)

    def set_feature_map_bucket_size(self, bucket_size: int) -> None:
        self.attention.set_feature_map_bucket_size(bucket_size)

    def set_feature_map_max_padding_ratio(self, ratio: float) -> None:
        self.attention.set_feature_map_max_padding_ratio(ratio)

    def set_differential_merge_execution(self, execution: str) -> None:
        self.attention.set_differential_merge_execution(execution)

    def set_differential_merge_min_tokens(self, minimum: int) -> None:
        self.attention.set_differential_merge_min_tokens(minimum)

    def set_differential_merge_compile_mode(self, mode: str) -> None:
        self.attention.set_differential_merge_compile_mode(mode)

    def set_population_balance_mode(self, mode: str) -> None:
        if mode not in {"exact", "combined"}:
            raise ValueError("population_balance_mode must be exact or combined")
        self.population_balance_mode = mode

    def set_residual_norm_execution(self, execution: str) -> None:
        if execution not in _RESIDUAL_NORM_EXECUTIONS:
            raise ValueError(
                "residual_norm_execution must be torch, torch_bf16, fla, or auto"
            )
        self.residual_norm_execution = execution

    def set_residual_norm_min_tokens(self, minimum: int) -> None:
        if minimum < 1:
            raise ValueError("residual_norm_min_tokens must be positive")
        self.residual_norm_min_tokens = int(minimum)


class BetaExpertBlock(nn.Module):
    """Sparse balanced expert assignment with independently optimized beta."""
    def __init__(self, d_model: int, d_hidden: int, n_experts: int, top_k: int = 1,
                 ffn_expansion: float = 2.67, fuse_nbeats_heads: bool = True,
                 expert_attention_backend: str = "mage_differential"):
        super().__init__()
        self.identity = nn.Parameter(torch.randn(n_experts, d_model) / d_model**0.5)
        # Beta is updated by the explicit load-balancing rule, not AdamW.
        # Keep the state_dict key so existing snapshots remain loadable.
        self.register_buffer('beta',torch.zeros(n_experts))
        self.expert_attention_backend = backend_spec(
            "attention", expert_attention_backend,
        ).name
        self.experts = nn.ModuleList([
            create_backend(
                "attention", self.expert_attention_backend,
                d_model=d_model, d_hidden=d_hidden, n_heads=1,
                ffn_expansion=ffn_expansion,
            )
            for _ in range(n_experts)
        ])
        if top_k not in (0, 1):
            raise ValueError('BrainWM positive-affinity routing does not use top-k')
        self.backcast = nn.Linear(d_model, d_model)
        self.forecast = nn.Linear(d_model, d_model)
        self.fuse_nbeats_heads = bool(fuse_nbeats_heads)
        self.register_buffer("usage_ema", torch.zeros(n_experts))

    def _nbeats_projection(
        self, features: Tensor, *, return_forecast: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        if not return_forecast or not self.fuse_nbeats_heads:
            backcast = self.backcast(features)
            forecast = self.forecast(features) if return_forecast else None
            return backcast, forecast
        with profile_range("brainwm.nbeats/fused_projection"):
            backcast, forecast = _fused_nbeats_pair(
                self.backcast, self.forecast, features,
            )
        return backcast, forecast

    def forward(self, x: Tensor, token_mask: Tensor | None = None,
                *, return_forecast: bool = True) -> tuple[Tensor, Tensor | None]:
        prefix=getattr(self,"profile_prefix","brainwm.backbone_block")
        with profile_range(f"{prefix}/cell_pool"):
            if token_mask is None:cell=x.mean(1)
            else:cell=(x*token_mask.unsqueeze(-1)).sum(1)/token_mask.sum(1,keepdim=True).clamp_min(1)
        with profile_range(f"{prefix}/beta_routing"):
            affinity = torch.einsum("bd,ed->be", nn.functional.normalize(cell, dim=-1),
                                   nn.functional.normalize(self.identity, dim=-1)) + self.beta
            probabilities=affinity.softmax(-1)
            positive=affinity.clamp_min(0)
            positive_mass=positive.sum(-1,keepdim=True)
            fallback=positive_mass.squeeze(-1).eq(0)
            route_weights=positive/positive_mass.clamp_min(1e-8)
            if fallback.any():
                fallback_index=affinity.argmax(-1)
                route_weights[fallback]=0
                route_weights[fallback,fallback_index[fallback]]=1
            importance=probabilities.mean(0)
            self.routing_balance_loss=(importance.square().mean() * len(self.experts)**2 - 1.0)
        if self.training:
            with torch.no_grad():
                # Load is activation frequency, not soft routing mass. A cell
                # activates every positive-affinity expert exactly once.
                usage=route_weights.detach().gt(0).float().mean(0)
                if dist.is_initialized():
                    dist.all_reduce(usage, op=dist.ReduceOp.SUM)
                    usage.div_(dist.get_world_size())
                self.usage_ema.mul_(0.99).add_(0.01 * usage)
        with profile_range(f"{prefix}/positive_experts"):
            mixed = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                selected=torch.nonzero(route_weights[:,i].gt(0),as_tuple=False).flatten()
                if not len(selected):continue
                local=expert(x[selected],None if token_mask is None else token_mask[selected])
                mixed[selected]+=local*route_weights[selected,i,None,None]
        with profile_range(f"{prefix}/nbeats_backcast"):
            backcast, forecast = self._nbeats_projection(
                mixed, return_forecast=return_forecast,
            )
            residual = x - backcast
        return residual, forecast

    @torch.no_grad()
    def update_beta_balance(self, rate: float = 0.01) -> None:
        target = self.usage_ema.mean()
        self.beta.add_(rate * (target - self.usage_ema))


class BrainWMFoundationMoE(nn.Module):
    """Position-free model with add-on modality experts."""

    @property
    def fuse_development_attention(self) -> bool:
        return self._fuse_development_attention

    @fuse_development_attention.setter
    def fuse_development_attention(self, value: bool) -> None:
        self._fuse_development_attention = bool(value)
        config = getattr(self, "model_config", None)
        if config is not None:
            config["fuse_development_attention"] = (
                self._fuse_development_attention
            )

    @property
    def fuse_development_target_attention(self) -> bool:
        return self._fuse_development_target_attention

    @fuse_development_target_attention.setter
    def fuse_development_target_attention(self, value: bool) -> None:
        self._fuse_development_target_attention = bool(value)
        config = getattr(self, "model_config", None)
        if config is not None:
            config["fuse_development_target_attention"] = (
                self._fuse_development_target_attention
            )
    def __init__(self, vocab_size: int, d_model: int = 512, d_hidden: int = 1024,
                 n_blocks: int = 8, n_experts: int = 8, top_k: int = 1,
                 modality_dims: Mapping[str, int] | None = None,
                 annotation_classes: int = 0, anatomy_classes: int = 0,
                 ffn_expansion: float = 2.67,
                 categorical_modalities: set[str] | None = None,
                 moe_backend: str = "easy_deepspeed", attention_heads: int = 1,
                 attention_backend: str = "flash_bla_differential",
                 expert_attention_backend: str = "mage_differential",
                 flow_backend: str = "fmt",
                 compile_backend: str = "torch_compile",
                 flow_depth: int = 4, flow_attention_mode: str = "auto",
                 flow_sdpa_backend: str = "flash",
                 packed_group_cells: int = 32,
                 packed_attention_budget: int | None = None,
                 packed_attention_backend: str = "auto",
                 expert_execution: str = "batched",
                 attention_projection_execution: str = "fused",
                 fuse_development_attention: bool = True,
                 fuse_development_target_attention: bool = True,
                 fuse_nbeats_heads: bool = True,
                 residual_norm_execution: str = "torch",
                 residual_norm_min_tokens: int = DEFAULT_FLA_RESIDUAL_NORM_MIN_TOKENS,
                 sinkhorn_epsilon: float = 0.05,
                 sinkhorn_iterations: int = 50,
                 sinkhorn_compile: bool = False,
                 sinkhorn_compile_mode: str = "reduce-overhead",
                 sinkhorn_compile_buckets: tuple[int, ...] = (8, 16, 32, 64),
                 feature_map_execution: str = "torch",
                 feature_map_min_tokens: int = DEFAULT_COMPILED_FEATURE_MAP_MIN_TOKENS,
                 feature_map_compile_mode: str = "max-autotune-no-cudagraphs",
                 feature_map_bucket_size: int = DEFAULT_FEATURE_MAP_BUCKET_SIZE,
                 feature_map_max_padding_ratio: float = DEFAULT_FEATURE_MAP_MAX_PADDING_RATIO,
                 differential_merge_execution: str = "torch",
                 differential_merge_min_tokens: int = DEFAULT_COMPILED_DIFFERENTIAL_MERGE_MIN_TOKENS,
                 differential_merge_compile_mode: str = "max-autotune-no-cudagraphs",
                 population_balance_mode: str = "exact",
                 router_execution: str = "eager",
                 router_min_tokens: int = 256,
                 router_compile_mode: str = "max-autotune-no-cudagraphs",
                 router_compile_dynamic: bool = False):
        super().__init__()
        self.d_model = d_model
        self.attention_heads = int(attention_heads)
        # Resolve implementation names up front.  This gives a useful error
        # before allocating model weights and records the exact choices in the
        # self-describing checkpoint configuration below.
        self.attention_backend = backend_spec("attention", attention_backend).name
        self.expert_attention_backend = backend_spec(
            "attention", expert_attention_backend,
        ).name
        self.moe_backend = backend_spec("moe", moe_backend).name
        self.flow_backend = backend_spec("flow", flow_backend).name
        self.compile_backend = backend_spec("compile", compile_backend).name
        self.position_embedding = False
        if packed_group_cells < 1:
            raise ValueError("packed_group_cells must be positive")
        self.packed_group_cells = int(packed_group_cells)
        if packed_attention_budget is not None and packed_attention_budget < 1:
            raise ValueError("packed_attention_budget must be positive")
        self.packed_attention_budget = (
            None if packed_attention_budget is None
            else int(packed_attention_budget)
        )
        if packed_attention_backend not in {"auto","bucket","flash_bla"}:
            raise ValueError(
                f"unknown packed attention backend: {packed_attention_backend}"
            )
        self.packed_attention_backend=packed_attention_backend
        self.fuse_nbeats_heads = bool(fuse_nbeats_heads)
        if residual_norm_execution not in _RESIDUAL_NORM_EXECUTIONS:
            raise ValueError(
                "residual_norm_execution must be torch, torch_bf16, fla, or auto"
            )
        self.residual_norm_execution = residual_norm_execution
        if residual_norm_min_tokens < 1:
            raise ValueError("residual_norm_min_tokens must be positive")
        self.residual_norm_min_tokens = int(residual_norm_min_tokens)
        if feature_map_execution not in _FEATURE_MAP_EXECUTIONS:
            raise ValueError(
                "feature_map_execution must be torch, compile, bucket, or auto"
            )
        if feature_map_min_tokens < 1:
            raise ValueError("feature_map_min_tokens must be positive")
        if feature_map_compile_mode not in _TORCH_COMPILE_MODES:
            raise ValueError(
                f"unknown feature-map compile mode {feature_map_compile_mode!r}"
            )
        if feature_map_bucket_size < 1:
            raise ValueError("feature_map_bucket_size must be positive")
        if feature_map_max_padding_ratio < 1.0:
            raise ValueError("feature_map_max_padding_ratio must be at least 1")
        self.feature_map_execution = feature_map_execution
        self.feature_map_min_tokens = int(feature_map_min_tokens)
        self.feature_map_compile_mode = feature_map_compile_mode
        self.feature_map_bucket_size = int(feature_map_bucket_size)
        self.feature_map_max_padding_ratio = float(feature_map_max_padding_ratio)
        if differential_merge_execution not in _DIFFERENTIAL_MERGE_EXECUTIONS:
            raise ValueError(
                "differential_merge_execution must be torch, compile, or auto"
            )
        if differential_merge_min_tokens < 1:
            raise ValueError("differential_merge_min_tokens must be positive")
        if differential_merge_compile_mode not in _TORCH_COMPILE_MODES:
            raise ValueError(
                "unknown differential-merge compile mode "
                f"{differential_merge_compile_mode!r}"
            )
        self.differential_merge_execution = differential_merge_execution
        self.differential_merge_min_tokens = int(differential_merge_min_tokens)
        self.differential_merge_compile_mode = differential_merge_compile_mode
        if router_execution not in _ROUTER_EXECUTIONS:
            raise ValueError("router_execution must be eager, compile, or auto")
        if router_min_tokens < 1:
            raise ValueError("router_min_tokens must be positive")
        if router_compile_mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown router compile mode {router_compile_mode!r}")
        self.router_execution = router_execution
        self.router_min_tokens = int(router_min_tokens)
        self.router_compile_mode = router_compile_mode
        self.router_compile_dynamic = bool(router_compile_dynamic)
        if population_balance_mode not in {"exact", "combined"}:
            raise ValueError(
                "population_balance_mode must be exact or combined"
            )
        self.population_balance_mode = population_balance_mode
        self.fuse_development_attention=bool(fuse_development_attention)
        # Execution-only policy for the inference target stream.  It is kept
        # separate so target fusion can be benchmarked without changing the
        # trainable main/source population or checkpoint tensors.
        self.fuse_development_target_attention=bool(
            fuse_development_target_attention
        )
        # OT compilation is an execution accelerator.  Keep the runtime plain
        # Python so compiled callables never enter the module state dict.
        self._sinkhorn_runtime = SinkhornRuntime(
            epsilon=sinkhorn_epsilon,
            iterations=sinkhorn_iterations,
            enabled=sinkhorn_compile,
            mode=sinkhorn_compile_mode,
            buckets=tuple(sinkhorn_compile_buckets),
        )
        self.model_config = {
            "vocab_size": int(vocab_size), "d_model": int(d_model),
            "d_hidden": int(d_hidden), "n_blocks": int(n_blocks),
            "n_experts": int(n_experts), "top_k": int(top_k),
            "modality_dims": dict(modality_dims or {}),
            "annotation_classes": int(annotation_classes),
            "anatomy_classes": int(anatomy_classes),
            "ffn_expansion": float(ffn_expansion),
            "categorical_modalities": sorted(categorical_modalities or set()),
            "moe_backend": self.moe_backend,
            "attention_heads": int(attention_heads),
            "attention_backend": self.attention_backend,
            "expert_attention_backend": self.expert_attention_backend,
            "flow_backend": self.flow_backend,
            "compile_backend": self.compile_backend,
            "flow_depth": int(flow_depth),
            "flow_attention_mode": flow_attention_mode,
            "flow_sdpa_backend": flow_sdpa_backend,
            "packed_group_cells": self.packed_group_cells,
            "packed_attention_budget": self.packed_attention_budget,
            "packed_attention_backend": self.packed_attention_backend,
            "expert_execution": expert_execution,
            "attention_projection_execution": attention_projection_execution,
            "fuse_development_attention": self.fuse_development_attention,
            "fuse_development_target_attention": (
                self.fuse_development_target_attention
            ),
            "fuse_nbeats_heads": self.fuse_nbeats_heads,
            "residual_norm_execution": self.residual_norm_execution,
            "residual_norm_min_tokens": self.residual_norm_min_tokens,
            "feature_map_execution": self.feature_map_execution,
            "feature_map_min_tokens": self.feature_map_min_tokens,
            "feature_map_compile_mode": self.feature_map_compile_mode,
            "feature_map_bucket_size": self.feature_map_bucket_size,
            "feature_map_max_padding_ratio": self.feature_map_max_padding_ratio,
            "differential_merge_execution": self.differential_merge_execution,
            "differential_merge_min_tokens": self.differential_merge_min_tokens,
            "differential_merge_compile_mode": self.differential_merge_compile_mode,
            "router_execution": self.router_execution,
            "router_min_tokens": self.router_min_tokens,
            "router_compile_mode": self.router_compile_mode,
            "router_compile_dynamic": self.router_compile_dynamic,
            "population_balance_mode": self.population_balance_mode,
            "sinkhorn_epsilon": float(sinkhorn_epsilon),
            "sinkhorn_iterations": int(sinkhorn_iterations),
            "sinkhorn_compile": bool(sinkhorn_compile),
            "sinkhorn_compile_mode": sinkhorn_compile_mode,
            "sinkhorn_compile_buckets": list(
                self._sinkhorn_runtime.buckets
            ),
        }
        self.gene = nn.Embedding(vocab_size, d_model)
        self.expression = ModalityAdapter(d_model)
        categorical_modalities=categorical_modalities or set()
        self.modalities = nn.ModuleDict({name: ModalityAdapter(d_model,dim or 1,name in categorical_modalities)
                                         for name, dim in (modality_dims or {}).items()})
        # Built lazily for one joint forward. The packed tensor is not a
        # parameter, so checkpoint keys and per-modality optimizer state stay
        # unchanged; it is only a differentiable view over the original
        # embedding tables and is cleared after the forward/backward graph.
        self._cached_categorical_pack = None
        block_kwargs = {
            "d_model": d_model,
            "d_hidden": d_hidden,
            "n_experts": n_experts,
            "ffn_expansion": ffn_expansion,
            "top_k": top_k,
            "attention_heads": attention_heads,
            "attention_backend": self.attention_backend,
            "expert_attention_backend": self.expert_attention_backend,
            "expert_execution": expert_execution,
            "attention_projection_execution": attention_projection_execution,
            "fuse_nbeats_heads": self.fuse_nbeats_heads,
            "residual_norm_execution": self.residual_norm_execution,
            "residual_norm_min_tokens": self.residual_norm_min_tokens,
            "feature_map_execution": self.feature_map_execution,
            "feature_map_min_tokens": self.feature_map_min_tokens,
            "feature_map_compile_mode": self.feature_map_compile_mode,
            "feature_map_bucket_size": self.feature_map_bucket_size,
            "feature_map_max_padding_ratio": self.feature_map_max_padding_ratio,
            "differential_merge_execution": self.differential_merge_execution,
            "differential_merge_min_tokens": self.differential_merge_min_tokens,
            "differential_merge_compile_mode": self.differential_merge_compile_mode,
            "router_execution": self.router_execution,
            "router_min_tokens": self.router_min_tokens,
            "router_compile_mode": self.router_compile_mode,
            "router_compile_dynamic": self.router_compile_dynamic,
            "population_balance_mode": self.population_balance_mode,
        }
        # Every registered MoE factory receives the same named contract. A
        # backend can consume the fields it supports and keep its own
        # parameter layout; the core model never branches on implementation.
        self.blocks = nn.ModuleList(
            create_backend("moe", self.moe_backend, **block_kwargs)
            for _ in range(n_blocks)
        )
        for index, block in enumerate(self.blocks):
            block.profile_prefix=f"brainwm.backbone/block_{index:02d}"
        self.norm = nn.LayerNorm(d_model)
        self.expression_head = nn.Linear(d_model, 1)
        self.flow_transformer = create_backend(
            "flow", self.flow_backend,
            width=d_model,
            heads=attention_heads,
            depth=flow_depth,
            expansion=ffn_expansion,
            attention_mode=flow_attention_mode,
            sdpa_backend=flow_sdpa_backend,
        )
        self.capabilities = None
        self.spatial_attention=SpatialCellAttention(d_model)
        self.canonical_annotation_head = nn.Linear(d_model,annotation_classes) if annotation_classes else None
        self.anatomy_head = nn.Linear(d_model,anatomy_classes) if anatomy_classes else None

    @contextmanager
    def _prepared_backbone_weights(self):
        # Preparation is an optional execution hook.  Discover it by protocol
        # instead of by concrete class so a registered MoE/attention backend can
        # reuse the same lifetime contract without changing this model.
        cleanup = []
        try:
            self._prepare_categorical_pack()
            for block in self.blocks:
                attention = getattr(block, "attention", None)
                prepare = getattr(attention, "prepare_projection", None)
                clear = getattr(attention, "clear_prepared_projection", None)
                if callable(prepare):
                    prepare()
                    cleanup.append(clear)
                experts = getattr(getattr(block, "moe", None), "deepspeed_moe", None)
                experts = getattr(experts, "experts", None)
                prepare = getattr(experts, "prepare_weights", None)
                clear = getattr(experts, "clear_prepared_weights", None)
                if callable(prepare):
                    prepare()
                    cleanup.append(clear)
                prepare = getattr(block, "prepare_nbeats_heads", None)
                clear = getattr(block, "clear_nbeats_heads", None)
                if callable(prepare):
                    prepare()
                    cleanup.append(clear)
            yield
        finally:
            for clear in reversed(cleanup):
                if callable(clear):
                    clear()
            self._clear_categorical_pack()

    def _prepare_categorical_pack(self) -> None:
        """Pack categorical tables for one joint forward with one embedding call."""
        names = tuple(
            name for name, adapter in self.modalities.items()
            if adapter.category is not None
        )
        if not names:
            self._cached_categorical_pack = None
            return
        offsets=[]
        total=0
        for name in names:
            offsets.append(total)
            total += self.modalities[name].category.num_embeddings
        weights=torch.cat(
            [self.modalities[name].category.weight for name in names], dim=0
        )
        null=torch.stack([self.modalities[name].null for name in names])
        mask=torch.stack([self.modalities[name].mask for name in names])
        offset_tensor=torch.as_tensor(offsets,device=weights.device,dtype=torch.long)
        self._cached_categorical_pack=(names,offset_tensor,weights,null,mask)

    def _clear_categorical_pack(self) -> None:
        self._cached_categorical_pack = None

    def _modality_context(self, batch_size: int, device: torch.device,
                          modality: Mapping[str, Tensor] | None,
                          observed_map: Mapping[str, Tensor] | None,
                          masked_map: Mapping[str, Tensor] | None) -> Tensor | None:
        """Encode metadata while keeping direct calls free of stale caches."""
        owns_pack=self._cached_categorical_pack is None
        if owns_pack:
            self._prepare_categorical_pack()
        try:
            return self._modality_context_impl(
                batch_size,device,modality,observed_map,masked_map,
            )
        finally:
            if owns_pack:
                self._clear_categorical_pack()

    def _apply_activation_dtype(self, value: Tensor) -> Tensor:
        """Keep CUDA activations in the requested autocast dtype.

        ``nn.Embedding`` and a few metadata additions are intentionally kept
        in their parameter dtype by PyTorch.  Under BF16 autocast that can
        promote the whole residual stream to FP32 before the first block.  The
        opt-in mixed-precision backends establish one BF16 state boundary;
        legacy ``torch`` execution remains byte-for-byte unchanged.
        """
        token_count = int(value.numel() // value.shape[-1]) if value.ndim else 0
        selected = self.residual_norm_execution
        if selected == "auto":
            selected = "fla" if token_count >= self.residual_norm_min_tokens else "torch"
        if (selected not in {"torch_bf16", "fla"}
                or not value.is_cuda
                or not value.is_floating_point()
                or not torch.is_autocast_enabled("cuda")):
            return value
        target = torch.get_autocast_dtype("cuda")
        return value if value.dtype == target else value.to(dtype=target)

    def _modality_context_impl(self, batch_size: int, device: torch.device,
                               modality: Mapping[str, Tensor] | None,
                               observed_map: Mapping[str, Tensor] | None,
                               masked_map: Mapping[str, Tensor] | None) -> Tensor | None:
        """Encode all metadata once, batching categorical lookups by offset."""
        modality=modality or {}
        observed_map=observed_map or {}
        masked_map=masked_map or {}
        context=None
        pack=self._cached_categorical_pack
        categorical_names=() if pack is None else pack[0]
        if pack is not None:
            names,offsets,weights,null,mask_embedding=pack
            ids=[];observed=[];masked=[]
            for index,name in enumerate(names):
                value=modality.get(name)
                seen=observed_map.get(name)
                if seen is None:
                    if value is not None:
                        raise ValueError(f'{name} requires an explicit observation mask')
                    seen=torch.zeros(batch_size,dtype=torch.bool,device=device)
                else:
                    seen=seen.to(device=device,dtype=torch.bool).reshape(batch_size)
                if value is None:
                    encoded_ids=torch.zeros(batch_size,dtype=torch.long,device=device)
                else:
                    raw=value.to(device=device)
                    encoded_ids=(raw.squeeze(-1).long() if raw.ndim>1 else raw.long())
                    if encoded_ids.shape != (batch_size,):
                        raise ValueError(f'{name} values must have shape [cells] or [cells, 1]')
                    encoded_ids=encoded_ids.clamp(
                        0,self.modalities[name].category.num_embeddings-1,
                    )
                ids.append(encoded_ids+offsets[index])
                observed.append(seen)
                masked_value=masked_map.get(name)
                if masked_value is None:
                    masked.append(torch.zeros_like(seen))
                else:
                    masked.append(masked_value.to(device=device,dtype=torch.bool).reshape(batch_size))
            with profile_range("brainwm.modality/categorical_packed"):
                flat_ids=torch.stack(ids,dim=0).reshape(-1)
                encoded=nn.functional.embedding(flat_ids,weights).view(
                    len(names),batch_size,self.d_model,
                )
                seen=torch.stack(observed,dim=0)
                add=torch.where(seen.unsqueeze(-1),encoded,null[:,None,:])
                effective=torch.stack(masked,dim=0) & seen
                add=torch.where(effective.unsqueeze(-1),mask_embedding[:,None,:],add)
                context=add.sum(dim=0)
        for name,adapter in self.modalities.items():
            if name in categorical_names:
                continue
            value=modality.get(name)
            seen=observed_map.get(name)
            masked_value=masked_map.get(name)
            if seen is None:
                if value is not None:
                    raise ValueError(f'{name} requires an explicit observation mask')
                seen=torch.zeros(batch_size,dtype=torch.bool,device=device)
            with profile_range(f"brainwm.modality/{name}"):
                add=adapter(value,seen.to(device),None if masked_value is None else masked_value.to(device))
            if add.shape!=(batch_size,self.d_model):
                raise ValueError('sparse modality context must have shape [cells, d_model]')
            context=add if context is None else context+add
        return context

    def set_expert_execution(self, execution: str) -> None:
        setters = [getattr(block, "set_expert_execution", None) for block in self.blocks]
        if not setters or not all(callable(setter) for setter in setters):
            raise ValueError(
                f"MoE backend {self.moe_backend!r} does not expose expert execution"
            )
        for block in self.blocks:
            block.set_expert_execution(execution)
        self.model_config["expert_execution"] = execution

    def set_router_execution(
        self, execution: str, *, min_tokens: int | None = None,
        mode: str | None = None, dynamic: bool | None = None,
    ) -> None:
        """Select the optional fixed-shape gate projection accelerator."""
        if execution not in _ROUTER_EXECUTIONS:
            raise ValueError("router_execution must be eager, compile, or auto")
        if min_tokens is not None and min_tokens < 1:
            raise ValueError("router_min_tokens must be positive")
        if mode is not None and mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown router compile mode {mode!r}")
        self.router_execution = execution
        if min_tokens is not None:
            self.router_min_tokens = int(min_tokens)
        if mode is not None:
            self.router_compile_mode = mode
        if dynamic is not None:
            self.router_compile_dynamic = bool(dynamic)
        self.model_config["router_execution"] = self.router_execution
        self.model_config["router_min_tokens"] = self.router_min_tokens
        self.model_config["router_compile_mode"] = self.router_compile_mode
        self.model_config["router_compile_dynamic"] = self.router_compile_dynamic
        for block in self.blocks:
            setter = getattr(block, "set_router_execution", None)
            if callable(setter):
                setter(
                    execution,
                    min_tokens=self.router_min_tokens,
                    mode=self.router_compile_mode,
                    dynamic=self.router_compile_dynamic,
                )

    def router_runtime_status(self) -> dict[str, object]:
        """Return execution-only router telemetry without touching parameters."""
        backends = [
            getattr(block, "_last_router_backend", "unsupported")
            for block in self.blocks
        ]
        return {
            "execution": self.router_execution,
            "min_tokens": self.router_min_tokens,
            "compile_mode": self.router_compile_mode,
            "dynamic": self.router_compile_dynamic,
            "backends": backends,
        }

    def set_attention_projection_execution(self, execution: str) -> None:
        setters = [
            getattr(getattr(block, "attention", None), "set_projection_execution", None)
            for block in self.blocks
        ]
        if not setters or not all(callable(setter) for setter in setters):
            raise ValueError(
                f"MoE backend {self.moe_backend!r} does not expose attention projections"
            )
        for setter in setters:
            setter(execution)
        self.model_config["attention_projection_execution"] = execution

    def set_feature_map_execution(self, execution: str) -> None:
        """Select the Flash-BLA feature-map execution backend."""
        if execution not in _FEATURE_MAP_EXECUTIONS:
            raise ValueError(
                "feature_map_execution must be torch, compile, bucket, or auto"
            )
        self.feature_map_execution = execution
        self.model_config["feature_map_execution"] = execution
        for block in self.blocks:
            if hasattr(block, "set_feature_map_execution"):
                block.set_feature_map_execution(execution)

    def set_feature_map_min_tokens(self, minimum: int) -> None:
        if minimum < 1:
            raise ValueError("feature_map_min_tokens must be positive")
        self.feature_map_min_tokens = int(minimum)
        self.model_config["feature_map_min_tokens"] = self.feature_map_min_tokens
        for block in self.blocks:
            if hasattr(block, "set_feature_map_min_tokens"):
                block.set_feature_map_min_tokens(minimum)

    def set_feature_map_compile_mode(self, mode: str) -> None:
        if mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown feature-map compile mode {mode!r}")
        self.feature_map_compile_mode = mode
        self.model_config["feature_map_compile_mode"] = mode
        for block in self.blocks:
            if hasattr(block, "set_feature_map_compile_mode"):
                block.set_feature_map_compile_mode(mode)

    def set_feature_map_bucket_size(self, bucket_size: int) -> None:
        if bucket_size < 1:
            raise ValueError("feature_map_bucket_size must be positive")
        self.feature_map_bucket_size = int(bucket_size)
        self.model_config["feature_map_bucket_size"] = self.feature_map_bucket_size
        for block in self.blocks:
            if hasattr(block, "set_feature_map_bucket_size"):
                block.set_feature_map_bucket_size(bucket_size)

    def set_feature_map_max_padding_ratio(self, ratio: float) -> None:
        if ratio < 1.0:
            raise ValueError("feature_map_max_padding_ratio must be at least 1")
        self.feature_map_max_padding_ratio = float(ratio)
        self.model_config["feature_map_max_padding_ratio"] = (
            self.feature_map_max_padding_ratio
        )
        for block in self.blocks:
            if hasattr(block, "set_feature_map_max_padding_ratio"):
                block.set_feature_map_max_padding_ratio(ratio)

    def set_differential_merge_execution(self, execution: str) -> None:
        """Select the tensor-only differential attention merge backend."""
        if execution not in _DIFFERENTIAL_MERGE_EXECUTIONS:
            raise ValueError(
                "differential_merge_execution must be torch, compile, or auto"
            )
        self.differential_merge_execution = execution
        self.model_config["differential_merge_execution"] = execution
        for block in self.blocks:
            setter = getattr(block, "set_differential_merge_execution", None)
            if callable(setter):
                setter(execution)

    def set_differential_merge_min_tokens(self, minimum: int) -> None:
        if minimum < 1:
            raise ValueError("differential_merge_min_tokens must be positive")
        self.differential_merge_min_tokens = int(minimum)
        self.model_config["differential_merge_min_tokens"] = int(minimum)
        for block in self.blocks:
            setter = getattr(block, "set_differential_merge_min_tokens", None)
            if callable(setter):
                setter(minimum)

    def set_differential_merge_compile_mode(self, mode: str) -> None:
        if mode not in _TORCH_COMPILE_MODES:
            raise ValueError(f"unknown differential-merge compile mode {mode!r}")
        self.differential_merge_compile_mode = mode
        self.model_config["differential_merge_compile_mode"] = mode
        for block in self.blocks:
            setter = getattr(block, "set_differential_merge_compile_mode", None)
            if callable(setter):
                setter(mode)

    def set_population_balance_mode(self, mode: str) -> None:
        if mode not in {"exact", "combined"}:
            raise ValueError("population_balance_mode must be exact or combined")
        self.population_balance_mode = mode
        self.model_config["population_balance_mode"] = mode
        for block in self.blocks:
            if hasattr(block, "set_population_balance_mode"):
                block.set_population_balance_mode(mode)

    def set_residual_norm_execution(self, execution: str) -> None:
        """Select the execution-only residual LayerNorm backend."""
        if execution not in _RESIDUAL_NORM_EXECUTIONS:
            raise ValueError(
                "residual_norm_execution must be torch, torch_bf16, fla, or auto"
            )
        self.residual_norm_execution = execution
        self.model_config["residual_norm_execution"] = execution
        for block in self.blocks:
            if hasattr(block, "set_residual_norm_execution"):
                block.set_residual_norm_execution(execution)

    def set_residual_norm_min_tokens(self, minimum: int) -> None:
        """Set the token threshold used by the automatic FLA selector."""
        if minimum < 1:
            raise ValueError("residual_norm_min_tokens must be positive")
        self.residual_norm_min_tokens = int(minimum)
        self.model_config["residual_norm_min_tokens"] = self.residual_norm_min_tokens
        for block in self.blocks:
            if hasattr(block, "set_residual_norm_min_tokens"):
                block.set_residual_norm_min_tokens(minimum)

    def set_flow_sdpa_backend(self,backend: str) -> None:
        setter = getattr(self.flow_transformer, "set_sdpa_backend", None)
        if not callable(setter):
            raise ValueError(
                f"flow backend {self.flow_backend!r} does not expose SDPA selection"
            )
        setter(backend)
        self.model_config["flow_sdpa_backend"]=backend

    def set_compile_backend(self, backend: str) -> None:
        """Select the registered compiler used by optional flow compilation.

        This is an execution policy and does not alter parameters.  A caller
        that has already compiled the flow path must call ``set_flow_compile``
        again so the cached callable is rebuilt under the new policy.
        """
        self.compile_backend = backend_spec("compile", backend).name
        self.model_config["compile_backend"] = self.compile_backend
        object.__setattr__(self, "_compiled_flow_forward", None)

    def backend_manifest(self) -> dict[str, object]:
        """Return the concrete backend choices and capability declarations."""
        return {
            "attention": backend_spec(
                "attention", self.attention_backend
            ).manifest(),
            "expert_attention": backend_spec(
                "attention", self.expert_attention_backend
            ).manifest(),
            "moe": backend_spec("moe", self.moe_backend).manifest(),
            "flow": backend_spec("flow", self.flow_backend).manifest(),
            "compile": backend_spec("compile", self.compile_backend).manifest(),
        }

    def set_sinkhorn_compile(
        self,
        enabled: bool,
        *,
        mode: str | None = None,
        buckets: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Toggle fixed-bucket TorchInductor acceleration for developmental OT."""
        current = self._sinkhorn_runtime
        selected_mode = current.mode if mode is None else mode
        selected_buckets = current.buckets if buckets is None else tuple(buckets)
        self._sinkhorn_runtime = SinkhornRuntime(
            epsilon=current.epsilon,
            iterations=current.iterations,
            enabled=bool(enabled),
            mode=selected_mode,
            buckets=selected_buckets,
            max_cached_graphs=current.max_cached_graphs,
        )
        self.model_config["sinkhorn_compile"] = bool(enabled)
        self.model_config["sinkhorn_compile_mode"] = selected_mode
        self.model_config["sinkhorn_compile_buckets"] = list(
            self._sinkhorn_runtime.buckets
        )

    def sinkhorn_runtime_status(self) -> dict[str, object]:
        """Expose bounded OT compiler telemetry for logs and A/B reports."""
        return self._sinkhorn_runtime.status()

    def _sinkhorn_plan(self, source: Tensor, target: Tensor) -> Tensor:
        runtime = self._sinkhorn_runtime
        if runtime is None:
            return sinkhorn_transport_plan(
                source, target, epsilon=0.05, iterations=50,
            )
        return runtime.plan(source, target)

    def set_flow_compile(
        self, enabled: bool, *, mode: str = "reduce-overhead",
        dynamic: bool = True,
    ) -> None:
        """Compile only the stable FMT call without changing model weights.

        The sparse backbone contains dynamic CSR layouts and a custom Triton
        operator, so compiling the whole model is prone to graph breaks.  The
        population Flow Matching Transformer has a fixed module schedule and
        is a good boundary for PyTorch 2.x compilation.  The compiled bound
        function is deliberately kept outside ``nn.Module`` registration;
        checkpoint keys and optimizer state therefore remain identical.
        """
        valid_modes = {
            "default", "reduce-overhead", "max-autotune",
            "max-autotune-no-cudagraphs",
        }
        if mode not in valid_modes:
            raise ValueError(
                f"unknown torch.compile mode {mode!r}; expected one of "
                f"{sorted(valid_modes)}"
            )
        # Bypass Module.__setattr__: an OptimizedModule would otherwise become
        # a registered child and alter the state-dict namespace.
        object.__setattr__(self, "_compiled_flow_forward", None)
        object.__setattr__(
            self, "_flow_compile_config",
            {"enabled": bool(enabled), "mode": mode, "dynamic": bool(dynamic)},
        )
        if not enabled:
            return
        if self.compile_backend != "eager" and getattr(torch, "compile", None) is None:
            raise RuntimeError(
                f"compile backend {self.compile_backend!r} requires PyTorch 2.x"
            )
        compiled = compile_callable(
            self.compile_backend,
            self.flow_transformer.forward,
            mode=mode,
            dynamic=bool(dynamic),
            fullgraph=False,
        )
        object.__setattr__(self, "_compiled_flow_forward", compiled)

    @property
    def flow_compile_enabled(self) -> bool:
        return bool(
            (self.__dict__.get("_flow_compile_config") or {}).get("enabled", False)
        )

    def set_fuse_development_attention(self, enabled: bool) -> None:
        """Set the main/source joint-attention execution policy."""
        self.fuse_development_attention = bool(enabled)

    def set_fuse_development_target_attention(self, enabled: bool) -> None:
        """Set whether the inference-only target joins the Flash-BLA call."""
        self.fuse_development_target_attention = bool(enabled)

    def set_fuse_nbeats_heads(self, enabled: bool) -> None:
        """Select the one-GEMM N-BEATS head schedule without changing weights."""
        self.fuse_nbeats_heads = bool(enabled)
        self.model_config["fuse_nbeats_heads"] = self.fuse_nbeats_heads
        for block in self.blocks:
            setter = getattr(block, "set_fuse_nbeats_heads", None)
            if setter is not None:
                setter(self.fuse_nbeats_heads)
            elif hasattr(block, "fuse_nbeats_heads"):
                block.fuse_nbeats_heads = self.fuse_nbeats_heads

    @classmethod
    def from_checkpoint(cls, checkpoint, *, map_location="cpu", strict=True):
        """Reconstruct a model from a self-describing training checkpoint."""
        payload = torch.load(checkpoint, map_location=map_location)
        config = payload.get("model_config")
        if config is None:
            raise ValueError("checkpoint has no model_config; supply the original architecture manually")
        config = dict(config)
        config["categorical_modalities"] = set(config.get("categorical_modalities", ()))
        model = cls(**config)
        model.load_state_dict(payload["model"], strict=strict)
        return model, payload

    def attach_capabilities(self, expression_dim: int, annotation_dims=(3, 24, 137)) -> None:
        from .unified_capabilities import UnifiedCapabilityHeads
        self.capabilities = UnifiedCapabilityHeads(self.d_model, expression_dim, annotation_dims)

    def add_modality(self, name: str, value_dim: int) -> None:
        if name in self.modalities:
            raise ValueError(f"modality already exists: {name}")
        self.modalities[name] = ModalityAdapter(self.d_model, value_dim)
        self._clear_categorical_pack()

    def _can_fuse_development_attention(
        self,main:SparseTokenBatch,source:SparseTokenBatch,
    )->bool:
        return bool(
            self.fuse_development_attention
            and all(
                hasattr(block, "attention")
                and callable(getattr(block, "finish_varlen", None))
                and callable(getattr(block.attention, "forward_varlen", None))
                for block in self.blocks
            )
            and main.packed and source.packed
            and main.values.is_cuda and source.values.is_cuda
            and main.values.device==source.values.device
            and self.packed_attention_backend in {"auto","flash_bla"}
        )

    def _can_fuse_development_triple_attention(
        self, main: SparseTokenBatch, source: SparseTokenBatch,
        target: SparseTokenBatch,
    ) -> bool:
        """Return whether main/source/target can share one varlen call."""
        return bool(
            self.fuse_development_target_attention
            and self._can_fuse_development_attention(main, source)
            and target.packed
            and target.values.is_cuda
            and target.values.device == main.values.device
        )

    def forward(self, batch: BrainWMInput) -> dict[str, Tensor]:
        if isinstance(batch,dict) and 'joint_batch' in batch:
            # The same parameters serve the main, source, and target encoders.
            # Stack per-expert tensors once for all three invocations, then drop
            # the temporary tensors before backward; autograd retains only the
            # references required by the active graph.
            with self._prepared_backbone_weights():
                development=batch.get('development')
                source_output=None
                target_output=None
                if (development is not None and
                        self._can_fuse_development_triple_attention(
                            batch['joint_batch'],development[0],development[1],
                        )):
                    with profile_range("brainwm.forward/joint_gradient_encoders"):
                        output,source_output,target_output=(
                            self._forward_sparse_flash_triple(
                                batch['joint_batch'],development[0],development[1],
                            )
                        )
                elif (development is not None and
                      self._can_fuse_development_attention(
                          batch['joint_batch'],development[0],
                      )):
                    with profile_range("brainwm.forward/joint_gradient_encoders"):
                        output,source_output=self._forward_sparse_flash_pair(
                            batch['joint_batch'],development[0],
                        )
                else:
                    with profile_range("brainwm.forward/joint_expression"):
                        output=self.forward_sparse(batch['joint_batch'])
                if development is not None:
                    source,target,t0,t1=development
                    if source_output is None:
                        with profile_range("brainwm.forward/development_source_encoder"):
                            source_output=self.forward_sparse(source, encode_only=True)
                    if target_output is None:
                        with profile_range("brainwm.forward/development_target_encoder"):
                            # The target branch is detached by construction. Inference
                            # mode also removes autograd version-counter metadata,
                            # reducing CPU launch overhead for the repeated backbone.
                            with torch.inference_mode():
                                target_output=self.forward_sparse(target, encode_only=True)
                                target_state=target_output['cell_embedding'].detach()
                    else:
                        target_state=target_output['cell_embedding'].detach()
                    # Development execution fields are telemetry, not model
                    # outputs.  Keep them explicit for every fallback path.
                    output.setdefault("development_attention_fused", False)
                    output.setdefault("development_attention_streams", 1)
                    output.setdefault("development_target_attention_fused", False)
                    for name in (
                        'attention_sequence_count',
                        'attention_max_seqlen',
                        'attention_max_seqlen_bucket',
                    ):
                        output[f'development_source_{name}']=source_output[name]
                        output[f'development_target_{name}']=target_output[name]
                    source_state=source_output['cell_embedding']
                    from .losses import sinkhorn_transport_plan,barycentric_transport_target
                    with profile_range("brainwm.forward/development_sinkhorn"):
                        with torch.no_grad():
                            plan=self._sinkhorn_plan(
                                source_state.detach(),target_state,
                            )
                            endpoint=barycentric_transport_target(plan,target_state)
                    with profile_range("brainwm.forward/development_interpolation"):
                        with torch.no_grad():
                            tau=(torch.rand((len(source_state),1),device=source_state.device)
                                 if self.training else torch.full((len(source_state),1),.5,device=source_state.device))
                            state=(1-tau)*source_state.detach()+tau*endpoint
                        source_time=torch.full((len(source_state),1),float(t0),device=source_state.device)
                        target_time=torch.full_like(source_time,float(t1))
                    with profile_range("brainwm.forward/development_fmt"):
                        output['flow_velocity']=self.velocity_at(
                            source_state,state,tau,source_time,target_time
                        )
                    with profile_range("brainwm.forward/development_targets"):
                        output['flow_target']=endpoint-source_state.detach()
                        output['flow_endpoint']=state+(1-tau)*output['flow_velocity']
                        output['flow_endpoint_target']=endpoint
                        output['developmental_observed']=torch.ones(
                            len(source_state),dtype=torch.bool,device=source_state.device
                        )
                return output
        if isinstance(batch,dict) and 'flow_context' in batch:
            with profile_range("brainwm.forward/flow_only"):
                return {'flow_velocity':self.velocity_at(batch['flow_context'],batch['flow_state'],batch['flow_tau'],batch['source_time'],batch['target_time'])}
        if hasattr(batch,'gene_index') and hasattr(batch,'row_ptr'):
            with profile_range("brainwm.forward/sparse"):
                return self.forward_sparse(batch)
        if batch.expression.ndim != 2:
            raise ValueError("expression must have shape (batch, genes)")
        b, g = batch.expression.shape
        ids = torch.arange(g, device=batch.expression.device).expand(b, -1)
        with profile_range("brainwm.dense/token_embedding"):
            x = self.gene(ids) + self.expression(
                batch.expression.unsqueeze(-1),batch.expression_observed
            )
        modality_context=self._modality_context(
            b,x.device,batch.modality,batch.modality_observed,batch.modality_masked,
        )
        if modality_context is not None:
            x=x+modality_context.unsqueeze(1)
        x = self._apply_activation_dtype(x)
        if batch.spatial_relation is not None:
            raise ValueError('use bounded cell neighbor_indices; dense gene-axis relations are unsupported')
        forecasts = []
        for block in self.blocks:
            with profile_range(block.profile_prefix):
                x, forecast = block(x)
            forecasts.append(forecast)
        with profile_range("brainwm.dense/final_norm_pool"):
            x = self.norm(x)
            cell = x.mean(1)
        with profile_range("brainwm.dense/spatial_attention"):
            spatial_cell=self.spatial_attention(
                cell,batch.neighbor_indices,batch.neighbor_mask
            )
        with profile_range("brainwm.dense/spatial_broadcast"):
            x=x+(spatial_cell-cell).unsqueeze(1)
            cell=spatial_cell
        with profile_range("brainwm.dense/development"):
            flow_pred,flow_loss_target,flow_endpoint,flow_endpoint_target = \
                self.development_outputs(cell,batch)
        with profile_range("brainwm.dense/expression_reconstruction_head"):
            expression_reconstruction=self.expression_head(x).squeeze(-1)
        with profile_range("brainwm.dense/forecast_reduce"):
            forecast=torch.stack(forecasts).sum(0)
        output = {"tokens": x, "cell_embedding": cell,
                "expression_reconstruction": expression_reconstruction,
                "forecast": forecast,
                "flow_velocity": flow_pred,
                "flow_target": flow_loss_target,
                "flow_endpoint": flow_endpoint,
                "flow_endpoint_target": flow_endpoint_target}
        if self.canonical_annotation_head is not None:
            with profile_range("brainwm.head/cell_annotation"):
                output['canonical_annotation_logits']=self.canonical_annotation_head(cell)
        if self.anatomy_head is not None:
            with profile_range("brainwm.head/anatomy"):
                output['anatomy_logits']=self.anatomy_head(cell)
        with profile_range("brainwm.head/expert_balance"):
            output['expert_balance_loss']=torch.stack(
                [block.routing_balance_loss for block in self.blocks]
            ).mean()
        if self.capabilities is not None:
            with profile_range("brainwm.head/legacy_capabilities"):
                cap = self.capabilities(cell, batch.expression,
                    flow_time=batch.source_time, source_time=batch.source_time,
                    target_time=batch.target_time,
                    developmental_observed=batch.developmental_observed,
                    completion_observed=(batch.expression_observed.any(1)
                                         if batch.expression_observed is not None else None))
                output.update({"annotation_logits": cap.annotation_logits,
                               "capability_endpoint": cap.endpoint,
                               "capability_velocity": cap.velocity,
                               "completed_expression": cap.completed_expression})
        return output

    def velocity_at(self, context: Tensor, state: Tensor, tau: Tensor,
                    source_time: Tensor, target_time: Tensor) -> Tensor:
        """Conditional normalized-time velocity; inference needs no target state."""
        times=torch.cat([tau,source_time/365.25,(target_time-source_time)/365.25],dim=-1)
        compiled = self.__dict__.get("_compiled_flow_forward")
        if compiled is None:
            return self.flow_transformer(context, state, times)
        try:
            return compiled(context, state, times)
        except Exception as exc:
            # Dynamic population sizes and optional third-party kernels can
            # invalidate a compiled graph.  Disable the optional accelerator
            # once and continue with the numerically identical eager path;
            # strict CI can request a hard failure through the environment.
            import os
            if os.environ.get("BRAINWM_COMPILE_STRICT", "0") == "1":
                raise
            warnings.warn(
                f"torch.compile FMT path disabled after {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "_compiled_flow_forward", None)
            return self.flow_transformer(context, state, times)

    def predict_development(
        self,
        context: Tensor,
        source_state: Tensor,
        source_time: Tensor | float,
        target_time: Tensor | float,
        *,
        steps: int = 10,
        solver: str = "heun",
        developmental_observed: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Integrate the learned normalized-time developmental velocity field.

        Rows without developmental information, and rows whose source and
        target ages are equal, remain exact identities. Times are expressed in
        days; the integration variable itself is normalized to [0, 1].
        """
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        if context.ndim != 2 or source_state.shape != context.shape:
            raise ValueError("context and source_state must have shape [cells, d_model]")
        if context.shape[1] != self.d_model:
            raise ValueError(f"development state width must be {self.d_model}")
        if not torch.isfinite(context).all() or not torch.isfinite(source_state).all():
            raise ValueError("development context and source state must be finite")

        cells = context.shape[0]
        def expand_time(value: Tensor | float, name: str) -> Tensor:
            out = torch.as_tensor(value, device=context.device, dtype=context.dtype)
            if out.ndim == 0:
                out = out.expand(cells)
            elif out.shape in {(cells,), (cells, 1)}:
                out = out.reshape(cells)
            else:
                raise ValueError(f"{name} must be scalar, [cells], or [cells, 1]")
            return out

        t0 = expand_time(source_time, "source_time")
        t1 = expand_time(target_time, "target_time")
        if developmental_observed is None:
            observed = torch.isfinite(t0) & torch.isfinite(t1)
        else:
            observed = torch.as_tensor(developmental_observed, device=context.device)
            if observed.shape not in {(cells,), (cells, 1)}:
                raise ValueError("developmental_observed must have shape [cells]")
            observed = observed.reshape(cells).bool()
            observed = observed & torch.isfinite(t0) & torch.isfinite(t1)
        if ((t1 < t0) & observed).any():
            raise ValueError("observed developmental prediction requires target_time >= source_time")

        active = observed & (t1 > t0)
        safe_t0 = torch.where(active, t0, torch.zeros_like(t0)).unsqueeze(-1)
        safe_t1 = torch.where(active, t1, safe_t0.squeeze(-1)).unsqueeze(-1)
        state = source_state
        trajectory = [state]
        step_size = 1.0 / steps
        active_column = active.unsqueeze(-1)
        for index in range(steps):
            tau = state.new_full((cells, 1), index / steps)
            k1 = self.velocity_at(context, state, tau, safe_t0, safe_t1)
            if solver == "euler":
                proposal = state + step_size * k1
            else:
                predictor = state + step_size * k1
                next_tau = state.new_full((cells, 1), (index + 1) / steps)
                k2 = self.velocity_at(context, predictor, next_tau, safe_t0, safe_t1)
                proposal = state + 0.5 * step_size * (k1 + k2)
            state = torch.where(active_column, proposal, source_state)
            if not torch.isfinite(state).all():
                raise FloatingPointError("development rollout produced non-finite states")
            trajectory.append(state)
        if return_trajectory:
            return state, torch.stack(trajectory)
        return state

    def development_outputs(self, cell: Tensor, batch: BrainWMInput):
        prediction,target=torch.zeros_like(cell),torch.zeros_like(cell)
        endpoint,endpoint_target=torch.zeros_like(cell),torch.zeros_like(cell)
        if batch.developmental_observed is None:return prediction,target,endpoint,endpoint_target
        observed=batch.developmental_observed.to(cell.device,dtype=torch.bool)
        if observed.shape!=(len(cell),):raise ValueError('development mask must be [cells]')
        if not observed.any():return prediction,target,endpoint,endpoint_target
        if any(v is None for v in [batch.source_state,batch.target_state,batch.source_time,batch.target_time]):
            raise ValueError('observed development requires population endpoints and ages')
        if batch.source_state.shape!=cell.shape or batch.target_state.shape!=cell.shape:
            raise ValueError('development endpoint states must be [cells, d_model]')
        # Select the active rows once and scatter the four outputs back with
        # index_copy. Boolean assignment creates CopySlices/index_put nodes in
        # autograd for every output, which is costly on sparse development
        # batches and obscures the actual Flow-Matching compute.
        indices=observed.nonzero(as_tuple=True)[0]
        x0=batch.source_state.to(cell).index_select(0,indices)
        x1=batch.target_state.to(cell).index_select(0,indices)
        t0=batch.source_time.to(cell).reshape(len(cell),1).index_select(0,indices)
        t1=batch.target_time.to(cell).reshape(len(cell),1).index_select(0,indices)
        if not all(torch.isfinite(v).all() for v in [x0,x1,t0,t1]) or not (t1>t0).all():
            raise ValueError('development endpoints and positive age intervals must be finite')
        tau=(torch.rand_like(t0) if self.training else torch.full_like(t0,.5)) if batch.flow_tau is None else batch.flow_tau.to(cell).reshape(len(cell),1).index_select(0,indices)
        if not torch.isfinite(tau).all() or not ((tau>=0)&(tau<=1)).all():
            raise ValueError('flow_tau must be finite in [0,1]')
        xt=(1-tau)*x0+tau*x1
        velocity=self.velocity_at(cell.index_select(0,indices),xt,tau,t0,t1)
        prediction=prediction.index_copy(0,indices,velocity)
        target=target.index_copy(0,indices,x1-x0)
        endpoint=endpoint.index_copy(0,indices,xt+(1-tau)*velocity)
        endpoint_target=endpoint_target.index_copy(0,indices,x1)
        return prediction,target,endpoint,endpoint_target

    def forward_sparse(self, batch: SparseTokenBatch, *, encode_only: bool = False) -> dict[str, Tensor]:
        """Run sparse reader tokens without allocating [B, vocab_size]."""
        if batch.packed:
            with profile_range("brainwm.sparse_packed/total"):
                return self._forward_sparse_packed(batch, encode_only=encode_only)
        with profile_range("brainwm.sparse_padded/layout"):
            lengths = batch.row_ptr[1:] - batch.row_ptr[:-1]
            if not torch.all(lengths == lengths[0]):
                raise ValueError('forward_sparse requires equal token counts per batch; pad in the collator')
            l = int(lengths[0]); b = batch.batch_size
            ids = batch.gene_index.view(b, l).long()
            values = batch.values.view(b, l, 1).float()
            token_mask = getattr(batch, 'token_mask', None)
            if token_mask is None:
                token_mask = torch.ones(b*l, dtype=torch.bool, device=ids.device)
            token_mask = token_mask.view(b, l)
            masked=None if batch.expression_masked is None else batch.expression_masked.view(b,l)
        with profile_range("brainwm.sparse_padded/token_embedding"):
            x = self.gene(ids) + self.expression(values, token_mask, masked)
        modality_context=self._modality_context(
            b,x.device,batch.modality,batch.modality_observed,batch.modality_masked,
        )
        # Context is cell-level. Summing before broadcasting avoids one full
        # [cells, genes, width] read/write for every metadata modality.
        if modality_context is not None:
            with profile_range("brainwm.sparse_padded/context_broadcast"):
                x = x + modality_context.unsqueeze(1)
        x = self._apply_activation_dtype(x)
        forecast_sum = None
        for block in self.blocks:
            with profile_range(block.profile_prefix):
                x, forecast = block(
                    x, token_mask, return_forecast=not encode_only,
                )
            if not encode_only:
                with profile_range("brainwm.sparse_padded/forecast_accumulate"):
                    forecast_sum = forecast if forecast_sum is None else forecast_sum + forecast
        with profile_range("brainwm.sparse_padded/final_norm_pool"):
            x = self.norm(x)
            cell = (x * token_mask.unsqueeze(-1)).sum(1) \
                / token_mask.sum(1, keepdim=True).clamp_min(1)
        pre_spatial_cell=cell
        with profile_range("brainwm.sparse_padded/spatial_attention"):
            spatial_cell=self.spatial_attention(
                cell,batch.neighbor_indices,batch.neighbor_mask,
                indices_validated=batch.neighbor_indices_validated,
                edges_present=batch.neighbor_edges_present,
            )
        with profile_range("brainwm.sparse_padded/spatial_broadcast"):
            x=x+(spatial_cell-cell).unsqueeze(1)
            cell=spatial_cell
        if encode_only:
            maximum_sequence_length = l
            maximum_sequence_bucket = 1 << (maximum_sequence_length - 1).bit_length()
            return {
                "tokens": x,
                "cell_embedding": cell,
                "pre_spatial_embedding": pre_spatial_cell,
                "attention_allocated_tokens": b*l,
                "attention_bucket_count": 1,
                # Keep the padded and packed encode-only contracts identical.
                # The joint development path consumes these fields for loader
                # and throughput telemetry regardless of attention backend.
                "attention_backend": "padded",
                "attention_sequence_count": b,
                "attention_max_seqlen": maximum_sequence_length,
                "attention_max_seqlen_bucket": maximum_sequence_bucket,
            }
        with profile_range("brainwm.sparse_padded/development"):
            flow_pred,flow_target,flow_endpoint,flow_endpoint_target=\
                self.development_outputs(cell,batch)
        with profile_range("brainwm.head/expression_reconstruction"):
            expression_reconstruction=self.expression_head(x).squeeze(-1)
        with profile_range("brainwm.head/expression_completion"):
            completed_expression=self.expression_head(forecast_sum).squeeze(-1)
        with profile_range("brainwm.head/expert_balance"):
            expert_balance_loss=torch.stack(
                [block.routing_balance_loss for block in self.blocks]
            ).mean()
        output = {"tokens": x, "cell_embedding": cell, "pre_spatial_embedding": pre_spatial_cell,
                "expression_reconstruction": expression_reconstruction,
                "completed_expression": completed_expression,
                "forecast": forecast_sum,
                "flow_velocity": flow_pred, "flow_target": flow_target,
                "flow_endpoint": flow_endpoint, "flow_endpoint_target": flow_endpoint_target,
                "attention_allocated_tokens": b*l,
                "attention_bucket_count": 1,
                "differential_merge_backend": (
                    getattr(
                        self.blocks[0].attention,
                        "_last_differential_merge_backend",
                        "torch",
                    )
                    if self.blocks and hasattr(self.blocks[0], "attention")
                    else "torch"
                ),
                "expert_balance_loss": expert_balance_loss}
        if self.canonical_annotation_head is not None:
            with profile_range("brainwm.head/cell_annotation"):
                output['canonical_annotation_logits']=self.canonical_annotation_head(cell)
        if self.anatomy_head is not None:
            with profile_range("brainwm.head/anatomy"):
                output['anatomy_logits']=self.anatomy_head(cell)
        return output

    def _prepare_sparse_packed(self,batch:SparseTokenBatch)->_PackedSparseState:
        with profile_range("brainwm.sparse_packed/layout_build"):
            row_ptr=batch.row_ptr.long()
            lengths=batch.row_lengths
            if (lengths is None or lengths.ndim != 1
                    or lengths.numel() != batch.batch_size):
                lengths=row_ptr[1:]-row_ptr[:-1]
            else:
                lengths=lengths.to(device=row_ptr.device,dtype=torch.long)
            token_lengths=batch.token_lengths
            if (token_lengths is None or token_lengths.ndim != 1
                    or token_lengths.numel() != batch.values.numel()):
                token_lengths=lengths[batch.token_row_ids()]
            else:
                token_lengths=token_lengths.to(
                    device=row_ptr.device,dtype=torch.long,
                )
            b=batch.batch_size
            token_mask=batch.token_mask
            if not batch.packed_layout_validated:
                if int(row_ptr[-1])!=batch.values.numel():
                    raise ValueError('packed row_ptr does not cover tokens')
                if token_mask is not None and not bool(token_mask.all()):
                    raise ValueError('packed batches cannot contain padding tokens')
            masked=batch.expression_masked
            attention_backend=self.packed_attention_backend
            supports_varlen = all(
                callable(
                    getattr(getattr(block, "attention", None), "forward_varlen", None)
                )
                for block in self.blocks
            )
            if attention_backend=="auto":
                # A plugin attention backend may provide only the ordinary
                # padded protocol. Let ``auto`` select compatible buckets
                # instead of assuming every CUDA backend implements the
                # Flash-BLA varlen ABI.
                attention_backend = (
                    "flash_bla"
                    if batch.values.is_cuda and supports_varlen
                    else "bucket"
                )
            if attention_backend=="flash_bla" and not batch.values.is_cuda:
                raise RuntimeError('the flash_bla packed attention backend requires CUDA')
            if attention_backend=="flash_bla" and not supports_varlen:
                raise RuntimeError(
                    "flash_bla packed attention requires every selected "
                    "attention backend to expose forward_varlen"
                )
            if batch.packed_min_seqlen is None:
                minimum_sequence_length=int(lengths.min())
            else:
                minimum_sequence_length=batch.packed_min_seqlen
            if attention_backend=="flash_bla" and minimum_sequence_length<=0:
                raise ValueError('flash_bla packed attention requires non-empty cells')
            if batch.packed_max_seqlen is None:
                maximum_sequence_length=int(lengths.max())
            else:
                maximum_sequence_length=batch.packed_max_seqlen
            maximum_sequence_bucket=1 << (maximum_sequence_length-1).bit_length()
            if attention_backend=="bucket":
                layout=DifferentialLinearAttention.build_packed_layout(
                    row_ptr,batch.values.numel(),self.packed_group_cells,
                    self.packed_attention_budget,
                )
                cu_seqlens=None;max_seqlen=None
            else:
                layout=None
                cached_cu_seqlens=batch.cu_seqlens
                if (cached_cu_seqlens is None or cached_cu_seqlens.ndim != 1
                        or cached_cu_seqlens.numel() != row_ptr.numel()):
                    cu_seqlens=row_ptr.to(torch.int32)
                else:
                    cu_seqlens=cached_cu_seqlens.to(
                        device=row_ptr.device,dtype=torch.int32,
                    )
                max_seqlen=maximum_sequence_length
        with profile_range("brainwm.sparse_packed/token_embedding"):
            x=self.gene(batch.gene_index.long())+self.expression(
                batch.values[:,None].float(),
                # Packed CSR contains only real/sampled-measured tokens; all
                # rows are observed. Passing ``None`` avoids allocating a
                # full-length all-ones mask on every step.
                None,masked,
            )
        modality_context=self._modality_context(
            b,x.device,batch.modality,batch.modality_observed,batch.modality_masked,
        )
        with profile_range("brainwm.sparse_packed/context_broadcast"):
            row_ids=batch.token_row_ids()
            if modality_context is not None:
                x=add_csr_rows(x,modality_context,row_ids,row_ptr)
        x = self._apply_activation_dtype(x)
        # Flash-BLA only uses token lengths as a multiplicative feature-map
        # scale; the upstream varlen kernel consumes ``cu_seqlens`` instead.
        # Convert this immutable CSR sidecar once here rather than launching a
        # dtype conversion in every attention block (and for each fused
        # development stream).  Direct ``forward_varlen`` callers may still
        # provide integer lengths and retain the compatibility fallback above.
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            length_dtype = torch.get_autocast_dtype("cuda")
        else:
            length_dtype = x.dtype if x.is_floating_point() else torch.float32
        if token_lengths.dtype != length_dtype:
            token_lengths = token_lengths.to(dtype=length_dtype)
        return _PackedSparseState(
            batch=batch,x=x,row_ptr=row_ptr,lengths=lengths,row_ids=row_ids,
            token_lengths=token_lengths,
            layout=layout,cu_seqlens=cu_seqlens,max_seqlen=max_seqlen,
            attention_backend=attention_backend,
            minimum_sequence_length=minimum_sequence_length,
            maximum_sequence_length=maximum_sequence_length,
            maximum_sequence_bucket=maximum_sequence_bucket,
        )

    def _finish_sparse_packed(
        self,state:_PackedSparseState,x:Tensor,forecast_sum:Tensor | None,
        *,encode_only:bool,balance_losses:list[Tensor] | None=None,
        precomputed_cell:Tensor | None=None,
    )->dict[str,Tensor]:
        batch=state.batch;row_ptr=state.row_ptr;lengths=state.lengths
        row_ids=state.row_ids;layout=state.layout;b=batch.batch_size
        attention_backend=state.attention_backend
        maximum_sequence_length=state.maximum_sequence_length
        maximum_sequence_bucket=state.maximum_sequence_bucket
        if precomputed_cell is None:
            with profile_range("brainwm.sparse_packed/cell_pool"):
                if state.minimum_sequence_length > 0:
                    # Packed batches enforce one or more tokens per row.  The
                    # native mean reduction avoids a second division kernel;
                    # retain the guarded sum path for hand-built empty rows.
                    cell=torch.segment_reduce(x,"mean",offsets=row_ptr)
                else:
                    denominators=lengths.clamp_min(1).to(x.dtype).unsqueeze(-1)
                    cell=torch.segment_reduce(x,"sum",offsets=row_ptr)/denominators
        else:
            if (precomputed_cell.ndim != 2
                    or precomputed_cell.shape[0] != b
                    or precomputed_cell.shape[1] != x.shape[-1]):
                raise ValueError(
                    "precomputed_cell must have shape [batch, hidden]"
                )
            cell=precomputed_cell
        pre_spatial_cell=cell
        with profile_range("brainwm.sparse_packed/spatial_attention"):
            spatial_cell=self.spatial_attention(
                cell,batch.neighbor_indices,batch.neighbor_mask,
                indices_validated=batch.neighbor_indices_validated,
                edges_present=batch.neighbor_edges_present,
            )
        with profile_range("brainwm.sparse_packed/spatial_broadcast"):
            if spatial_cell is not cell:
                x=add_csr_rows(x,spatial_cell-cell,row_ids,row_ptr)
            cell=spatial_cell
        telemetry={
            "attention_allocated_tokens":(
                batch.values.numel() if layout is None else layout.allocated_tokens
            ),
            "attention_bucket_count":0 if layout is None else len(layout.groups),
            "attention_backend":attention_backend,
            "attention_sequence_count":b,
            "attention_max_seqlen":maximum_sequence_length,
            "attention_max_seqlen_bucket":maximum_sequence_bucket,
            "feature_map_backend": (
                getattr(self.blocks[0].attention, "_last_feature_map_backend", "torch")
                if self.blocks and hasattr(self.blocks[0], "attention") else "torch"
            ),
            "differential_merge_backend": (
                getattr(
                    self.blocks[0].attention,
                    "_last_differential_merge_backend",
                    "torch",
                )
                if self.blocks and hasattr(self.blocks[0], "attention") else "torch"
            ),
        }
        if encode_only:
            return {
                "tokens":x,"cell_embedding":cell,
                "pre_spatial_embedding":pre_spatial_cell,**telemetry,
            }
        with profile_range("brainwm.sparse_packed/development"):
            flow_pred,flow_target,flow_endpoint,flow_endpoint_target=\
                self.development_outputs(cell,batch)
        with profile_range("brainwm.head/expression_reconstruction"):
            expression_reconstruction=self.expression_head(x).squeeze(-1)
        with profile_range("brainwm.head/expression_completion"):
            completed_expression=self.expression_head(forecast_sum).squeeze(-1)
        with profile_range("brainwm.head/expert_balance"):
            if balance_losses is None:
                balance_losses=[block.routing_balance_loss for block in self.blocks]
            expert_balance_loss=torch.stack(balance_losses).mean()
            # Packed CSR makes every materialized expression token observed, so
            # the expression null vector has a mathematically zero gradient.
            # Keep that zero dependency explicit for DDP without restoring a
            # token-sized all-ones mask and torch.where allocation.
            expert_balance_loss=(expert_balance_loss+
                                 self.expression.null.sum()*0.0)
        output={"tokens":x,"cell_embedding":cell,
                "pre_spatial_embedding":pre_spatial_cell,
                "expression_reconstruction":expression_reconstruction,
                "completed_expression":completed_expression,
                "forecast":forecast_sum,"flow_velocity":flow_pred,
                "flow_target":flow_target,"flow_endpoint":flow_endpoint,
                "flow_endpoint_target":flow_endpoint_target,
                **telemetry,"expert_balance_loss":expert_balance_loss}
        if self.canonical_annotation_head is not None:
            with profile_range("brainwm.head/cell_annotation"):
                output['canonical_annotation_logits']=self.canonical_annotation_head(cell)
        if self.anatomy_head is not None:
            with profile_range("brainwm.head/anatomy"):
                output['anatomy_logits']=self.anatomy_head(cell)
        return output

    def _forward_sparse_packed(self,batch:SparseTokenBatch, *, encode_only: bool = False)->dict[str,Tensor]:
        state=self._prepare_sparse_packed(batch)
        x=state.x;layout=state.layout;row_ids=state.row_ids
        lengths=state.lengths;token_lengths=state.token_lengths
        cu_seqlens=state.cu_seqlens
        max_seqlen=state.max_seqlen;attention_backend=state.attention_backend
        forecast_sum=None
        if attention_backend=="flash_bla":
            for block in self.blocks:
                if not callable(
                    getattr(getattr(block, "attention", None), "forward_varlen", None)
                ):
                    raise RuntimeError(
                        'flash_bla packed tokens require the selected attention '
                        'backend to expose forward_varlen'
                    )
                with profile_range(block.profile_prefix):
                    x,forecast=block.forward_varlen(
                        x,cu_seqlens,row_ids,token_lengths,max_seqlen,
                        return_forecast=not encode_only,
                    )
                if not encode_only:
                    with profile_range("brainwm.sparse_packed/forecast_accumulate"):
                        forecast_sum=(forecast if forecast_sum is None else
                                      forecast_sum+forecast)
            with profile_range("brainwm.sparse_packed/final_norm"):
                x=self.norm(x)
        else:
            with profile_range("brainwm.sparse_packed/initial_pack"):
                x=layout.pack(x)
            for block in self.blocks:
                with profile_range(block.profile_prefix):
                    packed_forward = getattr(block, "forward_packed", None)
                    if callable(packed_forward):
                        x,forecast=packed_forward(
                            x,layout,return_forecast=not encode_only,
                        )
                    else:
                        # A plugin block only needs the ordinary PyTorch
                        # ``forward(bucket, mask)`` protocol.  Keep the
                        # optimized packed method as the fast path, while
                        # adapting simple Transformer/MoE implementations to
                        # the same length-bucket layout without new kernels.
                        if not callable(getattr(block, "forward", None)):
                            raise RuntimeError(
                                "bucket packed tokens require a block exposing "
                                "forward_packed or forward"
                            )
                        residual_buckets=[]
                        forecast_buckets=[]
                        for bucket,(_, token_mask) in zip(
                            x, layout.groups, strict=True,
                        ):
                            residual, block_forecast = block(
                                bucket,
                                token_mask,
                                return_forecast=not encode_only,
                            )
                            residual_buckets.append(residual)
                            if not encode_only:
                                forecast_buckets.append(block_forecast)
                        x=tuple(residual_buckets)
                        forecast=(
                            tuple(forecast_buckets) if not encode_only else None
                        )
                if not encode_only:
                    with profile_range("brainwm.sparse_packed/forecast_accumulate"):
                        forecast_sum=(forecast if forecast_sum is None else
                                      tuple(total+update for total,update in zip(forecast_sum,forecast)))
            with profile_range("brainwm.sparse_packed/final_norm_unpack"):
                x=layout.unpack(tuple(self.norm(bucket) for bucket in x))
            if forecast_sum is not None:
                with profile_range("brainwm.sparse_packed/forecast_unpack"):
                    forecast_sum=layout.unpack(forecast_sum)
        return self._finish_sparse_packed(
            state,x,forecast_sum,encode_only=encode_only,
        )

    def _forward_sparse_flash_pair(
        self,main_batch:SparseTokenBatch,source_batch:SparseTokenBatch,
    )->tuple[dict[str,Tensor],dict[str,Tensor]]:
        """Fuse main/source attention while preserving independent MoE routing."""
        main=self._prepare_sparse_packed(main_batch)
        source=self._prepare_sparse_packed(source_batch)
        if main.attention_backend!="flash_bla" or source.attention_backend!="flash_bla":
            raise ValueError("joint packed attention fusion requires Flash-BLA")
        if not all(hasattr(block,"finish_varlen") for block in self.blocks):
            raise RuntimeError("joint packed attention fusion requires easy_deepspeed blocks")
        main_tokens=main.x.shape[0]
        joint_cu_seqlens=torch.cat((
            main.cu_seqlens,
            source.cu_seqlens[1:]+main_tokens,
        ))
        joint_token_lengths=torch.cat((
            main.token_lengths,source.token_lengths,
        ))
        main_cells=main.batch.batch_size
        joint_row_ids=torch.cat((
            main.row_ids,source.row_ids+main_cells,
        ))
        joint_max_seqlen=max(main.max_seqlen,source.max_seqlen)
        # Keep one canonical token stream between blocks. The previous path
        # materialized a second copy of the main hidden state in every block;
        # those copies remained live for backward and dominated dense memory.
        joint_x=torch.cat((main.x,source.x),dim=0)
        main.x=joint_x[:main_tokens]
        source.x=joint_x[main_tokens:]
        forecast_sum=None;main_balance=[]
        for block in self.blocks:
            prefix=block.profile_prefix
            with profile_range(prefix):
                with profile_range(f"{prefix}/joint_gradient_attention"):
                    attended=block.attention.forward_varlen(
                        joint_x,joint_cu_seqlens,joint_token_lengths,
                        joint_max_seqlen,
                    )
                with profile_range(f"{prefix}/joint_finish"):
                    joint_x,forecast,balance=block.finish_varlen_pair(
                        joint_x,attended,joint_cu_seqlens,joint_row_ids,
                        main_tokens,main_cells,
                    )
                    main_balance.append(balance)
            with profile_range("brainwm.sparse_packed/forecast_accumulate"):
                forecast_sum=(forecast if forecast_sum is None else
                              forecast_sum+forecast)
        with profile_range("brainwm.sparse_packed/final_norm"):
            joint_x=self.norm(joint_x)
            main_x,source_x=joint_x.split(
                (main_tokens,joint_x.shape[0]-main_tokens),dim=0,
            )
        with profile_range("brainwm.sparse_packed/joint_cell_pool"):
            joint_row_ptr=torch.cat((
                main.row_ptr,
                source.row_ptr[1:]+main_tokens,
            ))
            joint_lengths=torch.cat((main.lengths,source.lengths))
            if (main.minimum_sequence_length > 0
                    and source.minimum_sequence_length > 0):
                joint_cell=torch.segment_reduce(
                    joint_x,"mean",offsets=joint_row_ptr,
                )
            else:
                joint_denominators=joint_lengths.clamp_min(1).to(
                    joint_x.dtype,
                ).unsqueeze(-1)
                joint_cell=torch.segment_reduce(
                    joint_x,"sum",offsets=joint_row_ptr,
                )/joint_denominators
            main_cell,source_cell=joint_cell.split(
                (main.batch.batch_size,source.batch.batch_size),dim=0,
            )
        main_output=self._finish_sparse_packed(
            main,main_x,forecast_sum,encode_only=False,
            balance_losses=main_balance,precomputed_cell=main_cell,
        )
        source_output=self._finish_sparse_packed(
            source,source_x,None,encode_only=True,precomputed_cell=source_cell,
        )
        main_output["development_attention_fused"]=True
        main_output["development_attention_streams"]=2
        main_output["development_target_attention_fused"]=False
        return main_output,source_output

    def _forward_sparse_flash_triple(
        self, main_batch: SparseTokenBatch, source_batch: SparseTokenBatch,
        target_batch: SparseTokenBatch,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Fuse main/source/target attention while keeping target inference-only.

        Flash-BLA treats each CSR row as an independent sequence, so appending
        populations cannot create cross-population attention.  The target
        stream is detached immediately after the shared attention call and its
        block finish runs under inference mode; consequently only main/source
        contribute gradients even though all three populations use one kernel
        launch per block.
        """
        main = self._prepare_sparse_packed(main_batch)
        source = self._prepare_sparse_packed(source_batch)
        with torch.inference_mode():
            target = self._prepare_sparse_packed(target_batch)
        if any(state.attention_backend != "flash_bla"
               for state in (main, source, target)):
            raise ValueError("triple packed attention fusion requires Flash-BLA")
        if not all(hasattr(block, "finish_varlen_pair") for block in self.blocks):
            raise RuntimeError(
                "triple packed attention fusion requires easy_deepspeed blocks"
            )

        main_tokens = int(main.x.shape[0])
        source_tokens = int(source.x.shape[0])
        target_start = main_tokens + source_tokens
        main_cells = main.batch.batch_size
        source_cells = source.batch.batch_size
        gradient_cu = torch.cat((
            main.cu_seqlens,
            source.cu_seqlens[1:] + main_tokens,
        ))
        gradient_row_ids = torch.cat((
            main.row_ids,
            source.row_ids + main_cells,
        ))
        joint_cu = torch.cat((
            gradient_cu,
            target.cu_seqlens[1:] + target_start,
        ))
        joint_lengths = torch.cat((
            main.token_lengths,
            source.token_lengths,
            target.token_lengths,
        ))
        joint_row_ids = torch.cat((
            gradient_row_ids,
            target.row_ids + main_cells + source_cells,
        ))
        joint_max_seqlen = max(
            main.max_seqlen, source.max_seqlen, target.max_seqlen,
        )
        joint_x = torch.cat((main.x, source.x, target.x), dim=0)
        forecast_sum = None
        main_balance = []
        for block in self.blocks:
            prefix = block.profile_prefix
            with profile_range(prefix):
                with profile_range(f"{prefix}/joint_triple_attention"):
                    attended = block.attention.forward_varlen(
                        joint_x, joint_cu, joint_lengths, joint_max_seqlen,
                    )
                with profile_range(f"{prefix}/joint_triple_finish"):
                    joint_x, forecast, balance = block.finish_varlen_triple(
                        joint_x, attended, joint_cu, joint_row_ids,
                        main_tokens, main_cells, source_cells,
                    )
                    main_balance.append(balance)
            with profile_range("brainwm.sparse_packed/forecast_accumulate"):
                forecast_sum = (
                    forecast if forecast_sum is None else forecast_sum + forecast
                )
            # Keep the joint stream as one buffer.  Flash-BLA receives the
            # population boundaries through ``joint_cu`` and therefore never
            # mixes main/source/target rows.  The target branch has no path to
            # a differentiable objective (its endpoint is consumed under
            # ``no_grad``), so the previous per-block ``cat(main,
            # target.detach())`` only copied the complete token buffer.  The
            # target is detached once at the final output boundary below.
            # Re-materialize the detached target suffix at each boundary. This
            # intentionally cuts the autograd graph before the next block;
            # leaving the suffix attached saves a copy but retains all target
            # activations through backward and increases peak memory.
            joint_x = concat_detached_suffix(
                joint_x[:target_start], joint_x[target_start:].detach(),
            )

        with profile_range("brainwm.sparse_packed/final_norm"):
            joint_x = self.norm(joint_x)
            main_x = joint_x[:main_tokens]
            source_x = joint_x[main_tokens:target_start]
            target_x = joint_x[target_start:].detach()
        with profile_range("brainwm.sparse_packed/joint_cell_pool"):
            joint_row_ptr = torch.cat((
                main.row_ptr,
                source.row_ptr[1:] + main_tokens,
                target.row_ptr[1:] + target_start,
            ))
            joint_lengths = torch.cat((
                main.lengths, source.lengths, target.lengths,
            ))
            if (main.minimum_sequence_length > 0
                    and source.minimum_sequence_length > 0
                    and target.minimum_sequence_length > 0):
                joint_cell = torch.segment_reduce(
                    joint_x, "mean", offsets=joint_row_ptr,
                )
            else:
                joint_denominators = joint_lengths.clamp_min(1).to(
                    joint_x.dtype,
                ).unsqueeze(-1)
                joint_cell = torch.segment_reduce(
                    joint_x, "sum", offsets=joint_row_ptr,
                ) / joint_denominators
            main_cell, source_cell, target_cell = torch.split(
                joint_cell,
                (main.batch.batch_size, source.batch.batch_size,
                 target.batch.batch_size),
                dim=0,
            )
            target_cell = target_cell.detach()
        main_output = self._finish_sparse_packed(
            main, main_x, forecast_sum, encode_only=False,
            balance_losses=main_balance, precomputed_cell=main_cell,
        )
        source_output = self._finish_sparse_packed(
            source, source_x, None, encode_only=True,
            precomputed_cell=source_cell,
        )
        with torch.inference_mode():
            target_output = self._finish_sparse_packed(
                target, target_x, None, encode_only=True,
                precomputed_cell=target_cell,
            )
        main_output["development_attention_fused"] = True
        main_output["development_attention_streams"] = 3
        main_output["development_target_attention_fused"] = True
        return main_output, source_output, target_output


def _build_easy_deepspeed_block(**kwargs):
    """Factory adapter for the common registered MoE block contract."""
    return DeepSpeedEasyMoEBlock(**kwargs)


def _build_mage_beta_block(**kwargs):
    """Build the legacy beta router while ignoring execution-only fields.

    The MAGE beta block has its own token-level differential attention and
    therefore does not consume the shared Flash-BLA projection knobs. Keeping
    the adapter here lets the model constructor remain backend agnostic.
    """
    supported = {
        key: kwargs[key]
        for key in ("d_model", "d_hidden", "n_experts", "top_k",
                    "ffn_expansion", "fuse_nbeats_heads",
                    "expert_attention_backend")
        if key in kwargs
    }
    return BetaExpertBlock(**supported)


def _build_mage_attention_expert(**kwargs):
    """Adapt the common attention factory contract to the legacy MAGE expert."""
    return MAGEAttentionExpert(
        d_model=int(kwargs["d_model"]),
        d_hidden=int(kwargs["d_hidden"]),
        ffn_expansion=float(kwargs.get("ffn_expansion", 2.67)),
    )


# Model-specific built-ins are registered after all classes are defined. The
# names preserve the historical checkpoint/config values; aliases make it
# straightforward to select the same implementation from an experiment file.
register_attention_backend(
    "flash_bla_differential",
    DifferentialLinearAttention,
    aliases=("differential_linear", "torch_differential"),
    capabilities={
        "linear_complexity": True,
        "flash_bla_varlen": True,
        "packed_csr": True,
        "differential_correction": True,
        "bf16": True,
        "backward": True,
        "state_dict_compatible": True,
    },
)
register_attention_backend(
    "torch_sdpa",
    TorchSDPAAttention,
    aliases=("pytorch_mha", "multihead_attention"),
    capabilities={
        "quadratic_complexity": True,
        "pytorch_sdpa": True,
        "packed_bucket": True,
        "differential_correction": False,
        "bf16": True,
        "backward": True,
        "state_dict_compatible": False,
    },
)
register_attention_backend(
    "mage_differential",
    _build_mage_attention_expert,
    aliases=("mage_expert",),
    capabilities={
        "linear_complexity": True,
        "differential_correction": True,
        "expert_internal": True,
        "bf16": True,
        "backward": True,
        "state_dict_compatible": True,
    },
)
register_moe_backend(
    "easy_deepspeed",
    _build_easy_deepspeed_block,
    aliases=("deepspeed", "sparse_moe"),
    capabilities={
        "sparse_dispatch": True,
        "top_k": (1, 2),
        "packed_csr": True,
        "joint_population_dispatch": True,
        "expert_execution": True,
        "attention_projection_execution": True,
        "feature_map_execution": True,
        "differential_merge_execution": True,
        "residual_norm_execution": True,
        "bf16": True,
        "ddp": True,
        "state_dict_compatible": True,
    },
)
register_moe_backend(
    "mage_beta",
    _build_mage_beta_block,
    aliases=("beta", "mage"),
    capabilities={
        "positive_affinity_routing": True,
        "beta_balance_update": True,
        "top_k": (0, 1),
        "packed_csr": False,
        "expert_execution": False,
        "attention_projection_execution": False,
        "feature_map_execution": False,
        "differential_merge_execution": False,
        "residual_norm_execution": False,
        "bf16": True,
        "ddp": True,
        "state_dict_compatible": True,
    },
)


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def freeze_for_modality_adaptation(model: BrainWMFoundationMoE, modality: str) -> None:
    """Freeze the foundation and expose only one newly added modality adapter."""
    if modality not in model.modalities:
        raise KeyError(f"unknown modality {modality!r}; call add_modality first")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.modalities[modality].parameters():
        p.requires_grad_(True)
