"""Optional fixed-shape acceleration for the BrainWM MoE router.

The router is deliberately kept separate from the model module.  Compiled
callables and CUDA graphs are execution caches, not parameters, so they must
never enter a checkpoint or an optimizer state.  The functions in this file
only cover the tensor-only part of routing (gate projection, softmax and
top-k); capacity construction and dispatch remain owned by the selected
MoE implementation.

CUDA Graph support is intentionally forward-only.  A graph captures the
exact tensor shapes and memory addresses used by one replay.  It is useful for
an apples-to-apples latency A/B, but it is not silently enabled for training
where dynamic batches, autograd and distributed dispatch need different
lifetimes.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence
import warnings

import torch
from torch import Tensor, nn

from .backends import compile_callable


_COMPILED_ROUTER_CACHE: "OrderedDict[tuple[object, ...], Callable]" = OrderedDict()
_COMPILED_ROUTER_CACHE_LIMIT = 32
_WARNED_FALLBACKS: set[str] = set()


def router_projection_topk(
    flat: Tensor,
    gate_weight: Tensor,
    top_k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference tensor-only router operation.

    This matches the part of DeepSpeed's ``TopKGate`` used by BrainWM's
    grouped expert path: the gate is evaluated in FP32, probabilities are
    sorted top-k values, and top-2 weights are renormalized.  Capacity and
    token dispatch deliberately happen outside this function.
    """
    if flat.ndim != 2 or gate_weight.ndim != 2:
        raise ValueError("router inputs must be rank-2 tensors")
    if flat.shape[-1] != gate_weight.shape[-1]:
        raise ValueError("router hidden dimensions do not match")
    if top_k not in (1, 2):
        raise ValueError("BrainWM router supports top_k=1 or top_k=2")
    logits = router_projection_logits(flat, gate_weight)
    probabilities = logits.softmax(dim=-1)
    route_weights, route_experts = torch.topk(
        probabilities, k=top_k, dim=-1, sorted=True,
    )
    if top_k > 1:
        route_weights = route_weights / route_weights.sum(
            dim=-1, keepdim=True,
        ).clamp_min(torch.finfo(route_weights.dtype).eps)
    return probabilities, route_experts, route_weights


def router_projection_logits(flat: Tensor, gate_weight: Tensor) -> Tensor:
    """Reference FP32 gate projection used by DeepSpeed's dispatcher.

    Keeping this boundary separate from top-k is deliberate: DeepSpeed owns
    capacity construction, token dropping and dispatch semantics. The
    optional compiled path can therefore fuse only the dense projection while
    the exact upstream top1gating/top2gating implementation remains in charge
    of routing decisions.
    """
    if flat.ndim != 2 or gate_weight.ndim != 2:
        raise ValueError("router inputs must be rank-2 tensors")
    if flat.shape[-1] != gate_weight.shape[-1]:
        raise ValueError("router hidden dimensions do not match")
    return torch.nn.functional.linear(
        flat.float(), gate_weight.float(), bias=None,
    )


def _cache_key(
    flat: Tensor,
    gate_weight: Tensor,
    top_k: int,
    mode: str,
    dynamic: bool,
) -> tuple[object, ...]:
    # Include shapes and devices so ``dynamic=False`` remains genuinely
    # fixed-shape.  Parameter values are inputs to the compiled callable and
    # therefore do not belong in the key.
    return (
        mode,
        bool(dynamic),
        int(top_k),
        tuple(flat.shape),
        tuple(gate_weight.shape),
        str(flat.device),
        str(flat.dtype),
        str(gate_weight.dtype),
    )


def _compiled_router(
    flat: Tensor,
    gate_weight: Tensor,
    top_k: int,
    *,
    mode: str,
    dynamic: bool,
) -> Callable:
    key = _cache_key(flat, gate_weight, top_k, mode, dynamic)
    cached = _COMPILED_ROUTER_CACHE.get(key)
    if cached is not None:
        _COMPILED_ROUTER_CACHE.move_to_end(key)
        return cached
    compiled = compile_callable(
        "torch_compile",
        lambda values, weights: router_projection_topk(values, weights, top_k),
        mode=mode,
        dynamic=bool(dynamic),
        fullgraph=True,
    )
    _COMPILED_ROUTER_CACHE[key] = compiled
    _COMPILED_ROUTER_CACHE.move_to_end(key)
    while len(_COMPILED_ROUTER_CACHE) > _COMPILED_ROUTER_CACHE_LIMIT:
        _COMPILED_ROUTER_CACHE.popitem(last=False)
    return compiled


def _compiled_router_logits(
    flat: Tensor,
    gate_weight: Tensor,
    *,
    mode: str,
    dynamic: bool,
) -> Callable:
    key = (
        "logits", mode, bool(dynamic), tuple(flat.shape),
        tuple(gate_weight.shape), str(flat.device), str(flat.dtype),
        str(gate_weight.dtype),
    )
    cached = _COMPILED_ROUTER_CACHE.get(key)
    if cached is not None:
        _COMPILED_ROUTER_CACHE.move_to_end(key)
        return cached
    compiled = compile_callable(
        "torch_compile",
        router_projection_logits,
        mode=mode,
        dynamic=bool(dynamic),
        fullgraph=True,
    )
    _COMPILED_ROUTER_CACHE[key] = compiled
    _COMPILED_ROUTER_CACHE.move_to_end(key)
    while len(_COMPILED_ROUTER_CACHE) > _COMPILED_ROUTER_CACHE_LIMIT:
        _COMPILED_ROUTER_CACHE.popitem(last=False)
    return compiled


def run_router_projection_logits(
    flat: Tensor,
    gate_weight: Tensor,
    *,
    execution: str = "eager",
    mode: str = "max-autotune-no-cudagraphs",
    dynamic: bool = False,
    min_tokens: int = 256,
) -> tuple[Tensor, str]:
    """Run the FP32 gate projection with an optional shape compiler.

    The return tag is intended for step telemetry. Compilation failures are
    isolated to this optional path and return the exact eager projection.
    """
    if execution not in {"eager", "compile", "auto"}:
        raise ValueError("router execution must be eager, compile, or auto")
    if min_tokens < 1:
        raise ValueError("router compile minimum token count must be positive")
    should_compile = execution == "compile" or (
        execution == "auto" and flat.is_cuda and flat.shape[0] >= min_tokens
    )
    if not should_compile:
        return router_projection_logits(flat, gate_weight), "eager"
    try:
        compiled = _compiled_router_logits(
            flat, gate_weight, mode=mode, dynamic=dynamic,
        )
        return compiled(flat, gate_weight), "torch_compile"
    except Exception as exc:  # pragma: no cover - compiler/device dependent
        warning_key = f"logits:{type(exc).__name__}:{str(exc)[:160]}"
        if warning_key not in _WARNED_FALLBACKS:
            _WARNED_FALLBACKS.add(warning_key)
            warnings.warn(
                "BrainWM compiled router projection disabled for this shape "
                f"and fell back to eager ({type(exc).__name__}: {exc})",
                RuntimeWarning,
                stacklevel=2,
            )
        return router_projection_logits(flat, gate_weight), "eager_fallback"


class CompiledTopKGate(nn.Module):
    """DeepSpeed-compatible gate with an optional compiled projection.

    The wrapper exposes the same public attributes as the DeepSpeed gate and
    owns only the original wg parameter. Replacing a gate therefore changes no
    state-dict key: gate.wg.weight remains unchanged. Capacity and dispatch
    are delegated to DeepSpeed's reviewed gating functions.
    """

    def __init__(
        self,
        gate: nn.Module,
        *,
        execution: str = "eager",
        min_tokens: int = 256,
        mode: str = "max-autotune-no-cudagraphs",
        dynamic: bool = False,
    ) -> None:
        super().__init__()
        self.wg = gate.wg
        self.k = int(gate.k)
        self.capacity_factor = gate.capacity_factor
        self.eval_capacity_factor = gate.eval_capacity_factor
        self.min_capacity = gate.min_capacity
        self.noisy_gate_policy = gate.noisy_gate_policy
        self.timers = gate.timers
        self.wall_clock_breakdown = gate.wall_clock_breakdown
        self.gate_time = gate.gate_time
        self.drop_tokens = gate.drop_tokens
        self.use_rts = gate.use_rts
        self.ep_group = gate.ep_group
        self.top2_2nd_expert_sampling = gate.top2_2nd_expert_sampling
        self.execution = execution
        self.min_tokens = int(min_tokens)
        self.mode = mode
        self.dynamic = bool(dynamic)
        self.last_backend = "eager"

    def forward(
        self,
        input: Tensor,
        used_token: Tensor = None,
        sparse_routes: bool = False,
        use_tutel: bool = False,
    ):
        # Import lazily so data-only utilities do not initialize DeepSpeed.
        from deepspeed.moe.sharded_moe import (
            TOPK_GATE_TIMER, multiplicative_jitter, top1gating,
            top2gating, topkgating,
        )
        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()
        input_fp32 = input.float()
        if self.noisy_gate_policy == "Jitter" and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        logits, self.last_backend = run_router_projection_logits(
            input_fp32, self.wg.weight, execution=self.execution,
            mode=self.mode, dynamic=self.dynamic, min_tokens=self.min_tokens,
        )
        if self.k == 1:
            gate_output = top1gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity, used_token,
                self.noisy_gate_policy if self.training else None,
                self.drop_tokens, self.use_rts, self.ep_group,
                sparse_routes, use_tutel,
            )
        elif self.k == 2:
            gate_output = top2gating(
                logits,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity, self.drop_tokens, self.ep_group,
                self.top2_2nd_expert_sampling, sparse_routes,
            )
        else:
            gate_output = topkgating(
                logits, self.k,
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity, self.drop_tokens, self.ep_group,
                sparse_routes=sparse_routes,
            )
        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)
        return gate_output


def run_router_projection_topk(
    flat: Tensor,
    gate_weight: Tensor,
    top_k: int,
    *,
    execution: str = "eager",
    mode: str = "max-autotune-no-cudagraphs",
    dynamic: bool = False,
    min_tokens: int = 32768,
) -> tuple[tuple[Tensor, Tensor, Tensor], str]:
    """Execute the router and return ``(outputs, backend_name)``.

    ``auto`` only compiles CUDA inputs at or above ``min_tokens``.  Explicit
    ``compile`` is useful for a controlled A/B and may be used on CPU too.
    Compilation failures are isolated to this optional path and fall back to
    the exact eager equation, with a warning emitted once per failure class.
    """
    if execution not in {"eager", "compile", "auto"}:
        raise ValueError("router execution must be eager, compile, or auto")
    if min_tokens < 1:
        raise ValueError("router compile minimum token count must be positive")
    should_compile = execution == "compile" or (
        execution == "auto" and flat.is_cuda and flat.shape[0] >= min_tokens
    )
    if not should_compile:
        return router_projection_topk(flat, gate_weight, top_k), "eager"
    try:
        compiled = _compiled_router(
            flat, gate_weight, top_k, mode=mode, dynamic=dynamic,
        )
        return compiled(flat, gate_weight), "torch_compile"
    except Exception as exc:  # pragma: no cover - depends on local compiler
        warning_key = f"{type(exc).__name__}:{str(exc)[:160]}"
        if warning_key not in _WARNED_FALLBACKS:
            _WARNED_FALLBACKS.add(warning_key)
            warnings.warn(
                "BrainWM compiled router disabled for this shape and fell "
                f"back to eager ({type(exc).__name__}: {exc})",
                RuntimeWarning,
                stacklevel=2,
            )
        return router_projection_topk(flat, gate_weight, top_k), "eager_fallback"


@dataclass
class FixedShapeCudaGraph:
    """Forward-only CUDA Graph replay for a tensor-only callable.

    Construct with :meth:`capture`; all inputs must be CUDA tensors and every
    replay must use the same shapes/dtypes.  ``replay`` copies new values into
    the captured input buffers and returns the captured output objects.  The
    class intentionally rejects autograd and non-tensor arguments so it cannot
    be mistaken for a general training wrapper.
    """

    graph: torch.cuda.CUDAGraph
    static_inputs: tuple[Tensor, ...]
    static_outputs: object

    @classmethod
    def capture(
        cls,
        function: Callable[..., object],
        sample_inputs: Sequence[Tensor],
        *,
        warmup: int = 3,
    ) -> "FixedShapeCudaGraph":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph capture requires CUDA")
        if warmup < 1:
            raise ValueError("CUDA Graph warmup must be positive")
        inputs = tuple(sample_inputs)
        if not inputs or any(not isinstance(value, Tensor) for value in inputs):
            raise TypeError("CUDA Graph inputs must be a non-empty tensor tuple")
        if any(not value.is_cuda for value in inputs):
            raise ValueError("all CUDA Graph inputs must be CUDA tensors")
        if any(value.requires_grad for value in inputs):
            raise ValueError("FixedShapeCudaGraph is forward-only; disable grad")

        static_inputs = tuple(value.detach().clone() for value in inputs)
        capture_stream = torch.cuda.Stream(device=static_inputs[0].device)
        current_stream = torch.cuda.current_stream(static_inputs[0].device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(warmup):
                function(*static_inputs)
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(static_inputs[0].device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream), torch.inference_mode():
            static_outputs = function(*static_inputs)
        torch.cuda.current_stream(static_inputs[0].device).wait_stream(capture_stream)
        return cls(graph, static_inputs, static_outputs)

    def replay(self, *inputs: Tensor):
        if len(inputs) != len(self.static_inputs):
            raise ValueError("CUDA Graph replay input count changed")
        for static, value in zip(self.static_inputs, inputs, strict=True):
            if not isinstance(value, Tensor) or not value.is_cuda:
                raise ValueError("CUDA Graph replay inputs must be CUDA tensors")
            if value.shape != static.shape or value.dtype != static.dtype:
                raise ValueError("CUDA Graph replay shape or dtype changed")
            static.copy_(value)
        self.graph.replay()
        return self.static_outputs


def clear_router_compile_cache() -> None:
    """Drop process-local compiled router callables between A/B campaigns."""
    _COMPILED_ROUTER_CACHE.clear()


__all__ = [
    "FixedShapeCudaGraph",
    "CompiledTopKGate",
    "clear_router_compile_cache",
    "router_projection_topk",
    "router_projection_logits",
    "run_router_projection_logits",
    "run_router_projection_topk",
]
