#!/usr/bin/env python
"""Fixed-shape BrainWM router A/B benchmark.

This measures only the dense FP32 gate projection. DeepSpeed capacity and
expert dispatch are intentionally excluded so the result isolates whether
TorchInductor amortizes the router GEMM/launch overhead. It never constructs a
training graph or updates parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainwm.router_acceleration import (  # noqa: E402
    FixedShapeCudaGraph,
    clear_router_compile_cache,
    router_projection_logits,
    run_router_projection_logits,
)


PRESETS = {
    "light": (64, 12),
    "normal": (256, 24),
    "ultra": (512, 32),
}


def _cuda_ms(function, values, weights, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function(values, weights)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="light")
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True,
                        help="also benchmark forward-only fixed-shape CUDA Graph replay")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune",
                 "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
    )
    args = parser.parse_args()
    if args.cells < 1 or args.iterations < 1 or args.warmup < 1:
        raise SystemExit("cells, iterations and warmup must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the router benchmark")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must be CUDA")
    hidden, experts = PRESETS[args.preset]
    torch.cuda.set_device(device)
    torch.manual_seed(20260913)
    values = torch.randn(args.cells, hidden, device=device, dtype=torch.bfloat16)
    weights = torch.randn(experts, hidden, device=device, dtype=torch.float32)
    clear_router_compile_cache()

    eager_fn = lambda x, w: run_router_projection_logits(
        x, w, execution="eager",
    )[0]
    eager_backend = "eager"
    for _ in range(args.warmup):
        eager_fn(values, weights)
    torch.cuda.synchronize(device)
    eager_ms = _cuda_ms(eager_fn, values, weights, args.iterations)
    eager_out = eager_fn(values, weights).detach()

    compile_started = time.perf_counter()
    compiled_out, compiled_backend = run_router_projection_logits(
        values, weights, execution="compile", mode=args.compile_mode,
    )
    torch.cuda.synchronize(device)
    compile_wall_s = time.perf_counter() - compile_started
    # One warmup call after graph construction prevents compilation work from
    # contaminating the steady-state event measurement.
    for _ in range(args.warmup):
        run_router_projection_logits(
            values, weights, execution="compile", mode=args.compile_mode,
        )
    torch.cuda.synchronize(device)
    compiled_fn = lambda x, w: run_router_projection_logits(
        x, w, execution="compile", mode=args.compile_mode,
    )[0]
    compiled_ms = _cuda_ms(compiled_fn, values, weights, args.iterations)
    compiled_out = compiled_fn(values, weights).detach()
    max_error = float((eager_out - compiled_out).abs().max().cpu())
    graph_report = {
        "enabled": bool(args.cuda_graph),
        "capture_wall_seconds": None,
        "ms": None,
        "speedup_eager_over_graph": None,
        "max_abs_error": None,
        "error": None,
    }
    if args.cuda_graph:
        try:
            graph_started = time.perf_counter()
            graph = FixedShapeCudaGraph.capture(
                router_projection_logits, (values, weights), warmup=max(1, args.warmup),
            )
            graph_report["capture_wall_seconds"] = time.perf_counter() - graph_started
            for _ in range(args.warmup):
                graph.replay(values, weights)
            torch.cuda.synchronize(device)
            graph_start = torch.cuda.Event(enable_timing=True)
            graph_end = torch.cuda.Event(enable_timing=True)
            graph_start.record()
            for _ in range(args.iterations):
                graph.replay(values, weights)
            graph_end.record()
            graph_end.synchronize()
            graph_report["ms"] = float(
                graph_start.elapsed_time(graph_end) / args.iterations
            )
            graph_out = graph.replay(values, weights).detach().clone()
            graph_report["max_abs_error"] = float(
                (eager_out - graph_out).abs().max().cpu()
            )
            if graph_report["ms"] > 0:
                graph_report["speedup_eager_over_graph"] = (
                    eager_ms / graph_report["ms"]
                )
        except Exception as exc:  # pragma: no cover - driver/runtime dependent
            graph_report["error"] = f"{type(exc).__name__}: {exc}"
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    report = {
        "format": "brainwm_router_ab_v1",
        "preset": args.preset,
        "cells": args.cells,
        "hidden": hidden,
        "experts": experts,
        "dtype": str(values.dtype),
        "device": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "compile_mode": args.compile_mode,
        "compile_backend": compiled_backend,
        "compile_wall_seconds": compile_wall_s,
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "speedup_eager_over_compiled": eager_ms / compiled_ms
        if compiled_ms > 0 else None,
        "max_abs_error": max_error,
        "peak_allocated_bytes": peak_bytes,
        "cuda_graph": graph_report,
        "correct": max_error <= 2e-3,
        "notes": (
            "Projection-only A/B; DeepSpeed top-k/capacity/dispatch and "
            "backward are intentionally excluded."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
