#!/usr/bin/env python3
"""Compile and warm fixed-size developmental Sinkhorn graphs.

The training process still owns the runtime cache.  This command is a small,
single-device preflight that populates the shared TorchInductor cache and
records the cost of each requested bucket.  It intentionally runs buckets
serially so a multi-GPU launch never starts one compiler worker per rank.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Keep an isolated warmup from consuming every host core.  These defaults are
# applied before importing torch; callers can override them in the environment.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from brainwm.losses import SinkhornRuntime


def normalize_buckets(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Return ascending unique positive compile buckets."""
    buckets = tuple(sorted({int(value) for value in values}))
    if not buckets or buckets[0] < 1:
        raise ValueError("Sinkhorn buckets must contain positive integers")
    return buckets


def warmup(
    *,
    device: torch.device,
    width: int,
    buckets: tuple[int, ...],
    epsilon: float,
    iterations: int,
    mode: str,
) -> dict[str, object]:
    if width < 1:
        raise ValueError("Sinkhorn width must be positive")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon must be positive and iterations >= 1")
    runtime = SinkhornRuntime(
        epsilon=epsilon,
        iterations=iterations,
        enabled=True,
        mode=mode,
        buckets=buckets,
        max_cached_graphs=len(buckets),
    )
    records: list[dict[str, object]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(20260913)
    for bucket in buckets:
        source = torch.randn(
            bucket, width, device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        target = torch.randn_like(source, generator=generator)
        started = time.perf_counter()
        # ``reduce-overhead`` may return a CUDA-Graph-owned buffer.  A later
        # replay can overwrite that buffer, so retain an ordinary tensor for
        # the post-warmup numerical check without adding a clone to training.
        plan = runtime.plan(source, target).clone()
        torch.cuda.synchronize(device)
        first_seconds = time.perf_counter() - started

        steady_ms: list[float] = []
        for _ in range(2):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            runtime.plan(source, target)
            end.record()
            end.synchronize()
            steady_ms.append(float(begin.elapsed_time(end)))
        if not bool(torch.isfinite(plan).all()):
            raise RuntimeError(f"non-finite Sinkhorn plan for bucket {bucket}")
        records.append(
            {
                "bucket": bucket,
                "width": width,
                "first_call_seconds": first_seconds,
                "steady_call_ms": sum(steady_ms) / len(steady_ms),
                "compile_count": runtime.compile_count,
                "fallback_count": runtime.fallback_count,
            }
        )
        print(json.dumps(records[-1]), flush=True)
        del source, target, plan

    return {
        "format": "brainwm_sinkhorn_warmup_v1",
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "device": str(device),
        "precision": "bf16",
        "width": width,
        "epsilon": epsilon,
        "iterations": iterations,
        "mode": mode,
        "buckets": list(buckets),
        "records": records,
        "runtime": runtime.status(),
        "torchinductor_compile_threads": os.environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="reduce-overhead",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Sinkhorn warmup")
    buckets = normalize_buckets(args.buckets)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("Sinkhorn warmup requires a CUDA device")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    receipt = warmup(
        device=device,
        width=args.width,
        buckets=buckets,
        epsilon=args.epsilon,
        iterations=args.iterations,
        mode=args.mode,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
