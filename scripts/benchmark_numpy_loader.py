#!/usr/bin/env python3
"""Measure NumPy-shard batch construction without starting a model.

The benchmark is intentionally separate from GPU profiling: it reports the
loader latency that can be hidden by ``CUDAPrefetcher`` and records the exact
worker/batch contract used for an A/B campaign.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
runtime_tmp = Path(os.environ.get("BRAINWM_TMPDIR", ROOT / "runtime/tmp"))
runtime_tmp.mkdir(parents=True, exist_ok=True)
for _name in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_name] = str(runtime_tmp)
sys.path.insert(0, str(ROOT / "src"))
from brainwm.numpy_shard_loader import shard_loader


def measure(args, workers: int) -> dict[str, object]:
    loader = shard_loader(
        args.shard_dir,
        split=args.split,
        workers=workers,
        pin_memory=False,
        batch_size=args.batch_size,
        max_padded_tokens=args.max_padded_tokens,
        packed=args.packed,
        packed_group_cells=args.packed_group_cells,
        shuffle=args.shuffle,
        seed=args.seed,
        balance_waves=args.balance_waves,
        zero_tokens_per_cell=args.zero_tokens_per_cell,
    )
    iterator = iter(loader)
    samples: list[float] = []
    batches: list[dict[str, int]] = []
    try:
        for _ in range(args.batches):
            started = time.perf_counter()
            batch = next(iterator)
            samples.append(time.perf_counter() - started)
            batches.append({
                "cells": int(batch.batch_size),
                "tokens": int(batch.values.numel()),
            })
    finally:
        # DataLoader workers are ordinary child processes. Explicitly shut
        # down before the next worker-count A/B so their IPC temp directories
        # and file descriptors do not overlap.
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        del iterator, loader
        gc.collect()
    steady = samples[1:] if len(samples) > 1 else samples
    return {
        "workers": int(workers),
        "batches": int(len(samples)),
        "first_batch_seconds": float(samples[0]) if samples else None,
        "steady_mean_seconds": float(statistics.mean(steady)) if steady else None,
        "steady_median_seconds": float(statistics.median(steady)) if steady else None,
        "steady_min_seconds": float(min(steady)) if steady else None,
        "steady_max_seconds": float(max(steady)) if steady else None,
        "cells": [item["cells"] for item in batches],
        "tokens": [item["tokens"] for item in batches],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-padded-tokens", type=int, default=720000)
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packed-group-cells", type=int, default=32)
    parser.add_argument("--zero-tokens-per-cell", type=int, default=16)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--balance-waves", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1 or any(value < 0 for value in args.workers):
        raise SystemExit("batches, batch-size, and workers must be valid positive values")
    report = {
        "format": "brainwm_numpy_loader_ab_v1",
        "shard_dir": str(args.shard_dir.resolve()),
        "split": args.split,
        "batch_size": args.batch_size,
        "max_padded_tokens": args.max_padded_tokens,
        "packed": bool(args.packed),
        "packed_group_cells": args.packed_group_cells,
        "zero_tokens_per_cell": args.zero_tokens_per_cell,
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "balance_waves": args.balance_waves,
        "results": [measure(args, workers) for workers in args.workers],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
