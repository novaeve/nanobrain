"""Profile training steps of the architecture on pre-built real batches (no data loading).

    python scripts/bench/profile_step.py --config runs/unified_v0_1m/config.json --out runs/profile

Prints (1) a CUDA-event breakdown of a step into block mask / forward / losses /
backward / optimizer, (2) GPU time per labelled forward region, (3) the top
kernels by GPU time, and writes a Chrome trace to --out.
"""

import argparse
import itertools
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from brainwm_unified import model as M
from brainwm_unified.config import DataConfig, ModelConfig, TrainConfig
from brainwm_unified.data import Subset, build_batch, load_level_maps
from brainwm_unified.objectives import TASKS, compute_losses
from brainwm_unified.train import MUON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.json of the run to mirror")
    ap.add_argument("--out", default="runs/profile")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--set", nargs="*", action="extend", default=[], help="model.<field>=<json> overrides")
    args = ap.parse_args()

    with open(args.config) as fh:
        run = json.load(fh)
    mcfg, dcfg = ModelConfig(**run["ModelConfig"]), DataConfig(**run["DataConfig"])
    tcfg = TrainConfig()
    for item in args.set:
        key, value = item.split("=", 1)
        section, name = key.split(".")
        setattr({"model": mcfg, "train": tcfg}[section], name, json.loads(value))
    device = torch.device("cuda")
    ds = Subset(dcfg.root)
    rng = np.random.default_rng(0)
    n_batches = args.warmup + args.steps
    batches = [
        M.pad_tokens({k: v.to(device) for k, v in build_batch(ds, dcfg, mcfg.n_genes, rng).items()})
        for _ in range(n_batches)
    ]
    maps = {
        k: [torch.as_tensor(m, device=device) for m in load_level_maps(os.path.join(dcfg.root, f"{k}_vocab.json"))]
        for k in ("cell_type", "anatomy")
    }
    panels = ds.panel_matrix(mcfg.n_genes).to(device)

    model = M.UnifiedModel(mcfg).to(device).train()
    M.enable_compiled_flex()
    muon, adamw = M.param_groups(model)
    opts = [
        MUON[tcfg.optimizer](muon, lr=1e-4, adjust_lr_fn="match_rms_adamw"),
        torch.optim.AdamW(adamw, lr=1e-4, fused=True),
    ]

    def step(b, timer=None):
        def mark(name):
            if timer is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                timer.append((name, e))

        mark("start")
        with record_function("phase.block_mask"):
            mask = M.attention_layout(b["tok_doc"], mcfg.attention_backend)
        mark("block_mask")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with record_function("phase.forward"):
                out = model(b, mask, args.loops)
            mark("forward")
            with record_function("phase.losses"):
                losses, _ = compute_losses(model, b, out, panels, maps, kl_weight=1.0)
                total = sum(losses[t] / losses[t].detach() for t in TASKS)
            mark("losses")
        with record_function("phase.backward"):
            for o in opts:
                o.zero_grad(set_to_none=True)
            total.backward()
        mark("backward")
        with record_function("phase.optimizer"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                o.step()
        mark("optimizer")

    for b in batches[: args.warmup]:
        step(b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    phases = defaultdict(list)
    walls = []
    for b in batches[args.warmup :]:
        timer = []
        t0 = time.perf_counter()
        step(b, timer)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        for (_, a), (name, e) in itertools.pairwise(timer):
            phases[name].append(a.elapsed_time(e))
    tokens = np.mean([int((b["tok_doc"] >= 0).sum()) for b in batches[args.warmup :]])
    cells = np.mean([int(b["tok_cls"].sum()) for b in batches[args.warmup :]])
    wall = float(np.median(walls))
    print(
        f"\n== step: median wall {wall:.3f} s, {cells:.0f} cells, {tokens / 1e3:.0f}k tokens, "
        f"{cells / wall:.0f} cells/s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    )
    for name, v in phases.items():
        ms = float(np.median(v))
        print(f"  {name:12s} {ms:9.1f} ms  {ms / 1e3 / wall * 100:5.1f}%")

    os.makedirs(args.out, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for b in batches[args.warmup : args.warmup + 2]:
            step(b)
        torch.cuda.synchronize()
    prof.export_chrome_trace(os.path.join(args.out, "trace.json"))
    events = prof.key_averages()

    def gpu_ms(e):
        return getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1e3 / 2

    def self_gpu_ms(e):
        return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) / 1e3 / 2

    print("\n== labelled regions (GPU ms per step; forward only, recompute in backward not included)")
    for e in sorted(
        (e for e in events if e.key.startswith(("phase.", "attn.", "moe.", "nbeats"))), key=lambda e: e.key
    ):
        print(f"  {e.key:22s} {gpu_ms(e):9.1f} ms   calls/step {e.count // 2}")
    kernels = sorted(
        (e for e in events if self_gpu_ms(e) > 0 and not e.key.startswith(("phase.", "attn.", "moe."))),
        key=self_gpu_ms,
        reverse=True,
    )
    busy = sum(self_gpu_ms(e) for e in kernels)
    print(f"\n== GPU busy {busy:.0f} ms of {wall * 1e3:.0f} ms wall per step ({busy / wall / 10:.0f}%); top kernels:")
    for e in kernels[:25]:
        print(f"  {self_gpu_ms(e):8.1f} ms  {e.count // 2:6d}x  {e.key[:90]}")


if __name__ == "__main__":
    main()
