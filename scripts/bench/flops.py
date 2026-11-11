"""Analytic FLOPs and MFU of one training step on real Cell x Gene batches.

    python scripts/bench/flops.py --config runs/unified_v0_1m/config.json --step-seconds 2.72

Counts matmul FLOPs (2 per multiply-add) for the encoder blocks, attention over each
cell's own tokens, the token embedding MLP and the heads; elementwise ops are ignored.
Model FLOPs = forward x 3 (forward + backward); hardware FLOPs add one more encoder
forward when every block is recomputed (activation checkpointing).
"""

import argparse
import json

import numpy as np

from brainwm_unified.config import DataConfig, ModelConfig
from brainwm_unified.data import Subset, build_batch

H20_BF16_TFLOPS = 148.0  # dense BF16 tensor-core peak of one H20


def block_flops(cfg: ModelConfig, tokens: int, sum_sq: int) -> dict:
    """Forward FLOPs of one encoder block call over `tokens` packed tokens."""
    d, h, k, e = cfg.d_model, cfg.expert_hidden, cfg.top_k, cfg.n_experts
    return {
        "attn.qkv+out": 2 * tokens * d * 4 * d,
        "attn.core (QK, PV)": 4 * d * sum_sq,
        "moe.router": 2 * tokens * d * e,
        "moe.routed experts": 2 * tokens * k * 3 * d * h,
        "moe.shared expert": 2 * tokens * 3 * d * h * cfg.n_shared_experts,
        "nbeats": 2 * tokens * 2 * d * d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batches", type=int, default=20, help="real batches to average the length statistics over")
    ap.add_argument("--loops", type=float, default=4.0, help="core loops per step (training samples 2-6, mean 4)")
    ap.add_argument("--step-seconds", type=float, default=None, help="measured step time for MFU")
    args = ap.parse_args()

    with open(args.config) as fh:
        run = json.load(fh)
    cfg, dcfg = ModelConfig(**run["ModelConfig"]), DataConfig(**run["DataConfig"])
    ds = Subset(dcfg.root)
    rng = np.random.default_rng(0)
    stats, all_lengths = [], []
    for _ in range(args.batches):
        b = build_batch(ds, dcfg, cfg.n_genes, rng)
        lengths = np.bincount(b["tok_doc"].numpy())
        all_lengths.append(lengths)
        masked = int(b["tok_masked"].sum())
        groups = b["grp_masked"].numpy()
        stats.append(
            (lengths.sum(), (lengths.astype(np.int64) ** 2).sum(), len(lengths), masked, int(groups.sum()), groups.size)
        )
    tokens, sum_sq, cells, masked, targets, members = (float(np.mean(c)) for c in zip(*stats))
    calls = cfg.n_prelude + cfg.n_core * args.loops + cfg.n_coda
    squares = np.sort(np.concatenate(all_lengths).astype(np.int64) ** 2)[::-1]
    long_share = squares[: len(squares) // 4].sum() / squares.sum()

    per_block = block_flops(cfg, int(tokens), int(sum_sq))
    d, z = cfg.d_model, cfg.z_dim
    fwd = {f"encoder: {k}": v * calls for k, v in per_block.items()}
    fwd["embed value MLP"] = 2 * tokens * (d + d * d)
    fwd["gene head (masked tokens)"] = 2 * masked * (d * d + d)
    fwd["integration decoder"] = 2 * cells * (2 * z * cfg.decoder_hidden + cfg.decoder_hidden * cfg.n_genes)
    fwd["spatial: neighbourhood + decoder"] = (
        2 * members * (z * cfg.spatial_d + cfg.spatial_layers * 12 * cfg.spatial_d**2)
    )
    fwd["spatial: decoder on targets"] = 2 * targets * (2 * z * cfg.decoder_hidden + cfg.decoder_hidden * cfg.n_genes)
    total_fwd = sum(fwd.values())
    encoder_fwd = sum(v for k, v in fwd.items() if k.startswith("encoder"))
    model = 3 * total_fwd
    hardware = model + (encoder_fwd if cfg.activation_checkpointing else 0)

    print(
        f"batch: {cells:.0f} cells, {tokens / 1e3:.0f}k tokens, sum L^2 {sum_sq / 1e9:.2f}e9, "
        f"block calls/step {calls:g}; the longest 25% of cells hold {long_share * 100:.0f}% of sum L^2"
    )
    print(f"{'forward component':40s} {'GFLOP':>9s} {'share':>7s}")
    for k, v in sorted(fwd.items(), key=lambda kv: -kv[1]):
        print(f"{k:40s} {v / 1e9:9.1f} {v / total_fwd * 100:6.1f}%")
    print(f"forward {total_fwd / 1e12:.2f} TFLOP; model (x3) {model / 1e12:.2f}; with recompute {hardware / 1e12:.2f}")
    if args.step_seconds:
        mfu = model / args.step_seconds / (H20_BF16_TFLOPS * 1e12)
        hfu = hardware / args.step_seconds / (H20_BF16_TFLOPS * 1e12)
        print(
            f"step {args.step_seconds:.2f} s -> MFU {mfu * 100:.1f}%, HFU {hfu * 100:.1f}% "
            f"(peak {H20_BF16_TFLOPS:g} TFLOPS bf16); ideal step at 100% MFU {model / H20_BF16_TFLOPS / 1e12:.2f} s"
        )


if __name__ == "__main__":
    main()
