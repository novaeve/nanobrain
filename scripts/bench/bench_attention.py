"""Time per-cell attention (forward + backward) on real cell lengths for several kernels.

    python scripts/bench/bench_attention.py --data /data/brainwm/unified_1m

Backends: FlexAttention with the boundary BlockMask (current), and PyTorch nested
jagged tensors through scaled_dot_product_attention (FlashAttention varlen).
Each is timed for 8 heads x 32 dims and 4 heads x 64 dims (same model width).
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from brainwm_unified import model as M
from brainwm_unified.config import DataConfig
from brainwm_unified.data import Subset, build_batch


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DataConfig.root)
    ap.add_argument("--d-model", type=int, default=256)
    args = ap.parse_args()

    ds = Subset(args.data)
    b = build_batch(ds, DataConfig(root=args.data), 65918, np.random.default_rng(0))
    doc = M.pad_tokens({k: v.cuda() for k, v in b.items()})["tok_doc"]
    lengths = torch.bincount(b["tok_doc"]).cuda()
    n_real = int(lengths.sum())
    print(
        f"{len(lengths)} cells, {n_real} tokens, length p50/p90/max "
        f"{int(lengths.median())}/{int(lengths.float().quantile(0.9))}/{int(lengths.max())}"
    )
    mask = M.block_mask(doc)
    flex = torch.compile(flex_attention, dynamic=True)
    offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])

    for heads in (8, 4):
        dh = args.d_model // heads
        q, k, v = (
            torch.randn(1, heads, doc.shape[0], dh, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )

        def run_flex(q=q, k=k, v=v):
            flex(q, k, v, block_mask=mask).float().square().mean().backward()

        # Jagged layout: (cells, ragged tokens, heads, dh) -> SDPA expects (B, H, T, dh)
        qj, kj, vj = (
            torch.randn(n_real, heads, dh, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
        )

        def run_jagged(qj=qj, kj=kj, vj=vj):
            nq, nk, nv = (torch.nested.nested_tensor_from_jagged(t, offsets).transpose(1, 2) for t in (qj, kj, vj))
            out = F.scaled_dot_product_attention(nq, nk, nv)
            out.transpose(1, 2).values().float().square().mean().backward()

        cu = offsets.to(torch.int32)
        max_len = int(lengths.max())

        def run_fa3(qj=qj, kj=kj, vj=vj, cu=cu, max_len=max_len):
            from flash_attn_interface import flash_attn_varlen_func

            out = flash_attn_varlen_func(qj, kj, vj, cu, cu, max_len, max_len)
            out = out[0] if isinstance(out, tuple) else out
            out.float().square().mean().backward()

        backends = [("flex", run_flex), ("jagged_sdpa", run_jagged)] + ([("fa3_varlen", run_fa3)] if dh >= 64 else [])
        for name, fn in backends:
            try:
                print(f"  heads={heads} dh={dh:3d} {name:12s} {timeit(fn):8.1f} ms  (fwd+bwd)", flush=True)
            except (RuntimeError, NotImplementedError) as e:  # report and try the next backend
                print(f"  heads={heads} dh={dh:3d} {name:12s} failed: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
