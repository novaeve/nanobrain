"""Routed-expert SwiGLU (gather + experts + gated combine), forward + backward, per backend.

    python scripts/bench/bench_moe.py --tokens 313000

Backends: torch (gather, torch._grouped_mm, permutation combine), deepgemm (same
with DeepGEMM grouped GEMMs), scattermoe (gather/scatter fused into Triton GEMMs).
"""

import argparse
import time

import torch
import torch.nn.functional as F

from brainwm_unified import deepinfra


def routed_reference(x, w13, w2, idx, gate, backend):
    k = idx.shape[1]
    flat = idx.flatten()
    counts = torch.zeros(w13.shape[0], dtype=torch.long, device=x.device).scatter_add_(0, flat, torch.ones_like(flat))
    order = torch.argsort(flat, stable=True)
    y = deepinfra.grouped_swiglu_fused(x[order // k], w13, w2, counts, backend)
    unsorted = torch.empty_like(y)
    unsorted[order] = y
    return (unsorted.view(x.shape[0], k, -1) * gate.unsqueeze(-1).to(y.dtype)).sum(1)


def routed_scattermoe(x, w13, w2, idx, gate):
    from scattermoe.parallel_experts import flatten_sort_count, parallel_linear

    k, h = idx.shape[1], w2.shape[-1]
    sorted_e, sorted_s, offsets = flatten_sort_count(idx, num_experts=w13.shape[0])
    gate_up = parallel_linear(x, w13.permute(0, 2, 1), k, sorted_e, sorted_s, offsets, grouped_out=True)
    hidden = F.silu(gate_up[:, :h]) * gate_up[:, h:]
    return parallel_linear(hidden, w2.permute(0, 2, 1), 1, sorted_e, sorted_s, offsets, grouped_in=True, gates=gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=313_000)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--backends", default="torch,deepgemm,scattermoe")
    args = ap.parse_args()

    torch.manual_seed(0)
    n, e, k, d, h = args.tokens, args.experts, args.top_k, args.d, args.h
    x0 = torch.randn(n, d, device="cuda", dtype=torch.bfloat16)
    w13_0 = torch.randn(e, 2 * h, d, device="cuda", dtype=torch.bfloat16) * d**-0.5
    w2_0 = torch.randn(e, d, h, device="cuda", dtype=torch.bfloat16) * h**-0.5
    scores = torch.rand(n, e, device="cuda")
    idx = scores.topk(k, -1).indices
    gate = scores.gather(-1, idx)
    gate = (gate / gate.sum(-1, keepdim=True)).bfloat16()
    flops = 3 * 2 * n * k * 3 * d * h  # forward + backward of three projections

    ref = None
    for name in args.backends.split(","):
        x, w13, w2 = (t.clone().requires_grad_() for t in (x0, w13_0, w2_0))

        def run(x=x, w13=w13, w2=w2, name=name):
            if name == "scattermoe":
                y = routed_scattermoe(x, w13, w2, idx, gate)
            else:
                y = routed_reference(x, w13, w2, idx, gate, name)
            y.float().square().mean().backward()
            return y

        try:
            y = run()
        except (ImportError, RuntimeError, ValueError) as err:  # a backend that is not built
            print(f"{name:11s} unavailable: {type(err).__name__}: {str(err)[:80]}")
            continue
        if ref is None:
            ref = (y.detach().float(), w13.grad.float())
        else:
            out_err = ((y.detach().float() - ref[0]).norm() / ref[0].norm()).item()
            grad_err = ((w13.grad.float() - ref[1]).norm() / ref[1].norm()).item()
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 10 * 1e3
        extra = (
            ""
            if ref is None or name == args.backends.split(",")[0]
            else f"  rel.err out {out_err:.1e} dW {grad_err:.1e}"
        )
        print(f"{name:11s} {ms:7.2f} ms  {flops / ms / 1e9:6.1f} TFLOPS{extra}")


if __name__ == "__main__":
    main()
