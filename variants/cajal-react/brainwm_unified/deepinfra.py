"""DeepInfra: thin adapters between our model and DeepSeek's open infrastructure.

Grouped expert compute for MoE: tokens arrive sorted by expert, with `counts`
tokens per expert, and every expert is a SwiGLU with weights stacked as
(n_experts, out, in). Backends:

  loop      one matmul per expert (reference; any device)
  torch     torch._grouped_mm (BF16, Hopper), differentiable
  deepgemm  DeepGEMM m-grouped BF16 GEMMs (third_party/DeepGEMM), when installed
"""

import torch
import torch.nn.functional as F

BACKENDS = ("loop", "torch", "deepgemm")


def _loop_linear(x, w, counts):
    out, start = [], 0
    for e, n in enumerate(counts.tolist()):
        out.append(x[start : start + n] @ w[e].T)
        start += n
    return torch.cat(out)


def _torch_linear(x, w, counts):
    offs = torch.cumsum(counts, 0).to(torch.int32)
    y = torch._grouped_mm(x.bfloat16(), w.bfloat16().transpose(-2, -1), offs=offs)
    return y.to(x.dtype)


def grouped_linear(x, w, counts, backend: str):
    """y[rows of expert e] = x[rows of expert e] @ w[e].T"""
    if backend == "loop":
        return _loop_linear(x, w, counts)
    if backend == "torch":
        return _torch_linear(x, w, counts)
    raise ValueError(f"unknown grouped GEMM backend {backend!r}")


class _DeepGEMMLinear(torch.autograd.Function):
    """DeepGEMM contiguous-layout grouped linear on expert-padded rows.

    forward  y  = x @ w[g].T   m_grouped_bf16_gemm_nt_contiguous
    backward dx = dy @ w[g]    m_grouped_bf16_gemm_nn_contiguous
             dw = dy_g.T @ x_g k_grouped_bf16_gemm_tn_contiguous
    Padding rows (layout == -1) must be zero in x and dy.
    """

    @staticmethod
    def forward(ctx, x, w, layout, host_ks):
        import deep_gemm

        y = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=torch.bfloat16)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(x, w, y, layout)
        ctx.save_for_backward(x, w, layout)
        ctx.host_ks = host_ks
        return y

    @staticmethod
    def backward(ctx, dy):
        import deep_gemm

        x, w, layout = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        deep_gemm.m_grouped_bf16_gemm_nn_contiguous(dy, w, dx, layout)
        dw = torch.zeros(w.shape, device=w.device, dtype=torch.float32)
        ks = torch.tensor(ctx.host_ks, device=w.device, dtype=torch.int32)
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(dy, x, dw, ctx.host_ks, ks, dw)
        return dx, dw.to(w.dtype), None, None


def _deepgemm_swiglu(x, w1, w3, w2, counts):
    """Pad each expert's rows to DeepGEMM's alignment, run the SwiGLU, drop the padding."""
    import deep_gemm

    align = deep_gemm.get_mk_alignment_for_contiguous_layout()
    host_ks = [-(-n // align) * align for n in counts.tolist()]
    padded_start = torch.tensor([0, *host_ks[:-1]], device=x.device).cumsum(0)
    start = torch.cumsum(counts, 0) - counts
    expert = torch.repeat_interleave(torch.arange(len(host_ks), device=x.device), counts)
    dest = padded_start[expert] + torch.arange(x.shape[0], device=x.device) - start[expert]
    total = sum(host_ks)
    layout = torch.full((total,), -1, device=x.device, dtype=torch.int32)
    layout[dest] = expert.to(torch.int32)
    x_pad = x.new_zeros(total, x.shape[1], dtype=torch.bfloat16).index_copy(0, dest, x.bfloat16())

    def linear(inp, w):
        return _DeepGEMMLinear.apply(inp, w.bfloat16().contiguous(), layout, host_ks)

    h = F.silu(linear(x_pad, w1)) * linear(x_pad, w3)
    return linear(h, w2)[dest].to(x.dtype)


def grouped_swiglu(x, w1, w3, w2, counts, backend: str = "loop"):
    """Per-expert SwiGLU on expert-sorted tokens."""
    if backend == "deepgemm":
        return _deepgemm_swiglu(x, w1, w3, w2, counts)
    h = F.silu(grouped_linear(x, w1, counts, backend)) * grouped_linear(x, w3, counts, backend)
    return grouped_linear(h, w2, counts, backend)


def default_backend(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 9:
        return "torch"
    return "loop"


def grouped_swiglu_fused(x, w13, w2, counts, backend: str = "loop"):
    """Per-expert SwiGLU with the gate and up projections fused: w13 is (E, 2h, d), w2 is (E, d, h)."""
    h = w2.shape[-1]
    if backend == "deepgemm":
        return _deepgemm_swiglu(x, w13[:, :h], w13[:, h:], w2, counts)
    gate_up = grouped_linear(x, w13, counts, backend)
    return grouped_linear(F.silu(gate_up[:, :h]) * gate_up[:, h:], w2, counts, backend)
