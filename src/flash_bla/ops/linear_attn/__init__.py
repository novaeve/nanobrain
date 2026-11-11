from . import varlen as _varlen

# Triton 3.6 supports persistent autotune results, while the pinned upstream
# decorators leave it disabled. Enabling the runtime option here keeps the
# vendored kernel byte-for-byte identical and prevents every DDP rank and
# resumed process from benchmarking the same configurations again.
for _kernel_name in (
    "_varlen_fwd_state_kernel",
    "_varlen_fwd_output_kernel",
    "_varlen_bwd_state_kernel",
    "_varlen_bwd_input_kernel",
):
    getattr(_varlen,_kernel_name).cache_results=True

varlen_linear_attention=_varlen.varlen_linear_attention

__all__ = ["varlen_linear_attention"]
