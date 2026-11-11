# Flash Bidirectional Linear Attention

The CUDA variable-length linear-attention kernel is an unmodified snapshot of:

- Repository: https://github.com/fla-org/flash-bidirectional-linear-attention
- Commit: `dd222769550d320bbb67edf3d78ddd5b8cfabb4d`
- Retrieved: 2026-09-12
- License: MIT, retained in `LICENSE`

Vendored upstream files and SHA-256 digests:

| File | SHA-256 |
| --- | --- |
| `utils.py` | `fd0538e8edda18d547fb8544a8c8215f52135fd9098d3c4c40edfd5d280e4ba3` |
| `ops/linear_attn/varlen.py` | `e3a565e23896194af06ae4cf875b91513922fa7aef4bb9ed452e533d0da2d766` |
| `LICENSE` | `8b524c0dba745a1869ec8d9ffb97a62cedbc87bb92441f073d74ba073e3633df` |

The three `__init__.py` files are local packaging shims because the upstream
`setup.py` does not include its namespace-only operator directories in a
normal wheel. The linear-attention shim also enables Triton 3.6's
`cache_results` option on the four upstream Autotuner objects. This persists
the selected configurations across DDP ranks and process restarts without
changing the pinned kernel source. BrainWM calls the upstream operator through
its public `varlen_linear_attention` function and does not maintain a separate
kernel.
