# Evaluation libraries (scikit-learn, XGBoost, cuML, scib-metrics, JAX CUDA, scvi-tools) live in the project's
# one environment, .venv (pyproject extras "eval" and "scvi"). JAX's cuSPARSE needs the venv's nvJitLink,
# not /usr/local/cuda's, or JAX falls back to the CPU. Do not import torch, JAX and cuML in one process
# (they crash together); each script uses only the set it needs.
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_nv=$_root/.venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$_nv/nvjitlink/lib:$_nv/cusparse/lib:$LD_LIBRARY_PATH
export PATH=$_root/.venv/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # share the GPU: allocate on demand
unset _nv _root
