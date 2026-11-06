# Source before running: CUDA 13 toolkit for DeepGEMM JIT, and the gcc-13
# runtime (libstdc++) that DeepGEMM was built against.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
