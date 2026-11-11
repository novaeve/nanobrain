#!/usr/bin/env bash
# Build FlashAttention-3 (Hopper) into the project venv: bf16, head dims 64/128, varlen + backward.
cd /data/brainwm/BrainWM-Unified
T=$PWD/toolchain
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$T/bin:$PATH
export CC=$T/bin/x86_64-conda-linux-gnu-gcc CXX=$T/bin/x86_64-conda-linux-gnu-g++ CPATH=$T/include LDFLAGS="-L$T/lib -Wl,-rpath,$T/lib"
export MAX_JOBS=32 NVCC_THREADS=2
for f in FP16 FP8 HDIM96 HDIM192 HDIM256 HDIMDIFF64 HDIMDIFF192 SM80 PAGEDKV APPENDKV SPLIT LOCAL SOFTCAP PACKGQA; do
  export FLASH_ATTENTION_DISABLE_$f=TRUE
done
cd third_party/flash-attention/hopper
start=$(date +%s)
nice -n 10 ../../../.venv/bin/python -m pip --version >/dev/null 2>&1 || true
nice -n 10 uv pip install -p ../../../.venv/bin/python --no-build-isolation . > ../../../fa3_build.log 2>&1
echo "EXIT=$? after $(( ($(date +%s) - start) / 60 )) min" >> ../../../fa3_build.log
