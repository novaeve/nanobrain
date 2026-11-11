#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export TMPDIR="$PWD/research-runs/cajal-react/controller-tmp"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export HF_HOME="$TMPDIR/huggingface"
export XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR"
exec .venv/bin/python -u -B scripts/cajal_react.py --execute
