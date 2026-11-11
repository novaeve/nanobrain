#!/usr/bin/env bash
# Lint + all tests on one GPU: gate.sh GPU
cd "$(dirname "$0")/.." && source env.sh && export PYTHONPATH=$PWD
.venv/bin/ruff format -q brainwm_unified nanobrain tests scripts && .venv/bin/ruff check -q brainwm_unified nanobrain tests scripts && echo "lint ok"
echo "tests (GPU $1): $(CUDA_VISIBLE_DEVICES=$1 timeout 1800 .venv/bin/python -m pytest -q -p no:warnings -p no:logging tests 2>&1 | tail -1)"
