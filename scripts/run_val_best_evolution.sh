#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
exec "$TASK_ROOT/.venv/bin/python" -u scripts/run_nanobrain_evolution.py \
  --config configs/nanobrain_graph_v2/test_driven_val_best_50g_20261029.json \
  --output results/nanobrain/test-driven-val-best-20261029 \
  --adaptive-queue --workers-per-gpu 5 "$@"
