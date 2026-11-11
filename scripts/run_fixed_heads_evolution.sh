#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
exec "$TASK_ROOT/.venv/bin/python" -u scripts/run_nanobrain_evolution.py \
  --config configs/nanobrain_graph_v2/backbone_training_fixed_heads_50g_20261030.json \
  --output results/nanobrain/backbone-train-fixed-heads-20261030 \
  --adaptive-queue --workers-per-gpu 5 "$@"
