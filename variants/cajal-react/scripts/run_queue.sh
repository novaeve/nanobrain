#!/usr/bin/env bash
# Run experiment configs one after another on one GPU, after an optional PID has exited.
#   scripts/run_queue.sh GPU [--after PID] CONFIG.toml ...
gpu=$1
shift
if [ "$1" = --after ]; then
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
  shift 2
fi
cd "$(dirname "$0")/.." && source env.sh && export PYTHONPATH=$PWD
for cfg in "$@"; do
  echo "$(date +%T) start $cfg on GPU $gpu"
  .venv/bin/python -m brainwm_unified.cli run "$cfg" --gpus "$gpu" --wait || echo "$(date +%T) FAILED $cfg"
done
echo "$(date +%T) queue done"
