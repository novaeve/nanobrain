#!/usr/bin/env bash
# Compatibility entry point for this fixed-head fork.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$TASK_ROOT/scripts/run_fixed_heads_evolution.sh" "$@"
