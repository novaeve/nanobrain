#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/env.sh"
mkdir -p "$root/checkpoints" "$root/logs" "$root/triton"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${BRAINWM_OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${BRAINWM_MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${BRAINWM_OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
echo "$$" > "$root/logs/light-full.pid"
telemetry_interval="${BRAINWM_TELEMETRY_INTERVAL:-10}"
cuda_timing_interval="${BRAINWM_CUDA_TIMING_INTERVAL:-$telemetry_interval}"
workers="${BRAINWM_WORKERS:-2}"
router_dynamic_args=(--no-router-compile-dynamic)
if [[ "${BRAINWM_ROUTER_COMPILE_DYNAMIC:-0}" == "1" ]]; then
  router_dynamic_args=(--router-compile-dynamic)
fi
sinkhorn_args=()
if [[ "${BRAINWM_COMPILE_SINKHORN:-1}" == "1" ]]; then
  sinkhorn_args+=(--compile-sinkhorn --sinkhorn-compile-buckets 8 16 32 64)
  if [[ "${BRAINWM_WARMUP_SINKHORN:-1}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
      TORCHINDUCTOR_COMPILE_THREADS=1 \
      "$root/.venv/bin/python" "$root/scripts/warmup_sinkhorn.py" \
      --width 64 --buckets 8 16 32 64 \
      --output "$root/results/training/light-full/sinkhorn_warmup.json"
  fi
fi
resume_args=()
if [[ -f "$root/checkpoints/light-full.pt" ]]; then
  resume_args=(--resume "$root/checkpoints/light-full.pt")
fi

exec "$root/.venv/bin/python" -u "$root/scripts/train_foundation_moe.py" \
  --preset light \
  --moe-backend "${BRAINWM_MOE_BACKEND:-easy_deepspeed}" \
  --attention-backend "${BRAINWM_ATTENTION_BACKEND:-flash_bla_differential}" \
  --expert-attention-backend "${BRAINWM_EXPERT_ATTENTION_BACKEND:-mage_differential}" \
  --flow-backend "${BRAINWM_FLOW_BACKEND:-fmt}" \
  --compile-backend "${BRAINWM_COMPILE_BACKEND:-torch_compile}" \
  --optimizer-backend "${BRAINWM_OPTIMIZER_BACKEND:-adamw}" \
  --benchmark-dir "$root/contracts/healthy_global_benchmark_v3" \
  --numpy-shards "$root/assets/v3_numpy_shards" \
  --batch-size 4096 \
  --max-padded-tokens 720000 \
  --packed-tokens \
  --packed-group-cells 32 \
  --packed-attention-backend auto \
  --expert-execution "${BRAINWM_EXPERT_EXECUTION:-batched}" \
  --router-execution "${BRAINWM_ROUTER_EXECUTION:-eager}" \
  --router-min-tokens "${BRAINWM_ROUTER_MIN_TOKENS:-32768}" \
  --router-compile-mode "${BRAINWM_ROUTER_COMPILE_MODE:-max-autotune-no-cudagraphs}" \
  "${router_dynamic_args[@]}" \
  --attention-projection-execution "${BRAINWM_ATTENTION_PROJECTION_EXECUTION:-fused_triplet}" \
  --residual-norm-execution "${BRAINWM_RESIDUAL_NORM_EXECUTION:-auto}" \
  --residual-norm-min-tokens "${BRAINWM_RESIDUAL_NORM_MIN_TOKENS:-131072}" \
  --feature-map-execution "${BRAINWM_FEATURE_MAP_EXECUTION:-auto}" \
  --feature-map-min-tokens "${BRAINWM_FEATURE_MAP_MIN_TOKENS:-131072}" \
  --feature-map-compile-mode "${BRAINWM_FEATURE_MAP_COMPILE_MODE:-max-autotune-no-cudagraphs}" \
  --feature-map-bucket-size "${BRAINWM_FEATURE_MAP_BUCKET_SIZE:-65536}" \
  --feature-map-max-padding-ratio "${BRAINWM_FEATURE_MAP_MAX_PADDING_RATIO:-1.10}" \
  --differential-merge-execution "${BRAINWM_DIFFERENTIAL_MERGE_EXECUTION:-auto}" \
  --differential-merge-min-tokens "${BRAINWM_DIFFERENTIAL_MERGE_MIN_TOKENS:-131072}" \
  --differential-merge-compile-mode "${BRAINWM_DIFFERENTIAL_MERGE_COMPILE_MODE:-max-autotune-no-cudagraphs}" \
  --population-balance-mode "${BRAINWM_POPULATION_BALANCE_MODE:-combined}" \
  --flow-sdpa-backend flash \
  --fuse-development-attention \
  --fuse-development-target-attention \
  --development-max-padded-tokens 60000 \
  --workers "$workers" \
  --allow-resume-batch-size-change \
  --cuda-prefetch \
  --cuda-timing-interval "$cuda_timing_interval" \
  --telemetry-interval "$telemetry_interval" \
  --gradient-health-execution "${BRAINWM_GRADIENT_HEALTH_EXECUTION:-auto}" \
  --precision bf16 \
  --learning-rate 2e-4 \
  --full-passes 5 \
  --checkpoint "$root/checkpoints/light-full.pt" \
  --state "$root/logs/light-full.state.json" \
  --metrics-dir "$root/results/training/light-full" \
  --checkpoint-every 25 \
  --version-checkpoint-every 1000 \
  --validation-every 1000 \
  --validation-rows 8192 \
  "${sinkhorn_args[@]}" \
  "${resume_args[@]}"
