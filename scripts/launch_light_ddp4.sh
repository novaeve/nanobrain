#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/env.sh"
run_id="${BRAINWM_RUN_ID:-light-full-v2}"
mkdir -p "$root/checkpoints" "$root/logs" "$root/triton" "$root/results/training/$run_id"

gpu_ids="${BRAINWM_GPU_IDS:-0,1}"
IFS=',' read -r -a gpu_id_array <<< "$gpu_ids"
gpu_count="${#gpu_id_array[@]}"
if [[ "$gpu_count" -lt 1 || "$gpu_count" -gt 4 ]]; then
  echo "BRAINWM_GPU_IDS must contain between one and four physical GPU IDs" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export OMP_NUM_THREADS="${BRAINWM_OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${BRAINWM_MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${BRAINWM_OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export TRITON_CACHE_DIR="$root/triton"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "$$" > "$root/logs/$run_id.pid"

# Phase timing synchronizes every step and is intended only for diagnosis.
# Production keeps the CUDA prefetch stream overlapped with model execution.
timing_args=()
if [[ "${BRAINWM_CUDA_PHASE_TIMING:-0}" == "1" ]]; then
  timing_args+=(--cuda-phase-timing)
fi
telemetry_interval="${BRAINWM_TELEMETRY_INTERVAL:-10}"
cuda_timing_interval="${BRAINWM_CUDA_TIMING_INTERVAL:-$telemetry_interval}"
workers="${BRAINWM_WORKERS:-2}"
router_dynamic_args=(--no-router-compile-dynamic)
if [[ "${BRAINWM_ROUTER_COMPILE_DYNAMIC:-0}" == "1" ]]; then
  router_dynamic_args=(--router-compile-dynamic)
fi

# PyTorch's reducer optimizations are intentionally opt-in.  Static-graph
# mode is only valid after a representative multi-GPU smoke has confirmed
# that the selected task/missingness pattern keeps the autograd graph stable.
ddp_tuning_args=()
if [[ "${BRAINWM_DDP_FIND_UNUSED_PARAMETERS:-1}" == "1" ]]; then
  ddp_tuning_args+=(--ddp-find-unused-parameters)
else
  ddp_tuning_args+=(--no-ddp-find-unused-parameters)
fi
if [[ "${BRAINWM_DDP_STATIC_GRAPH:-0}" == "1" ]]; then
  ddp_tuning_args+=(--ddp-static-graph)
fi
if [[ -n "${BRAINWM_DDP_BUCKET_CAP_MB:-}" ]]; then
  ddp_tuning_args+=(--ddp-bucket-cap-mb "$BRAINWM_DDP_BUCKET_CAP_MB")
fi

# Sinkhorn compilation is enabled by default for full runs. The first graph
# build is expensive, so warm the fixed Light-width graphs once, serially, before
# torchrun so ranks do not all invoke TorchInductor at the same time.
sinkhorn_args=()
if [[ "${BRAINWM_COMPILE_SINKHORN:-1}" == "1" ]]; then
  sinkhorn_args+=(--compile-sinkhorn --sinkhorn-compile-buckets 8 16 32 64)
  if [[ "${BRAINWM_WARMUP_SINKHORN:-1}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id_array[0]}" \
      TORCHINDUCTOR_COMPILE_THREADS=1 \
      "$root/.venv/bin/python" "$root/scripts/warmup_sinkhorn.py" \
      --width 64 --buckets 8 16 32 64 \
      --output "$root/results/training/$run_id/sinkhorn_warmup.json"
  fi
fi

if [[ "${BRAINWM_WARMUP_ATTENTION_KERNELS:-1}" == "1" ]]; then
  # Warm each selected device serially.  Triton autotune uses a shared cache;
  # concurrent rank warmups contend on its compile locks and can put the first
  # DDP step behind minutes of redundant compilation.
  if [[ "${BRAINWM_WARMUP_ALL_GPUS:-1}" == "1" ]]; then
    for physical_gpu in "${gpu_id_array[@]}"; do
      CUDA_VISIBLE_DEVICES="$physical_gpu" \
        "$root/.venv/bin/python" "$root/scripts/warmup_flash_bla.py" \
        --heads 8 --head-dim 64 --device cuda:0 \
        --output "$root/results/training/$run_id/flash_bla_warmup-gpu${physical_gpu}.json"
    done
  else
    "$root/.venv/bin/python" "$root/scripts/warmup_flash_bla.py" \
      --heads 8 --head-dim 64 --device cuda:0 \
      --output "$root/results/training/$run_id/flash_bla_warmup.json"
  fi
fi

resume_args=()
if [[ -e "$root/checkpoints/$run_id.pt" && ! -s "$root/checkpoints/$run_id.pt" ]]; then
  echo "Refusing to resume from empty checkpoint: $root/checkpoints/$run_id.pt" >&2
  exit 3
elif [[ -s "$root/checkpoints/$run_id.pt" ]]; then
  resume_args=(--resume "$root/checkpoints/$run_id.pt")
fi

exec "$root/.venv/bin/torchrun" --standalone --nproc-per-node="$gpu_count" \
  "$root/scripts/train_foundation_moe.py" \
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
  --precision bf16 \
  --learning-rate 2e-4 \
  --minimum-learning-rate 2e-5 \
  --warmup-fraction 0.01 \
  --full-passes 5 \
  --seed 20260912 \
  --shuffle-training \
  --shuffle-seed 20260912 \
  --shard-balance-waves 8 \
  --task-mask-probability 0.5 \
  --all-context-mask-probability 0.15 \
  --context-mask-probability 0.15 \
  "${ddp_tuning_args[@]}" \
  --cuda-prefetch \
  --cuda-timing-interval "$cuda_timing_interval" \
  --telemetry-interval "$telemetry_interval" \
  --gradient-health-execution "${BRAINWM_GRADIENT_HEALTH_EXECUTION:-auto}" \
  --checkpoint "$root/checkpoints/$run_id.pt" \
  --state "$root/logs/$run_id.state.json" \
  --metrics-dir "$root/results/training/$run_id" \
  --checkpoint-every 25 \
  --version-checkpoint-every 1000 \
  --validation-every 1000 \
  --validation-rows 8192 \
  --validation-shards 96 \
  --validation-seed 20260912 \
  --validation-development-groups 32 \
  "${sinkhorn_args[@]}" \
  "${timing_args[@]}" \
  "${resume_args[@]}"
