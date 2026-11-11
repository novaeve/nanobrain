#!/usr/bin/env bash
# Evaluate runs as they train: every 5 min, run zero-shot inference on each new milestone checkpoint
# (stepN.pt) of the given runs, and re-score the table whenever a prediction or a baseline is added.
#   scripts/eval_watch.sh GPU FINAL_STEP RUN=NAME... [-- EXTRA_PRED=NAME=PATH...]
# Writes runs/eval/<tag><run>_step<N>.npz, runs/eval/<tag>table_latest/summary.csv and timestamped copies.
# EVAL_DATA picks the evaluation set (default eval_10k), EVAL_TAG prefixes the outputs (one watcher per set),
# EVAL_RUNS is the directory holding the runs (default /data/brainwm/BrainWM-Unified/runs).
# EVAL_MODE=each scores every checkpoint once into <tag>scores/<name>/ and merges them (trajectories).
# EVAL_TASK=genes|spatial runs the imputation tasks instead (predict_eval --task, eval_imputation.py;
# implies EVAL_MODE=each); the default, embed, is annotation + integration.
# Exits once every run has FINAL_STEP scored and all four baselines are fitted.
gpu=$1 final=$2
shift 2
runs=() extra=()
while [ $# -gt 0 ]; do
  if [ "$1" = -- ]; then shift; extra=("$@"); break; fi
  runs+=("$1"); shift
done
cd "$(dirname "$0")/.." && source env.sh && export PYTHONPATH=$PWD
R=${EVAL_RUNS:-/data/brainwm/BrainWM-Unified/runs} E=runs/eval
DATA=${EVAL_DATA:-/data/brainwm/eval_10k} TAG=${EVAL_TAG:-} TASK=${EVAL_TASK:-embed}
[ "$TASK" = embed ] || EVAL_MODE=each
last=""
while true; do
  for spec in "${runs[@]}"; do
    run=${spec%%=*}
    for ck in "$R/$run"/step*.pt; do
      [ -f "$ck" ] && [ $(($(date +%s) - $(stat -c %Y "$ck"))) -gt 120 ] || continue # fully written
      s=$(basename "$ck" .pt)
      [ -f "$E/$TAG${run}_$s.npz" ] && continue
      echo "$(date +%T) predict $TAG$run $s" >> $E/${TAG}watch.log
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/predict_eval.py --checkpoint "$ck" --data "$DATA" --task "$TASK" \
        --out "$E/$TAG${run}_$s.npz" >> $E/${TAG}watch.log 2>&1
    done
  done
  preds=() done_all=1
  for spec in "${runs[@]}"; do
    run=${spec%%=*} name=${spec#*=}
    for f in "$E/$TAG${run}"_step*.npz; do
      [ -f "$f" ] || continue
      s=${f##*_step}; preds+=(--pred "$name@${s%.npz}=$f")
    done
    [ -f "$E/$TAG${run}_step$final.npz" ] || done_all=0
  done
  for p in "${extra[@]}"; do preds+=(--pred "$p"); done
  n_base=$(ls $E/baselines/lr_*.npz $E/baselines/xgb_*.pkl 2>/dev/null | wc -l)
  state="${preds[*]} $n_base"
  if [ "${EVAL_MODE:-all}" = each ]; then
    # Score each prediction once, cached per folder, then merge (scores are per embedding, so this
    # equals scoring them together; cost stays flat as checkpoints accumulate).
    for ((i = 1; i < ${#preds[@]}; i += 2)); do
      pair=${preds[i]} name=${preds[i]%%=*}
      [ -f "$E/${TAG}scores/$name/summary.csv" ] && continue
      echo "$(date +%T) score $name" >> $E/${TAG}watch.log
      if [ "$TASK" = embed ]; then
        (source scripts/eval_env.sh && CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_zero_shot.py --pred "$pair" \
          --data "$DATA" --out "$E/${TAG}scores/$name" > "$E/${TAG}scores_$name.log" 2>&1)
      else
        (source scripts/eval_env.sh && python scripts/eval_imputation.py --task "$TASK" --pred "$pair" \
          --data "$DATA" --cache $E/imputation_baselines --out "$E/${TAG}scores/$name" > "$E/${TAG}scores_$name.log" 2>&1)
      fi
    done
    .venv/bin/python scripts/collect_scores.py $E/${TAG}scores --out $E/${TAG}table_latest >> $E/${TAG}watch.log 2>&1
    last=$state
  elif [ ${#preds[@]} -gt 0 ] && [ "$state" != "$last" ]; then
    echo "$(date +%T) score: ${#preds[@]} prediction args, $n_base baselines" >> $E/${TAG}watch.log
    (source scripts/eval_env.sh && CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_zero_shot.py "${preds[@]}" \
      --data "$DATA" --out $E/${TAG}table_latest > $E/${TAG}table_latest.log 2>&1) &&
      cp $E/${TAG}table_latest/summary.csv "$E/${TAG}summary_$(date +%m%d_%H%M).csv" && last=$state
  fi
  [ "$TASK" = embed ] || n_base=4  # imputation baselines are computed by eval_imputation.py itself
  [ $done_all = 1 ] && [ "$n_base" = 4 ] && [ "$state" = "$last" ] && { echo "$(date +%T) all done" >> $E/${TAG}watch.log; exit 0; }
  sleep 300
done
