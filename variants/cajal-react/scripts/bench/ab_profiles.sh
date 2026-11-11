#!/usr/bin/env bash
# Interleaved A/B step profiles on one GPU: ab_profiles.sh GPU OLD_COMMIT REPS NAME1 "ARGS1" NAME2 "ARGS2" ...
# A name starting with "old:" runs model code from OLD_COMMIT; the rest run the working tree.
gpu=$1 old=$2 reps=$3
shift 3
args=("$@")
cd "$(dirname "$0")/../.." && source env.sh && export PYTHONPATH=$PWD
mkdir -p runs/ab && cp brainwm_unified/model.py runs/ab/model_new.py
for rep in $(seq "$reps"); do
  for ((i = 0; i < ${#args[@]}; i += 2)); do
    name=${args[i]} extra=${args[i + 1]}
    if [[ $name == old:* ]]; then git show "$old:brainwm_unified/model.py" >brainwm_unified/model.py; else cp runs/ab/model_new.py brainwm_unified/model.py; fi
    r=$(eval CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 2400 .venv/bin/python scripts/bench/profile_step.py \
      --config ../BrainWM-Unified/runs/unified_v0_1m/config.json --out "runs/ab/${name#old:}_$rep" --warmup 5 --steps 6 "$extra" 2>&1 |
      grep -E -m1 "== step|OutOfMemory")
    echo "$name rep$rep: $r"
  done
done
cp runs/ab/model_new.py brainwm_unified/model.py
echo AB_DONE
