#!/usr/bin/env bash
# Move a training run onto the current code (or another GPU) without losing work: if it is running,
# wait until it writes its next checkpoint, stop it, then resume from latest.pt with the argv saved
# in its record.json. Blocks until the resumed run ends.
#   scripts/resume_run.sh RUN_DIR GPU
run=${1%/}
gpu=$2
cd "$(dirname "$0")/.." && source env.sh && export PYTHONPATH=$PWD

py=.venv/bin/python  # the system python3 may predate shlex.join
step() { $py -c "import json, sys; print(json.load(open(sys.argv[1])).get('step', 0))" "$run/record.json"; }
# the training process itself, not a shell that launched it (its command line holds the same text)
pid=$(for p in $(pgrep -f -- "brainwm_unified.train .*--out $run( |$)"); do
  [[ $(ps -o comm= -p "$p") == python* ]] && echo "$p"; done | head -1)
if [ -n "$pid" ]; then
  start=$(step)
  while [ "$(step)" = "$start" ] && kill -0 "$pid" 2>/dev/null; do sleep 30; done
  kill "$pid" 2>/dev/null
  while kill -0 "$pid" 2>/dev/null; do sleep 5; done
fi
argv=$($py -c "import json, shlex, sys; print(shlex.join(json.load(open(sys.argv[1]))['argv']))" "$run/record.json")
[ -n "$argv" ] || { echo "no argv in $run/record.json" >&2; exit 1; }
echo "$(date +%T) resume $run from step $(step) on GPU $gpu" | tee -a "$run.log"
eval CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m brainwm_unified.train "$argv" --resume >> "$run.log" 2>&1
