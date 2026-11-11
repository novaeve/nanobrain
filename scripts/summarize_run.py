"""Print a run's per-task loss curve and latest gradient probe.

python scripts/summarize_run.py runs/NAME [--every 100]
"""

import argparse
import json
import os

from brainwm_unified.objectives import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--every", type=int, default=1, help="print every n-th logged row")
    args = ap.parse_args()

    with open(os.path.join(args.run, "train.jsonl")) as fh:
        rows = [json.loads(line) for line in fh]
    print("step loops " + " ".join(f"{t:>11s}" for t in TASKS) + "  ct_acc anat_acc  s/step")
    for r in rows[:: args.every] + rows[-1:]:
        losses = " ".join(f"{r['loss/' + t]:11.4f}" for t in TASKS)
        print(
            f"{r['step']:>4d} {r['loops']:>5d} {losses}  {r['cell_type/acc']:.3f}  {r['anatomy/acc']:.3f}"
            f"  {r['sec_per_step']:.3f}"
        )
    probe_path = os.path.join(args.run, "grad_probe.jsonl")
    if os.path.exists(probe_path):
        with open(probe_path) as fh:
            probes = [json.loads(line) for line in fh]
        if probes:
            last = probes[-1]
            print(f"\ngradient probe at step {last['step']}:")
            for k, v in last.items():
                if k != "step" and k.count("/") == 1:  # whole-encoder entries; per-layer ones carry a group
                    print(f"  {k:32s} {v: .4f}")


if __name__ == "__main__":
    main()
