#!/usr/bin/env python3
"""Report one structured snapshot for a self-managed BrainWM training run."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def tail_jsonl(path, count):
    records = deque(maxlen=count)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    return list(records)


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def gpu_snapshot(selected):
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    selected = set(selected)
    result = []
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        index = int(fields[0])
        if selected and index not in selected:
            continue
        result.append({
            "index": index,
            "name": fields[1],
            "utilization_percent": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
            "power_w": float(fields[5]),
            "temperature_c": int(fields[6]),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="light-full-v2")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.window < 1:
        raise SystemExit("--window must be positive")

    root = args.root.resolve()
    metrics = root / "results" / "training" / args.run_id
    pid_path = root / "logs" / f"{args.run_id}.pid"
    state_path = root / "logs" / f"{args.run_id}.state.json"
    checkpoint_path = root / "checkpoints" / f"{args.run_id}.pt"
    pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
    state = read_json(state_path)
    steps = tail_jsonl(metrics / "train_steps.jsonl", args.window)
    validations = tail_jsonl(metrics / "validation.jsonl", 1)
    events = tail_jsonl(metrics / "events.jsonl", 1)
    gpu_ids = []
    if events:
        value = events[-1].get("cuda_visible_devices", "")
        gpu_ids = [int(item) for item in value.split(",") if item.strip()]

    training = None
    if steps:
        last = steps[-1]
        average = lambda values: sum(values) / len(values)
        cells_s = average([row["distributed"]["global_cells_s"] for row in steps])
        distributed_rows=[row["distributed"] for row in steps]
        density_ratios=[]; step_time_ratios=[]
        for distributed in distributed_rows:
            densities=distributed.get("rank_token_density") or []
            times=distributed.get("rank_total_s") or []
            if densities and min(densities)>0:
                density_ratios.append(max(densities)/min(densities))
            if times and min(times)>0:
                step_time_ratios.append(max(times)/min(times))
        total_rows = int(last["train_rows"])
        total_passes = int((events[-1].get("argv") or {}).get("full_passes", 1)) if events else 1
        completed_rows = int(last["full_pass"]) * total_rows + int(last["rows_in_pass"])
        remaining_rows = max(0, total_passes * total_rows - completed_rows)
        training = {
            "step_window": [steps[0]["step"], last["step"]],
            "last_timestamp_utc": last["timestamp_utc"],
            "full_pass_index": last["full_pass"],
            "rows_in_pass": last["rows_in_pass"],
            "train_rows": total_rows,
            "campaign_progress_percent": 100 * float(last["progress_fraction"]),
            "average_global_cells_s": cells_s,
            "average_global_tokens_s": average([
                row["distributed"]["global_tokens_s"] for row in steps
            ]),
            "average_total_loss": average([
                row["optimization"]["total_loss"] for row in steps
            ]),
            "nonfinite_steps": sum(bool(row["optimization"]["nonfinite"]) for row in steps),
            "average_attention_utilization": average([
                row["batch"]["attention_token_utilization"] for row in steps
            ]),
            "average_rank_token_density_ratio": (
                average(density_ratios) if density_ratios else None
            ),
            "average_rank_step_time_ratio": (
                average(step_time_ratios) if step_time_ratios else None
            ),
            "peak_allocated_gib": max(
                row["gpu_memory_bytes"]["peak_allocated"] for row in steps
            ) / 2**30,
            "learning_rate": last["optimization"]["learning_rate"],
            "eta_hours_at_window_rate": remaining_rows / max(cells_s, 1e-12) / 3600,
        }

    checkpoint = None
    if checkpoint_path.exists():
        stat = checkpoint_path.stat()
        checkpoint = {
            "path": str(checkpoint_path),
            "bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
        }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "pid": pid,
        "process_alive": process_alive(pid),
        "state": state,
        "training": training,
        "latest_validation": validations[-1] if validations else None,
        "checkpoint": checkpoint,
        "gpus": gpu_snapshot(gpu_ids),
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
