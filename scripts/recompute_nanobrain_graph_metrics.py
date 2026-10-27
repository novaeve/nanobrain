#!/usr/bin/env python3
"""Re-evaluate archived graph metrics from predictions without fitting models.

Writes a derived campaign; never changes source predictions/tables/manifests.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brainwm_benchmark.metrics import evaluate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/nanobrain/final-100k-20260916/test-v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be new; preserve previous metric versions")
    args.output.mkdir(parents=True)
    frozen_sources = {}
    computed = {}
    graph_tasks = {"integration", "spatial_niche"}
    for table in sorted(args.source.glob("*/benchmark_table.csv")):
        frozen_sources[str(table)] = sha(table)
        rows = read(table)
        for row in rows:
            if row["task"] not in graph_tasks:
                continue
            path = Path(row["prediction_artifact"])
            key = (row["method"], row["task"])
            if key not in computed:
                frozen_sources[str(path)] = sha(path)
                with np.load(path, allow_pickle=False) as payload:
                    computed[key] = evaluate(row["task"], payload)
                print(f"evaluated {row['method']} {row['task']}", flush=True)
            value = computed[key][row["metric"]]
            row["value"] = float(value) if math.isfinite(value) else ""
            row["status"] = "complete" if math.isfinite(value) else "not_defined"
            row["notes"] += "; graph_metrics_v2: nearest non-self neighbors, recomputed from frozen predictions"
        write(args.output / table.parent.name / table.name, rows)
    final = args.source / "final_benchmark_long.csv"
    frozen_sources[str(final)] = sha(final)
    rows = read(final)
    deltas = []
    for row in rows:
        if row["task"] not in graph_tasks:
            continue
        old = row["mean"]
        if row["source_group"] == "nanobrain_three_seed":
            values = [m[row["metric"]] for (method, task), m in computed.items()
                      if method.startswith("nanobrain_candidate-") and task == row["task"]]
            if len(values) != 3:
                raise ValueError("expected three historical nanoBrain checkpoints")
            row["mean"] = statistics.mean(values)
            row["std"] = statistics.stdev(values)
        else:
            row["mean"] = computed[(row["method"], row["task"])][row["metric"]]
        if not math.isfinite(float(row["mean"])):
            row["mean"] = ""
            row["status"] = "not_defined"
        row["source_table"] = "receipt.json (frozen source predictions and evaluator hash)"
        row["notes"] += "; graph_metrics_v2 recomputation; weights and predictions unchanged"
        deltas.append(dict(method=row["method"], task=row["task"], metric=row["metric"],
                           old=old, new=row["mean"],
                           delta=float(row["mean"]) - float(old) if old and row["mean"] != "" else ""))
    write(args.output / "final_benchmark_long.csv", rows)
    write(args.output / "graph_metric_changes.csv", deltas)
    if any(sha(path) != value for path, value in frozen_sources.items()):
        raise RuntimeError("source artifact changed during re-evaluation")
    receipt = dict(status="complete", metric_version="graph_metrics_v2", source=str(args.source.resolve()),
                   training_runs=0, used_for_evolution_selection=False, test_policy="descriptive re-evaluation of previously published predictions",
                   graph_prediction_artifacts=len(computed), graph_metric_rows=len(deltas), total_metric_rows=len(rows),
                   source_artifacts_sha256=frozen_sources,
                   evaluator_sha256=sha(ROOT / "src/brainwm_benchmark/metrics.py"),
                   table_sha256=sha(args.output / "final_benchmark_long.csv"))
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(
        "# 历史 baseline 近邻指标复算\n\n"
        "修复近邻查询重复排除自身、跳过最近有效邻居的问题。仅从已有预测重算整合与空间指标，未训练模型。"
        "原 test 表与预测保持原样；本表为已公开 test 的描述性复评，不用于演化选择。\n\n"
        f"复算 {len(computed)} 份预测、{len(deltas)} 条指标；完整表 {len(rows)} 条。"
        "新实验使用 graph_metrics_v2，应与此版本比较。\n\n"
        "完整数据见 `final_benchmark_long.csv`，逐指标变化见 `graph_metric_changes.csv`。\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_artifacts_sha256"}, indent=2))


if __name__ == "__main__":
    main()
