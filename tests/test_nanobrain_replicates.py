import csv
import json

import pytest

from scripts import aggregate_nanobrain_replicates as aggregate


def _write_run(root, seed, candidate_id, offset=0.0, *, test_opened=False):
    root.mkdir()
    fields = [
        "task", "method", "status", "metric", "value", "direction",
        "stratum_type", "stratum_value", "eval_rows",
    ]
    with (root / "benchmark_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (task, metric) in enumerate(sorted(aggregate.KEY_METRICS)):
            writer.writerow({
                "task": task,
                "method": f"nanobrain_candidate-{candidate_id}",
                "status": "complete",
                "metric": metric,
                "value": 1.0 + index + offset,
                "direction": "max",
                "stratum_type": "overall",
                "stratum_value": "overall",
                "eval_rows": 10,
            })
    (root / "run_manifest.json").write_text(
        json.dumps({"test_opened": test_opened, "checkpoint": f"seed-{seed}.pt"}) + "\n",
        encoding="utf-8",
    )


def test_three_seed_aggregate_and_metric_directions(tmp_path, monkeypatch):
    arguments = ["aggregate_nanobrain_replicates.py"]
    for index, seed in enumerate((11, 22, 33)):
        candidate_id = f"candidate{index}"
        run = tmp_path / f"run{index}"
        _write_run(run, seed, candidate_id, offset=float(index))
        arguments.extend(["--run", f"{seed}:{candidate_id}:{run}"])
    output = tmp_path / "aggregate"
    arguments.extend(["--output", str(output)])
    monkeypatch.setattr("sys.argv", arguments)

    aggregate.main()

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["independent_seeds"] == [11, 22, 33]
    assert receipt["all_key_metrics_have_three_seeds"] is True
    assert receipt["key_metrics"]["spatial_completion.heldout_rmse"]["direction"] == "min"
    assert receipt["key_metrics"]["development.population_mse"]["direction"] == "min"
    assert all(metric["n_seeds"] == 3 for metric in receipt["key_metrics"].values())


def test_three_seed_aggregate_rejects_unsealed_source(tmp_path, monkeypatch):
    arguments = ["aggregate_nanobrain_replicates.py"]
    for index, seed in enumerate((11, 22, 33)):
        candidate_id = f"candidate{index}"
        run = tmp_path / f"run{index}"
        _write_run(run, seed, candidate_id, test_opened=index == 2)
        arguments.extend(["--run", f"{seed}:{candidate_id}:{run}"])
    arguments.extend(["--output", str(tmp_path / "aggregate")])
    monkeypatch.setattr("sys.argv", arguments)

    with pytest.raises(ValueError, match="sealed test"):
        aggregate.main()
