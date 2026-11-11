from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobrain.genome import CandidateGenome
from scripts import nanobrain_scheduler as scheduler
from scripts.run_nanobrain_baselines import _write_rows


def test_busy_cards_are_not_used_while_quiet_card_has_no_free_slot():
    snapshot = {0: dict(free_mb=10000, external_mb=80000, utilization=100),
                7: dict(free_mb=90000, external_mb=0, utilization=100)}
    assert scheduler.available_slots(snapshot, Counter(), per_gpu=2) == [7, 7]
    assert scheduler.available_slots(snapshot, Counter({7: 1}), per_gpu=2) == [7]
    assert scheduler.available_slots(snapshot, Counter({7: 2}), per_gpu=2) == []


def test_all_busy_fallback_chooses_least_external_load():
    snapshot = {0: dict(free_mb=10000, external_mb=80000, utilization=100),
                1: dict(free_mb=20000, external_mb=70000, utilization=90)}
    assert scheduler.available_slots(snapshot, Counter(), per_gpu=1) == [1]


def test_gpu_snapshot_excludes_our_process_memory(monkeypatch):
    outputs = iter(["0, UUID0, 10000, 100\n7, UUID7, 90000, 100\n",
                    "UUID0, 10, 80000\nUUID7, 20, 1000\n"])
    monkeypatch.setattr(scheduler.subprocess, "check_output", lambda *a, **k: next(outputs))
    result = scheduler.gpu_snapshot([0, 7], {20})
    assert result[0]["external_mb"] == 80000 and result[7]["external_mb"] == 0


def test_runtime_migration_archives_previous_source_and_refuses_model_changes(tmp_path):
    root = tmp_path / "root"
    output = tmp_path / "output"
    names = ["scripts/run_nanobrain_evolution.py", "scripts/nanobrain_scheduler.py"]
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("new code")
    old = output / "source_snapshot" / names[0]
    old.parent.mkdir(parents=True)
    old.write_text("old code")
    frozen = dict(protocol={"steps": 10000}, fingerprint={names[0]: "old", "model": "same"})
    fingerprint = {name: scheduler.sha(root / name) for name in names}
    fingerprint["model"] = "same"
    record = scheduler.migrate_scheduler_fingerprint(output, frozen, fingerprint, root)
    assert (Path(record["archive"]) / "source.before" / names[0]).read_text() == "old code"
    updated = json.loads((output / "protocol.json").read_text())
    assert updated["protocol"] == frozen["protocol"]
    assert updated["fingerprint"] == fingerprint
    assert len(updated["runtime_migrations"]) == 1
    with pytest.raises(ValueError, match="scientific files"):
        scheduler.migrate_scheduler_fingerprint(output, frozen, {**fingerprint, "model": "changed"}, root)


def queue_fixture(tmp_path, monkeypatch, *, adopt=False):
    clock = [0]
    launches = []
    processes = []
    snapshots = {0: dict(free_mb=10000, external_mb=80000, utilization=100),
                 7: dict(free_mb=90000, external_mb=0, utilization=0)}
    monkeypatch.setattr(scheduler, "gpu_snapshot", lambda *a: snapshots)
    monkeypatch.setattr(scheduler.time, "time", lambda: clock[0])
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    experiment = SimpleNamespace(args=SimpleNamespace(gpus=[0, 7], workers_per_gpu=2),
        protocol=dict(tasks=["annotation", "integration"], batch_size=8, validation_batch_size=8,
            precision="bf16", development_cells=4, validation_development_groups=1, eval_interval=500,
            loss_balance="fixed", loss_balance_decay=.99, checkpoint_policy="best_weights",
            checkpoint_selection="best_validation", evaluation_batch_size=8, evaluation_seed=5,
            selection_evaluation_split="test"), state={}, data=tmp_path, output=tmp_path)
    experiment.save = lambda *a: None
    specs = [dict(genome=CandidateGenome(seed=seed).to_dict()) for seed in (1, 2, 3)]

    class Process:
        def __init__(self, delay, complete):
            self.pid = 100 + len(processes)
            self.deadline = clock[0] + delay
            self.complete = complete
            self.done = False
            self.code = 0
            processes.append(self)

        def poll(self):
            if clock[0] < self.deadline:
                return None
            if not self.done:
                self.complete()
                self.done = True
            return self.code

    def receipt_file(item, steps):
        path = Path(item["directory"]) / "receipt.json"
        item["receipt"] = str(path)
        return json.loads(path.read_text()) if path.exists() else None

    def write_receipt(path, task):
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = path / "best.pt"
        checkpoint.write_bytes(b"selected validation weights")
        (path / "receipt.json").write_text(json.dumps(dict(training_task=task, best_step=500,
            completed_steps=10000, best_checkpoint=str(checkpoint), best_validation_quality=.8,
            quality=.7, parameter_count=10, trainable_parameter_count=10, runtime_seconds=10)))

    def launch_train(**kwargs):
        path = kwargs["output"] / kwargs["directory_names"][0]
        index = int(path.name.split("_")[-1])
        task = kwargs["task"]
        delay = 10 if task == "annotation" and index == 0 else 1
        process = Process(delay, lambda: write_receipt(path, task))
        launches.append(("training", task, index, kwargs["gpus"][0], clock[0]))
        return [dict(process=process, pid=process.pid, gpu=kwargs["gpus"][0],
                     directory=str(path), index=0)]

    def launch_evaluate(**kwargs):
        item = kwargs["candidates"][0]
        assert kwargs["evaluation_split"] == "test"
        assert item["selected_step"] == 500
        path = Path(item["directory"]) / "evaluation/benchmark_table.csv"
        item["evaluation"] = str(path)

        def write_table():
            path.parent.mkdir(parents=True, exist_ok=True)
            metric = "accuracy" if item["training_task"] == "annotation" else "integration_composite"
            _write_rows(path, [dict(method="nanobrain-" + item["candidate_id"], task=item["training_task"],
                metric=metric, value=.5, status="complete", dataset="fixture", training_regime="fixture",
                stratum_type="overall", stratum_value="overall")])

        process = Process(1, write_table)
        launches.append(("evaluation", item["training_task"], item["index"], kwargs["gpus"][0], clock[0]))
        return [dict(process=process, pid=process.pid, gpu=kwargs["gpus"][0])]

    monkeypatch.setattr(scheduler, "_load_training_receipt", receipt_file)
    monkeypatch.setattr(scheduler, "audit_training_receipt", lambda r: dict(passed=bool(r), reasons=[]))
    monkeypatch.setattr(scheduler, "_launch_candidates", launch_train)
    monkeypatch.setattr(scheduler, "_launch_evaluations", launch_evaluate)
    # Production uses the unchanged canonical method mapper; keep fixtures exact.
    monkeypatch.setattr(scheduler, "candidate_method", lambda cid: "nanobrain-" + cid)
    return experiment, specs, launches, clock, Process, write_receipt


def test_fast_slots_refill_and_cross_task_work_starts_before_slow_task_ends(tmp_path, monkeypatch):
    experiment, specs, launches, clock, _, _ = queue_fixture(tmp_path, monkeypatch)
    output = tmp_path / "round"
    selector = lambda receipt, policy: Path(receipt["best_checkpoint"])
    result = scheduler.run_task_queue(experiment, specs, output, 10000, selector)
    assert len(result["annotation"]) == len(result["integration"]) == 3
    assert all(gpu == 7 for _, _, _, gpu, _ in launches)
    assert next(t for phase, task, index, gpu, t in launches if phase == "training" and index == 2) < 10
    assert next(t for phase, task, index, gpu, t in launches if phase == "training" and task == "integration") < 10
    assert [r["index"] for r in result["annotation"]] == [0, 1, 2]
    assert all(r["checkpoint_selection"] == "best_validation" for rows in result.values() for r in rows)
    before = list(launches)
    again = scheduler.run_task_queue(experiment, specs, output, 10000, selector)
    assert launches == before and again == result


def test_existing_training_receipt_and_live_worker_are_not_retrained(tmp_path, monkeypatch):
    experiment, specs, launches, clock, Process, write_receipt = queue_fixture(tmp_path, monkeypatch)
    output = tmp_path / "round"
    write_receipt(output / "annotation/candidate_00", "annotation")
    worker = Process(5, lambda: write_receipt(output / "annotation/candidate_01", "annotation"))
    monkeypatch.setattr(scheduler, "adopt_worker", lambda item, records:
        dict(process=worker, pid=worker.pid, gpu=0, adopted=True)
        if item["training_task"] == "annotation" and item["index"] == 1 else None)
    result = scheduler.run_task_queue(experiment, specs, output, 10000,
                                     lambda receipt, policy: Path(receipt["best_checkpoint"]))
    trained = [(task, index) for phase, task, index, gpu, t in launches if phase == "training"]
    assert ("annotation", 0) not in trained and ("annotation", 1) not in trained
    assert len(result["annotation"]) == 3
    assert experiment.state["scheduler"]["counters"]["adopted_workers"] == 1
    assert experiment.state["scheduler"]["counters"]["reused_training"] == 1


def test_shared_fallback_has_five_total_slots_not_five_per_busy_gpu():
    snapshot = {gpu: dict(free_mb=20000, external_mb=70000, utilization=100) for gpu in range(8)}
    assert scheduler.available_slots(snapshot, Counter({0: 5}), per_gpu=5) == []
    assert len(scheduler.available_slots(snapshot, Counter({0: 4}), per_gpu=5)) == 1


@pytest.mark.parametrize("oom,permanent", [(True, False), (False, True)])
def test_evaluation_failure_retries_without_retraining_or_stopping_other_candidates(
        tmp_path, monkeypatch, oom, permanent):
    experiment, specs, launches, clock, Process, _ = queue_fixture(tmp_path, monkeypatch)
    original = scheduler._launch_evaluations
    seen = [0]

    def launch(**kwargs):
        jobs = original(**kwargs)
        item = kwargs["candidates"][0]
        if item["training_task"] == "annotation" and item["index"] == 0:
            seen[0] += 1
            if permanent or seen[0] == 1:
                proc = jobs[0]["process"]
                proc.code = 1
                log = Path(item["evaluation"]).parent / "run.log"

                def error():
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text("torch.OutOfMemoryError: CUDA out of memory" if oom else "RuntimeError: invalid output")

                proc.complete = error
        return jobs

    monkeypatch.setattr(scheduler, "_launch_evaluations", launch)
    result = scheduler.run_task_queue(experiment, specs, tmp_path / "round", 10000,
                                     lambda receipt, policy: Path(receipt["best_checkpoint"]))
    assert len([x for x in launches if x[0] == "training"]) == 6
    assert len(result["integration"]) == 3 and not any(x.get("failed") for x in result["integration"])
    if oom:
        assert not result["annotation"][0].get("failed")
        assert experiment.state["scheduler"]["counters"]["evaluation_retries"] == 1
    else:
        assert result["annotation"][0]["failed"]
        assert experiment.state["scheduler"]["counters"]["failed_evaluation"] == 1


def test_memory_budget_prevents_two_large_evaluations_on_small_free_gpu(tmp_path, monkeypatch):
    experiment, specs, launches, clock, _, _ = queue_fixture(tmp_path, monkeypatch)
    experiment.args.workers_per_gpu = 5
    monkeypatch.setattr(scheduler, "gpu_snapshot", lambda *a:
        {7: dict(free_mb=12288, external_mb=0, own_mb=0, utilization=0)})
    result = scheduler.run_task_queue(experiment, specs, tmp_path / "round", 10000,
                                     lambda receipt, policy: Path(receipt["best_checkpoint"]))
    starts = [x[4] for x in launches if x[0] == "evaluation"]
    assert len(starts) == 6 and len(set(starts)) == 6
    assert all(len(rows) == 3 for rows in result.values())
