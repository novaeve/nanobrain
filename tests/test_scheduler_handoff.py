"""A scheduling amendment must preserve live training and the scientific recipe."""

import json
from pathlib import Path

import pytest

from nanobrain.infra import contracts as C
from nanobrain.infra import handoff as H
from nanobrain.infra.workflow import Workflow


def stage_plan(tmp_path):
    directory = tmp_path / "jobs" / "candidate" / "0" / "joint"
    directory.mkdir(parents=True)
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    args = ["train", "--bundle", str(bundle), "--seed", "0"]
    stage = {
        "directory": str(directory),
        "mode": "train",
        "pid": 123,
        "process_start": "original-start",
        "gpu": 4,
        "reservation_mb": 49152,
        "arguments": args,
        "bundle": str(bundle),
        "bundle_sha256": C.sha(bundle),
        "attempt": 2,
    }
    plan = {
        "version": 1,
        "stages": [stage],
        "reservations": [stage],
        "previous_controller": {"pid": 999, "process_start": "controller-start"},
    }
    C.save(tmp_path / "scheduler-handoff.json", plan)
    return stage, args


def test_sixteen_slot_config_preserves_existing_scientific_settings(tmp_path):
    cfg = C.load_config()
    assert cfg["max_workers"] == 16 and cfg["per_gpu_workers"] == 3
    assert cfg["gpu_ids"] == list(range(5))
    assert cfg["cpu_probe_workers"] == 4 and cfg["task_evaluation_workers"] == 2
    assert cfg["generations"] == 30 and cfg["candidates_per_generation"] == 10
    assert cfg["stop_at"] == 10000 and cfg["seeds"] == [0]
    path = tmp_path / "config.json"
    for key, value in [
        ("max_workers", 17),
        ("per_gpu_workers", 4),
        ("gpu_ids", [5]),
        ("cpu_probe_workers", 9),
        ("task_evaluation_workers", 5),
    ]:
        invalid = dict(cfg, **{key: value})
        C.save(path, invalid)
        with pytest.raises(ValueError):
            C.load_config(path)


def test_running_stage_memory_is_reserved_and_receipt_is_reused_without_retraining(tmp_path, monkeypatch):
    stage, args = stage_plan(tmp_path)
    monkeypatch.setattr(H, "process_start", lambda pid: "original-start" if pid == 123 else None)
    workflow = Workflow(C.load_config(), tmp_path)
    key = stage["directory"]
    assert workflow.resources.leases[key]["pid"] == 123
    assert workflow.resources.leases[key]["reserve_mib"] == 49152
    # The old sandbox finishes after the new scheduler is constructed.
    artifact = Path(key) / "latest.pt"
    artifact.write_bytes(b"synthetic checkpoint")
    receipt = {"status": "complete", "seed": 0, "step": 10000, "artifact_hashes": {"latest.pt": C.sha(artifact)}}
    C.save(Path(key) / "receipt.json", receipt)
    from nanobrain.infra import sandbox

    monkeypatch.setattr(sandbox, "launch", lambda *a, **k: pytest.fail("restarted an already-running trainer"))
    retired = []
    monkeypatch.setattr(H, "retire_parent", lambda *a: retired.append(True))
    assert workflow.worker(Path(key), args, []) == receipt
    assert not workflow.resources.leases and not workflow.activity and retired == [True]


def test_adoption_rejects_recipe_mutation_and_never_attaches_to_a_reused_pid(tmp_path, monkeypatch):
    stage, args = stage_plan(tmp_path)
    with pytest.raises(ValueError, match="recipe changed"):
        H.wait_for_stage(stage, args + ["unexpected"], lambda d: {}, 1)
    monkeypatch.setattr(H, "process_start", lambda pid: "different-process")
    with pytest.raises(RuntimeError, match="exited without"):
        H.wait_for_stage(stage, args, lambda d: {}, 1)
    workflow = Workflow(C.load_config(), tmp_path)
    assert not workflow.adopted_stages and not workflow.resources.leases


def test_handoff_cannot_mount_other_campaign_jobs(tmp_path, monkeypatch):
    stage_plan(tmp_path)
    path = tmp_path / "scheduler-handoff.json"
    plan = json.loads(path.read_text())
    plan["reservations"][0]["directory"] = str(tmp_path.parent / "other-campaign")
    C.save(path, plan)
    with pytest.raises(ValueError, match="outside this campaign"):
        Workflow(C.load_config(), tmp_path)


@pytest.mark.parametrize("gpu", [5, 6, 7])
def test_a_live_worker_on_a_disallowed_gpu_cannot_be_adopted(tmp_path, gpu):
    stage_plan(tmp_path)
    path = tmp_path / "scheduler-handoff.json"
    plan = json.loads(path.read_text())
    plan["stages"][0]["gpu"] = gpu
    C.save(path, plan)
    with pytest.raises(ValueError, match="GPU is not authorized"):
        Workflow(C.load_config(), tmp_path)


@pytest.mark.parametrize("second_state", ["T", "R"])
def test_nested_handoffs_retire_only_verified_paused_controller_keepers(tmp_path, monkeypatch, second_state):
    from types import SimpleNamespace

    real_path = H.Path

    def path(value):
        if str(value).startswith("/proc/"):
            state = second_state if "/888/" in str(value) else "T"
            return SimpleNamespace(read_text=lambda: f"999 (python) {state}")
        return real_path(value)

    monkeypatch.setattr(H, "Path", path)
    monkeypatch.setattr(H, "process_start", lambda pid: "keeper" if pid in [999, 888] else "reused-pid")
    killed = []
    monkeypatch.setattr(H.os, "kill", lambda pid, sig: killed.append(pid))
    parents = [{"pid": pid, "process_start": "keeper"} for pid in [999, 888, 777]]
    plan = {"stages": [], "previous_controller": parents[0], "previous_controllers": parents}
    if second_state == "R":
        with pytest.raises(ValueError, match="explicitly paused"):
            H.retire_parent(tmp_path, plan)
        assert killed == []
    else:
        H.retire_parent(tmp_path, plan)
        assert sorted(killed) == [888, 999] and plan["retired_controller_pids"] == [888, 999]
