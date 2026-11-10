"""A scheduling amendment must preserve live training and the scientific recipe."""

import json
from pathlib import Path

import pytest

from cajal_research import contracts as C
from cajal_research import handoff as H
from cajal_research.workflow import Workflow


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


def test_twelve_slot_config_preserves_existing_scientific_settings(tmp_path):
    cfg = C.load_config()
    assert cfg["max_workers"] == 12 and cfg["per_gpu_workers"] == 3
    assert cfg["gpu_ids"] == [0, 1, 2, 3, 4]
    assert cfg["generations"] == 30 and cfg["candidates_per_generation"] == 10
    assert cfg["stop_at"] == 10000 and cfg["seeds"] == [0]
    path = tmp_path / "config.json"
    for key, value in [("max_workers", 13), ("per_gpu_workers", 4), ("gpu_ids", [5])]:
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
    from cajal_research import sandbox

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
