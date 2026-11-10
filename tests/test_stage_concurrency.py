"""Parallel task pipelines share a global stage budget and preserve running jobs."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from cajal_research import contracts as C
from cajal_research import handoff as H
from cajal_research.workflow import Workflow


def test_global_stage_cap_also_covers_cpu_stages(tmp_path, monkeypatch):
    cfg = C.load_config()
    workflow = Workflow(cfg, tmp_path)
    counters = {"active": 0, "peak": 0, "calls": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(cfg["max_workers"])

    def run(*args, **kwargs):
        with lock:
            counters["active"] += 1
            counters["calls"] += 1
            counters["peak"] = max(counters["peak"], counters["active"])
            call = counters["calls"]
        if call <= cfg["max_workers"]:
            try:
                barrier.wait(timeout=2)
            except threading.BrokenBarrierError:
                pass
        time.sleep(0.02)
        with lock:
            counters["active"] -= 1
        return {"complete": True}

    monkeypatch.setattr(workflow, "_run_stage", run)
    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(lambda i: workflow.worker(tmp_path / str(i), ["probe"], [], cpu=True), range(20)))
    assert len(results) == 20 and counters["peak"] == cfg["max_workers"]
    # Every released stage slot remains usable after completion.
    assert workflow.worker_slots.acquire(blocking=False)


def test_an_adopted_worker_occupies_a_global_slot_until_its_receipt(tmp_path, monkeypatch):
    d = tmp_path / "jobs/candidate/0/joint"
    d.mkdir(parents=True)
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    stage = {
        "directory": str(d),
        "mode": "train",
        "pid": 123,
        "process_start": "original",
        "gpu": 6,
        "reservation_mb": 32768,
        "attempt": 1,
        "bundle": str(bundle),
        "bundle_sha256": C.sha(bundle),
        "arguments": ["train", "--bundle", str(bundle), "--seed", "0"],
    }
    C.save(tmp_path / "scheduler-handoff.json", {"version": 1, "stages": [stage], "reservations": [stage]})
    monkeypatch.setattr(H, "process_start", lambda pid: "original" if pid == 123 else None)
    workflow = Workflow(C.load_config(), tmp_path)
    acquired = 0
    while workflow.worker_slots.acquire(blocking=False):
        acquired += 1
    assert acquired == C.load_config()["max_workers"] - 1
    for _ in range(acquired):
        workflow.worker_slots.release()
    artifact = d / "latest.pt"
    artifact.write_bytes(b"completed adopted model")
    C.save(d / "receipt.json", {"status": "complete", "artifact_hashes": {"latest.pt": C.sha(artifact)}})
    monkeypatch.setattr(H, "retire_parent", lambda *args: None)
    workflow.worker(d, stage["arguments"], [])
    acquired = 0
    while workflow.worker_slots.acquire(blocking=False):
        acquired += 1
    assert acquired == C.load_config()["max_workers"]


def test_only_the_last_finished_adopted_stage_retires_the_old_controller(tmp_path, monkeypatch):
    stage = tmp_path / "jobs/candidate/0/joint"
    stage.mkdir(parents=True)
    plan = {
        "stages": [{"directory": str(stage), "pid": 123, "process_start": "live"}],
        "previous_controller": {"pid": 999, "process_start": "gone"},
    }
    monkeypatch.setattr(H, "process_start", lambda pid: "live" if pid == 123 else None)
    saves = []
    monkeypatch.setattr(H, "save", lambda *args: saves.append(True))
    H.retire_parent(tmp_path, plan)
    assert not saves
    (stage / "receipt.json").write_text("{}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: H.retire_parent(tmp_path, plan), range(8)))
    assert len(saves) == 1 and plan["parent_retired_at"]
