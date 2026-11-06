"""Concurrent scheduling tests: fake GPU snapshots, mock LLM and mock training only."""

import copy
import itertools
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from cajal_research import contracts as C
from cajal_research.agent import ToolSession
from cajal_research.resources import ResourcePool
from cajal_research.store import Store


def test_ten_atomic_gpu_leases_stay_on_five_cards():
    cfg = C.load_config()
    pool = ResourcePool(cfg["gpu_ids"], cfg["per_gpu_workers"], cfg["resources"])
    pool.snapshot = {g: {"free_mib": 100000} for g in cfg["gpu_ids"]}
    pool.snapshot_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=10) as executor:
        cards = list(executor.map(lambda i: pool.acquire(str(i), 32768, 0), range(10)))
    assert Counter(cards) == dict.fromkeys(range(5), 2)
    assert pool.choose(32768, 0) is None
    for i in range(10):
        pool.release(str(i))
    assert pool.leases == {}


def test_pending_and_observed_memory_are_both_reserved():
    cfg = C.load_config()
    pool = ResourcePool([0], 2, cfg["resources"])
    pool.snapshot = {0: {"free_mib": 50000}}
    pool.leases = {"pending": {"gpu": 0, "reserve_mib": 32768, "observed_mib": 0, "host_reserve_mib": 0, "pid": None}}
    assert pool.choose(32768, 0) is None
    pool.snapshot = {0: {"free_mib": 20000}}
    pool.leases["pending"]["observed_mib"] = 80000
    assert pool.choose(32768, 0) is None
    pool.snapshot = {0: {"free_mib": 60000}}
    pool.leases["pending"]["observed_mib"] = 27000
    assert pool.choose(32768, 0) == 0


def test_refresh_ignores_unauthorized_gpus_and_counts_own_descendants(monkeypatch):
    from cajal_research import resources

    cfg = C.load_config()
    pool = ResourcePool(cfg["gpu_ids"], 2, cfg["resources"])
    pool.leases = {"own": {"gpu": 0, "pid": 100, "reserve_mib": 32768, "observed_mib": 0, "host_reserve_mib": 0}}

    def query(args, **kwargs):
        if "index,uuid,memory.free" in args[1]:
            return "\n".join(f"{i}, GPU-{i}, 50000" for i in range(8))
        return "GPU-0, 101, 27000\nGPU-0, 200, 10000\nGPU-7, 300, 10000"

    monkeypatch.setattr(resources.subprocess, "check_output", query)
    monkeypatch.setattr(resources, "parent_pid", lambda pid: 100 if pid == 101 else 0)
    pool.refresh()
    assert set(pool.snapshot) == set(range(5))
    assert pool.leases["own"]["observed_mib"] == 27000


def test_concurrent_llm_submissions_claim_semantics_once(tmp_path):
    store = Store(tmp_path / "lineage")
    root = store.stage(C.reference_bundle())
    seen = {C.semantic_id(C.reference_bundle())}
    lock = threading.Lock()
    sessions = [
        ToolSession(store, {}, None, [root], seen, {"proposal": i}, tmp_path / f"p{i}.json", seen_lock=lock)
        for i in range(10)
    ]
    bundle = C.reference_bundle()
    bundle["parent_ids"] = [root]
    bundle["train"] = {"lr": 0.0005}

    def write(session):
        try:
            return session.call("write_candidate", {"bundle": copy.deepcopy(bundle)})
        except ValueError as exc:
            assert "semantic duplicate" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(write, sessions))
    assert sum(r is not None for r in results) == 1
    assert len(store.nodes()) == 2


def test_parallel_generation_is_ordered_and_resumable(tmp_path, monkeypatch):
    from cajal_research import workflow as W

    cfg = C.load_config()
    cfg["generations"] = 1
    monkeypatch.setattr(W, "code_fingerprint", lambda: {"mock": "unchanged"})
    monkeypatch.setattr(W, "input_fingerprint", lambda cfg: {"mock": "unchanged"})
    workflow = W.Workflow(cfg, tmp_path)
    monkeypatch.setattr(workflow, "validate", lambda *a: {"status": "accepted"})

    def evaluate(identifier, generation):
        p = workflow.store.candidates / identifier / "result.json"
        if p.exists():
            return json.loads(p.read_text())
        bundle = workflow.store.bundle(identifier)
        index = round((bundle["train"].get("lr", 0.001) - 0.001) * 100000)
        metrics = dict.fromkeys(C.METRICS, 0.55 if not generation else 0.6 + index * 0.001)
        node = {
            "candidate_id": identifier,
            "generation": generation,
            "status": "complete",
            "all_seeds_passed": True,
            "metrics": metrics,
            "std": dict.fromkeys(C.METRICS, None),
            "seed_count": 1,
            "seed_results": [{"seed": 0, "metrics": metrics}],
        }
        workflow.store.put(identifier, "result", node)
        return node

    monkeypatch.setattr(workflow, "evaluate", evaluate)
    barrier = threading.Barrier(10)
    lock = threading.Lock()
    activity = {"current": 0, "peak": 0, "calls": 0}

    def lane(client, generation, index, root, champion, allowed, seen, seen_lock, context):
        assert context["current_champion"] == root == champion and allowed == [root]
        with lock:
            activity["current"] += 1
            activity["calls"] += 1
            activity["peak"] = max(activity["peak"], activity["current"])
        barrier.wait(timeout=20)
        b = C.reference_bundle()
        b["parent_ids"] = [root]
        b["train"] = {"lr": 0.001 + index / 100000}
        identifier = workflow.store.stage(b, origin={"generation": generation, "proposal": index})
        time.sleep((9 - index) * 0.005)
        result = evaluate(identifier, generation)
        with lock:
            activity["current"] -= 1
        return identifier, result

    monkeypatch.setattr(workflow, "run_proposal", lane)
    state = workflow.execute(object())
    assert activity["peak"] == activity["calls"] == 10 and len(state["generations"]) == 1
    decisions = [json.loads((tmp_path / "decisions" / f"g01-p{i:02d}.json").read_text()) for i in range(10)]
    for before, after in itertools.pairwise(decisions):
        assert before["next_champion"] == after["previous_champion"]
    snapshot = (tmp_path / "campaign.json").read_bytes()
    workflow.execute(object())
    assert activity["calls"] == 10 and (tmp_path / "campaign.json").read_bytes() == snapshot


def test_failed_worker_releases_its_gpu_lease(tmp_path, monkeypatch):
    from cajal_research import sandbox
    from cajal_research.workflow import Workflow

    workflow = Workflow(C.load_config(), tmp_path)
    leases = []
    monkeypatch.setattr(workflow.resources, "acquire", lambda *a: 2)
    monkeypatch.setattr(workflow.resources, "release", lambda key: leases.append(key))

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic process failure")

    monkeypatch.setattr(sandbox, "launch", fail)
    with pytest.raises(RuntimeError, match="synthetic process failure"):
        workflow.worker(tmp_path / "job", ["train"], [])
    assert leases == [str(tmp_path / "job")] and not workflow.activity
