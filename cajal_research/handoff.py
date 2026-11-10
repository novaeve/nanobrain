"""Audited scheduler handoff that leaves in-flight training in its old sandbox."""

import json
import os
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from .contracts import save, sha

_RETIRE_LOCK = threading.Lock()


def process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def load_handoff(output, pool):
    path = Path(output) / "scheduler-handoff.json"
    if not path.exists():
        return {}, {}, None
    plan = json.loads(path.read_text())
    if plan.get("version") != 1:
        raise ValueError("unknown scheduler handoff version")
    stages, reservations = {}, {}
    for row in plan.get("reservations", []):
        directory = Path(row["directory"]).resolve()
        if not directory.is_relative_to((Path(output) / "jobs").resolve()):
            raise ValueError("handoff stage is outside this campaign")
        reservations[str(directory)] = row
    for row in plan.get("stages", []):
        directory = Path(row["directory"]).resolve()
        if not directory.is_relative_to((Path(output) / "jobs").resolve()) or row["mode"] not in {
            "train",
            "predict",
            "score",
        }:
            raise ValueError("only an existing campaign GPU stage can be adopted")
        if (directory / "receipt.json").exists():
            continue
        if row["gpu"] not in pool.gpus:
            raise ValueError("adopted worker GPU is not authorized")
        if process_start(row["pid"]) != row["process_start"]:
            # A stopped worker resumes using the existing checkpoint through
            # the original worker entrypoint, never by attaching to a reused PID.
            continue
        key = str(directory)
        stages[key] = row
        pool.leases[key] = {
            "gpu": row["gpu"],
            "reserve_mib": row["reservation_mb"],
            "host_reserve_mib": 8192 if row["mode"] == "train" else 4096,
            "observed_mib": 0,
            "pid": row["pid"],
        }
    return stages, reservations, plan


def wait_for_stage(stage, args, verify_stage, timeout):
    if args != stage["arguments"] or sha(stage["bundle"]) != stage["bundle_sha256"]:
        raise ValueError("adopted training recipe changed")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (Path(stage["directory"]) / "receipt.json").exists():
            return verify_stage(stage["directory"])
        if process_start(stage["pid"]) != stage["process_start"]:
            raise RuntimeError("preserved training worker exited without a complete stage receipt")
        time.sleep(2)
    raise RuntimeError("preserved training stage timed out")


def retire_parent(output, plan):
    with _RETIRE_LOCK:
        _retire_parent(output, plan)


def _retire_parent(output, plan):
    if not plan or plan.get("parent_retired_at"):
        return
    for stage in plan["stages"]:
        if (
            not (Path(stage["directory"]) / "receipt.json").exists()
            and process_start(stage["pid"]) == stage["process_start"]
        ):
            return
    parent = plan["previous_controller"]
    if process_start(parent["pid"]) == parent["process_start"]:
        state = Path(f"/proc/{parent['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state not in {"T", "t"}:
            raise ValueError("retire only the explicitly paused previous scheduler")
        os.kill(parent["pid"], signal.SIGKILL)
    plan["parent_retired_at"] = datetime.now(UTC).isoformat()
    save(Path(output) / "scheduler-handoff.json", plan)
