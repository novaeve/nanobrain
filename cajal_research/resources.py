"""GPU leases adapted from the original ReAct scheduler, with observed-memory accounting."""

import subprocess
import threading
import time
from collections import Counter
from pathlib import Path


def parent_pid(pid):
    try:
        return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


class ResourcePool:
    def __init__(self, gpus, per_gpu, options):
        if not gpus or set(gpus) - set(range(7)):
            raise ValueError("only GPUs 0..6 are authorized")
        self.gpus, self.per_gpu, self.options = list(gpus), per_gpu, options
        self.condition = threading.Condition()
        self.leases, self.snapshot, self.snapshot_at = {}, {}, 0.0

    def refresh(self):
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=15
        )
        devices, uuids = {}, {}
        for line in raw.splitlines():
            gpu, uuid, free = [x.strip() for x in line.split(",")]
            gpu = int(gpu)
            if gpu in self.gpus:
                devices[gpu] = {"free_mib": int(free)}
                uuids[uuid] = gpu
        for lease in self.leases.values():
            lease["observed_mib"] = 0
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        roots = {v["pid"]: v for v in self.leases.values() if v.get("pid")}
        for line in raw.splitlines():
            uuid, pid, memory = [x.strip() for x in line.split(",")]
            if uuid not in uuids or not memory.isdigit():
                continue
            pid, visited = int(pid), set()
            while pid and pid not in visited:
                if pid in roots:
                    roots[pid]["observed_mib"] += int(memory)
                    break
                visited.add(pid)
                pid = parent_pid(pid)
        self.snapshot, self.snapshot_at = devices, time.monotonic()

    def choose(self, reserve, host_reserve):
        available = next(
            int(line.split()[1]) // 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if (
            available - sum(v["host_reserve_mib"] for v in self.leases.values())
            < self.options["minimum_host_available_mb"] + host_reserve
        ):
            return None
        counts = Counter(v["gpu"] for v in self.leases.values())
        choices = []
        for gpu, state in self.snapshot.items():
            leases = [v for v in self.leases.values() if v["gpu"] == gpu]
            observed = sum(v.get("observed_mib", 0) for v in leases)
            committed = sum(max(v["reserve_mib"], v.get("observed_mib", 0)) for v in leases)
            remaining = state["free_mib"] + observed - committed - self.options["gpu_memory_margin_mb"]
            if counts[gpu] < self.per_gpu and remaining >= reserve:
                choices.append((remaining, -counts[gpu], -gpu, gpu))
        return max(choices)[-1] if choices else None

    def acquire(self, key, reserve, host_reserve):
        with self.condition:
            while True:
                if time.monotonic() - self.snapshot_at > 3 or not self.snapshot:
                    try:
                        self.refresh()
                    except (OSError, subprocess.SubprocessError, ValueError):
                        self.snapshot = {}
                gpu = self.choose(reserve, host_reserve)
                if gpu is not None:
                    self.leases[key] = {
                        "gpu": gpu,
                        "reserve_mib": reserve,
                        "host_reserve_mib": host_reserve,
                        "observed_mib": 0,
                        "pid": None,
                    }
                    return gpu
                self.condition.wait(5)

    def started(self, key, pid):
        with self.condition:
            self.leases[key]["pid"] = pid
            self.snapshot_at = 0.0

    def release(self, key):
        with self.condition:
            self.leases.pop(key, None)
            self.snapshot_at = 0.0
            self.condition.notify_all()
