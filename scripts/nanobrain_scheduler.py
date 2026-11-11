"""Work-conserving task queue; training and scientific selection stay unchanged."""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from nanobrain.campaign import (
    audit_training_receipt, candidate_method, overall_metrics, primary_metric_record,
    read_metric_table,
)
from nanobrain.evolution import architecture_id, joint_score
from nanobrain.genome import CandidateGenome
from scripts.run_nanobrain_campaign import (
    _load_training_receipt, _launch_candidates, _launch_evaluations,
    _public_candidate, _write_json,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def migrate_scheduler_fingerprint(output, frozen, fingerprint, root):
    """Audited runtime-only migration; refuse changes to any training inputs."""
    before = frozen["fingerprint"]
    changed = {name for name in set(before) | set(fingerprint) if before.get(name) != fingerprint.get(name)}
    allowed = {"scripts/run_nanobrain_evolution.py", "scripts/nanobrain_scheduler.py"}
    if not changed or changed - allowed:
        raise ValueError(f"scheduler migration cannot change scientific files: {sorted(changed - allowed)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = output / "runtime_migrations" / stamp
    archive.mkdir(parents=True)
    _write_json(archive / "protocol.before.json", frozen)
    for relative in changed:
        previous = output / "source_snapshot" / relative
        if previous.exists():
            backup = archive / "source.before" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous, backup)
        if relative in fingerprint:
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, previous)
    record = dict(timestamp=datetime.now(timezone.utc).isoformat(), changed_files=sorted(changed),
                  reason="adaptive GPU queue; no budget, data, model, loss, seed or evaluator changes",
                  before={name: before.get(name) for name in changed},
                  after={name: fingerprint.get(name) for name in changed}, archive=str(archive))
    updated = dict(frozen, fingerprint=fingerprint,
                   runtime_migrations=[*frozen.get("runtime_migrations", []), record])
    _write_json(output / "protocol.json", updated)
    return record


def gpu_snapshot(gpus, own_pids):
    """Exclude our workers when measuring other jobs' memory pressure."""
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    devices = {}
    uuids = {}
    for line in query.splitlines():
        index, uuid, free, util = [x.strip() for x in line.split(",")]
        index = int(index)
        if index in gpus:
            devices[index] = dict(free_mb=int(free), utilization=int(util), external_mb=0, own_mb=0)
            uuids[uuid] = index
    query = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    for line in query.splitlines():
        if not line.strip():
            continue
        uuid, pid, memory = [x.strip() for x in line.split(",")]
        if uuid in uuids:
            key = "own_mb" if int(pid) in own_pids else "external_mb"
            devices[uuids[uuid]][key] += int(memory) if memory.isdigit() else 0
    return devices


def available_slots(snapshot, occupied, *, per_gpu=2, external_limit_mb=16384, minimum_free_mb=2048,
                    allow_shared=False):
    candidates = [gpu for gpu, state in snapshot.items()
                  if state["free_mb"] >= minimum_free_mb and occupied[gpu] < per_gpu]
    quiet = [gpu for gpu in candidates if snapshot[gpu]["external_mb"] <= external_limit_mb]
    # An unoccupied slot on a heavily shared device must not displace a
    # currently full quiet device: wait for that quiet worker to finish.
    quiet_devices = [gpu for gpu, state in snapshot.items()
                     if state["external_mb"] <= external_limit_mb and state["free_mb"] >= minimum_free_mb]
    if quiet_devices and not allow_shared:
        candidates = quiet
    elif candidates and not allow_shared:
        candidates = [min(candidates, key=lambda gpu: (snapshot[gpu]["external_mb"],
                                                      snapshot[gpu]["utilization"], gpu))]
    ordered = sorted(candidates, key=lambda gpu: (snapshot[gpu]["external_mb"], occupied[gpu],
                                                 snapshot[gpu]["utilization"], -snapshot[gpu]["free_mb"], gpu))
    slots = [gpu for slot in range(per_gpu) for gpu in ordered if occupied[gpu] + slot < per_gpu]
    if not quiet_devices and not allow_shared:
        # Shared cards are a fallback pool with a single concurrency limit,
        # rather than multiplying five workers by every heavily occupied GPU.
        slots = slots[:max(0, per_gpu - sum(occupied.values()))]
    return slots


class AdoptedProcess:
    """Observe an existing trainer without restarting its completed steps."""
    def __init__(self, pid):
        self.pid = int(pid)

    def poll(self):
        try:
            stat = (Path("/proc") / str(self.pid) / "stat").read_text().split()
            return 0 if stat[2] == "Z" else None
        except FileNotFoundError:
            return 0


def adopt_worker(item, records):
    for record in records:
        if Path(record["directory"]).resolve() != Path(item["directory"]).resolve():
            continue
        proc = Path("/proc") / str(record["pid"])
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")
            command = [x.decode() for x in command if x]
            output = command[command.index("--output") + 1]
            if not any(x.endswith("/scripts/train_nanobrain.py") for x in command):
                continue
            if Path(output).resolve() != Path(item["directory"]).resolve():
                continue
            genome = json.loads((Path(output) / "genome.json").read_text())
            budget = int(command[command.index("--steps") + 1])
            if genome != item["genome"] or budget != item["steps"]:
                raise RuntimeError("existing worker genome or budget mismatch")
            process = AdoptedProcess(record["pid"])
            if process.poll() is None:
                return dict(process=process, pid=process.pid, gpu=record["gpu"], adopted=True)
        except (FileNotFoundError, ValueError, IndexError):
            continue
    return None


def run_task_queue(experiment, specs, directory, steps, checkpoint_selector):
    tasks = experiment.protocol["tasks"]
    results = {task: {} for task in tasks}
    pending = deque()
    active = []
    counters = Counter()
    records_path = getattr(experiment.args, "adopt_workers", None)
    records = json.loads(Path(records_path).read_text())["workers"] if records_path else []
    events_path = experiment.output / "scheduler_events.jsonl"

    def event(kind, **fields):
        with events_path.open("a") as f:
            f.write(json.dumps(dict(event=kind, timestamp=datetime.now(timezone.utc).isoformat(),
                                    **fields), ensure_ascii=False) + "\n")

    def finish_failed(item, reasons, audit, phase="training"):
        result = _public_candidate(item)
        result.update(metrics={}, diagnostics={}, score=-1.0e9,
                      architecture_id=architecture_id(item["genome"]),
                      trainable_parameter_count=None, training_audit=audit,
                      failed=True, failure_reasons=reasons)
        _write_json(Path(item["directory"]) / "result.json", result)
        results[item["training_task"]][item["index"]] = result
        counters["failed_" + phase] += 1
        event(phase + "_failed", task=item["training_task"], index=item["index"], reasons=reasons)

    def prepare_evaluation(item):
        receipt = _load_training_receipt(item, steps)
        audit = audit_training_receipt(receipt or {})
        if receipt is not None and receipt.get("training_task", "joint") != item["training_task"]:
            raise RuntimeError("training receipt task mismatch")
        if not audit["passed"] or item.get("returncode", 0) != 0:
            finish_failed(item, audit["reasons"] or [f"training_exit_{item.get('returncode')}"], audit)
            return False
        selection = experiment.protocol.get("checkpoint_selection", "final_step_fixed_budget")
        checkpoint = checkpoint_selector(receipt, selection)
        item.update(checkpoint_selection=selection,
                    selected_step=receipt["best_step"],
                    selected_validation_quality=receipt["best_validation_quality"],
                    training_audit={**audit, "checkpoint": str(checkpoint)},
                    parameter_count=receipt["parameter_count"], runtime_seconds=receipt["runtime_seconds"])
        return True

    def finish_evaluation(item, code):
        if code != 0:
            log = Path(item["evaluation"]).parent / "run.log"
            detail = log.read_text(errors="replace") if log.exists() else ""
            retries = int(item.get("evaluation_retries", 0))
            is_oom = "out of memory" in detail.lower() or "OutOfMemoryError" in detail
            if retries < 3:
                item["evaluation_retries"] = retries + 1
                if is_oom:
                    item["evaluation_memory_mb"] = min(32768, 8192 * (2 ** (retries + 1)))
                if log.exists():
                    shutil.copy2(log, log.with_name(f"run.retry-{retries + 1}.log"))
                pending.append(("evaluation", item))
                counters["evaluation_retries"] += 1
                event("evaluation_retry", task=item["training_task"], index=item["index"],
                      oom=is_oom, attempt=retries + 1, memory_mb=item.get("evaluation_memory_mb", 8192))
                return
            if is_oom:
                # Lack of available GPU memory is not evidence of a bad model.
                # Keep its weights and wait for capacity while other jobs run.
                item["evaluation_memory_mb"] = 32768
                pending.append(("evaluation", item))
                event("evaluation_waiting_for_memory", task=item["training_task"], index=item["index"])
                return
            finish_failed(item, [f"evaluation_exit_{code}", detail[-1500:]], item["training_audit"], "evaluation")
            return
        item["evaluation_returncode"] = code
        rows = read_metric_table(item["evaluation"])
        if not rows or any(x.get("status") != "complete" for x in rows):
            finish_failed(item, ["incomplete_canonical_metric_table"], item["training_audit"], "evaluation")
            return
        metrics = overall_metrics(rows, candidate_method(item["candidate_id"]))
        result = _public_candidate(item)
        result.update(metrics=primary_metric_record(metrics),
                      diagnostics={f"{k[0]}.{k[1]}": v for k, v in metrics.items()},
                      architecture_id=architecture_id(item["genome"]),
                      checkpoint=item["training_audit"]["checkpoint"])
        result["trainable_parameter_count"] = json.loads(Path(item["receipt"]).read_text()).get("trainable_parameter_count")
        try:
            result["score"] = joint_score(result["metrics"], [item["training_task"]])
        except (KeyError, ValueError, TypeError) as exc:
            finish_failed(item, [f"invalid_evaluation_metrics:{exc}"], item["training_audit"], "evaluation")
            return
        for name in ("checkpoint", "evaluation"):
            result[name + "_sha256"] = sha(result[name])
        _write_json(Path(item["directory"]) / "result.json", result)
        results[item["training_task"]][item["index"]] = result
        counters["evaluations_complete"] += 1
        event("evaluation_complete", task=item["training_task"], index=item["index"], gpu=item.get("gpu"))

    for task in tasks:
        for index, spec in enumerate(specs):
            genome = CandidateGenome.from_dict(spec["genome"])
            path = directory / task / f"candidate_{index:02d}"
            saved_path = path / "result.json"
            if saved_path.is_file():
                saved = json.loads(saved_path.read_text())
                if saved["genome"] != genome.to_dict() or saved["steps"] != steps or saved.get("training_task", "joint") != task:
                    raise ValueError("cannot resume a candidate with changed budget, genome or task")
                if not saved.get("failed"):
                    for name in ("checkpoint", "evaluation"):
                        if sha(saved[name]) != saved[name + "_sha256"]:
                            raise ValueError("resumed artifact checksum mismatch")
                results[task][index] = saved
                counters["reused_results"] += 1
                continue
            item = dict(spec, index=index, directory=str(path), candidate_id=genome.genome_id,
                        genome=genome.to_dict(), steps=steps, training_task=task)
            receipt = _load_training_receipt(item, steps)
            if receipt is not None:
                item["returncode"] = 0
                if prepare_evaluation(item):
                    counters["reused_training"] += 1
                    eval_dir = path / ("test_selection_evaluation" if experiment.protocol.get("selection_evaluation_split") == "test" else "evaluation")
                    manifest_path = eval_dir / "run_manifest.json"
                    table = eval_dir / "benchmark_table.csv"
                    reused_evaluation = False
                    if manifest_path.exists() and table.exists():
                        manifest = json.loads(manifest_path.read_text())
                        expected_manifest = experiment.fingerprint.get("data_manifest") if hasattr(experiment, "fingerprint") else None
                        if (manifest.get("checkpoint") == item["training_audit"]["checkpoint"]
                                and manifest.get("evaluation_split") == experiment.protocol.get("selection_evaluation_split", "validation")
                                and manifest.get("population_evaluation_seed") == experiment.protocol["evaluation_seed"]
                                and manifest.get("data_manifest_sha256") == expected_manifest
                                and manifest.get("tasks_evaluated") == [task]):
                            item["evaluation"] = str(table)
                            finish_evaluation(item, 0)
                            counters["reused_evaluations"] += 1
                            reused_evaluation = True
                    if not reused_evaluation:
                        pending.append(("evaluation", item))
            else:
                adopted = adopt_worker(item, records)
                if adopted:
                    item.update(pid=adopted["pid"], gpu=adopted["gpu"])
                    active.append(dict(phase="training", item=item, started=time.time(), memory_mb=2048, **adopted))
                    counters["adopted_workers"] += 1
                    event("worker_adopted", pid=adopted["pid"], gpu=adopted["gpu"], task=task, index=index)
                else:
                    pending.append(("training", item))

    per_gpu = int(getattr(experiment.args, "workers_per_gpu", 2))
    max_workers = int(getattr(experiment.args, "max_workers", len(experiment.args.gpus) * per_gpu))
    training_reservation = int(experiment.protocol.get("training_memory_reservation_mb", 4096))
    guard_mb = int(experiment.protocol.get("gpu_memory_guard_mb", 4096))
    last_save = 0.0
    snapshot = {}
    snapshot_at = 0.0
    while pending or active:
        for job in list(active):
            code = job["process"].poll()
            if code is None:
                continue
            active.remove(job)
            if job.get("log_handle") is not None:
                job["log_handle"].close()
            item = job["item"]
            if job["phase"] == "training":
                item["returncode"] = code
                log = Path(item["directory"]) / "train.log"
                detail = log.read_text(errors="replace") if code != 0 and log.exists() else ""
                if "out of memory" in detail.lower() or "OutOfMemoryError" in detail:
                    retry = int(item.get("training_oom_retries", 0)) + 1
                    item.update(training_oom_retries=retry, training_memory_mb=min(49152, training_reservation * 2 ** retry))
                    shutil.copy2(log, log.with_name(f"train.oom-retry-{retry}.log"))
                    pending.append(("training", item))
                    counters["training_oom_retries"] += 1
                    event("training_oom_retry", task=item["training_task"], index=item["index"],
                          memory_mb=item["training_memory_mb"], attempt=retry)
                    continue
                if prepare_evaluation(item):
                    pending.appendleft(("evaluation", item))
            else:
                item["evaluation_elapsed_seconds"] = time.time() - job["started"]
                finish_evaluation(item, code)
        if pending:
            if not snapshot or time.time() - snapshot_at >= 5:
                try:
                    snapshot = gpu_snapshot(experiment.args.gpus, {x["pid"] for x in active})
                    snapshot_at = time.time()
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    event("gpu_query_retry", reason=str(exc)[:500])
            occupied = Counter(x["gpu"] for x in active)
            slots = available_slots(snapshot, occupied, per_gpu=per_gpu,
                                    allow_shared=getattr(experiment.args, "allow_shared_gpus", False))
            for gpu in slots:
                if not pending or len(active) >= max_workers:
                    break
                capacity = snapshot[gpu]["free_mb"] + snapshot[gpu].get("own_mb", 0) - guard_mb
                reserved = sum(x.get("memory_mb", training_reservation) for x in active if x["gpu"] == gpu)
                chosen = None
                for _ in range(len(pending)):
                    phase, item = pending.popleft()
                    memory_mb = int(item.get("evaluation_memory_mb", 8192)) if phase == "evaluation" else int(item.get("training_memory_mb", training_reservation))
                    if chosen is None and reserved + memory_mb <= capacity:
                        chosen = (phase, item, memory_mb)
                        break
                    pending.append((phase, item))
                if chosen is None:
                    continue
                phase, item, memory_mb = chosen
                if phase == "training":
                    genome = CandidateGenome.from_dict(item["genome"])
                    p = experiment.protocol
                    launched = _launch_candidates(genomes=[genome], gpus=[gpu], data=experiment.data,
                        output=Path(item["directory"]).parent, steps=steps, seed=genome.seed,
                        directory_names=[Path(item["directory"]).name], task=item["training_task"],
                        **{key: p[key] for key in ("batch_size", "validation_batch_size", "precision", "development_cells",
                            "validation_development_groups", "eval_interval", "loss_balance", "loss_balance_decay")},
                        checkpoint_policy=p.get("checkpoint_policy", "all"),
                        integration_adversarial_warmup_steps=p.get("integration_adversarial_warmup_steps", 500),
                        integration_logit_clip=p.get("integration_logit_clip", 20.),
                        stability_max_loss=p.get("stability_max_loss", 1.e8),
                        stability_max_gradient_norm=p.get("stability_max_gradient_norm", 1.e6))[0]
                    index = item["index"]
                    item.update(_public_candidate(launched))
                    item["index"] = index
                else:
                    p = experiment.protocol
                    launched = _launch_evaluations(candidates=[item], gpus=[gpu], data=experiment.data,
                        batch_size=p["evaluation_batch_size"], evaluation_seed=p["evaluation_seed"],
                        evaluation_split=p.get("selection_evaluation_split", "validation"),
                        output_name="test_selection_evaluation" if p.get("selection_evaluation_split") == "test" else "evaluation")[0]
                item.update(gpu=gpu, pid=launched["pid"])
                active.append(dict(phase=phase, item=item, process=launched["process"],
                    log_handle=launched.get("log_handle"), gpu=gpu, pid=launched["pid"],
                    memory_mb=memory_mb, started=time.time()))
                counters[phase + "_launched"] += 1
                event("job_started", phase=phase, task=item["training_task"], index=item["index"],
                      gpu=gpu, pid=launched["pid"])
        experiment.state["active_runs"] = [dict(candidate_id=x["item"]["candidate_id"],
            directory=x["item"]["directory"], task=x["item"]["training_task"], phase=x["phase"],
            pid=x["pid"], gpu=x["gpu"], steps=steps) for x in active]
        experiment.state["active_processes"] = len(active)
        experiment.state["scheduler"] = dict(policy="adaptive_work_queue", workers_per_gpu=per_gpu,
            max_workers=max_workers, allow_shared_gpus=getattr(experiment.args, "allow_shared_gpus", False),
            completed=sum(len(x) for x in results.values()), total=len(tasks) * len(specs), pending=len(pending),
            counters=dict(counters), gpu_snapshot=snapshot, directory=str(directory))
        if time.time() - last_save >= float(experiment.protocol.get("scheduler_save_interval_seconds", 10)) or not active:
            experiment.save("adaptive_queue_running")
            last_save = time.time()
        if pending or active:
            time.sleep(1)
    return {task: [results[task][i] for i in range(len(specs))] for task in tasks}
