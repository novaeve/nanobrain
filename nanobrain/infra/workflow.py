"""Independent Cajal campaign; no work starts until the explicit execute entrypoint."""

import concurrent.futures
import csv
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

from . import gate as selection
from .contracts import (
    METRIC_GROUPS,
    METRICS,
    ROOT,
    digest,
    reference_bundle,
    reference_document,
    resolved_document,
    save,
    semantic_id,
    sha,
    validate_bundle,
)


def plan(cfg):
    return {
        "status": "prepared_not_started",
        "reference_recipe": cfg["reference_recipe"],
        "initial_candidate": reference_bundle(),
        "reference_config": reference_document(cfg),
        "training": "one shared Cajal model, all five objectives, per seed",
        "weight_policy": cfg["weight_policy"],
        "external_weight_dependencies": [],
        "seeds": cfg["seeds"],
        "seed_count": len(cfg["seeds"]),
        "uncertainty_estimated": False,
        "schedule_steps": cfg["schedule_steps"],
        "fixed_stop": cfg["stop_at"],
        "generations": cfg["generations"],
        "candidates_per_generation": cfg["candidates_per_generation"],
        "joint_trainings_per_generation": len(cfg["seeds"]) * cfg["candidates_per_generation"],
        "maximum_joint_trainings": len(cfg["seeds"]) * cfg["generations"] * cfg["candidates_per_generation"],
        "reference_evaluation": cfg["reference_evaluation"],
        "reference_trainings": 0,
        "baseline_policy": "reuse_existing_Cajal_scores_without_refitting_or_rescoring",
        "gpu_ids": cfg["gpu_ids"],
        "max_workers": cfg["max_workers"],
        "per_gpu_workers": cfg["per_gpu_workers"],
        "cpu_probe_workers": cfg.get("cpu_probe_workers", 1),
        "task_evaluation_workers": cfg.get("task_evaluation_workers", 1),
        "model": cfg["model"],
        "reasoning_effort": cfg["reasoning_effort"],
        "generation_policy": "parallel cohort, ordered champion decisions",
        "metrics": list(METRICS),
        "feedback": "existing Cajal held-out research evaluation sets; sealed V4 test inaccessible",
        "automatic_scaling": False,
        "llm_called": False,
        "real_training_started": False,
    }


def baseline():
    return json.loads((ROOT / "reference/opponents.json").read_text())


def code_fingerprint():
    from .dependencies import verify

    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, env=env).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit the prepared code before any formal run")
    files = git("ls-files").splitlines()
    versions = {"python": platform.python_version()}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "scib-metrics", "jax", "cuml-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "files": {p: sha(ROOT / p) for p in files if (ROOT / p).is_file()},
        "environment": versions,
        "component_environment": verify(),
    }


def input_fingerprint(cfg):
    from .existing_reference import load

    result = {}
    for name, path in cfg["data"].items():
        if "test" in name or "test" in Path(path).name.lower():
            raise ValueError("the sealed test is outside this protocol")
        p = Path(path) / "manifest.json"
        result[name] = {"path": str(Path(path).resolve()), "manifest_sha256": sha(p)}
    for name, expected in json.loads((ROOT / "reference/input_fingerprints.json").read_text()).items():
        if name not in result or result[name] != expected:
            raise ValueError("Cajal data manifest or path changed since preparation")
    if cfg.get("trusted_teachers"):
        raise ValueError("historical teacher weights cannot be inputs to independent training")
    result["existing_reference"] = load(cfg)["artifact_hashes"]
    return result


class InfrastructureError(RuntimeError):
    """Transient resource failures preserve an unfinished candidate for resume."""


class Workflow:
    def __init__(self, cfg, output):
        # Construction is deliberately read-only; default plan never creates an output directory.
        self.cfg, self.output = cfg, Path(output).resolve()
        from .resources import ResourcePool

        self.activity_lock = threading.RLock()
        self.activity, self.status = {}, {}
        self.cpu_slots = threading.Semaphore(cfg.get("cpu_probe_workers", 2))
        self.resources = ResourcePool(cfg["gpu_ids"], cfg["per_gpu_workers"], cfg["resources"])
        from .handoff import load_handoff

        self.adopted_stages, self.stage_reservations, self.handoff_plan = load_handoff(self.output, self.resources)
        if len(self.adopted_stages) > cfg["max_workers"]:
            raise ValueError("live adopted stages exceed the global concurrency cap")
        self.worker_slots = threading.Semaphore(cfg["max_workers"])
        for _ in self.adopted_stages:
            self.worker_slots.acquire()

    def progress(self, **fields):
        with self.activity_lock:
            self.status.update(fields)
            self.status.update(
                updated_at=datetime.now(UTC).isoformat(),
                active_jobs=list(self.activity.values()),
                maximum_workers=self.cfg["max_workers"],
                gpu_ids=self.cfg["gpu_ids"],
                per_gpu_workers=self.cfg["per_gpu_workers"],
                cpu_probe_workers=self.cfg.get("cpu_probe_workers", 1),
                task_evaluation_workers=self.cfg.get("task_evaluation_workers", 1),
            )
            save(self.output / "status.json", self.status)

    @staticmethod
    def verify_stage(directory):
        directory = Path(directory)
        value = json.loads((directory / "receipt.json").read_text())
        if value.get("status") != "complete" or not value.get("artifact_hashes"):
            raise ValueError("incomplete stage receipt")
        for name, expected in value["artifact_hashes"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
                raise ValueError("stage artifact changed")
        return value

    def worker(self, directory, args, inputs, cpu=False):
        # Live adopted stages already occupy one global slot and a GPU lease.
        if str(directory) not in self.adopted_stages:
            self.worker_slots.acquire()
        try:
            return self._run_stage(directory, args, inputs, cpu)
        finally:
            self.worker_slots.release()

    def _run_stage(self, directory, args, inputs, cpu=False):
        from .sandbox import launch

        key = str(directory)
        reserve = (
            self.cfg["minimum_free_gpu_mb"]
            if args[0] == "train"
            else self.cfg["resources"]["evaluation_reservation_mb"]
        )
        host = 8192 if args[0] == "train" else 4096
        adopted = self.adopted_stages.pop(key, None)
        previous = self.stage_reservations.get(key, {})
        reserve = max(reserve, previous.get("reservation_mb", 0))
        for attempt in range(max(0, previous.get("attempt", 1) - 1), 3):
            with self.activity_lock:
                self.activity[key] = {
                    "directory": key,
                    "mode": args[0],
                    "phase": "waiting_cpu" if cpu else "waiting_gpu",
                    "attempt": attempt + 1,
                    "gpu": None,
                    "reservation_mb": 0 if cpu else reserve,
                }
            self.progress()
            if adopted:
                gpu = adopted["gpu"]
                with self.activity_lock:
                    self.activity[key].update(pid=adopted["pid"], gpu=gpu, phase="running", adopted=True)
                self.progress()
            elif cpu:
                self.cpu_slots.acquire()
                gpu = None
            else:
                gpu = self.resources.acquire(key, reserve, host)

            def started(pid, gpu=gpu):
                if not cpu:
                    self.resources.started(key, pid)
                with self.activity_lock:
                    self.activity[key].update(pid=pid, gpu=gpu, phase="running")
                self.progress()

            try:
                if adopted:
                    from .handoff import wait_for_stage

                    return wait_for_stage(adopted, args, self.verify_stage, self.cfg["job_timeout_seconds"])
                return launch(
                    ROOT,
                    directory,
                    args,
                    inputs=inputs,
                    gpu=gpu,
                    timeout=600 if cpu else self.cfg["job_timeout_seconds"],
                    on_started=started,
                )
            except RuntimeError as exc:
                log = Path(directory) / "worker.log"
                text = log.read_text(errors="replace").lower() if log.exists() else ""
                oom = any(x in text for x in ("out of memory", "resource exhausted", "cuda_error_out_of_memory"))
                if cpu or not oom:
                    raise
                log.rename(log.with_name(f"oom-attempt-{attempt}.log"))
                if attempt == 2:
                    raise InfrastructureError(
                        "GPU OOM after three attempts; candidate resource limit recorded"
                    ) from exc
                reserve = min(90112, int(reserve * 1.5))
            finally:
                if cpu:
                    self.cpu_slots.release()
                else:
                    self.resources.release(key)
                with self.activity_lock:
                    self.activity.pop(key, None)
                self.progress()
                if adopted:
                    from .handoff import retire_parent

                    retire_parent(self.output, self.handoff_plan)
                    adopted = None

    def validate(self, identifier, bundle):
        folder = self.store.candidates / identifier
        path = folder / "validation.json"
        if path.exists():
            return json.loads(path.read_text())
        try:
            validate_bundle(bundle)
            value = self.worker(
                self.output / "probes" / identifier,
                ["probe", "--bundle", str(folder / "bundle.json")],
                [folder / "bundle.json"],
                cpu=True,
            )
        except Exception as exc:  # noqa: BLE001 - record any candidate acceptance failure, then keep the archive
            value = {"status": "rejected_validation", "reason": str(exc)[:1500], "synthetic_only": True}
        self.store.put(identifier, "validation", value)
        return value

    def evaluate_seed(self, identifier, seed):
        if identifier == digest(reference_bundle()):
            from .existing_reference import seed_result

            return seed_result(self.cfg, identifier, seed)
        from .training_policy import validate_identity

        bundle = self.store.bundle(identifier)
        validate_bundle(bundle)
        folder = self.output / "jobs" / identifier / str(seed)
        result_path = folder / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            validate_identity(result, bundle, self.cfg, seed)
            for name, expected in result["artifact_hashes"].items():
                if sha(Path(name)) != expected:
                    raise ValueError("completed result artifact changed")
            return result
        bundle_path = self.store.candidates / identifier / "bundle.json"
        train_dir = folder / "joint"
        train_receipt = train_dir / "receipt.json"
        if train_receipt.exists():
            receipt = self.verify_stage(train_dir)
            if receipt.get("status") != "complete" or receipt.get("seed") != seed:
                raise ValueError("incompatible cached training receipt")
        else:
            self.worker(
                train_dir,
                ["train", "--bundle", str(bundle_path), "--seed", str(seed)],
                [bundle_path, Path(self.cfg["data"]["pretrain"])],
            )
        receipt = self.verify_stage(train_dir)
        validate_identity(receipt, bundle, self.cfg, seed)
        checkpoint = train_dir / "train/latest.pt"
        artifacts = {str(checkpoint): sha(checkpoint), str(train_receipt): sha(train_receipt)}

        def evaluate_task(task):
            data_key = "annotation" if task == "genes" else task
            pred_dir, score_dir = folder / (task + "-predict"), folder / (task + "-score")
            pred = pred_dir / "prediction.npz"
            if not (pred_dir / "receipt.json").exists():
                self.worker(
                    pred_dir,
                    ["predict", "--checkpoint", str(checkpoint), "--task", task],
                    [checkpoint, Path(self.cfg["data"][data_key])],
                )
            self.verify_stage(pred_dir)
            if not (score_dir / "receipt.json").exists():
                self.worker(
                    score_dir,
                    ["score", "--prediction", str(pred), "--task", task],
                    [pred_dir, Path(self.cfg["data"][data_key]), Path(self.cfg["data"]["pretrain"])],
                )
            self.verify_stage(score_dir)
            summary = score_dir / "scores/summary.csv"
            with summary.open() as fh:
                rows = list(csv.DictReader(fh))
            (row,) = [r for r in rows if r[next(iter(r))] == "candidate"]
            wanted = METRIC_GROUPS[task]
            task_metrics = {key: float(row[key]) for key in wanted}
            task_artifacts = {}
            for p in (pred, pred_dir / "receipt.json", score_dir / "receipt.json", summary):
                task_artifacts[str(p)] = sha(p)
            return task_metrics, task_artifacts

        metrics = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.get("task_evaluation_workers", 1)) as pool:
            for task_metrics, task_artifacts in pool.map(
                evaluate_task, ("annotation", "integration", "genes", "spatial")
            ):
                metrics.update(task_metrics)
                artifacts.update(task_artifacts)
        result = {
            "status": "complete",
            "seed": seed,
            "metrics": metrics,
            "artifact_hashes": artifacts,
            "candidate_id": identifier,
            "checkpoint_sha256": artifacts[str(checkpoint)],
            "joint_model_count": 1,
            "weight_policy": self.cfg["weight_policy"],
            "external_weight_dependencies": [],
            "training_identity": receipt["training_identity"],
            "training_parameters": json.loads(train_receipt.read_text()).get("record", {}).get("config", {}),
        }
        # The same checkpoint supplies ALL task predictions; no task-wise model selection.
        selection.check_metrics(metrics)
        save(result_path, result)
        return result

    def evaluate(self, identifier, generation):
        existing = self.store.candidates / identifier / "result.json"
        if existing.exists():
            node = json.loads(existing.read_text())
            if node.get("status") == "complete":
                for seed in self.cfg["seeds"]:
                    self.evaluate_seed(identifier, seed)
            return node
        try:
            results = [self.evaluate_seed(identifier, s) for s in self.cfg["seeds"]]
            node = dict(
                candidate_id=identifier,
                status="complete",
                generation=generation,
                **selection.aggregate(results, self.cfg["seeds"]),
            )
        except InfrastructureError as exc:
            if generation == 0:
                raise
            node = {
                "candidate_id": identifier,
                "status": "resource_limited",
                "generation": generation,
                "all_seeds_passed": False,
                "reason": str(exc),
                "scientific_failure_claimed": False,
            }
        except Exception as exc:  # noqa: BLE001 - a failed model must not terminate other candidates
            node = {
                "candidate_id": identifier,
                "status": "failed",
                "generation": generation,
                "all_seeds_passed": False,
                "reason": str(exc)[:2000],
            }
        self.store.put(identifier, "result", node)
        return node

    def execute(self, client):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "controller.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._execute_locked(client)

    def run_proposal(self, client, generation, index, root, champion, allowed, seen, seen_lock, context):
        from ..agent.react import ToolSession, propose

        origin = {"generation": generation, "proposal": index}
        submission = self.output / "proposals" / f"g{generation:02d}-p{index:02d}.json"
        if submission.exists():
            saved = json.loads(submission.read_text())
            if saved["origin"] != origin:
                raise ValueError("proposal origin changed")
            identifier = saved["candidate_id"]
            self.store.bundle(identifier)
        else:
            for attempt in range(3):
                session = ToolSession(
                    self.store, baseline(), self.validate, allowed, seen, origin, submission, seen_lock=seen_lock
                )
                try:
                    propose(
                        client,
                        session,
                        dict(context, generation=generation, proposal=index),
                        reasoning=self.cfg["reasoning_effort"],
                    )
                    identifier = session.submitted
                    break
                except Exception as exc:  # noqa: BLE001 - proposal failures are persisted and credentials redacted
                    secret = getattr(getattr(client, "config", None), "api_key", "")
                    reason = str(exc).replace(secret, "[redacted]") if secret else str(exc)
                    save(
                        submission.with_suffix(f".attempt{attempt}.json"),
                        {"status": "failed_proposal", "reason": reason[:1500]},
                    )
            else:
                raise RuntimeError(f"LLM proposal g{generation:02d}-p{index:02d} failed; progress preserved")
        return identifier, self.evaluate(identifier, generation)

    def _execute_locked(self, client):
        from .store import Store

        self.progress(
            status="preflight",
            generations=self.cfg["generations"],
            candidates_per_generation=self.cfg["candidates_per_generation"],
            model=self.cfg["model"],
            reasoning_effort=self.cfg["reasoning_effort"],
        )
        contract = {
            "protocol": self.cfg,
            "code": code_fingerprint(),
            "data": input_fingerprint(self.cfg),
            "reference": reference_document(self.cfg),
            "reference_bundle": reference_bundle(),
            "opponents": baseline(),
        }
        contract_path = self.output / "run-contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("resume requires identical code, data, config, reference and opponents")
        save(contract_path, contract)
        self.store = Store(self.output / "lineage")
        root = self.store.stage(reference_bundle(), origin={"generation": 0, "role": "existing_cajal_v1"})
        if self.validate(root, reference_bundle())["status"] != "accepted":
            raise ValueError("initial existing Cajal reference failed synthetic acceptance")
        reference = self.evaluate(root, 0)
        if reference["status"] != "complete":
            raise ValueError("initial reference requires the configured seed and all five tasks")
        best = baseline()["best_opponents"]
        state_path = self.output / "campaign.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"champion": root, "generations": [], "processed": []}
        )
        self.progress(status="evolution", completed_generations=len(state["generations"]), champion=state["champion"])
        for generation in range(1, self.cfg["generations"] + 1):
            if any(g["generation"] == generation for g in state["generations"]):
                continue
            nodes = self.store.nodes()
            allowed = [n["candidate_id"] for n in nodes if n.get("status") == "complete" and n.get("all_seeds_passed")]
            seen = {semantic_id(self.store.bundle(n["candidate_id"])) for n in nodes}
            seen_lock = threading.Lock()
            context = {
                "protocol": self.cfg["protocol_id"],
                "reference_candidate": root,
                "current_champion": state["champion"],
                "recommended_parents": selection.parents(nodes, self.store, best),
                "complete_evolution_dag": self.store.tree_summary(),
                "frozen_opponents": baseline(),
                "generation_policy": "parallel cohort; parents and champion frozen at generation start; decisions applied in proposal order",
            }
            completed, failures = {}, []
            self.progress(
                status="evolution",
                generation=generation,
                phase="parallel_proposals_and_evaluation",
                completed_candidates_in_generation=0,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg["max_workers"]) as pool:
                futures = {
                    pool.submit(
                        self.run_proposal,
                        client,
                        generation,
                        index,
                        root,
                        state["champion"],
                        allowed,
                        seen,
                        seen_lock,
                        context,
                    ): index
                    for index in range(self.cfg["candidates_per_generation"])
                }
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    try:
                        completed[index] = future.result()
                    except Exception as exc:  # noqa: BLE001 - finish other lanes and leave failed proposals resumable
                        failures.append({"proposal": index, "error": str(exc)[:1500]})
                    self.progress(completed_candidates_in_generation=len(completed), proposal_failures=failures)
            if failures:
                self.progress(status="interrupted", proposal_failures=failures)
                raise RuntimeError("parallel proposal failed; other completed candidates preserved for resume")
            identifiers = []
            # Completion order must not change the champion trajectory or a resumed decision.
            for index in range(self.cfg["candidates_per_generation"]):
                identifier, node = completed[index]
                identifiers.append(identifier)
                label = f"g{generation:02d}-p{index:02d}"
                if label not in state["processed"]:
                    decision_path = self.output / "decisions" / (label + ".json")
                    if decision_path.exists():
                        decision = json.loads(decision_path.read_text())
                        if decision["candidate_id"] != identifier or decision["previous_champion"] != state["champion"]:
                            raise ValueError("inconsistent saved champion decision")
                    else:
                        gate = {"passed": False, "reason": "incomplete candidate"}
                        if node["status"] == "complete":
                            gate = selection.promote(
                                node,
                                self.store.node(state["champion"]),
                                reference,
                                best,
                                self.store.bundle(identifier)["target_metrics"],
                            )
                        decision = {
                            "candidate_id": identifier,
                            "previous_champion": state["champion"],
                            "gate": gate,
                            "next_champion": identifier if gate["passed"] else state["champion"],
                        }
                        save(decision_path, decision)
                    state["champion"] = decision["next_champion"]
                    state["processed"].append(label)
                save(state_path, state)
            state["generations"].append(
                {"generation": generation, "candidates": identifiers, "champion": state["champion"]}
            )
            save(state_path, state)
            self.progress(
                completed_generations=len(state["generations"]), champion=state["champion"], phase="generation_complete"
            )
        self.report(state, root)
        self.progress(status="complete", completed_generations=len(state["generations"]), phase="complete")
        return state

    def report(self, state, root):
        with (self.output / "standard_metrics.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "candidate_id",
                    "generation",
                    "seed",
                    "dissociated_cells",
                    "spatial_groups",
                    "max_nonzero_tokens",
                    "zero_tokens",
                    *METRICS,
                ]
            )
            for node in self.store.nodes():
                if node.get("status") == "complete":
                    for result in node["seed_results"]:
                        data = result.get("training_parameters", {}).get("DataConfig", {})
                        writer.writerow(
                            [
                                node["candidate_id"],
                                node["generation"],
                                result["seed"],
                                *[
                                    data.get(k)
                                    for k in (
                                        "dissociated_cells",
                                        "spatial_groups",
                                        "max_nonzero_tokens",
                                        "zero_tokens",
                                    )
                                ],
                                *[result["metrics"][k] for k in METRICS],
                            ]
                        )
        save(
            self.output / "frozen-champion.json",
            {
                "candidate_id": state["champion"],
                "bundle": self.store.bundle(state["champion"]),
                "resolved_training_config": resolved_document(self.store.bundle(state["champion"]), self.cfg, 0),
                "weight_policy": self.cfg["weight_policy"],
                "external_weight_dependencies": [],
                "result": self.store.node(state["champion"]),
                "reference_candidate": root,
                "improvement_found": state["champion"] != root,
                "adoption": "research artifact only; no upstream recipe/tag adoption",
                "scaling_started": False,
            },
        )
        save(
            self.output / "status.json",
            {
                "status": "complete",
                "completed_generations": len(state["generations"]),
                "champion": state["champion"],
                "scaling_started": False,
            },
        )
