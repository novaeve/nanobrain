#!/usr/bin/env python3
"""Run/resume bounded LLM module evolution, paired confirmation and final reporting.

Reuses the established candidate trainer, baseline evaluator and process helpers.
Single-stage mode may intentionally select on test when frozen protocol requests it. All budgets and guards are frozen before training.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.api_config import APIConfig
from nanobrain.campaign import (
    PRIMARY_METRICS, audit_training_receipt, candidate_method, overall_metrics,
    primary_metric_record, read_metric_table,
)
from nanobrain.evolution import (
    architecture_id, joint_score, improvement_gate, confirm_replicates, validate_proposals,
    selected_objectives,
    has_module_change,
)
from nanobrain.genome import CandidateGenome, SEARCH_SPACE
from nanobrain.component_catalog import component_catalog
from nanobrain.llm_client import APIRequestError, OpenAICompatibleClient
from nanobrain.search import CandidateScore, pareto_front
from scripts.run_nanobrain_campaign import (
    _extract_json, _parse_gpus, _write_json, _launch_candidates, _wait_processes,
    _load_training_receipt, _public_candidate, _launch_evaluations, _finalize_evaluations,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def proposal_history(history, recent_generations=5, best_candidates=12):
    """Bound LLM context while retaining full history in per-generation files."""
    if recent_generations < 1 or best_candidates < 1:
        raise ValueError("history limits must be positive")
    candidates = [candidate for generation in history for candidate in generation.get("candidates", [])]
    compact_fields = ("architecture_id", "genome", "metrics", "score", "eligible", "parent_gate", "reference_gate", "failed", "failure_reasons")
    compact = lambda item: {key: item[key] for key in compact_fields if key in item}
    return {
        "completed_generations": len(history), "evaluated_candidates": len(candidates),
        "recent_generations": [dict(generation=g["generation"], candidates=[compact(c) for c in g["candidates"]])
                               for g in history[-recent_generations:]],
        "best_observed_candidates": [compact(c) for c in sorted(candidates, key=lambda c: -c["score"])[:best_candidates]],
    }


def summarize_runs(runs, tasks=None):
    if not runs:
        raise ValueError("cannot summarize an empty replicate group")
    result = copy.deepcopy(runs[0])
    result["replicates"] = runs
    required = [f"{task}.{metric}" for task, metric, _ in selected_objectives(tasks)]
    failures = []
    for run in runs:
        metrics = run.get("metrics", {})
        invalid = [key for key in required
                   if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key])]
        if run.get("failed") or invalid:
            failures.append(dict(seed=run.get("genome", {}).get("seed"),
                                 reasons=run.get("failure_reasons", []), invalid_metrics=invalid))
    if failures:
        # Never promote an architecture using only its successful seeds.
        result.update(failed=True, failure_reasons=failures, metrics={}, diagnostics={}, score=-1.0e9,
                      runtime_seconds=sum(float(x.get("runtime_seconds", 0.0) or 0.0) for x in runs))
        return result
    result["metrics"] = {key: statistics.mean(x["metrics"][key] for x in runs) for key in runs[0]["metrics"]}
    result["score"] = joint_score(result["metrics"], tasks)
    result["runtime_seconds"] = statistics.mean(x["runtime_seconds"] for x in runs)
    return result


def evaluation_checkpoint(receipt, selection):
    """Choose weights before the test evaluator is opened."""
    if selection not in {"best_validation", "final_step_fixed_budget"}:
        raise ValueError(f"unsupported checkpoint_selection: {selection}")
    field = "best_checkpoint" if selection == "best_validation" else "checkpoint"
    path = Path(receipt[field]) if receipt.get(field) else None
    if path is None or not path.is_file():
        raise RuntimeError(f"required {field} missing for {selection}")
    if selection == "best_validation" and int(receipt.get("best_step", 0)) < 1:
        raise RuntimeError("best validation checkpoint has no selected step")
    return path.resolve()


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.protocol = json.loads(args.config.read_text())
        tasks = self.protocol.get("tasks")
        selected_objectives(tasks)
        if self.protocol.get("training_mode", "joint") not in {"joint", "independent"}:
            raise ValueError("unsupported training mode")
        if self.protocol.get("training_mode") == "independent" and not tasks:
            raise ValueError("independent training requires an explicit task list")
        if self.protocol.get("reference_genome", {}).get("architecture") is not None and self.protocol.get("training_mode") != "independent":
            raise ValueError("graph protocol requires independent tasks")
        if self.protocol.get("gate", {}).get("tasks") != tasks:
            raise ValueError("gate task set must equal trained/evaluated task set")
        p = self.protocol
        if p.get("checkpoint_selection") == "best_validation":
            if p.get("checkpoint_policy") not in {"all", "best_weights"} or int(p.get("eval_interval", 0)) < 1:
                raise ValueError("best_validation requires retained best weights and periodic validation")
        for key in ("generations", "candidates_per_generation", "screen_steps", "confirmation_steps"):
            if not isinstance(p[key], int) or p[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if int(p.get("no_improvement_patience", 1)) < 0:
            raise ValueError("no_improvement_patience must be zero (disabled) or positive")
        if not 0 <= int(p.get("minimum_module_change_candidates", 0)) <= p["candidates_per_generation"]:
            raise ValueError("module-change quota exceeds candidate count")
        for key in ("propagation_seeds", "confirmation_seeds"):
            if len(p[key]) < 3 or len(set(p[key])) != len(p[key]):
                raise ValueError(f"{key} needs at least three unique seeds")
        if p["propagation_seeds"][0] != p["screen_seed"]:
            raise ValueError("the first propagation seed must be the screening seed")
        self.data = (ROOT / self.protocol["data"]).resolve()
        from nanobrain.data import verify_subset_manifest
        verify_subset_manifest(self.data, verify_hashes=True, data_splits=("train", "validation"))
        self.state_path = self.output / "campaign.json"
        self.state = {}
        self.api = APIConfig.from_env()
        if self.protocol.get("llm_model"):
            from dataclasses import replace
            self.api = replace(self.api, model=self.protocol["llm_model"])
        if self.protocol.get("llm_timeout_seconds"):
            from dataclasses import replace
            self.api = replace(self.api, timeout_seconds=float(self.protocol["llm_timeout_seconds"]))
        if p.get("llm_reasoning_effort") not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported llm_reasoning_effort")
        if not getattr(args, "task_suite_only", False):
            self.api.validate(require_credentials=True)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(self.protocol["threads_per_process"])
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.output.mkdir(parents=True, exist_ok=True)
        from nanobrain.baseline_components import REFERENCE_FILES
        files = (sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
                 + [ROOT / relative for relative in REFERENCE_FILES])
        from nanobrain.upstream_loader import upstream_fingerprint_files
        files += upstream_fingerprint_files()
        self.fingerprint = {str(p.relative_to(ROOT)): sha(p) for p in files}
        self.fingerprint["data_manifest"] = sha(self.data / "manifest.json")
        import importlib.metadata
        for package in ("torch", "transformers", "scvi-tools", "numpy", "scikit-learn"):
            self.fingerprint[f"package:{package}"] = importlib.metadata.version(package)
        for package in ("celltypist", "harmonypy", "scanorama", "bbknn", "scipy"):
            self.fingerprint[f"package:{package}"] = importlib.metadata.version(package)
        if self.protocol.get("reference_genome", {}).get("architecture") is not None:
            for package in ("jsonpatch", "jsonpointer"):
                self.fingerprint[f"package:{package}"] = importlib.metadata.version(package)
        if self.state_path.exists():
            if not args.resume:
                raise ValueError("campaign exists; use --resume to continue the identical protocol")
            frozen = json.loads((self.output / "protocol.json").read_text())
            if frozen["protocol"] != self.protocol:
                raise ValueError("code, data or protocol changed; use a fresh campaign directory")
            if frozen["fingerprint"] != self.fingerprint:
                if not getattr(args, "adopt_scheduler_update", False):
                    raise ValueError("code, data or protocol changed; use a fresh campaign directory")
                from scripts.nanobrain_scheduler import migrate_scheduler_fingerprint
                migrate_scheduler_fingerprint(self.output, frozen, self.fingerprint, ROOT)
            self.state = json.loads(self.state_path.read_text())
            self.state["training_gpus"] = list(args.gpus)
            if self.state.get("error"):
                self.state.setdefault("recovered_errors", []).append(dict(
                    error=self.state.pop("error"), recovered_at=utc(), previous_status=self.state.get("status")))
        else:
            if any(self.output.iterdir()):
                raise ValueError("fresh campaign output must be empty")
            for p in files:
                target = self.output / "source_snapshot" / p.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
            _write_json(self.output / "protocol.json", dict(protocol=self.protocol, fingerprint=self.fingerprint,
                         frozen_at=utc(), search_space=SEARCH_SPACE, historical_test_already_published=True))
            _write_json(self.output / "component_catalog.json", component_catalog())
            if self.protocol.get("reference_genome", {}).get("architecture") is not None:
                from nanobrain.architecture import graph_catalog
                _write_json(self.output / "graph_component_catalog.json", graph_catalog())
            self.state = dict(protocol_id=self.protocol["protocol_id"], started_at=utc(),
                              generations=[], reference=None, confirmations=[], selected=None,
                              training_gpus=args.gpus, api=self.api.redacted(),
                              llm_reasoning_effort=self.protocol.get("llm_reasoning_effort"),
                              test_used_for_selection=False, status="initialized")
            self.save()

    def save(self, status=None):
        # Historical campaigns are large. Queue heartbeats need not rewrite
        # the entire history every second; terminal transitions remain immediate.
        now = time.monotonic()
        if status == "adaptive_queue_running" and now - getattr(self, "_last_queue_save", 0.0) < 10.0:
            return
        if status == "adaptive_queue_running":
            self._last_queue_save = now
        if status:
            self.state["status"] = status
        self.state["updated_at"] = utc()
        _write_json(self.state_path, self.state)

    def run_batch(self, specs, directory, steps):
        if self.protocol.get("training_mode", "joint") == "joint":
            return self._run_single_batch(specs, directory, steps)
        tasks = self.protocol["tasks"]
        if getattr(self.args, "adaptive_queue", False):
            from scripts.nanobrain_scheduler import run_task_queue
            results = run_task_queue(self, specs, directory, steps, evaluation_checkpoint)
        else:
            results = {task: self._run_single_batch(specs, directory / task, steps, task=task) for task in tasks}
        merged = []
        from scripts.run_nanobrain_baselines import _write_rows
        for index, spec in enumerate(specs):
            records = {task: results[task][index] for task in tasks}
            genome = CandidateGenome.from_dict(spec["genome"])
            out = directory / f"candidate_{index:02d}"
            out.mkdir(parents=True, exist_ok=True)
            failed_tasks = {
                task: record.get("failure_reasons", ["task_training_failed"])
                for task, record in records.items() if record.get("failed")
            }
            if failed_tasks:
                result = dict(
                    spec, index=index, genome=genome.to_dict(), candidate_id=genome.genome_id,
                    architecture_id=architecture_id(genome.to_dict()), directory=str(out.resolve()),
                    steps=steps, task_runs=records, training_mode="independent", metrics={},
                    diagnostics={}, score=-1.0e9, parameter_count=None,
                    runtime_seconds=sum(float(x.get("runtime_seconds", 0.0) or 0.0) for x in records.values()),
                    training_audit={"passed": False, "failed_tasks": failed_tasks},
                    failed=True, failure_reasons=failed_tasks,
                )
                _write_json(out / "result.json", result)
                merged.append(result)
                continue
            bundle = out / "task_checkpoints.json"
            _write_json(bundle, dict(format="nanobrain_task_checkpoints_v1", genome=genome.to_dict(),
                                    tasks={task: dict(path=x["checkpoint"], sha256=x["checkpoint_sha256"],
                                                     checkpoint_selection=x.get("checkpoint_selection"),
                                                     selected_step=x.get("selected_step"))
                                           for task, x in records.items()}))
            table = out / "evaluation" / "benchmark_table.csv"
            rows = [row for x in records.values() for row in read_metric_table(x["evaluation"])]
            _write_rows(table, rows)
            metrics = overall_metrics(rows, candidate_method(genome.genome_id))
            result = dict(spec, index=index, genome=genome.to_dict(), candidate_id=genome.genome_id,
                          architecture_id=architecture_id(genome.to_dict()), directory=str(out.resolve()),
                          steps=steps, task_runs=records, training_mode="independent",
                          checkpoint=str(bundle.resolve()), checkpoint_sha256=sha(bundle),
                          evaluation=str(table.resolve()), evaluation_sha256=sha(table),
                          parameter_count=sum(x["parameter_count"] for x in records.values()),
                          runtime_seconds=sum(x["runtime_seconds"] for x in records.values()),
                          metrics=primary_metric_record(metrics),
                          diagnostics={f"{k[0]}.{k[1]}": v for k, v in metrics.items()},
                          training_audit=dict(passed=True, checkpoint=str(bundle.resolve())))
            result["score"] = joint_score(result["metrics"], tasks)
            _write_json(out / "result.json", result)
            merged.append(result)
        return merged

    def _run_single_batch(self, specs, directory, steps, task="joint"):
        """Execute GPU-sized waves; completed receipts support interruption recovery."""
        results = []
        for start in range(0, len(specs), len(self.args.gpus)):
            chunk = specs[start:start + len(self.args.gpus)]
            wave = []
            processes = []
            for i, spec in enumerate(chunk):
                index = start + i
                genome = CandidateGenome.from_dict(spec["genome"])
                out = directory / f"candidate_{index:02d}"
                result_path = out / "result.json"
                if result_path.is_file():
                    saved = json.loads(result_path.read_text())
                    if (saved["genome"] != genome.to_dict() or saved["steps"] != steps
                            or saved.get("training_task", "joint") != task):
                        raise ValueError("cannot resume a candidate with changed budget or genome")
                    if saved.get("failed"):
                        results.append(saved)
                        continue
                    for name in ("checkpoint", "evaluation"):
                        if sha(saved[name]) != saved[name + "_sha256"]:
                            raise ValueError("resumed artifact checksum mismatch")
                    results.append(saved)
                    continue
                item = dict(spec, index=index, directory=str(out), candidate_id=genome.genome_id,
                            genome=genome.to_dict(), steps=steps)
                receipt = _load_training_receipt(item, steps)
                if receipt is None:
                    launched = _launch_candidates(
                        genomes=[genome], gpus=[self.args.gpus[i]], data=self.data,
                        output=directory, steps=steps, seed=genome.seed,
                        batch_size=self.protocol["batch_size"],
                        validation_batch_size=self.protocol["validation_batch_size"],
                        precision=self.protocol["precision"],
                        development_cells=self.protocol["development_cells"],
                        validation_development_groups=self.protocol["validation_development_groups"],
                        eval_interval=self.protocol["eval_interval"],
                        loss_balance=self.protocol["loss_balance"],
                        loss_balance_decay=self.protocol["loss_balance_decay"],
                        integration_adversarial_warmup_steps=self.protocol.get("integration_adversarial_warmup_steps", 500),
                        integration_logit_clip=self.protocol.get("integration_logit_clip", 20.0),
                        stability_max_loss=self.protocol.get("stability_max_loss", 1.0e8),
                        stability_max_gradient_norm=self.protocol.get("stability_max_gradient_norm", 1.0e6),
                        directory_names=[out.name],
                        task=task,
                        checkpoint_policy=self.protocol.get("checkpoint_policy", "all"),
                    )[0]
                    item.update(launched)
                    item["index"] = index
                    processes.append(item)
                else:
                    item["returncode"] = 0
                wave.append(item)
            self.state["active_runs"] = [dict(candidate_id=x["candidate_id"], directory=x["directory"],
                                              pid=x.get("pid"), gpu=x.get("gpu"), steps=steps) for x in wave]
            self.save("training")
            if processes:
                _wait_processes(processes, manifest=self.state, manifest_path=self.state_path, phase="training")
            for item in wave:
                receipt = _load_training_receipt(item, steps)
                audit = audit_training_receipt(receipt or {})
                if receipt is not None and receipt.get("training_task", "joint") != task:
                    raise RuntimeError("training receipt task mismatch")
                if not audit["passed"] or item.get("returncode") != 0:
                    item["failed"] = True
                    item["failure_reasons"] = audit["reasons"] or [f"training_exit_{item.get('returncode')}"]
                    item["training_audit"] = audit
                    # A fixed reference failure leaves no valid comparison
                    # point; a proposed candidate is recorded and skipped.
                    if item.get("proposal_source") in {"fixed_reference", "matched_reference"}:
                        raise RuntimeError(f"reference training failed: {item['directory']}; {audit['reasons']}")
                    results.append(_public_candidate(item))
                    continue
                selection = self.protocol.get("checkpoint_selection", "final_step_fixed_budget")
                checkpoint = evaluation_checkpoint(receipt, selection)
                item["checkpoint_selection"] = selection
                item["selected_step"] = (receipt["best_step"] if selection == "best_validation"
                                         else receipt["completed_steps"])
                item["selected_validation_quality"] = (receipt["best_validation_quality"]
                                                         if selection == "best_validation" else receipt["quality"])
                item["training_audit"] = {**audit, "checkpoint": str(checkpoint.resolve())}
                item["parameter_count"] = receipt["parameter_count"]
                item["runtime_seconds"] = receipt["runtime_seconds"]
            successful = [item for item in wave if not item.get("failed")]
            if successful:
                evaluated = _launch_evaluations(candidates=successful, gpus=self.args.gpus, data=self.data,
                            batch_size=self.protocol["evaluation_batch_size"],
                            evaluation_seed=self.protocol["evaluation_seed"],
                            evaluation_split=self.protocol.get("selection_evaluation_split", "validation"),
                            output_name="test_selection_evaluation" if self.protocol.get("selection_evaluation_split") == "test" else "evaluation")
                _finalize_evaluations(evaluated, manifest=self.state, manifest_path=self.state_path)
            for item in wave:
                if item.get("failed"):
                    result = _public_candidate(item)
                    result.update(
                        metrics={}, diagnostics={}, score=-1.0e9,
                        architecture_id=architecture_id(item["genome"]),
                        training_task=task,
                        trainable_parameter_count=None,
                        training_audit=item["training_audit"],
                        failed=True,
                        failure_reasons=item["failure_reasons"],
                    )
                    _write_json(Path(item["directory"]) / "result.json", result)
                    # It was already appended above so keep one record per
                    # candidate and preserve index ordering on return.
                    continue
                if item.get("evaluation_returncode") != 0:
                    raise RuntimeError(f"candidate evaluator failed: {item['directory']}")
                rows = read_metric_table(item["evaluation"])
                if any(x.get("status") != "complete" for x in rows):
                    raise RuntimeError("incomplete canonical metric table")
                metrics = overall_metrics(rows, candidate_method(item["candidate_id"]))
                result = _public_candidate(item)
                result.update(metrics=primary_metric_record(metrics),
                              diagnostics={f"{k[0]}.{k[1]}": v for k, v in metrics.items()},
                              architecture_id=architecture_id(item["genome"]),
                              checkpoint=item["training_audit"]["checkpoint"])
                result["training_task"] = task
                result["trainable_parameter_count"] = json.loads(Path(item["receipt"]).read_text()).get("trainable_parameter_count")
                result["score"] = joint_score(result["metrics"], None if task == "joint" else [task])
                for name in ("checkpoint", "evaluation"):
                    result[name + "_sha256"] = sha(result[name])
                _write_json(Path(item["directory"]) / "result.json", result)
                results.append(result)
            self.state["active_runs"] = []
            self.save()
        return sorted(results, key=lambda x: int(x["index"]))

    def propose(self, parents, history, generation, seen):
        folder = self.output / f"generation_{generation:02d}"
        folder.mkdir(exist_ok=True)
        accepted = folder / "accepted_proposals.json"
        exhausted = folder / "proposal_exhausted.json"
        count = self.protocol["candidates_per_generation"]
        seed = self.protocol["screen_seed"]
        if accepted.is_file():
            return json.loads(accepted.read_text())["candidates"]
        if exhausted.is_file():
            return None
        context = lambda x: {k: x[k] for k in ("architecture_id", "genome", "metrics", "score",
                                               "runtime_seconds", "parameter_count", "diagnostics") if k in x}
        prompt = (
            "Propose module combinations for nanoBrain. Respond with JSON {\"candidates\": [...]} only. "
            f"Exactly {count} distinct candidates. Each needs parent_ids (one mutation parent or two "
            "crossover parents), target_tasks (one or more canonical task names), hypothesis, "
            "expected_tradeoff, and genome containing EVERY search_space field and no extra fields. " +
            ("Each candidate is instantiated and trained FROM SCRATCH INDEPENDENTLY for each task. "
             "Tasks have separate weights, optimizers and checkpoints. "
             if self.protocol.get("training_mode") == "independent" else "All five tasks train jointly from scratch. ") +
            "All candidates use the same data/seed/steps per task. The panel_patches encoder makes "
            "16 ordered gene-panel tokens per cell; other encoders make one pooled cell state. "
            "Attention runs before mean-pooling tokens. FMT attends across population cells. "
            "The legacy FMT maps requested flow_depth 1/2/3/4 to actual 2/2/4/4 layers; "
            "scfit_adaln uses exactly flow_depth blocks on each cell's two state/context tokens. "
            "Legacy mlp flow uses interpolation time only; other active flow modules also use age/interval. "
            "Keep flow active (not none). Components are imported baseline primitives trained from scratch, "
            "not the complete baselines or their pretrained checkpoints. Read component_catalog for sources. "
            "Explore both module-family replacements/combinations and hyperparameter changes, retaining strengths. "
            "Explain why the changed modules should improve the named metrics. Test-set metrics are available and are the selection signal in this protocol. "
            "The joint score and task guards below are fixed, not negotiable; runtime/parameters break ties. "
            "Every candidate must improve joint score and obey per-task regression bounds vs its parent "
            "AND the initial reference. High batch-classification accuracy is NOT integration success.\n"
            + json.dumps(dict(search_space={k: list(v) for k, v in SEARCH_SPACE.items()},
                              component_catalog=component_catalog(),
                              protocol=self.protocol, parents=[context(x) for x in parents],
                              initial_reference=context(self.state["reference"]),
                              previous_generations=proposal_history(history,
                                  int(self.protocol.get("llm_history_generations", 5)),
                                  int(self.protocol.get("llm_history_best_candidates", 12)))), ensure_ascii=False)
        )
        if self.protocol.get("reference_genome", {}).get("architecture") is not None:
            from nanobrain.architecture import graph_catalog
            from nanobrain.objectives import objective_catalog
            prompt = (
                'Propose nanoBrain computation graphs. Respond with JSON {"candidates": [...]} only. '
                f'Exactly {count} distinct candidates. Each has parent_ids, target_tasks, hypothesis, expected_tradeoff, '
                'and EITHER genome containing exactly architecture and learning_rate OR patch containing RFC6902 JSON Patch '
                'operations against one current parent genome (seed excluded). Prefer compact patches for small edits. '
                'Patch paths may only target /architecture/... or /learning_rate. Never return both genome and patch. '
                'Use ONLY the current parent IDs below. '
                'Each candidate trains FROM SCRATCH INDEPENDENTLY for each task; checkpoints/optimizers are separate. '
                'Architecture versions 2 and 3 describe backbone and velocity-field graphs. Version 3 also exposes '
                'batch_id[B,1] (technical batch IDs, unknown=-1) and nonnegative_features[B,G] '
                '(frozen train-normalization inverse, no biology labels). scpoli_conditional takes [features_or_state,batch_id]. '
                'scprint_value requires nonnegative_features; scfoundation_autobin/scprint_value produce gene tokens and need pooling. '
                'scpoli_prototype is annotation/niche only, without a pre graph; train-only prototypes are checkpointed. '
                'Optional architecture.endpoint selects an endpoint transport, requires version 3 and endpoint_transport objective; '
                'it is not a velocity node. You may insert, remove, '
                'reorder, replace and repeat compatible nodes, and construct parallel paths with add/subtract/concat/stack. '
                'The nodes list is topologically ordered; inputs reference prior nodes or declared external inputs. '
                'Backbone input is features[B,G]; flow inputs are state[P,H], context[P,H], time[P,3] '
                '(tau,source_years,delta_years). Both graph outputs must be rank-2 with architecture.width. '
                'Every node must contribute to its graph output; flow must depend on all three inputs. '
                'Each node has id,kind,name,inputs,params and optional repeat (1..8). Max 32 nodes/64 expanded layers. '
                'Each repeated layer has independent weights; repeat only unary shape-preserving nodes. '
                'hidden_dim may vary at projections; add/subtract require equal shapes; concat joins features; '
                'stack creates tokens from cell states. Token pooling is an explicit operation. '
                'Attention on a rank-2 cell state has one token; use patch tokens or stack for token interactions. '
                'Current metadata supports at most ONE nonidentity spatial hop along any path. '
                'Normalization, scBrain FFN, scFiT FFN, scFiT attention and scFiT time encoding are independently available. '
                'NEW components: head/celltypist_linear is the CellTypist linear logistic decision function, newly initialized. '
                'operation/harmony, operation/scanorama, operation/bbknn are cached dataset-level input components. '
                'They require inputs=["features"], repeat=1, and strength in [0,1]; no hidden_dim parameter. '
                'Use them ONLY in the backbone, before an encoder. They preserve the input gene width, not architecture.width. '
                'They run official algorithms independently within train/validation using expression and technical batch IDs only. '
                'They do not consume spatial geometry or biological labels. Parallel raw/corrected branches may be merged. '
                'Do not place them on learned embeddings, in heads, or in the flow graph. '
                'In every generation include at least one candidate for each of harmony, scanorama, bbknn and celltypist_linear; '
                'Retain this input-component coverage while exploring all 17 newly registered Graph V3 adapters across generations: '
                'scTab, STAGATE, scPoli, MIOFlow, PRESCIENT, scNym, SEDR, SpaGCN, scFoundation, scPRINT, scBERT and CellOT. '
                'Only explicit factory parameters in the catalog are allowed; input_dim/output_dim are inferred. '
                'Each head has its own name and params, and optional pre graph (input state[B,H]) before its output projection. '
                'Pairing, solver, and task loss params are configurable. '
                'Evaluators, split boundaries, label access, and required positive training objectives stay fixed. '
                'Test-set metrics are available and are the selection signal in this protocol. Judge using fixed joint score and per-task guards, not self-reported scores. '
                'Formatting/ID-only changes are not new architectures. Full-genome proposals require complete architecture objects; '
                'patch proposals are expanded and validated by the runner.\n' +
                json.dumps(dict(component_catalog=graph_catalog(), objective_catalog=objective_catalog(),
                    learning_rates=SEARCH_SPACE["learning_rate"], protocol=self.protocol,
                    parents=[context(x) for x in parents], initial_reference=context(self.state["reference"]),
                    previous_generations=proposal_history(history,
                        int(self.protocol.get("llm_history_generations", 5)),
                        int(self.protocol.get("llm_history_best_candidates", 12)))), ensure_ascii=False))
        (folder / "prompt.txt").write_text(prompt)
        client = OpenAICompatibleClient(self.api)
        attempt = 0
        semantic_attempts = 0
        feedback = ""
        collected: list[dict] = []
        collected_ids = set(seen)
        max_semantic_attempts = max(1, int(self.protocol.get("proposal_semantic_attempts", 8)))
        while True:
            attempt += 1
            self.state["api_generation"] = generation
            self.state["api_attempt"] = attempt
            self.save("waiting_for_llm")
            try:
                remaining = count - len(collected)
                request_options = {}
                if self.protocol.get("llm_reasoning_effort"):
                    request_options["reasoning_effort"] = self.protocol["llm_reasoning_effort"]
                _write_json(folder / f"llm_request_{attempt:03d}.json", dict(
                    model=getattr(self.api, "model", self.protocol.get("llm_model")),
                    max_tokens=int(self.protocol.get("llm_response_max_tokens", 8000)),
                    prompt_sha256=hashlib.sha256((prompt + feedback).encode()).hexdigest(),
                    **request_options))
                response = client.complete(
                    prompt + feedback +
                    f"\nThis is proposal fill request {attempt}. Return exactly {remaining} candidates "
                    "for this response when possible; do not repeat any architecture already present "
                    "in the parent or previous responses.",
                    max_tokens=int(self.protocol.get("llm_response_max_tokens", 8000)),
                    **request_options)
                (folder / f"llm_response_{attempt:03d}.txt").write_text(response)
                payload = _extract_json(response)
                raw = payload.get("candidates") if isinstance(payload, dict) else None
                if not isinstance(raw, list):
                    raise ValueError("response must contain a candidates array")
                accepted_this_response = 0
                rejection_reasons = []
                for proposal in raw:
                    if len(collected) >= count:
                        break
                    try:
                        # Validate each candidate independently.  A duplicate or
                        # malformed item no longer invalidates valid siblings.
                        one = validate_proposals(
                            {"candidates": [proposal]}, parents, count=1,
                            seed=seed, seen=collected_ids
                        )[0]
                        if self.protocol.get("tasks") and set(one["target_tasks"]) - set(self.protocol["tasks"]):
                            raise ValueError("target_tasks must belong to this campaign's evaluated tasks")
                    except (TypeError, ValueError, KeyError) as exc:
                        rejection_reasons.append(str(exc)[:300])
                        continue
                    collected.append(one)
                    collected_ids.add(one["architecture_id"])
                    accepted_this_response += 1
                required_components = set(self.protocol.get("required_component_coverage", []))
                def component_names(proposal):
                    a = proposal["genome"].get("architecture", {})
                    return ({n["name"] for n in a.get("backbone", {}).get("nodes", [])}
                            | {h["name"] for key, h in a.get("heads", {}).items()
                               if key in {"annotation", "integration", "niche"}})
                covered = set().union(*(component_names(x) for x in collected)) if collected else set()
                missing = required_components - covered
                if len(collected) >= count and missing:
                    # Make room for the missing families while retaining existing
                    # coverage. No local proposals substitute for LLM proposals.
                    for index in range(len(collected) - 1, -1, -1):
                        others = collected[:index] + collected[index + 1:]
                        other_coverage = set().union(*(component_names(x) for x in others)) if others else set()
                        if required_components & covered <= other_coverage:
                            collected.pop(index)
                            if count - len(collected) >= len(missing):
                                break
                    rejection_reasons.append("Missing required new component families: " + ", ".join(sorted(missing)))
                if len(collected) >= count:
                    module_changes = sum(has_module_change(x, parents) for x in collected)
                    required_changes = int(self.protocol.get("minimum_module_change_candidates", 0))
                    if module_changes >= required_changes:
                        _write_json(accepted, dict(candidates=collected, accepted_at=utc(), attempt=attempt,
                                                  source="llm", prompt_sha256=sha(folder / "prompt.txt")))
                        return collected
                    # The pool is already full; retaining only numerical
                    # edits cannot be repaired by asking for a zero-sized
                    # response. Preserve the incumbent and skip this round.
                    reason = "proposal batch needs more module-family changes, not only hyperparameter edits"
                    _write_json(exhausted, {
                        "status": "proposal_exhausted", "required": count,
                        "accepted": len(collected),
                        "accepted_architectures": [x["architecture_id"] for x in collected],
                        "attempts": attempt, "semantic_attempts": semantic_attempts + 1,
                        "rejections": [reason], "timestamp": utc(),
                        "incumbent_preserved": True,
                    })
                    self.save("proposal_exhausted")
                    return None
                semantic_attempts += 1
                if semantic_attempts >= max_semantic_attempts:
                    _write_json(exhausted, {
                        "status": "proposal_exhausted",
                        "required": count,
                        "accepted": len(collected),
                        "accepted_architectures": [x["architecture_id"] for x in collected],
                        "attempts": attempt,
                        "semantic_attempts": semantic_attempts,
                        "rejections": rejection_reasons,
                        "timestamp": utc(),
                        "incumbent_preserved": True,
                    })
                    self.save("proposal_exhausted")
                    return None
                feedback = "\nPrevious response yielded only valid candidates listed below or failed item checks. " \
                           "Return new, distinct candidates for the remaining slots. " + \
                           "; ".join(rejection_reasons[-5:])
            except Exception as exc:
                # Network/provider failures keep retrying; semantic responses use
                # the bounded path above and eventually skip the generation.
                if not isinstance(exc, APIRequestError):
                    semantic_attempts += 1
                    if semantic_attempts >= max_semantic_attempts:
                        _write_json(exhausted, {
                            "status": "proposal_exhausted", "required": count,
                            "accepted": len(collected), "attempts": attempt,
                            "semantic_attempts": semantic_attempts,
                            "reason": str(exc)[:1000], "timestamp": utc(),
                            "incumbent_preserved": True,
                        })
                        self.save("proposal_exhausted")
                        return None
                # User requested persistent API retries. Never silently substitute local proposals.
                message = str(exc).replace(self.api.api_key, "[redacted]") if self.api.api_key else str(exc)
                _write_json(folder / f"retry_{attempt:03d}.json", dict(error_type=type(exc).__name__,
                             reason=message[:1000], timestamp=utc()))
                feedback = "\nPrevious response failed validation. Correct this error: " + message[:1000]
                self.save("llm_retry")
                time.sleep(min(30.0, 2.0 * attempt))

    def run(self):
        """Single-stage final-budget evolution with explicit test selection.

        Every candidate is trained at evolution_steps for the configured
        evolution_seeds, evaluated on selection_evaluation_split, and compared
        immediately. There is no short screen, propagation stage, or final
        confirmation stage. This mode intentionally spends test labels during
        search when the protocol says selection_evaluation_split=test.
        """
        if self.state.get("status") == "complete":
            return
        p = self.protocol
        tasks = p["tasks"]
        steps = int(p.get("evolution_steps", p["confirmation_steps"]))
        seeds = [int(x) for x in p.get("evolution_seeds", p["confirmation_seeds"])]
        if len(seeds) < 1 or len(set(seeds)) != len(seeds):
            raise ValueError("evolution_seeds must contain unique seeds")
        selection_split = p.get("selection_evaluation_split", "validation")
        if selection_split not in {"validation", "test"}:
            raise ValueError("selection_evaluation_split must be validation or test")

        control = CandidateGenome.from_dict({**p["reference_genome"], "seed": seeds[0]}).to_dict()
        reference_runs = self.run_batch(
            [dict(genome={**control, "seed": seed}, proposal_source="single_stage_reference")
             for seed in seeds],
            self.output / "reference_single_stage", steps)
        reference = summarize_runs(reference_runs, tasks)
        if reference.get("failed"):
            raise RuntimeError(f"reference replicates failed: {reference['failure_reasons']}")
        reference["replicates"] = reference_runs
        reference["selection_split"] = selection_split
        self.state["reference"] = reference
        self.state["selection_split"] = selection_split
        self.state["test_used_for_selection"] = selection_split == "test"
        parents = [reference]
        incumbent = reference
        history = []
        seen = {reference["architecture_id"]}
        all_eligible = []

        completed_records = list(self.state.get("generations", []))
        for generation in range(int(p["generations"])):
            self.state["current_generation"] = generation
            if generation < len(completed_records):
                # A committed selection is the generation checkpoint. Restore
                # its lineage instead of re-running historical task queues.
                record = completed_records[generation]
                persisted = json.loads((self.output / f"generation_{generation:02d}" / "selection.json").read_text())
                if persisted != record or record.get("generation") != generation:
                    raise ValueError("completed generation checkpoint mismatch")
                if record.get("parent_ids") != [x["architecture_id"] for x in parents]:
                    raise ValueError("completed generation parent lineage mismatch")
                current = record.get("candidates", [])
                eligible = [item for item in current if item.get("eligible")]
                if any(item.get("failed") for item in eligible):
                    raise ValueError("failed candidate in restored parent lineage")
                eligible.sort(key=lambda item: (-item["score"], item.get("runtime_seconds", 0.0)))
                if eligible and eligible[0]["score"] > incumbent["score"]:
                    incumbent = eligible[0]
                if record.get("incumbent_id") != incumbent["architecture_id"]:
                    raise ValueError("completed generation incumbent mismatch")
                if record.get("next_parent_ids") != [incumbent["architecture_id"]]:
                    raise ValueError("completed generation next-parent mismatch")
                parents = [incumbent]
                all_eligible.extend(eligible)
                seen.update(item["architecture_id"] for item in current)
                history.append(dict(generation=generation, candidates=[
                    {k: item[k] for k in ("architecture_id", "genome", "metrics", "score", "eligible", "parent_gate", "reference_gate", "diagnostics", "failed", "failure_reasons") if k in item}
                    for item in current]))
                self.state["restored_generations"] = generation + 1
                print(f"Restored completed generation {generation}; no training or LLM request", flush=True)
                continue
            proposals = self.propose(parents, history, generation, seen)
            if proposals is None:
                record = dict(generation=generation, candidates=[],
                              parent_ids=[x["architecture_id"] for x in parents],
                              promoted_parent_ids=[], next_parent_ids=[incumbent["architecture_id"]],
                              incumbent_id=incumbent["architecture_id"],
                              status="proposal_exhausted_incumbent_retained")
                _write_json(self.output / f"generation_{generation:02d}" / "selection.json", record)
                self.state["generations"] = self.state["generations"][:generation] + [record]
                self.state["incumbent"] = incumbent["architecture_id"]
                self.save("generation_complete_proposal_exhausted")
                history.append(dict(generation=generation, candidates=[]))
                continue

            flat = []
            for proposal in proposals:
                base = dict(proposal)
                for seed in seeds:
                    spec = copy.deepcopy(base)
                    spec["genome"] = {**base["genome"], "seed": seed}
                    spec["proposal_source"] = "llm_single_stage_final"
                    spec["single_stage_seed"] = seed
                    flat.append(spec)
            raw_runs = self.run_batch(flat, self.output / f"generation_{generation:02d}", steps)
            grouped = {}
            for item in raw_runs:
                grouped.setdefault(item["architecture_id"], []).append(item)
            proposal_map = {x["architecture_id"]: x for x in proposals}
            current = []
            for architecture, runs in grouped.items():
                summary = summarize_runs(runs, tasks)
                source = proposal_map.get(architecture, {})
                for key in ("parent_ids", "target_tasks", "hypothesis", "expected_tradeoff", "changes", "proposal_patch", "proposal_source"):
                    if key in source:
                        summary[key] = source[key]
                summary["architecture_id"] = architecture
                summary["replicates"] = runs
                summary["selection_split"] = selection_split
                current.append(summary)
            current.sort(key=lambda x: (-x["score"], x.get("runtime_seconds", 0.0)))
            parent_map = {x["architecture_id"]: x for x in parents}
            eligible = []
            for item in current:
                seen.add(item["architecture_id"])
                if item.get("failed"):
                    item["reference_gate"] = item["parent_gate"] = dict(
                        passed=False, reasons=["replicate_training_or_metrics_failed"])
                    item["eligible"] = False
                    continue
                item["reference_gate"] = improvement_gate(item["metrics"], reference["metrics"], p["gate"])
                valid_parents = [parent_map[x] for x in item.get("parent_ids", []) if x in parent_map]
                parent = max(valid_parents or parents, key=lambda x: x["score"])
                item["parent_gate"] = improvement_gate(item["metrics"], parent["metrics"], p["gate"])
                item["eligible"] = item["reference_gate"]["passed"] and item["parent_gate"]["passed"]
                if item["eligible"]:
                    eligible.append(item)
            eligible.sort(key=lambda x: (-x["score"], x.get("runtime_seconds", 0.0)))
            all_eligible.extend(eligible)
            improved = bool(eligible) and eligible[0]["score"] > incumbent["score"]
            if improved:
                incumbent = eligible[0]
                parents = [incumbent]
                status = "accepted"
            else:
                status = "retained_incumbent_continue"
                parents = [incumbent]
            record = dict(generation=generation, candidates=current,
                          parent_ids=[x["architecture_id"] for x in parent_map.values()],
                          promoted_parent_ids=[x["architecture_id"] for x in eligible],
                          next_parent_ids=[incumbent["architecture_id"]],
                          incumbent_id=incumbent["architecture_id"], status=status,
                          selection_split=selection_split)
            _write_json(self.output / f"generation_{generation:02d}" / "selection.json", record)
            self.state["generations"] = self.state["generations"][:generation] + [record]
            self.state["incumbent"] = incumbent["architecture_id"]
            self.save("generation_complete")
            history.append(dict(generation=generation, candidates=[
                {k: x[k] for k in ("architecture_id", "genome", "metrics", "score", "eligible", "parent_gate", "reference_gate", "diagnostics", "failed", "failure_reasons") if k in x}
                for x in current]))

        self.state["selected"] = incumbent["architecture_id"]
        self.state["scientific_promotion_passed"] = incumbent["architecture_id"] != reference["architecture_id"]
        self.state["stop_reason"] = "generation_budget_exhausted"
        self.state["test_used_for_selection"] = selection_split == "test"
        self.state["single_stage"] = True
        self.state["final_selection"] = dict(selected=incumbent["architecture_id"],
            selection_split=selection_split, seeds=seeds, steps=steps,
            test_used_for_selection=selection_split == "test", frozen_at=utc())
        _write_json(self.output / "selection_freeze.json", self.state["final_selection"])
        self.save("complete")
        self._write_single_stage_report(reference, incumbent, steps, seeds, selection_split)

    def _write_single_stage_report(self, reference, incumbent, steps, seeds, selection_split):
        lines = ["# nanoBrain 单阶段最终预算演化", "", f"协议：{self.protocol['protocol_id']}", "",
                 f"状态：complete；完成代数：{len(self.state.get('generations', []))}；候选数：{sum(len(x.get('candidates', [])) for x in self.state.get('generations', []))}。", "",
                  f"训练预算：每候选每任务 {steps} steps；seed：{seeds}；没有短预算筛选、传播或末尾确认。", "",
                  f"权重选择：{self.protocol.get('checkpoint_selection', 'final_step_fixed_budget')}；每任务每 seed 独立按 validation quality 选择最佳权重，再计算其 test 成绩。", "",
                 f"选择划分：{selection_split}；test_used_for_selection={selection_split == 'test'}。", "",
                 "| 指标 | reference | selected |", "| --- | ---: | ---: |"]
        for key in sorted(set(reference["metrics"]) & set(incumbent["metrics"])):
            lines.append(f"| {key} | {reference['metrics'][key]:.6f} | {incumbent['metrics'][key]:.6f} |")
        lines += ["", "## 解释", "", "LLM 在每一代提案 prompt 中收到 parent、initial reference、近期候选及 best observed candidates 的逐任务 metrics、diagnostics、score 和 gate 信息。", "", "本协议若选择 test，则这些 test 指标也会进入 prompt 并驱动 parent promotion；该结果是 test-driven exploratory search，不能作为未触碰测试集的泛化报告。", ""]
        (self.output / "REPORT.md").write_text("\n".join(lines))

    def report(self):
        rows = []
        for generation in self.state.get("generations", []):
            for x in generation["candidates"]:
                rows.append(dict(generation=generation["generation"], architecture=x["architecture_id"],
                                 source=x["proposal_source"], score=x["score"], eligible=x["eligible"],
                                 **x["metrics"]))
        if rows:
            with (self.output / "search_history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        lines = ["# nanoBrain 100k LLM 模块演化", "", f"协议：{self.protocol['protocol_id']}", "",
                 f"状态：{self.state['status']}；完成代数：{len(self.state.get('generations', []))}；LLM 候选数：{len(rows)}。", "",
                 "固定 60k/20k/20k；搜索和选择只用 validation。历史 test 已公开，最终仅作冻结后的描述性复评。", "",
                 f"训练模式：{self.protocol.get('training_mode', 'joint')}；每任务独立训练时分别保存权重。短预算 {self.protocol['screen_steps']} steps；晋级父代前进行 3-seed 重测；确认阶段 3 seeds × {self.protocol['confirmation_steps']} steps。原始模型与候选使用配对 seed。", "",
                 f"晋级规则：{json.dumps(self.protocol['gate'], ensure_ascii=False)}；配置内任务的综合分与逐任务退化限制。", "",
                 "| 代 | 候选 | 综合分 | 可晋级 |", "| --- | --- | ---: | --- |"]
        for x in rows:
            lines.append(f"| {x['generation']} | {x['architecture']} | {x['score']:.6f} | {x['eligible']} |")
        lines += ["", f"停止原因：{self.state.get('stop_reason', '仍在执行')}",
                  f"多 seed 验证后晋级：{self.state.get('scientific_promotion_passed', '待验证')}",
                  f"保留架构：{self.state.get('selected', '待选择')}", ""]
        for confirmed in self.state.get("confirmations", []):
            gate = confirmed["gate"]
            lines += [f"## 配对长预算验证：{confirmed['screen_architecture_id']}", "",
                      f"通过晋级：{gate['passed']}；三个 seed 的综合分差值：{gate['paired_score_deltas']}", "",
                      "| 指标 | 原始模型均值 | 演化模型均值 |", "| --- | ---: | ---: |"]
            for metric, value in gate["mean_metrics"].items():
                lines.append(f"| {metric} | {gate['reference_mean_metrics'][metric]:.6f} | {value:.6f} |")
            lines.append("")
        long_rows = []
        groups = [("reference", self.state.get("confirmation_reference", []))]
        groups += [(x["screen_architecture_id"], x["runs"]) for x in self.state.get("confirmations", [])]
        for phase, candidates in [("validation", groups), ("test_reevaluation", list(self.state.get("test_reevaluation", {}).items()))]:
            for label, runs in candidates:
                for item in runs:
                    for row in read_metric_table(item["evaluation"]):
                        long_rows.append(dict(phase=phase, architecture=label, seed=item["genome"]["seed"], **row))
        if long_rows:
            with (self.output / "final_metrics_long.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(long_rows[0]))
                writer.writeheader(); writer.writerows(long_rows)
        for name, runs in self.state.get("test_reevaluation", {}).items():
            all_metrics = [overall_metrics(read_metric_table(x["evaluation"]), candidate_method(x["candidate_id"])) for x in runs]
            lines += [f"## {name}：冻结后 test 复评", "", "| 任务 | 指标 | Mean ± SD |", "| --- | --- | ---: |"]
            for key in sorted(set.intersection(*(set(m) for m in all_metrics))):
                values = [m[key] for m in all_metrics]
                lines.append(f"| {key[0]} | {key[1]} | {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f} |")
            lines.append("")
        (self.output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nanobrain_graph_v2/test_driven_val_best_50g_20261029.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=_parse_gpus, default=[3, 5, 6])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adaptive-queue", action="store_true",
                        help="share one nonblocking queue across tasks, training and evaluations")
    parser.add_argument("--workers-per-gpu", type=int, choices=tuple(range(1, 9)), default=5)
    parser.add_argument("--max-workers", type=int, default=None, help="hard global training/evaluation concurrency cap")
    parser.add_argument("--allow-shared-gpus", action="store_true",
                        help="admit jobs on all requested shared GPUs, subject to memory reservations")
    parser.add_argument("--adopt-workers", type=Path,
                        help="JSON record of live trainers to observe rather than restart")
    parser.add_argument("--adopt-scheduler-update", action="store_true",
                        help="on resume, audit and adopt changes to the two scheduler source files only")
    parser.add_argument("--task-suite-only", action="store_true",
                        help="train/evaluate the reference task suite only, without LLM; evaluation split follows the protocol")
    args = parser.parse_args()
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be positive")
    experiment = Experiment(args)
    try:
        if args.task_suite_only:
            genome = {**experiment.protocol["reference_genome"], "seed": experiment.protocol["screen_seed"]}
            runs = experiment.run_batch([dict(genome=genome, proposal_source="task_suite_preflight")],
                                        experiment.output / "task_suite", experiment.protocol["screen_steps"])
            experiment.state["task_suite"] = runs
            experiment.save("task_suite_complete")
        else:
            experiment.run()
    except Exception as exc:
        experiment.state["error"] = f"{type(exc).__name__}: {exc}"
        experiment.save("failed")
        raise


if __name__ == "__main__":
    main()
