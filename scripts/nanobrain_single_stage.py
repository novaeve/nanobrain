"""Single-stage, matched-budget evolution adapted from the existing test-driven run."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import statistics
from pathlib import Path

from nanobrain.evolution import architecture_id, improvement_gate
from nanobrain.genome import CandidateGenome
from nanobrain.campaign import candidate_method, overall_metrics, read_metric_table
from brainwm_benchmark.protocol import metric_role
from scripts.run_nanobrain_campaign import _write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def compact_generation(record):
    """Full provenance lives in selection.json and per-worker result.json."""
    keys = ("architecture_id", "candidate_id", "metrics", "score", "eligible", "failed",
            "failure_reasons", "parent_ids", "target_tasks", "runtime_seconds", "parameter_count",
            "parent_gate", "reference_gate", "directory")
    return {**record, "candidates": [{k: item[k] for k in keys if k in item}
                                      for item in record.get("candidates", [])],
            "state_record_format": "compact_generation_v1"}


def run_single_stage(self, summarize_runs):
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
    self.state["incumbent_result"] = reference
    self.state["selection_split"] = selection_split
    self.state["test_used_for_selection"] = selection_split == "test"
    parents = [reference]
    incumbent = reference
    history = []
    seen = {reference["architecture_id"]}

    for generation in range(int(p["generations"])):
        self.state["current_generation"] = generation
        proposals = self.propose(parents, history, generation, seen)
        if proposals is None:
            record = dict(generation=generation, candidates=[],
                          parent_ids=[x["architecture_id"] for x in parents],
                          promoted_parent_ids=[], next_parent_ids=[incumbent["architecture_id"]],
                          incumbent_id=incumbent["architecture_id"],
                          status="proposal_exhausted_incumbent_retained")
            _write_json(self.output / f"generation_{generation:02d}" / "selection.json", record)
            self.state["generations"] = self.state["generations"][:generation] + [compact_generation(record)]
            self.state["incumbent"] = incumbent["architecture_id"]
            self.state["incumbent_result"] = incumbent
            self.save("generation_complete_proposal_exhausted")
            self.report()
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
        self.state["generations"] = self.state["generations"][:generation] + [compact_generation(record)]
        self.state["incumbent"] = incumbent["architecture_id"]
        self.state["incumbent_result"] = incumbent
        self.save("generation_complete")
        self.report()
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
    write_single_stage_report(self)


def write_single_stage_report(experiment):
    p, state, output = experiment.protocol, experiment.state, experiment.output
    split = p.get("selection_evaluation_split", "validation")
    reference = state.get("reference")
    incumbent = state.get("incumbent_result", reference)
    lines = ["# nanoBrain V3 single-stage evolution", "",
             f"Status: {state.get('status')}; completed generations: {len(state.get('generations', []))}/{p['generations']}.", "",
             f"Selection split: **{split}**. Validation selects weights; {split} metrics select architectures and enter LLM prompts.", "",
             "Test-driven results are exploratory search results, not untouched holdout generalization." if split == "test"
             else "Architecture selection uses validation only.", "",
             f"Each generation: {p['candidates_per_generation']} architectures × {len(p['evolution_seeds'])} seeds × {len(p['tasks'])} tasks; {p['evolution_steps']} steps per task/seed.", "",
             "Search utilities are internal engineering rules. Standard scientific metrics are reported separately below.", "",
             "| Generation | Architecture | Internal search utility | Eligible |",
             "|---:|---|---:|---|"]
    for generation in state.get("generations", []):
        for item in generation.get("candidates", []):
            lines.append(f"| {generation['generation']} | {item['architecture_id']} | {item['score']:.6f} | {item.get('eligible', False)} |")
    full = []
    for label, item in [("reference", reference), ("selected", incumbent)]:
        if not item or item.get("failed"):
            continue
        runs = item.get("replicates", [])
        metrics = []
        for run in runs:
            table = read_metric_table(run["evaluation"])
            metrics.append(overall_metrics(table, candidate_method(run["candidate_id"])))
            full.extend(dict(architecture=label, seed=run["genome"]["seed"],
                             selection_split=split, **row) for row in table)
        lines += ["", f"## {label}: standard metrics", "",
                  "| Task | Metric | Mean | SD | Seeds |", "|---|---|---:|---:|---:|"]
        for key in sorted(set.intersection(*(set(m) for m in metrics))) if metrics else []:
            if metric_role(*key) != "benchmark":
                continue
            values = [m[key] for m in metrics]
            sd = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "NA"
            lines.append(f"| {key[0]} | {key[1]} | {statistics.mean(values):.6f} | {sd} | {len(values)} |")
    if full:
        standard = [r for r in full if r.get("status") == "complete"
                    and r.get("stratum_type") == "overall" and r.get("stratum_value") == "overall"
                    and metric_role(r.get("task", ""), r.get("metric", "")) == "benchmark"]
        for filename, rows in [("final_metrics_long.csv", full), ("final_standard_metrics_long.csv", standard)]:
            with (output / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(full[0]))
                writer.writeheader(); writer.writerows(rows)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
