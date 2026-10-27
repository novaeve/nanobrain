"""Frozen objectives, lineage checks and promotion rules for LLM evolution."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any

from .campaign import PRIMARY_METRICS, genome_diff
from .genome import CandidateGenome, SEARCH_SPACE


def architecture_id(genome: dict) -> str:
    value = CandidateGenome.from_dict(genome).to_dict()
    value.pop("seed")
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def selected_objectives(tasks=None):
    if tasks is None:
        # The public evolution score is the four-task core; spatial completion is optional.
        return tuple(x for x in PRIMARY_METRICS if x[0] in {"annotation", "integration", "spatial_niche", "development"})
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - {x[0] for x in PRIMARY_METRICS}:
        raise ValueError("invalid or duplicated evaluation tasks")
    return tuple(x for x in PRIMARY_METRICS if x[0] in tasks)


def joint_score(metrics: dict[str, float], tasks=None) -> float:
    """Five equally weighted, bounded utilities; higher is always better."""
    values = []
    for task, metric, direction in selected_objectives(tasks):
        key = f"{task}.{metric}"
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite primary metric: {key}")
        if direction == "min":
            if value < 0:
                raise ValueError(f"negative error: {key}")
            value = 1.0 / (1.0 + value)
        elif metric == "ari":
            if not -1 <= value <= 1:
                raise ValueError("ARI outside [-1, 1]")
            value = (value + 1.0) / 2.0
        elif not 0 <= value <= 1:
            raise ValueError(f"metric outside [0, 1]: {key}")
        values.append(value)
    return statistics.mean(values)


def improvement_gate(candidate: dict[str, float], reference: dict[str, float], policy: dict) -> dict:
    """Improve joint quality without material per-task regression vs reference."""
    reasons = []
    checks = []
    try:
        delta = joint_score(candidate, policy.get("tasks")) - joint_score(reference, policy.get("tasks"))
        if delta < float(policy["minimum_joint_gain"]):
            reasons.append("joint_gain_below_threshold")
        for task, metric, direction in selected_objectives(policy.get("tasks")):
            key = f"{task}.{metric}"
            observed, base = float(candidate[key]), float(reference[key])
            tolerance = (float(policy["max_score_drop"]) if direction == "max"
                         else float(policy["max_error_relative_increase"]) * max(abs(base), 1e-12))
            deterioration = base - observed if direction == "max" else observed - base
            passed = deterioration <= tolerance + 1e-12
            checks.append(dict(metric=key, candidate=observed, reference=base,
                               deterioration=deterioration, allowance=tolerance, passed=passed))
            if not passed:
                reasons.append(f"task_regression:{key}")
    except (KeyError, ValueError, TypeError) as exc:
        delta = None
        reasons.append(f"invalid_metrics:{exc}")
    return dict(passed=not reasons, joint_gain=delta, checks=checks, reasons=reasons)


def confirm_replicates(candidates: list[dict], references: list[dict], policy: dict) -> dict:
    if len(candidates) < 3 or len(candidates) != len(references):
        raise ValueError("confirmation requires at least three paired seeds")
    candidate_seeds = [int(x["genome"]["seed"]) for x in candidates]
    reference_seeds = [int(x["genome"]["seed"]) for x in references]
    if len(set(candidate_seeds)) != len(candidate_seeds) or set(candidate_seeds) != set(reference_seeds):
        raise ValueError("candidate and reference must have the same independent seeds")
    if len({architecture_id(x["genome"]) for x in candidates}) != 1:
        raise ValueError("candidate replicate architectures differ")
    if len({architecture_id(x["genome"]) for x in references}) != 1:
        raise ValueError("reference replicate architectures differ")
    reference_by_seed = {x["genome"]["seed"]: x for x in references}
    mean = lambda runs: {key: statistics.mean(x["metrics"][key] for x in runs)
                         for key in runs[0]["metrics"]}
    aggregate = improvement_gate(mean(candidates), mean(references), policy)
    deltas = [joint_score(x["metrics"], policy.get("tasks")) - joint_score(reference_by_seed[x["genome"]["seed"]]["metrics"], policy.get("tasks"))
              for x in candidates]
    wins = sum(value > 0 for value in deltas)
    per_seed = [improvement_gate(x["metrics"], reference_by_seed[x["genome"]["seed"]]["metrics"],
                                {**policy, "minimum_joint_gain": -1.0}) for x in candidates]
    return dict(passed=aggregate["passed"] and wins >= 2 and all(x["passed"] for x in per_seed),
                aggregate_gate=aggregate, paired_score_deltas=deltas, improving_seeds=wins,
                per_seed_regression_gates=per_seed, seeds=candidate_seeds,
                mean_metrics=mean(candidates), reference_mean_metrics=mean(references))


def validate_proposals(payload: Any, parents: list[dict], *, count: int, seed: int,
                       seen: set[str]) -> list[dict]:
    """Reject silent defaults, unregistered edits, fabricated ancestry and duplicates."""
    proposals = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(proposals, list) or len(proposals) != count:
        raise ValueError(f"expected exactly {count} candidates")
    parent_map = {x["architecture_id"]: x for x in parents}
    accepted = []
    identities = set(seen)
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("candidate must be an object")
        parent_ids = proposal.get("parent_ids")
        if not isinstance(parent_ids, list) or not 1 <= len(parent_ids) <= 2:
            raise ValueError("one mutation parent or two crossover parents required")
        if len(set(parent_ids)) != len(parent_ids) or any(x not in parent_map for x in parent_ids):
            raise ValueError("unknown or duplicated parent")
        tasks = proposal.get("target_tasks")
        if not isinstance(tasks, list) or not tasks or any(x not in {v[0] for v in PRIMARY_METRICS} for x in tasks):
            raise ValueError("target_tasks must identify at least one evaluator task")
        for field in ("hypothesis", "expected_tradeoff"):
            if not isinstance(proposal.get(field), str) or not proposal[field].strip():
                raise ValueError(f"missing {field}")
        raw = proposal.get("genome")
        graph_mode = all(parent_map[p]["genome"].get("architecture") is not None for p in parent_ids)
        if graph_mode and "patch" in proposal:
            import copy
            import jsonpatch
            if raw is not None or len(parent_ids) != 1:
                raise ValueError("JSON Patch proposal needs one parent and no genome field")
            patch = proposal["patch"]
            if not isinstance(patch, list) or not 1 <= len(patch) <= 64:
                raise ValueError("patch requires 1..64 operations")
            for op in patch:
                if not isinstance(op, dict) or op.get("op") not in {"add", "remove", "replace", "move", "copy", "test"}:
                    raise ValueError("unsupported JSON Patch operation")
                for path in [op.get("path"), *([op.get("from")] if op["op"] in {"move", "copy"} else [])]:
                    if not isinstance(path, str) or not (path == "/learning_rate" or path.startswith("/architecture/")):
                        raise ValueError("patch paths are restricted to architecture and learning_rate")
            raw = copy.deepcopy(parent_map[parent_ids[0]]["genome"]); raw.pop("seed", None)
            try:
                raw = jsonpatch.apply_patch(raw, patch, in_place=False)
            except (jsonpatch.JsonPatchException, KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"invalid architecture patch: {exc}") from exc
        if graph_mode:
            if not isinstance(raw, dict) or set(raw) != {"architecture", "learning_rate"}:
                raise ValueError("graph genome requires exactly architecture and learning_rate; seed is assigned by runner")
        else:
            if not isinstance(raw, dict) or set(raw) != set(SEARCH_SPACE):
                raise ValueError("genome must contain exactly the registered SEARCH_SPACE fields")
            if any(raw[key] not in choices for key, choices in SEARCH_SPACE.items()):
                raise ValueError("genome contains an unregistered search value")
        genome = CandidateGenome.from_dict({**raw, "seed": seed}).to_dict()
        if graph_mode and any(genome["architecture"]["input_dim"] != parent_map[p]["genome"]["architecture"]["input_dim"]
                              for p in parent_ids):
            raise ValueError("graph candidates cannot change the fixed data input contract")
        # All objectives remain enabled; the developmental component must model flow.
        if genome.get("flow") == "none":
            raise ValueError("this campaign requires an active mlp or fmt population flow")
        identity = architecture_id(genome)
        if identity in identities:
            raise ValueError("duplicate architecture (training seed does not count as a mutation)")
        identities.add(identity)
        changes = {p: genome_diff(parent_map[p]["genome"], genome) for p in parent_ids}
        for diff in changes.values():
            diff.pop("seed", None)
        accepted.append(dict(genome=genome, architecture_id=identity, parent_ids=parent_ids,
                             target_tasks=tasks, hypothesis=proposal["hypothesis"],
                             expected_tradeoff=proposal["expected_tradeoff"], changes=changes,
                             proposal_source="llm"))
        if "patch" in proposal:
            accepted[-1]["proposal_patch"] = proposal["patch"]
    return accepted


def has_module_change(proposal, parents):
    """Structural changes count; pure numerical/ID changes do not."""
    from .architecture import canonical_architecture, architecture_signature
    child = proposal["genome"]
    if child.get("architecture") is not None:
        current = canonical_architecture(child["architecture"])
        for parent in parents:
            if parent["architecture_id"] not in proposal["parent_ids"]:
                continue
            before = canonical_architecture(parent["genome"]["architecture"])
            if architecture_signature(current) != architecture_signature(before):
                return True
            if current["pairing"]["name"] != before["pairing"]["name"]:
                return True
            if any(current["heads"][k]["name"] != before["heads"][k]["name"] for k in current["heads"]):
                return True
        return False
    fields = {"encoder", "attention", "spatial", "residual", "pairing", "flow", "classification_head", "regression_head"}
    return any(fields & set(d) for d in proposal["changes"].values())

EVOLUTION_TASKS: tuple[str, ...] = ("annotation", "integration", "spatial_niche", "development")

# V1 metric alias retained for old public benchmark rows.
def joint_score(metrics: dict[str, float], tasks=None) -> float:
    values=[]
    for task,metric,direction in selected_objectives(tasks):
        key=f'{task}.{metric}'
        if key not in metrics and task=='annotation' and metric=='accuracy': key='annotation.f1_macro'
        value=float(metrics[key])
        if not math.isfinite(value): raise ValueError(f'nonfinite primary metric: {key}')
        if direction=='min':
            if value<0: raise ValueError(f'negative error: {key}')
            value=1.0/(1.0+value)
        elif metric=='ari':
            if not -1<=value<=1: raise ValueError('ARI outside [-1, 1]')
            value=(value+1.0)/2.0
        elif not 0<=value<=1: raise ValueError(f'metric outside [0, 1]: {key}')
        values.append(value)
    return statistics.mean(values)

def improvement_gate(candidate, reference, policy):
    reasons=[]; checks=[]; tasks=policy.get("tasks")
    def val(metrics, task, metric):
        return metrics.get(f"{task}.{metric}", metrics.get("annotation.f1_macro") if task=="annotation" and metric=="accuracy" else None)
    try:
        delta=joint_score(candidate,tasks)-joint_score(reference,tasks)
        if delta < float(policy.get("minimum_joint_gain",0.0)): reasons.append("joint_gain_below_threshold")
        for task,metric,direction in selected_objectives(tasks):
            key=f"{task}.{metric}"; observed=val(candidate,task,metric); base=val(reference,task,metric)
            if observed is None or base is None: raise KeyError(key)
            observed=float(observed); base=float(base)
            tolerance=float(policy.get("max_score_drop",0.0)) if direction=="max" else float(policy.get("max_error_relative_increase",0.0))*max(abs(base),1e-12)
            deterioration=base-observed if direction=="max" else observed-base; passed=deterioration<=tolerance+1e-12
            checks.append({"metric":key,"candidate":observed,"reference":base,"deterioration":deterioration,"allowance":tolerance,"passed":passed})
            if not passed: reasons.append(f"task_regression:{key}")
    except (KeyError,ValueError,TypeError) as exc:
        delta=None; reasons.append(f"invalid_metrics:{exc}")
    return {"passed":not reasons,"joint_gain":delta,"checks":checks,"reasons":reasons}
