"""Frozen validation contract and promotion gates for nanoBrain campaigns."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable


PRIMARY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("annotation", "accuracy", "max"),
    ("integration", "integration_composite", "max"),
    ("spatial_niche", "ari", "max"),
    ("spatial_completion", "heldout_rmse", "min"),
    ("development", "population_mse", "min"),
)


def candidate_method(candidate_id: str) -> str:
    return f"nanobrain_candidate-{candidate_id}"


def read_metric_table(path: str | Path) -> list[dict[str, str]]:
    table = Path(path)
    if table.is_dir():
        table = table / "benchmark_table.csv"
    if not table.is_file():
        raise ValueError(f"benchmark table does not exist: {table}")
    with table.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def overall_metrics(
    rows: Iterable[dict[str, str]], method: str
) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("method") != method or row.get("status") != "complete":
            continue
        if row.get("stratum_type") != "overall" or row.get("stratum_value") != "overall":
            continue
        key = (str(row.get("task", "")), str(row.get("metric", "")))
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if key in result and not math.isclose(result[key], value, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"conflicting duplicate metric for {method}: {key}")
        result[key] = value
    return result


def primary_metric_record(metrics: dict[tuple[str, str], float]) -> dict[str, float]:
    return {
        f"{task}.{metric}": float(metrics[(task, metric)])
        for task, metric, _ in PRIMARY_METRICS
        if (task, metric) in metrics
    }


def metric_objectives(metrics: dict[tuple[str, str], float]) -> tuple[float, ...]:
    missing = [
        f"{task}.{metric}"
        for task, metric, _ in PRIMARY_METRICS
        if (task, metric) not in metrics
    ]
    if missing:
        raise ValueError(f"missing primary validation metrics: {', '.join(missing)}")
    return tuple(
        float(metrics[(task, metric)]) if direction == "max" else -float(metrics[(task, metric)])
        for task, metric, direction in PRIMARY_METRICS
    )


def audit_training_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if receipt.get("status") != "complete":
        reasons.append("training_status_incomplete")
    if receipt.get("test_opened") is not False:
        reasons.append("test_split_not_sealed")
    if receipt.get("fit_split") != "train" or receipt.get("evaluation_split") != "validation":
        reasons.append("split_contract_mismatch")
    if (receipt.get("training_task", "joint") in {"joint", "development"}
            and receipt.get("development_contract") != "unpaired_cross_sectional_population_flow_matching"):
        reasons.append("development_contract_mismatch")
    if receipt.get("optimizer_changed_parameter") is not True:
        reasons.append("optimizer_update_unverified")
    if int(receipt.get("gradient_finite_parameters", 0)) < 1:
        reasons.append("finite_gradients_unverified")
    if int(receipt.get("gradient_nonzero_parameters", 0)) < 1:
        reasons.append("nonzero_gradients_unverified")
    stability = receipt.get("stability")
    if not isinstance(stability, dict) or stability.get("passed") is not True:
        reasons.append("training_stability_failed")
    try:
        quality = float(receipt.get("quality", math.nan))
    except (TypeError, ValueError):
        quality = math.nan
    if not math.isfinite(quality):
        reasons.append("validation_quality_nonfinite")
    checkpoint_value = receipt.get("best_checkpoint") or receipt.get("checkpoint")
    checkpoint = Path(str(checkpoint_value)) if checkpoint_value else None
    if checkpoint is None or not checkpoint.is_file():
        reasons.append("checkpoint_missing")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "checkpoint": str(checkpoint.resolve()) if checkpoint is not None and checkpoint.is_file() else None,
    }


def strongest_baselines(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = list(rows)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for task, metric, direction in PRIMARY_METRICS:
        candidates: list[tuple[float, str]] = []
        for row in rows:
            method = str(row.get("method", ""))
            if method.startswith("nanobrain_candidate-"):
                continue
            if row.get("task") != task or row.get("metric") != metric:
                continue
            if row.get("status") != "complete":
                continue
            if row.get("stratum_type") != "overall" or row.get("stratum_value") != "overall":
                continue
            try:
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                candidates.append((value, method))
        if not candidates:
            continue
        value, method = (max(candidates) if direction == "max" else min(candidates))
        result[(task, metric)] = {
            "method": method,
            "value": float(value),
            "direction": direction,
        }
    return result


def baseline_regression_gate(
    metrics: dict[tuple[str, str], float],
    baselines: dict[tuple[str, str], dict[str, Any]],
    *,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    tolerance = max(0.0, float(relative_tolerance))
    for task, metric, direction in PRIMARY_METRICS:
        key = (task, metric)
        baseline = baselines.get(key)
        candidate = metrics.get(key)
        if baseline is None:
            reasons.append(f"baseline_missing:{task}.{metric}")
            continue
        if candidate is None or not math.isfinite(float(candidate)):
            reasons.append(f"candidate_metric_missing:{task}.{metric}")
            continue
        reference = float(baseline["value"])
        allowance = tolerance * max(abs(reference), 1.0e-12)
        passed = (
            float(candidate) + allowance >= reference
            if direction == "max"
            else float(candidate) - allowance <= reference
        )
        checks.append(
            {
                "task": task,
                "metric": metric,
                "direction": direction,
                "candidate": float(candidate),
                "baseline_method": baseline["method"],
                "baseline": reference,
                "passed": bool(passed),
            }
        )
        if not passed:
            reasons.append(f"baseline_regression:{task}.{metric}")
    return {"passed": not reasons, "reasons": reasons, "checks": checks}


def genome_diff(parent: dict[str, Any] | None, child: dict[str, Any]) -> dict[str, Any]:
    if parent is None:
        return {key: {"from": None, "to": value} for key, value in sorted(child.items())}
    return {
        key: {"from": parent.get(key), "to": child.get(key)}
        for key in sorted(set(parent) | set(child))
        if parent.get(key) != child.get(key)
    }


__all__ = [
    "PRIMARY_METRICS",
    "audit_training_receipt",
    "baseline_regression_gate",
    "candidate_method",
    "genome_diff",
    "metric_objectives",
    "overall_metrics",
    "primary_metric_record",
    "read_metric_table",
    "strongest_baselines",
]

# Compatibility ledger helpers retained for the public baseline command.
def append_generation_baseline(path, record):
    import json
    from pathlib import Path
    target = Path(path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("format") != "nanobrain_generation_ledger_v1":
            raise ValueError(f"unsupported generation ledger: {target}")
        generations = list(payload.get("generations", []))
    else:
        payload = {"format": "nanobrain_generation_ledger_v1", "generations": []}
        generations = []
    generation = int(record["generation"])
    generations = [x for x in generations if int(x.get("generation", -1)) != generation]
    generations.append(dict(record)); generations.sort(key=lambda x: int(x["generation"]))
    payload["generations"] = generations
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

def load_generation_baseline(path, generation):
    import json
    from pathlib import Path
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "nanobrain_generation_ledger_v1":
        raise ValueError("unsupported generation ledger format")
    for record in payload.get("generations", []):
        if int(record.get("generation", -1)) == int(generation): return record
    raise KeyError(f"generation baseline not found: {generation}")

# Backward-compatible public baseline API; Graph V2 continues to use the frozen primary metrics above.
def strongest_baselines(rows):
    rows = list(rows)
    specs = [(task, metric, direction) for task, metric, direction in PRIMARY_METRICS]
    if any(str(r.get("task")) == "annotation" and str(r.get("metric")) == "f1_macro" for r in rows):
        specs.append(("annotation", "f1_macro", "max"))
    result = {}
    for task, metric, direction in specs:
        vals = []
        for row in rows:
            method = str(row.get("method", ""))
            if method.startswith("nanobrain_candidate-") or row.get("task") != task or row.get("metric") != metric:
                continue
            if row.get("status") not in {None, "", "complete"}: continue
            if row.get("stratum_type", "overall") != "overall" or row.get("stratum_value", "overall") != "overall": continue
            try: value = float(row["value"])
            except (KeyError, TypeError, ValueError): continue
            if math.isfinite(value): vals.append((value, method))
        if vals:
            value, method = (max(vals) if direction == "max" else min(vals))
            record = {"method": method, "value": float(value), "direction": direction}
            result[(task, metric)] = record
            result[f"{task}.{metric}"] = record
    return result

def baseline_regression_gate(candidate_metrics, baseline_metrics, *, relative_tolerance=0.0):
    checks=[]; reasons=[]; tolerance=max(0.0,float(relative_tolerance))
    def get(mapping, task, metric):
        return mapping.get((task,metric), mapping.get(f"{task}.{metric}"))
    for task, metric, direction in PRIMARY_METRICS:
        cand = candidate_metrics.get((task,metric), candidate_metrics.get(f"{task}.{metric}"))
        base = get(baseline_metrics, task, metric)
        # V1 used annotation.f1_macro; retain that input contract without changing V2 accuracy.
        if (cand is None or base is None) and task == "annotation" and metric == "accuracy":
            cand = candidate_metrics.get("annotation.f1_macro", candidate_metrics.get(("annotation","f1_macro")))
            base = get(baseline_metrics, "annotation", "f1_macro")
            metric = "f1_macro"
        key=f"{task}.{metric}"
        if base is None: reasons.append(f"baseline_missing:{key}"); continue
        if cand is None:
            reasons.append(f"candidate_metric_missing:{key}"); continue
        try: cand=float(cand); ref=float(base["value"])
        except (TypeError, ValueError, KeyError): reasons.append(f"candidate_metric_missing:{key}"); continue
        allowance=tolerance*max(abs(ref),1e-12)
        passed=(cand+allowance>=ref) if direction=="max" else (cand-allowance<=ref)
        checks.append({"metric":key,"direction":direction,"candidate":cand,"baseline":ref,"baseline_method":base.get("method"),"allowance":allowance,"passed":bool(passed)})
        if not passed: reasons.append(f"baseline_regression:{key}")
    return {"passed": not reasons, "reasons": reasons, "checks": checks}

def generation_baseline_record(rows, *, generation, evaluation_split="validation", split_id="BRAINWM-HUMAN-GLOBAL-6-2-2-V3"):
    rows=list(rows); best=strongest_baselines(rows)
    baseline_metrics={k:v for k,v in best.items() if isinstance(k,str)}
    methods=sorted({str(r.get("method")) for r in rows if r.get("method")})
    candidate_methods=sorted({str(r.get("method")) for r in rows if str(r.get("method","")).startswith("nanobrain_candidate-")})
    return {"format":"nanobrain_generation_baseline_v1","generation":int(generation),"split_id":split_id,"fit_split":"train","evaluation_split":str(evaluation_split),"test_used_for_selection":str(evaluation_split)=="test","status":"complete" if best else "incomplete","baseline_metrics":baseline_metrics,"methods":methods,"candidate_methods":candidate_methods}

# Compatibility surface: primary Graph V2 baselines remain tuple-keyed; V1 annotation F1 stays string-keyed.
def strongest_baselines(rows):
    rows=list(rows); specs=list(PRIMARY_METRICS)
    if any(str(r.get("task"))=="annotation" and str(r.get("metric"))=="f1_macro" for r in rows): specs.append(("annotation","f1_macro","max"))
    result={}
    for task,metric,direction in specs:
        vals=[]
        for row in rows:
            method=str(row.get("method",""))
            if method.startswith("nanobrain_candidate-") or row.get("task")!=task or row.get("metric")!=metric: continue
            if row.get("status") not in {None,"","complete"} or row.get("stratum_type","overall")!="overall" or row.get("stratum_value","overall")!="overall": continue
            try: value=float(row["value"])
            except (KeyError,TypeError,ValueError): continue
            if math.isfinite(value): vals.append((value,method))
        if vals:
            value,method=(max(vals) if direction=="max" else min(vals))
            rec={"method":method,"value":float(value),"direction":direction}
            result[(task,metric)]=rec
            if metric=="f1_macro": result[f"{task}.{metric}"]=rec
    return result

def baseline_regression_gate(candidate_metrics, baseline_metrics, *, relative_tolerance=0.0):
    checks=[]; reasons=[]; tolerance=max(0.0,float(relative_tolerance))
    for task,metric,direction in PRIMARY_METRICS:
        key=(task,metric); cand=candidate_metrics.get(key); base=baseline_metrics.get(key)
        if cand is None and task=="annotation" and metric=="accuracy":
            cand=candidate_metrics.get("annotation.f1_macro"); base=baseline_metrics.get("annotation.f1_macro"); metric="f1_macro"; key=(task,metric)
        label=f"{task}.{metric}"
        if base is None: reasons.append(f"baseline_missing:{label}"); continue
        if cand is None:
            cand=candidate_metrics.get(label)
        if cand is None: reasons.append(f"candidate_metric_missing:{label}"); continue
        try: cand=float(cand); ref=float(base["value"])
        except (TypeError,ValueError,KeyError): reasons.append(f"candidate_metric_missing:{label}"); continue
        allowance=tolerance*max(abs(ref),1e-12); passed=(cand+allowance>=ref) if direction=="max" else (cand-allowance<=ref)
        checks.append({"metric":label,"direction":direction,"candidate":cand,"baseline":ref,"baseline_method":base.get("method"),"allowance":allowance,"passed":bool(passed)})
        if not passed: reasons.append(f"baseline_regression:{label}")
    return {"passed":not reasons,"reasons":reasons,"checks":checks}

def generation_baseline_record(rows, *, generation, evaluation_split="validation", split_id="BRAINWM-HUMAN-GLOBAL-6-2-2-V3"):
    rows=list(rows); best=strongest_baselines(rows)
    bm={f"{k[0]}.{k[1]}":v for k,v in best.items() if isinstance(k,tuple)}
    methods=sorted({str(r.get("method")) for r in rows if r.get("method")})
    candidates=sorted({str(r.get("method")) for r in rows if str(r.get("method","")).startswith("nanobrain_candidate-")})
    return {"format":"nanobrain_generation_baseline_v1","generation":int(generation),"split_id":split_id,"fit_split":"train","evaluation_split":str(evaluation_split),"test_used_for_selection":str(evaluation_split)=="test","status":"complete" if bm else "incomplete","baseline_metrics":bm,"methods":methods,"candidate_methods":candidates}

# Optional spatial completion is not part of the four-task core gate unless supplied.
def baseline_regression_gate(candidate_metrics, baseline_metrics, *, relative_tolerance=0.0):
    checks=[]; reasons=[]; tolerance=max(0.0,float(relative_tolerance))
    for task,metric,direction in PRIMARY_METRICS:
        key=(task,metric); cand=candidate_metrics.get(key); base=baseline_metrics.get(key)
        if task=="spatial_completion" and cand is None and base is None: continue
        if cand is None and task=="annotation" and metric=="accuracy":
            cand=candidate_metrics.get("annotation.f1_macro"); base=baseline_metrics.get("annotation.f1_macro"); metric="f1_macro"; key=(task,metric)
        label=f"{task}.{metric}"
        if base is None: reasons.append(f"baseline_missing:{label}"); continue
        if cand is None: cand=candidate_metrics.get(label)
        if cand is None: reasons.append(f"candidate_metric_missing:{label}"); continue
        try: cand=float(cand); ref=float(base["value"])
        except (TypeError,ValueError,KeyError): reasons.append(f"candidate_metric_missing:{label}"); continue
        allowance=tolerance*max(abs(ref),1e-12); passed=(cand+allowance>=ref) if direction=="max" else (cand-allowance<=ref)
        checks.append({"metric":label,"direction":direction,"candidate":cand,"baseline":ref,"baseline_method":base.get("method"),"allowance":allowance,"passed":bool(passed)})
        if not passed: reasons.append(f"baseline_regression:{label}")
    return {"passed":not reasons,"reasons":reasons,"checks":checks}
