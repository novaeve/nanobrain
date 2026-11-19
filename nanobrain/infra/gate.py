"""Single-seed exploratory scores, archive, and a separate champion gate."""

import math
import statistics

from .contracts import METRIC_RANGES, METRIC_SIGN, METRIC_TOLERANCE, METRICS, digest


def check_metrics(metrics):
    if set(metrics) != set(METRICS):
        raise ValueError(f"all {len(METRICS)} metric columns of the campaign are required")
    for key, v in metrics.items():
        low, high = METRIC_RANGES[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"invalid value for metric {key}")


def aggregate(results, seeds):
    if len(results) != len(seeds) or {r["seed"] for r in results} != set(seeds):
        raise ValueError("every configured seed must have exactly one complete result")
    for row in results:
        if row.get("status") != "complete" or set(row["metrics"]) != set(METRICS):
            raise ValueError("missing task or failed seed")
        check_metrics(row["metrics"])
    return {
        "metrics": {k: statistics.mean(r["metrics"][k] for r in results) for k in METRICS},
        "std": {k: statistics.stdev(r["metrics"][k] for r in results) if len(results) > 1 else None for k in METRICS},
        "seed_count": len(results),
        "uncertainty_estimated": len(results) > 1,
        "all_seeds_passed": True,
        "seeds": list(seeds),
        "seed_results": results,
    }


def better(a, b, k):
    """How much a beats b on metric k, in its unit: positive is better whatever the metric's direction."""
    return METRIC_SIGN[k] * (a - b)


def margins(node, baselines):
    return [better(node["metrics"][k], baselines[k], k) / max(abs(baselines[k]), 1e-8) for k in METRICS]


def quality(node, baselines):
    v = margins(node, baselines)
    return min(v), sum(v) / len(v)


def promote(candidate, parent, reference, baselines, targets):
    tolerance = dict(METRIC_TOLERANCE)
    regressions = [k for k in METRICS if better(candidate["metrics"][k], parent["metrics"][k], k) < -tolerance[k]]
    # Always require improvement on the incumbent's weakest frontier dimension.
    weakest = METRICS[min(range(len(METRICS)), key=lambda i: margins(parent, baselines)[i])]
    target = weakest if weakest in targets else min(targets, key=lambda k: margins(parent, baselines)[METRICS.index(k)])
    gains = [
        better(candidate["metrics"][target], parent["metrics"][target], target),
        better(candidate["metrics"][target], reference["metrics"][target], target),
    ]
    passed = (
        not regressions
        and min(gains) > tolerance[target] + 1e-12
        and quality(candidate, baselines) > quality(parent, baselines)
    )
    return {
        "passed": passed,
        "target": target,
        "gains": gains,
        "tolerance": tolerance,
        "regressions": regressions,
        "evidence": "single_seed_exploratory",
        "statistical_significance_claimed": False,
    }


def parents(nodes, store, baseline, limit=12):
    valid = [n for n in nodes if n.get("status") == "complete" and n.get("all_seeds_passed")]
    ordered = sorted(valid, key=lambda n: quality(n, baseline), reverse=True)
    pareto = [
        n
        for n in ordered
        if not any(
            all(better(o["metrics"][k], n["metrics"][k], k) >= 0 for k in METRICS)
            and any(better(o["metrics"][k], n["metrics"][k], k) > 0 for k in METRICS)
            for o in ordered
        )
    ]
    picked = ordered[: max(1, limit // 3)]
    signature = lambda n: digest(
        {k: store.bundle(n["candidate_id"])[k] for k in ("model", "components", "plugins", "files")}
    )
    seen = {signature(n) for n in picked}
    for pool in (pareto, ordered[::-1], ordered):
        for n in pool:
            if len(picked) >= limit:
                break
            if n not in picked and (signature(n) not in seen or pool is ordered):
                picked.append(n)
                seen.add(signature(n))
    return [n["candidate_id"] for n in picked]
