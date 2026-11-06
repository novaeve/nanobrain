"""Complete paired seeds; relaxed archive and a separate conservative champion gate."""

import math
import statistics

from .contracts import METRICS, digest


def check_metrics(metrics):
    if set(metrics) != set(METRICS):
        raise ValueError("all six Cajal metric columns are required")
    for key, v in metrics.items():
        low = -1.0 if "Pearson" in key else 0.0
        if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= 1.0:
            raise ValueError("invalid Cajal metric")


def aggregate(results, seeds):
    if len(results) != len(seeds) or {r["seed"] for r in results} != set(seeds):
        raise ValueError("all three distinct paired seed results are required")
    for row in results:
        if row.get("status") != "complete" or set(row["metrics"]) != set(METRICS):
            raise ValueError("missing task or failed seed")
        check_metrics(row["metrics"])
    return {
        "metrics": {k: statistics.mean(r["metrics"][k] for r in results) for k in METRICS},
        "std": {k: statistics.stdev(r["metrics"][k] for r in results) for k in METRICS},
        "all_seeds_passed": True,
        "seeds": list(seeds),
        "seed_results": results,
    }


def margins(node, baselines):
    return [(node["metrics"][k] - baselines[k]) / max(abs(baselines[k]), 1e-8) for k in METRICS]


def quality(node, baselines):
    v = margins(node, baselines)
    return min(v), sum(v) / len(v)


def promote(candidate, parent, reference, baselines, targets, floor=0.005):
    tolerance = {k: max(floor, reference["std"][k], parent["std"][k], candidate["std"][k]) for k in METRICS}
    regressions = [k for k in METRICS if candidate["metrics"][k] < parent["metrics"][k] - tolerance[k]]
    # Always require improvement on the incumbent's weakest frontier dimension.
    weakest = METRICS[min(range(len(METRICS)), key=lambda i: margins(parent, baselines)[i])]
    target = weakest if weakest in targets else min(targets, key=lambda k: margins(parent, baselines)[METRICS.index(k)])
    gains = [
        candidate["metrics"][target] - parent["metrics"][target],
        candidate["metrics"][target] - reference["metrics"][target],
    ]
    passed = (
        not regressions
        and min(gains) > tolerance[target]
        and quality(candidate, baselines) > quality(parent, baselines)
    )
    return {"passed": passed, "target": target, "gains": gains, "tolerance": tolerance, "regressions": regressions}


def parents(nodes, store, baseline, limit=12):
    valid = [n for n in nodes if n.get("status") == "complete" and n.get("all_seeds_passed")]
    ordered = sorted(valid, key=lambda n: quality(n, baseline), reverse=True)
    pareto = [
        n
        for n in ordered
        if not any(
            all(o["metrics"][k] >= n["metrics"][k] for k in METRICS)
            and any(o["metrics"][k] > n["metrics"][k] for k in METRICS)
            for o in ordered
        )
    ]
    picked = ordered[: max(1, limit // 3)]
    signature = lambda n: digest({k: store.bundle(n["candidate_id"])[k] for k in ("model", "plugins", "files")})
    seen = {signature(n) for n in picked}
    for pool in (pareto, ordered[::-1], ordered):
        for n in pool:
            if len(picked) >= limit:
                break
            if n not in picked and (signature(n) not in seen or pool is ordered):
                picked.append(n)
                seen.add(signature(n))
    return [n["candidate_id"] for n in picked]
