"""GPU busy time and idle gaps from a Chrome trace written by scripts/bench/profile_step.py.

    python scripts/bench/trace_gaps.py runs/e2e/compiled/trace.json

Merges GPU kernel intervals into a busy timeline (overlapping kernels counted once),
then lists the largest idle gaps with the CPU-side operations that were running
when each gap started (syncs and launches show up here).
"""

import argparse
import bisect
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    with open(args.trace) as fh:
        events = json.load(fh)["traceEvents"]
    kernels = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    )
    cpu_ops = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in events
        if e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver") and e.get("dur", 0) > 0
    )
    if not kernels:
        raise SystemExit("no GPU kernel events in trace")

    merged = []
    for start, end, _ in kernels:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    span = merged[-1][1] - merged[0][0]
    busy = sum(e - s for s, e in merged)
    gaps = sorted(((merged[i + 1][0] - merged[i][1], merged[i][1]) for i in range(len(merged) - 1)), reverse=True)
    print(
        f"window {span / 1e3:.1f} ms, GPU busy {busy / 1e3:.1f} ms ({busy / span * 100:.1f}%), "
        f"idle {(span - busy) / 1e3:.1f} ms in {len(merged) - 1} gaps"
    )
    small = sum(g for g, _ in gaps if g < 50)
    print(
        f"idle in gaps < 50 us (launch overhead): {small / 1e3:.1f} ms; in larger gaps: {(span - busy - small) / 1e3:.1f} ms"
    )

    starts = [c[0] for c in cpu_ops]
    print(f"\nlargest {args.top} gaps and the CPU ops running at their start:")
    blame = Counter()
    for gap, at in gaps[: args.top]:
        i = bisect.bisect_right(starts, at)
        running = [name for s, e, name in cpu_ops[max(0, i - 200) : i] if s <= at <= e]
        label = " > ".join(running[-3:]) if running else "(no CPU op recorded)"
        print(f"  {gap / 1e3:8.2f} ms  {label[:110]}")
    for gap, at in gaps:
        if gap < 50:
            break
        i = bisect.bisect_right(starts, at)
        running = [name for s, e, name in cpu_ops[max(0, i - 200) : i] if s <= at <= e]
        blame[running[-1] if running else "(none)"] += gap
    print("\nidle time in gaps >= 50 us by innermost CPU op:")
    for name, total in blame.most_common(12):
        print(f"  {total / 1e3:8.2f} ms  {name[:100]}")


if __name__ == "__main__":
    main()
