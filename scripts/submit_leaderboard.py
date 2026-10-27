#!/usr/bin/env python3
"""Submit a nanoBrain receipt to the H20 leaderboard service."""
from __future__ import print_function
import argparse
import json
import urllib.request

def flatten(receipt):
    if isinstance(receipt.get("results"), list):
        return receipt["results"]
    if isinstance(receipt.get("leaderboard_rows"), list):
        return receipt["leaderboard_rows"]
    metrics = receipt.get("metrics", {})
    rows = []
    for key, value in metrics.items():
        if "." not in key:
            continue
        task, metric = key.split(".", 1)
        rows.append({"task": task, "metric": metric, "value": value})
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt")
    parser.add_argument("--endpoint", default="http://127.0.0.1:4173/api/submit")
    args = parser.parse_args()
    with open(args.receipt, "r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    rows = flatten(receipt)
    if not rows:
        raise SystemExit("receipt has no results or metrics")
    common = {k: receipt[k] for k in ("run_id", "model_name", "parameter_count", "dataset", "split_id", "commit_sha", "hardware", "submitted_at") if k in receipt}
    payload = {"submitted_at": receipt.get("submitted_at"), "results": [dict(common, **row) for row in rows]}
    raw = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(args.endpoint, data=raw, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        print(response.read().decode("utf-8"))

if __name__ == "__main__":
    main()
