#!/usr/bin/env python3
"""Static nanoBrain site plus a small receipt submission endpoint."""
from __future__ import print_function
import json
import os
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "leaderboard.json")
SUBMISSIONS = os.path.join(ROOT, "data", "submissions.jsonl")
LOCK = threading.Lock()
PRIMARY = {"annotation": "f1_macro", "integration": "integration_composite", "development": "gene_spearman", "spatial_niche": "ari"}

def now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def load_snapshot():
    try:
        with open(DATA, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (IOError, ValueError):
        return {"generated_at": now(), "source": "live submissions", "rows": []}

def write_snapshot(payload):
    temp = DATA + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temp, DATA)

def normalize(item, submitted_at):
    task = item.get("task")
    metric = item.get("metric")
    value = item.get("value", item.get("mean"))
    if task not in PRIMARY or not metric or value is None:
        raise ValueError("each result needs task, metric and value")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("metric value must be finite")
    row = dict(item)
    row["mean"] = value
    row.pop("value", None)
    row["task"] = task
    row["metric"] = metric
    row["status"] = "verified"
    row["source_group"] = "live_submission"
    row["submitted_at"] = item.get("submitted_at", submitted_at)
    row["parameter_count"] = item.get("parameter_count")
    row["run_id"] = item.get("run_id", "live-" + row["submitted_at"].replace(":", "").replace("-", ""))
    row.setdefault("direction", "min" if metric == "population_mse" else "max")
    row.setdefault("std", None)
    row.setdefault("n_runs", 1)
    row.setdefault("eval_rows", None)
    row.setdefault("display_name", item.get("model_name", item.get("method", "Submitted model")))
    row.setdefault("method", item.get("model_name", "submitted_model"))
    row.setdefault("method_family", "submitted")
    row.setdefault("training_regime", "receipt")
    return row

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[site] " + fmt % args)
    def do_POST(self):
        if self.path != "/api/submit":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            source_rows = body.get("results", body.get("rows", [body]))
            if not isinstance(source_rows, list):
                source_rows = [source_rows]
            submitted_at = body.get("submitted_at", now())
            rows = [normalize(item, submitted_at) for item in source_rows]
            with LOCK:
                snapshot = load_snapshot()
                existing = snapshot.get("rows", [])
                keys = {(r.get("run_id"), r.get("task"), r.get("metric")) for r in existing}
                accepted = [r for r in rows if (r.get("run_id"), r.get("task"), r.get("metric")) not in keys]
                existing.extend(accepted)
                snapshot = {"generated_at": now(), "source": "repository snapshot + live submissions", "rows": existing}
                write_snapshot(snapshot)
                with open(SUBMISSIONS, "a", encoding="utf-8") as handle:
                    for row in accepted:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            payload = {"accepted": len(accepted), "duplicates": len(rows) - len(accepted), "generated_at": snapshot["generated_at"]}
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (ValueError, TypeError, IOError, json.JSONDecodeError) as exc:
            raw = json.dumps({"accepted": 0, "error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("0.0.0.0", 4173), Handler)
    print("nanoBrain site listening on :4173")
    server.serve_forever()
