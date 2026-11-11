"""Content addressed immutable bundles and a complete, untruncated research DAG."""

import json
import threading
from functools import wraps
from pathlib import Path

from .contracts import METRICS, digest, save, validate_bundle


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return call


class Store:
    def __init__(self, root):
        self.lock = threading.RLock()
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.candidates.mkdir(parents=True, exist_ok=True)

    @locked
    def stage(self, bundle, parents=None, origin=None):
        validate_bundle(bundle, parents)
        identifier = digest(bundle)
        p = self.candidates / identifier / "bundle.json"
        if p.exists() and json.loads(p.read_text()) != bundle:
            raise ValueError("immutable artifact collision")
        if not p.exists():
            save(p, bundle)
            save(p.with_name("origin.json"), origin or {})
        return identifier

    @locked
    def resolve(self, identifier):
        if (
            not isinstance(identifier, str)
            or not 12 <= len(identifier) <= 64
            or any(c not in "0123456789abcdef" for c in identifier)
        ):
            raise ValueError("invalid artifact reference")
        matches = [p.name for p in self.candidates.glob(identifier + "*") if (p / "bundle.json").is_file()]
        if len(matches) != 1:
            raise ValueError("unknown or ambiguous artifact reference")
        return matches[0]

    @locked
    def bundle(self, identifier):
        identifier = self.resolve(identifier)
        b = json.loads((self.candidates / identifier / "bundle.json").read_text())
        if digest(b) != identifier:
            raise ValueError("candidate artifact was mutated")
        return b

    @locked
    def put(self, identifier, kind, value):
        self.bundle(identifier)
        if kind not in ("validation", "result"):
            raise ValueError("invalid artifact kind")
        p = self.candidates / identifier / (kind + ".json")
        if p.exists() and json.loads(p.read_text()) != value:
            raise ValueError("candidate receipts are immutable")
        if not p.exists():
            save(p, value)

    @locked
    def node(self, identifier):
        b = self.bundle(identifier)
        folder = self.candidates / identifier
        origin = json.loads((folder / "origin.json").read_text())
        if (folder / "result.json").exists():
            return json.loads((folder / "result.json").read_text())
        status = "not_evaluated"
        if (folder / "validation.json").exists():
            status = json.loads((folder / "validation.json").read_text())["status"]
        return dict(candidate_id=identifier, parent_ids=b["parent_ids"], status=status, **origin)

    @locked
    def nodes(self):
        return [self.node(p.name) for p in sorted(self.candidates.iterdir()) if (p / "bundle.json").exists()]

    @locked
    def tree_summary(self):
        nodes = self.nodes()
        result, edges = [], []
        # Full 64-character IDs avoid ambiguity as the archive grows.
        for n in nodes:
            i = n["candidate_id"]
            b = self.bundle(i)
            parents = [p for p in b["parent_ids"] if p != "0" * 64]
            edges.extend([p, i] for p in parents)
            result.append(
                {
                    "id": i,
                    "parents": parents,
                    "status": n["status"],
                    "generation": n.get("generation"),
                    "metrics": n.get("metrics", {}),
                    "model": b["model"],
                    "train": b["train"],
                    "data": b["data"],
                    "components": b["components"],
                    "plugins": [p["kind"] + "/" + p["name"] for p in b["plugins"]],
                    "hypothesis": b["hypothesis"][:100],
                    "reason": n.get("reason"),
                }
            )
        return {"complete": True, "truncated": False, "metric_order": list(METRICS), "nodes": result, "edges": edges}

    @locked
    def read(self, identifier, kind, offset=0):
        identifier = self.resolve(identifier)
        self.bundle(identifier)
        if kind not in ("bundle", "validation", "result") or type(offset) is not int or offset < 0:
            raise ValueError("invalid bounded artifact read")
        s = (self.candidates / identifier / (kind + ".json")).read_text()
        return {
            "candidate_id": identifier,
            "text": s[offset : offset + 8000],
            "total_chars": len(s),
            "next_offset": offset + 8000 if offset + 8000 < len(s) else None,
        }
