"""Freeze a campaign: lock what it is judged on before evolution starts.

Before a campaign starts its definition is open: users write the data, tasks, metrics, budget and search
space. Freezing records the SHA-256 of every file the campaign depends on and of each dataset manifest
in campaign.lock.json, which is committed next to the definition. Every start and resume verifies the
lock; any difference makes it a different campaign, which needs its own id.

The lock covers the campaign's identity (definition, agent brief, reference recipe, seed model, the task
and judge files the definition declares, and the data). Infrastructure code is pinned per run by the
run contract instead, so it can improve between campaigns without changing what a campaign means.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from . import campaign
from .contracts import ROOT, digest, save, sha

LOCK = "campaign.lock.json"


def lock_path(definition):
    return Path(definition["directory"]) / LOCK


def _key(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def locked_files(definition):
    folder = Path(definition["directory"])
    spec = definition["freeze"]
    files = {
        folder / "campaign.toml",
        folder / definition["campaign"]["agent_brief"],
        ROOT / definition["campaign"]["reference_recipe"],
        *(ROOT / p for p in spec["files"]),
    }
    seed = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z", "--", spec["seed_model"]], text=True)
    seed_files = [ROOT / p for p in seed.split("\0") if p]
    if not seed_files:
        raise ValueError(f"seed model {spec['seed_model']!r} has no tracked files")
    return sorted(files | set(seed_files), key=_key)


def snapshot(definition, cfg):
    files = {}
    for path in locked_files(definition):
        if not path.is_file():
            raise FileNotFoundError(f"locked file is missing: {_key(path)}")
        files[_key(path)] = sha(path)
    data = {
        name: {"path": str(Path(root).resolve()), "manifest_sha256": sha(Path(root) / "manifest.json")}
        for name, root in sorted(cfg["data"].items())
    }
    return {"campaign_id": definition["campaign"]["id"], "files": files, "data": data}


def freeze(cfg, definition=None):
    """Write the lock once. Re-freezing with different contents is refused: define a new campaign."""
    definition = definition or campaign.load()
    path, current = lock_path(definition), snapshot(definition, cfg)
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: existing[k] for k in current} != current:
            raise ValueError(
                f"campaign {current['campaign_id']} is already frozen with different contents; "
                "give the changed definition a new campaign id instead of re-freezing"
            )
        return existing
    lock = {"version": 1, **current, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds")}
    save(path, lock)
    return lock


def verify(cfg, definition=None):
    """Return the lock digest, or refuse if the campaign is unfrozen or anything it locks has changed."""
    definition = definition or campaign.load()
    path = lock_path(definition)
    if not path.is_file():
        raise ValueError(
            f"campaign {definition['campaign']['id']} is not frozen; run `nanobrain --freeze` and commit {LOCK}"
        )
    lock, current = json.loads(path.read_text()), snapshot(definition, cfg)
    if lock["campaign_id"] != current["campaign_id"]:
        raise ValueError("the lock belongs to a different campaign id")
    changed = [
        k for k in sorted(set(lock["files"]) | set(current["files"])) if lock["files"].get(k) != current["files"].get(k)
    ]
    changed += [
        f"data:{k}"
        for k in sorted(set(lock["data"]) | set(current["data"]))
        if lock["data"].get(k) != current["data"].get(k)
    ]
    if changed:
        raise ValueError("frozen campaign changed: " + ", ".join(changed[:20]))
    return digest({k: lock[k] for k in current})
