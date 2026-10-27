#!/usr/bin/env python3
"""Hash the complete frozen nanoBrain protocol before opening the test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "nanobrain_final_100k.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "nanobrain" / "final-100k-20260916" / "protocol_freeze.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("test_policy", {}).get("state") != "sealed":
        raise ValueError("protocol config must attest a sealed test split")

    paths = [config_path, ROOT / config["data"]["path"] / "manifest.json"]
    paths.extend(ROOT / value for value in config["nanobrain"]["frozen_checkpoints"])
    for method in config["development_baselines"].values():
        paths.append(ROOT / method["checkpoint"])
        if "preprocessing" in method:
            paths.append(ROOT / method["preprocessing"])
    source_files = sorted((ROOT / "src" / "nanobrain").glob("*.py"))
    source_files.extend(
        ROOT / "scripts" / name
        for name in (
            "run_nanobrain_baselines.py",
            "run_nanobrain_external_baselines.py",
            "aggregate_nanobrain_replicates.py",
        )
    )
    paths.extend(source_files)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze missing artifacts: {missing}")
    artifacts = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    payload = {
        "format": "nanobrain_protocol_freeze_receipt_v1",
        "protocol_id": config["protocol_id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "config": str(config_path.relative_to(ROOT)),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "test_opened": False,
        "post_freeze_policy": "no model, data, task, metric, or hyperparameter changes",
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), **payload}, indent=2))


if __name__ == "__main__":
    main()
