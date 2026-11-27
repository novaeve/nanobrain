"""Reuse the project's existing 10k Cajal reference without refitting it."""

import json
from pathlib import Path

from .contracts import DEFINITION, ROOT, canonical_document, digest, reference_bundle, resolved_document, sha
from .gate import check_metrics


def snapshot_path():
    return ROOT / DEFINITION["campaign"].get("reference_snapshot", "reference/existing_cajal_v1_10k.json")


def niche_reference():
    path = DEFINITION["campaign"].get("niche_reference")
    return json.loads((ROOT / path).read_text()) if path else None


def load(cfg, seed=0):
    snapshot = json.loads(snapshot_path().read_text())
    if (snapshot["seed"], snapshot["step"], snapshot["schedule_steps"]) != (
        seed,
        cfg["stop_at"],
        cfg["schedule_steps"],
    ):
        raise ValueError("existing Cajal reference must match the fixed seed and comparison budget")
    if snapshot["resolved_config_sha256"] != digest(
        canonical_document(resolved_document(reference_bundle(), cfg, seed, include_optional_tasks=False))
    ):
        raise ValueError("existing Cajal reference no longer matches the initial recipe")
    for name, expected in snapshot["data_fingerprints"].items():
        if (
            cfg["data"][name] != expected["path"]
            or sha(Path(expected["path"]) / "manifest.json") != expected["manifest_sha256"]
        ):
            raise ValueError("existing Cajal reference used different training or evaluation data")
    for path, expected in snapshot["artifact_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("existing Cajal reference artifact changed; do not silently rerun it")
    prior = niche_reference()
    if prior is not None:
        for name, expected in prior["data_manifest_sha256"].items():
            if sha(Path(cfg["data"][name]) / "manifest.json") != expected:
                raise ValueError("Niche prior used different training or evaluation data")
        snapshot["metrics"] = {**snapshot["metrics"], **prior["metrics"]}
        path = DEFINITION["campaign"]["niche_reference"]
        snapshot["artifact_hashes"][path] = sha(ROOT / path)
        snapshot["niche_metric_source"] = prior["method"]
    check_metrics(snapshot["metrics"])
    return snapshot


def seed_result(cfg, identifier, seed):
    snapshot = load(cfg, seed)
    result = {
        "status": "complete",
        "seed": seed,
        "candidate_id": identifier,
        "metrics": snapshot["metrics"],
        "artifact_hashes": snapshot["artifact_hashes"],
        "joint_model_count": 1,
        "checkpoint_sha256": snapshot["artifact_hashes"][snapshot["checkpoint"]],
        "training_parameters": snapshot["original_training_config"],
        "source": "existing_Cajal_project_result",
        "method": snapshot["method"],
        "training_executed_by_campaign": False,
        "evaluation_executed_by_campaign": False,
    }
    if "niche_metric_source" in snapshot:
        result["niche_metric_source"] = snapshot["niche_metric_source"]
    return result
