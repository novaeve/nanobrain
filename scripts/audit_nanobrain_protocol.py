#!/usr/bin/env python3
"""Audit frozen compact data without reading test task labels or scoring test."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nanobrain.data import verify_subset_manifest
from nanobrain.component_catalog import component_catalog


def audit(data: Path, contract: Path):
    # Hashes cover bytes, including sealed labels, without loading their values.
    manifest = verify_subset_manifest(data, verify_hashes=True)
    receipt = json.loads((contract / "split_receipt.json").read_text())
    samples = pd.read_parquet(contract / "samples.parquet")
    assert receipt["split_id"] == manifest["split_id"]
    assert samples.groupby("sample_key")["split"].nunique().max() == 1
    assert samples.groupby("split_group")["split"].nunique().max() == 1
    details, identities = {}, {}
    for code, split in enumerate(("train", "validation", "test")):
        load = lambda key: np.load(data / split / f"{key}.npy", allow_pickle=False)
        rows, refs, codes = load("row_id"), load("source_row_ref"), load("source_split")
        assert len(rows) == (60000 if split == "train" else 20000)
        assert np.all(codes == code) and np.all(refs >= 0)
        assert len(np.unique(refs)) == len(refs)
        assert np.all(rows == code * 100_000_000 + refs)
        assert refs.max() < receipt["counts_by_split"][split]["cells"]
        identities[split] = rows
        source_counts = dict(zip(*[v.tolist() for v in np.unique(load("source_id"), return_counts=True)]))
        details[split] = {"rows": len(rows), "source_split_code": code,
            "source_count": len(source_counts), "source_rows": source_counts,
            "known_group_rows": int((load("split_group") >= 0).sum()),
            "known_section_rows": int((load("section") >= 0).sum()),
            "xy_rows": int(np.isfinite(load("coordinates")[:, :2]).all(1).sum()),
            "neighbor_edges": int(load("neighbor_mask").sum())}
    overlap = {f"{a}/{b}": int(len(np.intersect1d(identities[a], identities[b])))
               for a, b in itertools.combinations(details, 2)}
    assert not any(overlap.values())
    # Compact IDs are split-offset namespaces, not independent physical row IDs.
    # Therefore do not mistake namespace disjointness for a donor leakage audit.
    source_exists = {s: Path(p).is_file() for s, p in manifest["source_manifests"].items()}
    identity = None
    if manifest.get("identity_repair"):
        identity = json.loads((data / manifest["identity_repair"]["receipt"]).read_text())
        for split in details:
            if identity["splits"][split]["source_refs_matched"] != details[split]["rows"]:
                raise ValueError("incomplete source row identity audit")
            source_path = Path(manifest["source_manifests"][split])
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != identity["source_manifests"][split]:
                raise ValueError("source manifest changed after identity recovery")
    result = {
        "format": "nanobrain_protocol_audit_v1",
        "data_manifest_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
        "checks_passed": ["all compact artifact hashes", "6:2:2 counts", "source split codes",
                          "unique in-split row references and valid bounds", "train-only normalization declaration",
                          "original sample and group assignments each belong to one split"],
        "splits": details, "compact_row_id_overlap": overlap,
        "source_manifests_available": source_exists,
        "source_sample_identity_recheck": ("all 100k source references mapped to canonical samples within the original split" if identity
            else "not performed: original metadata recovery receipt absent"),
        "canonical_identity_overlap": identity["overlap"] if identity else None,
        "compact_donor_disjointness": ("canonical dataset-qualified known donors checked; missing donor values excluded" if identity
            else "not certified: compact donor/group vocabularies are incomplete"),
        "test_task_label_values_read": False, "test_metrics_computed": False,
        "niche_target": manifest["niche_policy"],
        "historical_test_already_published": True,
        "evolution_input": "train fits task weights; validation metrics guide LLM proposals and promotion; test only after selection freeze",
        "requirements_origin": {"ratio": "user's earlier global 6:2:2 requirement",
            "subset": "docs/nanobrain_prompt.md: proportional 100k sample inside existing full split",
            "fixed_validation": "nanoXXX blueprint: fixed held-out, multi-seed confirmation",
            "independent_training": "latest user clarification",
            "rounds_and_scalar_score": "explicit configurable engineering choices"},
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "assets/nanobrain_v3_100k_metadata_v2")
    parser.add_argument("--contract", type=Path, default=ROOT / "contracts/healthy_global_benchmark_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.contract)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "component_catalog.json").write_text(json.dumps(component_catalog(), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"splits": result["splits"], "checks_passed": result["checks_passed"],
                      "source_manifests_available": result["source_manifests_available"]}, ensure_ascii=False))
