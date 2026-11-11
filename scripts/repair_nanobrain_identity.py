#!/usr/bin/env python3
"""Recover identity/geometry sidecars for the SAME compact rows and split.

The old dataset is immutable. Large unchanged files are hardlinked; replacements
use atomic rename, so no inode shared with the old dataset is ever overwritten.
Only NumPy metadata and frozen sample tables are read, never H5AD or test labels.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from nanobrain.paths import locate_v3_shards
from scripts.materialize_nanobrain_subset import _discover, _gather, _neighbors
from scripts.materialize_nanobrain_development_pairs import _atomic_npy, _atomic_json, _sha256
from scripts.build_v3_development_pairs import stable_id


def identifier(value):
    if value is None or pd.isna(value) or str(value).strip().lower() in {"", "unknown", "nan", "none", "<na>"}:
        return None
    return str(value).strip()


def repair(old: Path, output: Path, contract: Path, geometry: Path):
    if output.exists():
        raise FileExistsError(output)
    source = locate_v3_shards().root
    manifest = json.loads((old / "manifest.json").read_text())
    samples = pd.read_parquet(contract / "samples.parquet")
    samples["sample_hash"] = samples.sample_key.map(stable_id)
    if samples.sample_hash.duplicated().any():
        raise ValueError("sample hash collision")
    sample_lookup = samples.set_index("sample_hash", drop=False)
    proof = pd.read_parquet(geometry)
    verified = set(proof.loc[proof.niche_geometry_ready & proof.independent_section_key_verified, "sample_key"])
    shutil.copytree(old, output, copy_function=os.link)
    # Provenance is locally archived; no dependency on mutable upstream metadata.
    shutil.copy2(geometry, output / "verified_spatial_units.parquet")
    report = {"parent_manifest_sha256": _sha256(old / "manifest.json"), "source_root": str(source),
              "geometry_proof_sha256": _sha256(geometry), "same_rows_and_splits": True,
              "test_label_values_read": False, "splits": {}, "source_manifests": {}}
    identities = {}
    for split in ("train", "validation", "test"):
        target = output / split
        refs = np.load(old / split / "source_row_ref.npy", allow_pickle=False)
        shards = _discover(source, split)
        canonical_source = _gather(shards, refs, "row_source_id", np.int32)
        if not np.array_equal(canonical_source, np.load(old / split / "source_id.npy")):
            raise ValueError(f"source row correspondence mismatch: {split}")
        sample_ids = _gather(shards, refs, "evaluation_sample", np.int64)
        if not np.isin(sample_ids, sample_lookup.index).all():
            raise ValueError(f"unmatched sample IDs: {split}")
        rows = sample_lookup.loc[sample_ids].reset_index(drop=True)
        if not rows.split.eq(split).all():
            raise ValueError(f"compact row crosses frozen sample split: {split}")
        if not np.array_equal(rows.source_id.to_numpy(), canonical_source):
            raise ValueError(f"sample/source mismatch: {split}")
        donors = [f"{r.dataset}::{identifier(r.donor_id)}" if identifier(r.donor_id) else None
                  for r in rows.itertuples(index=False)]
        # These are input identity IDs, not labels or fitted vocabularies.
        groups = rows.split_group.map(stable_id).to_numpy(np.int64)
        donor_ids = np.array([stable_id(v) if v is not None else -1 for v in donors], dtype=np.int64)
        sections = np.array([stable_id("section::" + k) if k in verified else -1
                             for k in rows.sample_key], dtype=np.int64)
        coordinates = np.load(old / split / "coordinates.npy")
        neighbors, mask = _neighbors(coordinates, canonical_source, sections, k=8)
        arrays = {"sample": sample_ids, "split_group": groups, "donor": donor_ids,
                  "section": sections, "neighbors": neighbors, "neighbor_mask": mask}
        for key, value in arrays.items():
            _atomic_npy(target / f"{key}.npy", value)
            manifest["splits"][split]["files"][key] = {"path": f"{split}/{key}.npy",
                "sha256": _sha256(target / f"{key}.npy"), "shape": list(value.shape), "dtype": str(value.dtype)}
        identity = pd.DataFrame({"row_id": np.load(old / split / "row_id.npy"), "source_id": canonical_source,
            "source_row_ref": refs, "sample_key": rows.sample_key, "split_group_key": rows.split_group,
            "donor_key": donors, "section_verified": sections >= 0, "split": split})
        identity.to_parquet(target / "row_identity.parquet", index=False)
        manifest["splits"][split]["files"]["row_identity"] = {"path": f"{split}/row_identity.parquet",
            "sha256": _sha256(target / "row_identity.parquet")}
        manifest["splits"][split]["spatial_edges"] = int(mask.sum())
        identities[split] = identity
        report["splits"][split] = {"rows": len(refs), "source_refs_matched": len(refs),
            "sample_count": int(rows.sample_key.nunique()), "group_count": int(rows.split_group.nunique()),
            "known_donor_rows": int((donor_ids >= 0).sum()), "verified_geometry_rows": int((sections >= 0).sum()),
            "neighbor_edges": int(mask.sum())}
        manifest["source_manifests"][split] = str(source / f"{split}-manifest.json")
        report["source_manifests"][split] = _sha256(source / f"{split}-manifest.json")
        print(split, report["splits"][split], flush=True)
    report["overlap"] = {}
    for a, b in itertools.combinations(identities, 2):
        overlaps = {k: sorted(set(identities[a][k].dropna()) & set(identities[b][k].dropna()))
                    for k in ("sample_key", "split_group_key", "donor_key")}
        if overlaps["sample_key"] or overlaps["split_group_key"]:
            raise ValueError(f"sample/group leakage between {a}/{b}")
        report["overlap"][f"{a}/{b}"] = overlaps
    # Check every original byte is still intact after atomic replacements.
    for s in ("train", "validation", "test"):
        old_manifest = json.loads((old / "manifest.json").read_text())
        for info in old_manifest["splits"][s]["files"].values():
            if _sha256(old / info["path"]) != info["sha256"]:
                raise RuntimeError("immutable parent dataset changed")
    manifest["source_shards"] = str(source)
    manifest["source_manifest"] = str(source / "train-manifest.json")
    manifest["identity_repair"] = {"receipt": "identity_repair.json", "version": 2,
        "geometry": "only independently verified sample/section keys; absent geometry remains no-neighbor",
        "parent_manifest_sha256": report["parent_manifest_sha256"]}
    _atomic_json(output / "identity_repair.json", report)
    _atomic_json(output / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "assets/nanobrain_v3_100k")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=ROOT / "contracts/healthy_global_benchmark_v3")
    p.add_argument("--geometry", type=Path, default=Path('/data/brainwm/BrainWM-Training-Benchmark/contracts/human_semantics_v4/spatial_units.parquet'))
    a = p.parse_args(); repair(a.input, a.output, a.contract, a.geometry)
