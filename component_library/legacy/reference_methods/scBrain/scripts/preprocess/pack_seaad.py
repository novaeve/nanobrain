#!/usr/bin/env python3
"""Pack the SEA-AD MTG MERFISH h5ad file into the NPZ format used by scBrain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", required=True, help="Path to SEAAD_MTG_MERFISH*.h5ad")
    parser.add_argument("--output-npz", default="data/global_data.npz", help="Output packed NPZ path")
    parser.add_argument("--gene-count", type=int, default=140, help="Number of MERFISH genes to keep")
    parser.add_argument(
        "--gene-names-json",
        default="data/merfish_gene_names.json",
        help="Optional JSON output containing the gene order used by foundation-model wrappers",
    )
    return parser.parse_args()


def encode_categorical(values) -> tuple[np.ndarray, dict[str, int]]:
    raw = values.astype(str).to_numpy() if hasattr(values, "to_numpy") else np.asarray(values).astype(str)
    labels = sorted(set(raw))
    mapping = {label: idx for idx, label in enumerate(labels)}
    encoded = np.array([mapping[label] for label in raw], dtype=np.int64)
    return encoded, mapping


def read_matrix(adata, gene_count: int) -> np.ndarray:
    x = adata.X[:, :gene_count]
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def get_obs_float(obs, column: str, default: float = 0.0) -> np.ndarray:
    if column not in obs.columns:
        return np.full(len(obs), default, dtype=np.float32)
    values = obs[column].to_numpy().astype(np.float32)
    return values


def main() -> None:
    args = parse_args()
    input_h5ad = Path(args.input_h5ad)
    output_npz = Path(args.output_npz)
    gene_json = Path(args.gene_names_json) if args.gene_names_json else None

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if gene_json is not None:
        gene_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SEA-AD h5ad: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad, backed="r")
    obs = adata.obs.copy()

    required_obs = [
        "Class",
        "Subclass",
        "Supertype",
        "Donor ID",
        "Supertype confidence",
        "Continuous Pseudo-progression Score",
    ]
    missing = [col for col in required_obs if col not in obs.columns]
    if missing:
        raise KeyError(f"Missing required SEA-AD obs columns: {missing}")
    if "spatial" not in adata.obsm:
        raise KeyError("Missing adata.obsm['spatial']")

    print(f"Reading expression matrix with first {args.gene_count} genes")
    x = read_matrix(adata, args.gene_count)

    y_class, class_map = encode_categorical(obs["Class"])
    y_subclass, subclass_map = encode_categorical(obs["Subclass"])
    y_supertype, supertype_map = encode_categorical(obs["Supertype"])
    batch_donor, donor_map = encode_categorical(obs["Donor ID"])

    confidence = obs["Supertype confidence"].to_numpy().astype(np.float32)
    cps = obs["Continuous Pseudo-progression Score"].to_numpy().astype(np.float32)
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    if "X_spatial_tiled" in adata.obsm:
        spatial_tiled = np.asarray(adata.obsm["X_spatial_tiled"], dtype=np.float32)
    else:
        spatial_tiled = spatial.copy()

    depth_raw = get_obs_float(obs, "Normalized depth from pia", default=np.nan)
    has_depth = ~np.isnan(depth_raw)
    depth_norm = np.where(has_depth, depth_raw, np.float32(-1.0)).astype(np.float32)

    volume_raw = get_obs_float(obs, "Cell volume", default=np.nan)
    has_volume = ~np.isnan(volume_raw)
    if has_volume.any():
        valid = volume_raw[has_volume]
        vol_mean = float(valid.mean())
        vol_std = float(valid.std())
        cell_volume_norm = np.where(has_volume, (volume_raw - vol_mean) / (vol_std + 1e-6), 0.0)
    else:
        vol_mean = 0.0
        vol_std = 1.0
        cell_volume_norm = np.zeros_like(volume_raw, dtype=np.float32)
    cell_volume_norm = cell_volume_norm.astype(np.float32)

    gene_names = [str(gene) for gene in adata.var_names[: args.gene_count]]
    meta = {
        "num_class": len(class_map),
        "num_subclass": len(subclass_map),
        "num_supertype": len(supertype_map),
        "num_donor": len(donor_map),
        "gene_names": gene_names,
        "class_map": class_map,
        "subclass_map": subclass_map,
        "supertype_map": supertype_map,
        "donor_map": donor_map,
        "vol_mean": vol_mean,
        "vol_std": vol_std,
    }

    print(f"Saving packed NPZ: {output_npz}")
    np.savez_compressed(
        output_npz,
        X=x,
        y_class=y_class,
        y_subclass=y_subclass,
        y_supertype=y_supertype,
        batch_donor=batch_donor,
        y_supertype_confidence=confidence,
        spatial=spatial,
        spatial_tiled=spatial_tiled,
        cps=cps,
        depth_norm=depth_norm,
        has_depth=has_depth,
        cell_volume_norm=cell_volume_norm,
        has_volume=has_volume,
        meta=meta,
    )

    if gene_json is not None:
        gene_json.write_text(json.dumps(gene_names, indent=2) + "\n", encoding="utf-8")

    print("Packed SEA-AD dataset")
    print(f"  cells: {x.shape[0]:,}")
    print(f"  genes: {x.shape[1]:,}")
    print(f"  labels: {len(class_map)} / {len(subclass_map)} / {len(supertype_map)}")
    print(f"  donors: {len(donor_map)}")
    print(f"  output: {output_npz}")


if __name__ == "__main__":
    main()
