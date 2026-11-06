#!/usr/bin/env python3
"""Pack Kukanja MS/EAE Xenium h5ad files into the NPZ format used by scBrain."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc


EAE_ANNOT_COLS = [
    "Annotation 1 (broad)",
    "Annotation 2 (medium)",
    "Annotation 3 (medium with DA)",
    "Annotation 4 (high)",
]
EAE_LABEL_NAMES = [
    "annotation_broad",
    "annotation_medium",
    "annotation_medium_da",
    "annotation_high",
]
MS_OPTIONS = [
    ["Level0", "Level1_5", "Level2", "Level3"],
    ["Level1", "Level1_5", "Level2", "Level3"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ms", "eae"], required=True, help="Kukanja dataset variant")
    parser.add_argument("--input-h5ad", required=True, help="Path to the downloaded Kukanja h5ad file")
    parser.add_argument("--output-npz", required=True, help="Output packed NPZ path")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Rows per expression extraction chunk")
    return parser.parse_args()


def read_matrix_chunked(adata, chunk_size: int) -> np.ndarray:
    n_cells, n_genes = adata.shape
    x = np.empty((n_cells, n_genes), dtype=np.float32)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata.X[start:end]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        x[start:end] = np.asarray(chunk, dtype=np.float32)
        print(f"  processed {end:,}/{n_cells:,} cells")
    return x


def encode_labels(obs, annot_cols: list[str]) -> tuple[list[str], dict[str, dict[str, int]], dict[str, np.ndarray]]:
    label_names = [sanitize_label_name(col) for col in annot_cols]
    label_maps: dict[str, dict[str, int]] = {}
    label_arrays: dict[str, np.ndarray] = {}
    for col, name in zip(annot_cols, label_names):
        raw = obs[col].astype(str).to_numpy()
        labels = sorted(set(raw))
        mapping = {label: idx for idx, label in enumerate(labels)}
        label_maps[name] = mapping
        label_arrays[name] = np.array([mapping[label] for label in raw], dtype=np.int64)
        print(f"  {name}: {len(labels)} classes")
    return label_names, label_maps, label_arrays


def sanitize_label_name(column: str) -> str:
    return (
        column.lower()
        .replace("annotation 1 (broad)", "annotation_broad")
        .replace("annotation 2 (medium)", "annotation_medium")
        .replace("annotation 3 (medium with da)", "annotation_medium_da")
        .replace("annotation 4 (high)", "annotation_high")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def hierarchy_violations(obs, cols: list[str]) -> int:
    total = 0
    for parent_col, child_col in zip(cols[:-1], cols[1:]):
        mapping = obs.groupby(child_col, observed=True)[parent_col].nunique()
        total += int((mapping > 1).sum())
    return total


def choose_ms_hierarchy(obs) -> list[str]:
    valid_options = []
    for cols in MS_OPTIONS:
        if all(col in obs.columns for col in cols):
            valid_options.append((hierarchy_violations(obs, cols), cols))
    if not valid_options:
        raise KeyError(f"MS h5ad must contain one of these hierarchy definitions: {MS_OPTIONS}")
    valid_options.sort(key=lambda item: item[0])
    violations, cols = valid_options[0]
    print(f"Using MS hierarchy {cols} with {violations} parent-child violations")
    return cols


def encode_sample_ids(obs) -> tuple[np.ndarray, dict[str, int]]:
    for candidate in ["sample_id", "library_id", "sample", "donor"]:
        if candidate in obs.columns:
            raw = obs[candidate].astype(str).to_numpy()
            if len(set(raw)) <= 50:
                break
    else:
        raise KeyError("Could not find sample column. Expected one of sample_id, library_id, sample, donor.")

    labels = sorted(set(raw))
    mapping = {label: idx for idx, label in enumerate(labels)}
    sample_ids = np.array([mapping[label] for label in raw], dtype=np.int64)
    print(f"Using sample column with {len(labels)} samples")
    return sample_ids, mapping


def disease_score(obs, dataset: str) -> np.ndarray:
    if dataset == "eae" and "SCORE" in obs.columns:
        values = obs["SCORE"].to_numpy().astype(np.float32)
        return np.where(np.isnan(values), 0.0, values).astype(np.float32)

    for candidate in ["SCORE", "score", "disease_score"]:
        if candidate in obs.columns:
            values = obs[candidate].to_numpy().astype(np.float32)
            return np.where(np.isnan(values), 0.0, values).astype(np.float32)

    for candidate in ["type", "condition", "disease", "group"]:
        if candidate in obs.columns:
            raw = obs[candidate].astype(str).to_numpy()
            scores = []
            for value in raw:
                lowered = value.lower().strip()
                if any(token in lowered for token in ["ctrl", "control", "cntrl", "healthy"]):
                    scores.append(0.0)
                elif any(token in lowered for token in ["ms", "eae", "disease"]):
                    scores.append(1.0)
                else:
                    scores.append(0.5)
            return np.asarray(scores, dtype=np.float32)

    return np.zeros(len(obs), dtype=np.float32)


def normalize_spatial(spatial: np.ndarray) -> np.ndarray:
    spatial_norm = np.empty_like(spatial, dtype=np.float32)
    for dim in range(spatial.shape[1]):
        mn = spatial[:, dim].min()
        mx = spatial[:, dim].max()
        spatial_norm[:, dim] = (spatial[:, dim] - mn) / (mx - mn + 1e-12)
    return spatial_norm


def main() -> None:
    args = parse_args()
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Kukanja {args.dataset.upper()} h5ad: {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad, backed="r")
    obs = adata.obs.copy()

    if args.dataset == "ms":
        annot_cols = choose_ms_hierarchy(obs)
        label_names, label_maps, label_arrays = encode_labels(obs, annot_cols)
    else:
        missing = [col for col in EAE_ANNOT_COLS if col not in obs.columns]
        if missing:
            raise KeyError(f"Missing EAE annotation columns: {missing}")
        label_names = EAE_LABEL_NAMES
        label_maps = {}
        label_arrays = {}
        for col, name in zip(EAE_ANNOT_COLS, label_names):
            raw = obs[col].astype(str).to_numpy()
            labels = sorted(set(raw))
            mapping = {label: idx for idx, label in enumerate(labels)}
            label_maps[name] = mapping
            label_arrays[name] = np.array([mapping[label] for label in raw], dtype=np.int64)
            print(f"  {name}: {len(labels)} classes")
        annot_cols = EAE_ANNOT_COLS

    if "spatial" not in adata.obsm:
        raise KeyError("Missing adata.obsm['spatial']")

    x = read_matrix_chunked(adata, args.chunk_size)
    sample_ids, sample_map = encode_sample_ids(obs)
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    spatial_norm = normalize_spatial(spatial)
    score = disease_score(obs, args.dataset)

    meta = {
        "label_names": label_names,
        "num_samples": len(sample_map),
        "n_genes": x.shape[1],
        "gene_names": [str(gene) for gene in adata.var_names],
        "label_maps": {name: {str(k): int(v) for k, v in label_maps[name].items()} for name in label_names},
        "sample_map": {str(k): int(v) for k, v in sample_map.items()},
        "annot_col_names": annot_cols,
    }
    for name in label_names:
        meta[f"num_{name}"] = len(label_maps[name])

    save_dict = {
        "X": x,
        "sample_ids": sample_ids,
        "spatial": spatial,
        "spatial_norm": spatial_norm,
        "disease_score": score,
        "meta": meta,
    }
    for name in label_names:
        save_dict[f"y_{name}"] = label_arrays[name]

    print(f"Saving packed NPZ: {output_npz}")
    np.savez_compressed(output_npz, **save_dict)
    print("Packed Kukanja dataset")
    print(f"  cells: {x.shape[0]:,}")
    print(f"  genes: {x.shape[1]:,}")
    print(f"  levels: {label_names}")
    print(f"  output: {output_npz}")


if __name__ == "__main__":
    main()
