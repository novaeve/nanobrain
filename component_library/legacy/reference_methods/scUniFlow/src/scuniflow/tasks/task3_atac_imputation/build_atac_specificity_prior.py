import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "model.py").exists() and (p / "data.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_atac_biological import _load_rna_reference_labels  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an RNA-cluster-conditioned ATAC peak specificity prior for oracle/ablation gating."
    )
    parser.add_argument("--raw_atac_h5ad", required=True)
    parser.add_argument("--multiome_h5", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--rna_resolution", type=float, default=1.0)
    parser.add_argument("--rna_top_genes", type=int, default=2000)
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=3.0)
    return parser.parse_args()


def _binary_chunk(X, rows, start, end):
    chunk = X[rows, start:end]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk > 0, dtype=np.float32)


def main():
    args = parse_args()
    raw = sc.read_h5ad(args.raw_atac_h5ad)
    labels_map = _load_rna_reference_labels(
        multiome_h5=args.multiome_h5,
        target_obs_names=raw.obs_names,
        rna_resolution=args.rna_resolution,
        rna_top_genes=args.rna_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )
    obs_names = np.asarray(raw.obs_names).astype(str)
    keep = np.asarray([name in labels_map for name in obs_names], dtype=bool)
    if not np.any(keep):
        raise ValueError("No ATAC cells overlap RNA labels.")

    kept_obs_names = obs_names[keep]
    label_values = np.asarray([labels_map[name] for name in kept_obs_names], dtype=str)
    cluster_names = np.asarray(sorted(np.unique(label_values), key=lambda x: int(x) if x.isdigit() else x))
    cluster_to_idx = {name: i for i, name in enumerate(cluster_names)}
    obs_cluster_idx = np.asarray([cluster_to_idx[x] for x in label_values], dtype=np.int64)

    X = raw.X.tocsr() if sp.issparse(raw.X) else np.asarray(raw.X)
    row_indices = np.where(keep)[0].astype(np.int64)
    n_clusters = int(cluster_names.shape[0])
    n_features = int(raw.n_vars)
    cluster_freq = np.zeros((n_clusters, n_features), dtype=np.float32)
    global_sum = np.zeros((n_features,), dtype=np.float64)

    chunk_size = 512
    cluster_rows = [row_indices[obs_cluster_idx == c] for c in range(n_clusters)]
    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        global_chunk_sum = np.zeros((end - start,), dtype=np.float64)
        for c, rows in enumerate(cluster_rows):
            if rows.size == 0:
                continue
            bin_chunk = _binary_chunk(X, rows, start, end)
            freq = bin_chunk.mean(axis=0)
            cluster_freq[c, start:end] = freq.astype(np.float32, copy=False)
            global_chunk_sum += bin_chunk.sum(axis=0)
        global_sum[start:end] = global_chunk_sum

    global_freq = (global_sum / max(row_indices.size, 1)).astype(np.float32)
    eps = float(args.eps)
    specificity = np.log((cluster_freq + eps) / (global_freq[None, :] + eps)).astype(np.float32)
    clip = float(args.clip)
    if clip > 0:
        specificity = np.clip(specificity, -clip, clip).astype(np.float32, copy=False)

    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        specificity=specificity,
        cluster_freq=cluster_freq,
        global_freq=global_freq,
        cluster_names=cluster_names.astype(str),
        obs_names=kept_obs_names.astype(str),
        obs_cluster_idx=obs_cluster_idx.astype(np.int64),
        var_names=np.asarray(raw.var_names).astype(str),
        score_type=np.asarray(["log_cluster_freq_over_global_freq"], dtype=str),
    )

    summary = {
        "raw_atac_h5ad": str(Path(args.raw_atac_h5ad).resolve()),
        "multiome_h5": str(Path(args.multiome_h5).resolve()),
        "output_npz": str(output_npz),
        "n_cells": int(row_indices.size),
        "n_features": n_features,
        "n_clusters": n_clusters,
        "cluster_names": cluster_names.astype(str).tolist(),
        "score_type": "log((cluster_freq + eps) / (global_freq + eps))",
        "eps": eps,
        "clip": clip,
        "specificity_mean": float(specificity.mean()),
        "specificity_std": float(specificity.std()),
        "specificity_min": float(specificity.min()),
        "specificity_max": float(specificity.max()),
    }
    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
