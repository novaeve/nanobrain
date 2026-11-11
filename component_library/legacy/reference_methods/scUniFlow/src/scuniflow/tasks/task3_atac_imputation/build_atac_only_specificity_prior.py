import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an ATAC-only cluster-conditioned peak specificity prior for non-oracle gating."
    )
    parser.add_argument("--raw_atac_h5ad", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--max_cells", type=int, default=None)
    return parser.parse_args()


def _as_dense(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _binary_chunk(X, rows, start, end):
    chunk = X[rows, start:end]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk > 0, dtype=np.float32)


def _cluster_raw_atac(adata, n_pcs, n_neighbors, resolution, seed):
    cluster_adata = adata.copy()
    if sp.issparse(cluster_adata.X):
        # For 500 x 50k or 4.8k x 50k top50k this is acceptable and matches eval conventions.
        cluster_adata.X = cluster_adata.X.toarray().astype(np.float32, copy=False)
    else:
        cluster_adata.X = np.asarray(cluster_adata.X, dtype=np.float32)

    n_comps = max(2, min(int(n_pcs), cluster_adata.n_obs - 1, cluster_adata.n_vars - 1))
    sc.pp.pca(cluster_adata, n_comps=n_comps, zero_center=False, random_state=int(seed))
    sc.pp.neighbors(
        cluster_adata,
        n_neighbors=min(int(n_neighbors), max(2, cluster_adata.n_obs - 1)),
        n_pcs=min(30, n_comps),
    )
    sc.tl.leiden(
        cluster_adata,
        resolution=float(resolution),
        key_added="atac_only_leiden",
        random_state=int(seed),
    )
    return cluster_adata.obs["atac_only_leiden"].astype(str).to_numpy()


def main():
    args = parse_args()
    raw = sc.read_h5ad(args.raw_atac_h5ad)
    if args.max_cells is not None and raw.n_obs > int(args.max_cells):
        rng = np.random.RandomState(int(args.seed))
        keep = np.sort(rng.choice(raw.n_obs, size=int(args.max_cells), replace=False))
        raw = raw[keep].copy()

    labels = _cluster_raw_atac(
        raw,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
        seed=args.seed,
    )
    cluster_names = np.asarray(sorted(np.unique(labels), key=lambda x: int(x) if x.isdigit() else x))
    cluster_to_idx = {name: i for i, name in enumerate(cluster_names)}
    obs_cluster_idx = np.asarray([cluster_to_idx[x] for x in labels], dtype=np.int64)

    X = raw.X.tocsr() if sp.issparse(raw.X) else np.asarray(raw.X)
    n_clusters = int(cluster_names.shape[0])
    n_features = int(raw.n_vars)
    cluster_freq = np.zeros((n_clusters, n_features), dtype=np.float32)
    global_sum = np.zeros((n_features,), dtype=np.float64)
    cluster_rows = [np.where(obs_cluster_idx == c)[0].astype(np.int64) for c in range(n_clusters)]

    chunk_size = 512
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

    global_freq = (global_sum / max(raw.n_obs, 1)).astype(np.float32)
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
        obs_names=np.asarray(raw.obs_names).astype(str),
        obs_cluster_idx=obs_cluster_idx.astype(np.int64),
        var_names=np.asarray(raw.var_names).astype(str),
        score_type=np.asarray(["atac_only_log_cluster_freq_over_global_freq"], dtype=str),
    )

    summary = {
        "raw_atac_h5ad": str(Path(args.raw_atac_h5ad).resolve()),
        "output_npz": str(output_npz),
        "label_source": "ATAC-only Leiden on raw ATAC PCA",
        "n_cells": int(raw.n_obs),
        "n_features": n_features,
        "n_clusters": n_clusters,
        "cluster_names": cluster_names.astype(str).tolist(),
        "resolution": float(args.resolution),
        "n_pcs": int(args.n_pcs),
        "n_neighbors": int(args.n_neighbors),
        "seed": int(args.seed),
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
