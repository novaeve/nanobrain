import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an ATAC-only LSI-cluster-conditioned peak specificity prior."
    )
    parser.add_argument("--raw_atac_h5ad", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--n_components", type=int, default=30)
    parser.add_argument("--drop_first", type=int, default=1)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--max_cells", type=int, default=None)
    return parser.parse_args()


def _binary_chunk(X, rows, start, end):
    chunk = X[rows, start:end]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk > 0, dtype=np.float32)


def _safe_csr_float32(X):
    if sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(X, dtype=np.float32))


def _cluster_lsi(adata, n_components, drop_first, n_neighbors, resolution, seed):
    X = _safe_csr_float32(adata.X)
    n_components = max(2, min(int(n_components), adata.n_obs - 1, adata.n_vars - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
    X_lsi = svd.fit_transform(X).astype(np.float32, copy=False)

    drop_first = max(0, int(drop_first))
    if drop_first > 0 and X_lsi.shape[1] > drop_first:
        X_lsi_use = X_lsi[:, drop_first:]
    else:
        X_lsi_use = X_lsi

    cluster_adata = sc.AnnData(
        X=np.zeros((adata.n_obs, 1), dtype=np.float32),
        obs=adata.obs.copy(),
    )
    cluster_adata.obsm["X_lsi"] = X_lsi_use
    sc.pp.neighbors(
        cluster_adata,
        use_rep="X_lsi",
        n_neighbors=min(int(n_neighbors), max(2, adata.n_obs - 1)),
    )
    sc.tl.leiden(
        cluster_adata,
        resolution=float(resolution),
        key_added="atac_only_lsi_leiden",
        random_state=int(seed),
    )
    return (
        cluster_adata.obs["atac_only_lsi_leiden"].astype(str).to_numpy(),
        svd.explained_variance_ratio_.astype(np.float32),
    )


def main():
    args = parse_args()
    raw = sc.read_h5ad(args.raw_atac_h5ad)
    if args.max_cells is not None and raw.n_obs > int(args.max_cells):
        rng = np.random.RandomState(int(args.seed))
        keep = np.sort(rng.choice(raw.n_obs, size=int(args.max_cells), replace=False))
        raw = raw[keep].copy()

    labels, explained_variance_ratio = _cluster_lsi(
        raw,
        n_components=args.n_components,
        drop_first=args.drop_first,
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
        score_type=np.asarray(["atac_only_lsi_log_cluster_freq_over_global_freq"], dtype=str),
        lsi_explained_variance_ratio=explained_variance_ratio,
    )

    summary = {
        "raw_atac_h5ad": str(Path(args.raw_atac_h5ad).resolve()),
        "output_npz": str(output_npz),
        "label_source": "ATAC-only Leiden on raw ATAC LSI",
        "n_cells": int(raw.n_obs),
        "n_features": n_features,
        "n_clusters": n_clusters,
        "cluster_names": cluster_names.astype(str).tolist(),
        "resolution": float(args.resolution),
        "n_components": int(args.n_components),
        "drop_first": int(args.drop_first),
        "n_neighbors": int(args.n_neighbors),
        "seed": int(args.seed),
        "score_type": "log((cluster_freq + eps) / (global_freq + eps))",
        "eps": eps,
        "clip": clip,
        "lsi_explained_variance_ratio_first5": explained_variance_ratio[:5].astype(float).tolist(),
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
