import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "model.py").exists() and (p / "data.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="scUniFlow scATAC biological evaluation")
    parser.add_argument("--raw_atac_h5ad", type=str, required=True)
    parser.add_argument("--imputed_atac_h5ad", type=str, required=True)
    parser.add_argument("--multiome_h5", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--rna_resolution", type=float, default=1.0)
    parser.add_argument("--atac_resolution", type=float, default=1.0)
    parser.add_argument("--rna_top_genes", type=int, default=2000)
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--peak_chunk_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _take_dense_rows(X, indices):
    if sp.issparse(X):
        return X[indices].toarray().astype(np.float32, copy=False)
    return np.asarray(X[indices], dtype=np.float32)


def _cluster_with_pca_leiden(adata, resolution, n_pcs, n_neighbors, key_added, seed):
    n_comps = max(2, min(int(n_pcs), adata.n_obs - 1, adata.n_vars - 1))
    # Use the same representation and PCA convention for sparse and dense
    # matrices. Otherwise a raw sparse matrix and its dense copy can produce
    # different Leiden/ARI results even when values are identical.
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32, copy=False)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    sc.pp.pca(adata, n_comps=n_comps, zero_center=False, random_state=int(seed))
    sc.pp.neighbors(adata, n_neighbors=min(int(n_neighbors), max(2, adata.n_obs - 1)), n_pcs=min(30, n_comps))
    sc.tl.leiden(adata, resolution=float(resolution), key_added=key_added, random_state=int(seed))
    return adata.obs[key_added].astype(str).to_numpy()


def _load_rna_reference_labels(multiome_h5, target_obs_names, rna_resolution, rna_top_genes, n_pcs, n_neighbors, seed):
    adata_multiome = sc.read_10x_h5(multiome_h5, gex_only=False)
    adata_rna = adata_multiome[:, adata_multiome.var["feature_types"] == "Gene Expression"].copy()
    adata_rna.var_names_make_unique()
    common = np.intersect1d(np.asarray(target_obs_names).astype(str), np.asarray(adata_rna.obs_names).astype(str))
    if common.size == 0:
        raise ValueError("RNA 侧 barcode 与 ATAC 数据没有交集，无法生成 reference labels。")
    adata_rna = adata_rna[common].copy()
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    n_top = min(int(rna_top_genes), max(200, adata_rna.n_vars - 1))
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=n_top, flavor="seurat")
    if "highly_variable" in adata_rna.var:
        adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
    labels = _cluster_with_pca_leiden(
        adata_rna,
        resolution=rna_resolution,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        key_added="rna_leiden",
        seed=seed,
    )
    return {barcode: label for barcode, label in zip(adata_rna.obs_names.astype(str), labels)}


def _build_raw_metacells(raw_X, labels):
    metacells = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            continue
        if sp.issparse(raw_X):
            meta = np.asarray(raw_X[idx].mean(axis=0)).ravel().astype(np.float32, copy=False)
        else:
            meta = np.asarray(raw_X[idx], dtype=np.float32).mean(axis=0).astype(np.float32, copy=False)
        metacells[str(label)] = meta
    return metacells


def _compute_cellwise_metacell_pearson(X, labels, metacells, batch_size=128):
    corrs = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            continue
        meta = np.asarray(metacells[str(label)], dtype=np.float32)
        meta_center = meta - meta.mean()
        meta_norm = float(np.linalg.norm(meta_center))
        if meta_norm <= 0:
            continue
        for start in range(0, idx.size, batch_size):
            batch_idx = idx[start:start + batch_size]
            rows = _take_dense_rows(X, batch_idx)
            rows_center = rows - rows.mean(axis=1, keepdims=True)
            row_norm = np.linalg.norm(rows_center, axis=1)
            valid = row_norm > 0
            if not np.any(valid):
                continue
            corr = (rows_center[valid] @ meta_center) / (row_norm[valid] * meta_norm)
            corrs.extend(corr.astype(np.float64).tolist())
    if not corrs:
        return None
    return float(np.mean(corrs))


def _compute_peakwise_spearman(raw_X, imputed_X, chunk_size=256):
    if sp.issparse(raw_X):
        raw_csc = raw_X.tocsc()
    else:
        raw_csc = None

    n_features = raw_X.shape[1]
    corr_sum = 0.0
    corr_cnt = 0
    for start in tqdm(range(0, n_features, chunk_size), desc="Peak-wise Spearman", leave=False):
        end = min(start + chunk_size, n_features)
        if raw_csc is not None:
            raw_chunk = raw_csc[:, start:end].toarray().astype(np.float32, copy=False)
        else:
            raw_chunk = np.asarray(raw_X[:, start:end], dtype=np.float32)
        imp_chunk = np.asarray(imputed_X[:, start:end], dtype=np.float32)

        valid_cols = raw_chunk.sum(axis=0) > 0
        if not np.any(valid_cols):
            continue

        raw_valid = raw_chunk[:, valid_cols].astype(np.float64, copy=False)
        imp_valid = imp_chunk[:, valid_cols].astype(np.float64, copy=False)
        raw_ranks = rankdata(raw_valid, axis=0, method="average")
        imp_ranks = rankdata(imp_valid, axis=0, method="average")
        raw_ranks -= raw_ranks.mean(axis=0, keepdims=True)
        imp_ranks -= imp_ranks.mean(axis=0, keepdims=True)
        denom = np.sqrt(
            np.square(raw_ranks).sum(axis=0) * np.square(imp_ranks).sum(axis=0)
        )
        valid = denom > 0
        if not np.any(valid):
            continue
        corr = np.sum(raw_ranks[:, valid] * imp_ranks[:, valid], axis=0) / denom[valid]
        corr_sum += float(np.sum(corr))
        corr_cnt += int(corr.size)
    if corr_cnt == 0:
        return None
    return float(corr_sum / corr_cnt)


def main():
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    raw_atac = sc.read_h5ad(args.raw_atac_h5ad)
    imputed_atac = sc.read_h5ad(args.imputed_atac_h5ad)

    common_obs = np.intersect1d(np.asarray(raw_atac.obs_names).astype(str), np.asarray(imputed_atac.obs_names).astype(str))
    if common_obs.size == 0:
        raise ValueError("raw ATAC 与 imputed ATAC 没有共同细胞。")
    raw_atac = raw_atac[common_obs].copy()
    imputed_atac = imputed_atac[common_obs].copy()

    rna_label_map = _load_rna_reference_labels(
        multiome_h5=args.multiome_h5,
        target_obs_names=raw_atac.obs_names,
        rna_resolution=args.rna_resolution,
        rna_top_genes=args.rna_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )
    common_obs = np.array([obs for obs in raw_atac.obs_names.astype(str) if obs in rna_label_map], dtype=object)
    if common_obs.size == 0:
        raise ValueError("RNA labels 与 ATAC 细胞无法对齐。")

    raw_atac = raw_atac[common_obs].copy()
    imputed_atac = imputed_atac[common_obs].copy()
    rna_labels = np.asarray([rna_label_map[obs] for obs in common_obs], dtype=object)

    raw_metacells = _build_raw_metacells(raw_atac.X, rna_labels)
    meta_cell_pearson_raw = _compute_cellwise_metacell_pearson(raw_atac.X, rna_labels, raw_metacells)
    meta_cell_pearson_imputed = _compute_cellwise_metacell_pearson(imputed_atac.X, rna_labels, raw_metacells)
    peakwise_spearman = _compute_peakwise_spearman(
        raw_atac.X,
        imputed_atac.X,
        chunk_size=args.peak_chunk_size,
    )

    raw_cluster_adata = raw_atac.copy()
    imputed_cluster_adata = imputed_atac.copy()
    raw_cluster_labels = _cluster_with_pca_leiden(
        raw_cluster_adata,
        resolution=args.atac_resolution,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        key_added="raw_atac_leiden",
        seed=args.seed,
    )
    imputed_cluster_labels = _cluster_with_pca_leiden(
        imputed_cluster_adata,
        resolution=args.atac_resolution,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        key_added="imputed_atac_leiden",
        seed=args.seed,
    )

    summary = {
        "raw_atac_h5ad": str(Path(args.raw_atac_h5ad).resolve()),
        "imputed_atac_h5ad": str(Path(args.imputed_atac_h5ad).resolve()),
        "multiome_h5": str(Path(args.multiome_h5).resolve()),
        "n_cells": int(raw_atac.n_obs),
        "n_features": int(raw_atac.n_vars),
        "rna_resolution": float(args.rna_resolution),
        "atac_resolution": float(args.atac_resolution),
        "rna_top_genes": int(args.rna_top_genes),
        "n_pcs": int(args.n_pcs),
        "n_neighbors": int(args.n_neighbors),
        "peak_chunk_size": int(args.peak_chunk_size),
        "seed": int(args.seed),
        "n_rna_clusters": int(len(np.unique(rna_labels))),
        "meta_cell_pearson_raw": meta_cell_pearson_raw,
        "meta_cell_pearson_imputed": meta_cell_pearson_imputed,
        "peakwise_spearman_raw_vs_imputed": peakwise_spearman,
        "ari_raw_atac_vs_rna_label": float(adjusted_rand_score(rna_labels, raw_cluster_labels)),
        "ari_imputed_atac_vs_rna_label": float(adjusted_rand_score(rna_labels, imputed_cluster_labels)),
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {output_json}")


if __name__ == "__main__":
    main()
