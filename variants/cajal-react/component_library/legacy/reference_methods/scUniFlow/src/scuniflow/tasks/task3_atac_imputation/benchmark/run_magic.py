#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import magic
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def _dense_or_svd(X, n_components, seed):
    if sp.issparse(X):
        if n_components and n_components > 0 and X.shape[1] > n_components:
            svd = TruncatedSVD(n_components=min(n_components, X.shape[1] - 1), random_state=seed)
            return svd.fit_transform(X).astype(np.float32, copy=False), {
                "input_space": "truncated_svd",
                "n_components": int(svd.n_components),
                "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            }
        return X.toarray().astype(np.float32, copy=False), {"input_space": "dense_peaks"}
    arr = np.asarray(X, dtype=np.float32)
    if n_components and n_components > 0 and arr.shape[1] > n_components:
        svd = TruncatedSVD(n_components=min(n_components, arr.shape[1] - 1), random_state=seed)
        return svd.fit_transform(arr).astype(np.float32, copy=False), {
            "input_space": "truncated_svd",
            "n_components": int(svd.n_components),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        }
    return arr, {"input_space": "dense_peaks"}


def main():
    parser = argparse.ArgumentParser(description="Run MAGIC baseline on scATAC TF-IDF h5ad")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--t", default="3")
    parser.add_argument("--n_pca", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features_dense", type=int, default=20000)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    n_features = int(adata.n_vars)
    use_svd = n_features > int(args.max_features_dense)
    X_in, input_summary = _dense_or_svd(adata.X, int(args.n_pca) if use_svd else None, args.seed)

    t_value = "auto" if str(args.t).lower() == "auto" else int(args.t)
    magic_op = magic.MAGIC(
        knn=int(args.knn),
        t=t_value,
        n_pca=None if use_svd else int(args.n_pca),
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        solver="approximate" if use_svd else "exact",
    )
    X_magic = magic_op.fit_transform(X_in)
    X_magic = np.asarray(X_magic, dtype=np.float32)

    out = adata.copy()
    if X_magic.shape[1] == adata.n_vars:
        out.X = X_magic
    else:
        out.obsm["X_magic"] = X_magic
        out.uns["benchmark_matrix_key"] = "obsm:X_magic"
    out.uns["magic_benchmark"] = {
        "input": str(input_path),
        "knn": int(args.knn),
        "t": str(args.t),
        "n_pca": int(args.n_pca),
        "n_jobs": int(args.n_jobs),
        "seed": int(args.seed),
        "max_features_dense": int(args.max_features_dense),
        **input_summary,
    }
    out.write_h5ad(output_path)
    print(json.dumps(out.uns["magic_benchmark"], indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
