#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def _load_10x_peaks(path: Path):
    adata = sc.read_10x_h5(str(path), gex_only=False)
    if "feature_types" in adata.var.columns:
        adata = adata[:, adata.var["feature_types"].astype(str).to_numpy() == "Peaks"].copy()
    adata.var_names_make_unique()
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    else:
        adata.X = adata.X.tocsr().astype(np.float32)
    return adata


def main():
    parser = argparse.ArgumentParser(description="Prepare raw-count ATAC h5ad aligned to a TF-IDF h5ad")
    parser.add_argument("--raw_10x_h5", required=True)
    parser.add_argument("--tfidf_h5ad", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    raw_path = Path(args.raw_10x_h5).resolve()
    tfidf_path = Path(args.tfidf_h5ad).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _load_10x_peaks(raw_path)
    tfidf = sc.read_h5ad(tfidf_path)

    common_obs = np.intersect1d(raw.obs_names.astype(str), tfidf.obs_names.astype(str))
    common_var = np.intersect1d(raw.var_names.astype(str), tfidf.var_names.astype(str))
    if common_obs.size == 0:
        raise ValueError("raw 10x h5 and TF-IDF h5ad have no shared cells")
    if common_var.size == 0:
        raise ValueError("raw 10x h5 and TF-IDF h5ad have no shared peaks")

    ordered_obs = [x for x in tfidf.obs_names.astype(str) if x in set(common_obs)]
    ordered_var = [x for x in tfidf.var_names.astype(str) if x in set(common_var)]
    raw = raw[ordered_obs, ordered_var].copy()

    for col in tfidf.var.columns:
        raw.var[col] = tfidf.var.loc[ordered_var, col].to_numpy()
    for col in tfidf.obs.columns:
        raw.obs[col] = tfidf.obs.loc[ordered_obs, col].to_numpy()

    raw.uns["aligned_to_tfidf_h5ad"] = str(tfidf_path)
    raw.write_h5ad(output_path)

    summary = {
        "raw_10x_h5": str(raw_path),
        "tfidf_h5ad": str(tfidf_path),
        "output": str(output_path),
        "n_cells": int(raw.n_obs),
        "n_peaks": int(raw.n_vars),
        "nnz": int(raw.X.nnz) if sp.issparse(raw.X) else int(np.count_nonzero(raw.X)),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
