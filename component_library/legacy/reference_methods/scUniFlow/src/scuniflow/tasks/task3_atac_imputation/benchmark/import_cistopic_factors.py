#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.io
import scipy.sparse as sp
from scipy.stats import rankdata


def _read_names(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def _read_mtx(path):
    X = scipy.io.mmread(path)
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _rank_normalize_rows(X):
    X = np.asarray(X, dtype=np.float32)
    denom = max(X.shape[1] - 1, 1)
    for i in range(X.shape[0]):
        X[i] = (rankdata(X[i], method="average") - 1.0) / denom
    return X


def main():
    parser = argparse.ArgumentParser(description="Build cisTopic evaluation h5ad from low-rank factors.")
    parser.add_argument("--factor_dir", required=True)
    parser.add_argument("--reference_h5ad", required=True)
    parser.add_argument("--output_h5ad", required=True)
    parser.add_argument("--rank_normalize", action="store_true")
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    factor_dir = Path(args.factor_dir).resolve()
    ref = sc.read_h5ad(args.reference_h5ad)

    region_topic = _read_mtx(factor_dir / "region_topic.mtx")
    cell_topic = _read_mtx(factor_dir / "cell_topic.mtx")
    features = _read_names(factor_dir / "features.tsv")
    barcodes = _read_names(factor_dir / "barcodes.tsv")

    if region_topic.shape[0] != len(features):
        raise ValueError("region_topic rows do not match features.tsv")
    if cell_topic.shape[0] != len(barcodes):
        raise ValueError("cell_topic rows do not match barcodes.tsv")
    if region_topic.shape[1] != cell_topic.shape[1]:
        raise ValueError("region_topic and cell_topic have different topic counts")

    ref_obs = ref.obs_names.astype(str)
    ref_var = ref.var_names.astype(str)
    barcode_to_idx = {x: i for i, x in enumerate(barcodes)}
    feature_to_idx = {x: i for i, x in enumerate(features)}
    obs_idx = np.array([barcode_to_idx[x] for x in ref_obs], dtype=np.int64)
    var_idx = np.array([feature_to_idx[x] for x in ref_var], dtype=np.int64)

    cell_topic = cell_topic[obs_idx]
    region_topic = region_topic[var_idx]

    X = (cell_topic @ region_topic.T).astype(np.float32, copy=False)
    if args.rank_normalize:
        X = _rank_normalize_rows(X)

    out = ad.AnnData(X=X, obs=ref.obs.copy(), var=ref.var.copy())
    out.uns["benchmark_method"] = "cistopic"
    out.uns["benchmark_matrix_key"] = "X"
    out.uns["cistopic_factor_import"] = {
        "factor_dir": str(factor_dir),
        "rank_normalize": bool(args.rank_normalize),
        "n_topics": int(cell_topic.shape[1]),
    }

    output = Path(args.output_h5ad).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output)

    summary = {
        "method": "cistopic",
        "factor_dir": str(factor_dir),
        "output_h5ad": str(output),
        "rank_normalize": bool(args.rank_normalize),
        "n_cells": int(out.n_obs),
        "n_features": int(out.n_vars),
        "n_topics": int(cell_topic.shape[1]),
    }
    if args.summary_json:
        with open(Path(args.summary_json).resolve(), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
