#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.io
import scipy.sparse as sp


def _read_names(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser(description="Import R baseline MatrixMarket output as h5ad.")
    parser.add_argument("--matrix_mtx", required=True)
    parser.add_argument("--features_tsv", required=True)
    parser.add_argument("--barcodes_tsv", required=True)
    parser.add_argument("--reference_h5ad", required=True)
    parser.add_argument("--output_h5ad", required=True)
    parser.add_argument(
        "--orientation",
        default="features_by_cells",
        choices=["features_by_cells", "cells_by_features"],
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    matrix = scipy.io.mmread(args.matrix_mtx)
    if sp.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float32)
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
    X = matrix.T if args.orientation == "features_by_cells" else matrix

    obs_names = _read_names(args.barcodes_tsv)
    var_names = _read_names(args.features_tsv)
    if X.shape != (len(obs_names), len(var_names)):
        raise ValueError(
            f"matrix shape {X.shape} is inconsistent with "
            f"{len(obs_names)} cells and {len(var_names)} features"
        )

    ref = sc.read_h5ad(args.reference_h5ad)
    obs = ref.obs.loc[obs_names].copy() if set(obs_names).issubset(set(ref.obs_names)) else None
    var = ref.var.loc[var_names].copy() if set(var_names).issubset(set(ref.var_names)) else None
    out = ad.AnnData(X=X, obs=obs, var=var)
    out.obs_names = obs_names
    out.var_names = var_names
    out.uns["benchmark_method"] = args.method
    out.uns["benchmark_matrix_key"] = "X"
    out.uns["r_matrix_import"] = {
        "matrix_mtx": str(Path(args.matrix_mtx).resolve()),
        "features_tsv": str(Path(args.features_tsv).resolve()),
        "barcodes_tsv": str(Path(args.barcodes_tsv).resolve()),
        "orientation": args.orientation,
    }

    output = Path(args.output_h5ad).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(output)

    summary = {
        "method": args.method,
        "output_h5ad": str(output),
        "n_cells": int(out.n_obs),
        "n_features": int(out.n_vars),
        "matrix_sparse": bool(sp.issparse(X)),
    }
    if args.summary_json:
        with open(Path(args.summary_json).resolve(), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
