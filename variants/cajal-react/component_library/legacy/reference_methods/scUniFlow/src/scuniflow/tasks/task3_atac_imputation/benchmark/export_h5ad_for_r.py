#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.io
import scipy.sparse as sp


def _get_matrix(adata, key):
    if key == "X":
        return adata.X
    if key.startswith("layer:"):
        name = key.split(":", 1)[1]
        return adata.layers[name]
    raise ValueError("--matrix_key must be X or layer:<name>")


def main():
    parser = argparse.ArgumentParser(description="Export h5ad matrix for R baselines.")
    parser.add_argument("--input_h5ad", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--matrix_key", default="X")
    parser.add_argument(
        "--orientation",
        default="features_by_cells",
        choices=["features_by_cells", "cells_by_features"],
    )
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    X = _get_matrix(adata, args.matrix_key)
    if sp.issparse(X):
        X = X.tocsr().astype(np.float64)
        if args.binary:
            X = X.copy()
            X.data = np.ones_like(X.data)
    else:
        X = np.asarray(X, dtype=np.float64)
        if args.binary:
            X = (X > 0).astype(np.float64)

    matrix = X.T if args.orientation == "features_by_cells" else X
    scipy.io.mmwrite(outdir / "matrix.mtx", matrix)

    with open(outdir / "features.tsv", "w") as f:
        f.write("\n".join(map(str, adata.var_names)) + "\n")
    with open(outdir / "barcodes.tsv", "w") as f:
        f.write("\n".join(map(str, adata.obs_names)) + "\n")

    summary = {
        "input_h5ad": str(Path(args.input_h5ad).resolve()),
        "output_dir": str(outdir),
        "matrix_key": args.matrix_key,
        "orientation": args.orientation,
        "binary": bool(args.binary),
        "n_cells": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "nnz": int(X.nnz) if sp.issparse(X) else int(np.count_nonzero(X)),
    }
    with open(outdir / "export_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if args.summary_json:
        with open(Path(args.summary_json).resolve(), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
