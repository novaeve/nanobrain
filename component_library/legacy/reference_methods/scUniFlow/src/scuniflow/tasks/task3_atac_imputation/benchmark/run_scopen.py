#!/usr/bin/env python
import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from scopen.Main import estimate_rank, run_nmf, tf_idf_transform


def _load_h5ad(path):
    adata = sc.read_h5ad(path)
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    else:
        adata.X = adata.X.tocsr().astype(np.float32)
    return adata


def _load_10x_h5(path):
    adata = sc.read_10x_h5(str(path), gex_only=False)
    if "feature_types" in adata.var.columns:
        keep = adata.var["feature_types"].astype(str).to_numpy() == "Peaks"
        adata = adata[:, keep].copy()
    adata.var_names_make_unique()
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(np.asarray(adata.X, dtype=np.float32))
    else:
        adata.X = adata.X.tocsr().astype(np.float32)
    return adata


def _load_input(path, input_format):
    fmt = str(input_format).lower()
    if fmt == "h5ad":
        return _load_h5ad(path)
    if fmt == "10xh5":
        return _load_10x_h5(path)
    raise ValueError("run_scopen.py expects --input_format h5ad or 10Xh5")


def main():
    parser = argparse.ArgumentParser(description="Run scOpen and write an aligned h5ad output")
    parser.add_argument("--input", required=True, help="Aligned raw-count ATAC h5ad, or 10x h5")
    parser.add_argument("--input_format", default="h5ad")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--estimate_rank", action="store_true")
    parser.add_argument("--n_components", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--min_n_components", type=int, default=2)
    parser.add_argument("--max_n_components", type=int, default=30)
    parser.add_argument("--step_n_components", type=int, default=1)
    parser.add_argument("--init", default="nndsvd")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--nc", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--scopen_cmd", default=None, help="Kept for compatibility; not used by the direct wrapper")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[], help="Kept for compatibility")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    adata = _load_input(Path(args.input).resolve(), args.input_format)

    data = adata.X.T.tocsr()
    data.data[:] = 1.0
    tf_idf = tf_idf_transform(data=data)

    scopen_args = SimpleNamespace(
        output_dir=str(output_dir),
        output_prefix=args.output_prefix,
        alpha=args.alpha,
        max_iter=args.max_iter,
        min_n_components=args.min_n_components,
        max_n_components=args.max_n_components,
        step_n_components=args.step_n_components,
        init=args.init,
        random_state=args.random_state,
        nc=args.nc,
        verbose=args.verbose,
    )
    if args.estimate_rank:
        w_hat, h_hat = estimate_rank(data=tf_idf, args=scopen_args)
        n_components = int(w_hat.shape[1])
    else:
        res = run_nmf(
            (
                tf_idf,
                args.n_components,
                args.alpha,
                args.max_iter,
                args.verbose,
                args.random_state,
                args.init,
            )
        )
        w_hat, h_hat = res[0], res[1]
        n_components = int(args.n_components)

    imputed = np.dot(w_hat, h_hat).T.astype(np.float32, copy=False)
    out = ad.AnnData(X=imputed, obs=adata.obs.copy(), var=adata.var.copy())
    out.uns["scopen"] = {
        "input": str(Path(args.input).resolve()),
        "input_format": args.input_format,
        "estimate_rank": bool(args.estimate_rank),
        "n_components": n_components,
        "alpha": float(args.alpha),
        "max_iter": int(args.max_iter),
        "random_state": int(args.random_state),
    }
    out.write_h5ad(output)

    pd.DataFrame(w_hat, index=adata.var_names).to_csv(
        output_dir / f"{args.output_prefix}_peaks.txt",
        sep="\t",
    )
    pd.DataFrame(h_hat, columns=adata.obs_names).to_csv(
        output_dir / f"{args.output_prefix}_barcodes.txt",
        sep="\t",
    )

    summary = {
        "method": "scOpen",
        "input": str(Path(args.input).resolve()),
        "input_format": args.input_format,
        "output": str(output),
        "n_cells": int(out.n_obs),
        "n_peaks": int(out.n_vars),
        "estimate_rank": bool(args.estimate_rank),
        "n_components": n_components,
        "alpha": float(args.alpha),
        "max_iter": int(args.max_iter),
        "random_state": int(args.random_state),
    }
    with open(output.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
