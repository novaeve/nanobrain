#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import rankdata

TFIDF_LOG1P_SCALE = 10000.0


def _as_dense(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _rank_normalize_rows(X):
    X = np.asarray(X, dtype=np.float32)
    out = np.empty_like(X, dtype=np.float32)
    denom = max(X.shape[1] - 1, 1)
    for i in range(X.shape[0]):
        out[i] = (rankdata(X[i], method="average") - 1.0) / denom
    return out


def _inverse_expression_transform(X, transform):
    transform = str(transform)
    if transform == "identity":
        return np.asarray(X, dtype=np.float32)
    if transform == "tfidf_log1p_1e4":
        X = np.asarray(X, dtype=np.float32)
        return (np.expm1(np.clip(X, a_min=0.0, a_max=None)) / TFIDF_LOG1P_SCALE).astype(
            np.float32,
            copy=False,
        )
    raise ValueError(f"Unsupported inverse expression transform: {transform}")


def _load_matrix(method, path, matrix_key=None):
    adata = sc.read_h5ad(path)
    if matrix_key:
        key = str(matrix_key)
        if key == "X":
            return _as_dense(adata.X), adata
        if key.startswith("obsm:"):
            obsm_key = key.split(":", 1)[1]
            if obsm_key not in adata.obsm:
                raise KeyError(f"{path} does not contain obsm['{obsm_key}']")
            return _as_dense(adata.obsm[obsm_key]), adata
        if key.startswith("layer:"):
            layer_key = key.split(":", 1)[1]
            if layer_key not in adata.layers:
                raise KeyError(f"{path} does not contain layers['{layer_key}']")
            return _as_dense(adata.layers[layer_key]), adata
        raise ValueError("--matrix_key must be X, obsm:<key>, or layer:<key>")

    key = str(adata.uns.get("benchmark_matrix_key", ""))
    if key.startswith("obsm:"):
        return np.asarray(adata.obsm[key.split(":", 1)[1]], dtype=np.float32), adata
    if method.lower() == "scale" and "impute" in adata.obsm:
        return np.asarray(adata.obsm["impute"], dtype=np.float32), adata
    if method.lower() == "scale" and "imputed" in adata.obsm:
        return np.asarray(adata.obsm["imputed"], dtype=np.float32), adata
    if method.lower() == "scale" and "impute" in adata.layers:
        return _as_dense(adata.layers["impute"]), adata
    if method.lower() == "scale" and "imputed" in adata.layers:
        return _as_dense(adata.layers["imputed"]), adata
    return _as_dense(adata.X), adata


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark method output to evaluation h5ad")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "scUniFlow",
            "SCALE",
            "scOpen",
            "MAGIC",
            "SAVER",
            "scImpute",
            "scBFA",
            "cisTopic",
            "saver",
            "scimpute",
            "scbfa",
            "cistopic",
            "raw",
        ],
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference_h5ad", required=True, help="TF-IDF h5ad whose obs/var/order define evaluation space")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--matrix_key",
        default=None,
        help="Override matrix source: X, obsm:<key>, or layer:<key>.",
    )
    parser.add_argument(
        "--inverse_expr_transform",
        default="identity",
        choices=["identity", "tfidf_log1p_1e4"],
        help="Invert method output into the reference TF-IDF space before evaluation.",
    )
    parser.add_argument("--rank_normalize", action="store_true")
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    ref = sc.read_h5ad(args.reference_h5ad)
    X, src = _load_matrix(args.method, args.input, matrix_key=args.matrix_key)

    if X.shape[0] != src.n_obs:
        raise ValueError(f"matrix rows {X.shape[0]} do not match source cells {src.n_obs}")
    if X.shape[1] != src.n_vars:
        raise ValueError(
            f"matrix columns {X.shape[1]} do not match source peaks {src.n_vars}; "
            "cannot convert dimensionality-reduced output for biological eval"
        )

    ref_obs_names = ref.obs_names.astype(str)
    src_obs_names = src.obs_names.astype(str)
    ref_var_names = ref.var_names.astype(str)
    src_var_names = src.var_names.astype(str)

    common_obs = np.intersect1d(ref_obs_names, src_obs_names)
    common_var = np.intersect1d(ref_var_names, src_var_names)
    if common_obs.size == 0 or common_var.size == 0:
        raise ValueError("reference and method output cannot be aligned")

    obs_pos_src_all = src_obs_names.get_indexer(ref_obs_names)
    keep_ref_obs = obs_pos_src_all >= 0
    obs_pos_src = obs_pos_src_all[keep_ref_obs]
    if obs_pos_src.size == 0:
        raise ValueError("method output has no cells overlapping the reference")

    var_pos_src = src_var_names.get_indexer(ref_var_names)
    if np.any(var_pos_src < 0):
        missing_var = int(np.sum(var_pos_src < 0))
        raise ValueError(f"method output missing reference peaks: var={missing_var}")

    X = X[np.asarray(obs_pos_src), :][:, np.asarray(var_pos_src)].astype(np.float32, copy=False)
    X = _inverse_expression_transform(X, args.inverse_expr_transform)
    if args.rank_normalize:
        X = _rank_normalize_rows(X)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    out = ad.AnnData(X=X, obs=ref.obs.iloc[np.where(keep_ref_obs)[0]].copy(), var=ref.var.copy())
    out.uns["benchmark_conversion"] = {
        "method": args.method,
        "input": str(Path(args.input).resolve()),
        "matrix_key": args.matrix_key,
        "reference_h5ad": str(Path(args.reference_h5ad).resolve()),
        "inverse_expr_transform": str(args.inverse_expr_transform),
        "rank_normalize": bool(args.rank_normalize),
        "allow_obs_subset": True,
        "n_reference_cells": int(ref.n_obs),
        "n_overlap_cells": int(obs_pos_src.size),
    }
    out.write_h5ad(output)

    summary = {
        **out.uns["benchmark_conversion"],
        "output": str(output),
        "n_cells": int(out.n_obs),
        "n_peaks": int(out.n_vars),
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
