import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

THIS_DIR = Path(__file__).resolve().parent
SPAGE_ROOT = THIS_DIR / "SpaGE-master"
if str(SPAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPAGE_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from SpaGE.main import SpaGE
from common import (
    DEFAULT_REF_H5AD,
    DEFAULT_VISIUM_ROOT,
    build_target_prediction_adata,
    load_reference_metadata,
    load_visium_mouse_brain,
    save_json,
    set_seed,
    summarize_target_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpaGE baseline on mouse brain snRNA + Visium with holdout genes"
    )
    parser.add_argument("--ref_h5ad", default=str(DEFAULT_REF_H5AD))
    parser.add_argument("--visium_root", default=str(DEFAULT_VISIUM_ROOT))
    parser.add_argument("--sample_ids", default="", help="comma-separated Visium sample IDs; empty means all")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--panel_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_ref_cells", type=int, default=8000)
    parser.add_argument("--max_spots", type=int, default=0)
    parser.add_argument("--in_tissue_only", action="store_true")
    parser.add_argument("--observed_gene_count", type=int, default=1024)
    parser.add_argument("--target_gene_count", type=int, default=256)
    parser.add_argument("--min_nonzero_spots", type=int, default=10)
    parser.add_argument("--min_nonzero_ref_cells", type=int, default=20)
    parser.add_argument("--n_pv", type=int, default=30)
    parser.add_argument("--save_h5ad", action="store_true")
    parser.add_argument("--save_truth", action="store_true")
    return parser.parse_args()


def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _log_cpm_dense(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log1p((X / lib) * 1e6).astype(np.float32, copy=False)


def _sample_indices(n_total: int, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or n_total <= max_items:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_total, size=int(max_items), replace=False)).astype(np.int64)


def _pick_gene_panels(
    ref_subset_counts: np.ndarray,
    spatial_counts: np.ndarray,
    ref,
    observed_gene_count: int,
    target_gene_count: int,
    min_nonzero_spots: int,
    min_nonzero_ref_cells: int,
    seed: int,
):
    ref_nonzero = (ref_subset_counts > 0).sum(axis=0)
    spatial_nonzero = (spatial_counts > 0).sum(axis=0)
    ref_var = np.var(ref_subset_counts, axis=0)
    spatial_var = np.var(spatial_counts, axis=0)

    eligible = (
        (ref_nonzero >= int(min_nonzero_ref_cells))
        & (spatial_nonzero >= int(min_nonzero_spots))
        & (ref_var > 0)
        & (spatial_var > 0)
    )
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size < int(observed_gene_count) + int(target_gene_count):
        raise ValueError(
            "Not enough eligible genes for requested observed/target panel sizes: "
            f"eligible={eligible_idx.size}, observed={observed_gene_count}, target={target_gene_count}"
        )

    rng = np.random.RandomState(seed)
    chosen = rng.choice(
        eligible_idx,
        size=int(observed_gene_count) + int(target_gene_count),
        replace=False,
    )
    chosen = np.asarray(chosen, dtype=np.int64)
    target_idx = np.sort(chosen[: int(target_gene_count)])
    observed_idx = np.sort(chosen[int(target_gene_count) :])
    return observed_idx, target_idx


def main():
    args = parse_args()
    set_seed(int(args.panel_seed))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_h5ad = Path(args.ref_h5ad).resolve()
    visium_root = Path(args.visium_root).resolve()
    ref = load_reference_metadata(ref_h5ad)
    sample_ids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]

    spatial = load_visium_mouse_brain(
        visium_root=visium_root,
        ref=ref,
        sample_ids=sample_ids or None,
        in_tissue_only=bool(args.in_tissue_only),
        max_spots=int(args.max_spots),
        sample_seed=int(args.sample_seed),
    )
    spatial_counts = _to_dense_float32(spatial.X)

    ref_adata = sc.read_h5ad(ref_h5ad, backed="r")
    ref_rows = _sample_indices(ref_adata.n_obs, int(args.max_ref_cells), int(args.sample_seed))
    ref_subset_counts = _to_dense_float32(ref_adata[ref_rows, :].X)

    observed_idx, target_idx = _pick_gene_panels(
        ref_subset_counts=ref_subset_counts,
        spatial_counts=spatial_counts,
        ref=ref,
        observed_gene_count=int(args.observed_gene_count),
        target_gene_count=int(args.target_gene_count),
        min_nonzero_spots=int(args.min_nonzero_spots),
        min_nonzero_ref_cells=int(args.min_nonzero_ref_cells),
        seed=int(args.panel_seed),
    )

    observed_ids = ref.ensembl_ids[observed_idx]
    target_ids = ref.ensembl_ids[target_idx]

    spatial_obs = _log_cpm_dense(spatial_counts[:, observed_idx])
    spatial_target_truth = _log_cpm_dense(spatial_counts[:, target_idx])
    ref_obs_target = _log_cpm_dense(ref_subset_counts[:, np.concatenate([observed_idx, target_idx])])

    spatial_df = pd.DataFrame(
        spatial_obs,
        index=spatial.obs_names.astype(str),
        columns=observed_ids,
    )
    rna_df = pd.DataFrame(
        ref_obs_target,
        index=[f"ref_{i}" for i in ref_rows.tolist()],
        columns=np.concatenate([observed_ids, target_ids]),
    )

    n_pv = int(min(args.n_pv, len(observed_ids)))
    pred_df = SpaGE(
        Spatial_data=spatial_df,
        RNA_data=rna_df,
        n_pv=n_pv,
        genes_to_predict=target_ids.tolist(),
    )
    pred_df.index = spatial_df.index
    pred_df = pred_df.loc[:, target_ids.tolist()]
    pred_np = pred_df.to_numpy(dtype=np.float32, copy=False)

    metrics = summarize_target_metrics(pred_np, spatial_target_truth)
    summary = {
        **metrics,
        "method": "SpaGE",
        "ref_h5ad": str(ref_h5ad),
        "visium_root": str(visium_root),
        "sample_ids": sample_ids if sample_ids else "all",
        "n_ref_cells_used": int(ref_rows.shape[0]),
        "n_spots_used": int(spatial.n_obs),
        "n_genes_total": int(ref.ensembl_ids.shape[0]),
        "observed_gene_count": int(observed_idx.shape[0]),
        "target_gene_count": int(target_idx.shape[0]),
        "panel_seed": int(args.panel_seed),
        "sample_seed": int(args.sample_seed),
        "min_nonzero_spots": int(args.min_nonzero_spots),
        "min_nonzero_ref_cells": int(args.min_nonzero_ref_cells),
        "n_pv": int(n_pv),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(
        output_dir / "panel_info.json",
        {
            "observed_ensembl_ids": observed_ids.tolist(),
            "target_ensembl_ids": target_ids.tolist(),
            "observed_symbols": ref.gene_symbols[observed_idx].tolist(),
            "target_symbols": ref.gene_symbols[target_idx].tolist(),
        },
    )

    pred_df.to_csv(output_dir / "predicted_target_logcpm.csv.gz", compression="gzip")
    if args.save_truth:
        truth_df = pd.DataFrame(
            spatial_target_truth,
            index=spatial.obs_names.astype(str),
            columns=target_ids,
        )
        truth_df.to_csv(output_dir / "truth_target_logcpm.csv.gz", compression="gzip")

    if args.save_h5ad:
        pred_adata = build_target_prediction_adata(
            spatial=spatial,
            ref=ref,
            target_idx=target_idx,
            pred_targets=pred_np,
            truth_targets=spatial_target_truth,
        )
        pred_adata.write_h5ad(output_dir / "spage_predicted_target_genes.h5ad")

    print(f"[done] SpaGE summary -> {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
