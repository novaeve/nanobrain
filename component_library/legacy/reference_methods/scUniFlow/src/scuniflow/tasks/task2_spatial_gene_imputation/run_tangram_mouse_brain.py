import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp

THIS_DIR = Path(__file__).resolve().parent
TANGRAM_ROOT = THIS_DIR / "Tangram"
if str(TANGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(TANGRAM_ROOT))

import tangram as tg

from benchmark_common import (
    log_cpm_dense,
    prepare_benchmark_data,
    write_benchmark_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Tangram baseline on mouse brain snRNA + Visium holdout benchmark"
    )
    parser.add_argument("--ref_h5ad", default="data/all_cells_20200625.h5ad")
    parser.add_argument("--visium_root", default="data/mouse_brain_visium/mouse_brain_visium_wo_cloupe_data/rawdata")
    parser.add_argument("--sample_ids", default="")
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--save_h5ad", action="store_true")
    parser.add_argument("--save_truth", action="store_true")
    return parser.parse_args()


def _make_csr(X: np.ndarray):
    if sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(X, dtype=np.float32))


def main():
    args = parse_args()
    sample_ids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
    benchmark = prepare_benchmark_data(
        ref_h5ad=args.ref_h5ad,
        visium_root=args.visium_root,
        sample_ids=sample_ids or None,
        panel_seed=int(args.panel_seed),
        sample_seed=int(args.sample_seed),
        max_ref_cells=int(args.max_ref_cells),
        max_spots=int(args.max_spots),
        in_tissue_only=bool(args.in_tissue_only),
        observed_gene_count=int(args.observed_gene_count),
        target_gene_count=int(args.target_gene_count),
        min_nonzero_spots=int(args.min_nonzero_spots),
        min_nonzero_ref_cells=int(args.min_nonzero_ref_cells),
    )

    ref = benchmark.ref
    observed_idx = benchmark.observed_idx
    target_idx = benchmark.target_idx

    ref_full_logcpm = log_cpm_dense(benchmark.ref_subset_counts)
    spatial_obs_logcpm = log_cpm_dense(benchmark.spatial_counts[:, observed_idx])

    ad_sc = ad.AnnData(
        X=_make_csr(ref_full_logcpm),
        obs=pd.DataFrame(index=[f"ref_{i}" for i in benchmark.ref_rows.tolist()]),
        var=pd.DataFrame(index=ref.ensembl_ids.copy()),
    )
    ad_sp_map = ad.AnnData(
        X=_make_csr(spatial_obs_logcpm),
        obs=benchmark.spatial.obs.copy(),
        var=pd.DataFrame(index=ref.ensembl_ids[observed_idx].copy()),
    )
    # Avoid Tangram pulling in squidpy neighborhood preprocessing.
    if "spatial" in ad_sp_map.obsm:
        del ad_sp_map.obsm["spatial"]

    training_genes = ref.ensembl_ids[observed_idx].tolist()
    tg.pp_adatas(ad_sc, ad_sp_map, genes=training_genes, gene_to_lowercase=False)
    ad_map = tg.map_cells_to_space(
        adata_sc=ad_sc,
        adata_sp=ad_sp_map,
        device=args.device,
        num_epochs=int(args.num_epochs),
        learning_rate=float(args.learning_rate),
        mode="cells",
        density_prior="rna_count_based",
    )
    ad_ge = tg.project_genes(ad_map, ad_sc)

    target_ids = ref.ensembl_ids[target_idx].tolist()
    target_ids_lower = [x.lower() for x in target_ids]
    pred_target = np.asarray(ad_ge[:, target_ids_lower].X, dtype=np.float32)

    extra_summary = {
        "sample_ids": sample_ids if sample_ids else "all",
        "panel_seed": int(args.panel_seed),
        "sample_seed": int(args.sample_seed),
        "min_nonzero_spots": int(args.min_nonzero_spots),
        "min_nonzero_ref_cells": int(args.min_nonzero_ref_cells),
        "num_epochs": int(args.num_epochs),
        "learning_rate": float(args.learning_rate),
        "device": str(args.device),
    }
    train_hist = ad_map.uns.get("training_history", {})
    if "main_loss" in train_hist and len(train_hist["main_loss"]) > 0:
        extra_summary["tangram_train_score"] = float(train_hist["main_loss"][-1])

    write_benchmark_outputs(
        output_dir=args.output_dir,
        method="Tangram",
        benchmark=benchmark,
        pred_targets_logcpm=pred_target,
        extra_summary=extra_summary,
        save_h5ad=bool(args.save_h5ad),
        save_truth=bool(args.save_truth),
    )

    print(f"[done] Tangram summary -> {Path(args.output_dir).resolve() / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
