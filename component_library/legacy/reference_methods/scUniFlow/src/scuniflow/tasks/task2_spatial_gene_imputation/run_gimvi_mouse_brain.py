import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import anndata as ad
import numpy as np
import scvi

from benchmark_common import prepare_benchmark_data, write_benchmark_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gimVI baseline on mouse brain snRNA + Visium with holdout genes"
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
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--n_latent", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_h5ad", action="store_true")
    parser.add_argument("--save_truth", action="store_true")
    return parser.parse_args()


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

    observed_idx = benchmark.observed_idx
    target_idx = benchmark.target_idx
    ref = benchmark.ref

    ref_counts = benchmark.ref_subset_counts[:, np.concatenate([observed_idx, target_idx])].copy()
    spatial_counts = benchmark.spatial_counts[:, observed_idx].copy()

    adata_seq = ad.AnnData(
        X=ref_counts,
        obs={"modality": ["seq"] * ref_counts.shape[0]},
        var={"gene_id": ref.ensembl_ids[np.concatenate([observed_idx, target_idx])]},
    )
    adata_spatial = ad.AnnData(
        X=spatial_counts,
        obs={"modality": ["spatial"] * spatial_counts.shape[0]},
        var={"gene_id": ref.ensembl_ids[observed_idx]},
    )

    scvi.external.GIMVI.setup_anndata(adata_seq)
    scvi.external.GIMVI.setup_anndata(adata_spatial)

    model = scvi.external.GIMVI(
        adata_seq=adata_seq,
        adata_spatial=adata_spatial,
        generative_distributions=["zinb", "nb"],
        model_library_size=[True, False],
        n_latent=int(args.n_latent),
    )
    model.train(
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        accelerator=args.device,
    )

    imputed = model.get_imputed_values(
        [adata_seq, adata_spatial],
        deterministic=True,
        normalized=True,
        batch_size=int(args.batch_size),
    )
    spatial_imputed = np.asarray(imputed[1], dtype=np.float32)
    pred_target = spatial_imputed[:, observed_idx.shape[0] :]

    write_benchmark_outputs(
        output_dir=args.output_dir,
        method="gimVI",
        benchmark=benchmark,
        pred_targets_logcpm=pred_target,
        extra_summary={
            "sample_ids": sample_ids if sample_ids else "all",
            "panel_seed": int(args.panel_seed),
            "sample_seed": int(args.sample_seed),
            "min_nonzero_spots": int(args.min_nonzero_spots),
            "min_nonzero_ref_cells": int(args.min_nonzero_ref_cells),
            "max_epochs": int(args.max_epochs),
            "n_latent": int(args.n_latent),
            "batch_size": int(args.batch_size),
        },
        save_h5ad=bool(args.save_h5ad),
        save_truth=bool(args.save_truth),
    )
    print(f"[done] gimVI summary -> {args.output_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
