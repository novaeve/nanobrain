import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from benchmark_common import prepare_benchmark_data
from common import (
    DEFAULT_REF_H5AD,
    DEFAULT_VISIUM_ROOT,
    SpatialGeneImputationDataset,
    build_global_target_mask,
    build_target_prediction_adata,
    compute_empirical_prior_stats,
    load_model_from_checkpoint,
    load_reference_metadata,
    load_visium_mouse_brain,
    save_json,
    set_seed,
    summarize_target_metrics,
    transform_counts_array,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpaGE-style spatial unmeasured-gene prediction with scUniFlow"
    )
    parser.add_argument("--exp_dir", required=True, help="scUniFlow training run directory")
    parser.add_argument("--checkpoint", default="", help="checkpoint path; default <exp_dir>/checkpoints/checkpoint_best.pt")
    parser.add_argument("--ref_h5ad", default=str(DEFAULT_REF_H5AD))
    parser.add_argument("--visium_root", default=str(DEFAULT_VISIUM_ROOT))
    parser.add_argument("--sample_ids", default="", help="comma-separated Visium sample IDs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.01, help="kept for API consistency; routing is deterministic")
    parser.add_argument(
        "--expression_space",
        choices=["log1p", "logcpm"],
        default="log1p",
        help="Expression space used for model input and evaluation targets",
    )
    parser.add_argument("--panel_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_spots", type=int, default=0)
    parser.add_argument("--in_tissue_only", action="store_true")
    parser.add_argument(
        "--observed_gene_count",
        type=int,
        default=0,
        help="If >0, use benchmark-style gene panels and expose only this many observed genes",
    )
    parser.add_argument("--target_gene_count", type=int, default=1024)
    parser.add_argument("--target_holdout_ratio", type=float, default=0.1)
    parser.add_argument("--min_nonzero_spots", type=int, default=10)
    parser.add_argument("--max_ref_cells", type=int, default=8000)
    parser.add_argument("--min_nonzero_ref_cells", type=int, default=20)
    parser.add_argument("--target_genes", default="", help="comma-separated target genes (symbols or Ensembl IDs)")
    parser.add_argument("--target_gene_file", default="", help="one target gene token per line")
    parser.add_argument("--prior_mean", type=float, default=None)
    parser.add_argument("--prior_std", type=float, default=None)
    parser.add_argument("--save_h5ad", action="store_true")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def _read_target_tokens(args) -> list[str]:
    tokens = []
    if args.target_genes:
        tokens.extend(x.strip() for x in args.target_genes.split(","))
    if args.target_gene_file:
        with Path(args.target_gene_file).open() as f:
            tokens.extend(line.strip() for line in f)
    return [x for x in tokens if x]


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(int(args.panel_seed))

    exp_dir = Path(args.exp_dir).resolve()
    cfg_path = exp_dir / "config.yaml"
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (exp_dir / "checkpoints/checkpoint_best.pt").resolve()
    )
    if not checkpoint_path.exists():
        fallback = exp_dir / "checkpoints/checkpoint_last.pt"
        if fallback.exists():
            checkpoint_path = fallback.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    sample_ids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
    target_tokens = _read_target_tokens(args)
    visible_gene_idx = None
    panel_mode = "full_genes_except_target_visible"
    expression_space = str(args.expression_space)
    if int(args.observed_gene_count) > 0:
        if target_tokens:
            raise ValueError(
                "--target_genes/--target_gene_file are not supported together with "
                "--observed_gene_count benchmark panel mode"
            )
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
        spatial = benchmark.spatial
        target_idx = benchmark.target_idx
        observed_idx = benchmark.observed_idx
        target_mask_np = np.zeros(ref.ensembl_ids.shape[0], dtype=np.float32)
        target_mask_np[target_idx] = 1.0
        observed_mask_np = np.zeros(ref.ensembl_ids.shape[0], dtype=bool)
        observed_mask_np[observed_idx] = True
        visible_gene_idx = observed_idx.astype(np.int64, copy=False)
        panel_mode = "benchmark_observed_only_zero_fill"
    else:
        ref = load_reference_metadata(Path(args.ref_h5ad).resolve())
        spatial = load_visium_mouse_brain(
            visium_root=Path(args.visium_root).resolve(),
            ref=ref,
            sample_ids=sample_ids or None,
            in_tissue_only=bool(args.in_tissue_only),
            max_spots=int(args.max_spots),
            sample_seed=int(args.sample_seed),
        )
        target_mask_np, observed_mask_np, target_idx = build_global_target_mask(
            spatial=spatial,
            ref=ref,
            target_gene_count=int(args.target_gene_count),
            target_holdout_ratio=float(args.target_holdout_ratio),
            min_nonzero_spots=int(args.min_nonzero_spots),
            seed=int(args.panel_seed),
            target_genes=target_tokens or None,
        )

    dataset = SpatialGeneImputationDataset(
        spatial.X,
        visible_gene_idx=visible_gene_idx,
        expression_space=expression_space,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, cfg, ckpt = load_model_from_checkpoint(cfg_path, checkpoint_path, device=device)
    if int(cfg["model"]["n_genes"]) != int(ref.ensembl_ids.shape[0]):
        raise ValueError(
            f"n_genes mismatch between config ({cfg['model']['n_genes']}) and reference ({ref.ensembl_ids.shape[0]})"
        )

    prior_mean = args.prior_mean
    prior_std = args.prior_std
    auto_prior = None
    if expression_space == "logcpm" and (prior_mean is None or prior_std is None):
        auto_prior = compute_empirical_prior_stats(
            spatial.X,
            expression_space=expression_space,
        )
    if prior_mean is None:
        if expression_space == "logcpm":
            prior_mean, prior_std_auto = auto_prior
            if prior_std is None:
                prior_std = prior_std_auto
        else:
            prior_mean = ckpt.get("prior_mean", 0.0)
    if prior_std is None:
        if expression_space == "logcpm":
            _, prior_std = auto_prior
        else:
            prior_std = ckpt.get("prior_std", 1.0)
    prior_mean = float(0.0 if prior_mean is None else prior_mean)
    prior_std = float(max(1e-6, 1.0 if prior_std is None else prior_std))

    gene_ids = torch.from_numpy(ref.gene_ids.astype(np.int64))
    mask_template = torch.from_numpy(target_mask_np.astype(np.float32))
    if sp.issparse(spatial.X):
        truth_targets_np = transform_counts_array(
            spatial.X[:, target_idx].toarray().astype(np.float32, copy=False),
            expression_space=expression_space,
        )
    else:
        truth_targets_np = transform_counts_array(
            np.asarray(spatial.X[:, target_idx], dtype=np.float32),
            expression_space=expression_space,
        )

    pred_targets = []
    batch_iter = loader if args.no_progress else tqdm(loader, desc="spatial impute", unit="batch")
    for x_full, _batch_indices in batch_iter:
        x_full = x_full.to(device, non_blocking=True)
        bs = x_full.shape[0]
        gene_ids_b = gene_ids.unsqueeze(0).expand(bs, -1).to(device, non_blocking=True)
        mask_b = mask_template.unsqueeze(0).expand(bs, -1).to(device, non_blocking=True)
        pred_full = model.inference(
            gene_ids=gene_ids_b,
            known_genes=x_full,
            mask_indicator=mask_b,
            n_steps=int(args.n_steps),
            temperature=float(args.temperature),
            prior_mean=float(prior_mean),
            prior_std=float(prior_std),
        )
        pred_targets.append(pred_full[:, target_idx].detach().float().cpu().numpy())

    pred_targets_np = np.concatenate(pred_targets, axis=0)
    metrics = summarize_target_metrics(pred_targets_np, truth_targets_np)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = {
        **metrics,
        "exp_dir": str(exp_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_expression_space": str(ckpt.get("expression_transform", "log1p")),
        "config": str(cfg_path),
        "ref_h5ad": str(Path(args.ref_h5ad).resolve()),
        "visium_root": str(Path(args.visium_root).resolve()),
        "sample_ids": sample_ids if sample_ids else "all",
        "n_spots": int(spatial.n_obs),
        "n_genes_total": int(spatial.n_vars),
        "n_observed_genes": int(observed_mask_np.sum()),
        "n_target_genes": int(target_idx.shape[0]),
        "panel_seed": int(args.panel_seed),
        "sample_seed": int(args.sample_seed),
        "max_spots": int(args.max_spots),
        "max_ref_cells": int(args.max_ref_cells),
        "in_tissue_only": bool(args.in_tissue_only),
        "expression_space": expression_space,
        "evaluation_mode": panel_mode,
        "zero_filled_non_observed": bool(visible_gene_idx is not None),
        "target_gene_count_arg": int(args.target_gene_count),
        "observed_gene_count_arg": int(args.observed_gene_count),
        "target_holdout_ratio_arg": float(args.target_holdout_ratio),
        "min_nonzero_spots": int(args.min_nonzero_spots),
        "min_nonzero_ref_cells": int(args.min_nonzero_ref_cells),
        "n_steps": int(args.n_steps),
        "prior_mean": float(prior_mean),
        "prior_std": float(prior_std),
    }
    save_json(output_dir / "summary.json", run_summary)
    panel_info = {
        "target_ensembl_ids": ref.ensembl_ids[target_idx].tolist(),
        "target_symbols": ref.gene_symbols[target_idx].tolist(),
        "observed_gene_count": int(observed_mask_np.sum()),
        "target_gene_count": int(target_idx.shape[0]),
    }
    if visible_gene_idx is not None:
        observed_idx = np.flatnonzero(observed_mask_np)
        panel_info["observed_ensembl_ids"] = ref.ensembl_ids[observed_idx].tolist()
        panel_info["observed_symbols"] = ref.gene_symbols[observed_idx].tolist()
    save_json(output_dir / "panel_info.json", panel_info)

    if args.save_arrays:
        suffix = "logcpm" if expression_space == "logcpm" else "log1p"
        np.save(output_dir / f"pred_target_{suffix}.npy", pred_targets_np.astype(np.float32))
        np.save(output_dir / f"truth_target_{suffix}.npy", truth_targets_np.astype(np.float32))

    if args.save_h5ad:
        pred_adata = build_target_prediction_adata(
            spatial=spatial,
            ref=ref,
            target_idx=target_idx,
            pred_targets=pred_targets_np,
            truth_targets=truth_targets_np,
            expression_space=expression_space,
        )
        pred_adata.write_h5ad(output_dir / "predicted_target_genes.h5ad")

    print(f"[done] summary -> {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
