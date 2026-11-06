import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from common import (
    DownsamplePairDataset,
    ScUniFlowReadDepthEnhancer,
    load_counts_and_split,
    save_json,
    set_seed,
    summarize_reconstruction,
)


_SHARED_COMMON_PATH = Path(__file__).resolve().parents[1] / "read_depth_enhancement" / "common.py"
_SHARED_SPEC = importlib.util.spec_from_file_location("shared_rde_common", _SHARED_COMMON_PATH)
_SHARED_COMMON = importlib.util.module_from_spec(_SHARED_SPEC)
_SHARED_SPEC.loader.exec_module(_SHARED_COMMON)
derive_reference_labels = _SHARED_COMMON.derive_reference_labels
evaluate_clustering_metrics = _SHARED_COMMON.evaluate_clustering_metrics
materialize_downsampled_full_gene_counts = _SHARED_COMMON.materialize_downsampled_full_gene_counts


def normalize_log1p_counts(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    return np.log1p(counts / (totals + 1e-6) * float(target_sum)).astype(np.float32, copy=False)


def select_hvg_indices(counts: np.ndarray, n_top_genes: int, seed: int) -> np.ndarray:
    n_top_genes = int(n_top_genes)
    if n_top_genes <= 0 or n_top_genes >= counts.shape[1]:
        return np.arange(counts.shape[1], dtype=np.int64)
    adata = sc.AnnData(np.asarray(counts, dtype=np.float32).copy())
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=False)
        idx = np.where(np.asarray(adata.var["highly_variable"], dtype=bool))[0]
        if idx.size > 0:
            return np.asarray(idx[:n_top_genes], dtype=np.int64)
    except Exception as exc:
        print(f"[warn] scanpy HVG selection failed, falling back to variance: {exc}", flush=True)

    rng = np.random.RandomState(int(seed))
    var = np.asarray(adata.X).var(axis=0)
    jitter = rng.uniform(0.0, 1e-12, size=var.shape[0])
    return np.argsort(-(var + jitter))[:n_top_genes].astype(np.int64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate scUniFlow read depth enhancement with scFoundation-style clustering metrics"
    )
    parser.add_argument("--exp_dir", required=True, help="base scUniFlow experiment directory")
    parser.add_argument("--task_ckpt", required=True, help="task fine-tuned checkpoint")
    parser.add_argument("--data", required=True, help="dataset key or h5ad path")
    parser.add_argument("--output_dir", required=True, help="evaluation output directory")
    parser.add_argument("--config", default="", help="optional config yaml; defaults to <exp_dir>/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--cell_indices_path", default="", help="optional .npy file with exact original h5ad cell indices")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prediction_chunk_size", type=int, default=4096)
    parser.add_argument("--downsample_fold", type=float, default=-1.0, help="negative means reuse task checkpoint setting")
    parser.add_argument("--label_col", default="", help="obs column used as target labels")
    parser.add_argument("--target_labels_path", default="", help="optional .npy labels aligned to evaluated cells")
    parser.add_argument("--ref_resolution", type=float, default=0.4, help="used when label_col is empty")
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument(
        "--enhance_gene_mode",
        choices=["all", "hvg"],
        default="all",
        help="all uses the full scUniFlow prediction; hvg writes predictions only on selected HVGs and keeps non-HVG low-depth",
    )
    parser.add_argument("--enhance_n_top_genes", type=int, default=2000)
    parser.add_argument("--enhance_gene_indices_path", default="", help="optional .npy gene indices to enhance")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_arrays", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def run_prediction(model, loader, device, chunk_size: int, max_batches: int = 0):
    model.eval()
    pred_logs = []
    low_logs = []
    high_logs = []
    cell_indices = []
    n_batches = 0

    for batch in tqdm(loader, desc="eval"):
        gene_ids, x_low, x_high, log_total_low, log_total_high, log_fold, cell_idx = batch
        gene_ids = gene_ids.to(device, non_blocking=True)
        x_low = x_low.to(device, non_blocking=True)
        x_high = x_high.to(device, non_blocking=True)
        log_total_low = log_total_low.to(device, non_blocking=True)
        log_fold = log_fold.to(device, non_blocking=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(
                gene_ids,
                x_low,
                log_total_low=log_total_low,
                log_fold=log_fold,
                chunk_size=chunk_size,
            )

        pred_logs.append(pred.cpu().float().numpy())
        low_logs.append(x_low.cpu().float().numpy())
        high_logs.append(x_high.cpu().float().numpy())
        cell_indices.append(cell_idx.numpy())

        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    pred_log = np.concatenate(pred_logs, axis=0)
    low_log = np.concatenate(low_logs, axis=0)
    high_log = np.concatenate(high_logs, axis=0)
    cell_idx = np.concatenate(cell_indices, axis=0).astype(np.int64)
    return pred_log, low_log, high_log, cell_idx, n_batches


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.exp_dir).resolve()
    cfg_path = Path(args.config).resolve() if args.config else (exp_dir / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    split_seed = int(train_cfg.get("seed", 42))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    cell_seed = int(train_cfg.get("cell_seed", split_seed))
    max_cells = train_cfg.get("max_cells")

    task_ckpt = torch.load(args.task_ckpt, map_location="cpu")
    task_args = task_ckpt.get("train_args", {})
    downsample_fold = float(args.downsample_fold if args.downsample_fold > 0 else task_args.get("downsample_fold", 4.0))

    counts, gene_ids, gene_names, meta, split = load_counts_and_split(
        args.data,
        val_ratio=val_ratio,
        seed=split_seed,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )
    if args.cell_indices_path:
        eval_idx = np.load(args.cell_indices_path).astype(np.int64)
    elif args.split == "train":
        eval_idx = split.train_idx
    elif args.split == "val":
        eval_idx = split.val_idx
    else:
        eval_idx = np.concatenate([split.train_idx, split.val_idx], axis=0)

    expression_space = str(task_args.get("expression_space", "raw_log1p"))
    head_type = str(task_args.get("head_type", "gene_residual"))
    dataset = DownsamplePairDataset(
        counts,
        gene_ids,
        eval_idx,
        downsample_fold=downsample_fold,
        seed=split_seed,
        expression_space=expression_space,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = ScUniFlowReadDepthEnhancer(
        n_genes=int(model_cfg["n_genes"]),
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        head_hidden_dim=int(task_args.get("head_hidden_dim", 256)),
        head_dropout=float(task_args.get("head_dropout", 0.1)),
        head_type=head_type,
    )
    missing, unexpected = model.load_state_dict(task_ckpt["model_state_dict"], strict=False)
    allowed_missing_prefixes = ("backbone.vector_field.binary_mlp.",)
    bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint mismatch while loading scUniFlow RDE head: "
            f"missing={bad_missing}, unexpected={unexpected}"
        )
    if missing:
        print(
            "[load] ignored missing keys from newer vector_field module not used by RDE eval: "
            + ", ".join(missing),
            flush=True,
        )
    model = model.to(device)

    t0 = time.perf_counter()
    pred_log, low_log, high_log, cell_idx, used_batches = run_prediction(
        model,
        loader,
        device,
        chunk_size=args.prediction_chunk_size,
        max_batches=int(args.max_eval_batches),
    )
    print(f"[timing] prediction={time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    recon_metrics, _, _ = summarize_reconstruction(pred_log, high_log)
    print(f"[timing] reconstruction_metrics={time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    if expression_space == "normalized_log1p":
        low_counts, high_counts = materialize_downsampled_full_gene_counts(
            counts,
            cell_idx,
            downsample_fold=downsample_fold,
            seed=split_seed,
        )
        pred_norm = np.expm1(pred_log).clip(min=0.0)
        low_totals = low_counts.sum(axis=1, keepdims=True)
        est_high_totals = np.maximum(low_totals * float(downsample_fold), 1.0)
        pred_counts = (pred_norm / 1e4) * est_high_totals
    else:
        pred_counts = np.expm1(pred_log).clip(min=0.0)
        low_counts = np.expm1(low_log).clip(min=0.0)
        high_counts = np.expm1(high_log).clip(min=0.0)
    if args.enhance_gene_indices_path:
        enhance_gene_idx = np.load(args.enhance_gene_indices_path).astype(np.int64)
        enhance_gene_idx = np.unique(enhance_gene_idx)
    elif args.enhance_gene_mode == "hvg":
        t_hvg = time.perf_counter()
        enhance_gene_idx = select_hvg_indices(
            low_counts,
            n_top_genes=int(args.enhance_n_top_genes),
            seed=int(args.seed),
        )
        print(f"[timing] select_enhance_genes={time.perf_counter() - t_hvg:.2f}s", flush=True)
    else:
        enhance_gene_idx = np.arange(pred_counts.shape[1], dtype=np.int64)

    if args.enhance_gene_mode == "hvg" or args.enhance_gene_indices_path:
        hybrid_counts = np.asarray(low_counts, dtype=np.float32).copy()
        hybrid_counts[:, enhance_gene_idx] = pred_counts[:, enhance_gene_idx].astype(np.float32, copy=False)
        pred_counts = np.clip(hybrid_counts, 0.0, None)
        # Recompute reconstruction on the same normalized-log full-gene space used by
        # classical HVG-only baselines.
        recon_metrics, _, _ = summarize_reconstruction(
            normalize_log1p_counts(pred_counts),
            normalize_log1p_counts(high_counts),
        )
    print(f"[timing] build_count_matrices={time.perf_counter() - t0:.2f}s", flush=True)

    obs = meta["obs"].iloc[cell_idx].copy()
    if args.target_labels_path:
        target_labels = np.load(args.target_labels_path, allow_pickle=True).astype(str)
        if target_labels.shape[0] != pred_counts.shape[0]:
            raise ValueError(
                f"target_labels length mismatch: labels={target_labels.shape[0]} eval_cells={pred_counts.shape[0]}"
            )
        n_clusters = len(np.unique(target_labels))
        ref_info = {
            "target_mode": "shared_labels",
            "target_labels_path": str(Path(args.target_labels_path).resolve()),
            "n_clusters": int(n_clusters),
        }
    elif args.label_col:
        if args.label_col not in obs.columns:
            raise KeyError(f"label_col not found in obs: {args.label_col}")
        target_labels = obs[args.label_col].astype(str).to_numpy()
        n_clusters = len(np.unique(target_labels))
        ref_info = {"target_mode": "obs_label", "label_col": args.label_col}
    else:
        t0 = time.perf_counter()
        target_labels, ref_adata = derive_reference_labels(
            high_counts,
            ref_resolution=float(args.ref_resolution),
            n_top_genes=int(args.n_top_genes),
        )
        print(f"[timing] reference_labels={time.perf_counter() - t0:.2f}s", flush=True)
        n_clusters = len(np.unique(target_labels))
        ref_info = {
            "target_mode": "reference_leiden",
            "ref_resolution": float(args.ref_resolution),
            "n_clusters": int(n_clusters),
        }

    t0 = time.perf_counter()
    enhanced_metrics, _, enhanced_clusters = evaluate_clustering_metrics(
        pred_counts,
        target_labels=target_labels,
        n_clusters=n_clusters,
        n_top_genes=int(args.n_top_genes),
    )
    print(f"[timing] enhanced_clustering={time.perf_counter() - t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    low_metrics, _, low_clusters = evaluate_clustering_metrics(
        low_counts,
        target_labels=target_labels,
        n_clusters=n_clusters,
        n_top_genes=int(args.n_top_genes),
    )
    print(f"[timing] low_clustering={time.perf_counter() - t0:.2f}s", flush=True)

    summary = {
        "exp_dir": str(exp_dir),
        "config": str(cfg_path),
        "task_ckpt": str(Path(args.task_ckpt).resolve()),
        "data": str(args.data),
        "split": args.split,
        "cell_indices_path": str(Path(args.cell_indices_path).resolve()) if args.cell_indices_path else "",
        "target_labels_path": str(Path(args.target_labels_path).resolve()) if args.target_labels_path else "",
        "n_eval_cells": int(pred_counts.shape[0]),
        "n_genes": int(pred_counts.shape[1]),
        "downsample_fold": float(downsample_fold),
        "head_type": head_type,
        "expression_space": expression_space,
        "enhance_gene_mode": args.enhance_gene_mode,
        "enhance_n_genes": int(enhance_gene_idx.shape[0]),
        "enhance_n_top_genes": int(args.enhance_n_top_genes),
        "enhance_gene_indices_path": str(Path(args.enhance_gene_indices_path).resolve()) if args.enhance_gene_indices_path else "",
        "used_batches": int(used_batches),
        "max_eval_batches": int(args.max_eval_batches),
        "reconstruction": recon_metrics,
        "reference": ref_info,
        "low_depth_baseline": low_metrics,
        "enhanced": enhanced_metrics,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "run_args.json", vars(args))

    np.save(output_dir / "cell_indices.npy", cell_idx)
    np.save(output_dir / "target_labels.npy", np.asarray(target_labels))
    np.save(output_dir / "enhanced_clusters.npy", np.asarray(enhanced_clusters))
    np.save(output_dir / "low_clusters.npy", np.asarray(low_clusters))
    np.save(output_dir / "enhance_gene_indices.npy", enhance_gene_idx)
    if args.save_arrays:
        np.save(output_dir / "pred_counts.npy", pred_counts)
        np.save(output_dir / "low_counts.npy", low_counts)
        np.save(output_dir / "high_counts.npy", high_counts)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
