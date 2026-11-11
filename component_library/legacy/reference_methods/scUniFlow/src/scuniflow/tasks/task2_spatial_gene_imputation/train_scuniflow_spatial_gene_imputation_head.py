import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from benchmark_common import prepare_benchmark_data
from common import DEFAULT_REF_H5AD, DEFAULT_VISIUM_ROOT, summarize_target_metrics
from spatial_head_common import (
    ScUniFlowSpatialImputer,
    SpatialSpotTargetDataset,
    build_spot_spatial_block_split,
    build_spot_train_val_split,
    normalize_spatial_coords,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a task head for scUniFlow spatial gene imputation"
    )
    parser.add_argument("--exp_dir", required=True, help="base scUniFlow experiment directory")
    parser.add_argument("--checkpoint", default="", help="backbone checkpoint; default <exp_dir>/checkpoints/checkpoint_best.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="", help="override config yaml; default <exp_dir>/config.yaml")
    parser.add_argument("--ref_h5ad", default=str(DEFAULT_REF_H5AD))
    parser.add_argument("--visium_root", default=str(DEFAULT_VISIUM_ROOT))
    parser.add_argument("--sample_ids", default="ST8059048")
    parser.add_argument("--in_tissue_only", action="store_true")
    parser.add_argument("--max_spots", type=int, default=0)
    parser.add_argument("--max_ref_cells", type=int, default=8000)
    parser.add_argument("--observed_gene_count", type=int, default=1024)
    parser.add_argument("--target_gene_count", type=int, default=256)
    parser.add_argument("--min_nonzero_spots", type=int, default=10)
    parser.add_argument("--min_nonzero_ref_cells", type=int, default=20)
    parser.add_argument("--expression_space", choices=["log1p", "logcpm"], default="logcpm")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_mode", choices=["random_spots", "spatial_block"], default="random_spots")
    parser.add_argument(
        "--block_corner",
        choices=["auto", "lower_left", "lower_right", "upper_left", "upper_right"],
        default="auto",
    )
    parser.add_argument("--panel_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--loss_weight_mse", type=float, default=1.0)
    parser.add_argument("--loss_weight_pearson_all", type=float, default=0.1)
    parser.add_argument("--loss_weight_spot_pearson", type=float, default=0.1)
    parser.add_argument("--loss_weight_gene_pearson", type=float, default=0.0)
    parser.add_argument("--train_mamba", action="store_true")
    parser.add_argument("--train_router", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sample_ids_arg(arg: str) -> list[str]:
    return [x.strip() for x in str(arg).split(",") if x.strip()]


def _pearson_loss_dim(pred: torch.Tensor, target: torch.Tensor, dim: int):
    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    target_centered = target - target.mean(dim=dim, keepdim=True)
    pred_ss = torch.sum(pred_centered.square(), dim=dim)
    target_ss = torch.sum(target_centered.square(), dim=dim)
    denom = torch.sqrt(pred_ss * target_ss)
    valid = denom > 1e-8
    if not torch.any(valid):
        return pred.new_tensor(0.0), 0
    corr = torch.sum(pred_centered * target_centered, dim=dim)[valid] / denom[valid]
    return (1.0 - corr).mean(), int(valid.sum().item())


def _compute_loss_terms(model, batch, device, args):
    gene_ids, x_visible, mask_missing, target_idx, y_target, log_total_visible, spatial_xy, _ = batch
    gene_ids = gene_ids.to(device, non_blocking=True)
    x_visible = x_visible.to(device, non_blocking=True)
    mask_missing = mask_missing.to(device, non_blocking=True)
    target_idx = target_idx.to(device, non_blocking=True)
    y_target = y_target.to(device, non_blocking=True)
    log_total_visible = log_total_visible.to(device, non_blocking=True)
    spatial_xy = spatial_xy.to(device, non_blocking=True)

    pred = model(
        gene_ids=gene_ids,
        x_visible=x_visible,
        mask_missing=mask_missing,
        target_idx=target_idx,
        log_total_visible=log_total_visible,
        spatial_xy=spatial_xy,
    )
    mse = F.mse_loss(pred, y_target)

    if float(args.loss_weight_pearson_all) > 0.0:
        loss_pearson_all, valid_all = _pearson_loss_dim(pred.reshape(1, -1), y_target.reshape(1, -1), dim=1)
    else:
        loss_pearson_all = pred.new_zeros(())
        valid_all = 0

    if float(args.loss_weight_spot_pearson) > 0.0:
        loss_spot_pearson, valid_spot = _pearson_loss_dim(pred, y_target, dim=1)
    else:
        loss_spot_pearson = pred.new_zeros(())
        valid_spot = 0

    if float(args.loss_weight_gene_pearson) > 0.0:
        loss_gene_pearson, valid_gene = _pearson_loss_dim(pred, y_target, dim=0)
    else:
        loss_gene_pearson = pred.new_zeros(())
        valid_gene = 0

    total_loss = float(args.loss_weight_mse) * mse
    if float(args.loss_weight_pearson_all) > 0.0:
        total_loss = total_loss + float(args.loss_weight_pearson_all) * loss_pearson_all
    if float(args.loss_weight_spot_pearson) > 0.0:
        total_loss = total_loss + float(args.loss_weight_spot_pearson) * loss_spot_pearson
    if float(args.loss_weight_gene_pearson) > 0.0:
        total_loss = total_loss + float(args.loss_weight_gene_pearson) * loss_gene_pearson
    return {
        "pred": pred,
        "target": y_target,
        "total_loss": total_loss,
        "mse": mse,
        "loss_pearson_all": loss_pearson_all,
        "loss_spot_pearson": loss_spot_pearson,
        "loss_gene_pearson": loss_gene_pearson,
        "valid_flat_pearson": valid_all,
        "valid_spot_pearson": valid_spot,
        "valid_gene_pearson": valid_gene,
    }


def _init_metric_sums():
    return {
        "loss": 0.0,
        "mse": 0.0,
        "loss_pearson_all": 0.0,
        "loss_spot_pearson": 0.0,
        "loss_gene_pearson": 0.0,
    }


def _finalize_metric_sums(metric_sums: dict, total_items: int) -> dict:
    denom = max(int(total_items), 1)
    return {k: float(v) / denom for k, v in metric_sums.items()}


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    metric_sums = _init_metric_sums()
    total_spots = 0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = _compute_loss_terms(model, batch, device, args)
            loss = outputs["total_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = int(batch[1].shape[0])
        metric_sums["loss"] += float(outputs["total_loss"].item()) * bs
        metric_sums["mse"] += float(outputs["mse"].item()) * bs
        metric_sums["loss_pearson_all"] += float(outputs["loss_pearson_all"].item()) * bs
        metric_sums["loss_spot_pearson"] += float(outputs["loss_spot_pearson"].item()) * bs
        metric_sums["loss_gene_pearson"] += float(outputs["loss_gene_pearson"].item()) * bs
        total_spots += bs
    metrics = _finalize_metric_sums(metric_sums, total_spots)
    metrics["rmse"] = float(np.sqrt(max(metrics["mse"], 0.0)))
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    metric_sums = _init_metric_sums()
    total_spots = 0
    preds = []
    truths = []
    for batch in tqdm(loader, desc="val", leave=False):
        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = _compute_loss_terms(model, batch, device, args)

        bs = int(batch[1].shape[0])
        metric_sums["loss"] += float(outputs["total_loss"].item()) * bs
        metric_sums["mse"] += float(outputs["mse"].item()) * bs
        metric_sums["loss_pearson_all"] += float(outputs["loss_pearson_all"].item()) * bs
        metric_sums["loss_spot_pearson"] += float(outputs["loss_spot_pearson"].item()) * bs
        metric_sums["loss_gene_pearson"] += float(outputs["loss_gene_pearson"].item()) * bs
        total_spots += bs
        preds.append(outputs["pred"].detach().float().cpu().numpy())
        truths.append(outputs["target"].detach().float().cpu().numpy())

    metrics = _finalize_metric_sums(metric_sums, total_spots)
    metrics["rmse"] = float(np.sqrt(max(metrics["mse"], 0.0)))
    if preds:
        pred_np = np.concatenate(preds, axis=0)
        truth_np = np.concatenate(truths, axis=0)
        summary_metrics = summarize_target_metrics(pred_np, truth_np)
        metrics.update(summary_metrics)
    return metrics


def main():
    args = parse_args()
    set_seed(int(args.seed))

    exp_dir = Path(args.exp_dir).resolve()
    cfg_path = Path(args.config).resolve() if args.config else (exp_dir / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (exp_dir / "checkpoints" / "checkpoint_best.pt").resolve()
    )
    if not checkpoint_path.exists():
        fallback = (exp_dir / "checkpoints" / "checkpoint_last.pt").resolve()
        if fallback.exists():
            checkpoint_path = fallback
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_args.json", vars(args))

    sample_ids = _sample_ids_arg(args.sample_ids)
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

    spatial_xy = np.zeros((benchmark.spatial.n_obs, 2), dtype=np.float32)
    if "spatial" in benchmark.spatial.obsm:
        spatial_xy = normalize_spatial_coords(np.asarray(benchmark.spatial.obsm["spatial"], dtype=np.float32))
    if str(args.split_mode) == "spatial_block":
        split = build_spot_spatial_block_split(
            spatial_xy=spatial_xy,
            val_ratio=float(args.val_ratio),
            corner=str(args.block_corner),
        )
    else:
        split = build_spot_train_val_split(
            benchmark.spatial.n_obs,
            val_ratio=float(args.val_ratio),
            seed=int(args.split_seed),
        )

    train_ds = SpatialSpotTargetDataset(
        count_matrix=benchmark.spatial.X,
        gene_ids=benchmark.ref.gene_ids,
        spatial_xy=spatial_xy,
        indices=split.train_idx,
        observed_idx=benchmark.observed_idx,
        target_idx=benchmark.target_idx,
        expression_space=str(args.expression_space),
    )
    val_ds = SpatialSpotTargetDataset(
        count_matrix=benchmark.spatial.X,
        gene_ids=benchmark.ref.gene_ids,
        spatial_xy=spatial_xy,
        indices=split.val_idx,
        observed_idx=benchmark.observed_idx,
        target_idx=benchmark.target_idx,
        expression_space=str(args.expression_space),
    )

    pin_memory = bool(torch.cuda.is_available() and str(args.device) == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    model_cfg = cfg["model"]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = ScUniFlowSpatialImputer(
        n_genes=int(model_cfg["n_genes"]),
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        head_hidden_dim=int(args.head_hidden_dim),
        head_dropout=float(args.head_dropout),
    )
    model.load_backbone_checkpoint(str(checkpoint_path), device="cpu")
    model.freeze_all_backbone()
    if args.train_mamba:
        model.unfreeze_mamba()
    if args.train_router:
        model.unfreeze_router()
    model = model.to(device)

    param_groups = [{"params": model.head.parameters(), "lr": float(args.lr_head)}]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(args.lr_backbone)})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    split_info = {
        **split.split_info,
        "sample_ids": sample_ids,
        "obs_names_train": benchmark.spatial.obs_names[split.train_idx].astype(str).tolist(),
        "obs_names_val": benchmark.spatial.obs_names[split.val_idx].astype(str).tolist(),
        "n_spots_total": int(benchmark.spatial.n_obs),
        "n_genes_total": int(benchmark.ref.ensembl_ids.shape[0]),
        "n_observed_genes": int(benchmark.observed_idx.shape[0]),
        "n_target_genes": int(benchmark.target_idx.shape[0]),
        "expression_space": str(args.expression_space),
        "base_exp_dir": str(exp_dir),
        "base_checkpoint": str(checkpoint_path),
        "config": str(cfg_path),
        "ref_h5ad": str(Path(args.ref_h5ad).resolve()),
        "visium_root": str(Path(args.visium_root).resolve()),
        "max_spots": int(args.max_spots),
        "max_ref_cells": int(args.max_ref_cells),
        "in_tissue_only": bool(args.in_tissue_only),
    }
    save_json(output_dir / "split_info.json", split_info)
    save_json(
        output_dir / "panel_info.json",
        {
            "observed_ensembl_ids": benchmark.ref.ensembl_ids[benchmark.observed_idx].tolist(),
            "target_ensembl_ids": benchmark.ref.ensembl_ids[benchmark.target_idx].tolist(),
            "observed_symbols": benchmark.ref.gene_symbols[benchmark.observed_idx].tolist(),
            "target_symbols": benchmark.ref.gene_symbols[benchmark.target_idx].tolist(),
        },
    )

    best_val = float("inf")
    best_ckpt_path = output_dir / "best_model.pt"
    history_path = output_dir / "train_log.jsonl"
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        if len(val_ds) > 0:
            val_metrics = evaluate(model, val_loader, device, args)
        else:
            val_metrics = {
                "loss": float(train_metrics["loss"]),
                "mse": float(train_metrics["mse"]),
                "rmse": float(train_metrics["rmse"]),
                "pearson_all": None,
                "spot_pearson_mean": None,
                "gene_pearson_mean": None,
            }
        scheduler.step()

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_mse": float(train_metrics["mse"]),
            "train_rmse": float(train_metrics["rmse"]),
            "train_loss_pearson_all": float(train_metrics["loss_pearson_all"]),
            "train_loss_spot_pearson": float(train_metrics["loss_spot_pearson"]),
            "train_loss_gene_pearson": float(train_metrics["loss_gene_pearson"]),
            "val_loss": float(val_metrics["loss"]),
            "val_mse": float(val_metrics["mse"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_pearson_all": None if val_metrics.get("pearson_all") is None else float(val_metrics["pearson_all"]),
            "val_spot_pearson_mean": None if val_metrics.get("spot_pearson_mean") is None else float(val_metrics["spot_pearson_mean"]),
            "val_spot_spearman_mean": None if val_metrics.get("spot_spearman_mean") is None else float(val_metrics["spot_spearman_mean"]),
            "val_gene_pearson_mean": None if val_metrics.get("gene_pearson_mean") is None else float(val_metrics["gene_pearson_mean"]),
            "lr_head": float(optimizer.param_groups[0]["lr"]),
            "lr_backbone": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else 0.0,
        }
        with history_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "exp_dir": str(exp_dir),
            "base_checkpoint": str(checkpoint_path),
            "config_path": str(cfg_path),
            "expression_space": str(args.expression_space),
            "sample_ids": sample_ids,
            "panel_seed": int(args.panel_seed),
            "sample_seed": int(args.sample_seed),
            "split_seed": int(args.split_seed),
            "head_hidden_dim": int(args.head_hidden_dim),
            "head_dropout": float(args.head_dropout),
        }
        torch.save(ckpt_payload, output_dir / "last_model.pt")
        if float(val_metrics["loss"]) < best_val:
            best_val = float(val_metrics["loss"])
            torch.save(ckpt_payload, best_ckpt_path)

    save_json(
        output_dir / "summary.json",
        {
            "best_val_loss": float(best_val),
            "epochs": int(args.epochs),
            "base_exp_dir": str(exp_dir),
            "base_checkpoint": str(checkpoint_path),
            "best_model": str(best_ckpt_path),
            "last_model": str((output_dir / "last_model.pt").resolve()),
        },
    )


if __name__ == "__main__":
    main()
