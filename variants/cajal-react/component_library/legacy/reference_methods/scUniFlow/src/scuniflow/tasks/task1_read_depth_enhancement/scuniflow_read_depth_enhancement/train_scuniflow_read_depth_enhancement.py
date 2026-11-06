import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from common import (
    DownsamplePairDataset,
    ScUniFlowReadDepthEnhancer,
    load_counts_and_split,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a scUniFlow task head for read depth enhancement"
    )
    parser.add_argument("--exp_dir", required=True, help="base scUniFlow experiment directory")
    parser.add_argument("--checkpoint", required=True, help="trained scUniFlow backbone checkpoint")
    parser.add_argument("--data", required=True, help="dataset key or h5ad path")
    parser.add_argument("--output_dir", required=True, help="task fine-tune output directory")
    parser.add_argument("--config", default="", help="optional override config yaml; defaults to <exp_dir>/config.yaml")
    parser.add_argument("--train_indices_path", default="", help="optional .npy original h5ad cell indices for training")
    parser.add_argument("--val_indices_path", default="", help="optional .npy original h5ad cell indices for validation")
    parser.add_argument("--downsample_fold", type=float, default=4.0, help="high/low depth fold ratio")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--head_type", choices=["gene_residual", "cell_embedding"], default="gene_residual")
    parser.add_argument("--expression_space", choices=["raw_log1p", "normalized_log1p"], default="raw_log1p")
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--prediction_chunk_size", type=int, default=4096)
    parser.add_argument("--train_mamba", action="store_true", help="jointly fine-tune mamba encoder")
    parser.add_argument("--train_router", action="store_true", help="jointly fine-tune router parameters")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=-1, help="negative means reuse training.seed from config")
    parser.add_argument("--skip_eval_after_train", action="store_true", help="do not auto-run the existing RDE evaluation script after training")
    parser.add_argument("--eval_split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--eval_output_dir", default="", help="optional evaluation output directory; defaults to <output_dir>/eval_<split>")
    parser.add_argument("--eval_batch_size", type=int, default=0, help="evaluation batch size; <=0 means reuse training batch_size")
    parser.add_argument("--eval_label_col", default="", help="obs column used as target labels for evaluation")
    parser.add_argument("--eval_ref_resolution", type=float, default=0.4, help="reference Leiden resolution when eval_label_col is empty")
    parser.add_argument("--eval_n_top_genes", type=int, default=2000)
    parser.add_argument("--eval_max_eval_batches", type=int, default=0, help="0 means evaluate all batches")
    parser.add_argument("--eval_save_arrays", action="store_true")
    return parser.parse_args()


def run_post_train_eval(args, output_dir: Path, best_ckpt_path: Path, seed: int):
    eval_script = Path(__file__).resolve().parent / "eval_scuniflow_read_depth_enhancement.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    eval_output_dir = (
        Path(args.eval_output_dir).resolve()
        if args.eval_output_dir
        else (output_dir / f"eval_{args.eval_split}").resolve()
    )
    eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)

    cmd = [
        sys.executable,
        str(eval_script),
        "--exp_dir",
        str(Path(args.exp_dir).resolve()),
        "--task_ckpt",
        str(best_ckpt_path.resolve()),
        "--data",
        str(args.data),
        "--output_dir",
        str(eval_output_dir),
        "--split",
        str(args.eval_split),
        "--batch_size",
        str(eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--prediction_chunk_size",
        str(args.prediction_chunk_size),
        "--downsample_fold",
        str(args.downsample_fold),
        "--ref_resolution",
        str(args.eval_ref_resolution),
        "--n_top_genes",
        str(args.eval_n_top_genes),
        "--max_eval_batches",
        str(args.eval_max_eval_batches),
        "--device",
        str(args.device),
        "--seed",
        str(int(seed)),
    ]
    if args.config:
        cmd.extend(["--config", str(Path(args.config).resolve())])
    if args.train_indices_path and args.val_indices_path:
        if args.eval_split == "train":
            cmd.extend(["--cell_indices_path", str(Path(args.train_indices_path).resolve())])
        elif args.eval_split == "val":
            cmd.extend(["--cell_indices_path", str(Path(args.val_indices_path).resolve())])
    if args.eval_label_col:
        cmd.extend(["--label_col", str(args.eval_label_col)])
    if args.eval_save_arrays:
        cmd.append("--save_arrays")

    print("[eval] running post-train evaluation:")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return eval_output_dir


def train_one_epoch(model, loader, optimizer, scaler, device, chunk_size: int):
    model.train()
    total_loss = 0.0
    total_cells = 0
    for batch in tqdm(loader, desc="train", leave=False):
        gene_ids, x_low, x_high, log_total_low, log_total_high, log_fold, _ = batch
        gene_ids = gene_ids.to(device, non_blocking=True)
        x_low = x_low.to(device, non_blocking=True)
        x_high = x_high.to(device, non_blocking=True)
        log_total_low = log_total_low.to(device, non_blocking=True)
        log_fold = log_fold.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(
                gene_ids,
                x_low,
                log_total_low=log_total_low,
                log_fold=log_fold,
                chunk_size=chunk_size,
            )
            loss = F.mse_loss(pred, x_high)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = x_low.shape[0]
        total_loss += float(loss.item()) * bs
        total_cells += bs
    return total_loss / max(total_cells, 1)


@torch.no_grad()
def evaluate(model, loader, device, chunk_size: int):
    model.eval()
    total_loss = 0.0
    total_cells = 0
    for batch in tqdm(loader, desc="val", leave=False):
        gene_ids, x_low, x_high, log_total_low, log_total_high, log_fold, _ = batch
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
            loss = F.mse_loss(pred, x_high)
        bs = x_low.shape[0]
        total_loss += float(loss.item()) * bs
        total_cells += bs
    return total_loss / max(total_cells, 1)


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    cfg_path = Path(args.config).resolve() if args.config else (exp_dir / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    seed = int(train_cfg.get("seed", 42) if args.seed < 0 else args.seed)
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    cell_seed = int(train_cfg.get("cell_seed", seed))
    max_cells = train_cfg.get("max_cells")
    set_seed(seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_args.json", vars(args))

    counts, gene_ids, gene_names, meta, split = load_counts_and_split(
        args.data,
        val_ratio=val_ratio,
        seed=seed,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )
    if bool(args.train_indices_path or args.val_indices_path):
        if not (args.train_indices_path and args.val_indices_path):
            raise ValueError("Pass both --train_indices_path and --val_indices_path, or neither.")
        train_idx = np.load(args.train_indices_path).astype(np.int64)
        val_idx = np.load(args.val_indices_path).astype(np.int64)
        n_cells = counts.shape[0]
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("train/val index files must be non-empty.")
        if train_idx.min() < 0 or val_idx.min() < 0 or train_idx.max() >= n_cells or val_idx.max() >= n_cells:
            raise ValueError(f"train/val index out of range for n_cells={n_cells}")
        overlap = np.intersect1d(train_idx, val_idx)
        if overlap.size:
            raise ValueError(f"train/val index overlap: {overlap.size} cells")
        split.train_idx = train_idx
        split.val_idx = val_idx
    if int(model_cfg["n_genes"]) != int(meta["n_genes"]):
        raise ValueError(f"n_genes mismatch: config={model_cfg['n_genes']} data={meta['n_genes']}")

    train_ds = DownsamplePairDataset(
        counts,
        gene_ids,
        split.train_idx,
        downsample_fold=args.downsample_fold,
        seed=seed,
        expression_space=args.expression_space,
    )
    val_ds = DownsamplePairDataset(
        counts,
        gene_ids,
        split.val_idx,
        downsample_fold=args.downsample_fold,
        seed=seed,
        expression_space=args.expression_space,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = ScUniFlowReadDepthEnhancer(
        n_genes=int(model_cfg["n_genes"]),
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        head_hidden_dim=int(args.head_hidden_dim),
        head_dropout=float(args.head_dropout),
        head_type=str(args.head_type),
    )
    base_ckpt = model.load_backbone_checkpoint(args.checkpoint, device="cpu")
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
    scaler = GradScaler(enabled=(device.type == "cuda"))

    split_summary = {
        "split_mode": split.split_mode,
        "split_info": split.split_info,
        "n_train_cells": int(len(split.train_idx)),
        "n_val_cells": int(len(split.val_idx)),
        "n_genes": int(meta["n_genes"]),
        "downsample_fold": float(args.downsample_fold),
        "head_type": str(args.head_type),
        "expression_space": str(args.expression_space),
        "train_indices_path": str(Path(args.train_indices_path).resolve()) if args.train_indices_path else "",
        "val_indices_path": str(Path(args.val_indices_path).resolve()) if args.val_indices_path else "",
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "base_exp_dir": str(exp_dir),
        "config": str(cfg_path),
        "source_data": str(meta["source_path"]),
    }
    save_json(output_dir / "split_info.json", split_summary)

    best_val = float("inf")
    best_ckpt_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, chunk_size=args.prediction_chunk_size)
        val_loss = evaluate(model, val_loader, device, chunk_size=args.prediction_chunk_size)
        scheduler.step()

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr_head": float(optimizer.param_groups[0]["lr"]),
            "lr_backbone": float(optimizer.param_groups[-1]["lr"]) if len(optimizer.param_groups) > 1 else 0.0,
        }
        with open(output_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        ckpt_payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_args": vars(args),
            "base_checkpoint": str(Path(args.checkpoint).resolve()),
            "config_path": str(cfg_path),
            "base_checkpoint_meta": {
                "epoch": base_ckpt.get("epoch"),
                "step": base_ckpt.get("step"),
            },
            "val_loss": float(val_loss),
        }
        torch.save(ckpt_payload, output_dir / "last_model.pt")
        if val_loss < best_val:
            best_val = float(val_loss)
            torch.save(ckpt_payload, best_ckpt_path)
            print(f"  [best] {best_ckpt_path}")

    summary = {
        "best_val_loss": float(best_val),
        "epochs": int(args.epochs),
        "downsample_fold": float(args.downsample_fold),
    }
    if not args.skip_eval_after_train:
        eval_output_dir = run_post_train_eval(
            args,
            output_dir=output_dir,
            best_ckpt_path=best_ckpt_path,
            seed=seed,
        )
        summary["eval_output_dir"] = str(eval_output_dir)
        print(f"[eval] finished -> {eval_output_dir}", flush=True)
    save_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
