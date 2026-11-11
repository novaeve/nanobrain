import argparse
import json
from pathlib import Path

import numpy as np
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
    save_reconstruction_outputs,
    set_seed,
    summarize_reconstruction,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute scUniFlow read-depth enhancement reconstruction metrics only"
    )
    parser.add_argument("--exp_dir", required=True, help="base scUniFlow experiment directory")
    parser.add_argument("--task_ckpt", required=True, help="task fine-tuned checkpoint")
    parser.add_argument("--data", required=True, help="dataset key or h5ad path")
    parser.add_argument("--output_dir", required=True, help="output directory")
    parser.add_argument("--config", default="", help="optional config yaml; defaults to <exp_dir>/config.yaml")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prediction_chunk_size", type=int, default=4096)
    parser.add_argument("--downsample_fold", type=float, default=-1.0, help="negative means reuse task checkpoint setting")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell_indices_path", default="", help="optional .npy file of exact cell indices to evaluate")
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

    for batch in tqdm(loader, desc="eval_pcc"):
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

    counts, gene_ids, _, meta, split = load_counts_and_split(
        args.data,
        val_ratio=val_ratio,
        seed=split_seed,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )
    if args.cell_indices_path:
        eval_idx = np.load(args.cell_indices_path).astype(np.int64)
        split_name = f"{args.split}_from_cell_indices"
    elif args.split == "train":
        eval_idx = split.train_idx
        split_name = "train"
    elif args.split == "val":
        eval_idx = split.val_idx
        split_name = "val"
    else:
        eval_idx = np.concatenate([split.train_idx, split.val_idx], axis=0)
        split_name = "all"

    dataset = DownsamplePairDataset(counts, gene_ids, eval_idx, downsample_fold=downsample_fold, seed=split_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = ScUniFlowReadDepthEnhancer(
        n_genes=int(model_cfg["n_genes"]),
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
        head_hidden_dim=int(task_args.get("head_hidden_dim", 256)),
        head_dropout=float(task_args.get("head_dropout", 0.1)),
    )
    model.load_state_dict(task_ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    pred_log, low_log, high_log, cell_idx, used_batches = run_prediction(
        model,
        loader,
        device,
        chunk_size=args.prediction_chunk_size,
        max_batches=int(args.max_eval_batches),
    )
    cell_ids = meta["obs"].index[cell_idx].astype(str).tolist()
    metrics, cell_pccs, gene_pccs = summarize_reconstruction(pred_log, high_log)
    metrics.update(
        {
            "exp_dir": str(exp_dir),
            "config": str(cfg_path),
            "task_ckpt": str(Path(args.task_ckpt).resolve()),
            "data": str(args.data),
            "split": split_name,
            "n_eval_cells": int(pred_log.shape[0]),
            "n_genes": int(pred_log.shape[1]),
            "downsample_fold": float(downsample_fold),
            "used_batches": int(used_batches),
            "max_eval_batches": int(args.max_eval_batches),
            "metric_space": "log1p_raw_counts",
            "cell_indices_path": str(Path(args.cell_indices_path).resolve()) if args.cell_indices_path else "",
        }
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_reconstruction_outputs(output_dir, metrics, cell_ids, cell_pccs, gene_pccs)
    save_json(output_dir / "run_args.json", vars(args))
    np.save(output_dir / "cell_indices.npy", cell_idx)
    if args.save_arrays:
        np.save(output_dir / "pred_log.npy", pred_log)
        np.save(output_dir / "low_log.npy", low_log)
        np.save(output_dir / "high_log.npy", high_log)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
