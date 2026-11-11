import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_sea_ad_split
from train import make_dataloader, compute_log1p_nonzero_prior_stats, scRNAPipelineTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Masked-reconstruction eval for scUniFlow on a generic h5ad/val split"
    )
    parser.add_argument("--exp_dir", required=True, help="Experiment directory containing config.yaml")
    parser.add_argument("--config", default="", help="Optional explicit config yaml path; defaults to <exp_dir>/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate")
    parser.add_argument("--data", required=True, help="Dataset key or h5ad path")
    parser.add_argument("--batch_size", type=int, default=None, help="Override validation.batch_size from yaml")
    parser.add_argument("--max_batches", type=int, default=None, help="Override validation.max_batches from yaml; 0 means all batches")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--eval_seed", type=int, default=1234, help="Seed for deterministic inference noise")
    parser.add_argument("--result_json", default="", help="Optional explicit output json path")
    return parser.parse_args()


def build_result_path(exp_dir: Path, checkpoint: Path) -> Path:
    out_dir = exp_dir / "eval_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{checkpoint.stem}_full_val_eval.json"


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    cfg_path = Path(args.config).resolve() if args.config else (exp_dir / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config yaml: {cfg_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    val_cfg = cfg.get("validation", {})

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)

    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    split_seed = int(train_cfg.get("seed", 42))
    cell_seed = int(train_cfg.get("cell_seed", split_seed))
    max_cells = train_cfg.get("max_cells")

    train_data, val_data, _, meta = load_sea_ad_split(
        args.data,
        val_ratio=val_ratio,
        seed=split_seed,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )
    train_X, gene_ids = train_data
    val_X, _ = val_data
    if val_X.shape[0] == 0:
        raise RuntimeError("Validation split is empty; cannot run full val eval.")

    val_batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else val_cfg.get("batch_size", train_cfg.get("batch_size_per_gpu", 8))
    )
    val_loader, _ = make_dataloader(
        val_X,
        gene_ids,
        batch_size=val_batch_size,
        shuffle=False,
        world_size=1,
        nonzero_mask_ratio=float(val_cfg.get("nonzero_mask_ratio", 0.2)),
        zero_mask_ratio=float(val_cfg.get("zero_mask_ratio", 0.0)),
        return_index=True,
    )

    trainer = scRNAPipelineTrainer(
        n_genes=int(meta["n_genes"]),
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
        local_rank=0,
        exp_dir=str(exp_dir),
    )
    trainer.val_infer_n_steps = int(val_cfg.get("n_steps", cfg.get("inference", {}).get("n_steps", 10)))
    trainer.val_infer_temperature = float(val_cfg.get("temperature", cfg.get("inference", {}).get("temperature", 0.1)))
    trainer.val_nonzero_mask_ratio = float(val_cfg.get("nonzero_mask_ratio", 0.2))
    trainer.val_zero_mask_ratio = float(val_cfg.get("zero_mask_ratio", 0.0))
    trainer.val_mask_seed = int(val_cfg.get("mask_seed", 1234))
    trainer.val_noise_seed = int(val_cfg.get("noise_seed", args.eval_seed))
    cfg_max_batches = int(val_cfg.get("max_batches", 0))
    effective_max_batches = cfg_max_batches if args.max_batches is None else int(args.max_batches)
    trainer.val_max_batches = len(val_loader) if effective_max_batches <= 0 else min(effective_max_batches, len(val_loader))

    ckpt = torch.load(checkpoint, map_location=trainer.device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    trainer.raw_model.load_state_dict(state_dict, strict=True)
    trainer.val_prior_mean = float(ckpt.get("prior_mean", 0.0))
    trainer.val_prior_std = float(max(ckpt.get("prior_std", 1.0), 1e-6))
    if "prior_mean" not in ckpt or "prior_std" not in ckpt:
        mean, std = compute_log1p_nonzero_prior_stats(train_X)
        trainer.val_prior_mean = float(mean)
        trainer.val_prior_std = float(max(std, 1e-6))

    # Re-seed immediately before rollout so both checkpoints see identical inference noise.
    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)

    metrics = trainer.validate(val_loader, epoch=None)
    result = {
        "exp_dir": str(exp_dir),
        "config": str(cfg_path),
        "checkpoint": str(checkpoint),
        "data": str(args.data),
        "n_val_cells": int(val_X.shape[0]),
        "n_genes": int(meta["n_genes"]),
        "grid_size": int(model_cfg["grid_size"]),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "eval_seed": int(args.eval_seed),
        "val_noise_seed": int(trainer.val_noise_seed),
        "val_batch_size": int(val_batch_size),
        "val_max_batches_config": int(cfg_max_batches),
        "val_max_batches": int(trainer.val_max_batches),
        "prior_mean": float(trainer.val_prior_mean),
        "prior_std": float(trainer.val_prior_std),
        **metrics,
    }

    result_path = Path(args.result_json).resolve() if args.result_json else build_result_path(exp_dir, checkpoint)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {result_path}")


if __name__ == "__main__":
    main()
