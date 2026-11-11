import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    DEFAULT_GENE_INDEX,
    ReadDepthEnhancementDataset,
    ScFoundationReadDepthEnhancer,
    align_h5ad_to_gene_list,
    get_trainable_param_groups,
    load_finetune_state_dict,
    read_gene_list,
    set_seed,
    split_train_val_indices,
)


def resolve_amp_settings(device, precision: str):
    precision = str(precision).lower()
    if device.type != "cuda" or precision == "fp32":
        return False, torch.float32
    if precision == "bf16":
        return True, torch.bfloat16
    if precision == "fp16":
        return True, torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def _tensor_range_str(x: torch.Tensor):
    x = x.detach()
    finite = torch.isfinite(x)
    finite_count = int(finite.sum().item())
    total = int(x.numel())
    if finite_count == 0:
        return f"shape={tuple(x.shape)} finite=0/{total}"
    xf = x[finite].float()
    return (
        f"shape={tuple(x.shape)} finite={finite_count}/{total} "
        f"min={float(xf.min().item()):.6g} "
        f"max={float(xf.max().item()):.6g} "
        f"mean={float(xf.mean().item()):.6g}"
    )


def ensure_finite(name: str, x: torch.Tensor, batch_idx: int):
    if not torch.isfinite(x).all():
        raise RuntimeError(
            f"Non-finite tensor detected: {name} at batch={batch_idx}. "
            f"{_tensor_range_str(x)}"
        )


def grad_status(model: nn.Module):
    checked = 0
    non_finite = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None or not torch.is_tensor(grad) or not grad.is_floating_point():
            continue
        checked += 1
        if not torch.isfinite(grad).all():
            non_finite.append((name, _tensor_range_str(grad)))
            if len(non_finite) >= 8:
                break
    return checked, non_finite


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    amp_dtype,
    max_grad_norm: float,
    log_interval: int = 100,
    max_batches: int = 0,
):
    model.train()
    total_loss = 0.0
    n_cells = 0

    for batch_idx, (x, target) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        x_raw = x.clone()
        optimizer.zero_grad(set_to_none=True)

        ensure_finite("train/x", x, batch_idx)
        ensure_finite("train/target", target, batch_idx)

        with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            recon_pred = model(x, x_raw)
            loss = F.mse_loss(recon_pred, target)

        ensure_finite("train/recon_pred", recon_pred, batch_idx)
        ensure_finite("train/loss", loss, batch_idx)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        checked_grads, bad_grads = grad_status(model)
        if bad_grads:
            raise RuntimeError(
                f"Non-finite gradients detected at batch={batch_idx} "
                f"(checked={checked_grads}). Examples: {bad_grads}"
            )

        if max_grad_norm > 0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
        else:
            grad_norm = float("nan")

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n_cells += bs
        if batch_idx == 1 or (log_interval > 0 and batch_idx % log_interval == 0):
            print(
                f"  [train batch={batch_idx}/{len(loader)}] "
                f"loss={float(loss.item()):.6f} grad_norm={grad_norm:.6f}",
                flush=True,
            )
        if max_batches > 0 and batch_idx >= max_batches:
            break

    return total_loss / max(n_cells, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, amp_dtype, max_batches: int = 0):
    model.eval()
    total_loss = 0.0
    n_cells = 0

    for batch_idx, (x, target) in enumerate(tqdm(loader, desc="val", leave=False), start=1):
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        x_raw = x.clone()
        ensure_finite("val/x", x, batch_idx)
        ensure_finite("val/target", target, batch_idx)
        with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            recon_pred = model(x, x_raw)
            loss = F.mse_loss(recon_pred, target)
        ensure_finite("val/recon_pred", recon_pred, batch_idx)
        ensure_finite("val/loss", loss, batch_idx)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n_cells += bs
        if max_batches > 0 and batch_idx >= max_batches:
            break

    return total_loss / max(n_cells, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune scFoundation on read depth enhancement (x_low -> x_high)"
    )
    parser.add_argument("--pretrain_ckpt", required=True, help="scFoundation pretrained checkpoint path")
    parser.add_argument("--init_ckpt", default="", help="optional pbmc-finetuned checkpoint used as initialization")
    parser.add_argument("--data_path", required=True, help="input h5ad with raw counts")
    parser.add_argument("--save_dir", required=True, help="training output directory")
    parser.add_argument("--gene_index", default=str(DEFAULT_GENE_INDEX))
    parser.add_argument("--gene_name_col", default="gene_name")
    parser.add_argument("--layer", default="", help="optional h5ad layer name")
    parser.add_argument("--max_cells", type=int, default=0, help="0 means use all cells")
    parser.add_argument("--train_indices_path", default="", help="optional .npy original h5ad cell indices for training")
    parser.add_argument("--val_indices_path", default="", help="optional .npy original h5ad cell indices for validation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--freeze_emb", dest="freeze_emb", action="store_true")
    parser.add_argument("--no_freeze_emb", dest="freeze_emb", action="store_false")
    parser.set_defaults(freeze_emb=True)
    parser.add_argument("--unfreeze_last_n", type=int, default=2)
    parser.add_argument("--tgt_highres", type=float, default=4.0)
    parser.add_argument("--downsample_fold", type=float, default=4.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_train_batches", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max_val_batches", type=int, default=0, help="0 means full validation")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp, amp_dtype = resolve_amp_settings(device, args.precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(
        f"[setup] device={device} seed={args.seed} precision={args.precision} "
        f"use_amp={use_amp}",
        flush=True,
    )
    print(f"[data] reading gene index: {args.gene_index}", flush=True)
    gene_list = read_gene_list(args.gene_index)
    print(f"[data] target genes: {len(gene_list):,}", flush=True)
    print(f"[data] loading h5ad and aligning genes: {args.data_path}", flush=True)
    t0 = time.time()
    use_index_files = bool(args.train_indices_path or args.val_indices_path)
    if use_index_files and not (args.train_indices_path and args.val_indices_path):
        raise ValueError("Pass both --train_indices_path and --val_indices_path, or neither.")
    X, meta = align_h5ad_to_gene_list(
        args.data_path,
        gene_list,
        gene_name_col=args.gene_name_col,
        layer=args.layer or None,
        max_cells=(None if use_index_files or int(args.max_cells) <= 0 else int(args.max_cells)),
        cell_seed=args.seed,
    )
    print(
        f"[data] aligned matrix ready: cells={X.shape[0]:,} genes={X.shape[1]:,} "
        f"present_genes={meta['n_present_genes']:,} missing_genes={meta['n_missing_genes']:,} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    if use_index_files:
        train_idx = np.load(args.train_indices_path).astype(np.int64)
        val_idx = np.load(args.val_indices_path).astype(np.int64)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("train/val index files must be non-empty.")
        if train_idx.max(initial=-1) >= X.shape[0] or val_idx.max(initial=-1) >= X.shape[0]:
            raise ValueError("train/val index files contain indices outside the h5ad.")
        overlap = np.intersect1d(train_idx, val_idx)
        if overlap.size:
            raise ValueError(f"train/val index overlap is not allowed: {overlap.size} cells")
    else:
        train_idx, val_idx = split_train_val_indices(X.shape[0], args.val_ratio, args.seed)
    print(f"[split] train={len(train_idx):,} val={len(val_idx):,}", flush=True)

    with open(save_dir / "split_info.json", "w") as f:
        json.dump(
            {
                "n_total_cells": int(X.shape[0]),
                "n_train_cells": int(len(train_idx)),
                "n_val_cells": int(len(val_idx)),
                "n_target_genes": int(meta["n_target_genes"]),
                "n_present_genes": int(meta["n_present_genes"]),
                "n_missing_genes": int(meta["n_missing_genes"]),
                "seed": int(args.seed),
                "val_ratio": float(args.val_ratio),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    train_ds = ReadDepthEnhancementDataset(
        X,
        indices=train_idx,
        tgt_highres=args.tgt_highres,
        downsample_fold=args.downsample_fold,
        seed=args.seed,
    )
    val_ds = ReadDepthEnhancementDataset(
        X,
        indices=val_idx,
        tgt_highres=args.tgt_highres,
        downsample_fold=args.downsample_fold,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(
        f"[loader] batch_size={args.batch_size} num_workers={args.num_workers} "
        f"train_steps={len(train_loader):,} val_steps={len(val_loader):,}",
        flush=True,
    )

    print(f"[model] loading scFoundation backbone: {args.pretrain_ckpt}", flush=True)
    model = ScFoundationReadDepthEnhancer(
        ckpt_path=args.pretrain_ckpt,
        freeze_emb=args.freeze_emb,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    init_meta = None
    if args.init_ckpt:
        init_meta = load_finetune_state_dict(model, args.init_ckpt)
        print(f"[init] loaded finetuned checkpoint: {Path(args.init_ckpt).resolve()}")
    print("[train] start training", flush=True)
    optimizer = torch.optim.AdamW(
        get_trainable_param_groups(model, lr=args.lr, lr_head=args.lr_head),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    scaler = GradScaler(enabled=use_amp)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_grad_norm=float(args.max_grad_norm),
            log_interval=int(args.log_interval),
            max_batches=int(args.max_train_batches),
        )
        val_loss = evaluate(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_batches=int(args.max_val_batches),
        )
        scheduler.step()

        if not torch.isfinite(torch.tensor(train_loss)) or not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError(
                f"Non-finite epoch loss detected at epoch={epoch}: "
                f"train_loss={train_loss}, val_loss={val_loss}"
            )

        row = {
            "epoch": int(epoch),
            "train_recon_loss": float(train_loss),
            "val_recon_loss": float(val_loss),
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[-1]["lr"]),
            "precision": args.precision,
            "max_train_batches": int(args.max_train_batches),
            "max_val_batches": int(args.max_val_batches),
        }
        history.append(row)
        with open(save_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"Epoch {epoch:03d} | "
            f"train_recon_loss={train_loss:.4f} | "
            f"val_recon_loss={val_loss:.4f}"
        )

        payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "pretrain_ckpt": args.pretrain_ckpt,
            "init_ckpt": args.init_ckpt,
            "train_args": vars(args),
            "model_cfg": model.cfg,
            "val_recon_loss": float(val_loss),
            "init_checkpoint_meta": {
                "epoch": None if init_meta is None else init_meta.get("epoch"),
                "val_recon_loss": None if init_meta is None else init_meta.get("val_recon_loss"),
            },
        }
        torch.save(payload, save_dir / "last_model.pt")

        if val_loss < best_val:
            best_val = float(val_loss)
            torch.save(payload, save_dir / "best_model.pt")
            print(f"  [best] {save_dir / 'best_model.pt'}")

    with open(save_dir / "summary.json", "w") as f:
        json.dump(
            {
                "best_val_recon_loss": float(best_val),
                "epochs": int(args.epochs),
                "n_train_cells": int(len(train_idx)),
                "n_val_cells": int(len(val_idx)),
                "n_target_genes": int(meta["n_target_genes"]),
                "downsample_fold": float(args.downsample_fold),
                "init_ckpt": str(Path(args.init_ckpt).resolve()) if args.init_ckpt else "",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
