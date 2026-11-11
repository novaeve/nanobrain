import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "model.py").exists() and (p / "data.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_atac import (  # noqa: E402
    compute_accessibility_prior_stats,
    load_atac_h5ad,
    load_atac_h5ad_split,
    make_dataloader,
)
from train_atac import scATACPipelineTrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune only the scATAC intensity head under simulated dropout context."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--spectral_coords", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--dropout_ratio", type=float, default=0.4)
    parser.add_argument("--intensity_t", type=float, default=0.0)
    parser.add_argument("--pearson_weight", type=float, default=0.1)
    parser.add_argument("--cell_pearson_weight", type=float, default=0.1)
    parser.add_argument("--val_max_batches", type=int, default=32)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument(
        "--expr_transform",
        type=str,
        choices=["identity", "log1p", "tfidf_log1p_1e4", "logcpm"],
        default=None,
    )
    return parser.parse_args()


def pearson_loss(pred, target):
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = pred_c.norm() * target_c.norm()
    if float(denom.detach().item()) <= 0:
        return pred.new_tensor(0.0)
    return 1.0 - (pred_c * target_c).sum() / (denom + 1e-8)


def _sample_dropout_mask(x_1, dropout_ratio):
    mask = torch.zeros_like(x_1, dtype=torch.float32)
    ratio = float(dropout_ratio)
    for b in range(x_1.shape[0]):
        nz = (x_1[b] > 0).nonzero(as_tuple=True)[0]
        if nz.numel() == 0:
            continue
        n_drop = max(1, int(round(nz.numel() * ratio)))
        n_drop = min(n_drop, nz.numel())
        chosen = nz[torch.randperm(nz.numel(), device=x_1.device)[:n_drop]]
        mask[b, chosen] = 1.0
    return mask


def _one_step_intensity_loss(raw_model, g_id, x_1, dropout_mask, intensity_t, pearson_weight, cell_pearson_weight):
    observed = x_1.clone()
    observed[dropout_mask == 1] = 0.0

    with torch.no_grad():
        context, _ = raw_model.compute_context(
            g_id,
            observed,
            dropout_mask,
            temperature=0.0,
        )

    preds = []
    targets = []
    cell_losses = []
    t = torch.full((x_1.shape[0], 1), float(intensity_t), device=x_1.device, dtype=x_1.dtype)
    for b in range(x_1.shape[0]):
        m_idx = dropout_mask[b].nonzero(as_tuple=True)[0]
        if m_idx.numel() == 0:
            continue
        ctx_m = context[b, m_idx, :].unsqueeze(0)
        xt_m = observed[b, m_idx].unsqueeze(0)
        pred = raw_model.vector_field(xt_m, ctx_m, t[b].unsqueeze(0)).squeeze(0)
        target = x_1[b, m_idx]
        preds.append(pred)
        targets.append(target)
        if pred.numel() >= 2:
            cell_losses.append(pearson_loss(pred, target))

    if not preds:
        zero = x_1.new_tensor(0.0)
        return zero, {"huber": zero, "pearson": zero, "cell_pearson": zero, "n_masked": 0}

    flat_pred = torch.cat(preds)
    flat_target = torch.cat(targets)
    huber = F.huber_loss(flat_pred, flat_target, delta=0.5)
    pearson = pearson_loss(flat_pred, flat_target)
    cell_pearson = torch.stack(cell_losses).mean() if cell_losses else flat_pred.new_tensor(0.0)
    loss = huber + float(pearson_weight) * pearson + float(cell_pearson_weight) * cell_pearson
    metrics = {
        "huber": huber,
        "pearson": pearson,
        "cell_pearson": cell_pearson,
        "n_masked": int(flat_target.numel()),
    }
    return loss, metrics


@torch.no_grad()
def validate_one_step(raw_model, val_loader, device, dropout_ratio, intensity_t, pearson_weight, cell_pearson_weight, max_batches):
    raw_model.eval()
    total_loss = 0.0
    total_huber = 0.0
    total_pearson = 0.0
    total_cell_pearson = 0.0
    total_masked = 0
    n_batches = 0
    cell_corrs = []

    for batch in val_loader:
        if len(batch) == 5:
            g_id, x_1, _, _, _ = batch
        else:
            g_id, x_1, _, _ = batch
        g_id = g_id.to(device)
        x_1 = x_1.to(device)
        dropout_mask = _sample_dropout_mask(x_1, dropout_ratio)
        loss, metrics = _one_step_intensity_loss(
            raw_model,
            g_id,
            x_1,
            dropout_mask,
            intensity_t,
            pearson_weight,
            cell_pearson_weight,
        )
        n_masked = int(metrics["n_masked"])
        total_loss += float(loss.item())
        total_huber += float(metrics["huber"].item())
        total_pearson += float(metrics["pearson"].item())
        total_cell_pearson += float(metrics["cell_pearson"].item())
        total_masked += n_masked
        n_batches += 1

        observed = x_1.clone()
        observed[dropout_mask == 1] = 0.0
        context, _ = raw_model.compute_context(g_id, observed, dropout_mask, temperature=0.0)
        t = torch.full((x_1.shape[0], 1), float(intensity_t), device=device, dtype=x_1.dtype)
        for b in range(x_1.shape[0]):
            m_idx = dropout_mask[b].nonzero(as_tuple=True)[0]
            if m_idx.numel() < 2:
                continue
            pred = raw_model.vector_field(
                observed[b, m_idx].unsqueeze(0),
                context[b, m_idx, :].unsqueeze(0),
                t[b].unsqueeze(0),
            ).squeeze(0)
            target = x_1[b, m_idx]
            pred_c = pred.float() - pred.float().mean()
            target_c = target.float() - target.float().mean()
            denom = pred_c.norm() * target_c.norm()
            if float(denom.item()) > 0:
                cell_corrs.append(float(((pred_c * target_c).sum() / (denom + 1e-8)).item()))

        if n_batches >= int(max_batches):
            break

    denom = max(n_batches, 1)
    return {
        "val_loss": total_loss / denom,
        "val_huber": total_huber / denom,
        "val_pearson_loss": total_pearson / denom,
        "val_cell_pearson_loss": total_cell_pearson / denom,
        "val_intensity_pearson_cell": float(np.mean(cell_corrs)) if cell_corrs else None,
        "val_masked_nonzero": int(total_masked),
        "val_batches": int(n_batches),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("finetune_atac_dropout_intensity.py requires CUDA.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    expr_transform = str(args.expr_transform or train_cfg.get("expression_transform", "identity"))
    spectral_coords = args.spectral_coords or train_cfg.get("spectral_coords")
    if not spectral_coords:
        raise ValueError("必须提供 spectral_coords。")

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_cells = args.max_cells if args.max_cells is not None else train_cfg.get("max_cells")
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    val_cells = train_cfg.get("val_cells")
    cell_seed = int(train_cfg.get("cell_seed", seed))

    use_split = val_ratio > 0 or (val_cells is not None and int(val_cells) > 0)
    if use_split:
        train_data, val_data, _, meta = load_atac_h5ad_split(
            args.data,
            val_ratio=val_ratio,
            val_cells=val_cells,
            seed=seed,
            max_cells=max_cells,
            cell_seed=cell_seed,
        )
        train_X, feature_ids = train_data
        val_X, _ = val_data
    else:
        train_X, feature_ids, _, meta = load_atac_h5ad(
            args.data,
            max_cells=max_cells,
            cell_seed=cell_seed,
        )
        val_X = None

    n_features = int(meta["n_features"])
    prior_mean, prior_std = compute_accessibility_prior_stats(train_X, expr_transform=expr_transform)

    run_name = args.run_name or f"atac_dropout_intensity_ft_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path(args.output_dir or Path(train_cfg.get("experiment_dir", str(ROOT / "experiments"))) / run_name)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    batch_size = int(args.batch_size or train_cfg.get("batch_size_per_gpu", 8))
    train_loader, _ = make_dataloader(
        train_X,
        feature_ids,
        batch_size=batch_size,
        shuffle=True,
        world_size=1,
        nonzero_mask_ratio=0.0,
        zero_mask_ratio=0.0,
        expr_transform=expr_transform,
    )
    val_loader = None
    if val_X is not None and val_X.shape[0] > 0:
        val_loader, _ = make_dataloader(
            val_X,
            feature_ids,
            batch_size=max(1, min(batch_size, int(cfg.get("validation", {}).get("batch_size", batch_size)))),
            shuffle=False,
            world_size=1,
            nonzero_mask_ratio=0.0,
            zero_mask_ratio=0.0,
            return_index=True,
            expr_transform=expr_transform,
        )

    trainer = scATACPipelineTrainer(
        n_genes=n_features,
        d_model=model_cfg["d_model"],
        grid_size=model_cfg["grid_size"],
        local_rank=0,
        exp_dir=str(exp_dir),
    )
    trainer.raw_model.mamba_encoder.activation_checkpointing = bool(
        model_cfg.get("activation_checkpointing", True)
    )
    trainer.expression_transform = expr_transform
    trainer.val_prior_mean = float(prior_mean)
    trainer.val_prior_std = float(max(prior_std, 1e-6))

    coords = np.load(spectral_coords)
    if coords.shape != (n_features, 2):
        raise ValueError(f"spectral_coords shape {coords.shape} != ({n_features}, 2)")
    trainer.raw_model.init_from_spectral(coords, slot_order_mode="hilbert", gene_order_mode="hilbert")

    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    state = ckpt.get("model_state_dict", ckpt)
    trainer.raw_model.load_state_dict(state)

    raw = trainer.raw_model
    for param in raw.parameters():
        param.requires_grad = False
    for param in raw.vector_field.mlp.parameters():
        param.requires_grad = True

    trainable = [p for p in raw.vector_field.mlp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.min_lr),
    )

    meta_out = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "data": str(Path(args.data).resolve()),
        "spectral_coords": str(Path(spectral_coords).resolve()),
        "expression_transform": expr_transform,
        "dropout_ratio": float(args.dropout_ratio),
        "intensity_t": float(args.intensity_t),
        "trainable": "vector_field.mlp",
        "frozen": ["router", "mamba_encoder", "vector_field.time_embed", "vector_field.binary_mlp"],
        "prior_mean": float(prior_mean),
        "prior_std": float(prior_std),
    }
    with open(exp_dir / "finetune_info.json", "w") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    print(f"[finetune] exp_dir={exp_dir}")
    print(
        "[finetune] trainable=vector_field.mlp "
        f"dropout_ratio={args.dropout_ratio:.2f} intensity_t={args.intensity_t:.3f}"
    )

    best_val = None
    global_step = 0
    for epoch in range(1, int(args.epochs) + 1):
        raw.train()
        raw.router.eval()
        raw.mamba_encoder.eval()
        raw.vector_field.time_embed.eval()
        raw.vector_field.binary_mlp.eval()

        epoch_loss = 0.0
        epoch_huber = 0.0
        epoch_pearson = 0.0
        epoch_cell_pearson = 0.0
        epoch_masked = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Dropout intensity ft {epoch}/{args.epochs}", leave=False)
        for g_id, x_1, _, _ in pbar:
            g_id = g_id.to(trainer.device)
            x_1 = x_1.to(trainer.device)
            dropout_mask = _sample_dropout_mask(x_1, args.dropout_ratio)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = _one_step_intensity_loss(
                    raw,
                    g_id,
                    x_1,
                    dropout_mask,
                    args.intensity_t,
                    args.pearson_weight,
                    args.cell_pearson_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            global_step += 1
            n_batches += 1
            epoch_loss += float(loss.item())
            epoch_huber += float(metrics["huber"].item())
            epoch_pearson += float(metrics["pearson"].item())
            epoch_cell_pearson += float(metrics["cell_pearson"].item())
            epoch_masked += int(metrics["n_masked"])
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                huber=f"{metrics['huber'].item():.4f}",
                pearson=f"{metrics['pearson'].item():.4f}",
                n=int(metrics["n_masked"]),
            )

        scheduler.step()
        train_record = {
            "epoch": int(epoch),
            "step": int(global_step),
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_huber": epoch_huber / max(n_batches, 1),
            "train_pearson_loss": epoch_pearson / max(n_batches, 1),
            "train_cell_pearson_loss": epoch_cell_pearson / max(n_batches, 1),
            "train_masked_nonzero": int(epoch_masked),
        }

        val_record = {}
        if val_loader is not None:
            val_record = validate_one_step(
                raw,
                val_loader,
                trainer.device,
                args.dropout_ratio,
                args.intensity_t,
                args.pearson_weight,
                args.cell_pearson_weight,
                args.val_max_batches,
            )

        record = {**train_record, **val_record}
        with open(exp_dir / "finetune_log.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        val_key = val_record.get("val_intensity_pearson_cell")
        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"loss={train_record['train_loss']:.4f} "
            f"huber={train_record['train_huber']:.4f} "
            f"pearson_loss={train_record['train_pearson_loss']:.4f} "
            f"val_r={0.0 if val_key is None else val_key:.4f}"
        )

        payload = {
            "format_version": 2,
            "training_mode": "dropout_intensity_finetune",
            "phase": "dropout_intensity",
            "epoch": int(epoch),
            "step": int(global_step),
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "prior_mean": float(prior_mean),
            "prior_std": float(prior_std),
            "expression_transform": expr_transform,
            "finetune_info": meta_out,
            "record": record,
        }
        torch.save(payload, ckpt_dir / "checkpoint_last.pt")

        if val_key is not None and (best_val is None or float(val_key) > float(best_val)):
            best_val = float(val_key)
            torch.save(payload, ckpt_dir / "checkpoint_best.pt")
            with open(exp_dir / "best_metrics.json", "w") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

    torch.save(raw.state_dict(), exp_dir / "model_final.pt")
    print(f"最终模型 -> {exp_dir / 'model_final.pt'}")


if __name__ == "__main__":
    main()
