import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr as scipy_spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "model.py").exists() and (p / "data.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_atac import (  # noqa: E402
    compute_accessibility_prior_stats,
    load_atac_h5ad_split,
    make_dataloader,
)
from train_atac import (  # noqa: E402
    _summarize_depth_binned_pearson,
    scATACPipelineTrainer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose scATAC checkpoint validation behavior with rollout vs 1-step prediction."
    )
    parser.add_argument("--config", type=str, required=True, help="scATAC config yaml")
    parser.add_argument("--data", type=str, required=True, help="ATAC h5ad path")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint .pt path")
    parser.add_argument("--output_json", type=str, required=True, help="diagnostic JSON output path")
    parser.add_argument("--spectral_coords", type=str, default=None, help="override spectral coords .npy")
    parser.add_argument("--expr_transform", type=str, default=None, help="override expression_transform")
    parser.add_argument("--batch_size", type=int, default=None, help="override validation batch size")
    parser.add_argument("--max_batches", type=int, default=None, help="override validation max batches")
    parser.add_argument("--max_cells", type=int, default=None, help="optional data subsampling for quick checks")
    parser.add_argument("--local_rank", type=int, default=0, help="CUDA local rank after CUDA_VISIBLE_DEVICES")
    return parser.parse_args()


def _safe_corrcoef(x, y):
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return None if np.isnan(r) else r


def _safe_spearman(x, y):
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return None
    r = scipy_spearmanr(x, y).correlation
    if r is None or np.isnan(float(r)):
        return None
    return float(r)


class PredictionAccumulator:
    def __init__(self, prefix):
        self.prefix = prefix
        self.total_sq = 0.0
        self.total_abs = 0.0
        self.total_cnt = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.total_corr_cnt = 0
        self.cell_pearsons = []
        self.cell_spearmans = []
        self.cell_pearsons_top20 = []
        self.depth_proxy_values = []
        self.depth_proxy_pearsons = []
        self.cell_pred_means = []
        self.cell_true_means = []
        self.cell_pred_stds = []
        self.cell_true_stds = []
        self.masked_true_chunks = []
        self.masked_pred_chunks = []
        self.masked_score_chunks = []
        self.masked_nonzero_true_chunks = []
        self.masked_nonzero_pred_chunks = []
        self.masked_nonzero_score_chunks = []
        self.masked_zero_score_chunks = []

    def update(self, pred_full, true_full, eval_mask, score_full=None, corr_pred_full=None):
        pred_cpu = pred_full.detach().float().cpu().numpy()
        corr_cpu = (
            corr_pred_full.detach().float().cpu().numpy()
            if corr_pred_full is not None
            else pred_cpu
        )
        if score_full is None:
            score_cpu = pred_cpu
        else:
            score_cpu = score_full.detach().float().cpu().numpy()
        true_cpu = true_full.detach().float().cpu().numpy()
        mask_cpu = (eval_mask.detach().cpu().numpy() == 1)

        for b in range(true_cpu.shape[0]):
            mb = mask_cpu[b]
            if not np.any(mb):
                continue

            pred_masked = pred_cpu[b, mb].astype(np.float64, copy=False)
            score_masked = score_cpu[b, mb].astype(np.float64, copy=False)
            true_masked = true_cpu[b, mb].astype(np.float64, copy=False)
            diff = pred_masked - true_masked
            self.total_sq += float(np.sum(diff * diff))
            self.total_abs += float(np.sum(np.abs(diff)))
            self.total_cnt += int(true_masked.size)
            self.masked_pred_chunks.append(pred_masked.astype(np.float32, copy=False))
            self.masked_score_chunks.append(score_masked.astype(np.float32, copy=False))
            self.masked_true_chunks.append(true_masked.astype(np.float32, copy=False))
            zero_mask = mb & (true_cpu[b] <= 0)
            if np.any(zero_mask):
                self.masked_zero_score_chunks.append(score_cpu[b, zero_mask].astype(np.float32, copy=False))

            corr_mask = mb & (true_cpu[b] > 0)
            if int(np.sum(corr_mask)) < 2:
                continue

            pv = corr_cpu[b, corr_mask].astype(np.float64, copy=False)
            sv = score_cpu[b, corr_mask].astype(np.float64, copy=False)
            tv = true_cpu[b, corr_mask].astype(np.float64, copy=False)
            self.masked_nonzero_pred_chunks.append(pv.astype(np.float32, copy=False))
            self.masked_nonzero_score_chunks.append(sv.astype(np.float32, copy=False))
            self.masked_nonzero_true_chunks.append(tv.astype(np.float32, copy=False))

            self.sum_x += float(pv.sum())
            self.sum_y += float(tv.sum())
            self.sum_x2 += float(np.square(pv).sum())
            self.sum_y2 += float(np.square(tv).sum())
            self.sum_xy += float((pv * tv).sum())
            self.total_corr_cnt += int(tv.size)
            self.cell_pred_means.append(float(np.mean(pv)))
            self.cell_true_means.append(float(np.mean(tv)))
            self.cell_pred_stds.append(float(np.std(pv)))
            self.cell_true_stds.append(float(np.std(tv)))

            r = _safe_corrcoef(pv, tv)
            if r is not None:
                self.cell_pearsons.append(r)
                self.depth_proxy_values.append(float(np.count_nonzero(true_cpu[b] > 0)))
                self.depth_proxy_pearsons.append(r)

            sr = _safe_spearman(pv, tv)
            if sr is not None:
                self.cell_spearmans.append(sr)

            k = min(20, tv.size)
            if k >= 2:
                idx_top = np.argsort(tv)[-k:]
                r20 = _safe_corrcoef(pv[idx_top], tv[idx_top])
                if r20 is not None:
                    self.cell_pearsons_top20.append(r20)

    def summarize(self):
        p = self.prefix
        mse = self.total_sq / max(self.total_cnt, 1)
        denom = np.sqrt(
            max(self.total_corr_cnt * self.sum_x2 - self.sum_x**2, 0.0)
            * max(self.total_corr_cnt * self.sum_y2 - self.sum_y**2, 0.0)
        )
        pearson_all = None if denom <= 0 else float(
            (self.total_corr_cnt * self.sum_xy - self.sum_x * self.sum_y) / denom
        )

        if self.masked_true_chunks:
            y_true_all = np.concatenate(self.masked_true_chunks).astype(np.float32, copy=False)
            y_pred_all = np.concatenate(self.masked_pred_chunks).astype(np.float32, copy=False)
            y_score_all = np.concatenate(self.masked_score_chunks).astype(np.float32, copy=False)
            y_true_bin = (y_true_all > 0).astype(np.int32, copy=False)
            positive_rate = float(y_true_bin.mean()) if y_true_bin.size else 0.0
            y_pred_zero = y_pred_all[y_true_all <= 0]
            if y_pred_zero.size:
                pred_mean_zero_global = float(y_pred_zero.mean())
                pred_std_zero_global = float(y_pred_zero.std())
            else:
                pred_mean_zero_global = None
                pred_std_zero_global = None
            try:
                auroc = float(roc_auc_score(y_true_bin, y_score_all))
                auprc = float(average_precision_score(y_true_bin, y_score_all))
            except ValueError:
                auroc = 0.5
                auprc = positive_rate
        else:
            positive_rate = 0.0
            auroc = 0.5
            auprc = 0.0
            pred_mean_zero_global = None
            pred_std_zero_global = None

        if self.masked_nonzero_true_chunks:
            y_true_nz = np.concatenate(self.masked_nonzero_true_chunks).astype(np.float32, copy=False)
            y_pred_nz = np.concatenate(self.masked_nonzero_pred_chunks).astype(np.float32, copy=False)
            pred_mean_global = float(y_pred_nz.mean())
            true_mean_global = float(y_true_nz.mean())
            pred_std_global = float(y_pred_nz.std())
            true_std_global = float(y_true_nz.std())
        else:
            pred_mean_global = None
            true_mean_global = None
            pred_std_global = None
            true_std_global = None
        if self.masked_nonzero_score_chunks and self.masked_zero_score_chunks:
            y_score_nz = np.concatenate(self.masked_nonzero_score_chunks).astype(np.float32, copy=False)
            y_score_z = np.concatenate(self.masked_zero_score_chunks).astype(np.float32, copy=False)
            open_prob_nonzero_global = float(y_score_nz.mean())
            open_prob_zero_global = float(y_score_z.mean())
            binary_margin = float(open_prob_nonzero_global - open_prob_zero_global)
        else:
            open_prob_nonzero_global = None
            open_prob_zero_global = None
            binary_margin = None

        return {
            f"{p}_mse": float(mse),
            f"{p}_rmse": float(np.sqrt(mse)),
            f"{p}_mae": float(self.total_abs / max(self.total_cnt, 1)),
            f"{p}_pearson_cell": float(np.mean(self.cell_pearsons)) if self.cell_pearsons else None,
            f"{p}_pearson_all": pearson_all,
            f"{p}_spearman_cell": float(np.mean(self.cell_spearmans)) if self.cell_spearmans else None,
            f"{p}_pearson_top20": float(np.mean(self.cell_pearsons_top20)) if self.cell_pearsons_top20 else None,
            f"{p}_auroc": auroc,
            f"{p}_auprc": auprc,
            f"{p}_masked_total": int(self.total_cnt),
            f"{p}_masked_nonzero": int(sum(x.size for x in self.masked_nonzero_true_chunks)),
            f"{p}_positive_rate": positive_rate,
            f"{p}_pred_mean_nonzero_global": pred_mean_global,
            f"{p}_true_mean_nonzero_global": true_mean_global,
            f"{p}_pred_std_nonzero_global": pred_std_global,
            f"{p}_true_std_nonzero_global": true_std_global,
            f"{p}_pred_mean_zero_global": pred_mean_zero_global,
            f"{p}_pred_std_zero_global": pred_std_zero_global,
            f"{p}_open_prob_nonzero_global": open_prob_nonzero_global,
            f"{p}_open_prob_zero_global": open_prob_zero_global,
            f"{p}_binary_margin": binary_margin,
            f"{p}_pred_mean_nonzero_cellavg": float(np.mean(self.cell_pred_means)) if self.cell_pred_means else None,
            f"{p}_true_mean_nonzero_cellavg": float(np.mean(self.cell_true_means)) if self.cell_true_means else None,
            f"{p}_pred_std_nonzero_cellavg": float(np.mean(self.cell_pred_stds)) if self.cell_pred_stds else None,
            f"{p}_true_std_nonzero_cellavg": float(np.mean(self.cell_true_stds)) if self.cell_true_stds else None,
            f"{p}_depth_binned_pearson_cell": _summarize_depth_binned_pearson(
                self.depth_proxy_values,
                self.depth_proxy_pearsons,
            ),
        }


def _build_trainer(cfg, n_features, prior_mean, prior_std, spectral_coords, checkpoint_path, expr_transform, local_rank):
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic script currently requires CUDA. Set CUDA_VISIBLE_DEVICES first.")

    model_cfg = cfg["model"]
    val_cfg = cfg.get("validation", {})
    infer_cfg = cfg.get("inference", {})

    trainer = scATACPipelineTrainer(
        n_genes=n_features,
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        local_rank=int(local_rank),
        exp_dir=str(Path(checkpoint_path).resolve().parent.parent),
    )
    trainer.raw_model.mamba_encoder.activation_checkpointing = bool(
        model_cfg.get("activation_checkpointing", True)
    )
    trainer.val_infer_n_steps = int(val_cfg.get("n_steps", infer_cfg.get("n_steps", 10)))
    trainer.val_infer_temperature = float(val_cfg.get("temperature", infer_cfg.get("temperature", 0.1)))
    trainer.val_nonzero_mask_ratio = float(val_cfg.get("nonzero_mask_ratio", 0.2))
    trainer.val_zero_mask_ratio = float(val_cfg.get("zero_mask_ratio", 0.1))
    trainer.val_mask_seed = int(val_cfg.get("mask_seed", 1234))
    trainer.val_noise_seed = int(val_cfg.get("noise_seed", 5678))
    trainer.val_prior_mean = float(prior_mean)
    trainer.val_prior_std = float(max(prior_std, 1e-6))
    trainer.inference_binary_threshold = float(
        infer_cfg.get("open_threshold", infer_cfg.get("binary_threshold", 0.5))
    )
    trainer.expression_transform = expr_transform

    coords = np.load(spectral_coords)
    if coords.shape != (n_features, 2):
        raise ValueError(f"spectral_coords shape {coords.shape} != ({n_features}, 2)")
    trainer.raw_model.init_from_spectral(
        coords,
        slot_order_mode="hilbert",
        gene_order_mode="hilbert",
    )

    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    state = ckpt.get("model_state_dict", ckpt)
    trainer.raw_model.load_state_dict(state)
    trainer.raw_model.eval()
    trainer.model.eval()
    return trainer, ckpt


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    val_cfg = cfg.get("validation", {})
    expr_transform = str(args.expr_transform or train_cfg.get("expression_transform", "identity"))
    spectral_coords = args.spectral_coords or train_cfg.get("spectral_coords")
    if not spectral_coords:
        raise ValueError("spectral_coords is required, either in config or via --spectral_coords")

    seed = int(train_cfg.get("seed", 42))
    cell_seed = int(train_cfg.get("cell_seed", seed))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    val_cells = train_cfg.get("val_cells")
    max_cells = args.max_cells if args.max_cells is not None else train_cfg.get("max_cells")
    batch_size = int(args.batch_size or val_cfg.get("batch_size", train_cfg.get("batch_size_per_gpu", 1)))
    max_batches = int(args.max_batches or val_cfg.get("max_batches", 200))

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
    if val_X is None or val_X.shape[0] == 0:
        raise ValueError("Validation split is empty. Set training.val_ratio > 0.")

    prior_mean, prior_std = compute_accessibility_prior_stats(
        train_X,
        expr_transform=expr_transform,
    )
    trainer, ckpt = _build_trainer(
        cfg=cfg,
        n_features=int(meta["n_features"]),
        prior_mean=prior_mean,
        prior_std=prior_std,
        spectral_coords=spectral_coords,
        checkpoint_path=args.checkpoint,
        expr_transform=expr_transform,
        local_rank=args.local_rank,
    )
    trainer.val_max_batches = max_batches

    val_loader, _ = make_dataloader(
        val_X,
        feature_ids,
        batch_size=batch_size,
        shuffle=False,
        world_size=1,
        nonzero_mask_ratio=float(train_cfg.get("nonzero_mask_ratio", 0.2)),
        zero_mask_ratio=float(train_cfg.get("zero_mask_ratio", 0.1)),
        return_index=True,
        expr_transform=expr_transform,
    )

    rollout_acc = PredictionAccumulator("rollout")
    one_step_acc = PredictionAccumulator("one_step")
    total_eval_cells = 0
    global_cell_offset = 0

    with torch.no_grad():
        for n_batches, batch in enumerate(tqdm(val_loader, desc="Diagnose validation", total=max_batches), start=1):
            if len(batch) == 5:
                g_id, x_1, _x_0, _mask_ind, cell_idx = batch
            else:
                g_id, x_1, _x_0, _mask_ind = batch
                cell_idx = None

            g_id = g_id.to(trainer.device)
            x_1 = x_1.to(trainer.device)
            bs = x_1.shape[0]
            total_eval_cells += int(bs)

            if cell_idx is not None:
                eval_mask = trainer._make_val_mask(x_1, cell_indices=cell_idx)
            else:
                eval_mask = trainer._make_val_mask(x_1, global_offset=global_cell_offset)

            observed = x_1.clone()
            observed[eval_mask == 1] = 0.0
            init_masked_values = trainer._make_val_initial_masked_values(
                x_1,
                eval_mask,
                cell_indices=cell_idx,
                global_offset=global_cell_offset,
            )
            global_cell_offset += bs

            pred_rollout_intensity = trainer.raw_model.inference(
                g_id,
                observed,
                eval_mask,
                n_steps=trainer.val_infer_n_steps,
                temperature=trainer.val_infer_temperature,
                prior_mean=trainer.val_prior_mean,
                prior_std=trainer.val_prior_std,
                initial_masked_values=init_masked_values,
            )
            t_binary = torch.zeros((bs, 1), device=trainer.device, dtype=x_1.dtype)
            rollout_logits = trainer.raw_model.predict_binary_logits(
                g_id,
                observed,
                eval_mask,
                t_binary,
                temperature=0.0,
            )
            rollout_score = torch.sigmoid(rollout_logits)
            pred_rollout = torch.where(
                rollout_score > trainer.inference_binary_threshold,
                pred_rollout_intensity,
                torch.zeros_like(pred_rollout_intensity),
            )
            rollout_acc.update(
                pred_rollout,
                x_1,
                eval_mask,
                score_full=rollout_score,
                corr_pred_full=pred_rollout_intensity,
            )

            t0 = torch.zeros((bs, 1), device=trainer.device, dtype=x_1.dtype)
            x1_hats, binary_logits, _x1_targets, _x1_hats_dup, _x1_targets_dup, _ = trainer.raw_model.forward_train_dual(
                g_id,
                x_1,
                eval_mask,
                t0,
                init_masked_values,
                temperature=0.0,
            )
            pred_one_step = observed.clone()
            one_step_logit_full = torch.zeros_like(observed)
            for b in range(bs):
                m_idx = (eval_mask[b] == 1).nonzero(as_tuple=True)[0]
                if m_idx.numel() > 0:
                    pred_one_step[b, m_idx] = x1_hats[b].to(dtype=pred_one_step.dtype)
                    one_step_logit_full[b, m_idx] = binary_logits[b].to(dtype=one_step_logit_full.dtype)
            one_step_score = torch.sigmoid(one_step_logit_full)
            one_step_intensity = pred_one_step
            pred_one_step = torch.where(
                one_step_score > trainer.inference_binary_threshold,
                one_step_intensity,
                torch.zeros_like(one_step_intensity),
            )
            one_step_acc.update(
                pred_one_step,
                x_1,
                eval_mask,
                score_full=one_step_score,
                corr_pred_full=one_step_intensity,
            )

            if n_batches >= max_batches:
                break

    result = {
        "config": str(Path(args.config).resolve()),
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "spectral_coords": str(Path(spectral_coords).resolve()),
        "expression_transform": expr_transform,
        "checkpoint_epoch": int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else None,
        "checkpoint_step": int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else None,
        "n_features": int(meta["n_features"]),
        "train_cells": int(train_X.shape[0]),
        "val_cells_total": int(val_X.shape[0]),
        "eval_cells": int(total_eval_cells),
        "eval_batches": int(min(max_batches, len(val_loader))),
        "batch_size": int(batch_size),
        "max_batches": int(max_batches),
        "val_nonzero_mask_ratio": float(trainer.val_nonzero_mask_ratio),
        "val_zero_mask_ratio": float(trainer.val_zero_mask_ratio),
        "val_mask_seed": int(trainer.val_mask_seed),
        "val_noise_seed": int(trainer.val_noise_seed),
        "prior_mean": float(prior_mean),
        "prior_std": float(prior_std),
        "rollout_n_steps": int(trainer.val_infer_n_steps),
        "rollout_temperature": float(trainer.val_infer_temperature),
        "split_mode": meta.get("split_mode"),
        "split_info_effective": meta.get("split_info_effective"),
    }
    result.update(rollout_acc.summarize())
    result.update(one_step_acc.summarize())

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
