import argparse
import json
import os
import time
from pathlib import Path


def _configure_cpu_thread_env():
    defaults = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


_configure_cpu_thread_env()

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Subset
import yaml
from tqdm import tqdm

from data import (
    TemporalPairDataset,
    TopKExpressionDataset,
    _load_preprocessed_temporal_bundle,
    compute_log1p_nonzero_bin_edges,
    compute_log1p_nonzero_prior_stats,
    load_h5ad,
    load_h5ad_split,
    load_mouse_development_temporal_datasets,
    load_sea_ad,
    load_sea_ad_split,
    make_dataloader,
    make_temporal_dataloader,
)
from model import scFiTModel
from route_a.protocol import load_protocol_bundle
from route_a.train import RouteATrainer
from utils import (
    bin_expression_values,
    build_deterministic_mask,
    build_random_mask,
    cell_pearson_loss,
    masked_cross_entropy_loss,
    masked_metrics,
    mmd_rbf_loss,
    pearson_loss,
    sample_gaussian_prior,
    sample_t_stratified,
    seed_everything,
    type_classification_accuracy,
    type_classification_loss,
    weighted_huber_loss,
    cosine_direction_loss,
    pseudotime_regression_loss,
    pseudotime_ranking_loss,
)


try:
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def build_experiment_dir(root: str, run_name: str) -> str:
    exp_dir = os.path.join(root, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    return exp_dir


class GpuMemoryReservation:
    def __init__(self, device: torch.device, enabled: bool = False, reserve_gb: float = None, fraction_of_free: float = 0.8):
        self.device = device
        self.enabled = bool(enabled) and device.type == "cuda"
        self.reserve_gb = None if reserve_gb is None else float(reserve_gb)
        self.fraction_of_free = float(fraction_of_free)
        self.buffer = None
        self.reserved_bytes = 0

    def reserve(self):
        if not self.enabled:
            return

        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        if self.reserve_gb is not None:
            target_bytes = int(self.reserve_gb * (1024 ** 3))
        else:
            safe_fraction = min(max(self.fraction_of_free, 0.0), 0.95)
            target_bytes = int(free_bytes * safe_fraction)

        reserve_cap = int(free_bytes * 0.9)
        target_bytes = max(0, min(target_bytes, reserve_cap))
        if target_bytes <= 0:
            print("[gpu_reserve] skip: no free memory available to reserve")
            return

        try:
            self.buffer = torch.empty(target_bytes, dtype=torch.uint8, device=self.device)
            self.buffer.zero_()
            self.reserved_bytes = target_bytes
            print(
                "[gpu_reserve] "
                f"reserved={target_bytes / (1024 ** 3):.2f}GB "
                f"free_before={free_bytes / (1024 ** 3):.2f}GB "
                f"total={total_bytes / (1024 ** 3):.2f}GB"
            )
        except RuntimeError as exc:
            self.buffer = None
            self.reserved_bytes = 0
            print(f"[gpu_reserve] failed: {exc}")

    def release(self):
        if self.buffer is None:
            return
        self.buffer = None
        self.reserved_bytes = 0
        torch.cuda.empty_cache()
        print("[gpu_reserve] released reservation")


class Trainer:
    def __init__(self, model, device, exp_dir, cfg):
        self.model = model.to(device)
        self.device = device
        self.exp_dir = exp_dir
        self.log_path = os.path.join(exp_dir, "train_log.jsonl")

        train_cfg = cfg["training"]
        val_cfg = cfg["validation"]
        infer_cfg = cfg["inference"]
        model_cfg = cfg["model"]

        self.mask_ratio = float(train_cfg.get("mask_ratio", 0.2))
        self.val_mask_ratio = float(val_cfg.get("mask_ratio", self.mask_ratio))
        self.val_mask_seed = int(val_cfg.get("mask_seed", 1234))
        self.val_max_batches = int(val_cfg.get("max_batches", 50))
        self.infer_n_steps = int(infer_cfg.get("n_steps", 10))
        self.amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        self.monitor_cls_attention = bool(train_cfg.get("monitor_cls_attention", False))
        self.log_every_steps = int(train_cfg.get("log_every_steps", 100))
        self.cls_attention_monitor_max_gb = float(train_cfg.get("cls_attention_monitor_max_gb", 1.0))
        self._cls_monitor_skip_warned = False

        loss_cfg = train_cfg.get("loss_weights", {})
        self.loss_weighted = float(loss_cfg.get("weighted_huber", 1.0))
        self.loss_pearson = float(loss_cfg.get("pearson", 0.1))
        self.loss_cell_pearson = float(loss_cfg.get("cell_pearson", 0.1))
        self.loss_cross_entropy = float(loss_cfg.get("cross_entropy", 0.0))

        self.prior_mean = 0.0
        self.prior_std = 1.0
        self.bin_edges = None
        aux_cfg = model_cfg.get("aux_bin_ce", {})
        self.use_aux_bin_ce = bool(aux_cfg.get("enabled", False))
        self.n_bins = int(aux_cfg.get("n_bins", 0))

    def _should_collect_cls_attention(self, batch_size: int, seq_len: int) -> bool:
        if not self.monitor_cls_attention:
            return False
        if self.device.type != "cuda":
            return True

        # MultiheadAttention with need_weights=True materializes full attention weights
        # of shape (B, H, S, S), which is extremely expensive for long sequences.
        approx_bytes = batch_size * int(self.model.blocks[0].self_attn.num_heads) * seq_len * seq_len * 4
        approx_gb = approx_bytes / (1024 ** 3)
        if approx_gb <= self.cls_attention_monitor_max_gb:
            return True

        if not self._cls_monitor_skip_warned:
            print(
                "[cls_monitor] skip attention-weight logging: "
                f"estimated extra memory {approx_gb:.2f}GB exceeds "
                f"training.cls_attention_monitor_max_gb={self.cls_attention_monitor_max_gb:.2f}GB"
            )
            self._cls_monitor_skip_warned = True
        return False

    def _log(self, record: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, epoch: int, global_step: int, optimizer, scheduler=None, tag: str = None):
        ckpt_name = tag or f"epoch_{epoch}"
        path = os.path.join(self.exp_dir, "checkpoints", f"{ckpt_name}.pt")
        payload = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "step": int(global_step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
        }
        if self.bin_edges is not None:
            payload["bin_edges"] = self.bin_edges.detach().cpu()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, path)
        print(f"[checkpoint] {path}")

    def _prepare_batch(self, batch):
        if len(batch) == 4:
            gene_ids, x_1, valid_mask, cell_idx = batch
        else:
            gene_ids, x_1, valid_mask = batch
            cell_idx = None

        gene_ids = gene_ids.to(self.device, non_blocking=True)
        x_1 = x_1.to(self.device, non_blocking=True)
        valid_mask = valid_mask.to(self.device, non_blocking=True).bool()
        cls_mask = gene_ids.eq(self.model.cls_token_id)
        gene_valid_mask = valid_mask & (~cls_mask)
        padding_mask = ~valid_mask
        return gene_ids, x_1, gene_valid_mask, padding_mask, cell_idx

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        stats = []
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="validate", leave=False)):
            if self.val_max_batches > 0 and batch_idx >= self.val_max_batches:
                break
            gene_ids, x_1, gene_valid_mask, padding_mask, cell_idx = self._prepare_batch(batch)
            if cell_idx is None:
                cell_idx = torch.arange(gene_ids.shape[0], device=self.device)
            else:
                cell_idx = cell_idx.to(self.device, non_blocking=True)

            mask_indicator = build_deterministic_mask(
                gene_valid_mask,
                cell_idx,
                mask_ratio=self.val_mask_ratio,
                base_seed=self.val_mask_seed,
            )

            pred = self.model.inference(
                gene_ids,
                x_1,
                padding_mask=padding_mask,
                mask_indicator=mask_indicator,
                n_steps=self.infer_n_steps,
                prior_mean=self.prior_mean,
                prior_std=self.prior_std,
            )
            stats.append(masked_metrics(pred, x_1, mask_indicator))

        self.model.train()
        if not stats:
            return {}

        def mean_or_none(key):
            values = [s[key] for s in stats if s.get(key) is not None]
            if not values:
                return None
            return float(sum(values) / len(values))

        return {
            "val_mse": mean_or_none("val_mse"),
            "val_rmse": mean_or_none("val_rmse"),
            "val_mae": mean_or_none("val_mae"),
            "val_pearson_all": mean_or_none("val_pearson_all"),
            "val_pearson_cell": mean_or_none("val_pearson_cell"),
            "val_spearman_cell": mean_or_none("val_spearman_cell"),
            "val_pearson_top20_cell": mean_or_none("val_pearson_top20_cell"),
        }

    def train(self, train_loader, val_loader, epochs, lr, weight_decay, val_every_steps, save_every_epochs, resume=None):
        epochs = int(epochs)
        val_every_steps = int(val_every_steps)
        save_every_epochs = int(save_every_epochs)
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = max(epochs * steps_per_epoch, 1)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        global_step = 0
        start_epoch = 0
        if resume is not None:
            ckpt = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.prior_mean = float(ckpt.get("prior_mean", self.prior_mean))
            self.prior_std = float(ckpt.get("prior_std", self.prior_std))
            if "bin_edges" in ckpt:
                self.bin_edges = ckpt["bin_edges"].to(self.device)
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
            print(f"[resume] {resume} epoch={start_epoch} global_step={global_step}")

        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        overall_pbar = tqdm(total=total_steps, initial=global_step, desc="train")

        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss_sum = 0.0
            epoch_batches = 0
            epoch_pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)

            for batch_idx, batch in enumerate(epoch_pbar, start=1):
                gene_ids, x_1, gene_valid_mask, padding_mask, _ = self._prepare_batch(batch)
                t = sample_t_stratified(gene_ids.shape[0], self.device)
                x_0 = sample_gaussian_prior(gene_valid_mask, mean=self.prior_mean, std=self.prior_std)
                mask_indicator = build_random_mask(gene_valid_mask, self.mask_ratio)
                supervision_mask = mask_indicator & gene_valid_mask
                collect_cls_attention = (
                    ((global_step + 1) % self.log_every_steps == 0)
                    and self._should_collect_cls_attention(gene_ids.shape[0], gene_ids.shape[1])
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                    pred, _, bin_logits, _, _, _, cls_attention_stats = self.model.forward_train(
                        gene_ids,
                        x_1,
                        padding_mask=padding_mask,
                        mask_indicator=mask_indicator,
                        t=t,
                        x_0=x_0,
                        return_cls_attention=collect_cls_attention,
                    )
                    loss_weighted = weighted_huber_loss(pred, x_1, supervision_mask)
                    flat_pred = pred[supervision_mask]
                    flat_true = x_1[supervision_mask]
                    loss_pearson = pearson_loss(flat_pred, flat_true)
                    loss_cell, cell_stats = cell_pearson_loss(
                        pred,
                        x_1,
                        supervision_mask,
                        return_stats=True,
                    )
                    loss_ce = pred.new_tensor(0.0)
                    if self.use_aux_bin_ce and self.bin_edges is not None and bin_logits is not None:
                        target_bins = bin_expression_values(x_1, self.bin_edges)
                        loss_ce = masked_cross_entropy_loss(bin_logits, target_bins, supervision_mask)
                    loss = (
                        self.loss_weighted * loss_weighted
                        + self.loss_pearson * loss_pearson
                        + self.loss_cell_pearson * loss_cell
                        + self.loss_cross_entropy * loss_ce
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1
                epoch_loss_sum += float(loss.item())
                epoch_batches += 1
                overall_pbar.update(1)

                epoch_pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    wh=f"{loss_weighted.item():.4f}",
                    pr=f"{loss_pearson.item():.4f}",
                    cpr=f"{loss_cell.item():.4f}",
                    ce=f"{loss_ce.item():.4f}",
                    cpr_used=f"{cell_stats['used_ratio']:.2f}",
                    cls_attn=(
                        f"{float(cls_attention_stats['cls_attn_received_mean'].item()):.3f}"
                        if cls_attention_stats is not None and "cls_attn_received_mean" in cls_attention_stats
                        else "-"
                    ),
                )

                val_record = None
                if val_loader is not None and global_step % max(val_every_steps, 1) == 0:
                    val_metrics = self.validate(val_loader)
                    val_record = {
                        "epoch": epoch + 1,
                        "epoch_step": batch_idx,
                        "global_step": global_step,
                        **val_metrics,
                    }
                    print(
                        f"\n[val step={global_step}] "
                        f"rmse={val_metrics.get('val_rmse')} "
                        f"mae={val_metrics.get('val_mae')} "
                        f"pearson_cell={val_metrics.get('val_pearson_cell')} "
                        f"spearman_cell={val_metrics.get('val_spearman_cell')} "
                        f"pearson_top20={val_metrics.get('val_pearson_top20_cell')}"
                    )

                if global_step % self.log_every_steps == 0:
                    record = {
                        "epoch": epoch + 1,
                        "epoch_step": batch_idx,
                        "global_step": global_step,
                        "train_loss": float(loss.item()),
                        "weighted_huber_loss": float(loss_weighted.item()),
                        "pearson_loss": float(loss_pearson.item()),
                        "cell_pearson_loss": float(loss_cell.item()),
                        "cross_entropy_loss": float(loss_ce.item()),
                        "cell_pearson_used_ratio": float(cell_stats["used_ratio"]),
                        "cell_pearson_skipped_ratio": float(cell_stats["skipped_ratio"]),
                        "cell_pearson_used_cells": int(cell_stats["n_used"]),
                        "cell_pearson_skipped_cells": int(cell_stats["n_skipped"]),
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                    if cls_attention_stats is not None:
                        if "cls_attn_received_mean" in cls_attention_stats:
                            record["cls_attn_received_mean"] = float(cls_attention_stats["cls_attn_received_mean"].item())
                        if "cls_attn_received_last" in cls_attention_stats:
                            record["cls_attn_received_last"] = float(cls_attention_stats["cls_attn_received_last"].item())
                    if val_record is not None:
                        record.update({k: v for k, v in val_record.items() if k.startswith("val_")})
                    self._log(record)
                elif val_record is not None:
                    self._log(val_record)

            epoch_pbar.close()

            epoch_record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "epoch_train_loss": float(epoch_loss_sum / max(epoch_batches, 1)),
                "epoch_batches": int(epoch_batches),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            self._log(epoch_record)

            if (epoch + 1) % max(save_every_epochs, 1) == 0:
                self._save_checkpoint(epoch + 1, global_step, optimizer, scheduler=scheduler)

        overall_pbar.close()
        self._save_checkpoint(epochs, global_step, optimizer, scheduler=scheduler, tag="final")


class TemporalTrainer:
    def __init__(self, model, device, exp_dir, cfg, data_meta=None):
        self.model = model.to(device)
        self.device = device
        self.exp_dir = exp_dir
        self.log_path = os.path.join(exp_dir, "train_log.jsonl")
        self.cfg = cfg
        self.data_meta = data_meta or {}

        train_cfg = cfg["training"]
        val_cfg = cfg.get("validation", {})
        infer_cfg = cfg.get("inference", {})

        self.amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        self.log_every_steps = int(train_cfg.get("log_every_steps", 100))
        self.val_max_batches = int(val_cfg.get("max_batches", 50))
        self.infer_n_steps = int(infer_cfg.get("n_steps", 10))
        self.use_temporal_eval_metrics = bool(val_cfg.get("use_temporal_eval_metrics", True))
        self.monitor_cls_attention = bool(train_cfg.get("monitor_cls_attention", False))
        self.cls_attention_monitor_max_gb = float(train_cfg.get("cls_attention_monitor_max_gb", 1.0))
        self._cls_monitor_skip_warned = False
        self._pre_meta = None
        self._pre_metadata_df = None
        self._validation_subset_cache = {}

        loss_cfg = train_cfg.get("loss_weights", {})
        self.loss_recon = float(loss_cfg.get("recon", 1.0))
        self.loss_recon_pearson = float(loss_cfg.get("recon_pearson", loss_cfg.get("pearson", 0.1)))
        self.loss_velocity = float(loss_cfg.get("velocity", 0.5))
        self.loss_dist = float(loss_cfg.get("distribution", 0.1))
        self.loss_type = float(loss_cfg.get("type", 0.1))
        self.loss_pseudotime = float(loss_cfg.get("pseudotime", 0.0))
        self.mmd_min_group_size = int(train_cfg.get("mmd_min_group_size", 8))
        self.dist_bank_size = int(train_cfg.get("distribution_bank_size", 2048))
        self.dist_bank_sample_size = int(train_cfg.get("distribution_bank_sample_size", 256))
        self.dist_bank_warmup_min = int(train_cfg.get("distribution_bank_warmup_min", max(self.mmd_min_group_size, 32)))
        self.marker_idx_by_cell_type = None
        self._pred_distribution_bank = {}
        self._real_distribution_bank = {}
        self.pseudotime_targets = None
        self.pseudotime_loss_enabled = False
        self.pseudotime_rank_weight = 0.5
        self.pseudotime_reg_beta = 0.1
        self.pseudotime_rank_margin = 0.0

        preprocessed_dir = cfg.get("data", {}).get("preprocessed_dir")
        if preprocessed_dir:
            try:
                _, self._pre_metadata_df, self._pre_meta = _load_preprocessed_temporal_bundle(preprocessed_dir)
            except Exception as exc:
                print(f"[validation] failed to preload preprocessed bundle metadata: {exc}")
                self._pre_meta = None
                self._pre_metadata_df = None
        self.marker_idx_by_cell_type = self._load_marker_gene_indices()
        self._configure_pseudotime_loss()

        subset_cfg = val_cfg.get("subset", {}) or {}
        self.val_subset_enabled = bool(subset_cfg.get("enabled", False))
        self.val_subset_samples_per_group = int(subset_cfg.get("samples_per_group", 4))
        self.val_subset_seed = int(subset_cfg.get("seed", train_cfg.get("seed", 42)))

    def _configure_pseudotime_loss(self):
        pseudo_cfg = self.cfg.get("training", {}).get("pseudotime_loss", {}) or {}
        model_cfg = self.cfg.get("model", {}).get("pseudotime_head", {}) or {}
        enabled = bool(pseudo_cfg.get("enabled", False)) and self.loss_pseudotime > 0.0
        enabled = enabled and bool(model_cfg.get("enabled", False))
        self.pseudotime_rank_weight = float(pseudo_cfg.get("rank_weight", 0.5))
        self.pseudotime_reg_beta = float(pseudo_cfg.get("beta", 0.1))
        self.pseudotime_rank_margin = float(pseudo_cfg.get("margin", 0.0))
        if not enabled:
            return

        pseudo_targets = self._load_pseudotime_targets()
        if pseudo_targets is None:
            print("[pseudotime_loss] disabled: failed to load aligned pseudotime targets")
            return
        self.pseudotime_targets = pseudo_targets
        self.pseudotime_loss_enabled = True

    def _load_pseudotime_targets(self):
        if self._pre_metadata_df is None:
            return None
        eval_cfg = self.cfg.get("evaluation", {}) or {}
        assets_cfg = eval_cfg.get("assets", {}) or {}
        pseudo_path = str(assets_cfg.get("pseudotime_path", "") or "").strip()
        if not pseudo_path:
            return None
        try:
            from eval_temporal import _load_pseudotime_values

            pseudo_df = _load_pseudotime_values(pseudo_path)
        except Exception as exc:
            print(f"[pseudotime_loss] failed to load pseudotime asset: {exc}")
            return None

        metadata_df = self._pre_metadata_df.copy()
        if "obs_name" in metadata_df.columns:
            obs_names = metadata_df["obs_name"].astype(str).to_numpy()
        else:
            obs_names = metadata_df.index.to_numpy().astype(str)
        pseudo_map = dict(zip(pseudo_df["obs_name"].astype(str).tolist(), pseudo_df["pseudotime"].astype(np.float32).tolist()))
        targets = np.full((obs_names.shape[0],), np.nan, dtype=np.float32)
        for i, obs_name in enumerate(obs_names.tolist()):
            value = pseudo_map.get(obs_name)
            if value is not None:
                targets[i] = float(value)
        finite = np.isfinite(targets)
        if not finite.any():
            return None
        return targets

    def _should_collect_cls_attention(self, batch_size: int, seq_len: int) -> bool:
        if not self.monitor_cls_attention:
            return False
        if self.device.type != "cuda":
            return True
        approx_bytes = batch_size * int(self.model.blocks[0].self_attn.num_heads) * seq_len * seq_len * 4
        approx_gb = approx_bytes / (1024 ** 3)
        if approx_gb <= self.cls_attention_monitor_max_gb:
            return True
        if not self._cls_monitor_skip_warned:
            print(
                "[cls_monitor] skip attention-weight logging: "
                f"estimated extra memory {approx_gb:.2f}GB exceeds "
                f"training.cls_attention_monitor_max_gb={self.cls_attention_monitor_max_gb:.2f}GB"
            )
            self._cls_monitor_skip_warned = True
        return False

    def _log(self, record: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _fmt_console_metric(value):
        if value is None:
            return "NA"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "NA"

    @staticmethod
    def _filter_temporal_val_record(record: dict) -> dict:
        allowed_keys = {
            "epoch",
            "epoch_step",
            "global_step",
            "val_eval_error",
            "val_per_cell_r2_mean",
            "val_per_cell_r2_by_t_tgt",
            "val_distribution_mean_r2_mean",
            "val_distribution_mean_r2_by_t_tgt",
            "val_marker_mmd_mean",
            "val_marker_mmd_by_t_tgt",
            "val_cell_type_accuracy",
            "val_cell_type_accuracy_per_class",
            "val_classifier_baseline_accuracy",
            "val_pseudotime_spearman",
            "val_pseudotime_spearman_per_class",
            "val_real_pseudotime_vs_time_spearman",
            "val_by_regime",
            "val_baselines",
            "val_asset_mode",
            "val_predict_seconds",
            "val_metric_seconds",
            "val_asset_prepare_seconds",
            "val_core_metric_seconds",
            "val_eval_seconds",
            "val_subset_pairs",
            "val_subset_times",
            "val_subset_groups",
            "val_subset_samples_per_group",
        }
        return {k: v for k, v in record.items() if k in allowed_keys}

    def _load_marker_gene_indices(self):
        eval_cfg = self.cfg.get("evaluation", {}) or {}
        assets_cfg = eval_cfg.get("assets", {}) or {}
        marker_path = str(assets_cfg.get("marker_gene_path", "") or "").strip()
        if not marker_path or self._pre_meta is None:
            return None

        marker_file = Path(marker_path)
        if not marker_file.exists():
            print(f"[marker_mmd] marker file missing, fallback to full-HVG MMD: {marker_file}")
            return None

        hvg_gene_names = self._pre_meta.get("hvg_gene_names")
        if hvg_gene_names is None:
            hvg_gene_names = self._pre_meta.get("hvg_var_names")
        if hvg_gene_names is None:
            print("[marker_mmd] missing HVG gene names in preprocessed bundle, fallback to full-HVG MMD")
            return None

        try:
            with marker_file.open("r", encoding="utf-8") as f:
                marker_asset = json.load(f)
        except Exception as exc:
            print(f"[marker_mmd] failed to load marker file, fallback to full-HVG MMD: {exc}")
            return None

        if isinstance(marker_asset, dict) and "markers" in marker_asset and isinstance(marker_asset["markers"], dict):
            marker_asset = marker_asset["markers"]
        if not isinstance(marker_asset, dict):
            print("[marker_mmd] invalid marker asset format, fallback to full-HVG MMD")
            return None

        gene_names = np.asarray(hvg_gene_names).astype(str)
        gene_to_idx = {name: idx for idx, name in enumerate(gene_names.tolist())}
        gene_to_idx_lower = {name.lower(): idx for idx, name in enumerate(gene_names.tolist())}

        marker_idx = {}
        missing_all = []
        for cell_type_name, genes in marker_asset.items():
            indices = []
            for gene in genes:
                gene = str(gene)
                idx = gene_to_idx.get(gene)
                if idx is None:
                    idx = gene_to_idx_lower.get(gene.lower())
                if idx is not None:
                    indices.append(int(idx))
            unique_idx = np.asarray(sorted(set(indices)), dtype=np.int64)
            if unique_idx.size == 0:
                missing_all.append(str(cell_type_name))
            marker_idx[str(cell_type_name)] = unique_idx

        if not marker_idx:
            print("[marker_mmd] no usable marker genes found, fallback to full-HVG MMD")
            return None
        if missing_all:
            print(f"[marker_mmd] warning: no marker genes mapped for {len(missing_all)} cell types")
        return marker_idx

    def _save_checkpoint(self, epoch: int, global_step: int, optimizer, scheduler=None, tag: str = None):
        ckpt_name = tag or f"epoch_{epoch}"
        path = os.path.join(self.exp_dir, "checkpoints", f"{ckpt_name}.pt")
        payload = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "step": int(global_step),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, path)
        print(f"[checkpoint] {path}")

    @staticmethod
    def _unwrap_dataset(dataset):
        current = dataset
        while isinstance(current, Subset):
            current = current.dataset
        return current

    def _build_validation_subset_loader(self, val_loader):
        if not self.val_subset_enabled:
            return val_loader, None

        dataset = getattr(val_loader, "dataset", None)
        base_dataset = self._unwrap_dataset(dataset)
        tgt_time_raw = getattr(base_dataset, "tgt_time_raw", None)
        cell_type_id = getattr(base_dataset, "cell_type_id", None)
        if tgt_time_raw is None or cell_type_id is None:
            return val_loader, None

        cache_key = (id(base_dataset), len(base_dataset))
        cached = self._validation_subset_cache.get(cache_key)
        if cached is None:
            rng = np.random.RandomState(self.val_subset_seed)
            tgt_time_raw = np.asarray(tgt_time_raw).astype(str)
            cell_type_id = np.asarray(cell_type_id, dtype=np.int64)
            group_to_indices = {}
            for idx, key in enumerate(zip(tgt_time_raw.tolist(), cell_type_id.tolist())):
                group_to_indices.setdefault(key, []).append(idx)

            chosen = []
            for indices in group_to_indices.values():
                idx_arr = np.asarray(indices, dtype=np.int64)
                if idx_arr.shape[0] <= self.val_subset_samples_per_group:
                    chosen.append(idx_arr)
                else:
                    picked = np.sort(rng.choice(idx_arr, size=self.val_subset_samples_per_group, replace=False))
                    chosen.append(picked.astype(np.int64))

            subset_indices = np.unique(np.concatenate(chosen, axis=0)).astype(np.int64, copy=False)
            subset_times = np.unique(tgt_time_raw[subset_indices]).shape[0]
            subset_groups = len(group_to_indices)
            cached = {
                "indices": subset_indices,
                "pairs": int(subset_indices.shape[0]),
                "times": int(subset_times),
                "groups": int(subset_groups),
            }
            self._validation_subset_cache[cache_key] = cached

        subset_dataset = Subset(base_dataset, cached["indices"].tolist())
        loader_cfg = (self.cfg.get("data", {}) or {}).get("loader", {}) or {}
        val_batch_size = int(self.cfg.get("validation", {}).get("batch_size", getattr(val_loader, "batch_size", 16) or 16))
        subset_loader = make_temporal_dataloader(
            subset_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            group_by_tgt_time=False,
            drop_last=False,
            seed=self.val_subset_seed,
            num_workers=int(loader_cfg.get("num_workers", 0)),
            prefetch_factor=loader_cfg.get("prefetch_factor"),
            persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
        )
        return subset_loader, {
            "val_subset_pairs": cached["pairs"],
            "val_subset_times": cached["times"],
            "val_subset_groups": cached["groups"],
            "val_subset_samples_per_group": self.val_subset_samples_per_group,
        }

    @staticmethod
    def _distribution_group_key(time_value: float, cell_type_id: int):
        return (round(float(time_value), 6), int(cell_type_id))

    def _append_distribution_bank(self, bank: dict, key, values: torch.Tensor):
        if values is None or values.numel() == 0:
            return
        values_cpu = values.detach().float().cpu()
        existing = bank.get(key)
        merged = values_cpu if existing is None else torch.cat([existing, values_cpu], dim=0)
        if merged.shape[0] > self.dist_bank_size:
            merged = merged[-self.dist_bank_size :]
        bank[key] = merged.contiguous()

    def _sample_distribution_bank(self, bank: dict, key, max_items: int, device: torch.device):
        values = bank.get(key)
        if values is None or values.shape[0] == 0:
            return None
        if max_items > 0 and values.shape[0] > max_items:
            idx = torch.randperm(values.shape[0])[:max_items]
            values = values.index_select(0, idx)
        return values.to(device=device, dtype=torch.float32, non_blocking=True)

    def _extract_marker_group_expr(
        self,
        expr: torch.Tensor,
        gene_valid_mask: torch.Tensor,
        group_mask: torch.Tensor,
        marker_idx_np: np.ndarray,
    ):
        marker_idx = torch.as_tensor(marker_idx_np + 1, device=expr.device, dtype=torch.long)
        group_expr = expr[group_mask].index_select(dim=1, index=marker_idx)
        valid_group = gene_valid_mask[group_mask].index_select(dim=1, index=marker_idx)
        return group_expr * valid_group.to(group_expr.dtype)

    def _update_distribution_banks(
        self,
        pred_tgt: torch.Tensor,
        tgt_expr: torch.Tensor,
        gene_valid_mask: torch.Tensor,
        cell_type_ids: torch.Tensor,
        t_tgt: torch.Tensor,
    ):
        if not self.marker_idx_by_cell_type:
            return
        cell_type_names = np.asarray(self.data_meta.get("cell_type_names", []), dtype=object)
        if cell_type_names.size == 0:
            return

        with torch.no_grad():
            unique_times = torch.unique(t_tgt[:, 0])
            for time_value in unique_times:
                time_mask = torch.isclose(t_tgt[:, 0], time_value)
                for type_id in torch.unique(cell_type_ids[time_mask]):
                    group_mask = time_mask & (cell_type_ids == type_id)
                    if not bool(group_mask.any()):
                        continue
                    type_idx = int(type_id.item())
                    if type_idx < 0 or type_idx >= cell_type_names.shape[0]:
                        continue
                    cell_type_name = str(cell_type_names[type_idx])
                    marker_idx_np = self.marker_idx_by_cell_type.get(cell_type_name)
                    if marker_idx_np is None or marker_idx_np.size == 0:
                        continue

                    key = self._distribution_group_key(float(time_value.item()), type_idx)
                    pred_group = self._extract_marker_group_expr(pred_tgt, gene_valid_mask, group_mask, marker_idx_np)
                    tgt_group = self._extract_marker_group_expr(tgt_expr, gene_valid_mask, group_mask, marker_idx_np)
                    self._append_distribution_bank(self._pred_distribution_bank, key, pred_group)
                    self._append_distribution_bank(self._real_distribution_bank, key, tgt_group)

    def _prepare_batch(self, batch):
        if len(batch) == 10:
            gene_ids, src_expr, tgt_expr, valid_mask, t_src, t_tgt, cell_type_ids, pair_weight, src_idx, tgt_idx = batch
        else:
            gene_ids, src_expr, tgt_expr, valid_mask, t_src, t_tgt, cell_type_ids, pair_weight = batch
            src_idx = None
            tgt_idx = None

        gene_ids = gene_ids.to(self.device, non_blocking=True)
        src_expr = src_expr.to(self.device, non_blocking=True)
        tgt_expr = tgt_expr.to(self.device, non_blocking=True)
        valid_mask = valid_mask.to(self.device, non_blocking=True).bool()
        t_src = t_src.to(self.device, non_blocking=True).float().unsqueeze(1)
        t_tgt = t_tgt.to(self.device, non_blocking=True).float().unsqueeze(1)
        cell_type_ids = cell_type_ids.to(self.device, non_blocking=True).long()
        pair_weight = pair_weight.to(self.device, non_blocking=True).float()
        cls_mask = gene_ids.eq(self.model.cls_token_id)
        gene_valid_mask = valid_mask & (~cls_mask)
        padding_mask = ~valid_mask
        return {
            "gene_ids": gene_ids,
            "src_expr": src_expr,
            "tgt_expr": tgt_expr,
            "valid_mask": valid_mask,
            "gene_valid_mask": gene_valid_mask,
            "padding_mask": padding_mask,
            "t_src": t_src,
            "t_tgt": t_tgt,
            "cell_type_ids": cell_type_ids,
            "pair_weight": pair_weight,
            "src_idx": src_idx,
            "tgt_idx": tgt_idx,
        }

    def _get_batch_pseudotime_targets(self, tgt_idx):
        if not self.pseudotime_loss_enabled or tgt_idx is None or self.pseudotime_targets is None:
            return None
        if isinstance(tgt_idx, torch.Tensor):
            idx_np = tgt_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            idx_np = np.asarray(tgt_idx, dtype=np.int64)
        values = self.pseudotime_targets[idx_np]
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def _compute_losses(
        self,
        pred_tgt: torch.Tensor,
        tgt_expr: torch.Tensor,
        src_expr: torch.Tensor,
        gene_valid_mask: torch.Tensor,
        type_logits: torch.Tensor,
        pseudo_pred: torch.Tensor,
        pseudo_target: torch.Tensor,
        cell_type_ids: torch.Tensor,
        t_tgt: torch.Tensor,
    ):
        recon_huber = weighted_huber_loss(pred_tgt, tgt_expr, gene_valid_mask)
        recon_pearson = pearson_loss(pred_tgt[gene_valid_mask], tgt_expr[gene_valid_mask])
        displacement_pred = pred_tgt - src_expr
        displacement_true = tgt_expr - src_expr
        vel_cos = cosine_direction_loss(displacement_pred, displacement_true, mask=gene_valid_mask)
        type_loss = type_classification_loss(type_logits, cell_type_ids)
        pseudo_reg = pred_tgt.new_tensor(0.0)
        pseudo_rank = pred_tgt.new_tensor(0.0)
        pseudo_loss = pred_tgt.new_tensor(0.0)
        if self.pseudotime_loss_enabled and pseudo_pred is not None and pseudo_target is not None:
            valid_mask = torch.isfinite(pseudo_target)
            pseudo_reg = pseudotime_regression_loss(
                pseudo_pred,
                pseudo_target,
                valid_mask=valid_mask,
                beta=self.pseudotime_reg_beta,
            )
            pseudo_rank = pseudotime_ranking_loss(
                pseudo_pred,
                pseudo_target,
                valid_mask=valid_mask,
                margin=self.pseudotime_rank_margin,
            )
            pseudo_loss = pseudo_reg + self.pseudotime_rank_weight * pseudo_rank
        dist_loss = self._compute_distribution_loss(
            pred_tgt=pred_tgt,
            tgt_expr=tgt_expr,
            gene_valid_mask=gene_valid_mask,
            cell_type_ids=cell_type_ids,
            t_tgt=t_tgt,
        )
        total = (
            self.loss_recon * (recon_huber + self.loss_recon_pearson * recon_pearson)
            + self.loss_velocity * vel_cos
            + self.loss_dist * dist_loss
            + self.loss_type * type_loss
            + self.loss_pseudotime * pseudo_loss
        )
        metrics = {
            "recon_huber": float(recon_huber.item()),
            "recon_pearson": float(recon_pearson.item()),
            "vel_cos": float(vel_cos.item()),
            "dist_loss": float(dist_loss.item()),
            "dist_loss_surrogate": float(dist_loss.item()),
            "type_loss": float(type_loss.item()),
            "pseudo_reg_loss": float(pseudo_reg.item()),
            "pseudo_rank_loss": float(pseudo_rank.item()),
            "pseudo_loss": float(pseudo_loss.item()),
            "type_acc": float(type_classification_accuracy(type_logits, cell_type_ids)),
        }
        return total, metrics

    def _compute_distribution_loss(
        self,
        pred_tgt: torch.Tensor,
        tgt_expr: torch.Tensor,
        gene_valid_mask: torch.Tensor,
        cell_type_ids: torch.Tensor,
        t_tgt: torch.Tensor,
        use_memory_bank: bool = True,
    ):
        if self.marker_idx_by_cell_type:
            marker_loss = self._compute_marker_mmd_loss(
                pred_tgt=pred_tgt,
                tgt_expr=tgt_expr,
                gene_valid_mask=gene_valid_mask,
                cell_type_ids=cell_type_ids,
                t_tgt=t_tgt,
                use_memory_bank=use_memory_bank,
            )
            if marker_loss is not None:
                return marker_loss

        pred_for_dist = pred_tgt * gene_valid_mask.to(pred_tgt.dtype)
        tgt_for_dist = tgt_expr * gene_valid_mask.to(tgt_expr.dtype)
        return mmd_rbf_loss(
            pred_for_dist,
            tgt_for_dist,
            group_ids=t_tgt[:, 0],
            min_group_size=self.mmd_min_group_size,
        )

    def _compute_marker_mmd_loss(
        self,
        pred_tgt: torch.Tensor,
        tgt_expr: torch.Tensor,
        gene_valid_mask: torch.Tensor,
        cell_type_ids: torch.Tensor,
        t_tgt: torch.Tensor,
        use_memory_bank: bool = True,
    ):
        cell_type_names = np.asarray(self.data_meta.get("cell_type_names", []), dtype=object)
        if cell_type_names.size == 0:
            return None

        losses = []
        unique_times = torch.unique(t_tgt[:, 0])
        for time_value in unique_times:
            time_mask = torch.isclose(t_tgt[:, 0], time_value)
            time_type_ids = torch.unique(cell_type_ids[time_mask])
            for type_id in time_type_ids:
                group_mask = time_mask & (cell_type_ids == type_id)
                if int(group_mask.sum().item()) < int(self.mmd_min_group_size):
                    continue

                type_idx = int(type_id.item())
                if type_idx < 0 or type_idx >= cell_type_names.shape[0]:
                    continue
                cell_type_name = str(cell_type_names[type_idx])
                marker_idx_np = self.marker_idx_by_cell_type.get(cell_type_name)
                if marker_idx_np is None or marker_idx_np.size == 0:
                    continue

                pred_group = self._extract_marker_group_expr(pred_tgt, gene_valid_mask, group_mask, marker_idx_np)
                tgt_group = self._extract_marker_group_expr(tgt_expr, gene_valid_mask, group_mask, marker_idx_np)

                if use_memory_bank:
                    key = self._distribution_group_key(float(time_value.item()), type_idx)
                    hist_limit = max(self.dist_bank_sample_size - pred_group.shape[0], 0)
                    hist_pred = self._sample_distribution_bank(
                        self._pred_distribution_bank,
                        key,
                        max_items=hist_limit,
                        device=pred_tgt.device,
                    )
                    hist_real = self._sample_distribution_bank(
                        self._real_distribution_bank,
                        key,
                        max_items=self.dist_bank_sample_size,
                        device=pred_tgt.device,
                    )

                    pred_ref = pred_group if hist_pred is None else torch.cat([pred_group, hist_pred], dim=0)
                    real_ref = tgt_group.detach()
                    if hist_real is not None:
                        real_ref = torch.cat([real_ref, hist_real], dim=0)

                    if pred_ref.shape[0] >= self.dist_bank_warmup_min and real_ref.shape[0] >= self.dist_bank_warmup_min:
                        losses.append(mmd_rbf_loss(pred_ref, real_ref, group_ids=None, min_group_size=self.mmd_min_group_size))
                        continue

                losses.append(mmd_rbf_loss(pred_group, tgt_group, group_ids=None, min_group_size=self.mmd_min_group_size))

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _build_temporal_eval_collected(self, dataset, prediction_payload):
        base_dataset = self._unwrap_dataset(dataset)
        metadata_df = getattr(base_dataset, "metadata", None)
        expression = getattr(base_dataset, "expression", None)
        if metadata_df is None or expression is None:
            return None
        if "obs_name" not in metadata_df.columns:
            if metadata_df.index.name == "obs_name":
                metadata_df = metadata_df.reset_index()
            else:
                metadata_df = metadata_df.copy()
                metadata_df.insert(0, "obs_name", metadata_df.index.to_numpy().astype(str))

        tgt_index = np.concatenate(prediction_payload["tgt_index"], axis=0)
        src_index = np.concatenate(prediction_payload["src_index"], axis=0)
        cell_type_names = np.asarray(self.data_meta.get("cell_type_names", []), dtype=object)
        pre_meta = self._pre_meta or {}

        collected = {
            "pred_expr": np.concatenate(prediction_payload["pred_expr"], axis=0),
            "tgt_expr": np.concatenate(prediction_payload["tgt_expr"], axis=0),
            "src_expr": np.concatenate(prediction_payload["src_expr"], axis=0),
            "t_src": np.concatenate(prediction_payload["t_src"], axis=0),
            "t_tgt": np.concatenate(prediction_payload["t_tgt"], axis=0),
            "cell_type_id": np.concatenate(prediction_payload["cell_type_id"], axis=0),
            "src_index": src_index,
            "tgt_index": tgt_index,
            "metadata_df": metadata_df,
            "preprocessed_expr": expression,
            "cell_type_names": cell_type_names,
            "hvg_gene_names": pre_meta.get("hvg_gene_names"),
            "hvg_var_names": pre_meta.get("hvg_var_names"),
            "time_norm_denom": float(pre_meta.get("time_norm_denom", getattr(dataset, "time_norm_denom", 1.0))),
            "src_time_raw": metadata_df.iloc[src_index]["author_day"].astype(str).to_numpy(),
            "tgt_time_raw": metadata_df.iloc[tgt_index]["author_day"].astype(str).to_numpy(),
        }
        return collected

    @torch.no_grad()
    def validate(self, val_loader):
        val_start = time.time()
        self.model.eval()
        stats = []
        prediction_payload = {
            "pred_expr": [],
            "tgt_expr": [],
            "src_expr": [],
            "t_src": [],
            "t_tgt": [],
            "cell_type_id": [],
            "src_index": [],
            "tgt_index": [],
        }
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="validate", leave=False)):
            if self.val_max_batches > 0 and batch_idx >= self.val_max_batches:
                break
            payload = self._prepare_batch(batch)
            pred_tgt = self.model.temporal_inference(
                payload["gene_ids"],
                payload["src_expr"],
                payload["padding_mask"],
                t_src=payload["t_src"],
                t_tgt=payload["t_tgt"],
                cell_type_ids=payload["cell_type_ids"],
                n_steps=self.infer_n_steps,
            )
            recon = masked_metrics(pred_tgt, payload["tgt_expr"], payload["gene_valid_mask"])
            dist_loss = self._compute_distribution_loss(
                pred_tgt=pred_tgt,
                tgt_expr=payload["tgt_expr"],
                gene_valid_mask=payload["gene_valid_mask"],
                cell_type_ids=payload["cell_type_ids"],
                t_tgt=payload["t_tgt"],
                use_memory_bank=False,
            )
            displacement_pred = pred_tgt - payload["src_expr"]
            displacement_true = payload["tgt_expr"] - payload["src_expr"]
            vel_cos = cosine_direction_loss(displacement_pred, displacement_true, mask=payload["gene_valid_mask"])
            stats.append(
                {
                    **recon,
                    "val_dist_loss": float(dist_loss.item()),
                    "val_vel_cos": float(vel_cos.item()),
                }
            )

            if self.use_temporal_eval_metrics and payload["src_idx"] is not None and payload["tgt_idx"] is not None:
                prediction_payload["pred_expr"].append(pred_tgt[:, 1:].detach().cpu().numpy().astype(np.float32))
                prediction_payload["tgt_expr"].append(payload["tgt_expr"][:, 1:].detach().cpu().numpy().astype(np.float32))
                prediction_payload["src_expr"].append(payload["src_expr"][:, 1:].detach().cpu().numpy().astype(np.float32))
                prediction_payload["t_src"].append(payload["t_src"][:, 0].detach().cpu().numpy().astype(np.float32))
                prediction_payload["t_tgt"].append(payload["t_tgt"][:, 0].detach().cpu().numpy().astype(np.float32))
                prediction_payload["cell_type_id"].append(payload["cell_type_ids"].detach().cpu().numpy().astype(np.int64))
                prediction_payload["src_index"].append(payload["src_idx"].detach().cpu().numpy().astype(np.int64))
                prediction_payload["tgt_index"].append(payload["tgt_idx"].detach().cpu().numpy().astype(np.int64))

        self.model.train()
        if not stats:
            return {}
        keys = stats[0].keys()
        result = {}
        for key in keys:
            values = [s[key] for s in stats if s.get(key) is not None]
            result[key] = None if not values else float(sum(values) / len(values))
        collect_done = time.time()
        result["val_predict_seconds"] = float(collect_done - val_start)

        if self.use_temporal_eval_metrics and prediction_payload["pred_expr"]:
            try:
                from eval_temporal import compute_temporal_metrics

                collected = self._build_temporal_eval_collected(val_loader.dataset, prediction_payload)
                if collected is not None:
                    eval_metrics, latent_meta = compute_temporal_metrics(
                        collected,
                        self.cfg,
                        output_dir=Path(self.exp_dir) / "validation_eval_cache",
                        verbose=False,
                    )
                    timing_meta = latent_meta.get("timing", {}) if isinstance(latent_meta, dict) else {}
                    result.update(
                        {
                            "val_per_cell_r2_mean": eval_metrics.get("per_cell_pearson_r2_mean"),
                            "val_per_cell_r2_by_t_tgt": eval_metrics.get("per_cell_pearson_r2_by_t_tgt"),
                            "val_distribution_mean_r2_mean": eval_metrics.get("distribution_mean_r2_mean"),
                            "val_distribution_mean_r2_by_t_tgt": eval_metrics.get("distribution_mean_r2_by_t_tgt"),
                            "val_marker_mmd_mean": eval_metrics.get("marker_mmd_mean"),
                            "val_marker_mmd_by_t_tgt": eval_metrics.get("marker_mmd_by_t_tgt"),
                            "val_cell_type_accuracy": eval_metrics.get("cell_type_accuracy"),
                            "val_cell_type_accuracy_per_class": eval_metrics.get("cell_type_accuracy_per_class"),
                            "val_classifier_baseline_accuracy": eval_metrics.get("classifier_baseline_accuracy"),
                            "val_pseudotime_spearman": eval_metrics.get("pseudotime_spearman"),
                            "val_pseudotime_spearman_per_class": eval_metrics.get("pseudotime_spearman_per_class"),
                            "val_real_pseudotime_vs_time_spearman": eval_metrics.get("real_pseudotime_vs_time_spearman"),
                            "val_by_regime": eval_metrics.get("by_regime"),
                            "val_baselines": eval_metrics.get("baselines"),
                            "val_asset_mode": latent_meta.get("mode"),
                            "val_metric_seconds": timing_meta.get("total_metric_seconds"),
                            "val_asset_prepare_seconds": timing_meta.get("asset_prepare_seconds"),
                            "val_core_metric_seconds": timing_meta.get("core_metric_seconds"),
                        }
                    )
            except Exception as exc:
                result["val_eval_error"] = str(exc)
        result["val_eval_seconds"] = float(time.time() - val_start)
        return result

    def train(self, train_loader, val_loader, epochs, lr, weight_decay, val_every_steps, save_every_epochs, resume=None):
        epochs = int(epochs)
        val_every_steps = int(val_every_steps)
        save_every_epochs = int(save_every_epochs)
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = max(epochs * steps_per_epoch, 1)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        global_step = 0
        start_epoch = 0
        if resume is not None:
            ckpt = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
            print(f"[resume] {resume} epoch={start_epoch} global_step={global_step}")

        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        overall_pbar = tqdm(total=total_steps, initial=global_step, desc="train")

        from utils import build_mask_states

        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss_sum = 0.0
            epoch_batches = 0
            epoch_pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)

            for batch_idx, batch in enumerate(epoch_pbar, start=1):
                payload = self._prepare_batch(batch)
                t_flow = sample_t_stratified(payload["gene_ids"].shape[0], self.device)
                mask_states = build_mask_states(
                    payload["padding_mask"],
                    payload["gene_valid_mask"],
                    cls_mask=payload["gene_ids"].eq(self.model.cls_token_id),
                )
                x_t = (1.0 - t_flow) * payload["src_expr"] + t_flow * payload["tgt_expr"]
                collect_cls_attention = (
                    ((global_step + 1) % self.log_every_steps == 0)
                    and self._should_collect_cls_attention(payload["gene_ids"].shape[0], payload["gene_ids"].shape[1])
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                    pred_tgt, _, type_logits, pseudo_pred, _, cls_attention_stats = self.model.forward_with_aux(
                        payload["gene_ids"],
                        x_t,
                        mask_states,
                        payload["padding_mask"],
                        t_flow,
                        t_src=payload["t_src"],
                        t_tgt=payload["t_tgt"],
                        cell_type_ids=payload["cell_type_ids"],
                        return_cls_attention=collect_cls_attention,
                    )
                    pseudo_target = self._get_batch_pseudotime_targets(payload["tgt_idx"])
                    loss, metrics = self._compute_losses(
                        pred_tgt=pred_tgt,
                        tgt_expr=payload["tgt_expr"],
                        src_expr=payload["src_expr"],
                        gene_valid_mask=payload["gene_valid_mask"],
                        type_logits=type_logits,
                        pseudo_pred=pseudo_pred,
                        pseudo_target=pseudo_target,
                        cell_type_ids=payload["cell_type_ids"],
                        t_tgt=payload["t_tgt"],
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                self._update_distribution_banks(
                    pred_tgt=pred_tgt.detach(),
                    tgt_expr=payload["tgt_expr"].detach(),
                    gene_valid_mask=payload["gene_valid_mask"],
                    cell_type_ids=payload["cell_type_ids"],
                    t_tgt=payload["t_tgt"],
                )

                global_step += 1
                epoch_loss_sum += float(loss.item())
                epoch_batches += 1
                overall_pbar.update(1)

                val_record = None
                if val_loader is not None and global_step % max(val_every_steps, 1) == 0:
                    eval_loader, subset_meta = self._build_validation_subset_loader(val_loader)
                    val_metrics = self.validate(eval_loader)
                    if subset_meta is not None:
                        val_metrics.update(subset_meta)
                    val_record = {
                        "epoch": epoch + 1,
                        "epoch_step": batch_idx,
                        "global_step": global_step,
                        **val_metrics,
                    }
                    print(
                        f"\n[val step={global_step}] "
                        f"per_cell_r2={self._fmt_console_metric(val_metrics.get('val_per_cell_r2_mean'))} "
                        f"dist_mean_r2={self._fmt_console_metric(val_metrics.get('val_distribution_mean_r2_mean'))} "
                        f"marker_mmd={self._fmt_console_metric(val_metrics.get('val_marker_mmd_mean'))} "
                        f"type_acc={self._fmt_console_metric(val_metrics.get('val_cell_type_accuracy'))} "
                        f"pseudotime={self._fmt_console_metric(val_metrics.get('val_pseudotime_spearman'))} "
                        f"eval_s={self._fmt_console_metric(val_metrics.get('val_eval_seconds'))}"
                    )

                if global_step % self.log_every_steps == 0:
                    record = {
                        "epoch": epoch + 1,
                        "epoch_step": batch_idx,
                        "global_step": global_step,
                        "train_loss": float(loss.item()),
                        **metrics,
                        "lr": float(scheduler.get_last_lr()[0]),
                    }
                    if cls_attention_stats is not None:
                        if "cls_attn_received_mean" in cls_attention_stats:
                            record["cls_attn_received_mean"] = float(cls_attention_stats["cls_attn_received_mean"].item())
                        if "cls_attn_received_last" in cls_attention_stats:
                            record["cls_attn_received_last"] = float(cls_attention_stats["cls_attn_received_last"].item())
                    if val_record is not None:
                        record.update(self._filter_temporal_val_record(val_record))
                    self._log(record)
                elif val_record is not None:
                    self._log(self._filter_temporal_val_record(val_record))

            epoch_pbar.close()

            epoch_record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "epoch_train_loss": float(epoch_loss_sum / max(epoch_batches, 1)),
                "epoch_batches": int(epoch_batches),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            self._log(epoch_record)

            if (epoch + 1) % max(save_every_epochs, 1) == 0:
                self._save_checkpoint(epoch + 1, global_step, optimizer, scheduler=scheduler)

        overall_pbar.close()
        self._save_checkpoint(epochs, global_step, optimizer, scheduler=scheduler, tag="final")


def parse_args():
    parser = argparse.ArgumentParser(description="Train scFiT")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg["training"].get("seed", 42)))

    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    train_cfg = cfg["training"]
    reserve_cfg = train_cfg.get("gpu_reservation", {})
    split_cfg = data_cfg.get("split", {}) or {}
    pairing_cfg = cfg.get("pairing", {}) or {}
    protocol_cfg = cfg.get("protocol", {}) or {}
    loader_cfg = data_cfg.get("loader", {}) or {}

    data_key = str(data_cfg.get("source", "mtg_toy"))
    run_name = exp_cfg.get("run_name") or f"{data_key}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = build_experiment_dir(exp_cfg.get("output_dir", "experiments"), run_name)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_reservation = GpuMemoryReservation(
        device=device,
        enabled=bool(reserve_cfg.get("enabled", False)),
        reserve_gb=reserve_cfg.get("reserve_gb"),
        fraction_of_free=float(reserve_cfg.get("fraction_of_free", 0.8)),
    )
    gpu_reservation.reserve()

    max_cells = data_cfg.get("max_cells")
    cell_seed = int(data_cfg.get("cell_seed", train_cfg.get("seed", 42)))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))

    protocol_path = str(protocol_cfg.get("path", "") or "").strip()
    if protocol_path:
        bundle = load_protocol_bundle(protocol_path)
        model = scFiTModel(
            n_genes=int(bundle["bundle_meta"]["n_genes"]),
            top_k_genes=int(bundle["bundle_meta"]["fixed_panel_size"]),
            d_model=int(cfg["model"]["d_model"]),
            nhead=int(cfg["model"]["nhead"]),
            d_ff=int(cfg["model"]["d_ff"]),
            nlayers=int(cfg["model"]["nlayers"]),
            dropout=float(cfg["model"]["dropout"]),
            value_max=float(cfg["model"].get("value_encoder", {}).get("max_value", 32.0)),
            share_adaln_projection=bool(cfg["model"].get("adaln", {}).get("share_projection", False)),
            aux_bin_ce=False,
            n_bins=0,
            n_cell_types=0,
            pseudotime_head=False,
        )
        gpu_reservation.release()
        trainer = RouteATrainer(model=model, device=device, exp_dir=exp_dir, cfg=cfg, bundle=bundle)
        print("=" * 60)
        print(f"exp_dir={exp_dir}")
        print(f"device={device}")
        print(f"protocol={bundle['protocol_name']}")
        print(f"observed_bins={len(bundle['observed_time_bins'])} heldout_bins={len(bundle['heldout_time_bins'])}")
        print(f"model_input_dim={bundle['bundle_meta']['fixed_panel_size']} total_gene_space={bundle['bundle_meta']['n_genes']}")
        print("=" * 60)
        trainer.train(
            epochs=int(train_cfg["epochs"]),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            save_every_epochs=int(train_cfg["save_every_epochs"]),
            resume=train_cfg.get("resume"),
        )
        return

    temporal_mode = str(pairing_cfg.get("strategy", "") or "").strip().lower() == "latent_knn"

    if temporal_mode:
        top_k = int(cfg["model"]["top_k_genes"])
        temporal_input_mode = str(pairing_cfg.get("input_mode", "dynamic_topk") or "dynamic_topk").strip().lower()
        train_dataset, val_dataset, _, meta = load_mouse_development_temporal_datasets(
            data_key,
            input_mode=temporal_input_mode,
            preprocessed_dir=data_cfg.get("preprocessed_dir"),
            pair_table_path=pairing_cfg.get("pair_table_path"),
            uce_npz_path=data_cfg.get("uce_npz"),
            time_col=data_cfg.get("time_col", "author_day"),
            time_numeric_col=data_cfg.get("time_numeric_col"),
            cell_type_col=data_cfg.get("cell_type_col", "author_cell_type"),
            top_k=top_k,
            pairing_k=int(pairing_cfg.get("k", 5)),
            future_only=bool(pairing_cfg.get("future_only", True)),
            similarity_chunk_size=int(pairing_cfg.get("similarity_chunk_size", 512)),
            train_src_min_time=pairing_cfg.get("train", {}).get("src_min_time", 10.0),
            train_src_max_time=pairing_cfg.get("train", {}).get("src_max_time", 14.0),
            train_tgt_min_time=pairing_cfg.get("train", {}).get("tgt_min_time", 10.0),
            train_tgt_max_time=pairing_cfg.get("train", {}).get("tgt_max_time", 14.0),
            val_src_min_time=pairing_cfg.get("val", {}).get("src_min_time", 12.0),
            val_src_max_time=pairing_cfg.get("val", {}).get("src_max_time", 14.0),
            val_tgt_min_time=pairing_cfg.get("val", {}).get("tgt_min_time", 14.0),
            val_tgt_max_time=pairing_cfg.get("val", {}).get("tgt_max_time", 19.0),
            max_cells=max_cells,
            cell_seed=cell_seed,
            train_cache_path=pairing_cfg.get("train_cache_path"),
            val_cache_path=pairing_cfg.get("val_cache_path"),
            force_rebuild_pairs=bool(pairing_cfg.get("force_rebuild_pairs", False)),
        )
        n_genes = int(meta["n_genes"])
        train_loader = make_temporal_dataloader(
            train_dataset,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
            group_by_tgt_time=bool(pairing_cfg.get("group_by_tgt_time", True)),
            drop_last=bool(pairing_cfg.get("drop_last", False)),
            seed=int(train_cfg.get("seed", 42)),
            num_workers=int(loader_cfg.get("num_workers", 0)),
            prefetch_factor=loader_cfg.get("prefetch_factor"),
            persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
        )
        val_loader = make_temporal_dataloader(
            val_dataset,
            batch_size=int(cfg.get("validation", {}).get("batch_size", cfg["training"]["batch_size"])),
            shuffle=False,
            group_by_tgt_time=bool(pairing_cfg.get("group_by_tgt_time", True)),
            drop_last=False,
            seed=int(train_cfg.get("seed", 42)),
            num_workers=int(loader_cfg.get("num_workers", 0)),
            prefetch_factor=loader_cfg.get("prefetch_factor"),
            persistent_workers=bool(loader_cfg.get("persistent_workers", False)),
        )
        model = scFiTModel(
            n_genes=n_genes,
            top_k_genes=top_k,
            d_model=int(cfg["model"]["d_model"]),
            nhead=int(cfg["model"]["nhead"]),
            d_ff=int(cfg["model"]["d_ff"]),
            nlayers=int(cfg["model"]["nlayers"]),
            dropout=float(cfg["model"]["dropout"]),
            value_max=float(cfg["model"].get("value_encoder", {}).get("max_value", 32.0)),
            share_adaln_projection=bool(cfg["model"].get("adaln", {}).get("share_projection", False)),
            aux_bin_ce=False,
            n_bins=0,
            n_cell_types=int(cfg["model"].get("n_cell_types", len(meta.get("cell_type_names", [])))),
            pseudotime_head=bool(cfg["model"].get("pseudotime_head", {}).get("enabled", False)),
        )
        gpu_reservation.release()
        trainer = TemporalTrainer(model, device, exp_dir, cfg, data_meta=meta)

        print("=" * 60)
        print(f"exp_dir={exp_dir}")
        print(f"device={device}")
        print(f"n_genes={n_genes} top_k={top_k}")
        print(
            f"input_mode={meta['pairing'].get('input_mode', temporal_input_mode)} "
            f"train_pairs={meta['pairing']['train_pairs']} val_pairs={meta['pairing']['val_pairs']}"
        )
        if meta.get("n_cells_aligned") is not None:
            print(f"aligned_cells={meta.get('n_cells_aligned')}")
        if meta['pairing'].get("pair_table_path") is not None:
            print(f"pair_table={meta['pairing']['pair_table_path']}")
        if meta['pairing'].get("train_cache_path") is not None:
            print(f"pair_cache_train={meta['pairing']['train_cache_path']}")
        if meta['pairing'].get("val_cache_path") is not None:
            print(f"pair_cache_val={meta['pairing']['val_cache_path']}")
        print("=" * 60)

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=int(train_cfg["epochs"]),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            val_every_steps=int(train_cfg["val_every_steps"]),
            save_every_epochs=int(train_cfg["save_every_epochs"]),
            resume=train_cfg.get("resume"),
        )
        return

    split_strategy = str(split_cfg.get("strategy", "") or "").strip().lower()
    if not split_strategy:
        split_strategy = "donor" if data_key in ("mtg_full", "dlpfc_full") or val_ratio > 0 else "none"
    use_split = split_strategy not in ("", "none", "full")
    if use_split:
        if split_strategy == "donor" and data_key in ("mtg_full", "dlpfc_full") and not split_cfg:
            train_data, val_data, _, meta = load_sea_ad_split(
                data_key,
                val_ratio=val_ratio,
                seed=int(train_cfg.get("seed", 42)),
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
        else:
            train_data, val_data, _, meta = load_h5ad_split(
                data_key,
                split_strategy=split_strategy,
                split_column=split_cfg.get("column"),
                donor_csv=split_cfg.get("donor_csv"),
                val_ratio=val_ratio,
                seed=int(train_cfg.get("seed", 42)),
                max_cells=max_cells,
                cell_seed=cell_seed,
                time_col=split_cfg.get("time_col"),
                time_numeric_col=split_cfg.get("time_numeric_col"),
                train_max_time=split_cfg.get("train_max_time"),
                val_min_time=split_cfg.get("val_min_time"),
                val_max_time=split_cfg.get("val_max_time"),
                train_values=split_cfg.get("train_values"),
                val_values=split_cfg.get("val_values"),
            )
        train_X, gene_ids = train_data
        val_X, _ = val_data
        n_genes = int(meta["n_genes"])
        prior_mean, prior_std = compute_log1p_nonzero_prior_stats(train_X)
    else:
        if data_key in ("mtg_full", "dlpfc_full", "mtg_toy", "dlpfc_toy"):
            count_matrix, gene_ids, _, meta = load_sea_ad(
                data_key,
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
        else:
            count_matrix, gene_ids, _, meta = load_h5ad(
                data_key,
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
        train_X = count_matrix
        val_X = None
        n_genes = int(meta["n_genes"])
        prior_mean, prior_std = compute_log1p_nonzero_prior_stats(train_X)

    aux_cfg = cfg["model"].get("aux_bin_ce", {})
    use_aux_bin_ce = bool(aux_cfg.get("enabled", False))
    n_bins = int(aux_cfg.get("n_bins", 0))
    bin_edges = compute_log1p_nonzero_bin_edges(train_X, n_bins=n_bins) if use_aux_bin_ce and n_bins > 1 else None

    top_k = int(cfg["model"]["top_k_genes"])
    train_dataset = TopKExpressionDataset(train_X, gene_ids, top_k=top_k, return_index=False)
    train_loader = make_dataloader(train_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)

    val_loader = None
    if val_X is not None and val_X.shape[0] > 0:
        val_dataset = TopKExpressionDataset(val_X, gene_ids, top_k=top_k, return_index=True)
        val_loader = make_dataloader(val_dataset, batch_size=int(cfg["validation"]["batch_size"]), shuffle=False)

    model = scFiTModel(
        n_genes=n_genes,
        top_k_genes=top_k,
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        d_ff=int(cfg["model"]["d_ff"]),
        nlayers=int(cfg["model"]["nlayers"]),
        dropout=float(cfg["model"]["dropout"]),
        value_max=float(cfg["model"].get("value_encoder", {}).get("max_value", 32.0)),
        share_adaln_projection=bool(cfg["model"].get("adaln", {}).get("share_projection", False)),
        aux_bin_ce=use_aux_bin_ce,
        n_bins=n_bins,
        n_cell_types=int(cfg["model"].get("n_cell_types", len(meta.get("cell_type_names", [])))),
        pseudotime_head=bool(cfg["model"].get("pseudotime_head", {}).get("enabled", False)),
    )
    gpu_reservation.release()
    trainer = Trainer(model, device, exp_dir, cfg)
    trainer.prior_mean = float(prior_mean)
    trainer.prior_std = float(prior_std)
    if bin_edges is not None:
        trainer.bin_edges = torch.tensor(bin_edges, device=device, dtype=torch.float32)

    print("=" * 60)
    print(f"exp_dir={exp_dir}")
    print(f"device={device}")
    print(f"n_genes={n_genes} top_k={top_k}")
    print(f"train_cells={train_X.shape[0]} train_batches_per_epoch={len(train_loader)}")
    print(f"prior_mean={trainer.prior_mean:.6f} prior_std={trainer.prior_std:.6f}")
    if trainer.bin_edges is not None:
        print(f"aux_bin_ce=True n_bins={n_bins}")
    print("=" * 60)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        val_every_steps=int(train_cfg["val_every_steps"]),
        save_every_epochs=int(train_cfg["save_every_epochs"]),
        resume=train_cfg.get("resume"),
    )


if __name__ == "__main__":
    main()
