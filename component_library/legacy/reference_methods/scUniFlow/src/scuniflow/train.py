import argparse
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp

import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import scRNAModel
from data import (
    compute_expression_prior_stats,
    load_sea_ad,
    load_sea_ad_split,
    scRNADataset,
)
from paths import ensure_tmp_root


BEST_SELECTION_METRICS = {
    "val_mse": "min",
    "val_rmse": "min",
    "val_mae": "min",
    "val_pearson_cell": "max",
    "val_pearson_all": "max",
    "val_spearman_cell": "max",
    "val_pearson_top20": "max",
}


# ======================================================================
# DDP 工具
# ======================================================================

def setup_ddp():
    """初始化分布式环境，返回 (rank, local_rank, world_size)。
    若未通过 torchrun 启动则回退到单卡模式。"""
    if "RANK" in os.environ:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def should_update_best_checkpoint(current_metrics, best_metrics):
    """
    Update best checkpoint only if more than half of comparable validation
    metrics improved in the right direction.
    """
    if best_metrics is None:
        return True, {
            "improved_metrics": [],
            "worsened_metrics": [],
            "unchanged_metrics": [],
            "comparable_metrics": 0,
            "improved_count": 0,
            "reason": "first_validation",
        }

    improved_metrics = []
    worsened_metrics = []
    unchanged_metrics = []

    for metric_name, direction in BEST_SELECTION_METRICS.items():
        cur_val = current_metrics.get(metric_name)
        best_val = best_metrics.get(metric_name)
        if cur_val is None or best_val is None:
            continue

        cur_val = float(cur_val)
        best_val = float(best_val)
        if direction == "min":
            if cur_val < best_val:
                improved_metrics.append(metric_name)
            elif cur_val > best_val:
                worsened_metrics.append(metric_name)
            else:
                unchanged_metrics.append(metric_name)
        elif direction == "max":
            if cur_val > best_val:
                improved_metrics.append(metric_name)
            elif cur_val < best_val:
                worsened_metrics.append(metric_name)
            else:
                unchanged_metrics.append(metric_name)
        else:
            raise ValueError(f"Unsupported selection direction: {direction}")

    comparable_metrics = (
        len(improved_metrics) + len(worsened_metrics) + len(unchanged_metrics)
    )
    should_update = comparable_metrics > 0 and (len(improved_metrics) * 2 > comparable_metrics)
    summary = {
        "improved_metrics": improved_metrics,
        "worsened_metrics": worsened_metrics,
        "unchanged_metrics": unchanged_metrics,
        "comparable_metrics": comparable_metrics,
        "improved_count": len(improved_metrics),
        "reason": "majority_improved" if should_update else "majority_not_improved",
    }
    return should_update, summary


class DistributedEvalSampler(Sampler):
    """Shard eval data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("DistributedEvalSampler requires initialized torch.distributed")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("DistributedEvalSampler requires initialized torch.distributed")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        total = len(self.dataset)
        if self.rank >= total:
            return 0
        return (total - 1 - self.rank) // self.num_replicas + 1


def pearson_loss(pred, target):
    """全局 Pearson 相关损失：对常数预测梯度最大（=1.0），防止模型停留在平凡解。"""
    pred_c = pred - pred.mean()
    tgt_c  = target - target.mean()
    return 1.0 - (pred_c * tgt_c).sum() / (
        pred_c.norm() * tgt_c.norm() + 1e-8
    )


def cell_pearson_loss(x1_hats, x1_targets):
    """Cell-wise Pearson 损失：直接对齐 val_pearson_cell 的计算方式。"""
    losses = []
    for b in range(len(x1_hats)):
        pred = x1_hats[b]
        true = x1_targets[b]
        if pred.shape[0] < 2:
            continue
        pred_c = pred - pred.mean()
        true_c = true - true.mean()
        r = (pred_c * true_c).sum() / (
            pred_c.norm() * true_c.norm() + 1e-8
        )
        losses.append(1.0 - r)
    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=x1_hats[0].device)


def weighted_recon_loss(x1_hats, x1_targets, top_k=20):
    """加权重建损失：对每个细胞 top_k 高表达基因额外加 3 倍权重。"""
    B = len(x1_hats)
    max_len = max(p.shape[0] for p in x1_hats)
    device = x1_hats[0].device

    pred_pad = torch.zeros(B, max_len, device=device)
    true_pad = torch.zeros(B, max_len, device=device)
    mask = torch.zeros(B, max_len, device=device)
    for i, (p, t) in enumerate(zip(x1_hats, x1_targets)):
        n = p.shape[0]
        pred_pad[i, :n] = p
        true_pad[i, :n] = t
        mask[i, :n] = 1.0

    base_loss = F.huber_loss(pred_pad, true_pad, delta=0.5, reduction='none')

    # top_k 加权
    weights = mask.clone()
    true_masked = true_pad.clone()
    true_masked[mask == 0] = -1e9  # padding 不参与 topk
    k = min(top_k, max_len)
    _, topk_idx = true_masked.topk(k, dim=1)
    weights.scatter_(1, topk_idx, 3.0)
    weights *= mask  # padding 权重归零

    return (base_loss * weights).sum() / mask.sum()


def sample_t_stratified(B, device):
    """分层采样：每个区间分配固定数量样本，消除 batch 间方差。"""
    bins = [
        (0.0, 0.2, int(B * 0.40)),
        (0.2, 0.4, int(B * 0.20)),
        (0.4, 0.6, int(B * 0.15)),
        (0.6, 0.8, int(B * 0.15)),
        (0.8, 1.0, B - int(B*0.40) - int(B*0.20) - int(B*0.15) - int(B*0.15)),
    ]
    parts = []
    for lo, hi, n in bins:
        parts.append(torch.rand(n, device=device) * (hi - lo) + lo)
    t = torch.cat(parts)[torch.randperm(B, device=device)]
    return t.unsqueeze(1)  # (B, 1)


def make_dataloader(
    count_matrix,
    gene_ids,
    batch_size,
    shuffle=True,
    world_size=1,
    nonzero_mask_ratio=0.2,
    zero_mask_ratio=0.0,
    return_index=False,
    expr_transform="log1p",
):
    dataset = scRNADataset(
        count_matrix,
        gene_ids,
        nonzero_mask_ratio=nonzero_mask_ratio,
        zero_mask_ratio=zero_mask_ratio,
        return_index=return_index,
        expr_transform=expr_transform,
    )
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True,
        ), sampler
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, pin_memory=True,
    ), None

def compute_empirical_prior_stats(count_matrix, expr_transform):
    """Compute empirical Gaussian prior stats in the chosen expression space."""
    return compute_expression_prior_stats(count_matrix, expr_transform=expr_transform)


def resolve_epoch_training_config(train_cfg, steps_per_epoch):
    """Read epoch-based training config with step-based validation support."""
    steps_per_epoch = max(1, int(steps_per_epoch))

    epochs = train_cfg.get("epochs")
    if epochs is None:
        legacy_p1 = int(train_cfg.get("phase1_warmup", {}).get("steps", 0))
        legacy_p2 = int(train_cfg.get("phase2_finetune", {}).get("steps", 0))
        legacy_total_steps = max(legacy_p1 + legacy_p2, legacy_p2, legacy_p1)
        if legacy_total_steps <= 0:
            raise ValueError("training.epochs 未设置，且找不到可转换的旧版 phase step 配置。")
        epochs = max(1, int(math.ceil(legacy_total_steps / steps_per_epoch)))

    lr = train_cfg.get("lr")
    if lr is None:
        lr = train_cfg.get("phase2_finetune", {}).get(
            "lr_base",
            train_cfg.get("phase1_warmup", {}).get("lr", 5e-4),
        )

    min_lr = float(train_cfg.get("min_lr", 1e-6))

    val_every_steps = train_cfg.get("val_every_steps")
    if val_every_steps is not None:
        val_every_steps = max(1, int(val_every_steps))
    else:
        val_every_epochs = train_cfg.get("val_every_epochs")
        if val_every_epochs is None:
            val_every_steps = steps_per_epoch
        else:
            val_every_steps = max(1, int(val_every_epochs)) * steps_per_epoch

    save_every_epochs = train_cfg.get("save_every_epochs")
    if save_every_epochs is None:
        legacy_save_steps = int(train_cfg.get("save_every_steps", steps_per_epoch))
        save_every_epochs = max(1, int(math.ceil(legacy_save_steps / steps_per_epoch)))

    log_every_steps = max(1, int(train_cfg.get("log_every_steps", 100)))

    return {
        "epochs": int(epochs),
        "lr": float(lr),
        "min_lr": float(min_lr),
        "val_every_steps": int(val_every_steps),
        "log_every_steps": int(log_every_steps),
        "save_every_epochs": int(save_every_epochs),
    }


# ======================================================================
# Trainer
# ======================================================================

class scRNAPipelineTrainer:
    def __init__(self, n_genes=32738, d_model=512, grid_size=181,
                 n_layers=8, d_state_row=64, d_state_col=16,
                 activation_checkpointing=True,
                 local_rank=0, exp_dir="experiments"):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.exp_dir = exp_dir
        self.ckpt_dir = os.path.join(exp_dir, "checkpoints")
        self.log_path = os.path.join(exp_dir, "train_log.jsonl")
        if is_main():
            os.makedirs(self.ckpt_dir, exist_ok=True)
            open(self.log_path, "a").close()

        self.model = scRNAModel(
            n_genes=n_genes,
            d_model=d_model,
            grid_size=grid_size,
            n_layers=n_layers,
            d_state_row=d_state_row,
            d_state_col=d_state_col,
            activation_checkpointing=activation_checkpointing,
        ).to(self.device)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank],
                             find_unused_parameters=True)

        # Validation defaults aligned with current reconstruction eval script.
        self.val_nonzero_mask_ratio = 0.2
        self.val_zero_mask_ratio = 0.0
        self.val_mask_seed = 1234
        self.val_noise_seed = 5678
        self.val_infer_n_steps = 10
        self.val_infer_temperature = 0.01
        self.val_prior_mean = 0.0
        self.val_prior_std = 1.0
        self.val_max_batches = 200
        self.debug_t_bucket_stats = False
        self.debug_t_bucket_every = 1000
        self.loss_weighted_x1 = 1.0
        self.loss_pearson_weight = 0.1
        self.loss_cell_pearson_weight = 0.1
        self.prediction_target = "x1"

    @property
    def raw_model(self):
        """获取去掉 DDP 包装的原始模型。"""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _make_val_mask(self, x_1, cell_indices=None, global_offset=0):
        """Mask policy aligned with eval script defaults (currently 20% nonzero + 0% zero)."""
        B, G = x_1.shape
        m = torch.zeros((B, G), dtype=torch.float32, device=x_1.device)
        for b in range(B):
            g = torch.Generator()
            if cell_indices is not None:
                cell_id = int(cell_indices[b])
            else:
                cell_id = int(global_offset + b)
            g.manual_seed(int(self.val_mask_seed + cell_id))

            nz = (x_1[b] > 0).nonzero(as_tuple=True)[0]
            z = (x_1[b] <= 0).nonzero(as_tuple=True)[0]

            n_mask_nz = int(nz.numel() * self.val_nonzero_mask_ratio)
            if n_mask_nz > 0 and nz.numel() > 0:
                perm = torch.randperm(nz.numel(), generator=g)
                chosen = nz[perm[:n_mask_nz].to(nz.device)]
                m[b, chosen] = 1.0

            n_mask_z = int(z.numel() * self.val_zero_mask_ratio)
            if n_mask_z > 0 and z.numel() > 0:
                perm = torch.randperm(z.numel(), generator=g)
                chosen = z[perm[:n_mask_z].to(z.device)]
                m[b, chosen] = 1.0
        return m

    def _make_val_initial_masked_values(self, x_1, eval_mask, cell_indices=None, global_offset=0):
        """
        Build deterministic masked initialization for validation.
        Each cell gets its own fixed Gaussian draw keyed by val_noise_seed + cell_id,
        so validation is reproducible across checkpoints and repeated runs.
        """
        init = x_1.clone()
        prior_mean = float(self.val_prior_mean)
        prior_std = float(max(self.val_prior_std, 1e-6))

        for b in range(x_1.shape[0]):
            masked_idx = (eval_mask[b] == 1).nonzero(as_tuple=True)[0]
            if masked_idx.numel() == 0:
                continue

            if cell_indices is not None:
                cell_id = int(cell_indices[b])
            else:
                cell_id = int(global_offset + b)

            g = torch.Generator(device="cpu")
            g.manual_seed(int(self.val_noise_seed + cell_id))
            noise = torch.randn(
                masked_idx.numel(),
                generator=g,
                dtype=torch.float32,
            ).to(device=x_1.device, dtype=x_1.dtype)
            init[b, masked_idx] = noise * prior_std + prior_mean

        return init

    def _log_t_bucket_velocity_stats(self, preds, targets, t, prefix="train"):
        """
        Debug helper: bucket by t and report mean(x1_pred), mean(x1_true) on masked positions.
        preds/targets are per-cell tensors from forward_train.
        """
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        sum_pred = [0.0] * 5
        sum_true = [0.0] * 5
        cnt = [0] * 5

        for b in range(len(preds)):
            tb = float(t[b].item())
            bi = min(int(tb / 0.2), 4)
            vp = preds[b].detach().float()
            vt = targets[b].detach().float()
            if vp.numel() == 0:
                continue
            sum_pred[bi] += float(vp.sum().item())
            sum_true[bi] += float(vt.sum().item())
            cnt[bi] += int(vp.numel())

        for i in range(5):
            if cnt[i] == 0:
                continue
            l = bins[i]
            r = bins[i + 1]
            mp = sum_pred[i] / cnt[i]
            mt = sum_true[i] / cnt[i]
            self._print(
                f"[{prefix}] t∈[{l:.1f},{r:.1f}): "
                f"x1_pred_mean={mp:.4f}, x1_true_mean={mt:.4f}, n={cnt[i]}"
            )

    def _trainable_parameter_groups(self):
        raw = self.raw_model
        return [
            {"params": raw.router.gene_embedding.parameters()},
            {"params": raw.mamba_encoder.parameters()},
            {"params": raw.vector_field.parameters()},
        ]

    def _compute_train_losses(self, g_id, x_1, x_0, mask_ind, t):
        raw = self.raw_model
        if getattr(self, "prediction_target", "x1") == "velocity":
            preds, targets, x1_hats, x1_targets, _ = raw.forward_train_velocity(
                g_id, x_1, mask_ind, t, x_0, temperature=0.0
            )
        else:
            preds, targets, x1_hats, x1_targets, _ = raw.forward_train(
                g_id, x_1, mask_ind, t, x_0, temperature=0.0
            )
        flat_preds = torch.cat(preds, dim=-1)
        flat_targets = torch.cat(targets, dim=-1)
        loss_x1 = F.huber_loss(flat_preds, flat_targets, delta=0.5)
        if getattr(self, "prediction_target", "x1") == "velocity":
            loss_weighted = loss_x1
        else:
            loss_weighted = weighted_recon_loss(x1_hats, x1_targets, top_k=20)
        loss_pearson = pearson_loss(flat_preds, flat_targets)
        loss_cell_pearson = cell_pearson_loss(preds, targets)
        loss = (
            self.loss_weighted_x1 * loss_weighted
            + self.loss_pearson_weight * loss_pearson
            + self.loss_cell_pearson_weight * loss_cell_pearson
        )
        return {
            "preds": preds,
            "targets": targets,
            "loss": loss,
            "x1_loss": loss_x1,
            "weighted_x1_loss": loss_weighted,
            "pearson_loss": loss_pearson,
            "cell_pearson_loss": loss_cell_pearson,
        }

    def _clip_gradients(self):
        raw = self.raw_model
        torch.nn.utils.clip_grad_norm_(
            list(raw.router.gene_embedding.parameters()) +
            list(raw.mamba_encoder.parameters()) +
            list(raw.vector_field.parameters()),
            max_norm=1.0,
        )

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, val_loader, epoch=None, temperature=0.1):
        from scipy.stats import spearmanr as scipy_spearmanr
        del temperature
        self.model.eval()
        n_batches = 0
        total_sq, total_abs, total_cnt = 0.0, 0.0, 0
        sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
        cell_pearsons, cell_spearmans, cell_pearsons_top20 = [], [], []

        with torch.no_grad():
            global_cell_offset = 0
            if dist.is_initialized():
                max_batches_local = max(1, int(math.ceil(self.val_max_batches / dist.get_world_size())))
            else:
                max_batches_local = self.val_max_batches

            desc = "Validate"
            if epoch is not None:
                desc = f"Validate epoch {epoch}"
            for batch in tqdm(val_loader, desc=desc, leave=False, total=max_batches_local):
                if len(batch) == 5:
                    g_id, x_1, x_0, mask_ind, cell_idx = batch
                else:
                    g_id, x_1, x_0, mask_ind = batch
                    cell_idx = None
                g_id = g_id.to(self.device)
                x_1 = x_1.to(self.device)
                raw = self.raw_model
                n_batches += 1

                bs = x_1.shape[0]
                if cell_idx is not None:
                    eval_mask = self._make_val_mask(x_1, cell_indices=cell_idx)
                else:
                    eval_mask = self._make_val_mask(x_1, global_offset=global_cell_offset)
                global_cell_offset += bs

                observed = x_1.clone()
                observed[eval_mask == 1] = 0.0
                init_masked_values = self._make_val_initial_masked_values(
                    x_1,
                    eval_mask,
                    cell_indices=cell_idx,
                    global_offset=global_cell_offset - bs,
                )

                if getattr(self, "prediction_target", "x1") == "velocity":
                    pred_full = raw.inference_velocity(
                        g_id,
                        observed,
                        eval_mask,
                        n_steps=self.val_infer_n_steps,
                        temperature=self.val_infer_temperature,
                        prior_mean=self.val_prior_mean,
                        prior_std=self.val_prior_std,
                        initial_masked_values=init_masked_values,
                    )
                else:
                    pred_full = raw.inference(
                        g_id,
                        observed,
                        eval_mask,
                        n_steps=self.val_infer_n_steps,
                        temperature=self.val_infer_temperature,
                        prior_mean=self.val_prior_mean,
                        prior_std=self.val_prior_std,
                        initial_masked_values=init_masked_values,
                    )

                mask_bool = eval_mask == 1
                if mask_bool.any():
                    diff = pred_full[mask_bool].float() - x_1[mask_bool].float()
                    total_sq += float(torch.sum(diff * diff).item())
                    total_abs += float(torch.sum(torch.abs(diff)).item())
                    total_cnt += int(mask_bool.sum().item())
                    px = pred_full[mask_bool].float().cpu().numpy().astype(np.float64)
                    py = x_1[mask_bool].float().cpu().numpy().astype(np.float64)
                    sum_x += float(px.sum())
                    sum_y += float(py.sum())
                    sum_x2 += float(np.square(px).sum())
                    sum_y2 += float(np.square(py).sum())
                    sum_xy += float((px * py).sum())

                pred_cpu = pred_full.cpu().numpy()
                true_cpu = x_1.cpu().numpy()
                mask_cpu = mask_bool.cpu().numpy()
                for b in range(bs):
                    mb = mask_cpu[b]
                    if mb.sum() < 2:
                        continue
                    pv = pred_cpu[b, mb].astype(np.float64)
                    tv = true_cpu[b, mb].astype(np.float64)
                    if np.std(pv) > 0 and np.std(tv) > 0:
                        r = float(np.corrcoef(pv, tv)[0, 1])
                        if not np.isnan(r):
                            cell_pearsons.append(r)
                    sr = scipy_spearmanr(pv, tv).correlation
                    if sr is not None and not np.isnan(float(sr)):
                        cell_spearmans.append(float(sr))
                    k = min(20, tv.shape[0])
                    if k >= 2:
                        idx_top = np.argsort(tv)[-k:]
                        pv_top, tv_top = pv[idx_top], tv[idx_top]
                        if np.std(pv_top) > 0 and np.std(tv_top) > 0:
                            r20 = float(np.corrcoef(pv_top, tv_top)[0, 1])
                            if not np.isnan(r20):
                                cell_pearsons_top20.append(r20)

                if n_batches >= max_batches_local:
                    break

        self.model.train()

        # DDP 下跨 rank 聚合统计量
        if dist.is_initialized():
            def _reduce(x):
                t = torch.tensor(x, dtype=torch.float64, device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                return t.item()
            total_sq   = _reduce(total_sq)
            total_abs  = _reduce(total_abs)
            total_cnt  = int(_reduce(total_cnt))
            sum_x      = _reduce(sum_x)
            sum_y      = _reduce(sum_y)
            sum_x2     = _reduce(sum_x2)
            sum_y2     = _reduce(sum_y2)
            sum_xy     = _reduce(sum_xy)
            # cell-level 列表：收集所有 rank 的结果
            for lst in (cell_pearsons, cell_spearmans, cell_pearsons_top20):
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, lst)
                lst.clear()
                for sub in gathered:
                    lst.extend(sub)

        mse = total_sq / max(total_cnt, 1)
        mae = total_abs / max(total_cnt, 1)
        denom = np.sqrt(
            max(total_cnt * sum_x2 - sum_x ** 2, 0.0) *
            max(total_cnt * sum_y2 - sum_y ** 2, 0.0)
        )
        pearson_all = None if denom <= 0 else float(
            (total_cnt * sum_xy - sum_x * sum_y) / denom
        )
        return {
            "val_mse":           mse,
            "val_rmse":          float(np.sqrt(mse)),
            "val_mae":           mae,
            "val_pearson_cell":  float(np.mean(cell_pearsons))    if cell_pearsons    else None,
            "val_pearson_all":   pearson_all,
            "val_spearman_cell": float(np.mean(cell_spearmans))   if cell_spearmans   else None,
            "val_pearson_top20": float(np.mean(cell_pearsons_top20)) if cell_pearsons_top20 else None,
        }

    # ------------------------------------------------------------------
    # 日志与 checkpoint（仅 rank 0 执行）
    # ------------------------------------------------------------------
    def _log(self, record: dict):
        if not is_main():
            return
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _plot_loss_curve(self):
        if not is_main():
            return
        if not os.path.exists(self.log_path):
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            records = []
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if not records:
                return

            train_steps, train_losses = [], []
            val_steps, val_losses = [], []
            corr_steps, val_corrs = [], []
            for r in records:
                loss_val = r.get("train_loss") or r.get("x1_loss") or r.get("cfm_loss")
                if loss_val is not None:
                    train_steps.append(r["step"])
                    train_losses.append(loss_val)
                if "val_rmse" in r:
                    val_steps.append(r["step"])
                    val_losses.append(r["val_rmse"])
                if "val_pearson_cell" in r and r["val_pearson_cell"] is not None:
                    corr_steps.append(r["step"])
                    val_corrs.append(r["val_pearson_cell"])

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(train_steps, train_losses, label="train_loss", alpha=0.7)
            if val_losses:
                axes[0].plot(val_steps, val_losses, label="val_rmse", marker="o", markersize=4)
            axes[0].set_xlabel("step")
            axes[0].set_ylabel("loss")
            axes[0].set_title("Loss Curve")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if val_corrs:
                axes[1].plot(corr_steps, val_corrs, label="val cell Pearson",
                             marker="o", markersize=4, color="green")
                axes[1].set_xlabel("step")
                axes[1].set_ylabel("Pearson r")
                axes[1].set_title("Val Cell-wise Pearson")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(self.exp_dir, "loss_curve.png"), dpi=100)
            plt.close()
        except Exception as e:
            self._print(f"  [plot] 绘图失败: {e}")

    def _save_checkpoint(self, epoch, step, tag=None, **opt_states):
        if not is_main():
            return
        name = tag or "checkpoint_last"
        path = os.path.join(self.ckpt_dir, f"{name}.pt")
        payload = {
            "format_version": 2,
            "training_mode": "single_stage_epoch",
            "phase": "single",
            "epoch": int(epoch),
            "step": step,
            "model_state_dict": self.raw_model.state_dict(),
            "prior_mean": self.val_prior_mean,
            "prior_std": self.val_prior_std,
            "expression_transform": getattr(self, "expression_transform", "log1p"),
            "prediction_target": getattr(self, "prediction_target", "x1"),
        }
        payload.update(opt_states)
        torch.save(payload, path)
        print(f"  [checkpoint] {path}")

    def _save_best_metrics(self, metrics: dict):
        if not is_main():
            return
        path = os.path.join(self.exp_dir, "best_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  [best metrics] {path}")

    def _print(self, msg):
        if is_main():
            print(msg)

    # ------------------------------------------------------------------
    # 单阶段 epoch 训练
    # ------------------------------------------------------------------
    def train_epochs(
        self,
        train_loader,
        train_sampler=None,
        val_loader=None,
        epochs=80,
        lr=5e-4,
        min_lr=1e-6,
        val_every_steps=0,
        log_every_steps=0,
        save_every_epochs=5,
        resume_ckpt=None,
    ):
        self._print("====== 单阶段 epoch 训练 ======")
        self.model.train()

        optimizer = optim.AdamW(
            self._trainable_parameter_groups(),
            lr=lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs)),
            eta_min=min_lr,
        )

        start_epoch = 0
        global_step = 0
        best_record = None
        best_selection_summary = None
        if resume_ckpt is not None:
            ckpt = torch.load(resume_ckpt, map_location=self.device)
            self.raw_model.load_state_dict(ckpt["model_state_dict"])
            ckpt_mode = ckpt.get("training_mode")
            if ckpt_mode == "single_stage_epoch":
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0))
                global_step = int(ckpt.get("step", 0))
                best_record = ckpt.get("best_record")
                best_selection_summary = ckpt.get("best_selection_summary")
                self._print(
                    f"  [resume] 从 {resume_ckpt} 恢复，epoch={start_epoch}, step={global_step}"
                )
            else:
                legacy_phase = ckpt.get("phase", "unknown")
                self._print(
                    f"  [resume legacy] 从 {resume_ckpt} 加载模型权重（旧 phase={legacy_phase}），"
                    "优化器与 epoch 调度将从头开始。"
                )

        if start_epoch >= epochs:
            self._print(
                f"  [skip] checkpoint 已完成 epoch={start_epoch}，目标 epochs={epochs}。"
            )
            return

        for epoch in range(start_epoch + 1, epochs + 1):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            epoch_metrics = {
                "train_loss": 0.0,
                "x1_loss": 0.0,
                "weighted_x1_loss": 0.0,
                "pearson_loss": 0.0,
                "cell_pearson_loss": 0.0,
            }
            metric_sources = {
                "train_loss": "loss",
                "x1_loss": "x1_loss",
                "weighted_x1_loss": "weighted_x1_loss",
                "pearson_loss": "pearson_loss",
                "cell_pearson_loss": "cell_pearson_loss",
            }
            n_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                disable=not is_main(),
                leave=False,
            )
            steps_in_epoch = len(train_loader)
            for batch_idx, (g_id, x_1, x_0, mask_ind) in enumerate(pbar, start=1):
                g_id = g_id.to(self.device)
                x_1 = x_1.to(self.device)
                x_0 = x_0.to(self.device)
                mask_ind = mask_ind.to(self.device)

                t = sample_t_stratified(g_id.shape[0], self.device)
                optimizer.zero_grad()

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = self._compute_train_losses(g_id, x_1, x_0, mask_ind, t)

                if self.debug_t_bucket_stats and global_step % self.debug_t_bucket_every == 0:
                    self._log_t_bucket_velocity_stats(
                        losses["preds"], losses["targets"], t, prefix=f"epoch={epoch} step={global_step}"
                    )

                losses["loss"].backward()
                self._clip_gradients()
                optimizer.step()

                global_step += 1
                n_batches += 1
                for metric_key, loss_key in metric_sources.items():
                    epoch_metrics[metric_key] += float(losses[loss_key].item())

                pbar.set_postfix(
                    loss=f"{losses['loss'].item():.4f}",
                    x1=f"{losses['x1_loss'].item():.4f}",
                    w_x1=f"{losses['weighted_x1_loss'].item():.4f}",
                    pr_x1=f"{losses['pearson_loss'].item():.4f}",
                    pr_cell=f"{losses['cell_pearson_loss'].item():.4f}",
                )

                should_log = int(log_every_steps) > 0 and global_step % int(log_every_steps) == 0

                should_run_val = (
                    bool(val_loader)
                    and int(val_every_steps) > 0
                    and global_step % int(val_every_steps) == 0
                )
                is_last_train_step = epoch == epochs and batch_idx == steps_in_epoch
                if bool(val_loader) and is_last_train_step and (
                    int(val_every_steps) <= 0 or global_step % int(val_every_steps) != 0
                ):
                    should_run_val = True

                vm = None
                if should_run_val:
                    vm = self.validate(val_loader, epoch=epoch)
                    if is_main():
                        print(
                            f"\n[val step={global_step} epoch={epoch}] "
                            f"rmse={vm.get('val_rmse', 0):.4f}  "
                            f"mae={vm.get('val_mae', 0):.4f}  "
                            f"pearson_cell={vm.get('val_pearson_cell') or 0:.4f}  "
                            f"pearson_all={vm.get('val_pearson_all') or 0:.4f}  "
                            f"spearman_cell={vm.get('val_spearman_cell') or 0:.4f}  "
                            f"pearson_top20={vm.get('val_pearson_top20') or 0:.4f}"
                        )

                if should_log or vm is not None:
                    step_record = {
                        "phase": "single",
                        "record_type": "step",
                        "epoch": epoch,
                        "epoch_step": batch_idx,
                        "step": global_step,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": float(losses["loss"].item()),
                        "x1_loss": float(losses["x1_loss"].item()),
                        "weighted_x1_loss": float(losses["weighted_x1_loss"].item()),
                        "pearson_loss": float(losses["pearson_loss"].item()),
                        "cell_pearson_loss": float(losses["cell_pearson_loss"].item()),
                    }
                    if vm is not None:
                        step_record.update(vm)
                        should_update_best, selection_summary = should_update_best_checkpoint(
                            current_metrics=step_record,
                            best_metrics=best_record,
                        )
                        step_record["best_selection_summary"] = selection_summary
                        if should_update_best:
                            best_record = dict(step_record)
                            best_selection_summary = dict(selection_summary)
                            self._save_checkpoint(
                                epoch,
                                global_step,
                                tag="checkpoint_best",
                                optimizer_state_dict=optimizer.state_dict(),
                                scheduler_state_dict=scheduler.state_dict(),
                                best_record=best_record,
                                best_selection_summary=best_selection_summary,
                            )
                            self._save_best_metrics(best_record)
                    self._log(step_record)
                    self._plot_loss_curve()

            pbar.close()
            scheduler.step()

            for key in epoch_metrics:
                epoch_metrics[key] /= max(n_batches, 1)

            lr_cur = float(scheduler.get_last_lr()[0])
            record = {
                "phase": "single",
                "record_type": "epoch_train",
                "epoch": epoch,
                "step": global_step,
                "lr": lr_cur,
                **epoch_metrics,
            }

            if is_main():
                summary = (
                    f"[epoch {epoch}/{epochs}] "
                    f"loss={record['train_loss']:.4f} "
                    f"x1={record['x1_loss']:.4f} "
                    f"w_x1={record['weighted_x1_loss']:.4f} "
                    f"pearson={record['pearson_loss']:.4f} "
                    f"cell_pearson={record['cell_pearson_loss']:.4f} "
                    f"lr={lr_cur:.2e}"
                )
                print(summary)

            if int(save_every_epochs) > 0 and (
                epoch % int(save_every_epochs) == 0 or epoch == epochs
            ):
                self._save_checkpoint(
                    epoch,
                    global_step,
                    tag="checkpoint_last",
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    best_record=best_record,
                    best_selection_summary=best_selection_summary,
                )

        self._save_checkpoint(
            epochs,
            global_step,
            tag="checkpoint_last",
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            best_record=best_record,
            best_selection_summary=best_selection_summary,
        )


# ======================================================================
# 入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scUniFlow 单阶段 epoch 硬路由训练")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data", type=str, default="mtg_toy",
                        help="数据键名或 h5ad 路径")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="单卡 batch size，覆盖 config 中的 batch_size_per_gpu")
    parser.add_argument("--phase", type=str, default=None,
                        help="已弃用；当前仅保留兼容，支持 all/single 或不传")
    parser.add_argument("--run_name", type=str, default=None,
                        help="实验名称，默认为 {data}_{timestamp}")
    parser.add_argument("--spectral_coords", type=str, default=None,
                        help="augmented_gene_coords.npy 路径，覆盖 config 中的 spectral_coords")
    parser.add_argument("--max_cells", type=int, default=None,
                        help="仅抽样部分细胞做快速验证；对 train/val 总数生效")
    parser.add_argument("--debug_t_bucket_stats", action="store_true",
                        help="打印训练时按 t 分桶的 x1_pred/x1_true 均值统计")
    parser.add_argument("--debug_t_bucket_every", type=int, default=1000,
                        help="每多少 step 打印一次 t 分桶统计")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 路径恢复训练（.pt 文件）")
    parser.add_argument(
        "--expr_transform",
        type=str,
        choices=["log1p", "logcpm"],
        default=None,
        help="表达空间变换，覆盖 config 中 training.expression_transform",
    )
    args = parser.parse_args()

    if args.phase not in (None, "all", "single"):
        raise ValueError("当前训练已重构为单阶段 epoch 模式，不再支持 --phase 1/2。请移除该参数或使用 --phase all。")

    rank, local_rank, world_size = setup_ddp()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    expr_transform = str(args.expr_transform or train_cfg.get("expression_transform", "log1p"))
    train_cfg["expression_transform"] = expr_transform
    prediction_target = str(train_cfg.get("prediction_target", "x1")).lower()
    if prediction_target not in ("x1", "velocity"):
        raise ValueError("training.prediction_target must be one of: x1, velocity")
    train_cfg["prediction_target"] = prediction_target

    run_name = args.run_name or f"{args.data}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(train_cfg.get("experiment_dir", "experiments"), run_name)
    if is_main():
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    if world_size > 1:
        dist.barrier()

    # CLI 参数覆盖 config；未指定则从 config 读取
    batch_size = args.batch_size or int(train_cfg.get("batch_size_per_gpu", 32))
    val_cfg_early = cfg.get("validation", {})
    val_batch_size = int(val_cfg_early.get("batch_size", batch_size))
    spectral_coords = args.spectral_coords or train_cfg.get("spectral_coords")

    train_nonzero_mask_ratio = float(train_cfg.get("nonzero_mask_ratio", 0.2))
    train_zero_mask_ratio = float(train_cfg.get("zero_mask_ratio", 0.0))
    seed = train_cfg.get("seed", 42)
    max_cells = args.max_cells if args.max_cells is not None else train_cfg.get("max_cells")
    cell_seed = int(train_cfg.get("cell_seed", seed))

    val_ratio = train_cfg.get("val_ratio", 0.1)

    if is_main():
        print("=" * 60)
        print(f"实验目录: {exp_dir}")
        print(f"GPU 数量: {world_size}")
        print(f"表达空间: {expr_transform}")
        print(f"预测目标: {prediction_target}")
        print("=" * 60)

    # ---- 数据加载：只读一次磁盘，按 rank 切分行 ----
    use_split = args.data in ("mtg_full", "dlpfc_full") or val_ratio > 0
    val_loader = None
    train_sampler = None

    if rank == 0:
        if use_split:
            try:
                train_data, val_data, _, meta = load_sea_ad_split(
                    args.data,
                    val_ratio=val_ratio,
                    seed=seed,
                    max_cells=max_cells,
                    cell_seed=cell_seed,
                )
                train_X, gene_ids = train_data
                val_X, _ = val_data
                n_genes = meta["n_genes"]
                train_prior_mean, train_prior_std = compute_empirical_prior_stats(
                    train_X,
                    expr_transform=expr_transform,
                )

                split_path = os.path.join(exp_dir, "split_info.json")
                with open(split_path, "w") as f:
                    json.dump({
                        "train_donors": meta["train_donors"],
                        "val_donors": meta["val_donors"],
                        **meta["split_info_effective"],
                    }, f, indent=2, ensure_ascii=False)
                print(f"[saved] 划分信息 -> {split_path}")

                if world_size > 1:
                    import scipy.sparse as sp
                    import tempfile, pickle
                    shard_dir = tempfile.mkdtemp(prefix="scmam_shards_", dir=ensure_tmp_root())
                    n_train = train_X.shape[0]
                    chunk = n_train // world_size
                    for r in range(world_size):
                        start = r * chunk
                        end = n_train if r == world_size - 1 else (r + 1) * chunk
                        shard_path = os.path.join(shard_dir, f"train_{r}.pkl")
                        with open(shard_path, "wb") as sf:
                            pickle.dump(train_X[start:end], sf)
                    info_path = os.path.join(shard_dir, "info.pkl")
                    with open(info_path, "wb") as sf:
                        pickle.dump({
                            "gene_ids": gene_ids, "n_genes": n_genes,
                            "val_X": val_X, "shard_dir": shard_dir,
                            "train_prior_mean": train_prior_mean,
                            "train_prior_std": train_prior_std,
                        }, sf)
                    shard_dir_bytes = shard_dir.encode()
                else:
                    shard_dir_bytes = b""
            except (FileNotFoundError, KeyError):
                print("[warn] 无法按 donor 划分，回退到无验证集模式")
                count_matrix, gene_ids, _, meta = load_sea_ad(
                    args.data,
                    max_cells=max_cells,
                    cell_seed=cell_seed,
                )
                n_genes = meta["n_genes"]
                train_prior_mean, train_prior_std = compute_empirical_prior_stats(
                    count_matrix,
                    expr_transform=expr_transform,
                )
                use_split = False
                if world_size > 1:
                    import scipy.sparse as sp
                    import tempfile, pickle
                    shard_dir = tempfile.mkdtemp(prefix="scmam_shards_", dir=ensure_tmp_root())
                    n_total = count_matrix.shape[0]
                    chunk = n_total // world_size
                    for r in range(world_size):
                        start = r * chunk
                        end = n_total if r == world_size - 1 else (r + 1) * chunk
                        shard_path = os.path.join(shard_dir, f"train_{r}.pkl")
                        with open(shard_path, "wb") as sf:
                            pickle.dump(count_matrix[start:end], sf)
                    info_path = os.path.join(shard_dir, "info.pkl")
                    with open(info_path, "wb") as sf:
                        pickle.dump({
                            "gene_ids": gene_ids, "n_genes": n_genes,
                            "val_X": None, "shard_dir": shard_dir,
                            "train_prior_mean": train_prior_mean,
                            "train_prior_std": train_prior_std,
                        }, sf)
                    shard_dir_bytes = shard_dir.encode()
                else:
                    shard_dir_bytes = b""
        else:
            count_matrix, gene_ids, _, meta = load_sea_ad(
                args.data,
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
            n_genes = meta["n_genes"]
            train_prior_mean, train_prior_std = compute_empirical_prior_stats(
                count_matrix,
                expr_transform=expr_transform,
            )
            if world_size > 1:
                import tempfile, pickle
                shard_dir = tempfile.mkdtemp(prefix="scmam_shards_", dir=ensure_tmp_root())
                n_total = count_matrix.shape[0]
                chunk = n_total // world_size
                for r in range(world_size):
                    start = r * chunk
                    end = n_total if r == world_size - 1 else (r + 1) * chunk
                    shard_path = os.path.join(shard_dir, f"train_{r}.pkl")
                    with open(shard_path, "wb") as sf:
                        pickle.dump(count_matrix[start:end], sf)
                info_path = os.path.join(shard_dir, "info.pkl")
                with open(info_path, "wb") as sf:
                    pickle.dump({
                        "gene_ids": gene_ids, "n_genes": n_genes,
                        "val_X": None, "shard_dir": shard_dir,
                        "train_prior_mean": train_prior_mean,
                        "train_prior_std": train_prior_std,
                    }, sf)
                shard_dir_bytes = shard_dir.encode()
            else:
                shard_dir_bytes = b""

    if world_size > 1:
        import pickle
        obj_list = [shard_dir_bytes if rank == 0 else b""]
        dist.broadcast_object_list(obj_list, src=0)
        shard_dir = obj_list[0].decode()

        info_path = os.path.join(shard_dir, "info.pkl")
        with open(info_path, "rb") as sf:
            info = pickle.load(sf)
        gene_ids = info["gene_ids"]
        n_genes = info["n_genes"]
        train_prior_mean = float(info.get("train_prior_mean", 0.0))
        train_prior_std = float(max(info.get("train_prior_std", 1.0), 1e-6))

        my_shard_path = os.path.join(shard_dir, f"train_{rank}.pkl")
        with open(my_shard_path, "rb") as sf:
            my_train_X = pickle.load(sf)
        print(f"[rank {rank}] 加载分片: {my_train_X.shape[0]:,} cells")

        train_ds = scRNADataset(
            my_train_X,
            gene_ids,
            x0_prior="standard_normal",
            nonzero_mask_ratio=train_nonzero_mask_ratio,
            zero_mask_ratio=train_zero_mask_ratio,
            expr_transform=expr_transform,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )

        val_X = info["val_X"]
        if val_X is not None:
            val_ds = scRNADataset(
                val_X,
                gene_ids,
                x0_prior="standard_normal",
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                return_index=True,
                expr_transform=expr_transform,
            )
            val_sampler = DistributedEvalSampler(val_ds) if world_size > 1 else None
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=val_batch_size, shuffle=False,
                sampler=val_sampler, num_workers=4, pin_memory=True,
            )

        del my_train_X
        dist.barrier()
    else:
        if use_split:
            train_loader, train_sampler = make_dataloader(
                train_X, gene_ids,
                batch_size=batch_size, shuffle=True, world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                expr_transform=expr_transform,
            )
            val_loader, _ = make_dataloader(
                val_X, gene_ids,
                batch_size=val_batch_size, shuffle=False, world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                return_index=True,
                expr_transform=expr_transform,
            )
        else:
            train_loader, train_sampler = make_dataloader(
                count_matrix, gene_ids,
                batch_size=batch_size, shuffle=True, world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                expr_transform=expr_transform,
            )

    # ---- 模型 ----
    if is_main():
        print("=" * 60)
        print(f"模型: n_genes={n_genes}, d_model={model_cfg['d_model']}, "
              f"grid_size={model_cfg['grid_size']}, "
              f"n_layers={model_cfg.get('n_layers', 8)}, "
              f"d_state_row={model_cfg.get('d_state_row', 64)}, "
              f"d_state_col={model_cfg.get('d_state_col', 16)}")
        print("=" * 60)

    trainer = scRNAPipelineTrainer(
        n_genes=n_genes,
        d_model=model_cfg["d_model"],
        grid_size=model_cfg["grid_size"],
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        local_rank=local_rank,
        exp_dir=exp_dir,
    )
    loss_cfg = train_cfg.get("loss_weights", {})
    trainer.loss_weighted_x1 = float(loss_cfg.get("weighted_x1", 1.0))
    trainer.loss_pearson_weight = float(loss_cfg.get("pearson", 0.1))
    trainer.loss_cell_pearson_weight = float(loss_cfg.get("cell_pearson", 0.1))
    trainer.prediction_target = prediction_target
    trainer.debug_t_bucket_stats = bool(args.debug_t_bucket_stats)
    trainer.debug_t_bucket_every = max(1, int(args.debug_t_bucket_every))
    val_cfg = cfg.get("validation", {})
    trainer.val_infer_n_steps = int(val_cfg.get("n_steps", cfg.get("inference", {}).get("n_steps", 10)))
    trainer.val_infer_temperature = float(val_cfg.get("temperature", cfg.get("inference", {}).get("temperature", 0.01)))
    trainer.val_nonzero_mask_ratio = float(val_cfg.get("nonzero_mask_ratio", 0.2))
    trainer.val_zero_mask_ratio = float(val_cfg.get("zero_mask_ratio", 0.0))
    trainer.val_mask_seed = int(val_cfg.get("mask_seed", 1234))
    trainer.val_noise_seed = int(val_cfg.get("noise_seed", 5678))
    val_max_batches = int(val_cfg.get("max_batches", 200))
    trainer.val_max_batches = val_max_batches
    trainer.expression_transform = expr_transform
    trainer.val_prior_mean = float(train_prior_mean)
    trainer.val_prior_std = float(max(train_prior_std, 1e-6))
    if is_main():
        print(
            f"[val prior] using empirical Gaussian from training data "
            f"({expr_transform}): "
            f"mean={trainer.val_prior_mean:.6f}, std={trainer.val_prior_std:.6f}"
        )

    # ---- 固定硬路由初始化：只加载现成谱坐标文件 ----
    if not spectral_coords:
        raise ValueError("硬路由训练必须提供 spectral_coords；请先单独生成坐标文件")
    coords = np.load(spectral_coords)
    assert coords.shape == (n_genes, 2), \
        f"spectral_coords shape {coords.shape} != ({n_genes}, 2)"
    trainer.raw_model.init_from_spectral(
        coords,
        slot_order_mode="hilbert",
        gene_order_mode="hilbert",
    )
    if is_main():
        routing_meta_path = os.path.join(exp_dir, "routing_info.json")
        with open(routing_meta_path, "w") as f:
            json.dump({
                "mode": "file",
                "coords_source": os.path.abspath(spectral_coords),
                "slot_order_mode": "hilbert",
                "gene_order_mode": "hilbert",
                "n_genes": int(n_genes),
                "grid_size": int(model_cfg["grid_size"]),
            }, f, indent=2, ensure_ascii=False)
        print(f"[init] hard routing ready from: {os.path.abspath(spectral_coords)}")
        print(f"[saved] 路由信息 -> {routing_meta_path}")
    if world_size > 1:
        dist.barrier()

    # ---- 训练 ----
    train_schedule = resolve_epoch_training_config(train_cfg, steps_per_epoch=len(train_loader))
    if is_main():
        print(
            "[train schedule] "
            f"epochs={train_schedule['epochs']}, "
            f"steps/epoch={len(train_loader)}, "
            f"lr={train_schedule['lr']:.2e}, "
            f"min_lr={train_schedule['min_lr']:.2e}, "
            f"log_every_steps={train_schedule['log_every_steps']}, "
            f"val_every_steps={train_schedule['val_every_steps']}, "
            f"save_every_epochs={train_schedule['save_every_epochs']}"
        )

    trainer.train_epochs(
        train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        epochs=train_schedule["epochs"],
        lr=train_schedule["lr"],
        min_lr=train_schedule["min_lr"],
        val_every_steps=train_schedule["val_every_steps"],
        log_every_steps=train_schedule["log_every_steps"],
        save_every_epochs=train_schedule["save_every_epochs"],
        resume_ckpt=args.resume,
    )

    # 最终模型保存
    if is_main():
        final_path = os.path.join(exp_dir, "model_final.pt")
        torch.save(trainer.raw_model.state_dict(), final_path)
        print(f"\n最终模型 -> {final_path}")

    cleanup_ddp()
