import argparse
import json
import math
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from scipy.stats import spearmanr as scipy_spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "model.py").exists() and (p / "data.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train as base_train
from paths import ensure_tmp_root
from data_atac import (
    compute_accessibility_prior_stats,
    load_atac_h5ad,
    load_atac_h5ad_split,
    make_dataloader,
    scATACDataset,
)

DEFAULT_AUPRC_GUARD_MULTIPLIER = 2.0


def _summarize_depth_binned_pearson(depths, correlations, n_bins=4):
    if not depths or not correlations or len(depths) != len(correlations):
        return None

    depth_arr = np.asarray(depths, dtype=np.float64)
    corr_arr = np.asarray(correlations, dtype=np.float64)
    if depth_arr.size == 0:
        return None

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(depth_arr, quantiles)
    edges = np.maximum.accumulate(edges)

    summary = {}
    for bin_idx in range(n_bins):
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        if bin_idx == n_bins - 1:
            mask = (depth_arr >= lo) & (depth_arr <= hi)
        else:
            mask = (depth_arr >= lo) & (depth_arr < hi)
        label = f"Q{bin_idx + 1}"
        if not np.any(mask):
            summary[label] = {
                "n_cells": 0,
                "depth_min": lo,
                "depth_max": hi,
                "mean_pearson": None,
            }
            continue
        summary[label] = {
            "n_cells": int(mask.sum()),
            "depth_min": lo,
            "depth_max": hi,
            "mean_pearson": float(np.mean(corr_arr[mask])),
        }
    return summary


def should_update_best_checkpoint_atac(current_metrics, best_metrics, guard_multiplier=DEFAULT_AUPRC_GUARD_MULTIPLIER):
    cur_auprc = current_metrics.get("val_binary_auprc", current_metrics.get("val_auprc"))
    pos_rate = current_metrics.get("val_positive_rate")
    binary_margin = current_metrics.get("val_binary_margin")
    pearson_cell = current_metrics.get("val_intensity_pearson_cell", current_metrics.get("val_pearson_cell"))
    if cur_auprc is None or pos_rate is None:
        return False, {
            "selection_metric": "val_binary_auprc",
            "guard_multiplier": float(guard_multiplier),
            "guard_threshold": None,
            "guard_passed": False,
            "reason": "missing_selection_metric",
        }

    cur_auprc = float(cur_auprc)
    pos_rate = float(pos_rate)
    guard_threshold = float(guard_multiplier * pos_rate)
    auprc_guard_passed = bool(cur_auprc >= guard_threshold)
    margin_guard_threshold = 0.02
    margin_guard_passed = bool(
        binary_margin is not None
        and np.isfinite(float(binary_margin))
        and float(binary_margin) > margin_guard_threshold
    )
    pearson_guard_passed = bool(
        pearson_cell is not None
        and np.isfinite(float(pearson_cell))
    )
    guard_passed = bool(auprc_guard_passed and margin_guard_passed and pearson_guard_passed)

    summary = {
        "selection_metric": "val_binary_auprc",
        "guard_multiplier": float(guard_multiplier),
        "guard_threshold": guard_threshold,
        "guard_passed": guard_passed,
        "auprc_guard_passed": auprc_guard_passed,
        "margin_guard_passed": margin_guard_passed,
        "pearson_guard_passed": pearson_guard_passed,
        "current_auprc": cur_auprc,
        "positive_rate": pos_rate,
        "binary_margin": None if binary_margin is None else float(binary_margin),
        "margin_guard_threshold": float(margin_guard_threshold),
    }

    if not guard_passed:
        failed = []
        if not auprc_guard_passed:
            failed.append("auprc")
        if not margin_guard_passed:
            failed.append("binary_margin")
        if not pearson_guard_passed:
            failed.append("pearson_cell")
        summary["reason"] = "guard_failed:" + ",".join(failed)
        return False, summary

    if best_metrics is None:
        summary["reason"] = "first_guarded_validation"
        return True, summary

    best_auprc = best_metrics.get("val_binary_auprc", best_metrics.get("val_auprc"))
    if best_auprc is None:
        summary["reason"] = "best_missing_selection_metric"
        return True, summary

    best_auprc = float(best_auprc)
    summary["best_auprc"] = best_auprc
    should_update = cur_auprc > best_auprc
    summary["reason"] = "better_auprc" if should_update else "not_better_auprc"
    return should_update, summary


class scATACPipelineTrainer(base_train.scRNAPipelineTrainer):
    def _compute_train_losses(self, g_id, x_1, x_0, mask_ind, t):
        raw = self.raw_model
        preds, binary_logits, targets, x1_hats, x1_targets, _ = raw.forward_train_dual(
            g_id, x_1, mask_ind, t, x_0, temperature=0.0
        )
        flat_preds = torch.cat(preds, dim=-1)
        flat_binary_logits = torch.cat(binary_logits, dim=-1)
        flat_targets = torch.cat(targets, dim=-1)

        nz_preds = []
        nz_targets = []
        for pred, target in zip(x1_hats, x1_targets):
            nz = target > 0
            if int(nz.sum().item()) > 0:
                nz_preds.append(pred[nz])
                nz_targets.append(target[nz])

        if nz_targets:
            flat_nz_preds = torch.cat(nz_preds, dim=-1)
            flat_nz_targets = torch.cat(nz_targets, dim=-1)
        else:
            flat_nz_preds = flat_preds.new_empty(0)
            flat_nz_targets = flat_targets.new_empty(0)

        loss_x1 = base_train.F.huber_loss(flat_preds, flat_targets, delta=0.5)
        if flat_nz_targets.numel() > 0:
            loss_weighted = base_train.F.huber_loss(flat_nz_preds, flat_nz_targets, delta=0.5)
        else:
            loss_weighted = flat_preds.new_tensor(0.0)

        if flat_nz_targets.numel() > 1:
            loss_pearson = base_train.pearson_loss(flat_nz_preds, flat_nz_targets)
        else:
            loss_pearson = flat_preds.new_tensor(0.0)

        cell_nz_losses = []
        for pred, target in zip(nz_preds, nz_targets):
            if pred.numel() < 2:
                continue
            pred_c = pred - pred.mean()
            target_c = target - target.mean()
            r = (pred_c * target_c).sum() / (
                pred_c.norm() * target_c.norm() + 1e-8
            )
            cell_nz_losses.append(1.0 - r)
        if cell_nz_losses:
            loss_cell_pearson = torch.stack(cell_nz_losses).mean()
        else:
            loss_cell_pearson = flat_preds.new_tensor(0.0)

        # Compatibility metric: pearson_loss is already masked-nonzero only.
        loss_pearson_nonzero = loss_pearson
        binary_target = (flat_targets > 0).to(flat_binary_logits.dtype)
        n_pos = binary_target.sum()
        n_neg = binary_target.numel() - n_pos
        pos_weight = (n_neg / (n_pos + 1e-6)).clamp_min(1e-6).detach()
        loss_binary = base_train.F.binary_cross_entropy_with_logits(
            flat_binary_logits,
            binary_target,
            pos_weight=pos_weight,
        )
        loss_binary_rank = flat_preds.new_tensor(0.0)

        loss = (
            self.loss_weighted_x1 * loss_weighted
            + self.loss_pearson_weight * loss_pearson
            + self.loss_cell_pearson_weight * loss_cell_pearson
            + float(getattr(self, "loss_pearson_nonzero_weight", 0.0)) * loss_pearson_nonzero
            + float(getattr(self, "loss_binary_weight", 0.0)) * loss_binary
            + float(getattr(self, "loss_binary_rank_weight", 0.0)) * loss_binary_rank
        )
        return {
            "preds": preds,
            "targets": targets,
            "loss": loss,
            "x1_loss": loss_x1,
            "weighted_x1_loss": loss_weighted,
            "pearson_loss": loss_pearson,
            "cell_pearson_loss": loss_cell_pearson,
            "pearson_nonzero_loss": loss_pearson_nonzero,
            "binary_loss": loss_binary,
            "binary_rank_loss": loss_binary_rank,
        }

    @torch.no_grad()
    def validate(self, val_loader, epoch=None, temperature=0.1):
        del temperature
        self.model.eval()
        n_batches = 0
        total_eval_cells = 0
        total_sq, total_abs, total_cnt = 0.0, 0.0, 0
        sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
        total_corr_cnt = 0
        cell_pearsons, cell_spearmans, cell_pearsons_top20 = [], [], []
        masked_true_chunks, masked_pred_chunks = [], []
        masked_score_chunks = []
        masked_nonzero_true_chunks, masked_nonzero_pred_chunks = [], []
        masked_nonzero_score_chunks, masked_zero_score_chunks = [], []
        depth_proxy_values, depth_proxy_pearsons = [], []
        cell_pred_means, cell_true_means = [], []
        cell_pred_stds, cell_true_stds = [], []
        open_threshold = float(getattr(self, "inference_binary_threshold", 0.5))

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
                del x_0, mask_ind

                g_id = g_id.to(self.device)
                x_1 = x_1.to(self.device)
                raw = self.raw_model
                n_batches += 1

                bs = x_1.shape[0]
                total_eval_cells += bs
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

                pred_intensity = raw.inference(
                    g_id,
                    observed,
                    eval_mask,
                    n_steps=self.val_infer_n_steps,
                    temperature=self.val_infer_temperature,
                    prior_mean=self.val_prior_mean,
                    prior_std=self.val_prior_std,
                    initial_masked_values=init_masked_values,
                )
                t_binary = torch.zeros((bs, 1), device=self.device, dtype=x_1.dtype)
                binary_logit_full = raw.predict_binary_logits(
                    g_id,
                    observed,
                    eval_mask,
                    t_binary,
                    temperature=0.0,
                )
                p_open_full = torch.sigmoid(binary_logit_full)
                pred_full = torch.where(
                    p_open_full > open_threshold,
                    pred_intensity,
                    torch.zeros_like(pred_intensity),
                )

                mask_bool = eval_mask == 1
                if mask_bool.any():
                    pred_masked = pred_full[mask_bool].float()
                    score_masked = p_open_full[mask_bool].float()
                    true_masked = x_1[mask_bool].float()
                    diff = pred_masked - true_masked
                    total_sq += float(torch.sum(diff * diff).item())
                    total_abs += float(torch.sum(torch.abs(diff)).item())
                    total_cnt += int(mask_bool.sum().item())

                    px = pred_masked.cpu().numpy().astype(np.float64)
                    ps = score_masked.cpu().numpy().astype(np.float64)
                    py = true_masked.cpu().numpy().astype(np.float64)

                    masked_pred_chunks.append(px.astype(np.float32))
                    masked_score_chunks.append(ps.astype(np.float32))
                    masked_true_chunks.append(py.astype(np.float32))

                pred_cpu = pred_intensity.cpu().numpy()
                score_cpu = p_open_full.cpu().numpy()
                true_cpu = x_1.cpu().numpy()
                mask_cpu = mask_bool.cpu().numpy()
                for b in range(bs):
                    mb = mask_cpu[b]
                    if mb.sum() < 2:
                        continue

                    zero_mask = mb & (true_cpu[b] <= 0)
                    if zero_mask.any():
                        masked_zero_score_chunks.append(score_cpu[b, zero_mask].astype(np.float32))

                    # Correlation metrics are evaluated only on masked nonzero targets.
                    corr_mask = mb & (true_cpu[b] > 0)
                    if corr_mask.sum() < 2:
                        continue

                    pv = pred_cpu[b, corr_mask].astype(np.float64)
                    sv = score_cpu[b, corr_mask].astype(np.float64)
                    tv = true_cpu[b, corr_mask].astype(np.float64)
                    masked_nonzero_pred_chunks.append(pv.astype(np.float32))
                    masked_nonzero_score_chunks.append(sv.astype(np.float32))
                    masked_nonzero_true_chunks.append(tv.astype(np.float32))
                    sum_x += float(pv.sum())
                    sum_y += float(tv.sum())
                    sum_x2 += float(np.square(pv).sum())
                    sum_y2 += float(np.square(tv).sum())
                    sum_xy += float((pv * tv).sum())
                    total_corr_cnt += int(tv.shape[0])
                    cell_pred_means.append(float(np.mean(pv)))
                    cell_true_means.append(float(np.mean(tv)))
                    cell_pred_stds.append(float(np.std(pv)))
                    cell_true_stds.append(float(np.std(tv)))

                    if np.std(pv) > 0 and np.std(tv) > 0:
                        r = float(np.corrcoef(pv, tv)[0, 1])
                        if not np.isnan(r):
                            cell_pearsons.append(r)
                            depth_proxy_values.append(float(np.count_nonzero(true_cpu[b] > 0)))
                            depth_proxy_pearsons.append(r)
                        sr = scipy_spearmanr(pv, tv).correlation
                        if sr is not None and not np.isnan(float(sr)):
                            cell_spearmans.append(float(sr))
                    k = min(20, tv.shape[0])
                    if k >= 2:
                        idx_top = np.argsort(tv)[-k:]
                        pv_top = pv[idx_top]
                        tv_top = tv[idx_top]
                        if np.std(pv_top) > 0 and np.std(tv_top) > 0:
                            r20 = float(np.corrcoef(pv_top, tv_top)[0, 1])
                            if not np.isnan(r20):
                                cell_pearsons_top20.append(r20)

                if n_batches >= max_batches_local:
                    break

        self.model.train()

        if dist.is_initialized():
            def _reduce(value):
                tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                return tensor.item()

            total_sq = _reduce(total_sq)
            total_abs = _reduce(total_abs)
            total_cnt = int(_reduce(total_cnt))
            total_eval_cells = int(_reduce(total_eval_cells))
            total_corr_cnt = int(_reduce(total_corr_cnt))
            sum_x = _reduce(sum_x)
            sum_y = _reduce(sum_y)
            sum_x2 = _reduce(sum_x2)
            sum_y2 = _reduce(sum_y2)
            sum_xy = _reduce(sum_xy)

            for lst in (
                cell_pearsons,
                cell_spearmans,
                cell_pearsons_top20,
                depth_proxy_values,
                depth_proxy_pearsons,
                cell_pred_means,
                cell_true_means,
                cell_pred_stds,
                cell_true_stds,
            ):
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, lst)
                lst.clear()
                for sub in gathered:
                    lst.extend(sub)

            gathered_true = [None] * dist.get_world_size()
            gathered_pred = [None] * dist.get_world_size()
            gathered_score = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_true, masked_true_chunks)
            dist.all_gather_object(gathered_pred, masked_pred_chunks)
            dist.all_gather_object(gathered_score, masked_score_chunks)
            masked_true_chunks = []
            masked_pred_chunks = []
            masked_score_chunks = []
            for sub in gathered_true:
                masked_true_chunks.extend(sub)
            for sub in gathered_pred:
                masked_pred_chunks.extend(sub)
            for sub in gathered_score:
                masked_score_chunks.extend(sub)

            gathered_true_nz = [None] * dist.get_world_size()
            gathered_pred_nz = [None] * dist.get_world_size()
            gathered_score_nz = [None] * dist.get_world_size()
            gathered_score_z = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_true_nz, masked_nonzero_true_chunks)
            dist.all_gather_object(gathered_pred_nz, masked_nonzero_pred_chunks)
            dist.all_gather_object(gathered_score_nz, masked_nonzero_score_chunks)
            dist.all_gather_object(gathered_score_z, masked_zero_score_chunks)
            masked_nonzero_true_chunks = []
            masked_nonzero_pred_chunks = []
            masked_nonzero_score_chunks = []
            masked_zero_score_chunks = []
            for sub in gathered_true_nz:
                masked_nonzero_true_chunks.extend(sub)
            for sub in gathered_pred_nz:
                masked_nonzero_pred_chunks.extend(sub)
            for sub in gathered_score_nz:
                masked_nonzero_score_chunks.extend(sub)
            for sub in gathered_score_z:
                masked_zero_score_chunks.extend(sub)

        mse = total_sq / max(total_cnt, 1)
        mae = total_abs / max(total_cnt, 1)
        denom = np.sqrt(
            max(total_corr_cnt * sum_x2 - sum_x ** 2, 0.0) *
            max(total_corr_cnt * sum_y2 - sum_y ** 2, 0.0)
        )
        pearson_all = None if denom <= 0 else float(
            (total_corr_cnt * sum_xy - sum_x * sum_y) / denom
        )
        depth_binned_pearson = _summarize_depth_binned_pearson(depth_proxy_values, depth_proxy_pearsons)

        if masked_true_chunks:
            y_true_all = np.concatenate(masked_true_chunks).astype(np.float32, copy=False)
            y_pred_all = np.concatenate(masked_pred_chunks).astype(np.float32, copy=False)
            y_score_all = np.concatenate(masked_score_chunks).astype(np.float32, copy=False)
            y_true_bin = (y_true_all > 0).astype(np.int32, copy=False)
            val_masked_nonzero = int(y_true_bin.sum())
            val_positive_rate = float(y_true_bin.mean()) if y_true_bin.size > 0 else 0.0
            val_binary_open_rate = float((y_score_all > open_threshold).mean()) if y_score_all.size > 0 else 0.0
            y_pred_zero = y_pred_all[y_true_all <= 0]
            if y_pred_zero.size > 0:
                pred_mean_zero_global = float(y_pred_zero.mean())
                pred_std_zero_global = float(y_pred_zero.std())
            else:
                pred_mean_zero_global = None
                pred_std_zero_global = None
            try:
                val_auroc = float(roc_auc_score(y_true_bin, y_score_all))
                val_auprc = float(average_precision_score(y_true_bin, y_score_all))
            except ValueError:
                val_auroc = 0.5
                val_auprc = val_positive_rate
        else:
            val_auroc = 0.5
            val_auprc = 0.0
            val_masked_nonzero = 0
            val_positive_rate = 0.0
            val_binary_open_rate = 0.0
            pred_mean_zero_global = None
            pred_std_zero_global = None

        if masked_nonzero_true_chunks:
            y_true_nonzero = np.concatenate(masked_nonzero_true_chunks).astype(np.float32, copy=False)
            y_pred_nonzero = np.concatenate(masked_nonzero_pred_chunks).astype(np.float32, copy=False)
            pred_mean_global = float(y_pred_nonzero.mean())
            true_mean_global = float(y_true_nonzero.mean())
            pred_std_global = float(y_pred_nonzero.std())
            true_std_global = float(y_true_nonzero.std())
        else:
            pred_mean_global = None
            true_mean_global = None
            pred_std_global = None
            true_std_global = None
        if masked_nonzero_score_chunks and masked_zero_score_chunks:
            y_score_nonzero = np.concatenate(masked_nonzero_score_chunks).astype(np.float32, copy=False)
            y_score_zero = np.concatenate(masked_zero_score_chunks).astype(np.float32, copy=False)
            open_prob_nonzero_global = float(y_score_nonzero.mean())
            open_prob_zero_global = float(y_score_zero.mean())
            binary_margin = float(open_prob_nonzero_global - open_prob_zero_global)
        else:
            open_prob_nonzero_global = None
            open_prob_zero_global = None
            binary_margin = None

        val_rmse = float(np.sqrt(mse))
        val_pearson_cell = float(np.mean(cell_pearsons)) if cell_pearsons else None
        val_spearman_cell = float(np.mean(cell_spearmans)) if cell_spearmans else None
        val_pearson_top20 = float(np.mean(cell_pearsons_top20)) if cell_pearsons_top20 else None
        val_pred_mean_nonzero_cellavg = float(np.mean(cell_pred_means)) if cell_pred_means else None
        val_true_mean_nonzero_cellavg = float(np.mean(cell_true_means)) if cell_true_means else None
        val_pred_std_nonzero_cellavg = float(np.mean(cell_pred_stds)) if cell_pred_stds else None
        val_true_std_nonzero_cellavg = float(np.mean(cell_true_stds)) if cell_true_stds else None

        return {
            # Endpoint metrics use the final hard-gated output.
            "val_mse": mse,
            "val_rmse": val_rmse,
            "val_mae": mae,
            "val_endpoint_mse": mse,
            "val_endpoint_rmse": val_rmse,
            "val_endpoint_mae": mae,
            "val_endpoint_pred_mean_zero_global": pred_mean_zero_global,
            "val_endpoint_pred_std_zero_global": pred_std_zero_global,
            "val_endpoint_open_rate": val_binary_open_rate,
            "val_open_threshold": float(open_threshold),

            # Intensity metrics ignore the binary gate and are evaluated only on masked nonzero targets.
            "val_pearson_cell": val_pearson_cell,
            "val_pearson_all": pearson_all,
            "val_spearman_cell": val_spearman_cell,
            "val_pearson_top20": val_pearson_top20,
            "val_intensity_pearson_cell": val_pearson_cell,
            "val_intensity_pearson_all": pearson_all,
            "val_intensity_spearman_cell": val_spearman_cell,
            "val_intensity_pearson_top20": val_pearson_top20,
            "val_intensity_pred_mean_nonzero_global": pred_mean_global,
            "val_intensity_true_mean_nonzero_global": true_mean_global,
            "val_intensity_pred_std_nonzero_global": pred_std_global,
            "val_intensity_true_std_nonzero_global": true_std_global,
            "val_intensity_pred_mean_nonzero_cellavg": val_pred_mean_nonzero_cellavg,
            "val_intensity_true_mean_nonzero_cellavg": val_true_mean_nonzero_cellavg,
            "val_intensity_pred_std_nonzero_cellavg": val_pred_std_nonzero_cellavg,
            "val_intensity_true_std_nonzero_cellavg": val_true_std_nonzero_cellavg,

            # Binary metrics use p_open.
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "val_binary_auroc": val_auroc,
            "val_binary_auprc": val_auprc,
            "val_binary_open_rate": val_binary_open_rate,
            "val_eval_cells": int(total_eval_cells),
            "val_masked_total": int(total_cnt),
            "val_masked_nonzero": int(val_masked_nonzero),
            "val_positive_rate": float(val_positive_rate),
            # Backward-compatible mixed names kept for older scripts.
            "val_pred_mean_nonzero_global": pred_mean_global,
            "val_true_mean_nonzero_global": true_mean_global,
            "val_pred_std_nonzero_global": pred_std_global,
            "val_true_std_nonzero_global": true_std_global,
            "val_pred_mean_zero_global": pred_mean_zero_global,
            "val_pred_std_zero_global": pred_std_zero_global,
            "val_open_prob_nonzero_global": open_prob_nonzero_global,
            "val_open_prob_zero_global": open_prob_zero_global,
            "val_binary_margin": binary_margin,
            "val_pred_mean_nonzero_cellavg": val_pred_mean_nonzero_cellavg,
            "val_true_mean_nonzero_cellavg": val_true_mean_nonzero_cellavg,
            "val_pred_std_nonzero_cellavg": val_pred_std_nonzero_cellavg,
            "val_true_std_nonzero_cellavg": val_true_std_nonzero_cellavg,
            "val_depth_binned_pearson_cell": depth_binned_pearson,
        }

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
        self._print("====== 单阶段 epoch 训练 (scATAC) ======")
        self.model.train()

        optimizer = base_train.optim.AdamW(
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
        fallback_best_record = None
        fallback_best_selection_summary = None
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
                fallback_best_record = ckpt.get("fallback_best_record")
                fallback_best_selection_summary = ckpt.get("fallback_best_selection_summary")
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

        guard_multiplier = float(getattr(self, "val_auprc_guard_multiplier", DEFAULT_AUPRC_GUARD_MULTIPLIER))

        for epoch in range(start_epoch + 1, epochs + 1):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            epoch_metrics = {
                "train_loss": 0.0,
                "x1_loss": 0.0,
                "weighted_x1_loss": 0.0,
                "pearson_loss": 0.0,
                "cell_pearson_loss": 0.0,
                "pearson_nonzero_loss": 0.0,
                "binary_loss": 0.0,
                "binary_rank_loss": 0.0,
            }
            metric_sources = {
                "train_loss": "loss",
                "x1_loss": "x1_loss",
                "weighted_x1_loss": "weighted_x1_loss",
                "pearson_loss": "pearson_loss",
                "cell_pearson_loss": "cell_pearson_loss",
                "pearson_nonzero_loss": "pearson_nonzero_loss",
                "binary_loss": "binary_loss",
                "binary_rank_loss": "binary_rank_loss",
            }
            n_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                disable=not base_train.is_main(),
                leave=False,
            )
            steps_in_epoch = len(train_loader)
            for batch_idx, (g_id, x_1, x_0, mask_ind) in enumerate(pbar, start=1):
                g_id = g_id.to(self.device)
                x_1 = x_1.to(self.device)
                x_0 = x_0.to(self.device)
                mask_ind = mask_ind.to(self.device)

                t = base_train.sample_t_stratified(g_id.shape[0], self.device)
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
                    pr_nz=f"{losses['pearson_nonzero_loss'].item():.4f}",
                    bce=f"{losses['binary_loss'].item():.4f}",
                    rank=f"{losses['binary_rank_loss'].item():.4f}",
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
                    if base_train.is_main():
                        print(
                            f"\n[val step={global_step} epoch={epoch}] "
                            f"endpoint_rmse={vm.get('val_endpoint_rmse', 0):.4f}  "
                            f"endpoint_mae={vm.get('val_endpoint_mae', 0):.4f}  "
                            f"int_pearson_cell={vm.get('val_intensity_pearson_cell') or 0:.4f}  "
                            f"int_pearson_all={vm.get('val_intensity_pearson_all') or 0:.4f}  "
                            f"int_spearman_cell={vm.get('val_intensity_spearman_cell') or 0:.4f}  "
                            f"int_pearson_top20={vm.get('val_intensity_pearson_top20') or 0:.4f}  "
                            f"binary_auprc={vm.get('val_binary_auprc', 0):.4f}"
                        )
                        print(
                            "  "
                            f"eval_cells={vm.get('val_eval_cells', 0)}  "
                            f"masked_total={vm.get('val_masked_total', 0)}  "
                            f"masked_nonzero={vm.get('val_masked_nonzero', 0)}  "
                            f"positive_rate={vm.get('val_positive_rate', 0):.4f}"
                        )
                        print(
                            "  "
                            f"pred_mean_nz={vm.get('val_pred_mean_nonzero_global') or 0:.4f}  "
                            f"true_mean_nz={vm.get('val_true_mean_nonzero_global') or 0:.4f}  "
                            f"pred_std_nz={vm.get('val_pred_std_nonzero_global') or 0:.4f}  "
                            f"true_std_nz={vm.get('val_true_std_nonzero_global') or 0:.4f}"
                        )
                        print(
                            "  "
                            f"pred_mean_zero={vm.get('val_pred_mean_zero_global') or 0:.4f}  "
                            f"pred_std_zero={vm.get('val_pred_std_zero_global') or 0:.4f}  "
                            f"binary_margin={vm.get('val_binary_margin') or 0:.4f}  "
                            f"binary_open_rate={vm.get('val_binary_open_rate') or 0:.4f}"
                        )
                        print(
                            "  "
                            f"p_open_nz={vm.get('val_open_prob_nonzero_global') or 0:.4f}  "
                            f"p_open_zero={vm.get('val_open_prob_zero_global') or 0:.4f}"
                        )
                        depth_bins = vm.get("val_depth_binned_pearson_cell") or {}
                        if depth_bins:
                            printable = []
                            for name, info in depth_bins.items():
                                mean_r = info.get("mean_pearson")
                                printable.append(
                                    f"{name}:n={info.get('n_cells', 0)},r={0.0 if mean_r is None else mean_r:.4f}"
                                )
                            print("  depth_binned_pearson:", " | ".join(printable))

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
                        "pearson_nonzero_loss": float(losses["pearson_nonzero_loss"].item()),
                        "binary_loss": float(losses["binary_loss"].item()),
                        "binary_rank_loss": float(losses["binary_rank_loss"].item()),
                    }
                    if vm is not None:
                        step_record.update(vm)

                        fallback_should_update = (
                            fallback_best_record is None
                            or (
                                step_record.get("val_binary_auprc") is not None
                                and float(step_record["val_binary_auprc"]) > float(
                                    fallback_best_record.get("val_binary_auprc", fallback_best_record.get("val_auprc", -1.0))
                                )
                            )
                        )
                        if fallback_should_update:
                            fallback_best_record = dict(step_record)
                            fallback_best_selection_summary = {
                                "selection_metric": "val_binary_auprc",
                                "reason": "best_binary_auprc_regardless_of_guard",
                            }
                            self._save_checkpoint(
                                epoch,
                                global_step,
                                tag="checkpoint_best_fallback",
                                optimizer_state_dict=optimizer.state_dict(),
                                scheduler_state_dict=scheduler.state_dict(),
                                best_record=best_record,
                                best_selection_summary=best_selection_summary,
                                fallback_best_record=fallback_best_record,
                                fallback_best_selection_summary=fallback_best_selection_summary,
                            )

                        should_update_best, selection_summary = should_update_best_checkpoint_atac(
                            current_metrics=step_record,
                            best_metrics=best_record,
                            guard_multiplier=guard_multiplier,
                        )
                        step_record["best_selection_summary"] = selection_summary
                        if should_update_best:
                            best_record = dict(step_record)
                            best_selection_summary = dict(selection_summary)
                            best_record["best_guard_satisfied"] = True
                            self._save_checkpoint(
                                epoch,
                                global_step,
                                tag="checkpoint_best",
                                optimizer_state_dict=optimizer.state_dict(),
                                scheduler_state_dict=scheduler.state_dict(),
                                best_record=best_record,
                                best_selection_summary=best_selection_summary,
                                fallback_best_record=fallback_best_record,
                                fallback_best_selection_summary=fallback_best_selection_summary,
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

            if base_train.is_main():
                summary = (
                    f"[epoch {epoch}/{epochs}] "
                    f"loss={record['train_loss']:.4f} "
                    f"x1={record['x1_loss']:.4f} "
                    f"w_x1={record['weighted_x1_loss']:.4f} "
                    f"pearson={record['pearson_loss']:.4f} "
                    f"cell_pearson={record['cell_pearson_loss']:.4f} "
                    f"pearson_nz={record['pearson_nonzero_loss']:.4f} "
                    f"binary={record['binary_loss']:.4f} "
                    f"rank={record['binary_rank_loss']:.4f} "
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
                    fallback_best_record=fallback_best_record,
                    fallback_best_selection_summary=fallback_best_selection_summary,
                )

        if best_record is None and fallback_best_record is not None and base_train.is_main():
            fallback_payload = dict(fallback_best_record)
            fallback_payload["best_guard_satisfied"] = False
            fallback_payload["best_selection_summary"] = {
                **(fallback_best_selection_summary or {}),
                "reason": "fallback_highest_binary_auprc_without_guard",
                "guard_multiplier": float(guard_multiplier),
            }
            self._save_best_metrics(fallback_payload)
            print(
                "  [checkpoint] no guarded best checkpoint; "
                "kept checkpoint_best_fallback.pt and did not create checkpoint_best.pt"
            )

        self._save_checkpoint(
            epochs,
            global_step,
            tag="checkpoint_last",
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            best_record=best_record,
            best_selection_summary=best_selection_summary,
            fallback_best_record=fallback_best_record,
            fallback_best_selection_summary=fallback_best_selection_summary,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="scUniFlow scATAC 单阶段 epoch 训练")
    parser.add_argument("--config", type=str, default=str(ROOT / "config_atac.yaml"))
    parser.add_argument("--data", type=str, required=True, help="TF-IDF ATAC h5ad 路径")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--spectral_coords", type=str, default=None)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--debug_t_bucket_stats", action="store_true")
    parser.add_argument("--debug_t_bucket_every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--expr_transform",
        type=str,
        choices=["identity", "log1p", "tfidf_log1p_1e4", "logcpm"],
        default=None,
        help="ATAC 连续值空间，TF-IDF 默认使用 identity",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = base_train.setup_ddp()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    expr_transform = str(args.expr_transform or train_cfg.get("expression_transform", "identity"))
    train_cfg["expression_transform"] = expr_transform

    run_name = args.run_name or f"atac_{Path(args.data).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(train_cfg.get("experiment_dir", str(ROOT / "experiments")), run_name)
    if base_train.is_main():
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    if world_size > 1:
        dist.barrier()

    batch_size = args.batch_size or int(train_cfg.get("batch_size_per_gpu", 8))
    val_cfg_early = cfg.get("validation", {})
    val_batch_size = int(val_cfg_early.get("batch_size", batch_size))
    spectral_coords = args.spectral_coords or train_cfg.get("spectral_coords")
    train_nonzero_mask_ratio = float(train_cfg.get("nonzero_mask_ratio", 0.2))
    train_zero_mask_ratio = float(train_cfg.get("zero_mask_ratio", 0.1))
    seed = int(train_cfg.get("seed", 42))
    max_cells = args.max_cells if args.max_cells is not None else train_cfg.get("max_cells")
    cell_seed = int(train_cfg.get("cell_seed", seed))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    val_cells = train_cfg.get("val_cells")

    if base_train.is_main():
        print("=" * 60)
        print(f"实验目录: {exp_dir}")
        print(f"GPU 数量: {world_size}")
        print(f"表达空间: {expr_transform}")
        print("=" * 60)

    use_split = val_ratio > 0
    val_loader = None
    train_sampler = None

    if rank == 0:
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
            n_features = meta["n_features"]
            train_prior_mean, train_prior_std = compute_accessibility_prior_stats(
                train_X,
                expr_transform=expr_transform,
            )

            split_path = os.path.join(exp_dir, "split_info.json")
            with open(split_path, "w") as f:
                json.dump(
                    {
                        "train_donors": meta["train_donors"],
                        "val_donors": meta["val_donors"],
                        **meta["split_info_effective"],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[saved] 划分信息 -> {split_path}")
        else:
            train_X, feature_ids, _, meta = load_atac_h5ad(
                args.data,
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
            val_X = None
            n_features = meta["n_features"]
            train_prior_mean, train_prior_std = compute_accessibility_prior_stats(
                train_X,
                expr_transform=expr_transform,
            )

        if world_size > 1:
            shard_dir = tempfile.mkdtemp(prefix="scmam_atac_shards_", dir=ensure_tmp_root())
            n_train = train_X.shape[0]
            chunk = max(1, n_train // world_size)
            for shard_rank in range(world_size):
                start = shard_rank * chunk
                end = n_train if shard_rank == world_size - 1 else min((shard_rank + 1) * chunk, n_train)
                shard_path = os.path.join(shard_dir, f"train_{shard_rank}.pkl")
                with open(shard_path, "wb") as sf:
                    pickle.dump(train_X[start:end], sf)
            info_path = os.path.join(shard_dir, "info.pkl")
            with open(info_path, "wb") as sf:
                pickle.dump(
                    {
                        "feature_ids": feature_ids,
                        "n_features": n_features,
                        "val_X": val_X,
                        "shard_dir": shard_dir,
                        "train_prior_mean": train_prior_mean,
                        "train_prior_std": train_prior_std,
                    },
                    sf,
                )
            shard_dir_bytes = shard_dir.encode()
        else:
            shard_dir_bytes = b""

    if world_size > 1:
        obj_list = [shard_dir_bytes if rank == 0 else b""]
        dist.broadcast_object_list(obj_list, src=0)
        shard_dir = obj_list[0].decode()

        info_path = os.path.join(shard_dir, "info.pkl")
        with open(info_path, "rb") as sf:
            info = pickle.load(sf)
        feature_ids = info["feature_ids"]
        n_features = int(info["n_features"])
        train_prior_mean = float(info.get("train_prior_mean", 0.0))
        train_prior_std = float(max(info.get("train_prior_std", 1.0), 1e-6))

        my_shard_path = os.path.join(shard_dir, f"train_{rank}.pkl")
        with open(my_shard_path, "rb") as sf:
            my_train_X = pickle.load(sf)
        print(f"[rank {rank}] 加载分片: {my_train_X.shape[0]:,} cells")

        train_ds = scATACDataset(
            my_train_X,
            feature_ids,
            x0_prior="standard_normal",
            nonzero_mask_ratio=train_nonzero_mask_ratio,
            zero_mask_ratio=train_zero_mask_ratio,
            expr_transform=expr_transform,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_X = info["val_X"]
        if val_X is not None:
            val_ds = scATACDataset(
                val_X,
                feature_ids,
                x0_prior="standard_normal",
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                return_index=True,
                expr_transform=expr_transform,
            )
            val_sampler = base_train.DistributedEvalSampler(val_ds)
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=val_batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True,
            )

        del my_train_X
        dist.barrier()
    else:
        if use_split:
            train_loader, train_sampler = make_dataloader(
                train_X,
                feature_ids,
                batch_size=batch_size,
                shuffle=True,
                world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                expr_transform=expr_transform,
            )
            val_loader, _ = make_dataloader(
                val_X,
                feature_ids,
                batch_size=val_batch_size,
                shuffle=False,
                world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                return_index=True,
                expr_transform=expr_transform,
            )
        else:
            train_loader, train_sampler = make_dataloader(
                train_X,
                feature_ids,
                batch_size=batch_size,
                shuffle=True,
                world_size=1,
                nonzero_mask_ratio=train_nonzero_mask_ratio,
                zero_mask_ratio=train_zero_mask_ratio,
                expr_transform=expr_transform,
            )

    if base_train.is_main():
        print("=" * 60)
        print(
            f"模型: n_features={n_features}, d_model={model_cfg['d_model']}, "
            f"grid_size={model_cfg['grid_size']}, "
            f"n_layers={model_cfg.get('n_layers', 8)}, "
            f"d_state_row={model_cfg.get('d_state_row', 64)}, "
            f"d_state_col={model_cfg.get('d_state_col', 16)}"
        )
        print("=" * 60)

    trainer = scATACPipelineTrainer(
        n_genes=n_features,
        d_model=model_cfg["d_model"],
        grid_size=model_cfg["grid_size"],
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        local_rank=local_rank,
        exp_dir=exp_dir,
    )
    trainer.raw_model.mamba_encoder.activation_checkpointing = bool(
        model_cfg.get("activation_checkpointing", True)
    )
    loss_cfg = train_cfg.get("loss_weights", {})
    trainer.loss_weighted_x1 = float(loss_cfg.get("weighted_x1", 1.0))
    trainer.loss_pearson_weight = float(loss_cfg.get("pearson", 0.1))
    trainer.loss_cell_pearson_weight = float(loss_cfg.get("cell_pearson", 0.1))
    trainer.loss_pearson_nonzero_weight = float(loss_cfg.get("pearson_nonzero", 0.0))
    trainer.loss_binary_weight = float(loss_cfg.get("binary", 0.0))
    trainer.loss_binary_rank_weight = float(loss_cfg.get("binary_rank", 0.0))
    trainer.debug_t_bucket_stats = bool(args.debug_t_bucket_stats)
    trainer.debug_t_bucket_every = max(1, int(args.debug_t_bucket_every))
    val_cfg = cfg.get("validation", {})
    trainer.val_infer_n_steps = int(val_cfg.get("n_steps", cfg.get("inference", {}).get("n_steps", 10)))
    trainer.val_infer_temperature = float(val_cfg.get("temperature", cfg.get("inference", {}).get("temperature", 0.1)))
    infer_cfg = cfg.get("inference", {})
    trainer.inference_binary_threshold = float(
        infer_cfg.get("open_threshold", infer_cfg.get("binary_threshold", 0.5))
    )
    trainer.val_nonzero_mask_ratio = float(val_cfg.get("nonzero_mask_ratio", 0.2))
    trainer.val_zero_mask_ratio = float(val_cfg.get("zero_mask_ratio", 0.1))
    trainer.val_mask_seed = int(val_cfg.get("mask_seed", 1234))
    trainer.val_noise_seed = int(val_cfg.get("noise_seed", 5678))
    trainer.val_max_batches = int(val_cfg.get("max_batches", 200))
    trainer.val_auprc_guard_multiplier = float(
        val_cfg.get("auprc_guard_multiplier", DEFAULT_AUPRC_GUARD_MULTIPLIER)
    )
    trainer.expression_transform = expr_transform
    trainer.val_prior_mean = float(train_prior_mean)
    trainer.val_prior_std = float(max(train_prior_std, 1e-6))
    if base_train.is_main():
        print(
            f"[train nonzero stats] ({expr_transform}) "
            f"mean={float(train_prior_mean):.6f}, std={float(train_prior_std):.6f}"
        )
        print(
            f"[val prior] using empirical Gaussian from training data ({expr_transform}): "
            f"mean={trainer.val_prior_mean:.6f}, std={trainer.val_prior_std:.6f}"
        )
        print(
            f"[best checkpoint] selection_metric=val_binary_auprc, "
            f"guard=val_binary_auprc >= {trainer.val_auprc_guard_multiplier:.2f} * val_positive_rate"
        )
        print(
            "[model speed] "
            f"activation_checkpointing={trainer.raw_model.mamba_encoder.activation_checkpointing}"
        )

    if not spectral_coords:
        raise ValueError(
            "scATAC training requires spectral_coords; run "
            "src/scuniflow/tasks/task3_atac_imputation/generate_atac_coords.py first."
        )
    coords = np.load(spectral_coords)
    if coords.shape != (n_features, 2):
        raise ValueError(f"spectral_coords shape {coords.shape} != ({n_features}, 2)")
    trainer.raw_model.init_from_spectral(
        coords,
        slot_order_mode="hilbert",
        gene_order_mode="hilbert",
    )
    if base_train.is_main():
        routing_meta_path = os.path.join(exp_dir, "routing_info.json")
        with open(routing_meta_path, "w") as f:
            json.dump(
                {
                    "mode": "file",
                    "coords_source": os.path.abspath(spectral_coords),
                    "slot_order_mode": "hilbert",
                    "gene_order_mode": "hilbert",
                    "n_features": int(n_features),
                    "grid_size": int(model_cfg["grid_size"]),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[init] hard routing ready from: {os.path.abspath(spectral_coords)}")
        print(f"[saved] 路由信息 -> {routing_meta_path}")
    if world_size > 1:
        dist.barrier()

    train_schedule = base_train.resolve_epoch_training_config(train_cfg, steps_per_epoch=len(train_loader))
    if base_train.is_main():
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

    if base_train.is_main():
        final_path = os.path.join(exp_dir, "model_final.pt")
        torch.save(trainer.raw_model.state_dict(), final_path)
        print(f"\n最终模型 -> {final_path}")

    base_train.cleanup_ddp()


if __name__ == "__main__":
    main()
