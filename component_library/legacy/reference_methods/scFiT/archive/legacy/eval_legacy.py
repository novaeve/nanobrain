import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    TopKExpressionDataset,
    compute_log1p_nonzero_prior_stats,
    load_h5ad,
    load_h5ad_split,
    load_sea_ad,
    load_sea_ad_split,
)
from model import scFiTModel
from utils import build_deterministic_mask, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate scFiT with donor-level split")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _cfg_value(value, default):
    return default if value is None or value == "" else value


def _safe_pearson_np(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return None
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def _subsample_cells_per_donor(count_matrix, donor_ids, max_cells_per_donor: int, seed: int):
    donor_ids = np.asarray(donor_ids)
    if donor_ids.ndim != 1:
        raise ValueError("donor_ids 必须是一维数组")
    if count_matrix.shape[0] != donor_ids.shape[0]:
        raise ValueError("count_matrix 与 donor_ids 的 cell 数不一致")

    max_cells_per_donor = int(max_cells_per_donor)
    if max_cells_per_donor <= 0:
        raise ValueError("max_cells_per_donor 必须大于 0")

    rng = np.random.RandomState(seed)
    keep_indices = []
    donor_counts = {}
    for donor in np.unique(donor_ids):
        idx = np.flatnonzero(donor_ids == donor)
        take = min(len(idx), max_cells_per_donor)
        chosen = np.sort(rng.choice(idx, size=take, replace=False))
        keep_indices.append(chosen)
        donor_counts[str(donor)] = int(take)

    if keep_indices:
        keep_indices = np.sort(np.concatenate(keep_indices))
        count_matrix = count_matrix[keep_indices]
        donor_ids = donor_ids[keep_indices]
    else:
        keep_indices = np.array([], dtype=np.int64)
        count_matrix = count_matrix[:0]
        donor_ids = donor_ids[:0]

    return count_matrix, donor_ids, {
        "max_cells_per_donor": max_cells_per_donor,
        "seed": int(seed),
        "n_selected_cells": int(len(keep_indices)),
        "n_selected_donors": int(len(donor_counts)),
        "cells_per_donor": donor_counts,
    }


class MetricAggregator:
    def __init__(self):
        self.masked_count = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.cell_pearsons = []
        self.cell_spearmans = []
        self.cell_top20_pearsons = []

    def update(self, pred: torch.Tensor, target: torch.Tensor, supervision_mask: torch.Tensor):
        mask = supervision_mask.bool()
        if not mask.any():
            return

        pred_masked = pred[mask].detach().float().cpu()
        target_masked = target[mask].detach().float().cpu()
        diff = pred_masked - target_masked

        self.masked_count += int(diff.numel())
        self.sum_abs += float(diff.abs().sum().item())
        self.sum_sq += float((diff * diff).sum().item())
        self.sum_x += float(pred_masked.sum().item())
        self.sum_y += float(target_masked.sum().item())
        self.sum_x2 += float((pred_masked * pred_masked).sum().item())
        self.sum_y2 += float((target_masked * target_masked).sum().item())
        self.sum_xy += float((pred_masked * target_masked).sum().item())

        for b in range(pred.shape[0]):
            idx = supervision_mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue

            pred_b = pred[b, idx].detach().float().cpu().numpy().astype(np.float64)
            tgt_b = target[b, idx].detach().float().cpu().numpy().astype(np.float64)

            pearson = _safe_pearson_np(pred_b, tgt_b)
            if pearson is not None:
                self.cell_pearsons.append(pearson)

            sp = spearmanr(pred_b, tgt_b).correlation
            if sp is not None and not np.isnan(sp):
                self.cell_spearmans.append(float(sp))

            k = min(20, tgt_b.shape[0])
            if k >= 2:
                top_idx = np.argsort(tgt_b)[-k:]
                top20 = _safe_pearson_np(pred_b[top_idx], tgt_b[top_idx])
                if top20 is not None:
                    self.cell_top20_pearsons.append(top20)

    def finalize(self):
        if self.masked_count <= 0:
            return {
                "val_mse": 0.0,
                "val_rmse": 0.0,
                "val_mae": 0.0,
                "val_pearson_all": None,
                "val_pearson_cell": None,
                "val_spearman_cell": None,
                "val_pearson_top20_cell": None,
                "masked_positions": 0,
                "cells_used_for_corr": 0,
                "cells_used_for_spearman": 0,
                "cells_used_for_top20": 0,
            }

        pearson_all = None
        num = self.masked_count * self.sum_xy - self.sum_x * self.sum_y
        den_x = self.masked_count * self.sum_x2 - self.sum_x * self.sum_x
        den_y = self.masked_count * self.sum_y2 - self.sum_y * self.sum_y
        den = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        if den > 0:
            pearson_all = float(num / den)

        mse = self.sum_sq / self.masked_count
        return {
            "val_mse": float(mse),
            "val_rmse": float(math.sqrt(mse)),
            "val_mae": float(self.sum_abs / self.masked_count),
            "val_pearson_all": pearson_all,
            "val_pearson_cell": float(np.mean(self.cell_pearsons)) if self.cell_pearsons else None,
            "val_spearman_cell": float(np.mean(self.cell_spearmans)) if self.cell_spearmans else None,
            "val_pearson_top20_cell": float(np.mean(self.cell_top20_pearsons)) if self.cell_top20_pearsons else None,
            "masked_positions": int(self.masked_count),
            "cells_used_for_corr": int(len(self.cell_pearsons)),
            "cells_used_for_spearman": int(len(self.cell_spearmans)),
            "cells_used_for_top20": int(len(self.cell_top20_pearsons)),
        }


def _build_loader(cfg):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    split_cfg = data_cfg.get("split", {}) or {}

    data_key = str(data_cfg.get("source", "mtg_full"))
    max_cells = data_cfg.get("max_cells")
    cell_seed = int(data_cfg.get("cell_seed", train_cfg.get("seed", 42)))
    split = str(eval_cfg.get("split", "val"))
    max_cells_per_donor = _cfg_value(eval_cfg.get("max_cells_per_donor"), None)
    donor_sample_seed = int(_cfg_value(eval_cfg.get("donor_sample_seed"), eval_cfg.get("seed", train_cfg.get("seed", 42))))

    val_ratio = float(_cfg_value(eval_cfg.get("val_ratio"), train_cfg.get("val_ratio", 0.1)))
    split_seed = int(_cfg_value(eval_cfg.get("split_seed"), train_cfg.get("seed", 42)))
    split_strategy = str(split_cfg.get("strategy", "") or "").strip().lower()
    if not split_strategy:
        split_strategy = "donor" if data_key in ("mtg_full", "dlpfc_full") else "none"

    if split == "full":
        if data_key in ("mtg_full", "dlpfc_full", "mtg_toy", "dlpfc_toy"):
            count_matrix, gene_ids, ensembl_ids, meta = load_sea_ad(
                data_key,
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
        else:
            count_matrix, gene_ids, ensembl_ids, meta = load_h5ad(
                data_key,
                max_cells=max_cells,
                cell_seed=cell_seed,
            )
        donor_ids = None
        if max_cells_per_donor is not None:
            if "obs" not in meta or "donor_id" not in meta["obs"]:
                raise ValueError("full split 按 donor 采样需要 meta['obs']['donor_id']")
            donor_ids = np.asarray(meta["obs"]["donor_id"].values)
        prior_source = count_matrix
    else:
        if split_strategy == "donor" and data_key in ("mtg_full", "dlpfc_full") and not split_cfg:
            train_data, val_data, ensembl_ids, meta = load_sea_ad_split(
                data_key,
                val_ratio=val_ratio,
                seed=split_seed,
                max_cells=max_cells,
                cell_seed=cell_seed,
                return_donor_ids=max_cells_per_donor is not None,
            )
        else:
            train_data, val_data, ensembl_ids, meta = load_h5ad_split(
                data_key,
                split_strategy=split_strategy,
                split_column=split_cfg.get("column"),
                donor_csv=split_cfg.get("donor_csv"),
                val_ratio=val_ratio,
                seed=split_seed,
                max_cells=max_cells,
                cell_seed=cell_seed,
                return_group_ids=max_cells_per_donor is not None and split_strategy == "donor",
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
        train_donor_ids = meta.get("train_obs_donors", meta.get("train_obs_group_ids"))
        val_donor_ids = meta.get("val_obs_donors", meta.get("val_obs_group_ids"))
        if split == "train":
            count_matrix = train_X
            donor_ids = train_donor_ids
        elif split == "val":
            count_matrix = val_X
            donor_ids = val_donor_ids
        else:
            raise ValueError(f"unsupported evaluation.split={split}")
        prior_source = train_X

    donor_sample_info = None
    if max_cells_per_donor is not None:
        if donor_ids is None:
            raise ValueError("当前 split 无法获取 donor_ids，不能按 donor 采样")
        count_matrix, donor_ids, donor_sample_info = _subsample_cells_per_donor(
            count_matrix,
            donor_ids,
            max_cells_per_donor=int(max_cells_per_donor),
            seed=donor_sample_seed,
        )

    top_k = int(cfg["model"]["top_k_genes"])
    dataset = TopKExpressionDataset(
        count_matrix,
        gene_ids,
        top_k=top_k,
        return_index=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", cfg.get("validation", {}).get("batch_size", 16))),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    meta = dict(meta)
    meta["eval_split"] = split
    meta["eval_cells"] = int(count_matrix.shape[0])
    meta["eval_top_k"] = top_k
    meta["split_seed"] = split_seed
    meta["val_ratio"] = val_ratio
    meta["donor_sample_info"] = donor_sample_info
    return loader, meta, prior_source


def _build_model(cfg, n_genes: int, device: torch.device, n_cell_types: int = 0):
    model_cfg = cfg["model"]
    aux_cfg = model_cfg.get("aux_bin_ce", {})
    model = scFiTModel(
        n_genes=n_genes,
        top_k_genes=int(model_cfg["top_k_genes"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        d_ff=int(model_cfg["d_ff"]),
        nlayers=int(model_cfg["nlayers"]),
        dropout=float(model_cfg["dropout"]),
        value_max=float(model_cfg.get("value_encoder", {}).get("max_value", 32.0)),
        share_adaln_projection=bool(model_cfg.get("adaln", {}).get("share_projection", False)),
        aux_bin_ce=bool(aux_cfg.get("enabled", False)),
        n_bins=int(aux_cfg.get("n_bins", 0)),
        n_cell_types=int(model_cfg.get("n_cell_types", n_cell_types)),
        pseudotime_head=bool(model_cfg.get("pseudotime_head", {}).get("enabled", False)),
    )
    return model.to(device)


def _load_checkpoint(model, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    prior_mean = ckpt.get("prior_mean") if isinstance(ckpt, dict) else None
    prior_std = ckpt.get("prior_std") if isinstance(ckpt, dict) else None
    return ckpt, prior_mean, prior_std


@torch.no_grad()
def evaluate(model, loader, cfg, device, prior_mean: float, prior_std: float):
    eval_cfg = cfg.get("evaluation", {})
    amp = bool(eval_cfg.get("amp", cfg.get("training", {}).get("amp", True))) and device.type == "cuda"
    mask_ratio = float(eval_cfg.get("mask_ratio", cfg.get("validation", {}).get("mask_ratio", 0.2)))
    mask_seed = int(eval_cfg.get("mask_seed", cfg.get("validation", {}).get("mask_seed", 1234)))
    n_steps = int(eval_cfg.get("n_steps", cfg.get("inference", {}).get("n_steps", 10)))
    max_batches = int(eval_cfg.get("max_batches", 0))

    model.eval()
    aggregator = MetricAggregator()
    total_cells = 0
    total_batches = 0

    iterator = tqdm(loader, desc="eval", leave=False)
    for batch_idx, batch in enumerate(iterator, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break

        gene_ids, x_1, valid_mask, cell_idx = batch
        gene_ids = gene_ids.to(device, non_blocking=True)
        x_1 = x_1.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True).bool()
        cell_idx = cell_idx.to(device, non_blocking=True)

        cls_mask = gene_ids.eq(model.cls_token_id)
        gene_valid_mask = valid_mask & (~cls_mask)
        padding_mask = ~valid_mask
        mask_indicator = build_deterministic_mask(
            gene_valid_mask,
            cell_idx,
            mask_ratio=mask_ratio,
            base_seed=mask_seed,
        )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model.inference(
                gene_ids,
                x_1,
                padding_mask=padding_mask,
                mask_indicator=mask_indicator,
                n_steps=n_steps,
                prior_mean=prior_mean,
                prior_std=prior_std,
            )

        supervision_mask = mask_indicator & gene_valid_mask
        aggregator.update(pred, x_1, supervision_mask)
        total_cells += int(gene_ids.shape[0])
        total_batches += 1

    metrics = aggregator.finalize()
    metrics["eval_batches"] = int(total_batches)
    metrics["eval_cells"] = int(total_cells)
    return metrics


def _build_result_path(cfg, checkpoint_path: str):
    eval_cfg = cfg.get("evaluation", {})
    explicit = str(eval_cfg.get("save_json", "") or "").strip()
    if explicit:
        return os.path.abspath(explicit)

    result_dir = os.path.abspath(str(eval_cfg.get("result_dir", "eval_results")))
    os.makedirs(result_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    split = str(eval_cfg.get("split", "val"))
    mask_seed = int(eval_cfg.get("mask_seed", cfg.get("validation", {}).get("mask_seed", 1234)))
    filename = f"eval_{ckpt_stem}_{split}_maskseed{mask_seed}.json"
    return os.path.join(result_dir, filename)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pairing_cfg = cfg.get("pairing", {}) or {}
    pairing_strategy = str(pairing_cfg.get("strategy", "") or "").strip().lower()
    if pairing_strategy == "latent_knn":
        from eval_temporal import run_temporal_eval

        run_temporal_eval(cfg, config_path=args.config)
        return

    eval_cfg = cfg.get("evaluation", {})
    checkpoint = str(eval_cfg.get("checkpoint", "") or "").strip()
    if not checkpoint:
        raise ValueError("请在 config.yaml 的 evaluation.checkpoint 中提供 checkpoint 路径")

    seed_everything(int(eval_cfg.get("seed", cfg.get("training", {}).get("seed", 42))))

    device = _resolve_device(str(eval_cfg.get("device", "cuda")))
    loader, data_meta, prior_source = _build_loader(cfg)
    model = _build_model(cfg, int(data_meta["n_genes"]), device, n_cell_types=len(data_meta.get("cell_type_names", [])))
    ckpt, prior_mean, prior_std = _load_checkpoint(model, checkpoint, device)

    if prior_mean is None or prior_std is None:
        prior_mean, prior_std = compute_log1p_nonzero_prior_stats(prior_source)

    t0 = time.time()
    metrics = evaluate(model, loader, cfg, device, float(prior_mean), float(prior_std))
    elapsed = time.time() - t0

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(checkpoint),
        "device": str(device),
        "prior_mean": float(prior_mean),
        "prior_std": float(prior_std),
        "elapsed_seconds": float(elapsed),
        "data": {
            "source": cfg.get("data", {}).get("source", "mtg_full"),
            "split": data_meta.get("eval_split"),
            "split_strategy": data_meta.get("split_strategy"),
            "eval_cells": int(data_meta.get("eval_cells", 0)),
            "top_k_genes": int(data_meta.get("eval_top_k", cfg["model"]["top_k_genes"])),
            "val_ratio": float(data_meta.get("val_ratio", _cfg_value(eval_cfg.get("val_ratio"), cfg.get("training", {}).get("val_ratio", 0.1)))),
            "split_seed": int(data_meta.get("split_seed", _cfg_value(eval_cfg.get("split_seed"), cfg.get("training", {}).get("seed", 42)))),
            "split_info": data_meta.get("split_info"),
            "split_info_effective": data_meta.get("split_info_effective"),
            "train_donors": data_meta.get("train_donors"),
            "val_donors": data_meta.get("val_donors"),
            "train_split_values": data_meta.get("train_split_values"),
            "val_split_values": data_meta.get("val_split_values"),
            "donor_sample_info": data_meta.get("donor_sample_info"),
        },
        "evaluation": {
            "batch_size": int(eval_cfg.get("batch_size", cfg.get("validation", {}).get("batch_size", 16))),
            "num_workers": int(eval_cfg.get("num_workers", 4)),
            "mask_ratio": float(eval_cfg.get("mask_ratio", cfg.get("validation", {}).get("mask_ratio", 0.2))),
            "mask_seed": int(eval_cfg.get("mask_seed", cfg.get("validation", {}).get("mask_seed", 1234))),
            "n_steps": int(eval_cfg.get("n_steps", cfg.get("inference", {}).get("n_steps", 10))),
            "max_batches": int(eval_cfg.get("max_batches", 0)),
            "max_cells_per_donor": (
                None
                if _cfg_value(eval_cfg.get("max_cells_per_donor"), None) is None
                else int(eval_cfg.get("max_cells_per_donor"))
            ),
            "donor_sample_seed": int(_cfg_value(eval_cfg.get("donor_sample_seed"), eval_cfg.get("seed", cfg.get("training", {}).get("seed", 42)))),
        },
        "metrics": metrics,
    }
    if isinstance(ckpt, dict):
        result["checkpoint_meta"] = {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step", ckpt.get("step")),
        }

    result_path = _build_result_path(cfg, checkpoint)
    result["result_json"] = result_path
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    with open(result_path, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
