import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "model.py").exists() and (p / "data.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_atac import (
    _transform_accessibility_numpy,
    compute_accessibility_prior_stats,
    load_atac_h5ad,
    load_atac_h5ad_split,
)
from train_atac import scATACPipelineTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="scUniFlow scATAC 推理脚本")
    parser.add_argument("--config", type=str, default=str(ROOT / "config_atac.yaml"))
    parser.add_argument("--data", type=str, required=True, help="TF-IDF ATAC h5ad 路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的 checkpoint 路径")
    parser.add_argument("--output", type=str, required=True, help="输出 *_imputed.h5ad 路径")
    parser.add_argument("--mode", type=str, default="all_zero", choices=["masked_eval", "all_zero"])
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--spectral_coords", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument(
        "--expr_transform",
        type=str,
        choices=["identity", "log1p", "tfidf_log1p_1e4", "logcpm"],
        default=None,
    )
    return parser.parse_args()


def _build_trainer(cfg, n_features, prior_mean, prior_std, spectral_coords, checkpoint_path, expr_transform, exp_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("impute_atac.py 当前要求可用 CUDA 环境。请先设置 CUDA_VISIBLE_DEVICES。")

    model_cfg = cfg["model"]
    val_cfg = cfg.get("validation", {})
    infer_cfg = cfg.get("inference", {})

    trainer = scATACPipelineTrainer(
        n_genes=n_features,
        d_model=model_cfg["d_model"],
        grid_size=model_cfg["grid_size"],
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        local_rank=0,
        exp_dir=exp_dir,
    )
    trainer.val_infer_n_steps = int(val_cfg.get("n_steps", infer_cfg.get("n_steps", 10)))
    trainer.val_infer_temperature = float(val_cfg.get("temperature", infer_cfg.get("temperature", 0.1)))
    trainer.val_nonzero_mask_ratio = float(val_cfg.get("nonzero_mask_ratio", 0.2))
    trainer.val_zero_mask_ratio = float(val_cfg.get("zero_mask_ratio", 0.1))
    trainer.val_mask_seed = int(val_cfg.get("mask_seed", 1234))
    trainer.val_noise_seed = int(val_cfg.get("noise_seed", 5678))
    trainer.expression_transform = expr_transform
    trainer.val_prior_mean = float(prior_mean)
    trainer.val_prior_std = float(max(prior_std, 1e-6))
    trainer.inference_hard_gate = bool(infer_cfg.get("hard_gate", True))
    trainer.inference_binary_threshold = float(
        infer_cfg.get("open_threshold", infer_cfg.get("binary_threshold", 0.5))
    )
    trainer.inference_gate_strategy = str(infer_cfg.get("gate_strategy", "threshold")).lower()
    if trainer.inference_gate_strategy in {"per_cell_topk", "top-k"}:
        trainer.inference_gate_strategy = "top_k"
    if trainer.inference_gate_strategy in {"depth_scaled_topk", "depth_scaled_top-k"}:
        trainer.inference_gate_strategy = "depth_scaled_top_k"
    if trainer.inference_gate_strategy not in {"threshold", "top_k", "threshold_top_k", "depth_scaled_top_k"}:
        raise ValueError(
            "inference.gate_strategy must be 'threshold', 'top_k', 'threshold_top_k', "
            "or 'depth_scaled_top_k', "
            f"got {trainer.inference_gate_strategy!r}"
        )
    trainer.inference_top_k = infer_cfg.get("top_k", infer_cfg.get("per_cell_top_k", None))
    if trainer.inference_top_k is not None:
        trainer.inference_top_k = int(trainer.inference_top_k)
        if trainer.inference_top_k <= 0:
            raise ValueError("inference.top_k must be positive when provided.")
    trainer.inference_top_k_base_rate = float(infer_cfg.get("top_k_base_rate", 0.05))
    trainer.inference_top_k_min = int(infer_cfg.get("top_k_min", 50))
    trainer.inference_top_k_max = int(infer_cfg.get("top_k_max", 2000))
    trainer.inference_specificity_alpha = float(infer_cfg.get("specificity_alpha", 0.0))
    trainer.inference_specificity_min = infer_cfg.get("specificity_min", None)
    if trainer.inference_specificity_min is not None:
        trainer.inference_specificity_min = float(trainer.inference_specificity_min)
    if trainer.inference_top_k_min < 0 or trainer.inference_top_k_max <= 0:
        raise ValueError("inference.top_k_min must be >= 0 and top_k_max must be > 0.")
    if trainer.inference_top_k_min > trainer.inference_top_k_max:
        raise ValueError("inference.top_k_min cannot exceed top_k_max.")
    trainer.inference_intensity_mode = str(infer_cfg.get("intensity_mode", "rollout")).lower()
    if trainer.inference_intensity_mode not in {"rollout", "one_step"}:
        raise ValueError(
            "inference.intensity_mode must be 'rollout' or 'one_step', "
            f"got {trainer.inference_intensity_mode!r}"
        )
    trainer.inference_intensity_t = float(infer_cfg.get("intensity_t", 0.0))

    coords = np.load(spectral_coords)
    if coords.shape != (n_features, 2):
        raise ValueError(f"spectral_coords shape {coords.shape} != ({n_features}, 2)")
    trainer.raw_model.init_from_spectral(
        coords,
        slot_order_mode="hilbert",
        gene_order_mode="hilbert",
    )

    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.raw_model.load_state_dict(ckpt["model_state_dict"])
    trainer.raw_model.eval()
    trainer.model.eval()
    return trainer, ckpt


def _build_gate(
    candidate_mask,
    p_open,
    threshold,
    strategy="threshold",
    top_k=None,
    observed_nonzero_counts=None,
    top_k_base_rate=0.05,
    top_k_min=50,
    top_k_max=2000,
    rank_score=None,
    specificity_score=None,
    specificity_min=None,
):
    """Build binary gate over candidate zero positions."""
    strategy = str(strategy).lower()
    if strategy == "threshold":
        return candidate_mask & (p_open > float(threshold))

    if strategy == "depth_scaled_top_k":
        if observed_nonzero_counts is None:
            raise ValueError("observed_nonzero_counts is required for depth_scaled_top_k.")
        per_cell_k = (
            observed_nonzero_counts.float() * float(top_k_base_rate)
        ).long().clamp(min=int(top_k_min), max=int(top_k_max))
    elif top_k is not None:
        per_cell_k = torch.full(
            (candidate_mask.shape[0],),
            int(top_k),
            device=candidate_mask.device,
            dtype=torch.long,
        )
    else:
        raise ValueError(f"inference.top_k is required for gate_strategy={strategy!r}")

    base_mask = candidate_mask
    if strategy == "threshold_top_k":
        base_mask = candidate_mask & (p_open > float(threshold))
    if specificity_score is not None and specificity_min is not None:
        base_mask = base_mask & (specificity_score > float(specificity_min))

    gate = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if rank_score is None:
        rank_score = p_open
    for b in range(candidate_mask.shape[0]):
        idx = base_mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        n_keep = min(int(per_cell_k[b].item()), int(idx.numel()))
        if n_keep <= 0:
            continue
        scores = rank_score[b, idx]
        chosen_local = torch.topk(scores, k=n_keep, largest=True).indices
        gate[b, idx[chosen_local]] = True
    return gate


def _load_specificity_prior(path, output_obs_names, var_names):
    if not path:
        return None
    payload = np.load(path, allow_pickle=True)
    specificity = np.asarray(payload["specificity"], dtype=np.float32)
    prior_var_names = np.asarray(payload["var_names"]).astype(str)
    cur_var_names = np.asarray(var_names).astype(str)
    if prior_var_names.shape[0] != cur_var_names.shape[0] or np.any(prior_var_names != cur_var_names):
        raise ValueError("specificity prior var_names do not match current h5ad peak order.")

    prior_obs_names = np.asarray(payload["obs_names"]).astype(str)
    prior_obs_to_pos = {name: i for i, name in enumerate(prior_obs_names)}
    prior_cluster_idx = np.asarray(payload["obs_cluster_idx"], dtype=np.int64)
    output_obs_names = np.asarray(output_obs_names).astype(str)
    cell_cluster_idx = np.full((output_obs_names.shape[0],), -1, dtype=np.int64)
    missing = 0
    for i, name in enumerate(output_obs_names):
        pos = prior_obs_to_pos.get(name)
        if pos is None:
            missing += 1
            continue
        cell_cluster_idx[i] = int(prior_cluster_idx[pos])
    if missing:
        raise ValueError(f"specificity prior missing labels for {missing} output cells.")

    return {
        "path": str(Path(path).resolve()),
        "specificity": specificity,
        "cell_cluster_idx": cell_cluster_idx,
        "cluster_names": np.asarray(payload["cluster_names"]).astype(str),
    }


@torch.no_grad()
def _predict_intensity(trainer, g_id, observed, eval_mask, selected_cell_indices=None, x_ref=None):
    """Predict intensity for eval_mask positions using rollout or one-step mode."""
    mode = getattr(trainer, "inference_intensity_mode", "rollout")
    if mode == "rollout":
        if x_ref is None:
            x_ref = observed
        init_masked_values = trainer._make_val_initial_masked_values(
            x_ref,
            eval_mask,
            cell_indices=selected_cell_indices,
        )
        return trainer.raw_model.inference(
            g_id,
            observed,
            eval_mask,
            n_steps=trainer.val_infer_n_steps,
            temperature=trainer.val_infer_temperature,
            prior_mean=trainer.val_prior_mean,
            prior_std=trainer.val_prior_std,
            initial_masked_values=init_masked_values,
        )

    context, _ = trainer.raw_model.compute_context(
        g_id,
        observed,
        eval_mask,
        temperature=0.0,
    )
    pred_full = observed.clone()
    t_value = float(getattr(trainer, "inference_intensity_t", 0.0))
    t = torch.full((observed.shape[0], 1), t_value, device=observed.device, dtype=observed.dtype)
    for b in range(observed.shape[0]):
        m_idx = eval_mask[b].nonzero(as_tuple=True)[0]
        if m_idx.numel() == 0:
            continue
        ctx_m = context[b, m_idx, :].unsqueeze(0)
        xt_m = observed[b, m_idx].unsqueeze(0)
        pred = trainer.raw_model.vector_field(xt_m, ctx_m, t[b].unsqueeze(0))
        pred_full[b, m_idx] = torch.clamp(pred.squeeze(0), min=0.0)
    return pred_full


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    expr_transform = str(args.expr_transform or train_cfg.get("expression_transform", "identity"))
    spectral_coords = args.spectral_coords or train_cfg.get("spectral_coords")
    if not spectral_coords:
        raise ValueError("必须提供 spectral_coords。")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 42))
    cell_seed = int(train_cfg.get("cell_seed", seed))
    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    val_cells = train_cfg.get("val_cells")

    if args.split == "all":
        count_matrix, feature_ids, _, meta = load_atac_h5ad(
            args.data,
            max_cells=args.max_cells,
            cell_seed=cell_seed,
        )
        prior_source = count_matrix
        selected_indices = np.arange(count_matrix.shape[0], dtype=np.int64)
        output_obs = meta["obs"].copy()
        split_info = None
    else:
        train_data, val_data, _, meta = load_atac_h5ad_split(
            args.data,
            val_ratio=val_ratio,
            val_cells=val_cells,
            seed=seed,
            max_cells=args.max_cells,
            cell_seed=cell_seed,
        )
        train_X, feature_ids = train_data
        val_X, _ = val_data
        prior_source = train_X
        if args.split == "train":
            count_matrix = train_X
            selected_indices = np.asarray(meta["train_indices"], dtype=np.int64)
        else:
            count_matrix = val_X
            selected_indices = np.asarray(meta["val_indices"], dtype=np.int64)
        output_obs = meta["obs"].iloc[selected_indices].copy()
        split_info = meta.get("split_info_effective")

    n_cells, n_features = count_matrix.shape
    prior_mean, prior_std = compute_accessibility_prior_stats(
        prior_source,
        expr_transform=expr_transform,
    )
    trainer, ckpt = _build_trainer(
        cfg=cfg,
        n_features=n_features,
        prior_mean=prior_mean,
        prior_std=prior_std,
        spectral_coords=spectral_coords,
        checkpoint_path=args.checkpoint,
        expr_transform=expr_transform,
        exp_dir=str(output_path.parent),
    )
    infer_cfg = cfg.get("inference", {})
    specificity_prior = _load_specificity_prior(
        infer_cfg.get("specificity_prior", None),
        output_obs.index.astype(str),
        meta["var"].index.astype(str),
    )
    specificity_tensor = None
    specificity_cell_cluster_idx = None
    if specificity_prior is not None:
        specificity_tensor = torch.from_numpy(specificity_prior["specificity"]).to(trainer.device)
        specificity_cell_cluster_idx = specificity_prior["cell_cluster_idx"]
        print(
            "[specificity] loaded "
            f"{specificity_prior['path']} "
            f"clusters={specificity_tensor.shape[0]} peaks={specificity_tensor.shape[1]} "
            f"alpha={trainer.inference_specificity_alpha}"
        )

    memmap_fd, memmap_path = tempfile.mkstemp(
        prefix=f"{output_path.stem}_",
        suffix=".float32.memmap",
        dir=str(output_path.parent),
    )
    os.close(memmap_fd)
    imputed_mm = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(n_cells, n_features))
    feature_ids_t = torch.as_tensor(feature_ids, dtype=torch.long, device=trainer.device).unsqueeze(0)

    total_masked = 0
    total_nonzero_masked = 0
    total_filled_zero = 0
    total_candidate_zero = 0
    sum_p_open_candidate_zero = 0.0
    sum_pred_filled_zero = 0.0
    pbar = tqdm(range(0, n_cells, args.batch_size), desc=f"Impute ({args.mode})", leave=False)
    with torch.no_grad():
        for start in pbar:
            end = min(start + args.batch_size, n_cells)
            rows = count_matrix[start:end]
            if sp.issparse(rows):
                x_raw = rows.toarray().astype(np.float32, copy=False)
            else:
                x_raw = np.asarray(rows, dtype=np.float32)

            x_np = _transform_accessibility_numpy(x_raw, expr_transform=expr_transform)
            x_t = torch.from_numpy(x_np).to(trainer.device)
            g_id = feature_ids_t.expand(end - start, -1)

            t_binary = torch.zeros((end - start, 1), device=trainer.device, dtype=x_t.dtype)
            specificity_score = None
            rank_score = None
            if specificity_tensor is not None:
                cluster_idx = torch.as_tensor(
                    specificity_cell_cluster_idx[start:end],
                    dtype=torch.long,
                    device=trainer.device,
                )
                specificity_score = specificity_tensor.index_select(0, cluster_idx).to(dtype=x_t.dtype)
            if args.mode == "all_zero":
                candidate_mask = x_t <= 0
                candidate_mask_f = candidate_mask.to(torch.float32)

                # Stage 1: classify which observed-zero peaks should be opened.
                binary_logits = trainer.raw_model.predict_binary_logits(
                    g_id,
                    x_t,
                    candidate_mask_f,
                    t_binary,
                    temperature=0.0,
                )
                p_open = torch.sigmoid(binary_logits)
                if specificity_score is not None:
                    rank_score = binary_logits + float(trainer.inference_specificity_alpha) * specificity_score
                gate_bool = _build_gate(
                    candidate_mask,
                    p_open,
                    threshold=trainer.inference_binary_threshold,
                    strategy=trainer.inference_gate_strategy,
                    top_k=trainer.inference_top_k,
                    observed_nonzero_counts=(x_t > 0).sum(dim=1),
                    top_k_base_rate=trainer.inference_top_k_base_rate,
                    top_k_min=trainer.inference_top_k_min,
                    top_k_max=trainer.inference_top_k_max,
                    rank_score=rank_score,
                    specificity_score=specificity_score,
                    specificity_min=trainer.inference_specificity_min,
                )

                # Stage 2: only predicted-open zeros are reconstructed. Predicted-closed
                # zeros remain explicit zero context instead of being treated as unknown.
                eval_mask = gate_bool.to(torch.float32)
                observed = x_t.clone()
                observed[eval_mask == 1] = 0.0
                cell_idx = torch.as_tensor(selected_indices[start:end], dtype=torch.long)
                pred_intensity = _predict_intensity(
                    trainer,
                    g_id,
                    observed,
                    eval_mask,
                    selected_cell_indices=cell_idx,
                    x_ref=x_t,
                )
                pred_full = torch.zeros_like(x_t)
                pred_full[gate_bool] = pred_intensity[gate_bool]
            else:
                cell_idx = torch.as_tensor(selected_indices[start:end], dtype=torch.long)
                eval_mask = trainer._make_val_mask(x_t, cell_indices=cell_idx)
                observed = x_t.clone()
                observed[eval_mask == 1] = 0.0
                pred_intensity = _predict_intensity(
                    trainer,
                    g_id,
                    observed,
                    eval_mask,
                    selected_cell_indices=cell_idx,
                    x_ref=x_t,
                )
                binary_logits = trainer.raw_model.predict_binary_logits(
                    g_id,
                    observed,
                    eval_mask,
                    t_binary,
                    temperature=0.0,
                )
                p_open = torch.sigmoid(binary_logits)
                if specificity_score is not None:
                    rank_score = binary_logits + float(trainer.inference_specificity_alpha) * specificity_score
                gate_bool = _build_gate(
                    eval_mask == 1,
                    p_open,
                    threshold=trainer.inference_binary_threshold,
                    strategy=trainer.inference_gate_strategy,
                    top_k=trainer.inference_top_k,
                    observed_nonzero_counts=(observed > 0).sum(dim=1),
                    top_k_base_rate=trainer.inference_top_k_base_rate,
                    top_k_min=trainer.inference_top_k_min,
                    top_k_max=trainer.inference_top_k_max,
                    rank_score=rank_score,
                    specificity_score=specificity_score,
                    specificity_min=trainer.inference_specificity_min,
                )
                if trainer.inference_hard_gate:
                    pred_full = gate_bool.to(pred_intensity.dtype) * pred_intensity
                else:
                    pred_full = p_open * pred_intensity

            mask_np = eval_mask.cpu().numpy().astype(bool, copy=False)
            pred_np = pred_full.cpu().numpy().astype(np.float32, copy=False)
            p_open_np = p_open.cpu().numpy().astype(np.float32, copy=False)
            gate_np = gate_bool.cpu().numpy().astype(bool, copy=False)
            out_np = x_np.copy()
            out_np[mask_np] = pred_np[mask_np]
            imputed_mm[start:end] = out_np

            total_masked += int(mask_np.sum())
            total_nonzero_masked += int(np.count_nonzero(x_np[mask_np] > 0))
            if args.mode == "all_zero":
                candidate_zero = candidate_mask.cpu().numpy().astype(bool, copy=False)
            else:
                candidate_zero = mask_np & (x_np <= 0)
            filled_zero = candidate_zero & gate_np
            total_candidate_zero += int(candidate_zero.sum())
            total_filled_zero += int(filled_zero.sum())
            if candidate_zero.any():
                sum_p_open_candidate_zero += float(p_open_np[candidate_zero].sum())
            if filled_zero.any():
                sum_pred_filled_zero += float(pred_np[filled_zero].sum())

    summary = {
        "mode": args.mode,
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "spectral_coords": str(Path(spectral_coords).resolve()),
        "input_data": str(Path(args.data).resolve()),
        "expression_transform": expr_transform,
        "n_cells": int(n_cells),
        "n_features": int(n_features),
        "masked_positions": int(total_masked),
        "intensity_masked_positions": int(total_masked),
        "masked_nonzero_positions": int(total_nonzero_masked),
        "candidate_zero_positions": int(total_candidate_zero),
        "filled_zero_positions": int(total_filled_zero),
        "filled_zero_rate": float(total_filled_zero / max(total_candidate_zero, 1)),
        "mean_p_open_candidate_zero": float(sum_p_open_candidate_zero / max(total_candidate_zero, 1)),
        "mean_pred_filled_zero": float(sum_pred_filled_zero / max(total_filled_zero, 1)),
        "hard_gate": bool(trainer.inference_hard_gate),
        "open_threshold": float(trainer.inference_binary_threshold),
        "gate_strategy": str(trainer.inference_gate_strategy),
        "top_k": None if trainer.inference_top_k is None else int(trainer.inference_top_k),
        "top_k_base_rate": float(trainer.inference_top_k_base_rate),
        "top_k_min": int(trainer.inference_top_k_min),
        "top_k_max": int(trainer.inference_top_k_max),
        "specificity_prior": None if specificity_prior is None else str(specificity_prior["path"]),
        "specificity_alpha": float(trainer.inference_specificity_alpha),
        "specificity_min": trainer.inference_specificity_min,
        "intensity_mode": str(getattr(trainer, "inference_intensity_mode", "rollout")),
        "intensity_t": float(getattr(trainer, "inference_intensity_t", 0.0)),
        "prior_mean": float(prior_mean),
        "prior_std": float(prior_std),
        "checkpoint_epoch": int(ckpt.get("epoch", 0)),
        "checkpoint_step": int(ckpt.get("step", 0)),
        "split_info": split_info,
    }

    imputed_array = np.asarray(imputed_mm, dtype=np.float32).copy()
    adata_out = ad.AnnData(
        X=imputed_array,
        obs=output_obs,
        var=meta["var"].copy(),
    )
    adata_out.uns["scuniflow_imputation"] = summary
    adata_out.write_h5ad(output_path)

    del adata_out
    del imputed_array
    del imputed_mm
    os.remove(memmap_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
