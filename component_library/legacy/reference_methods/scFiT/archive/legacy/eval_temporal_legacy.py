import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
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
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tqdm import tqdm

from data import _load_preprocessed_temporal_bundle, load_mouse_development_temporal_datasets, make_temporal_dataloader
from model import scFiTModel
from utils import seed_everything


try:
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


_SERIALIZED_ASSET_CACHE = {}
_REAL_PCA_CACHE = {}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _cfg_value(value, default):
    return default if value is None or value == "" else value


def _stage_pbar(enabled: bool, total: int, desc: str):
    return tqdm(total=total, desc=desc, leave=False) if enabled else None


def _ensure_obs_name_column(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if "obs_name" in metadata_df.columns:
        return metadata_df
    normalized = metadata_df.copy()
    if normalized.index.name == "obs_name":
        normalized = normalized.reset_index()
        return normalized
    index_values = normalized.index.to_numpy()
    normalized.insert(0, "obs_name", np.asarray(index_values).astype(str))
    return normalized


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


def _safe_pearson_r2_np(x: np.ndarray, y: np.ndarray):
    corr = _safe_pearson_np(x, y)
    if corr is None:
        return None
    return float(corr * corr)


def _safe_spearman_np(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return None
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    corr = spearmanr(x, y).correlation
    if corr is None or np.isnan(corr):
        return None
    return float(corr)


def _build_model(cfg, n_genes: int, device: torch.device, top_k_genes: int, n_cell_types: int = 0):
    model_cfg = cfg["model"]
    model = scFiTModel(
        n_genes=n_genes,
        top_k_genes=top_k_genes,
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        d_ff=int(model_cfg["d_ff"]),
        nlayers=int(model_cfg["nlayers"]),
        dropout=float(model_cfg["dropout"]),
        value_max=float(model_cfg.get("value_encoder", {}).get("max_value", 32.0)),
        share_adaln_projection=bool(model_cfg.get("adaln", {}).get("share_projection", False)),
        aux_bin_ce=False,
        n_bins=0,
        n_cell_types=int(model_cfg.get("n_cell_types", n_cell_types)),
        pseudotime_head=bool(model_cfg.get("pseudotime_head", {}).get("enabled", False)),
    )
    return model.to(device)


def _load_checkpoint(model, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    return ckpt


def _build_temporal_eval_loader(cfg):
    data_cfg = cfg.get("data", {})
    pairing_cfg = cfg.get("pairing", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}
    loader_cfg = data_cfg.get("loader", {}) or {}

    split = str(eval_cfg.get("split", "val"))
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported evaluation.split={split}")

    train_dataset, val_dataset, _, meta = load_mouse_development_temporal_datasets(
        path_or_key=str(data_cfg.get("source", "ignored")),
        input_mode=str(pairing_cfg.get("input_mode", "fixed_hvg")),
        preprocessed_dir=data_cfg.get("preprocessed_dir"),
        pair_table_path=pairing_cfg.get("pair_table_path"),
        uce_npz_path=data_cfg.get("uce_npz"),
        time_col=data_cfg.get("time_col", "author_day"),
        time_numeric_col=data_cfg.get("time_numeric_col"),
        cell_type_col=data_cfg.get("cell_type_col", "author_cell_type"),
        top_k=int(cfg["model"]["top_k_genes"]),
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
        max_cells=data_cfg.get("max_cells"),
        cell_seed=int(data_cfg.get("cell_seed", cfg.get("training", {}).get("seed", 42))),
        train_cache_path=pairing_cfg.get("train_cache_path"),
        val_cache_path=pairing_cfg.get("val_cache_path"),
        force_rebuild_pairs=bool(pairing_cfg.get("force_rebuild_pairs", False)),
    )
    dataset = train_dataset if split == "train" else val_dataset
    loader = make_temporal_dataloader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", cfg.get("validation", {}).get("batch_size", 16))),
        shuffle=False,
        group_by_tgt_time=bool(pairing_cfg.get("group_by_tgt_time", True)),
        drop_last=False,
        seed=int(eval_cfg.get("seed", cfg.get("training", {}).get("seed", 42))),
        num_workers=int(eval_cfg.get("num_workers", loader_cfg.get("num_workers", 0))),
        prefetch_factor=eval_cfg.get("prefetch_factor", loader_cfg.get("prefetch_factor")),
        persistent_workers=bool(eval_cfg.get("persistent_workers", loader_cfg.get("persistent_workers", False))),
    )
    return loader, meta, dataset, split


def _collect_temporal_predictions(model, loader, meta, cfg, device):
    eval_cfg = cfg.get("evaluation", {})
    n_steps = int(eval_cfg.get("n_steps", cfg.get("inference", {}).get("n_steps", 10)))
    max_batches = int(eval_cfg.get("max_batches", 0))

    model.eval()
    preds = []
    tgts = []
    srcs = []
    t_src = []
    t_tgt = []
    cell_type_ids = []
    src_idx = []
    tgt_idx = []

    iterator = tqdm(loader, desc="temporal-eval", leave=False)
    for batch_idx, batch in enumerate(iterator, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        if len(batch) != 10:
            raise ValueError("Temporal eval expects dataset with return_index=True")

        batch_gene_ids, batch_src, batch_tgt, batch_valid_mask, batch_t_src, batch_t_tgt, batch_cell_type_ids, _, batch_src_idx, batch_tgt_idx = batch
        batch_gene_ids = batch_gene_ids.to(device, non_blocking=True)
        batch_src = batch_src.to(device, non_blocking=True)
        batch_valid_mask = batch_valid_mask.to(device, non_blocking=True).bool()
        batch_t_src_tensor = batch_t_src.to(device, non_blocking=True).float().unsqueeze(1)
        batch_t_tgt_tensor = batch_t_tgt.to(device, non_blocking=True).float().unsqueeze(1)
        batch_cell_type_ids = batch_cell_type_ids.to(device, non_blocking=True).long()
        padding_mask = ~batch_valid_mask

        with torch.no_grad():
            pred_tgt = model.temporal_inference(
                batch_gene_ids,
                batch_src,
                padding_mask=padding_mask,
                t_src=batch_t_src_tensor,
                t_tgt=batch_t_tgt_tensor,
                cell_type_ids=batch_cell_type_ids,
                n_steps=n_steps,
            )

        preds.append(pred_tgt[:, 1:].detach().cpu().numpy().astype(np.float32))
        tgts.append(batch_tgt[:, 1:].numpy().astype(np.float32))
        srcs.append(batch_src[:, 1:].detach().cpu().numpy().astype(np.float32))
        t_src.append(batch_t_src.numpy().astype(np.float32))
        t_tgt.append(batch_t_tgt.numpy().astype(np.float32))
        cell_type_ids.append(batch_cell_type_ids.detach().cpu().numpy().astype(np.int64))
        src_idx.append(batch_src_idx.numpy().astype(np.int64))
        tgt_idx.append(batch_tgt_idx.numpy().astype(np.int64))

    result = {
        "pred_expr": np.concatenate(preds, axis=0),
        "tgt_expr": np.concatenate(tgts, axis=0),
        "src_expr": np.concatenate(srcs, axis=0),
        "t_src": np.concatenate(t_src, axis=0),
        "t_tgt": np.concatenate(t_tgt, axis=0),
        "cell_type_id": np.concatenate(cell_type_ids, axis=0),
        "src_index": np.concatenate(src_idx, axis=0),
        "tgt_index": np.concatenate(tgt_idx, axis=0),
    }

    pre_expr, metadata_df, pre_meta = _load_preprocessed_temporal_bundle(cfg.get("data", {}).get("preprocessed_dir"))
    result["metadata_df"] = metadata_df
    result["preprocessed_expr"] = pre_expr
    result["cell_type_names"] = np.asarray(meta.get("cell_type_names", []), dtype=object)
    result["hvg_gene_names"] = pre_meta.get("hvg_gene_names")
    result["hvg_var_names"] = pre_meta.get("hvg_var_names")
    result["time_norm_denom"] = float(pre_meta.get("time_norm_denom", 1.0))
    result["src_time_raw"] = metadata_df.iloc[result["src_index"]]["author_day"].astype(str).to_numpy()
    result["tgt_time_raw"] = metadata_df.iloc[result["tgt_index"]]["author_day"].astype(str).to_numpy()
    return result


def _resolve_target_cell_type_names(collected, cfg):
    cell_type_names = np.asarray(collected.get("cell_type_names", []), dtype=object)
    cell_type_ids = np.asarray(collected["cell_type_id"], dtype=np.int64)
    if cell_type_names.size > 0 and np.all((cell_type_ids >= 0) & (cell_type_ids < cell_type_names.shape[0])):
        return cell_type_names[cell_type_ids].astype(str)

    cell_type_col = cfg.get("data", {}).get("cell_type_col", "author_cell_type")
    metadata_df = _ensure_obs_name_column(collected["metadata_df"])
    tgt_index = np.asarray(collected["tgt_index"], dtype=np.int64)
    return metadata_df.iloc[tgt_index][cell_type_col].astype(str).to_numpy()


def _load_serialized_asset(path: str):
    path = str(path)
    cached = _SERIALIZED_ASSET_CACHE.get(path)
    if cached is not None:
        return cached

    asset_path = Path(path)
    if not asset_path.exists():
        raise FileNotFoundError(f"asset not found: {asset_path}")

    if asset_path.suffix == ".npz":
        asset = np.load(asset_path, allow_pickle=True)
    elif asset_path.suffix == ".npy":
        asset = np.load(asset_path, allow_pickle=True)
    elif asset_path.suffix in {".pkl", ".pickle", ".joblib"}:
        try:
            with open(asset_path, "rb") as f:
                asset = pickle.load(f)
        except Exception:
            import joblib

            asset = joblib.load(asset_path)
    elif asset_path.suffix in {".json"}:
        with open(asset_path, "r", encoding="utf-8") as f:
            asset = json.load(f)
    else:
        raise ValueError(f"unsupported asset format: {asset_path}")

    _SERIALIZED_ASSET_CACHE[path] = asset
    return asset


def _load_pca_transform(path: str):
    if not path:
        return None
    asset = _load_serialized_asset(path)
    if hasattr(asset, "transform"):
        return asset
    if isinstance(asset, np.lib.npyio.NpzFile):
        keys = set(asset.files)
        required = {"mean_", "components_"}
        if not required.issubset(keys):
            raise ValueError("PCA npz must contain `mean_` and `components_`.")
        return {
            "mean_": np.asarray(asset["mean_"], dtype=np.float32),
            "components_": np.asarray(asset["components_"], dtype=np.float32),
            "whiten": bool(asset["whiten"][()]) if "whiten" in keys else False,
            "explained_variance_": np.asarray(asset["explained_variance_"], dtype=np.float32) if "explained_variance_" in keys else None,
        }
    raise ValueError(f"Unsupported PCA asset at {path}")


def _transform_with_pca(x: np.ndarray, pca_asset):
    x = np.asarray(x, dtype=np.float32)
    if hasattr(pca_asset, "transform"):
        return np.asarray(pca_asset.transform(x), dtype=np.float32)

    x_centered = x - np.asarray(pca_asset["mean_"], dtype=np.float32)
    transformed = x_centered @ np.asarray(pca_asset["components_"], dtype=np.float32).T
    if bool(pca_asset.get("whiten", False)):
        exp_var = np.asarray(pca_asset.get("explained_variance_"), dtype=np.float32)
        if exp_var is None:
            raise ValueError("Whitened PCA asset requires `explained_variance_`.")
        transformed = transformed / np.sqrt(np.maximum(exp_var, 1e-8))
    return transformed.astype(np.float32, copy=False)


def _load_real_pca_embeddings(collected, cfg):
    eval_cfg = cfg.get("evaluation", {}) or {}
    assets_cfg = eval_cfg.get("assets", {}) or {}
    pca_cfg = assets_cfg.get("pca", {}) or {}
    pca_path = str(pca_cfg.get("path", "") or "").strip()
    real_embedding_path = str(pca_cfg.get("real_embedding_path", "") or "").strip()

    if not pca_path:
        return None, None, {"mode": "disabled", "reason": "No PCA transform configured."}

    cache_key = (pca_path, real_embedding_path, cfg.get("data", {}).get("preprocessed_dir"))
    cached = _REAL_PCA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pca_asset = _load_pca_transform(pca_path)
    metadata_df = collected["metadata_df"]

    if real_embedding_path:
        real_asset = _load_serialized_asset(real_embedding_path)
        if isinstance(real_asset, np.lib.npyio.NpzFile):
            keys = set(real_asset.files)
            if "embedding" in keys:
                real_pca = np.asarray(real_asset["embedding"], dtype=np.float32)
            elif "X_pca" in keys:
                real_pca = np.asarray(real_asset["X_pca"], dtype=np.float32)
            else:
                raise ValueError("real PCA npz must contain `embedding` or `X_pca`.")
            if "obs_names" in keys:
                obs_names = real_asset["obs_names"].astype(str)
                current_obs = metadata_df["obs_name"].astype(str).to_numpy()
                row_by_obs = {name: idx for idx, name in enumerate(obs_names.tolist())}
                take = [row_by_obs[name] for name in current_obs]
                real_pca = real_pca[np.asarray(take, dtype=np.int64)]
        else:
            real_pca = np.asarray(real_asset, dtype=np.float32)
    else:
        expr = collected["preprocessed_expr"]
        batch_size = int(pca_cfg.get("transform_batch_size", 4096))
        chunks = []
        for start in range(0, expr.shape[0], batch_size):
            stop = min(start + batch_size, expr.shape[0])
            chunks.append(_transform_with_pca(np.asarray(expr[start:stop], dtype=np.float32), pca_asset))
        real_pca = np.concatenate(chunks, axis=0)

    meta = {"mode": "fixed_pca", "pca_path": pca_path, "real_embedding_path": real_embedding_path or None}
    _REAL_PCA_CACHE[cache_key] = (pca_asset, real_pca.astype(np.float32, copy=False), meta)
    return _REAL_PCA_CACHE[cache_key]


def _transform_predicted_to_pca(collected, context):
    pca_asset = context.get("pca_asset")
    if pca_asset is None:
        return None
    return _transform_with_pca(collected["pred_expr"], pca_asset)


def _load_marker_index_by_cell_type(collected, cfg):
    eval_cfg = cfg.get("evaluation", {}) or {}
    assets_cfg = eval_cfg.get("assets", {}) or {}
    marker_path = str(assets_cfg.get("marker_gene_path", "") or "").strip()
    if not marker_path:
        return None, {"mode": "disabled", "reason": "No marker_gene_path configured."}

    marker_asset = _load_serialized_asset(marker_path)
    if not isinstance(marker_asset, dict):
        raise ValueError("marker gene asset must be a JSON object mapping cell type to marker genes.")
    if "markers" in marker_asset and isinstance(marker_asset["markers"], dict):
        marker_asset = marker_asset["markers"]

    hvg_gene_names = collected.get("hvg_gene_names")
    if hvg_gene_names is None:
        hvg_gene_names = collected.get("hvg_var_names")
    if hvg_gene_names is None:
        raise ValueError("Cannot resolve marker gene indices: missing HVG gene names.")

    hvg_gene_names = np.asarray(hvg_gene_names).astype(str)
    gene_to_idx = {name: idx for idx, name in enumerate(hvg_gene_names.tolist())}
    gene_to_idx_lower = {name.lower(): idx for idx, name in enumerate(hvg_gene_names.tolist())}

    marker_idx = {}
    for cell_type, genes in marker_asset.items():
        indices = []
        for gene in genes:
            gene = str(gene)
            idx = gene_to_idx.get(gene)
            if idx is None:
                idx = gene_to_idx_lower.get(gene.lower())
            if idx is not None:
                indices.append(int(idx))
        marker_idx[str(cell_type)] = np.asarray(sorted(set(indices)), dtype=np.int64)
    return marker_idx, {"mode": "fixed_markers", "path": marker_path}


def _load_pseudotime_values(path: str):
    pseudo_path = Path(path)
    if not pseudo_path.exists():
        raise FileNotFoundError(f"pseudotime file not found: {pseudo_path}")
    if pseudo_path.suffix in {".csv", ".gz"}:
        df = pd.read_csv(pseudo_path)
    elif pseudo_path.suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(pseudo_path)
    elif pseudo_path.suffix == ".npz":
        data = np.load(pseudo_path, allow_pickle=True)
        if "obs_names" not in data.files or "pseudotime" not in data.files:
            raise ValueError("pseudotime npz must contain `obs_names` and `pseudotime`.")
        df = pd.DataFrame({"obs_name": data["obs_names"].astype(str), "pseudotime": data["pseudotime"].astype(np.float32)})
    else:
        raise ValueError(f"unsupported pseudotime format: {pseudo_path}")
    if "obs_name" not in df.columns:
        raise ValueError("pseudotime file must contain `obs_name` column.")
    if "pseudotime" not in df.columns:
        raise ValueError("pseudotime file must contain `pseudotime` column.")
    return df[["obs_name", "pseudotime"]].copy()


def _prepare_eval_context(collected, cfg, output_dir=None):
    del output_dir
    pca_asset, real_pca, pca_meta = _load_real_pca_embeddings(collected, cfg)
    marker_idx_by_cell_type, marker_meta = _load_marker_index_by_cell_type(collected, cfg)

    eval_cfg = cfg.get("evaluation", {}) or {}
    assets_cfg = eval_cfg.get("assets", {}) or {}
    pseudo_path = str(assets_cfg.get("pseudotime_path", "") or "").strip()
    pseudotime_df = _load_pseudotime_values(pseudo_path) if pseudo_path else None

    classifier_path = str(assets_cfg.get("cell_type_knn_path", "") or "").strip()
    classifier = _load_serialized_asset(classifier_path) if classifier_path else None

    return {
        "pca_asset": pca_asset,
        "real_pca": real_pca,
        "pca_meta": pca_meta,
        "marker_idx_by_cell_type": marker_idx_by_cell_type,
        "marker_meta": marker_meta,
        "pseudotime_df": pseudotime_df,
        "classifier": classifier,
        "classifier_path": classifier_path or None,
    }


def _compute_per_cell_r2_metrics(collected, cfg):
    pred_expr = collected["pred_expr"]
    metadata_df = collected["metadata_df"]
    real_expr = collected["preprocessed_expr"]
    tgt_time_raw = collected["tgt_time_raw"]
    cell_type_col = cfg.get("data", {}).get("cell_type_col", "author_cell_type")
    target_cell_type = _resolve_target_cell_type_names(collected, cfg)

    metrics_by_time = {}
    all_scores = []
    for time_name in np.unique(tgt_time_raw):
        time_mask = tgt_time_raw == time_name
        time_scores = []
        for cell_type_name in np.unique(target_cell_type[time_mask]):
            pred_mask = time_mask & (target_cell_type == cell_type_name)
            pred_group = pred_expr[pred_mask]
            real_mask = (
                metadata_df["author_day"].astype(str).to_numpy() == str(time_name)
            ) & (metadata_df[cell_type_col].astype(str).to_numpy() == str(cell_type_name))
            real_idx = np.flatnonzero(real_mask)
            if pred_group.shape[0] == 0 or real_idx.shape[0] == 0:
                continue
            real_group = np.asarray(real_expr[real_idx], dtype=np.float32)
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(real_group)
            nearest_local = nn.kneighbors(pred_group, return_distance=False).reshape(-1)
            matched = real_group[nearest_local]
            for pred_vec, tgt_vec in zip(pred_group, matched):
                score = _safe_pearson_r2_np(pred_vec.astype(np.float64), tgt_vec.astype(np.float64))
                if score is not None:
                    time_scores.append(score)
                    all_scores.append(score)
        metrics_by_time[str(time_name)] = None if not time_scores else float(np.mean(time_scores))
    return {
        "per_cell_pearson_r2_mean": None if not all_scores else float(np.mean(all_scores)),
        "per_cell_pearson_r2_by_t_tgt": metrics_by_time,
    }


def _compute_distribution_mean_r2_metrics(collected):
    pred_expr = collected["pred_expr"]
    metadata_df = collected["metadata_df"]
    real_expr = collected["preprocessed_expr"]
    tgt_time_raw = collected["tgt_time_raw"]

    metrics_by_time = {}
    scores = []
    for time_name in np.unique(tgt_time_raw):
        pred_mask = tgt_time_raw == time_name
        real_mask = metadata_df["author_day"].astype(str).to_numpy() == str(time_name)
        if pred_mask.sum() == 0 or real_mask.sum() == 0:
            metrics_by_time[str(time_name)] = None
            continue
        pred_mean = pred_expr[pred_mask].mean(axis=0)
        real_mean = np.asarray(real_expr[real_mask], dtype=np.float32).mean(axis=0)
        score = _safe_pearson_r2_np(pred_mean.astype(np.float64), real_mean.astype(np.float64))
        metrics_by_time[str(time_name)] = score
        if score is not None:
            scores.append(score)
    return {
        "distribution_mean_r2_mean": None if not scores else float(np.mean(scores)),
        "distribution_mean_r2_by_t_tgt": metrics_by_time,
    }


def _compute_sigma_from_median(x: np.ndarray, y: np.ndarray, max_samples: int, rng: np.random.RandomState):
    if x.shape[0] > max_samples:
        x = x[rng.choice(x.shape[0], size=max_samples, replace=False)]
    if y.shape[0] > max_samples:
        y = y[rng.choice(y.shape[0], size=max_samples, replace=False)]
    merged = np.concatenate([x, y], axis=0)
    if merged.shape[0] < 2:
        return None
    dist_sq = pairwise_distances(merged, metric="sqeuclidean")
    upper = dist_sq[np.triu_indices(dist_sq.shape[0], k=1)]
    upper = upper[upper > 0]
    if upper.size == 0:
        return None
    sigma_sq = float(np.median(upper))
    if sigma_sq <= 0:
        return None
    return float(np.sqrt(sigma_sq))


def _rbf_mmd(x: np.ndarray, y: np.ndarray, sigma: float):
    if x.shape[0] < 2 or y.shape[0] < 2 or sigma is None or sigma <= 0:
        return None
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = np.exp(-pairwise_distances(x, x, metric="sqeuclidean") * gamma)
    k_yy = np.exp(-pairwise_distances(y, y, metric="sqeuclidean") * gamma)
    k_xy = np.exp(-pairwise_distances(x, y, metric="sqeuclidean") * gamma)
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def _compute_marker_mmd_metrics(collected, cfg, context):
    marker_idx_by_cell_type = context.get("marker_idx_by_cell_type")
    if marker_idx_by_cell_type is None:
        return {"marker_mmd_mean": None, "marker_mmd_by_t_tgt": None}

    eval_cfg = cfg.get("evaluation", {}) or {}
    metric_cfg = (eval_cfg.get("metrics", {}) or {}).get("marker_mmd", {}) or {}
    min_real_cells = int(metric_cfg.get("min_real_cells", 32))
    max_cells_per_set = int(metric_cfg.get("max_cells_per_set", 256))
    max_cells_for_sigma = int(metric_cfg.get("max_cells_for_sigma", 256))
    rng = np.random.RandomState(int(eval_cfg.get("seed", cfg.get("training", {}).get("seed", 42))))

    pred_expr = collected["pred_expr"]
    metadata_df = collected["metadata_df"]
    real_expr = collected["preprocessed_expr"]
    tgt_time_raw = collected["tgt_time_raw"]
    cell_type_col = cfg.get("data", {}).get("cell_type_col", "author_cell_type")
    target_cell_type = _resolve_target_cell_type_names(collected, cfg)

    metrics_by_time = {}
    all_scores = []
    for time_name in np.unique(tgt_time_raw):
        time_mask = tgt_time_raw == time_name
        time_scores = []
        for cell_type_name in np.unique(target_cell_type[time_mask]):
            marker_idx = marker_idx_by_cell_type.get(str(cell_type_name))
            if marker_idx is None or marker_idx.size == 0:
                continue

            pred_mask = time_mask & (target_cell_type == cell_type_name)
            real_mask = (
                metadata_df["author_day"].astype(str).to_numpy() == str(time_name)
            ) & (metadata_df[cell_type_col].astype(str).to_numpy() == str(cell_type_name))

            pred_group = pred_expr[pred_mask][:, marker_idx]
            real_group = np.asarray(real_expr[real_mask], dtype=np.float32)[:, marker_idx]
            if pred_group.shape[0] < 2 or real_group.shape[0] < min_real_cells:
                continue

            if pred_group.shape[0] > max_cells_per_set:
                pred_group = pred_group[rng.choice(pred_group.shape[0], size=max_cells_per_set, replace=False)]
            if real_group.shape[0] > max_cells_per_set:
                real_group = real_group[rng.choice(real_group.shape[0], size=max_cells_per_set, replace=False)]

            sigma = _compute_sigma_from_median(
                pred_group,
                real_group,
                max_samples=max_cells_for_sigma,
                rng=rng,
            )
            score = _rbf_mmd(pred_group.astype(np.float64), real_group.astype(np.float64), sigma=sigma)
            if score is not None:
                time_scores.append(score)
                all_scores.append(score)
        metrics_by_time[str(time_name)] = None if not time_scores else float(np.mean(time_scores))
    return {
        "marker_mmd_mean": None if not all_scores else float(np.mean(all_scores)),
        "marker_mmd_by_t_tgt": metrics_by_time,
    }


def _compute_cell_type_accuracy_metrics(collected, cfg, context):
    real_pca = context.get("real_pca")
    pca_asset = context.get("pca_asset")
    if real_pca is None or pca_asset is None:
        return {
            "cell_type_accuracy": None,
            "cell_type_accuracy_per_class": None,
            "classifier_baseline_accuracy": None,
            "cell_type_classifier_source": None,
        }

    pred_pca = _transform_predicted_to_pca(collected, context)
    eval_cfg = cfg.get("evaluation", {}) or {}
    metric_cfg = (eval_cfg.get("metrics", {}) or {}).get("cell_type_accuracy", {}) or {}
    knn_k = int(metric_cfg.get("knn_k", 10))
    cell_type_col = cfg.get("data", {}).get("cell_type_col", "author_cell_type")

    metadata_df = collected["metadata_df"]
    y_real = metadata_df[cell_type_col].astype(str).to_numpy()
    effective_k = min(knn_k, real_pca.shape[0])
    clf = context.get("classifier")
    clf_source = context.get("classifier_path")
    if clf is None:
        clf = KNeighborsClassifier(n_neighbors=effective_k, metric="euclidean")
        clf.fit(real_pca, y_real)
        clf_source = "fitted_from_fixed_pca"

    y_pred = clf.predict(pred_pca)
    target_names = _resolve_target_cell_type_names(collected, cfg)
    acc = float(np.mean(y_pred == target_names))

    per_class = {}
    for class_name in np.unique(target_names):
        mask = target_names == class_name
        per_class[str(class_name)] = float(np.mean(y_pred[mask] == target_names[mask])) if mask.any() else None

    baseline_preds = np.asarray(clf.predict(real_pca), dtype=object)
    baseline_acc = float(np.mean(baseline_preds == y_real))
    return {
        "cell_type_accuracy": acc,
        "cell_type_accuracy_per_class": per_class,
        "classifier_baseline_accuracy": baseline_acc,
        "cell_type_classifier_source": clf_source,
        "cell_type_knn_k": effective_k,
    }


def _compute_pseudotime_metrics(collected, cfg, context):
    real_pca = context.get("real_pca")
    pca_asset = context.get("pca_asset")
    pseudo_df = context.get("pseudotime_df")
    if real_pca is None or pca_asset is None or pseudo_df is None:
        return {
            "pseudotime_spearman": None,
            "pseudotime_spearman_per_class": None,
            "real_pseudotime_vs_time_spearman": None,
        }

    eval_cfg = cfg.get("evaluation", {}) or {}
    metric_cfg = (eval_cfg.get("metrics", {}) or {}).get("pseudotime", {}) or {}
    k = int(metric_cfg.get("knn_k", 10))

    pred_pca = _transform_predicted_to_pca(collected, context)
    metadata_df = _ensure_obs_name_column(collected["metadata_df"]).copy()
    metadata_df["aligned_row"] = np.arange(metadata_df.shape[0], dtype=np.int64)

    real_df = metadata_df.merge(pseudo_df, how="inner", on="obs_name")
    if real_df.empty:
        raise ValueError("No overlap between pseudotime file and real cells.")

    real_indices = real_df["aligned_row"].to_numpy(dtype=np.int64)
    real_pca_valid = real_pca[real_indices]
    real_pseudotime = real_df["pseudotime"].to_numpy(dtype=np.float32)
    real_time = real_df["author_day_numeric"].to_numpy(dtype=np.float32)
    real_spearman = _safe_spearman_np(real_time.astype(np.float64), real_pseudotime.astype(np.float64))

    nn = NearestNeighbors(n_neighbors=min(k, real_pca_valid.shape[0]), metric="euclidean")
    nn.fit(real_pca_valid)
    distances, indices = nn.kneighbors(pred_pca)
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    pred_pseudotime = np.sum(real_pseudotime[indices] * weights, axis=1)

    pred_time = collected["t_tgt"] * float(collected["time_norm_denom"])
    overall = _safe_spearman_np(pred_time.astype(np.float64), pred_pseudotime.astype(np.float64))
    target_names = _resolve_target_cell_type_names(collected, cfg)
    per_class = {}
    for class_name in np.unique(target_names):
        mask = target_names == class_name
        per_class[str(class_name)] = _safe_spearman_np(
            pred_time[mask].astype(np.float64),
            pred_pseudotime[mask].astype(np.float64),
        )
    return {
        "pseudotime_spearman": overall,
        "pseudotime_spearman_per_class": per_class,
        "real_pseudotime_vs_time_spearman": real_spearman,
        "pseudotime_knn_k": k,
    }


def _build_eval_regime_masks(collected, cfg):
    pairing_cfg = cfg.get("pairing", {}) or {}
    train_cfg = pairing_cfg.get("train", {}) or {}
    train_tgt_min = train_cfg.get("tgt_min_time")
    train_tgt_max = train_cfg.get("tgt_max_time")
    if train_tgt_min is None or train_tgt_max is None:
        return {}

    t_abs = collected["t_tgt"] * float(collected["time_norm_denom"])
    iid_mask = (t_abs >= float(train_tgt_min)) & (t_abs <= float(train_tgt_max))
    ood_mask = ~iid_mask
    regime_masks = {}
    if iid_mask.any():
        regime_masks["iid"] = iid_mask
    if ood_mask.any():
        regime_masks["ood"] = ood_mask
    return regime_masks


def _subset_collected(collected, mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    subset = {}
    for key, value in collected.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == mask.shape:
            subset[key] = value[mask]
        else:
            subset[key] = value
    return subset


def _build_observed_baseline(collected, cfg):
    eval_cfg = cfg.get("evaluation", {}) or {}
    baseline_cfg = eval_cfg.get("baselines", {}) or {}
    rng = np.random.RandomState(int(baseline_cfg.get("observed_seed", eval_cfg.get("seed", cfg.get("training", {}).get("seed", 42)))))

    metadata_df = collected["metadata_df"]
    real_expr = collected["preprocessed_expr"]
    cell_type_col = cfg.get("data", {}).get("cell_type_col", "author_cell_type")
    target_names = _resolve_target_cell_type_names(collected, cfg)

    observed_pred = np.zeros_like(collected["pred_expr"], dtype=np.float32)
    for i, (time_name, cell_type_name) in enumerate(zip(collected["tgt_time_raw"], target_names)):
        real_mask = (
            metadata_df["author_day"].astype(str).to_numpy() == str(time_name)
        ) & (metadata_df[cell_type_col].astype(str).to_numpy() == str(cell_type_name))
        real_idx = np.flatnonzero(real_mask)
        if real_idx.size == 0:
            observed_pred[i] = collected["tgt_expr"][i]
            continue
        chosen = int(rng.choice(real_idx))
        observed_pred[i] = np.asarray(real_expr[chosen], dtype=np.float32)

    baseline = dict(collected)
    baseline["pred_expr"] = observed_pred
    return baseline


def _compute_temporal_metrics_core(collected, cfg, context):
    return {
        **_compute_per_cell_r2_metrics(collected, cfg),
        **_compute_distribution_mean_r2_metrics(collected),
        **_compute_marker_mmd_metrics(collected, cfg, context),
        **_compute_cell_type_accuracy_metrics(collected, cfg, context),
        **_compute_pseudotime_metrics(collected, cfg, context),
    }


def compute_temporal_metrics(collected, cfg, output_dir=None, include_regimes=True, include_baselines=True, verbose=False):
    t0 = time.time()
    stage_total = 2 + int(include_regimes) + int(include_baselines and bool((cfg.get("evaluation", {}) or {}).get("baselines", {}).get("enabled", False)))
    pbar = _stage_pbar(verbose, total=stage_total, desc="eval-metrics")
    if verbose:
        print("[eval] preparing evaluation assets...")
    context = _prepare_eval_context(collected, cfg, output_dir=output_dir)
    context_ready = time.time()
    if pbar is not None:
        pbar.update(1)
    if verbose:
        print(f"[eval] assets ready in {context_ready - t0:.1f}s; computing core metrics...")
    metrics = _compute_temporal_metrics_core(collected, cfg, context)
    core_done = time.time()
    if pbar is not None:
        pbar.update(1)
    metrics["asset_status"] = {
        "pca": context.get("pca_meta"),
        "markers": context.get("marker_meta"),
        "classifier_path": context.get("classifier_path"),
        "has_pseudotime": context.get("pseudotime_df") is not None,
    }

    if include_regimes:
        if verbose:
            print("[eval] computing IID/OOD splits...")
        regime_metrics = {}
        for regime_name, mask in _build_eval_regime_masks(collected, cfg).items():
            if mask.any():
                regime_metrics[regime_name] = _compute_temporal_metrics_core(_subset_collected(collected, mask), cfg, context)
        if regime_metrics:
            metrics["by_regime"] = regime_metrics
        if pbar is not None:
            pbar.update(1)

    eval_cfg = cfg.get("evaluation", {}) or {}
    baseline_cfg = eval_cfg.get("baselines", {}) or {}
    if include_baselines and bool(baseline_cfg.get("enabled", False)):
        if verbose:
            print("[eval] computing baselines...")
        identity = dict(collected)
        identity["pred_expr"] = collected["src_expr"]
        observed = _build_observed_baseline(collected, cfg)
        metrics["baselines"] = {
            "identity": _compute_temporal_metrics_core(identity, cfg, context),
            "observed": _compute_temporal_metrics_core(observed, cfg, context),
        }
        if pbar is not None:
            pbar.update(1)
    elif pbar is not None and stage_total > pbar.n:
        pbar.update(stage_total - pbar.n)
    if verbose:
        print(f"[eval] metric computation finished in {time.time() - t0:.1f}s")
    if pbar is not None:
        pbar.close()
    timing = {
        "asset_prepare_seconds": float(context_ready - t0),
        "core_metric_seconds": float(core_done - context_ready),
        "total_metric_seconds": float(time.time() - t0),
    }
    return metrics, {"mode": "fixed_hvg_eval_assets", "timing": timing}


def _build_result_path(cfg, checkpoint_path: str):
    eval_cfg = cfg.get("evaluation", {})
    explicit = str(eval_cfg.get("save_json", "") or "").strip()
    if explicit:
        return os.path.abspath(explicit)
    result_dir = os.path.abspath(str(eval_cfg.get("result_dir", "eval_results")))
    os.makedirs(result_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    split = str(eval_cfg.get("split", "val"))
    filename = f"eval_temporal_{ckpt_stem}_{split}.json"
    return os.path.join(result_dir, filename)


def run_temporal_eval(cfg, config_path: str):
    eval_cfg = cfg.get("evaluation", {})
    checkpoint = str(eval_cfg.get("checkpoint", "") or "").strip()
    if not checkpoint:
        raise ValueError("Please provide evaluation.checkpoint for temporal evaluation.")

    seed_everything(int(eval_cfg.get("seed", cfg.get("training", {}).get("seed", 42))))
    device = _resolve_device(str(eval_cfg.get("device", "cuda")))
    loader, data_meta, dataset, split = _build_temporal_eval_loader(cfg)
    model = _build_model(
        cfg,
        n_genes=int(data_meta["n_genes"]),
        device=device,
        top_k_genes=int(data_meta.get("fixed_panel_size", cfg["model"]["top_k_genes"])),
        n_cell_types=len(data_meta.get("cell_type_names", [])),
    )
    ckpt = _load_checkpoint(model, checkpoint, device)

    t0 = time.time()
    print("[eval] collecting predictions...")
    collected = _collect_temporal_predictions(model, loader, data_meta, cfg, device)
    collect_elapsed = time.time() - t0
    print(f"[eval] collected {collected['pred_expr'].shape[0]} predictions in {collect_elapsed:.1f}s")
    output_dir = Path(os.path.dirname(_build_result_path(cfg, checkpoint)))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, asset_meta = compute_temporal_metrics(collected, cfg, output_dir=output_dir, verbose=True)
    elapsed = time.time() - t0
    metric_timing = asset_meta.get("timing", {}) if isinstance(asset_meta, dict) else {}

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": os.path.abspath(config_path),
        "checkpoint": os.path.abspath(checkpoint),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "collection_seconds": float(collect_elapsed),
        "metric_seconds": float(metric_timing.get("total_metric_seconds", max(elapsed - collect_elapsed, 0.0))),
        "asset_prepare_seconds": float(metric_timing.get("asset_prepare_seconds", 0.0)),
        "core_metric_seconds": float(metric_timing.get("core_metric_seconds", 0.0)),
        "data": {
            "source": cfg.get("data", {}).get("source"),
            "preprocessed_dir": cfg.get("data", {}).get("preprocessed_dir"),
            "pair_table_path": cfg.get("pairing", {}).get("pair_table_path"),
            "split": split,
            "eval_pairs": int(collected["pred_expr"].shape[0]),
            "fixed_panel_size": int(collected["pred_expr"].shape[1]),
        },
        "asset_source": asset_meta,
        "metrics": metrics,
    }
    if isinstance(ckpt, dict):
        result["checkpoint_meta"] = {
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step", ckpt.get("step")),
        }

    result_path = _build_result_path(cfg, checkpoint)
    result["result_json"] = result_path
    text = json.dumps(result, indent=2, ensure_ascii=False)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    return result


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return run_temporal_eval(cfg, config_path=config_path)


def cli():
    parser = argparse.ArgumentParser(description="Temporal evaluation for scFiT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
