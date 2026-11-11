import os
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr
from torch.utils.data import Dataset

def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "data.py").exists() and (parent / "model.py").exists():
            return parent
    raise FileNotFoundError(f"Could not locate scUniFlow repo root from: {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_sea_ad, split_cells_train_val, split_obs_donors_train_val
from model import scRNAModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    split_mode: str
    split_info: dict
    obs: object


def build_train_val_split(obs, val_ratio: float, seed: int) -> SplitBundle:
    n_cells = obs.shape[0]
    split_mode = "random_cells"
    if val_ratio > 0 and "donor_id" in obs.columns:
        obs_donors = obs["donor_id"].astype(str).to_numpy()
        unique_donors = np.unique(obs_donors)
        if unique_donors.size >= 2:
            train_donors, val_donors, split_info = split_obs_donors_train_val(
                obs_donors,
                val_ratio=val_ratio,
                seed=seed,
            )
            val_mask = np.isin(obs_donors, val_donors)
            train_mask = ~val_mask
            if train_mask.any() and val_mask.any():
                split_mode = "donor"
                train_idx = np.where(train_mask)[0].astype(np.int64)
                val_idx = np.where(val_mask)[0].astype(np.int64)
            else:
                train_idx, val_idx, split_info = split_cells_train_val(
                    n_cells,
                    val_ratio=val_ratio,
                    seed=seed,
                )
        else:
            train_idx, val_idx, split_info = split_cells_train_val(
                n_cells,
                val_ratio=val_ratio,
                seed=seed,
            )
    else:
        train_idx, val_idx, split_info = split_cells_train_val(
            n_cells,
            val_ratio=val_ratio,
            seed=seed,
        )
    return SplitBundle(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        split_mode=split_mode,
        split_info=split_info,
        obs=obs,
    )


def load_counts_and_split(path_or_key: str, val_ratio: float, seed: int, max_cells=None, cell_seed: int = 42):
    counts, gene_ids, gene_names, meta = load_sea_ad(
        path_or_key,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )
    split = build_train_val_split(meta["obs"], val_ratio=val_ratio, seed=seed)
    return counts, gene_ids, gene_names, meta, split


class DownsamplePairDataset(Dataset):
    def __init__(
        self,
        count_matrix,
        gene_ids,
        indices,
        downsample_fold: float = 4.0,
        seed: int = 42,
        expression_space: str = "raw_log1p",
    ):
        self.count_matrix = count_matrix
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.downsample_fold = float(downsample_fold)
        self.keep_prob = 1.0 / max(float(downsample_fold), 1.0)
        self.seed = int(seed)
        self.expression_space = str(expression_space)
        if self.expression_space not in {"raw_log1p", "normalized_log1p"}:
            raise ValueError(f"Unsupported expression_space: {self.expression_space}")
        self.sparse = sp.issparse(count_matrix)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        cell_idx = int(self.indices[idx])
        if self.sparse:
            high_raw = self.count_matrix.getrow(cell_idx).toarray().ravel().astype(np.float32)
        else:
            high_raw = np.asarray(self.count_matrix[cell_idx], dtype=np.float32).ravel()

        high_counts = np.rint(np.clip(high_raw, 0.0, None)).astype(np.int64)
        rng = np.random.RandomState(self.seed + cell_idx)
        low_counts = rng.binomial(high_counts, self.keep_prob).astype(np.float32)
        high_counts = high_counts.astype(np.float32)

        if self.expression_space == "normalized_log1p":
            low_total = float(low_counts.sum())
            high_total = float(high_counts.sum())
            x_low = np.log1p(low_counts / (low_total + 1e-6) * 1e4).astype(np.float32)
            x_high = np.log1p(high_counts / (high_total + 1e-6) * 1e4).astype(np.float32)
        else:
            x_low = np.log1p(low_counts).astype(np.float32)
            x_high = np.log1p(high_counts).astype(np.float32)
        log_total_low = np.float32(np.log10(float(low_counts.sum()) + 1.0))
        log_total_high = np.float32(np.log10(float(high_counts.sum()) + 1.0))
        log_fold = np.float32(math.log(max(self.downsample_fold, 1.0)))

        return (
            torch.from_numpy(self.gene_ids.copy()),
            torch.from_numpy(x_low),
            torch.from_numpy(x_high),
            torch.tensor(log_total_low, dtype=torch.float32),
            torch.tensor(log_total_high, dtype=torch.float32),
            torch.tensor(log_fold, dtype=torch.float32),
            torch.tensor(cell_idx, dtype=torch.long),
        )


class ReadDepthTaskHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context, x_low, log_total_low, log_fold, chunk_size: int = 4096):
        bsz, n_genes, _ = context.shape
        preds = []
        low_feat = log_total_low.view(bsz, 1, 1)
        fold_feat = log_fold.view(bsz, 1, 1)
        for start in range(0, n_genes, int(chunk_size)):
            end = min(start + int(chunk_size), n_genes)
            ctx = context[:, start:end, :]
            x_chunk = x_low[:, start:end].unsqueeze(-1)
            feat = torch.cat(
                [
                    ctx,
                    x_chunk,
                    low_feat.expand(-1, end - start, -1),
                    fold_feat.expand(-1, end - start, -1),
                ],
                dim=-1,
            )
            delta = self.net(feat).squeeze(-1)
            preds.append(torch.clamp(x_low[:, start:end] + delta, min=0.0))
        return torch.cat(preds, dim=1)


class CellEmbeddingReadDepthHead(nn.Module):
    def __init__(self, d_model: int, n_genes: int, hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + 2),
            nn.Linear(d_model + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, context, log_total_low, log_fold):
        pooled = context.mean(dim=1)
        feat = torch.cat(
            [
                pooled,
                log_total_low.view(-1, 1),
                log_fold.view(-1, 1),
            ],
            dim=1,
        )
        return self.net(feat)


class ScUniFlowReadDepthEnhancer(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int,
        grid_size: int,
        n_layers: int = 8,
        d_state_row: int = 64,
        d_state_col: int = 16,
        activation_checkpointing: bool = True,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        head_type: str = "gene_residual",
    ):
        super().__init__()
        self.head_type = str(head_type)
        self.backbone = scRNAModel(
            n_genes=n_genes,
            d_model=d_model,
            grid_size=grid_size,
            n_layers=n_layers,
            d_state_row=d_state_row,
            d_state_col=d_state_col,
            activation_checkpointing=activation_checkpointing,
        )
        if self.head_type == "gene_residual":
            self.head = ReadDepthTaskHead(d_model=d_model, hidden_dim=head_hidden_dim, dropout=head_dropout)
        elif self.head_type == "cell_embedding":
            self.head = CellEmbeddingReadDepthHead(
                d_model=d_model,
                n_genes=n_genes,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_mamba(self):
        for p in self.backbone.mamba_encoder.parameters():
            p.requires_grad = True

    def unfreeze_router(self):
        for p in self.backbone.router.parameters():
            p.requires_grad = True

    def load_backbone_checkpoint(self, checkpoint_path: str, device=None):
        ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("vector_field.binary_mlp.",)
        bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
        if bad_missing or unexpected:
            raise RuntimeError(
                "Unexpected backbone checkpoint mismatch while loading scUniFlow: "
                f"missing={bad_missing}, unexpected={unexpected}"
            )
        if missing:
            print(
                "[load] ignored missing keys from newer vector_field module not used by RDE head: "
                + ", ".join(missing),
                flush=True,
            )
        return ckpt

    def forward(self, gene_ids, x_low, log_total_low, log_fold, chunk_size: int = 4096):
        mask = torch.zeros_like(x_low)
        requires_backbone_grad = any(p.requires_grad for p in self.backbone.parameters())
        if requires_backbone_grad:
            context, _ = self.backbone.compute_context(gene_ids, x_low, mask)
        else:
            with torch.no_grad():
                context, _ = self.backbone.compute_context(gene_ids, x_low, mask)
        if self.head_type == "cell_embedding":
            pred = self.head(context, log_total_low=log_total_low, log_fold=log_fold)
        else:
            pred = self.head(
                context,
                x_low,
                log_total_low=log_total_low,
                log_fold=log_fold,
                chunk_size=chunk_size,
            )
        return pred


def postprocess_counts_matrix(raw_counts: np.ndarray, n_top_genes: int = 2000):
    matrix = np.asarray(raw_counts, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D count matrix, got shape={matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Empty count matrix, got shape={matrix.shape}")

    adata = sc.AnnData(matrix)

    # Drop genes that are identically zero across all cells before HVG selection.
    gene_sums = np.asarray(adata.X.sum(axis=0)).ravel()
    keep_nonzero = gene_sums > 0
    if keep_nonzero.any():
        adata = adata[:, keep_nonzero].copy()
    else:
        # Keep one dummy feature so downstream steps can still run and report
        # degenerate clustering metrics instead of crashing outright.
        adata = sc.AnnData(np.zeros((adata.n_obs, 1), dtype=np.float32))

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Scanpy's HVG routine can fail when all normalized dispersions are absent or
    # non-finite. Fall back to simple variance ranking in that case.
    log_x = np.asarray(adata.X, dtype=np.float32)
    gene_var = np.var(log_x, axis=0)
    keep_variable = np.isfinite(gene_var) & (gene_var > 0)
    if keep_variable.any():
        adata = adata[:, keep_variable].copy()
        gene_var = gene_var[keep_variable]
    else:
        gene_var = np.zeros((adata.n_vars,), dtype=np.float32)

    n_top = int(min(max(1, n_top_genes), adata.shape[1]))
    if adata.shape[1] > n_top:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
        except Exception:
            top_idx = np.argsort(-gene_var)[:n_top]
            adata = adata[:, np.sort(top_idx)].copy()

    sc.pp.scale(adata)
    if adata.n_obs >= 2 and adata.n_vars >= 2:
        n_comps = int(min(50, adata.n_obs - 1, adata.n_vars - 1))
        if n_comps >= 1:
            sc.tl.pca(adata, n_comps=n_comps)
        else:
            adata.obsm["X_pca"] = np.zeros((adata.n_obs, 1), dtype=np.float32)
    else:
        adata.obsm["X_pca"] = np.zeros((adata.n_obs, 1), dtype=np.float32)

    if adata.n_obs >= 2:
        n_neighbors = int(min(15, max(1, adata.n_obs - 1)))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    return adata


def cluster_k_leiden(adata, n_cluster: int, max_steps: int = 20, min_resolution: float = 0.0, max_resolution: float = 2.0):
    if adata.n_obs == 0:
        raise ValueError("Cannot cluster empty AnnData")
    if adata.n_obs == 1 or "neighbors" not in adata.uns:
        key = f"cluster_{n_cluster}"
        adata.obs[key] = np.array(["0"] * adata.n_obs, dtype=object)
        return 0.0, key

    lo = float(min_resolution)
    hi = float(max_resolution)
    final_resolution = hi
    key = f"cluster_{n_cluster}"
    try:
        for _ in range(int(max_steps)):
            mid = lo + (hi - lo) / 2.0
            sc.tl.leiden(adata, resolution=mid, random_state=42)
            n_found = adata.obs["leiden"].nunique()
            final_resolution = mid
            if n_found > n_cluster:
                hi = mid
            elif n_found < n_cluster:
                lo = mid
            else:
                break
        sc.tl.leiden(adata, resolution=final_resolution, random_state=42, key_added=key)
    except Exception:
        adata.obs[key] = np.array(["0"] * adata.n_obs, dtype=object)
        final_resolution = 0.0
    return final_resolution, key


def _safe_metric(metric_fn, X, labels):
    try:
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(metric_fn(X, labels))
    except Exception:
        return float("nan")


def evaluate_clustering_metrics(
    enhanced_counts: np.ndarray,
    target_labels,
    n_clusters: int,
    n_top_genes: int = 2000,
):
    adata = postprocess_counts_matrix(enhanced_counts, n_top_genes=n_top_genes)
    resolution, cluster_key = cluster_k_leiden(adata, int(n_clusters))
    pred_clusters = adata.obs[cluster_key].astype(str).to_numpy()
    target_labels = np.asarray(target_labels).astype(str)
    x_pca = np.asarray(adata.obsm["X_pca"])

    metrics = {
        "nmi": float(normalized_mutual_info_score(pred_clusters, target_labels)),
        "ari": float(adjusted_rand_score(pred_clusters, target_labels)),
        "silhouette_target": _safe_metric(silhouette_score, x_pca, target_labels),
        "silhouette_cluster": _safe_metric(silhouette_score, x_pca, pred_clusters),
        "calinski_harabasz": _safe_metric(calinski_harabasz_score, x_pca, pred_clusters),
        "davies_bouldin": _safe_metric(davies_bouldin_score, x_pca, pred_clusters),
        "resolution": float(resolution),
        "n_clusters_target": int(n_clusters),
        "n_clusters_pred": int(len(np.unique(pred_clusters))),
    }
    return metrics, adata, pred_clusters


def derive_reference_labels(raw_counts: np.ndarray, ref_resolution: float = 0.4, n_top_genes: int = 2000):
    adata = postprocess_counts_matrix(raw_counts, n_top_genes=n_top_genes)
    sc.tl.leiden(adata, resolution=float(ref_resolution), random_state=42)
    labels = adata.obs["leiden"].astype(str).to_numpy()
    return labels, adata


def safe_pearsonr(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(pearsonr(x, y)[0])


def safe_nanmean(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(finite.mean())


def safe_nanmedian(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.median(finite))


def _pearsonr_axis_chunked(x: np.ndarray, y: np.ndarray, axis: int, chunk_size: int = 2048):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"Expected matching shapes, got {x.shape} and {y.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got ndim={x.ndim}")

    if axis == 1:
        n_items = x.shape[0]
        out = np.full(n_items, np.nan, dtype=np.float32)
        for start in range(0, n_items, int(chunk_size)):
            end = min(start + int(chunk_size), n_items)
            xs = x[start:end]
            ys = y[start:end]
            if xs.shape[1] < 2:
                continue
            xc = xs - xs.mean(axis=1, keepdims=True)
            yc = ys - ys.mean(axis=1, keepdims=True)
            num = np.sum(xc * yc, axis=1)
            den = np.sqrt(np.sum(xc * xc, axis=1) * np.sum(yc * yc, axis=1))
            valid = den > 1e-8
            out[start:end][valid] = (num[valid] / den[valid]).astype(np.float32)
        return out

    if axis == 0:
        n_items = x.shape[1]
        out = np.full(n_items, np.nan, dtype=np.float32)
        for start in range(0, n_items, int(chunk_size)):
            end = min(start + int(chunk_size), n_items)
            xs = x[:, start:end]
            ys = y[:, start:end]
            if xs.shape[0] < 2:
                continue
            xc = xs - xs.mean(axis=0, keepdims=True)
            yc = ys - ys.mean(axis=0, keepdims=True)
            num = np.sum(xc * yc, axis=0)
            den = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc, axis=0))
            valid = den > 1e-8
            out[start:end][valid] = (num[valid] / den[valid]).astype(np.float32)
        return out

    raise ValueError(f"Unsupported axis={axis}; expected 0 or 1")


def summarize_reconstruction(preds: np.ndarray, targets: np.ndarray):
    cell_pccs = _pearsonr_axis_chunked(preds, targets, axis=1)
    gene_pccs = _pearsonr_axis_chunked(preds, targets, axis=0)
    valid_cell_pccs = int(np.isfinite(cell_pccs).sum())
    valid_gene_pccs = int(np.isfinite(gene_pccs).sum())

    metrics = {
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "mae": float(np.mean(np.abs(preds - targets))),
        "mean_cell_pcc": safe_nanmean(cell_pccs),
        "median_cell_pcc": safe_nanmedian(cell_pccs),
        "mean_gene_pcc": safe_nanmean(gene_pccs),
        "median_gene_pcc": safe_nanmedian(gene_pccs),
        "valid_cell_pccs": valid_cell_pccs,
        "valid_gene_pccs": valid_gene_pccs,
        "total_cells": int(cell_pccs.shape[0]),
        "total_genes": int(gene_pccs.shape[0]),
    }
    return metrics, cell_pccs, gene_pccs


def save_reconstruction_outputs(
    save_dir: str | Path,
    metrics: dict,
    cell_ids: list[str],
    cell_pccs: np.ndarray,
    gene_pccs: np.ndarray,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "summary.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame({"cell_id": cell_ids, "cell_pcc": cell_pccs}).to_csv(
        save_dir / "cell_metrics.csv", index=False
    )
    pd.DataFrame({"gene_idx": np.arange(len(gene_pccs)), "gene_pcc": gene_pccs}).to_csv(
        save_dir / "gene_metrics.csv", index=False
    )

    np.save(save_dir / "cell_pccs.npy", cell_pccs)
    np.save(save_dir / "gene_pccs.npy", gene_pccs)


def save_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
