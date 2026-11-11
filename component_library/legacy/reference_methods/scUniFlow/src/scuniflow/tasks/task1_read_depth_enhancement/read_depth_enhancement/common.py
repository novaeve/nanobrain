import os
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split


def _find_named_root(start: Path, name: str, required_files: tuple[str, ...] = ()) -> Path:
    for parent in [start, *start.parents]:
        if parent.name != name:
            continue
        if all((parent / rel).exists() for rel in required_files):
            return parent
    raise FileNotFoundError(f"Could not locate {name} root from: {start}")


THIS_FILE = Path(__file__).resolve()
SCUNIFLOW_ROOT = _find_named_root(THIS_FILE, "scUniFlow", required_files=("data.py", "model.py"))
WORKSPACE_ROOT = SCUNIFLOW_ROOT.parent
SCFOUNDATION_ROOT = WORKSPACE_ROOT / "scFoundation"
if not ((SCFOUNDATION_ROOT / "models.ckpt").exists() and (SCFOUNDATION_ROOT / "model").exists()):
    raise FileNotFoundError(f"Could not locate usable scFoundation root at: {SCFOUNDATION_ROOT}")
SCFOUNDATION_MODEL_ROOT = SCFOUNDATION_ROOT / "model"

sys.path.insert(0, str(SCUNIFLOW_ROOT))
sys.path.insert(0, str(SCFOUNDATION_ROOT))
sys.path.insert(0, str(SCFOUNDATION_MODEL_ROOT))

from data import load_sea_ad, split_cells_train_val, split_obs_donors_train_val  # noqa: E402
from load import load_model_frommmf, getEncoerDecoderData  # noqa: E402


DEFAULT_GENE_INDEX = SCFOUNDATION_ROOT / "OS_scRNA_gene_index.19264.tsv"
EXTRA_TOKENS = 2


@dataclass
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    split_mode: str
    split_info: dict
    obs: object


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_gene_list(gene_index_path: str | Path) -> list[str]:
    gene_df = pd.read_csv(gene_index_path, sep="\t", header=0)
    return gene_df["gene_name"].astype(str).tolist()


def _select_matrix(adata, layer: str | None = None):
    if layer:
        if layer not in adata.layers:
            raise KeyError(f"layer not found in h5ad: {layer}")
        return adata.layers[layer]
    return adata.X


def align_h5ad_to_gene_list(
    data_path: str | Path,
    gene_list: list[str],
    gene_name_col: str = "gene_name",
    layer: str | None = None,
    max_cells: int | None = None,
    cell_seed: int = 42,
):
    adata = sc.read_h5ad(data_path)
    if max_cells is not None and adata.n_obs > int(max_cells):
        # Keep the same subsampling semantics as scUniFlow/data.py.
        rng = np.random.RandomState(int(cell_seed))
        keep = np.sort(rng.choice(adata.n_obs, size=int(max_cells), replace=False))
        adata = adata[keep].copy()

    if gene_name_col in adata.var.columns:
        source_genes = adata.var[gene_name_col].astype(str).tolist()
    else:
        source_genes = adata.var_names.astype(str).tolist()

    src_name_to_idx = {}
    for idx, gene_name in enumerate(source_genes):
        if gene_name not in src_name_to_idx:
            src_name_to_idx[gene_name] = idx

    dst_cols = []
    src_cols = []
    missing = 0
    for dst_idx, gene_name in enumerate(gene_list):
        src_idx = src_name_to_idx.get(gene_name)
        if src_idx is None:
            missing += 1
            continue
        dst_cols.append(dst_idx)
        src_cols.append(src_idx)

    X_src = _select_matrix(adata, layer=layer)
    aligned = np.zeros((adata.n_obs, len(gene_list)), dtype=np.float32)
    if src_cols:
        subset = X_src[:, src_cols]
        subset = subset.toarray() if issparse(subset) else np.asarray(subset)
        aligned[:, dst_cols] = subset.astype(np.float32, copy=False)

    meta = {
        "n_cells": int(adata.n_obs),
        "n_target_genes": int(len(gene_list)),
        "n_present_genes": int(len(src_cols)),
        "n_missing_genes": int(missing),
        "obs_names": adata.obs_names.astype(str).tolist(),
        "obs": adata.obs.copy(),
        "src_cols": np.asarray(src_cols, dtype=np.int64),
        "dst_cols": np.asarray(dst_cols, dtype=np.int64),
        "source_genes": source_genes,
    }
    return aligned, meta


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


def load_full_gene_counts_and_split(
    path_or_key: str,
    val_ratio: float,
    seed: int,
    max_cells: int | None = None,
    cell_seed: int = 42,
):
    counts, gene_ids, gene_names, meta = load_sea_ad(
        path_or_key,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )
    split = build_train_val_split(meta["obs"], val_ratio=val_ratio, seed=seed)
    return counts, gene_ids, gene_names, meta, split


def materialize_downsampled_full_gene_counts(
    count_matrix,
    indices: np.ndarray,
    downsample_fold: float = 4.0,
    seed: int = 42,
):
    indices = np.asarray(indices, dtype=np.int64)
    keep_prob = 1.0 / max(float(downsample_fold), 1.0)
    sparse = issparse(count_matrix)

    high_list = []
    low_list = []
    for cell_idx in indices:
        if sparse:
            high_raw = count_matrix.getrow(int(cell_idx)).toarray().ravel().astype(np.float32)
        else:
            high_raw = np.asarray(count_matrix[int(cell_idx)], dtype=np.float32).ravel()

        high_counts = np.rint(np.clip(high_raw, 0.0, None)).astype(np.int64)
        rng = np.random.RandomState(int(seed) + int(cell_idx))
        low_counts = rng.binomial(high_counts, keep_prob).astype(np.float32)

        high_list.append(high_counts.astype(np.float32))
        low_list.append(low_counts)

    high = np.stack(high_list, axis=0)
    low = np.stack(low_list, axis=0)
    return low, high


def split_train_val_indices(n_cells: int, val_ratio: float, seed: int):
    indices = np.arange(n_cells)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(val_ratio),
        shuffle=True,
        random_state=int(seed),
    )
    return np.asarray(train_idx), np.asarray(val_idx)


class ReadDepthDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, tgt_highres: float = 4.0):
        self.X = np.asarray(X, dtype=np.float32)
        self.tgt_highres = np.float32(tgt_highres)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        total = float(x.sum())
        x_norm = np.log1p(x / (total + 1e-6) * 1e4).astype(np.float32)
        log_total = np.float32(np.log10(total + 1.0))
        x_input = np.concatenate(
            [x_norm, np.array([self.tgt_highres, log_total], dtype=np.float32)]
        ).astype(np.float32)
        return torch.from_numpy(x_input)


class ReadDepthEnhancementDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        indices: np.ndarray | None = None,
        tgt_highres: float = 4.0,
        downsample_fold: float = 4.0,
        seed: int = 42,
        return_cell_idx: bool = False,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(self.X.shape[0], dtype=np.int64)
        )
        self.tgt_highres = np.float32(tgt_highres)
        self.downsample_fold = float(downsample_fold)
        self.keep_prob = 1.0 / max(float(downsample_fold), 1.0)
        self.seed = int(seed)
        self.return_cell_idx = bool(return_cell_idx)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        cell_idx = int(self.indices[idx])
        high = np.asarray(self.X[cell_idx], dtype=np.float32)
        high_counts = np.rint(np.clip(high, 0.0, None)).astype(np.int64)

        rng = np.random.RandomState(self.seed + cell_idx)
        low_counts = rng.binomial(high_counts, self.keep_prob).astype(np.float32)
        high_counts = high_counts.astype(np.float32)

        low_total = float(low_counts.sum())
        high_total = float(high_counts.sum())
        low_norm = np.log1p(low_counts / (low_total + 1e-6) * 1e4).astype(np.float32)
        high_norm = np.log1p(high_counts / (high_total + 1e-6) * 1e4).astype(np.float32)
        log_total_low = np.float32(np.log10(low_total + 1.0))
        x_input = np.concatenate(
            [low_norm, np.array([self.tgt_highres, log_total_low], dtype=np.float32)]
        ).astype(np.float32)
        if self.return_cell_idx:
            return torch.from_numpy(x_input), torch.from_numpy(high_norm), torch.tensor(cell_idx, dtype=torch.long)
        return torch.from_numpy(x_input), torch.from_numpy(high_norm)


class ScFoundationReadDepthEnhancer(nn.Module):
    """
    scFoundation encoder-decoder fine-tuned for read depth enhancement /
    full-expression reconstruction, following the current scFoundation recon-only setup.
    """

    def __init__(self, ckpt_path: str, freeze_emb: bool = True, unfreeze_last_n: int = 2):
        super().__init__()
        model, cfg = load_model_frommmf(ckpt_path, key="gene")
        self.cfg = cfg

        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        self.decoder_embed = model.decoder_embed
        self.decoder = model.decoder
        self.recon_norm = model.norm
        self.to_final = model.to_final

        self._freeze(freeze_emb=freeze_emb, unfreeze_last_n=unfreeze_last_n)

    def _freeze(self, freeze_emb: bool, unfreeze_last_n: int):
        if freeze_emb:
            for p in self.token_emb.parameters():
                p.requires_grad = False
            for p in self.pos_emb.parameters():
                p.requires_grad = False

        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_last_n > 0:
            blocks = list(self.encoder.transformer_encoder)
            for block in blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor | None = None):
        if x_raw is None:
            x_raw = x

        (enc_data, enc_pos, enc_pad, enc_labels,
         dec_data, dec_pad, _, _, dec_pos) = getEncoerDecoderData(
            x.float(), x_raw.float(), self.cfg
        )

        tok = self.token_emb(enc_data.unsqueeze(2).float(), output_weight=0)
        tok = tok + self.pos_emb(enc_pos)
        enc_out = self.encoder(tok, enc_pad)

        dec_tok = self.token_emb(dec_data.unsqueeze(2).float())
        dec_tok = dec_tok + self.pos_emb(dec_pos)
        batch_idx, gene_idx = enc_labels.nonzero(as_tuple=True)
        dec_tok[batch_idx, gene_idx] = enc_out[~enc_pad].to(dec_tok.dtype)

        dec_tok = self.decoder_embed(dec_tok)
        dec_out = self.decoder(dec_tok, padding_mask=dec_pad)
        dec_out = self.recon_norm(dec_out)
        pred = self.to_final(dec_out).squeeze(-1)
        return pred[:, :x.shape[1] - EXTRA_TOKENS]


def get_trainable_param_groups(model: ScFoundationReadDepthEnhancer, lr: float, lr_head: float):
    backbone_modules = [model.token_emb, model.pos_emb, model.encoder]
    head_modules = [model.decoder_embed, model.decoder, model.recon_norm, model.to_final]

    backbone_params = []
    for module in backbone_modules:
        backbone_params.extend([p for p in module.parameters() if p.requires_grad])

    head_params = []
    for module in head_modules:
        head_params.extend([p for p in module.parameters() if p.requires_grad])

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(lr)})
    if head_params:
        groups.append({"params": head_params, "lr": float(lr_head)})
    return groups


def load_finetune_state_dict(model: nn.Module, checkpoint_path: str | Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    return ckpt


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


def save_eval_outputs(
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


def postprocess_expression_matrix(expr: np.ndarray, n_top_genes: int = 2000):
    # scanpy preprocessing mutates AnnData.X in place. Copy here so metric
    # computation never changes cached count matrices or saved arrays.
    adata = sc.AnnData(np.asarray(expr, dtype=np.float32).copy())
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = int(min(max(1, n_top_genes), adata.shape[1]))
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    return adata


def cluster_k_leiden(adata, n_cluster: int, max_steps: int = 20, min_resolution: float = 0.0, max_resolution: float = 2.0):
    lo = float(min_resolution)
    hi = float(max_resolution)
    final_resolution = hi
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
    sc.tl.leiden(adata, resolution=final_resolution, random_state=42, key_added=f"cluster_{n_cluster}")
    return final_resolution, f"cluster_{n_cluster}"


def _safe_metric(metric_fn, X, labels):
    try:
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(metric_fn(X, labels))
    except Exception:
        return float("nan")


def evaluate_clustering_metrics(expr: np.ndarray, target_labels, n_clusters: int, n_top_genes: int = 2000):
    adata = postprocess_expression_matrix(expr, n_top_genes=n_top_genes)
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


def derive_reference_labels(expr: np.ndarray, ref_resolution: float = 0.4, n_top_genes: int = 2000):
    adata = postprocess_expression_matrix(expr, n_top_genes=n_top_genes)
    sc.tl.leiden(adata, resolution=float(ref_resolution), random_state=42)
    labels = adata.obs["leiden"].astype(str).to_numpy()
    return labels, adata
