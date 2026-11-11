import json
import math
import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from torch.utils.data import Dataset


DEFAULT_REF_H5AD = Path("data/all_cells_20200625.h5ad")
DEFAULT_VISIUM_ROOT = Path(
    "data/mouse_brain_visium/mouse_brain_visium_wo_cloupe_data/rawdata"
)


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "data.py").exists() and (parent / "model.py").exists():
            return parent
    raise FileNotFoundError(f"Could not locate scUniFlow repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_scuniflow_model_class():
    gene_mamba_path = ROOT / 'gene_mamba.py'
    gene_mamba_spec = importlib.util.spec_from_file_location('gene_mamba', gene_mamba_path)
    if gene_mamba_spec is None or gene_mamba_spec.loader is None:
        raise ImportError(f'Failed to load scUniFlow gene_mamba module from {gene_mamba_path}')
    gene_mamba_module = importlib.util.module_from_spec(gene_mamba_spec)
    sys.modules['gene_mamba'] = gene_mamba_module
    gene_mamba_spec.loader.exec_module(gene_mamba_module)

    model_path = ROOT / 'model.py'
    spec = importlib.util.spec_from_file_location('scuniflow_task3_model_loader', model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load scUniFlow model module from {model_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scRNAModel


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_gene_ids(ensembl_ids: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(ensembl_ids))).astype(np.int64)


def _hilbert_distances(rows: np.ndarray, cols: np.ndarray, grid_size: int) -> np.ndarray:
    x = rows.astype(np.int64, copy=True)
    y = cols.astype(np.int64, copy=True)
    order = int(math.ceil(math.log2(max(grid_size, 2))))
    side = 1 << order
    d = np.zeros_like(x, dtype=np.int64)
    s = side // 2

    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += (s * s) * ((3 * rx) ^ ry)

        swap_mask = ry == 0
        flip_mask = swap_mask & (rx == 1)
        if np.any(flip_mask):
            x[flip_mask] = (side - 1) - x[flip_mask]
            y[flip_mask] = (side - 1) - y[flip_mask]
        if np.any(swap_mask):
            tmp = x[swap_mask].copy()
            x[swap_mask] = y[swap_mask]
            y[swap_mask] = tmp
        s //= 2

    return d


def _ordered_slot_indices(grid_size: int, order_mode: str = "hilbert"):
    n_slots = grid_size * grid_size
    slot_lin = np.arange(n_slots, dtype=np.int64)
    slot_rows = slot_lin // grid_size
    slot_cols = slot_lin % grid_size

    if order_mode == "hilbert":
        slot_key = _hilbert_distances(slot_rows, slot_cols, grid_size)
        slot_order = np.lexsort((slot_lin, slot_cols, slot_rows, slot_key))
    elif order_mode == "row_major":
        slot_order = np.lexsort((slot_lin, slot_cols, slot_rows))
    else:
        raise ValueError(f"Unsupported order_mode: {order_mode}")
    return slot_order, slot_rows, slot_cols


def build_ordered_gene_coords(n_genes: int, grid_size: int, order_mode: str = "hilbert"):
    n_slots = grid_size * grid_size
    if n_genes > n_slots:
        raise ValueError(
            f"Grid {grid_size}x{grid_size} has only {n_slots} slots for {n_genes} genes"
        )

    slot_order, slot_rows, slot_cols = _ordered_slot_indices(
        grid_size=grid_size,
        order_mode=order_mode,
    )
    chosen = slot_order[:n_genes]
    return np.stack([slot_rows[chosen], slot_cols[chosen]], axis=1).astype(np.float32)


@dataclass
class ReferenceData:
    gene_ids: np.ndarray
    ensembl_ids: np.ndarray
    gene_symbols: np.ndarray
    source_path: str
    n_cells: int


def load_reference_metadata(ref_h5ad: Path) -> ReferenceData:
    adata = sc.read_h5ad(ref_h5ad, backed="r")
    ensembl_ids = np.asarray(adata.var_names, dtype=object)
    if "SYMBOL" in adata.var.columns:
        gene_symbols = np.asarray(adata.var["SYMBOL"], dtype=object)
    elif "feature_name" in adata.var.columns:
        gene_symbols = np.asarray(adata.var["feature_name"], dtype=object)
    else:
        gene_symbols = ensembl_ids.copy()
    return ReferenceData(
        gene_ids=make_gene_ids(ensembl_ids),
        ensembl_ids=ensembl_ids,
        gene_symbols=gene_symbols,
        source_path=str(ref_h5ad),
        n_cells=int(adata.n_obs),
    )


def _load_single_visium(sample_dir: Path, in_tissue_only: bool) -> ad.AnnData:
    adata = sc.read_visium(sample_dir)
    adata.var_names_make_unique()
    adata.obs_names = [f"{sample_dir.name}:{idx}" for idx in adata.obs_names.astype(str)]
    adata.obs["sample_id"] = sample_dir.name
    if in_tissue_only and "in_tissue" in adata.obs.columns:
        keep = np.asarray(adata.obs["in_tissue"]).astype(int) == 1
        adata = adata[keep].copy()
    return adata


def load_visium_mouse_brain(
    visium_root: Path,
    ref: ReferenceData,
    sample_ids: Optional[Iterable[str]] = None,
    in_tissue_only: bool = True,
    max_spots: int = 0,
    sample_seed: int = 42,
) -> ad.AnnData:
    sample_dirs = sorted(
        p for p in visium_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if sample_ids:
        wanted = {str(x) for x in sample_ids}
        sample_dirs = [p for p in sample_dirs if p.name in wanted]
    if not sample_dirs:
        raise FileNotFoundError(f"No Visium sample directories found under {visium_root}")

    adatas = [_load_single_visium(p, in_tissue_only=in_tissue_only) for p in sample_dirs]
    spatial = ad.concat(adatas, join="outer", merge="same")
    spatial.var_names_make_unique()

    if "gene_ids" not in spatial.var.columns:
        raise ValueError("Visium var['gene_ids'] is required for Ensembl alignment")

    spatial_ids = np.asarray(spatial.var["gene_ids"], dtype=object)
    if set(spatial_ids.tolist()) != set(ref.ensembl_ids.tolist()):
        raise ValueError("Reference and Visium gene universes do not match")

    order = pd.Index(spatial_ids).get_indexer(ref.ensembl_ids)
    if np.any(order < 0):
        raise ValueError("Failed to align some reference genes to Visium genes")
    spatial = spatial[:, order].copy()
    spatial.var.index = ref.ensembl_ids.copy()
    spatial.var["ensembl_id"] = ref.ensembl_ids.copy()
    spatial.var["SYMBOL_ref"] = ref.gene_symbols.copy()

    if max_spots and spatial.n_obs > int(max_spots):
        rng = np.random.RandomState(sample_seed)
        keep = np.sort(rng.choice(spatial.n_obs, size=int(max_spots), replace=False))
        spatial = spatial[keep].copy()

    return spatial


def build_global_target_mask(
    spatial: ad.AnnData,
    ref: ReferenceData,
    target_gene_count: int,
    target_holdout_ratio: float,
    min_nonzero_spots: int,
    seed: int,
    target_genes: Optional[Iterable[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonzero_counts = np.asarray((spatial.X > 0).sum(axis=0)).ravel()
    candidate = nonzero_counts >= int(min_nonzero_spots)

    if target_genes:
        resolved = resolve_gene_tokens(target_genes, ref)
        target_mask = np.isin(ref.ensembl_ids, resolved)
    else:
        candidate_idx = np.flatnonzero(candidate)
        if candidate_idx.size == 0:
            raise ValueError("No candidate genes remain after min_nonzero_spots filtering")
        if int(target_gene_count) > 0:
            n_target = min(int(target_gene_count), int(candidate_idx.size))
        else:
            ratio = float(target_holdout_ratio)
            if not (0.0 < ratio < 1.0):
                raise ValueError("target_holdout_ratio must be in (0, 1) when target_gene_count<=0")
            n_target = max(1, int(round(candidate_idx.size * ratio)))
        rng = np.random.RandomState(seed)
        chosen = np.sort(rng.choice(candidate_idx, size=n_target, replace=False))
        target_mask = np.zeros(ref.ensembl_ids.shape[0], dtype=bool)
        target_mask[chosen] = True

    observed_mask = ~target_mask
    target_idx = np.flatnonzero(target_mask)
    if target_idx.size == 0:
        raise ValueError("Target gene set is empty")
    return target_mask.astype(np.float32), observed_mask, target_idx


def resolve_gene_tokens(tokens: Iterable[str], ref: ReferenceData) -> np.ndarray:
    clean = [str(x).strip() for x in tokens if str(x).strip()]
    if not clean:
        return np.array([], dtype=object)

    ensembl_set = {str(x) for x in ref.ensembl_ids.tolist()}
    symbol_to_ids = {}
    for gid, sym in zip(ref.ensembl_ids.tolist(), ref.gene_symbols.tolist()):
        symbol_to_ids.setdefault(str(sym), []).append(str(gid))

    resolved = []
    missing = []
    ambiguous = []
    for token in clean:
        if token in ensembl_set:
            resolved.append(token)
            continue
        ids = symbol_to_ids.get(token, [])
        if len(ids) == 1:
            resolved.append(ids[0])
        elif len(ids) > 1:
            ambiguous.append((token, ids[:5]))
        else:
            missing.append(token)

    if ambiguous or missing:
        parts = []
        if missing:
            parts.append(f"missing={missing[:10]}")
        if ambiguous:
            parts.append(f"ambiguous={ambiguous[:5]}")
        raise ValueError("Could not resolve target genes: " + "; ".join(parts))
    return np.asarray(resolved, dtype=object)


def transform_counts_array(
    counts: np.ndarray,
    expression_space: str = "log1p",
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    counts = np.clip(counts, a_min=0.0, a_max=None)
    if expression_space == "log1p":
        return np.log1p(counts).astype(np.float32, copy=False)
    if expression_space == "logcpm":
        if counts.ndim == 1:
            lib = float(max(counts.sum(), 1.0))
            return np.log1p((counts / lib) * 1e6).astype(np.float32, copy=False)
        lib = counts.sum(axis=1, keepdims=True)
        lib = np.maximum(lib, 1.0)
        return np.log1p((counts / lib) * 1e6).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported expression_space: {expression_space}")


def compute_empirical_prior_stats(
    count_matrix,
    expression_space: str = "log1p",
) -> tuple[float, float]:
    if sp.issparse(count_matrix):
        X = count_matrix.toarray().astype(np.float32, copy=False)
    else:
        X = np.asarray(count_matrix, dtype=np.float32)
    expr = transform_counts_array(X, expression_space=expression_space)
    nz = expr[expr > 0]
    if nz.size == 0:
        return 0.0, 1.0
    return float(nz.mean()), float(max(nz.std(), 1e-6))


class SpatialGeneImputationDataset(Dataset):
    def __init__(
        self,
        count_matrix,
        visible_gene_idx: Optional[np.ndarray] = None,
        expression_space: str = "log1p",
    ):
        self.count_matrix = count_matrix
        self.sparse = sp.issparse(count_matrix)
        self.visible_gene_idx = None
        self.expression_space = str(expression_space)
        if visible_gene_idx is not None:
            self.visible_gene_idx = np.asarray(visible_gene_idx, dtype=np.int64)

    def __len__(self):
        return self.count_matrix.shape[0]

    def __getitem__(self, idx):
        if self.sparse:
            raw = self.count_matrix.getrow(idx).toarray().ravel().astype(np.float32)
        else:
            raw = np.asarray(self.count_matrix[idx], dtype=np.float32).ravel()
        if self.visible_gene_idx is not None:
            visible_raw = np.zeros_like(raw, dtype=np.float32)
            visible_raw[self.visible_gene_idx] = raw[self.visible_gene_idx]
            raw = visible_raw
        x = transform_counts_array(raw, expression_space=self.expression_space)
        return torch.from_numpy(x), torch.tensor(int(idx), dtype=torch.long)


def load_model_from_checkpoint(
    cfg_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    scRNAModel = _load_scuniflow_model_class()

    with cfg_path.open() as f:
        cfg = json.load(f) if cfg_path.suffix == ".json" else None
    if cfg is None:
        import yaml

        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = scRNAModel(
        n_genes=int(model_cfg["n_genes"]),
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg, ckpt


def safe_pearson(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return None
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return None
    return float(r)


def safe_spearman(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return None
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return None
    r = spearmanr(x, y).correlation
    if r is None or np.isnan(r):
        return None
    return float(r)


def summarize_target_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    diff = pred - true

    spot_pearsons = []
    spot_spearmans = []
    for i in range(pred.shape[0]):
        r = safe_pearson(pred[i], true[i])
        if r is not None:
            spot_pearsons.append(r)
        sr = safe_spearman(pred[i], true[i])
        if sr is not None:
            spot_spearmans.append(sr)

    gene_pearsons = []
    gene_spearmans = []
    for j in range(pred.shape[1]):
        r = safe_pearson(pred[:, j], true[:, j])
        if r is not None:
            gene_pearsons.append(r)
        sr = safe_spearman(pred[:, j], true[:, j])
        if sr is not None:
            gene_spearmans.append(sr)

    pearson_all = safe_pearson(pred.reshape(-1), true.reshape(-1))
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    return {
        "target_gene_count": int(pred.shape[1]),
        "spot_count": int(pred.shape[0]),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "pearson_all": pearson_all,
        "spot_pearson_mean": float(np.mean(spot_pearsons)) if spot_pearsons else None,
        "spot_spearman_mean": float(np.mean(spot_spearmans)) if spot_spearmans else None,
        "gene_pearson_mean": float(np.mean(gene_pearsons)) if gene_pearsons else None,
        "gene_spearman_mean": float(np.mean(gene_spearmans)) if gene_spearmans else None,
        "n_spots_for_pearson": int(len(spot_pearsons)),
        "n_genes_for_pearson": int(len(gene_pearsons)),
    }


def build_target_prediction_adata(
    spatial: ad.AnnData,
    ref: ReferenceData,
    target_idx: np.ndarray,
    pred_targets: np.ndarray,
    truth_targets: Optional[np.ndarray] = None,
    expression_space: str = "log1p",
) -> ad.AnnData:
    var = pd.DataFrame(index=ref.ensembl_ids[target_idx].copy())
    var["SYMBOL"] = ref.gene_symbols[target_idx].copy()
    out = ad.AnnData(
        X=np.asarray(pred_targets, dtype=np.float32),
        obs=spatial.obs.copy(),
        var=var,
    )
    if "spatial" in spatial.obsm:
        out.obsm["spatial"] = np.asarray(spatial.obsm["spatial"]).copy()
    pred_key = "pred_logcpm" if expression_space == "logcpm" else "pred_log1p"
    truth_key = "truth_logcpm" if expression_space == "logcpm" else "truth_log1p"
    out.layers[pred_key] = np.asarray(pred_targets, dtype=np.float32)
    if truth_targets is not None:
        out.layers[truth_key] = np.asarray(truth_targets, dtype=np.float32)
    out.uns["expression_space"] = str(expression_space)
    return out


def generate_ordered_coords_for_reference(ref_h5ad: Path, output_path: Path, grid_size: int):
    ref = load_reference_metadata(ref_h5ad)
    coords = build_ordered_gene_coords(
        n_genes=int(ref.ensembl_ids.shape[0]),
        grid_size=int(grid_size),
        order_mode="hilbert",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, coords)
    return ref, coords
