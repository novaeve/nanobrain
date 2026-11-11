import csv
import os
import signal
import time
import torch
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from torch.utils.data import Dataset

try:
    from paths import DATA_ROOT, SEA_AD_ROOT
except ImportError:  # package import fallback
    from .paths import DATA_ROOT, SEA_AD_ROOT

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")


_interrupted = False

def _sigint_handler(signum, frame):
    global _interrupted
    _interrupted = True
    print("\n[!] 收到中断信号，将在当前 I/O 操作完成后退出...")

def _check_interrupt():
    global _interrupted
    if _interrupted:
        _interrupted = False
        raise KeyboardInterrupt("用户中断")


# ---------------------------------------------------------------------------
# SEA-AD h5ad 数据加载接口
# ---------------------------------------------------------------------------

DEFAULT_PATHS = {
    "mtg_full": DATA_ROOT / "external" / "MTG" / "92b37feb-aa2c-40d7-bd90-0a9b5ddb3b27.h5ad",
    "dlpfc_full": SEA_AD_ROOT / "DLPFC" / "dad4819b-4c14-439c-b32a-2c8d68bd22e1.h5ad",
    "mtg_toy": SEA_AD_ROOT / "MTG" / "sea_ad_mtg_toy_50k.h5ad",
    "dlpfc_toy": SEA_AD_ROOT / "DLPFC" / "sea_ad_dlpfc_toy_50k.h5ad",
}


def load_sea_ad(
    path_or_key: str = "mtg_toy",
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
):
    """
    从 SEA-AD h5ad 文件加载全量基因数据，保持稀疏格式以节省内存和加载时间。

    不做列重排序——保持 h5ad 文件原始列顺序，通过 gene_ids 记录
    每列对应的全局基因编号（Ensembl ID 字典序排名）。

    Returns
    -------
    count_matrix : scipy.sparse.csr_matrix, shape (N_cells, N_genes)
    gene_ids : np.ndarray, shape (N_genes,)
        每列基因在 Ensembl ID 字典序中的排名，作为模型的基因编号。
    gene_names : np.ndarray, shape (N_genes,)
        每列对应的 Ensembl ID（原始列顺序）。
    adata_meta : dict
    """
    import scanpy as sc

    h5ad_path = DEFAULT_PATHS.get(path_or_key, Path(path_or_key).expanduser())
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"h5ad 文件不存在: {h5ad_path}")

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)
    try:
        t0 = time.time()
        print(f"[load] 正在加载 {h5ad_path} ...")
        adata = sc.read_h5ad(h5ad_path, backed="r" if backed else None)
        if backed:
            adata = adata.to_memory()
        t1 = time.time()
        print(f"[load] 维度: {adata.shape[0]} cells × {adata.shape[1]} genes "
              f"({t1 - t0:.1f}s)")
        _check_interrupt()

        if max_cells is not None and adata.shape[0] > int(max_cells):
            rng = np.random.RandomState(cell_seed)
            keep_idx = np.sort(rng.choice(adata.shape[0], size=int(max_cells), replace=False))
            adata = adata[keep_idx].copy()
            print(f"[load] 子采样到 {adata.shape[0]} cells (seed={cell_seed})")

        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float32)
        count_matrix = X.astype(np.float32) if X.dtype != np.float32 else X

        ensembl_ids = np.array(adata.var.index)
        # gene_ids[i] = 列 i 在 Ensembl 字典序中的排名
        gene_ids = np.argsort(np.argsort(ensembl_ids)).astype(np.int64)

        n_genes = count_matrix.shape[1]
        sparse_bytes = (count_matrix.data.nbytes + count_matrix.indices.nbytes
                        + count_matrix.indptr.nbytes)
        print(f"[load] 就绪 ({t1 - t0:.1f}s), "
              f"稀疏内存: {sparse_bytes / 1e9:.1f} GB")

        adata_meta = {
            "obs": adata.obs,
            "ensembl_ids": ensembl_ids,
            "source_path": h5ad_path,
            "n_genes": n_genes,
        }
        if "feature_name" in adata.var.columns:
            adata_meta["gene_symbols"] = adata.var["feature_name"].values

        return count_matrix, gene_ids, ensembl_ids, adata_meta
    finally:
        signal.signal(signal.SIGINT, prev_handler)


def load_sea_ad_by_donor(
    donor_dir: str | Path = SEA_AD_ROOT / "MTG" / "donor_h5ad",
    donor_id: str = None,
    **kwargs,
):
    """加载按 donor 拆分的 h5ad 文件。"""
    donor_path = Path(donor_dir)
    if not donor_path.is_dir():
        raise FileNotFoundError(f"Donor 目录不存在: {donor_dir}")

    available = sorted(p.stem for p in donor_path.glob("*.h5ad"))
    if donor_id is None:
        return available

    h5ad_file = donor_path / f"{donor_id}.h5ad"
    if not h5ad_file.exists():
        raise FileNotFoundError(
            f"Donor 文件 {h5ad_file} 不存在。可用: {available[:5]}..."
        )
    return load_sea_ad(str(h5ad_file), **kwargs)


# ---------------------------------------------------------------------------
# Donor-level train/val 划分
# ---------------------------------------------------------------------------

def split_donors_train_val(
    donor_csv: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    按 donor 粒度划分 train/val，使验证集细胞总数接近 val_ratio。
    采用「从小到大贪心装箱」策略。
    """
    donors = []
    with open(donor_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            donors.append((row["donor_id"], int(row["n_cells"])))

    rng = np.random.RandomState(seed)
    rng.shuffle(donors)
    donors.sort(key=lambda x: x[1])

    total_cells = sum(n for _, n in donors)
    target_val = total_cells * val_ratio

    val_donors, val_cells = [], 0
    remaining = []
    for did, n in donors:
        if val_cells < target_val:
            val_donors.append(did)
            val_cells += n
        else:
            remaining.append(did)

    train_donors = remaining
    train_cells = total_cells - val_cells

    split_info = {
        "train_donors": len(train_donors),
        "val_donors": len(val_donors),
        "train_cells": train_cells,
        "val_cells": val_cells,
        "val_ratio_actual": val_cells / total_cells,
    }
    return train_donors, val_donors, split_info


def split_obs_donors_train_val(
    obs_donors,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    根据 adata.obs['donor_id'] 的实际 donor 分布做 train/val donor split。
    """
    donor_ids, counts = np.unique(np.asarray(obs_donors).astype(str), return_counts=True)
    donors = list(zip(donor_ids.tolist(), counts.astype(int).tolist()))

    rng = np.random.RandomState(seed)
    rng.shuffle(donors)
    donors.sort(key=lambda x: x[1])

    total_cells = sum(n for _, n in donors)
    target_val = total_cells * val_ratio

    val_donors, val_cells = [], 0
    remaining = []
    for did, n in donors:
        if val_cells < target_val:
            val_donors.append(did)
            val_cells += n
        else:
            remaining.append(did)

    train_donors = remaining
    train_cells = total_cells - val_cells

    split_info = {
        "train_donors": len(train_donors),
        "val_donors": len(val_donors),
        "train_cells": train_cells,
        "val_cells": val_cells,
        "val_ratio_actual": val_cells / max(total_cells, 1),
    }
    return train_donors, val_donors, split_info


def split_cells_train_val(
    n_cells: int,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    当 donor-level split 不适用时，回退到 cell-level 随机划分。
    """
    if n_cells <= 1 or val_ratio <= 0:
        train_idx = np.arange(n_cells, dtype=np.int64)
        val_idx = np.array([], dtype=np.int64)
    else:
        n_val = int(round(n_cells * val_ratio))
        n_val = max(1, min(n_val, n_cells - 1))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_cells)
        val_idx = np.sort(perm[:n_val]).astype(np.int64)
        train_idx = np.sort(perm[n_val:]).astype(np.int64)

    split_info = {
        "train_donors": 0,
        "val_donors": 0,
        "train_cells": int(train_idx.shape[0]),
        "val_cells": int(val_idx.shape[0]),
        "val_ratio_actual": (
            float(val_idx.shape[0] / max(train_idx.shape[0] + val_idx.shape[0], 1))
        ),
    }
    return train_idx, val_idx, split_info


def load_sea_ad_split(
    path_or_key: str = "mtg_full",
    donor_csv: str = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
):
    """
    加载 h5ad 并按 donor 划分为 train / val 两组（稀疏格式）。

    Returns
    -------
    train_data : (csr_matrix, gene_ids)
    val_data : (csr_matrix, gene_ids)
    gene_names : np.ndarray
    meta : dict
    """
    import scanpy as sc

    h5ad_path = DEFAULT_PATHS.get(path_or_key, path_or_key)
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"h5ad 文件不存在: {h5ad_path}")

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)
    try:
        t0 = time.time()
        print(f"[load] 正在加载 {h5ad_path} ...")
        adata = sc.read_h5ad(h5ad_path, backed="r" if backed else None)
        if backed:
            adata = adata.to_memory()
        t1 = time.time()
        print(f"[load] 维度: {adata.shape[0]} cells × {adata.shape[1]} genes "
              f"({t1 - t0:.1f}s)")
        _check_interrupt()

        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float32)
        X = X.astype(np.float32) if X.dtype != np.float32 else X

        ensembl_ids = np.array(adata.var.index)
        gene_ids = np.argsort(np.argsort(ensembl_ids)).astype(np.int64)

        n_genes = X.shape[1]
        split_mode = "random_cells"
        train_donors = []
        val_donors = []

        if val_ratio > 0 and "donor_id" in adata.obs.columns:
            obs_donors = adata.obs["donor_id"].astype(str).to_numpy()
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
                    train_X = X[train_mask]
                    val_X = X[val_mask]
                    print(
                        f"[split] donor-level train: {split_info['train_donors']} donors / "
                        f"{split_info['train_cells']:,} cells"
                    )
                    print(
                        f"[split] donor-level val:   {split_info['val_donors']} donors / "
                        f"{split_info['val_cells']:,} cells  "
                        f"({split_info['val_ratio_actual']:.1%})"
                    )
                else:
                    print(
                        "[split] donor-level 划分退化为空 train/val，"
                        "回退到 cell-level 随机划分"
                    )
                    train_idx, val_idx, split_info = split_cells_train_val(
                        X.shape[0],
                        val_ratio=val_ratio,
                        seed=seed,
                    )
                    train_X = X[train_idx]
                    val_X = X[val_idx]
            else:
                print(
                    f"[split] 仅检测到 {unique_donors.size} 个 donor，"
                    "回退到 cell-level 随机划分"
                )
                train_idx, val_idx, split_info = split_cells_train_val(
                    X.shape[0],
                    val_ratio=val_ratio,
                    seed=seed,
                )
                train_X = X[train_idx]
                val_X = X[val_idx]
        else:
            if val_ratio > 0:
                print(
                    "[split] 数据缺少 donor_id 或未启用 donor split，"
                    "回退到 cell-level 随机划分"
                )
                train_idx, val_idx, split_info = split_cells_train_val(
                    X.shape[0],
                    val_ratio=val_ratio,
                    seed=seed,
                )
                train_X = X[train_idx]
                val_X = X[val_idx]
            else:
                train_X = X
                val_X = X[:0]
                split_info = {
                    "train_donors": 0,
                    "val_donors": 0,
                    "train_cells": int(train_X.shape[0]),
                    "val_cells": int(val_X.shape[0]),
                    "val_ratio_actual": 0.0,
                }
                split_mode = "train_only"

        if split_mode == "random_cells":
            print(
                f"[split] random cell train: {split_info['train_cells']:,} cells"
            )
            print(
                f"[split] random cell val:   {split_info['val_cells']:,} cells  "
                f"({split_info['val_ratio_actual']:.1%})"
            )

        if max_cells is not None:
            max_cells = int(max_cells)
            total_cells = train_X.shape[0] + val_X.shape[0]
            if total_cells > max_cells:
                rng = np.random.RandomState(cell_seed)
                target_train = max(1, int(round(max_cells * (train_X.shape[0] / total_cells))))
                target_train = min(target_train, train_X.shape[0])
                target_val = max_cells - target_train
                target_val = min(target_val, val_X.shape[0])
                if target_val == 0 and val_X.shape[0] > 0:
                    target_val = 1
                    target_train = min(max_cells - 1, train_X.shape[0])

                train_idx = np.sort(rng.choice(train_X.shape[0], size=target_train, replace=False))
                val_idx = np.sort(rng.choice(val_X.shape[0], size=target_val, replace=False)) if target_val > 0 else np.array([], dtype=np.int64)
                train_X = train_X[train_idx]
                val_X = val_X[val_idx] if target_val > 0 else val_X[:0]
                print(
                    f"[split] 子采样到 train={train_X.shape[0]:,}, val={val_X.shape[0]:,} "
                    f"(max_cells={max_cells}, seed={cell_seed})"
                )

        t2 = time.time()
        print(f"[split] 实际 train: {train_X.shape[0]:,} cells, "
              f"val: {val_X.shape[0]:,} cells ({t2 - t0:.1f}s)")
        _check_interrupt()

        meta = {
            "obs": adata.obs,
            "ensembl_ids": ensembl_ids,
            "source_path": h5ad_path,
            "n_genes": n_genes,
            "split_mode": split_mode,
            "split_info": split_info,
            "split_info_effective": {
                **split_info,
                "train_cells": int(train_X.shape[0]),
                "val_cells": int(val_X.shape[0]),
                "val_ratio_actual": (
                    float(val_X.shape[0] / max(train_X.shape[0] + val_X.shape[0], 1))
                ),
            },
            "train_donors": train_donors,
            "val_donors": val_donors,
        }
        if "feature_name" in adata.var.columns:
            meta["gene_symbols"] = adata.var["feature_name"].values

        return (train_X, gene_ids), (val_X, gene_ids), ensembl_ids, meta
    finally:
        signal.signal(signal.SIGINT, prev_handler)


# ---------------------------------------------------------------------------
# Dataset & DataLoader — 支持稀疏和稠密输入
# ---------------------------------------------------------------------------

def _transform_counts_numpy(
    counts: np.ndarray,
    expr_transform: str = "log1p",
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    counts = np.clip(counts, a_min=0.0, a_max=None)
    if expr_transform == "log1p":
        return np.log1p(counts).astype(np.float32, copy=False)
    if expr_transform == "logcpm":
        if counts.ndim == 1:
            lib = float(max(counts.sum(), 1.0))
            return np.log1p((counts / lib) * 1e6).astype(np.float32, copy=False)
        lib = counts.sum(axis=1, keepdims=True)
        lib = np.maximum(lib, 1.0)
        return np.log1p((counts / lib) * 1e6).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported expr_transform: {expr_transform}")


def compute_expression_prior_stats(
    count_matrix,
    expr_transform: str = "log1p",
) -> tuple[float, float]:
    if sp.issparse(count_matrix):
        row_sums = np.asarray(count_matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
        total_sum = 0.0
        total_sq = 0.0
        total_n = 0
        for row_idx in range(count_matrix.shape[0]):
            start = int(count_matrix.indptr[row_idx])
            end = int(count_matrix.indptr[row_idx + 1])
            if start == end:
                continue
            row_raw = count_matrix.data[start:end].astype(np.float32, copy=False)
            if expr_transform == "log1p":
                row_expr = np.log1p(row_raw)
            elif expr_transform == "logcpm":
                lib = float(max(row_sums[row_idx], 1.0))
                row_expr = np.log1p((row_raw / lib) * 1e6)
            else:
                raise ValueError(f"Unsupported expr_transform: {expr_transform}")
            total_sum += float(row_expr.sum())
            total_sq += float(np.square(row_expr).sum())
            total_n += int(row_expr.size)
        if total_n == 0:
            return 0.0, 1.0
        mean = total_sum / total_n
        var = max(total_sq / total_n - mean * mean, 1e-12)
        return float(mean), float(np.sqrt(var))

    expr = _transform_counts_numpy(count_matrix, expr_transform=expr_transform)
    nz = expr[expr > 0]
    if nz.size == 0:
        return 0.0, 1.0
    return float(nz.mean()), float(max(nz.std(), 1e-6))

class scRNADataset(Dataset):
    def __init__(
        self,
        count_matrix,
        gene_ids,
        x0_prior: str = "empirical",
        prior_mean: float = None,
        prior_std: float = None,
        nonzero_mask_ratio: float = 0.2,
        zero_mask_ratio: float = 0.0,
        return_index: bool = False,
        expr_transform: str = "log1p",
    ):
        """
        count_matrix: scipy.sparse.csr_matrix 或 np.ndarray, (N_cells, N_genes)
        gene_ids: (N_genes,)
        """
        self.sparse = sp.issparse(count_matrix)
        if self.sparse:
            self.counts = count_matrix.astype(np.float32)
        else:
            self.counts = np.asarray(count_matrix, dtype=np.float32)
        self.gene_ids = gene_ids
        self.x0_prior = x0_prior
        self.nonzero_mask_ratio = float(nonzero_mask_ratio)
        self.zero_mask_ratio = float(zero_mask_ratio)
        self.return_index = bool(return_index)
        self.expr_transform = str(expr_transform)

        # x0 prior statistics are computed in the chosen expression space.
        if prior_mean is not None and prior_std is not None:
            self.expr_mean = float(prior_mean)
            self.expr_std = float(max(prior_std, 1e-6))
        else:
            self.expr_mean, self.expr_std = compute_expression_prior_stats(
                self.counts,
                expr_transform=self.expr_transform,
            )

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, idx):
        if self.sparse:
            x_raw = torch.from_numpy(
                self.counts.getrow(idx).toarray().ravel()
            )
        else:
            x_raw = torch.tensor(self.counts[idx], dtype=torch.float32)

        x_1 = torch.from_numpy(
            _transform_counts_numpy(
                x_raw.cpu().numpy(),
                expr_transform=self.expr_transform,
            )
        )
        g_id = torch.tensor(self.gene_ids, dtype=torch.long)
        if self.x0_prior == "empirical":
            x_0 = torch.randn_like(x_1) * self.expr_std + self.expr_mean
        else:
            x_0 = torch.randn_like(x_1)

        nonzero_idx = x_1.nonzero(as_tuple=True)[0]
        zero_idx = (x_1 == 0).nonzero(as_tuple=True)[0]
        mask_indicator = torch.zeros(len(x_1), dtype=torch.float32)
        if len(nonzero_idx) > 0:
            n_mask = max(1, int(len(nonzero_idx) * self.nonzero_mask_ratio))
            perm = torch.randperm(len(nonzero_idx))[:n_mask]
            chosen = nonzero_idx[perm]
            mask_indicator[chosen] = 1.0
        if self.zero_mask_ratio > 0 and len(zero_idx) > 0:
            n_mask_zero = int(len(zero_idx) * self.zero_mask_ratio)
            if n_mask_zero > 0:
                perm = torch.randperm(len(zero_idx))[:n_mask_zero]
                chosen = zero_idx[perm]
                mask_indicator[chosen] = 1.0

        if self.return_index:
            return g_id, x_1, x_0, mask_indicator, torch.tensor(idx, dtype=torch.long)
        return g_id, x_1, x_0, mask_indicator


def get_dataloader(
    count_matrix,
    gene_ids,
    batch_size=64,
    shuffle=True,
    x0_prior: str = "empirical",
    prior_mean: float = None,
    prior_std: float = None,
    nonzero_mask_ratio: float = 0.2,
    zero_mask_ratio: float = 0.0,
    return_index: bool = False,
    expr_transform: str = "log1p",
):
    dataset = scRNADataset(
        count_matrix,
        gene_ids,
        x0_prior=x0_prior,
        prior_mean=prior_mean,
        prior_std=prior_std,
        nonzero_mask_ratio=nonzero_mask_ratio,
        zero_mask_ratio=zero_mask_ratio,
        return_index=return_index,
        expr_transform=expr_transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4,
    )
