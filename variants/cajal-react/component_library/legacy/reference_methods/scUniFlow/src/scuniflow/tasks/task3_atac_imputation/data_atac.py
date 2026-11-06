import os
import signal
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

TFIDF_LOG1P_SCALE = 10000.0


DEFAULT_ATAC_PATHS = {}

_interrupted = False


def _sigint_handler(signum, frame):
    del signum, frame
    global _interrupted
    _interrupted = True
    print("\n[!] 收到中断信号，将在当前 I/O 操作完成后退出...")


def _check_interrupt():
    global _interrupted
    if _interrupted:
        _interrupted = False
        raise KeyboardInterrupt("用户中断")


def split_obs_donors_train_val(obs_donors, val_ratio: float = 0.1, seed: int = 42):
    donor_ids, counts = np.unique(np.asarray(obs_donors).astype(str), return_counts=True)
    donors = list(zip(donor_ids.tolist(), counts.astype(int).tolist()))

    rng = np.random.RandomState(seed)
    rng.shuffle(donors)
    donors.sort(key=lambda x: x[1])

    total_cells = sum(n for _, n in donors)
    target_val = total_cells * val_ratio

    val_donors, val_cells = [], 0
    remaining = []
    for donor_id, n_cells in donors:
        if val_cells < target_val:
            val_donors.append(donor_id)
            val_cells += n_cells
        else:
            remaining.append(donor_id)

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


def split_cells_train_val(n_cells: int, val_ratio: float = 0.1, seed: int = 42):
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
        "val_ratio_actual": float(val_idx.shape[0] / max(train_idx.shape[0] + val_idx.shape[0], 1)),
    }
    return train_idx, val_idx, split_info


def split_cells_train_val_fixed(n_cells: int, val_cells: int, seed: int = 42):
    val_cells = int(val_cells)
    if n_cells <= 1 or val_cells <= 0:
        train_idx = np.arange(n_cells, dtype=np.int64)
        val_idx = np.array([], dtype=np.int64)
    else:
        n_val = max(1, min(val_cells, n_cells - 1))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_cells)
        val_idx = np.sort(perm[:n_val]).astype(np.int64)
        train_idx = np.sort(perm[n_val:]).astype(np.int64)

    split_info = {
        "train_donors": 0,
        "val_donors": 0,
        "train_cells": int(train_idx.shape[0]),
        "val_cells": int(val_idx.shape[0]),
        "val_ratio_actual": float(val_idx.shape[0] / max(train_idx.shape[0] + val_idx.shape[0], 1)),
    }
    return train_idx, val_idx, split_info


def _resolve_h5ad_path(path_or_key: str) -> str:
    return DEFAULT_ATAC_PATHS.get(path_or_key, path_or_key)


def _feature_ids_from_var(var_names) -> np.ndarray:
    feature_ids = np.asarray(var_names).astype(str)
    return np.argsort(np.argsort(feature_ids)).astype(np.int64)


def load_atac_h5ad(
    path_or_key: str,
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
):
    import scanpy as sc

    h5ad_path = _resolve_h5ad_path(path_or_key)
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"h5ad 文件不存在: {h5ad_path}")

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)
    try:
        t0 = time.time()
        print(f"[load] 正在加载 ATAC h5ad: {h5ad_path} ...")
        adata = sc.read_h5ad(h5ad_path, backed="r" if backed else None)
        if backed:
            adata = adata.to_memory()
        t1 = time.time()
        print(f"[load] 维度: {adata.shape[0]} cells × {adata.shape[1]} peaks ({t1 - t0:.1f}s)")
        _check_interrupt()

        if max_cells is not None and adata.shape[0] > int(max_cells):
            rng = np.random.RandomState(cell_seed)
            keep_idx = np.sort(rng.choice(adata.shape[0], size=int(max_cells), replace=False))
            adata = adata[keep_idx].copy()
            print(f"[load] 子采样到 {adata.shape[0]} cells (seed={cell_seed})")

        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float32)
        else:
            X = X.tocsr()
        X = X.astype(np.float32) if X.dtype != np.float32 else X

        feature_names = np.asarray(adata.var_names).astype(str)
        feature_ids = _feature_ids_from_var(feature_names)
        sparse_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        print(f"[load] 就绪，稀疏内存: {sparse_bytes / 1e9:.2f} GB")

        meta = {
            "obs": adata.obs,
            "var": adata.var,
            "feature_names": feature_names,
            "source_path": h5ad_path,
            "n_features": int(X.shape[1]),
        }
        return X, feature_ids, feature_names, meta
    finally:
        signal.signal(signal.SIGINT, prev_handler)


def load_atac_h5ad_split(
    path_or_key: str,
    val_ratio: float = 0.1,
    val_cells: int = None,
    seed: int = 42,
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
):
    import scanpy as sc

    h5ad_path = _resolve_h5ad_path(path_or_key)
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"h5ad 文件不存在: {h5ad_path}")

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)
    try:
        t0 = time.time()
        print(f"[load] 正在加载 ATAC h5ad: {h5ad_path} ...")
        adata = sc.read_h5ad(h5ad_path, backed="r" if backed else None)
        if backed:
            adata = adata.to_memory()
        t1 = time.time()
        print(f"[load] 维度: {adata.shape[0]} cells × {adata.shape[1]} peaks ({t1 - t0:.1f}s)")
        _check_interrupt()

        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float32)
        else:
            X = X.tocsr()
        X = X.astype(np.float32) if X.dtype != np.float32 else X

        feature_names = np.asarray(adata.var_names).astype(str)
        feature_ids = _feature_ids_from_var(feature_names)

        split_mode = "random_cells"
        train_donors = []
        val_donors = []
        train_obs_idx = np.arange(X.shape[0], dtype=np.int64)
        val_obs_idx = np.array([], dtype=np.int64)

        if val_cells is not None and int(val_cells) > 0:
            split_mode = "random_cells_fixed"
            train_obs_idx, val_obs_idx, split_info = split_cells_train_val_fixed(
                X.shape[0],
                val_cells=int(val_cells),
                seed=seed,
            )
            train_X = X[train_obs_idx]
            val_X = X[val_obs_idx]
            print(f"[split] fixed-cell train: {split_info['train_cells']:,} cells")
            print(f"[split] fixed-cell val:   {split_info['val_cells']:,} cells ({split_info['val_ratio_actual']:.1%})")
        elif val_ratio > 0 and "donor_id" in adata.obs.columns:
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
                    train_obs_idx = np.where(train_mask)[0].astype(np.int64)
                    val_obs_idx = np.where(val_mask)[0].astype(np.int64)
                    train_X = X[train_mask]
                    val_X = X[val_mask]
                    print(
                        f"[split] donor-level train: {split_info['train_donors']} donors / "
                        f"{split_info['train_cells']:,} cells"
                    )
                    print(
                        f"[split] donor-level val:   {split_info['val_donors']} donors / "
                        f"{split_info['val_cells']:,} cells ({split_info['val_ratio_actual']:.1%})"
                    )
                else:
                    print("[split] donor-level 划分退化为空，回退到 cell-level 随机划分")
                    train_idx, val_idx, split_info = split_cells_train_val(
                        X.shape[0],
                        val_ratio=val_ratio,
                        seed=seed,
                    )
                    train_obs_idx = train_idx
                    val_obs_idx = val_idx
                    train_X = X[train_idx]
                    val_X = X[val_idx]
            else:
                print(f"[split] 仅检测到 {unique_donors.size} 个 donor，回退到 cell-level 随机划分")
                train_idx, val_idx, split_info = split_cells_train_val(
                    X.shape[0],
                    val_ratio=val_ratio,
                    seed=seed,
                )
                train_obs_idx = train_idx
                val_obs_idx = val_idx
                train_X = X[train_idx]
                val_X = X[val_idx]
        else:
            if val_ratio > 0:
                print("[split] 数据缺少 donor_id，回退到 cell-level 随机划分")
                train_idx, val_idx, split_info = split_cells_train_val(
                    X.shape[0],
                    val_ratio=val_ratio,
                    seed=seed,
                )
                train_obs_idx = train_idx
                val_obs_idx = val_idx
                train_X = X[train_idx]
                val_X = X[val_idx]
            else:
                train_X = X
                val_X = X[:0]
                train_obs_idx = np.arange(X.shape[0], dtype=np.int64)
                val_obs_idx = np.array([], dtype=np.int64)
                split_mode = "train_only"
                split_info = {
                    "train_donors": 0,
                    "val_donors": 0,
                    "train_cells": int(train_X.shape[0]),
                    "val_cells": int(val_X.shape[0]),
                    "val_ratio_actual": 0.0,
                }

        if split_mode == "random_cells":
            print(f"[split] random cell train: {split_info['train_cells']:,} cells")
            print(f"[split] random cell val:   {split_info['val_cells']:,} cells ({split_info['val_ratio_actual']:.1%})")

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
                val_idx = (
                    np.sort(rng.choice(val_X.shape[0], size=target_val, replace=False))
                    if target_val > 0
                    else np.array([], dtype=np.int64)
                )
                train_X = train_X[train_idx]
                val_X = val_X[val_idx] if target_val > 0 else val_X[:0]
                train_obs_idx = train_obs_idx[train_idx]
                val_obs_idx = val_obs_idx[val_idx] if target_val > 0 else val_obs_idx[:0]
                print(
                    f"[split] 子采样到 train={train_X.shape[0]:,}, val={val_X.shape[0]:,} "
                    f"(max_cells={max_cells}, seed={cell_seed})"
                )

        meta = {
            "obs": adata.obs,
            "var": adata.var,
            "feature_names": feature_names,
            "source_path": h5ad_path,
            "n_features": int(X.shape[1]),
            "split_mode": split_mode,
            "split_info": split_info,
            "split_info_effective": {
                **split_info,
                "train_cells": int(train_X.shape[0]),
                "val_cells": int(val_X.shape[0]),
                "val_ratio_actual": float(val_X.shape[0] / max(train_X.shape[0] + val_X.shape[0], 1)),
            },
            "train_donors": train_donors,
            "val_donors": val_donors,
            "train_indices": train_obs_idx.astype(np.int64),
            "val_indices": val_obs_idx.astype(np.int64),
        }
        return (train_X, feature_ids), (val_X, feature_ids), feature_names, meta
    finally:
        signal.signal(signal.SIGINT, prev_handler)


def _transform_accessibility_numpy(values: np.ndarray, expr_transform: str = "identity") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, a_min=0.0, a_max=None)
    if expr_transform == "identity":
        return values.astype(np.float32, copy=False)
    if expr_transform == "log1p":
        return np.log1p(values).astype(np.float32, copy=False)
    if expr_transform == "tfidf_log1p_1e4":
        return np.log1p(values * TFIDF_LOG1P_SCALE).astype(np.float32, copy=False)
    if expr_transform == "logcpm":
        if values.ndim == 1:
            lib = float(max(values.sum(), 1.0))
            return np.log1p((values / lib) * 1e6).astype(np.float32, copy=False)
        lib = values.sum(axis=1, keepdims=True)
        lib = np.maximum(lib, 1.0)
        return np.log1p((values / lib) * 1e6).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported expr_transform: {expr_transform}")


def compute_accessibility_prior_stats(count_matrix, expr_transform: str = "identity") -> tuple[float, float]:
    if sp.issparse(count_matrix):
        total_sum = 0.0
        total_sq = 0.0
        total_n = 0
        for row_idx in range(count_matrix.shape[0]):
            start = int(count_matrix.indptr[row_idx])
            end = int(count_matrix.indptr[row_idx + 1])
            if start == end:
                continue
            row_raw = count_matrix.data[start:end].astype(np.float32, copy=False)
            if expr_transform == "logcpm":
                lib = float(max(count_matrix[row_idx].sum(), 1.0))
                row_expr = np.log1p((row_raw / lib) * 1e6).astype(np.float32, copy=False)
            else:
                # Prior stats must be computed in the same transformed space used by the model.
                row_expr = _transform_accessibility_numpy(row_raw, expr_transform=expr_transform)
            total_sum += float(row_expr.sum())
            total_sq += float(np.square(row_expr).sum())
            total_n += int(row_expr.size)
        if total_n == 0:
            return 0.0, 1.0
        mean = total_sum / total_n
        var = max(total_sq / total_n - mean * mean, 1e-12)
        return float(mean), float(np.sqrt(var))

    expr = _transform_accessibility_numpy(count_matrix, expr_transform=expr_transform)
    nz = expr[expr > 0]
    if nz.size == 0:
        return 0.0, 1.0
    return float(nz.mean()), float(max(nz.std(), 1e-6))


class scATACDataset(Dataset):
    def __init__(
        self,
        count_matrix,
        feature_ids,
        x0_prior: str = "empirical",
        prior_mean: float = None,
        prior_std: float = None,
        nonzero_mask_ratio: float = 0.2,
        zero_mask_ratio: float = 0.1,
        return_index: bool = False,
        expr_transform: str = "identity",
    ):
        self.sparse = sp.issparse(count_matrix)
        if self.sparse:
            self.counts = count_matrix.tocsr().astype(np.float32)
        else:
            self.counts = np.asarray(count_matrix, dtype=np.float32)
        self.feature_ids = np.asarray(feature_ids, dtype=np.int64)
        self.x0_prior = x0_prior
        self.nonzero_mask_ratio = float(nonzero_mask_ratio)
        self.zero_mask_ratio = float(zero_mask_ratio)
        self.return_index = bool(return_index)
        self.expr_transform = str(expr_transform)

        if prior_mean is not None and prior_std is not None:
            self.expr_mean = float(prior_mean)
            self.expr_std = float(max(prior_std, 1e-6))
        else:
            self.expr_mean, self.expr_std = compute_accessibility_prior_stats(
                self.counts,
                expr_transform=self.expr_transform,
            )

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, idx):
        if self.sparse:
            x_raw = self.counts.getrow(idx).toarray().ravel().astype(np.float32, copy=False)
        else:
            x_raw = np.asarray(self.counts[idx], dtype=np.float32)

        x_1 = torch.from_numpy(
            _transform_accessibility_numpy(
                x_raw,
                expr_transform=self.expr_transform,
            )
        )
        feature_id = torch.tensor(self.feature_ids, dtype=torch.long)
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
            return feature_id, x_1, x_0, mask_indicator, torch.tensor(idx, dtype=torch.long)
        return feature_id, x_1, x_0, mask_indicator


def make_dataloader(
    count_matrix,
    feature_ids,
    batch_size,
    shuffle=True,
    world_size=1,
    x0_prior: str = "empirical",
    prior_mean: float = None,
    prior_std: float = None,
    nonzero_mask_ratio: float = 0.2,
    zero_mask_ratio: float = 0.1,
    return_index: bool = False,
    expr_transform: str = "identity",
):
    dataset = scATACDataset(
        count_matrix,
        feature_ids,
        x0_prior=x0_prior,
        prior_mean=prior_mean,
        prior_std=prior_std,
        nonzero_mask_ratio=nonzero_mask_ratio,
        zero_mask_ratio=zero_mask_ratio,
        return_index=return_index,
        expr_transform=expr_transform,
    )
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=shuffle)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
        return loader, sampler

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
    return loader, None
