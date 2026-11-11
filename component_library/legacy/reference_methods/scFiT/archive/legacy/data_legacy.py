import csv
import hashlib
import json
import os
import re
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


_NUMBA_CACHE_DIR = Path(os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/scfit_numba_cache"))
_MPLCONFIGDIR = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/scfit_matplotlib"))
_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)


_interrupted = False


def _limit_worker_threads(_worker_id):
    _configure_cpu_thread_env()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _sigint_handler(signum, frame):
    del signum, frame
    global _interrupted
    if not _interrupted:
        _interrupted = True
        print("\n[!] 收到 Ctrl+C，中断当前数据加载...")
    raise KeyboardInterrupt("用户中断")


def _check_interrupt():
    global _interrupted
    if _interrupted:
        _interrupted = False
        raise KeyboardInterrupt("用户中断")


@contextmanager
def _interruptible_io():
    global _interrupted
    _interrupted = False
    prev_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            signal.siginterrupt(signal.SIGINT, True)
        except AttributeError:
            pass
        yield
    finally:
        _interrupted = False
        signal.signal(signal.SIGINT, prev_handler)
        try:
            signal.siginterrupt(signal.SIGINT, False)
        except AttributeError:
            pass


DEFAULT_PATHS = {
    "mtg_full": "/data1/SEA-AD/MTG/92b37feb-aa2c-40d7-bd90-0a9b5ddb3b27.h5ad",
    "dlpfc_full": "/data1/SEA-AD/DLPFC/dad4819b-4c14-439c-b32a-2c8d68bd22e1.h5ad",
    "mtg_toy": "/data1/SEA-AD/MTG/sea_ad_mtg_toy_50k.h5ad",
    "dlpfc_toy": "/data1/SEA-AD/DLPFC/sea_ad_dlpfc_toy_50k.h5ad",
}
DEFAULT_UCE_NPZ_PATH = "data/external/jax_uce_flow_data.npz"
PAIR_CACHE_DIR = Path(__file__).parent / "pair_cache"
PAIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_h5ad_path(path_or_key: str) -> str:
    h5ad_path = DEFAULT_PATHS.get(path_or_key, path_or_key)
    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"h5ad 文件不存在: {h5ad_path}")
    return h5ad_path


def _to_float_or_nan(value) -> float:
    if value is None:
        return np.nan
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        s = str(value).strip()
    except Exception:
        return np.nan
    if not s or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except ValueError:
        match = re.search(r"([0-9]+(?:\\.[0-9]+)?)", s)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
    return np.nan


def _parse_e_token(token: str):
    token = str(token).strip()
    if token.startswith("E"):
        token = token[1:]
    if not token:
        return None
    if token.isdigit():
        if len(token) == 3:
            return float(token[0] + "." + token[1:])
        if len(token) >= 4:
            return float(token[:2] + "." + token[2:])
        return float(token)
    if "." in token:
        left, right = token.split(".", 1)
        if left.isdigit() and right.isdigit():
            if len(left) >= 3:
                merged = left + right
                scale = 10 ** (len(merged) - 2)
                return float(merged) / float(scale)
            return float(token)
    try:
        return float(token)
    except ValueError:
        return None


def parse_time_value(value) -> float:
    if value is None:
        return np.nan
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.nan
    if s in {"P0", "P0000"}:
        return 19.0
    if s.startswith("E") and "-E" in s:
        left, right = s.split("-E", 1)
        left_v = _parse_e_token(left)
        right_v = _parse_e_token("E" + right)
        if left_v is None or right_v is None:
            return np.nan
        return 0.5 * (left_v + right_v)
    if s.startswith("E"):
        parsed = _parse_e_token(s)
        return np.nan if parsed is None else float(parsed)
    return _to_float_or_nan(s)


def _extract_time_numeric(obs, time_col: str = None, time_numeric_col: str = None):
    if time_numeric_col and time_numeric_col in obs.columns:
        values = obs[time_numeric_col].values
        return np.array([_to_float_or_nan(v) for v in values], dtype=np.float64)
    if time_col and time_col in obs.columns:
        values = obs[time_col].values
        return np.array([parse_time_value(v) for v in values], dtype=np.float64)
    raise ValueError("缺少可用的时间列，请提供 data.split.time_col 或 data.split.time_numeric_col")


def _build_count_matrix_and_meta(adata, h5ad_path: str):
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float32)
    count_matrix = X.astype(np.float32) if X.dtype != np.float32 else X

    ensembl_ids = np.array(adata.var.index)
    gene_ids = np.argsort(np.argsort(ensembl_ids)).astype(np.int64)

    meta = {
        "obs": adata.obs.copy(),
        "ensembl_ids": ensembl_ids,
        "source_path": h5ad_path,
        "n_genes": int(count_matrix.shape[1]),
    }
    if "feature_name" in adata.var.columns:
        meta["gene_symbols"] = adata.var["feature_name"].values
    elif "gene_short_name" in adata.var.columns:
        meta["gene_symbols"] = adata.var["gene_short_name"].values
    return count_matrix, gene_ids, ensembl_ids, meta


def load_h5ad(
    path_or_key: str = "mtg_toy",
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
):
    import scanpy as sc

    h5ad_path = _resolve_h5ad_path(path_or_key)

    with _interruptible_io():
        t0 = time.time()
        print(f"[load] 正在加载 {h5ad_path} ...")
        adata = sc.read_h5ad(h5ad_path, backed="r" if backed else None)
        if backed:
            adata = adata.to_memory()
        t1 = time.time()
        print(f"[load] 维度: {adata.shape[0]} cells x {adata.shape[1]} genes ({t1 - t0:.1f}s)")
        _check_interrupt()

        if max_cells is not None and adata.shape[0] > int(max_cells):
            rng = np.random.RandomState(cell_seed)
            keep_idx = np.sort(rng.choice(adata.shape[0], size=int(max_cells), replace=False))
            adata = adata[keep_idx].copy()
            print(f"[load] 子采样到 {adata.shape[0]} cells (seed={cell_seed})")

        return _build_count_matrix_and_meta(adata, h5ad_path)


def split_donors_train_val(
    donor_csv: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
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
    for donor_id, n_cells in donors:
        if val_cells < target_val:
            val_donors.append(donor_id)
            val_cells += n_cells
        else:
            remaining.append(donor_id)

    train_donors = remaining
    split_info = {
        "train_donors": len(train_donors),
        "val_donors": len(val_donors),
        "train_cells": total_cells - val_cells,
        "val_cells": val_cells,
        "val_ratio_actual": val_cells / max(total_cells, 1),
    }
    return train_donors, val_donors, split_info


def split_groups_train_val(
    group_ids,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    group_ids = np.asarray(group_ids).astype(str)
    uniq_groups, counts = np.unique(group_ids, return_counts=True)
    groups = list(zip(uniq_groups.tolist(), counts.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda x: x[1])

    total_cells = int(sum(n for _, n in groups))
    target_val = total_cells * float(val_ratio)

    val_groups, val_cells = [], 0
    train_groups = []
    for group_id, n_cells in groups:
        if val_cells < target_val:
            val_groups.append(group_id)
            val_cells += n_cells
        else:
            train_groups.append(group_id)

    split_info = {
        "split_strategy": "group_ratio",
        "split_column": None,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "train_cells": total_cells - val_cells,
        "val_cells": val_cells,
        "val_ratio_actual": val_cells / max(total_cells, 1),
    }
    return train_groups, val_groups, split_info


def _subsample_split_matrices(train_X, val_X, max_cells: int = None, cell_seed: int = 42):
    if max_cells is None:
        return train_X, val_X
    max_cells = int(max_cells)
    total_cells = train_X.shape[0] + val_X.shape[0]
    if total_cells <= max_cells:
        return train_X, val_X

    rng = np.random.RandomState(cell_seed)
    target_train = max(1, int(round(max_cells * (train_X.shape[0] / max(total_cells, 1)))))
    target_train = min(target_train, train_X.shape[0])
    target_val = min(max_cells - target_train, val_X.shape[0])
    train_idx = np.sort(rng.choice(train_X.shape[0], size=target_train, replace=False))
    if target_val > 0:
        val_idx = np.sort(rng.choice(val_X.shape[0], size=target_val, replace=False))
        return train_X[train_idx], val_X[val_idx]
    return train_X[train_idx], val_X[:0]


def load_h5ad_split(
    path_or_key: str,
    split_strategy: str = "donor",
    split_column: str = None,
    donor_csv: str = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
    return_group_ids: bool = False,
    time_col: str = None,
    time_numeric_col: str = None,
    train_max_time: float = None,
    val_min_time: float = None,
    val_max_time: float = None,
    train_values=None,
    val_values=None,
):
    import scanpy as sc

    h5ad_path = _resolve_h5ad_path(path_or_key)
    split_strategy = str(split_strategy or "donor").strip().lower()

    with _interruptible_io():
        t0 = time.time()
        print(f"[load] 正在加载 {h5ad_path} ...")
        adata = sc.read_h5ad(h5ad_path, backed="r" if backed else None)
        if backed:
            adata = adata.to_memory()
        t1 = time.time()
        print(f"[load] 维度: {adata.shape[0]} cells x {adata.shape[1]} genes ({t1 - t0:.1f}s)")
        _check_interrupt()

        X, gene_ids, ensembl_ids, base_meta = _build_count_matrix_and_meta(adata, h5ad_path)

        split_info = {
            "split_strategy": split_strategy,
            "split_column": split_column,
        }
        train_values_meta = None
        val_values_meta = None
        group_values = None

        if split_strategy == "donor":
            split_column = split_column or "donor_id"
            if donor_csv is not None and split_column == "donor_id":
                train_groups, val_groups, donor_split_info = split_donors_train_val(
                    donor_csv,
                    val_ratio=val_ratio,
                    seed=seed,
                )
            else:
                if split_column not in adata.obs.columns:
                    raise ValueError(f"缺少分组列: {split_column}")
                group_values = np.asarray(adata.obs[split_column].values).astype(str)
                train_groups, val_groups, donor_split_info = split_groups_train_val(
                    group_values,
                    val_ratio=val_ratio,
                    seed=seed,
                )
            split_info.update(donor_split_info)
            split_info["split_column"] = split_column
            train_values_meta = list(train_groups)
            val_values_meta = list(val_groups)
            if group_values is None:
                group_values = np.asarray(adata.obs[split_column].values).astype(str)
            val_mask = np.isin(group_values, val_groups)
            train_mask = ~val_mask
        elif split_strategy == "random":
            n_cells = X.shape[0]
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_cells)
            n_val = int(round(n_cells * float(val_ratio)))
            val_idx = np.sort(perm[:n_val])
            train_idx = np.sort(perm[n_val:])
            train_mask = np.zeros((n_cells,), dtype=bool)
            val_mask = np.zeros((n_cells,), dtype=bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            split_info.update(
                {
                    "train_cells": int(train_mask.sum()),
                    "val_cells": int(val_mask.sum()),
                    "val_ratio_actual": float(val_mask.mean()),
                }
            )
        elif split_strategy == "time_threshold":
            time_numeric = _extract_time_numeric(
                adata.obs,
                time_col=time_col or split_column,
                time_numeric_col=time_numeric_col,
            )
            finite_mask = np.isfinite(time_numeric)
            raw_values = None
            raw_time_col = time_col or split_column
            if raw_time_col and raw_time_col in adata.obs.columns:
                raw_values = np.asarray(adata.obs[raw_time_col].astype(str).values)

            if train_values is not None:
                if raw_values is None:
                    raise ValueError("time_threshold 按取值切分需要提供可用的 time_col")
                train_value_set = {str(v) for v in train_values}
                train_mask = finite_mask & np.isin(raw_values, list(train_value_set))
                train_values_meta = sorted(train_value_set)
            else:
                train_mask = finite_mask.copy()
                if train_max_time is not None:
                    train_mask &= time_numeric <= float(train_max_time)

            if val_values is not None:
                if raw_values is None:
                    raise ValueError("time_threshold 按取值切分需要提供可用的 time_col")
                val_value_set = {str(v) for v in val_values}
                val_mask = finite_mask & np.isin(raw_values, list(val_value_set))
                val_values_meta = sorted(val_value_set)
            else:
                lower = val_min_time if val_min_time is not None else train_max_time
                val_mask = finite_mask.copy()
                if lower is not None:
                    val_mask &= time_numeric > float(lower)
                if val_max_time is not None:
                    val_mask &= time_numeric <= float(val_max_time)

            overlap = train_mask & val_mask
            if overlap.any():
                raise ValueError("train/val 时间切分发生重叠，请检查 train_max_time 与 val_min_time/val_values")

            split_info.update(
                {
                    "split_column": raw_time_col,
                    "time_numeric_col": time_numeric_col,
                    "train_max_time": None if train_max_time is None else float(train_max_time),
                    "val_min_time": None if val_min_time is None else float(val_min_time),
                    "val_max_time": None if val_max_time is None else float(val_max_time),
                    "n_unparsed_time_cells": int((~finite_mask).sum()),
                    "train_cells": int(train_mask.sum()),
                    "val_cells": int(val_mask.sum()),
                    "excluded_cells": int((~train_mask & ~val_mask).sum()),
                }
            )
        else:
            raise ValueError(f"unsupported split_strategy={split_strategy}")

        train_X = X[train_mask]
        val_X = X[val_mask]
        train_X, val_X = _subsample_split_matrices(
            train_X,
            val_X,
            max_cells=max_cells,
            cell_seed=cell_seed,
        )

        meta = {
            **base_meta,
            "split_strategy": split_strategy,
            "split_info": split_info,
            "split_info_effective": {
                **split_info,
                "train_cells": int(train_X.shape[0]),
                "val_cells": int(val_X.shape[0]),
                "val_ratio_actual": float(val_X.shape[0] / max(train_X.shape[0] + val_X.shape[0], 1)),
            },
            "train_split_values": train_values_meta,
            "val_split_values": val_values_meta,
        }
        if split_strategy == "donor":
            meta["train_donors"] = train_values_meta
            meta["val_donors"] = val_values_meta
        if return_group_ids and group_values is not None:
            meta["train_obs_group_ids"] = np.asarray(group_values[train_mask])
            meta["val_obs_group_ids"] = np.asarray(group_values[val_mask])

        return (train_X, gene_ids), (val_X, gene_ids), ensembl_ids, meta


def load_sea_ad(
    path_or_key: str = "mtg_toy",
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
):
    return load_h5ad(
        path_or_key=path_or_key,
        backed=backed,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )


def load_sea_ad_split(
    path_or_key: str = "mtg_full",
    donor_csv: str = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
    return_donor_ids: bool = False,
):
    if donor_csv is None:
        project_dir = Path(__file__).parent
        candidates = sorted(project_dir.glob("donor_cell_counts*.csv"))
        if not candidates:
            raise FileNotFoundError("未找到 donor_cell_counts CSV 文件")
        donor_csv = str(candidates[0])

    train_data, val_data, ensembl_ids, meta = load_h5ad_split(
        path_or_key=path_or_key,
        split_strategy="donor",
        split_column="donor_id",
        donor_csv=donor_csv,
        val_ratio=val_ratio,
        seed=seed,
        backed=backed,
        max_cells=max_cells,
        cell_seed=cell_seed,
        return_group_ids=return_donor_ids,
    )
    if return_donor_ids:
        meta["train_obs_donors"] = meta.pop("train_obs_group_ids")
        meta["val_obs_donors"] = meta.pop("val_obs_group_ids")
    return train_data, val_data, ensembl_ids, meta


def _ensure_npz_path(npz_path: str = None) -> str:
    resolved = str(npz_path or DEFAULT_UCE_NPZ_PATH)
    if not Path(resolved).exists():
        raise FileNotFoundError(f"UCE npz 文件不存在: {resolved}")
    return resolved


def load_uce_npz(npz_path: str = None) -> dict:
    resolved = _ensure_npz_path(npz_path)
    data = np.load(resolved, allow_pickle=True)
    payload = {
        "path": resolved,
        "obs_names": np.asarray(data["obs_names"]).astype(str),
        "z": np.asarray(data["z"], dtype=np.float32),
    }
    if "time_raw" in data.files:
        payload["time_raw"] = np.asarray(data["time_raw"]).astype(str)
    if "time_numeric" in data.files:
        payload["time_numeric"] = np.asarray(data["time_numeric"], dtype=np.float32)
    if "cell_type" in data.files:
        payload["cell_type"] = np.asarray(data["cell_type"]).astype(str)
    return payload


def _sample_rows(count_matrix, keep_idx: np.ndarray):
    if keep_idx is None:
        return count_matrix
    if sp.issparse(count_matrix):
        return count_matrix[keep_idx]
    return np.asarray(count_matrix)[keep_idx]


def load_h5ad_with_uce(
    path_or_key: str,
    uce_npz_path: str = None,
    time_col: str = "author_day",
    time_numeric_col: str = None,
    cell_type_col: str = "author_cell_type",
    backed: bool = False,
    max_cells: int = None,
    cell_seed: int = 42,
):
    count_matrix, gene_ids, ensembl_ids, meta = load_h5ad(
        path_or_key=path_or_key,
        backed=backed,
        max_cells=None,
        cell_seed=cell_seed,
    )
    obs = meta["obs"]
    if cell_type_col not in obs.columns:
        raise ValueError(f"缺少 cell type 列: {cell_type_col}")

    uce = load_uce_npz(uce_npz_path)
    h5ad_obs_names = np.asarray(obs.index).astype(str)
    uce_index = {name: idx for idx, name in enumerate(uce["obs_names"])}

    keep_h5ad_idx = []
    keep_uce_idx = []
    for i, name in enumerate(h5ad_obs_names):
        j = uce_index.get(name)
        if j is None:
            continue
        keep_h5ad_idx.append(i)
        keep_uce_idx.append(j)

    if not keep_h5ad_idx:
        raise ValueError("h5ad 与 UCE npz 没有可对齐的 obs_names")

    keep_h5ad_idx = np.asarray(keep_h5ad_idx, dtype=np.int64)
    keep_uce_idx = np.asarray(keep_uce_idx, dtype=np.int64)

    if max_cells is not None and len(keep_h5ad_idx) > int(max_cells):
        rng = np.random.RandomState(cell_seed)
        chosen = np.sort(rng.choice(len(keep_h5ad_idx), size=int(max_cells), replace=False))
        keep_h5ad_idx = keep_h5ad_idx[chosen]
        keep_uce_idx = keep_uce_idx[chosen]

    aligned_counts = _sample_rows(count_matrix, keep_h5ad_idx)
    aligned_obs = obs.iloc[keep_h5ad_idx].copy()
    aligned_obs_names = np.asarray(aligned_obs.index).astype(str)
    aligned_z = uce["z"][keep_uce_idx].astype(np.float32, copy=False)
    aligned_time_raw = np.asarray(aligned_obs[time_col].astype(str).values)
    aligned_time_numeric = _extract_time_numeric(
        aligned_obs,
        time_col=time_col,
        time_numeric_col=time_numeric_col,
    ).astype(np.float32)
    aligned_cell_type = np.asarray(aligned_obs[cell_type_col].astype(str).values)

    cell_type_names = sorted(np.unique(aligned_cell_type).tolist())
    cell_type_to_id = {name: idx for idx, name in enumerate(cell_type_names)}
    aligned_cell_type_id = np.asarray([cell_type_to_id[name] for name in aligned_cell_type], dtype=np.int64)

    aligned_meta = {
        **meta,
        "obs": aligned_obs,
        "obs_names": aligned_obs_names,
        "time_col": time_col,
        "time_numeric_col": time_numeric_col,
        "cell_type_col": cell_type_col,
        "time_raw": aligned_time_raw,
        "time_numeric": aligned_time_numeric,
        "cell_type": aligned_cell_type,
        "cell_type_id": aligned_cell_type_id,
        "cell_type_names": np.asarray(cell_type_names, dtype=object),
        "cell_type_to_id": cell_type_to_id,
        "z": aligned_z,
        "uce_npz_path": uce["path"],
        "aligned_h5ad_indices": keep_h5ad_idx,
        "aligned_uce_indices": keep_uce_idx,
        "alignment_digest": hashlib.md5(keep_uce_idx.tobytes()).hexdigest()[:12],
        "n_cells_aligned": int(len(keep_h5ad_idx)),
    }
    return aligned_counts, gene_ids, ensembl_ids, aligned_meta


def _time_window_mask(
    time_numeric: np.ndarray,
    lower: float = None,
    upper: float = None,
    values: Sequence[str] = None,
    raw_values: np.ndarray = None,
) -> np.ndarray:
    mask = np.isfinite(time_numeric)
    if values is not None:
        if raw_values is None:
            raise ValueError("按离散时间取值筛选时需要提供 raw_values")
        value_set = {str(v) for v in values}
        mask &= np.isin(raw_values, list(value_set))
        return mask
    if lower is not None:
        mask &= time_numeric >= float(lower)
    if upper is not None:
        mask &= time_numeric <= float(upper)
    return mask


def _sorted_unique_time_bins(time_raw: np.ndarray, time_numeric: np.ndarray) -> List[Tuple[str, float]]:
    uniq = np.unique(time_raw)
    bins = []
    for value in uniq:
        mask = time_raw == value
        bins.append((str(value), float(np.mean(time_numeric[mask]))))
    bins.sort(key=lambda x: (x[1], x[0]))
    return bins


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    np.exp(x, out=x)
    denom = np.maximum(np.sum(x, axis=1, keepdims=True), 1e-12)
    return x / denom


def _compute_topk_cosine_pairs(
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    k: int,
    chunk_size: int = 512,
):
    if z_src.ndim != 2 or z_tgt.ndim != 2:
        raise ValueError("z_src/z_tgt 必须是二维矩阵")
    if z_src.shape[1] != z_tgt.shape[1]:
        raise ValueError("z_src/z_tgt 的特征维度不一致")
    if z_tgt.shape[0] <= 0:
        raise ValueError("tgt 候选池为空")

    k_eff = max(1, min(int(k), int(z_tgt.shape[0])))
    z_src_n = _l2_normalize_rows(np.asarray(z_src, dtype=np.float32))
    z_tgt_n = _l2_normalize_rows(np.asarray(z_tgt, dtype=np.float32))

    all_idx = []
    all_sim = []
    for start in range(0, z_src_n.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), z_src_n.shape[0])
        sims = z_src_n[start:end] @ z_tgt_n.T
        if k_eff < sims.shape[1]:
            idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
            sim = np.take_along_axis(sims, idx, axis=1)
        else:
            idx = np.tile(np.arange(sims.shape[1], dtype=np.int64), (sims.shape[0], 1))
            sim = sims
        order = np.argsort(-sim, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        sim = np.take_along_axis(sim, order, axis=1)
        all_idx.append(idx.astype(np.int64, copy=False))
        all_sim.append(sim.astype(np.float32, copy=False))

    return np.concatenate(all_idx, axis=0), np.concatenate(all_sim, axis=0)


def _default_pair_cache_path(
    dataset_path: str,
    uce_npz_path: str,
    alignment_digest: str,
    split_name: str,
    time_col: str,
    cell_type_col: str,
    src_min_time: float,
    src_max_time: float,
    tgt_min_time: float,
    tgt_max_time: float,
    k: int,
    future_only: bool,
) -> str:
    payload = {
        "dataset_path": str(dataset_path),
        "uce_npz_path": str(uce_npz_path),
        "alignment_digest": str(alignment_digest),
        "split_name": str(split_name),
        "time_col": str(time_col),
        "cell_type_col": str(cell_type_col),
        "src_min_time": src_min_time,
        "src_max_time": src_max_time,
        "tgt_min_time": tgt_min_time,
        "tgt_max_time": tgt_max_time,
        "k": int(k),
        "future_only": bool(future_only),
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    stem = f"pairs_{split_name}_{digest}.npz"
    return str(PAIR_CACHE_DIR / stem)


def build_temporal_knn_pairs(
    *,
    meta: dict,
    split_name: str,
    src_min_time: float = None,
    src_max_time: float = None,
    tgt_min_time: float = None,
    tgt_max_time: float = None,
    src_values: Sequence[str] = None,
    tgt_values: Sequence[str] = None,
    future_only: bool = True,
    k: int = 5,
    similarity_chunk_size: int = 512,
    cache_path: str = None,
    force_rebuild: bool = False,
):
    if "z" not in meta:
        raise ValueError("meta 中缺少 UCE embedding `z`，请先使用 load_h5ad_with_uce()")

    resolved_cache = cache_path or _default_pair_cache_path(
        dataset_path=meta["source_path"],
        uce_npz_path=meta["uce_npz_path"],
        alignment_digest=meta.get("alignment_digest", "full"),
        split_name=split_name,
        time_col=meta["time_col"],
        cell_type_col=meta["cell_type_col"],
        src_min_time=src_min_time,
        src_max_time=src_max_time,
        tgt_min_time=tgt_min_time,
        tgt_max_time=tgt_max_time,
        k=k,
        future_only=future_only,
    )
    cache_file = Path(resolved_cache)
    if cache_file.exists() and not force_rebuild:
        loaded = np.load(cache_file, allow_pickle=True)
        return {
            "src_index": loaded["src_index"].astype(np.int64),
            "tgt_index": loaded["tgt_index"].astype(np.int64),
            "src_time": loaded["src_time"].astype(np.float32),
            "tgt_time": loaded["tgt_time"].astype(np.float32),
            "cell_type_id": loaded["cell_type_id"].astype(np.int64),
            "pair_weight": loaded["pair_weight"].astype(np.float32),
            "src_time_raw": loaded["src_time_raw"].astype(str),
            "tgt_time_raw": loaded["tgt_time_raw"].astype(str),
            "cache_path": str(cache_file),
            "split_name": str(loaded["split_name"][0]),
            "n_pairs": int(len(loaded["src_index"])),
        }

    z = np.asarray(meta["z"], dtype=np.float32)
    time_numeric = np.asarray(meta["time_numeric"], dtype=np.float32)
    time_raw = np.asarray(meta["time_raw"]).astype(str)
    cell_type = np.asarray(meta["cell_type"]).astype(str)
    cell_type_id = np.asarray(meta["cell_type_id"], dtype=np.int64)

    src_mask_base = _time_window_mask(
        time_numeric,
        lower=src_min_time,
        upper=src_max_time,
        values=src_values,
        raw_values=time_raw,
    )
    tgt_mask_base = _time_window_mask(
        time_numeric,
        lower=tgt_min_time,
        upper=tgt_max_time,
        values=tgt_values,
        raw_values=time_raw,
    )

    if not src_mask_base.any():
        raise ValueError(f"{split_name}: 没有满足 src 时间范围的细胞")
    if not tgt_mask_base.any():
        raise ValueError(f"{split_name}: 没有满足 tgt 时间范围的细胞")

    src_indices_all = []
    tgt_indices_all = []
    src_times_all = []
    tgt_times_all = []
    cell_type_ids_all = []
    pair_weights_all = []
    src_time_raw_all = []
    tgt_time_raw_all = []

    for time_bin_raw, time_bin_numeric in _sorted_unique_time_bins(time_raw, time_numeric):
        src_time_mask = src_mask_base & (time_raw == time_bin_raw)
        if not src_time_mask.any():
            continue

        for ct_name in np.unique(cell_type[src_time_mask]):
            src_idx = np.where(src_time_mask & (cell_type == ct_name))[0]
            if src_idx.size == 0:
                continue

            tgt_mask = tgt_mask_base & (cell_type == ct_name)
            if future_only:
                tgt_mask &= time_numeric > float(time_bin_numeric)
            tgt_idx = np.where(tgt_mask)[0]
            if tgt_idx.size == 0:
                continue

            nn_local_idx, nn_sim = _compute_topk_cosine_pairs(
                z[src_idx],
                z[tgt_idx],
                k=k,
                chunk_size=similarity_chunk_size,
            )
            nn_global_idx = tgt_idx[nn_local_idx]
            nn_weights = _softmax_rows(nn_sim.astype(np.float32, copy=True))

            repeat_src = np.repeat(src_idx[:, None], nn_global_idx.shape[1], axis=1)
            repeat_src_time = np.repeat(time_numeric[src_idx][:, None], nn_global_idx.shape[1], axis=1)
            repeat_src_time_raw = np.repeat(time_raw[src_idx][:, None], nn_global_idx.shape[1], axis=1)
            repeat_ct = np.repeat(cell_type_id[src_idx][:, None], nn_global_idx.shape[1], axis=1)

            src_indices_all.append(repeat_src.reshape(-1))
            tgt_indices_all.append(nn_global_idx.reshape(-1))
            src_times_all.append(repeat_src_time.reshape(-1))
            tgt_times_all.append(time_numeric[nn_global_idx].reshape(-1))
            cell_type_ids_all.append(repeat_ct.reshape(-1))
            pair_weights_all.append(nn_weights.reshape(-1))
            src_time_raw_all.append(repeat_src_time_raw.reshape(-1))
            tgt_time_raw_all.append(time_raw[nn_global_idx].reshape(-1))

    if not src_indices_all:
        raise ValueError(f"{split_name}: 没有构造出任何 temporal k-NN 配对")

    pair_payload = {
        "src_index": np.concatenate(src_indices_all).astype(np.int64, copy=False),
        "tgt_index": np.concatenate(tgt_indices_all).astype(np.int64, copy=False),
        "src_time": np.concatenate(src_times_all).astype(np.float32, copy=False),
        "tgt_time": np.concatenate(tgt_times_all).astype(np.float32, copy=False),
        "cell_type_id": np.concatenate(cell_type_ids_all).astype(np.int64, copy=False),
        "pair_weight": np.concatenate(pair_weights_all).astype(np.float32, copy=False),
        "src_time_raw": np.concatenate(src_time_raw_all).astype(str),
        "tgt_time_raw": np.concatenate(tgt_time_raw_all).astype(str),
        "split_name": np.array([split_name], dtype=object),
        "cache_path": str(cache_file),
        "n_pairs": int(sum(len(x) for x in src_indices_all)),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **{k: v for k, v in pair_payload.items() if k not in {"cache_path", "n_pairs"}})
    return pair_payload


class _TopKTokenizerMixin:
    def __init__(
        self,
        count_matrix,
        gene_ids,
        top_k: int = 2048,
    ):
        self.sparse = sp.issparse(count_matrix)
        self.counts = count_matrix.astype(np.float32) if self.sparse else np.asarray(count_matrix, dtype=np.float32)
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.top_k = int(top_k)
        self.n_genes = int(len(self.gene_ids))
        self.pad_token_id = self.n_genes
        self.cls_token_id = self.n_genes + 1
        self.gene_id_to_var_idx = np.empty((self.n_genes,), dtype=np.int64)
        self.gene_id_to_var_idx[self.gene_ids] = np.arange(self.n_genes, dtype=np.int64)

    def __len__(self):
        return self.counts.shape[0]

    def _get_row(self, idx: int) -> np.ndarray:
        if self.sparse:
            return self.counts.getrow(idx).toarray().ravel().astype(np.float32, copy=False)
        return self.counts[idx].astype(np.float32, copy=False)

    def _tokenize_cell(self, idx: int):
        x_raw = self._get_row(idx)
        x_log = np.log1p(x_raw)

        nonzero_idx = np.flatnonzero(x_log > 0)
        chosen = np.array([], dtype=np.int64)
        if nonzero_idx.size > 0:
            expr_nz = x_log[nonzero_idx]
            gene_nz = self.gene_ids[nonzero_idx]
            order = np.lexsort((gene_nz, -expr_nz))
            chosen = nonzero_idx[order[: self.top_k]]

        seq_len = self.top_k + 1
        subset_gene_ids = np.full((seq_len,), self.pad_token_id, dtype=np.int64)
        subset_values = np.zeros((seq_len,), dtype=np.float32)
        valid_mask = np.zeros((seq_len,), dtype=np.bool_)

        subset_gene_ids[0] = self.cls_token_id
        subset_values[0] = 0.0
        valid_mask[0] = True

        n_valid = min(len(chosen), self.top_k)
        if n_valid > 0:
            subset_gene_ids[1 : 1 + n_valid] = self.gene_ids[chosen[:n_valid]]
            subset_values[1 : 1 + n_valid] = x_log[chosen[:n_valid]]
            valid_mask[1 : 1 + n_valid] = True

        return (
            torch.from_numpy(subset_gene_ids).long(),
            torch.from_numpy(subset_values).float(),
            torch.from_numpy(valid_mask),
        )

    def _tokenize_cell_with_var_idx(self, idx: int):
        x_raw = self._get_row(idx)
        x_log = np.log1p(x_raw)

        nonzero_idx = np.flatnonzero(x_log > 0)
        chosen = np.array([], dtype=np.int64)
        if nonzero_idx.size > 0:
            expr_nz = x_log[nonzero_idx]
            gene_nz = self.gene_ids[nonzero_idx]
            order = np.lexsort((gene_nz, -expr_nz))
            chosen = nonzero_idx[order[: self.top_k]]

        seq_len = self.top_k + 1
        subset_gene_ids = np.full((seq_len,), self.pad_token_id, dtype=np.int64)
        subset_values = np.zeros((seq_len,), dtype=np.float32)
        valid_mask = np.zeros((seq_len,), dtype=np.bool_)
        subset_var_idx = np.full((seq_len,), -1, dtype=np.int64)

        subset_gene_ids[0] = self.cls_token_id
        subset_values[0] = 0.0
        valid_mask[0] = True

        n_valid = min(len(chosen), self.top_k)
        if n_valid > 0:
            subset_gene_ids[1 : 1 + n_valid] = self.gene_ids[chosen[:n_valid]]
            subset_values[1 : 1 + n_valid] = x_log[chosen[:n_valid]]
            valid_mask[1 : 1 + n_valid] = True
            subset_var_idx[1 : 1 + n_valid] = chosen[:n_valid]

        return subset_gene_ids, subset_values, valid_mask, subset_var_idx


def compute_log1p_nonzero_prior_stats(count_matrix):
    if sp.issparse(count_matrix):
        nz_raw = count_matrix.data.astype(np.float32, copy=False)
    else:
        arr = np.asarray(count_matrix, dtype=np.float32)
        nz_raw = arr[arr > 0]

    if nz_raw.size == 0:
        return 0.0, 1.0

    nz_log = np.log1p(nz_raw)
    mean = float(nz_log.mean())
    std = float(max(nz_log.std(), 1e-6))
    return mean, std


def compute_log1p_nonzero_bin_edges(count_matrix, n_bins: int = 51):
    n_bins = int(n_bins)
    if n_bins <= 1:
        return np.zeros((0,), dtype=np.float32)

    if sp.issparse(count_matrix):
        nz_raw = count_matrix.data.astype(np.float32, copy=False)
    else:
        arr = np.asarray(count_matrix, dtype=np.float32)
        nz_raw = arr[arr > 0]

    if nz_raw.size == 0:
        return np.linspace(0.0, 1.0, n_bins - 1, dtype=np.float32)

    nz_log = np.log1p(nz_raw)
    quantiles = np.linspace(0.0, 1.0, n_bins - 1)
    edges = np.quantile(nz_log, quantiles).astype(np.float32)
    edges = np.maximum.accumulate(edges)
    return edges


def _load_preprocessed_temporal_bundle(preprocessed_dir: str):
    pre_dir = Path(preprocessed_dir)
    if not pre_dir.exists():
        raise FileNotFoundError(f"预处理目录不存在: {pre_dir}")

    metadata_path = pre_dir / "dataset_metadata.json"
    metadata_df_path = pre_dir / "cell_metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"缺少 dataset_metadata.json: {metadata_path}")
    if not metadata_df_path.exists():
        raise FileNotFoundError(f"缺少 cell_metadata.pkl: {metadata_df_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        dataset_meta = json.load(f)

    artifacts = dataset_meta.get("artifacts", {})
    expr_path = Path(artifacts.get("expression_memmap", pre_dir / "expression_log1p_clipped.npy"))
    obs_names_path = Path(artifacts.get("obs_names", pre_dir / "obs_names.npy"))
    hvg_gene_indices_path = Path(artifacts.get("hvg_gene_indices", pre_dir / "hvg_gene_indices.npy"))
    hvg_var_names_path = Path(artifacts.get("hvg_var_names", pre_dir / "hvg_var_names.npy"))
    hvg_gene_names_path = Path(artifacts.get("hvg_gene_names", pre_dir / "hvg_gene_names.npy"))

    expression = np.load(expr_path, mmap_mode="r")
    metadata_df = pd.read_pickle(metadata_df_path)
    obs_names = np.load(obs_names_path, allow_pickle=True).astype(str)
    hvg_gene_indices = np.load(hvg_gene_indices_path, allow_pickle=True).astype(np.int64)
    hvg_var_names = np.load(hvg_var_names_path, allow_pickle=True).astype(str) if hvg_var_names_path.exists() else None
    hvg_gene_names = np.load(hvg_gene_names_path, allow_pickle=True).astype(str) if hvg_gene_names_path.exists() else None

    if metadata_df.shape[0] != expression.shape[0]:
        raise ValueError("预处理 metadata 行数与表达矩阵行数不一致")
    if obs_names.shape[0] != expression.shape[0]:
        raise ValueError("obs_names 长度与表达矩阵行数不一致")
    metadata_index = metadata_df.index.to_numpy().astype(str)
    if not np.array_equal(metadata_index, obs_names):
        raise ValueError("cell_metadata.pkl 的 index 与 obs_names.npy 不一致")

    time_values = metadata_df["author_day_numeric"].to_numpy(dtype=np.float32)
    finite_time = time_values[np.isfinite(time_values)]
    if finite_time.size == 0:
        raise ValueError("预处理 metadata 中没有有效的 author_day_numeric")
    time_norm_denom = float(dataset_meta.get("time_norm_denom", np.max(finite_time)))
    time_norm_denom = max(time_norm_denom, 1e-6)

    total_gene_space = int(dataset_meta.get("original_n_vars", np.max(hvg_gene_indices) + 1))
    bundle_meta = {
        **dataset_meta,
        "source_path": dataset_meta.get("input_h5ad", str(pre_dir)),
        "preprocessed_dir": str(pre_dir),
        "obs_names": obs_names,
        "hvg_gene_indices": hvg_gene_indices,
        "hvg_var_names": hvg_var_names,
        "hvg_gene_names": hvg_gene_names,
        "time_norm_denom": time_norm_denom,
        "n_genes": int(total_gene_space),
        "fixed_panel_size": int(hvg_gene_indices.shape[0]),
    }
    return expression, metadata_df, bundle_meta


def _load_pair_index_from_npz(path: str):
    pair_path = Path(path)
    if not pair_path.exists():
        raise FileNotFoundError(f"pair table 不存在: {pair_path}")
    loaded = np.load(pair_path, allow_pickle=True)
    pair_index = {}
    for key in loaded.files:
        value = loaded[key]
        if value.dtype.kind in {"U", "S", "O"}:
            pair_index[key] = value.astype(str)
        else:
            pair_index[key] = value
    pair_index["cache_path"] = str(pair_path)
    pair_index["n_pairs"] = int(len(pair_index["src_index"]))
    return pair_index


def _filter_pair_index(
    pair_index: dict,
    *,
    src_min_time: float = None,
    src_max_time: float = None,
    tgt_min_time: float = None,
    tgt_max_time: float = None,
    src_values: Sequence[str] = None,
    tgt_values: Sequence[str] = None,
):
    src_time = np.asarray(pair_index["src_time"], dtype=np.float32)
    tgt_time = np.asarray(pair_index["tgt_time"], dtype=np.float32)
    src_time_raw = np.asarray(pair_index["src_time_raw"]).astype(str)
    tgt_time_raw = np.asarray(pair_index["tgt_time_raw"]).astype(str)

    mask = _time_window_mask(
        src_time,
        lower=src_min_time,
        upper=src_max_time,
        values=src_values,
        raw_values=src_time_raw,
    )
    mask &= _time_window_mask(
        tgt_time,
        lower=tgt_min_time,
        upper=tgt_max_time,
        values=tgt_values,
        raw_values=tgt_time_raw,
    )

    filtered = {}
    for key, value in pair_index.items():
        if isinstance(value, np.ndarray) and value.shape[0] == mask.shape[0]:
            filtered[key] = value[mask]
        else:
            filtered[key] = value
    filtered["n_pairs"] = int(mask.sum())
    return filtered


class TopKExpressionDataset(_TopKTokenizerMixin, Dataset):
    def __init__(
        self,
        count_matrix,
        gene_ids,
        top_k: int = 2048,
        return_index: bool = False,
    ):
        super().__init__(
            count_matrix=count_matrix,
            gene_ids=gene_ids,
            top_k=top_k,
        )
        self.return_index = bool(return_index)

    def __getitem__(self, idx):
        item = self._tokenize_cell(int(idx))
        if self.return_index:
            return item + (torch.tensor(idx, dtype=torch.long),)
        return item


class TemporalPairDataset(_TopKTokenizerMixin, Dataset):
    def __init__(
        self,
        count_matrix,
        gene_ids,
        pair_index: dict,
        top_k: int = 2048,
        return_index: bool = False,
    ):
        super().__init__(
            count_matrix=count_matrix,
            gene_ids=gene_ids,
            top_k=top_k,
        )
        self.return_index = bool(return_index)
        self.src_index = np.asarray(pair_index["src_index"], dtype=np.int64)
        self.tgt_index = np.asarray(pair_index["tgt_index"], dtype=np.int64)
        self.src_time = np.asarray(pair_index["src_time"], dtype=np.float32)
        self.tgt_time = np.asarray(pair_index["tgt_time"], dtype=np.float32)
        self.cell_type_id = np.asarray(pair_index["cell_type_id"], dtype=np.int64)
        self.pair_weight = np.asarray(pair_index["pair_weight"], dtype=np.float32)
        self.src_time_raw = np.asarray(pair_index["src_time_raw"]).astype(str)
        self.tgt_time_raw = np.asarray(pair_index["tgt_time_raw"]).astype(str)
        self.group_labels = self.tgt_time_raw

        n_pairs = len(self.src_index)
        expected = {
            "tgt_index": len(self.tgt_index),
            "src_time": len(self.src_time),
            "tgt_time": len(self.tgt_time),
            "cell_type_id": len(self.cell_type_id),
            "pair_weight": len(self.pair_weight),
        }
        for key, length in expected.items():
            if length != n_pairs:
                raise ValueError(f"pair_index[{key}] 长度与 src_index 不一致")

    def __len__(self):
        return len(self.src_index)

    def __getitem__(self, idx):
        idx = int(idx)
        src_idx = int(self.src_index[idx])
        tgt_idx = int(self.tgt_index[idx])
        tgt_gene_ids_np, tgt_values_np, valid_mask_np, tgt_var_idx_np = self._tokenize_cell_with_var_idx(tgt_idx)
        src_raw = self._get_row(src_idx)
        src_log = np.log1p(src_raw)
        src_values_np = np.zeros_like(tgt_values_np, dtype=np.float32)
        src_values_np[0] = 0.0
        valid_gene_positions = tgt_var_idx_np >= 0
        if valid_gene_positions.any():
            src_values_np[valid_gene_positions] = src_log[tgt_var_idx_np[valid_gene_positions]]

        item = (
            torch.from_numpy(tgt_gene_ids_np).long(),
            torch.from_numpy(src_values_np).float(),
            torch.from_numpy(tgt_values_np).float(),
            torch.from_numpy(valid_mask_np),
            torch.tensor(self.src_time[idx], dtype=torch.float32),
            torch.tensor(self.tgt_time[idx], dtype=torch.float32),
            torch.tensor(self.cell_type_id[idx], dtype=torch.long),
            torch.tensor(self.pair_weight[idx], dtype=torch.float32),
        )
        if self.return_index:
            item = item + (
                torch.tensor(src_idx, dtype=torch.long),
                torch.tensor(tgt_idx, dtype=torch.long),
            )
        return item


class FixedHVGTemporalPairDataset(Dataset):
    def __init__(
        self,
        expression_matrix,
        metadata_df: pd.DataFrame,
        hvg_gene_ids,
        pair_index: dict,
        time_norm_denom: float,
        return_index: bool = False,
    ):
        self.expression = expression_matrix
        self.metadata = metadata_df
        self.return_index = bool(return_index)

        self.hvg_gene_ids = np.asarray(hvg_gene_ids, dtype=np.int64)
        if self.hvg_gene_ids.ndim != 1:
            raise ValueError("hvg_gene_ids 必须是一维数组")
        self.n_panel_genes = int(self.hvg_gene_ids.shape[0])
        self.n_total_genes = int(max(np.max(self.hvg_gene_ids) + 1, self.n_panel_genes))
        self.pad_token_id = self.n_total_genes
        self.cls_token_id = self.n_total_genes + 1
        self.time_norm_denom = max(float(time_norm_denom), 1e-6)

        self.src_index = np.asarray(pair_index["src_index"], dtype=np.int64)
        self.tgt_index = np.asarray(pair_index["tgt_index"], dtype=np.int64)
        self.pair_weight = np.asarray(pair_index["pair_weight"], dtype=np.float32)
        self.cell_type_id = np.asarray(pair_index["cell_type_id"], dtype=np.int64)
        self.src_time_raw = np.asarray(pair_index["src_time_raw"]).astype(str)
        self.tgt_time_raw = np.asarray(pair_index["tgt_time_raw"]).astype(str)
        self.group_labels = self.tgt_time_raw

        n_pairs = len(self.src_index)
        for key, values in {
            "tgt_index": self.tgt_index,
            "pair_weight": self.pair_weight,
            "cell_type_id": self.cell_type_id,
            "src_time_raw": self.src_time_raw,
            "tgt_time_raw": self.tgt_time_raw,
        }.items():
            if len(values) != n_pairs:
                raise ValueError(f"pair_index[{key}] 长度与 src_index 不一致")

        if self.expression.shape[0] != self.metadata.shape[0]:
            raise ValueError("固定 HVG 表达矩阵与 metadata 行数不一致")

        self.time_numeric = self.metadata["author_day_numeric"].to_numpy(dtype=np.float32)
        self.valid_mask_np = np.ones((self.n_panel_genes + 1,), dtype=np.bool_)
        self.gene_ids_np = np.concatenate(
            [
                np.asarray([self.cls_token_id], dtype=np.int64),
                self.hvg_gene_ids.astype(np.int64, copy=False),
            ]
        )

    def __len__(self):
        return len(self.src_index)

    def __getitem__(self, idx):
        idx = int(idx)
        src_idx = int(self.src_index[idx])
        tgt_idx = int(self.tgt_index[idx])

        src_values = np.asarray(self.expression[src_idx], dtype=np.float32)
        tgt_values = np.asarray(self.expression[tgt_idx], dtype=np.float32)
        src_seq = np.empty((self.n_panel_genes + 1,), dtype=np.float32)
        tgt_seq = np.empty((self.n_panel_genes + 1,), dtype=np.float32)
        src_seq[0] = 0.0
        tgt_seq[0] = 0.0
        src_seq[1:] = src_values
        tgt_seq[1:] = tgt_values

        t_src = float(self.time_numeric[src_idx] / self.time_norm_denom)
        t_tgt = float(self.time_numeric[tgt_idx] / self.time_norm_denom)

        item = (
            torch.from_numpy(self.gene_ids_np).long(),
            torch.from_numpy(src_seq).float(),
            torch.from_numpy(tgt_seq).float(),
            torch.from_numpy(self.valid_mask_np),
            torch.tensor(t_src, dtype=torch.float32),
            torch.tensor(t_tgt, dtype=torch.float32),
            torch.tensor(self.cell_type_id[idx], dtype=torch.long),
            torch.tensor(self.pair_weight[idx], dtype=torch.float32),
        )
        if self.return_index:
            item = item + (
                torch.tensor(src_idx, dtype=torch.long),
                torch.tensor(tgt_idx, dtype=torch.long),
            )
        return item


def load_mouse_development_temporal_datasets(
    path_or_key: str,
    *,
    input_mode: str = "dynamic_topk",
    preprocessed_dir: str = None,
    pair_table_path: str = None,
    uce_npz_path: str = None,
    time_col: str = "author_day",
    time_numeric_col: str = None,
    cell_type_col: str = "author_cell_type",
    top_k: int = 2048,
    pairing_k: int = 5,
    future_only: bool = True,
    similarity_chunk_size: int = 512,
    train_src_min_time: float = 10.0,
    train_src_max_time: float = 14.0,
    train_tgt_min_time: float = 10.0,
    train_tgt_max_time: float = 14.0,
    val_src_min_time: float = 12.0,
    val_src_max_time: float = 14.0,
    val_tgt_min_time: float = 14.0,
    val_tgt_max_time: float = 19.0,
    max_cells: int = None,
    cell_seed: int = 42,
    train_cache_path: str = None,
    val_cache_path: str = None,
    force_rebuild_pairs: bool = False,
):
    input_mode = str(input_mode or "dynamic_topk").strip().lower()
    if input_mode in {"fixed_hvg", "preprocessed_hvg", "fixed_panel"}:
        if preprocessed_dir is None:
            raise ValueError("固定 HVG 模式需要提供 preprocessed_dir")
        if pair_table_path is None:
            raise ValueError("固定 HVG 模式需要提供 pair_table_path")

        expression_matrix, metadata_df, pre_meta = _load_preprocessed_temporal_bundle(preprocessed_dir)
        pair_table = _load_pair_index_from_npz(pair_table_path)
        train_pairs = _filter_pair_index(
            pair_table,
            src_min_time=train_src_min_time,
            src_max_time=train_src_max_time,
            tgt_min_time=train_tgt_min_time,
            tgt_max_time=train_tgt_max_time,
        )
        val_pairs = _filter_pair_index(
            pair_table,
            src_min_time=val_src_min_time,
            src_max_time=val_src_max_time,
            tgt_min_time=val_tgt_min_time,
            tgt_max_time=val_tgt_max_time,
        )
        if train_pairs["n_pairs"] <= 0:
            raise ValueError("固定 HVG 模式下 train split 没有配对样本")
        if val_pairs["n_pairs"] <= 0:
            raise ValueError("固定 HVG 模式下 val split 没有配对样本")

        train_dataset = FixedHVGTemporalPairDataset(
            expression_matrix=expression_matrix,
            metadata_df=metadata_df,
            hvg_gene_ids=pre_meta["hvg_gene_indices"],
            pair_index=train_pairs,
            time_norm_denom=pre_meta["time_norm_denom"],
            return_index=True,
        )
        val_dataset = FixedHVGTemporalPairDataset(
            expression_matrix=expression_matrix,
            metadata_df=metadata_df,
            hvg_gene_ids=pre_meta["hvg_gene_indices"],
            pair_index=val_pairs,
            time_norm_denom=pre_meta["time_norm_denom"],
            return_index=True,
        )

        cell_type_values = metadata_df[cell_type_col].astype(str).to_numpy() if cell_type_col in metadata_df.columns else None
        cell_type_names = sorted(np.unique(cell_type_values[cell_type_values != ""]).tolist()) if cell_type_values is not None else []
        meta = dict(pre_meta)
        meta["cell_type_col"] = cell_type_col
        meta["cell_type_names"] = np.asarray(cell_type_names, dtype=object)
        meta["pairing"] = {
            "strategy": "latent_knn",
            "input_mode": input_mode,
            "k": int(pairing_k),
            "future_only": bool(future_only),
            "pair_table_path": str(pair_table_path),
            "train_pairs": int(train_pairs["n_pairs"]),
            "val_pairs": int(val_pairs["n_pairs"]),
            "train_time_window": {
                "src_min_time": train_src_min_time,
                "src_max_time": train_src_max_time,
                "tgt_min_time": train_tgt_min_time,
                "tgt_max_time": train_tgt_max_time,
            },
            "val_time_window": {
                "src_min_time": val_src_min_time,
                "src_max_time": val_src_max_time,
                "tgt_min_time": val_tgt_min_time,
                "tgt_max_time": val_tgt_max_time,
            },
        }
        return train_dataset, val_dataset, pre_meta.get("hvg_var_names"), meta

    count_matrix, gene_ids, ensembl_ids, meta = load_h5ad_with_uce(
        path_or_key=path_or_key,
        uce_npz_path=uce_npz_path,
        time_col=time_col,
        time_numeric_col=time_numeric_col,
        cell_type_col=cell_type_col,
        max_cells=max_cells,
        cell_seed=cell_seed,
    )

    train_pairs = build_temporal_knn_pairs(
        meta=meta,
        split_name="train",
        src_min_time=train_src_min_time,
        src_max_time=train_src_max_time,
        tgt_min_time=train_tgt_min_time,
        tgt_max_time=train_tgt_max_time,
        future_only=future_only,
        k=pairing_k,
        similarity_chunk_size=similarity_chunk_size,
        cache_path=train_cache_path,
        force_rebuild=force_rebuild_pairs,
    )
    val_pairs = build_temporal_knn_pairs(
        meta=meta,
        split_name="val",
        src_min_time=val_src_min_time,
        src_max_time=val_src_max_time,
        tgt_min_time=val_tgt_min_time,
        tgt_max_time=val_tgt_max_time,
        future_only=future_only,
        k=pairing_k,
        similarity_chunk_size=similarity_chunk_size,
        cache_path=val_cache_path,
        force_rebuild=force_rebuild_pairs,
    )

    train_dataset = TemporalPairDataset(
        count_matrix=count_matrix,
        gene_ids=gene_ids,
        pair_index=train_pairs,
        top_k=top_k,
        return_index=True,
    )
    val_dataset = TemporalPairDataset(
        count_matrix=count_matrix,
        gene_ids=gene_ids,
        pair_index=val_pairs,
        top_k=top_k,
        return_index=True,
    )

    meta = dict(meta)
    meta["pairing"] = {
        "strategy": "latent_knn",
        "input_mode": input_mode,
        "k": int(pairing_k),
        "future_only": bool(future_only),
        "similarity_chunk_size": int(similarity_chunk_size),
        "train_cache_path": train_pairs["cache_path"],
        "val_cache_path": val_pairs["cache_path"],
        "train_pairs": int(train_pairs["n_pairs"]),
        "val_pairs": int(val_pairs["n_pairs"]),
        "train_time_window": {
            "src_min_time": train_src_min_time,
            "src_max_time": train_src_max_time,
            "tgt_min_time": train_tgt_min_time,
            "tgt_max_time": train_tgt_max_time,
        },
        "val_time_window": {
            "src_min_time": val_src_min_time,
            "src_max_time": val_src_max_time,
            "tgt_min_time": val_tgt_min_time,
            "tgt_max_time": val_tgt_max_time,
        },
    }
    return train_dataset, val_dataset, ensembl_ids, meta


class GroupedBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        group_labels: Sequence[str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.group_labels = np.asarray(group_labels).astype(str)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.group_to_indices = {
            group: np.flatnonzero(self.group_labels == group).astype(np.int64)
            for group in np.unique(self.group_labels)
        }

    def __iter__(self):
        rng = np.random.RandomState(self.seed + int(time.time()) % 100000)
        group_items = list(self.group_to_indices.items())
        if self.shuffle:
            rng.shuffle(group_items)
        batches = []
        for _, indices in group_items:
            idx = indices.copy()
            if self.shuffle:
                rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                batch = idx[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.group_to_indices.values():
            n = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n += 1
            total += n
        return total


def make_temporal_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    group_by_tgt_time: bool = True,
    drop_last: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    prefetch_factor: int = None,
    persistent_workers: bool = False,
):
    loader_kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["worker_init_fn"] = _limit_worker_threads

    if group_by_tgt_time and hasattr(dataset, "group_labels"):
        batch_sampler = GroupedBatchSampler(
            dataset.group_labels,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs,
    )


def make_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers: int = 0,
    prefetch_factor: int = None,
    persistent_workers: bool = False,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["worker_init_fn"] = _limit_worker_threads
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)
