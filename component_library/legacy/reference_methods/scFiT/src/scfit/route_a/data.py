import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


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


def load_preprocessed_temporal_bundle(preprocessed_dir: str):
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
    hvg_gene_names = (
        np.load(hvg_gene_names_path, allow_pickle=True).astype(str) if hvg_gene_names_path.exists() else None
    )

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


_load_preprocessed_temporal_bundle = load_preprocessed_temporal_bundle
