#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.lib.format import open_memmap


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a fixed-HVG temporal dataset for scFiT.")
    parser.add_argument(
        "--input-h5ad",
        type=str,
        default="data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad",
        help="Path to source AnnData file.",
    )
    parser.add_argument(
        "--hvg-dir",
        type=str,
        default="results/cache/hvg_cache/neural_crest_pns_glia_rawcounts_seuratv3",
        help="Directory containing ranked HVG artifacts from stage 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/neural_crest_pns_glia_hvg2048",
        help="Directory to store the preprocessed matrix and metadata.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2048,
        help="Number of HVGs to keep.",
    )
    parser.add_argument(
        "--source-layer",
        type=str,
        default="raw.X",
        help="Matrix source to preprocess. Use raw counts layer by default.",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Library-size normalization target.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=0.0,
        help="Minimum value after log1p.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=10.0,
        help="Maximum value after log1p.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Number of cells per preprocessing chunk.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="author_day",
        help="Obs column storing developmental time.",
    )
    return parser.parse_args()


def _to_float_or_nan(value) -> float:
    if value is None:
        return np.nan
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except ValueError:
        match = re.search(r"([0-9]+(?:\\.[0-9]+)?)", s)
        if match:
            return float(match.group(1))
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


def _load_topk_hvg(hvg_dir: Path, k: int):
    ranked_indices_path = hvg_dir / "hvg_ranked_gene_indices.npy"
    ranked_var_names_path = hvg_dir / "hvg_ranked_var_names.txt"
    ranked_gene_names_path = hvg_dir / "hvg_ranked_gene_names.txt"
    if not ranked_indices_path.exists():
        raise FileNotFoundError(f"Missing ranked HVG indices: {ranked_indices_path}")
    ranked_indices = np.load(ranked_indices_path)
    k_eff = min(int(k), int(ranked_indices.shape[0]))
    topk_indices = ranked_indices[:k_eff].astype(np.int64, copy=False)

    ranked_var_names = np.loadtxt(ranked_var_names_path, dtype=str, ndmin=1)
    ranked_gene_names = np.loadtxt(ranked_gene_names_path, dtype=str, ndmin=1)
    topk_var_names = ranked_var_names[:k_eff]
    topk_gene_names = ranked_gene_names[:k_eff]

    np.save(hvg_dir / "hvg_gene_indices.npy", topk_indices)
    with (hvg_dir / "hvg_var_names.txt").open("w", encoding="utf-8") as f:
        for item in topk_var_names:
            f.write(f"{item}\n")
    with (hvg_dir / "hvg_gene_names.txt").open("w", encoding="utf-8") as f:
        for item in topk_gene_names:
            f.write(f"{item}\n")

    return topk_indices, topk_var_names, topk_gene_names


def _get_matrix(adata, source_layer: str):
    if source_layer and source_layer != "X":
        if source_layer not in adata.layers:
            raise KeyError(f"Layer not found: {source_layer}")
        return adata.layers[source_layer], f"layers[{source_layer}]"
    return adata.X, "X"


def _row_normalize_log1p_clip(matrix_chunk, target_sum: float, clip_min: float, clip_max: float):
    if sp.issparse(matrix_chunk):
        dense = matrix_chunk.toarray().astype(np.float32, copy=False)
    else:
        dense = np.asarray(matrix_chunk, dtype=np.float32)
    libsize = dense.sum(axis=1, keepdims=True)
    libsize = np.maximum(libsize, 1e-8)
    dense = dense * (float(target_sum) / libsize)
    np.log1p(dense, out=dense)
    if clip_min is not None or clip_max is not None:
        dense = np.clip(dense, clip_min, clip_max, out=dense)
    return dense.astype(np.float32, copy=False)


def main():
    args = parse_args()
    t0 = time.time()

    hvg_dir = Path(args.hvg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hvg_indices, hvg_var_names, hvg_gene_names = _load_topk_hvg(hvg_dir, args.k)
    print(f"[prep] using top-{len(hvg_indices)} HVGs from {hvg_dir}")

    adata = ad.read_h5ad(args.input_h5ad, backed="r")
    matrix, matrix_name = _get_matrix(adata, args.source_layer)
    n_cells = int(adata.n_obs)
    n_genes = int(len(hvg_indices))
    original_n_vars = int(adata.n_vars)
    print(f"[prep] input={args.input_h5ad} shape=({n_cells}, {adata.n_vars}) source={matrix_name}")

    memmap_path = output_dir / "expression_log1p_clipped.npy"
    expr_memmap = open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(n_cells, n_genes))

    chunk_size = max(int(args.chunk_size), 1)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = matrix[start:end, hvg_indices]
        expr_memmap[start:end] = _row_normalize_log1p_clip(
            chunk,
            target_sum=args.target_sum,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )
        if start == 0 or end == n_cells or (start // chunk_size) % 10 == 0:
            print(f"[prep] processed cells {start}:{end}")

    expr_memmap.flush()

    obs = adata.obs.copy()
    obs_names = np.asarray(adata.obs_names).astype(str)
    metadata = pd.DataFrame(index=obs_names)
    metadata.index.name = "obs_name"
    metadata["author_day"] = obs[args.time_col].astype(str).to_numpy() if args.time_col in obs.columns else ""
    metadata["author_day_numeric"] = np.array([parse_time_value(v) for v in metadata["author_day"].to_numpy()], dtype=np.float32)
    finite_time = metadata["author_day_numeric"].to_numpy(dtype=np.float32)
    finite_time = finite_time[np.isfinite(finite_time)]
    time_norm_denom = float(np.max(finite_time)) if finite_time.size > 0 else 1.0
    for col in ("author_cell_type", "cell_type", "donor_id"):
        metadata[col] = obs[col].astype(str).to_numpy() if col in obs.columns else ""

    metadata_pickle_path = output_dir / "cell_metadata.pkl"
    metadata_csv_path = output_dir / "cell_metadata.csv.gz"
    metadata.to_pickle(metadata_pickle_path)
    metadata.to_csv(metadata_csv_path)

    np.save(output_dir / "obs_names.npy", obs_names)
    np.save(output_dir / "author_day_numeric.npy", metadata["author_day_numeric"].to_numpy(dtype=np.float32))
    np.save(output_dir / "hvg_gene_indices.npy", hvg_indices)
    np.save(output_dir / "hvg_var_names.npy", hvg_var_names)
    np.save(output_dir / "hvg_gene_names.npy", hvg_gene_names)

    dataset_meta = {
        "input_h5ad": str(args.input_h5ad),
        "hvg_dir": str(hvg_dir),
        "output_dir": str(output_dir),
        "matrix_source_used": matrix_name,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "original_n_vars": original_n_vars,
        "target_sum": float(args.target_sum),
        "clip_min": float(args.clip_min),
        "clip_max": float(args.clip_max),
        "chunk_size": chunk_size,
        "time_col": args.time_col,
        "time_norm_denom": time_norm_denom,
        "artifacts": {
            "expression_memmap": str(memmap_path),
            "cell_metadata_pickle": str(metadata_pickle_path),
            "cell_metadata_csv": str(metadata_csv_path),
            "obs_names": str(output_dir / "obs_names.npy"),
            "author_day_numeric": str(output_dir / "author_day_numeric.npy"),
            "hvg_gene_indices": str(output_dir / "hvg_gene_indices.npy"),
            "hvg_var_names": str(output_dir / "hvg_var_names.npy"),
            "hvg_gene_names": str(output_dir / "hvg_gene_names.npy"),
        },
        "elapsed_seconds": float(time.time() - t0),
    }

    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_meta, f, ensure_ascii=False, indent=2)

    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Preprocessed HVG Dataset\n\n"
            f"- input_h5ad: `{args.input_h5ad}`\n"
            f"- source_matrix: `{matrix_name}`\n"
            f"- cells: `{n_cells}`\n"
            f"- fixed_hvg_genes: `{n_genes}`\n"
            f"- normalization: `library-size -> {args.target_sum}, log1p, clip[{args.clip_min}, {args.clip_max}]`\n"
            f"- expression_memmap: `{memmap_path.name}`\n"
            f"- cell_metadata: `{metadata_pickle_path.name}` / `{metadata_csv_path.name}`\n"
        )

    print(f"[prep] wrote dataset to {output_dir}")
    print(f"[prep] finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
