#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_temporal_knn_pairs, load_h5ad_with_uce


def parse_args():
    parser = argparse.ArgumentParser(description="Build offline temporal kNN pair tables for scFiT.")
    parser.add_argument(
        "--source-h5ad",
        type=str,
        default="data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad",
        help="Source h5ad used for UCE alignment in pooled-future mode.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="data/processed/neural_crest_pns_glia_hvg2048",
        help="Directory produced by prepare_temporal_hvg_dataset.py",
    )
    parser.add_argument(
        "--uce-npz",
        type=str,
        default="data/external/jax_uce_flow_data.npz",
        help="Path to UCE embedding NPZ.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/cache/pair_cache/neural_crest_pns_glia_hvg2048_adjacent_k5.npz",
        help="Output pair table path (.npz).",
    )
    parser.add_argument(
        "--cell-type-col",
        type=str,
        default="author_cell_type",
        help="Metadata column used for within-type pairing.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors per source cell.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=1,
        help="Max time-bin step distance. Default 1 means adjacent bins only.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="adjacent_bins",
        choices=["adjacent_bins", "pooled_future"],
        help="Pairing mode. `adjacent_bins` keeps the original per-step table; `pooled_future` matches the training-time all-future semantics.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Source chunk size for cosine similarity blocks.",
    )
    return parser.parse_args()


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    np.exp(x, out=x)
    denom = np.maximum(np.sum(x, axis=1, keepdims=True), 1e-12)
    return x / denom


def _compute_topk_cosine_pairs(z_src: np.ndarray, z_tgt: np.ndarray, k: int, chunk_size: int = 512):
    if z_src.ndim != 2 or z_tgt.ndim != 2:
        raise ValueError("z_src/z_tgt must be 2D.")
    if z_src.shape[1] != z_tgt.shape[1]:
        raise ValueError("z_src/z_tgt feature dims mismatch.")
    if z_tgt.shape[0] == 0:
        raise ValueError("empty tgt pool.")

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


def _build_adjacent_bins_pair_table(args):
    pre_dir = Path(args.preprocessed_dir)
    output_path = Path(args.output_path)
    metadata = pd.read_pickle(pre_dir / "cell_metadata.pkl")
    obs_names = np.load(pre_dir / "obs_names.npy", allow_pickle=True).astype(str)
    if metadata.shape[0] != obs_names.shape[0]:
        raise ValueError("metadata rows and obs_names length mismatch")
    if not np.array_equal(metadata.index.to_numpy().astype(str), obs_names):
        raise ValueError("metadata index is not aligned with obs_names")

    uce = np.load(args.uce_npz, allow_pickle=True)
    uce_obs_names = uce["obs_names"].astype(str)
    uce_z = np.asarray(uce["z"], dtype=np.float32)

    pre_row_by_obs = {name: idx for idx, name in enumerate(obs_names.tolist())}
    aligned_pre_idx = []
    aligned_uce_idx = []
    for uce_idx, obs_name in enumerate(uce_obs_names.tolist()):
        pre_idx = pre_row_by_obs.get(obs_name)
        if pre_idx is None:
            continue
        aligned_pre_idx.append(pre_idx)
        aligned_uce_idx.append(uce_idx)
    aligned_pre_idx = np.asarray(aligned_pre_idx, dtype=np.int64)
    aligned_uce_idx = np.asarray(aligned_uce_idx, dtype=np.int64)

    aligned_meta = metadata.iloc[aligned_pre_idx].copy()
    aligned_meta["pre_idx"] = aligned_pre_idx
    aligned_meta["uce_idx"] = aligned_uce_idx
    aligned_meta["obs_name"] = obs_names[aligned_pre_idx]
    aligned_meta["time_raw"] = aligned_meta["author_day"].astype(str)
    aligned_meta["time_numeric"] = aligned_meta["author_day_numeric"].astype(np.float32)
    aligned_meta["cell_type"] = aligned_meta[args.cell_type_col].astype(str)

    valid_mask = (
        aligned_meta["cell_type"].ne("")
        & aligned_meta["time_raw"].ne("")
        & np.isfinite(aligned_meta["time_numeric"].to_numpy(dtype=np.float32))
    )
    aligned_meta = aligned_meta.loc[valid_mask].reset_index(drop=True)

    cell_type_names = sorted(aligned_meta["cell_type"].unique().tolist())
    cell_type_to_id = {name: idx for idx, name in enumerate(cell_type_names)}
    aligned_meta["cell_type_id"] = aligned_meta["cell_type"].map(cell_type_to_id).astype(np.int64)

    time_bins = []
    for raw_value, group in aligned_meta.groupby("time_raw", sort=False):
        time_bins.append((str(raw_value), float(group["time_numeric"].mean())))
    time_bins.sort(key=lambda x: (x[1], x[0]))
    time_bin_names = [name for name, _ in time_bins]
    if len(time_bin_names) <= 1:
        raise ValueError("Not enough time bins to build temporal pairs.")

    aligned_z = uce_z[aligned_meta["uce_idx"].to_numpy(dtype=np.int64)]

    src_index_all = []
    tgt_index_all = []
    pair_weight_all = []
    src_time_all = []
    tgt_time_all = []
    src_time_raw_all = []
    tgt_time_raw_all = []
    cell_type_id_all = []
    cell_type_name_all = []
    step_all = []

    total_group_jobs = 0
    built_group_jobs = 0
    for step in range(1, max(int(args.max_step), 1) + 1):
        for src_bin_pos in range(0, len(time_bin_names) - step):
            tgt_bin_pos = src_bin_pos + step
            src_bin = time_bin_names[src_bin_pos]
            tgt_bin = time_bin_names[tgt_bin_pos]
            src_bin_mask = aligned_meta["time_raw"].eq(src_bin).to_numpy()
            tgt_bin_mask = aligned_meta["time_raw"].eq(tgt_bin).to_numpy()
            src_cts = np.unique(aligned_meta.loc[src_bin_mask, "cell_type"].to_numpy())
            tgt_cts = np.unique(aligned_meta.loc[tgt_bin_mask, "cell_type"].to_numpy())
            common_cts = np.intersect1d(src_cts, tgt_cts)
            total_group_jobs += int(common_cts.shape[0])

            for ct_name in common_cts.tolist():
                src_mask = src_bin_mask & aligned_meta["cell_type"].eq(ct_name).to_numpy()
                tgt_mask = tgt_bin_mask & aligned_meta["cell_type"].eq(ct_name).to_numpy()
                src_rows = np.where(src_mask)[0]
                tgt_rows = np.where(tgt_mask)[0]
                if src_rows.size == 0 or tgt_rows.size == 0:
                    continue

                nn_local_idx, nn_sim = _compute_topk_cosine_pairs(
                    aligned_z[src_rows],
                    aligned_z[tgt_rows],
                    k=int(args.k),
                    chunk_size=int(args.chunk_size),
                )
                nn_global_rows = tgt_rows[nn_local_idx]
                nn_weights = _softmax_rows(nn_sim.astype(np.float32, copy=True))

                src_pre = aligned_meta.iloc[src_rows]["pre_idx"].to_numpy(dtype=np.int64)
                tgt_pre = aligned_meta.iloc[nn_global_rows.reshape(-1)]["pre_idx"].to_numpy(dtype=np.int64).reshape(nn_global_rows.shape)
                src_time = aligned_meta.iloc[src_rows]["time_numeric"].to_numpy(dtype=np.float32)
                tgt_time = aligned_meta.iloc[nn_global_rows.reshape(-1)]["time_numeric"].to_numpy(dtype=np.float32).reshape(nn_global_rows.shape)
                src_time_raw = aligned_meta.iloc[src_rows]["time_raw"].to_numpy(dtype=str)
                tgt_time_raw = aligned_meta.iloc[nn_global_rows.reshape(-1)]["time_raw"].to_numpy(dtype=str).reshape(nn_global_rows.shape)
                ct_id = int(cell_type_to_id[ct_name])

                src_index_all.append(np.repeat(src_pre[:, None], nn_global_rows.shape[1], axis=1).reshape(-1))
                tgt_index_all.append(tgt_pre.reshape(-1))
                pair_weight_all.append(nn_weights.reshape(-1))
                src_time_all.append(np.repeat(src_time[:, None], nn_global_rows.shape[1], axis=1).reshape(-1))
                tgt_time_all.append(tgt_time.reshape(-1))
                src_time_raw_all.append(np.repeat(src_time_raw[:, None], nn_global_rows.shape[1], axis=1).reshape(-1))
                tgt_time_raw_all.append(tgt_time_raw.reshape(-1))
                cell_type_id_all.append(np.full((src_rows.size * nn_global_rows.shape[1],), ct_id, dtype=np.int64))
                cell_type_name_all.append(np.full((src_rows.size * nn_global_rows.shape[1],), ct_name, dtype=object))
                step_all.append(np.full((src_rows.size * nn_global_rows.shape[1],), step, dtype=np.int64))
                built_group_jobs += 1

                if built_group_jobs % 20 == 0:
                    print(
                        f"[pairs] built {built_group_jobs}/{max(total_group_jobs, built_group_jobs)} "
                        f"groups; latest={src_bin}->{tgt_bin} / {ct_name}"
                    )

    if not src_index_all:
        raise ValueError("No temporal pairs were constructed.")

    pair_payload = {
        "src_index": np.concatenate(src_index_all).astype(np.int64, copy=False),
        "tgt_index": np.concatenate(tgt_index_all).astype(np.int64, copy=False),
        "pair_weight": np.concatenate(pair_weight_all).astype(np.float32, copy=False),
        "src_time": np.concatenate(src_time_all).astype(np.float32, copy=False),
        "tgt_time": np.concatenate(tgt_time_all).astype(np.float32, copy=False),
        "src_time_raw": np.concatenate(src_time_raw_all).astype(str),
        "tgt_time_raw": np.concatenate(tgt_time_raw_all).astype(str),
        "cell_type_id": np.concatenate(cell_type_id_all).astype(np.int64, copy=False),
        "cell_type_name": np.concatenate(cell_type_name_all).astype(str),
        "step_distance": np.concatenate(step_all).astype(np.int64, copy=False),
    }
    summary = {
        "preprocessed_dir": str(pre_dir),
        "uce_npz": str(args.uce_npz),
        "output_path": str(args.output_path),
        "n_preprocessed_cells": int(obs_names.shape[0]),
        "n_uce_cells": int(uce_obs_names.shape[0]),
        "n_aligned_cells": int(aligned_pre_idx.shape[0]),
        "n_missing_uce_cells": int(obs_names.shape[0] - aligned_pre_idx.shape[0]),
        "n_time_bins": int(len(time_bin_names)),
        "time_bins": time_bin_names,
        "cell_type_col": args.cell_type_col,
        "n_cell_types": int(len(cell_type_names)),
        "k": int(args.k),
        "max_step": int(args.max_step),
        "chunk_size": int(args.chunk_size),
        "n_pairs": int(pair_payload["src_index"].shape[0]),
        "n_unique_src": int(np.unique(pair_payload["src_index"]).shape[0]),
        "n_unique_tgt": int(np.unique(pair_payload["tgt_index"]).shape[0]),
    }
    return pair_payload, summary


def _build_pooled_future_pair_table(args):
    counts, _, _, meta = load_h5ad_with_uce(
        path_or_key=args.source_h5ad,
        uce_npz_path=args.uce_npz,
        time_col="author_day",
        time_numeric_col=None,
        cell_type_col=args.cell_type_col,
        backed=False,
        max_cells=None,
        cell_seed=42,
    )
    del counts
    pair_payload = build_temporal_knn_pairs(
        meta=meta,
        split_name="all_future",
        src_min_time=None,
        src_max_time=None,
        tgt_min_time=None,
        tgt_max_time=None,
        future_only=True,
        k=int(args.k),
        similarity_chunk_size=int(args.chunk_size),
        cache_path=None,
        force_rebuild=True,
    )
    aligned_to_pre_idx = np.asarray(meta["aligned_h5ad_indices"], dtype=np.int64)
    src_index = aligned_to_pre_idx[np.asarray(pair_payload["src_index"], dtype=np.int64)]
    tgt_index = aligned_to_pre_idx[np.asarray(pair_payload["tgt_index"], dtype=np.int64)]
    cell_type_names = np.asarray(meta.get("cell_type_names", []), dtype=object)
    cell_type_id = np.asarray(pair_payload["cell_type_id"], dtype=np.int64)
    if cell_type_names.size > 0:
        cell_type_name = cell_type_names[cell_type_id].astype(str)
    else:
        cell_type_name = np.full(cell_type_id.shape, "", dtype=object)

    output_payload = {
        "src_index": src_index.astype(np.int64, copy=False),
        "tgt_index": tgt_index.astype(np.int64, copy=False),
        "pair_weight": np.asarray(pair_payload["pair_weight"], dtype=np.float32),
        "src_time": np.asarray(pair_payload["src_time"], dtype=np.float32),
        "tgt_time": np.asarray(pair_payload["tgt_time"], dtype=np.float32),
        "src_time_raw": np.asarray(pair_payload["src_time_raw"]).astype(str),
        "tgt_time_raw": np.asarray(pair_payload["tgt_time_raw"]).astype(str),
        "cell_type_id": cell_type_id.astype(np.int64, copy=False),
        "cell_type_name": np.asarray(cell_type_name).astype(str),
    }
    summary = {
        "preprocessed_dir": str(args.preprocessed_dir),
        "source_h5ad": str(args.source_h5ad),
        "uce_npz": str(args.uce_npz),
        "output_path": str(args.output_path),
        "mode": "pooled_future",
        "n_preprocessed_cells": int(np.load(Path(args.preprocessed_dir) / "obs_names.npy", allow_pickle=True).shape[0]),
        "n_aligned_cells": int(meta.get("n_cells_aligned", len(aligned_to_pre_idx))),
        "n_missing_uce_cells": int(np.load(Path(args.preprocessed_dir) / "obs_names.npy", allow_pickle=True).shape[0] - len(aligned_to_pre_idx)),
        "n_time_bins": int(len(np.unique(np.asarray(meta["time_raw"]).astype(str)))),
        "time_bins": sorted(np.unique(np.asarray(meta["time_raw"]).astype(str)).tolist()),
        "cell_type_col": args.cell_type_col,
        "n_cell_types": int(len(cell_type_names)),
        "k": int(args.k),
        "chunk_size": int(args.chunk_size),
        "future_only": True,
        "n_pairs": int(output_payload["src_index"].shape[0]),
        "n_unique_src": int(np.unique(output_payload["src_index"]).shape[0]),
        "n_unique_tgt": int(np.unique(output_payload["tgt_index"]).shape[0]),
    }
    return output_payload, summary


def main():
    args = parse_args()
    t0 = time.time()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "pooled_future":
        pair_payload, summary = _build_pooled_future_pair_table(args)
    else:
        pair_payload, summary = _build_adjacent_bins_pair_table(args)

    np.savez_compressed(output_path, **pair_payload)
    summary["mode"] = str(args.mode)
    summary["elapsed_seconds"] = float(time.time() - t0)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[pairs] wrote {pair_payload['src_index'].shape[0]} pairs to {output_path}")
    if "n_aligned_cells" in summary and "n_missing_uce_cells" in summary:
        print(f"[pairs] aligned_cells={summary['n_aligned_cells']} missing_uce={summary['n_missing_uce_cells']}")
    print(f"[pairs] finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
