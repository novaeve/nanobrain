#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import _load_preprocessed_temporal_bundle, load_mouse_development_temporal_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare fixed evaluation assets for temporal scFiT experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="./config_mouse_temporal_single_h100.yaml",
        help="Temporal training/eval config used to define the training split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/eval_assets/legacy/neural_crest_temporal_fixed",
        help="Directory to save PCA / marker / classifier / pseudotime assets.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA dimensions.",
    )
    parser.add_argument(
        "--marker-top-k",
        type=int,
        default=50,
        help="Number of marker genes to keep for each cell type.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="k for the fixed cell-type kNN classifier.",
    )
    parser.add_argument(
        "--dpt-neighbors",
        type=int,
        default=15,
        help="Neighbors used to build the DPT graph in PCA space.",
    )
    parser.add_argument(
        "--root-obs-name",
        type=str,
        default="",
        help="Optional explicit root cell obs_name for DPT. Default: first cell from earliest training time.",
    )
    parser.add_argument(
        "--transform-batch-size",
        type=int,
        default=4096,
        help="Batch size when transforming the full expression matrix into PCA space.",
    )
    return parser.parse_args()


def _load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_train_cell_indices(cfg):
    data_cfg = cfg.get("data", {})
    pairing_cfg = cfg.get("pairing", {}) or {}
    train_dataset, _, _, _ = load_mouse_development_temporal_datasets(
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
    train_cell_indices = np.unique(
        np.concatenate(
            [
                np.asarray(train_dataset.src_index, dtype=np.int64),
                np.asarray(train_dataset.tgt_index, dtype=np.int64),
            ]
        )
    )
    return train_cell_indices


def _fit_pca(train_expr: np.ndarray, n_components: int, seed: int):
    n_components = min(int(n_components), int(train_expr.shape[0]), int(train_expr.shape[1]))
    if n_components < 2:
        raise ValueError("PCA requires at least 2 components.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    train_pca = pca.fit_transform(train_expr)
    return pca, train_pca.astype(np.float32, copy=False)


def _transform_all_cells(expr, pca: PCA, batch_size: int):
    outputs = []
    for start in range(0, expr.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), expr.shape[0])
        chunk = np.asarray(expr[start:stop], dtype=np.float32)
        outputs.append(np.asarray(pca.transform(chunk), dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def _compute_group_stats(train_expr: np.ndarray, labels: np.ndarray):
    stats = {}
    for label in np.unique(labels):
        mask = labels == label
        group = train_expr[mask]
        stats[str(label)] = {
            "n_cells": int(group.shape[0]),
            "mean": group.mean(axis=0).astype(np.float32),
            "var": group.var(axis=0).astype(np.float32),
        }
    return stats


def _select_pairwise_markers(train_expr: np.ndarray, labels: np.ndarray, gene_names, top_k: int):
    gene_names = np.asarray(gene_names).astype(str)
    group_stats = _compute_group_stats(train_expr, labels)
    cell_types = sorted(group_stats.keys())
    markers = {}

    for cell_type in cell_types:
        mean_a = group_stats[cell_type]["mean"]
        var_a = group_stats[cell_type]["var"]
        n_a = max(group_stats[cell_type]["n_cells"], 1)
        pair_scores = []
        pair_diffs = []
        for other in cell_types:
            if other == cell_type:
                continue
            mean_b = group_stats[other]["mean"]
            var_b = group_stats[other]["var"]
            n_b = max(group_stats[other]["n_cells"], 1)
            diff = mean_a - mean_b
            se = np.sqrt(var_a / float(n_a) + var_b / float(n_b) + 1e-8)
            score = diff / se
            pair_scores.append(score)
            pair_diffs.append(diff)

        stacked_scores = np.stack(pair_scores, axis=0)
        stacked_diffs = np.stack(pair_diffs, axis=0)
        min_pair_score = stacked_scores.min(axis=0)
        mean_pair_diff = stacked_diffs.mean(axis=0)
        all_positive = (stacked_diffs > 0).all(axis=0)

        candidate_idx = np.flatnonzero(all_positive)
        if candidate_idx.size < int(top_k):
            candidate_idx = np.arange(gene_names.shape[0], dtype=np.int64)

        order = np.lexsort(
            (
                -mean_pair_diff[candidate_idx],
                -min_pair_score[candidate_idx],
            )
        )
        chosen = candidate_idx[order[: int(top_k)]]
        markers[cell_type] = [str(g) for g in gene_names[chosen]]
    return markers


def _fit_cell_type_knn(train_pca: np.ndarray, train_labels: np.ndarray, k: int):
    effective_k = min(int(k), int(train_pca.shape[0]))
    clf = KNeighborsClassifier(n_neighbors=effective_k, metric="euclidean")
    clf.fit(train_pca, train_labels)
    return clf, effective_k


def _compute_dpt(train_pca: np.ndarray, train_meta: pd.DataFrame, n_neighbors: int, root_obs_name: str = ""):
    import anndata as ad
    import scanpy as sc

    obs = train_meta.copy()
    obs.index = obs["obs_name"].astype(str)
    adata = ad.AnnData(X=np.zeros((train_pca.shape[0], 1), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.asarray(train_pca, dtype=np.float32)

    effective_neighbors = max(2, min(int(n_neighbors), int(train_pca.shape[0]) - 1))
    sc.pp.neighbors(adata, n_neighbors=effective_neighbors, use_rep="X_pca")
    sc.tl.diffmap(adata)

    if root_obs_name:
        if root_obs_name not in adata.obs_names:
            raise ValueError(f"root obs_name not found in training cells: {root_obs_name}")
        root_idx = int(np.flatnonzero(adata.obs_names.to_numpy() == str(root_obs_name))[0])
    else:
        time_values = adata.obs["author_day_numeric"].to_numpy(dtype=np.float32)
        earliest = np.nanmin(time_values)
        candidate_idx = np.flatnonzero(np.isclose(time_values, earliest, equal_nan=False))
        if candidate_idx.size == 0:
            raise ValueError("Failed to locate an earliest-time training cell for DPT root.")
        root_idx = int(candidate_idx[0])
        root_obs_name = str(adata.obs_names[root_idx])

    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)

    pseudotime = adata.obs["dpt_pseudotime"].to_numpy(dtype=np.float32)
    finite = pseudotime[np.isfinite(pseudotime)]
    if finite.size == 0:
        raise ValueError("DPT returned no finite pseudotime values.")
    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    if max_v > min_v:
        pseudotime = (pseudotime - min_v) / (max_v - min_v)
    else:
        pseudotime = np.zeros_like(pseudotime, dtype=np.float32)
    return pseudotime.astype(np.float32, copy=False), root_obs_name, effective_neighbors


def main():
    args = parse_args()
    t0 = time.time()

    cfg = _load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expr, metadata_df, pre_meta = _load_preprocessed_temporal_bundle(cfg["data"]["preprocessed_dir"])
    metadata_df = metadata_df.copy()
    if "obs_name" not in metadata_df.columns:
        metadata_df = metadata_df.reset_index()
        if "obs_name" not in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={metadata_df.columns[0]: "obs_name"})

    train_cell_idx = _load_train_cell_indices(cfg)
    train_cell_idx = np.asarray(train_cell_idx, dtype=np.int64)
    np.save(out_dir / "train_cell_indices.npy", train_cell_idx)

    train_meta = metadata_df.iloc[train_cell_idx].copy().reset_index(drop=True)
    train_expr = np.asarray(expr[train_cell_idx], dtype=np.float32)
    train_labels = train_meta[cfg["data"].get("cell_type_col", "author_cell_type")].astype(str).to_numpy()

    print(f"[assets] train cells used for fitting: {train_expr.shape[0]}")
    print(f"[assets] fitting PCA ({args.pca_components} components) on training cells only")
    pca, train_pca = _fit_pca(
        train_expr=train_expr,
        n_components=args.pca_components,
        seed=int(cfg.get("training", {}).get("seed", 42)),
    )
    pca_model_path = out_dir / "pca_model.pkl"
    joblib.dump(pca, pca_model_path)

    print("[assets] transforming full fixed-HVG matrix into the frozen PCA space")
    all_pca = _transform_all_cells(expr, pca, batch_size=args.transform_batch_size)
    pca_embedding_path = out_dir / "pca_embeddings_all_cells.npz"
    np.savez_compressed(
        pca_embedding_path,
        X_pca=all_pca.astype(np.float32, copy=False),
        obs_names=metadata_df["obs_name"].astype(str).to_numpy(),
        train_cell_indices=train_cell_idx,
    )

    print(f"[assets] selecting top-{args.marker_top_k} pairwise markers per cell type on training cells")
    hvg_gene_names = pre_meta.get("hvg_gene_names")
    if hvg_gene_names is None:
        hvg_gene_names = pre_meta.get("hvg_var_names")
    marker_dict = _select_pairwise_markers(
        train_expr=train_expr,
        labels=train_labels,
        gene_names=hvg_gene_names,
        top_k=args.marker_top_k,
    )
    marker_path = out_dir / "marker_genes_by_cell_type.json"
    with marker_path.open("w", encoding="utf-8") as f:
        json.dump({"markers": marker_dict}, f, ensure_ascii=False, indent=2)

    print(f"[assets] fitting kNN classifier (k={args.knn_k}) in frozen PCA space")
    clf, effective_k = _fit_cell_type_knn(train_pca, train_labels, k=args.knn_k)
    classifier_path = out_dir / "cell_type_knn.pkl"
    joblib.dump(clf, classifier_path)
    classifier_train_acc = float(np.mean(clf.predict(train_pca) == train_labels))
    classifier_all_real_acc = float(
        np.mean(
            clf.predict(all_pca.astype(np.float32, copy=False))
            == metadata_df[cfg["data"].get("cell_type_col", "author_cell_type")].astype(str).to_numpy()
        )
    )

    print("[assets] computing DPT pseudotime on training cells only")
    pseudotime, root_obs_name, effective_neighbors = _compute_dpt(
        train_pca=train_pca,
        train_meta=train_meta,
        n_neighbors=args.dpt_neighbors,
        root_obs_name=args.root_obs_name,
    )
    pseudotime_path = out_dir / "pseudotime_train_dpt.npz"
    np.savez_compressed(
        pseudotime_path,
        obs_names=train_meta["obs_name"].astype(str).to_numpy(),
        pseudotime=pseudotime.astype(np.float32, copy=False),
        root_obs_name=np.asarray([root_obs_name], dtype=object),
    )

    summary = {
        "config_path": str(Path(args.config).resolve()),
        "preprocessed_dir": cfg["data"]["preprocessed_dir"],
        "pair_table_path": cfg["pairing"]["pair_table_path"],
        "output_dir": str(out_dir.resolve()),
        "n_total_cells": int(expr.shape[0]),
        "n_train_cells_fit": int(train_expr.shape[0]),
        "n_hvg_genes": int(train_expr.shape[1]),
        "pca_components": int(pca.n_components_),
        "pca_model_path": str(pca_model_path.resolve()),
        "pca_embedding_path": str(pca_embedding_path.resolve()),
        "marker_gene_path": str(marker_path.resolve()),
        "cell_type_knn_path": str(classifier_path.resolve()),
        "pseudotime_path": str(pseudotime_path.resolve()),
        "cell_type_knn_k": int(effective_k),
        "cell_type_knn_train_acc": classifier_train_acc,
        "cell_type_knn_all_real_acc": classifier_all_real_acc,
        "dpt_root_obs_name": root_obs_name,
        "dpt_neighbors": int(effective_neighbors),
        "train_time_min": float(np.nanmin(train_meta["author_day_numeric"].to_numpy(dtype=np.float32))),
        "train_time_max": float(np.nanmax(train_meta["author_day_numeric"].to_numpy(dtype=np.float32))),
        "cell_types": sorted(pd.unique(train_labels).astype(str).tolist()),
        "elapsed_seconds": float(time.time() - t0),
    }
    with (out_dir / "asset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
