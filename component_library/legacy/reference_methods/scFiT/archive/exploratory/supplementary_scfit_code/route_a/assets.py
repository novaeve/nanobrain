import json
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from .protocol import load_protocol_bundle


def _select_markers(train_expr: np.ndarray, train_labels: np.ndarray, gene_names, top_k: int = 50):
    gene_names = np.asarray(gene_names).astype(str)
    markers = {}
    unique_labels = sorted(np.unique(train_labels).tolist())
    for label in tqdm(unique_labels, desc="marker-select", leave=False):
        mask = train_labels == label
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        pos_mean = train_expr[mask].mean(axis=0)
        neg_mean = train_expr[~mask].mean(axis=0)
        score = pos_mean - neg_mean
        top_idx = np.argsort(score)[-top_k:][::-1]
        markers[str(label)] = gene_names[top_idx].tolist()
    return markers


def prepare_route_a_assets(
    protocol_path: str,
    output_dir: str,
    pca_components: int = 50,
    knn_k: int = 10,
    marker_top_k: int = 50,
):
    bundle = load_protocol_bundle(protocol_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[route_a_assets] protocol={bundle['protocol_name']}")
    print(f"[route_a_assets] output_dir={out_dir}")

    observed_indices = np.concatenate([bundle["time_to_indices"][t] for t in bundle["observed_time_bins"]], axis=0)
    expr = bundle["expression"]
    metadata_df = bundle["metadata_df"]
    print(f"[route_a_assets] loading observed cells: n_observed={observed_indices.shape[0]}")
    train_expr = np.asarray(expr[observed_indices], dtype=np.float32)

    print(f"[route_a_assets] fitting PCA: n_components={int(pca_components)}")
    pca = PCA(n_components=int(pca_components), random_state=42)
    train_pca = pca.fit_transform(train_expr)
    joblib.dump(pca, out_dir / "pca_model.pkl")

    print("[route_a_assets] transforming all cells into PCA space")
    all_pca = np.empty((expr.shape[0], int(pca_components)), dtype=np.float32)
    chunk_size = 2048
    for start in tqdm(range(0, expr.shape[0], chunk_size), desc="pca-transform", leave=False):
        end = min(start + chunk_size, expr.shape[0])
        all_pca[start:end] = pca.transform(np.asarray(expr[start:end], dtype=np.float32)).astype(np.float32, copy=False)
    np.savez_compressed(
        out_dir / "pca_embeddings_all_cells.npz",
        X_pca=all_pca.astype(np.float32),
        obs_names=metadata_df["obs_name"].astype(str).to_numpy(),
        observed_indices=observed_indices.astype(np.int64),
    )

    cell_type_col = bundle["cell_type_col"]
    marker_path = out_dir / "marker_genes_by_cell_type.json"
    classifier_path = out_dir / "cell_type_knn.pkl"
    classifier_path_str = None
    if cell_type_col is not None:
        print(f"[route_a_assets] fitting cell type kNN: k={min(int(knn_k), train_pca.shape[0])}")
        train_labels = metadata_df.iloc[observed_indices][cell_type_col].astype(str).to_numpy()
        clf = KNeighborsClassifier(n_neighbors=min(int(knn_k), train_pca.shape[0]), metric="euclidean")
        clf.fit(train_pca, train_labels)
        joblib.dump(clf, classifier_path)
        classifier_path_str = str(classifier_path.resolve())
        print(f"[route_a_assets] selecting marker genes: top_k={int(marker_top_k)}")
        markers = _select_markers(
            train_expr=train_expr,
            train_labels=train_labels,
            gene_names=bundle["bundle_meta"]["hvg_gene_names"],
            top_k=int(marker_top_k),
        )
        marker_path.write_text(json.dumps({"markers": markers}, ensure_ascii=False, indent=2))
        sample_n = min(train_pca.shape[0], 2000)
        sample_idx = np.random.RandomState(42).choice(train_pca.shape[0], size=sample_n, replace=False)
        print(f"[route_a_assets] estimating classifier accuracy on sample_n={sample_n}")
        train_acc = float(np.mean(clf.predict(train_pca[sample_idx]) == train_labels[sample_idx]))
    else:
        train_acc = None
        marker_path.write_text(json.dumps({"markers": {}}, ensure_ascii=False, indent=2))

    summary = {
        "protocol_name": bundle["protocol_name"],
        "protocol_path": bundle["protocol_path"],
        "output_dir": str(out_dir.resolve()),
        "pca_model_path": str((out_dir / "pca_model.pkl").resolve()),
        "pca_embedding_path": str((out_dir / "pca_embeddings_all_cells.npz").resolve()),
        "cell_type_knn_path": classifier_path_str,
        "marker_gene_path": str(marker_path.resolve()),
        "n_total_cells": int(metadata_df.shape[0]),
        "n_observed_cells_fit": int(train_expr.shape[0]),
        "observed_time_bins": bundle["observed_time_bins"],
        "heldout_time_bins": bundle["heldout_time_bins"],
        "cell_type_knn_train_acc": train_acc,
        "elapsed_seconds": float(time.time() - t0),
    }
    (out_dir / "asset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[route_a_assets] done in {summary['elapsed_seconds']:.2f}s")
    return summary
