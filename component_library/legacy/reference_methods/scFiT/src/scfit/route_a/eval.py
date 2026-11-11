import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from scfit.utils import sinkhorn_distance


SUPPORTED_EVAL_METRICS = {
    "wasserstein_hvg",
    "wasserstein_pca50",
    "distribution_mean_r2",
    "nn_matched_per_cell_r2",
    "cell_type_accuracy",
    "marker_mmd",
}


def _safe_pearson_r2(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr * corr)


def _rbf_mmd(x: np.ndarray, y: np.ndarray):
    if x.shape[0] < 2 or y.shape[0] < 2:
        return None
    merged = np.concatenate([x, y], axis=0)
    dist_sq = np.sum((merged[:, None, :] - merged[None, :, :]) ** 2, axis=-1)
    upper = dist_sq[np.triu_indices(dist_sq.shape[0], k=1)]
    upper = upper[upper > 0]
    if upper.size == 0:
        return None
    sigma_sq = float(np.median(upper))
    gamma = 1.0 / max(2.0 * sigma_sq, 1e-8)
    k_xx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    k_yy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    k_xy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def _torch_sinkhorn_np(x: np.ndarray, y: np.ndarray, epsilon: float, max_iter: int) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return float(sinkhorn_distance(x_t, y_t, epsilon=epsilon, max_iter=max_iter).item())


def _normalize_metric_names(metric_names=None):
    if metric_names is None:
        return set(SUPPORTED_EVAL_METRICS)
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    normalized = set()
    for name in metric_names:
        key = str(name).strip()
        if not key:
            continue
        if key == "full":
            return set(SUPPORTED_EVAL_METRICS)
        if key not in SUPPORTED_EVAL_METRICS:
            raise ValueError(f"unsupported eval metric: {key}")
        normalized.add(key)
    if not normalized:
        raise ValueError("metric_names resolved to an empty set")
    return normalized


def load_eval_assets(assets_dir: str) -> Dict:
    assets_path = Path(assets_dir)
    summary_path = assets_path / "asset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"asset summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    pca = joblib.load(summary["pca_model_path"])
    clf = None
    clf_path = summary.get("cell_type_knn_path")
    if clf_path:
        clf_path = Path(clf_path)
        if clf_path.exists():
            clf = joblib.load(clf_path)
    marker_payload = json.loads(Path(summary["marker_gene_path"]).read_text())
    real_pca_npz = np.load(summary["pca_embedding_path"], allow_pickle=True)
    return {
        "summary": summary,
        "pca": pca,
        "classifier": clf,
        "marker_dict": marker_payload.get("markers", {}),
        "real_pca": np.asarray(real_pca_npz["X_pca"], dtype=np.float32),
        "real_pca_obs_names": real_pca_npz["obs_names"].astype(str),
    }


def save_prediction_artifact(output_dir: str, artifact: Dict):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[route_a_eval] saving artifact -> {out_dir}")
    arrays = {}
    for time_bin, pred in tqdm(artifact["pred_by_time"].items(), desc="save-pred", leave=False):
        key = time_bin.replace(".", "_")
        arrays[f"pred__{key}"] = np.asarray(pred, dtype=np.float32)
    for time_bin, real in tqdm(artifact["real_by_time"].items(), desc="save-real", leave=False):
        key = time_bin.replace(".", "_")
        arrays[f"real__{key}"] = np.asarray(real, dtype=np.float32)
    for time_bin, obs_names in tqdm(artifact["obs_names_by_time"].items(), desc="save-obs", leave=False):
        key = time_bin.replace(".", "_")
        arrays[f"obs__{key}"] = np.asarray(obs_names).astype(str)
    np.savez_compressed(out_dir / "artifact_arrays.npz", **arrays)
    metadata = {
        "protocol_name": artifact["protocol_name"],
        "method_name": artifact["method_name"],
        "ordered_time_bins": [str(x) for x in artifact["ordered_time_bins"]],
        "heldout_time_bins": [str(x) for x in artifact["heldout_time_bins"]],
        "time_display_map": {str(k): str(v) for k, v in (artifact.get("time_display_map") or {}).items()},
    }
    (out_dir / "artifact_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return str(out_dir)


def _load_artifact(artifact_dir: str) -> Dict:
    artifact_path = Path(artifact_dir)
    summary = json.loads((artifact_path / "artifact_summary.json").read_text())
    arrays = np.load(artifact_path / "artifact_arrays.npz", allow_pickle=True)
    pred_by_time = {}
    real_by_time = {}
    obs_names_by_time = {}
    for time_bin in summary["heldout_time_bins"]:
        key = str(time_bin).replace(".", "_")
        pred_by_time[time_bin] = np.asarray(arrays[f"pred__{key}"], dtype=np.float32)
        real_by_time[time_bin] = np.asarray(arrays[f"real__{key}"], dtype=np.float32)
        obs_names_by_time[time_bin] = arrays[f"obs__{key}"].astype(str)
    return {
        **summary,
        "pred_by_time": pred_by_time,
        "real_by_time": real_by_time,
        "obs_names_by_time": obs_names_by_time,
    }


def evaluate_artifact_dir(
    artifact_dir: str,
    assets_dir: str,
    metadata_df: pd.DataFrame,
    hvg_gene_names,
    epsilon: float = 0.05,
    max_iter: int = 100,
    metric_names=None,
):
    artifact = _load_artifact(artifact_dir)
    metric_names = _normalize_metric_names(metric_names)
    needs_pca = bool(
        {"wasserstein_pca50", "cell_type_accuracy", "marker_mmd"} & metric_names
    )
    needs_real_obs_lookup = bool({"wasserstein_pca50", "cell_type_accuracy", "marker_mmd"} & metric_names)
    needs_marker = "marker_mmd" in metric_names
    assets = load_eval_assets(assets_dir) if needs_pca else {}
    print(f"[route_a_eval] artifact={artifact_dir}")
    print(f"[route_a_eval] assets={assets_dir}")
    metadata_df = metadata_df.copy()
    if "obs_name" not in metadata_df.columns:
        metadata_df = metadata_df.reset_index().rename(columns={metadata_df.columns[0]: "obs_name"})
    real_pca_map = {obs: i for i, obs in enumerate(assets["real_pca_obs_names"].tolist())} if needs_real_obs_lookup else {}
    hvg_index = {str(g): idx for idx, g in enumerate(np.asarray(hvg_gene_names).astype(str))} if needs_marker else {}
    cell_type_col = "author_cell_type" if "author_cell_type" in metadata_df.columns else None

    per_time = {}
    time_display_map = {str(k): str(v) for k, v in (artifact.get("time_display_map") or {}).items()}
    w_hvg_all = []
    w_pca_all = []
    dist_r2_all = []
    nn_r2_all = []
    type_acc_all = []
    marker_mmd_all = []
    clf = assets.get("classifier")
    pca = assets.get("pca")

    for time_bin in tqdm(artifact["heldout_time_bins"], desc="eval-heldout", leave=False):
        pred = np.asarray(artifact["pred_by_time"][time_bin], dtype=np.float32)
        real = np.asarray(artifact["real_by_time"][time_bin], dtype=np.float32)
        obs_names = artifact["obs_names_by_time"][time_bin]
        if pred.shape[0] == 0 or real.shape[0] == 0:
            continue

        pred_pca = None
        real_meta = None
        real_pca = None
        if needs_pca:
            pred_pca = pca.transform(pred)
        if needs_real_obs_lookup:
            real_meta = metadata_df.set_index("obs_name").loc[list(obs_names)].reset_index()
            real_idx = np.asarray([real_pca_map[x] for x in real_meta["obs_name"].astype(str)], dtype=np.int64)
            real_pca = assets["real_pca"][real_idx]

        wasserstein_hvg = _torch_sinkhorn_np(pred, real, epsilon=epsilon, max_iter=max_iter) if "wasserstein_hvg" in metric_names else None
        wasserstein_pca50 = (
            _torch_sinkhorn_np(pred_pca, real_pca, epsilon=epsilon, max_iter=max_iter)
            if "wasserstein_pca50" in metric_names
            else None
        )
        dist_mean_r2 = (
            _safe_pearson_r2(pred.mean(axis=0), real.mean(axis=0))
            if "distribution_mean_r2" in metric_names
            else None
        )

        nn_matched_r2 = None
        if "nn_matched_per_cell_r2" in metric_names:
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(real)
            matched = real[nn.kneighbors(pred, return_distance=False).reshape(-1)]
            nn_scores = []
            for pred_vec, real_vec in zip(pred, matched):
                score = _safe_pearson_r2(pred_vec.astype(np.float64), real_vec.astype(np.float64))
                if score is not None:
                    nn_scores.append(score)
            nn_matched_r2 = None if not nn_scores else float(np.mean(nn_scores))

        type_acc = None
        marker_mmd = None
        if (
            {"cell_type_accuracy", "marker_mmd"} & metric_names
            and cell_type_col is not None
            and clf is not None
            and real_meta is not None
            and not real_meta.empty
        ):
            pred_labels = np.asarray(clf.predict(pred_pca)).astype(str)
            real_labels = real_meta[cell_type_col].astype(str).to_numpy()
            if "cell_type_accuracy" in metric_names:
                nn_real_pca = NearestNeighbors(n_neighbors=1, metric="euclidean")
                nn_real_pca.fit(real_pca)
                nearest_real = nn_real_pca.kneighbors(pred_pca, return_distance=False).reshape(-1)
                matched_real_labels = real_labels[nearest_real]
                type_acc = float(np.mean(pred_labels == matched_real_labels))

            if "marker_mmd" in metric_names:
                class_scores = []
                for cell_type_name, marker_genes in assets["marker_dict"].items():
                    marker_idx = [hvg_index[g] for g in marker_genes if g in hvg_index]
                    if not marker_idx:
                        continue
                    pred_mask = pred_labels == str(cell_type_name)
                    real_mask = real_labels == str(cell_type_name)
                    if pred_mask.sum() < 2 or real_mask.sum() < 2:
                        continue
                    score = _rbf_mmd(pred[pred_mask][:, marker_idx], real[real_mask][:, marker_idx])
                    if score is not None:
                        class_scores.append(score)
                if class_scores:
                    marker_mmd = float(np.mean(class_scores))

        display_key = time_display_map.get(str(time_bin), str(time_bin))
        per_time[display_key] = {
            "wasserstein_hvg": wasserstein_hvg,
            "wasserstein_pca50": wasserstein_pca50,
            "distribution_mean_r2": dist_mean_r2,
            "nn_matched_per_cell_r2": nn_matched_r2,
            "cell_type_accuracy": type_acc,
            "marker_mmd": marker_mmd,
        }
        if wasserstein_hvg is not None:
            w_hvg_all.append(wasserstein_hvg)
        if wasserstein_pca50 is not None:
            w_pca_all.append(wasserstein_pca50)
        if dist_mean_r2 is not None:
            dist_r2_all.append(dist_mean_r2)
        if nn_matched_r2 is not None:
            nn_r2_all.append(nn_matched_r2)
        if type_acc is not None:
            type_acc_all.append(type_acc)
        if marker_mmd is not None:
            marker_mmd_all.append(marker_mmd)

    metrics = {
        "protocol_name": artifact["protocol_name"],
        "method_name": artifact["method_name"],
        "wasserstein_hvg_mean": None if not w_hvg_all else float(np.mean(w_hvg_all)),
        "wasserstein_pca50_mean": None if not w_pca_all else float(np.mean(w_pca_all)),
        "distribution_mean_r2_mean": None if not dist_r2_all else float(np.mean(dist_r2_all)),
        "nn_matched_per_cell_r2_mean": None if not nn_r2_all else float(np.mean(nn_r2_all)),
        "cell_type_accuracy_mean": None if not type_acc_all else float(np.mean(type_acc_all)),
        "marker_mmd_mean": None if not marker_mmd_all else float(np.mean(marker_mmd_all)),
        "time_display_map": time_display_map,
        "per_time": per_time,
    }
    metrics_path = Path(artifact_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[route_a_eval] wrote metrics -> {metrics_path}")
    return metrics
