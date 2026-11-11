import numpy as np
from scipy.stats import spearmanr


def safe_pearson(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return None
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return None
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return None
    return float(r)


def init_metric_accumulator():
    return {
        "total_sq": 0.0,
        "total_abs": 0.0,
        "total_cnt": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "cell_pearsons": [],
        "cell_spearmans": [],
        "cell_pearsons_top20": [],
    }


def update_metric_accumulator(stats, pred_cpu: np.ndarray, true_cpu: np.ndarray, mask_cpu: np.ndarray):
    mask_bool = mask_cpu.astype(bool, copy=False)
    cnt = int(mask_bool.sum())
    if cnt > 0:
        pred_mask = pred_cpu[mask_bool].astype(np.float64)
        true_mask = true_cpu[mask_bool].astype(np.float64)
        diff = pred_mask - true_mask
        stats["total_sq"] += float(np.square(diff).sum())
        stats["total_abs"] += float(np.abs(diff).sum())
        stats["total_cnt"] += cnt
        stats["sum_x"] += float(pred_mask.sum())
        stats["sum_y"] += float(true_mask.sum())
        stats["sum_x2"] += float(np.square(pred_mask).sum())
        stats["sum_y2"] += float(np.square(true_mask).sum())
        stats["sum_xy"] += float((pred_mask * true_mask).sum())

    for b in range(pred_cpu.shape[0]):
        mb = mask_bool[b]
        if mb.sum() < 2:
            continue
        pv = pred_cpu[b, mb].astype(np.float64)
        tv = true_cpu[b, mb].astype(np.float64)
        r = safe_pearson(pv, tv)
        if r is not None:
            stats["cell_pearsons"].append(r)
        sr = spearmanr(pv, tv).correlation
        if sr is not None and not np.isnan(sr):
            stats["cell_spearmans"].append(float(sr))
        k = min(20, tv.shape[0])
        if k >= 2:
            idx_top = np.argsort(tv)[-k:]
            r20 = safe_pearson(pv[idx_top], tv[idx_top])
            if r20 is not None:
                stats["cell_pearsons_top20"].append(r20)


def summarize_metric_accumulator(stats):
    mse = stats["total_sq"] / max(stats["total_cnt"], 1)
    mae = stats["total_abs"] / max(stats["total_cnt"], 1)
    rmse = float(np.sqrt(mse))
    denom = np.sqrt(
        (stats["total_cnt"] * stats["sum_x2"] - stats["sum_x"] * stats["sum_x"])
        * (stats["total_cnt"] * stats["sum_y2"] - stats["sum_y"] * stats["sum_y"])
    )
    pearson_all = None if denom <= 0 else float(
        (stats["total_cnt"] * stats["sum_xy"] - stats["sum_x"] * stats["sum_y"]) / denom
    )
    return {
        "l2_rmse": rmse,
        "mse": float(mse),
        "mae": float(mae),
        "pearson_all": pearson_all,
        "pearson_cell_mean": (
            float(np.mean(stats["cell_pearsons"])) if stats["cell_pearsons"] else None
        ),
        "spearman_cell_mean": (
            float(np.mean(stats["cell_spearmans"])) if stats["cell_spearmans"] else None
        ),
        "pearson_top20_cell_mean": (
            float(np.mean(stats["cell_pearsons_top20"])) if stats["cell_pearsons_top20"] else None
        ),
        "n_cells_for_corr": int(len(stats["cell_pearsons"])),
        "n_eval_positions": int(stats["total_cnt"]),
    }


def build_topk_nonzero_eval_mask(true_cpu: np.ndarray, base_mask_cpu: np.ndarray, top_k: int) -> np.ndarray:
    topk_mask = np.zeros_like(base_mask_cpu, dtype=bool)
    if top_k <= 0:
        return topk_mask

    for i in range(true_cpu.shape[0]):
        nz = np.flatnonzero(true_cpu[i] > 0)
        if nz.size == 0:
            continue
        k = min(int(top_k), int(nz.size))
        if k == nz.size:
            top_idx = nz
        else:
            vals = true_cpu[i, nz]
            top_idx = nz[np.argpartition(vals, -k)[-k:]]
        topk_mask[i, top_idx] = True

    return topk_mask & base_mask_cpu.astype(bool, copy=False)
