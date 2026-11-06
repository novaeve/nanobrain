"""Imputation metrics shared by model predictions and baselines (NumPy only, no torch).

Values are log1p(CP10K). A ragged set of rows is given as flat arrays plus `ptr` offsets
(row r is [ptr[r], ptr[r + 1])), e.g. the masked tokens of each cell or the panel genes of
each hidden spatial cell. Predictions are reduced to per-row sufficient statistics
(row_stats), small enough to store per checkpoint; `summarize` turns them into the metrics.
"""

import numpy as np

STATS = ("n", "sp", "st", "spp", "stt", "spt", "sse", "n_nz", "sse_nz")


def row_stats(pred: np.ndarray, true: np.ndarray, ptr: np.ndarray) -> dict[str, np.ndarray]:
    """Per row: count, sums of pred, true, their squares and product, squared error (all and non-zero targets)."""
    n = np.diff(ptr)
    row = np.repeat(np.arange(len(n)), n)
    p, t = pred.astype(np.float64), true.astype(np.float64)
    nz = t > 0

    def s(w):
        return np.bincount(row, w, len(n))

    return {
        "n": n.astype(np.float64),
        "sp": s(p),
        "st": s(t),
        "spp": s(p * p),
        "stt": s(t * t),
        "spt": s(p * t),
        "sse": s((p - t) ** 2),
        "n_nz": s(nz.astype(np.float64)),
        "sse_nz": s(np.where(nz, (p - t) ** 2, 0.0)),
    }


def _pearson(n, sp, st, spp, stt, spt):
    cov = spt - sp * st / n
    var = (spp - sp * sp / n) * (stt - st * st / n)
    return np.where(var > 1e-12, cov / np.sqrt(np.maximum(var, 1e-300)), np.nan)


def summarize(stats: dict, min_len: int = 10) -> dict[str, float]:
    """Median per-row Pearson (rows with >= min_len entries), MSE over all and non-zero targets, global Pearson."""
    st = {k: np.asarray(stats[k], np.float64) for k in STATS}
    ok = st["n"] >= min_len
    r = _pearson(*(st[k][ok] for k in ("n", "sp", "st", "spp", "stt", "spt")))
    tot = {k: st[k].sum() for k in STATS}
    return {
        "pearson_cell_median": float(np.nanmedian(r)),
        "mse": float(tot["sse"] / tot["n"]),
        "mse_nonzero": float(tot["sse_nz"] / max(tot["n_nz"], 1.0)),
        "pearson_global": float(_pearson(*(tot[k] for k in ("n", "sp", "st", "spp", "stt", "spt")))),
    }


def log1p_cp10k(values: np.ndarray, lib: float) -> np.ndarray:
    return np.log1p(values / max(lib, 1.0) * 1e4).astype(np.float32)


def panel_profile(row_ptr, gene_index, values, cell: int, genes: np.ndarray) -> np.ndarray:
    """log1p CP10K of `cell` (library = all its counts) over the sorted `genes`, zeros where not detected."""
    lo, hi = row_ptr[cell], row_ptr[cell + 1]
    g, v = gene_index[lo:hi], values[lo:hi]
    out = np.zeros(len(genes), np.float32)
    pos = np.searchsorted(genes, g)
    hit = (pos < len(genes)) & (genes[np.minimum(pos, len(genes) - 1)] == g)
    out[pos[hit]] = log1p_cp10k(np.asarray(v, np.float32), float(np.sum(v)))[hit]
    return out
