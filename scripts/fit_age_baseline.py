"""Opponent for the age task: ridge regression of log days post fertilization on log1p-CP10K expression, with a
split-conformal interval (residual quantiles on a held-back calibration part of the training cells).

    .venv/bin/python scripts/fit_age_baseline.py --train-root /data/brainwm/unified_1m \
        --holdout runs/age/<run>/age_holdout.npz --eval-root /data/brainwm/eval_10k \
        --out runs/eval/age/ridge_eval_10k.npz

Training cells: a seeded sample of cells with a closed age label outside the held-out groups (the cells whose
ages the model's loss sees). Writes age_log_lo / age_log_hi / age_log_centre for every evaluation cell.
"""

import argparse
import os

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from brainwm_unified import age
from brainwm_unified.data import Subset


def features(ds: Subset, cells: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """log1p of counts per 10k over all genes, as a sparse cells x genes matrix."""
    starts, ends = ds.row_ptr[cells], ds.row_ptr[cells + 1]
    idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    vals = np.asarray(ds.values[idx], np.float64)
    genes = np.asarray(ds.gene_index[idx], np.int64)
    indptr = np.concatenate([[0], np.cumsum(ends - starts)])
    x = sp.csr_matrix((vals, genes, indptr), shape=(len(cells), n_genes))
    lib = np.asarray(x.sum(1)).ravel()
    x = sp.diags(1e4 / np.maximum(lib, 1)) @ x
    x.data = np.log1p(x.data)
    return x.tocsr()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", default="/data/brainwm/unified_1m")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--calibration", type=float, default=0.2)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.1, help="interval miscoverage")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train = Subset(args.train_root)
    lo = np.load(os.path.join(args.train_root, "age_lower_dpf.npy"))
    hi = np.load(os.path.join(args.train_root, "age_upper_dpf.npy"))
    h = np.load(args.holdout)
    group = np.load(os.path.join(args.train_root, str(h["unit"]) + ".npy"))
    ok = np.flatnonzero(np.isfinite(lo) & np.isfinite(hi) & ~np.isin(group, h["ids"]))
    rng = np.random.default_rng(args.seed)
    cells = np.sort(rng.choice(ok, size=min(args.n_train, len(ok)), replace=False))
    ev = Subset(args.eval_root)
    n_genes = int(max(train.panel_gene_index.max(), ev.panel_gene_index.max())) + 1  # one feature space
    y = np.log(age.label_center(lo[cells], hi[cells]))
    cal = rng.random(len(cells)) < args.calibration
    x = features(train, cells, n_genes)
    model = Ridge(alpha=args.ridge_alpha, solver="sparse_cg").fit(x[~cal], y[~cal])
    resid = y[cal] - model.predict(x[cal])
    q_lo, q_hi = np.quantile(resid, [args.alpha / 2, 1 - args.alpha / 2])  # split conformal, two-sided
    centre = model.predict(features(ev, np.arange(ev.n_cells), n_genes))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, age_log_lo=centre + q_lo, age_log_hi=centre + q_hi, age_log_centre=centre)
    print(
        f"ridge on {(~cal).sum()} cells, calibration {cal.sum()}: residual quantiles {q_lo:.3f} {q_hi:.3f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
