"""SCTransform-style opponents: analytic Pearson residuals (Lause, Berens & Kobak 2021, the closed-form
sctransform), alone and with Harmony, embedding evaluation sets for eval_zero_shot.py.

    .venv/bin/python scripts/fit_pearson.py --out runs/eval/pearson --eval /data/brainwm/eval_integration_10k

Features are the 2000 HVGs every other opponent uses (eval_zero_shot.py --prepare). Each design changes
one thing against an embedding that already exists, so the effect of each step can be read off:

  Pearson PCA                  residuals -> 50-PC PCA, both fitted on the pretraining subset and
                               applied to the evaluation cells (zero-shot, like Cajal). Against
                               "PCA (unintegrated)" (log1p CP10K -> scale -> PCA, same fit): the
                               normalisation alone.
  Pearson + Harmony            Harmony on the evaluation cells' Pearson PCs with their own batches
  (transductive)               (it sees the evaluation set, like scVI (transductive)).
  log PCA + Harmony            Harmony on "PCA (unintegrated)": against the line above, Pearson
  (transductive)               residuals versus log normalisation under the same integration.

Residuals: mu_cg = n_c p_g with n_c the cell's total UMIs over all genes and p_g the gene's share of
all pretraining UMIs; r = (x - mu) / sqrt(mu + mu^2 / theta), theta = 100, clipped to +-sqrt(N), N the
pretraining cell count (the paper's defaults, as in scanpy's normalize_pearson_residuals). Writes
predict_eval.py-format npz files (mu) via fit_scvi.save_embedding.
"""

import argparse
import os
import pickle
import time

import harmonypy
import numpy as np
import pandas as pd
from eval_zero_shot import load_counts, log_normalize
from fit_scvi import hvg_counts, log, save_embedding
from sklearn.decomposition import PCA

THETA = 100.0


def library(root: str) -> np.ndarray:
    """Total UMIs per cell over every measured gene."""
    row_ptr = np.load(os.path.join(root, "row_ptr.npy"))
    values = np.load(os.path.join(root, "values.npy"), mmap_mode="r")
    cum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    return np.diff(cum[row_ptr])


def residuals(x, n: np.ndarray, p: np.ndarray, clip: float) -> np.ndarray:
    """Analytic Pearson residuals of a cells x genes count matrix (dense float32 out)."""
    mu = n[:, None] * p[None, :]
    sd = np.sqrt(mu + mu * mu / THETA)
    r = np.divide(x.toarray() - mu, sd, out=np.zeros_like(mu), where=sd > 0)  # mu = 0 (empty cell or gene): 0
    return np.clip(r, -clip, clip).astype(np.float32)


def harmony(pcs: np.ndarray, batch: np.ndarray, seed: int) -> np.ndarray:
    ho = harmonypy.run_harmony(pcs, pd.DataFrame({"batch": batch.astype(str)}), ["batch"], random_state=seed)
    z = np.asarray(ho.Z_corr)
    return (z if z.shape[0] == len(pcs) else z.T).astype(np.float32)  # harmonypy 2.x: cells x dims; 0.0.x: dims x cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", default="/data/brainwm/unified_1m")
    ap.add_argument("--features", default="runs/eval/baselines/features.pkl", help="eval_zero_shot --prepare cache")
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-cells", type=int, default=200_000, help="cells the PCA is fitted on (as for the log PCA)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.features, "rb") as fh:
        prep = pickle.load(fh)
    genes = np.asarray(prep["genes"])

    model_path = os.path.join(args.out, "pearson_pca.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as fh:
            m = pickle.load(fh)
    else:
        log("loading pretraining counts")
        x = hvg_counts(args.pretrain, genes)
        n_all = library(args.pretrain)
        p = np.asarray(x.sum(0)).ravel() / n_all.sum()  # each HVG's share of all UMIs
        clip = float(np.sqrt(x.shape[0]))
        idx = np.sort(np.random.default_rng(args.seed).choice(x.shape[0], args.fit_cells, replace=False))
        r = residuals(x[idx], n_all[idx], p, clip)
        pca = PCA(50, random_state=args.seed).fit(r)
        m = {"genes": genes, "p": p, "clip": clip, "pca": pca}
        with open(model_path, "wb") as fh:
            pickle.dump(m, fh)
        log(f"Pearson PCA fitted on {len(idx)} cells; {pca.explained_variance_ratio_.sum():.3f} variance in 50 PCs")

    for root in args.eval:
        tag = os.path.basename(os.path.normpath(root))
        batch = np.load(os.path.join(root, "integration_batch.npy"))
        x = hvg_counts(root, genes)
        t = time.time()
        pcs = m["pca"].transform(residuals(x, library(root), m["p"], m["clip"])).astype(np.float32)
        save_embedding(os.path.join(args.out, f"{tag}_pearson_pca.npz"), root, pcs, time.time() - t)
        t = time.time()
        save_embedding(
            os.path.join(args.out, f"{tag}_pearson_harmony_transductive.npz"),
            root,
            harmony(pcs, batch, args.seed),
            time.time() - t,
        )
        # the same Harmony on the "PCA (unintegrated)" opponent, computed exactly as eval_zero_shot.py does
        logx = log_normalize(load_counts(root))[:, genes].toarray()
        log_pcs = prep["pca"].transform(prep["scaler"].transform(logx)).astype(np.float32)
        t = time.time()
        save_embedding(
            os.path.join(args.out, f"{tag}_log_harmony_transductive.npz"),
            root,
            harmony(log_pcs, batch, args.seed),
            time.time() - t,
        )


if __name__ == "__main__":
    main()
