"""scVI and scANVI baselines (scvi-tools on the GPU): fit on the pretraining subset, embed evaluation sets.

    .venv/bin/python scripts/fit_scvi.py --out runs/eval/scvi \\
        --eval /data/brainwm/eval_integration_10k /data/brainwm/eval_10k

Features: the 2000 HVGs the LogReg / XGBoost baselines use (eval_zero_shot.py --prepare), raw
counts; batch = integration_batch. scANVI is trained twice from the scVI model, once on the cell-type
labels and once on the anatomy (structure) labels, so it competes on both annotation tasks; its
integration latent is the cell-type model's. Settings follow scvi-tools'
atlas-integration guidance (2 layers, 30 latent dimensions, negative binomial); everything else,
minibatch size and epochs included, is scvi-tools' default (its epoch heuristic assumes minibatches of
128; larger ones would cut the number of updates and understate the baseline).

Evaluation cells are embedded zero-shot, as Cajal's are: the encoders do not take the batch
(scvi-tools' default), so the query's batch field only satisfies the registry. As a reference,
scVI is also fitted on each evaluation set itself with its own batches (transductive, the usual
scIB protocol; it sees the evaluation cells, Cajal does not). Each embedding is written as a
predict_eval.py-format npz (mu; scANVI also fine cell-type and anatomy logits) for eval_zero_shot.py.
Fitted models and finished embeddings are reused when the script runs again.
"""

import argparse
import json
import os
import pickle
import time

import anndata as ad
import numpy as np
import scvi
from scipy import sparse

N_GENES = 65918
SCVI_KW = {"n_layers": 2, "n_latent": 30, "gene_likelihood": "nb"}
LABELS = ("cell_type", "anatomy")  # one scANVI per label kind


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def hvg_counts(root: str, genes: np.ndarray, chunk: int = 50_000_000) -> sparse.csr_matrix:
    """Raw counts of every cell over `genes` (columns in that order), read in chunks of entries."""
    row_ptr = np.load(os.path.join(root, "row_ptr.npy"))
    gene_index = np.load(os.path.join(root, "gene_index.npy"), mmap_mode="r")
    values = np.load(os.path.join(root, "values.npy"), mmap_mode="r")
    col = np.full(N_GENES, -1, np.int64)
    col[genes] = np.arange(len(genes))
    rows, cols, vals = [], [], []
    row_of = np.repeat(np.arange(len(row_ptr) - 1), np.diff(row_ptr)) if row_ptr[-1] < chunk else None
    for lo in range(0, int(row_ptr[-1]), chunk):
        hi = min(lo + chunk, int(row_ptr[-1]))
        c = col[np.asarray(gene_index[lo:hi])]
        keep = c >= 0
        idx = np.flatnonzero(keep) + lo
        r = row_of[idx] if row_of is not None else np.searchsorted(row_ptr, idx, side="right") - 1
        rows.append(r)
        cols.append(c[keep])
        vals.append(np.asarray(values[lo:hi], np.float32)[keep])
    n = len(row_ptr) - 1
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, len(genes))
    )


def anndata(root: str, genes: np.ndarray, batch=None, labels=None) -> ad.AnnData:
    def arr(name):
        return np.load(os.path.join(root, name + ".npy"))

    n = len(arr("cell_type"))
    obs = {
        "batch": (arr("integration_batch") if batch is None else np.full(n, batch)).astype(str),
        **{k: (arr(k) if labels is None else np.full(n, labels)).astype(str) for k in LABELS},
    }
    x = hvg_counts(root, genes)
    a = ad.AnnData(X=x, obs=obs)
    a.var_names = [str(g) for g in genes]
    a.obs_names = [str(i) for i in range(a.n_obs)]
    return a


def save_embedding(path: str, root: str, mu: np.ndarray, seconds: float, **logits):
    def arr(name):
        return np.load(os.path.join(root, name + ".npy"))

    np.savez_compressed(
        path,
        mu=mu.astype(np.float32),
        cell_type=arr("cell_type"),
        anatomy=arr("anatomy"),
        batch=arr("integration_batch"),
        **{k: arr(k) for k in ("dataset", "technology", "assay")},
        **logits,
        step=0,
        seconds=seconds,
    )
    log(f"wrote {path} {mu.shape}")


def fine_logits(model, query: ad.AnnData, n_classes: int) -> np.ndarray:
    """scANVI's label probabilities placed at the global label ids (log; ids it never saw get -30)."""
    prob = model.predict(query, soft=True)
    out = np.full((query.n_obs, n_classes), -30.0, np.float32)
    for name in prob.columns:
        if name != "unknown":
            out[:, int(name)] = np.log(np.clip(prob[name].to_numpy(), 1e-12, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", default="/data/brainwm/unified_1m")
    ap.add_argument("--features", default="runs/eval/baselines/features.pkl", help="eval_zero_shot --prepare cache")
    ap.add_argument("--eval", nargs="+", required=True, help="evaluation set directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128, help="scvi-tools' default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    scvi.settings.seed = args.seed
    os.makedirs(args.out, exist_ok=True)
    with open(args.features, "rb") as fh:
        genes = np.asarray(pickle.load(fh)["genes"])
    train = {"accelerator": "gpu", "devices": 1, "batch_size": args.batch_size}

    log("loading the pretraining subset")
    ref = anndata(args.pretrain, genes)
    seen_batch = ref.obs["batch"].iloc[0]
    log(f"{ref.n_obs} cells x {ref.n_vars} genes, {ref.obs['batch'].nunique()} batches")
    scvi_path = os.path.join(args.out, "scvi_model")
    if os.path.exists(scvi_path):
        model = scvi.model.SCVI.load(scvi_path, adata=ref)
    else:
        scvi.model.SCVI.setup_anndata(ref, batch_key="batch")
        model = scvi.model.SCVI(ref, **SCVI_KW)
        t = time.time()
        model.train(**train)
        log(f"scVI trained in {time.time() - t:.0f} s")
        model.save(scvi_path)
    scanvi = {}
    for kind in LABELS:
        path = os.path.join(args.out, "scanvi_model" if kind == "cell_type" else f"scanvi_{kind}_model")
        if os.path.exists(path):
            scanvi[kind] = scvi.model.SCANVI.load(path, adata=ref)
            continue
        scanvi[kind] = scvi.model.SCANVI.from_scvi_model(model, unlabeled_category="unknown", labels_key=kind)
        t = time.time()
        scanvi[kind].train(n_samples_per_label=100, **train)
        log(f"scANVI ({kind}) trained in {time.time() - t:.0f} s")
        scanvi[kind].save(path)
    record = {"settings": SCVI_KW, "batch_size": args.batch_size, "genes": len(genes), "cells": ref.n_obs}

    for root in args.eval:
        tag = os.path.basename(os.path.normpath(root))
        n_classes = {}
        for kind in LABELS:
            with open(os.path.join(root, f"{kind}_vocab.json")) as fh:
                n_classes[kind] = len(json.load(fh)[kind])
        query = anndata(root, genes, batch=seen_batch, labels="unknown")  # encoders ignore the batch
        out = os.path.join(args.out, f"{tag}_scvi.npz")
        if not os.path.exists(out):
            t = time.time()
            save_embedding(out, root, model.get_latent_representation(query), time.time() - t)
        out = os.path.join(args.out, f"{tag}_scanvi.npz")
        if not os.path.exists(out) or "anatomy_logits_0" not in np.load(out).files:
            t = time.time()
            mu = scanvi["cell_type"].get_latent_representation(query)
            logits = {f"{k}_logits_0": fine_logits(scanvi[k], query, n_classes[k]) for k in LABELS}
            save_embedding(out, root, mu, time.time() - t, **logits)

        out = os.path.join(args.out, f"{tag}_scvi_transductive.npz")
        if os.path.exists(out):
            continue
        own = anndata(root, genes)  # transductive reference: fitted on the evaluation set with its batches
        scvi.model.SCVI.setup_anndata(own, batch_key="batch")
        local = scvi.model.SCVI(own, **SCVI_KW)
        t = time.time()
        local.train(**train)
        save_embedding(out, root, local.get_latent_representation(), time.time() - t)
    with open(os.path.join(args.out, "settings.json"), "w") as fh:
        json.dump(record, fh, indent=1)


if __name__ == "__main__":
    main()
