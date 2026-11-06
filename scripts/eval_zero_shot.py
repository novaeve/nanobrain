"""Score zero-shot outputs against ML baselines on an evaluation subset (eval env, GPU, no torch).

    source scripts/eval_env.sh
    python scripts/eval_zero_shot.py --pred Cajal@4000=runs/eval/cajal_step4000.npz --out runs/eval/step4000

Annotation: Cajal's label heads vs logistic regression and XGBoost trained on the same
pretraining labels (unified_1m), all on the same evaluation cells. Baseline features are
log1p(CP10K) expression of 2000 highly variable genes chosen on the pretraining set.
Metrics at the fine and coarse levels: accuracy, balanced accuracy, macro-F1.
Logistic regression runs on the GPU (cuML, L-BFGS), XGBoost on the GPU (hist).

Integration: scib-metrics (JAX on the GPU; k-NN by cuML brute force) on each model's latent
mean vs unintegrated PCA of the same features (batch = integration_batch, label = cell type).
Several models (--pred NAME=PATH, repeatable) are scored together, one table row each.

Baselines are fitted once and cached under --cache (later checkpoints score in seconds):
    eval_zero_shot.py --prepare; then eval_zero_shot.py --fit lr/cell_type (etc., in parallel).
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

N_GENES = 65918


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_counts(root: str) -> sparse.csr_matrix:
    def arr(name):
        return np.load(os.path.join(root, name + ".npy"), mmap_mode="r")

    return sparse.csr_matrix(
        (arr("values"), arr("gene_index"), arr("row_ptr")), shape=(len(arr("row_ptr")) - 1, N_GENES)
    )


def log_normalize(x: sparse.csr_matrix) -> sparse.csr_matrix:
    x = x.astype(np.float32)
    totals = np.asarray(x.sum(1)).ravel()
    x.data *= np.repeat(np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0), np.diff(x.indptr))
    np.log1p(x.data, out=x.data)
    return x


def highly_variable(x: sparse.csr_matrix, n: int) -> np.ndarray:
    """Top-n genes by variance of log-normalized expression."""
    mean = np.asarray(x.mean(0)).ravel()
    sq = np.asarray(x.multiply(x).mean(0)).ravel()
    return np.sort(np.argsort(-(sq - mean**2))[:n])


def level_parents(root: str) -> dict:
    """Fine label -> coarse label index, for each label kind."""
    with open(os.path.join(root, "cell_type_vocab.json")) as fh:
        ct = json.load(fh)
    with open(os.path.join(root, "anatomy_vocab.json")) as fh:
        an = json.load(fh)
    return {
        "cell_type": np.array([c["parent_lineage_index"] for c in ct["cell_type"]]),
        "anatomy": np.array([a["parent_region_index"] for a in an["anatomy"]]),
    }


def scores(true, pred) -> dict:
    keep = true >= 0
    t, p = true[keep], pred[keep]
    return {
        "accuracy": accuracy_score(t, p),
        "balanced_accuracy": balanced_accuracy_score(t, p),
        "macro_f1": f1_score(t, p, average="macro", labels=np.unique(t)),
    }


KINDS = ("cell_type", "anatomy")
MODELS = [f"{m}/{k}" for m in ("lr", "xgb") for k in KINDS]


def prepare_features(args):
    """HVGs, scaler and PCA fitted on the pretraining subset; its scaled features cached as a .npy."""
    path = os.path.join(args.cache, "features.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    log("loading pretraining counts")
    x = log_normalize(load_counts(args.pretrain))
    genes = highly_variable(x, args.hvg)
    feats = x[:, genes].toarray()
    del x
    scaler = StandardScaler().fit(feats)
    z = scaler.transform(feats).astype(np.float32)
    pca = PCA(50, random_state=0).fit(z[np.random.default_rng(0).choice(len(z), 200_000, replace=False)])
    os.makedirs(args.cache, exist_ok=True)
    np.save(os.path.join(args.cache, "train_features.npy"), z)
    prep = {"genes": genes, "scaler": scaler, "pca": pca}
    with open(path, "wb") as fh:
        pickle.dump(prep, fh)
    log(f"features cached: {z.shape}")
    return prep


class LinearClassifier:
    """A fitted linear classifier as plain arrays (argmax of x W^T + b), so it loads in any library version."""

    def __init__(self, coef, intercept, classes):
        self.coef, self.intercept, self.classes = (np.asarray(a) for a in (coef, intercept, classes))

    @classmethod
    def from_model(cls, model):
        return cls(model.coef_, model.intercept_, model.classes_)

    def save(self, path: str):
        np.savez(path, coef=self.coef, intercept=self.intercept, classes=self.classes)

    @classmethod
    def load(cls, path: str):
        z = np.load(path)
        return cls(z["coef"], z["intercept"], z["classes"])

    def predict(self, x):
        return self.classes[np.argmax(np.asarray(x) @ self.coef.T + self.intercept, axis=1)]


def fit_model(args, name: str):
    """Fit one baseline ("lr/<kind>" or "xgb/<kind>") on the cached pretraining features."""
    model, kind = name.split("/")
    z = np.load(os.path.join(args.cache, "train_features.npy"), mmap_mode="r")
    y = np.load(os.path.join(args.pretrain, kind + ".npy")).astype(np.int64)
    log(f"{name}: fitting on {z.shape}")
    if model == "lr":
        from cuml.linear_model import LogisticRegression as GPULogisticRegression

        lr = GPULogisticRegression(max_iter=300, C=1.0, output_type="numpy").fit(np.asarray(z), y)
        LinearClassifier.from_model(lr).save(os.path.join(args.cache, name.replace("/", "_") + ".npz"))
        log(f"{name}: done")
        return
    else:
        import xgboost as xgb

        classes, y_enc = np.unique(y, return_inverse=True)
        val = np.random.default_rng(0).random(len(y)) < 0.05  # early stopping on held-out pretraining cells
        clf = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            device=args.xgb_device,
            early_stopping_rounds=20,
            eval_metric="mlogloss",
        )
        clf.fit(z[~val], y_enc[~val], eval_set=[(z[val], y_enc[val])], verbose=50)
        fitted = (clf, classes)
    with open(os.path.join(args.cache, name.replace("/", "_") + ".pkl"), "wb") as fh:
        pickle.dump(fitted, fh)
    log(f"{name}: done")


def load_baselines(args) -> dict:
    """Features plus every fitted baseline; one still fitting is left out (its row shows '-')."""
    models = dict(prepare_features(args))
    for name in MODELS:
        path = os.path.join(args.cache, name.replace("/", "_") + ".pkl")
        arrays = os.path.join(args.cache, name.replace("/", "_") + ".npz")
        if name.startswith("lr/") and os.path.exists(arrays):  # linear baselines are stored as arrays
            models[name] = LinearClassifier.load(arrays)
            continue
        if not os.path.exists(path):
            log(f"{name} not fitted yet; skipped")
            continue
        with open(path, "rb") as fh:
            models[name] = pickle.load(fh)
    return models


def gpu_neighbors(x: np.ndarray, k: int):
    """Exact k-NN on the GPU (cuML brute force), in scib-metrics' format."""
    from cuml.neighbors import NearestNeighbors
    from scib_metrics.nearest_neighbors import NeighborsResults

    dist, idx = NearestNeighbors(n_neighbors=k, algorithm="brute", output_type="numpy").fit(x).kneighbors(x)
    return NeighborsResults(indices=np.asarray(idx), distances=np.asarray(dist))


def summary_table(results: dict, integration):
    """One row per method, one column per metric; NaN (shown '-') where a method does not do a task."""
    import pandas as pd

    rows = {}
    for key, r in results["annotation"].items():
        kind, name = key.split("/", 1)
        label = {"cell_type": "Cell type", "anatomy": "Region"}[kind]
        for level in ("coarse", "fine"):
            rows.setdefault(name, {})[f"{label} {level} macro-F1"] = r[level]["macro_f1"]
            rows[name][f"{label} {level} bal.acc"] = r[level]["balanced_accuracy"]
    for emb in integration.index:
        if emb == "Metric Type":
            continue
        for col, metric in (
            ("Bio conservation", "Integration bio"),
            ("Batch correction", "Integration batch"),
            ("Total", "Integration total"),
        ):
            rows.setdefault(emb, {})[metric] = float(integration.loc[emb, col])
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[], help="NAME=npz from scripts/predict_eval.py (repeatable)")
    ap.add_argument("--data", default="/data/brainwm/eval_10k")
    ap.add_argument("--pretrain", default="/data/brainwm/unified_1m")
    ap.add_argument("--out", help="output directory for scoring")
    ap.add_argument("--cache", default="runs/eval/baselines")
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--xgb-device", default="cuda")
    ap.add_argument("--prepare", action="store_true", help="only cache the pretraining features")
    ap.add_argument("--fit", choices=MODELS, help="only fit this baseline (run the four in parallel)")
    args = ap.parse_args()

    if args.prepare:
        return prepare_features(args)
    if args.fit:
        return fit_model(args, args.fit)
    if not (args.pred and args.out):
        ap.error("scoring needs --pred NAME=PATH and --out")
    preds = {name: np.load(path) for name, path in (p.split("=", 1) for p in args.pred)}
    ref = next(iter(preds.values()))
    for p in preds.values():  # every model must have scored the same cells
        assert np.array_equal(p["cell_type"], ref["cell_type"]) and np.array_equal(p["batch"], ref["batch"])
    parents = level_parents(args.data)
    models = load_baselines(args)
    x = log_normalize(load_counts(args.data))[:, models["genes"]].toarray()
    z = models["scaler"].transform(x).astype(np.float32)

    results = {"steps": {n: int(p["step"]) for n, p in preds.items()}, "cells": len(z), "annotation": {}}
    for kind in KINDS:
        true = ref[kind].astype(np.int64)
        guesses = {}
        for name, p in preds.items():
            n_levels = len([k for k in p.files if k.startswith(kind + "_logits_")])
            if n_levels == 0:  # an embedding without this label head (e.g. scVI): integration only
                continue
            if n_levels == 1:  # fine labels only (e.g. scANVI): coarse from the fine prediction
                guesses[name] = (p[f"{kind}_logits_0"].argmax(-1), None)
                continue
            coarse_level = 0 if kind == "cell_type" else n_levels - 2  # lineage; region
            guesses[name] = (
                p[f"{kind}_logits_{n_levels - 1}"].argmax(-1),
                p[f"{kind}_logits_{coarse_level}"].argmax(-1),
            )
        if f"lr/{kind}" in models:
            guesses["LogReg"] = (np.asarray(models[f"lr/{kind}"].predict(z)), None)
        if f"xgb/{kind}" in models:
            clf, classes = models[f"xgb/{kind}"]
            guesses["XGBoost"] = (classes[clf.predict(z)], None)
        for name, (f, c) in guesses.items():
            c = parents[kind][f] if c is None else c  # baselines: coarse label of the fine prediction
            results["annotation"][f"{kind}/{name}"] = {
                "fine": scores(true, f),
                "coarse": scores(parents[kind][true], c),
            }

    log("integration metrics")
    import anndata as ad
    from scib_metrics.benchmark import Benchmarker

    adata = ad.AnnData(obs={"batch": ref["batch"].astype(str), "cell_type": ref["cell_type"].astype(str)})
    adata.obsm["PCA (unintegrated)"] = models["pca"].transform(z)
    for name, p in preds.items():
        adata.obsm[name] = p["mu"]
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=["PCA (unintegrated)", *preds],
        pre_integrated_embedding_obsm_key="PCA (unintegrated)",
        n_jobs=16,
    )
    bm.prepare(neighbor_computer=gpu_neighbors)
    bm.benchmark()
    table = bm.get_results(min_max_scale=False)
    results["integration"] = json.loads(table.to_json(orient="index"))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    summary = summary_table(results, table)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from brainwm_unified import record

    record.write(
        args.out,
        "eval",
        predictions=dict(p.split("=", 1) for p in args.pred),
        steps=results["steps"],
        data=record.data_fingerprint(args.data),
        baselines=sorted(k for k in models if "/" in k),
        summary=json.loads(summary.to_json(orient="index")),
    )
    print(f"{results['cells']} evaluation cells; checkpoints {results['steps']}")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}", na_rep="-"))


if __name__ == "__main__":
    main()
