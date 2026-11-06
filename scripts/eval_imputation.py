"""Score imputation predictions against simple baselines (eval env; NumPy + scikit-learn, no torch).

    python scripts/eval_imputation.py --task genes --data /data/brainwm/eval_10k \
        --pred Cajal@10000=runs/eval/genes_cajal_step10000.npz --out runs/eval/genes_scores/Cajal@10000
    python scripts/eval_imputation.py --task spatial --data /data/brainwm/eval_spatial --pred ... --out ...

Predictions come from predict_eval.py --task genes|spatial (per-cell sufficient statistics).

Gene imputation (the training gene mask, fixed seed, on every evaluation cell):
  gene mean      each gene's mean log1p CP10K over pretraining cells whose panel measures it
  kNN            same-panel pretraining cells nearest to the cell's visible expression (PCA of the
                 panel's most variable genes, masked genes zeroed); masked values = their mean
Spatial imputation (each centre cell hidden, rebuilt from its first `--neighbors` neighbours):
  neighbour mean the neighbours' mean over the centre's panel, averaged in log1p CP10K space and in CP10K
                 space then log1p (the form of the model's decoded expectation); cell type = their
                 majority label (the neighbours' labels are given to it, not to the model)

Metrics (brainwm_unified.impute_metrics): median per-cell Pearson, MSE (all and non-zero targets),
global Pearson; spatial adds the hidden cell's fine cell-type accuracy and macro-F1 (cells whose
type occurs in pretraining). Baselines are cached under --cache. Writes <out>/summary.csv, one row
per prediction and baseline, in the format collect_scores.py merges.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from brainwm_unified.impute_metrics import STATS, panel_profile, row_stats, summarize

METRICS = {
    "pearson_cell_median": "Pearson (cell median)",
    "mse": "MSE",
    "mse_nonzero": "MSE non-zero",
    "pearson_global": "Pearson (global)",
}
TASK_NAME = {"genes": "Gene imputation", "spatial": "Spatial imputation"}


class Cells:
    """The arrays of a subset directory that scoring needs (values memory-mapped)."""

    def __init__(self, root: str):
        def load(name, mmap=None):
            return np.load(os.path.join(root, name + ".npy"), mmap_mode=mmap)

        self.root = root
        self.row_ptr = load("row_ptr").astype(np.int64)
        self.gene_index = load("gene_index", "r")
        self.values = load("values", "r")
        self.panel_id = load("panel_id").astype(np.int64)
        self.panel_ptr = load("panel_gene_ptr").astype(np.int64)
        self.panel_index = load("panel_gene_index").astype(np.int64)
        self.cell_type = load("cell_type").astype(np.int64)
        self.neighbors = load("neighbors").astype(np.int64)
        self.n = len(self.row_ptr) - 1

    def panel(self, p: int) -> np.ndarray:
        return np.sort(self.panel_index[self.panel_ptr[p] : self.panel_ptr[p + 1]])

    def csr(self, cells: np.ndarray, n_genes: int) -> sparse.csr_matrix:
        """log1p CP10K rows of `cells` over all genes."""
        lo, hi = self.row_ptr[cells], self.row_ptr[cells + 1]
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(cells) else np.zeros(0, np.int64)
        vals = np.asarray(self.values[idx], np.float32)
        lens = hi - lo
        lib = np.maximum(np.add.reduceat(vals, np.r_[0, np.cumsum(lens)[:-1]]) if len(vals) else 0, 1.0)
        data = np.log1p(vals / np.repeat(lib, lens) * 1e4).astype(np.float32)
        ptr = np.r_[0, np.cumsum(lens)]
        return sparse.csr_matrix((data, np.asarray(self.gene_index[idx], np.int64), ptr), shape=(len(cells), n_genes))


def gene_means(ref: Cells, n_genes: int, n_cells: int, seed: int) -> np.ndarray:
    """Per gene: mean log1p CP10K over sampled reference cells whose panel measures it (0 if none)."""
    rng = np.random.default_rng(seed)
    cells = np.sort(rng.choice(ref.n, size=min(n_cells, ref.n), replace=False))
    total = np.zeros(n_genes)
    for chunk in np.array_split(cells, max(1, len(cells) // 5000)):
        total += np.asarray(ref.csr(chunk, n_genes).sum(0)).ravel()
    measured = np.zeros(n_genes)
    panels, counts = np.unique(ref.panel_id[cells], return_counts=True)
    for p, c in zip(panels, counts):
        measured[ref.panel(p)] += c
    return np.where(measured > 0, total / np.maximum(measured, 1), 0.0)


def knn_genes(ev: Cells, ref: Cells, mask: dict, n_genes: int, args) -> np.ndarray:
    """kNN prediction for every masked token, panel by panel; cells whose panel the reference lacks get NaN."""
    rng = np.random.default_rng(args.seed)
    pred = np.full(len(mask["gene"]), np.nan, np.float32)
    ptr = mask["ptr"]
    for p in np.unique(ev.panel_id):
        q = np.flatnonzero(ev.panel_id == p)
        pool = np.flatnonzero(ref.panel_id == p)
        if len(pool) < args.k + 1:
            continue
        pool = np.sort(rng.choice(pool, size=min(args.knn_ref, len(pool)), replace=False))
        R = ref.csr(pool, n_genes)
        mean = np.asarray(R.mean(0)).ravel()
        var = np.asarray(R.multiply(R).mean(0)).ravel() - mean**2
        top = np.argsort(var)[::-1][: args.knn_genes]
        Q = ev.csr(q, n_genes).tolil()
        for i, c in enumerate(q):  # hide the masked genes from the query
            Q[i, mask["gene"][ptr[c] : ptr[c + 1]]] = 0
        Q = Q.tocsr()
        pca = PCA(min(args.knn_pcs, len(top), len(pool) - 1), random_state=0).fit(R[:, top].toarray())
        nn = NearestNeighbors(n_neighbors=args.k).fit(pca.transform(R[:, top].toarray()))
        _, nbrs = nn.kneighbors(pca.transform(Q[:, top].toarray()))
        for i, c in enumerate(q):
            sl = slice(ptr[c], ptr[c + 1])
            pred[sl] = np.asarray(R[nbrs[i]][:, mask["gene"][sl]].mean(0)).ravel()
    return pred


def spatial_baselines(ev: Cells, centers: np.ndarray, k: int, seen_types: np.ndarray) -> dict[str, dict]:
    """Neighbour means over the centre's panel, averaged in log space and in CP10K space (then log1p, the
    form of the model's decoded expectation), each with the neighbours' majority cell type (their labels known)."""
    log_mean, lin_mean, trues, majority = [], [], [], []
    for c in centers:
        genes = ev.panel(ev.panel_id[c])
        nb = ev.neighbors[c, :k]
        nb = nb[nb >= 0]
        prof = np.stack([panel_profile(ev.row_ptr, ev.gene_index, ev.values, m, genes) for m in nb])
        log_mean.append(prof.mean(0))
        lin_mean.append(np.log1p(np.expm1(prof).mean(0)))
        trues.append(panel_profile(ev.row_ptr, ev.gene_index, ev.values, c, genes))
        majority.append(np.bincount(ev.cell_type[nb]).argmax())
    ptr = np.r_[0, np.cumsum([len(x) for x in trues])]
    true = np.concatenate(trues)
    types = cell_type_scores(np.asarray(majority), ev.cell_type[centers], seen_types)
    return {
        "neighbour mean (log space)": summarize(row_stats(np.concatenate(log_mean), true, ptr)) | types,
        "neighbour mean (CP10K space)": summarize(row_stats(np.concatenate(lin_mean), true, ptr)) | types,
    }


def cell_type_scores(pred: np.ndarray, true: np.ndarray, seen: np.ndarray) -> dict:
    keep = np.isin(true, seen)
    return {
        "cell_type_acc": float((pred[keep] == true[keep]).mean()),
        "cell_type_f1": float(f1_score(true[keep], pred[keep], average="macro")),
    }


def columns(task: str) -> list[tuple[str, str]]:
    cols = [(k, f"{TASK_NAME[task]} {v}") for k, v in METRICS.items()]
    if task == "spatial":
        cols += [("cell_type_acc", "Spatial cell type acc"), ("cell_type_f1", "Spatial cell type macro-F1")]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["genes", "spatial"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrain", default="/data/brainwm/unified_1m")
    ap.add_argument("--pred", action="append", default=[], help="NAME=PATH from predict_eval.py --task (repeatable)")
    ap.add_argument("--mask", default=None, help="genes: masked tokens (default: next to the first --pred)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="runs/eval/imputation_baselines")
    ap.add_argument("--seed", type=int, default=0, help="the mask seed predictions were made with")
    ap.add_argument("--n-genes", type=int, default=65918)
    ap.add_argument("--mean-cells", type=int, default=100_000, help="genes: reference cells for gene means")
    ap.add_argument("--knn-ref", type=int, default=5000, help="genes: reference cells per panel for kNN")
    ap.add_argument("--knn-genes", type=int, default=2000)
    ap.add_argument("--knn-pcs", type=int, default=50)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--neighbors", type=int, default=16, help="spatial: neighbours used (DataConfig.neighbors)")
    args = ap.parse_args()

    ev = Cells(args.data)
    rows = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        z = np.load(path)
        if str(z["task"]) != args.task:
            raise SystemExit(f"{path} holds task {z['task']}, not {args.task}")
        rows[name] = summarize({k: z[k] for k in STATS})
        if args.task == "spatial":
            fine = max(
                (k for k in z.files if k.startswith("cell_type_logits_")), key=lambda k: int(k.rsplit("_", 1)[1])
            )
            seen = np.unique(np.load(os.path.join(args.pretrain, "cell_type.npy")))
            rows[name] |= cell_type_scores(z[fine].argmax(-1), z["cell_type"], seen)

    os.makedirs(args.cache, exist_ok=True)
    cache = os.path.join(
        args.cache, f"{args.task}_{os.path.basename(os.path.normpath(args.data))}_seed{args.seed}.json"
    )
    if os.path.exists(cache):
        with open(cache) as fh:
            rows |= json.load(fh)
    else:
        ref = Cells(args.pretrain)
        base = {}
        if args.task == "genes":
            mask_file = args.mask or os.path.join(
                os.path.dirname(os.path.abspath(args.pred[0].split("=", 1)[1])),
                f"genes_mask_{os.path.basename(os.path.normpath(args.data))}_seed{args.seed}.npz",
            )
            mask = dict(np.load(mask_file))
            means = gene_means(ref, args.n_genes, args.mean_cells, args.seed)
            base["gene mean"] = summarize(row_stats(means[mask["gene"]], mask["target"], mask["ptr"]))
            knn = knn_genes(ev, ref, mask, args.n_genes, args)
            missing = np.isnan(knn)
            knn[missing] = means[mask["gene"][missing]]  # panels absent from the reference fall back to the mean
            base[f"kNN (k={args.k}, same panel)"] = summarize(row_stats(knn, mask["target"], mask["ptr"]))
            base[f"kNN (k={args.k}, same panel)"]["fallback_tokens"] = float(missing.mean())
        else:
            centers = np.flatnonzero(ev.neighbors[:, 0] >= 0)
            seen = np.unique(np.load(os.path.join(args.pretrain, "cell_type.npy")))
            base |= spatial_baselines(ev, centers, args.neighbors, seen)
        with open(cache, "w") as fh:
            json.dump(base, fh, indent=1)
        rows |= base

    os.makedirs(args.out, exist_ok=True)
    cols = columns(args.task)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", *(label for _, label in cols)])
        for name, m in rows.items():
            w.writerow([name, *(m.get(k, "") for k, _ in cols)])
    for name, m in rows.items():
        print(f"{name:28s} " + " ".join(f"{k}={m[k]:.4f}" for k, _ in cols if k in m))


if __name__ == "__main__":
    main()
