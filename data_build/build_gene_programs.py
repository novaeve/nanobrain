"""Data-driven gene programs for the GeneProgram mask: co-expression modules of variable genes.

    .venv/bin/python data_build/build_gene_programs.py --data /data/brainwm/unified_1m

Samples cells, log-normalises them (log1p CP10K), keeps the most variable genes, clusters
genes by their correlation profiles (k-means) and keeps programs whose genes are, on
average, correlated with each other. Writes <data>/gene_programs.npy: program id per gene
(-1 for genes in no program) and gene_programs.json with the members and their coherence.
"""

import argparse
import json
import os

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

N_GENES = 65918


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/brainwm/unified_1m")
    ap.add_argument("--cells", type=int, default=200_000)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--programs", type=int, default=100)
    ap.add_argument("--min-coherence", type=float, default=0.1, help="mean within-program correlation to keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    def arr(name):
        return np.load(os.path.join(args.data, name + ".npy"), mmap_mode="r")

    ptr = arr("row_ptr")
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(len(ptr) - 1, size=min(args.cells, len(ptr) - 1), replace=False))
    lo, hi = ptr[rows], ptr[rows + 1]
    take = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
    new_ptr = np.concatenate([[0], np.cumsum(hi - lo)])
    x = sparse.csr_matrix((arr("values")[take].astype(np.float32), arr("gene_index")[take], new_ptr),
                          shape=(len(rows), N_GENES))
    totals = np.asarray(x.sum(1)).ravel()
    x.data *= np.repeat(np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0), np.diff(x.indptr))
    np.log1p(x.data, out=x.data)

    mean = np.asarray(x.mean(0)).ravel()
    var = np.asarray(x.multiply(x).mean(0)).ravel() - mean**2
    genes = np.sort(np.argsort(-var)[: args.hvg])
    z = x[:, genes].toarray()
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-6)
    corr = (z.T @ z) / len(z)

    labels = KMeans(args.programs, n_init=4, random_state=args.seed).fit_predict(corr)
    program = np.full(N_GENES, -1, np.int32)
    kept = []
    for k in range(args.programs):
        idx = np.flatnonzero(labels == k)
        if len(idx) < 3:
            continue
        sub = corr[np.ix_(idx, idx)]
        coherence = float((sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
        if coherence < args.min_coherence:
            continue
        program[genes[idx]] = len(kept)
        kept.append({"id": len(kept), "genes": genes[idx].tolist(), "coherence": round(coherence, 4)})
    np.save(os.path.join(args.data, "gene_programs.npy"), program)
    with open(os.path.join(args.data, "gene_programs.json"), "w") as fh:
        json.dump({"cells": len(rows), "hvg": args.hvg, "kmeans": args.programs, "seed": args.seed,
                   "min_coherence": args.min_coherence, "programs": kept}, fh)
    sizes = [len(p["genes"]) for p in kept]
    print(f"{len(kept)} programs over {sum(sizes)} genes (size median {int(np.median(sizes))}, "
          f"max {max(sizes)}); coherence median {np.median([p['coherence'] for p in kept]):.3f}")


if __name__ == "__main__":
    main()
