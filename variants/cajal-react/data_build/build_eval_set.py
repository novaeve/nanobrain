"""Build a balanced evaluation set (default 10K cells) from V4 cells that are NOT in unified_1m.

Reuses build_unified_1m for metadata, the expression writer and vocabularies, so indices
match the pretraining subset. Cells are drawn so that cell type, anatomy, dataset and
technology are each as even as possible (IPF). Only train and validation are read (never the sealed
test split). Cells must carry a cell type and an anatomy label that unified_1m also
contains. Spatial MERFISH cells may be drawn as single cells; every cell is stored
expression-only (spatial NaN, no neighbours), since the evaluations are per cell.

--design spatial instead takes whole tiles of verified MERFISH sections (the pretraining tile grid)
that hold no unified_1m or --disjoint-from cell, one tile at a time round-robin over datasets and
then sections, until --n cells; neighbours are found within the chosen tiles as in unified_1m.
This set is for spatial imputation (hide a cell, rebuild it from its neighbours).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_unified_1m as B  # noqa: E402


# Margins evened by IPF, the last one most exactly. Datasets (hence platforms) are spread
# within each cell type x anatomy stratum by the base weights, and each dataset is capped at
# a share of the set: forcing datasets even as a margin conflicts with the label margins
# (many types live in few datasets) and makes IPF diverge; a one-sided cap is always feasible.
BALANCE = ("anatomy", "cell_type")
SPREAD = "dataset"
CAPS = {"dataset": 0.05}


def water_fill(avail: np.ndarray, total: float) -> np.ndarray:
    """Most even per-class targets summing to `total`: min(avail, cap) with the cap solved by bisection."""
    lo, hi = 0.0, float(avail.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.minimum(avail, mid).sum() < total else (lo, mid)
    return np.minimum(avail, hi)


def select_balanced(meta, cand, n, iters=200):
    """Inclusion probabilities that even out the BALANCE margins and respect CAPS (capped IPF).

    p_i = min(1, w_i * prod_j u_j[class_j(i)]), with base weight w_i = 1 / size of the cell's
    (cell type, anatomy, SPREAD) stratum so that datasets share each label stratum equally.
    Each pass first shrinks the factors of classes over their CAPS share (never grows them),
    then rescales each BALANCE margin's factors toward its water-filled target.
    """
    names = (*CAPS, *BALANCE)
    keys = [np.unique(meta[k][cand], return_inverse=True)[1] for k in names]
    strata = np.unique(np.stack([meta[k][cand] for k in (*BALANCE, SPREAD)]), axis=1, return_inverse=True)[1]
    base = 1.0 / np.bincount(strata)[strata]
    targets = [np.full(k.max() + 1, CAPS[name] * n) if name in CAPS else water_fill(np.bincount(k).astype(float), n)
               for name, k in zip(names, keys)]
    factors = [np.ones(len(t)) for t in targets]

    def probs():
        return np.minimum(1.0, base * np.prod([f[kk] for f, kk in zip(factors, keys)], axis=0))

    for _ in range(iters):
        for j, (name, k, t) in enumerate(zip(names, keys, targets)):
            got = np.bincount(k, weights=probs(), minlength=len(t))
            ratio = np.where(got > 0, t / np.maximum(got, 1e-12), 1.0)
            factors[j] *= np.minimum(ratio, 1.0) if name in CAPS else ratio
    p = probs()
    p *= n / p.sum()  # exact expected size; the margins stay near their targets
    p = np.minimum(p, 1.0)
    fit = {name: float(np.abs(np.bincount(k, weights=p, minlength=len(t)) - t).sum() / n)
           for name, k, t in zip(names, keys, targets) if name in BALANCE}
    fit["max_capped_share"] = {name: float(np.bincount(k, weights=p).max() / n)
                               for name, k in zip(names, keys) if name in CAPS}
    return p, fit


def select_shared(meta, cand, n, rng, min_batches=3, min_cells=20):
    """Integration design: only cell types that at least `min_batches` batches each hold `min_cells` of,
    and equal shares per (cell type, batch) pair (water-filled), so batch and biology are not confounded."""
    ct, batch = meta["cell_type"][cand], meta["integration_batch"][cand]
    ok = batch >= 0
    pair = ct.astype(np.int64) * 100_000 + batch
    codes, inv, size = np.unique(pair[ok], return_inverse=True, return_counts=True)
    big = size >= min_cells
    pair_type = codes // 100_000
    n_batches = {t: k for t, k in zip(*np.unique(pair_type[big], return_counts=True))}
    keep = big & np.array([n_batches.get(t, 0) >= min_batches for t in pair_type])
    quota = np.zeros(len(codes))
    quota[keep] = water_fill(size[keep].astype(float), n)
    rows = cand[ok]
    order = np.argsort(inv, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(codes)))]
    picked = [rng.choice(rows[order[bounds[i]:bounds[i + 1]]], size=int(round(quota[i])), replace=False)
              for i in np.flatnonzero(quota >= 0.5)]
    info = {"cell_types": int(len(np.unique(pair_type[keep]))), "pairs": int(keep.sum()),
            "batches": int(len(np.unique(codes[keep] % 100_000))), "min_batches": min_batches, "min_cells": min_cells}
    return np.concatenate(picked), info


def select_tiles(meta, blocked, n, tile_um, rng):
    """Spatial design: whole tiles without a blocked row, round-robin over datasets then sections."""
    cand, tiles, tile_of = B.spatial_tiles(meta, tile_um)
    bad = np.bincount(tile_of, weights=blocked[cand].astype(float), minlength=len(tiles)) > 0
    size = np.bincount(tile_of, minlength=len(tiles))
    first = np.zeros(len(tiles), np.int64)
    first[tile_of[::-1]] = cand[::-1]  # any member row, for the tile's dataset and section
    pool = {}  # dataset -> section -> shuffled good tiles
    for t in rng.permutation(np.flatnonzero(~bad & (size > B.N_NEIGHBORS))):
        d, sct = int(meta["dataset"][first[t]]), int(meta["section"][first[t]])
        pool.setdefault(d, {}).setdefault(sct, []).append(t)
    chosen, total = [], 0
    datasets = list(rng.permutation(sorted(pool)))
    turn = {d: 0 for d in datasets}
    while total < n and datasets:
        for d in list(datasets):
            secs = [k for k in sorted(pool[d]) if pool[d][k]]
            if not secs:
                datasets.remove(d)
                continue
            t = pool[d][secs[turn[d] % len(secs)]].pop()
            turn[d] += 1
            chosen.append(t)
            total += size[t]
            if total >= n:
                break
    chosen = np.array(sorted(chosen))
    keep = np.isin(tile_of, chosen)
    info = {"method": "whole held-out tiles, round-robin over datasets then sections", "tile_um": tile_um,
            "tiles": int(len(chosen)), "tiles_available": int((~bad).sum()), "tiles_blocked": int(bad.sum())}
    return cand[keep], np.searchsorted(chosen, tile_of[keep]).astype(np.int32), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/brainwm/BrainWM-Training-Benchmark")
    ap.add_argument("--pretrain", default="/data/brainwm/unified_1m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--design", choices=["annotation", "integration", "spatial"], default="annotation",
                    help="annotation: balance cell types and regions; integration: shared cell types, balanced per "
                    "batch; spatial: whole held-out tiles with neighbourhoods")
    ap.add_argument("--tile-um", type=float, default=500.0, help="spatial design: tile size (as unified_1m)")
    ap.add_argument("--disjoint-from", nargs="*", default=[], help="eval sets (dirs with source_row.npy) to exclude")
    args = ap.parse_args()
    src = os.path.join(args.root, "assets/v4_numpy_shards")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    vocab = json.load(open(os.path.join(src, "joint_metadata_vocab.json")))

    prefixes = B.shard_prefixes(src)
    meta = B.load_metadata(prefixes)
    start = np.r_[0, np.cumsum(np.bincount(meta["shard"], minlength=len(prefixes)))[:-1]]
    used = np.load(os.path.join(args.pretrain, "source_row.npy"))
    exclude = np.zeros(len(meta["row"]), bool)
    exclude[start[used[:, 0]] + used[:, 1]] = True
    seen_ct = np.unique(np.load(os.path.join(args.pretrain, "cell_type.npy")))
    seen_an = np.unique(np.load(os.path.join(args.pretrain, "anatomy.npy")))
    exclude |= ~np.isin(meta["cell_type"], seen_ct) | ~np.isin(meta["anatomy"], seen_an)
    in_source = exclude.copy()  # rows already used by unified_1m or another eval set
    for other in args.disjoint_from:
        rows = np.load(os.path.join(other, "source_row.npy"))
        exclude[start[rows[:, 0]] + rows[:, 1]] = True
        in_source[start[rows[:, 0]] + rows[:, 1]] = True
    B.log("excluded %d rows (in unified_1m, in --disjoint-from sets, or label unseen in unified_1m)" % exclude.sum())

    # Verified-section MERFISH cells are ordinary single cells here (no neighbourhoods needed).
    cand = np.flatnonzero((meta["cell_type"] >= 0) & (meta["anatomy"] >= 0) & ~exclude)
    tile = None
    if args.design == "spatial":  # whole tiles: cells with labels unseen in unified_1m stay in their tile
        order, tile, caps = select_tiles(meta, in_source, args.n, args.tile_um, rng)
        B.log("spatial design: %d cells, %s" % (len(order), caps))
    elif args.design == "integration":
        order, caps = select_shared(meta, cand, args.n, rng)
        caps["method"] = "shared cell types; water-filled quota per (cell type, batch) pair"
        B.log("integration design: %d cells, %s" % (len(order), caps))
    else:
        p, fit = select_balanced(meta, cand, args.n)
        order = cand[rng.random(len(cand)) < p]
        caps = {"method": "IPF over " + ", ".join(BALANCE) + "; " + SPREAD + " spread within strata; caps "
                + str(CAPS), "margin_misfit_fraction": fit}
        B.log("balanced: %d candidates -> %d cells; margin misfit %s" % (len(cand), len(order), fit))
    perm = rng.permutation(len(order))
    order = order[perm]
    assert not (in_source if tile is not None else exclude)[order].any()
    n = len(order)

    nnz_total = B.write_expression(prefixes, meta, order, args.out)
    for k in B.META:
        name = "panel_id" if k == "row_source_id" else k
        np.save(os.path.join(args.out, name + ".npy"), meta[k][order].astype(np.int32))
    if tile is not None:
        xyz = meta["xyz"][order].astype(np.float32)
        nb, nd = B.neighbors(xyz[:, :2], meta["section"][order], np.ones(n, bool))
        caps["cells_with_full_neighbors"] = int((nb >= 0).all(1).sum())
        np.save(os.path.join(args.out, "spatial.npy"), xyz)
        np.save(os.path.join(args.out, "tile_id.npy"), tile[perm])
        np.save(os.path.join(args.out, "neighbors.npy"), nb)
        np.save(os.path.join(args.out, "neighbor_dist.npy"), nd)
    else:
        np.save(os.path.join(args.out, "spatial.npy"), np.full((n, 3), np.nan, np.float32))
        np.save(os.path.join(args.out, "tile_id.npy"), np.full(n, -1, np.int32))
        np.save(os.path.join(args.out, "neighbors.npy"), np.full((n, B.N_NEIGHBORS), -1, np.int32))
        np.save(os.path.join(args.out, "neighbor_dist.npy"), np.full((n, B.N_NEIGHBORS), np.nan, np.float32))
    np.save(os.path.join(args.out, "panel_gene_index.npy"), np.load(os.path.join(src, "panel_gene_index.npy")))
    np.save(os.path.join(args.out, "panel_gene_ptr.npy"), np.load(os.path.join(src, "panel_ptr.npy")))
    np.save(os.path.join(args.out, "source_row.npy"),
            np.stack([meta["shard"][order], meta["row"][order]], 1).astype(np.int64))
    np.save(os.path.join(args.out, "was_spatial.npy"), meta["graph"][order])
    B.build_vocabs(args.root, vocab, args.out)

    def counts(arr, names):
        u, c = np.unique(arr, return_counts=True)
        return {str(names[i]) if 0 <= i < len(names) else str(i): int(x) for i, x in zip(u, c)}

    names = {k: B.vocab_list(vocab[k]) for k in ("cell_type", "anatomy", "integration_batch",
                                                  "technology", "dataset", "life_stage", "assay")}
    manifest = {
        "rows": n, "nonzeros": nnz_total, "genes": 65918, "seed": args.seed,
        "value_scale": "raw integer UMI/read counts (V4 values)",
        "source": "V4 train+validation rows not in %s; sealed test split not read" % args.pretrain,
        "sampling": dict(caps, n_target=args.n),
        "was_spatial_rows": int(meta["graph"][order].sum()),
        "counts": {k: counts(meta[k][order], names[k]) for k in names},
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    B.log("done: %d rows" % n)


if __name__ == "__main__":
    main()
