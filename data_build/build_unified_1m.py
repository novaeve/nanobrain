"""Build the ~1M-cell balanced BrainWM pretraining subset from V4 NumPy shards.

Spatial part: whole square tiles of verified MERFISH sections (neighbours stay
inside the subset). Expression-only part: labelled cells balanced by
cell_type x anatomy with per-dataset caps and floors.
"""
import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

META = ["cell_type", "anatomy", "dataset", "technology", "assay", "donor", "section",
        "life_stage", "integration_batch", "cl_id", "uberon_id", "row_source_id"]
SPATIAL_ASSAY = 2  # joint_metadata_vocab: assay "spatialRNA"
N_NEIGHBORS = 16


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def shard_prefixes(src):
    out = []
    for split in ("train", "validation"):
        for vp in sorted(glob.glob(os.path.join(src, split + "-shard-*-values.npy"))):
            out.append(vp[: -len("values.npy")])
    return out


def load_metadata(prefixes):
    cols = defaultdict(list)
    for s, pre in enumerate(prefixes):
        ptr = np.load(pre + "row_ptr.npy")
        n = len(ptr) - 1
        for k in META:
            cols[k].append(np.load(pre + k + ".npy").astype(np.int64))
        cols["graph"].append(np.load(pre + "spatial_graph_eligible.npy"))
        cols["xyz"].append(np.load(pre + "spatial.npy").astype(np.float32))
        cols["nnz"].append(np.diff(ptr))
        cols["shard"].append(np.full(n, s, np.int32))
        cols["row"].append(np.arange(n, dtype=np.int64))
    meta = {k: np.concatenate(v) for k, v in cols.items()}
    log("metadata rows %d from %d shards" % (len(meta["row"]), len(prefixes)))
    return meta


def spatial_tiles(meta, tile_um):
    """Labelled verified-section cells and their square tile: (cand rows, tile keys, tile index per cand)."""
    xy = meta["xyz"][:, :2]
    cand = np.where(meta["graph"] & (meta["cell_type"] >= 0) & (meta["anatomy"] >= 0)
                    & (meta["section"] >= 0) & np.isfinite(xy).all(1))[0]
    sec = meta["section"][cand]
    tx = np.floor(xy[cand, 0] / tile_um).astype(np.int64)
    ty = np.floor(xy[cand, 1] / tile_um).astype(np.int64)
    key = (sec << 40) + ((tx - tx.min()) << 20) + (ty - ty.min())
    tiles, tile_of = np.unique(key, return_inverse=True)
    return cand, tiles, tile_of


def select_spatial(meta, target, tile_um, rng):
    cand, tiles, tile_of = spatial_tiles(meta, tile_um)
    sec = meta["section"][cand]
    tile_sec = (tiles >> 40)
    tile_size = np.bincount(tile_of)
    sections = np.unique(sec)
    quota = target / len(sections)
    chosen = []
    for s in sections:
        ts = np.where(tile_sec == s)[0]
        ts = ts[rng.permutation(len(ts))]
        got = 0
        for t in ts:
            if got >= quota:
                break
            chosen.append(t)
            got += tile_size[t]
    chosen = np.array(sorted(chosen))
    keep = np.isin(tile_of, chosen)
    idx = cand[keep]
    tile_id = np.searchsorted(chosen, tile_of[keep]).astype(np.int32)
    log("spatial: %d candidate cells, %d sections, %d tiles chosen, %d cells"
        % (len(cand), len(sections), len(chosen), len(idx)))
    return idx, tile_id


def select_expression(meta, exclude, target, dataset_cap_frac, dataset_floor, rng):
    lab = (meta["cell_type"] >= 0) & (meta["anatomy"] >= 0)
    lab[exclude] = False
    lab &= ~meta["graph"]  # verified-section MERFISH cells only enter as whole tiles
    cand = np.where(lab)[0]
    group = meta["cell_type"][cand] * 1000 + meta["anatomy"][cand]
    _, g_of = np.unique(group, return_inverse=True)
    n_g = np.bincount(g_of).astype(np.float64)
    _, d_of = np.unique(meta["dataset"][cand], return_inverse=True)
    n_d = np.bincount(d_of).astype(np.float64)
    cap = dataset_cap_frac * target
    s_d = np.ones(len(n_d))

    def probs(c):
        return np.minimum(1.0, np.minimum(1.0, c / n_g[g_of]) * s_d[d_of])

    def solve_c():
        lo, hi = 1.0, n_g.max()
        for _ in range(60):
            mid = (lo + hi) / 2
            if probs(mid).sum() < target:
                lo = mid
            else:
                hi = mid
        return hi

    for _ in range(30):
        c = solve_c()
        e_d = np.bincount(d_of, weights=probs(c), minlength=len(n_d))
        floor = np.minimum(dataset_floor, n_d)
        adj = np.where(e_d > cap, cap / e_d, np.where(e_d < floor, floor / np.maximum(e_d, 1e-9), 1.0))
        if np.all(np.abs(adj - 1) < 1e-3):
            break
        s_d *= adj
    c = solve_c()
    p = probs(c)
    idx = cand[rng.random(len(cand)) < p]
    log("expression: %d candidates, %d groups, cap c=%.1f, %d cells" % (len(cand), len(n_g), c, len(idx)))
    return idx, {"group_cap_c": float(c), "dataset_cap": float(cap), "dataset_floor": int(dataset_floor)}


def write_expression(prefixes, meta, order, out):
    nnz = meta["nnz"][order]
    row_ptr = np.zeros(len(order) + 1, np.int64)
    np.cumsum(nnz, out=row_ptr[1:])
    np.save(os.path.join(out, "row_ptr.npy"), row_ptr)
    total = int(row_ptr[-1])
    vals = np.lib.format.open_memmap(os.path.join(out, "values.npy"), "w+", np.float32, (total,))
    genes = np.lib.format.open_memmap(os.path.join(out, "gene_index.npy"), "w+", np.int32, (total,))
    shard = meta["shard"][order]
    for s in np.unique(shard):
        pos = np.where(shard == s)[0]
        pre = prefixes[s]
        sptr = np.load(pre + "row_ptr.npy")
        sv = np.load(pre + "values.npy", mmap_mode="r")
        sg = np.load(pre + "gene_index.npy", mmap_mode="r")
        rows = meta["row"][order[pos]]
        lens = sptr[rows + 1] - sptr[rows]
        src = np.repeat(sptr[rows] - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())
        dst = np.repeat(row_ptr[pos] - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())
        vals[dst] = sv[src]
        genes[dst] = sg[src]
    vals.flush()
    genes.flush()
    log("expression written: %d rows, %d nonzeros" % (len(order), total))
    return total


def neighbors(xy, section, is_spatial):
    nb = np.full((len(xy), N_NEIGHBORS), -1, np.int32)
    nd = np.full((len(xy), N_NEIGHBORS), np.nan, np.float32)
    for s in np.unique(section[is_spatial]):
        ids = np.where(is_spatial & (section == s))[0]
        k = min(N_NEIGHBORS + 1, len(ids))
        d, j = cKDTree(xy[ids]).query(xy[ids], k=k)
        d, j = d[:, 1:], ids[j[:, 1:]]  # drop self
        nb[ids, : k - 1] = j
        nd[ids, : k - 1] = d
    return nb, nd


def obo_ancestors(path, relations=("is_a", "part_of")):
    parents, names = defaultdict(set), {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id: "):
            cur = line[4:]
        elif cur and line.startswith("name: "):
            names[cur] = line[6:]
        elif cur and line.startswith("is_a: ") and "is_a" in relations:
            parents[cur].add(line[6:].split(" ")[0])
        elif cur and line.startswith("relationship: "):
            rel, tgt = line[14:].split(" ")[:2]
            if rel in relations:
                parents[cur].add(tgt)

    def anc(t, seen=None):
        seen = set() if seen is None else seen
        for p in parents.get(t, ()):
            if p not in seen:
                seen.add(p)
                anc(p, seen)
        return seen
    return anc, names


ANATOMY_LEVELS = [
    ("division", [("UBERON:0001890", "forebrain"), ("UBERON:0001891", "midbrain"),
                  ("UBERON:0002028", "hindbrain"), ("UBERON:0002240", "spinal cord")]),
    ("region", [("UBERON:0001893", "telencephalon"), ("UBERON:0001894", "diencephalon"),
                ("UBERON:0001891", "midbrain"), ("UBERON:0002037", "cerebellum"),
                ("UBERON:0000988", "pons"), ("UBERON:0001896", "medulla oblongata"),
                ("UBERON:0002240", "spinal cord")]),
]


# Terms whose UBERON is_a/part_of chain does not reach the coarse nodes above.
# Values are (division, region) names; reviewed by hand on 2026-11-03.
ANATOMY_OVERRIDES = {
    "UBERON:0001851": ("forebrain", "telencephalon"),       # "cortex" as annotated in brain atlases
    "UBERON:0002420": ("forebrain", "telencephalon"),       # basal ganglion
    "UBERON:0010011": ("forebrain", "telencephalon"),       # collection of basal ganglia
    "UBERON:0002743": ("forebrain", "telencephalon"),       # basal forebrain
    "UBERON:0004023": ("forebrain", "telencephalon"),       # ganglionic eminence
    "UBERON:0004024": ("forebrain", "telencephalon"),
    "UBERON:0004025": ("forebrain", "telencephalon"),
    "UBERON:0004026": ("forebrain", "telencephalon"),
    "UBERON:0002474": ("hindbrain", "cerebellum"),          # cerebellar peduncular complex
    "UBERON:0005290": ("hindbrain", "medulla oblongata"),   # myelencephalon
}


def vocab_list(d):
    return [n for n, _ in sorted(d.items(), key=lambda kv: kv[1])]


def build_vocabs(root, vocab, out):
    import pandas as pd
    ann = pd.read_parquet(os.path.join(root, "contracts/human_semantics_v4/annotation_vocabulary.parquet"))
    ct_ids = vocab_list(vocab["cell_type"])
    lineage = dict(zip(ann["label"], ann["lineage"]))
    cl_anc, cl_names = obo_ancestors(os.path.join(root, "assets/reference_ontology/cl.obo"), ("is_a",))
    lin_names = sorted(set(lineage[c] for c in ct_ids))
    ct = {
        "levels": ["lineage", "cell_type"],
        "lineage": lin_names,
        "cell_type": [{"index": i, "cl_id": c, "name": cl_names.get(c, c), "lineage": lineage[c],
                       "parent_lineage_index": lin_names.index(lineage[c]),
                       "cl_is_a_ancestors": sorted(a for a in cl_anc(c) if a.startswith("CL:"))}
                      for i, c in enumerate(ct_ids)],
    }
    json.dump(ct, open(os.path.join(out, "cell_type_vocab.json"), "w"), indent=1)
    ub_anc, ub_names = obo_ancestors(os.path.join(root, "..", "reference_ontology_dl", "uberon-basic.obo"))
    an_ids = vocab_list(vocab["anatomy"])
    an = {"levels": [lv for lv, _ in ANATOMY_LEVELS] + ["anatomy"]}
    for lv, nodes in ANATOMY_LEVELS:
        an[lv] = [name for _, name in nodes] + ["other"]
    entries = []
    for i, u in enumerate(an_ids):
        a = ub_anc(u) | {u}
        e = {"index": i, "uberon_id": u, "name": ub_names.get(u, u),
             "mapping_source": "manual_override" if u in ANATOMY_OVERRIDES else "uberon_is_a_part_of"}
        for k_lv, (lv, nodes) in enumerate(ANATOMY_LEVELS):
            if u in ANATOMY_OVERRIDES:
                names = [name for _, name in nodes]
                e["parent_" + lv + "_index"] = names.index(ANATOMY_OVERRIDES[u][k_lv])
                continue
            hit = [k for k, (nid, _) in enumerate(nodes) if nid in a]
            e["parent_" + lv + "_index"] = hit[0] if hit else len(nodes)
        entries.append(e)
    an["anatomy"] = entries
    json.dump(an, open(os.path.join(out, "anatomy_vocab.json"), "w"), indent=1)
    return ct, an


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/brainwm/BrainWM-Training-Benchmark")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-spatial", type=int, default=300000)
    ap.add_argument("--n-expression", type=int, default=700000)
    ap.add_argument("--tile-um", type=float, default=500.0)
    ap.add_argument("--dataset-cap-frac", type=float, default=0.06)
    ap.add_argument("--dataset-floor", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()
    src = os.path.join(args.root, "assets/v4_numpy_shards")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    vocab = json.load(open(os.path.join(src, "joint_metadata_vocab.json")))

    prefixes = shard_prefixes(src)
    meta = load_metadata(prefixes)
    sp_idx, sp_tile = select_spatial(meta, args.n_spatial, args.tile_um, rng)
    ex_idx, caps = select_expression(meta, sp_idx, args.n_expression, args.dataset_cap_frac,
                                     args.dataset_floor, rng)

    sel = np.concatenate([sp_idx, ex_idx])
    tile = np.concatenate([sp_tile, np.full(len(ex_idx), -1, np.int32)])
    perm = rng.permutation(len(sel))
    order, tile = sel[perm], tile[perm]
    n = len(order)

    nnz_total = write_expression(prefixes, meta, order, args.out)
    for k in META:
        name = "panel_id" if k == "row_source_id" else k
        np.save(os.path.join(args.out, name + ".npy"), meta[k][order].astype(np.int32))
    is_sp = tile >= 0
    xyz = meta["xyz"][order].copy()
    xyz[~is_sp] = np.nan
    np.save(os.path.join(args.out, "spatial.npy"), xyz)
    np.save(os.path.join(args.out, "tile_id.npy"), tile)
    nb, nd = neighbors(xyz[:, :2], meta["section"][order], is_sp)
    np.save(os.path.join(args.out, "neighbors.npy"), nb)
    np.save(os.path.join(args.out, "neighbor_dist.npy"), nd)
    np.save(os.path.join(args.out, "panel_gene_index.npy"), np.load(os.path.join(src, "panel_gene_index.npy")))
    np.save(os.path.join(args.out, "panel_gene_ptr.npy"), np.load(os.path.join(src, "panel_ptr.npy")))
    np.save(os.path.join(args.out, "source_row.npy"),
            np.stack([meta["shard"][order], meta["row"][order]], 1).astype(np.int64))
    ct, an = build_vocabs(args.root, vocab, args.out)

    def counts(arr, names):
        u, c = np.unique(arr, return_counts=True)
        return {str(names[i]) if 0 <= i < len(names) else str(i): int(x) for i, x in zip(u, c)}

    names = {k: vocab_list(vocab[k]) for k in ("cell_type", "anatomy", "integration_batch",
                                                "technology", "dataset", "life_stage", "assay")}
    manifest = {
        "rows": n, "spatial_rows": int(is_sp.sum()), "expression_only_rows": int((~is_sp).sum()),
        "nonzeros": nnz_total, "genes": 65918, "value_scale": "raw integer UMI/read counts (V4 values)",
        "seed": args.seed, "tile_um": args.tile_um, "n_tiles": int(tile.max() + 1),
        "sampling": dict(caps, n_spatial_target=args.n_spatial, n_expression_target=args.n_expression),
        "counts": {k: counts(meta[k if k != "assay" else "assay"][order], names[k]) for k in names},
        "spatial_sections": int(len(np.unique(meta["section"][order][is_sp]))),
        "cells_with_full_neighbors": int((nb[is_sp] >= 0).all(1).sum()),
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    log("done: %d rows" % n)


if __name__ == "__main__":
    main()
