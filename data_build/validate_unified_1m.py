"""Validate a unified_1m directory against its file contract; exits non-zero on failure."""
import json
import os
import sys

import numpy as np

d = sys.argv[1]
L = lambda name, **kw: np.load(os.path.join(d, name + ".npy"), **kw)
fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


ptr = L("row_ptr")
n = len(ptr) - 1
vals, genes = L("values", mmap_mode="r"), L("gene_index", mmap_mode="r")
check(ptr[0] == 0 and np.all(np.diff(ptr) >= 0), "row_ptr starts at 0 and is monotone")
check(ptr[-1] == len(vals) == len(genes), "row_ptr[-1] == len(values) == len(gene_index)")
check(vals.dtype == np.float32 and genes.dtype == np.int32 and ptr.dtype == np.int64, "CSR dtypes")
step = max(1, len(genes) // 50_000_000)
check(int(genes[::step].min()) >= 0 and int(genes[::step].max()) < 65918, "gene_index in [0, 65918) (sampled)")
v = np.asarray(vals[::step])
check(np.isfinite(v).all() and (v > 0).all() and np.all(v == np.round(v)), "values positive integer counts (sampled)")

ct_vocab = json.load(open(os.path.join(d, "cell_type_vocab.json")))
an_vocab = json.load(open(os.path.join(d, "anatomy_vocab.json")))
ct, an = L("cell_type"), L("anatomy")
for name in ["cell_type", "anatomy", "integration_batch", "dataset", "technology", "assay", "donor",
             "section", "life_stage", "cl_id", "uberon_id", "panel_id", "tile_id"]:
    a = L(name)
    check(a.shape == (n,) and a.dtype == np.int32, "%s shape (n,) int32" % name)
check(ct.min() >= 0 and ct.max() < len(ct_vocab["cell_type"]), "cell_type labelled and in vocab")
check(an.min() >= 0 and an.max() < len(an_vocab["anatomy"]), "anatomy labelled and in vocab")
panel_ptr = L("panel_gene_ptr")
check(L("panel_id").min() >= 0 and L("panel_id").max() < len(panel_ptr) - 1, "panel_id indexes panel_gene_ptr")

tile, xyz, sec = L("tile_id"), L("spatial"), L("section")
sp = tile >= 0
check(xyz.shape == (n, 3) and np.isfinite(xyz[sp, :2]).all() and np.isnan(xyz[~sp]).all(),
      "spatial coords finite for tiles, NaN otherwise")
nb, nd = L("neighbors"), L("neighbor_dist")
check(nb.shape == (n, 16) and nb.dtype == np.int32, "neighbors (n,16) int32")
check((nb[~sp] == -1).all(), "non-spatial rows have no neighbours")
ids = np.where(sp)[0]
nbs = nb[ids]
valid = nbs >= 0
check((nbs[valid] < n).all(), "neighbour indices in range")
check((nbs != ids[:, None]).all(), "self excluded from neighbours")
check((sec[nbs[valid]] == np.repeat(sec[ids], valid.sum(1))).all(), "neighbours share the section")
check(sp[nbs[valid]].all(), "neighbours are spatial rows")
check(np.all(np.diff(np.where(valid, nd[ids], np.inf), axis=1)[valid[:, 1:]] >= 0), "neighbour distances sorted")
check(900_000 <= n <= 1_100_000, "total rows ~1M (got %d)" % n)
print("rows", n, "spatial", int(sp.sum()), "nonzeros", int(ptr[-1]))
sys.exit(1 if fails else 0)
