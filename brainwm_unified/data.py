"""In-memory pretraining subset and per-step batch construction.

Every step mixes dissociated cells and spatial groups (a centre cell plus its
neighbours), so all five objectives receive data on every step.
"""

import json
import os

import numpy as np
import torch

from . import masking as M
from .config import DataConfig

CELL_ARRAYS = ("cell_type", "anatomy", "integration_batch", "panel_id")


# Panels measuring at least this many genes count as whole-transcriptome (vs targeted, e.g. MERFISH).
WHOLE_TRANSCRIPTOME = 10_000


def load_level_maps(path: str) -> list[np.ndarray]:
    """Fine-label -> level-label maps, coarse to fine; the last map is the identity.

    The vocabulary lists its levels coarse to fine under "levels"; each entry of
    the finest level carries `parent_<level>_index` for every coarser level.
    """
    with open(path) as fh:
        vocab = json.load(fh)
    *coarse, fine = vocab["levels"]
    entries = sorted(vocab[fine], key=lambda e: e["index"])
    maps = [np.asarray([e[f"parent_{level}_index"] for e in entries], dtype=np.int64) for level in coarse]
    maps.append(np.arange(len(entries), dtype=np.int64))
    return maps


class Subset:
    """The subset arrays, loaded fully into memory."""

    def __init__(self, root: str, value_scale: str = "counts"):
        def load(name):
            return np.load(os.path.join(root, name + ".npy"))

        self.root = root
        self.value_scale = value_scale
        self.values = load("values").astype(np.float32, copy=False)
        self.gene_index = load("gene_index").astype(np.int64, copy=False)
        self.row_ptr = load("row_ptr").astype(np.int64, copy=False)
        for name in CELL_ARRAYS:
            setattr(self, name, load(name).astype(np.int64, copy=False))
        self.spatial = load("spatial").astype(np.float32, copy=False)
        self.neighbors = load("neighbors").astype(np.int64, copy=False)
        self.neighbor_dist = load("neighbor_dist").astype(np.float32, copy=False)
        self.panel_gene_ptr = load("panel_gene_ptr").astype(np.int64, copy=False)
        self.panel_gene_index = load("panel_gene_index").astype(np.int64, copy=False)

        self.n_cells = len(self.row_ptr) - 1
        is_spatial = ~np.isnan(self.spatial[:, 0])
        self.dissociated = np.flatnonzero(~is_spatial)
        self.spatial_centers = np.flatnonzero(is_spatial & (self.neighbors[:, 0] >= 0))

    def panel_genes(self, panel: int) -> np.ndarray:
        return self.panel_gene_index[self.panel_gene_ptr[panel] : self.panel_gene_ptr[panel + 1]]

    def gene_rates(self, n_genes: int) -> np.ndarray:
        """Each gene's UMIs per UMI of the cells whose panel measures it (the Pearson-residual p_g)."""
        lib = np.diff(np.concatenate([[0.0], np.cumsum(self.values, dtype=np.float64)])[self.row_ptr])
        num = np.bincount(self.gene_index, weights=self.values, minlength=n_genes)
        den = np.zeros(n_genes)
        panel_lib = np.bincount(self.panel_id, weights=lib, minlength=len(self.panel_gene_ptr) - 1)
        for p, total in enumerate(panel_lib):
            den[self.panel_genes(p)] += total
        return np.divide(num, den, out=np.zeros(n_genes), where=den > 0)

    def panel_matrix(self, n_genes: int) -> torch.Tensor:
        """Boolean (n_panels, n_genes): which genes each panel measures."""
        n_panels = len(self.panel_gene_ptr) - 1
        mat = torch.zeros(n_panels, n_genes, dtype=torch.bool)
        for p in range(n_panels):
            mat[p, torch.from_numpy(self.panel_genes(p))] = True
        return mat


def sample_cells(ds: Subset, cfg: DataConfig, rng: np.random.Generator):
    """One step's dissociated cells and spatial group centres."""
    diss = rng.choice(ds.dissociated, size=min(cfg.dissociated_cells, len(ds.dissociated)), replace=False)
    centers = rng.choice(ds.spatial_centers, size=min(cfg.spatial_groups, len(ds.spatial_centers)), replace=False)
    return diss, centers


def restrict(ds: Subset, fraction: float, seed: int = 20261104) -> Subset:
    """Keep a fixed random fraction of the dissociated cells and spatial group centres (data scaling).

    The subset depends only on `seed` and `fraction`, and smaller fractions are nested in larger ones.
    Neighbours of a kept centre are still read, as they belong to its spatial group.
    """
    if fraction >= 1.0:
        return ds
    rng = np.random.default_rng(seed)
    for name in ("dissociated", "spatial_centers"):
        pool = getattr(ds, name)
        order = pool[rng.permutation(len(pool))]
        setattr(ds, name, np.sort(order[: max(1, round(fraction * len(pool)))]))
    return ds


def mask_resources(ds: Subset, cfg: DataConfig) -> M.Resources:
    """Gene programs (if built) and the gene sets of targeted-panel platforms, for the richer masks."""
    path = cfg.gene_programs or os.path.join(ds.root, "gene_programs.npy")
    programs = np.load(path) if os.path.exists(path) else None
    spatial_panels = np.unique(ds.panel_id[~np.isnan(ds.spatial[:, 0])])
    panels = [ds.panel_genes(p) for p in spatial_panels if len(ds.panel_genes(p)) < WHOLE_TRANSCRIPTOME]
    return M.Resources(gene_program=programs, panels=panels)


def build_batch(
    ds: Subset,
    cfg: DataConfig,
    n_genes: int,
    rng: np.random.Generator,
    diss: np.ndarray | None = None,
    centers: np.ndarray | None = None,
    plan: M.MaskPlan | None = None,
    res: M.Resources | None = None,
    progress: float = 0.0,
) -> dict:
    """Turn one step's cells (sampled from `rng` unless given) into packed token and cell tensors.

    Masks come from `plan` (default: cfg.masks); `progress` (0..1) selects curriculum stage weights.
    Cells come out in increasing index order (`cells` below), whatever order they were given in.
    """
    if diss is None or centers is None:
        diss, centers = sample_cells(ds, cfg, rng)
    plan = plan or M.load_plan(cfg.masks)
    res = res or M.Resources()
    k = min(cfg.neighbors, ds.neighbors.shape[1])
    group_ids = np.full((len(centers), k + 1), -1, dtype=np.int64)
    group_ids[:, 0] = centers
    group_ids[:, 1:] = ds.neighbors[centers, :k]

    cells = np.unique(np.concatenate([diss, group_ids[group_ids >= 0]]))
    pos_of = {int(c): i for i, c in enumerate(cells)}

    tok_gene, tok_in, tok_target, tok_masked, tok_doc, tok_cls = [], [], [], [], [], []
    cnt_gene, cnt_val, cnt_ptr = [], [], [0]
    for i, c in enumerate(cells):
        lo, hi = ds.row_ptr[c], ds.row_ptr[c + 1]
        genes, vals = ds.gene_index[lo:hi], ds.values[lo:hi]
        cnt_gene.append(genes)
        cnt_val.append(vals)
        cnt_ptr.append(cnt_ptr[-1] + len(genes))

        if ds.value_scale == "counts":
            lib = max(float(vals.sum()), 1.0)
            norm = np.log1p(vals / lib * 1e4)
        else:
            norm = vals
        raw = vals
        if len(genes) > cfg.max_nonzero_tokens:
            keep = rng.choice(len(genes), cfg.max_nonzero_tokens, replace=False)
            genes, norm, raw = genes[keep], norm[keep], raw[keep]
        panel = ds.panel_genes(ds.panel_id[c])
        cand = rng.choice(panel, size=min(len(panel), 2 * cfg.zero_tokens + len(genes)), replace=False)
        zeros = np.setdiff1d(cand, genes, assume_unique=False)[: cfg.zero_tokens]

        g = np.concatenate([genes, zeros])
        v = np.concatenate([norm, np.zeros(len(zeros), np.float32)]).astype(np.float32)
        row = M.Row(
            genes=g,
            values=v,
            raw=np.concatenate([raw, np.zeros(len(zeros), np.float32)]) if ds.value_scale == "counts" else None,
            lib=float(vals.sum()),
            whole_transcriptome=len(panel) >= WHOLE_TRANSCRIPTOME,
        )
        states, evidence = plan.rows.pick(rng, progress)(row, rng, res)
        kept = states != M.HIDDEN
        masked = states[kept] == M.TARGET
        g, v = g[kept], v[kept]
        v_in = v if evidence is None else evidence[kept]

        # CLS token first; its gene id is the extra vocabulary slot n_genes.
        tok_gene += [np.array([n_genes]), g]
        tok_target += [np.zeros(1, np.float32), v]
        tok_in += [np.zeros(1, np.float32), np.where(masked, 0.0, v_in).astype(np.float32)]
        tok_masked += [np.zeros(1, bool), masked]
        tok_cls += [np.ones(1, bool), np.zeros(len(g), bool)]
        tok_doc.append(np.full(len(g) + 1, i, dtype=np.int64))

    # Spatial groups: member positions in `cells`, relative coordinates, neighbour mask.
    members = np.vectorize(lambda c: pos_of.get(int(c), -1), otypes=[np.int64])(group_ids)
    xyz = ds.spatial[np.where(group_ids >= 0, group_ids, 0)]
    rel = xyz - xyz[:, :1]
    scale = np.nanmean(ds.neighbor_dist[centers, :k], axis=1, keepdims=True)
    rel = np.nan_to_num(rel / np.maximum(scale, 1e-6)[:, :, None], nan=0.0).astype(np.float32)
    rel[members < 0] = 0.0
    neigh_masked = np.zeros_like(members, dtype=bool)
    for gi in range(len(members)):
        valid = np.flatnonzero(members[gi] >= 0)
        neigh_masked[gi] = plan.groups.pick(rng, progress)(rel[gi], valid, rng)

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x))

    return {
        "tok_gene": t(np.concatenate(tok_gene)),
        "tok_in": t(np.concatenate(tok_in)),
        "tok_target": t(np.concatenate(tok_target)),
        "tok_masked": t(np.concatenate(tok_masked)),
        "tok_cls": t(np.concatenate(tok_cls)),
        "tok_doc": t(np.concatenate(tok_doc)),
        "cnt_gene": t(np.concatenate(cnt_gene)),
        "cnt_val": t(np.concatenate(cnt_val)),
        "cnt_ptr": t(np.asarray(cnt_ptr, dtype=np.int64)),
        "cell_type": t(ds.cell_type[cells]),
        "anatomy": t(ds.anatomy[cells]),
        "batch": t(ds.integration_batch[cells]),
        "panel": t(ds.panel_id[cells]),
        "grp_members": t(members),
        "grp_rel": t(rel),
        "grp_masked": t(neigh_masked),
    }


class BatchStream(torch.utils.data.IterableDataset):
    """Endless stream of batches; each DataLoader worker draws from its own seed.

    With `total_steps`, each batch knows its training progress for curriculum masks: worker w of W
    makes batches w, w + W, ... after `seed_offset` (the DataLoader returns them in that order).
    """

    def __init__(self, ds: Subset, cfg: DataConfig, n_genes: int, seed_offset: int = 0, total_steps: int = 0):
        self.ds, self.cfg, self.n_genes, self.seed_offset, self.total_steps = ds, cfg, n_genes, seed_offset, total_steps
        self.plan = M.load_plan(cfg.masks)
        self.res = mask_resources(ds, cfg) if cfg.masks != "cajal_v0" else M.Resources()

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, n_workers = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.cfg.seed, self.seed_offset, worker])
        step = self.seed_offset + worker
        while True:
            progress = step / self.total_steps if self.total_steps else 0.0
            yield build_batch(self.ds, self.cfg, self.n_genes, rng, plan=self.plan, res=self.res, progress=progress)
            step += n_workers
