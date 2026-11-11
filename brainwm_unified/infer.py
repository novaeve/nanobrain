"""Forward-only inference: cell latents and label predictions for a list of cells.

Evaluation is a prefill-like workload (many cells, every token at once), so it follows
LLM prefill practice rather than the training step: no masking, batches filled to a
token budget instead of a fixed cell count (no activations are kept, so batches can be
far larger), batches prepared by DataLoader workers while the GPU runs, inference_mode,
and only the modules the outputs need (encoder, latent, label heads; the gene head,
expression decoder and neighbourhood model are skipped).

The two imputation tasks have their own entry points: predict_genes (the training gene mask,
fixed by a seed, on every cell; the gene head fills the masked values) and predict_spatial
(each centre cell hidden in turn and rebuilt from its spatial neighbours).
"""

import numpy as np
import torch

from . import masking
from . import model as M
from .config import DataConfig, ModelConfig
from .data import Subset, build_batch
from .impute_metrics import panel_profile


def token_count(ds: Subset, cells: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Tokens a cell becomes (upper bound: measured zeros are also limited by its panel)."""
    nonzero = np.minimum(ds.row_ptr[cells + 1] - ds.row_ptr[cells], cfg.max_nonzero_tokens)
    return 1 + nonzero + cfg.zero_tokens


def plan_batches(ds: Subset, cells: np.ndarray, cfg: DataConfig, token_budget: int) -> list[np.ndarray]:
    """Length-sorted cells cut into consecutive chunks of at most `token_budget` tokens."""
    order = cells[np.argsort(token_count(ds, cells, cfg), kind="stable")]
    tokens = token_count(ds, order, cfg)
    chunks, start, total = [], 0, 0
    for i, t in enumerate(tokens):
        if total + t > token_budget and i > start:
            chunks.append(order[start:i])
            start, total = i, 0
        total += t
    chunks.append(order[start:])
    return chunks


class EvalBatches(torch.utils.data.Dataset):
    """Chunk i -> an unmasked batch; its random choices (zeros, >cap subsampling) use seed (seed, i)."""

    def __init__(self, ds: Subset, cfg: DataConfig, n_genes: int, chunks: list[np.ndarray], seed: int = 0):
        self.ds, self.cfg, self.n_genes, self.chunks, self.seed = ds, cfg, n_genes, chunks, seed

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        b = build_batch(self.ds, self.cfg, self.n_genes, rng, self.chunks[i], np.zeros(0, np.int64))
        b["tok_in"] = b["tok_target"].clone()
        b["tok_masked"] = torch.zeros_like(b["tok_masked"])
        b["cells"] = torch.from_numpy(np.unique(self.chunks[i]))  # build_batch's cell order
        return b


class MaskedEvalBatches(EvalBatches):
    """Chunk i -> a batch with mask plan `masks` (default the training plan's gene mask), seeded by (seed, i)."""

    def __init__(self, *args, masks: str = "cajal_v0", **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = masking.load_plan(masks)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        b = build_batch(self.ds, self.cfg, self.n_genes, rng, self.chunks[i], np.zeros(0, np.int64), plan=self.plan)
        b["cells"] = torch.from_numpy(np.unique(self.chunks[i]))
        return b


class SpatialEvalBatches(torch.utils.data.Dataset):
    """Chunk i -> centres with their neighbourhoods, every token visible, only the centre of each group hidden."""

    def __init__(self, ds: Subset, cfg: DataConfig, n_genes: int, chunks: list[np.ndarray], seed: int = 0):
        self.ds, self.cfg, self.n_genes, self.chunks, self.seed = ds, cfg, n_genes, chunks, seed

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        b = build_batch(self.ds, self.cfg, self.n_genes, rng, np.zeros(0, np.int64), self.chunks[i])
        b["tok_in"] = b["tok_target"].clone()
        b["tok_masked"] = torch.zeros_like(b["tok_masked"])
        b["grp_masked"] = torch.zeros_like(b["grp_masked"])
        b["grp_masked"][:, 0] = True
        b["centers"] = torch.from_numpy(self.chunks[i])
        return b


def spatial_chunks(ds: Subset, centers: np.ndarray, cfg: DataConfig, token_budget: int) -> list[np.ndarray]:
    """Centres in spatial order (neighbourhoods overlap, so each cell is encoded once per chunk), cut so
    that the chunk's distinct cells stay within `token_budget` tokens."""
    k = min(cfg.neighbors, ds.neighbors.shape[1])
    order = centers[np.lexsort((ds.spatial[centers, 1], ds.spatial[centers, 0]))]
    chunks, cur, seen, total = [], [], set(), 0
    for c in order:
        new = [m for m in (c, *ds.neighbors[c, :k]) if m >= 0 and m not in seen]
        add = int(token_count(ds, np.asarray(new, np.int64), cfg).sum()) if new else 0
        if cur and total + add > token_budget:
            chunks.append(np.asarray(cur, np.int64))
            cur, seen, total = [], set(), 0
            new = [m for m in (c, *ds.neighbors[c, :k]) if m >= 0]
            add = int(token_count(ds, np.asarray(new, np.int64), cfg).sum())
        cur.append(c)
        seen.update(new)
        total += add
    chunks.append(np.asarray(cur, np.int64))
    return chunks


@torch.inference_mode()
def forward_batch(model, b: dict, loops: int) -> dict:
    """Latent mean and per-level label logits for one device batch."""
    layout = M.attention_layout(b["tok_doc"], model.cfg.attention_backend)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.encoder(b, layout, loops)
        mu = model.to_latent(h[b["tok_cls"]]).float().chunk(2, -1)[0]
        return {
            "mu": mu,
            "cell_type": [x.float() for x in model.cell_type_head(mu)],
            "anatomy": [x.float() for x in model.anatomy_head(mu)],
        }


@torch.inference_mode()
def predict(
    model,
    ds: Subset,
    cells: np.ndarray,
    cfg: DataConfig,
    token_budget: int = 1 << 19,
    num_workers: int = 8,
    seed: int = 0,
    loops: int | None = None,
) -> dict:
    """Latents and label logits for `cells`, returned in the order given."""
    model.eval()
    cells = np.asarray(cells, dtype=np.int64)
    loader = torch.utils.data.DataLoader(
        EvalBatches(ds, cfg, model.cfg.n_genes, plan_batches(ds, cells, cfg, token_budget), seed),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers else None,
    )
    device = next(model.parameters()).device
    parts = []
    for b in loader:
        got = b.pop("cells")
        b = M.pad_tokens({k: v.to(device, non_blocking=True) for k, v in b.items()})
        out = forward_batch(model, b, loops or model.cfg.loops)
        parts.append((got, out))
    done = torch.cat([g for g, _ in parts]).numpy()
    where = np.argsort(done)[np.searchsorted(done, cells, sorter=np.argsort(done))]

    def gather(get):
        return torch.cat([get(o).cpu() for _, o in parts])[torch.from_numpy(where)].numpy()

    n_levels = {k: len(parts[0][1][k]) for k in ("cell_type", "anatomy")}
    return {
        "cells": cells,
        "mu": gather(lambda o: o["mu"]),
        **{k: [gather(lambda o, k=k, i=i: o[k][i]) for i in range(n)] for k, n in n_levels.items()},
    }


def load_model(checkpoint: str, device="cuda", **overrides):
    """Model from a training checkpoint (its saved config, with optional ModelConfig overrides).

    Older checkpoints load too: per-expert weights are stacked, and models with head dim
    below 64 (FA3's minimum) run SDPA varlen attention, uncompiled (nested tensors do not compile).
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = state["config"]["ModelConfig"]
    if saved["d_model"] // saved["n_heads"] < 64 and "attention_backend" not in overrides:
        overrides = {**overrides, "attention_backend": "jagged", "compile_blocks": False}
    cfg = ModelConfig(**{**saved, **overrides})
    model = M.UnifiedModel(cfg)
    model.load_state_dict(M.upgrade_state_dict(state["model"]))
    return model.to(device).eval(), state


def _loader(dataset, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers else None,
    )


@torch.inference_mode()
def predict_genes(
    model,
    ds: Subset,
    cells: np.ndarray,
    cfg: DataConfig,
    token_budget: int = 1 << 19,
    num_workers: int = 8,
    seed: int = 0,
    loops: int | None = None,
    masks: str = "cajal_v0",
) -> dict:
    """Gene imputation: masked tokens of every cell (mask plan `masks`), their true and predicted values.

    Returns flat token arrays sorted by cell (cell, gene, target, pred; values log1p CP10K) and
    `ptr`, the offsets of each cell of `cells` (in the order given) into them.
    """
    model.eval()
    cells = np.asarray(cells, dtype=np.int64)
    batches = MaskedEvalBatches(
        ds, cfg, model.cfg.n_genes, plan_batches(ds, cells, cfg, token_budget), seed, masks=masks
    )
    device = next(model.parameters()).device
    parts = []
    for b in _loader(batches, num_workers):
        got = b.pop("cells")
        masked = b["tok_masked"]
        cell, gene, target = got[b["tok_doc"][masked]], b["tok_gene"][masked], b["tok_target"][masked]
        b = M.pad_tokens({k: v.to(device, non_blocking=True) for k, v in b.items()})
        layout = M.attention_layout(b["tok_doc"], model.cfg.attention_backend)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encoder(b, layout, loops or model.cfg.loops)
            pred = model.impute(h, b).cpu()
        parts.append((cell, gene, target, pred))
    cell, gene, target, pred = (torch.cat(x).numpy() for x in zip(*parts))
    rank = np.empty(ds.n_cells, np.int64)
    rank[cells] = np.arange(len(cells))
    order = np.argsort(rank[cell], kind="stable")
    counts = np.bincount(rank[cell], minlength=len(cells))
    return {
        "cells": cells,
        "cell": cell[order],
        "gene": gene[order].astype(np.int32),
        "target": target[order],
        "pred": pred[order],
        "ptr": np.r_[0, np.cumsum(counts)],
    }


@torch.inference_mode()
def predict_spatial(
    model,
    ds: Subset,
    centers: np.ndarray,
    cfg: DataConfig,
    token_budget: int = 1 << 19,
    num_workers: int = 8,
    seed: int = 0,
    loops: int | None = None,
    readout: str = "mean",
    n_samples: int = 64,
    source: str = "neighbours",
) -> dict:
    """Spatial imputation: each centre hidden, its latent predicted from its neighbours, then decoded.

    The prediction is log1p CP10K over the centre's panel genes, from the decoder's gene proportions
    (under the centre's batch) renormalised to that panel. readout "mean" is log1p of the expected
    CP10K; "expected_log" is the expected log1p CP10K under the model's negative binomial (n_samples
    draws at the neighbours' mean library size; the hidden cell's own library is not used), the point
    estimate a log-space metric rewards. source picks the latent that is decoded: "neighbours" (the
    task: the neighbourhood model's prediction), or two diagnostics that bound it, "self" (the hidden
    cell's own latent mean from its own tokens: what the latent bottleneck allows) and "neighbour_mean"
    (the plain mean of the neighbours' latent means: what the neighbourhood model adds). Returns, per centre in the order given,
    flat panel-gene arrays (gene, true, pred) with offsets `ptr`, and fine/coarse cell-type logits
    of the predicted latent.
    """
    if readout not in ("mean", "expected_log"):
        raise ValueError(f"readout {readout!r}; valid: mean, expected_log")
    if source not in ("neighbours", "self", "neighbour_mean"):
        raise ValueError(f"source {source!r}; valid: neighbours, self, neighbour_mean")
    model.eval()
    centers = np.asarray(centers, dtype=np.int64)
    k = min(cfg.neighbors, ds.neighbors.shape[1])
    lib = np.diff(np.concatenate([[0.0], np.cumsum(ds.values, dtype=np.float64)])[ds.row_ptr])  # library per cell
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    batches = SpatialEvalBatches(ds, cfg, model.cfg.n_genes, spatial_chunks(ds, centers, cfg, token_budget), seed)
    device = next(model.parameters()).device
    out = {}
    for b in _loader(batches, num_workers):
        chunk = b.pop("centers").numpy()
        b = M.pad_tokens({k: v.to(device, non_blocking=True) for k, v in b.items()})
        layout = M.attention_layout(b["tok_doc"], model.cfg.attention_backend)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encoder(b, layout, loops or model.cfg.loops)
            mu = model.to_latent(h[b["tok_cls"]]).float().chunk(2, -1)[0]
            members, masked = b["grp_members"], b["grp_masked"]
            if source == "self":
                z = mu[members[:, 0]]
            elif source == "neighbour_mean":
                valid = (members >= 0) & ~masked
                z = (mu[members.clamp(min=0)] * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
            else:
                z = model.neighborhood(mu[members.clamp(min=0)], b["grp_rel"], masked, members < 0)[:, 0].float()
            logits = model.decoder(z, b["batch"][members[:, 0]], task="spatial").float()
            types = [x.float() for x in model.cell_type_head(z)]
        for gi, c in enumerate(chunk):
            genes = np.sort(ds.panel_genes(ds.panel_id[c]))
            idx = torch.from_numpy(genes).to(device)
            rho = torch.softmax(logits[gi, idx], -1)
            if readout == "mean":
                value = torch.log1p(rho * 1e4)
            else:
                nb = ds.neighbors[c, :k]
                size = max(float(lib[nb[nb >= 0]].mean()), 1.0)
                theta = torch.exp(model.decoder.log_theta[idx].float()).clamp(max=1e6)
                rate = torch._standard_gamma(theta.expand(n_samples, -1).contiguous(), generator=gen) * (
                    rho * size / theta
                )
                value = torch.log1p(torch.poisson(rate, generator=gen) / size * 1e4).mean(0)
            out[int(c)] = (
                genes.astype(np.int32),
                panel_profile(ds.row_ptr, ds.gene_index, ds.values, c, genes),
                value.cpu().numpy(),
                [t[gi].cpu().numpy() for t in types],
            )
    genes, true, pred, types = zip(*(out[int(c)] for c in centers))
    return {
        "centers": centers,
        "gene": np.concatenate(genes),
        "true": np.concatenate(true),
        "pred": np.concatenate(pred),
        "ptr": np.r_[0, np.cumsum([len(g) for g in genes])],
        "cell_type": [np.stack([t[i] for t in types]) for i in range(len(types[0]))],
    }
