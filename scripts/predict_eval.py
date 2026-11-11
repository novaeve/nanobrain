"""Zero-shot outputs of a checkpoint on an evaluation subset, saved for scoring.

    python scripts/predict_eval.py --checkpoint runs/cajal_1m/latest.pt --data /data/brainwm/eval_10k \
        --out runs/eval/cajal_step30000.npz

Writes the latent mean (integration), per-level label logits (annotation) and the cells'
true labels and batches, so scoring needs no GPU or model code.

--task genes: the training gene mask (fixed seed) on every cell, imputed by the gene head.
--task spatial: every centre with a full neighbourhood hidden and rebuilt from its neighbours
(an eval set of whole spatial tiles, e.g. /data/brainwm/eval_spatial). Both store per-cell
sufficient statistics (brainwm_unified.impute_metrics) and are scored by eval_imputation.py.
"""

import argparse
import os
import time

import numpy as np
import torch

from brainwm_unified import impute_metrics, infer
from brainwm_unified.config import DataConfig
from brainwm_unified.data import Subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="/data/brainwm/eval_10k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--token-budget", type=int, default=1 << 19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["embed", "genes", "spatial"], default="embed")
    ap.add_argument("--readout", choices=["mean", "expected_log"], default="mean", help="spatial: point estimate")
    ap.add_argument(
        "--source",
        choices=["neighbours", "self", "neighbour_mean"],
        default="neighbours",
        help="spatial: latent decoded (self / neighbour_mean are diagnostics, not the task)",
    )
    args = ap.parse_args()

    ds = Subset(args.data)
    cells = np.arange(ds.n_cells)
    model, state = infer.load_model(args.checkpoint)
    if args.task != "embed":
        return save_imputation(args, ds, model, state)
    t = time.perf_counter()
    res = infer.predict(model, ds, cells, DataConfig(root=args.data), args.token_budget, seed=args.seed)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        mu=res["mu"],
        **{f"cell_type_logits_{i}": x for i, x in enumerate(res["cell_type"])},
        **{f"anatomy_logits_{i}": x for i, x in enumerate(res["anatomy"])},
        cell_type=ds.cell_type,
        anatomy=ds.anatomy,
        batch=ds.integration_batch,
        **{k: np.load(os.path.join(args.data, k + ".npy")) for k in ("dataset", "technology", "assay")},
        step=state["step"],
        seconds=seconds,
    )
    print(f"{len(cells)} cells in {seconds:.1f} s (step {state['step']}) -> {args.out}")


def mask_path(out: str, data: str, seed: int) -> str:
    """Where the gene task's masked tokens are kept, next to the predictions."""
    return os.path.join(os.path.dirname(os.path.abspath(out)), f"genes_mask_{os.path.basename(data)}_seed{seed}.npz")


def save_imputation(args, ds, model, state):
    cfg = DataConfig(root=args.data)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t = time.perf_counter()
    if args.task == "genes":
        res = infer.predict_genes(model, ds, np.arange(ds.n_cells), cfg, args.token_budget, seed=args.seed)
        rows, pred, true, extra = res["cells"], res["pred"], res["target"], {}
        mask = mask_path(args.out, args.data, args.seed)
        if not os.path.exists(mask):  # the masked tokens depend on data and seed only: baselines reuse them
            np.savez_compressed(mask, cell=res["cell"], gene=res["gene"], target=res["target"], ptr=res["ptr"])
    else:
        if not len(ds.spatial_centers):
            raise SystemExit(f"{args.data} has no spatial centres with neighbours")
        res = infer.predict_spatial(
            model,
            ds,
            ds.spatial_centers,
            cfg,
            args.token_budget,
            seed=args.seed,
            readout=args.readout,
            source=args.source,
        )
        rows, pred, true = res["centers"], res["pred"], res["true"]
        extra = {f"cell_type_logits_{i}": x for i, x in enumerate(res["cell_type"])}
        extra["cell_type"] = ds.cell_type[rows]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    stats = impute_metrics.row_stats(pred, true, res["ptr"])
    np.savez_compressed(
        args.out, task=args.task, rows=rows, seed=args.seed, step=state["step"], seconds=seconds, **stats, **extra
    )
    print(f"{args.task}: {len(rows)} rows in {seconds:.1f} s (step {state['step']}) {impute_metrics.summarize(stats)}")


if __name__ == "__main__":
    main()
