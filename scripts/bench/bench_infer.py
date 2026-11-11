"""Inference throughput on an evaluation subset: training-style evaluation vs brainwm_unified.infer.

    python scripts/bench/bench_infer.py --data /data/brainwm/eval_10k --checkpoint runs/cajal_1m/latest.pt

"train-style": 128 cells per batch (a training step's count), batches built in the main
process, torch.no_grad and the full training forward. "infer": infer.predict. Both see
the same cells; zero-token sampling differs, so predictions are compared by agreement.
"""

import argparse
import time

import numpy as np
import torch

from brainwm_unified import infer
from brainwm_unified import model as M
from brainwm_unified.config import DataConfig
from brainwm_unified.data import Subset, build_batch


def train_style(model, ds, cells, cfg, batch_cells=128):
    model.eval()
    mus, preds = [], []
    with torch.no_grad():
        for i in range(0, len(cells), batch_cells):
            chunk = cells[i : i + batch_cells]
            b = build_batch(ds, cfg, model.cfg.n_genes, np.random.default_rng(i), chunk, np.zeros(0, np.int64))
            order = np.unique(chunk)
            b = M.pad_tokens({k: v.cuda(non_blocking=True) for k, v in b.items()})
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b, M.attention_layout(b["tok_doc"], model.cfg.attention_backend))
                ct = model.cell_type_head(out["mu"])[-1].argmax(-1)
            pos = np.searchsorted(order, chunk)
            mus.append(out["mu"][pos].float().cpu())
            preds.append(ct[pos].cpu())
    return torch.cat(mus).numpy(), torch.cat(preds).numpy()


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def compare_backends(args, ds, cfg, cells):
    """Same cells and tokens (same seed) through infer.predict with two attention backends."""
    results = {}
    for backend in ("fa3", args.compare):
        model, _ = infer.load_model(args.checkpoint, attention_backend=backend)
        infer.predict(model, ds, cells[:2048], cfg, args.token_budget, args.workers)  # compile warm-up
        res, t = timed(lambda m=model: infer.predict(m, ds, cells, cfg, args.token_budget, args.workers))
        results[backend] = res
        print(f"{backend:8s} {len(cells) / t:7.0f} cells/s ({t:5.1f} s)")
    a, b = results["fa3"], results[args.compare]
    cos = np.sum(a["mu"] * b["mu"], 1) / (np.linalg.norm(a["mu"], axis=1) * np.linalg.norm(b["mu"], axis=1))
    agree = {k: np.mean(a[k][-1].argmax(-1) == b[k][-1].argmax(-1)) * 100 for k in ("cell_type", "anatomy")}
    print(
        f"{args.compare} vs fa3: cell-type agreement {agree['cell_type']:.2f}%, anatomy {agree['anatomy']:.2f}%, "
        f"latent cosine median {np.median(cos):.5f} min {cos.min():.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/brainwm/eval_10k")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--token-budget", type=int, default=1 << 19)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument(
        "--compare", default=None, help="attention backend to compare with fa3 on the same tokens, e.g. jagged"
    )
    args = ap.parse_args()

    ds, cfg = Subset(args.data), DataConfig(root=args.data)
    cells = np.arange(ds.n_cells)
    if args.compare:
        return compare_backends(args, ds, cfg, cells)
    model, _ = infer.load_model(args.checkpoint)
    print(f"{len(cells)} cells, token budget {args.token_budget}, {args.workers} workers")

    # Warm-up compiles the eval-mode graphs for both paths.
    train_style(model, ds, cells[:256], cfg)
    infer.predict(model, ds, cells[:2048], cfg, args.token_budget, args.workers)

    for rep in range(args.repeats):
        (mu_a, pred_a), t_a = timed(lambda: train_style(model, ds, cells, cfg))
        res, t_b = timed(lambda: infer.predict(model, ds, cells, cfg, args.token_budget, args.workers))
        pred_b = res["cell_type"][-1].argmax(-1)
        cos = np.sum(mu_a * res["mu"], 1) / (np.linalg.norm(mu_a, axis=1) * np.linalg.norm(res["mu"], axis=1))
        print(
            f"rep {rep}: train-style {len(cells) / t_a:7.0f} cells/s ({t_a:6.1f} s) | infer {len(cells) / t_b:7.0f} "
            f"cells/s ({t_b:6.1f} s) | x{t_a / t_b:.2f} | cell-type agreement {np.mean(pred_a == pred_b) * 100:.1f}% "
            f"| latent cosine median {np.median(cos):.4f}"
        )


if __name__ == "__main__":
    main()
