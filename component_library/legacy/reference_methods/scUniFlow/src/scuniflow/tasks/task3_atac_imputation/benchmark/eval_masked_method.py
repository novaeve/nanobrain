#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_dense_rows(X, rows):
    if sp.issparse(X):
        return X[rows].toarray().astype(np.float32, copy=False)
    return np.asarray(X[rows], dtype=np.float32)


def _split_val_indices(n_cells, val_cells, seed):
    n_val = max(1, min(int(val_cells), int(n_cells) - 1))
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(int(n_cells))
    return np.sort(perm[:n_val]).astype(np.int64)


def _make_mask(row, local_cell_idx, nonzero_ratio, zero_ratio, mask_seed):
    row_t = torch.from_numpy(np.asarray(row, dtype=np.float32))
    mask = torch.zeros(row_t.numel(), dtype=torch.bool)
    g = torch.Generator()
    g.manual_seed(int(mask_seed) + int(local_cell_idx))

    nz = (row_t > 0).nonzero(as_tuple=True)[0]
    z = (row_t <= 0).nonzero(as_tuple=True)[0]

    n_nz = int(nz.numel() * float(nonzero_ratio))
    if n_nz > 0 and nz.numel() > 0:
        perm = torch.randperm(nz.numel(), generator=g)
        mask[nz[perm[:n_nz]]] = True

    n_z = int(z.numel() * float(zero_ratio))
    if n_z > 0 and z.numel() > 0:
        perm = torch.randperm(z.numel(), generator=g)
        mask[z[perm[:n_z]]] = True

    return mask.numpy()


def main():
    parser = argparse.ArgumentParser(description="Masked AUROC/AUPRC for scATAC benchmark outputs")
    parser.add_argument("--raw_atac_h5ad", required=True)
    parser.add_argument("--imputed_atac_h5ad", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--val_cells", type=int, default=500)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_eval_cells", type=int, default=32)
    parser.add_argument("--nonzero_mask_ratio", type=float, default=0.2)
    parser.add_argument("--zero_mask_ratio", type=float, default=0.02)
    parser.add_argument("--mask_seed", type=int, default=1234)
    args = parser.parse_args()

    raw = sc.read_h5ad(args.raw_atac_h5ad)
    imp = sc.read_h5ad(args.imputed_atac_h5ad)

    raw_obs = raw.obs_names.astype(str)
    imp_obs = imp.obs_names.astype(str)
    raw_var = raw.var_names.astype(str)
    imp_var = imp.var_names.astype(str)

    var_pos_imp = imp_var.get_indexer(raw_var)
    if np.any(var_pos_imp < 0):
        raise ValueError(f"imputed output missing raw peaks: {int(np.sum(var_pos_imp < 0))}")

    val_idx = _split_val_indices(raw.n_obs, args.val_cells, args.split_seed)
    obs_pos_imp_val = imp_obs.get_indexer(raw_obs[val_idx])
    present = obs_pos_imp_val >= 0
    if not np.any(present):
        raise ValueError("No validation cells overlap the imputed output")
    val_positions = np.where(present)[0].astype(np.int64)
    n_eval = max(1, min(int(args.max_eval_cells), val_positions.size))
    eval_val_positions = val_positions[:n_eval]
    eval_val_idx = val_idx[eval_val_positions]
    eval_imp_idx = obs_pos_imp_val[eval_val_positions]

    raw_rows = _as_dense_rows(raw.X, eval_val_idx)
    imp_rows = _as_dense_rows(imp.X, eval_imp_idx)[:, var_pos_imp]

    y_true_chunks = []
    y_score_chunks = []
    masked_total = 0
    for row_i in range(raw_rows.shape[0]):
        mask = _make_mask(
            raw_rows[row_i],
            local_cell_idx=eval_val_positions[row_i],
            nonzero_ratio=args.nonzero_mask_ratio,
            zero_ratio=args.zero_mask_ratio,
            mask_seed=args.mask_seed,
        )
        if not np.any(mask):
            continue
        y_true_chunks.append((raw_rows[row_i, mask] > 0).astype(np.int32, copy=False))
        y_score_chunks.append(imp_rows[row_i, mask].astype(np.float32, copy=False))
        masked_total += int(mask.sum())

    if not y_true_chunks:
        raise ValueError("No masked positions selected")

    y_true = np.concatenate(y_true_chunks)
    y_score = np.concatenate(y_score_chunks)
    positive_rate = float(y_true.mean()) if y_true.size else 0.0
    try:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    except ValueError:
        auroc = 0.5
        auprc = positive_rate

    summary = {
        "raw_atac_h5ad": str(Path(args.raw_atac_h5ad).resolve()),
        "imputed_atac_h5ad": str(Path(args.imputed_atac_h5ad).resolve()),
        "n_reference_cells": int(raw.n_obs),
        "n_reference_peaks": int(raw.n_vars),
        "val_cells": int(args.val_cells),
        "split_seed": int(args.split_seed),
        "max_eval_cells": int(args.max_eval_cells),
        "eval_cells": int(eval_val_idx.size),
        "n_imputed_overlap_val_cells": int(val_positions.size),
        "nonzero_mask_ratio": float(args.nonzero_mask_ratio),
        "zero_mask_ratio": float(args.zero_mask_ratio),
        "mask_seed": int(args.mask_seed),
        "masked_total": int(masked_total),
        "masked_nonzero": int(y_true.sum()),
        "positive_rate": positive_rate,
        "auroc": auroc,
        "auprc": auprc,
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
