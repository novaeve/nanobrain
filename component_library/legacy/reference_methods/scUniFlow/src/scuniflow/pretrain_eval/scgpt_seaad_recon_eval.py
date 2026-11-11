import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
from scipy.sparse import issparse

import sys
sys.path.insert(0, "external/scGPT")

from scgpt.model.model import TransformerModel
from scgpt.utils.util import load_pretrained
from donor_eval_utils import (
    DEFAULT_DONOR_CSV,
    DEFAULT_DONOR_DIR,
    pick_eval_donor,
    sample_balanced_val_donors,
    sample_single_donor_cells,
    split_val_donors,
)
from scgpt_eval_utils import (
    SimpleGeneVocab,
    evaluate_mlm_metrics,
    prepare_dataloader,
    random_mask_value,
    summarize_mask_stats,
    tokenize_and_pad_batch,
)

DEFAULT_RESULT_DIR = Path("results/pretrain_eval")


def build_result_path(args, donor_id: str, sampled_cells: int, per_donor_n: int) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_subset_mode == "balanced_val_donors":
        filename = (
            f"scgpt_pretrained_recon_{Path(args.ckpt_dir).name}_"
            f"{'fullseq' if args.include_zero_gene else 'nonzero'}_"
            f"{args.eval_space}_seed{args.split_seed}_{donor_id}_"
            f"perdonor{per_donor_n}_total{sampled_cells}_mask{args.mask_ratio:.2f}.json"
        )
    else:
        filename = (
            f"scgpt_pretrained_recon_{Path(args.ckpt_dir).name}_"
            f"{'fullseq' if args.include_zero_gene else 'nonzero'}_"
            f"{args.eval_space}_seed{args.split_seed}_{donor_id}_"
            f"n{sampled_cells}_mask{args.mask_ratio:.2f}.json"
        )
    return result_dir / filename.replace("/", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="external/scGPT/checkpoint/scGPT_human")
    parser.add_argument("--max_seq_len", type=int, default=1201)
    parser.add_argument("--eval_space", type=str, default="log1p", choices=["log1p", "binned"])
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="Mask ratio for nonzero gene tokens. If --zero_mask_ratio is unset, zero tokens follow this same ratio.")
    parser.add_argument("--zero_mask_ratio", type=float, default=None, help="Optional separate mask ratio for zero-value gene tokens. Default: use --mask_ratio.")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--mask_seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--donor_csv", type=str, default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", type=str, default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--eval_donor", type=str, default="")
    parser.add_argument(
        "--eval_subset_mode",
        type=str,
        default="balanced_val_donors",
        choices=["balanced_val_donors", "single_donor"],
        help="balanced_val_donors: sample the same number of cells from every val donor; single_donor: evaluate one donor only.",
    )
    parser.add_argument("--include_zero_gene", action="store_true", help="If set, keep zero-expression genes in the token sequence before truncation.")
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    donor_csv = Path(args.donor_csv)
    donor_dir = Path(args.donor_dir)
    ckpt_dir = Path(args.ckpt_dir)

    print(
        f"[setup] ckpt_dir={ckpt_dir} subset_mode={args.eval_subset_mode} "
        f"include_zero_gene={args.include_zero_gene}",
        flush=True,
    )

    if args.eval_subset_mode == "balanced_val_donors":
        adata, donor_stats, per_donor_n = sample_balanced_val_donors(
            donor_csv=donor_csv,
            donor_dir=donor_dir,
            split_seed=args.split_seed,
            sample_seed=args.sample_seed,
            concat_join="inner",
        )
        donor_id = "balanced_val_donors"
        donor_cells = int(sum(x["donor_cells_total"] for x in donor_stats))
        n = int(adata.n_obs)
    else:
        if args.eval_donor:
            donor_id = args.eval_donor
            donor_cells = sc.read_h5ad(donor_dir / f"{donor_id}.h5ad", backed="r").n_obs
        else:
            val_donors = split_val_donors(donor_csv, val_ratio=0.1, seed=args.split_seed)
            donor_id, donor_cells = pick_eval_donor(val_donors, donor_dir, min_cells=args.sample_n)
        adata, _, n = sample_single_donor_cells(
            donor_path=donor_dir / f"{donor_id}.h5ad",
            sample_n=args.sample_n,
            sample_seed=args.sample_seed,
        )
        donor_stats = [
            {
                "donor_id": donor_id,
                "donor_cells_total": int(donor_cells),
                "sampled_cells": int(n),
                "sample_seed": int(args.sample_seed),
            }
        ]
        per_donor_n = int(n)

    vocab = SimpleGeneVocab.from_file(ckpt_dir / "vocab.json")
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])

    with (ckpt_dir / "args.json").open() as f:
        mcfg = json.load(f)

    if "feature_name" not in adata.var.columns:
        raise RuntimeError("SEA-AD donor h5ad has no feature_name for mapping to scGPT vocab.")

    gene_names = adata.var["feature_name"].astype(str).to_numpy()
    in_vocab = np.array([g in vocab for g in gene_names])
    kept_gene_names = gene_names[in_vocab]
    gene_ids = np.array(vocab(kept_gene_names.tolist()), dtype=int)

    X = adata.X[:, in_vocab]
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    X = np.log1p(X)

    n_bins = int(args.n_bins if args.n_bins > 0 else mcfg.get("n_bins", 51))
    if args.eval_space == "binned":
        X_eval = np.zeros_like(X, dtype=np.int64)
        # per-cell bin edges for inverse mapping to log1p
        bin_edges = np.zeros((X.shape[0], n_bins - 1), dtype=np.float32)
        for i in range(X.shape[0]):
            row = X[i]
            if row.max() == 0:
                X_eval[i] = np.zeros_like(row, dtype=np.int64)
                continue
            if row.min() <= 0:
                non_zero_ids = np.nonzero(row)[0]
                non_zero_row = row[non_zero_ids]
                edges = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1)).astype(np.float32)
                digits = np.digitize(non_zero_row, edges).astype(np.int64)
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = digits
            else:
                edges = np.quantile(row, np.linspace(0, 1, n_bins - 1)).astype(np.float32)
                binned_row = np.digitize(row, edges).astype(np.int64)
            binned_row = np.clip(binned_row, 0, n_bins - 1)
            X_eval[i] = binned_row
            bin_edges[i] = edges
    else:
        # Directly evaluate in continuous log1p space
        X_eval = X.astype(np.float32)
        bin_edges = None

    max_seq_len = args.max_seq_len
    pad_value = int(mcfg.get("pad_value", -2))
    mask_value = int(mcfg.get("mask_value", -1))

    print("[prep] tokenizing evaluation matrix", flush=True)
    tokenized = tokenize_and_pad_batch(
        X_eval,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=args.include_zero_gene,
        show_progress=True,
        progress_desc="tokenizing",
    )

    print("[prep] applying random masks", flush=True)
    masked_values = random_mask_value(
        tokenized["values"],
        mask_ratio=args.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
        base_seed=args.mask_seed,
        keep_first_n_tokens=1,
        zero_mask_ratio=args.zero_mask_ratio,
    )
    mask_stats = summarize_mask_stats(
        original_values=tokenized["values"],
        masked_values=masked_values,
        pad_value=pad_value,
        mask_value=mask_value,
        keep_first_n_tokens=1,
    )

    data_pt = {
        "gene_ids": tokenized["genes"],
        "values": masked_values,
        "target_values": tokenized["values"],
        "batch_labels": torch.zeros(tokenized["genes"].shape[0], dtype=torch.long),
    }
    if bin_edges is not None:
        data_pt["bin_edges"] = torch.from_numpy(bin_edges).float()

    loader = prepare_dataloader(data_pt=data_pt, batch_size=args.batch_size)

    print(
        f"[eval] donor={donor_id} sampled_cells={n} batch_size={args.batch_size} "
        f"total_batches={len(loader)} mask_ratio={args.mask_ratio} "
        f"zero_mask_ratio={args.zero_mask_ratio if args.zero_mask_ratio is not None else args.mask_ratio} "
        f"seq_len={max_seq_len} include_zero_gene={args.include_zero_gene}",
        flush=True,
    )

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=mcfg["embsize"],
        nhead=mcfg["nheads"],
        d_hid=mcfg["d_hid"],
        nlayers=mcfg["nlayers"],
        nlayers_cls=mcfg.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=mcfg["dropout"],
        pad_token=mcfg["pad_token"],
        pad_value=pad_value,
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=mcfg.get("input_emb_style", "continuous"),
        n_input_bins=n_bins,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    load_pretrained(model, state, verbose=False)
    model.to(device)

    print(f"[eval] starting model forward passes on device={device}", flush=True)
    metrics = evaluate_mlm_metrics(
        model=model,
        loader=loader,
        vocab=vocab,
        device=device,
        pad_token="<pad>",
        mask_value=mask_value,
        show_progress=True,
        eval_space=args.eval_space,
        n_bins=n_bins,
    )

    out = {
        "checkpoint": str(ckpt_dir),
        "dataset": "SEA-AD MTG donor_h5ad",
        "donor_id": donor_id,
        "donor_cells_total": int(donor_cells),
        "eval_subset_mode": args.eval_subset_mode,
        "balanced_per_donor_cells": int(per_donor_n),
        "donor_sampling": donor_stats,
        "sampled_cells": int(n),
        "split_rule": "scUniFlow donor split val_ratio=0.1 seed=42",
        "unseen_for_scuniflow_train": True,
        "matched_genes_in_vocab": int(in_vocab.sum()),
        "all_genes": int(len(in_vocab)),
        "max_seq_len": int(max_seq_len),
        "include_zero_gene": bool(args.include_zero_gene),
        "eval_space": args.eval_space,
        "n_bins": int(n_bins if args.eval_space == "binned" else 0),
        "mask_ratio_target": args.mask_ratio,
        "nonzero_mask_ratio_target": float(args.mask_ratio),
        "zero_mask_ratio_target": float(args.zero_mask_ratio if args.zero_mask_ratio is not None else args.mask_ratio),
        "batch_size": int(args.batch_size),
        **mask_stats,
        **metrics,
    }

    result_path = build_result_path(args, donor_id, sampled_cells=n, per_donor_n=per_donor_n)
    out["result_json"] = str(result_path)
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    print(f"[done] wrote result json to {result_path}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
