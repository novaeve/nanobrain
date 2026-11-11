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
from tqdm.auto import tqdm

import sys
sys.path.insert(0, "data/SEA-AD/MJM")

from src.models.scGPT.scGPT import TransformerModel
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


DEFAULT_EXPERIMENT_ROOT = Path("data/SEA-AD/MJM/experiments")
DEFAULT_MERFISH_H5AD = Path("data/SEA-AD/MTG/mtg_spatial_data/SEAAD_MTG_MERFISH.2024-12-11.h5ad")
DEFAULT_RESULT_DIR = Path("results/pretrain_eval")

PRETRAINED_DIRS = {
    "brain": Path("external/scGPT/checkpoint/brain"),
    "human": Path("external/scGPT/checkpoint/human"),
}

FINETUNED_MODEL_DIRS = {
    "brain": "scGPT_brain",
    "human": "scGPT_human",
}


def bin_expression_matrix(
    x_log: np.ndarray,
    n_bins: int,
    show_progress: bool = True,
    desc: str = "binning cells",
):
    x_eval = np.zeros_like(x_log, dtype=np.int64)
    bin_edges = np.zeros((x_log.shape[0], n_bins - 1), dtype=np.float32)
    row_iter = range(x_log.shape[0])
    if show_progress:
        row_iter = tqdm(row_iter, total=x_log.shape[0], desc=desc, unit="cell")
    for i in row_iter:
        row = x_log[i]
        if row.max() == 0:
            x_eval[i] = np.zeros_like(row, dtype=np.int64)
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
        x_eval[i] = binned_row
        bin_edges[i] = edges
    return x_eval, bin_edges


def resolve_finetuned_ckpt(args) -> Path:
    if args.finetuned_ckpt:
        ckpt = Path(args.finetuned_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt}")
        return ckpt

    ckpt = (
        Path(args.experiment_root)
        / FINETUNED_MODEL_DIRS[args.variant]
        / "SEA_AD_MTP_ST"
        / "checkpoints"
        / f"{args.tuning_mode}_s{args.seed}"
        / "best_model.pth"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt}")
    return ckpt


def build_merfish_panel_matrix(adata, gene_panel_names):
    donor_gene_names = (
        adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in adata.var.columns
        else adata.var.index.astype(str).to_numpy()
    )
    donor_gene_to_col = {}
    for i, g in enumerate(donor_gene_names):
        if g not in donor_gene_to_col:
            donor_gene_to_col[g] = i

    matched_genes = [g for g in gene_panel_names if g in donor_gene_to_col]
    missing_genes = [g for g in gene_panel_names if g not in donor_gene_to_col]
    src_cols = [donor_gene_to_col[g] for g in matched_genes]
    dst_cols = [gene_panel_names.index(g) for g in matched_genes]

    X_sub = adata.X[:, src_cols]
    if issparse(X_sub):
        X_sub = X_sub.toarray()
    X_sub = X_sub.astype(np.float32)

    X_panel = np.zeros((adata.n_obs, len(gene_panel_names)), dtype=np.float32)
    X_panel[:, dst_cols] = X_sub
    return X_panel, matched_genes, missing_genes


def build_full_gene_matrix(adata, vocab: SimpleGeneVocab):
    if "feature_name" not in adata.var.columns:
        raise RuntimeError("SEA-AD donor h5ad has no feature_name for mapping to scGPT vocab.")

    donor_gene_names = adata.var["feature_name"].astype(str).to_numpy()
    in_vocab = np.array([g in vocab for g in donor_gene_names])
    kept_gene_names = donor_gene_names[in_vocab]
    if kept_gene_names.size == 0:
        raise RuntimeError("No donor genes overlap with scGPT vocab.")

    gene_ids = np.array(vocab(kept_gene_names.tolist()), dtype=np.int64)
    x_full = adata.X[:, in_vocab]
    if issparse(x_full):
        x_full = x_full.toarray()
    x_full = x_full.astype(np.float32)

    missing_gene_names = donor_gene_names[~in_vocab].tolist()
    return x_full, gene_ids, kept_gene_names.tolist(), missing_gene_names


def load_finetuned_scgpt_model(pretrained_dir: Path, finetuned_ckpt: Path, variant: str):
    vocab = SimpleGeneVocab.from_file(pretrained_dir / "vocab.json")
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])

    with (pretrained_dir / "args.json").open() as f:
        mcfg = json.load(f)

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
        pad_value=mcfg.get("pad_value", -2),
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=mcfg.get("input_emb_style", "continuous"),
        n_input_bins=int(mcfg.get("n_bins", 51)),
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )

    state = torch.load(finetuned_ckpt, map_location="cpu")
    gene_ids = state.get("gene_ids", None)
    backbone_state = {
        k[len("backbone."):]: v
        for k, v in state.items()
        if k.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to cleanly load fine-tuned scGPT {variant} checkpoint.\n"
            f"missing={missing}\nunexpected={unexpected}"
        )

    if gene_ids is None:
        raise RuntimeError("Fine-tuned scGPT checkpoint does not contain top-level gene_ids.")

    return model, vocab, mcfg, gene_ids.detach().cpu().numpy().astype(np.int64)


def build_result_path(args, donor_id: str, sampled_cells: int, per_donor_n: int) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_subset_mode == "balanced_val_donors":
        filename = (
            f"scgpt_finetuned_recon_{args.variant}_{args.tuning_mode}_{args.gene_space}_"
            f"seed{args.seed}_{donor_id}_perdonor{per_donor_n}_total{sampled_cells}_"
            f"mask{args.mask_ratio:.2f}.json"
        )
    else:
        filename = (
            f"scgpt_finetuned_recon_{args.variant}_{args.tuning_mode}_{args.gene_space}_"
            f"seed{args.seed}_{donor_id}_n{sampled_cells}_mask{args.mask_ratio:.2f}.json"
        )
    filename = filename.replace("/", "_")
    return result_dir / filename


def main():
    parser = argparse.ArgumentParser(description="Reconstruction eval for SEA-AD fine-tuned scGPT checkpoints.")
    parser.add_argument("--variant", type=str, default="brain", choices=["brain", "human"])
    parser.add_argument("--tuning_mode", type=str, default="freeze", choices=["freeze", "full"])
    parser.add_argument("--seed", type=int, default=3028)
    parser.add_argument("--finetuned_ckpt", type=str, default="")
    parser.add_argument("--pretrained_dir", type=str, default="")
    parser.add_argument("--merfish_h5ad", type=str, default=str(DEFAULT_MERFISH_H5AD))
    parser.add_argument("--max_seq_len", type=int, default=0, help="<=0 means auto: full uses all kept genes + <cls>; merfish140 uses 141.")
    parser.add_argument("--gene_space", type=str, default="full", choices=["full", "merfish140"])
    parser.add_argument("--eval_space", type=str, default="binned", choices=["log1p", "binned"])
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="Mask ratio for nonzero gene tokens. If --zero_mask_ratio is unset, zero tokens follow this same ratio.")
    parser.add_argument("--zero_mask_ratio", type=float, default=None, help="Optional separate mask ratio for zero-value gene tokens. Default: use --mask_ratio.")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--mask_seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--experiment_root", type=str, default=str(DEFAULT_EXPERIMENT_ROOT))
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
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    donor_csv = Path(args.donor_csv)
    donor_dir = Path(args.donor_dir)
    finetuned_ckpt = resolve_finetuned_ckpt(args)
    pretrained_dir = Path(args.pretrained_dir) if args.pretrained_dir else PRETRAINED_DIRS[args.variant]
    print(
        f"[setup] variant={args.variant} tuning_mode={args.tuning_mode} "
        f"subset_mode={args.eval_subset_mode} gene_space={args.gene_space}",
        flush=True,
    )
    print(f"[setup] fine-tuned checkpoint: {finetuned_ckpt}", flush=True)

    if args.eval_subset_mode == "balanced_val_donors":
        adata, donor_stats, per_donor_n = sample_balanced_val_donors(
            donor_csv=donor_csv,
            donor_dir=donor_dir,
            split_seed=args.split_seed,
            sample_seed=args.sample_seed,
            concat_join="outer",
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
        donor_stats = [{
            "donor_id": donor_id,
            "donor_cells_total": int(donor_cells),
            "sampled_cells": int(n),
            "sample_seed": int(args.sample_seed),
        }]
        per_donor_n = int(n)

    print(f"[model] loading fine-tuned scGPT backbone from {finetuned_ckpt}", flush=True)
    model, vocab, mcfg, checkpoint_gene_ids = load_finetuned_scgpt_model(pretrained_dir, finetuned_ckpt, args.variant)

    if args.gene_space == "merfish140":
        print(f"[data] loading MERFISH panel from {args.merfish_h5ad}", flush=True)
        merfish_adata = sc.read_h5ad(Path(args.merfish_h5ad), backed="r")
        gene_panel_names = [str(g) for g in merfish_adata.var_names if not str(g).startswith("Blank")]
        if len(gene_panel_names) != 140:
            raise RuntimeError(f"Expected 140 MERFISH genes, got {len(gene_panel_names)} from {args.merfish_h5ad}")

        print("[data] projecting sampled cells to MERFISH-140 gene panel", flush=True)
        x_input, matched_genes, missing_genes = build_merfish_panel_matrix(adata, gene_panel_names)
        X_log = np.log1p(x_input.astype(np.float32))
        gene_ids = checkpoint_gene_ids
        if len(gene_ids) != len(gene_panel_names):
            raise RuntimeError(
                f"Checkpoint gene_ids length {len(gene_ids)} does not match MERFISH panel size {len(gene_panel_names)}."
            )
        gene_space_meta = {
            "gene_space": "merfish140",
            "dataset": "SEA-AD MTG donor_h5ad on MERFISH-140 panel",
            "gene_panel_size": int(len(gene_panel_names)),
            "matched_genes_in_input": int(len(matched_genes)),
            "missing_genes_in_input": missing_genes,
            "used_checkpoint_gene_ids": True,
            "checkpoint_gene_ids_length": int(len(checkpoint_gene_ids)),
        }
    else:
        print("[data] building full-gene input matrix from donor h5ad", flush=True)
        x_input, gene_ids, kept_gene_names, missing_gene_names = build_full_gene_matrix(adata, vocab)
        X_log = np.log1p(x_input.astype(np.float32))
        matched_genes = kept_gene_names
        missing_genes = missing_gene_names
        print(
            f"[data] full-gene overlap with vocab: {len(kept_gene_names)}/{len(kept_gene_names) + len(missing_gene_names)}",
            flush=True,
        )
        print(
            f"[data] checkpoint was fine-tuned with {len(checkpoint_gene_ids)} tracked genes; "
            f"full eval uses {len(gene_ids)} donor genes mapped through pretrained vocab",
            flush=True,
        )
        gene_space_meta = {
            "gene_space": "full",
            "dataset": "SEA-AD MTG donor_h5ad full gene space",
            "genes_in_vocab": int(len(kept_gene_names)),
            "genes_missing_from_vocab": int(len(missing_gene_names)),
            "missing_gene_names_preview": missing_gene_names[:50],
            "used_checkpoint_gene_ids": False,
            "checkpoint_gene_ids_length": int(len(checkpoint_gene_ids)),
        }

    n_bins = int(args.n_bins if args.n_bins > 0 else mcfg.get("n_bins", 51))
    if args.eval_space == "binned":
        print(f"[prep] binning expression matrix into {n_bins} bins", flush=True)
        X_eval, bin_edges = bin_expression_matrix(
            X_log,
            n_bins=n_bins,
            show_progress=True,
            desc="binning cells",
        )
    else:
        X_eval = X_log.astype(np.float32)
        bin_edges = None

    pad_value = int(mcfg.get("pad_value", -2))
    mask_value = int(mcfg.get("mask_value", -1))
    effective_max_seq_len = int(args.max_seq_len) if args.max_seq_len and args.max_seq_len > 0 else int(len(gene_ids) + 1)

    print("[prep] tokenizing evaluation matrix", flush=True)
    tokenized = tokenize_and_pad_batch(
        X_eval,
        gene_ids,
        max_len=effective_max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
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
        f"[eval] variant={args.variant} tuning_mode={args.tuning_mode} donor={donor_id} "
        f"sampled_cells={n} batch_size={args.batch_size} total_batches={len(loader)} "
        f"mask_ratio={args.mask_ratio} "
        f"zero_mask_ratio={args.zero_mask_ratio if args.zero_mask_ratio is not None else args.mask_ratio} "
        f"seq_len={effective_max_seq_len} "
        f"genes_used={len(gene_ids)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        progress_desc="ft-scGPT eval",
    )

    out = {
        "finetuned_checkpoint": str(finetuned_ckpt),
        "pretrained_dir": str(pretrained_dir),
        "variant": args.variant,
        "tuning_mode": args.tuning_mode,
        "donor_id": donor_id,
        "donor_cells_total": int(donor_cells),
        "eval_subset_mode": args.eval_subset_mode,
        "balanced_per_donor_cells": int(per_donor_n),
        "donor_sampling": donor_stats,
        "sampled_cells": int(n),
        "split_rule": "scUniFlow donor split val_ratio=0.1 seed=42",
        "unseen_for_scuniflow_train": True,
        "max_seq_len": int(effective_max_seq_len),
        "eval_space": args.eval_space,
        "n_bins": int(n_bins if args.eval_space == "binned" else 0),
        "mask_ratio_target": args.mask_ratio,
        "nonzero_mask_ratio_target": float(args.mask_ratio),
        "zero_mask_ratio_target": float(args.zero_mask_ratio if args.zero_mask_ratio is not None else args.mask_ratio),
        "batch_size": int(args.batch_size),
        **mask_stats,
        **gene_space_meta,
        **metrics,
    }

    result_path = build_result_path(args, donor_id, sampled_cells=n, per_donor_n=per_donor_n)
    out["result_json"] = str(result_path)
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    print(f"[done] wrote result json to {result_path}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
