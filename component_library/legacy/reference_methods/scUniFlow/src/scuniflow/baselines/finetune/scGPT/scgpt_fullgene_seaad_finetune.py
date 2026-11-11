import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import json
import time
import shutil
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import issparse
from tqdm.auto import tqdm

import sys

THIS_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = THIS_DIR.parent
EVAL_DIR = FINETUNE_DIR.parent
REPO_ROOT = EVAL_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, "external/scGPT")

from data import split_donors_train_val
from scgpt.model.model import TransformerModel
from scgpt.utils.util import load_pretrained
from scgpt.loss import masked_mse_loss, masked_relative_error
from donor_eval_utils import DEFAULT_DONOR_CSV, DEFAULT_DONOR_DIR, sample_single_donor_cells
from scgpt_eval_utils import SimpleGeneVocab, prepare_dataloader, random_mask_value, summarize_mask_stats


DEFAULT_RESULT_ROOT = Path("./src/scuniflow/baselines/finetune/outputs")

PRETRAINED_DIRS = {
    "brain": Path("external/scGPT/checkpoint/brain"),
    "human": Path("external/scGPT/checkpoint/human"),
}


def load_donor_cell_count_map(donor_csv: Path):
    donor_counts = {}
    with donor_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            donor_counts[row["donor_id"]] = int(row["n_cells"])
    return donor_counts


def allocate_cells_across_donors(donor_ids, donor_counts, total_cells: int, seed: int):
    available = []
    for donor_id in donor_ids:
        n = int(donor_counts.get(donor_id, 0))
        if n > 0:
            available.append((donor_id, n))
    if not available:
        return []

    total_available = sum(n for _, n in available)
    if total_cells is None or total_cells <= 0 or total_cells >= total_available:
        return [
            {"donor_id": donor_id, "alloc_cells": int(n), "donor_cells_total": int(n)}
            for donor_id, n in available
        ]

    donor_ids_arr = [x[0] for x in available]
    counts = np.array([x[1] for x in available], dtype=np.int64)
    raw = counts.astype(np.float64) * (float(total_cells) / float(total_available))
    alloc = np.floor(raw).astype(np.int64)
    remaining = int(total_cells - alloc.sum())

    frac = raw - alloc.astype(np.float64)
    rng = np.random.RandomState(seed)
    tie = rng.random(len(available))
    order = np.lexsort((-tie, -frac))
    for idx in order:
        if remaining <= 0:
            break
        if alloc[idx] < counts[idx]:
            alloc[idx] += 1
            remaining -= 1

    if remaining > 0:
        capacity = counts - alloc
        order2 = np.argsort(-capacity)
        for idx in order2:
            if remaining <= 0:
                break
            if capacity[idx] <= 0:
                continue
            take = min(int(capacity[idx]), remaining)
            alloc[idx] += take
            remaining -= take

    plan = []
    for donor_id, donor_total, donor_alloc in zip(donor_ids_arr, counts.tolist(), alloc.tolist()):
        if donor_alloc > 0:
            plan.append(
                {
                    "donor_id": donor_id,
                    "alloc_cells": int(donor_alloc),
                    "donor_cells_total": int(donor_total),
                }
            )
    return plan


def build_subset_from_donors(
    donor_ids,
    donor_counts,
    donor_dir: Path,
    total_cells: int,
    donor_seed: int,
    cell_seed: int,
    split_label: str,
):
    plan = allocate_cells_across_donors(
        donor_ids=donor_ids,
        donor_counts=donor_counts,
        total_cells=total_cells,
        seed=donor_seed,
    )
    if not plan:
        raise RuntimeError(f"No donors selected for {split_label}.")

    print(
        f"[data] {split_label}: total_cells_budget={total_cells}, "
        f"selected_donors={len(plan)}, donor_seed={donor_seed}, cell_seed={cell_seed}",
        flush=True,
    )

    adatas = []
    donor_stats = []
    donor_iter = tqdm(plan, total=len(plan), desc=f"loading {split_label} donors", unit="donor")
    for donor_idx, donor_plan in enumerate(donor_iter):
        donor_id = donor_plan["donor_id"]
        alloc_cells = int(donor_plan["alloc_cells"])
        donor_path = donor_dir / f"{donor_id}.h5ad"
        if not donor_path.exists():
            print(f"[warn] skip missing donor file: {donor_path}", flush=True)
            continue
        sampled_adata, donor_total, sampled_n = sample_single_donor_cells(
            donor_path=donor_path,
            sample_n=alloc_cells,
            sample_seed=int(cell_seed + donor_idx),
        )
        sampled_adata.obs = sampled_adata.obs.copy()
        sampled_adata.obs["_subset_donor_id"] = donor_id
        adatas.append(sampled_adata)
        donor_stats.append(
            {
                "donor_id": donor_id,
                "donor_cells_total": int(donor_total),
                "allocated_cells": int(alloc_cells),
                "sampled_cells": int(sampled_n),
                "sample_seed": int(cell_seed + donor_idx),
            }
        )

    if not adatas:
        raise RuntimeError(f"No donor data could be loaded for {split_label}.")

    print(f"[data] concatenating {split_label} donor subsets", flush=True)
    adata = (
        adatas[0].concatenate(
            *adatas[1:],
            join="inner",
            batch_key=f"_{split_label}_batch",
            index_unique=None,
        )
        if len(adatas) > 1
        else adatas[0]
    )
    print(f"[data] {split_label} subset ready: {adata.n_obs} cells × {adata.n_vars} genes", flush=True)
    return adata, donor_stats


def build_full_gene_inputs(adata, vocab: SimpleGeneVocab):
    if "feature_name" not in adata.var.columns:
        raise RuntimeError("SEA-AD donor h5ad has no feature_name for mapping to scGPT vocab.")

    donor_gene_names = adata.var["feature_name"].astype(str).to_numpy()
    in_vocab = np.array([g in vocab for g in donor_gene_names])
    kept_gene_names = donor_gene_names[in_vocab]
    missing_gene_names = donor_gene_names[~in_vocab].tolist()
    if kept_gene_names.size == 0:
        raise RuntimeError("No donor genes overlap with scGPT vocab.")

    gene_ids = np.array(vocab(kept_gene_names.tolist()), dtype=np.int64)
    x = adata.X[:, in_vocab]
    if not issparse(x):
        x = np.asarray(x, dtype=np.float32)
    elif x.dtype != np.float32:
        x = x.astype(np.float32)
    return x, gene_ids, kept_gene_names.tolist(), missing_gene_names


def bin_log1p_values(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    log_values = np.log1p(values.astype(np.float32))
    edges = np.quantile(log_values, np.linspace(0, 1, n_bins - 1)).astype(np.float32)
    digits = np.digitize(log_values, edges).astype(np.int64)
    return np.clip(digits, 0, n_bins - 1)


def tokenize_and_pad_matrix(
    matrix,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: SimpleGeneVocab,
    pad_token: str,
    pad_value: int,
    input_space: str,
    n_bins: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    trunc_seed: int = 1234,
    show_progress: bool = True,
    progress_desc: str = "tokenize",
):
    cls_id = vocab[cls_token]
    pad_id = vocab[pad_token]
    n_cells = matrix.shape[0]

    genes = []
    values = []
    row_iter = range(n_cells)
    if show_progress:
        row_iter = tqdm(row_iter, total=n_cells, desc=progress_desc, unit="cell")

    for i in row_iter:
        if issparse(matrix):
            row = matrix.getrow(i)
            if include_zero_gene:
                raw_vals = row.toarray().ravel().astype(np.float32)
                gids = gene_ids
            else:
                raw_vals = row.data.astype(np.float32, copy=False)
                gids = gene_ids[row.indices]
        else:
            row = np.asarray(matrix[i]).ravel().astype(np.float32)
            if include_zero_gene:
                raw_vals = row
                gids = gene_ids
            else:
                nz = np.nonzero(row)[0]
                raw_vals = row[nz]
                gids = gene_ids[nz]

        if input_space == "binned":
            proc_vals = bin_log1p_values(raw_vals, n_bins=n_bins)
            proc_tensor = torch.from_numpy(proc_vals).float()
        else:
            proc_vals = np.log1p(raw_vals.astype(np.float32))
            proc_tensor = torch.from_numpy(proc_vals).float()

        gid_tensor = torch.from_numpy(np.asarray(gids, dtype=np.int64)).long()

        if append_cls:
            gid_tensor = torch.cat([torch.tensor([cls_id], dtype=torch.long), gid_tensor])
            proc_tensor = torch.cat([torch.tensor([0.0], dtype=torch.float32), proc_tensor])

        if len(gid_tensor) > max_len:
            rng = np.random.RandomState(trunc_seed + i)
            if append_cls:
                idx = rng.choice(len(gid_tensor) - 1, max_len - 1, replace=False) + 1
                idx = np.insert(idx, 0, 0)
            else:
                idx = rng.choice(len(gid_tensor), max_len, replace=False)
            idx = np.sort(idx)
            gid_tensor = gid_tensor[idx]
            proc_tensor = proc_tensor[idx]

        if len(gid_tensor) < max_len:
            gid_tensor = torch.cat(
                [gid_tensor, torch.full((max_len - len(gid_tensor),), pad_id, dtype=gid_tensor.dtype)]
            )
            proc_tensor = torch.cat(
                [proc_tensor, torch.full((max_len - len(proc_tensor),), pad_value, dtype=proc_tensor.dtype)]
            )

        genes.append(gid_tensor)
        values.append(proc_tensor)

    return {"genes": torch.stack(genes, dim=0), "values": torch.stack(values, dim=0)}


def prepare_epoch_data(tokenized, batch_size, mask_ratio, zero_mask_ratio, mask_value, pad_value, mask_seed, shuffle):
    masked_values = random_mask_value(
        tokenized["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
        base_seed=mask_seed,
        keep_first_n_tokens=1,
        zero_mask_ratio=zero_mask_ratio,
    )
    data_pt = {
        "gene_ids": tokenized["genes"],
        "values": masked_values,
        "target_values": tokenized["values"],
        "batch_labels": torch.zeros(tokenized["genes"].shape[0], dtype=torch.long),
    }
    loader = prepare_dataloader(data_pt=data_pt, batch_size=batch_size, shuffle=shuffle)
    mask_stats = summarize_mask_stats(
        original_values=tokenized["values"],
        masked_values=masked_values,
        pad_value=pad_value,
        mask_value=mask_value,
        keep_first_n_tokens=1,
    )
    return data_pt, loader, mask_stats


def evaluate_epoch(model, loader, device, vocab, pad_token, mask_value, amp):
    model.eval()
    total_mse = 0.0
    total_mre = 0.0
    total_batches = 0
    with torch.no_grad():
        it = tqdm(loader, total=len(loader), desc="valid", unit="batch")
        for batch in it:
            input_gene_ids = batch["gene_ids"].to(device)
            input_values = batch["values"].to(device)
            target_values = batch["target_values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=False,
                    MVC=False,
                    ECS=False,
                )
                masked_positions = input_values.eq(mask_value)
                loss = masked_mse_loss(output_dict["mlm_output"], target_values, masked_positions)

            mre = masked_relative_error(output_dict["mlm_output"], target_values, masked_positions)
            total_mse += float(loss.item())
            total_mre += float(mre.item())
            total_batches += 1
            it.set_postfix(mse=f"{loss.item():.4f}", mre=f"{mre.item():.4f}")

    return {
        "valid_mse": total_mse / max(total_batches, 1),
        "valid_mre": total_mre / max(total_batches, 1),
    }


def train_epoch(model, loader, optimizer, scaler, device, vocab, pad_token, mask_value, amp, grad_clip):
    model.train()
    total_mse = 0.0
    total_mre = 0.0
    total_batches = 0

    it = tqdm(loader, total=len(loader), desc="train", unit="batch")
    for batch in it:
        input_gene_ids = batch["gene_ids"].to(device)
        input_values = batch["values"].to(device)
        target_values = batch["target_values"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=False,
                MVC=False,
                ECS=False,
            )
            masked_positions = input_values.eq(mask_value)
            loss = masked_mse_loss(output_dict["mlm_output"], target_values, masked_positions)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            mre = masked_relative_error(output_dict["mlm_output"], target_values, masked_positions)

        total_mse += float(loss.item())
        total_mre += float(mre.item())
        total_batches += 1
        it.set_postfix(mse=f"{loss.item():.4f}", mre=f"{mre.item():.4f}")

    return {
        "train_mse": total_mse / max(total_batches, 1),
        "train_mre": total_mre / max(total_batches, 1),
    }


def build_output_dir(args) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        run_name = args.run_name or (
            f"seaad_{args.variant}_{args.input_space}_"
            f"{'fullseq' if args.include_zero_gene else 'official'}_"
            f"train{args.train_total_cells}_"
            f"val{args.val_total_cells}_"
            f"s{args.seed}"
        )
        out_dir = Path(args.result_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_checkpoint_dir(
    out_dir: Path,
    model,
    pretrained_dir: Path,
    pretrained_cfg: dict,
    effective_max_seq_len: int,
    args,
    best_metrics: dict,
    summary: dict,
):
    shutil.copy2(pretrained_dir / "vocab.json", out_dir / "vocab.json")
    saved_args = dict(pretrained_cfg)
    saved_args["max_seq_len"] = int(effective_max_seq_len)
    saved_args["input_style"] = args.input_space
    saved_args["fast_transformer"] = bool(args.use_fast_transformer)
    saved_args["load_model"] = str(pretrained_dir)
    saved_args["pad_token"] = saved_args.get("pad_token", "<pad>")
    saved_args["pad_value"] = int(saved_args.get("pad_value", -2))
    saved_args["mask_value"] = int(saved_args.get("mask_value", -1))
    (out_dir / "args.json").write_text(json.dumps(saved_args, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "finetune_config.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained scGPT on SEA-AD full-gene reconstruction with controllable donor subsets.")
    parser.add_argument("--variant", type=str, default="brain", choices=["brain", "human"])
    parser.add_argument("--pretrained_dir", type=str, default="")
    parser.add_argument("--donor_csv", type=str, default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", type=str, default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_total_cells", type=int, default=8192)
    parser.add_argument("--val_total_cells", type=int, default=4096)
    parser.add_argument("--train_donor_seed", type=int, default=1234)
    parser.add_argument("--val_donor_seed", type=int, default=2345)
    parser.add_argument("--train_cell_seed", type=int, default=3456)
    parser.add_argument("--val_cell_seed", type=int, default=4567)
    parser.add_argument("--input_space", type=str, default="binned", choices=["binned", "log1p"])
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--include_zero_gene", action="store_true", help="Use all genes including zeros. This is much slower and more memory-intensive.")
    parser.add_argument("--max_seq_len", type=int, default=1201, help="Official-style default is 1201 (= 1200 genes + <cls>). Use <=0 with --include_zero_gene for full-length sequences.")
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="Mask ratio for nonzero gene tokens. If --zero_mask_ratio is unset, zero tokens follow this same ratio.")
    parser.add_argument("--zero_mask_ratio", type=float, default=None, help="Optional separate mask ratio for zero-value gene tokens. Default: use --mask_ratio.")
    parser.add_argument("--mask_seed", type=int, default=5678)
    parser.add_argument("--val_mask_seed", type=int, default=6789)
    parser.add_argument("--trunc_seed", type=int, default=7890)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp during fine-tuning.")
    parser.add_argument("--use_fast_transformer", action="store_true", help="Enable flash-attn backend if available.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    donor_dir = Path(args.donor_dir)
    donor_csv = Path(args.donor_csv)
    pretrained_dir = Path(args.pretrained_dir) if args.pretrained_dir else PRETRAINED_DIRS[args.variant]
    out_dir = build_output_dir(args)
    donor_counts = load_donor_cell_count_map(donor_csv)

    print(
        f"[setup] variant={args.variant} input_space={args.input_space} "
        f"include_zero_gene={args.include_zero_gene} epochs={args.epochs}",
        flush=True,
    )
    print(f"[setup] pretrained_dir={pretrained_dir}", flush=True)
    print(f"[setup] output_dir={out_dir}", flush=True)

    train_donors, val_donors, split_info = split_donors_train_val(
        str(donor_csv),
        val_ratio=0.1,
        seed=args.split_seed,
    )
    print(
        f"[split] train donors={len(train_donors)} val donors={len(val_donors)} "
        f"val_ratio_actual={split_info['val_ratio_actual']:.2%}",
        flush=True,
    )

    train_adata, train_donor_stats = build_subset_from_donors(
        donor_ids=train_donors,
        donor_counts=donor_counts,
        donor_dir=donor_dir,
        total_cells=args.train_total_cells,
        donor_seed=args.train_donor_seed,
        cell_seed=args.train_cell_seed,
        split_label="train",
    )
    val_adata, val_donor_stats = build_subset_from_donors(
        donor_ids=val_donors,
        donor_counts=donor_counts,
        donor_dir=donor_dir,
        total_cells=args.val_total_cells,
        donor_seed=args.val_donor_seed,
        cell_seed=args.val_cell_seed,
        split_label="val",
    )

    vocab = SimpleGeneVocab.from_file(pretrained_dir / "vocab.json")
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])

    with (pretrained_dir / "args.json").open() as f:
        pretrained_cfg = json.load(f)

    train_x, gene_ids, kept_gene_names, missing_gene_names = build_full_gene_inputs(train_adata, vocab)
    val_x, val_gene_ids, _, _ = build_full_gene_inputs(val_adata, vocab)
    if len(gene_ids) != len(val_gene_ids) or not np.array_equal(gene_ids, val_gene_ids):
        raise RuntimeError("Train/val gene_id mapping mismatch after donor subset alignment.")

    effective_max_seq_len = int(args.max_seq_len)
    if effective_max_seq_len <= 0:
        effective_max_seq_len = int(len(gene_ids) + 1)

    print(
        f"[data] gene overlap with vocab: {len(kept_gene_names)}/{len(kept_gene_names) + len(missing_gene_names)}",
        flush=True,
    )
    print(
        f"[data] train cells={train_x.shape[0]} val cells={val_x.shape[0]} "
        f"genes_used={len(gene_ids)} effective_max_seq_len={effective_max_seq_len}",
        flush=True,
    )

    print("[prep] tokenizing train subset", flush=True)
    tokenized_train = tokenize_and_pad_matrix(
        matrix=train_x,
        gene_ids=gene_ids,
        max_len=effective_max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=int(pretrained_cfg.get("pad_value", -2)),
        input_space=args.input_space,
        n_bins=int(args.n_bins if args.n_bins > 0 else pretrained_cfg.get("n_bins", 51)),
        append_cls=True,
        include_zero_gene=args.include_zero_gene,
        trunc_seed=args.trunc_seed,
        show_progress=True,
        progress_desc="tokenizing train",
    )

    print("[prep] tokenizing val subset", flush=True)
    tokenized_val = tokenize_and_pad_matrix(
        matrix=val_x,
        gene_ids=gene_ids,
        max_len=effective_max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=int(pretrained_cfg.get("pad_value", -2)),
        input_space=args.input_space,
        n_bins=int(args.n_bins if args.n_bins > 0 else pretrained_cfg.get("n_bins", 51)),
        append_cls=True,
        include_zero_gene=args.include_zero_gene,
        trunc_seed=args.trunc_seed + 1000000,
        show_progress=True,
        progress_desc="tokenizing val",
    )

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=pretrained_cfg["embsize"],
        nhead=pretrained_cfg["nheads"],
        d_hid=pretrained_cfg["d_hid"],
        nlayers=pretrained_cfg["nlayers"],
        nlayers_cls=pretrained_cfg.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=pretrained_cfg["dropout"],
        pad_token=pretrained_cfg["pad_token"],
        pad_value=int(pretrained_cfg.get("pad_value", -2)),
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=pretrained_cfg.get("input_emb_style", "continuous"),
        n_input_bins=int(args.n_bins if args.n_bins > 0 else pretrained_cfg.get("n_bins", 51)),
        explicit_zero_prob=False,
        use_fast_transformer=bool(args.use_fast_transformer),
        pre_norm=bool(pretrained_cfg.get("pre_norm", False)),
    )
    pretrained_state = torch.load(pretrained_dir / "best_model.pt", map_location="cpu")
    load_pretrained(model, pretrained_state, verbose=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and torch.cuda.is_available())
    model.to(device)
    print(f"[setup] device={device} amp={amp}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-4 if amp else 1e-8,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    history = []
    history_path = out_dir / "history.jsonl"
    pad_value = int(pretrained_cfg.get("pad_value", -2))
    mask_value = int(pretrained_cfg.get("mask_value", -1))
    best_val_mse = float("inf")
    best_metrics = {}

    fixed_val_data, fixed_val_loader, val_mask_stats = prepare_epoch_data(
        tokenized=tokenized_val,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        zero_mask_ratio=args.zero_mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
        mask_seed=args.val_mask_seed,
        shuffle=False,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch] {epoch}/{args.epochs}", flush=True)
        epoch_data, train_loader, train_mask_stats = prepare_epoch_data(
            tokenized=tokenized_train,
            batch_size=args.batch_size,
            mask_ratio=args.mask_ratio,
            zero_mask_ratio=args.zero_mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
            mask_seed=args.mask_seed + epoch * 100000,
            shuffle=True,
        )

        t0 = time.time()
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            vocab=vocab,
            pad_token="<pad>",
            mask_value=mask_value,
            amp=amp,
            grad_clip=args.grad_clip,
        )
        valid_metrics = evaluate_epoch(
            model=model,
            loader=fixed_val_loader,
            device=device,
            vocab=vocab,
            pad_token="<pad>",
            mask_value=mask_value,
            amp=amp,
        )
        epoch_metrics = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 2),
            "train_mask_ratio_actual_nonpad": train_mask_stats["mask_ratio_actual_nonpad"],
            "train_nonzero_mask_ratio_actual_nonpad": train_mask_stats["nonzero_mask_ratio_actual_nonpad"],
            "train_zero_mask_ratio_actual_nonpad": train_mask_stats["zero_mask_ratio_actual_nonpad"],
            "val_mask_ratio_actual_nonpad": val_mask_stats["mask_ratio_actual_nonpad"],
            "val_nonzero_mask_ratio_actual_nonpad": val_mask_stats["nonzero_mask_ratio_actual_nonpad"],
            "val_zero_mask_ratio_actual_nonpad": val_mask_stats["zero_mask_ratio_actual_nonpad"],
            **train_metrics,
            **valid_metrics,
        }
        history.append(epoch_metrics)
        with history_path.open("a") as f:
            f.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")

        print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2), flush=True)

        if valid_metrics["valid_mse"] < best_val_mse:
            best_val_mse = valid_metrics["valid_mse"]
            best_metrics = dict(epoch_metrics)
            summary = {
                "variant": args.variant,
                "pretrained_dir": str(pretrained_dir),
                "dataset": "SEA-AD MTG donor_h5ad full gene space",
                "split_rule": f"scUniFlow donor split val_ratio=0.1 seed={args.split_seed}",
                "include_zero_gene": bool(args.include_zero_gene),
                "effective_max_seq_len": int(effective_max_seq_len),
                "input_space": args.input_space,
                "n_bins": int(args.n_bins if args.input_space == "binned" else 0),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "mask_ratio_target": float(args.mask_ratio),
                "nonzero_mask_ratio_target": float(args.mask_ratio),
                "zero_mask_ratio_target": float(args.zero_mask_ratio if args.zero_mask_ratio is not None else args.mask_ratio),
                "train_cells": int(train_x.shape[0]),
                "val_cells": int(val_x.shape[0]),
                "train_total_cells_target": int(args.train_total_cells),
                "val_total_cells_target": int(args.val_total_cells),
                "genes_in_vocab": int(len(kept_gene_names)),
                "genes_missing_from_vocab": int(len(missing_gene_names)),
                "missing_gene_names_preview": missing_gene_names[:50],
                "train_donor_sampling": train_donor_stats,
                "val_donor_sampling": val_donor_stats,
                "best_metrics": best_metrics,
            }
            save_checkpoint_dir(
                out_dir=out_dir,
                model=model,
                pretrained_dir=pretrained_dir,
                pretrained_cfg=pretrained_cfg,
                effective_max_seq_len=effective_max_seq_len,
                args=args,
                best_metrics=best_metrics,
                summary=summary,
            )
            print(f"[saved] best checkpoint updated at {out_dir}", flush=True)

    final_summary = {
        "output_dir": str(out_dir),
        "best_model_pt": str(out_dir / "best_model.pt"),
        "history_jsonl": str(history_path),
        "best_metrics": best_metrics,
        "train_cells": int(train_x.shape[0]),
        "val_cells": int(val_x.shape[0]),
        "genes_in_vocab": int(len(kept_gene_names)),
        "effective_max_seq_len": int(effective_max_seq_len),
        "input_space": args.input_space,
        "include_zero_gene": bool(args.include_zero_gene),
    }
    (out_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
