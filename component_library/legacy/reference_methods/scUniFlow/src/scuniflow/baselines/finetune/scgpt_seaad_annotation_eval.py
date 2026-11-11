import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

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
    prepare_dataloader,
    summarize_mask_stats,
)


DEFAULT_RESULT_DIR = Path("results/pretrain_eval")
DEFAULT_LABEL_KEYS = ("Class", "Subclass", "Supertype")
DEFAULT_TASK_NAMES = ("class", "subclass", "supertype")


class AnnotationHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def official_tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    show_progress: bool = False,
    progress_desc: str = "tokenize: scan",
):
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match number of gene_ids ({len(gene_ids)})."
        )

    tokenized_data = []
    row_iter = range(len(data))
    if show_progress:
        row_iter = tqdm(row_iter, total=len(data), desc=progress_desc, unit="cell")
    for i in row_iter:
        row = data[i]
        if include_zero_gene:
            values = row
            genes = gene_ids
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
        tokenized_data.append((genes, values))
    return tokenized_data


def official_pad_batch(
    batch,
    max_len: int,
    vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    show_progress: bool = False,
    progress_desc: str = "tokenize: pad",
):
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    gene_ids_list = []
    values_list = []

    batch_iter = range(len(batch))
    if show_progress:
        batch_iter = tqdm(batch_iter, total=len(batch), desc=progress_desc, unit="cell")
    for i in batch_iter:
        gene_ids, values = batch[i]

        if len(gene_ids) > max_len:
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[idx]
            values = values[idx]
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full((max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )

        gene_ids_list.append(gene_ids)
        values_list.append(values)

    return {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }


def official_tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    show_progress: bool = False,
    progress_desc: str = "tokenizing",
):
    cls_id = vocab[cls_token]
    tokenized_data = official_tokenize_batch(
        data,
        gene_ids,
        return_pt=True,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        show_progress=show_progress,
        progress_desc=f"{progress_desc}: scan",
    )
    return official_pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        show_progress=show_progress,
        progress_desc=f"{progress_desc}: pad",
    )


def official_random_mask_value(
    values,
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
):
    if isinstance(values, torch.Tensor):
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        if n_mask <= 0:
            continue
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()


def parse_csv_list(text: str):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_result_path(args, donor_id: str, sampled_cells: int, per_donor_n: int) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    eval_space_tag = args.eval_space if args.eval_space != "auto" else "auto"
    seq_tag = "fullseq" if args.include_zero_gene else "nonzero"
    if args.eval_subset_mode == "balanced_val_donors":
        filename = (
            f"scgpt_annotation_{Path(args.ckpt_dir).name}_{seq_tag}_{eval_space_tag}_"
            f"seed{args.split_seed}_{donor_id}_perdonor{per_donor_n}_total{sampled_cells}_"
            f"mask{args.mask_ratio:.2f}.json"
        )
    else:
        filename = (
            f"scgpt_annotation_{Path(args.ckpt_dir).name}_{seq_tag}_{eval_space_tag}_"
            f"seed{args.split_seed}_{donor_id}_n{sampled_cells}_mask{args.mask_ratio:.2f}.json"
        )
    return result_dir / filename.replace("/", "_")


def resolve_eval_space(args, saved_args: dict):
    if args.eval_space != "auto":
        return args.eval_space
    style = str(saved_args.get("input_style", "log1p")).lower()
    if style in ("binned", "log1p"):
        return style
    return "log1p"


def load_label_spaces(ckpt_dir: Path):
    label_path = ckpt_dir / "label_spaces.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Annotation label space file not found: {label_path}")
    with label_path.open() as f:
        label_spaces = json.load(f)
    if not isinstance(label_spaces, dict) or not label_spaces:
        raise RuntimeError(f"Invalid or empty label space json: {label_path}")
    return label_spaces


def encode_annotation_labels(obs, label_keys, task_names, label_spaces):
    encoded = {}
    valid_mask = np.ones(obs.shape[0], dtype=bool)
    dropped = {}

    for obs_key, task_name in zip(label_keys, task_names):
        if obs_key not in obs.columns:
            raise KeyError(f"Annotation column not found in adata.obs: {obs_key}")
        mapping = {name: idx for idx, name in enumerate(label_spaces[task_name])}
        values = obs[obs_key].astype(str).to_numpy()
        ids = np.array([mapping.get(v, -1) for v in values], dtype=np.int64)
        is_valid = ids >= 0
        if not np.all(is_valid):
            dropped[task_name] = sorted(set(values[~is_valid].tolist()))
        valid_mask &= is_valid
        encoded[task_name] = ids

    for task_name in encoded:
        encoded[task_name] = encoded[task_name][valid_mask]
    return encoded, valid_mask, dropped


def compute_official_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_name: str):
    return {
        f"{task_name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{task_name}_precision": float(precision_score(y_true, y_pred, average="macro")),
        f"{task_name}_recall": float(recall_score(y_true, y_pred, average="macro")),
        f"{task_name}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{task_name}_micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }


def build_full_gene_matrix(adata, vocab: SimpleGeneVocab):
    if "feature_name" not in adata.var.columns:
        raise RuntimeError("SEA-AD donor h5ad has no feature_name for mapping to scGPT vocab.")

    gene_names = adata.var["feature_name"].astype(str).to_numpy()
    in_vocab = np.array([g in vocab for g in gene_names])
    kept_gene_names = gene_names[in_vocab]
    gene_ids = np.array(vocab(kept_gene_names.tolist()), dtype=np.int64)

    x = adata.X[:, in_vocab]
    if issparse(x):
        x = x.toarray()
    x = x.astype(np.float32)
    return x, gene_ids, in_vocab, kept_gene_names.tolist(), gene_names[~in_vocab].tolist()


def bin_expression_matrix(
    x_log: np.ndarray,
    n_bins: int,
    show_progress: bool = True,
    desc: str = "binning cells",
):
    x_eval = np.zeros_like(x_log, dtype=np.int64)
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
        x_eval[i] = np.clip(binned_row, 0, n_bins - 1)
    return x_eval


def load_annotation_model(ckpt_dir: Path):
    vocab = SimpleGeneVocab.from_file(ckpt_dir / "vocab.json")
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])

    with (ckpt_dir / "args.json").open() as f:
        saved_args = json.load(f)

    label_spaces = load_label_spaces(ckpt_dir)
    task_names = tuple(saved_args.get("annotation_task_names", list(label_spaces.keys())))
    label_keys = tuple(saved_args.get("annotation_label_keys", DEFAULT_LABEL_KEYS[: len(task_names)]))
    if len(task_names) != len(label_keys):
        raise RuntimeError("Checkpoint annotation_task_names and annotation_label_keys length mismatch.")

    backbone = TransformerModel(
        ntoken=len(vocab),
        d_model=saved_args["embsize"],
        nhead=saved_args["nheads"],
        d_hid=saved_args["d_hid"],
        nlayers=saved_args["nlayers"],
        nlayers_cls=saved_args.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=saved_args["dropout"],
        pad_token=saved_args["pad_token"],
        pad_value=int(saved_args.get("pad_value", -2)),
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=saved_args.get("input_emb_style", "continuous"),
        n_input_bins=int(saved_args.get("n_bins", 51)),
        explicit_zero_prob=False,
        use_fast_transformer=bool(saved_args.get("fast_transformer", False)),
        pre_norm=bool(saved_args.get("pre_norm", False)),
    )
    backbone_state = torch.load(ckpt_dir / "best_model.pt", map_location="cpu")
    load_pretrained(backbone, backbone_state, verbose=False)

    annotation_heads = nn.ModuleDict()
    for task_name in task_names:
        annotation_heads[task_name] = AnnotationHead(
            d_model=saved_args["embsize"],
            n_classes=len(label_spaces[task_name]),
            dropout=0.2,
        )
    head_state = torch.load(ckpt_dir / "annotation_heads.pt", map_location="cpu")
    missing, unexpected = annotation_heads.load_state_dict(head_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to cleanly load annotation heads.\nmissing={missing}\nunexpected={unexpected}"
        )

    return backbone, annotation_heads, vocab, saved_args, label_keys, task_names, label_spaces


def predict_annotation(
    backbone,
    annotation_heads,
    loader,
    vocab,
    pad_token,
    device,
    amp,
    show_progress: bool = True,
    progress_desc: str = "annotation eval",
):
    backbone.eval()
    annotation_heads.eval()

    predictions = {task_name: [] for task_name in annotation_heads.keys()}
    with torch.no_grad():
        batch_iter = loader
        if show_progress:
            batch_iter = tqdm(loader, total=len(loader), desc=progress_desc, unit="batch")
        for batch_data in batch_iter:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = backbone(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if getattr(backbone, "use_batch_labels", False) else None,
                    CLS=False,
                    MVC=False,
                    ECS=False,
                )
            cell_emb = output_dict["cell_emb"]
            for task_name, head in annotation_heads.items():
                logits = head(cell_emb)
                predictions[task_name].append(logits.argmax(1).cpu().numpy())

    for task_name in predictions:
        predictions[task_name] = np.concatenate(predictions[task_name], axis=0)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Fine-tuned joint scGPT checkpoint directory.")
    parser.add_argument("--max_seq_len", type=int, default=0, help="<=0 means use checkpoint args max_seq_len.")
    parser.add_argument("--eval_space", type=str, default="auto", choices=["auto", "log1p", "binned"])
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="Official scGPT test() mask ratio.")
    parser.add_argument("--zero_mask_ratio", type=float, default=None, help="Ignored in official-compatible annotation eval.")
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
    )
    parser.add_argument("--include_zero_gene", action="store_true")
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--annotation_label_keys", type=str, default="")
    parser.add_argument("--annotation_task_names", type=str, default="")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    donor_csv = Path(args.donor_csv)
    donor_dir = Path(args.donor_dir)

    print(
        f"[setup] ckpt_dir={ckpt_dir} subset_mode={args.eval_subset_mode} include_zero_gene={args.include_zero_gene}",
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
        donor_stats = [{
            "donor_id": donor_id,
            "donor_cells_total": int(donor_cells),
            "sampled_cells": int(n),
            "sample_seed": int(args.sample_seed),
        }]
        per_donor_n = int(n)

    print(f"[model] loading annotation checkpoint from {ckpt_dir}", flush=True)
    backbone, annotation_heads, vocab, saved_args, label_keys, task_names, label_spaces = load_annotation_model(ckpt_dir)

    if args.annotation_label_keys:
        label_keys = tuple(parse_csv_list(args.annotation_label_keys))
    if args.annotation_task_names:
        task_names = tuple(parse_csv_list(args.annotation_task_names))
    if len(label_keys) != len(task_names):
        raise ValueError("annotation_label_keys and annotation_task_names must have the same length.")

    eval_space = resolve_eval_space(args, saved_args)
    amp = bool(saved_args.get("amp", False) and torch.cuda.is_available())
    n_bins = int(args.n_bins if args.n_bins > 0 else saved_args.get("n_bins", 51))
    max_seq_len = int(args.max_seq_len if args.max_seq_len > 0 else saved_args.get("max_seq_len", 1201))
    pad_value = int(saved_args.get("pad_value", -2))
    mask_value = int(saved_args.get("mask_value", -1))

    if args.zero_mask_ratio is not None:
        print("[warn] zero_mask_ratio is ignored to stay aligned with official scGPT annotation test().", flush=True)

    labels_all, valid_mask, dropped_labels = encode_annotation_labels(
        adata.obs,
        label_keys=label_keys,
        task_names=task_names,
        label_spaces=label_spaces,
    )
    if not np.all(valid_mask):
        adata = adata[valid_mask].copy()
    if adata.n_obs == 0:
        raise RuntimeError("No cells remain after applying checkpoint label space filtering.")

    print(
        f"[annotation] eval cells after label filtering={adata.n_obs} dropped={n - adata.n_obs}",
        flush=True,
    )

    x, gene_ids, in_vocab, kept_gene_names, missing_gene_names = build_full_gene_matrix(adata, vocab)
    x_log = np.log1p(x.astype(np.float32))
    if eval_space == "binned":
        print(f"[prep] binning expression matrix into {n_bins} bins", flush=True)
        x_eval = bin_expression_matrix(x_log, n_bins=n_bins, show_progress=True, desc="binning cells")
    else:
        x_eval = x_log.astype(np.float32)

    print("[prep] tokenizing evaluation matrix", flush=True)
    np.random.seed(args.mask_seed)
    tokenized = official_tokenize_and_pad_batch(
        x_eval,
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

    print("[prep] applying official scGPT random masks", flush=True)
    np.random.seed(args.mask_seed)
    masked_values = official_random_mask_value(
        tokenized["values"],
        mask_ratio=args.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    mask_stats = summarize_mask_stats(
        original_values=tokenized["values"],
        masked_values=masked_values,
        pad_value=pad_value,
        mask_value=mask_value,
        keep_first_n_tokens=0,
    )

    data_pt = {
        "gene_ids": tokenized["genes"],
        "values": masked_values,
        "target_values": tokenized["values"],
        "batch_labels": torch.zeros(tokenized["genes"].shape[0], dtype=torch.long),
    }
    for task_name, values in labels_all.items():
        data_pt[f"labels_{task_name}"] = torch.from_numpy(values).long()
    loader = prepare_dataloader(data_pt=data_pt, batch_size=args.batch_size)

    print(
        f"[eval] donor={donor_id} sampled_cells={adata.n_obs} batch_size={args.batch_size} "
        f"total_batches={len(loader)} mask_ratio={args.mask_ratio} "
        f"seq_len={max_seq_len} include_zero_gene={args.include_zero_gene} eval_space={eval_space} amp={amp}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    annotation_heads.to(device)

    print(f"[eval] starting annotation forward passes on device={device}", flush=True)
    predictions = predict_annotation(
        backbone=backbone,
        annotation_heads=annotation_heads,
        loader=loader,
        vocab=vocab,
        pad_token="<pad>",
        device=device,
        amp=amp,
        show_progress=True,
        progress_desc="annotation eval",
    )

    metrics = {}
    for task_name in task_names:
        metrics.update(compute_official_metrics(labels_all[task_name], predictions[task_name], task_name))

    out = {
        "checkpoint": str(ckpt_dir),
        "dataset": "SEA-AD MTG donor_h5ad",
        "donor_id": donor_id,
        "donor_cells_total": int(donor_cells),
        "eval_subset_mode": args.eval_subset_mode,
        "balanced_per_donor_cells": int(per_donor_n),
        "donor_sampling": donor_stats,
        "sampled_cells_before_label_filter": int(n),
        "sampled_cells": int(adata.n_obs),
        "split_rule": "scUniFlow donor split val_ratio=0.1 seed=42",
        "unseen_for_scuniflow_train": True,
        "matched_genes_in_vocab": int(in_vocab.sum()),
        "all_genes": int(len(in_vocab)),
        "max_seq_len": int(max_seq_len),
        "include_zero_gene": bool(args.include_zero_gene),
        "eval_space": eval_space,
        "n_bins": int(n_bins if eval_space == "binned" else 0),
        "mask_ratio_target": float(args.mask_ratio),
        "official_scgpt_masking": True,
        "batch_size": int(args.batch_size),
        "annotation_label_keys": list(label_keys),
        "annotation_task_names": list(task_names),
        "label_space_sizes": {k: int(len(v)) for k, v in label_spaces.items()},
        "dropped_labels": dropped_labels,
        "genes_missing_from_vocab": int(len(missing_gene_names)),
        "missing_gene_names_preview": missing_gene_names[:50],
        **mask_stats,
        **metrics,
    }

    result_path = build_result_path(args, donor_id, sampled_cells=adata.n_obs, per_donor_n=per_donor_n)
    out["result_json"] = str(result_path)
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    print(f"[done] wrote result json to {result_path}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
