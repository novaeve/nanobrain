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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

import sys

THIS_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = THIS_DIR.parent
EVAL_DIR = FINETUNE_DIR.parent
REPO_ROOT = EVAL_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, "external/scGPT")

from data import split_donors_train_val
from scgpt.model.model import TransformerModel
from scgpt.utils.util import load_pretrained
from scgpt.loss import masked_mse_loss, masked_relative_error
from donor_eval_utils import DEFAULT_DONOR_CSV, DEFAULT_DONOR_DIR, sample_single_donor_cells
from scgpt_eval_utils import SimpleGeneVocab, prepare_dataloader, random_mask_value, summarize_mask_stats

# Reuse full-gene donor subset and tokenization helpers.
from scgpt_fullgene_seaad_finetune import (
    load_donor_cell_count_map,
    allocate_cells_across_donors,
    build_subset_from_donors,
    build_full_gene_inputs,
    bin_log1p_values,
    tokenize_and_pad_matrix,
)


DEFAULT_RESULT_ROOT = Path("./src/scuniflow/baselines/finetune/outputs")

PRETRAINED_DIRS = {
    "brain": Path("external/scGPT/checkpoint/brain"),
    "human": Path("external/scGPT/checkpoint/human"),
}

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


def parse_csv_list(text: str):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def get_series_categories(series):
    if hasattr(series.dtype, "categories"):
        return [str(x) for x in series.dtype.categories.tolist()]
    return sorted(str(x) for x in pd_unique(series))


def pd_unique(series):
    # Avoid importing pandas just for unique extraction.
    arr = np.asarray(series.astype(str))
    return np.unique(arr)


def build_label_spaces(adata, label_keys, task_names):
    label_spaces = {}
    for obs_key, task_name in zip(label_keys, task_names):
        if obs_key not in adata.obs.columns:
            raise KeyError(f"Annotation column not found in adata.obs: {obs_key}")
        label_spaces[task_name] = get_series_categories(adata.obs[obs_key].astype("category"))
    return label_spaces


def encode_annotation_labels(adata, label_keys, task_names, label_spaces):
    encoded = {}
    valid_mask = np.ones(adata.n_obs, dtype=bool)
    dropped = {}

    for obs_key, task_name in zip(label_keys, task_names):
        space = label_spaces[task_name]
        mapping = {name: idx for idx, name in enumerate(space)}
        values = adata.obs[obs_key].astype(str).to_numpy()
        ids = np.array([mapping.get(v, -1) for v in values], dtype=np.int64)
        is_valid = ids >= 0
        if not np.all(is_valid):
            dropped[task_name] = sorted(set(values[~is_valid].tolist()))
        valid_mask &= is_valid
        encoded[task_name] = ids

    for task_name in encoded:
        encoded[task_name] = encoded[task_name][valid_mask]
    return encoded, valid_mask, dropped


def filter_adata_and_labels(adata, labels, valid_mask):
    if valid_mask is None or np.all(valid_mask):
        return adata, labels
    adata = adata[valid_mask].copy()
    return adata, labels


def prepare_epoch_data_joint(
    tokenized,
    batch_size,
    recon_enabled,
    mask_ratio,
    zero_mask_ratio,
    mask_value,
    pad_value,
    mask_seed,
    shuffle,
    annotation_labels=None,
):
    if recon_enabled:
        input_values = random_mask_value(
            tokenized["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
            base_seed=mask_seed,
            keep_first_n_tokens=1,
            zero_mask_ratio=zero_mask_ratio,
        )
        mask_stats = summarize_mask_stats(
            original_values=tokenized["values"],
            masked_values=input_values,
            pad_value=pad_value,
            mask_value=mask_value,
            keep_first_n_tokens=1,
        )
    else:
        input_values = tokenized["values"].clone()
        mask_stats = {
            "mask_ratio_actual_nonpad": None,
            "zero_mask_ratio_actual_nonpad": None,
            "nonzero_mask_ratio_actual_nonpad": None,
            "n_mask_candidates_nonpad": 0,
            "n_zero_mask_candidates_nonpad": 0,
            "n_nonzero_mask_candidates_nonpad": 0,
            "n_masked_nonpad": 0,
            "n_zero_masked_nonpad": 0,
            "n_nonzero_masked_nonpad": 0,
        }

    data_pt = {
        "gene_ids": tokenized["genes"],
        "values": input_values,
        "target_values": tokenized["values"],
        "batch_labels": torch.zeros(tokenized["genes"].shape[0], dtype=torch.long),
    }
    if annotation_labels:
        for task_name, values in annotation_labels.items():
            data_pt[f"labels_{task_name}"] = torch.from_numpy(values).long()

    loader = prepare_dataloader(data_pt=data_pt, batch_size=batch_size, shuffle=shuffle)
    return data_pt, loader, mask_stats


def safe_auc(y_true, prob):
    present_classes = np.unique(y_true)
    if present_classes.size < 2:
        return {"macro": None, "micro": None, "n_present_classes": int(present_classes.size)}

    prob_present = prob[:, present_classes]
    row_sum = prob_present.sum(axis=1, keepdims=True)
    prob_present = np.divide(prob_present, np.clip(row_sum, 1e-12, None))
    y_bin = label_binarize(y_true, classes=present_classes)

    if y_bin.ndim == 1:
        y_bin = np.stack([1 - y_bin, y_bin], axis=1)

    try:
        macro = float(roc_auc_score(y_bin, prob_present, average="macro", multi_class="ovr"))
    except ValueError:
        macro = None

    try:
        micro = float(roc_auc_score(y_bin, prob_present, average="micro", multi_class="ovr"))
    except ValueError:
        micro = None

    return {"macro": macro, "micro": micro, "n_present_classes": int(present_classes.size)}


def summarize_task_metrics(task_name, y_true, logits):
    y_pred = logits.argmax(axis=1)
    y_prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    auc = safe_auc(y_true, y_prob)

    return {
        f"{task_name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{task_name}_macro_precision": float(macro_p),
        f"{task_name}_macro_recall": float(macro_r),
        f"{task_name}_macro_f1": float(macro_f1),
        f"{task_name}_micro_precision": float(micro_p),
        f"{task_name}_micro_recall": float(micro_r),
        f"{task_name}_micro_f1": float(micro_f1),
        f"{task_name}_weighted_precision": float(weighted_p),
        f"{task_name}_weighted_recall": float(weighted_r),
        f"{task_name}_weighted_f1": float(weighted_f1),
        f"{task_name}_auc_macro_present": auc["macro"],
        f"{task_name}_auc_micro_present": auc["micro"],
        f"{task_name}_n_present_classes": auc["n_present_classes"],
        f"{task_name}_n_pred_classes": int(np.unique(y_pred).size),
    }


def aggregate_annotation_metrics(task_names, logits_all, labels_all):
    if not task_names:
        return {}

    metrics = {}
    macro_f1s = []
    accuracies = []
    for task_name in task_names:
        task_metrics = summarize_task_metrics(task_name, labels_all[task_name], logits_all[task_name])
        metrics.update(task_metrics)
        macro_f1s.append(task_metrics[f"{task_name}_macro_f1"])
        accuracies.append(task_metrics[f"{task_name}_accuracy"])

    metrics["annotation_mean_macro_f1"] = float(np.mean(macro_f1s)) if macro_f1s else None
    metrics["annotation_mean_accuracy"] = float(np.mean(accuracies)) if accuracies else None

    if len(task_names) > 1:
        exact = np.ones_like(labels_all[task_names[0]], dtype=bool)
        for task_name in task_names:
            exact &= logits_all[task_name].argmax(axis=1) == labels_all[task_name]
        metrics["annotation_joint_exact_match"] = float(np.mean(exact))
    return metrics


def extract_backbone_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    if "backbone_state_dict" in checkpoint_obj:
        return checkpoint_obj["backbone_state_dict"]
    return checkpoint_obj


def run_model(
    backbone,
    annotation_heads,
    batch,
    device,
    use_recon,
    use_annotation,
    task_names,
    vocab,
    pad_token,
    mask_value,
    amp,
):
    input_gene_ids = batch["gene_ids"].to(device)
    input_values = batch["values"].to(device)
    target_values = batch["target_values"].to(device)
    src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

    with torch.cuda.amp.autocast(enabled=amp):
        output_dict = backbone(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            CLS=False,
            MVC=False,
            ECS=False,
        )

        loss_recon = None
        mre = None
        masked_positions = None
        if use_recon:
            masked_positions = input_values.eq(mask_value)
            loss_recon = masked_mse_loss(output_dict["mlm_output"], target_values, masked_positions)
            mre = masked_relative_error(output_dict["mlm_output"], target_values, masked_positions)

        logits = {}
        losses_annotation = {}
        if use_annotation:
            cell_emb = output_dict["cell_emb"]
            for task_name in task_names:
                task_logits = annotation_heads[task_name](cell_emb)
                task_labels = batch[f"labels_{task_name}"].to(device)
                logits[task_name] = task_logits
                losses_annotation[task_name] = F.cross_entropy(task_logits, task_labels)

    return {
        "output_dict": output_dict,
        "loss_recon": loss_recon,
        "mre": mre,
        "logits": logits,
        "losses_annotation": losses_annotation,
        "masked_positions": masked_positions,
    }


def train_epoch(
    backbone,
    annotation_heads,
    loader,
    optimizer,
    scaler,
    device,
    vocab,
    pad_token,
    mask_value,
    amp,
    grad_clip,
    use_recon,
    use_annotation,
    task_names,
    recon_loss_weight,
    annotation_loss_weight,
):
    backbone.train()
    annotation_heads.train()

    total_loss = 0.0
    total_recon = 0.0
    total_mre = 0.0
    total_batches = 0
    total_samples = 0
    total_ann = {task_name: 0.0 for task_name in task_names}
    total_ann_acc = {task_name: 0.0 for task_name in task_names}

    it = tqdm(loader, total=len(loader), desc="train", unit="batch")
    for batch in it:
        batch_size = batch["gene_ids"].shape[0]
        total_samples += int(batch_size)
        optimizer.zero_grad(set_to_none=True)

        model_out = run_model(
            backbone=backbone,
            annotation_heads=annotation_heads,
            batch=batch,
            device=device,
            use_recon=use_recon,
            use_annotation=use_annotation,
            task_names=task_names,
            vocab=vocab,
            pad_token=pad_token,
            mask_value=mask_value,
            amp=amp,
        )

        loss = 0.0
        postfix = {}

        if use_recon:
            loss = loss + recon_loss_weight * model_out["loss_recon"]
            total_recon += float(model_out["loss_recon"].item())
            total_mre += float(model_out["mre"].item())
            postfix["mse"] = f"{model_out['loss_recon'].item():.4f}"

        if use_annotation:
            ann_loss = sum(model_out["losses_annotation"].values()) / max(len(task_names), 1)
            loss = loss + annotation_loss_weight * ann_loss
            postfix["cls"] = f"{ann_loss.item():.4f}"
            for task_name in task_names:
                logits = model_out["logits"][task_name]
                labels = batch[f"labels_{task_name}"].to(device)
                total_ann[task_name] += float(
                    F.cross_entropy(logits, labels, reduction="sum").item()
                )
                total_ann_acc[task_name] += float(
                    (logits.argmax(dim=1) == labels).sum().item()
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            params = list(backbone.parameters()) + list(annotation_heads.parameters())
            torch.nn.utils.clip_grad_norm_(params, grad_clip, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_batches += 1
        postfix["loss"] = f"{loss.item():.4f}"
        it.set_postfix(**postfix)

    metrics = {
        "train_loss": total_loss / max(total_batches, 1),
    }
    if use_recon:
        metrics.update(
            {
                "train_mse": total_recon / max(total_batches, 1),
                "train_mre": total_mre / max(total_batches, 1),
            }
        )
    if use_annotation:
        ann_losses = []
        ann_accs = []
        for task_name in task_names:
            metrics[f"train_loss_{task_name}"] = total_ann[task_name] / max(total_samples, 1)
            metrics[f"train_acc_{task_name}"] = total_ann_acc[task_name] / max(total_samples, 1)
            ann_losses.append(metrics[f"train_loss_{task_name}"])
            ann_accs.append(metrics[f"train_acc_{task_name}"])
        metrics["train_loss_annotation_mean"] = float(np.mean(ann_losses)) if ann_losses else None
        metrics["train_acc_annotation_mean"] = float(np.mean(ann_accs)) if ann_accs else None
    return metrics


def evaluate_epoch(
    backbone,
    annotation_heads,
    loader,
    device,
    vocab,
    pad_token,
    mask_value,
    amp,
    use_recon,
    use_annotation,
    task_names,
):
    backbone.eval()
    annotation_heads.eval()

    total_recon = 0.0
    total_mre = 0.0
    total_batches = 0
    total_samples = 0
    total_ann = {task_name: 0.0 for task_name in task_names}
    logits_all = {task_name: [] for task_name in task_names}
    labels_all = {task_name: [] for task_name in task_names}

    with torch.no_grad():
        it = tqdm(loader, total=len(loader), desc="valid", unit="batch")
        for batch in it:
            batch_size = batch["gene_ids"].shape[0]
            total_samples += int(batch_size)

            model_out = run_model(
                backbone=backbone,
                annotation_heads=annotation_heads,
                batch=batch,
                device=device,
                use_recon=use_recon,
                use_annotation=use_annotation,
                task_names=task_names,
                vocab=vocab,
                pad_token=pad_token,
                mask_value=mask_value,
                amp=amp,
            )

            postfix = {}
            if use_recon:
                total_recon += float(model_out["loss_recon"].item())
                total_mre += float(model_out["mre"].item())
                postfix["mse"] = f"{model_out['loss_recon'].item():.4f}"

            if use_annotation:
                for task_name in task_names:
                    logits = model_out["logits"][task_name]
                    labels = batch[f"labels_{task_name}"].to(device)
                    total_ann[task_name] += float(
                        F.cross_entropy(logits, labels, reduction="sum").item()
                    )
                    logits_all[task_name].append(logits.cpu().numpy())
                    labels_all[task_name].append(labels.cpu().numpy())
                ann_loss = sum(total_ann.values()) / max(total_samples * max(len(task_names), 1), 1)
                postfix["cls"] = f"{ann_loss:.4f}"

            total_batches += 1
            it.set_postfix(**postfix)

    metrics = {}
    if use_recon:
        metrics.update(
            {
                "valid_mse": total_recon / max(total_batches, 1),
                "valid_mre": total_mre / max(total_batches, 1),
            }
        )
    if use_annotation:
        logits_cat = {task_name: np.concatenate(logits_all[task_name], axis=0) for task_name in task_names}
        labels_cat = {task_name: np.concatenate(labels_all[task_name], axis=0) for task_name in task_names}
        for task_name in task_names:
            metrics[f"valid_loss_{task_name}"] = total_ann[task_name] / max(total_samples, 1)
        metrics["valid_loss_annotation_mean"] = float(
            np.mean([metrics[f"valid_loss_{task_name}"] for task_name in task_names])
        )
        metrics.update(aggregate_annotation_metrics(task_names, logits_cat, labels_cat))
    return metrics


def build_output_dir(args) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        label_tag = "-".join(args.annotation_task_names) if args.annotation_task_names else "nolabel"
        run_name = args.run_name or (
            f"seaad_{args.variant}_{args.training_tasks}_{label_tag}_{args.input_space}_"
            f"{'fullseq' if args.include_zero_gene else 'official'}_"
            f"train{args.train_total_cells}_val{args.val_total_cells}_s{args.seed}"
        )
        out_dir = Path(args.result_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_checkpoint_dir(
    out_dir: Path,
    backbone,
    annotation_heads,
    pretrained_dir: Path,
    pretrained_cfg: dict,
    effective_max_seq_len: int,
    args,
    best_metrics: dict,
    summary: dict,
    label_spaces,
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
    saved_args["training_tasks"] = args.training_tasks
    saved_args["annotation_label_keys"] = args.annotation_label_keys
    saved_args["annotation_task_names"] = args.annotation_task_names
    (out_dir / "args.json").write_text(json.dumps(saved_args, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "finetune_config.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "label_spaces.json").write_text(json.dumps(label_spaces, ensure_ascii=False, indent=2) + "\n")

    backbone_state = backbone.state_dict()
    annotation_state = annotation_heads.state_dict()

    torch.save(backbone_state, out_dir / "best_model.pt")
    torch.save(backbone_state, out_dir / "best_backbone_model.pt")
    torch.save(annotation_state, out_dir / "annotation_heads.pt")
    torch.save(
        {
            "backbone_state_dict": backbone_state,
            "annotation_heads_state_dict": annotation_state,
            "training_tasks": args.training_tasks,
            "annotation_label_keys": args.annotation_label_keys,
            "annotation_task_names": args.annotation_task_names,
            "label_spaces": label_spaces,
            "best_metrics": best_metrics,
        },
        out_dir / "best_joint_model.pt",
    )
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n")


def make_selection_score(metrics, mode, use_recon, use_annotation):
    if mode == "valid_mse":
        return -float(metrics["valid_mse"])
    if mode == "annotation_mean_macro_f1":
        value = metrics.get("annotation_mean_macro_f1")
        return float("-inf") if value is None else float(value)
    if mode == "joint_exact_match":
        value = metrics.get("annotation_joint_exact_match")
        return float("-inf") if value is None else float(value)
    if mode == "auto":
        if use_annotation:
            value = metrics.get("annotation_mean_macro_f1")
            return float("-inf") if value is None else float(value)
        if use_recon:
            return -float(metrics["valid_mse"])
    raise ValueError(f"Unsupported model_selection_metric: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained scGPT on SEA-AD full-gene reconstruction and/or annotation."
    )
    parser.add_argument("--variant", type=str, default="brain", choices=["brain", "human"])
    parser.add_argument("--pretrained_dir", type=str, default="")
    parser.add_argument("--donor_csv", type=str, default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", type=str, default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_total_cells", type=int, default=12000)
    parser.add_argument("--val_total_cells", type=int, default=2000)
    parser.add_argument("--train_donor_seed", type=int, default=1234)
    parser.add_argument("--val_donor_seed", type=int, default=2345)
    parser.add_argument("--train_cell_seed", type=int, default=3456)
    parser.add_argument("--val_cell_seed", type=int, default=4567)
    parser.add_argument("--input_space", type=str, default="binned", choices=["binned", "log1p"])
    parser.add_argument("--n_bins", type=int, default=51)
    parser.add_argument("--include_zero_gene", action="store_true", help="Use all genes including zeros.")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1201,
        help="Official-style default is 1201 (= 1200 genes + <cls>). Use <=0 with --include_zero_gene for full-length sequences.",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--zero_mask_ratio", type=float, default=None)
    parser.add_argument("--mask_seed", type=int, default=5678)
    parser.add_argument("--val_mask_seed", type=int, default=6789)
    parser.add_argument("--trunc_seed", type=int, default=7890)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_fast_transformer", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument(
        "--training_tasks",
        type=str,
        default="joint",
        choices=["recon", "annotation", "joint"],
        help="Recon only, annotation only, or joint training.",
    )
    parser.add_argument(
        "--annotation_label_keys",
        type=str,
        default=",".join(DEFAULT_LABEL_KEYS),
        help="Comma-separated adata.obs column names used for annotation.",
    )
    parser.add_argument(
        "--annotation_task_names",
        type=str,
        default=",".join(DEFAULT_TASK_NAMES),
        help="Comma-separated task names aligned with --annotation_label_keys.",
    )
    parser.add_argument("--annotation_head_dropout", type=float, default=0.2)
    parser.add_argument("--recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--annotation_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--model_selection_metric",
        type=str,
        default="auto",
        choices=["auto", "valid_mse", "annotation_mean_macro_f1", "joint_exact_match"],
    )
    args = parser.parse_args()

    args.annotation_label_keys = tuple(parse_csv_list(args.annotation_label_keys))
    args.annotation_task_names = tuple(parse_csv_list(args.annotation_task_names))
    if len(args.annotation_label_keys) != len(args.annotation_task_names):
        raise ValueError("--annotation_label_keys and --annotation_task_names must have the same length.")

    use_recon = args.training_tasks in ("recon", "joint")
    use_annotation = args.training_tasks in ("annotation", "joint")
    if not use_recon and not use_annotation:
        raise ValueError("At least one task must be enabled.")
    if use_annotation and len(args.annotation_task_names) == 0:
        raise ValueError("Annotation training requires at least one label column/task name.")
    if args.freeze_backbone and not use_annotation:
        raise ValueError("--freeze_backbone requires annotation heads to remain trainable.")

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
        f"[setup] variant={args.variant} training_tasks={args.training_tasks} "
        f"input_space={args.input_space} include_zero_gene={args.include_zero_gene} epochs={args.epochs}",
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

    label_spaces = {}
    train_labels = None
    val_labels = None
    dropped_train_labels = {}
    dropped_val_labels = {}
    if use_annotation:
        label_spaces = build_label_spaces(train_adata, args.annotation_label_keys, args.annotation_task_names)
        train_labels, train_valid_mask, dropped_train_labels = encode_annotation_labels(
            train_adata,
            args.annotation_label_keys,
            args.annotation_task_names,
            label_spaces,
        )
        val_labels, val_valid_mask, dropped_val_labels = encode_annotation_labels(
            val_adata,
            args.annotation_label_keys,
            args.annotation_task_names,
            label_spaces,
        )
        train_adata, train_labels = filter_adata_and_labels(train_adata, train_labels, train_valid_mask)
        val_adata, val_labels = filter_adata_and_labels(val_adata, val_labels, val_valid_mask)
        if train_adata.n_obs == 0 or val_adata.n_obs == 0:
            raise RuntimeError("No cells remain after annotation label filtering. Check label columns.")

        print(
            f"[annotation] train cells after label filtering={train_adata.n_obs} "
            f"val cells after label filtering={val_adata.n_obs}",
            flush=True,
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

    backbone = TransformerModel(
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
    load_pretrained(backbone, extract_backbone_state_dict(pretrained_state), verbose=False)

    annotation_heads = nn.ModuleDict()
    if use_annotation:
        for task_name in args.annotation_task_names:
            annotation_heads[task_name] = AnnotationHead(
                d_model=pretrained_cfg["embsize"],
                n_classes=len(label_spaces[task_name]),
                dropout=args.annotation_head_dropout,
            )

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and torch.cuda.is_available())
    backbone.to(device)
    annotation_heads.to(device)
    print(f"[setup] device={device} amp={amp} freeze_backbone={args.freeze_backbone}", flush=True)

    trainable_params = [p for p in list(backbone.parameters()) + list(annotation_heads.parameters()) if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Disable --freeze_backbone or enable annotation heads.")
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-4 if amp else 1e-8,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    history = []
    history_path = out_dir / "history.jsonl"
    pad_value = int(pretrained_cfg.get("pad_value", -2))
    mask_value = int(pretrained_cfg.get("mask_value", -1))
    best_selection_score = float("-inf")
    best_metrics = {}

    fixed_val_data, fixed_val_loader, val_mask_stats = prepare_epoch_data_joint(
        tokenized=tokenized_val,
        batch_size=args.batch_size,
        recon_enabled=use_recon,
        mask_ratio=args.mask_ratio,
        zero_mask_ratio=args.zero_mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
        mask_seed=args.val_mask_seed,
        shuffle=False,
        annotation_labels=val_labels,
    )
    del fixed_val_data

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch] {epoch}/{args.epochs}", flush=True)
        epoch_data, train_loader, train_mask_stats = prepare_epoch_data_joint(
            tokenized=tokenized_train,
            batch_size=args.batch_size,
            recon_enabled=use_recon,
            mask_ratio=args.mask_ratio,
            zero_mask_ratio=args.zero_mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
            mask_seed=args.mask_seed + epoch * 100000,
            shuffle=True,
            annotation_labels=train_labels,
        )
        del epoch_data

        t0 = time.time()
        train_metrics = train_epoch(
            backbone=backbone,
            annotation_heads=annotation_heads,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            vocab=vocab,
            pad_token="<pad>",
            mask_value=mask_value,
            amp=amp,
            grad_clip=args.grad_clip,
            use_recon=use_recon,
            use_annotation=use_annotation,
            task_names=args.annotation_task_names,
            recon_loss_weight=args.recon_loss_weight,
            annotation_loss_weight=args.annotation_loss_weight,
        )
        valid_metrics = evaluate_epoch(
            backbone=backbone,
            annotation_heads=annotation_heads,
            loader=fixed_val_loader,
            device=device,
            vocab=vocab,
            pad_token="<pad>",
            mask_value=mask_value,
            amp=amp,
            use_recon=use_recon,
            use_annotation=use_annotation,
            task_names=args.annotation_task_names,
        )

        epoch_metrics = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 2),
            "training_tasks": args.training_tasks,
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

        selection_score = make_selection_score(
            metrics=epoch_metrics,
            mode=args.model_selection_metric,
            use_recon=use_recon,
            use_annotation=use_annotation,
        )
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_metrics = dict(epoch_metrics)
            summary = {
                "variant": args.variant,
                "pretrained_dir": str(pretrained_dir),
                "dataset": "SEA-AD MTG donor_h5ad full gene space",
                "split_rule": f"scUniFlow donor split val_ratio=0.1 seed={args.split_seed}",
                "training_tasks": args.training_tasks,
                "annotation_label_keys": list(args.annotation_label_keys),
                "annotation_task_names": list(args.annotation_task_names),
                "label_spaces": label_spaces,
                "dropped_train_labels": dropped_train_labels,
                "dropped_val_labels": dropped_val_labels,
                "freeze_backbone": bool(args.freeze_backbone),
                "include_zero_gene": bool(args.include_zero_gene),
                "effective_max_seq_len": int(effective_max_seq_len),
                "input_space": args.input_space,
                "n_bins": int(args.n_bins if args.input_space == "binned" else 0),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "recon_loss_weight": float(args.recon_loss_weight),
                "annotation_loss_weight": float(args.annotation_loss_weight),
                "model_selection_metric": args.model_selection_metric,
                "mask_ratio_target": float(args.mask_ratio if use_recon else 0.0),
                "nonzero_mask_ratio_target": float(args.mask_ratio if use_recon else 0.0),
                "zero_mask_ratio_target": float(
                    (args.zero_mask_ratio if args.zero_mask_ratio is not None else args.mask_ratio)
                    if use_recon
                    else 0.0
                ),
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
                backbone=backbone,
                annotation_heads=annotation_heads,
                pretrained_dir=pretrained_dir,
                pretrained_cfg=pretrained_cfg,
                effective_max_seq_len=effective_max_seq_len,
                args=args,
                best_metrics=best_metrics,
                summary=summary,
                label_spaces=label_spaces,
            )
            print(f"[saved] best checkpoint updated at {out_dir}", flush=True)

    final_summary = {
        "output_dir": str(out_dir),
        "best_model_pt": str(out_dir / "best_model.pt"),
        "best_backbone_model_pt": str(out_dir / "best_backbone_model.pt"),
        "best_joint_model_pt": str(out_dir / "best_joint_model.pt"),
        "annotation_heads_pt": str(out_dir / "annotation_heads.pt"),
        "history_jsonl": str(history_path),
        "best_metrics": best_metrics,
        "training_tasks": args.training_tasks,
        "train_cells": int(train_x.shape[0]),
        "val_cells": int(val_x.shape[0]),
        "genes_in_vocab": int(len(kept_gene_names)),
        "effective_max_seq_len": int(effective_max_seq_len),
        "input_space": args.input_space,
        "include_zero_gene": bool(args.include_zero_gene),
        "annotation_task_names": list(args.annotation_task_names),
        "label_space_sizes": {k: len(v) for k, v in label_spaces.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
