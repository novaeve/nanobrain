import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertForMaskedLM


THIS_DIR = Path(__file__).resolve().parent
FINETUNE_GENEFORMER_DIR = THIS_DIR / "finetune" / "Geneformer"
sys.path.insert(0, str(FINETUNE_GENEFORMER_DIR))

from donor_eval_utils import (
    DEFAULT_DONOR_CSV,
    DEFAULT_DONOR_DIR,
    pick_eval_donor,
    sample_balanced_val_donors,
    sample_single_donor_cells,
    split_val_donors,
)
from geneformer_seaad_joint_finetune import (
    DEFAULT_GENEFORMER_CODE_DIR,
    GeneformerReconAnnotationModel,
    PreparedGeneformerDataset,
    aggregate_annotation_metrics,
    build_prepared_examples,
    encode_annotation_labels,
    get_default_seq_len,
    load_geneformer_modules,
    make_collate_fn,
    parse_csv_list,
    prepare_geneformer_adata,
    summarize_gene_overlap,
    tokenize_adata_to_dataset,
)


DEFAULT_RESULT_DIR = Path("results/pretrain_eval")


def build_result_path(
    args,
    donor_id: str,
    sampled_cells: int,
    per_donor_n: int,
    tokenizer_seq_len: int,
    effective_mlm_probability: float,
) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    task_tag = args.eval_tasks
    if args.eval_subset_mode == "balanced_val_donors":
        filename = (
            f"geneformer_eval_{Path(args.ckpt_dir).name}_{task_tag}_"
            f"seed{args.split_seed}_{donor_id}_seq{tokenizer_seq_len}_perdonor{per_donor_n}_total{sampled_cells}_"
            f"mask{effective_mlm_probability:.4f}.json"
        )
    else:
        filename = (
            f"geneformer_eval_{Path(args.ckpt_dir).name}_{task_tag}_"
            f"seed{args.split_seed}_{donor_id}_seq{tokenizer_seq_len}_n{sampled_cells}_mask{effective_mlm_probability:.4f}.json"
        )
    return result_dir / filename.replace("/", "_")


def load_checkpoint_metadata(ckpt_dir: Path):
    manifest_path = ckpt_dir / "tokenizer_manifest.json"
    config_path = ckpt_dir / "finetune_config.json"
    label_path = ckpt_dir / "label_spaces.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Tokenizer manifest not found: {manifest_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Fine-tune config not found: {config_path}")

    with manifest_path.open() as f:
        manifest = json.load(f)
    with config_path.open() as f:
        finetune_cfg = json.load(f)
    if label_path.exists():
        with label_path.open() as f:
            label_spaces = json.load(f)
    else:
        label_spaces = {}
    return manifest, finetune_cfg, label_spaces


def resolve_eval_tasks(requested: str, label_spaces: dict):
    has_annotation = bool(label_spaces)
    if requested == "auto":
        return "joint" if has_annotation else "recon"
    if requested == "annotation" and not has_annotation:
        raise RuntimeError("Checkpoint has no annotation heads/label spaces, cannot run annotation eval.")
    if requested == "joint" and not has_annotation:
        raise RuntimeError("Checkpoint has no annotation heads/label spaces, cannot run joint eval.")
    return requested


def resolve_tokenizer_seq_len(args, manifest: dict, finetune_cfg: dict) -> int:
    if int(args.seq_len) > 0:
        return int(args.seq_len)
    saved_seq_len = manifest.get("tokenizer_model_input_size", finetune_cfg.get("seq_len", 0))
    if int(saved_seq_len) > 0:
        return int(saved_seq_len)
    return int(get_default_seq_len(manifest["model_version"]))


def resolve_effective_mlm_probability(args, tokenizer_seq_len: int) -> tuple[float, dict]:
    if int(args.full_length_seq_len) > 0:
        total_full_ratio = float(args.full_length_nonzero_mask_ratio) + float(args.full_length_zero_mask_ratio)
        effective_mlm_probability = total_full_ratio * float(tokenizer_seq_len) / float(args.full_length_seq_len)
        if effective_mlm_probability <= 0:
            raise ValueError("Scaled effective MLM probability must be > 0.")
        if effective_mlm_probability > 1:
            raise ValueError(
                f"Scaled effective MLM probability is > 1 ({effective_mlm_probability:.4f}). "
                "Please check full-length mask ratios and seq_len."
            )
        metadata = {
            "mask_ratio_mode": "scaled_from_full_length_total_budget",
            "full_length_seq_len": int(args.full_length_seq_len),
            "full_length_nonzero_mask_ratio": float(args.full_length_nonzero_mask_ratio),
            "full_length_zero_mask_ratio": float(args.full_length_zero_mask_ratio),
            "effective_mlm_probability": float(effective_mlm_probability),
            "note": (
                "Geneformer token sequences contain expressed-gene ranks only; zero-expression mask budget "
                "is folded into the total effective MLM probability."
            ),
        }
        return float(effective_mlm_probability), metadata

    metadata = {
        "mask_ratio_mode": "direct",
        "effective_mlm_probability": float(args.mlm_probability),
    }
    return float(args.mlm_probability), metadata


def evaluate_masked_recon(backbone, loader, device, amp, topk=(1, 5)):
    backbone.eval()
    total_loss = 0.0
    total_positions = 0
    total_correct = {int(k): 0 for k in topk}

    with torch.no_grad():
        it = tqdm(loader, total=len(loader), desc="geneformer recon eval", unit="batch")
        for batch in it:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            masked_positions = mlm_labels.ne(-100)
            n_masked = int(masked_positions.sum().item())
            if n_masked == 0:
                continue

            with torch.cuda.amp.autocast(enabled=amp):
                logits = backbone(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    mlm_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

            total_loss += float(loss.item())
            total_positions += n_masked

            masked_logits = logits[masked_positions]
            masked_labels = mlm_labels[masked_positions]
            max_k = max(int(k) for k in topk)
            topk_idx = masked_logits.topk(max_k, dim=-1).indices
            for k in topk:
                matches = topk_idx[:, : int(k)].eq(masked_labels.unsqueeze(1)).any(dim=1)
                total_correct[int(k)] += int(matches.sum().item())

            postfix = {"masked": total_positions}
            if 1 in total_correct and total_positions > 0:
                postfix["acc@1"] = f"{total_correct[1] / total_positions:.4f}"
            it.set_postfix(**postfix)

    metrics = {
        "masked_n_eval_positions": int(total_positions),
        "masked_ce_loss": float(total_loss / max(total_positions, 1)),
    }
    for k, v in sorted(total_correct.items()):
        metrics[f"masked_accuracy_top{k}"] = float(v / max(total_positions, 1))
    return metrics


def predict_annotation(model, loader, device, amp, task_names):
    model.eval()
    logits_all = {task_name: [] for task_name in task_names}
    labels_all = {task_name: [] for task_name in task_names}

    with torch.no_grad():
        it = tqdm(loader, total=len(loader), desc="geneformer annotation eval", unit="batch")
        for batch in it:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                _, annotation_logits = model.encode(input_ids, attention_mask)
            for task_name in task_names:
                logits_all[task_name].append(annotation_logits[task_name].cpu().numpy())
                labels_all[task_name].append(batch[f"labels_{task_name}"].cpu().numpy())

    logits_all = {task_name: np.concatenate(logits_all[task_name], axis=0) for task_name in task_names}
    labels_all = {task_name: np.concatenate(labels_all[task_name], axis=0) for task_name in task_names}
    return logits_all, labels_all


def main():
    parser = argparse.ArgumentParser(
        description="Joint eval for fine-tuned Geneformer checkpoints on SEA-AD."
    )
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Fine-tuned Geneformer checkpoint directory.")
    parser.add_argument("--geneformer_code_dir", type=str, default="")
    parser.add_argument("--donor_csv", type=str, default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", type=str, default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--eval_donor", type=str, default="")
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--mask_seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gene_id_col", type=str, default="index")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=0,
        help="Tokenizer model_input_size override. 0 means use checkpoint-saved value or model-version default.",
    )
    parser.add_argument("--annotation_label_keys", type=str, default="")
    parser.add_argument("--annotation_task_names", type=str, default="")
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument(
        "--full_length_seq_len",
        type=int,
        default=0,
        help="If >0, derive eval MLM probability by scaling a full-length mask budget down to current seq_len.",
    )
    parser.add_argument(
        "--full_length_nonzero_mask_ratio",
        type=float,
        default=0.20,
        help="Nonzero mask ratio defined on the full-length reference sequence when using --full_length_seq_len.",
    )
    parser.add_argument(
        "--full_length_zero_mask_ratio",
        type=float,
        default=0.02,
        help="Zero-value mask ratio defined on the full-length reference sequence when using --full_length_seq_len.",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="auto",
        choices=["auto", "recon", "annotation", "joint"],
        help="auto: infer from checkpoint; recon: MLM only; annotation: label eval only; joint: run both.",
    )
    parser.add_argument(
        "--eval_subset_mode",
        type=str,
        default="balanced_val_donors",
        choices=["balanced_val_donors", "single_donor"],
    )
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    donor_csv = Path(args.donor_csv)
    donor_dir = Path(args.donor_dir)

    manifest, finetune_cfg, label_spaces = load_checkpoint_metadata(ckpt_dir)
    args.eval_tasks = resolve_eval_tasks(args.eval_tasks, label_spaces)
    tokenizer_seq_len = resolve_tokenizer_seq_len(args, manifest, finetune_cfg)
    effective_mlm_probability, mask_ratio_meta = resolve_effective_mlm_probability(args, tokenizer_seq_len)

    default_label_keys = tuple(finetune_cfg.get("annotation_label_keys", []))
    default_task_names = tuple(finetune_cfg.get("annotation_task_names", []))
    args.annotation_label_keys = tuple(parse_csv_list(args.annotation_label_keys)) if args.annotation_label_keys else default_label_keys
    args.annotation_task_names = tuple(parse_csv_list(args.annotation_task_names)) if args.annotation_task_names else default_task_names
    if args.eval_tasks in ("annotation", "joint") and len(args.annotation_label_keys) != len(args.annotation_task_names):
        raise ValueError("annotation_label_keys and annotation_task_names must have the same length.")

    print(
        f"[setup] ckpt_dir={ckpt_dir} subset_mode={args.eval_subset_mode} "
        f"eval_tasks={args.eval_tasks} tokenizer_seq_len={tokenizer_seq_len} "
        f"effective_mlm_probability={effective_mlm_probability:.4f}",
        flush=True,
    )
    if mask_ratio_meta["mask_ratio_mode"] != "direct":
        print(
            "[setup] scaled MLM budget from full length: "
            f"full_length_seq_len={mask_ratio_meta['full_length_seq_len']} "
            f"nonzero_ratio={mask_ratio_meta['full_length_nonzero_mask_ratio']:.4f} "
            f"zero_ratio={mask_ratio_meta['full_length_zero_mask_ratio']:.4f} "
            f"-> effective_mlm_probability={mask_ratio_meta['effective_mlm_probability']:.4f}",
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
        sampled_cells_before_filter = int(adata.n_obs)
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
        sampled_cells_before_filter = int(n)

    geneformer_code_dir = Path(args.geneformer_code_dir) if args.geneformer_code_dir else Path(
        manifest.get("geneformer_code_dir", str(DEFAULT_GENEFORMER_CODE_DIR))
    )
    module_info = load_geneformer_modules(geneformer_code_dir)
    TranscriptomeTokenizer = module_info["TranscriptomeTokenizer"]
    tokenizer_files = module_info[manifest["model_version"]]

    label_keys_for_prep = args.annotation_label_keys if args.eval_tasks in ("annotation", "joint") else ()
    adata = prepare_geneformer_adata(adata, args.gene_id_col, label_keys=label_keys_for_prep)

    metadata_cols = ["unique_cell_id"]
    if args.eval_tasks in ("annotation", "joint"):
        metadata_cols.extend(list(args.annotation_label_keys))

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={k: k for k in metadata_cols},
        nproc=1,
        chunk_size=512,
        model_input_size=tokenizer_seq_len,
        model_version=manifest["model_version"],
        use_h5ad_index=False,
        keep_counts=False,
        gene_median_file=Path(manifest.get("gene_median_file", tokenizer_files["gene_median_file"])),
        token_dictionary_file=Path(manifest.get("token_dictionary_file", tokenizer_files["token_dictionary_file"])),
        gene_mapping_file=Path(manifest.get("gene_mapping_file", tokenizer_files["gene_mapping_file"])),
    )
    gene_stats = summarize_gene_overlap(adata, tokenizer)

    print("[prep] tokenizing evaluation subset with Geneformer tokenizer", flush=True)
    tokenized = tokenize_adata_to_dataset(
        adata=adata,
        tokenizer=tokenizer,
        temp_h5ad_path=ckpt_dir / "_tmp_joint_eval_subset.h5ad",
        metadata_cols=metadata_cols,
        target_sum=10000.0,
        save_to_disk_path=None,
    )

    encoded_labels = {}
    dropped_labels = {}
    if args.eval_tasks in ("annotation", "joint"):
        tokenized, encoded_labels, dropped_labels = encode_annotation_labels(
            dataset=tokenized,
            label_keys=args.annotation_label_keys,
            task_names=args.annotation_task_names,
            label_spaces=label_spaces,
        )
        if len(tokenized) == 0:
            raise RuntimeError("No cells remain after label-space filtering.")

    backbone_dir = ckpt_dir / "best_backbone"
    if not backbone_dir.exists():
        raise FileNotFoundError(f"Backbone directory not found: {backbone_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and torch.cuda.is_available())

    out = {
        "checkpoint": str(ckpt_dir),
        "dataset": "SEA-AD MTG donor_h5ad",
        "donor_id": donor_id,
        "donor_cells_total": int(donor_cells),
        "eval_subset_mode": args.eval_subset_mode,
        "balanced_per_donor_cells": int(per_donor_n),
        "donor_sampling": donor_stats,
        "sampled_cells_before_label_filter": int(sampled_cells_before_filter),
        "sampled_cells": int(len(tokenized)),
        "split_rule": f"scUniFlow donor split val_ratio=0.1 seed={args.split_seed}",
        "unseen_for_scuniflow_train": True,
        "model_version": manifest["model_version"],
        "tokenizer_seq_len": int(tokenizer_seq_len),
        "eval_tasks": args.eval_tasks,
        "matched_genes_mapped": int(gene_stats["mapped_genes"]),
        "usable_genes": int(gene_stats["usable_genes"]),
        "all_genes": int(gene_stats["input_genes"]),
        "batch_size": int(args.batch_size),
        **mask_ratio_meta,
    }

    if args.eval_tasks in ("recon", "joint"):
        recon_examples = build_prepared_examples(
            dataset=tokenized,
            encoded_labels={},
            task_names=(),
            use_recon=True,
            mlm_probability=effective_mlm_probability,
            mask_token_id=int(tokenizer.gene_token_dict["<mask>"]),
            vocab_size=int(len(tokenizer.gene_token_dict)),
            special_token_ids={
                int(tokenizer.gene_token_dict["<pad>"]),
                int(tokenizer.gene_token_dict["<mask>"]),
                int(tokenizer.gene_token_dict.get("<cls>", -1)),
                int(tokenizer.gene_token_dict.get("<eos>", -1)),
            },
            seed=args.mask_seed,
        )
        recon_dataset = PreparedGeneformerDataset(recon_examples, ())
        recon_loader = DataLoader(
            recon_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=make_collate_fn((), pad_token_id=int(tokenizer.gene_token_dict["<pad>"])),
        )
        recon_model = BertForMaskedLM.from_pretrained(str(backbone_dir))
        recon_model.to(device)
        print(
            f"[eval] recon donor={donor_id} sampled_cells={len(tokenized)} batches={len(recon_loader)} "
            f"mlm_probability={effective_mlm_probability} device={device} amp={amp}",
            flush=True,
        )
        recon_metrics = evaluate_masked_recon(recon_model, recon_loader, device=device, amp=amp)
        out.update(
            {
                "mlm_probability": float(effective_mlm_probability),
                **recon_metrics,
            }
        )

    if args.eval_tasks in ("annotation", "joint"):
        model = GeneformerReconAnnotationModel(
            pretrained_model_dir=backbone_dir,
            num_labels_by_task={task_name: len(label_spaces[task_name]) for task_name in args.annotation_task_names},
            annotation_head_dropout=float(finetune_cfg.get("annotation_head_dropout", 0.2)),
            pooling=str(finetune_cfg.get("pooling", "cls")),
            freeze_layers=0,
            freeze_embeddings=False,
        )
        joint_state = torch.load(ckpt_dir / "best_joint_model.pt", map_location="cpu")
        state_dict = joint_state["model_state_dict"] if "model_state_dict" in joint_state else joint_state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to cleanly load fine-tuned Geneformer joint checkpoint.\nmissing={missing}\nunexpected={unexpected}"
            )
        model.to(device)

        annotation_examples = build_prepared_examples(
            dataset=tokenized,
            encoded_labels=encoded_labels,
            task_names=args.annotation_task_names,
            use_recon=False,
            mlm_probability=0.0,
            mask_token_id=int(tokenizer.gene_token_dict["<mask>"]),
            vocab_size=int(len(tokenizer.gene_token_dict)),
            special_token_ids={
                int(tokenizer.gene_token_dict["<pad>"]),
                int(tokenizer.gene_token_dict["<mask>"]),
                int(tokenizer.gene_token_dict.get("<cls>", -1)),
                int(tokenizer.gene_token_dict.get("<eos>", -1)),
            },
            seed=0,
        )
        annotation_dataset = PreparedGeneformerDataset(annotation_examples, args.annotation_task_names)
        annotation_loader = DataLoader(
            annotation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=make_collate_fn(
                args.annotation_task_names,
                pad_token_id=int(tokenizer.gene_token_dict["<pad>"]),
            ),
        )

        print(
            f"[eval] annotation donor={donor_id} sampled_cells={len(tokenized)} batches={len(annotation_loader)} "
            f"device={device} amp={amp}",
            flush=True,
        )
        logits_all, labels_all = predict_annotation(
            model=model,
            loader=annotation_loader,
            device=device,
            amp=amp,
            task_names=args.annotation_task_names,
        )
        ann_metrics = aggregate_annotation_metrics(args.annotation_task_names, logits_all, labels_all)
        for task_name in args.annotation_task_names:
            ann_metrics[f"valid_loss_{task_name}"] = float(
                F.cross_entropy(
                    torch.from_numpy(logits_all[task_name]),
                    torch.from_numpy(labels_all[task_name]),
                ).item()
            )

        out.update(
            {
                "annotation_label_keys": list(args.annotation_label_keys),
                "annotation_task_names": list(args.annotation_task_names),
                "label_space_sizes": {k: int(len(v)) for k, v in label_spaces.items()},
                "dropped_labels": dropped_labels,
                **ann_metrics,
            }
        )

    result_path = build_result_path(
        args,
        donor_id,
        sampled_cells=len(tokenized),
        per_donor_n=per_donor_n,
        tokenizer_seq_len=tokenizer_seq_len,
        effective_mlm_probability=effective_mlm_probability,
    )
    out["result_json"] = str(result_path)
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    print(f"[done] wrote result json to {result_path}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
