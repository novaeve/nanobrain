import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    DEFAULT_GENE_INDEX,
    ReadDepthEnhancementDataset,
    ScFoundationReadDepthEnhancer,
    align_h5ad_to_gene_list,
    derive_reference_labels,
    evaluate_clustering_metrics,
    load_finetune_state_dict,
    load_full_gene_counts_and_split,
    materialize_downsampled_full_gene_counts,
    read_gene_list,
    save_eval_outputs,
    set_seed,
    summarize_reconstruction,
)


@torch.no_grad()
def run_inference(model, loader, device, max_batches: int = 0):
    model.eval()
    all_preds = []
    all_targets = []
    all_low = []
    all_cell_idx = []
    n_batches = 0

    for x, target, cell_idx in tqdm(loader, desc="eval"):
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast("cuda", enabled=(device.type == "cuda")):
            recon_pred = model(x, x)

        all_preds.append(recon_pred.cpu().float().numpy())
        all_targets.append(target.cpu().float().numpy())
        all_low.append(x[:, :-2].cpu().float().numpy())
        all_cell_idx.append(cell_idx.numpy())
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_low, axis=0),
        np.concatenate(all_cell_idx, axis=0).astype(np.int64),
        n_batches,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate scFoundation read depth enhancement checkpoint"
    )
    parser.add_argument("--ckpt_path", required=True, help="fine-tuned checkpoint path")
    parser.add_argument("--data_path", required=True, help="evaluation h5ad")
    parser.add_argument("--save_dir", required=True, help="evaluation output directory")
    parser.add_argument("--pretrain_ckpt", default="", help="optional override for scFoundation pretrained checkpoint")
    parser.add_argument("--gene_index", default=str(DEFAULT_GENE_INDEX))
    parser.add_argument("--gene_name_col", default="gene_name")
    parser.add_argument("--layer", default="", help="optional h5ad layer name")
    parser.add_argument("--batch_size", type=int, default=0, help="0 means use training batch size")
    parser.add_argument("--max_batches", type=int, default=0, help="0 means use all batches")
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--cell_indices_path", default="", help="optional .npy file with exact original h5ad cell indices")
    parser.add_argument("--tgt_highres", type=float, default=-1.0, help="negative means reuse training setting")
    parser.add_argument("--downsample_fold", type=float, default=-1.0, help="negative means reuse training setting")
    parser.add_argument(
        "--eval_max_cells",
        type=int,
        default=-1,
        help="negative means reuse training setting; 0 means use full dataset without subsampling",
    )
    parser.add_argument(
        "--eval_val_ratio",
        type=float,
        default=-1.0,
        help="negative means reuse training setting",
    )
    parser.add_argument(
        "--eval_cell_seed",
        type=int,
        default=-1,
        help="negative means reuse training seed/cell_seed setting",
    )
    parser.add_argument("--label_col", default="", help="optional obs column used as clustering target")
    parser.add_argument("--target_labels_path", default="", help="optional .npy labels aligned to evaluated cells")
    parser.add_argument("--ref_resolution", type=float, default=0.4)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_arrays", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    train_args = ckpt.get("train_args", {})
    pretrain_ckpt = args.pretrain_ckpt or ckpt.get("pretrain_ckpt")
    if not pretrain_ckpt:
        raise ValueError("pretrain checkpoint path is missing; pass --pretrain_ckpt explicitly")

    batch_size = int(args.batch_size or train_args.get("batch_size", 32))
    tgt_highres = float(args.tgt_highres if args.tgt_highres >= 0 else train_args.get("tgt_highres", 4.0))
    downsample_fold = float(args.downsample_fold if args.downsample_fold >= 0 else train_args.get("downsample_fold", 4.0))
    split_seed = int(train_args.get("seed", args.seed))
    val_ratio = float(train_args.get("val_ratio", 0.1))
    max_cells = int(train_args.get("max_cells", 0) or 0)
    cell_seed = int(train_args.get("cell_seed", split_seed))
    eval_val_ratio = float(args.eval_val_ratio if args.eval_val_ratio >= 0 else val_ratio)
    eval_cell_seed = int(args.eval_cell_seed if args.eval_cell_seed >= 0 else cell_seed)
    eval_max_cells = int(args.eval_max_cells if args.eval_max_cells >= 0 else max_cells)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    gene_list = read_gene_list(args.gene_index)
    if args.cell_indices_path and eval_max_cells > 0:
        raise ValueError("--cell_indices_path expects original h5ad indices; do not combine it with eval_max_cells > 0")
    max_cells_arg = None if args.cell_indices_path or eval_max_cells <= 0 else eval_max_cells
    full_counts, _, _, full_meta, split = load_full_gene_counts_and_split(
        args.data_path,
        val_ratio=eval_val_ratio,
        seed=split_seed,
        max_cells=max_cells_arg,
        cell_seed=eval_cell_seed,
    )
    X, meta = align_h5ad_to_gene_list(
        args.data_path,
        gene_list,
        gene_name_col=args.gene_name_col,
        layer=args.layer or None,
        max_cells=max_cells_arg,
        cell_seed=eval_cell_seed,
    )

    full_obs_names = full_meta["obs"].index.astype(str).tolist()
    aligned_obs_names = meta["obs_names"]
    if aligned_obs_names != full_obs_names:
        raise RuntimeError(
            "Aligned scFoundation view and full-gene scUniFlow view do not share the same "
            "cell ordering after subsampling. Please check max_cells/cell_seed alignment."
        )

    if args.cell_indices_path:
        eval_idx = np.load(args.cell_indices_path).astype(np.int64)
    elif args.split == "all":
        eval_idx = np.arange(X.shape[0])
    else:
        eval_idx = split.train_idx if args.split == "train" else split.val_idx

    dataset = ReadDepthEnhancementDataset(
        X,
        indices=eval_idx,
        tgt_highres=tgt_highres,
        downsample_fold=downsample_fold,
        seed=split_seed,
        return_cell_idx=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ScFoundationReadDepthEnhancer(
        ckpt_path=pretrain_ckpt,
        freeze_emb=bool(train_args.get("freeze_emb", True)),
        unfreeze_last_n=int(train_args.get("unfreeze_last_n", 2)),
    ).to(device)
    load_finetune_state_dict(model, args.ckpt_path)

    preds, targets, low_inputs, used_cell_idx, used_batches = run_inference(
        model, loader, device, max_batches=int(args.max_batches)
    )
    cell_ids = [full_obs_names[int(i)] for i in used_cell_idx]
    metrics, cell_pccs, gene_pccs = summarize_reconstruction(preds, targets)
    full_low_counts, full_high_counts = materialize_downsampled_full_gene_counts(
        full_counts,
        used_cell_idx,
        downsample_fold=downsample_fold,
        seed=split_seed,
    )

    # Map scFoundation predictions back into the full-gene space. For genes outside the
    # scFoundation vocabulary overlap, we keep the low-depth baseline values unchanged.
    pred_norm = np.expm1(preds).clip(min=0.0)
    low_totals = full_low_counts.sum(axis=1, keepdims=True)
    est_high_totals = np.maximum(low_totals * float(downsample_fold), 1.0)
    pred_count_est = (pred_norm / 1e4) * est_high_totals

    src_cols = np.asarray(meta["src_cols"], dtype=np.int64)
    dst_cols = np.asarray(meta["dst_cols"], dtype=np.int64)
    enhanced_full_counts = full_low_counts.copy()
    if src_cols.size > 0:
        enhanced_full_counts[:, src_cols] = pred_count_est[:, dst_cols]

    metrics.update(
        {
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "pretrain_ckpt": str(Path(pretrain_ckpt).resolve()),
            "data_path": str(Path(args.data_path).resolve()),
            "split": args.split,
            "n_eval_cells": int(preds.shape[0]),
            "n_genes": int(full_high_counts.shape[1]),
            "n_model_genes": int(preds.shape[1]),
            "n_vocab_overlap_genes": int(src_cols.size),
            "used_batches": int(used_batches),
            "batch_size": int(batch_size),
            "max_batches": int(args.max_batches),
            "tgt_highres": float(tgt_highres),
            "downsample_fold": float(downsample_fold),
            "eval_max_cells": int(eval_max_cells),
            "eval_val_ratio": float(eval_val_ratio),
            "eval_cell_seed": int(eval_cell_seed),
            "cell_indices_path": str(Path(args.cell_indices_path).resolve()) if args.cell_indices_path else "",
            "target_labels_path": str(Path(args.target_labels_path).resolve()) if args.target_labels_path else "",
            "clustering_gene_space": "full_gene_scuniflow_protocol",
            "reconstruction_gene_space": "scfoundation_aligned_vocab",
        }
    )

    obs_eval = full_meta["obs"].iloc[used_cell_idx].copy()
    if args.target_labels_path:
        target_labels = np.load(args.target_labels_path, allow_pickle=True).astype(str)
        if target_labels.shape[0] != preds.shape[0]:
            raise ValueError(
                f"target_labels length mismatch: labels={target_labels.shape[0]} eval_cells={preds.shape[0]}"
            )
        n_clusters = len(np.unique(target_labels))
        reference_info = {
            "target_mode": "shared_labels",
            "target_labels_path": str(Path(args.target_labels_path).resolve()),
            "n_clusters": int(n_clusters),
        }
    elif args.label_col:
        if args.label_col not in obs_eval.columns:
            raise KeyError(f"label_col not found in obs: {args.label_col}")
        target_labels = obs_eval[args.label_col].astype(str).to_numpy()
        n_clusters = len(np.unique(target_labels))
        reference_info = {"target_mode": "obs_label", "label_col": args.label_col}
    else:
        target_labels, _ = derive_reference_labels(
            full_high_counts,
            ref_resolution=float(args.ref_resolution),
            n_top_genes=int(args.n_top_genes),
        )
        n_clusters = len(np.unique(target_labels))
        reference_info = {
            "target_mode": "reference_leiden",
            "ref_resolution": float(args.ref_resolution),
            "n_clusters": int(n_clusters),
        }

    enhanced_metrics, _, enhanced_clusters = evaluate_clustering_metrics(
        enhanced_full_counts,
        target_labels=target_labels,
        n_clusters=n_clusters,
        n_top_genes=int(args.n_top_genes),
    )
    low_metrics, _, low_clusters = evaluate_clustering_metrics(
        full_low_counts,
        target_labels=target_labels,
        n_clusters=n_clusters,
        n_top_genes=int(args.n_top_genes),
    )
    metrics["clustering"] = {
        "reference": reference_info,
        "low_depth_baseline": low_metrics,
        "enhanced": enhanced_metrics,
    }

    save_eval_outputs(save_dir, metrics, cell_ids, cell_pccs, gene_pccs)
    np.save(save_dir / "target_labels.npy", np.asarray(target_labels))
    np.save(save_dir / "enhanced_clusters.npy", np.asarray(enhanced_clusters))
    np.save(save_dir / "low_clusters.npy", np.asarray(low_clusters))

    if args.save_arrays:
        np.save(save_dir / "preds.npy", preds)
        np.save(save_dir / "targets.npy", targets)
        np.save(save_dir / "low_inputs.npy", low_inputs)
        np.save(save_dir / "full_low_counts.npy", full_low_counts)
        np.save(save_dir / "full_high_counts.npy", full_high_counts)
        np.save(save_dir / "enhanced_full_counts.npy", enhanced_full_counts)

    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {save_dir}")


if __name__ == "__main__":
    main()
