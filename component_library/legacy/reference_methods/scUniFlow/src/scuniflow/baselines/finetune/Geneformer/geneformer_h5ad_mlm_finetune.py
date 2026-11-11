import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = THIS_DIR.parent
EVAL_DIR = FINETUNE_DIR.parent
REPO_ROOT = EVAL_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(THIS_DIR))

from geneformer_seaad_joint_finetune import (  # noqa: E402
    GeneformerReconAnnotationModel,
    PreparedGeneformerDataset,
    build_prepared_examples,
    ensure_obs_n_counts,
    ensure_unique_cell_ids,
    evaluate_epoch,
    find_pretrained_weight_file,
    load_geneformer_modules,
    make_collate_fn,
    make_selection_score,
    mask_sequence,
    prepare_geneformer_adata,
    resolve_tokenizer_seq_len,
    tokenize_adata_to_dataset,
    train_epoch,
)


DEFAULT_GENEFORMER_CODE_DIR = Path("external/Geneformer")
DEFAULT_RESULT_ROOT = Path("./src/scuniflow/baselines/finetune/outputs/Geneformer")
DEFAULT_MODEL_FILE = Path(
    "external/model_zoo/"
    "4014c514594e479d8fd84d2c65dd94fc/Research/model.safetensors"
)
DEFAULT_MODEL_CONFIG = Path("external/Geneformer/config.json")


def sample_indices(indices: np.ndarray, n_cells: int, seed: int):
    if n_cells is None or n_cells <= 0 or n_cells >= len(indices):
        return np.array(indices, copy=True)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(indices, size=int(n_cells), replace=False))


def split_train_val_indices(n_obs: int, val_ratio: float, seed: int):
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_obs)
    n_val = max(1, int(round(n_obs * float(val_ratio))))
    n_val = min(n_val, n_obs - 1)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def build_output_dir(args) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        dataset_stem = Path(args.h5ad_path).stem
        run_name = args.run_name or (
            f"{dataset_stem}_geneformer_{args.model_version.lower()}_mlm_"
            f"train{args.train_total_cells}_val{args.val_total_cells}_s{args.seed}"
        )
        out_dir = Path(args.result_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_pretrained_model_dir(args, out_dir: Path) -> Path:
    if args.pretrained_model_dir:
        model_dir = Path(args.pretrained_model_dir).resolve()
        if find_pretrained_weight_file(model_dir) is None:
            raise FileNotFoundError(
                f"No pretrained weights found in {model_dir}; expected pytorch_model.bin or model.safetensors."
            )
        return model_dir

    model_file = Path(args.pretrained_model_file).resolve()
    config_file = Path(args.pretrained_config).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Geneformer weight file not found: {model_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"Geneformer config file not found: {config_file}")

    model_dir = out_dir / "_pretrained_geneformer"
    model_dir.mkdir(parents=True, exist_ok=True)
    dst_weight = model_dir / model_file.name
    if not dst_weight.exists():
        try:
            dst_weight.symlink_to(model_file)
        except OSError:
            shutil.copy2(model_file, dst_weight)
    shutil.copy2(config_file, model_dir / "config.json")
    return model_dir


def save_best_checkpoint(
    out_dir: Path,
    model,
    tokenizer_files,
    tokenizer_seq_len,
    args,
    best_metrics,
    summary,
):
    backbone_dir = out_dir / "best_backbone"
    if backbone_dir.exists():
        shutil.rmtree(backbone_dir)
    model.backbone.save_pretrained(str(backbone_dir))

    tokenizer_manifest = {
        "model_version": args.model_version,
        "geneformer_code_dir": str(args.geneformer_code_dir),
        "gene_median_file": str(tokenizer_files["gene_median_file"]),
        "token_dictionary_file": str(tokenizer_files["token_dictionary_file"]),
        "gene_mapping_file": str(tokenizer_files["gene_mapping_file"]),
        "tokenizer_model_input_size": int(tokenizer_seq_len),
    }

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    (out_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "finetune_config.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "tokenizer_manifest.json").write_text(json.dumps(tokenizer_manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Geneformer on a single h5ad file with MLM reconstruction."
    )
    parser.add_argument("--h5ad_path", type=str, required=True)
    parser.add_argument("--geneformer_code_dir", type=str, default=str(DEFAULT_GENEFORMER_CODE_DIR))
    parser.add_argument("--pretrained_model_dir", type=str, default="")
    parser.add_argument("--pretrained_model_file", type=str, default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--pretrained_config", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--model_version", type=str, default="V2", choices=["V1", "V2"])
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_cell_seed", type=int, default=3456)
    parser.add_argument("--val_cell_seed", type=int, default=4567)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--train_total_cells", type=int, default=12000)
    parser.add_argument("--val_total_cells", type=int, default=2000)
    parser.add_argument("--train_indices_path", default="", help="optional .npy original h5ad cell indices for training")
    parser.add_argument("--val_indices_path", default="", help="optional .npy original h5ad cell indices for validation")
    parser.add_argument("--gene_id_col", type=str, default="gene_id")
    parser.add_argument("--target_sum", type=float, default=10000.0)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--mask_seed", type=int, default=5678)
    parser.add_argument("--val_mask_seed", type=int, default=6789)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--save_tokenized_dataset", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = build_output_dir(args)
    pretrained_model_dir = ensure_pretrained_model_dir(args, out_dir)
    module_info = load_geneformer_modules(Path(args.geneformer_code_dir))
    tokenizer_files = module_info[args.model_version]
    TranscriptomeTokenizer = module_info["TranscriptomeTokenizer"]
    tokenizer_seq_len = resolve_tokenizer_seq_len(args.model_version, args.seq_len)

    print(
        f"[setup] h5ad_path={args.h5ad_path} model_version={args.model_version} "
        f"epochs={args.epochs} batch_size={args.batch_size} tokenizer_seq_len={tokenizer_seq_len}",
        flush=True,
    )
    print(f"[setup] pretrained_model_dir={pretrained_model_dir}", flush=True)
    print(f"[setup] output_dir={out_dir}", flush=True)

    adata = ad.read_h5ad(args.h5ad_path)
    use_index_files = bool(args.train_indices_path or args.val_indices_path)
    if use_index_files and not (args.train_indices_path and args.val_indices_path):
        raise ValueError("Pass both --train_indices_path and --val_indices_path, or neither.")
    if use_index_files:
        train_idx = np.load(args.train_indices_path).astype(np.int64)
        val_idx = np.load(args.val_indices_path).astype(np.int64)
        if train_idx.size == 0 or val_idx.size == 0:
            raise ValueError("train/val index files must be non-empty.")
        if train_idx.max(initial=-1) >= adata.n_obs or val_idx.max(initial=-1) >= adata.n_obs:
            raise ValueError("train/val index files contain indices outside the h5ad.")
        overlap = np.intersect1d(train_idx, val_idx)
        if overlap.size:
            raise ValueError(f"train/val index overlap is not allowed: {overlap.size} cells")
    else:
        train_idx, val_idx = split_train_val_indices(adata.n_obs, args.val_ratio, args.split_seed)
        train_idx = sample_indices(train_idx, args.train_total_cells, args.train_cell_seed)
        val_idx = sample_indices(val_idx, args.val_total_cells, args.val_cell_seed)
    train_adata = adata[train_idx].copy()
    val_adata = adata[val_idx].copy()

    train_adata = prepare_geneformer_adata(train_adata, args.gene_id_col, ())
    val_adata = prepare_geneformer_adata(val_adata, args.gene_id_col, ())
    ensure_obs_n_counts(train_adata)
    ensure_obs_n_counts(val_adata)
    ensure_unique_cell_ids(train_adata, "train")
    ensure_unique_cell_ids(val_adata, "val")

    tokenizer_metadata_cols = ["unique_cell_id"]
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={col: col for col in tokenizer_metadata_cols},
        nproc=args.nproc,
        chunk_size=args.chunk_size,
        model_input_size=tokenizer_seq_len,
        model_version=args.model_version,
        use_h5ad_index=False,
        keep_counts=False,
        gene_median_file=tokenizer_files["gene_median_file"],
        token_dictionary_file=tokenizer_files["token_dictionary_file"],
        gene_mapping_file=tokenizer_files["gene_mapping_file"],
    )

    tokenized_root = out_dir / "tokenized"
    tokenized_train_path = tokenized_root / "train.dataset" if args.save_tokenized_dataset else None
    tokenized_val_path = tokenized_root / "val.dataset" if args.save_tokenized_dataset else None

    print("[prep] tokenizing train subset with official Geneformer tokenizer", flush=True)
    tokenized_train = tokenize_adata_to_dataset(
        adata=train_adata,
        tokenizer=tokenizer,
        temp_h5ad_path=out_dir / "_tmp" / "train_subset.h5ad",
        metadata_cols=tokenizer_metadata_cols,
        target_sum=args.target_sum,
        save_to_disk_path=tokenized_train_path,
    )
    print("[prep] tokenizing val subset with official Geneformer tokenizer", flush=True)
    tokenized_val = tokenize_adata_to_dataset(
        adata=val_adata,
        tokenizer=tokenizer,
        temp_h5ad_path=out_dir / "_tmp" / "val_subset.h5ad",
        metadata_cols=tokenizer_metadata_cols,
        target_sum=args.target_sum,
        save_to_disk_path=tokenized_val_path,
    )
    print(
        f"[data] tokenized train cells={len(tokenized_train)} val cells={len(tokenized_val)} "
        f"train_max_len={max(tokenized_train['length']) if len(tokenized_train) else 0} "
        f"val_max_len={max(tokenized_val['length']) if len(tokenized_val) else 0}",
        flush=True,
    )

    model = GeneformerReconAnnotationModel(
        pretrained_model_dir=pretrained_model_dir,
        num_labels_by_task={},
        annotation_head_dropout=0.0,
        pooling=args.pooling,
        freeze_layers=args.freeze_layers,
        freeze_embeddings=args.freeze_embeddings,
    )
    vocab_size = int(model.backbone.config.vocab_size)
    pad_token_id = int(tokenizer.gene_token_dict["<pad>"])
    mask_token_id = int(tokenizer.gene_token_dict["<mask>"])
    cls_token_id = int(tokenizer.gene_token_dict.get("<cls>", -1))
    eos_token_id = int(tokenizer.gene_token_dict.get("<eos>", -1))
    special_token_ids = {pad_token_id, mask_token_id}
    if cls_token_id >= 0:
        special_token_ids.add(cls_token_id)
    if eos_token_id >= 0:
        special_token_ids.add(eos_token_id)

    print("[prep] building masked train examples", flush=True)
    train_examples = build_prepared_examples(
        dataset=tokenized_train,
        encoded_labels={},
        task_names=(),
        use_recon=True,
        mlm_probability=args.mlm_probability,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        special_token_ids=special_token_ids,
        seed=args.mask_seed,
    )
    print("[prep] building masked val examples", flush=True)
    val_examples = build_prepared_examples(
        dataset=tokenized_val,
        encoded_labels={},
        task_names=(),
        use_recon=True,
        mlm_probability=args.mlm_probability,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        special_token_ids=special_token_ids,
        seed=args.val_mask_seed,
    )

    train_loader = DataLoader(
        PreparedGeneformerDataset(train_examples, ()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate_fn((), pad_token_id=pad_token_id),
    )
    val_loader = DataLoader(
        PreparedGeneformerDataset(val_examples, ()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn((), pad_token_id=pad_token_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    history_path = out_dir / "history.jsonl"
    best_score = float("-inf")
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch] {epoch}/{args.epochs}", flush=True)
        t0 = time.time()
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            use_recon=True,
            use_annotation=False,
            task_names=(),
            mlm_loss_weight=1.0,
            annotation_loss_weight=0.0,
            grad_clip=args.grad_clip,
        )
        valid_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            use_recon=True,
            use_annotation=False,
            task_names=(),
        )
        metrics = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 2),
            **train_metrics,
            **valid_metrics,
        }
        with history_path.open("a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

        score = make_selection_score(metrics, "valid_mlm_loss", use_recon=True, use_annotation=False)
        if score > best_score:
            best_score = score
            best_metrics = dict(metrics)
            summary = {
                "dataset": "single h5ad Geneformer MLM reconstruction",
                "h5ad_path": str(args.h5ad_path),
                "pretrained_model_dir": str(pretrained_model_dir),
                "model_version": args.model_version,
                "tokenizer_seq_len": int(tokenizer_seq_len),
                "gene_id_col": args.gene_id_col,
                "train_cells": int(len(train_idx)),
                "val_cells": int(len(val_idx)),
                "tokenized_train_cells": int(len(tokenized_train)),
                "tokenized_val_cells": int(len(tokenized_val)),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "mlm_probability": float(args.mlm_probability),
                "best_metrics": best_metrics,
            }
            save_best_checkpoint(
                out_dir=out_dir,
                model=model,
                tokenizer_files=tokenizer_files,
                tokenizer_seq_len=tokenizer_seq_len,
                args=args,
                best_metrics=best_metrics,
                summary=summary,
            )
            print(f"[saved] best checkpoint updated at {out_dir}", flush=True)

    final_summary = {
        "output_dir": str(out_dir),
        "best_backbone": str(out_dir / "best_backbone"),
        "best_model_pt": str(out_dir / "best_model.pt"),
        "history_jsonl": str(history_path),
        "best_metrics": best_metrics,
        "train_cells": int(len(train_idx)),
        "val_cells": int(len(val_idx)),
        "tokenized_train_cells": int(len(tokenized_train)),
        "tokenized_val_cells": int(len(tokenized_val)),
    }
    (out_dir / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
