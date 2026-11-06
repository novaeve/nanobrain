import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
SCUNIFLOW_ROOT = next(p for p in THIS_FILE.parents if (p / "model.py").exists() and (p / "data.py").exists())
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SCUNIFLOW_ROOT))

from common import (  # noqa: E402
    derive_reference_labels,
    evaluate_clustering_metrics,
    materialize_downsampled_full_gene_counts,
    save_eval_outputs,
    set_seed,
    summarize_reconstruction,
)
from data import split_cells_train_val  # noqa: E402


DEFAULT_RUN_ROOT = SCUNIFLOW_ROOT / "eval" / "task2" / "read_depth_enhancement" / "runs"
DEFAULT_DATA_PATH = "data/pbmc68k_raw_counts.h5ad"
DEFAULT_CKPTS = {
    "scgpt": DEFAULT_RUN_ROOT / "pbmc68k_scgpt_rde_from_pbmc_recon_e3_fold4_12k_e10" / "best_model.pt",
    "geneformer": DEFAULT_RUN_ROOT / "pbmc68k_geneformer_rde_from_pbmc_mlm_e3_fold4_12k_e10" / "best_model.pt",
    "uce": DEFAULT_RUN_ROOT / "pbmc68k_uce_rde_fold4_12k_2k_e8" / "best_model.pt",
    "nicheformer": DEFAULT_RUN_ROOT / "pbmc68k_nicheformer_rde_fold4_12k_2k_e8" / "best_model.pt",
}


def get_matrix(adata, layer: str):
    if not layer:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer not found in adata.layers: {layer}")
    return adata.layers[layer]


def sample_indices(indices: np.ndarray, n_cells: int, seed: int):
    if n_cells is None or n_cells <= 0 or n_cells >= len(indices):
        return np.array(indices, copy=True)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(indices, size=int(n_cells), replace=False))


def namespace_from_dict(values: dict):
    return SimpleNamespace(**dict(values))


class TupleWithCellIdx(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
        self.indices = np.asarray(base.indices, dtype=np.int64)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return (*item, torch.tensor(int(self.indices[idx]), dtype=torch.long))


class DictWithCellIdx(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
        self.indices = np.asarray(base.indices, dtype=np.int64)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        item["cell_idx"] = torch.tensor(int(self.indices[idx]), dtype=torch.long)
        return item


def collate_uce_eval(batch):
    from train_uce_read_depth_enhancement import collate_uce

    cell_idx = torch.stack([x.pop("cell_idx") for x in batch])
    tokens, mask, target = collate_uce(batch)
    return tokens, mask, target, cell_idx


def resolve_eval_indices(n_obs: int, train_args: dict, args):
    if args.eval_index_mode == "full_random":
        train_idx, val_idx, _ = split_cells_train_val(
            n_obs,
            val_ratio=args.eval_val_ratio,
            seed=args.seed,
        )
        if args.split == "all":
            return np.arange(n_obs, dtype=np.int64)
        return np.asarray(train_idx if args.split == "train" else val_idx, dtype=np.int64)

    all_idx = np.arange(n_obs, dtype=np.int64)

    if int(train_args.get("max_cells", 0) or 0) > 0:
        max_cells = int(train_args["max_cells"])
        cell_seed = int(train_args.get("cell_seed", train_args.get("seed", args.seed)))
        keep = sample_indices(all_idx, max_cells, cell_seed)
        local_train, local_val, _ = split_cells_train_val(
            len(keep),
            val_ratio=float(train_args.get("val_ratio", args.eval_val_ratio)),
            seed=int(train_args.get("seed", args.seed)),
        )
        train_idx = keep[local_train]
        val_idx = keep[local_val]
    elif int(train_args.get("train_total_cells", 0) or 0) > 0 or int(train_args.get("val_total_cells", 0) or 0) > 0:
        train_idx, val_idx, _ = split_cells_train_val(
            n_obs,
            val_ratio=float(train_args.get("val_ratio", args.eval_val_ratio)),
            seed=int(train_args.get("split_seed", train_args.get("seed", args.seed))),
        )
        train_idx = sample_indices(
            train_idx,
            int(train_args.get("train_total_cells", 0) or 0),
            int(train_args.get("train_cell_seed", train_args.get("seed", args.seed))),
        )
        val_idx = sample_indices(
            val_idx,
            int(train_args.get("val_total_cells", 0) or 0),
            int(train_args.get("val_cell_seed", train_args.get("seed", args.seed))),
        )
    else:
        train_idx, val_idx, _ = split_cells_train_val(
            n_obs,
            val_ratio=float(train_args.get("val_ratio", args.eval_val_ratio)),
            seed=int(train_args.get("seed", args.seed)),
        )

    if args.split == "all":
        return np.sort(np.concatenate([train_idx, val_idx])).astype(np.int64)
    return np.asarray(train_idx if args.split == "train" else val_idx, dtype=np.int64)


def detect_scgpt_components(adata, matrix, eval_idx, train_args, ckpt, downsample_fold):
    import train_scgpt_read_depth_enhancement as scgpt_rde

    gene_col = scgpt_rde.detect_gene_col(adata, train_args.get("gene_col", "feature_name"))
    gene_names = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    model = scgpt_rde.ScGPTRDEModel(
        Path(train_args["finetuned_dir"]),
        adata.n_vars,
        bool(train_args.get("freeze_backbone", True)),
        int(train_args.get("head_hidden", 2048)),
        float(train_args.get("head_dropout", 0.1)),
        bool(train_args.get("use_fast_transformer", False)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    gene_mapping = scgpt_rde.build_gene_mapping(gene_names, model.vocab)
    dataset = scgpt_rde.ScGPTRDEDataset(
        matrix=matrix,
        indices=eval_idx,
        gene_mapping=gene_mapping,
        vocab=model.vocab,
        max_seq_len=int(train_args.get("max_seq_len", 1201)),
        n_bins=int(train_args.get("n_bins", 51)),
        downsample_fold=downsample_fold,
        seed=int(train_args.get("seed", 42)),
        pad_token="<pad>",
        cls_token="<cls>",
        pad_value=int(model.cfg.get("pad_value", -2)),
        input_space=train_args.get("input_space", "binned"),
    )
    return model, TupleWithCellIdx(dataset), None, gene_mapping


def detect_geneformer_components(adata, matrix, eval_idx, train_args, ckpt, downsample_fold):
    import train_geneformer_read_depth_enhancement as geneformer_rde

    code_dir = Path(train_args.get("geneformer_code_dir", geneformer_rde.DEFAULT_GENEFORMER_CODE_DIR))
    token_dict = geneformer_rde.load_pickle(code_dir / "geneformer" / "token_dictionary_gc104M.pkl")
    median_dict = geneformer_rde.load_pickle(code_dir / "geneformer" / "gene_median_dictionary_gc104M.pkl")
    mapping_dict = geneformer_rde.load_pickle(code_dir / "geneformer" / "ensembl_mapping_dict_gc104M.pkl")
    gene_col = geneformer_rde.detect_gene_col(adata, train_args.get("gene_id_col", "gene_id"))
    gene_ids = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    gene_mapping = geneformer_rde.build_gene_mapping(gene_ids, token_dict, median_dict, mapping_dict)
    model = geneformer_rde.GeneformerRDEModel(
        Path(train_args["finetuned_dir"]),
        adata.n_vars,
        bool(train_args.get("freeze_backbone", True)),
        int(train_args.get("head_hidden", 2048)),
        float(train_args.get("head_dropout", 0.1)),
        train_args.get("pooling", "cls"),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    dataset = geneformer_rde.GeneformerRDEDataset(
        matrix=matrix,
        indices=eval_idx,
        gene_mapping=gene_mapping,
        max_seq_len=int(train_args.get("seq_len", 4096)),
        downsample_fold=downsample_fold,
        seed=int(train_args.get("seed", 42)),
        pad_token_id=int(token_dict["<pad>"]),
        cls_token_id=int(token_dict["<cls>"]),
        eos_token_id=int(token_dict["<eos>"]),
        target_sum=float(train_args.get("target_sum", 10000.0)),
    )
    return model, TupleWithCellIdx(dataset), None, gene_mapping


def detect_uce_components(adata, matrix, eval_idx, train_args, ckpt, downsample_fold):
    import train_uce_read_depth_enhancement as uce_rde

    gene_col = uce_rde.detect_gene_col(adata, train_args.get("gene_col", "feature_name"))
    gene_names = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    uce_mapping = uce_rde.build_uce_gene_mapping(
        gene_names=gene_names,
        protein_embeddings_dir=Path(train_args["protein_embeddings_dir"]),
        species_chrom_csv_path=Path(train_args["species_chrom_csv_path"]),
        offset_pkl_path=Path(train_args["offset_pkl_path"]),
        species=train_args.get("species", "human"),
    )
    model = uce_rde.UCERDEModel(namespace_from_dict(train_args), output_dim=adata.n_vars)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    dataset = uce_rde.UCERDEDataset(
        matrix=matrix,
        indices=eval_idx,
        uce_mapping=uce_mapping,
        downsample_fold=downsample_fold,
        seed=int(train_args.get("seed", 42)),
        sample_size=int(train_args.get("sample_size", 1024)),
        pad_length=int(train_args.get("pad_length", 1536)),
        pad_token_idx=int(train_args.get("pad_token_idx", 0)),
        chrom_token_right_idx=int(train_args.get("chrom_token_right_idx", 2)),
        cls_token_idx=int(train_args.get("cls_token_idx", 3)),
        chrom_token_offset=int(train_args.get("chrom_token_offset", 143574)),
    )
    return model, DictWithCellIdx(dataset), collate_uce_eval, uce_mapping


def detect_nicheformer_components(adata, matrix, eval_idx, train_args, ckpt, downsample_fold):
    import train_nicheformer_read_depth_enhancement as niche_rde

    backbone_ckpt = torch.load(train_args["ckpt_path"], map_location="cpu", weights_only=False)
    n_tokens = int(backbone_ckpt["hyper_parameters"]["n_tokens"])
    symbol_token_map = niche_rde.load_symbol_token_map(Path(train_args["gene_map"]), n_tokens=n_tokens)
    gene_col = niche_rde.detect_gene_col(adata, train_args.get("gene_col", "feature_name"))
    gene_names = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    gene_mapping = niche_rde.build_gene_mapping(gene_names, symbol_token_map)
    model = niche_rde.NicheformerRDEModel(
        Path(train_args["ckpt_path"]),
        adata.n_vars,
        bool(train_args.get("freeze_encoder", True)),
        int(train_args.get("head_hidden", 2048)),
        float(train_args.get("head_dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    dataset = niche_rde.NicheformerRDEDataset(
        matrix=matrix,
        indices=eval_idx,
        gene_mapping=gene_mapping,
        downsample_fold=downsample_fold,
        seed=int(train_args.get("seed", 42)),
        context_length=int(train_args.get("context_length", 1500)),
        sample_size=int(train_args.get("sample_size", 1200)),
        pad_token_id=int(train_args.get("pad_token_id", 1)),
        cls_token_id=int(train_args.get("cls_token_id", 3)),
    )
    return model, TupleWithCellIdx(dataset), None, gene_mapping


def build_components(model_name, adata, matrix, eval_idx, train_args, ckpt, downsample_fold):
    builders = {
        "scgpt": detect_scgpt_components,
        "geneformer": detect_geneformer_components,
        "uce": detect_uce_components,
        "nicheformer": detect_nicheformer_components,
    }
    return builders[model_name](adata, matrix, eval_idx, train_args, ckpt, downsample_fold)


@torch.no_grad()
def run_prediction(model_name, model, loader, device, amp: bool, max_batches: int = 0):
    model.eval()
    preds, targets, cell_indices = [], [], []
    n_batches = 0
    for batch in tqdm(loader, desc=f"eval {model_name}", unit="batch"):
        if model_name == "scgpt":
            genes, values, target, cell_idx = batch
            genes = genes.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            with autocast(enabled=amp):
                pred = model(genes, values)
        elif model_name == "geneformer":
            input_ids, attention_mask, target, cell_idx = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            with autocast(enabled=amp):
                pred = model(input_ids, attention_mask)
        elif model_name == "uce":
            tokens, mask, target, cell_idx = batch
            tokens = tokens.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with autocast(enabled=amp):
                pred = model(tokens, mask)
        elif model_name == "nicheformer":
            tokens, target, cell_idx = batch
            tokens = tokens.to(device, non_blocking=True)
            with autocast(enabled=amp):
                pred = model(tokens)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        preds.append(pred.detach().cpu().float().numpy())
        targets.append(target.cpu().float().numpy())
        cell_indices.append(cell_idx.cpu().numpy())
        n_batches += 1
        if max_batches > 0 and n_batches >= max_batches:
            break

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(cell_indices, axis=0).astype(np.int64),
        n_batches,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline RDE checkpoints with scFoundation/scUniFlow-style metrics.")
    parser.add_argument("--model", choices=["scgpt", "geneformer", "uce", "nicheformer"], required=True)
    parser.add_argument("--ckpt_path", default="", help="RDE checkpoint; defaults to the known PBMC run for --model")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--layer", default="")
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--cell_indices_path", default="", help="optional .npy file with exact original h5ad cell indices")
    parser.add_argument(
        "--eval_index_mode",
        choices=["full_random", "train_script"],
        default="full_random",
        help="full_random reproduces the old full-PBMC random split; train_script reconstructs the RDE script's own split",
    )
    parser.add_argument("--eval_max_cells", type=int, default=0, help="0 means full h5ad")
    parser.add_argument("--eval_val_ratio", type=float, default=0.1)
    parser.add_argument("--eval_cell_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downsample_fold", type=float, default=-1.0, help="negative means reuse checkpoint arg")
    parser.add_argument("--batch_size", type=int, default=0, help="0 means reuse checkpoint arg")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--label_col", default="")
    parser.add_argument("--target_labels_path", default="", help="optional .npy labels aligned to evaluated cells")
    parser.add_argument("--ref_resolution", type=float, default=0.4)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--skip_clustering", action="store_true", help="only compute reconstruction metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_arrays", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.ckpt_path or DEFAULT_CKPTS[args.model]).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}))
    if not train_args:
        raise KeyError(f"Checkpoint has no saved args: {ckpt_path}")

    downsample_fold = float(args.downsample_fold if args.downsample_fold >= 0 else train_args.get("downsample_fold", 4.0))
    batch_size = int(args.batch_size or train_args.get("batch_size", 8))
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    adata = ad.read_h5ad(args.data_path)
    if args.cell_indices_path and args.eval_max_cells > 0:
        raise ValueError("--cell_indices_path expects original h5ad indices; do not combine it with --eval_max_cells")
    if not args.cell_indices_path and args.eval_max_cells > 0 and adata.n_obs > args.eval_max_cells:
        keep = sample_indices(np.arange(adata.n_obs), args.eval_max_cells, args.eval_cell_seed)
        adata = adata[keep].copy()

    matrix = get_matrix(adata, args.layer or train_args.get("layer", ""))
    if args.cell_indices_path:
        eval_idx = np.load(args.cell_indices_path).astype(np.int64)
    else:
        eval_idx = resolve_eval_indices(adata.n_obs, train_args, args)

    model, dataset, collate_fn, gene_mapping = build_components(
        args.model,
        adata,
        matrix,
        eval_idx,
        train_args,
        ckpt,
        downsample_fold,
    )
    model = model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    t0 = time.perf_counter()
    preds, targets, used_cell_idx, used_batches = run_prediction(
        args.model,
        model,
        loader,
        device,
        amp=amp,
        max_batches=int(args.max_batches),
    )
    print(f"[timing] prediction={time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    recon_metrics, cell_pccs, gene_pccs = summarize_reconstruction(preds, targets)
    print(f"[timing] reconstruction_metrics={time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    full_low_counts, full_high_counts = materialize_downsampled_full_gene_counts(
        matrix,
        used_cell_idx,
        downsample_fold=downsample_fold,
        seed=int(train_args.get("seed", args.seed)),
    )
    print(f"[timing] materialize_low_high={time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    pred_norm = np.expm1(preds).clip(min=0.0)
    low_totals = full_low_counts.sum(axis=1, keepdims=True)
    est_high_totals = np.maximum(low_totals * float(downsample_fold), 1.0)
    enhanced_full_counts = (pred_norm / 1e4) * est_high_totals
    print(f"[timing] build_enhanced_counts={time.perf_counter() - t0:.2f}s", flush=True)

    clustering = None
    if args.skip_clustering:
        target_labels = np.asarray([], dtype=str)
        enhanced_clusters = np.asarray([], dtype=str)
        low_clusters = np.asarray([], dtype=str)
    elif args.target_labels_path:
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
        if args.label_col not in adata.obs.columns:
            raise KeyError(f"label_col not found in obs: {args.label_col}")
        target_labels = adata.obs.iloc[used_cell_idx][args.label_col].astype(str).to_numpy()
        n_clusters = len(np.unique(target_labels))
        reference_info = {"target_mode": "obs_label", "label_col": args.label_col, "n_clusters": int(n_clusters)}
    else:
        t0 = time.perf_counter()
        target_labels, _ = derive_reference_labels(
            full_high_counts,
            ref_resolution=float(args.ref_resolution),
            n_top_genes=int(args.n_top_genes),
        )
        print(f"[timing] reference_labels={time.perf_counter() - t0:.2f}s", flush=True)
        n_clusters = len(np.unique(target_labels))
        reference_info = {
            "target_mode": "reference_leiden",
            "ref_resolution": float(args.ref_resolution),
            "n_clusters": int(n_clusters),
        }

    if not args.skip_clustering:
        t0 = time.perf_counter()
        enhanced_metrics, _, enhanced_clusters = evaluate_clustering_metrics(
            enhanced_full_counts,
            target_labels=target_labels,
            n_clusters=n_clusters,
            n_top_genes=int(args.n_top_genes),
        )
        print(f"[timing] enhanced_clustering={time.perf_counter() - t0:.2f}s", flush=True)
        t0 = time.perf_counter()
        low_metrics, _, low_clusters = evaluate_clustering_metrics(
            full_low_counts,
            target_labels=target_labels,
            n_clusters=n_clusters,
            n_top_genes=int(args.n_top_genes),
        )
        print(f"[timing] low_clustering={time.perf_counter() - t0:.2f}s", flush=True)
        clustering = {
            "reference": reference_info,
            "low_depth_baseline": low_metrics,
            "enhanced": enhanced_metrics,
        }

    cell_ids = adata.obs_names.astype(str).to_numpy()[used_cell_idx].tolist()
    metrics = dict(recon_metrics)
    metrics.update(
        {
            "model": args.model,
            "checkpoint": str(ckpt_path),
            "data_path": str(Path(args.data_path).resolve()),
            "split": args.split,
            "n_eval_cells": int(preds.shape[0]),
            "n_genes": int(full_high_counts.shape[1]),
            "n_model_genes": int(preds.shape[1]),
            "n_mapped_genes": int(gene_mapping.get("n_mapped_genes", -1)),
            "used_batches": int(used_batches),
            "batch_size": int(batch_size),
            "max_batches": int(args.max_batches),
            "downsample_fold": float(downsample_fold),
            "eval_max_cells": int(args.eval_max_cells),
            "eval_val_ratio": float(args.eval_val_ratio),
            "eval_cell_seed": int(args.eval_cell_seed),
            "eval_index_mode": args.eval_index_mode,
            "cell_indices_path": str(Path(args.cell_indices_path).resolve()) if args.cell_indices_path else "",
            "target_labels_path": str(Path(args.target_labels_path).resolve()) if args.target_labels_path else "",
            "clustering_gene_space": "full_gene_scuniflow_protocol",
            "reconstruction_gene_space": "full_gene_model_output",
            "train_best_metrics": ckpt.get("best_metrics", {}),
            "clustering": clustering,
        }
    )

    save_eval_outputs(save_dir, metrics, cell_ids, cell_pccs, gene_pccs)
    np.save(save_dir / "target_labels.npy", np.asarray(target_labels))
    np.save(save_dir / "enhanced_clusters.npy", np.asarray(enhanced_clusters))
    np.save(save_dir / "low_clusters.npy", np.asarray(low_clusters))

    if args.save_arrays:
        np.save(save_dir / "preds.npy", preds)
        np.save(save_dir / "targets.npy", targets)
        np.save(save_dir / "full_low_counts.npy", full_low_counts)
        np.save(save_dir / "full_high_counts.npy", full_high_counts)
        np.save(save_dir / "enhanced_full_counts.npy", enhanced_full_counts)

    with (save_dir / "run_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {save_dir}")


if __name__ == "__main__":
    main()
