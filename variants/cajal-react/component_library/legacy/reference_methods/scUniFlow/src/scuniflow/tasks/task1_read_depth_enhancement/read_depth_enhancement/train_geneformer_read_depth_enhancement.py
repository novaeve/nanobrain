import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import issparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertModel


DEFAULT_FINETUNED_DIR = Path(
    "./src/scuniflow/baselines/finetune/outputs/Geneformer/"
    "pbmc68k_geneformer_v2_mlm_12k_2k_e3"
)
DEFAULT_GENEFORMER_CODE_DIR = Path("external/Geneformer")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_gene_col(adata, requested: str):
    if requested and requested != "auto":
        if requested == "index":
            return "index"
        if requested not in adata.var.columns:
            raise KeyError(f"Gene column not found in adata.var: {requested}")
        return requested
    for candidate in ("gene_id", "gene_ids", "ensembl_id"):
        if candidate in adata.var.columns:
            return candidate
    return "index"


def get_matrix(adata, layer: str):
    if not layer:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer not found in adata.layers: {layer}")
    return adata.layers[layer]


def row_to_dense(matrix, row_idx: int):
    row = matrix.getrow(row_idx).toarray().ravel() if issparse(matrix) else np.asarray(matrix[row_idx]).ravel()
    return np.asarray(row, dtype=np.float32)


def log_norm(counts):
    counts = np.asarray(counts, dtype=np.float32)
    total = float(counts.sum())
    return np.log1p(counts / (total + 1e-6) * 1e4).astype(np.float32)


def split_train_val_indices(n_obs: int, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_obs)
    n_val = max(1, int(round(n_obs * float(val_ratio))))
    n_val = min(n_val, n_obs - 1)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def sample_indices(indices, n_cells: int, seed: int):
    if n_cells is None or n_cells <= 0 or n_cells >= len(indices):
        return np.array(indices, copy=True)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(indices, size=int(n_cells), replace=False))


def load_pickle(path: Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def build_gene_mapping(gene_ids, token_dict, median_dict, mapping_dict):
    source_cols = []
    token_ids = []
    medians = []
    for idx, gene in enumerate(gene_ids):
        gene = str(gene)
        mapped = gene if gene in token_dict else mapping_dict.get(gene.upper(), mapping_dict.get(gene, None))
        if mapped is None or mapped not in token_dict or mapped not in median_dict:
            continue
        source_cols.append(idx)
        token_ids.append(int(token_dict[mapped]))
        medians.append(float(median_dict[mapped]))
    if not source_cols:
        raise RuntimeError("No PBMC genes overlap with Geneformer token/median dictionaries.")
    return {
        "source_cols": np.asarray(source_cols, dtype=np.int64),
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "medians": np.asarray(medians, dtype=np.float32),
        "n_input_genes": int(len(gene_ids)),
        "n_mapped_genes": int(len(source_cols)),
    }


class GeneformerRDEDataset(Dataset):
    def __init__(
        self,
        matrix,
        indices,
        gene_mapping,
        max_seq_len: int,
        downsample_fold: float,
        seed: int,
        pad_token_id: int,
        cls_token_id: int,
        eos_token_id: int,
        target_sum: float,
    ):
        self.matrix = matrix
        self.indices = np.asarray(indices, dtype=np.int64)
        self.map = gene_mapping
        self.max_seq_len = int(max_seq_len)
        self.keep_prob = 1.0 / max(float(downsample_fold), 1.0)
        self.seed = int(seed)
        self.pad_token_id = int(pad_token_id)
        self.cls_token_id = int(cls_token_id)
        self.eos_token_id = int(eos_token_id)
        self.target_sum = float(target_sum)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cell_idx = int(self.indices[idx])
        high_raw = row_to_dense(self.matrix, cell_idx)
        high_counts = np.rint(np.clip(high_raw, 0.0, None)).astype(np.int64)
        rng = np.random.RandomState(self.seed + cell_idx)
        low_counts = rng.binomial(high_counts, self.keep_prob).astype(np.float32)

        sub = low_counts[self.map["source_cols"]]
        total = float(sub.sum())
        scaled = sub / (total + 1e-6) * self.target_sum
        ranked_values = scaled / np.clip(self.map["medians"], 1e-12, None)
        nz = np.nonzero(ranked_values > 0)[0]
        if nz.size:
            order = nz[np.argsort(-ranked_values[nz])]
            room = max(0, self.max_seq_len - 2)
            order = order[:room]
            tokens = self.map["token_ids"][order]
            input_ids = np.concatenate([[self.cls_token_id], tokens, [self.eos_token_id]]).astype(np.int64)
        else:
            input_ids = np.asarray([self.cls_token_id, self.eos_token_id], dtype=np.int64)
        attention = np.ones(len(input_ids), dtype=np.int64)
        if len(input_ids) < self.max_seq_len:
            pad_n = self.max_seq_len - len(input_ids)
            input_ids = np.concatenate([input_ids, np.full(pad_n, self.pad_token_id, dtype=np.int64)])
            attention = np.concatenate([attention, np.zeros(pad_n, dtype=np.int64)])
        return torch.from_numpy(input_ids).long(), torch.from_numpy(attention).long(), torch.from_numpy(log_norm(high_counts))


class GeneformerRDEModel(nn.Module):
    def __init__(self, finetuned_dir: Path, output_dim: int, freeze_backbone: bool, head_hidden: int, head_dropout: float, pooling: str):
        super().__init__()
        self.backbone = BertModel.from_pretrained(str(finetuned_dir / "best_backbone"))
        self.pooling = pooling
        hidden = int(self.backbone.config.hidden_size)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, output_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def train_one_epoch(model, loader, optimizer, scaler, device, amp, grad_clip):
    model.train()
    total = 0.0
    n = 0
    for input_ids, attention_mask, target in tqdm(loader, desc="train", unit="batch"):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            pred = model(input_ids, attention_mask)
            loss = F.mse_loss(pred, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item()) * input_ids.size(0)
        n += input_ids.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    total = 0.0
    n = 0
    for input_ids, attention_mask, target in tqdm(loader, desc="valid", unit="batch"):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(enabled=amp):
            pred = model(input_ids, attention_mask)
            loss = F.mse_loss(pred, target)
        total += float(loss.item()) * input_ids.size(0)
        n += input_ids.size(0)
    return total / max(n, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Geneformer RDE head: low-depth PBMC -> high-depth expression.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--finetuned_dir", default=str(DEFAULT_FINETUNED_DIR))
    parser.add_argument("--geneformer_code_dir", default=str(DEFAULT_GENEFORMER_CODE_DIR))
    parser.add_argument("--gene_id_col", default="gene_id")
    parser.add_argument("--layer", default="")
    parser.add_argument("--max_cells", type=int, default=12000)
    parser.add_argument("--train_indices_path", default="", help="optional .npy original h5ad cell indices for training")
    parser.add_argument("--val_indices_path", default="", help="optional .npy original h5ad cell indices for validation")
    parser.add_argument("--val_ratio", type=float, default=0.142857)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell_seed", type=int, default=42)
    parser.add_argument("--downsample_fold", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--target_sum", type=float, default=10000.0)
    parser.add_argument("--head_hidden", type=int, default=2048)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    print(f"[setup] device={device} amp={amp} save_dir={save_dir}", flush=True)

    code_dir = Path(args.geneformer_code_dir)
    token_dict = load_pickle(code_dir / "geneformer" / "token_dictionary_gc104M.pkl")
    median_dict = load_pickle(code_dir / "geneformer" / "gene_median_dictionary_gc104M.pkl")
    mapping_dict = load_pickle(code_dir / "geneformer" / "ensembl_mapping_dict_gc104M.pkl")

    adata = ad.read_h5ad(args.data_path)
    use_index_files = bool(args.train_indices_path or args.val_indices_path)
    if use_index_files and not (args.train_indices_path and args.val_indices_path):
        raise ValueError("Pass both --train_indices_path and --val_indices_path, or neither.")
    if not use_index_files and args.max_cells > 0 and adata.n_obs > args.max_cells:
        keep = sample_indices(np.arange(adata.n_obs), args.max_cells, args.cell_seed)
        adata = adata[keep].copy()
    gene_col = detect_gene_col(adata, args.gene_id_col)
    gene_ids = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    matrix = get_matrix(adata, args.layer)
    gene_mapping = build_gene_mapping(gene_ids, token_dict, median_dict, mapping_dict)
    print(f"[data] cells={adata.n_obs} genes={adata.n_vars} mapped={gene_mapping['n_mapped_genes']}/{gene_mapping['n_input_genes']}", flush=True)
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
        train_idx, val_idx = split_train_val_indices(adata.n_obs, args.val_ratio, args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)}", flush=True)

    model = GeneformerRDEModel(Path(args.finetuned_dir), adata.n_vars, args.freeze_backbone, args.head_hidden, args.head_dropout, args.pooling).to(device)
    ds_kwargs = dict(
        matrix=matrix,
        gene_mapping=gene_mapping,
        max_seq_len=args.seq_len,
        downsample_fold=args.downsample_fold,
        seed=args.seed,
        pad_token_id=int(token_dict["<pad>"]),
        cls_token_id=int(token_dict["<cls>"]),
        eos_token_id=int(token_dict["<eos>"]),
        target_sum=args.target_sum,
    )
    train_ds = GeneformerRDEDataset(indices=train_idx, **ds_kwargs)
    val_ds = GeneformerRDEDataset(indices=val_idx, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    param_groups.append({"params": head_params, "lr": args.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=amp)

    best_val = float("inf")
    best_metrics = {}
    history_path = save_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        print(f"[epoch] {epoch}/{args.epochs}", flush=True)
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, amp, args.grad_clip)
        val_loss = evaluate(model, val_loader, device, amp)
        metrics = {"epoch": epoch, "seconds": round(time.time() - t0, 2), "train_recon_loss": float(train_loss), "valid_recon_loss": float(val_loss)}
        with history_path.open("a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_metrics = dict(metrics)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "gene_mapping": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gene_mapping.items()},
                    "best_metrics": best_metrics,
                },
                save_dir / "best_model.pt",
            )
            (save_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n")
            print(f"[saved] best checkpoint updated: {save_dir / 'best_model.pt'}", flush=True)

    summary = {"output_dir": str(save_dir), "best_model_pt": str(save_dir / "best_model.pt"), "best_metrics": best_metrics, "train_cells": int(len(train_idx)), "val_cells": int(len(val_idx)), "mapped_genes": int(gene_mapping["n_mapped_genes"]), "output_genes": int(adata.n_vars)}
    (save_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
