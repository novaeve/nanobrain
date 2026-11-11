import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

THIS_FILE = Path(__file__).resolve()
SCUNIFLOW_ROOT = next(p for p in THIS_FILE.parents if (p / "model.py").exists() and (p / "data.py").exists())
WORKSPACE_ROOT = SCUNIFLOW_ROOT.parent
CELLFM_ROOT = WORKSPACE_ROOT / "CellFM"
sys.path.insert(0, str(CELLFM_ROOT))

import mindspore as ms  # noqa: E402
import mindspore.nn as nn  # noqa: E402
import mindspore.ops as ops  # noqa: E402
from config import Config  # noqa: E402
from model import CellFM  # noqa: E402


DEFAULT_CKPT = Path(
    "external/model_zoo/"
    "4014c514594e479d8fd84d2c65dd94fc/Research/base_weight.ckpt"
)
DEFAULT_GENE_MAP = CELLFM_ROOT / "csv" / "gene_info.csv"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ms.set_seed(seed)


def detect_gene_col(adata, requested: str):
    if requested and requested != "auto":
        if requested == "index":
            return "index"
        if requested not in adata.var.columns:
            raise KeyError(f"Gene column not found in adata.var: {requested}")
        return requested
    for candidate in ("feature_name", "gene_name", "gene_symbols"):
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


def load_cellfm_symbol_map(gene_map_path: Path):
    df = pd.read_csv(gene_map_path)
    if "idx" in df.columns:
        gene_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
        return {str(g).upper(): int(i) for g, i in zip(df[gene_col], df["idx"])}
    gene_col = "HGNC_gene" if "HGNC_gene" in df.columns else df.columns[0]
    return {str(g).upper(): int(i + 1) for i, g in enumerate(df[gene_col].astype(str))}


def build_gene_mapping(gene_names, symbol_map):
    source_cols = []
    token_ids = []
    for idx, gene in enumerate(gene_names):
        token_id = symbol_map.get(str(gene).upper())
        if token_id is None:
            continue
        source_cols.append(idx)
        token_ids.append(token_id)
    if not source_cols:
        raise RuntimeError("No PBMC genes could be mapped to CellFM gene ids.")
    return {
        "source_cols": np.asarray(source_cols, dtype=np.int64),
        "token_ids": np.asarray(token_ids, dtype=np.int32),
        "n_input_genes": int(len(gene_names)),
        "n_mapped_genes": int(len(source_cols)),
        "n_cellfm_genes": int(max(symbol_map.values())),
    }


class CellFMRDEDataset(Dataset):
    def __init__(
        self,
        matrix,
        indices,
        gene_mapping,
        downsample_fold: float,
        seed: int,
        nonz_len: int,
    ):
        self.matrix = matrix
        self.indices = np.asarray(indices, dtype=np.int64)
        self.map = gene_mapping
        self.keep_prob = 1.0 / max(float(downsample_fold), 1.0)
        self.seed = int(seed)
        self.nonz_len = int(nonz_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cell_idx = int(self.indices[idx])
        high_raw = row_to_dense(self.matrix, cell_idx)
        high_counts = np.rint(np.clip(high_raw, 0.0, None)).astype(np.int64)
        rng = np.random.RandomState(self.seed + cell_idx)
        low_counts = rng.binomial(high_counts, self.keep_prob).astype(np.float32)

        cols = self.map["source_cols"]
        low_sub = np.asarray(low_counts[cols], dtype=np.float32)
        nonzero = np.where(low_sub > 0)[0]
        if nonzero.size > self.nonz_len:
            weights = np.log1p(low_sub[nonzero])
            weights = weights / weights.sum()
            nonzero = np.sort(rng.choice(nonzero, size=self.nonz_len, replace=False, p=weights))

        expr = np.zeros(self.nonz_len, dtype=np.float32)
        gene = np.zeros(self.nonz_len, dtype=np.int32)
        zero_idx = np.zeros(self.nonz_len + 1, dtype=np.float32)
        zero_idx[0] = 1.0
        if nonzero.size:
            values = log_norm(low_counts)[cols[nonzero]]
            n = min(self.nonz_len, nonzero.size)
            expr[:n] = values[:n]
            gene[:n] = self.map["token_ids"][nonzero[:n]]
            zero_idx[1: 1 + n] = 1.0

        target = log_norm(high_counts)
        return expr, gene, zero_idx, target


class CellFMRDE(nn.Cell):
    def __init__(self, n_cellfm_genes: int, output_dim: int, cfg, freeze_encoder: bool, head_hidden: int):
        super().__init__()
        self.backbone = CellFM(n_cellfm_genes, cfg)
        if freeze_encoder:
            for p in self.backbone.trainable_params():
                p.requires_grad = False
        self.head = nn.SequentialCell(
            nn.LayerNorm((cfg.enc_dims,)),
            nn.Dense(cfg.enc_dims, head_hidden),
            nn.GELU(),
            nn.Dense(head_hidden, output_dim),
        )

    def construct(self, expr, gene, zero_idx):
        _, _, cls_token = self.backbone.forward(expr, gene, zero_idx)
        return self.head(cls_token.astype(ms.float32))


class MSELossCell(nn.Cell):
    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network
        self.loss = nn.MSELoss()

    def construct(self, expr, gene, zero_idx, target):
        pred = self.network(expr, gene, zero_idx)
        return self.loss(pred, target)


def to_ms(batch):
    expr, gene, zero_idx, target = batch
    return (
        ms.Tensor(expr.numpy(), ms.float32),
        ms.Tensor(gene.numpy(), ms.int32),
        ms.Tensor(zero_idx.numpy(), ms.float32),
        ms.Tensor(target.numpy(), ms.float32),
    )


def train_one_epoch(model, loss_net, optimizer, loader):
    grad_fn = ms.value_and_grad(loss_net, None, optimizer.parameters)
    model.set_train(True)
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", unit="batch"):
        expr, gene, zero_idx, target = to_ms(batch)
        loss, grads = grad_fn(expr, gene, zero_idx, target)
        optimizer(grads)
        bs = int(expr.shape[0])
        total += float(loss.asnumpy()) * bs
        n += bs
    return total / max(n, 1)


def evaluate(model, loader):
    model.set_train(False)
    mse = nn.MSELoss()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="valid", unit="batch"):
        expr, gene, zero_idx, target = to_ms(batch)
        pred = model(expr, gene, zero_idx)
        loss = mse(pred, target)
        bs = int(expr.shape[0])
        total += float(loss.asnumpy()) * bs
        n += bs
    return total / max(n, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CellFM + RDE head on PBMC read-depth enhancement.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT))
    parser.add_argument("--gene_map", default=str(DEFAULT_GENE_MAP))
    parser.add_argument("--gene_col", default="feature_name")
    parser.add_argument("--layer", default="")
    parser.add_argument("--device_target", default="CPU", choices=["CPU", "GPU", "Ascend"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--train_total_cells", type=int, default=12000)
    parser.add_argument("--val_total_cells", type=int, default=2000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_cell_seed", type=int, default=3456)
    parser.add_argument("--val_cell_seed", type=int, default=4567)
    parser.add_argument("--downsample_fold", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--nonz_len", type=int, default=2048)
    parser.add_argument("--head_hidden", type=int, default=2048)
    parser.add_argument("--enc_dims", type=int, default=1536)
    parser.add_argument("--enc_nlayers", type=int, default=40)
    parser.add_argument("--enc_num_heads", type=int, default=48)
    return parser.parse_args()


def main():
    args = parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")

    adata = ad.read_h5ad(args.data_path)
    gene_col = detect_gene_col(adata, args.gene_col)
    gene_names = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    matrix = get_matrix(adata, args.layer)
    symbol_map = load_cellfm_symbol_map(Path(args.gene_map))
    gene_mapping = build_gene_mapping(gene_names, symbol_map)
    print(f"[data] cells={adata.n_obs} genes={adata.n_vars} mapped={gene_mapping['n_mapped_genes']}/{gene_mapping['n_input_genes']}", flush=True)

    train_idx, val_idx = split_train_val_indices(adata.n_obs, args.val_ratio, args.split_seed)
    train_idx = sample_indices(train_idx, args.train_total_cells, args.train_cell_seed)
    val_idx = sample_indices(val_idx, args.val_total_cells, args.val_cell_seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)}", flush=True)

    ds_kwargs = dict(matrix=matrix, gene_mapping=gene_mapping, downsample_fold=args.downsample_fold, seed=args.seed, nonz_len=args.nonz_len)
    train_ds = CellFMRDEDataset(indices=train_idx, **ds_kwargs)
    val_ds = CellFMRDEDataset(indices=val_idx, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cfg = Config()
    cfg.enc_dims = args.enc_dims
    cfg.enc_nlayers = args.enc_nlayers
    cfg.enc_num_heads = args.enc_num_heads
    cfg.nonz_len = args.nonz_len
    cfg.add_zero = False
    cfg.pad_zero = True
    cfg.label = False

    model = CellFMRDE(gene_mapping["n_cellfm_genes"], adata.n_vars, cfg, args.freeze_encoder, args.head_hidden)
    params = ms.load_checkpoint(str(args.ckpt_path))
    prefixed_params = dict(params)
    prefixed_params.update({f"backbone.{k}": v for k, v in params.items()})
    not_loaded, _ = ms.load_param_into_net(model, prefixed_params, strict_load=False)
    print(f"[model] loaded CellFM checkpoint, not_loaded={len(not_loaded)}", flush=True)

    trainable = model.trainable_params()
    optimizer = nn.AdamWeightDecay(trainable, learning_rate=args.lr, weight_decay=args.weight_decay)
    loss_net = MSELossCell(model)

    best_val = float("inf")
    best_metrics = {}
    history_path = save_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        print(f"[epoch] {epoch}/{args.epochs}", flush=True)
        t0 = time.time()
        train_loss = train_one_epoch(model, loss_net, optimizer, train_loader)
        val_loss = evaluate(model, val_loader)
        metrics = {"epoch": epoch, "seconds": round(time.time() - t0, 2), "train_recon_loss": float(train_loss), "valid_recon_loss": float(val_loss)}
        with history_path.open("a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_metrics = dict(metrics)
            ms.save_checkpoint(model, str(save_dir / "best_model.ckpt"))
            (save_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n")
            print(f"[saved] best checkpoint updated: {save_dir / 'best_model.ckpt'}", flush=True)

    summary = {
        "output_dir": str(save_dir),
        "best_model_ckpt": str(save_dir / "best_model.ckpt"),
        "best_metrics": best_metrics,
        "train_cells": int(len(train_idx)),
        "val_cells": int(len(val_idx)),
        "mapped_genes": int(gene_mapping["n_mapped_genes"]),
        "output_genes": int(adata.n_vars),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
