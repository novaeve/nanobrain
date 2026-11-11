import argparse
import json
import os
import random
import sys
import time
import types
import inspect
from types import SimpleNamespace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import issparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

THIS_FILE = Path(__file__).resolve()
SCUNIFLOW_ROOT = next(p for p in THIS_FILE.parents if (p / "model.py").exists() and (p / "data.py").exists())
WORKSPACE_ROOT = SCUNIFLOW_ROOT.parent
NICHEFORMER_SRC = WORKSPACE_ROOT / "nicheformer" / "src"
sys.path.insert(0, str(NICHEFORMER_SRC))

import importlib.metadata as importlib_metadata  # noqa: E402

_orig_version = importlib_metadata.version


def _local_version(name):
    if name == "nicheformer":
        return "local"
    return _orig_version(name)


importlib_metadata.version = _local_version


try:
    import pytorch_lightning  # noqa: F401
except ModuleNotFoundError:
    pl_stub = types.ModuleType("pytorch_lightning")

    class _LightningModule(nn.Module):
        def save_hyperparameters(self, *args, ignore=None, **kwargs):
            ignore = set(ignore or [])
            frame = inspect.currentframe().f_back
            params = {
                k: v
                for k, v in frame.f_locals.items()
                if k != "self" and k not in ignore
            }
            params.update(kwargs)
            self.hparams = SimpleNamespace(**params)

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, map_location="cpu", **kwargs):
            ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            hparams = dict(ckpt.get("hyper_parameters", {}))
            hparams.update(kwargs)
            allowed = {
                k
                for k in inspect.signature(cls.__init__).parameters
                if k != "self"
            }
            init_kwargs = {k: v for k, v in hparams.items() if k in allowed}
            model = cls(**init_kwargs)
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(
                    f"[nicheformer] loaded checkpoint with "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
            return model

        def log(self, *args, **kwargs):
            return None

        def freeze(self):
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def _seed_everything(seed, *args, **kwargs):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    pl_stub.LightningModule = _LightningModule
    pl_stub.seed_everything = _seed_everything
    sys.modules["pytorch_lightning"] = pl_stub

from nicheformer.models._nicheformer import Nicheformer  # noqa: E402


DEFAULT_CKPT = Path(
    "external/model_zoo/"
    "4014c514594e479d8fd84d2c65dd94fc/Research/nicheformer.ckpt"
)
DEFAULT_GENE_MAP = Path("external/CellFM/csv/genemap.csv")


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


def load_symbol_token_map(gene_map_path: Path, n_tokens: int):
    df = pd.read_csv(gene_map_path)
    gene_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    id_col = "idx" if "idx" in df.columns else df.columns[1]
    mapping = {}
    for gene, idx in zip(df[gene_col].astype(str), df[id_col].astype(int)):
        # Nicheformer reserves low token ids for special/context tokens. The
        # public checkpoint in this workspace does not include its original gene
        # dictionary, so we reuse the local symbol map and avoid reserved ids.
        token_id = int(idx)
        if 10 <= token_id < int(n_tokens):
            mapping[gene.upper()] = token_id
    return mapping


def build_gene_mapping(gene_names, symbol_token_map):
    source_cols = []
    token_ids = []
    for idx, gene in enumerate(gene_names):
        token_id = symbol_token_map.get(str(gene).upper())
        if token_id is None:
            continue
        source_cols.append(idx)
        token_ids.append(token_id)
    if not source_cols:
        raise RuntimeError("No PBMC genes could be mapped to Nicheformer token ids.")
    return {
        "source_cols": np.asarray(source_cols, dtype=np.int64),
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "n_input_genes": int(len(gene_names)),
        "n_mapped_genes": int(len(source_cols)),
    }


class NicheformerRDEDataset(Dataset):
    def __init__(
        self,
        matrix,
        indices,
        gene_mapping,
        downsample_fold: float,
        seed: int,
        context_length: int,
        sample_size: int,
        pad_token_id: int,
        cls_token_id: int,
    ):
        self.matrix = matrix
        self.indices = np.asarray(indices, dtype=np.int64)
        self.map = gene_mapping
        self.keep_prob = 1.0 / max(float(downsample_fold), 1.0)
        self.seed = int(seed)
        self.context_length = int(context_length)
        self.sample_size = int(sample_size)
        self.pad_token_id = int(pad_token_id)
        self.cls_token_id = int(cls_token_id)

    def __len__(self):
        return len(self.indices)

    def _make_tokens(self, low_counts, rng):
        cols = self.map["source_cols"]
        expr = np.asarray(low_counts[cols], dtype=np.float64)
        nonzero = np.where(expr > 0)[0]
        if nonzero.size == 0:
            chosen = rng.choice(np.arange(len(cols)), size=min(self.sample_size, len(cols)), replace=False)
        else:
            weights = np.log1p(expr[nonzero])
            weights = weights / weights.sum()
            n_take = min(self.sample_size, nonzero.size)
            chosen = rng.choice(nonzero, size=n_take, replace=False, p=weights)
        # Highest expression first is a simple deterministic h5ad-to-token bridge.
        chosen = chosen[np.argsort(-expr[chosen])]
        tokens = np.full(self.context_length, self.pad_token_id, dtype=np.int64)
        tokens[0] = self.cls_token_id
        room = self.context_length - 1
        tokens[1: 1 + min(room, len(chosen))] = self.map["token_ids"][chosen[:room]]
        return tokens

    def __getitem__(self, idx):
        cell_idx = int(self.indices[idx])
        high_raw = row_to_dense(self.matrix, cell_idx)
        high_counts = np.rint(np.clip(high_raw, 0.0, None)).astype(np.int64)
        rng = np.random.RandomState(self.seed + cell_idx)
        low_counts = rng.binomial(high_counts, self.keep_prob).astype(np.float32)
        tokens = self._make_tokens(low_counts, rng)
        return torch.from_numpy(tokens).long(), torch.from_numpy(log_norm(high_counts))


class NicheformerRDEModel(nn.Module):
    def __init__(self, ckpt_path: Path, output_dim: int, freeze_encoder: bool, head_hidden: int, head_dropout: float):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hparams = dict(ckpt["hyper_parameters"])
        allowed = {
            k
            for k in inspect.signature(Nicheformer.__init__).parameters
            if k != "self"
        }
        init_kwargs = {k: v for k, v in hparams.items() if k in allowed}
        self.backbone = Nicheformer(**init_kwargs)
        missing, unexpected = self.backbone.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print(
                f"[nicheformer] manual checkpoint load: "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
        dim_model = int(self.backbone.hparams.dim_model)
        self.head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, output_dim),
        )

    def forward(self, tokens):
        attention_mask = tokens.eq(1)
        out = self.backbone(tokens, attention_mask)
        hidden = out["transformer_output"]
        valid = (~attention_mask).unsqueeze(-1).to(hidden.dtype)
        # Exclude CLS if there is at least one gene token.
        gene_hidden = hidden[:, 1:, :]
        gene_valid = valid[:, 1:, :]
        pooled = (gene_hidden * gene_valid).sum(dim=1) / gene_valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def train_one_epoch(model, loader, optimizer, scaler, device, amp, grad_clip):
    model.train()
    total = 0.0
    n = 0
    for tokens, target in tqdm(loader, desc="train", unit="batch"):
        tokens = tokens.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            pred = model(tokens)
            loss = F.mse_loss(pred, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item()) * tokens.size(0)
        n += tokens.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    total = 0.0
    n = 0
    for tokens, target in tqdm(loader, desc="valid", unit="batch"):
        tokens = tokens.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(enabled=amp):
            pred = model(tokens)
            loss = F.mse_loss(pred, target)
        total += float(loss.item()) * tokens.size(0)
        n += tokens.size(0)
    return total / max(n, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Nicheformer + RDE head on PBMC read-depth enhancement.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT))
    parser.add_argument("--gene_map", default=str(DEFAULT_GENE_MAP))
    parser.add_argument("--gene_col", default="feature_name")
    parser.add_argument("--layer", default="")
    parser.add_argument("--train_total_cells", type=int, default=12000)
    parser.add_argument("--val_total_cells", type=int, default=2000)
    parser.add_argument("--train_indices_path", default="", help="optional .npy original h5ad cell indices for training")
    parser.add_argument("--val_indices_path", default="", help="optional .npy original h5ad cell indices for validation")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_cell_seed", type=int, default=3456)
    parser.add_argument("--val_cell_seed", type=int, default=4567)
    parser.add_argument("--downsample_fold", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--context_length", type=int, default=1500)
    parser.add_argument("--sample_size", type=int, default=1200)
    parser.add_argument("--pad_token_id", type=int, default=1)
    parser.add_argument("--cls_token_id", type=int, default=3)
    parser.add_argument("--head_hidden", type=int, default=2048)
    parser.add_argument("--head_dropout", type=float, default=0.1)
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

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    n_tokens = int(ckpt["hyper_parameters"]["n_tokens"])
    symbol_token_map = load_symbol_token_map(Path(args.gene_map), n_tokens=n_tokens)

    adata = ad.read_h5ad(args.data_path)
    gene_col = detect_gene_col(adata, args.gene_col)
    gene_names = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    matrix = get_matrix(adata, args.layer)
    gene_mapping = build_gene_mapping(gene_names, symbol_token_map)
    print(f"[data] cells={adata.n_obs} genes={adata.n_vars} mapped={gene_mapping['n_mapped_genes']}/{gene_mapping['n_input_genes']}", flush=True)

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
    print(f"[split] train={len(train_idx)} val={len(val_idx)}", flush=True)

    ds_kwargs = dict(
        matrix=matrix,
        gene_mapping=gene_mapping,
        downsample_fold=args.downsample_fold,
        seed=args.seed,
        context_length=args.context_length,
        sample_size=args.sample_size,
        pad_token_id=args.pad_token_id,
        cls_token_id=args.cls_token_id,
    )
    train_ds = NicheformerRDEDataset(indices=train_idx, **ds_kwargs)
    val_ds = NicheformerRDEDataset(indices=val_idx, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = NicheformerRDEModel(Path(args.ckpt_path), adata.n_vars, args.freeze_encoder, args.head_hidden, args.head_dropout).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
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
                    "gene_names": gene_names,
                    "gene_mapping": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gene_mapping.items()},
                    "best_metrics": best_metrics,
                    "mapping_note": "Uses local CellFM symbol token map because the Nicheformer gene dictionary is not present in this workspace.",
                },
                save_dir / "best_model.pt",
            )
            (save_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n")
            print(f"[saved] best checkpoint updated: {save_dir / 'best_model.pt'}", flush=True)

    summary = {
        "output_dir": str(save_dir),
        "best_model_pt": str(save_dir / "best_model.pt"),
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
