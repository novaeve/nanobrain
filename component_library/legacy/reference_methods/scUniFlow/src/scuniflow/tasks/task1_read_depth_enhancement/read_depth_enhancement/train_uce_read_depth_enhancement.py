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
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import issparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

THIS_FILE = Path(__file__).resolve()
SCUNIFLOW_ROOT = next(p for p in THIS_FILE.parents if (p / "model.py").exists() and (p / "data.py").exists())
WORKSPACE_ROOT = SCUNIFLOW_ROOT.parent
UCE_ROOT = WORKSPACE_ROOT / "UCE"
sys.path.insert(0, str(UCE_ROOT))

from model import TransformerModel  # noqa: E402


DEFAULT_UCE_MODEL_DIR = Path("external/uce/model")
DEFAULT_RESULT_ROOT = Path("./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_matrix(adata, layer: str):
    if not layer:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer not found in adata.layers: {layer}")
    return adata.layers[layer]


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


def split_train_val_indices(n_obs: int, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_obs)
    n_val = max(1, int(round(n_obs * float(val_ratio))))
    n_val = min(n_val, n_obs - 1)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def sample_indices(indices: np.ndarray, n_cells: int, seed: int):
    if n_cells is None or n_cells <= 0 or n_cells >= len(indices):
        return np.array(indices, copy=True)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(indices, size=int(n_cells), replace=False))


def row_to_dense(matrix, row_idx: int):
    row = matrix.getrow(row_idx).toarray().ravel() if issparse(matrix) else np.asarray(matrix[row_idx]).ravel()
    return np.asarray(row, dtype=np.float32)


def log_norm(counts: np.ndarray):
    counts = np.asarray(counts, dtype=np.float32)
    total = float(counts.sum())
    return np.log1p(counts / (total + 1e-6) * 1e4).astype(np.float32)


def build_uce_gene_mapping(
    gene_names,
    protein_embeddings_dir: Path,
    species_chrom_csv_path: Path,
    offset_pkl_path: Path,
    species: str,
):
    import pickle

    pe_path = protein_embeddings_dir / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
    species_to_pe = torch.load(pe_path, map_location="cpu")
    pe_genes = [str(g).upper() for g in species_to_pe.keys()]
    gene_to_pe_local_idx = {gene: idx for idx, gene in enumerate(pe_genes)}

    with open(offset_pkl_path, "rb") as f:
        species_offsets = pickle.load(f)
    offset = int(species_offsets[species])

    chrom_df = pd.read_csv(species_chrom_csv_path)
    chrom_df = chrom_df[chrom_df["species"] == species].copy()
    chrom_df["gene_symbol"] = chrom_df["gene_symbol"].astype(str).str.upper()
    chrom_df["spec_chrom"] = pd.Categorical(chrom_df["species"].astype(str) + "_" + chrom_df["chromosome"].astype(str))
    chrom_df = chrom_df.drop_duplicates("gene_symbol").set_index("gene_symbol")

    source_cols = []
    token_ids = []
    chroms = []
    starts = []
    for idx, gene in enumerate(gene_names):
        key = str(gene).upper()
        local_idx = gene_to_pe_local_idx.get(key)
        if local_idx is None or key not in chrom_df.index:
            continue
        source_cols.append(idx)
        token_ids.append(local_idx + offset)
        chroms.append(int(chrom_df.loc[key, "spec_chrom"].codes if hasattr(chrom_df.loc[key, "spec_chrom"], "codes") else 0))
        starts.append(float(chrom_df.loc[key, "start"]))

    if not source_cols:
        raise RuntimeError("No PBMC genes could be mapped to UCE protein/chromosome tokens.")

    # Recompute chrom categorical codes after filtering, avoiding pandas scalar categorical quirks.
    mapped_chrom_labels = []
    for idx in source_cols:
        key = str(gene_names[idx]).upper()
        mapped_chrom_labels.append(str(chrom_df.loc[key, "species"]) + "_" + str(chrom_df.loc[key, "chromosome"]))
    chroms = pd.Categorical(mapped_chrom_labels).codes.astype(np.int64)

    return {
        "source_cols": np.asarray(source_cols, dtype=np.int64),
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "chroms": np.asarray(chroms, dtype=np.int64),
        "starts": np.asarray(starts, dtype=np.float32),
        "n_input_genes": int(len(gene_names)),
        "n_mapped_genes": int(len(source_cols)),
    }


class UCERDEDataset(Dataset):
    def __init__(
        self,
        matrix,
        indices,
        uce_mapping,
        downsample_fold: float,
        seed: int,
        sample_size: int,
        pad_length: int,
        pad_token_idx: int,
        chrom_token_right_idx: int,
        cls_token_idx: int,
        chrom_token_offset: int,
    ):
        self.matrix = matrix
        self.indices = np.asarray(indices, dtype=np.int64)
        self.map = uce_mapping
        self.keep_prob = 1.0 / max(float(downsample_fold), 1.0)
        self.seed = int(seed)
        self.sample_size = int(sample_size)
        self.pad_length = int(pad_length)
        self.pad_token_idx = int(pad_token_idx)
        self.chrom_token_right_idx = int(chrom_token_right_idx)
        self.cls_token_idx = int(cls_token_idx)
        self.chrom_token_offset = int(chrom_token_offset)

    def __len__(self):
        return len(self.indices)

    def _make_sentence(self, low_counts: np.ndarray, rng: np.random.RandomState):
        cols = self.map["source_cols"]
        expr = np.asarray(low_counts[cols], dtype=np.float64)
        weights = np.log1p(np.clip(expr, 0.0, None))
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / weights.sum()

        choice = rng.choice(np.arange(len(weights)), size=self.sample_size, replace=True, p=weights)
        chroms = self.map["chroms"][choice]
        starts = self.map["starts"][choice]
        token_ids = self.map["token_ids"][choice]

        order = np.lexsort((starts, chroms))
        chroms = chroms[order]
        starts = starts[order]
        token_ids = token_ids[order]

        sentence = np.full(self.pad_length, self.pad_token_idx, dtype=np.int64)
        sentence[0] = self.cls_token_idx
        cursor = 1
        for chrom in rng.permutation(np.unique(chroms)):
            loc = np.where(chroms == chrom)[0]
            loc = loc[np.argsort(starts[loc])]
            needed = len(loc) + 2
            if cursor + needed > self.pad_length:
                room = max(0, self.pad_length - cursor - 2)
                loc = loc[:room]
                needed = len(loc) + 2
            if needed <= 1 or cursor >= self.pad_length:
                break
            sentence[cursor] = int(chrom) + self.chrom_token_offset
            cursor += 1
            if len(loc):
                sentence[cursor: cursor + len(loc)] = token_ids[loc]
                cursor += len(loc)
            if cursor < self.pad_length:
                sentence[cursor] = self.chrom_token_right_idx
                cursor += 1
            if cursor >= self.pad_length:
                break
        mask = np.zeros(self.pad_length, dtype=np.float32)
        mask[:cursor] = 1.0
        return sentence, mask

    def __getitem__(self, idx):
        cell_idx = int(self.indices[idx])
        high_raw = row_to_dense(self.matrix, cell_idx)
        high_counts = np.rint(np.clip(high_raw, 0.0, None)).astype(np.int64)
        rng = np.random.RandomState(self.seed + cell_idx)
        low_counts = rng.binomial(high_counts, self.keep_prob).astype(np.float32)
        sentence, mask = self._make_sentence(low_counts, rng)
        return {
            "tokens": torch.from_numpy(sentence).long(),
            "mask": torch.from_numpy(mask).float(),
            "target": torch.from_numpy(log_norm(high_counts)),
        }


def collate_uce(batch):
    max_len = max(int(x["mask"].sum().item()) for x in batch)
    max_len = max(max_len, 1)
    tokens = torch.stack([x["tokens"][:max_len] for x in batch])
    mask = torch.stack([x["mask"][:max_len] for x in batch])
    target = torch.stack([x["target"] for x in batch])
    return tokens, mask, target


class UCERDEModel(nn.Module):
    def __init__(self, args, output_dim: int):
        super().__init__()
        self.encoder = TransformerModel(
            token_dim=args.token_dim,
            d_model=1280,
            nhead=20,
            d_hid=args.d_hid,
            nlayers=args.nlayers,
            dropout=args.dropout,
            output_dim=args.output_dim,
        )
        empty_pe = torch.zeros(args.pe_vocab_size, args.token_dim)
        self.encoder.pe_embedding = nn.Embedding.from_pretrained(empty_pe, freeze=True)
        state = torch.load(args.model_loc, map_location="cpu")
        self.encoder.load_state_dict(state, strict=True)

        all_tokens = torch.load(args.token_file, map_location="cpu")
        if all_tokens.shape[0] == 143574:
            torch.manual_seed(23)
            chrom_tokens = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
            all_tokens = torch.vstack((all_tokens, chrom_tokens))
        self.encoder.pe_embedding = nn.Embedding.from_pretrained(all_tokens.float(), freeze=True)

        if args.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(args.output_dim),
            nn.Linear(args.output_dim, args.head_hidden),
            nn.GELU(),
            nn.Dropout(args.head_dropout),
            nn.Linear(args.head_hidden, output_dim),
        )

    def forward(self, token_ids, mask):
        token_emb = self.encoder.pe_embedding(token_ids.long()).permute(1, 0, 2)
        token_emb = F.normalize(token_emb, dim=2)
        _, embedding = self.encoder(token_emb, mask)
        return self.head(embedding)


def train_one_epoch(model, loader, optimizer, scaler, device, amp, grad_clip):
    model.train()
    total_loss = 0.0
    total_cells = 0
    for tokens, mask, target in tqdm(loader, desc="train", unit="batch"):
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            pred = model(tokens, mask)
            loss = F.mse_loss(pred, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = tokens.size(0)
        total_loss += float(loss.item()) * bs
        total_cells += bs
    return total_loss / max(total_cells, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    total_loss = 0.0
    total_cells = 0
    for tokens, mask, target in tqdm(loader, desc="valid", unit="batch"):
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast(enabled=amp):
            pred = model(tokens, mask)
            loss = F.mse_loss(pred, target)
        bs = tokens.size(0)
        total_loss += float(loss.item()) * bs
        total_cells += bs
    return total_loss / max(total_cells, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune UCE + RDE head on PBMC read-depth enhancement.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_dir", default=str(DEFAULT_UCE_MODEL_DIR))
    parser.add_argument("--model_loc", default=str(DEFAULT_UCE_MODEL_DIR / "4layer_model.torch"))
    parser.add_argument("--token_file", default=str(DEFAULT_UCE_MODEL_DIR / "all_tokens.torch"))
    parser.add_argument("--protein_embeddings_dir", default=str(DEFAULT_UCE_MODEL_DIR / "protein_embeddings"))
    parser.add_argument("--species_chrom_csv_path", default=str(DEFAULT_UCE_MODEL_DIR / "species_chrom.csv"))
    parser.add_argument("--offset_pkl_path", default=str(DEFAULT_UCE_MODEL_DIR / "species_offsets.pkl"))
    parser.add_argument("--species", default="human")
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
    parser.add_argument("--sample_size", type=int, default=1024)
    parser.add_argument("--pad_length", type=int, default=1536)
    parser.add_argument("--pad_token_idx", type=int, default=0)
    parser.add_argument("--chrom_token_right_idx", type=int, default=2)
    parser.add_argument("--cls_token_idx", type=int, default=3)
    parser.add_argument("--chrom_token_offset", type=int, default=143574)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--output_dim", type=int, default=1280)
    parser.add_argument("--d_hid", type=int, default=5120)
    parser.add_argument("--token_dim", type=int, default=5120)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pe_vocab_size", type=int, default=145469)
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

    adata = ad.read_h5ad(args.data_path)
    gene_col = detect_gene_col(adata, args.gene_col)
    gene_names = adata.var_names.astype(str).tolist() if gene_col == "index" else adata.var[gene_col].astype(str).tolist()
    matrix = get_matrix(adata, args.layer)
    print(f"[data] cells={adata.n_obs} genes={adata.n_vars} gene_col={gene_col}", flush=True)

    uce_mapping = build_uce_gene_mapping(
        gene_names=gene_names,
        protein_embeddings_dir=Path(args.protein_embeddings_dir),
        species_chrom_csv_path=Path(args.species_chrom_csv_path),
        offset_pkl_path=Path(args.offset_pkl_path),
        species=args.species,
    )
    print(f"[data] UCE mapped genes={uce_mapping['n_mapped_genes']}/{uce_mapping['n_input_genes']}", flush=True)

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
        uce_mapping=uce_mapping,
        downsample_fold=args.downsample_fold,
        seed=args.seed,
        sample_size=args.sample_size,
        pad_length=args.pad_length,
        pad_token_idx=args.pad_token_idx,
        chrom_token_right_idx=args.chrom_token_right_idx,
        cls_token_idx=args.cls_token_idx,
        chrom_token_offset=args.chrom_token_offset,
    )
    train_ds = UCERDEDataset(indices=train_idx, **ds_kwargs)
    val_ds = UCERDEDataset(indices=val_idx, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_uce, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_uce, pin_memory=device.type == "cuda")

    model = UCERDEModel(args, output_dim=adata.n_vars).to(device)
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
        metrics = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 2),
            "train_recon_loss": float(train_loss),
            "valid_recon_loss": float(val_loss),
        }
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
                    "uce_mapping": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in uce_mapping.items()},
                    "best_metrics": best_metrics,
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
        "uce_mapped_genes": int(uce_mapping["n_mapped_genes"]),
        "output_genes": int(adata.n_vars),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
