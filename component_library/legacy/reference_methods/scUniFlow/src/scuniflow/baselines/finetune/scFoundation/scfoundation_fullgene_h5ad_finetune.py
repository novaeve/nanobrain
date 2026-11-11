import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, issparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = THIS_DIR.parent
EVAL_DIR = FINETUNE_DIR.parent
REPO_ROOT = EVAL_DIR.parent
SCFOUNDATION_ROOT = Path("external/scFoundation")
SCFOUNDATION_MODEL_DIR = SCFOUNDATION_ROOT / "model"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SCFOUNDATION_MODEL_DIR))

from pretrainmodels.mae_autobin import MaeAutobin
from pretrainmodels.performer import PerformerModule
from pretrainmodels.transformer import pytorchTransformerModule


EXTRA_TOKENS = 2
DEFAULT_RESULT_ROOT = Path("./src/scuniflow/baselines/finetune/outputs/scFoundation")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def detect_gene_col(adata, requested: str) -> str:
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


def split_train_val_indices(n_obs: int, val_ratio: float, seed: int):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if n_obs < 2:
        raise ValueError(f"Need at least 2 cells to split, got {n_obs}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_obs)
    n_val = max(1, int(round(n_obs * val_ratio)))
    n_val = min(n_val, n_obs - 1)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def sample_indices(indices: np.ndarray, n_cells: int, seed: int):
    if n_cells is None or n_cells <= 0 or n_cells >= len(indices):
        return np.array(indices, copy=True)
    rng = np.random.RandomState(seed)
    sampled = rng.choice(indices, size=n_cells, replace=False)
    return np.sort(sampled)


def build_gene_mapping(source_gene_names, target_gene_list):
    source_to_idx = {}
    for idx, gene in enumerate(source_gene_names):
        if gene not in source_to_idx:
            source_to_idx[gene] = idx

    target_positions = []
    source_positions = []
    missing_gene_names = []
    for target_idx, gene in enumerate(target_gene_list):
        src_idx = source_to_idx.get(gene)
        if src_idx is None:
            missing_gene_names.append(gene)
            continue
        target_positions.append(target_idx)
        source_positions.append(src_idx)

    return {
        "target_positions": np.array(target_positions, dtype=np.int64),
        "source_positions": np.array(source_positions, dtype=np.int64),
        "missing_gene_names": missing_gene_names,
    }


def align_matrix_to_target_genes(matrix, mapping, full_gene_count: int):
    source_positions = mapping["source_positions"]
    target_positions = mapping["target_positions"]

    if issparse(matrix):
        subset = matrix[:, source_positions]
        subset = subset.tocsr() if not isinstance(subset, csr_matrix) else subset
        remapped_indices = np.take(target_positions, subset.indices)
        aligned = csr_matrix(
            (subset.data, remapped_indices, subset.indptr),
            shape=(matrix.shape[0], full_gene_count),
        )
        return aligned

    subset = np.asarray(matrix[:, source_positions], dtype=np.float32)
    aligned = np.zeros((matrix.shape[0], full_gene_count), dtype=np.float32)
    aligned[:, target_positions] = subset
    return aligned


def build_model_from_config(config):
    encoder_cfg = config["encoder"]
    decoder_cfg = config["decoder"]

    if encoder_cfg["module_type"] == "transformer":
        encoder = pytorchTransformerModule(
            max_seq_len=config["seq_len"],
            dim=encoder_cfg["hidden_dim"],
            depth=encoder_cfg["depth"],
            heads=encoder_cfg["heads"],
            norm_first=encoder_cfg.get("norm_first", False),
        )
    else:
        raise NotImplementedError(
            f"Unsupported encoder module_type: {encoder_cfg['module_type']}"
        )

    if decoder_cfg["module_type"] == "performer":
        decoder = PerformerModule(
            max_seq_len=config["seq_len"],
            dim=decoder_cfg["hidden_dim"],
            depth=decoder_cfg["depth"],
            heads=decoder_cfg["heads"],
            dim_head=decoder_cfg.get("dim_head", 64),
        )
    elif decoder_cfg["module_type"] == "transformer":
        decoder = pytorchTransformerModule(
            max_seq_len=config["seq_len"],
            dim=decoder_cfg["hidden_dim"],
            depth=decoder_cfg["depth"],
            heads=decoder_cfg["heads"],
            norm_first=decoder_cfg.get("norm_first", False),
        )
    else:
        raise NotImplementedError(
            f"Unsupported decoder module_type: {decoder_cfg['module_type']}"
        )

    model = MaeAutobin(
        num_tokens=config["n_class"],
        max_seq_len=config["seq_len"],
        embed_dim=encoder_cfg["hidden_dim"],
        decoder_embed_dim=decoder_cfg["hidden_dim"],
        bin_alpha=config["bin_alpha"],
        bin_num=config["bin_num"],
        pad_token_id=config["pad_token_id"],
        mask_token_id=config["mask_token_id"],
    )
    model.encoder = encoder
    model.decoder = decoder
    return model


def load_gene_branch_config(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gene_branch = ckpt["gene"]
    raw_cfg = gene_branch["config"]
    model_type = raw_cfg["model"]

    cfg = {}
    cfg.update(raw_cfg["model_config"][model_type])
    cfg.update(raw_cfg["dataset_config"]["rnaseq"])
    cfg["model_type"] = model_type
    cfg["pos_embed"] = False
    cfg["device"] = "cpu"

    state_dict = {}
    for key, value in gene_branch["state_dict"].items():
        state_dict[str(key).split("model.", 1)[1]] = value
    return cfg, state_dict


def gather_data(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = int(value_nums.max().item())

    fake_data = torch.full(
        (data.shape[0], max_num),
        pad_token_id,
        device=data.device,
        dtype=data.dtype,
    )
    padded_data = torch.hstack([data, fake_data])

    fake_label = torch.full(
        (labels.shape[0], max_num),
        True,
        device=labels.device,
        dtype=torch.bool,
    )
    none_labels = ~labels
    score = labels.float()
    score[none_labels] = torch.tensor(-float("inf"), device=labels.device)

    tmp_data = torch.tensor(
        [(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)],
        device=labels.device,
        dtype=score.dtype,
    )
    score += tmp_data
    score = torch.hstack([score, fake_label.float()])

    gather_idx = score.topk(max_num).indices
    new_data = torch.gather(padded_data, 1, gather_idx)
    padding_labels = new_data.eq(pad_token_id)
    return new_data, padding_labels


def get_encoder_decoder_data(data, data_raw, config):
    decoder_data = data.clone().detach()
    decoder_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)

    encoder_labels = data_raw > 0
    encoder_data, encoder_padding = gather_data(
        decoder_data, encoder_labels, config["pad_token_id"]
    )
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_pos_ids, _ = gather_data(
        data_gene_ids, encoder_labels, config["pad_token_id"]
    )
    decoder_pos_ids = data_gene_ids

    encoder_pos_ids[encoder_padding] = config["seq_len"]
    decoder_pos_ids[decoder_padding] = config["seq_len"]

    return (
        encoder_data,
        encoder_pos_ids,
        encoder_padding,
        encoder_labels,
        decoder_data,
        decoder_padding,
        decoder_pos_ids,
    )


class ScFoundationReconDataset(Dataset):
    def __init__(self, matrix, tgt_highres: float = 4.0):
        self.matrix = matrix
        self.tgt_highres = np.float32(tgt_highres)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        row = self.matrix[idx]
        if issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row).ravel()

        row = row.astype(np.float32, copy=False)
        total = float(row.sum())
        x_norm = np.log1p(row / (total + 1e-6) * 1e4).astype(np.float32)
        log_total = np.float32(np.log10(total + 1.0))
        x_input = np.concatenate(
            [x_norm, np.array([self.tgt_highres, log_total], dtype=np.float32)]
        ).astype(np.float32)
        return torch.from_numpy(x_input)


class ScFoundationReconOnly(nn.Module):
    def __init__(self, ckpt_path: Path, device: torch.device, freeze_emb: bool = True, unfreeze_last_n: int = 2):
        super().__init__()
        model_cfg, model_state = load_gene_branch_config(ckpt_path)
        model = build_model_from_config(model_cfg)
        model.load_state_dict(model_state)

        self.cfg = model_cfg
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        self.decoder_embed = model.decoder_embed
        self.decoder = model.decoder
        self.recon_norm = model.norm
        self.to_final = model.to_final
        self._freeze(freeze_emb=freeze_emb, unfreeze_last_n=unfreeze_last_n)
        self.to(device)

    def _freeze(self, freeze_emb: bool, unfreeze_last_n: int):
        if freeze_emb:
            for param in self.token_emb.parameters():
                param.requires_grad = False
            for param in self.pos_emb.parameters():
                param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0 and hasattr(self.encoder, "transformer_encoder"):
            blocks = list(self.encoder.transformer_encoder)
            for block in blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor):
        (
            enc_data,
            enc_pos,
            enc_pad,
            enc_labels,
            dec_data,
            dec_pad,
            dec_pos,
        ) = get_encoder_decoder_data(x.float(), x.float(), self.cfg)

        tok = self.token_emb(enc_data.unsqueeze(2).float(), output_weight=0)
        tok = tok + self.pos_emb(enc_pos)
        enc_out = self.encoder(tok, enc_pad)

        dec_tok = self.token_emb(dec_data.unsqueeze(2).float())
        dec_tok = dec_tok + self.pos_emb(dec_pos)
        batch_idx, gene_idx = enc_labels.nonzero(as_tuple=True)
        dec_tok[batch_idx, gene_idx] = enc_out[~enc_pad].to(dec_tok.dtype)

        dec_tok = self.decoder_embed(dec_tok)
        dec_out = self.decoder(dec_tok, padding_mask=dec_pad)
        dec_out = self.recon_norm(dec_out)
        pred = self.to_final(dec_out).squeeze(-1)
        return pred[:, : x.shape[1] - EXTRA_TOKENS]


def get_trainable_params(*modules):
    params = []
    for module in modules:
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def train_one_epoch(model, loader, optimizer, scaler, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_cells = 0

    for x in tqdm(loader, total=len(loader), desc="train", unit="batch"):
        x = x.to(device, non_blocking=True)
        target = x[:, :-EXTRA_TOKENS]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            recon_pred = model(x)
            loss = F.mse_loss(recon_pred, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_batches += 1
        total_cells += batch_size

    mean_loss = total_loss / max(total_cells, 1)
    return {
        "train_recon_loss": mean_loss,
        "train_batches": total_batches,
        "train_cells": total_cells,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_cells = 0

    for x in tqdm(loader, total=len(loader), desc="valid", unit="batch"):
        x = x.to(device, non_blocking=True)
        target = x[:, :-EXTRA_TOKENS]

        with autocast(enabled=device.type == "cuda"):
            recon_pred = model(x)
            loss = F.mse_loss(recon_pred, target)

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_batches += 1
        total_cells += batch_size

    mean_loss = total_loss / max(total_cells, 1)
    return {
        "valid_recon_loss": mean_loss,
        "valid_batches": total_batches,
        "valid_cells": total_cells,
    }


def build_output_dir(args) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        dataset_stem = Path(args.h5ad_path).stem
        run_name = args.run_name or (
            f"{dataset_stem}_scfoundation_recon_"
            f"train{args.train_total_cells if args.train_total_cells > 0 else 'all'}_"
            f"val{args.val_total_cells if args.val_total_cells > 0 else 'all'}_"
            f"s{args.seed}"
        )
        out_dir = Path(args.result_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_checkpoint(out_dir: Path, model, optimizer, args, cfg, best_metrics, summary):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "args": vars(args),
            "best_metrics": best_metrics,
        },
        out_dir / "best_model.pt",
    )
    (out_dir / "args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n"
    )
    (out_dir / "model_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    )
    (out_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n"
    )
    (out_dir / "finetune_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune scFoundation on a single h5ad file for reconstruction."
    )
    parser.add_argument("--h5ad_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=str(SCFOUNDATION_ROOT / "models.ckpt"))
    parser.add_argument(
        "--gene_index",
        type=str,
        default=str(SCFOUNDATION_ROOT / "OS_scRNA_gene_index.19264.tsv"),
    )
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
    parser.add_argument("--gene_col", type=str, default="auto")
    parser.add_argument("--matrix_layer", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--freeze_emb", dest="freeze_emb", action="store_true")
    parser.add_argument("--no_freeze_emb", dest="freeze_emb", action="store_false")
    parser.set_defaults(freeze_emb=True)
    parser.add_argument("--unfreeze_last_n", type=int, default=2)
    parser.add_argument("--tgt_highres", type=float, default=4.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = build_output_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    print(f"[setup] h5ad_path={args.h5ad_path}", flush=True)
    print(f"[setup] ckpt_path={args.ckpt_path}", flush=True)
    print(f"[setup] output_dir={out_dir}", flush=True)
    print(f"[setup] device={device} amp={amp}", flush=True)

    adata = ad.read_h5ad(args.h5ad_path)
    gene_col = detect_gene_col(adata, args.gene_col)
    source_gene_names = (
        adata.var_names.astype(str).tolist()
        if gene_col == "index"
        else adata.var[gene_col].astype(str).tolist()
    )
    matrix = get_matrix(adata, args.matrix_layer)

    gene_df = pd.read_csv(args.gene_index, sep="\t")
    target_gene_list = gene_df["gene_name"].astype(str).tolist()
    gene_mapping = build_gene_mapping(source_gene_names, target_gene_list)

    print(
        f"[data] loaded cells={adata.n_obs} genes={adata.n_vars} "
        f"gene_col={gene_col} matrix_layer={(args.matrix_layer or 'X')}",
        flush=True,
    )
    print(
        f"[data] scFoundation gene overlap: "
        f"{len(gene_mapping['source_positions'])}/{len(target_gene_list)}",
        flush=True,
    )

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
        train_idx, val_idx = split_train_val_indices(
            n_obs=adata.n_obs,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )
        train_idx = sample_indices(train_idx, args.train_total_cells, args.train_cell_seed)
        val_idx = sample_indices(val_idx, args.val_total_cells, args.val_cell_seed)

    print(
        f"[split] train cells={len(train_idx)} val cells={len(val_idx)} "
        f"val_ratio={args.val_ratio}",
        flush=True,
    )

    train_matrix = align_matrix_to_target_genes(
        matrix[train_idx], gene_mapping, len(target_gene_list)
    )
    val_matrix = align_matrix_to_target_genes(
        matrix[val_idx], gene_mapping, len(target_gene_list)
    )

    train_ds = ScFoundationReconDataset(train_matrix, tgt_highres=args.tgt_highres)
    val_ds = ScFoundationReconDataset(val_matrix, tgt_highres=args.tgt_highres)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ScFoundationReconOnly(
        ckpt_path=Path(args.ckpt_path),
        device=device,
        freeze_emb=args.freeze_emb,
        unfreeze_last_n=args.unfreeze_last_n,
    )
    backbone_params = get_trainable_params(
        model.token_emb, model.pos_emb, model.encoder
    )
    head_params = get_trainable_params(
        model.decoder_embed, model.decoder, model.recon_norm, model.to_final
    )

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = GradScaler(enabled=amp)

    best_val_recon_loss = float("inf")
    best_metrics = {}
    history_path = out_dir / "history.jsonl"

    shutil.copy2(args.gene_index, out_dir / "gene_index.tsv")

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch] {epoch}/{args.epochs}", flush=True)
        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
        )
        valid_metrics = evaluate(model=model, loader=val_loader, device=device)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 2),
            **train_metrics,
            **valid_metrics,
        }
        with history_path.open("a") as f:
            f.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
        print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2), flush=True)

        if valid_metrics["valid_recon_loss"] < best_val_recon_loss:
            best_val_recon_loss = valid_metrics["valid_recon_loss"]
            best_metrics = dict(epoch_metrics)
            summary = {
                "dataset": "single h5ad reconstruction",
                "h5ad_path": str(args.h5ad_path),
                "ckpt_path": str(args.ckpt_path),
                "gene_index": str(args.gene_index),
                "gene_col": gene_col,
                "matrix_layer": args.matrix_layer or "X",
                "split_rule": f"random cell split val_ratio={args.val_ratio} seed={args.split_seed}",
                "train_cells": int(len(train_idx)),
                "val_cells": int(len(val_idx)),
                "train_total_cells_target": int(args.train_total_cells),
                "val_total_cells_target": int(args.val_total_cells),
                "model_gene_count": int(len(target_gene_list)),
                "genes_found_in_input": int(len(gene_mapping["source_positions"])),
                "genes_missing_from_input": int(len(gene_mapping["missing_gene_names"])),
                "missing_gene_names_preview": gene_mapping["missing_gene_names"][:50],
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "lr_head": float(args.lr_head),
                "freeze_emb": bool(args.freeze_emb),
                "unfreeze_last_n": int(args.unfreeze_last_n),
                "tgt_highres": float(args.tgt_highres),
                "best_metrics": best_metrics,
            }
            save_checkpoint(
                out_dir=out_dir,
                model=model,
                optimizer=optimizer,
                args=args,
                cfg=model.cfg,
                best_metrics=best_metrics,
                summary=summary,
            )
            print(
                f"[saved] best checkpoint updated at {out_dir}",
                flush=True,
            )

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": model.cfg,
            "args": vars(args),
            "best_metrics": best_metrics,
        },
        out_dir / "last_model.pt",
    )
    final_summary = {
        "output_dir": str(out_dir),
        "best_model_pt": str(out_dir / "best_model.pt"),
        "last_model_pt": str(out_dir / "last_model.pt"),
        "history_jsonl": str(history_path),
        "best_metrics": best_metrics,
        "train_cells": int(len(train_idx)),
        "val_cells": int(len(val_idx)),
        "model_gene_count": int(len(target_gene_list)),
        "genes_found_in_input": int(len(gene_mapping["source_positions"])),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
