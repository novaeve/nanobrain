import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from masked_recon_metrics import init_metric_accumulator, summarize_metric_accumulator, update_metric_accumulator


class SimpleGeneVocab:
    def __init__(self, token_to_idx):
        self.stoi = dict(token_to_idx)
        self.default_index = None

    @classmethod
    def from_file(cls, file_path: Path):
        with Path(file_path).open() as f:
            data = json.load(f)
        return cls(data)

    def __contains__(self, token):
        return token in self.stoi

    def __getitem__(self, token):
        return self.stoi[token]

    def __len__(self):
        return len(self.stoi)

    def set_default_index(self, idx):
        self.default_index = idx

    def append_token(self, token):
        if token not in self.stoi:
            self.stoi[token] = len(self.stoi)

    def __call__(self, tokens):
        out = []
        for t in tokens:
            if t in self.stoi:
                out.append(self.stoi[t])
            elif self.default_index is not None:
                out.append(self.default_index)
            else:
                raise KeyError(f"Token {t} not in vocab and no default index set")
        return out


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(data_pt, batch_size=64, shuffle=False):
    return DataLoader(
        dataset=SeqDataset(data_pt),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: SimpleGeneVocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    show_progress: bool = False,
    progress_desc: str = "tokenize",
):
    cls_id = vocab[cls_token]
    pad_id = vocab[pad_token]

    tokenized = []
    row_iter = range(len(data))
    if show_progress:
        row_iter = tqdm(row_iter, total=len(data), desc=f"{progress_desc}: scan", unit="cell")
    for i in row_iter:
        row = data[i]
        if include_zero_gene:
            vals = row
            gids = gene_ids
        else:
            nz = np.nonzero(row)[0]
            vals = row[nz]
            gids = gene_ids[nz]
        if append_cls:
            gids = np.insert(gids, 0, cls_id)
            vals = np.insert(vals, 0, 0)
        tokenized.append(
            (
                torch.from_numpy(gids).long(),
                torch.from_numpy(vals).float(),
            )
        )

    max_ori_len = max(len(g) for g, _ in tokenized)
    max_len = min(max_ori_len, max_len)
    genes, values = [], []

    token_iter = tokenized
    if show_progress:
        token_iter = tqdm(tokenized, total=len(tokenized), desc=f"{progress_desc}: pad", unit="cell")
    for g, v in token_iter:
        if len(g) > max_len:
            if append_cls:
                idx = np.random.choice(len(g) - 1, max_len - 1, replace=False) + 1
                idx = np.insert(idx, 0, 0)
            else:
                idx = np.random.choice(len(g), max_len, replace=False)
            g = g[idx]
            v = v[idx]
        if len(g) < max_len:
            g = torch.cat([g, torch.full((max_len - len(g),), pad_id, dtype=g.dtype)])
            v = torch.cat([v, torch.full((max_len - len(v),), pad_value, dtype=v.dtype)])
        genes.append(g)
        values.append(v)

    return {"genes": torch.stack(genes, dim=0), "values": torch.stack(values, dim=0)}


def random_mask_value(
    values,
    mask_ratio=0.15,
    mask_value=-1,
    pad_value=0,
    base_seed=1234,
    keep_first_n_tokens=1,
    zero_mask_ratio=None,
    nonzero_mask_ratio=None,
):
    values = values.clone().detach().numpy()
    for i in range(len(values)):
        rng = np.random.RandomState(base_seed + i)
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        if keep_first_n_tokens > 0:
            non_padding_idx = non_padding_idx[non_padding_idx >= keep_first_n_tokens]
        if zero_mask_ratio is None and nonzero_mask_ratio is None:
            n_mask = int(len(non_padding_idx) * mask_ratio)
            if n_mask <= 0:
                continue
            mask_idx = rng.choice(non_padding_idx, n_mask, replace=False)
            row[mask_idx] = mask_value
            continue

        eff_nonzero_mask_ratio = float(mask_ratio if nonzero_mask_ratio is None else nonzero_mask_ratio)
        eff_zero_mask_ratio = float(mask_ratio if zero_mask_ratio is None else zero_mask_ratio)

        nonzero_idx = non_padding_idx[row[non_padding_idx] != 0]
        zero_idx = non_padding_idx[row[non_padding_idx] == 0]

        n_mask_nonzero = int(len(nonzero_idx) * eff_nonzero_mask_ratio)
        if n_mask_nonzero > 0:
            mask_nonzero_idx = rng.choice(nonzero_idx, n_mask_nonzero, replace=False)
            row[mask_nonzero_idx] = mask_value

        n_mask_zero = int(len(zero_idx) * eff_zero_mask_ratio)
        if n_mask_zero > 0:
            mask_zero_idx = rng.choice(zero_idx, n_mask_zero, replace=False)
            row[mask_zero_idx] = mask_value
    return torch.from_numpy(values).float()


def summarize_mask_stats(
    original_values: torch.Tensor,
    masked_values: torch.Tensor,
    pad_value: int,
    mask_value: int,
    keep_first_n_tokens: int = 1,
):
    original = original_values.detach().cpu().numpy()
    masked = masked_values.detach().cpu().numpy()

    candidate_mask = original != pad_value
    if keep_first_n_tokens > 0:
        candidate_mask[:, :keep_first_n_tokens] = False

    zero_mask = candidate_mask & (original == 0)
    nonzero_mask = candidate_mask & (original != 0)
    applied_mask = masked == mask_value

    n_candidates = int(candidate_mask.sum())
    n_zero_candidates = int(zero_mask.sum())
    n_nonzero_candidates = int(nonzero_mask.sum())
    n_masked = int((applied_mask & candidate_mask).sum())
    n_zero_masked = int((applied_mask & zero_mask).sum())
    n_nonzero_masked = int((applied_mask & nonzero_mask).sum())

    return {
        "mask_ratio_actual_nonpad": float(n_masked / max(n_candidates, 1)),
        "zero_mask_ratio_actual_nonpad": (
            float(n_zero_masked / max(n_zero_candidates, 1)) if n_zero_candidates > 0 else None
        ),
        "nonzero_mask_ratio_actual_nonpad": (
            float(n_nonzero_masked / max(n_nonzero_candidates, 1)) if n_nonzero_candidates > 0 else None
        ),
        "n_mask_candidates_nonpad": n_candidates,
        "n_zero_mask_candidates_nonpad": n_zero_candidates,
        "n_nonzero_mask_candidates_nonpad": n_nonzero_candidates,
        "n_masked_nonpad": n_masked,
        "n_zero_masked_nonpad": n_zero_masked,
        "n_nonzero_masked_nonpad": n_nonzero_masked,
    }


def invert_bins_to_log1p(values_arr: np.ndarray, bin_edges_arr: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.zeros_like(values_arr, dtype=np.float32)
    for i in range(values_arr.shape[0]):
        edges = bin_edges_arr[i]
        rep = np.zeros((n_bins,), dtype=np.float32)
        if edges.size > 0:
            for b in range(1, n_bins):
                lo = edges[b - 1] if (b - 1) < edges.size else edges[-1]
                hi = edges[b] if b < edges.size else edges[-1]
                rep[b] = 0.5 * (lo + hi)
        idx = np.rint(values_arr[i]).astype(np.int64)
        idx = np.clip(idx, 0, n_bins - 1)
        out[i] = rep[idx]
    return out


def evaluate_mlm_metrics(
    model,
    loader,
    vocab,
    device,
    pad_token,
    mask_value,
    show_progress=True,
    eval_space="log1p",
    n_bins=51,
    progress_desc="scGPT eval",
):
    model.eval()
    stats = init_metric_accumulator()

    with torch.no_grad():
        it = loader
        if show_progress:
            it = tqdm(loader, total=len(loader), desc=progress_desc, unit="batch")
        for batch_data in it:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            bin_edges = batch_data.get("bin_edges", None)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=False,
                    MVC=False,
                    ECS=False,
                )
                output_values = output_dict["mlm_output"]
                masked_positions = input_values.eq(mask_value)
                if eval_space == "binned":
                    if bin_edges is None:
                        raise RuntimeError("bin_edges missing for binned->log1p evaluation")
                    pred_np = output_values.detach().cpu().numpy().astype(np.float32)
                    true_np = target_values.detach().cpu().numpy().astype(np.float32)
                    edges_np = bin_edges.detach().cpu().numpy().astype(np.float32)
                    pred_for_corr = invert_bins_to_log1p(pred_np, edges_np, n_bins=n_bins)
                    true_for_corr = invert_bins_to_log1p(true_np, edges_np, n_bins=n_bins)
                else:
                    pred_for_corr = output_values.detach().cpu().numpy().astype(np.float32)
                    true_for_corr = target_values.detach().cpu().numpy().astype(np.float32)

            update_metric_accumulator(
                stats,
                pred_cpu=pred_for_corr,
                true_cpu=true_for_corr,
                mask_cpu=masked_positions.detach().cpu().numpy(),
            )

    summary = summarize_metric_accumulator(stats)
    return {
        "masked_l2_rmse": summary["l2_rmse"],
        "masked_mse": summary["mse"],
        "masked_mae": summary["mae"],
        "masked_pearson_all": summary["pearson_all"],
        "masked_pearson_cell_mean": summary["pearson_cell_mean"],
        "masked_spearman_cell_mean": summary["spearman_cell_mean"],
        "masked_pearson_top20_cell_mean": summary["pearson_top20_cell_mean"],
        "n_cells_for_corr": summary["n_cells_for_corr"],
        "masked_n_eval_positions": summary["n_eval_positions"],
    }
