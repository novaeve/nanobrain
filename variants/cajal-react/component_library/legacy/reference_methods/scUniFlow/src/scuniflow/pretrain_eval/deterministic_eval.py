import argparse
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import sparse as sp
from torch.utils.data import DataLoader, Dataset

from data import load_sea_ad, load_sea_ad_split
from model import scRNAModel

DEFAULT_RESULT_DIR = "results/pretrain_eval"


@dataclass
class EvalBatch:
    gene_ids: torch.Tensor
    x_1: torch.Tensor
    x_0: torch.Tensor
    mask_indicator: torch.Tensor
    t: torch.Tensor


class DeterministicEvalDataset(Dataset):
    def __init__(
        self,
        count_matrix,
        gene_ids,
        base_seed=42,
        nonzero_mask_rate=0.20,
        zero_mask_rate=0.02,
        fixed_t=0.5,
        t_mode="fixed",
    ):
        self.sparse = sp.issparse(count_matrix)
        self.counts = count_matrix.astype(np.float32) if self.sparse else np.asarray(count_matrix, dtype=np.float32)
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.base_seed = int(base_seed)
        self.nonzero_mask_rate = float(nonzero_mask_rate)
        self.zero_mask_rate = float(zero_mask_rate)
        self.fixed_t = float(fixed_t)
        self.t_mode = t_mode

    def __len__(self):
        return self.counts.shape[0]

    def _make_generator(self, idx: int, offset: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.base_seed + idx * 1009 + offset)
        return g

    def __getitem__(self, idx):
        if self.sparse:
            x_raw = torch.from_numpy(self.counts.getrow(idx).toarray().ravel())
        else:
            x_raw = torch.tensor(self.counts[idx], dtype=torch.float32)

        x_1 = torch.log1p(x_raw)
        gene_ids = torch.from_numpy(self.gene_ids)

        x_0 = torch.randn(x_1.shape, generator=self._make_generator(idx, 1), dtype=x_1.dtype)
        rand_vals = torch.rand(x_1.shape, generator=self._make_generator(idx, 2), dtype=x_1.dtype)

        is_nonzero = x_raw > 0
        mask_nonzero = is_nonzero & (rand_vals < self.nonzero_mask_rate)
        mask_zero = (~is_nonzero) & (rand_vals < self.zero_mask_rate)
        mask_indicator = (mask_nonzero | mask_zero).to(torch.float32)

        if self.t_mode == "fixed":
            t = torch.tensor([self.fixed_t], dtype=torch.float32)
        else:
            t = torch.rand((1,), generator=self._make_generator(idx, 3), dtype=torch.float32)

        return gene_ids, x_1, x_0, mask_indicator, t


def build_loader(args, cfg):
    val_ratio = cfg["training"].get("val_ratio", 0.1)
    seed = cfg["training"].get("seed", 42)

    if args.split == "val":
        _, val_data, _, _ = load_sea_ad_split(args.data, val_ratio=val_ratio, seed=seed)
        count_matrix, gene_ids = val_data
    elif args.split == "train":
        train_data, _, _, _ = load_sea_ad_split(args.data, val_ratio=val_ratio, seed=seed)
        count_matrix, gene_ids = train_data
    else:
        count_matrix, gene_ids, _, _ = load_sea_ad(args.data)

    dataset = DeterministicEvalDataset(
        count_matrix=count_matrix,
        gene_ids=gene_ids,
        base_seed=args.eval_seed,
        nonzero_mask_rate=args.nonzero_mask_rate,
        zero_mask_rate=args.zero_mask_rate,
        fixed_t=args.fixed_t,
        t_mode=args.t_mode,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _masked_vectors(pred, true, mask_indicator):
    masked_pred = []
    masked_true = []
    for b in range(pred.shape[0]):
        m_idx = mask_indicator[b].nonzero(as_tuple=True)[0]
        if len(m_idx) == 0:
            continue
        masked_pred.append(pred[b, m_idx].detach().float().cpu())
        masked_true.append(true[b, m_idx].detach().float().cpu())
    return masked_pred, masked_true


def _summarize_correlations(masked_pred, masked_true):
    cell_corrs = []
    for pred, true in zip(masked_pred, masked_true):
        if len(pred) < 2:
            continue
        r = torch.corrcoef(torch.stack([pred, true]))[0, 1].item()
        if not np.isnan(r):
            cell_corrs.append(r)
    return float(np.mean(cell_corrs)) if cell_corrs else None


def evaluate(model, loader, device, args):
    model.eval()

    total_teacher_loss = 0.0
    total_rollout_loss = 0.0
    n_teacher_batches = 0
    n_rollout_batches = 0

    teacher_pred_all = []
    teacher_true_all = []
    rollout_pred_all = []
    rollout_true_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            eval_batch = EvalBatch(
                gene_ids=batch[0].to(device),
                x_1=batch[1].to(device),
                x_0=batch[2].to(device),
                mask_indicator=batch[3].to(device),
                t=batch[4].to(device),
            )

            if args.mode in ("teacher", "both"):
                preds, targets, _ = model.forward_train(
                    eval_batch.gene_ids,
                    eval_batch.x_1,
                    eval_batch.mask_indicator,
                    eval_batch.t,
                    eval_batch.x_0,
                    temperature=args.temperature,
                )
                flat_p = torch.cat(preds, dim=-1)
                flat_t = torch.cat(targets, dim=-1)
                teacher_loss = F.huber_loss(flat_p, flat_t, delta=1.0)
                total_teacher_loss += float(teacher_loss.item())
                n_teacher_batches += 1

                x_t_full = ((1 - eval_batch.t) * eval_batch.x_0 + eval_batch.t * eval_batch.x_1).float()
                teacher_full = eval_batch.x_1.clone()
                for b in range(eval_batch.x_1.shape[0]):
                    m_idx = eval_batch.mask_indicator[b].nonzero(as_tuple=True)[0]
                    if len(m_idx) == 0:
                        continue
                    teacher_full[b, m_idx] = x_t_full[b, m_idx] + preds[b].float() * (1 - eval_batch.t[b])

                pred_list, true_list = _masked_vectors(teacher_full, eval_batch.x_1, eval_batch.mask_indicator)
                teacher_pred_all.extend(pred_list)
                teacher_true_all.extend(true_list)

            if args.mode in ("rollout", "both"):
                observed = eval_batch.x_1.clone()
                observed[eval_batch.mask_indicator == 1] = 0.0
                rollout_full = model.inference(
                    eval_batch.gene_ids,
                    observed,
                    eval_batch.mask_indicator,
                    n_steps=args.n_steps,
                    temperature=args.inference_temperature,
                )

                mask = eval_batch.mask_indicator == 1
                if mask.any():
                    rollout_loss = F.huber_loss(rollout_full[mask], eval_batch.x_1[mask], delta=1.0)
                    total_rollout_loss += float(rollout_loss.item())
                    n_rollout_batches += 1

                pred_list, true_list = _masked_vectors(rollout_full, eval_batch.x_1, eval_batch.mask_indicator)
                rollout_pred_all.extend(pred_list)
                rollout_true_all.extend(true_list)

    result = {
        "mode": args.mode,
        "split": args.split,
        "data": args.data,
        "checkpoint": args.checkpoint,
        "eval_seed": args.eval_seed,
        "t_mode": args.t_mode,
        "fixed_t": args.fixed_t,
        "nonzero_mask_rate": args.nonzero_mask_rate,
        "zero_mask_rate": args.zero_mask_rate,
        "temperature": args.temperature,
        "n_steps": args.n_steps,
        "inference_temperature": args.inference_temperature,
    }

    if args.mode in ("teacher", "both"):
        result["teacher_loss"] = total_teacher_loss / max(n_teacher_batches, 1)
        result["teacher_corr"] = _summarize_correlations(teacher_pred_all, teacher_true_all)
        result["teacher_batches"] = n_teacher_batches

    if args.mode in ("rollout", "both"):
        result["rollout_loss"] = total_rollout_loss / max(n_rollout_batches, 1)
        result["rollout_corr"] = _summarize_correlations(rollout_pred_all, rollout_true_all)
        result["rollout_batches"] = n_rollout_batches

    return result


def load_model(args, cfg, device):
    model_cfg = cfg["model"]
    model = scRNAModel(
        n_genes=model_cfg["n_genes"],
        d_model=model_cfg["d_model"],
        grid_size=model_cfg["grid_size"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    return model


def build_result_path(args):
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    filename = (
        f"deterministic_eval_{ckpt_stem}_{args.split}_{args.mode}_"
        f"seed{args.eval_seed}_nz{args.nonzero_mask_rate:.2f}_z{args.zero_mask_rate:.2f}.json"
    )
    return os.path.join(result_dir, filename.replace("/", "_"))


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic reconstruction evaluation for scUniFlow")
    parser.add_argument("--checkpoint", required=True, help="checkpoint path, e.g. phase3_final.pt")
    parser.add_argument("--config", default="./config.yaml", help="config yaml path")
    parser.add_argument("--data", default="mtg_full", help="dataset key or h5ad path")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val", help="evaluation split")
    parser.add_argument("--batch_size", type=int, default=8, help="evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--max_batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--mode", choices=["teacher", "rollout", "both"], default="both")
    parser.add_argument("--temperature", type=float, default=0.1, help="routing temperature for teacher-forced eval")
    parser.add_argument("--n_steps", type=int, default=10, help="ODE rollout steps")
    parser.add_argument("--inference_temperature", type=float, default=0.01, help="routing temperature for inference")
    parser.add_argument("--eval_seed", type=int, default=1234, help="base seed for deterministic masks and x0")
    parser.add_argument("--fixed_t", type=float, default=0.5, help="fixed t value when t_mode=fixed")
    parser.add_argument("--t_mode", choices=["fixed", "random"], default="fixed", help="fixed is recommended for fair comparison")
    parser.add_argument("--nonzero_mask_rate", type=float, default=0.20, help="mask rate for nonzero genes")
    parser.add_argument("--zero_mask_rate", type=float, default=0.02, help="mask rate for zero genes")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR, help="directory to save metrics json when --save_json is not set")
    parser.add_argument("--save_json", default="", help="optional explicit path to save metrics json")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.eval_seed)
    np.random.seed(args.eval_seed)

    loader = build_loader(args, cfg)
    model = load_model(args, cfg, device)
    result = evaluate(model, loader, device, args)
    result_path = os.path.abspath(args.save_json) if args.save_json else build_result_path(args)
    result["result_json"] = result_path

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    with open(result_path, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
