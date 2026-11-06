import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import json
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
import yaml
from scipy.sparse import issparse
from tqdm.auto import tqdm

import sys
sys.path.insert(0, ".")

from model import scRNAModel
from donor_eval_utils import (
    DEFAULT_DONOR_CSV,
    DEFAULT_DONOR_DIR,
    pick_eval_donor,
    sample_balanced_val_donors,
    sample_single_donor_cells,
    split_val_donors,
)
from masked_recon_metrics import (
    build_topk_nonzero_eval_mask,
    init_metric_accumulator,
    summarize_metric_accumulator,
    update_metric_accumulator,
)

DEFAULT_RESULT_DIR = Path("results/pretrain_eval")


def make_gene_ids(ensembl_ids: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(ensembl_ids)).astype(np.int64)


def load_donor_rows(donor_csv: Path):
    rows = []
    with donor_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["donor_id"], int(row["n_cells"])))
    return rows


def resolve_train_donors(exp_dir: Path, donor_csv: Path, split_seed: int):
    split_info_path = exp_dir / "split_info.json"
    if split_info_path.exists():
        with split_info_path.open() as f:
            split_info = json.load(f)
        train_donors = split_info.get("train_donors")
        if isinstance(train_donors, list) and train_donors:
            return train_donors, "exp_split_info"
        print(
            "[prior] split_info.json exists but does not contain explicit train donor ids; "
            "falling back to split_seed reconstruction",
            flush=True,
        )

    all_donor_ids = [did for did, _ in load_donor_rows(donor_csv)]
    val_donors = {did for did, _ in split_val_donors(donor_csv, val_ratio=0.1, seed=split_seed)}
    train_donors = [did for did in all_donor_ids if did not in val_donors]
    if not train_donors:
        raise RuntimeError("Failed to reconstruct train donors from donor CSV and split_seed.")
    return train_donors, "split_seed_reconstructed"


def compute_log1p_nonzero_prior_stats_from_donors(
    donor_ids,
    donor_dir: Path,
    no_progress: bool = False,
):
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0

    donor_iter = donor_ids
    if not no_progress:
        donor_iter = tqdm(donor_ids, total=len(donor_ids), desc="train prior", unit="donor")

    for donor_id in donor_iter:
        donor_path = donor_dir / f"{donor_id}.h5ad"
        if not donor_path.exists():
            raise FileNotFoundError(f"Training donor file not found: {donor_path}")

        adata = sc.read_h5ad(donor_path)
        X = adata.X
        if issparse(X):
            nz_raw = X.data.astype(np.float64, copy=False)
        else:
            arr = np.asarray(X, dtype=np.float32)
            nz_raw = arr[arr > 0].astype(np.float64, copy=False)

        if nz_raw.size > 0:
            nz_log = np.log1p(nz_raw)
            total_count += int(nz_log.size)
            total_sum += float(nz_log.sum())
            total_sumsq += float(np.square(nz_log).sum())

        del adata

    if total_count == 0:
        return 0.0, 1.0, 0

    mean = total_sum / total_count
    var = max(total_sumsq / total_count - mean * mean, 0.0)
    std = max(var ** 0.5, 1e-6)
    return float(mean), float(std), int(total_count)


def random_mask_mixed(
    xb: np.ndarray,
    nonzero_ratio: float,
    zero_ratio: float,
    base_seed: int,
    global_offset: int = 0,
):
    """
    Mask non-zero and zero genes for each cell, using per-cell seed alignment:
    seed = base_seed + global_cell_index
    """
    bs, n_genes = xb.shape
    m = np.zeros((bs, n_genes), dtype=np.float32)
    for i in range(bs):
        rng = np.random.RandomState(base_seed + global_offset + i)
        nz = np.where(xb[i] > 0)[0]
        z = np.where(xb[i] <= 0)[0]
        n_mask_nz = int(len(nz) * nonzero_ratio)
        if n_mask_nz > 0 and len(nz) > 0:
            idx_nz = rng.choice(nz, size=n_mask_nz, replace=False)
            m[i, idx_nz] = 1.0
        n_mask_z = int(len(z) * zero_ratio)
        if n_mask_z > 0 and len(z) > 0:
            idx_z = rng.choice(z, size=n_mask_z, replace=False)
            m[i, idx_z] = 1.0
    return m


def build_result_path(args, checkpoint: Path, donor_id: str, sampled_cells: int, per_donor_n: int) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    exp_name = Path(args.exp_dir).name
    if args.eval_subset_mode == "balanced_val_donors":
        filename = (
            f"scuniflow_recon_{exp_name}_{checkpoint.stem}_{donor_id}_"
            f"perdonor{per_donor_n}_total{sampled_cells}_"
            f"masknz{args.nonzero_mask_ratio:.2f}_maskz{args.zero_mask_ratio:.2f}.json"
        )
    else:
        filename = (
            f"scuniflow_recon_{exp_name}_{checkpoint.stem}_{donor_id}_"
            f"n{sampled_cells}_masknz{args.nonzero_mask_ratio:.2f}_maskz{args.zero_mask_ratio:.2f}.json"
        )
    return result_dir / filename.replace("/", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="results/pretrain/flowv5")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--balanced_per_donor_n", type=int, default=0, help="If >0, cap balanced_val_donors sampling to this many cells per donor.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nonzero_mask_ratio", type=float, default=0.2)
    parser.add_argument("--zero_mask_ratio", type=float, default=0.0)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--mask_seed", type=int, default=1234)
    parser.add_argument(
        "--fair_topk_nonzero",
        type=int,
        default=1200,
        help="Also report fair-comparison metrics restricted to the top-K nonzero genes by true expression per cell. Use <=0 to disable.",
    )
    parser.add_argument("--eval_donor", type=str, default="")
    parser.add_argument(
        "--eval_subset_mode",
        type=str,
        default="balanced_val_donors",
        choices=["balanced_val_donors", "single_donor"],
        help="balanced_val_donors: sample the same number of cells from every val donor; single_donor: evaluate one donor only.",
    )
    parser.add_argument("--amp", action="store_true", help="use autocast in inference")
    parser.add_argument("--no_progress", action="store_true", help="disable progress bar")
    parser.add_argument("--debug_step0_stats", action="store_true", help="print first-step inference stats")
    parser.add_argument("--prior_mean", type=float, default=None, help="x0 prior mean in log1p space")
    parser.add_argument("--prior_std", type=float, default=None, help="x0 prior std in log1p space")
    parser.add_argument("--velocity_bias", type=float, default=0.0, help="additive bias correction to v_pred at each Euler step")
    parser.add_argument("--init_from_x1_true", action="store_true", help="diagnostic only: initialize masked x_t from x1_true")
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    if (args.prior_mean is None) != (args.prior_std is None):
        raise ValueError("Please provide both --prior_mean and --prior_std together, or neither.")

    donor_csv = DEFAULT_DONOR_CSV
    donor_dir = DEFAULT_DONOR_DIR

    # 默认优先使用新的单阶段 epoch checkpoint，同时兼容旧版 phase2 命名
    exp_dir = Path(args.exp_dir)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        checkpoint_last = exp_dir / "checkpoints/checkpoint_last.pt"
        checkpoint_legacy = exp_dir / "checkpoints/phase2_final.pt"
        checkpoint = checkpoint_last if checkpoint_last.exists() else checkpoint_legacy
    cfg_path = exp_dir / "config.yaml"

    sample_n = args.sample_n
    split_seed = args.split_seed
    sample_seed = args.sample_seed
    mask_seed = args.mask_seed
    nonzero_mask_ratio = args.nonzero_mask_ratio
    zero_mask_ratio = args.zero_mask_ratio
    batch_size = args.batch_size

    if args.eval_subset_mode == "balanced_val_donors":
        adata, donor_stats, per_donor_n = sample_balanced_val_donors(
            donor_csv=donor_csv,
            donor_dir=donor_dir,
            split_seed=split_seed,
            sample_seed=sample_seed,
            concat_join="outer",
            per_donor_n=args.balanced_per_donor_n,
        )
        donor_id = "balanced_val_donors"
        donor_cells = int(sum(x["donor_cells_total"] for x in donor_stats))
        n = int(adata.n_obs)
    else:
        if args.eval_donor:
            donor_id = args.eval_donor
            donor_cells = sc.read_h5ad(donor_dir / f"{donor_id}.h5ad", backed="r").n_obs
        else:
            val_donors = split_val_donors(donor_csv, val_ratio=0.1, seed=split_seed)
            donor_id, donor_cells = pick_eval_donor(val_donors, donor_dir, min_cells=sample_n)
        adata, _, n = sample_single_donor_cells(
            donor_path=donor_dir / f"{donor_id}.h5ad",
            sample_n=sample_n,
            sample_seed=sample_seed,
        )
        donor_stats = [
            {
                "donor_id": donor_id,
                "donor_cells_total": int(donor_cells),
                "sampled_cells": int(n),
                "sample_seed": int(sample_seed),
            }
        ]
        per_donor_n = int(n)

    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    X_log = np.log1p(X)

    ensembl_ids = np.array(adata.var.index).astype(str)
    gene_ids = make_gene_ids(ensembl_ids)

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = scRNAModel(
        n_genes=model_cfg["n_genes"],
        d_model=model_cfg["d_model"],
        grid_size=model_cfg["grid_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # x0 prior：显式 CLI 覆盖优先；否则基于训练 donor 集实时重算，避免依赖 checkpoint 内部统计量
    prior_nonzero_count = None
    if args.prior_mean is not None and args.prior_std is not None:
        prior_mean = float(args.prior_mean)
        prior_std  = float(max(args.prior_std, 1e-6))
        prior_source = "cli"
    else:
        train_donors, prior_split_source = resolve_train_donors(
            exp_dir=exp_dir,
            donor_csv=donor_csv,
            split_seed=split_seed,
        )
        prior_mean, prior_std, prior_nonzero_count = compute_log1p_nonzero_prior_stats_from_donors(
            donor_ids=train_donors,
            donor_dir=donor_dir,
            no_progress=args.no_progress,
        )
        prior_source = f"train_data:{prior_split_source}"

    print(f"[prior] source={prior_source}, mean={prior_mean:.6f}, std={prior_std:.6f}")

    # 评估：固定 mask；同时输出全 masked positions 和公平对比 top-K nonzero masked positions 的指标
    overall_stats = init_metric_accumulator()
    fair_stats = init_metric_accumulator() if args.fair_topk_nonzero > 0 else None

    total_batches = (n + batch_size - 1) // batch_size
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(
        f"[eval] donor={donor_id} subset_mode={args.eval_subset_mode} sampled_cells={n} batch_size={batch_size} "
        f"total_batches={total_batches} n_steps={args.n_steps} amp={args.amp} "
        f"amp_dtype={str(amp_dtype).replace('torch.', '') if args.amp else 'disabled'}"
    )

    with torch.inference_mode():
        starts = range(0, n, batch_size)
        if not args.no_progress:
            starts = tqdm(starts, total=total_batches, desc="scUniFlow eval", unit="batch")
        for start in starts:
            end = min(start + batch_size, n)
            xb = X_log[start:end]
            bs = end - start

            mb = random_mask_mixed(
                xb,
                nonzero_ratio=nonzero_mask_ratio,
                zero_ratio=zero_mask_ratio,
                base_seed=mask_seed,
                global_offset=start,
            )

            x1 = torch.from_numpy(xb).to(device)
            m = torch.from_numpy(mb).to(device)
            g = torch.from_numpy(np.tile(gene_ids, (bs, 1))).long().to(device)

            observed = x1.clone()
            observed[m == 1] = 0.0

            with torch.amp.autocast(
                "cuda",
                dtype=amp_dtype,
                enabled=(args.amp and torch.cuda.is_available()),
            ):
                pred = model.inference(
                    g,
                    observed,
                    m,
                    n_steps=args.n_steps if args.n_steps > 0 else cfg["inference"].get("n_steps", 10),
                    temperature=args.temperature if args.temperature > 0 else cfg["inference"].get("temperature", 0.01),
                    x1_true=x1,
                    debug_step0_stats=args.debug_step0_stats,
                    prior_mean=prior_mean,
                    prior_std=prior_std,
                    velocity_bias=args.velocity_bias,
                    init_from_x1_true=args.init_from_x1_true,
                )

            mask_bool = m == 1
            pred_cpu = pred.detach().cpu().numpy().astype(np.float32)
            true_cpu = x1.detach().cpu().numpy().astype(np.float32)
            mask_cpu = mask_bool.detach().cpu().numpy()
            update_metric_accumulator(overall_stats, pred_cpu=pred_cpu, true_cpu=true_cpu, mask_cpu=mask_cpu)

            if fair_stats is not None:
                fair_mask_cpu = build_topk_nonzero_eval_mask(
                    true_cpu=true_cpu,
                    base_mask_cpu=mask_cpu,
                    top_k=args.fair_topk_nonzero,
                )
                update_metric_accumulator(
                    fair_stats,
                    pred_cpu=pred_cpu,
                    true_cpu=true_cpu,
                    mask_cpu=fair_mask_cpu,
                )

    overall_metrics = summarize_metric_accumulator(overall_stats)
    fair_metrics = summarize_metric_accumulator(fair_stats) if fair_stats is not None else None

    out = {
        "checkpoint": str(checkpoint),
        "dataset": "SEA-AD MTG donor_h5ad",
        "donor_id": donor_id,
        "donor_cells_total": int(donor_cells),
        "eval_subset_mode": args.eval_subset_mode,
        "balanced_per_donor_cells": int(per_donor_n),
        "donor_sampling": donor_stats,
        "sampled_cells": int(n),
        "split_rule": f"scUniFlow donor split val_ratio=0.1 seed={split_seed}",
        "unseen_for_scuniflow_train": True,
        "n_genes": int(X_log.shape[1]),
        "nonzero_mask_ratio_target": nonzero_mask_ratio,
        "zero_mask_ratio_target": zero_mask_ratio,
        "mask_policy": "mixed_nonzero_zero_per_cell",
        "batch_size": batch_size,
        "n_steps": args.n_steps,
        "temperature": args.temperature,
        "amp": bool(args.amp),
        "fair_topk_nonzero": int(args.fair_topk_nonzero),
        "x0_prior": "custom",
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "prior_source": prior_source,
        "prior_nonzero_count": int(prior_nonzero_count) if prior_nonzero_count is not None else None,
        "velocity_bias": float(args.velocity_bias),
        "init_from_x1_true": bool(args.init_from_x1_true),
        "masked_l2_rmse": overall_metrics["l2_rmse"],
        "masked_mse": overall_metrics["mse"],
        "masked_mae": overall_metrics["mae"],
        "masked_pearson_all": overall_metrics["pearson_all"],
        "masked_pearson_cell_mean": overall_metrics["pearson_cell_mean"],
        "masked_spearman_cell_mean": overall_metrics["spearman_cell_mean"],
        "masked_pearson_top20_cell_mean": overall_metrics["pearson_top20_cell_mean"],
        "n_cells_for_corr": overall_metrics["n_cells_for_corr"],
        "masked_n_eval_positions": overall_metrics["n_eval_positions"],
    }
    if fair_metrics is not None:
        out.update(
            {
                f"masked_top{args.fair_topk_nonzero}_nonzero_l2_rmse": fair_metrics["l2_rmse"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_mse": fair_metrics["mse"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_mae": fair_metrics["mae"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_pearson_all": fair_metrics["pearson_all"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_pearson_cell_mean": fair_metrics["pearson_cell_mean"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_spearman_cell_mean": fair_metrics["spearman_cell_mean"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_pearson_top20_cell_mean": fair_metrics["pearson_top20_cell_mean"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_n_cells_for_corr": fair_metrics["n_cells_for_corr"],
                f"masked_top{args.fair_topk_nonzero}_nonzero_n_eval_positions": fair_metrics["n_eval_positions"],
            }
        )

    result_path = build_result_path(
        args,
        checkpoint=checkpoint,
        donor_id=donor_id,
        sampled_cells=n,
        per_donor_n=per_donor_n,
    )
    out["result_json"] = str(result_path)
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    print(f"[done] wrote result json to {result_path}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
