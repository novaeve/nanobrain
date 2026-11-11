import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import scanpy as sc


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = Path(__file__).resolve().parents[2]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from donor_eval_utils import split_val_donors


DEFAULT_DONOR_CSV = ROOT / "donor_cell_counts_full_mtg.csv"
DEFAULT_DONOR_DIR = Path("data/external/MTG/donor_h5ad")
DEFAULT_OUT_DIR = Path("data")


def read_donor_rows(donor_csv: Path):
    rows = []
    with donor_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((str(row["donor_id"]), int(row["n_cells"])))
    return rows


def choose_donors(rows, donor_dir: Path, exclude_ids, n_donors: int, min_cells: int, seed: int):
    exclude_ids = set(map(str, exclude_ids))
    candidates = [
        (did, n)
        for did, n in rows
        if did not in exclude_ids and n >= int(min_cells) and (donor_dir / f"{did}.h5ad").exists()
    ]
    if not candidates:
        raise RuntimeError("No donor candidates available for requested subset.")
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(candidates))
    selected = [candidates[i] for i in order[: min(int(n_donors), len(candidates))]]
    selected.sort(key=lambda x: x[0])
    return selected


def sample_donor_raw_counts(donor_path: Path, sample_n: int, sample_seed: int, role: str, donor_id: str):
    t0 = time.time()
    print(f"[load] {role} donor {donor_id}: {donor_path}", flush=True)
    adata = sc.read_h5ad(donor_path)
    total_n = int(adata.n_obs)
    n = min(int(sample_n), total_n)
    if n < total_n:
        rng = np.random.RandomState(int(sample_seed))
        idx = np.sort(rng.choice(total_n, size=n, replace=False))
        adata = adata[idx].copy()
    else:
        adata = adata.copy()

    if adata.raw is None:
        raise ValueError(f"{donor_path} has no .raw; cannot build raw-count RDE subset.")
    raw_x = adata.raw.X
    raw_var = adata.raw.var.copy()
    if raw_x.shape[1] != adata.n_vars:
        # SEA-AD MTG donor files currently have aligned raw.var/var, but fail
        # loudly if this changes instead of silently mixing gene spaces.
        if list(raw_var.index.astype(str)) != list(adata.var.index.astype(str)):
            raise ValueError(
                f"raw.X gene space does not match .X for {donor_path}: "
                f"raw={raw_x.shape[1]} var={adata.n_vars}"
            )
    adata.X = raw_x.copy()
    adata.var = raw_var
    adata.raw = None
    adata.obs = adata.obs.copy()
    adata.obs["_rde_subset_role"] = str(role)
    adata.obs["_rde_source_donor_id"] = str(donor_id)
    adata.obs["_rde_sample_seed"] = int(sample_seed)
    print(f"[sample] {role} donor {donor_id}: {n}/{total_n} cells ({time.time() - t0:.1f}s)", flush=True)
    return adata, {"donor_id": str(donor_id), "role": str(role), "total_cells": total_n, "sampled_cells": n}


def summarize_obs(adata, columns):
    out = {}
    for col in columns:
        if col not in adata.obs.columns:
            continue
        values = adata.obs[col].astype(str)
        counts = values.value_counts(dropna=False)
        out[col] = {
            "n_unique": int(values.nunique()),
            "top_values": [
                {"label": str(idx), "count": int(cnt)}
                for idx, cnt in counts.head(12).items()
            ],
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a raw-count SEA-AD MTG train/eval subset for scUniFlow RDE."
    )
    parser.add_argument("--donor_csv", default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=20260502)
    parser.add_argument("--train_donors_n", type=int, default=16)
    parser.add_argument("--val_donors_n", type=int, default=8)
    parser.add_argument("--train_per_donor_n", type=int, default=512)
    parser.add_argument("--val_per_donor_n", type=int, default=512)
    parser.add_argument("--min_cells", type=int, default=512)
    parser.add_argument("--join", default="inner", choices=["inner", "outer"])
    parser.add_argument("--subset_name", default="seaad_mtg_rde_raw_train16x512_val8x512_s42")
    parser.add_argument("--output_h5ad", default="")
    parser.add_argument("--output_manifest", default="")
    parser.add_argument("--output_train_indices", default="")
    parser.add_argument("--output_val_indices", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    donor_csv = Path(args.donor_csv).resolve()
    donor_dir = Path(args.donor_dir).resolve()
    if not donor_csv.exists():
        raise FileNotFoundError(f"donor_csv not found: {donor_csv}")
    if not donor_dir.exists():
        raise FileNotFoundError(f"donor_dir not found: {donor_dir}")

    out_dir = DEFAULT_OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = Path(args.output_h5ad).resolve() if args.output_h5ad else out_dir / f"{args.subset_name}.h5ad"
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else output_h5ad.with_suffix(".manifest.json")
    )
    output_train_indices = (
        Path(args.output_train_indices).resolve()
        if args.output_train_indices
        else output_h5ad.with_suffix(".train_indices.npy")
    )
    output_val_indices = (
        Path(args.output_val_indices).resolve()
        if args.output_val_indices
        else output_h5ad.with_suffix(".val_indices.npy")
    )

    rows = read_donor_rows(donor_csv)
    val_pool = split_val_donors(donor_csv, val_ratio=0.1, seed=int(args.split_seed))
    val_pool_ids = {did for did, _ in val_pool}
    train_donors = choose_donors(
        rows,
        donor_dir=donor_dir,
        exclude_ids=val_pool_ids,
        n_donors=int(args.train_donors_n),
        min_cells=int(args.min_cells),
        seed=int(args.sample_seed),
    )
    val_donors = choose_donors(
        val_pool,
        donor_dir=donor_dir,
        exclude_ids=set(),
        n_donors=int(args.val_donors_n),
        min_cells=int(args.min_cells),
        seed=int(args.sample_seed) + 17,
    )

    adatas = []
    donor_stats = []
    for donor_i, (donor_id, _) in enumerate(train_donors):
        adata, stats = sample_donor_raw_counts(
            donor_dir / f"{donor_id}.h5ad",
            sample_n=int(args.train_per_donor_n),
            sample_seed=int(args.sample_seed) + donor_i,
            role="train",
            donor_id=donor_id,
        )
        adatas.append(adata)
        donor_stats.append(stats)
    val_offset = len(train_donors)
    for donor_i, (donor_id, _) in enumerate(val_donors):
        adata, stats = sample_donor_raw_counts(
            donor_dir / f"{donor_id}.h5ad",
            sample_n=int(args.val_per_donor_n),
            sample_seed=int(args.sample_seed) + val_offset + donor_i,
            role="val",
            donor_id=donor_id,
        )
        adatas.append(adata)
        donor_stats.append(stats)

    print("[concat] merging donor subsets", flush=True)
    adata = (
        adatas[0].concatenate(*adatas[1:], join=str(args.join), batch_key="_rde_concat_batch", index_unique=None)
        if len(adatas) > 1
        else adatas[0]
    )
    adata.obs = adata.obs.copy()
    adata.obs["_rde_subset_name"] = str(args.subset_name)
    roles = adata.obs["_rde_subset_role"].astype(str).to_numpy()
    train_indices = np.where(roles == "train")[0].astype(np.int64)
    val_indices = np.where(roles == "val")[0].astype(np.int64)
    if train_indices.size == 0 or val_indices.size == 0:
        raise RuntimeError(f"empty train/val split: train={train_indices.size}, val={val_indices.size}")

    manifest = {
        "subset_name": str(args.subset_name),
        "source": "SEA-AD MTG donor_h5ad raw.X",
        "output_h5ad": str(output_h5ad),
        "output_manifest": str(output_manifest),
        "output_train_indices": str(output_train_indices),
        "output_val_indices": str(output_val_indices),
        "shape": {"n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)},
        "split_policy": {
            "mode": "train_donors_vs_original_val_donors",
            "split_seed": int(args.split_seed),
            "sample_seed": int(args.sample_seed),
            "train_donors_n": int(len(train_donors)),
            "val_donors_n": int(len(val_donors)),
            "train_per_donor_n": int(args.train_per_donor_n),
            "val_per_donor_n": int(args.val_per_donor_n),
            "join": str(args.join),
        },
        "donor_stats": donor_stats,
        "obs_summary": summarize_obs(
            adata,
            columns=["_rde_subset_role", "donor_id", "Class", "Subclass", "Supertype", "cell_type", "disease", "sex"],
        ),
        "var_columns": list(map(str, adata.var.columns.tolist())),
    }

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    np.save(output_train_indices, train_indices)
    np.save(output_val_indices, val_indices)
    with output_manifest.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[saved] {output_h5ad} shape={adata.shape}", flush=True)
    print(f"[saved] {output_train_indices} n={train_indices.size}", flush=True)
    print(f"[saved] {output_val_indices} n={val_indices.size}", flush=True)
    print(f"[saved] {output_manifest}", flush=True)


if __name__ == "__main__":
    main()
