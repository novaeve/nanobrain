import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import argparse
import json
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from donor_eval_utils import sample_balanced_val_donors


DEFAULT_DONOR_CSV = ROOT / "donor_cell_counts_full_mtg.csv"
DEFAULT_DONOR_DIR = Path("data/external/MTG/donor_h5ad")
DEFAULT_OUT_DIR = Path("data")


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
                for idx, cnt in counts.head(10).items()
            ],
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a donor-balanced SEA-AD MTG subset for read depth enhancement evaluation."
    )
    parser.add_argument("--donor_csv", default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--per_donor_n", type=int, default=512)
    parser.add_argument("--join", default="inner", choices=["inner", "outer"])
    parser.add_argument("--subset_name", default="seaad_mtg_rde_eval_balanced_val_512pd")
    parser.add_argument("--output_h5ad", default="")
    parser.add_argument("--output_manifest", default="")
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
    output_h5ad = (
        Path(args.output_h5ad).resolve()
        if args.output_h5ad
        else (out_dir / f"{args.subset_name}.h5ad")
    )
    output_manifest = (
        Path(args.output_manifest).resolve()
        if args.output_manifest
        else output_h5ad.with_suffix(".manifest.json")
    )

    adata, donor_stats, per_donor_n = sample_balanced_val_donors(
        donor_csv=donor_csv,
        donor_dir=donor_dir,
        split_seed=int(args.split_seed),
        sample_seed=int(args.sample_seed),
        concat_join=str(args.join),
        per_donor_n=int(args.per_donor_n),
        donor_obs_key="_rde_eval_donor_id",
        batch_key="_rde_eval_batch",
        progress_desc="loading SEA-AD eval donors",
    )

    adata.obs = adata.obs.copy()
    adata.obs["_rde_subset_name"] = str(args.subset_name)
    adata.obs["_rde_subset_role"] = "eval"
    adata.obs["_rde_subset_source"] = "SEA-AD MTG balanced validation donors"

    obs_summary = summarize_obs(
        adata,
        columns=[
            "donor_id",
            "Class",
            "Subclass",
            "Supertype",
            "cell_type",
            "disease",
            "sex",
            "tissue",
        ],
    )

    manifest = {
        "subset_name": str(args.subset_name),
        "source": "SEA-AD MTG donor_h5ad",
        "output_h5ad": str(output_h5ad),
        "output_manifest": str(output_manifest),
        "shape": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
        },
        "split_policy": {
            "mode": "balanced_val_donors",
            "split_seed": int(args.split_seed),
            "sample_seed": int(args.sample_seed),
            "per_donor_n": int(per_donor_n),
            "join": str(args.join),
        },
        "donor_stats": donor_stats,
        "obs_summary": obs_summary,
        "var_columns": list(map(str, adata.var.columns.tolist())),
    }

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad)
    with output_manifest.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[saved] {output_h5ad}  shape={adata.shape}")
    print(f"[saved] {output_manifest}")
    print(
        f"[info] donors={len(donor_stats)} per_donor_n={per_donor_n} "
        f"total_cells={adata.n_obs} genes={adata.n_vars}"
    )


if __name__ == "__main__":
    main()
