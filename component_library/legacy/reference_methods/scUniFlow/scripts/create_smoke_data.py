#!/usr/bin/env python
"""Create tiny synthetic inputs for the pretraining smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic smoke-test data.")
    parser.add_argument("--out-dir", default="data/smoke", help="Directory for smoke.h5ad.")
    parser.add_argument(
        "--coords-dir",
        default="data/gene_coords",
        help="Directory for smoke_coords.npy.",
    )
    parser.add_argument("--n-cells", type=int, default=8)
    parser.add_argument("--n-genes", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_genes > args.grid_size * args.grid_size:
        raise ValueError("n_genes must fit into grid_size * grid_size")

    out_dir = Path(args.out_dir)
    coords_dir = Path(args.coords_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    counts = rng.poisson(lam=2.0, size=(args.n_cells, args.n_genes)).astype(np.float32)
    counts[counts < 1] = 0

    split = max(1, args.n_cells // 2)
    donors = ["smoke_donor_a"] * split + ["smoke_donor_b"] * (args.n_cells - split)
    obs = pd.DataFrame(
        {"donor_id": donors},
        index=[f"smoke_cell_{i}" for i in range(args.n_cells)],
    )
    var = pd.DataFrame(
        {"feature_name": [f"SMOKE_GENE_{i}" for i in range(args.n_genes)]},
        index=[f"ENSGSMOKE{i:06d}" for i in range(args.n_genes)],
    )
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)

    h5ad_path = out_dir / "smoke.h5ad"
    adata.write_h5ad(h5ad_path)

    coords = np.array(
        [(i // args.grid_size, i % args.grid_size) for i in range(args.n_genes)],
        dtype=np.float32,
    )
    coords_path = coords_dir / "smoke_coords.npy"
    np.save(coords_path, coords)

    print(f"wrote {h5ad_path}")
    print(f"wrote {coords_path}")


if __name__ == "__main__":
    main()
