"""
Step 9: Generate deterministic routing coordinates from the current gene order.

This script does NOT use the protein heterogeneity graph. Instead, it creates
an external coordinate file that can be consumed by hard-routing training.

The output indexing follows the same convention as scUniFlow gene_ids:
rank order after sorting Ensembl IDs alphabetically, i.e.
gene_ids = argsort(argsort(ensembl_ids)).

Examples
--------
python ./string_spectral/step9_ordered_coords.py \
    --h5ad data/pbmc68k_raw_counts.h5ad \
    --grid_size 181 \
    --order_mode row_major

python ./string_spectral/step9_ordered_coords.py \
    --h5ad data/pbmc68k_raw_counts.h5ad \
    --order_mode hilbert \
    --output_npy ./data/gene_coords/pbmc68k_ordered_hilbert_coords.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from model import build_ordered_gene_coords


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_sorted_gene_ids(h5ad_path: Path):
    adata = sc.read_h5ad(h5ad_path, backed="r")
    ensembl_ids = np.asarray(adata.var_names).astype(str)
    sorted_ids = sorted(ensembl_ids.tolist())

    feature_names = None
    if "feature_name" in adata.var.columns:
        feature_series = adata.var["feature_name"].astype(str)
        feature_name_map = {
            str(eid): str(name)
            for eid, name in zip(adata.var_names.astype(str), feature_series)
        }
        feature_names = [feature_name_map.get(eid, "") for eid in sorted_ids]

    return sorted_ids, feature_names


def default_output_paths(h5ad_path: Path, order_mode: str):
    stem = h5ad_path.stem
    out_npy = RESULTS_DIR / f"{stem}_ordered_{order_mode}_coords.npy"
    out_meta = RESULTS_DIR / f"{stem}_ordered_{order_mode}_meta.json"
    return out_npy, out_meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard-routing coords from the current gene order"
    )
    parser.add_argument("--h5ad", type=str, required=True,
                        help="Input h5ad path")
    parser.add_argument("--grid_size", type=int, default=181,
                        help="Grid size, must match scUniFlow model.grid_size")
    parser.add_argument("--order_mode", type=str, default="row_major",
                        choices=["row_major", "hilbert"],
                        help="How to place the existing gene order onto the 2-D grid")
    parser.add_argument("--output_npy", type=str, default=None,
                        help="Optional output .npy path")
    parser.add_argument("--output_meta", type=str, default=None,
                        help="Optional output .json metadata path")
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad).resolve()
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad does not exist: {h5ad_path}")

    default_npy, default_meta = default_output_paths(h5ad_path, args.order_mode)
    out_npy = Path(args.output_npy).resolve() if args.output_npy else default_npy
    out_meta = Path(args.output_meta).resolve() if args.output_meta else default_meta
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    sorted_ids, feature_names = load_sorted_gene_ids(h5ad_path)
    coords = build_ordered_gene_coords(
        n_genes=len(sorted_ids),
        grid_size=int(args.grid_size),
        order_mode=args.order_mode,
    )

    np.save(out_npy, coords)

    meta = {
        "source_h5ad": str(h5ad_path),
        "generation_mode": "ordered_gene_coords",
        "order_mode": args.order_mode,
        "grid_size": int(args.grid_size),
        "n_genes": int(len(sorted_ids)),
        "coord_shape": list(coords.shape),
        "coord_min": float(coords.min()) if coords.size > 0 else 0.0,
        "coord_max": float(coords.max()) if coords.size > 0 else 0.0,
        "indexing": "sorted_ensembl_rank",
        "sorted_gene_ids": sorted_ids,
    }
    if feature_names is not None:
        meta["sorted_feature_names"] = feature_names

    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[saved] {out_npy}  shape={coords.shape}")
    print(f"[saved] {out_meta}")
    print(f"[info] order_mode={args.order_mode} grid_size={args.grid_size} n_genes={len(sorted_ids)}")


if __name__ == "__main__":
    main()
