#!/usr/bin/env python3
"""
Generate augmented spectral coordinates using local heterogeneous network data only.

This wraps step8_augmented_graph.py but intentionally skips the Ensembl REST
genomic-proximity step so the workflow remains reproducible in offline or
restricted-network environments.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from step8_augmented_graph import (
    STRING_THR,
    RESULTS_DIR,
    add_go_edges,
    add_gtex_edges,
    assemble_coords,
    assign_border_positions,
    build_string_graph,
    default_output_paths,
    load_model_genes,
    spectral_embed,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented gene spectral coords from local STRING/GTEx/GO data"
    )
    parser.add_argument("--h5ad", type=str, required=True, help="Input h5ad path")
    parser.add_argument(
        "--grid_size",
        type=int,
        required=True,
        help="Grid size, must match scUniFlow model.grid_size",
    )
    parser.add_argument("--output_npy", type=str, default=None)
    parser.add_argument("--output_meta", type=str, default=None)
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad).resolve()
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad does not exist: {h5ad_path}")

    default_npy, default_meta = default_output_paths(h5ad_path)
    out_npy = Path(args.output_npy).resolve() if args.output_npy else default_npy
    out_meta = Path(args.output_meta).resolve() if args.output_meta else default_meta
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    print("=" * 65)
    print("Local augmented graph -> spectral coords for all genes")
    print("=" * 65)
    print(f"[input] h5ad={h5ad_path}")
    print(f"[input] grid_size={args.grid_size}")

    sorted_ids, symbol_of = load_model_genes(h5ad_path)

    print("\n[Case A] STRING thr=200")
    G_graph, isolated_after_string = build_string_graph(sorted_ids, symbol_of)

    print("\n[Case B1] GTEx brain co-expression")
    G_graph, isolated_after_gtex = add_gtex_edges(G_graph, isolated_after_string)

    print("\n[Case B3] GO BP Jaccard similarity")
    G_graph, isolated_final = add_go_edges(G_graph, isolated_after_gtex, symbol_of)

    print("\n[summary] After local augmentation:")
    print(f"  after STRING: {len(sorted_ids) - len(isolated_after_string):,} connected")
    print(f"  after GTEx:   {len(sorted_ids) - len(isolated_after_gtex):,} connected")
    print(f"  after GO:     {len(sorted_ids) - len(isolated_final):,} connected")
    print(f"  isolated border genes: {len(isolated_final):,}")

    print("\n[Spectral embedding]")
    coords_dict = spectral_embed(G_graph, grid_size=int(args.grid_size))

    print(f"\n[Case C] Placing {len(isolated_final):,} isolated genes on grid border")
    border_dict = assign_border_positions(isolated_final, grid_size=int(args.grid_size))

    coords = assemble_coords(
        sorted_ids,
        coords_dict=coords_dict,
        border_dict=border_dict,
        grid_size=int(args.grid_size),
    )
    np.save(out_npy, coords)

    meta = {
        "source_h5ad": str(h5ad_path),
        "grid_size": int(args.grid_size),
        "n_genes": int(len(sorted_ids)),
        "string_thr": int(STRING_THR),
        "augmentation_mode": "local_string_gtex_go",
        "used_sources": ["STRING", "GTEx_brain", "GO_BP"],
        "skipped_sources": ["genomic_proximity_ensembl_rest"],
        "n_connected_after_string": int(len(sorted_ids) - len(isolated_after_string)),
        "n_connected_after_gtex": int(len(sorted_ids) - len(isolated_after_gtex)),
        "n_connected_after_go": int(len(sorted_ids) - len(isolated_final)),
        "n_isolated_border": int(len(isolated_final)),
        "coord_shape": list(coords.shape),
        "coord_min": float(coords.min()) if coords.size > 0 else 0.0,
        "coord_max": float(coords.max()) if coords.size > 0 else 0.0,
        "sorted_gene_list": sorted_ids,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[saved] {out_npy} shape={coords.shape}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()
