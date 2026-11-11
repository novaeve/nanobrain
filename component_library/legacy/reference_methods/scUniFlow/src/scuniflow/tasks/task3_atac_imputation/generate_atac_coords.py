import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import scanpy as sc


HUMAN_CHROM_LENGTHS = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
}
MOUSE_CHROM_LENGTHS = {
    "chr1": 195471971,
    "chr2": 182113224,
    "chr3": 160039680,
    "chr4": 156508116,
    "chr5": 151834684,
    "chr6": 149736546,
    "chr7": 145441459,
    "chr8": 129401213,
    "chr9": 124595110,
    "chr10": 130694993,
    "chr11": 122082543,
    "chr12": 120129022,
    "chr13": 120421639,
    "chr14": 124902244,
    "chr15": 104043685,
    "chr16": 98207768,
    "chr17": 94987271,
    "chr18": 90702639,
    "chr19": 61431566,
    "chrX": 171031299,
    "chrY": 91744698,
}
GENOME_SPECS = {
    "hg38": {
        "chrom_lengths": HUMAN_CHROM_LENGTHS,
        "chrom_order": [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
    },
    "mm10": {
        "chrom_lengths": MOUSE_CHROM_LENGTHS,
        "chrom_order": [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"],
    },
}


def parse_peak_name(peak_name: str):
    chrom, pos = str(peak_name).rsplit(":", 1)
    start, end = pos.split("-")
    return chrom, int(start), int(end)


def main():
    parser = argparse.ArgumentParser(
        description="Generate grid-scaled scATAC routing coordinates for scUniFlow"
    )
    parser.add_argument("--h5ad", required=True, help="Input TF-IDF h5ad path")
    parser.add_argument("--grid_size", type=int, required=True, help="Must match model.grid_size")
    parser.add_argument("--output_npy", required=True, help="Output coords .npy path")
    parser.add_argument("--output_meta", default=None, help="Optional metadata .json path")
    parser.add_argument(
        "--genome",
        choices=sorted(GENOME_SPECS.keys()),
        default="hg38",
        help="Reference genome for chromosome lengths/order",
    )
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad).resolve()
    output_npy = Path(args.output_npy).resolve()
    output_meta = (
        Path(args.output_meta).resolve()
        if args.output_meta is not None
        else output_npy.with_suffix(".json")
    )
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(h5ad_path, backed="r")
    n_peaks = adata.n_vars
    grid_size = int(args.grid_size)
    coords = np.zeros((n_peaks, 2), dtype=np.float32)
    unknown_count = 0
    genome_spec = GENOME_SPECS[str(args.genome)]
    chrom_lengths = genome_spec["chrom_lengths"]
    chrom_order = genome_spec["chrom_order"]
    chrom_rank = {chrom: idx for idx, chrom in enumerate(chrom_order)}

    var = adata.var
    has_columns = "chrom" in var.columns and "start" in var.columns

    for idx, peak_name in enumerate(adata.var_names.astype(str)):
        if has_columns:
            chrom = str(var.iloc[idx]["chrom"])
            start = int(var.iloc[idx]["start"])
        else:
            chrom, start, _ = parse_peak_name(peak_name)

        if chrom not in chrom_rank or chrom not in chrom_lengths:
            coords[idx] = np.array([grid_size - 1, grid_size - 1], dtype=np.float32)
            unknown_count += 1
            continue

        chrom_frac = chrom_rank[chrom] / max(len(chrom_order) - 1, 1)
        pos_frac = start / max(chrom_lengths[chrom], 1)
        coords[idx, 0] = np.clip(chrom_frac * (grid_size - 1), 0.0, grid_size - 1)
        coords[idx, 1] = np.clip(pos_frac * (grid_size - 1), 0.0, grid_size - 1)

    np.save(output_npy, coords)

    meta = {
        "source_h5ad": str(h5ad_path),
        "grid_size": grid_size,
        "n_peaks": int(n_peaks),
        "coord_shape": list(coords.shape),
        "coord_min": float(coords.min()) if coords.size > 0 else 0.0,
        "coord_max": float(coords.max()) if coords.size > 0 else 0.0,
        "unknown_chrom_count": int(unknown_count),
        "coord_space": "grid_scaled_genomic",
        "genome": str(args.genome),
        "note": "Coordinates are scaled to [0, grid_size-1] because scUniFlow hard routing expects grid coordinates.",
    }
    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[saved] {output_npy} shape={coords.shape}")
    print(f"[saved] {output_meta}")


if __name__ == "__main__":
    main()
