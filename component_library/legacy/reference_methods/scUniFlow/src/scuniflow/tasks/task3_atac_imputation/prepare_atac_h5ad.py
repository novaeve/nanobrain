import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import scanpy as sc


def parse_peak_name(peak_name: str):
    chrom, pos = str(peak_name).rsplit(":", 1)
    start, end = pos.split("-")
    return chrom, int(start), int(end)


def read_bed_peak_table(bed_path: Path) -> pd.DataFrame:
    bed_df = pd.read_csv(
        bed_path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    bed_df["peak_name"] = (
        bed_df["chrom"].astype(str)
        + ":"
        + bed_df["start"].astype(str)
        + "-"
        + bed_df["end"].astype(str)
    )
    return bed_df


def load_input_matrix(input_path: Path):
    input_path = input_path.resolve()
    if input_path.suffix == ".h5ad":
        adata = sc.read_h5ad(input_path)
    elif input_path.suffix == ".h5":
        adata = sc.read_10x_h5(str(input_path), gex_only=False)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    if "feature_types" in adata.var.columns:
        feature_types = adata.var["feature_types"].astype(str)
        if (feature_types == "Peaks").any():
            adata = adata[:, feature_types == "Peaks"].copy()
    return adata


def sparse_column_variance(matrix):
    matrix = matrix.tocsc().astype(np.float32)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    return mean_sq - np.square(mean)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare scATAC TF-IDF h5ad for scUniFlow scATAC training"
    )
    parser.add_argument("--input", required=True, help="Input 10x .h5 or .h5ad path")
    parser.add_argument("--output", required=True, help="Output TF-IDF .h5ad path")
    parser.add_argument(
        "--peak-bed",
        default=None,
        help="Optional BED with peak locations; if provided, use it as the coordinate source",
    )
    parser.add_argument("--min-peaks-per-cell", type=int, default=200)
    parser.add_argument("--min-cells-per-peak", type=int, default=10)
    parser.add_argument(
        "--top-n-peaks",
        type=int,
        default=None,
        help="Optional top-N variable peaks after TF-IDF preprocessing",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    peak_bed_path = Path(args.peak_bed).resolve() if args.peak_bed else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = load_input_matrix(input_path)
    print(f"[load] raw shape: {adata.shape}")

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    else:
        adata.X = adata.X.tocsr().astype(np.float32)

    sc.pp.filter_cells(adata, min_genes=int(args.min_peaks_per_cell))
    sc.pp.filter_genes(adata, min_cells=int(args.min_cells_per_peak))
    print(f"[qc] filtered shape: {adata.shape}")

    tfidf = TfidfTransformer(norm="l1", use_idf=True, smooth_idf=True)
    adata.X = tfidf.fit_transform(adata.X).tocsr().astype(np.float32)
    print("[tfidf] finished sparse TF-IDF transform")

    if args.top_n_peaks is not None and adata.n_vars > int(args.top_n_peaks):
        var = sparse_column_variance(adata.X)
        keep_idx = np.argsort(var)[-int(args.top_n_peaks):]
        keep_idx = np.sort(keep_idx)
        adata = adata[:, keep_idx].copy()
        print(f"[select] kept top-{int(args.top_n_peaks)} variable peaks -> {adata.shape}")

    if peak_bed_path is not None:
        bed_df = read_bed_peak_table(peak_bed_path)
        bed_map = bed_df.set_index("peak_name")[["chrom", "start", "end"]]
        peak_index = pd.Index(adata.var_names.astype(str), name="peak_name")
        missing = peak_index.difference(bed_map.index)
        if len(missing) > 0:
            raise ValueError(
                f"{peak_bed_path} 中有 {len(missing)} 个 peak 无法与矩阵 var_names 对齐，"
                f"例如: {missing[:5].tolist()}"
            )
        coord_df = bed_map.loc[peak_index]
        chroms = coord_df["chrom"].astype(str).to_numpy()
        starts = coord_df["start"].astype(np.int64).to_numpy()
        ends = coord_df["end"].astype(np.int64).to_numpy()
        print(f"[bed] loaded coordinates from {peak_bed_path}")
    else:
        chroms, starts, ends = [], [], []
        for peak in adata.var_names.astype(str):
            chrom, start, end = parse_peak_name(peak)
            chroms.append(chrom)
            starts.append(start)
            ends.append(end)

    adata.var["chrom"] = chroms
    adata.var["start"] = np.asarray(starts, dtype=np.int64)
    adata.var["end"] = np.asarray(ends, dtype=np.int64)

    adata.write_h5ad(output_path)
    print(f"[saved] {output_path}")
    print(f"[final] shape: {adata.shape}")


if __name__ == "__main__":
    main()
