#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(description="Compute and freeze an HVG ranking/panel for scFiT.")
    parser.add_argument(
        "--input-h5ad",
        type=str,
        default="data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad",
        help="Path to the source AnnData file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cache/hvg_cache/neural_crest_pns_glia",
        help="Directory to store ranked HVG artifacts and coverage reports.",
    )
    parser.add_argument(
        "--raw-layer",
        type=str,
        default="raw.X",
        help="Layer containing raw counts. If missing, fall back to adata.X.",
    )
    parser.add_argument(
        "--flavor",
        type=str,
        default="seurat_v3",
        choices=["seurat_v3", "seurat", "cell_ranger"],
        help="Scanpy HVG flavor.",
    )
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="raw_counts",
        choices=["raw_counts", "normalize_log1p"],
        help=(
            "How to prepare the matrix before HVG. "
            "`raw_counts` matches Scanpy's recommended seurat_v3 usage; "
            "`normalize_log1p` follows the explicit library-size-normalize + log1p workflow."
        ),
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Target library size used when preprocess-mode=normalize_log1p.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
        help="Candidate panel sizes to summarize before the final k is fixed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional final panel size. If provided, also materialize hvg_gene_names.txt and hvg_gene_indices.npy.",
    )
    parser.add_argument(
        "--marker-genes",
        type=str,
        nargs="*",
        default=["Sox10", "Phox2b"],
        help="Marker genes to check against the ranked HVG list.",
    )
    return parser.parse_args()


def _resolve_symbol_column(var) -> str | None:
    for col in ("feature_name", "gene_short_name", "gene_name", "symbol"):
        if col in var.columns:
            return col
    return None


def _build_work_adata(adata: ad.AnnData, raw_layer: str, preprocess_mode: str, target_sum: float):
    if raw_layer and raw_layer in adata.layers:
        base_matrix = adata.layers[raw_layer]
        matrix_source = f"layers[{raw_layer}]"
    else:
        base_matrix = adata.X
        matrix_source = "X"

    if sp.issparse(base_matrix):
        work_matrix = base_matrix.copy().tocsr().astype(np.float32)
    else:
        work_matrix = np.asarray(base_matrix, dtype=np.float32).copy()

    work_adata = ad.AnnData(X=work_matrix, var=adata.var.copy())
    if preprocess_mode == "normalize_log1p":
        sc.pp.normalize_total(work_adata, target_sum=float(target_sum))
        sc.pp.log1p(work_adata)
    return work_adata, matrix_source, base_matrix


def _rank_genes(work_adata: ad.AnnData, flavor: str):
    n_vars = int(work_adata.n_vars)
    sc.pp.highly_variable_genes(
        work_adata,
        n_top_genes=n_vars,
        flavor=flavor,
        inplace=True,
        subset=False,
    )

    var = work_adata.var
    if "highly_variable_rank" in var.columns and np.isfinite(var["highly_variable_rank"].to_numpy(dtype=np.float64)).any():
        rank_values = var["highly_variable_rank"].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(rank_values)
        if "variances_norm" in var.columns:
            fallback = -var["variances_norm"].to_numpy(dtype=np.float64)
        elif "dispersions_norm" in var.columns:
            fallback = -var["dispersions_norm"].to_numpy(dtype=np.float64)
        else:
            fallback = np.zeros((n_vars,), dtype=np.float64)
        sentinel = np.nanmax(rank_values[finite_mask]) + n_vars if finite_mask.any() else float(n_vars)
        primary = np.where(finite_mask, rank_values, sentinel)
        order = np.lexsort((fallback, primary))
    elif "variances_norm" in var.columns:
        order = np.argsort(-var["variances_norm"].to_numpy(dtype=np.float64), kind="mergesort")
    elif "dispersions_norm" in var.columns:
        order = np.argsort(-var["dispersions_norm"].to_numpy(dtype=np.float64), kind="mergesort")
    else:
        raise RuntimeError("Scanpy HVG output missing ranking columns.")

    return order.astype(np.int64, copy=False)


def _nnz_per_cell(matrix, gene_indices: np.ndarray) -> np.ndarray:
    if gene_indices.size == 0:
        return np.zeros((matrix.shape[0],), dtype=np.int64)
    if sp.issparse(matrix):
        view = matrix[:, gene_indices]
        view = view.tocsr()
        return np.diff(view.indptr).astype(np.int64, copy=False)
    dense = np.asarray(matrix[:, gene_indices])
    return np.count_nonzero(dense > 0, axis=1).astype(np.int64, copy=False)


def _summarize_counts(values: np.ndarray) -> dict:
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    summary = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    q = np.percentile(values, percentiles)
    for p, v in zip(percentiles, q):
        key = "min" if p == 0 else "max" if p == 100 else f"p{p}"
        summary[key] = float(v)
    return summary


def _write_lines(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def _marker_membership(marker_genes, symbol_names, panel_order, candidate_ks):
    symbol_to_rank = {}
    for rank, gene in enumerate(symbol_names[panel_order], start=1):
        symbol_to_rank.setdefault(str(gene), rank)

    report = {}
    for gene in marker_genes:
        rank = symbol_to_rank.get(str(gene))
        report[str(gene)] = {
            "rank": None if rank is None else int(rank),
            "present": rank is not None,
            "present_by_k": {str(k): bool(rank is not None and rank <= int(k)) for k in candidate_ks},
        }
    return report


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hvg] loading {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    symbol_col = _resolve_symbol_column(adata.var)
    symbol_names = (
        adata.var[symbol_col].astype(str).to_numpy()
        if symbol_col is not None
        else adata.var_names.astype(str).to_numpy()
    )
    var_names = adata.var_names.astype(str).to_numpy()

    work_adata, matrix_source, base_matrix = _build_work_adata(
        adata=adata,
        raw_layer=args.raw_layer,
        preprocess_mode=args.preprocess_mode,
        target_sum=args.target_sum,
    )
    print(
        "[hvg] matrix_source="
        f"{matrix_source} preprocess_mode={args.preprocess_mode} flavor={args.flavor} "
        f"shape={work_adata.shape}"
    )

    rank_order = _rank_genes(work_adata, args.flavor)
    ranked_var_names = var_names[rank_order]
    ranked_symbol_names = symbol_names[rank_order]

    candidate_ks = sorted({int(k) for k in args.candidate_k if int(k) > 0})
    if args.top_k is not None:
        candidate_ks = sorted(set(candidate_ks + [int(args.top_k)]))

    coverage = {}
    for k in candidate_ks:
        k_eff = min(int(k), int(rank_order.size))
        nnz = _nnz_per_cell(base_matrix, rank_order[:k_eff])
        coverage[str(k_eff)] = {
            "panel_size": int(k_eff),
            "usable_genes_per_cell": _summarize_counts(nnz),
            "fraction_cells_with_all_panel_nonzero": float(np.mean(nnz >= k_eff)),
            "fraction_cells_with_at_least_half_panel_nonzero": float(np.mean(nnz >= max(k_eff // 2, 1))),
        }

    marker_report = _marker_membership(
        marker_genes=args.marker_genes,
        symbol_names=symbol_names,
        panel_order=rank_order,
        candidate_ks=candidate_ks,
    )

    metadata = {
        "input_h5ad": str(args.input_h5ad),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "raw_layer_requested": args.raw_layer,
        "matrix_source_used": matrix_source,
        "preprocess_mode": args.preprocess_mode,
        "flavor": args.flavor,
        "target_sum": float(args.target_sum),
        "symbol_column": symbol_col,
        "candidate_k": candidate_ks,
        "top_k_materialized": None if args.top_k is None else int(args.top_k),
        "marker_genes": [str(g) for g in args.marker_genes],
        "scanpy_version": sc.__version__,
        "elapsed_seconds": float(time.time() - t0),
    }

    _write_lines(out_dir / "var_names_all.txt", var_names)
    _write_lines(out_dir / "gene_symbols_all.txt", symbol_names)
    _write_lines(out_dir / "hvg_ranked_var_names.txt", ranked_var_names)
    _write_lines(out_dir / "hvg_ranked_gene_names.txt", ranked_symbol_names)
    np.save(out_dir / "hvg_ranked_gene_indices.npy", rank_order)

    if args.top_k is not None:
        k_eff = min(int(args.top_k), int(rank_order.size))
        _write_lines(out_dir / "hvg_gene_names.txt", ranked_symbol_names[:k_eff])
        _write_lines(out_dir / "hvg_var_names.txt", ranked_var_names[:k_eff])
        np.save(out_dir / "hvg_gene_indices.npy", rank_order[:k_eff])

    with (out_dir / "hvg_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (out_dir / "hvg_coverage_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"coverage": coverage, "markers": marker_report}, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# HVG Coverage Summary",
        "",
        f"- input_h5ad: `{args.input_h5ad}`",
        f"- shape: `{adata.n_obs} x {adata.n_vars}`",
        f"- matrix_source_used: `{matrix_source}`",
        f"- preprocess_mode: `{args.preprocess_mode}`",
        f"- flavor: `{args.flavor}`",
        f"- symbol_column: `{symbol_col}`",
        "",
        "## Candidate k Coverage",
        "",
        "| k | mean usable genes | p10 | p50 | p90 | frac >= k/2 | frac >= k |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(coverage.keys(), key=lambda x: int(x)):
        stats = coverage[key]["usable_genes_per_cell"]
        md_lines.append(
            f"| {key} | {stats['mean']:.1f} | {stats['p10']:.1f} | {stats['p50']:.1f} | "
            f"{stats['p90']:.1f} | {coverage[key]['fraction_cells_with_at_least_half_panel_nonzero']:.4f} | "
            f"{coverage[key]['fraction_cells_with_all_panel_nonzero']:.4f} |"
        )

    if marker_report:
        md_lines.extend(["", "## Marker Check", ""])
        for gene, report in marker_report.items():
            by_k = ", ".join(f"top-{k}:{'Y' if present else 'N'}" for k, present in report["present_by_k"].items())
            md_lines.append(f"- `{gene}`: rank={report['rank']}, present={report['present']}, {by_k}")

    with (out_dir / "hvg_coverage_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[hvg] wrote artifacts to {out_dir}")
    print(f"[hvg] finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
