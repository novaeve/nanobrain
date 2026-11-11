#!/usr/bin/env python
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.io import mmread, mmwrite

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
SCUNIFLOW_ROOT = next(p for p in THIS_FILE.parents if (p / "model.py").exists() and (p / "data.py").exists())
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(SCUNIFLOW_ROOT))

from common import (  # noqa: E402
    derive_reference_labels,
    evaluate_clustering_metrics,
    materialize_downsampled_full_gene_counts,
    save_eval_outputs,
    set_seed,
    summarize_reconstruction,
)


DEFAULT_DATA_PATH = "data/pbmc68k_raw_counts.h5ad"
DEFAULT_IDX_DIR = THIS_DIR / "shared_eval_indices"
DEFAULT_VAL_IDX = DEFAULT_IDX_DIR / "pbmc68k_scuniflow_val_seed42_indices.npy"
DEFAULT_REF_LABELS = DEFAULT_IDX_DIR / "pbmc68k_scuniflow_val_seed42_ref_labels_res0.4_n9.npy"
DEFAULT_RSCRIPT = "Rscript"
DEFAULT_SCATAC_BENCHMARK = SCUNIFLOW_ROOT / "scATAC" / "benchmark"
DEFAULT_LOW_HIGH_CACHE_DIR = DEFAULT_IDX_DIR / "pbmc68k_scuniflow_val_fold4_seed42_low_high_cache"


def get_matrix(adata, layer: str):
    if not layer:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer not found in adata.layers: {layer}")
    return adata.layers[layer]


def normalize_log1p_counts(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    return np.log1p(counts / (totals + 1e-6) * float(target_sum)).astype(np.float32, copy=False)


def _cache_manifest_matches(manifest: dict, args, eval_idx: np.ndarray, n_obs: int, n_vars: int) -> bool:
    return (
        manifest.get("data_path") == str(Path(args.data_path).resolve())
        and manifest.get("layer", "") == str(args.layer or "")
        and float(manifest.get("downsample_fold", -1.0)) == float(args.downsample_fold)
        and int(manifest.get("seed", -1)) == int(args.seed)
        and int(manifest.get("n_obs", -1)) == int(n_obs)
        and int(manifest.get("n_vars", -1)) == int(n_vars)
        and manifest.get("cell_indices_path") == str(Path(args.cell_indices_path).resolve())
        and manifest.get("n_eval_cells") == int(eval_idx.shape[0])
    )


def _atomic_save_npy(path: Path, array: np.ndarray):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def load_or_materialize_low_high(matrix, eval_idx: np.ndarray, args, n_obs: int, n_vars: int):
    cache_dir = Path(args.low_high_cache_dir).resolve() if args.low_high_cache_dir else None
    if cache_dir is not None:
        manifest_path = cache_dir / "manifest.json"
        low_path = cache_dir / "low_counts.npy"
        high_path = cache_dir / "high_counts.npy"
        idx_path = cache_dir / "cell_indices.npy"
        if not args.refresh_low_high_cache and manifest_path.exists() and low_path.exists() and high_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)
            if _cache_manifest_matches(manifest, args, eval_idx, n_obs=n_obs, n_vars=n_vars):
                print(f"[cache] loading low/high counts from {cache_dir}", flush=True)
                t0 = time.perf_counter()
                low_counts = np.load(low_path)
                high_counts = np.load(high_path)
                print(f"[timing] load_low_high_cache={time.perf_counter() - t0:.2f}s", flush=True)
                return low_counts.astype(np.float32, copy=False), high_counts.astype(np.float32, copy=False)
            print(f"[cache] ignoring stale cache manifest: {cache_dir}", flush=True)

    t0 = time.perf_counter()
    low_counts, high_counts = materialize_downsampled_full_gene_counts(
        matrix,
        eval_idx,
        downsample_fold=float(args.downsample_fold),
        seed=int(args.seed),
    )
    print(f"[timing] materialize_low_high={time.perf_counter() - t0:.2f}s", flush=True)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "data_path": str(Path(args.data_path).resolve()),
            "layer": str(args.layer or ""),
            "cell_indices_path": str(Path(args.cell_indices_path).resolve()),
            "downsample_fold": float(args.downsample_fold),
            "seed": int(args.seed),
            "n_obs": int(n_obs),
            "n_vars": int(n_vars),
            "n_eval_cells": int(eval_idx.shape[0]),
            "low_counts": "low_counts.npy",
            "high_counts": "high_counts.npy",
        }
        print(f"[cache] saving low/high counts to {cache_dir}", flush=True)
        t0 = time.perf_counter()
        _atomic_save_npy(cache_dir / "low_counts.npy", low_counts.astype(np.float32, copy=False))
        _atomic_save_npy(cache_dir / "high_counts.npy", high_counts.astype(np.float32, copy=False))
        _atomic_save_npy(cache_dir / "cell_indices.npy", eval_idx.astype(np.int64, copy=False))
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        tmp_manifest.replace(manifest_path)
        print(f"[timing] save_low_high_cache={time.perf_counter() - t0:.2f}s", flush=True)
    return low_counts, high_counts


def select_hvg_indices(counts: np.ndarray, n_top_genes: int, seed: int) -> np.ndarray:
    n_top_genes = int(n_top_genes)
    if n_top_genes <= 0 or n_top_genes >= counts.shape[1]:
        return np.arange(counts.shape[1], dtype=np.int64)

    # scanpy preprocessing mutates AnnData.X in place; keep the shared low-count
    # matrix untouched because it is reused for non-HVG genes and low baseline.
    adata = sc.AnnData(np.asarray(counts, dtype=np.float32).copy())
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=False)
        idx = np.where(np.asarray(adata.var["highly_variable"], dtype=bool))[0]
        if idx.size > 0:
            return np.asarray(idx[:n_top_genes], dtype=np.int64)
    except Exception as exc:
        print(f"[warn] scanpy HVG selection failed, falling back to variance: {exc}", flush=True)

    rng = np.random.RandomState(int(seed))
    var = np.asarray(adata.X).var(axis=0)
    jitter = rng.uniform(0.0, 1e-12, size=var.shape[0])
    return np.argsort(-(var + jitter))[:n_top_genes].astype(np.int64)


def resolve_method_genes(args, low_counts: np.ndarray) -> np.ndarray:
    if args.method_gene_indices_path:
        idx = np.load(args.method_gene_indices_path).astype(np.int64)
        return np.unique(idx)
    if args.method_gene_mode == "all":
        return np.arange(low_counts.shape[1], dtype=np.int64)
    if args.method_gene_mode == "hvg":
        return select_hvg_indices(low_counts, args.method_n_top_genes, args.seed)
    raise ValueError(f"Unsupported method_gene_mode: {args.method_gene_mode}")


def write_r_input(input_dir: Path, counts_cells_by_genes: np.ndarray, gene_names: list[str], cell_names: list[str]):
    input_dir.mkdir(parents=True, exist_ok=True)
    # R wrappers expect genes x cells MatrixMarket plus one name per row/column.
    mtx = sparse.coo_matrix(np.asarray(counts_cells_by_genes, dtype=np.float32).T)
    mmwrite(input_dir / "matrix.mtx", mtx)
    (input_dir / "features.tsv").write_text("\n".join(gene_names) + "\n")
    (input_dir / "barcodes.tsv").write_text("\n".join(cell_names) + "\n")


def read_r_output(output_mtx: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    x = mmread(output_mtx)
    if sparse.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    # R output is genes x cells; Python evaluation uses cells x genes.
    arr = arr.T
    if arr.shape != expected_shape:
        raise ValueError(f"R output shape mismatch: got {arr.shape}, expected {expected_shape}")
    return np.clip(arr, 0.0, None)


def run_magic(low_counts: np.ndarray, est_high_totals: np.ndarray, args) -> np.ndarray:
    import magic

    if args.magic_input_space == "log1p_norm":
        x_in = normalize_log1p_counts(low_counts)
    elif args.magic_input_space == "raw_counts":
        x_in = np.asarray(low_counts, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported magic_input_space: {args.magic_input_space}")

    t_value = "auto" if str(args.magic_t).lower() == "auto" else int(args.magic_t)
    n_pca = None if int(args.magic_n_pca) <= 0 else min(int(args.magic_n_pca), x_in.shape[0] - 1, x_in.shape[1] - 1)
    op = magic.MAGIC(
        knn=int(args.magic_knn),
        t=t_value,
        n_pca=n_pca,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        solver=args.magic_solver,
        verbose=True,
    )
    out = np.asarray(op.fit_transform(x_in), dtype=np.float32)
    if args.magic_input_space == "log1p_norm":
        norm = np.expm1(out).clip(min=0.0)
        counts = norm / 1e4 * np.asarray(est_high_totals, dtype=np.float32).reshape(-1, 1)
        return np.clip(counts, 0.0, None)
    return np.clip(out, 0.0, None)


def run_scvi(low_counts: np.ndarray, est_high_totals: np.ndarray, args) -> np.ndarray:
    import torch
    import scvi

    adata = ad.AnnData(sparse.csr_matrix(np.asarray(low_counts, dtype=np.float32)))
    scvi.settings.seed = int(args.seed)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(
        adata,
        n_hidden=int(args.scvi_n_hidden),
        n_latent=int(args.scvi_n_latent),
        n_layers=int(args.scvi_n_layers),
        gene_likelihood=args.scvi_gene_likelihood,
    )
    accelerator = "gpu" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    model.train(
        max_epochs=int(args.scvi_epochs),
        batch_size=int(args.scvi_batch_size),
        accelerator=accelerator,
        devices=1 if accelerator == "gpu" else "auto",
        enable_progress_bar=True,
    )
    norm = model.get_normalized_expression(
        adata=adata,
        n_samples=int(args.scvi_n_samples),
        return_mean=True,
        library_size=1e4,
    )
    norm = np.asarray(norm, dtype=np.float32)
    counts = norm / 1e4 * np.asarray(est_high_totals, dtype=np.float32).reshape(-1, 1)
    return np.clip(counts, 0.0, None)


def run_r_method(method: str, low_counts: np.ndarray, gene_names: list[str], cell_names: list[str], args, work_dir: Path) -> np.ndarray:
    rscript = Path(args.rscript)
    if not rscript.exists():
        raise FileNotFoundError(f"Rscript not found: {rscript}")

    input_dir = work_dir / "r_input"
    output_mtx = work_dir / f"{method}_imputed.mtx"
    write_r_input(input_dir, low_counts, gene_names, cell_names)

    if method == "saver":
        cmd = [
            str(rscript),
            str(DEFAULT_SCATAC_BENCHMARK / "run_saver.R"),
            "--input_dir", str(input_dir),
            "--output_mtx", str(output_mtx),
            "--source_dir", str(DEFAULT_SCATAC_BENCHMARK / "SAVER"),
            "--ncores", str(args.r_ncores),
        ]
        if int(args.saver_npred) > 0:
            cmd += ["--npred", str(args.saver_npred)]
        if args.saver_size_factor_one:
            cmd += ["--size_factor_one"]
    elif method == "scimpute":
        cmd = [
            str(rscript),
            str(DEFAULT_SCATAC_BENCHMARK / "run_scimpute.R"),
            "--input_dir", str(input_dir),
            "--work_dir", str(work_dir / "scimpute_work"),
            "--output_mtx", str(output_mtx),
            "--source_dir", str(DEFAULT_SCATAC_BENCHMARK / "scImpute"),
            "--ncores", str(args.r_ncores),
            "--kcluster", str(args.scimpute_kcluster),
            "--drop_thre", str(args.scimpute_drop_thre),
        ]
    else:
        raise ValueError(f"Unsupported R method: {method}")

    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return read_r_output(output_mtx, expected_shape=low_counts.shape)


def run_method(method: str, low_counts: np.ndarray, est_high_totals: np.ndarray, gene_names: list[str], cell_names: list[str], args, work_dir: Path):
    if args.reuse_imputed_path:
        print(f"[reuse] loading imputed selected counts from {args.reuse_imputed_path}", flush=True)
        arr = np.load(args.reuse_imputed_path).astype(np.float32)
        if arr.shape != low_counts.shape:
            raise ValueError(f"reuse_imputed_path shape mismatch: got {arr.shape}, expected {low_counts.shape}")
        return np.clip(arr, 0.0, None)

    if method == "low":
        return np.asarray(low_counts, dtype=np.float32)
    if method == "magic":
        return run_magic(low_counts, est_high_totals, args)
    if method == "scvi":
        return run_scvi(low_counts, est_high_totals, args)
    if method in {"saver", "scimpute"}:
        return run_r_method(method, low_counts, gene_names, cell_names, args, work_dir)
    raise ValueError(f"Unsupported method: {method}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate classical imputation baselines on the fixed PBMC RDE val split.")
    parser.add_argument("--method", choices=["low", "magic", "saver", "scimpute", "scvi"], required=True)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--layer", default="")
    parser.add_argument("--cell_indices_path", default=str(DEFAULT_VAL_IDX))
    parser.add_argument("--target_labels_path", default=str(DEFAULT_REF_LABELS))
    parser.add_argument("--downsample_fold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_resolution", type=float, default=0.4)
    parser.add_argument("--n_top_genes", type=int, default=2000, help="HVGs used inside clustering metric.")
    parser.add_argument("--skip_clustering", action="store_true")
    parser.add_argument("--method_gene_mode", choices=["all", "hvg"], default="hvg")
    parser.add_argument("--method_n_top_genes", type=int, default=2000, help="Genes imputed by the method when method_gene_mode=hvg.")
    parser.add_argument("--method_gene_indices_path", default="")
    parser.add_argument("--gene_name_col", default="", help="Optional var column for gene names; defaults to var_names.")
    parser.add_argument("--work_dir", default="", help="Temporary method work dir; defaults to save_dir/work.")
    parser.add_argument("--low_high_cache_dir", default=str(DEFAULT_LOW_HIGH_CACHE_DIR))
    parser.add_argument("--refresh_low_high_cache", action="store_true")
    parser.add_argument("--reuse_imputed_path", default="")
    parser.add_argument("--save_imputed", action="store_true")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--magic_knn", type=int, default=5)
    parser.add_argument("--magic_t", default="3")
    parser.add_argument("--magic_n_pca", type=int, default=100)
    parser.add_argument("--magic_solver", choices=["exact", "approximate"], default="exact")
    parser.add_argument("--magic_input_space", choices=["log1p_norm", "raw_counts"], default="log1p_norm")

    parser.add_argument("--scvi_epochs", type=int, default=100)
    parser.add_argument("--scvi_batch_size", type=int, default=256)
    parser.add_argument("--scvi_n_hidden", type=int, default=128)
    parser.add_argument("--scvi_n_latent", type=int, default=20)
    parser.add_argument("--scvi_n_layers", type=int, default=2)
    parser.add_argument("--scvi_n_samples", type=int, default=1)
    parser.add_argument("--scvi_gene_likelihood", choices=["zinb", "nb", "poisson"], default="zinb")

    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT", DEFAULT_RSCRIPT))
    parser.add_argument("--r_ncores", type=int, default=8)
    parser.add_argument("--saver_npred", type=int, default=0)
    parser.add_argument("--saver_size_factor_one", action="store_true")
    parser.add_argument("--scimpute_kcluster", type=int, default=9)
    parser.add_argument("--scimpute_drop_thre", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).resolve() if args.work_dir else save_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[load] {args.data_path}", flush=True)
    adata = ad.read_h5ad(args.data_path)
    print(f"[load] shape={adata.n_obs} cells x {adata.n_vars} genes ({time.perf_counter() - t0:.2f}s)", flush=True)
    matrix = get_matrix(adata, args.layer)
    eval_idx = np.load(args.cell_indices_path).astype(np.int64)
    cell_names = adata.obs_names.astype(str).to_numpy()[eval_idx].tolist()
    if args.gene_name_col:
        if args.gene_name_col not in adata.var.columns:
            raise KeyError(f"gene_name_col not found in adata.var: {args.gene_name_col}")
        all_gene_names = adata.var[args.gene_name_col].astype(str).tolist()
    else:
        all_gene_names = adata.var_names.astype(str).tolist()

    low_counts, high_counts = load_or_materialize_low_high(
        matrix,
        eval_idx,
        args,
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
    )

    if args.method == "low":
        gene_idx = np.arange(low_counts.shape[1], dtype=np.int64)
        selected_gene_names = all_gene_names
        low_selected = low_counts
        selected_est_high_totals = np.maximum(low_selected.sum(axis=1) * float(args.downsample_fold), 1.0)
    else:
        t0 = time.perf_counter()
        gene_idx = resolve_method_genes(args, low_counts)
        print(f"[timing] select_method_genes={time.perf_counter() - t0:.2f}s", flush=True)
        selected_gene_names = [all_gene_names[int(i)] for i in gene_idx]
        low_selected = low_counts[:, gene_idx]
        selected_est_high_totals = np.maximum(low_selected.sum(axis=1) * float(args.downsample_fold), 1.0)
    print(
        f"[setup] method={args.method} cells={low_counts.shape[0]} full_genes={low_counts.shape[1]} "
        f"method_genes={gene_idx.size} gene_mode={args.method_gene_mode}",
        flush=True,
    )

    t0 = time.perf_counter()
    imputed_selected = run_method(
        args.method,
        low_selected,
        selected_est_high_totals,
        selected_gene_names,
        cell_names,
        args,
        work_dir,
    )
    print(f"[timing] method={time.perf_counter() - t0:.2f}s", flush=True)

    enhanced_counts = np.asarray(low_counts, dtype=np.float32).copy()
    enhanced_counts[:, gene_idx] = imputed_selected.astype(np.float32, copy=False)
    enhanced_counts = np.clip(enhanced_counts, 0.0, None)

    pred_log = normalize_log1p_counts(enhanced_counts)
    high_log = normalize_log1p_counts(high_counts)
    recon_metrics, cell_pccs, gene_pccs = summarize_reconstruction(pred_log, high_log)

    clustering = None
    if args.skip_clustering:
        target_labels = np.asarray([], dtype=str)
        enhanced_clusters = np.asarray([], dtype=str)
        low_clusters = np.asarray([], dtype=str)
    elif args.target_labels_path:
        target_labels = np.load(args.target_labels_path, allow_pickle=True).astype(str)
        if target_labels.shape[0] != high_counts.shape[0]:
            raise ValueError(f"target_labels length mismatch: {target_labels.shape[0]} vs {high_counts.shape[0]}")
        n_clusters = len(np.unique(target_labels))
        reference_info = {
            "target_mode": "shared_labels",
            "target_labels_path": str(Path(args.target_labels_path).resolve()),
            "n_clusters": int(n_clusters),
        }
    else:
        target_labels, _ = derive_reference_labels(high_counts, ref_resolution=args.ref_resolution, n_top_genes=args.n_top_genes)
        n_clusters = len(np.unique(target_labels))
        reference_info = {
            "target_mode": "reference_leiden",
            "ref_resolution": float(args.ref_resolution),
            "n_clusters": int(n_clusters),
        }

    if not args.skip_clustering:
        t0 = time.perf_counter()
        enhanced_metrics, _, enhanced_clusters = evaluate_clustering_metrics(
            enhanced_counts,
            target_labels=target_labels,
            n_clusters=n_clusters,
            n_top_genes=int(args.n_top_genes),
        )
        print(f"[timing] enhanced_clustering={time.perf_counter() - t0:.2f}s", flush=True)
        t0 = time.perf_counter()
        low_metrics, _, low_clusters = evaluate_clustering_metrics(
            low_counts,
            target_labels=target_labels,
            n_clusters=n_clusters,
            n_top_genes=int(args.n_top_genes),
        )
        print(f"[timing] low_clustering={time.perf_counter() - t0:.2f}s", flush=True)
        clustering = {
            "reference": reference_info,
            "low_depth_baseline": low_metrics,
            "enhanced": enhanced_metrics,
        }

    metrics = dict(recon_metrics)
    metrics.update(
        {
            "model": args.method,
            "method_family": "classical_imputation",
            "data_path": str(Path(args.data_path).resolve()),
            "n_eval_cells": int(high_counts.shape[0]),
            "n_genes": int(high_counts.shape[1]),
            "n_method_genes": int(gene_idx.size),
            "method_gene_mode": args.method_gene_mode,
            "method_n_top_genes": int(args.method_n_top_genes),
            "downsample_fold": float(args.downsample_fold),
            "cell_indices_path": str(Path(args.cell_indices_path).resolve()),
            "target_labels_path": str(Path(args.target_labels_path).resolve()) if args.target_labels_path else "",
            "clustering_gene_space": "full_gene_scuniflow_protocol",
            "reconstruction_gene_space": "full_gene_enhanced_low_filled",
            "clustering": clustering,
        }
    )

    save_eval_outputs(save_dir, metrics, cell_names, cell_pccs, gene_pccs)
    np.save(save_dir / "method_gene_indices.npy", gene_idx)
    np.save(save_dir / "target_labels.npy", np.asarray(target_labels))
    np.save(save_dir / "enhanced_clusters.npy", np.asarray(enhanced_clusters))
    np.save(save_dir / "low_clusters.npy", np.asarray(low_clusters))
    if args.save_imputed:
        np.save(save_dir / "imputed_selected_counts.npy", imputed_selected.astype(np.float32, copy=False))
    if args.save_arrays:
        np.save(save_dir / "full_low_counts.npy", low_counts)
        np.save(save_dir / "full_high_counts.npy", high_counts)
        np.save(save_dir / "enhanced_full_counts.npy", enhanced_counts)

    with (save_dir / "run_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[saved] {save_dir}")


if __name__ == "__main__":
    main()
