import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from common import (
    DEFAULT_REF_H5AD,
    DEFAULT_VISIUM_ROOT,
    build_target_prediction_adata,
    load_reference_metadata,
    load_visium_mouse_brain,
    save_json,
    set_seed,
    summarize_target_metrics,
)


def to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def log_cpm_dense(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log1p((X / lib) * 1e6).astype(np.float32, copy=False)


def sample_indices(n_total: int, max_items: int, seed: int) -> np.ndarray:
    if max_items <= 0 or n_total <= max_items:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_total, size=int(max_items), replace=False)).astype(np.int64)


def pick_gene_panels(
    ref_subset_counts: np.ndarray,
    spatial_counts: np.ndarray,
    observed_gene_count: int,
    target_gene_count: int,
    min_nonzero_spots: int,
    min_nonzero_ref_cells: int,
    seed: int,
):
    ref_nonzero = (ref_subset_counts > 0).sum(axis=0)
    spatial_nonzero = (spatial_counts > 0).sum(axis=0)
    ref_var = np.var(ref_subset_counts, axis=0)
    spatial_var = np.var(spatial_counts, axis=0)

    eligible = (
        (ref_nonzero >= int(min_nonzero_ref_cells))
        & (spatial_nonzero >= int(min_nonzero_spots))
        & (ref_var > 0)
        & (spatial_var > 0)
    )
    eligible_idx = np.flatnonzero(eligible)
    need = int(observed_gene_count) + int(target_gene_count)
    if eligible_idx.size < need:
        raise ValueError(
            f"Not enough eligible genes: eligible={eligible_idx.size}, required={need}"
        )

    rng = np.random.RandomState(seed)
    chosen = rng.choice(eligible_idx, size=need, replace=False)
    chosen = np.asarray(chosen, dtype=np.int64)
    target_idx = np.sort(chosen[: int(target_gene_count)])
    observed_idx = np.sort(chosen[int(target_gene_count) :])
    return observed_idx, target_idx


@dataclass
class BenchmarkData:
    ref_h5ad: Path
    visium_root: Path
    ref: Any
    spatial: Any
    spatial_counts: np.ndarray
    ref_rows: np.ndarray
    ref_subset_counts: np.ndarray
    observed_idx: np.ndarray
    target_idx: np.ndarray


def prepare_benchmark_data(
    ref_h5ad: str | Path = DEFAULT_REF_H5AD,
    visium_root: str | Path = DEFAULT_VISIUM_ROOT,
    sample_ids: Optional[list[str]] = None,
    panel_seed: int = 42,
    sample_seed: int = 42,
    max_ref_cells: int = 8000,
    max_spots: int = 0,
    in_tissue_only: bool = True,
    observed_gene_count: int = 1024,
    target_gene_count: int = 256,
    min_nonzero_spots: int = 10,
    min_nonzero_ref_cells: int = 20,
) -> BenchmarkData:
    set_seed(int(panel_seed))
    ref_h5ad = Path(ref_h5ad).resolve()
    visium_root = Path(visium_root).resolve()
    ref = load_reference_metadata(ref_h5ad)
    spatial = load_visium_mouse_brain(
        visium_root=visium_root,
        ref=ref,
        sample_ids=sample_ids or None,
        in_tissue_only=bool(in_tissue_only),
        max_spots=int(max_spots),
        sample_seed=int(sample_seed),
    )
    spatial_counts = to_dense_float32(spatial.X)

    ref_adata = sc.read_h5ad(ref_h5ad, backed="r")
    ref_rows = sample_indices(ref_adata.n_obs, int(max_ref_cells), int(sample_seed))
    ref_subset_counts = to_dense_float32(ref_adata[ref_rows, :].X)
    observed_idx, target_idx = pick_gene_panels(
        ref_subset_counts=ref_subset_counts,
        spatial_counts=spatial_counts,
        observed_gene_count=int(observed_gene_count),
        target_gene_count=int(target_gene_count),
        min_nonzero_spots=int(min_nonzero_spots),
        min_nonzero_ref_cells=int(min_nonzero_ref_cells),
        seed=int(panel_seed),
    )

    return BenchmarkData(
        ref_h5ad=ref_h5ad,
        visium_root=visium_root,
        ref=ref,
        spatial=spatial,
        spatial_counts=spatial_counts,
        ref_rows=ref_rows,
        ref_subset_counts=ref_subset_counts,
        observed_idx=observed_idx,
        target_idx=target_idx,
    )


def write_benchmark_outputs(
    output_dir: str | Path,
    method: str,
    benchmark: BenchmarkData,
    pred_targets_logcpm: np.ndarray,
    extra_summary: Optional[dict] = None,
    save_h5ad: bool = False,
    save_truth: bool = True,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref = benchmark.ref
    spatial = benchmark.spatial
    target_idx = benchmark.target_idx
    observed_idx = benchmark.observed_idx
    truth_target_logcpm = log_cpm_dense(benchmark.spatial_counts[:, target_idx])

    pred_targets_logcpm = np.asarray(pred_targets_logcpm, dtype=np.float32)
    metrics = summarize_target_metrics(pred_targets_logcpm, truth_target_logcpm)
    summary = {
        **metrics,
        "method": method,
        "ref_h5ad": str(benchmark.ref_h5ad),
        "visium_root": str(benchmark.visium_root),
        "n_ref_cells_used": int(benchmark.ref_rows.shape[0]),
        "n_spots_used": int(spatial.n_obs),
        "n_genes_total": int(ref.ensembl_ids.shape[0]),
        "observed_gene_count": int(observed_idx.shape[0]),
        "target_gene_count": int(target_idx.shape[0]),
    }
    if extra_summary:
        summary.update(extra_summary)
    save_json(output_dir / "summary.json", summary)

    target_ids = ref.ensembl_ids[target_idx]
    observed_ids = ref.ensembl_ids[observed_idx]
    save_json(
        output_dir / "panel_info.json",
        {
            "observed_ensembl_ids": observed_ids.tolist(),
            "target_ensembl_ids": target_ids.tolist(),
            "observed_symbols": ref.gene_symbols[observed_idx].tolist(),
            "target_symbols": ref.gene_symbols[target_idx].tolist(),
        },
    )

    pred_df = pd.DataFrame(
        pred_targets_logcpm,
        index=spatial.obs_names.astype(str),
        columns=target_ids,
    )
    pred_df.to_csv(output_dir / "predicted_target_logcpm.csv.gz", compression="gzip")

    if save_truth:
        truth_df = pd.DataFrame(
            truth_target_logcpm,
            index=spatial.obs_names.astype(str),
            columns=target_ids,
        )
        truth_df.to_csv(output_dir / "truth_target_logcpm.csv.gz", compression="gzip")

    if save_h5ad:
        pred_adata = build_target_prediction_adata(
            spatial=spatial,
            ref=ref,
            target_idx=target_idx,
            pred_targets=pred_targets_logcpm,
            truth_targets=truth_target_logcpm,
        )
        pred_adata.write_h5ad(output_dir / f"{method.lower()}_predicted_target_genes.h5ad")


def dump_r_inputs(
    output_dir: str | Path,
    benchmark: BenchmarkData,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = benchmark.ref

    observed_idx = benchmark.observed_idx
    target_idx = benchmark.target_idx
    observed_ids = ref.ensembl_ids[observed_idx]
    target_ids = ref.ensembl_ids[target_idx]

    spatial_obs_logcpm = log_cpm_dense(benchmark.spatial_counts[:, observed_idx])
    spatial_target_truth = log_cpm_dense(benchmark.spatial_counts[:, target_idx])
    ref_all_logcpm = log_cpm_dense(benchmark.ref_subset_counts[:, np.concatenate([observed_idx, target_idx])])
    spatial_obs_raw = benchmark.spatial_counts[:, observed_idx].astype(np.float32, copy=False)
    spatial_target_raw = benchmark.spatial_counts[:, target_idx].astype(np.float32, copy=False)
    ref_all_raw = benchmark.ref_subset_counts[:, np.concatenate([observed_idx, target_idx])].astype(np.float32, copy=False)

    pd.DataFrame(
        spatial_obs_logcpm,
        index=benchmark.spatial.obs_names.astype(str),
        columns=observed_ids,
    ).to_csv(output_dir / "spatial_observed_logcpm.csv.gz", compression="gzip")
    pd.DataFrame(
        spatial_obs_raw,
        index=benchmark.spatial.obs_names.astype(str),
        columns=observed_ids,
    ).to_csv(output_dir / "spatial_observed_raw_counts.csv.gz", compression="gzip")

    pd.DataFrame(
        spatial_target_truth,
        index=benchmark.spatial.obs_names.astype(str),
        columns=target_ids,
    ).to_csv(output_dir / "spatial_target_truth_logcpm.csv.gz", compression="gzip")
    pd.DataFrame(
        spatial_target_raw,
        index=benchmark.spatial.obs_names.astype(str),
        columns=target_ids,
    ).to_csv(output_dir / "spatial_target_truth_raw_counts.csv.gz", compression="gzip")

    pd.DataFrame(
        ref_all_logcpm,
        index=[f"ref_{i}" for i in benchmark.ref_rows.tolist()],
        columns=np.concatenate([observed_ids, target_ids]),
    ).to_csv(output_dir / "reference_observed_plus_target_logcpm.csv.gz", compression="gzip")
    pd.DataFrame(
        ref_all_raw,
        index=[f"ref_{i}" for i in benchmark.ref_rows.tolist()],
        columns=np.concatenate([observed_ids, target_ids]),
    ).to_csv(output_dir / "reference_observed_plus_target_raw_counts.csv.gz", compression="gzip")

    save_json(
        output_dir / "panel_info.json",
        {
            "observed_ensembl_ids": observed_ids.tolist(),
            "target_ensembl_ids": target_ids.tolist(),
            "observed_symbols": ref.gene_symbols[observed_idx].tolist(),
            "target_symbols": ref.gene_symbols[target_idx].tolist(),
        },
    )
