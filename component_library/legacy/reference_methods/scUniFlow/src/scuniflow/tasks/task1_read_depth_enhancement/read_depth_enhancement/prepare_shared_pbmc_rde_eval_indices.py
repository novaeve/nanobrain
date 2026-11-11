import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import yaml

THIS_FILE = Path(__file__).resolve()
SCUNIFLOW_ROOT = next(p for p in THIS_FILE.parents if (p / "model.py").exists() and (p / "data.py").exists())
sys.path.insert(0, str(SCUNIFLOW_ROOT))

from data import split_cells_train_val


DEFAULT_RUN_ROOT = Path("./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs")
DEFAULT_OUT = Path("./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/shared_eval_indices")


def sample_indices(indices, n_cells: int, seed: int):
    indices = np.asarray(indices, dtype=np.int64)
    if n_cells is None or n_cells <= 0 or n_cells >= len(indices):
        return indices.copy()
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(indices, size=int(n_cells), replace=False)).astype(np.int64)


def max_cells_train_val(n_obs: int, args: dict):
    all_idx = np.arange(n_obs, dtype=np.int64)
    keep = sample_indices(
        all_idx,
        int(args.get("max_cells", 0) or 0),
        int(args.get("cell_seed", args.get("seed", 42))),
    )
    train_local, val_local, _ = split_cells_train_val(
        len(keep),
        val_ratio=float(args.get("val_ratio", 0.1)),
        seed=int(args.get("seed", 42)),
    )
    return keep[train_local], keep[val_local]


def sampled_train_val(n_obs: int, args: dict):
    train_idx, val_idx, _ = split_cells_train_val(
        n_obs,
        val_ratio=float(args.get("val_ratio", 0.1)),
        seed=int(args.get("split_seed", args.get("seed", 42))),
    )
    train_idx = sample_indices(
        train_idx,
        int(args.get("train_total_cells", 0) or 0),
        int(args.get("train_cell_seed", args.get("seed", 42))),
    )
    val_idx = sample_indices(
        val_idx,
        int(args.get("val_total_cells", 0) or 0),
        int(args.get("val_cell_seed", args.get("seed", 42))),
    )
    return train_idx, val_idx


def main():
    parser = argparse.ArgumentParser(description="Prepare PBMC RDE shared held-out eval cell indices.")
    parser.add_argument("--data_path", default="data/pbmc68k_raw_counts.h5ad")
    parser.add_argument("--scuniflow_config", default="results/pretrain/pbmc68k_augmented_181/config.yaml")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_cells", type=int, default=997, help="sample size from the not-trained-by-any pool; 0 means all")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.data_path)
    n_obs = int(adata.n_obs)
    all_idx = np.arange(n_obs, dtype=np.int64)

    train_sets = {}
    val_sets = {}

    cfg = yaml.safe_load(open(args.scuniflow_config))
    train_cfg = cfg["training"]
    sc_train, sc_val, _ = split_cells_train_val(
        n_obs,
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
        seed=int(train_cfg.get("seed", 42)),
    )
    train_sets["scUniFlow"] = np.asarray(sc_train, dtype=np.int64)
    val_sets["scUniFlow"] = np.asarray(sc_val, dtype=np.int64)

    max_cell_runs = {
        "scFoundation": DEFAULT_RUN_ROOT / "pbmc68k_scfoundation_rde_from_pbmc_recon_12k_2k" / "args.json",
        "scGPT": DEFAULT_RUN_ROOT / "pbmc68k_scgpt_rde_from_pbmc_recon_e3_fold4_12k_e10" / "args.json",
        "Geneformer": DEFAULT_RUN_ROOT / "pbmc68k_geneformer_rde_from_pbmc_mlm_e3_fold4_12k_e10" / "args.json",
    }
    for name, path in max_cell_runs.items():
        run_args = json.load(open(path))
        train_sets[name], val_sets[name] = max_cells_train_val(n_obs, run_args)

    sampled_runs = {
        "UCE": DEFAULT_RUN_ROOT / "pbmc68k_uce_rde_fold4_12k_2k_e8" / "args.json",
        "Nicheformer": DEFAULT_RUN_ROOT / "pbmc68k_nicheformer_rde_fold4_12k_2k_e8" / "args.json",
    }
    for name, path in sampled_runs.items():
        run_args = json.load(open(path))
        train_sets[name], val_sets[name] = sampled_train_val(n_obs, run_args)

    train_union = np.unique(np.concatenate(list(train_sets.values())))
    heldout = np.setdiff1d(all_idx, train_union, assume_unique=False)

    rng = np.random.RandomState(int(args.seed))
    if args.n_cells and args.n_cells > 0 and args.n_cells < len(heldout):
        shared = np.sort(rng.choice(heldout, size=int(args.n_cells), replace=False)).astype(np.int64)
    else:
        shared = heldout.astype(np.int64)

    common_val = val_sets["scUniFlow"].copy()
    for values in val_sets.values():
        common_val = np.intersect1d(common_val, values)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pbmc68k_rde_shared_heldout997_indices.npy", shared)
    np.save(out_dir / "pbmc68k_rde_not_trained_by_any_pool_indices.npy", heldout)
    np.save(out_dir / "pbmc68k_rde_common_model_val280_indices.npy", common_val)

    summary = {
        "data_path": str(Path(args.data_path).resolve()),
        "n_obs": n_obs,
        "shared_indices": str(out_dir / "pbmc68k_rde_shared_heldout997_indices.npy"),
        "not_trained_by_any_pool_indices": str(out_dir / "pbmc68k_rde_not_trained_by_any_pool_indices.npy"),
        "common_model_val_indices": str(out_dir / "pbmc68k_rde_common_model_val280_indices.npy"),
        "n_shared": int(len(shared)),
        "n_not_trained_by_any_pool": int(len(heldout)),
        "n_common_model_val": int(len(common_val)),
        "train_set_sizes": {k: int(len(v)) for k, v in train_sets.items()},
        "val_set_sizes": {k: int(len(v)) for k, v in val_sets.items()},
        "shared_train_overlaps": {k: int(len(np.intersect1d(shared, v))) for k, v in train_sets.items()},
        "shared_val_overlaps": {k: int(len(np.intersect1d(shared, v))) for k, v in val_sets.items()},
        "seed": int(args.seed),
    }
    with open(out_dir / "pbmc68k_rde_shared_heldout997_manifest.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
