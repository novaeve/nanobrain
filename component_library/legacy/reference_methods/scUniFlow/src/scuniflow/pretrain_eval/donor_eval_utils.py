import csv
import time
from pathlib import Path

import numpy as np
import scanpy as sc
from tqdm.auto import tqdm


DEFAULT_DONOR_CSV = Path("./donor_cell_counts_full_mtg.csv")
DEFAULT_DONOR_DIR = Path("data/SEA-AD/MTG/donor_h5ad")


def split_val_donors(donor_csv: Path, val_ratio: float = 0.1, seed: int = 42):
    rows = []
    with donor_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["donor_id"], int(row["n_cells"])))

    rng = np.random.RandomState(seed)
    rng.shuffle(rows)
    rows.sort(key=lambda x: x[1])

    total = sum(n for _, n in rows)
    target = total * val_ratio

    val = []
    val_cells = 0
    for did, n in rows:
        if val_cells < target:
            val.append((did, n))
            val_cells += n

    return val


def pick_eval_donor(val_donors, donor_dir: Path, min_cells: int = 1000):
    candidates = sorted([x for x in val_donors if x[1] >= min_cells], key=lambda x: x[1])
    if not candidates:
        raise RuntimeError("No val donor has enough cells.")
    for did, n in candidates:
        if (donor_dir / f"{did}.h5ad").exists():
            return did, n
    raise RuntimeError("No donor file exists for val donors.")


def sample_single_donor_cells(donor_path: Path, sample_n: int, sample_seed: int):
    t0 = time.time()
    print(f"[data] loading donor file: {donor_path.name}", flush=True)
    adata = sc.read_h5ad(donor_path, backed="r")

    total_n = int(adata.n_obs)
    if sample_n is None:
        n = total_n
    else:
        n = min(int(sample_n), total_n)

    if n >= total_n:
        sampled = adata.to_memory()
    else:
        rng = np.random.RandomState(sample_seed)
        idx = np.sort(rng.choice(total_n, size=n, replace=False))
        sampled = adata[idx].to_memory()

    print(
        f"[data] donor {donor_path.stem}: sampled {n}/{adata.n_obs} cells "
        f"(seed={sample_seed}, {time.time() - t0:.1f}s)",
        flush=True,
    )
    return sampled, total_n, int(n)


def sample_balanced_val_donors(
    donor_csv: Path,
    donor_dir: Path,
    split_seed: int,
    sample_seed: int,
    concat_join: str,
    per_donor_n: int | None = None,
    donor_obs_key: str = "_eval_donor_id",
    batch_key: str = "_balanced_eval_batch",
    progress_desc: str = "loading val donors",
):
    val_donors = split_val_donors(donor_csv, val_ratio=0.1, seed=split_seed)
    existing = [(did, n) for did, n in val_donors if (donor_dir / f"{did}.h5ad").exists()]
    if not existing:
        raise RuntimeError("No val donor files exist for balanced evaluation.")

    inferred_per_donor_n = min(n for _, n in existing)
    if per_donor_n is None or int(per_donor_n) <= 0:
        per_donor_n = inferred_per_donor_n
    else:
        per_donor_n = min(int(per_donor_n), inferred_per_donor_n)
    print(
        f"[data] balanced validation subset: {len(existing)} donors, "
        f"{per_donor_n} cells per donor, split_seed={split_seed}, sample_seed={sample_seed}",
        flush=True,
    )
    adatas = []
    donor_stats = []
    donor_iter = tqdm(existing, total=len(existing), desc=progress_desc, unit="donor")
    for donor_idx, (donor_id, donor_cells_total) in enumerate(donor_iter):
        donor_path = donor_dir / f"{donor_id}.h5ad"
        donor_seed = int(sample_seed + donor_idx)
        sampled_adata, _, sampled_n = sample_single_donor_cells(
            donor_path=donor_path,
            sample_n=per_donor_n,
            sample_seed=donor_seed,
        )
        sampled_adata.obs = sampled_adata.obs.copy()
        sampled_adata.obs[donor_obs_key] = donor_id
        adatas.append(sampled_adata)
        donor_stats.append(
            {
                "donor_id": donor_id,
                "donor_cells_total": int(donor_cells_total),
                "sampled_cells": int(sampled_n),
                "sample_seed": donor_seed,
            }
        )

    print("[data] concatenating sampled donor subsets", flush=True)
    adata = (
        adatas[0].concatenate(
            *adatas[1:],
            join=concat_join,
            batch_key=batch_key,
            index_unique=None,
        )
        if len(adatas) > 1
        else adatas[0]
    )
    print(f"[data] balanced subset ready: {adata.n_obs} cells", flush=True)
    return adata, donor_stats, per_donor_n
