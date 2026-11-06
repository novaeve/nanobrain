#!/usr/bin/env python3
"""Run an end-to-end smoke test on a tiny synthetic Route A dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import yaml


def _write_synthetic_bundle(root: Path) -> tuple[Path, Path]:
    bundle = root / "bundle"
    config_dir = root / "configs"
    bundle.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    time_bins = ["0", "1", "2"]
    rows = []
    obs_names = []
    expr_parts = []
    for time_index, time_bin in enumerate(time_bins):
        for cell_index in range(8):
            obs_name = f"cell_{time_bin}_{cell_index}"
            cell_type = "type_a" if cell_index < 4 else "type_b"
            base = np.linspace(0.2, 1.0, 8, dtype=np.float32)
            trend = time_index * np.linspace(0.05, 0.25, 8, dtype=np.float32)
            type_shift = (0.3 if cell_type == "type_b" else 0.0) * np.eye(1, 8, 3, dtype=np.float32).ravel()
            noise = rng.normal(0, 0.03, 8).astype(np.float32)
            expr_parts.append(np.clip(base + trend + type_shift + noise, 0, None))
            obs_names.append(obs_name)
            rows.append(
                {
                    "obs_name": obs_name,
                    "author_day": time_bin,
                    "author_day_numeric": float(time_bin),
                    "author_cell_type": cell_type,
                }
            )

    expression = np.asarray(expr_parts, dtype=np.float32)
    metadata = pd.DataFrame(rows, index=obs_names)
    gene_names = np.asarray([f"Gene{i}" for i in range(expression.shape[1])], dtype=str)

    np.save(bundle / "expression_log1p_clipped.npy", expression)
    metadata.to_pickle(bundle / "cell_metadata.pkl")
    metadata.to_csv(bundle / "cell_metadata.csv.gz", compression="gzip")
    np.save(bundle / "obs_names.npy", np.asarray(obs_names, dtype=str))
    np.save(bundle / "author_day_numeric.npy", metadata["author_day_numeric"].to_numpy(dtype=np.float32))
    np.save(bundle / "hvg_gene_indices.npy", np.arange(expression.shape[1], dtype=np.int64))
    np.save(bundle / "hvg_var_names.npy", gene_names)
    np.save(bundle / "hvg_gene_names.npy", gene_names)
    (bundle / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "input_h5ad": "synthetic_smoke",
                "output_dir": str(bundle),
                "matrix_source_used": "synthetic",
                "n_cells": int(expression.shape[0]),
                "n_genes": int(expression.shape[1]),
                "original_n_vars": int(expression.shape[1]),
                "time_norm_denom": 2.0,
                "artifacts": {
                    "expression_memmap": str(bundle / "expression_log1p_clipped.npy"),
                    "obs_names": str(bundle / "obs_names.npy"),
                    "hvg_gene_indices": str(bundle / "hvg_gene_indices.npy"),
                    "hvg_var_names": str(bundle / "hvg_var_names.npy"),
                    "hvg_gene_names": str(bundle / "hvg_gene_names.npy"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    protocol_path = config_dir / "protocol.yaml"
    config_path = config_dir / "config.yaml"
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "protocol_name": "synthetic_smoke_v1",
                "data_bundle_path": str(bundle),
                "eligible_time_bins": time_bins,
                "observed_time_bins": ["0", "1"],
                "heldout_time_bins": ["2"],
                "task_type": "three_forecasting",
                "source_policy": "previous_observed",
                "direction_mode": "forward_only",
                "eval_time_bins": ["2"],
                "normalization": "synthetic",
                "hvg_dim": 8,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "source": "synthetic_smoke",
                    "preprocessed_dir": str(bundle),
                    "time_col": "author_day",
                    "cell_type_col": "author_cell_type",
                },
                "protocol": {"path": str(protocol_path)},
                "experiment": {"output_dir": str(root / "results" / "experiments"), "run_name": "smoke_run"},
                "model": {
                    "top_k_genes": 8,
                    "d_model": 16,
                    "nhead": 2,
                    "d_ff": 32,
                    "nlayers": 1,
                    "dropout": 0.0,
                    "value_encoder": {"max_value": 4.0},
                    "adaln": {"share_projection": False},
                },
                "training": {
                    "batch_size": 4,
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "epochs": 1,
                    "iters_per_epoch": 1,
                    "seed": 42,
                    "log_every_steps": 1,
                    "val_every_steps": 999,
                    "save_every_epochs": 1,
                    "amp": False,
                    "grad_clip": 1.0,
                    "resume": None,
                    "gpu_reservation": {"enabled": False, "reserve_gb": None, "fraction_of_free": 0.3},
                    "loss_weights": {
                        "cfm": 1.0,
                        "wasserstein_hvg": 0.1,
                        "wasserstein_pca50": 0.05,
                        "mean": 0.1,
                        "variance": 0.0,
                    },
                    "variance_loss_use_log": True,
                    "variance_loss_eps": 1e-4,
                },
                "validation": {"max_cells_per_time": 4, "metric_names": ["wasserstein_pca50"]},
                "inference": {"n_steps": 2, "micro_batch_size": 4},
                "ot": {"epsilon": 0.05, "max_iter": 20},
                "evaluation": {"assets_dir": str(root / "results" / "assets"), "batch_size": 4, "max_cells_per_time": 4},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return protocol_path, config_path


def _run(cmd: list[str]) -> None:
    print("[smoke]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic end-to-end smoke test.")
    parser.add_argument("--work-dir", default="/tmp/scfit_smoke", help="Temporary output directory.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    protocol_path, config_path = _write_synthetic_bundle(work_dir)
    artifact_dir = work_dir / "results" / "experiments" / "smoke_run" / "eval_artifacts" / "final"
    assets_dir = work_dir / "results" / "assets"

    _run(
        [
            sys.executable,
            "scripts/prepare_route_a_assets.py",
            "--protocol",
            str(protocol_path),
            "--output-dir",
            str(assets_dir),
            "--pca-components",
            "4",
            "--knn-k",
            "3",
            "--marker-top-k",
            "3",
        ]
    )
    _run([sys.executable, "scripts/train.py", "--config", str(config_path)])
    _run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--protocol",
            str(protocol_path),
            "--artifact-dir",
            str(artifact_dir),
            "--assets-dir",
            str(assets_dir),
            "--epsilon",
            "0.05",
            "--max-iter",
            "20",
        ]
    )
    print(f"[smoke] completed successfully; outputs: {work_dir}")


if __name__ == "__main__":
    main()
