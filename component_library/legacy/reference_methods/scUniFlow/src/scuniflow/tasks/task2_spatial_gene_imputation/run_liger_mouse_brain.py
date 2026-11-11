import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from benchmark_common import dump_r_inputs, prepare_benchmark_data, write_benchmark_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and run LIGER baseline on mouse brain snRNA + Visium holdout benchmark"
    )
    parser.add_argument("--ref_h5ad", default="data/all_cells_20200625.h5ad")
    parser.add_argument("--visium_root", default="data/mouse_brain_visium/mouse_brain_visium_wo_cloupe_data/rawdata")
    parser.add_argument("--sample_ids", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--panel_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_ref_cells", type=int, default=8000)
    parser.add_argument("--max_spots", type=int, default=0)
    parser.add_argument("--in_tissue_only", action="store_true")
    parser.add_argument("--observed_gene_count", type=int, default=1024)
    parser.add_argument("--target_gene_count", type=int, default=256)
    parser.add_argument("--min_nonzero_spots", type=int, default=10)
    parser.add_argument("--min_nonzero_ref_cells", type=int, default=20)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--save_h5ad", action="store_true")
    parser.add_argument("--save_truth", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sample_ids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
    benchmark = prepare_benchmark_data(
        ref_h5ad=args.ref_h5ad,
        visium_root=args.visium_root,
        sample_ids=sample_ids or None,
        panel_seed=int(args.panel_seed),
        sample_seed=int(args.sample_seed),
        max_ref_cells=int(args.max_ref_cells),
        max_spots=int(args.max_spots),
        in_tissue_only=bool(args.in_tissue_only),
        observed_gene_count=int(args.observed_gene_count),
        target_gene_count=int(args.target_gene_count),
        min_nonzero_spots=int(args.min_nonzero_spots),
        min_nonzero_ref_cells=int(args.min_nonzero_ref_cells),
    )

    output_dir = Path(args.output_dir).resolve()
    input_dir = output_dir / "r_inputs"
    dump_r_inputs(input_dir, benchmark)

    rscript = shutil.which(args.rscript)
    runner = Path(__file__).resolve().parent / "run_liger_mouse_brain.R"
    if rscript is None:
        raise RuntimeError(
            "Rscript not found. Inputs are already prepared at "
            f"{input_dir}. After installing R/LIGER, run:\n"
            f"  {args.rscript} {runner} {input_dir} {output_dir / 'predicted_target_logcpm.csv.gz'}"
        )

    pred_path = output_dir / "predicted_target_logcpm.csv.gz"
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([rscript, str(runner), str(input_dir), str(pred_path)], check=True, env=env)

    pred_df = pd.read_csv(pred_path, index_col=0)
    write_benchmark_outputs(
        output_dir=output_dir,
        method="LIGER",
        benchmark=benchmark,
        pred_targets_logcpm=pred_df.to_numpy(dtype="float32", copy=False),
        extra_summary={
            "sample_ids": sample_ids if sample_ids else "all",
            "panel_seed": int(args.panel_seed),
            "sample_seed": int(args.sample_seed),
            "min_nonzero_spots": int(args.min_nonzero_spots),
            "min_nonzero_ref_cells": int(args.min_nonzero_ref_cells),
        },
        save_h5ad=bool(args.save_h5ad),
        save_truth=bool(args.save_truth),
    )
    print(f"[done] LIGER summary -> {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
