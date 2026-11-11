#!/usr/bin/env python
import argparse
import json
from pathlib import Path


DEFAULT_SUMMARIES = {
    "scUniFlow": "./src/scuniflow/tasks/task1_read_depth_enhancement/scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head/eval_scuniflow_val_shared_common/summary.json",
    "scGPT head e1": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_scgpt_rde_scuniflow_train12k_val6858_e1/eval_scuniflow_val_shared_common/summary.json",
    "Geneformer head e1": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_geneformer_rde_scuniflow_train12k_val6858_ft1_head1/eval_scuniflow_val_shared_common/summary.json",
    "UCE head e8": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_uce_rde_scuniflow_train12k_val6858_e8/eval_scuniflow_val_shared_common/summary.json",
    "Nicheformer head e8": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_nicheformer_rde_scuniflow_train12k_val6858_e8/eval_scuniflow_val_shared_common/summary.json",
    "Low-depth": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_classical_lowdepth_val_shared_common/summary.json",
    "MAGIC": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_magic_hvg2000_val_shared_common/summary.json",
    "SAVER": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_saver_hvg2000_val_shared_common/summary.json",
    "scImpute": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_scimpute_hvg2000_val_shared_common/summary.json",
    "scVI": "./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/runs/pbmc68k_scvi_hvg2000_val_shared_common/summary.json",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def fmt(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "-"


def row_from_summary(name: str, path: Path):
    if not path.exists():
        return [name, "-", "-", "-", "-", "-", "-", "missing"]
    data = load_json(path)
    enhanced = (data.get("clustering") or {}).get("enhanced") or data.get("enhanced") or {}
    recon = data.get("reconstruction") or data
    return [
        name,
        fmt(enhanced.get("nmi")),
        fmt(enhanced.get("ari")),
        fmt(enhanced.get("calinski_harabasz")),
        fmt(enhanced.get("davies_bouldin")),
        fmt(recon.get("mean_cell_pcc")),
        fmt(recon.get("median_cell_pcc")),
        str(path),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect RDE summary.json files into one markdown table.")
    parser.add_argument("--output", default="./src/scuniflow/tasks/task1_read_depth_enhancement/read_depth_enhancement/rde_results_summary.md")
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Additional entry as NAME=PATH. Can be passed multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    entries = dict(DEFAULT_SUMMARIES)
    for item in args.summary:
        if "=" not in item:
            raise ValueError(f"--summary expects NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        entries[name] = path

    lines = [
        "# RDE Results Summary",
        "",
        "Fixed eval split: `pbmc68k_scuniflow_val_seed42_indices.npy`; shared target labels: `pbmc68k_scuniflow_val_seed42_ref_labels_res0.4_n9.npy`.",
        "",
        "| Model | NMI (↑) | ARI (↑) | C-H (↑) | D-B (↓) | Mean PCC (↑) | Median PCC (↑) | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, path in entries.items():
        row = row_from_summary(name, Path(path))
        lines.append("| " + " | ".join(row) + " |")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
