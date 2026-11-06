import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scfit.route_a.benchmark_registry import (  # noqa: E402
    BENCHMARK_PREPROCESSED_ROOT,
    BENCHMARK_PROTOCOL_ROOT,
    get_benchmark_task_spec,
    list_supported_benchmark_tasks,
)
from scfit.paths import project_root  # noqa: E402


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _canonical_time_label(value) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        rounded = float(value)
        if abs(rounded - round(rounded)) < 1e-8:
            return str(int(round(rounded)))
        return str(rounded)
    s = str(value).strip()
    try:
        rounded = float(s)
        if abs(rounded - round(rounded)) < 1e-8:
            return str(int(round(rounded)))
    except ValueError:
        pass
    return s


def _parse_hour_window_midpoint(label: str) -> float:
    s = str(label)
    if not s.startswith("hrs_"):
        raise ValueError(f"unsupported drosophila time label: {label}")
    left, right = s.replace("hrs_", "", 1).split("_", 1)
    return 0.5 * (float(left) + float(right))


def _resolve_numeric_time(metadata_df: pd.DataFrame, spec: dict) -> np.ndarray:
    if spec["dataset_key"] == "drosophila":
        time_values = metadata_df[spec["time_label_col"]].tolist()
        return np.asarray([_parse_hour_window_midpoint(x) for x in time_values], dtype=np.float32)
    if spec["dataset_key"] == "wot":
        time_values = metadata_df[spec["time_label_col"]].tolist()
        return np.asarray([float(x) for x in time_values], dtype=np.float32)

    if spec.get("time_numeric_col") and spec["time_numeric_col"] in metadata_df.columns:
        return metadata_df[spec["time_numeric_col"]].to_numpy(dtype=np.float32)

    for col in spec.get("time_numeric_candidates", []) or []:
        if col in metadata_df.columns:
            series = metadata_df[col]
            if pd.api.types.is_numeric_dtype(series):
                return series.to_numpy(dtype=np.float32)

    raise ValueError(f"unable to infer numeric time for dataset={spec['dataset_key']}")


def _normalize_expression(expr_df: pd.DataFrame, normalization_mode: str) -> np.ndarray:
    expr = expr_df.to_numpy(dtype=np.float32, copy=True)
    if normalization_mode == "provided_log1p_hvg_clip_0_10":
        return np.clip(expr, 0.0, 10.0).astype(np.float32, copy=False)
    if normalization_mode == "librarysize_10000_log1p_clip_0_10_from_count_hvg":
        totals = expr.sum(axis=1, keepdims=True)
        totals = np.clip(totals, 1e-8, None)
        expr = expr / totals * 10000.0
        expr = np.log1p(expr)
        return np.clip(expr, 0.0, 10.0).astype(np.float32, copy=False)
    raise ValueError(f"unsupported normalization_mode={normalization_mode}")


def _load_gene_names(gene_file: Path) -> np.ndarray:
    gene_df = pd.read_csv(gene_file)
    if gene_df.shape[1] == 0:
        raise ValueError(f"empty gene list: {gene_file}")
    gene_names = gene_df.iloc[:, -1].astype(str).to_numpy()
    return gene_names


def _load_metadata(expr_index: np.ndarray, spec: dict, split_key: str) -> pd.DataFrame:
    if "meta_file" in spec:
        meta_file = Path(spec["meta_file"])
    else:
        meta_file = Path(spec["data_dir"]) / f"{split_key}-{spec['meta_suffix']}"
    meta_df = pd.read_csv(meta_file)
    obs_name_col = spec["obs_name_col"]
    if obs_name_col not in meta_df.columns and meta_df.columns[0] == obs_name_col:
        pass
    if obs_name_col not in meta_df.columns:
        raise ValueError(f"metadata missing obs_name_col={obs_name_col}: {meta_file}")
    meta_df = meta_df.set_index(obs_name_col)
    missing = np.setdiff1d(expr_index.astype(str), meta_df.index.astype(str), assume_unique=False)
    if missing.size > 0:
        raise ValueError(f"metadata missing {missing.size} cells from expression matrix")
    meta_df = meta_df.loc[expr_index.astype(str)].copy()
    return meta_df


def _build_output_metadata(meta_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=meta_df.index.astype(str))
    out["author_day"] = meta_df[spec["time_label_col"]].map(_canonical_time_label).astype(str)
    out["author_day_numeric"] = _resolve_numeric_time(meta_df, spec).astype(np.float32)
    if spec.get("cell_type_col") and spec["cell_type_col"] in meta_df.columns:
        out["author_cell_type"] = meta_df[spec["cell_type_col"]].astype(str).to_numpy()
        out["cell_type"] = out["author_cell_type"].to_numpy()
    if spec.get("donor_id_col") and spec["donor_id_col"] in meta_df.columns:
        out["donor_id"] = meta_df[spec["donor_id_col"]].astype(str).to_numpy()
    return out


def _ordered_time_bins(metadata_df: pd.DataFrame):
    time_df = (
        metadata_df.groupby("author_day", as_index=False)["author_day_numeric"]
        .median()
        .sort_values(["author_day_numeric", "author_day"], kind="stable")
    )
    return time_df["author_day"].astype(str).tolist()


def _build_protocol_manifest(metadata_df: pd.DataFrame, spec: dict, bundle_dir: Path):
    ordered_bins = _ordered_time_bins(metadata_df)
    split_train = spec["train_idx"]
    split_test = spec["test_idx"]
    observed_bins = [ordered_bins[i] for i in split_train]
    heldout_bins = [ordered_bins[i] for i in split_test]
    return {
        "protocol_name": spec["protocol_name"],
        "data_bundle_path": _repo_relative(bundle_dir),
        "eligible_time_bins": ordered_bins,
        "observed_time_bins": observed_bins,
        "heldout_time_bins": heldout_bins,
        "task_type": spec["task_type"],
        "source_policy": spec["source_policy"],
        "direction_mode": spec["direction_mode"],
        "eval_time_bins": heldout_bins,
        "display_time_mode": "index_0based",
        "normalization": spec["normalization_mode"],
        "hvg_dim": int(metadata_df.attrs["hvg_dim"]),
    }


def prepare_bundle(dataset: str, split: str, output_dir: str = None, protocol_output: str = None):
    spec = get_benchmark_task_spec(dataset, split)
    data_dir = Path(spec["data_dir"])
    expr_file = data_dir / f"{split}-{spec['expr_suffix']}"
    gene_file = data_dir / f"{split}-{spec['gene_suffix']}"
    if not expr_file.exists():
        raise FileNotFoundError(f"missing expression file: {expr_file}")
    if not gene_file.exists():
        raise FileNotFoundError(f"missing gene file: {gene_file}")

    bundle_dir = (
        Path(output_dir)
        if output_dir
        else BENCHMARK_PREPROCESSED_ROOT / f"{spec['dataset_key']}_{split}_hvg2000"
    )
    protocol_path = (
        Path(protocol_output)
        if protocol_output
        else BENCHMARK_PROTOCOL_ROOT / f"{spec['dataset_key']}_{split}.yaml"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark_bundle] dataset={dataset} split={split}")
    print(f"[benchmark_bundle] expr_file={expr_file}")
    expr_df = pd.read_csv(expr_file, index_col=0)
    expr_df.index = expr_df.index.astype(str)
    gene_names = _load_gene_names(gene_file)
    if expr_df.shape[1] != gene_names.shape[0]:
        raise ValueError(
            f"gene count mismatch: expr={expr_df.shape[1]} gene_file={gene_names.shape[0]}"
        )
    expr_df.columns = gene_names

    meta_df = _load_metadata(expr_df.index.to_numpy(), spec, split_key=split)
    out_meta = _build_output_metadata(meta_df, spec)
    expr_np = _normalize_expression(expr_df, normalization_mode=spec["normalization_mode"])

    np.save(bundle_dir / "expression_log1p_clipped.npy", expr_np)
    out_meta.to_pickle(bundle_dir / "cell_metadata.pkl")
    out_meta.to_csv(bundle_dir / "cell_metadata.csv.gz", compression="gzip")
    np.save(bundle_dir / "obs_names.npy", out_meta.index.to_numpy(dtype=str))
    np.save(bundle_dir / "author_day_numeric.npy", out_meta["author_day_numeric"].to_numpy(dtype=np.float32))
    np.save(bundle_dir / "hvg_gene_indices.npy", np.arange(expr_np.shape[1], dtype=np.int64))
    np.save(bundle_dir / "hvg_var_names.npy", gene_names.astype(str))
    np.save(bundle_dir / "hvg_gene_names.npy", gene_names.astype(str))

    dataset_meta = {
        "input_h5ad": _repo_relative(expr_file),
        "output_dir": _repo_relative(bundle_dir),
        "matrix_source_used": spec["normalization_mode"],
        "n_cells": int(expr_np.shape[0]),
        "n_genes": int(expr_np.shape[1]),
        "original_n_vars": int(expr_np.shape[1]),
        "target_sum": 10000.0 if "librarysize_10000" in spec["normalization_mode"] else None,
        "clip_min": 0.0,
        "clip_max": 10.0,
        "chunk_size": 2048,
        "time_col": "author_day",
        "time_norm_denom": float(np.nanmax(out_meta["author_day_numeric"].to_numpy(dtype=np.float32))),
        "artifacts": {
            "expression_memmap": str((bundle_dir / "expression_log1p_clipped.npy").resolve()),
            "cell_metadata_pickle": str((bundle_dir / "cell_metadata.pkl").resolve()),
            "cell_metadata_csv": str((bundle_dir / "cell_metadata.csv.gz").resolve()),
            "obs_names": str((bundle_dir / "obs_names.npy").resolve()),
            "author_day_numeric": str((bundle_dir / "author_day_numeric.npy").resolve()),
            "hvg_gene_indices": str((bundle_dir / "hvg_gene_indices.npy").resolve()),
            "hvg_var_names": str((bundle_dir / "hvg_var_names.npy").resolve()),
            "hvg_gene_names": str((bundle_dir / "hvg_gene_names.npy").resolve()),
        },
    }
    (bundle_dir / "dataset_metadata.json").write_text(json.dumps(dataset_meta, ensure_ascii=False, indent=2))
    (bundle_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {spec['protocol_name']}",
                "",
                f"- dataset: `{dataset}`",
                f"- split: `{split}`",
                f"- cells: `{expr_np.shape[0]}`",
                f"- genes: `{expr_np.shape[1]}`",
                f"- normalization: `{spec['normalization_mode']}`",
            ]
        ),
        encoding="utf-8",
    )

    out_meta.attrs["hvg_dim"] = int(expr_np.shape[1])
    manifest = _build_protocol_manifest(out_meta, spec, bundle_dir=bundle_dir)
    protocol_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    summary = {
        "dataset": dataset,
        "split": split,
        "bundle_dir": str(bundle_dir.resolve()),
        "protocol_path": str(protocol_path.resolve()),
        "n_cells": int(expr_np.shape[0]),
        "n_genes": int(expr_np.shape[1]),
        "observed_time_bins": manifest["observed_time_bins"],
        "heldout_time_bins": manifest["heldout_time_bins"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare Route A bundle/protocol from official benchmark csv")
    parser.add_argument("--dataset", help="Benchmark dataset key")
    parser.add_argument("--split", help="Benchmark split key")
    parser.add_argument("--output-dir", default=None, help="Optional bundle output directory")
    parser.add_argument("--protocol-output", default=None, help="Optional protocol yaml output path")
    parser.add_argument("--list", action="store_true", help="List supported dataset/split pairs")
    args = parser.parse_args()

    if args.list:
        print(json.dumps(list_supported_benchmark_tasks(), ensure_ascii=False, indent=2))
        return

    if not args.dataset or not args.split:
        parser.error("--dataset and --split are required unless --list is used")

    prepare_bundle(
        dataset=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        protocol_output=args.protocol_output,
    )


if __name__ == "__main__":
    main()
