import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT / "benchmark"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run scATAC imputation, convert output for benchmark evaluation, and run biological eval."
    )
    parser.add_argument("--dataset", required=True, choices=["mouse_top5k", "mouse_top50k", "mouse_full", "pbmc_top50k", "pbmc_full"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Reference/input TF-IDF h5ad used by impute_atac.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Imputed h5ad output path")
    parser.add_argument("--mode", default="all_zero", choices=["masked_eval", "all_zero"])
    parser.add_argument("--split", default="val", choices=["all", "train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--spectral_coords", type=str, default=None)
    parser.add_argument("--expr_transform", type=str, default=None)
    parser.add_argument("--inverse_expr_transform", type=str, default=None)
    parser.add_argument("--rank_normalize", action="store_true")
    parser.add_argument("--skip_impute", action="store_true", help="Reuse --output and only run conversion/eval.")
    parser.add_argument("--skip_convert", action="store_true", help="Reuse *_eval.h5ad and only run biological eval.")
    parser.add_argument("--skip_eval", action="store_true", help="Only run imputation and conversion.")
    parser.add_argument("--eval_h5ad", type=str, default=None)
    parser.add_argument("--conversion_summary_json", type=str, default=None)
    parser.add_argument("--biological_eval_json", type=str, default=None)
    return parser.parse_args()


def _run(cmd):
    print("[run]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def _read_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _default_eval_paths(output):
    output = Path(output).resolve()
    stem = output.stem
    eval_h5ad = output.with_name(f"{stem}_eval.h5ad")
    summary_json = output.with_name(f"{stem}_eval.summary.json")
    bio_json = output.with_name(f"{stem}.biological_eval.json")
    return eval_h5ad, summary_json, bio_json


def main():
    args = parse_args()
    cfg = _read_config(args.config)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    eval_h5ad_default, summary_json_default, bio_json_default = _default_eval_paths(output)
    eval_h5ad = Path(args.eval_h5ad).resolve() if args.eval_h5ad else eval_h5ad_default
    summary_json = (
        Path(args.conversion_summary_json).resolve()
        if args.conversion_summary_json
        else summary_json_default
    )
    bio_json = (
        Path(args.biological_eval_json).resolve()
        if args.biological_eval_json
        else bio_json_default
    )

    train_cfg = cfg.get("training", {})
    expr_transform = args.expr_transform or str(train_cfg.get("expression_transform", "identity"))
    inverse_expr_transform = args.inverse_expr_transform or expr_transform
    if inverse_expr_transform not in {"identity", "tfidf_log1p_1e4"}:
        raise ValueError(
            "convert_for_eval.py only supports inverse_expr_transform in "
            "{'identity', 'tfidf_log1p_1e4'}; got "
            f"{inverse_expr_transform!r}"
        )

    if not args.skip_impute:
        impute_cmd = [
            sys.executable,
            ROOT / "impute_atac.py",
            "--config",
            args.config,
            "--data",
            args.data,
            "--checkpoint",
            args.checkpoint,
            "--output",
            output,
            "--mode",
            args.mode,
            "--split",
            args.split,
            "--batch_size",
            int(args.batch_size),
        ]
        if args.max_cells is not None:
            impute_cmd += ["--max_cells", int(args.max_cells)]
        if args.spectral_coords:
            impute_cmd += ["--spectral_coords", args.spectral_coords]
        if args.expr_transform:
            impute_cmd += ["--expr_transform", args.expr_transform]
        _run(impute_cmd)

    if not args.skip_convert:
        convert_cmd = [
            sys.executable,
            BENCHMARK_ROOT / "convert_for_eval.py",
            "--method",
            "scUniFlow",
            "--input",
            output,
            "--reference_h5ad",
            args.data,
            "--inverse_expr_transform",
            inverse_expr_transform,
            "--output",
            eval_h5ad,
            "--summary_json",
            summary_json,
        ]
        if args.rank_normalize:
            convert_cmd.append("--rank_normalize")
        _run(convert_cmd)

    if not args.skip_eval:
        _run([
            BENCHMARK_ROOT / "eval_method.sh",
            args.dataset,
            eval_h5ad,
            bio_json,
        ])

    summary = {
        "dataset": args.dataset,
        "config": str(Path(args.config).resolve()),
        "data": str(Path(args.data).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "imputed_h5ad": str(output),
        "eval_h5ad": str(eval_h5ad),
        "conversion_summary_json": str(summary_json),
        "biological_eval_json": str(bio_json),
        "mode": args.mode,
        "split": args.split,
        "expr_transform": expr_transform,
        "inverse_expr_transform": inverse_expr_transform,
        "rank_normalize": bool(args.rank_normalize),
        "skip_impute": bool(args.skip_impute),
        "skip_convert": bool(args.skip_convert),
        "skip_eval": bool(args.skip_eval),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
