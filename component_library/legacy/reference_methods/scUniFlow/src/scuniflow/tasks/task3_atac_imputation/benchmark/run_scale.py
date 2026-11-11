#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run SCALE and normalize its output location")
    parser.add_argument("--input", required=True, help="Raw-count ATAC h5ad")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--output", required=True, help="Final h5ad path for benchmark evaluation")
    parser.add_argument("--n_feature", type=int, default=-1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale_cmd", default="SCALE.py")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    output = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.scale_cmd,
        "--data_list", str(Path(args.input).resolve()),
        "--outdir", str(outdir),
        "--impute",
        "--n_feature", str(args.n_feature),
        "--gpu", str(args.gpu),
        "--seed", str(args.seed),
    ] + list(args.extra_args)
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    candidates = [outdir / "adata.h5ad", outdir / "imputed.h5ad", outdir / "scale_output.h5ad"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        h5ads = sorted(outdir.glob("*.h5ad"))
        if not h5ads:
            raise FileNotFoundError(f"No h5ad output found under {outdir}")
        src = h5ads[0]
    if src.resolve() != output.resolve():
        shutil.copy2(src, output)

    summary = {
        "method": "SCALE",
        "command": cmd,
        "source_output": str(src),
        "output": str(output),
    }
    with open(output.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
