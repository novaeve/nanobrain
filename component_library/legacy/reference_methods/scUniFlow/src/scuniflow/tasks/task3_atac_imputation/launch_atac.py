import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="scUniFlow scATAC 训练启动器")
    parser.add_argument("--config", default=str(ROOT / "config_atac.yaml"), help="config.yaml 路径")
    parser.add_argument("--data", required=True, help="TF-IDF ATAC h5ad 路径")
    parser.add_argument("--run_name", default=None, help="实验名称")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--spectral_coords", default=None)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--expr_transform", default=None, choices=["identity", "log1p", "logcpm"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    n_gpus = int(train_cfg.get("n_gpus", 1))
    gpu_ids = str(train_cfg.get("gpu_ids", "0"))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

    train_args = [
        "--config",
        args.config,
        "--data",
        args.data,
    ]
    if args.run_name:
        train_args += ["--run_name", args.run_name]
    if args.batch_size is not None:
        train_args += ["--batch_size", str(args.batch_size)]
    if args.spectral_coords is not None:
        train_args += ["--spectral_coords", args.spectral_coords]
    if args.max_cells is not None:
        train_args += ["--max_cells", str(args.max_cells)]
    if args.resume is not None:
        train_args += ["--resume", args.resume]
    if args.expr_transform is not None:
        train_args += ["--expr_transform", args.expr_transform]

    train_script = str(ROOT / "train_atac.py")
    if n_gpus > 1:
        master_port = int(train_cfg.get("master_port", 29500))
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            f"--master_port={master_port}",
            train_script,
        ] + train_args
    else:
        cmd = [sys.executable, train_script] + train_args

    print(f"CUDA_VISIBLE_DEVICES={gpu_ids}")
    print("CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
