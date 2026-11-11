"""
训练启动器 — 从 config.yaml 读取所有参数后调用 torchrun。

用法:
    python launch.py --config experiments/flow_v3/config.yaml --data mtg_full
    python launch.py --config experiments/flow_v3/config.yaml --data mtg_full --run_name my_exp
"""

import argparse
import os
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="scUniFlow 单阶段 epoch 训练启动器")
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument("--data", default="mtg_toy",
                        help="数据键名或 h5ad 路径（默认 mtg_toy）")
    parser.add_argument("--run_name", default=None,
                        help="实验名称，默认为 {data}_{timestamp}")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    n_gpus    = int(train_cfg.get("n_gpus", 1))
    gpu_ids   = str(train_cfg.get("gpu_ids", "0"))

    # 设置 CUDA_VISIBLE_DEVICES，让进程只看到指定的卡
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")  # 数据加载可能超过默认480s

    # 构造 train.py 的参数列表
    train_args = [
        "--config", args.config,
        "--data",   args.data,
    ]
    if args.run_name:
        train_args += ["--run_name", args.run_name]

    # 构造完整命令
    if n_gpus > 1:
        master_port = int(train_cfg.get("master_port", 29500))
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            f"--master_port={master_port}",
            "train.py",
        ] + train_args
    else:
        cmd = [sys.executable, "train.py"] + train_args

    print(f"CUDA_VISIBLE_DEVICES={gpu_ids}")
    print("CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
