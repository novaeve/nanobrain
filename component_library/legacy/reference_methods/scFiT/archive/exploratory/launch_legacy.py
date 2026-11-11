import argparse
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="scFiT 训练启动器")
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml.safe_load(f)

    cmd = [sys.executable, "train.py", "--config", args.config]

    print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
