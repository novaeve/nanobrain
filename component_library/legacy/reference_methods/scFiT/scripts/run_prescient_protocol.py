import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scfit.route_a.prescient import run_prescient_protocol


def main():
    parser = argparse.ArgumentParser(description="Run PRESCIENT on a Route A protocol")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--scnode-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assets-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-dim", type=int, default=100)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--train-epochs", type=int, default=2000)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--train-sd", type=float, default=0.5)
    parser.add_argument("--train-tau", type=float, default=1e-6)
    parser.add_argument("--train-clip", type=float, default=0.25)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=0)
    args = parser.parse_args()

    run_prescient_protocol(
        protocol_path=args.protocol,
        scnode_root=args.scnode_root,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir or None,
        device=args.device,
        k_dim=args.k_dim,
        layers=args.layers,
        train_epochs=args.train_epochs,
        train_lr=args.train_lr,
        train_sd=args.train_sd,
        train_tau=args.train_tau,
        train_clip=args.train_clip,
        pretrain_epochs=args.pretrain_epochs,
        num_steps=(None if int(args.num_steps) <= 0 else int(args.num_steps)),
    )


if __name__ == "__main__":
    main()
