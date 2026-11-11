import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scfit.route_a.scnode import run_scnode_protocol


def _parse_int_list(value: str):
    value = str(value or "").strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run scNODE on a Route A protocol")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--scnode-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assets-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--drift-latent-size", default="")
    parser.add_argument("--enc-latent-list", default="")
    parser.add_argument("--dec-latent-list", default="")
    parser.add_argument("--latent-coeff", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-iters", type=int, default=200)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_scnode_protocol(
        protocol_path=args.protocol,
        scnode_root=args.scnode_root,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir or None,
        device=args.device,
        latent_dim=args.latent_dim,
        drift_latent_size=_parse_int_list(args.drift_latent_size),
        enc_latent_list=_parse_int_list(args.enc_latent_list),
        dec_latent_list=_parse_int_list(args.dec_latent_list),
        latent_coeff=args.latent_coeff,
        epochs=args.epochs,
        iters=args.iters,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrain_iters=args.pretrain_iters,
        pretrain_lr=args.pretrain_lr,
    )


if __name__ == "__main__":
    main()
