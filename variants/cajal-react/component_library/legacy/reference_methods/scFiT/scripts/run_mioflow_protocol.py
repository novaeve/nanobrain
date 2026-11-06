import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scfit.route_a.mioflow import run_mioflow_protocol


def _parse_int_list(value: str):
    value = str(value or "").strip()
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run MIOFlow on a Route A protocol")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--scnode-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assets-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--n-epochs-emb", type=int, default=1000)
    parser.add_argument("--gae-embedded-dim", type=int, default=50)
    parser.add_argument("--encoder-layers", default="50,50,50")
    parser.add_argument("--layers", default="50,50,50")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-local-epochs", type=int, default=40)
    parser.add_argument("--n-global-epochs", type=int, default=40)
    parser.add_argument("--n-post-local-epochs", type=int, default=0)
    parser.add_argument("--lambda-density", type=float, default=35.0)
    args = parser.parse_args()

    run_mioflow_protocol(
        protocol_path=args.protocol,
        scnode_root=args.scnode_root,
        output_dir=args.output_dir,
        assets_dir=args.assets_dir or None,
        device=args.device,
        pca_dims=args.pca_dims,
        n_epochs_emb=args.n_epochs_emb,
        gae_embedded_dim=args.gae_embedded_dim,
        encoder_layers=_parse_int_list(args.encoder_layers),
        layers=_parse_int_list(args.layers),
        batch_size=args.batch_size,
        n_local_epochs=args.n_local_epochs,
        n_global_epochs=args.n_global_epochs,
        n_post_local_epochs=args.n_post_local_epochs,
        lambda_density=args.lambda_density,
    )


if __name__ == "__main__":
    main()
