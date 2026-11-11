import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from route_a.eval import evaluate_artifact_dir
from route_a.protocol import load_protocol_bundle


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate a Route A prediction artifact.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing artifact_summary.json and artifact_arrays.npz")
    parser.add_argument("--protocol", required=True, help="Protocol yaml/json used to build the artifact")
    parser.add_argument("--assets-dir", required=True, help="Route A eval assets directory")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn epsilon")
    parser.add_argument("--max-iter", type=int, default=100, help="Sinkhorn max iterations")
    args = parser.parse_args()

    bundle = load_protocol_bundle(args.protocol)
    metrics = evaluate_artifact_dir(
        artifact_dir=args.artifact_dir,
        assets_dir=args.assets_dir,
        metadata_df=bundle["metadata_df"],
        hvg_gene_names=bundle["bundle_meta"]["hvg_gene_names"],
        epsilon=float(args.epsilon),
        max_iter=int(args.max_iter),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
