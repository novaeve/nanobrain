import argparse
import json

from route_a.eval import evaluate_artifact_dir
from route_a.protocol import load_protocol_bundle


def main():
    parser = argparse.ArgumentParser(description="Evaluate Route A prediction artifact")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()

    bundle = load_protocol_bundle(args.protocol)
    metrics = evaluate_artifact_dir(
        artifact_dir=args.artifact_dir,
        assets_dir=args.assets_dir,
        metadata_df=bundle["metadata_df"],
        hvg_gene_names=bundle["bundle_meta"]["hvg_gene_names"],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
