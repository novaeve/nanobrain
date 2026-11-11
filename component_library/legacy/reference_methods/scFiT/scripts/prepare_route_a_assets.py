import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scfit.route_a.assets import prepare_route_a_assets


def main():
    parser = argparse.ArgumentParser(description="Prepare Route A evaluation assets")
    parser.add_argument("--protocol", required=True, help="Path to protocol manifest")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--marker-top-k", type=int, default=50)
    args = parser.parse_args()

    summary = prepare_route_a_assets(
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        pca_components=args.pca_components,
        knn_k=args.knn_k,
        marker_top_k=args.marker_top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
