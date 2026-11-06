import argparse
from pathlib import Path

from common import DEFAULT_REF_H5AD, generate_ordered_coords_for_reference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate deterministic ordered gene coords for mouse brain scUniFlow training"
    )
    parser.add_argument("--ref_h5ad", default=str(DEFAULT_REF_H5AD))
    parser.add_argument(
        "--output",
        default="./src/scuniflow/tasks/task2_spatial_gene_imputation/mouse_brain_ordered_coords_177.npy",
    )
    parser.add_argument("--grid_size", type=int, default=177)
    return parser.parse_args()


def main():
    args = parse_args()
    ref, coords = generate_ordered_coords_for_reference(
        ref_h5ad=Path(args.ref_h5ad).resolve(),
        output_path=Path(args.output).resolve(),
        grid_size=int(args.grid_size),
    )
    print(
        f"[saved] coords shape={coords.shape} for {ref.n_cells} reference cells -> {Path(args.output).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
