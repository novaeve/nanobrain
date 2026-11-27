"""Score predictions of 16-neighbor cell-type proportions from center expression."""

import argparse
import csv
from pathlib import Path

import numpy as np

METRIC = "Niche composition MAE"
N_TYPES = 167
N_NEIGHBORS = 16


def read_spatial_rows(data):
    """Select centers by their finite 2D coordinates, without reading neighbor labels."""
    root = Path(data)
    spatial = np.load(root / "spatial.npy", mmap_mode="r")
    source = np.load(root / "source_row.npy", mmap_mode="r")
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("expected one spatial coordinate row per cell with at least two columns")
    if source.shape != (len(spatial), 2) or source.dtype.kind not in "iu":
        raise ValueError("expected two integer source identifiers per cell")
    spatial_mask = np.isfinite(spatial[:, :2]).all(axis=1)
    rows = np.flatnonzero(spatial_mask)
    if not len(rows):
        raise ValueError("dataset has no spatial cells")
    return rows, spatial_mask, source[rows]


def read_spatial_labels(data):
    """Validate the frozen spatial graph and read each center's fine neighbor labels."""
    root = Path(data)
    rows, spatial_mask, source = read_spatial_rows(root)
    n = len(spatial_mask)
    neighbors = np.load(root / "neighbors.npy", mmap_mode="r")
    sections = np.load(root / "section.npy", mmap_mode="r")
    labels = np.load(root / "cell_type.npy", mmap_mode="r")
    if neighbors.shape != (n, N_NEIGHBORS) or neighbors.dtype.kind not in "iu":
        raise ValueError(f"expected {N_NEIGHBORS} integer neighbor indices per cell")
    if sections.shape != (n,) or sections.dtype.kind not in "iu":
        raise ValueError("expected one integer section identifier per cell")
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError("expected one integer fine cell-type label per cell")
    graph = neighbors[rows]
    if np.any(graph < 0) or np.any(graph >= n) or not spatial_mask[graph].all():
        raise ValueError("each spatial cell must have 16 in-range spatial neighbors")
    if np.any(graph == rows[:, None]) or np.any(np.diff(np.sort(graph, axis=1), axis=1) == 0):
        raise ValueError("spatial neighbors must be distinct and exclude the center")
    if np.any(sections[rows] < 0) or np.any(sections[graph] != sections[rows, None]):
        raise ValueError("spatial neighbors must be in the center's known tissue section")
    classes = labels[graph]
    if np.any(classes < 0) or np.any(classes >= N_TYPES):
        raise ValueError("all 16 neighbors of each spatial cell must have a known fine cell type")
    return rows, classes, source


def composition(classes):
    """Turn neighbor labels into an empirical distribution for each center."""
    target = np.zeros((len(classes), N_TYPES), dtype=np.float32)
    np.add.at(target, (np.arange(len(classes))[:, None], classes), 1 / N_NEIGHBORS)
    return target


def score(data, prediction):
    rows, classes, source = read_spatial_labels(data)
    with np.load(prediction, allow_pickle=False) as saved:
        if saved["rows"].dtype.kind not in "iu" or not np.array_equal(saved["rows"], rows):
            raise ValueError("prediction rows do not match the ordered spatial rows of the evaluation dataset")
        if saved["source_rows"].dtype.kind not in "iu" or not np.array_equal(saved["source_rows"], source):
            raise ValueError("prediction source rows do not match the evaluation dataset")
        probabilities = saved["probabilities"]
    if (
        probabilities.shape != (len(rows), N_TYPES)
        or probabilities.dtype.kind != "f"
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(f"expected one normalized {N_TYPES}-class probability vector per spatial row")
    target = composition(classes)
    return float(np.abs(probabilities - target).mean(dtype=np.float64))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--method", default="candidate")
    parser.add_argument("--out", required=True, type=Path, help="directory for summary.csv")
    args = parser.parse_args(argv)
    value = score(args.data, args.prediction)
    args.out.mkdir(parents=True, exist_ok=True)
    temporary = args.out / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", METRIC])
        writer.writerow([args.method, value])
    temporary.replace(args.out / "summary.csv")
    print(f"{args.method}: {METRIC} = {value:.8f}")


if __name__ == "__main__":
    main()
