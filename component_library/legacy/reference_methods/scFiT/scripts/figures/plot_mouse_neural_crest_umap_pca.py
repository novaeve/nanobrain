import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import yaml
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

TASK_CONFIG = {
    "interpolation": {
        "protocol_path": ROOT / "configs" / "protocols" / "mouse" / "interpolation.yaml",
        "scflow_dir": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_interpolation_v1" / "eval_artifacts" / "final",
        "baseline_dir": ROOT / "results" / "eval_results" / "mouse" / "scnode_interpolation",
        "task_label": "Interpolation",
    },
    "near": {
        "protocol_path": ROOT / "configs" / "protocols" / "mouse" / "near_extrapolation.yaml",
        "scflow_dir": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_near_v1" / "eval_artifacts" / "final",
        "baseline_dir": ROOT / "results" / "eval_results" / "mouse" / "scnode_near",
        "task_label": "Near extrapolation",
    },
    "far": {
        "protocol_path": ROOT / "configs" / "protocols" / "mouse" / "far_extrapolation.yaml",
        "scflow_dir": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_far_v1" / "eval_artifacts" / "final",
        "baseline_dir": ROOT / "results" / "eval_results" / "mouse" / "scnode_far",
        "task_label": "Far extrapolation",
    },
}

DATASET_BUNDLE = ROOT / "data" / "processed" / "neural_crest_pns_glia_hvg4096"
GREY = "#c7c7c7"
RANDOM_STATE = 42


def sanitize_time_bin(time_bin: str) -> str:
    return str(time_bin).replace(".", "_")


def parse_time_value(value: str) -> float:
    s = str(value).strip()
    if s in {"P0", "P0000"}:
        return 19.0
    if s.startswith("E"):
        s = s[1:]
        if s.isdigit():
            if len(s) == 3:
                return float(f"{s[0]}.{s[1:]}")
            if len(s) >= 4:
                return float(f"{s[:2]}.{s[2:]}")
        return float(s)
    return float(s)


def make_time_palette(time_bins):
    ordered = sorted(time_bins, key=parse_time_value)
    cmap = plt.get_cmap("turbo")
    denom = max(len(ordered) - 1, 1)
    return {tp: cmap(idx / denom) for idx, tp in enumerate(ordered)}


def load_protocol(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_bundle(bundle_dir: Path):
    metadata_df = pd.read_pickle(bundle_dir / "cell_metadata.pkl")
    expr = np.load(bundle_dir / "expression_log1p_clipped.npy", mmap_mode="r")
    metadata_df = metadata_df.copy()
    metadata_df["author_day"] = metadata_df["author_day"].astype(str)
    return expr, metadata_df


def print_status(message: str):
    print(f"[plot_mouse_neural_crest_umap_pca] {message}", flush=True)


def load_artifact_dir(artifact_dir: Path):
    with (artifact_dir / "artifact_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    arrays = np.load(artifact_dir / "artifact_arrays.npz", allow_pickle=True)
    return summary, arrays


def collect_true_cells(expr, metadata_df, ordered_time_bins, max_cells=None):
    frames = []
    for tp in ordered_time_bins:
        mask = metadata_df["author_day"].to_numpy() == tp
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        if max_cells is not None and idx.size > max_cells:
            rng = np.random.default_rng(RANDOM_STATE + int(parse_time_value(tp) * 100))
            idx = np.sort(rng.choice(idx, size=max_cells, replace=False))
        frames.append(
            pd.DataFrame(
                {
                    "time_bin": tp,
                    "kind": "all_true",
                    "cell_index": idx,
                }
            )
        )
    if not frames:
        raise ValueError("No true cells found for ordered_time_bins")
    info = pd.concat(frames, ignore_index=True)
    matrix = np.asarray(expr[info["cell_index"].to_numpy()], dtype=np.float32)
    return matrix, info


def collect_after_removal(expr, metadata_df, observed_time_bins, heldout_time_bins, max_cells_per_timepoint=None):
    frames = []
    heldout_set = set(heldout_time_bins)
    for tp in observed_time_bins:
        mask = metadata_df["author_day"].to_numpy() == tp
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        if max_cells_per_timepoint is not None and idx.size > max_cells_per_timepoint:
            rng = np.random.default_rng(RANDOM_STATE + int(parse_time_value(tp) * 100))
            idx = np.sort(rng.choice(idx, size=max_cells_per_timepoint, replace=False))
        frames.append(
            pd.DataFrame(
                {
                    "time_bin": tp,
                    "kind": "heldout" if tp in heldout_set else "train",
                    "cell_index": idx,
                }
            )
        )
    info = pd.concat(frames, ignore_index=True)
    matrix = np.asarray(expr[info["cell_index"].to_numpy()], dtype=np.float32)
    return matrix, info


def build_predicted_panel(artifact_arrays, heldout_time_bins, method_label, max_cells_per_timepoint=None):
    matrices = []
    rows = []
    for tp in heldout_time_bins:
        key = f"pred__{sanitize_time_bin(tp)}"
        if key not in artifact_arrays.files:
            raise KeyError(f"Missing prediction array: {key}")
        arr = np.asarray(artifact_arrays[key], dtype=np.float32)
        if max_cells_per_timepoint is not None and arr.shape[0] > max_cells_per_timepoint:
            rng = np.random.default_rng(RANDOM_STATE + int(parse_time_value(tp) * 100))
            take = np.sort(rng.choice(arr.shape[0], size=max_cells_per_timepoint, replace=False))
            arr = arr[take]
        matrices.append(arr)
        rows.append(pd.DataFrame({"time_bin": tp, "kind": method_label}, index=np.arange(arr.shape[0])))
    info = pd.concat(rows, ignore_index=True)
    matrix = np.concatenate(matrices, axis=0)
    return matrix, info


def fit_embedding(all_true_matrix, embedding_type: str, max_cells_for_fit: int):
    n_cells = all_true_matrix.shape[0]
    if n_cells > max_cells_for_fit:
        rng = np.random.default_rng(RANDOM_STATE)
        fit_idx = np.sort(rng.choice(n_cells, size=max_cells_for_fit, replace=False))
        fit_matrix = all_true_matrix[fit_idx]
    else:
        fit_matrix = all_true_matrix

    if embedding_type == "pca":
        embedder = PCA(n_components=2, random_state=RANDOM_STATE)
    elif embedding_type == "umap":
        pca50 = PCA(n_components=50, random_state=RANDOM_STATE)
        fit_pca = pca50.fit_transform(fit_matrix)
        embedder = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.3,
            metric="euclidean",
            random_state=RANDOM_STATE,
            transform_seed=RANDOM_STATE,
            low_memory=True,
            verbose=True,
        )
        embedder.fit(fit_pca)
        return {"type": "umap", "pca50": pca50, "umap": embedder}
    else:
        raise ValueError(f"Unsupported embedding_type={embedding_type}")

    embedder.fit(fit_matrix)
    return {"type": "pca", "embedder": embedder}


def transform_embedding(embedder_bundle, matrix: np.ndarray):
    if embedder_bundle["type"] == "pca":
        return embedder_bundle["embedder"].transform(matrix)
    if embedder_bundle["type"] == "umap":
        pca50 = embedder_bundle["pca50"]
        umap_embedder = embedder_bundle["umap"]
        transformed = pca50.transform(matrix)
        return umap_embedder.transform(transformed)
    raise ValueError(f"Unknown embedder type: {embedder_bundle['type']}")


def scatter_panel(ax, coords, info, palette, emphasize_train_as_grey=False, point_size=3.0, alpha=0.75):
    if emphasize_train_as_grey:
        train_mask = info["kind"].to_numpy() == "train"
        if np.any(train_mask):
            ax.scatter(
                coords[train_mask, 0],
                coords[train_mask, 1],
                s=point_size,
                c=GREY,
                alpha=0.5,
                linewidths=0,
                rasterized=True,
            )
        for tp in sorted(info.loc[~train_mask, "time_bin"].unique(), key=parse_time_value):
            mask = (~train_mask) & (info["time_bin"].to_numpy() == tp)
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=point_size,
                    c=[palette[tp]],
                    alpha=alpha,
                    linewidths=0,
                    rasterized=True,
                )
    else:
        for tp in sorted(info["time_bin"].unique(), key=parse_time_value):
            mask = info["time_bin"].to_numpy() == tp
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                c=[palette[tp]],
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_legends(fig, palette, heldout_time_bins, train_label=True):
    handles = []
    if train_label:
        handles.append(Line2D([0], [0], marker="o", color="w", label="training points", markerfacecolor=GREY, markersize=6))
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="w", label=tp, markerfacecolor=palette[tp], markersize=6)
            for tp in heldout_time_bins
        ]
    )
    fig.legend(
    handles=handles,
    loc="center left",
    bbox_to_anchor=(0.92, 0.5),
    frameon=False,
    title="Timepoint",
    fontsize=14,
    title_fontsize=13,
    markerscale=2,
    )


def render_task(task_name: str, output_dir: Path, embedding_type: str, max_fit_cells: int, max_plot_cells_per_timepoint: int):
    cfg = TASK_CONFIG[task_name]
    protocol = load_protocol(cfg["protocol_path"])
    ordered_time_bins = [str(x) for x in protocol["eligible_time_bins"]]
    observed_time_bins = [str(x) for x in protocol["observed_time_bins"]]
    heldout_time_bins = [str(x) for x in protocol["heldout_time_bins"]]

    print_status(f"Loading dataset bundle for task={task_name}")
    expr, metadata_df = load_dataset_bundle(DATASET_BUNDLE)
    scflow_summary, scflow_arrays = load_artifact_dir(cfg["scflow_dir"])
    baseline_summary, baseline_arrays = load_artifact_dir(cfg["baseline_dir"])

    palette = make_time_palette(ordered_time_bins)
    print_status(f"Preparing panel matrices for task={task_name}, embedding={embedding_type}")
    all_true_matrix, all_true_info = collect_true_cells(
        expr, metadata_df, ordered_time_bins, max_cells=max_plot_cells_per_timepoint
    )
    removal_matrix, removal_info = collect_after_removal(
        expr,
        metadata_df,
        observed_time_bins,
        heldout_time_bins,
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )
    scflow_matrix, scflow_info = build_predicted_panel(
        scflow_arrays,
        heldout_time_bins,
        "scFiT",
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )
    baseline_matrix, baseline_info = build_predicted_panel(
        baseline_arrays,
        heldout_time_bins,
        baseline_summary.get("method_name", "Baseline"),
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )

    print_status(
        "Counts "
        f"all_true={all_true_matrix.shape[0]}, after_removal={removal_matrix.shape[0]}, "
        f"scFiT={scflow_matrix.shape[0]}, baseline={baseline_matrix.shape[0]}"
    )
    print_status(f"Fitting {embedding_type.upper()} embedding for task={task_name}")
    embedder = fit_embedding(all_true_matrix, embedding_type=embedding_type, max_cells_for_fit=max_fit_cells)
    print_status(f"Transforming panels for task={task_name}, embedding={embedding_type}")
    all_true_coords = transform_embedding(embedder, all_true_matrix)
    removal_coords = transform_embedding(embedder, removal_matrix)
    scflow_coords = transform_embedding(embedder, scflow_matrix)
    baseline_coords = transform_embedding(embedder, baseline_matrix)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=False)
    scatter_panel(axes[0], all_true_coords, all_true_info, palette, emphasize_train_as_grey=False)
    scatter_panel(axes[1], removal_coords, removal_info, palette, emphasize_train_as_grey=True)
    scatter_panel(
        axes[2],
        scflow_coords,
        scflow_info,
        palette,
        emphasize_train_as_grey=False,
    )
    scatter_panel(
        axes[3],
        baseline_coords,
        baseline_info,
        palette,
        emphasize_train_as_grey=False,
    )

    add_legends(fig, palette, heldout_time_bins, train_label=True)
    fig.subplots_adjust(right=0.9, wspace=0.08, top=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mouse_neural_crest_{task_name}_{embedding_type}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "task": task_name,
        "embedding": embedding_type,
        "protocol_name": protocol["protocol_name"],
        "scflow_method": scflow_summary.get("method_name", "scFiT"),
        "baseline_method": baseline_summary.get("method_name", "baseline"),
        "heldout_time_bins": heldout_time_bins,
        "observed_time_bins": observed_time_bins,
        "output_path": str(out_path),
    }
    with (output_dir / f"mouse_neural_crest_{task_name}_{embedding_type}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render standalone Route A UMAP/PCA figures for mouse_neural_crest")
    parser.add_argument("--tasks", nargs="+", default=["interpolation", "near", "far"], choices=sorted(TASK_CONFIG.keys()))
    parser.add_argument("--embedding", choices=["umap", "pca", "both"], default="both")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "figures" / "umap_pca_panels"))
    parser.add_argument("--max-fit-cells", type=int, default=15000)
    parser.add_argument("--max-plot-cells-per-timepoint", type=int, default=800)
    args = parser.parse_args()

    embeddings = [args.embedding] if args.embedding != "both" else ["pca", "umap"]
    jobs = [(task_name, embedding_type) for task_name in args.tasks for embedding_type in embeddings]
    output_dir = Path(args.output_dir)

    print_status(f"Starting {len(jobs)} plotting jobs")
    for idx, (task_name, embedding_type) in enumerate(tqdm(jobs, desc="Plotting", unit="figure"), start=1):
        print_status(f"[{idx}/{len(jobs)}] task={task_name}, embedding={embedding_type}")
        out_path = render_task(
            task_name,
            output_dir,
            embedding_type,
            args.max_fit_cells,
            args.max_plot_cells_per_timepoint,
        )
        print_status(f"Completed [{idx}/{len(jobs)}]: {out_path}")


if __name__ == "__main__":
    main()
