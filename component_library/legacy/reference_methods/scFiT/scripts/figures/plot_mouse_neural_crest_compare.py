import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    umap = None


ROOT = Path(__file__).resolve().parents[2]
DATASET_BUNDLE = ROOT / "data" / "processed" / "neural_crest_pns_glia_hvg4096"
CACHE_DIR = ROOT / "results" / "cache" / "figures"
RANDOM_STATE = 42

OBSERVED_GREY = "#d8d8d8"
SOURCE_GREY = "#777777"
TEXT_GREY = "#404040"
PANEL_BG = "#fcfcfc"
FIGURE_PALETTE = [
    "#B6B3D6",
    "#CFCCE3",
    "#D5D3DE",
    "#D5D1D1",
    "#F6DFD6",
    "#F8B2A2",
    "#F1837A",
    "#E9687A",
]

TASK_CONFIG = {
    "interpolation": {
        "protocol_path": ROOT / "configs" / "protocols" / "mouse" / "interpolation.yaml",
        "scfit_dir": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_interpolation_v1" / "eval_artifacts" / "final",
        "baseline_dir": ROOT / "results" / "eval_results" / "mouse" / "scnode_interpolation",
        "task_label": "Interpolation",
        "display_label": "Interpolation",
        "display_time_bins": ["E1125", "E1325", "E1525", "E1675", "E1825"],
        "mean_wd": {"scFiT": 3.673, "scNODE": 5.498},
    },
    "near": {
        "protocol_path": ROOT / "configs" / "protocols" / "mouse" / "near_extrapolation.yaml",
        "scfit_dir": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_near_v1" / "eval_artifacts" / "final",
        "baseline_dir": ROOT / "results" / "eval_results" / "mouse" / "scnode_near",
        "task_label": "Near Extrapolation",
        "display_label": "Near extrap.",
        "display_time_bins": ["E1625", "E1650", "E1675", "E1700"],
        "mean_wd": {"scFiT": 2.991, "scNODE": 5.536},
    },
    "far": {
        "protocol_path": ROOT / "configs" / "protocols" / "mouse" / "far_extrapolation.yaml",
        "scfit_dir": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_far_v1" / "eval_artifacts" / "final",
        "baseline_dir": ROOT / "results" / "eval_results" / "mouse" / "scnode_far",
        "task_label": "Far Extrapolation",
        "display_label": "Far extrap.",
        "display_time_bins": ["E1500", "E1600", "E1700", "E1800", "P0000"],
        "mean_wd": {"scFiT": 7.482, "scNODE": 9.283},
    },
}


def print_status(message: str):
    print(f"[plot_mouse_neural_crest_compare] {message}", flush=True)


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


def sorted_time_bins(time_bins):
    return sorted([str(x) for x in time_bins], key=parse_time_value)


def sample_indices(indices: np.ndarray, time_bin: str, max_cells: int = None) -> np.ndarray:
    if max_cells is None or indices.size <= max_cells:
        return indices
    seed = RANDOM_STATE + int(round(parse_time_value(time_bin) * 100))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_cells, replace=False))


def make_time_palette(time_bins):
    ordered = sorted_time_bins(time_bins)
    if len(ordered) == 1:
        return {ordered[0]: FIGURE_PALETTE[-1]}
    color_idx = np.linspace(0, len(FIGURE_PALETTE) - 1, len(ordered)).round().astype(int)
    return {tp: FIGURE_PALETTE[idx] for tp, idx in zip(ordered, color_idx)}


def load_protocol(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_bundle(bundle_dir: Path):
    metadata_df = pd.read_pickle(bundle_dir / "cell_metadata.pkl").copy()
    expr = np.load(bundle_dir / "expression_log1p_clipped.npy", mmap_mode="r")
    metadata_df["author_day"] = metadata_df["author_day"].astype(str)
    return expr, metadata_df


def load_artifact_dir(artifact_dir: Path):
    with (artifact_dir / "artifact_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    arrays = np.load(artifact_dir / "artifact_arrays.npz", allow_pickle=True)
    return summary, arrays


def cache_key(task_name: str, embedding_type: str, max_fit_cells: int, max_plot_cells_per_timepoint: int) -> str:
    return f"{task_name}__{embedding_type}__fit{max_fit_cells}__plot{max_plot_cells_per_timepoint}"


def cache_paths(task_name: str, embedding_type: str, max_fit_cells: int, max_plot_cells_per_timepoint: int):
    key = cache_key(task_name, embedding_type, max_fit_cells, max_plot_cells_per_timepoint)
    return (
        CACHE_DIR / f"{key}.npz",
        CACHE_DIR / f"{key}.json",
    )


def collect_fit_matrix(expr, metadata_df, ordered_time_bins, max_cells_per_timepoint):
    frames = []
    for tp in ordered_time_bins:
        idx = np.flatnonzero(metadata_df["author_day"].to_numpy() == tp)
        if idx.size == 0:
            continue
        idx = sample_indices(idx, tp, max_cells=max_cells_per_timepoint)
        frames.append(np.asarray(expr[idx], dtype=np.float32))
    if not frames:
        raise ValueError("No cells found while building fit matrix")
    return np.concatenate(frames, axis=0)


def collect_context_cells(expr, metadata_df, observed_time_bins, source_time_bins, max_cells_per_timepoint):
    rows = []
    matrices = []
    source_set = set(str(x) for x in source_time_bins)
    for tp in sorted_time_bins(observed_time_bins):
        idx = np.flatnonzero(metadata_df["author_day"].to_numpy() == tp)
        if idx.size == 0:
            continue
        idx = sample_indices(idx, tp, max_cells=max_cells_per_timepoint)
        matrices.append(np.asarray(expr[idx], dtype=np.float32))
        rows.append(
            pd.DataFrame(
                {
                    "time_bin": tp,
                    "kind": "source" if tp in source_set else "observed",
                },
                index=np.arange(idx.size),
            )
        )
    info = pd.concat(rows, ignore_index=True)
    matrix = np.concatenate(matrices, axis=0)
    return matrix, info


def collect_true_targets(expr, metadata_df, heldout_time_bins, max_cells_per_timepoint):
    rows = []
    matrices = []
    for tp in sorted_time_bins(heldout_time_bins):
        idx = np.flatnonzero(metadata_df["author_day"].to_numpy() == tp)
        if idx.size == 0:
            continue
        idx = sample_indices(idx, tp, max_cells=max_cells_per_timepoint)
        matrices.append(np.asarray(expr[idx], dtype=np.float32))
        rows.append(pd.DataFrame({"time_bin": tp, "kind": "target"}, index=np.arange(idx.size)))
    info = pd.concat(rows, ignore_index=True)
    matrix = np.concatenate(matrices, axis=0)
    return matrix, info


def build_predicted_panel(artifact_arrays, heldout_time_bins, method_label, max_cells_per_timepoint):
    rows = []
    matrices = []
    for tp in sorted_time_bins(heldout_time_bins):
        key = f"pred__{sanitize_time_bin(tp)}"
        if key not in artifact_arrays.files:
            raise KeyError(f"Missing prediction array: {key}")
        arr = np.asarray(artifact_arrays[key], dtype=np.float32)
        if max_cells_per_timepoint is not None and arr.shape[0] > max_cells_per_timepoint:
            seed = RANDOM_STATE + int(round(parse_time_value(tp) * 100))
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(arr.shape[0], size=max_cells_per_timepoint, replace=False))
            arr = arr[keep]
        matrices.append(arr)
        rows.append(pd.DataFrame({"time_bin": tp, "kind": method_label}, index=np.arange(arr.shape[0])))
    info = pd.concat(rows, ignore_index=True)
    matrix = np.concatenate(matrices, axis=0)
    return matrix, info


def fit_embedding(matrix: np.ndarray, embedding_type: str, max_fit_cells: int):
    n_cells = matrix.shape[0]
    if n_cells > max_fit_cells:
        rng = np.random.default_rng(RANDOM_STATE)
        fit_idx = np.sort(rng.choice(n_cells, size=max_fit_cells, replace=False))
        fit_matrix = matrix[fit_idx]
    else:
        fit_matrix = matrix

    if embedding_type == "pca":
        embedder = PCA(n_components=2, random_state=RANDOM_STATE)
        embedder.fit(fit_matrix)
        return {"type": "pca", "embedder": embedder}

    if embedding_type == "umap":
        if umap is None:
            raise ImportError("umap-learn is required when --embedding umap is requested")
        pca50 = PCA(n_components=50, random_state=RANDOM_STATE)
        fit_pca = pca50.fit_transform(fit_matrix)
        umap_embedder = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.3,
            metric="euclidean",
            random_state=RANDOM_STATE,
            transform_seed=RANDOM_STATE,
            low_memory=True,
            verbose=False,
        )
        umap_embedder.fit(fit_pca)
        return {"type": "umap", "pca50": pca50, "umap": umap_embedder}

    raise ValueError(f"Unsupported embedding_type={embedding_type}")


def transform_embedding(embedder_bundle, matrix: np.ndarray):
    if embedder_bundle["type"] == "pca":
        return embedder_bundle["embedder"].transform(matrix)
    if embedder_bundle["type"] == "umap":
        transformed = embedder_bundle["pca50"].transform(matrix)
        return embedder_bundle["umap"].transform(transformed)
    raise ValueError(f"Unknown embedder type: {embedder_bundle['type']}")


def resolve_generation_sources(protocol: dict, target_time_bin: str):
    observed_time_bins = sorted_time_bins(protocol["observed_time_bins"])
    target_numeric = parse_time_value(target_time_bin)
    prev_time = None
    next_time = None
    for tp in observed_time_bins:
        numeric = parse_time_value(tp)
        if numeric < target_numeric:
            prev_time = tp
        elif numeric > target_numeric and next_time is None:
            next_time = tp
            break

    policy = str(protocol["source_policy"])
    if policy == "previous_observed":
        if prev_time is None:
            raise ValueError(f"Missing previous observed time for {target_time_bin}")
        return [prev_time]
    if policy == "bidirectional_linear":
        if prev_time is None or next_time is None:
            raise ValueError(f"Interpolation target {target_time_bin} needs both observed neighbors")
        return [prev_time, next_time]
    raise ValueError(f"Unsupported source_policy={policy}")


def prepare_task_data(task_name: str, embedding_type: str, max_fit_cells: int, max_plot_cells_per_timepoint: int):
    cfg = TASK_CONFIG[task_name]
    protocol = load_protocol(cfg["protocol_path"])
    ordered_time_bins = sorted_time_bins(protocol["eligible_time_bins"])
    observed_time_bins = sorted_time_bins(protocol["observed_time_bins"])
    heldout_time_bins = sorted_time_bins(protocol["heldout_time_bins"])
    display_time_bins = [str(x) for x in cfg["display_time_bins"]]
    if not set(display_time_bins).issubset(set(heldout_time_bins)):
        missing = sorted(set(display_time_bins) - set(heldout_time_bins), key=parse_time_value)
        raise ValueError(f"display_time_bins not held-out in task={task_name}: {missing}")

    source_time_bins = sorted_time_bins(
        {
            src
            for target_time_bin in display_time_bins
            for src in resolve_generation_sources(protocol, target_time_bin)
        }
    )
    palette = make_time_palette(display_time_bins)
    npz_path, meta_path = cache_paths(
        task_name,
        embedding_type,
        max_fit_cells=max_fit_cells,
        max_plot_cells_per_timepoint=max_plot_cells_per_timepoint,
    )

    if npz_path.exists() and meta_path.exists():
        print_status(f"Loading cached embedding for task={task_name}")
        arrays = np.load(npz_path, allow_pickle=False)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "task_name": task_name,
            "task_label": cfg["task_label"],
            "display_label": cfg.get("display_label", cfg["task_label"]),
            "display_time_bins": display_time_bins,
            "source_time_bins": source_time_bins,
            "palette": palette,
            "context_coords": arrays["context_coords"],
            "context_info": pd.DataFrame(
                {
                    "time_bin": arrays["context_time_bin"].astype(str),
                    "kind": arrays["context_kind"].astype(str),
                }
            ),
            "truth_coords": arrays["truth_coords"],
            "truth_info": pd.DataFrame({"time_bin": arrays["truth_time_bin"].astype(str), "kind": "target"}),
            "scfit_coords": arrays["scfit_coords"],
            "scfit_info": pd.DataFrame({"time_bin": arrays["scfit_time_bin"].astype(str), "kind": "scFiT"}),
            "baseline_coords": arrays["baseline_coords"],
            "baseline_info": pd.DataFrame(
                {
                    "time_bin": arrays["baseline_time_bin"].astype(str),
                    "kind": meta.get("baseline_label", "scNODE"),
                }
            ),
            "baseline_label": meta.get("baseline_label", "scNODE"),
            "scfit_label": meta.get("scfit_label", "scFiT"),
            "protocol_name": protocol["protocol_name"],
            "mean_wd": dict(cfg["mean_wd"]),
        }

    print_status(f"Loading data for task={task_name}")
    expr, metadata_df = load_dataset_bundle(DATASET_BUNDLE)
    scfit_summary, scfit_arrays = load_artifact_dir(cfg["scfit_dir"])
    baseline_summary, baseline_arrays = load_artifact_dir(cfg["baseline_dir"])

    fit_matrix = collect_fit_matrix(
        expr,
        metadata_df,
        ordered_time_bins,
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )
    context_matrix, context_info = collect_context_cells(
        expr,
        metadata_df,
        observed_time_bins,
        source_time_bins,
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )
    truth_matrix, truth_info = collect_true_targets(
        expr,
        metadata_df,
        display_time_bins,
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )
    scfit_matrix, scfit_info = build_predicted_panel(
        scfit_arrays,
        display_time_bins,
        method_label="scFiT",
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )
    baseline_matrix, baseline_info = build_predicted_panel(
        baseline_arrays,
        display_time_bins,
        method_label=baseline_summary.get("method_name", "scNODE"),
        max_cells_per_timepoint=max_plot_cells_per_timepoint,
    )

    print_status(
        f"Fitting {embedding_type.upper()} for task={task_name} "
        f"(fit={fit_matrix.shape[0]}, context={context_matrix.shape[0]}, truth={truth_matrix.shape[0]})"
    )
    embedder = fit_embedding(fit_matrix, embedding_type=embedding_type, max_fit_cells=max_fit_cells)
    context_coords = transform_embedding(embedder, context_matrix)
    truth_coords = transform_embedding(embedder, truth_matrix)
    scfit_coords = transform_embedding(embedder, scfit_matrix)
    baseline_coords = transform_embedding(embedder, baseline_matrix)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        context_coords=context_coords.astype(np.float32),
        context_time_bin=context_info["time_bin"].to_numpy(dtype=str),
        context_kind=context_info["kind"].to_numpy(dtype=str),
        truth_coords=truth_coords.astype(np.float32),
        truth_time_bin=truth_info["time_bin"].to_numpy(dtype=str),
        scfit_coords=scfit_coords.astype(np.float32),
        scfit_time_bin=scfit_info["time_bin"].to_numpy(dtype=str),
        baseline_coords=baseline_coords.astype(np.float32),
        baseline_time_bin=baseline_info["time_bin"].to_numpy(dtype=str),
    )
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task_name": task_name,
                "embedding_type": embedding_type,
                "max_fit_cells": max_fit_cells,
                "max_plot_cells_per_timepoint": max_plot_cells_per_timepoint,
                "scfit_label": scfit_summary.get("method_name", "scFiT"),
                "baseline_label": baseline_summary.get("method_name", "scNODE"),
            },
            f,
            indent=2,
        )
    print_status(f"Cached embedding for task={task_name} -> {npz_path.name}")

    return {
        "task_name": task_name,
        "task_label": cfg["task_label"],
        "display_label": cfg.get("display_label", cfg["task_label"]),
        "display_time_bins": display_time_bins,
        "source_time_bins": source_time_bins,
        "palette": palette,
        "context_coords": context_coords,
        "context_info": context_info,
        "truth_coords": truth_coords,
        "truth_info": truth_info,
        "scfit_coords": scfit_coords,
        "scfit_info": scfit_info,
        "baseline_coords": baseline_coords,
        "baseline_info": baseline_info,
        "baseline_label": baseline_summary.get("method_name", "scNODE"),
        "scfit_label": scfit_summary.get("method_name", "scFiT"),
        "protocol_name": protocol["protocol_name"],
        "mean_wd": dict(cfg["mean_wd"]),
    }


def scatter_context(ax, context_coords, context_info):
    kinds = context_info["kind"].to_numpy()
    observed_mask = kinds == "observed"
    source_mask = kinds == "source"

    if np.any(observed_mask):
        ax.scatter(
            context_coords[observed_mask, 0],
            context_coords[observed_mask, 1],
            s=2.5,
            c=OBSERVED_GREY,
            alpha=0.15,
            linewidths=0,
            rasterized=True,
        )
    if np.any(source_mask):
        ax.scatter(
            context_coords[source_mask, 0],
            context_coords[source_mask, 1],
            s=3.1,
            c=SOURCE_GREY,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )


def scatter_targets(ax, coords, info, palette, point_size=3.8, alpha=0.66):
    time_bins = sorted_time_bins(info["time_bin"].unique())
    for tp in time_bins:
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


def style_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def compute_limits(task_data):
    stacks = [
        task_data["context_coords"],
        task_data["truth_coords"],
        task_data["scfit_coords"],
        task_data["baseline_coords"],
    ]
    merged = np.concatenate(stacks, axis=0)
    x_min, y_min = merged.min(axis=0)
    x_max, y_max = merged.max(axis=0)
    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    pad_x = 0.05 * dx
    pad_y = 0.05 * dy
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


def render_compare_figure(task_names, output_path: Path, embedding_type: str, max_fit_cells: int, max_plot_cells_per_timepoint: int):
    prepared = [
        prepare_task_data(
            task_name,
            embedding_type=embedding_type,
            max_fit_cells=max_fit_cells,
            max_plot_cells_per_timepoint=max_plot_cells_per_timepoint,
        )
        for task_name in task_names
    ]

    fig = plt.figure(figsize=(4.35, 4.75), constrained_layout=False)
    gs = fig.add_gridspec(
        len(prepared) + 1,
        3,
        height_ratios=[0.11] + [1.0] * len(prepared),
        left=0.035,
        right=0.995,
        top=0.985,
        bottom=0.050,
        wspace=0.035,
        hspace=0.14,
    )

    header_axes = [fig.add_subplot(gs[0, col_idx]) for col_idx in range(3)]
    axes = np.array(
        [[fig.add_subplot(gs[row_idx + 1, col_idx]) for col_idx in range(3)] for row_idx in range(len(prepared))]
    )

    column_titles = ["Target", "scFiT", "scNODE"]
    for col_idx, title in enumerate(column_titles):
        header_axes[col_idx].axis("off")
        header_axes[col_idx].text(
            0.5,
            0.36,
            title,
            ha="center",
            va="center",
            fontsize=8.7,
            fontweight="bold",
            color="black",
        )

    for row_idx, task_data in enumerate(prepared):
        xlim, ylim = compute_limits(task_data)
        row_axes = axes[row_idx]

        scatter_context(row_axes[0], task_data["context_coords"], task_data["context_info"])
        scatter_targets(row_axes[0], task_data["truth_coords"], task_data["truth_info"], task_data["palette"], point_size=3.8, alpha=0.66)

        scatter_context(row_axes[1], task_data["context_coords"], task_data["context_info"])
        scatter_targets(row_axes[1], task_data["scfit_coords"], task_data["scfit_info"], task_data["palette"], point_size=3.5, alpha=0.62)

        scatter_context(row_axes[2], task_data["context_coords"], task_data["context_info"])
        scatter_targets(row_axes[2], task_data["baseline_coords"], task_data["baseline_info"], task_data["palette"], point_size=3.5, alpha=0.62)

        for ax in row_axes:
            style_axis(ax, xlim=xlim, ylim=ylim)

        row_axes[1].text(
            0.03,
            0.04,
            f"WD {task_data['mean_wd']['scFiT']:.3f}",
            transform=row_axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=6.3,
            color=TEXT_GREY,
            bbox={
                "facecolor": "white",
                "alpha": 0.84,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.18",
            },
        )
        row_axes[2].text(
            0.03,
            0.04,
            f"WD {task_data['mean_wd']['scNODE']:.3f}",
            transform=row_axes[2].transAxes,
            ha="left",
            va="bottom",
            fontsize=6.3,
            color=TEXT_GREY,
            bbox={
                "facecolor": "white",
                "alpha": 0.84,
                "edgecolor": "none",
                "boxstyle": "round,pad=0.18",
            },
        )

    for row_idx, task_data in enumerate(prepared):
        pos = axes[row_idx, 0].get_position()
        fig.text(
            pos.x0,
            pos.y1 + 0.002,
            task_data["display_label"],
            ha="left",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="black",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=650, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    meta = {
        "tasks": list(task_names),
        "embedding": embedding_type,
        "output_path": str(output_path),
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render the combined mouse neural crest qualitative comparison figure")
    parser.add_argument("--tasks", nargs="+", default=["interpolation", "near", "far"], choices=sorted(TASK_CONFIG.keys()))
    parser.add_argument("--embedding", choices=["pca", "umap"], default="pca")
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "results" / "figures" / "mouse_neural_crest_compare_v2.png"),
    )
    parser.add_argument("--max-fit-cells", type=int, default=15000)
    parser.add_argument("--max-plot-cells-per-timepoint", type=int, default=800)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    print_status(f"Rendering compare figure to {output_path}")
    out_path = render_compare_figure(
        task_names=args.tasks,
        output_path=output_path,
        embedding_type=args.embedding,
        max_fit_cells=args.max_fit_cells,
        max_plot_cells_per_timepoint=args.max_plot_cells_per_timepoint,
    )
    print_status(f"Saved compare figure: {out_path}")


if __name__ == "__main__":
    main()
