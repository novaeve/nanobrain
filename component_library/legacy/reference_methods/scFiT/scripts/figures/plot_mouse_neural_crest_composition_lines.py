import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "results" / "cache" / "figures" / "mouse_neural_crest_near_composition_cache.json"
BOOTSTRAP_CACHE_PATH = ROOT / "results" / "cache" / "figures" / "mouse_neural_crest_near_composition_bootstrap.json"
OUTPUT_DIR = ROOT / "results" / "figures" / "composition"
DATASET_DIR = ROOT / "data" / "processed" / "neural_crest_pns_glia_hvg4096"
SCFIT_ARTIFACT_PATH = (
    ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_near_v1" / "eval_artifacts" / "final" / "artifact_arrays.npz"
)
SCNODE_ARTIFACT_PATH = ROOT / "results" / "eval_results" / "mouse" / "scnode_near" / "artifact_arrays.npz"

TIME_BINS = ["E1625", "E1650", "E1675", "E1700"]
FOCAL_CELL_TYPES = [
    "Sympathetic neurons",
    "Dorsal root ganglion neurons",
    "Parasympathetic neurons",
]
CELL_TYPE_TITLES = {
    "Sympathetic neurons": "Sympathetic",
    "Dorsal root ganglion neurons": "DRG",
    "Parasympathetic neurons": "Parasympathetic",
}
METHOD_STYLES = {
    "Real": {"color": "#D5D1D1", "marker": "o", "linestyle": "-", "linewidth": 2.3, "markersize": 5.8},
    "scFiT": {"color": "#E9687A", "marker": "o", "linestyle": "-", "linewidth": 2.1, "markersize": 5.2},
    "scNODE": {"color": "#B6B3D6", "marker": "s", "linestyle": "--", "linewidth": 2.0, "markersize": 5.0},
}
METHOD_LABELS = {
    "Real": "Real",
    "scFiT": "scFiT",
    "scNODE": "scNODE",
}
METHOD_FILL_ALPHA = {"Real": 0.10, "scFiT": 0.16, "scNODE": 0.14}
TEXT_COLOR = "#333333"
GRID_COLOR = "#DADADA"
PANEL_BG = "#FBFBFA"
BOOTSTRAP_ROUNDS = 2000
CONFIDENCE_INTERVAL = (2.5, 97.5)
RANDOM_SEED = 42


def _configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.6,
        }
    )


def _style_axis(ax):
    ax.set_facecolor(PANEL_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.grid(True, axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)


def _load_summary():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"composition cache not found: {CACHE_PATH}. "
            "Run plot_mouse_neural_crest_composition.py first or pass --cache-path."
        )
    return json.loads(CACHE_PATH.read_text())


def _build_sample_sizes():
    if not (DATASET_DIR / "cell_metadata.csv.gz").exists():
        raise FileNotFoundError(f"cell metadata not found: {DATASET_DIR / 'cell_metadata.csv.gz'}")
    meta = pd.read_csv(DATASET_DIR / "cell_metadata.csv.gz", usecols=["author_day"])
    meta["author_day"] = meta["author_day"].astype(str)
    sample_sizes = {"Real": {}}
    for time_bin in TIME_BINS:
        sample_sizes["Real"][time_bin] = int((meta["author_day"] == time_bin).sum())

    for method_name, artifact_path in [("scFiT", SCFIT_ARTIFACT_PATH), ("scNODE", SCNODE_ARTIFACT_PATH)]:
        if not artifact_path.exists():
            raise FileNotFoundError(f"{method_name} artifact not found: {artifact_path}")
        arrays = np.load(artifact_path, allow_pickle=True)
        sample_sizes[method_name] = {}
        for time_bin in TIME_BINS:
            sample_sizes[method_name][time_bin] = int(arrays[f"pred__{time_bin}"].shape[0])
    return sample_sizes


def _build_bootstrap_cache(summary: dict):
    rng = np.random.default_rng(RANDOM_SEED)
    sample_sizes = _build_sample_sizes()
    payload = {
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "confidence_interval": list(CONFIDENCE_INTERVAL),
        "sample_sizes": sample_sizes,
        "intervals": {},
    }

    for method_name, time_map in summary["compositions"].items():
        payload["intervals"][method_name] = {}
        for time_bin, cell_type_map in time_map.items():
            probs = np.asarray([cell_type_map[cell_type] for cell_type in summary["cell_type_order"]], dtype=np.float64)
            n = sample_sizes[method_name][time_bin]
            sampled = rng.multinomial(n=n, pvals=probs, size=BOOTSTRAP_ROUNDS) / float(n)
            payload["intervals"][method_name][time_bin] = {}
            for idx, cell_type in enumerate(summary["cell_type_order"]):
                low, high = np.percentile(sampled[:, idx], CONFIDENCE_INTERVAL)
                payload["intervals"][method_name][time_bin][cell_type] = {
                    "low": float(low),
                    "high": float(high),
                }
    return payload


def _load_or_create_bootstrap_cache(summary: dict):
    if BOOTSTRAP_CACHE_PATH.exists():
        return json.loads(BOOTSTRAP_CACHE_PATH.read_text())
    payload = _build_bootstrap_cache(summary)
    BOOTSTRAP_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _plot_panel(ax, summary: dict, bootstrap_summary: dict, cell_type: str):
    _style_axis(ax)
    x = np.arange(len(TIME_BINS))
    for method_name in ["Real", "scFiT", "scNODE"]:
        y = [
            summary["compositions"][method_name][time_bin][cell_type] * 100.0
            for time_bin in TIME_BINS
        ]
        low = [
            bootstrap_summary["intervals"][method_name][time_bin][cell_type]["low"] * 100.0
            for time_bin in TIME_BINS
        ]
        high = [
            bootstrap_summary["intervals"][method_name][time_bin][cell_type]["high"] * 100.0
            for time_bin in TIME_BINS
        ]
        ax.fill_between(
            x,
            low,
            high,
            color=METHOD_STYLES[method_name]["color"],
            alpha=METHOD_FILL_ALPHA[method_name],
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(x, y, label=method_name, **METHOD_STYLES[method_name])
        ax.lines[-1].set_label(METHOD_LABELS[method_name])

    ax.set_title(CELL_TYPE_TITLES[cell_type], pad=6, color=TEXT_COLOR, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_BINS)
    ax.set_ylim(10, 60)
    ax.set_yticks(np.arange(10, 61, 10))


def _render(summary: dict, bootstrap_summary: dict):
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), sharex=True, sharey=False, constrained_layout=False)
    for ax, cell_type in zip(axes, FOCAL_CELL_TYPES):
        _plot_panel(ax, summary, bootstrap_summary, cell_type)

    axes[0].set_ylabel("Cell proportion (%)", color=TEXT_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.28, wspace=0.18)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "mouse_neural_crest_near_composition_lines.png"
    pdf_path = OUTPUT_DIR / "mouse_neural_crest_near_composition_lines.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Plot focal cell-type composition trajectories with bootstrap bands.")
    parser.add_argument("--cache-path", default=str(CACHE_PATH), help="Composition cache JSON from plot_mouse_neural_crest_composition.py.")
    parser.add_argument("--bootstrap-cache-path", default=str(BOOTSTRAP_CACHE_PATH), help="Bootstrap cache JSON path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for output figures.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Processed mouse data bundle directory.")
    parser.add_argument("--scfit-artifact", default=str(SCFIT_ARTIFACT_PATH), help="scFiT artifact_arrays.npz.")
    parser.add_argument("--scnode-artifact", default=str(SCNODE_ARTIFACT_PATH), help="scNODE artifact_arrays.npz.")
    args = parser.parse_args()

    globals().update(
        {
            "CACHE_PATH": Path(args.cache_path),
            "BOOTSTRAP_CACHE_PATH": Path(args.bootstrap_cache_path),
            "OUTPUT_DIR": Path(args.output_dir),
            "DATASET_DIR": Path(args.dataset_dir),
            "SCFIT_ARTIFACT_PATH": Path(args.scfit_artifact),
            "SCNODE_ARTIFACT_PATH": Path(args.scnode_artifact),
        }
    )

    _configure_style()
    summary = _load_summary()
    bootstrap_summary = _load_or_create_bootstrap_cache(summary)
    png_path, pdf_path = _render(summary, bootstrap_summary)
    payload = {
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "time_bins": TIME_BINS,
        "cell_types": FOCAL_CELL_TYPES,
        "bootstrap_cache": str(BOOTSTRAP_CACHE_PATH),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
