import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "processed" / "neural_crest_pns_glia_hvg4096"
ASSET_DIR = ROOT / "results" / "eval_assets" / "mouse" / "near_extrapolation"
CACHE_DIR = ROOT / "results" / "cache" / "figures"
OUTPUT_DIR = ROOT / "results" / "figures" / "composition"

TIME_BINS = ["E1625", "E1650", "E1675", "E1700"]
METHODS = {
    "Real": None,
    "scFiT": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_near_v1" / "eval_artifacts" / "final" / "artifact_arrays.npz",
    "scNODE": ROOT / "results" / "eval_results" / "mouse" / "scnode_near" / "artifact_arrays.npz",
}

CELL_TYPE_ORDER = [
    "Sympathetic neurons",
    "Dorsal root ganglion neurons",
    "Parasympathetic neurons",
    "Neural crest (PNS neurons)",
    "Enteric neurons",
    "Otic sensory neurons",
]
CELL_TYPE_COLORS = {
    "Sympathetic neurons": "#D98C7C",
    "Dorsal root ganglion neurons": "#7FA7C9",
    "Parasympathetic neurons": "#9C8AB5",
    "Neural crest (PNS neurons)": "#A7B88A",
    "Enteric neurons": "#D4B06A",
    "Otic sensory neurons": "#78B7B2",
}
TEXT_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"
PANEL_BG = "#FBFBFA"


def _configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
        }
    )


def _cache_path() -> Path:
    return CACHE_DIR / "mouse_neural_crest_near_composition_cache.json"


def _require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_real_composition(meta: pd.DataFrame) -> dict:
    payload = {}
    for time_bin in TIME_BINS:
        subset = meta.loc[meta["author_day"] == time_bin, "author_cell_type"]
        freq = subset.value_counts(normalize=True).reindex(CELL_TYPE_ORDER, fill_value=0.0)
        payload[time_bin] = {cell_type: float(freq[cell_type]) for cell_type in CELL_TYPE_ORDER}
    return payload


def _predict_composition(npz_path: Path, pca_model, knn_model) -> dict:
    arrays = np.load(npz_path, allow_pickle=True)
    payload = {}
    for time_bin in TIME_BINS:
        pred = np.asarray(arrays[f"pred__{time_bin}"], dtype=np.float32)
        pred_pca = pca_model.transform(pred)
        labels = knn_model.predict(pred_pca)
        freq = pd.Series(labels).value_counts(normalize=True).reindex(CELL_TYPE_ORDER, fill_value=0.0)
        payload[time_bin] = {cell_type: float(freq[cell_type]) for cell_type in CELL_TYPE_ORDER}
    return payload


def _compute_total_variation(real_comp: dict, pred_comp: dict) -> float:
    tvds = []
    for time_bin in TIME_BINS:
        real_vec = np.asarray([real_comp[time_bin][ct] for ct in CELL_TYPE_ORDER], dtype=np.float64)
        pred_vec = np.asarray([pred_comp[time_bin][ct] for ct in CELL_TYPE_ORDER], dtype=np.float64)
        tvds.append(0.5 * np.abs(real_vec - pred_vec).sum())
    return float(np.mean(tvds))


def _build_cache() -> dict:
    _require_file(DATASET_DIR / "cell_metadata.csv.gz", "cell metadata")
    _require_file(ASSET_DIR / "pca_model.pkl", "PCA model")
    _require_file(ASSET_DIR / "cell_type_knn.pkl", "cell type kNN model")
    for method_name, npz_path in METHODS.items():
        if npz_path is not None:
            _require_file(npz_path, f"{method_name} artifact")

    meta = pd.read_csv(DATASET_DIR / "cell_metadata.csv.gz")
    meta["author_day"] = meta["author_day"].astype(str)
    pca_model = joblib.load(ASSET_DIR / "pca_model.pkl")
    knn_model = joblib.load(ASSET_DIR / "cell_type_knn.pkl")

    compositions = {"Real": _load_real_composition(meta)}
    for method_name, npz_path in METHODS.items():
        if npz_path is None:
            continue
        compositions[method_name] = _predict_composition(npz_path, pca_model, knn_model)

    summary = {
        "time_bins": TIME_BINS,
        "cell_type_order": CELL_TYPE_ORDER,
        "compositions": compositions,
        "mean_tvd": {
            "scFiT": _compute_total_variation(compositions["Real"], compositions["scFiT"]),
            "scNODE": _compute_total_variation(compositions["Real"], compositions["scNODE"]),
        },
    }
    return summary


def _load_or_create_cache() -> dict:
    cache_path = _cache_path()
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    summary = _build_cache()
    cache_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _style_axis(ax):
    ax.set_facecolor(PANEL_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.grid(True, axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _plot_method_panel(ax, method_name: str, summary: dict):
    _style_axis(ax)
    comp = summary["compositions"][method_name]
    x = np.arange(len(TIME_BINS))
    bottom = np.zeros(len(TIME_BINS), dtype=np.float64)
    for cell_type in CELL_TYPE_ORDER:
        values = np.asarray([comp[tp][cell_type] * 100.0 for tp in TIME_BINS], dtype=np.float64)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.72,
            color=CELL_TYPE_COLORS[cell_type],
            edgecolor="white",
            linewidth=0.7,
            label=cell_type,
        )
        bottom += values

    title = method_name
    if method_name in summary["mean_tvd"]:
        title = f"{method_name}\nMean TVD={summary['mean_tvd'][method_name]:.3f}"
    ax.set_title(title, pad=8, color=TEXT_COLOR, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_BINS, rotation=0)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{v:d}" for v in range(0, 101, 20)], color=TEXT_COLOR)
    ax.tick_params(axis="x", colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)


def _render(summary: dict):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True, constrained_layout=False)
    for ax, method_name in zip(axes, METHODS):
        _plot_method_panel(ax, method_name, summary)

    axes[0].set_ylabel("Cell proportion (%)", color=TEXT_COLOR)
    for ax in axes[1:]:
        ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.28, wspace=0.12)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "mouse_neural_crest_near_composition.png"
    pdf_path = OUTPUT_DIR / "mouse_neural_crest_near_composition.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Plot near-extrapolation cell-type composition bars.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Processed mouse data bundle directory.")
    parser.add_argument("--assets-dir", default=str(ASSET_DIR), help="Evaluation assets directory with PCA and kNN files.")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="Directory for computed composition cache.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for output figures.")
    parser.add_argument("--scfit-artifact", default=str(METHODS["scFiT"]), help="scFiT artifact_arrays.npz.")
    parser.add_argument("--scnode-artifact", default=str(METHODS["scNODE"]), help="scNODE artifact_arrays.npz.")
    args = parser.parse_args()

    globals().update(
        {
            "DATASET_DIR": Path(args.dataset_dir),
            "ASSET_DIR": Path(args.assets_dir),
            "CACHE_DIR": Path(args.cache_dir),
            "OUTPUT_DIR": Path(args.output_dir),
            "METHODS": {
                "Real": None,
                "scFiT": Path(args.scfit_artifact),
                "scNODE": Path(args.scnode_artifact),
            },
        }
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_style()
    summary = _load_or_create_cache()
    png_path, pdf_path = _render(summary)
    summary.update({"output_png": str(png_path), "output_pdf": str(pdf_path)})
    (OUTPUT_DIR / "mouse_neural_crest_near_composition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
