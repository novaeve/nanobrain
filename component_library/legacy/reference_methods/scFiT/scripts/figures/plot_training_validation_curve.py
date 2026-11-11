import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "figures" / "training_curves"

TASKS = [
    {
        "name": "Interpolation",
        "key": "interpolation",
        "log_path": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_interpolation_v1" / "train_log.jsonl",
    },
    {
        "name": "Near Extrap.",
        "key": "near",
        "log_path": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_near_v1" / "train_log.jsonl",
    },
    {
        "name": "Far Extrap.",
        "key": "far",
        "log_path": ROOT / "results" / "experiments" / "mouse" / "neural_crest_route_a_far_v1" / "train_log.jsonl",
    },
]

PCA50_COLOR = "#B6B3D6"
R2_COLOR = "#E9687A"
BEST_COLOR = "#F1837A"
GRID_COLOR = "#D8D8D8"


def _load_validation_rows(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"training log not found: {log_path}")
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    val_rows = [row for row in rows if "val_wasserstein_pca50_mean" in row]
    if not val_rows:
        raise ValueError(f"no validation rows found in {log_path}")
    return val_rows


def _prepare_task(task_cfg):
    val_rows = _load_validation_rows(task_cfg["log_path"])
    steps = np.asarray([row["global_step"] for row in val_rows], dtype=np.int64)
    pca50 = np.asarray([row["val_wasserstein_pca50_mean"] for row in val_rows], dtype=np.float64)
    r2 = np.asarray([row["val_distribution_mean_r2_mean"] for row in val_rows], dtype=np.float64)
    best_idx = int(np.argmin(pca50))
    return {
        **task_cfg,
        "steps": steps,
        "pca50": pca50,
        "r2": r2,
        "best_idx": best_idx,
        "best_step": int(steps[best_idx]),
        "best_pca50": float(pca50[best_idx]),
        "best_r2": float(r2[best_idx]),
        "n_validation_points": int(len(val_rows)),
    }


def _configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def _style_axis(ax):
    ax.grid(True, axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", pad=3)


def _tick_labels(steps):
    return [f"{step // 1000}k" if step % 1000 == 0 else str(step) for step in steps]


def _save_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _render_single_task(task):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(5.2, 3.3),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 0.9]},
        constrained_layout=False,
    )
    ax_top, ax_bottom = axes

    for ax in axes:
        _style_axis(ax)
        ax.axvline(task["best_step"], color=BEST_COLOR, linestyle="--", linewidth=1.0, alpha=0.9)

    ax_top.plot(task["steps"], task["pca50"], color=PCA50_COLOR, marker="o", markersize=4.6, linewidth=1.9)
    ax_top.scatter([task["best_step"]], [task["best_pca50"]], color=BEST_COLOR, s=34, zorder=4)
    ax_top.text(
        0.72,
        0.93,
        f"Best checkpoint\nstep={task['best_step']}, WD={task['best_pca50']:.3f}",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "boxstyle": "round,pad=0.2"},
    )
    ax_top.set_ylabel("Val PCA50 WD ↓")

    ax_bottom.plot(task["steps"], task["r2"], color=R2_COLOR, marker="o", markersize=4.2, linewidth=1.7)
    ax_bottom.scatter([task["best_step"]], [task["best_r2"]], color=BEST_COLOR, s=30, zorder=4)
    ax_bottom.set_ylabel(r"Val Mean $R^2$ ↑")
    ax_bottom.set_xlabel("Global step")
    ax_bottom.set_xticks(task["steps"].tolist())
    ax_bottom.set_xticklabels(_tick_labels(task["steps"]))

    fig.subplots_adjust(left=0.12, right=0.985, top=0.98, bottom=0.16, hspace=0.16)
    stem = f"{task['key']}_validation_curve"
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "task": task["key"],
        "log_path": str(task["log_path"]),
        "output_png": str(OUTPUT_DIR / f"{stem}.png"),
        "output_pdf": str(OUTPUT_DIR / f"{stem}.pdf"),
        "best_step": task["best_step"],
        "best_val_wasserstein_pca50_mean": task["best_pca50"],
        "best_step_val_distribution_mean_r2_mean": task["best_r2"],
        "n_validation_points": task["n_validation_points"],
    }


def _render_combined(tasks):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.4, 3.35),
        sharex="col",
        constrained_layout=False,
    )

    for col, task in enumerate(tasks):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        for ax in (ax_top, ax_bottom):
            _style_axis(ax)
            ax.axvline(task["best_step"], color=BEST_COLOR, linestyle="--", linewidth=0.95, alpha=0.9)

        ax_top.plot(task["steps"], task["pca50"], color=PCA50_COLOR, marker="o", markersize=3.8, linewidth=1.7)
        ax_top.scatter([task["best_step"]], [task["best_pca50"]], color=BEST_COLOR, s=24, zorder=4)
        ax_top.set_title(task["name"], pad=6, fontweight="bold")
        ax_top.text(
            0.03,
            0.05,
            f"best {task['best_step'] // 1000 if task['best_step'] % 1000 == 0 else task['best_step']}k, WD={task['best_pca50']:.3f}" if task["best_step"] % 1000 == 0 else f"best {task['best_step']}, WD={task['best_pca50']:.3f}",
            transform=ax_top.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "boxstyle": "round,pad=0.15"},
        )

        ax_bottom.plot(task["steps"], task["r2"], color=R2_COLOR, marker="o", markersize=3.6, linewidth=1.55)
        ax_bottom.scatter([task["best_step"]], [task["best_r2"]], color=BEST_COLOR, s=22, zorder=4)
        ax_bottom.set_xticks(task["steps"].tolist())
        ax_bottom.set_xticklabels(_tick_labels(task["steps"]))
        ax_bottom.set_xlabel("Global step")

    axes[0, 0].set_ylabel("Val PCA50 WD ↓")
    axes[1, 0].set_ylabel(r"Val Mean $R^2$ ↑")

    for col in (1, 2):
        axes[0, col].set_ylabel("")
        axes[1, col].set_ylabel("")

    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.18, wspace=0.16, hspace=0.17)
    stem = "all_tasks_validation_curves"
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "output_png": str(OUTPUT_DIR / f"{stem}.png"),
        "output_pdf": str(OUTPUT_DIR / f"{stem}.pdf"),
        "tasks": [
            {
                "task": task["key"],
                "best_step": task["best_step"],
                "best_val_wasserstein_pca50_mean": task["best_pca50"],
                "best_step_val_distribution_mean_r2_mean": task["best_r2"],
            }
            for task in tasks
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Plot Route A validation curves from train_log.jsonl files.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for output figures.")
    parser.add_argument("--interpolation-log", default=str(TASKS[0]["log_path"]), help="Interpolation train_log.jsonl.")
    parser.add_argument("--near-log", default=str(TASKS[1]["log_path"]), help="Near extrapolation train_log.jsonl.")
    parser.add_argument("--far-log", default=str(TASKS[2]["log_path"]), help="Far extrapolation train_log.jsonl.")
    args = parser.parse_args()

    globals().update(
        {
            "OUTPUT_DIR": Path(args.output_dir),
            "TASKS": [
                {**TASKS[0], "log_path": Path(args.interpolation_log)},
                {**TASKS[1], "log_path": Path(args.near_log)},
                {**TASKS[2], "log_path": Path(args.far_log)},
            ],
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_style()

    tasks = [_prepare_task(task_cfg) for task_cfg in TASKS]
    single_summaries = [_render_single_task(task) for task in tasks]
    combined_summary = _render_combined(tasks)

    payload = {
        "single_task_figures": single_summaries,
        "combined_figure": combined_summary,
    }
    _save_json(OUTPUT_DIR / "validation_curve_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
