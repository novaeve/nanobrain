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
SMOOTH_WINDOW = 9

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

LOSS_COLOR = "#93ACCE"
CFM_COLOR = "#E8B9B7"
RAW_COLOR = "#B8B8B8"
GRID_COLOR = "#D8D8D8"


def _configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
        }
    )


def _load_train_rows(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"training log not found: {log_path}")
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    train_rows = [row for row in rows if "loss" in row and "cfm" in row]
    if not train_rows:
        raise ValueError(f"no training rows found in {log_path}")
    deduped_by_step = {}
    duplicate_steps = {}
    for row in train_rows:
        step = int(row["global_step"])
        if step in deduped_by_step:
            duplicate_steps[step] = duplicate_steps.get(step, 1) + 1
        deduped_by_step[step] = row
    deduped_rows = [deduped_by_step[step] for step in sorted(deduped_by_step)]
    return deduped_rows, duplicate_steps


def _smooth(y, window):
    if window <= 1 or len(y) < 2:
        return np.asarray(y, dtype=np.float64)
    half = window // 2
    padded = np.pad(np.asarray(y, dtype=np.float64), (half, half), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _tick_labels(steps):
    labels = []
    for step in steps:
        if step >= 1000 and step % 1000 == 0:
            labels.append(f"{step // 1000}k")
        else:
            labels.append(str(step))
    return labels


def _style_axis(ax):
    ax.grid(True, axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", pad=3)


def _prepare_task(task_cfg):
    train_rows, duplicate_steps = _load_train_rows(task_cfg["log_path"])
    steps = np.asarray([row["global_step"] for row in train_rows], dtype=np.int64)
    loss = np.asarray([row["loss"] for row in train_rows], dtype=np.float64)
    cfm = np.asarray([row["cfm"] for row in train_rows], dtype=np.float64)
    return {
        **task_cfg,
        "steps": steps,
        "loss": loss,
        "cfm": cfm,
        "loss_smooth": _smooth(loss, SMOOTH_WINDOW),
        "cfm_smooth": _smooth(cfm, SMOOTH_WINDOW),
        "n_points": int(len(train_rows)),
        "duplicate_steps": {str(k): int(v) for k, v in sorted(duplicate_steps.items())},
    }


def _render_combined(tasks):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.4, 3.5),
        sharex="col",
        constrained_layout=False,
    )

    for col, task in enumerate(tasks):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        for ax in (ax_top, ax_bottom):
            _style_axis(ax)

        ax_top.plot(task["steps"], task["loss"], color=RAW_COLOR, linewidth=0.9, alpha=0.55, zorder=1)
        ax_top.plot(task["steps"], task["loss_smooth"], color=LOSS_COLOR, linewidth=1.9, zorder=2)
        ax_top.set_title(task["name"], pad=6, fontweight="bold")
        ax_top.text(
            0.03,
            0.06,
            f"window={SMOOTH_WINDOW}",
            transform=ax_top.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.1,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "boxstyle": "round,pad=0.15"},
        )

        ax_bottom.plot(task["steps"], task["cfm"], color=RAW_COLOR, linewidth=0.9, alpha=0.55, zorder=1)
        ax_bottom.plot(task["steps"], task["cfm_smooth"], color=CFM_COLOR, linewidth=1.85, zorder=2)
        tick_steps = np.asarray([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
        ax_bottom.set_xticks(tick_steps.tolist())
        ax_bottom.set_xticklabels(_tick_labels(tick_steps))
        ax_bottom.set_xlabel("Global step")

    axes[0, 0].set_ylabel("Train loss")
    axes[1, 0].set_ylabel("Train CFM")
    for col in (1, 2):
        axes[0, col].set_ylabel("")
        axes[1, col].set_ylabel("")

    axes[0, 0].legend(
        ["raw", "smoothed"],
        frameon=False,
        loc="upper right",
        handlelength=1.8,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.18, wspace=0.17, hspace=0.20)
    stem = "all_tasks_training_dynamics"
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "output_png": str(OUTPUT_DIR / f"{stem}.png"),
        "output_pdf": str(OUTPUT_DIR / f"{stem}.pdf"),
        "smooth_window": int(SMOOTH_WINDOW),
        "tasks": [
            {
                "task": task["key"],
                "n_points": task["n_points"],
                "n_duplicate_steps": int(len(task["duplicate_steps"])),
                "loss_start": float(task["loss"][0]),
                "loss_end": float(task["loss"][-1]),
                "cfm_start": float(task["cfm"][0]),
                "cfm_end": float(task["cfm"][-1]),
                "duplicate_steps": task["duplicate_steps"],
            }
            for task in tasks
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Plot Route A training loss dynamics from train_log.jsonl files.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for output figures.")
    parser.add_argument("--interpolation-log", default=str(TASKS[0]["log_path"]), help="Interpolation train_log.jsonl.")
    parser.add_argument("--near-log", default=str(TASKS[1]["log_path"]), help="Near extrapolation train_log.jsonl.")
    parser.add_argument("--far-log", default=str(TASKS[2]["log_path"]), help="Far extrapolation train_log.jsonl.")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW, help="Moving-average smoothing window.")
    args = parser.parse_args()

    globals().update(
        {
            "OUTPUT_DIR": Path(args.output_dir),
            "SMOOTH_WINDOW": int(args.smooth_window),
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
    payload = _render_combined(tasks)
    (OUTPUT_DIR / "training_dynamics_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
