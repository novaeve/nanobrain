import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scfit.model import scFiTModel  # noqa: E402
from scfit.route_a.protocol import load_protocol_bundle, resolve_generation_sources  # noqa: E402
from scfit.route_a.train import RouteATrainer, parse_time_from_bundle  # noqa: E402
from scfit.utils import sinkhorn_distance  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "mouse" / "config_mouse_route_a_near.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "figures" / "mouse_near_rollout"

BACKGROUND_COLOR = "#D5D1D1"
TARGET_COLOR = "#E9687A"
PANEL_BG = "#fcfcfc"
ROLLOUT_STEP_COLORS = ["#B6B3D6", "#CFCCE3", "#F8B2A2", "#F1837A"]


def _configure_cpu_thread_env():
    defaults = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/matplotlib",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _load_cfg(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_assets(assets_dir: Path) -> Dict:
    summary = json.loads((assets_dir / "asset_summary.json").read_text())
    pca = joblib.load(summary["pca_model_path"])
    return {
        "summary": summary,
        "pca": pca,
    }


def _build_model(cfg: Dict, bundle: Dict):
    return scFiTModel(
        n_genes=int(bundle["bundle_meta"]["n_genes"]),
        top_k_genes=int(bundle["bundle_meta"]["fixed_panel_size"]),
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        d_ff=int(cfg["model"]["d_ff"]),
        nlayers=int(cfg["model"]["nlayers"]),
        dropout=float(cfg["model"]["dropout"]),
        value_max=float(cfg["model"].get("value_encoder", {}).get("max_value", 32.0)),
        share_adaln_projection=bool(cfg["model"].get("adaln", {}).get("share_projection", False)),
        time_conditioning_mode=str(cfg["model"].get("time_conditioning", {}).get("mode", "adaln")),
        aux_bin_ce=False,
        n_bins=0,
        n_cell_types=0,
        pseudotime_head=False,
    )


def _select_best_log_entry(log_path: Path, metric: str, mode: str):
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    val_rows = [row for row in rows if metric in row and row[metric] is not None]
    if not val_rows:
        raise ValueError(f"no validation entries with metric={metric} in {log_path}")
    if mode == "min":
        return min(val_rows, key=lambda row: float(row[metric]))
    if mode == "max":
        return max(val_rows, key=lambda row: float(row[metric]))
    raise ValueError(f"unsupported mode={mode}")


def _resolve_checkpoint(exp_dir: Path, cfg: Dict, mode: str, explicit_path: str = None) -> Path:
    checkpoints_dir = exp_dir / "checkpoints"
    if mode == "path":
        if not explicit_path:
            raise ValueError("--checkpoint-path is required when checkpoint-mode=path")
        ckpt_path = Path(explicit_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        return ckpt_path
    if mode == "final":
        ckpt_path = checkpoints_dir / "final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"final checkpoint not found: {ckpt_path}")
        return ckpt_path
    if mode == "best_pca50":
        log_path = exp_dir / "train_log.jsonl"
        best = _select_best_log_entry(log_path, metric="val_wasserstein_pca50_mean", mode="min")
        global_step = int(best["global_step"])
        steps_per_epoch = int(cfg["training"].get("iters_per_epoch", cfg["training"].get("steps_per_epoch", 200)))
        if global_step % steps_per_epoch != 0:
            raise ValueError(
                f"best validation step={global_step} does not align with epoch checkpoints "
                f"(steps_per_epoch={steps_per_epoch})"
            )
        epoch = global_step // steps_per_epoch
        ckpt_path = checkpoints_dir / f"epoch_{epoch}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        return ckpt_path
    raise ValueError(f"unsupported checkpoint mode: {mode}")


def _sample_indices(rng: np.random.Generator, idx: np.ndarray, n: int, replace: bool = False):
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("cannot sample from empty index array")
    if replace:
        return rng.choice(idx, size=int(n), replace=True).astype(np.int64)
    if idx.size <= int(n):
        return idx.copy()
    return np.sort(rng.choice(idx, size=int(n), replace=False)).astype(np.int64)


def _parse_display_steps(raw: str, n_steps: int, max_panels: int = 6) -> List[int]:
    max_step = int(n_steps)
    if raw:
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
        values = sorted(set(values))
        return [v for v in values if 0 <= v <= max_step]
    if max_step <= 3:
        return list(range(max_step + 1))
    values = [0, max(1, round(max_step * 0.3)), max(1, round(max_step * 0.7)), max_step]
    return sorted(set(int(v) for v in values))


def _build_rollout_states(
    trainer: RouteATrainer,
    src_expr_np: np.ndarray,
    t_src: float,
    t_tgt: float,
    n_steps: int,
) -> List[np.ndarray]:
    trainer.model.eval()
    outputs = []
    micro_bs = max(1, int(trainer.infer_micro_batch_size))
    dt = 1.0 / max(int(n_steps), 1)
    all_chunks_per_step: List[List[np.ndarray]] = [[] for _ in range(int(n_steps) + 1)]
    n_chunks = (src_expr_np.shape[0] + micro_bs - 1) // micro_bs
    with torch.no_grad():
        for start in tqdm(
            range(0, src_expr_np.shape[0], micro_bs),
            total=n_chunks,
            desc="rollout-chunks",
            leave=False,
        ):
            end = min(start + micro_bs, src_expr_np.shape[0])
            x = torch.as_tensor(src_expr_np[start:end], dtype=torch.float32, device=trainer.device)
            all_chunks_per_step[0].append(x.detach().cpu().numpy().astype(np.float32))
            for step in range(int(n_steps)):
                t_flow = torch.full((x.shape[0], 1), step * dt, dtype=torch.float32, device=trainer.device)
                _, vel = trainer._predict_target_and_velocity(x, t_flow=t_flow, t_src=t_src, t_tgt=t_tgt)
                x = torch.clamp(x + vel * dt, min=0.0)
                all_chunks_per_step[step + 1].append(x.detach().cpu().numpy().astype(np.float32))
    for chunks in all_chunks_per_step:
        outputs.append(np.concatenate(chunks, axis=0))
    return outputs


def _sinkhorn_np(x: np.ndarray, y: np.ndarray, epsilon: float, max_iter: int) -> float:
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return float(sinkhorn_distance(x_t, y_t, epsilon=epsilon, max_iter=max_iter).item())


def _mean_expr_r2(pred: np.ndarray, real: np.ndarray):
    if pred.shape[0] == 0 or real.shape[0] == 0:
        return None
    x = pred.mean(axis=0)
    y = real.mean(axis=0)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr * corr)


def _fit_embedder(embedding: str, context_pca50: np.ndarray, random_seed: int):
    if embedding == "pca":
        embedder = PCA(n_components=2, random_state=random_seed)
        embedder.fit(context_pca50)
        return embedder
    if embedding == "umap":
        try:
            os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
            import umap
        except Exception as exc:
            print(f"[rollout_vis] failed to import UMAP ({exc}); falling back to PCA")
            embedder = PCA(n_components=2, random_state=random_seed)
            embedder.fit(context_pca50)
            return embedder

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.35,
            metric="euclidean",
            random_state=random_seed,
            transform_seed=random_seed,
            low_memory=True,
        )
        reducer.fit(context_pca50)
        return reducer
    raise ValueError(f"unsupported embedding={embedding}")


def _transform_2d(embedder, x: np.ndarray) -> np.ndarray:
    return np.asarray(embedder.transform(x), dtype=np.float32)


def _panel_label(step: int, n_steps: int) -> str:
    if step == 0:
        return "Observed source"
    if step == n_steps:
        return "Final"
    return f"k={step}"


def _scatter_panel(ax, background_xy, target_xy, rollout_xy, label_text, step_color):
    ax.set_facecolor(PANEL_BG)
    ax.scatter(
        background_xy[:, 0],
        background_xy[:, 1],
        s=2.3,
        c=BACKGROUND_COLOR,
        alpha=0.20,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        target_xy[:, 0],
        target_xy[:, 1],
        s=4.9,
        facecolors="none",
        edgecolors=TARGET_COLOR,
        alpha=0.62,
        linewidths=0.48,
        rasterized=True,
    )
    ax.scatter(
        rollout_xy[:, 0],
        rollout_xy[:, 1],
        s=4.6,
        c=[step_color],
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    ax.text(
        0.04,
        0.96,
        label_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.1,
        fontweight="bold",
        color="black",
        bbox={
            "facecolor": "white",
            "alpha": 0.90,
            "edgecolor": "none",
            "boxstyle": "round,pad=0.18",
        },
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _parse_target_times(raw: str, bundle: Dict) -> List[str]:
    if raw:
        target_times = [str(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        target_times = [str(bundle["eval_time_bins"][0])]
    invalid = [t for t in target_times if t not in bundle["eval_time_bins"]]
    if invalid:
        raise ValueError(f"target times must be in eval_time_bins, got invalid={invalid}")
    return target_times


def _render_single_target_figure(
    *,
    output_dir: Path,
    stem: str,
    source_time: str,
    target_time: str,
    embedding: str,
    display_steps: List[int],
    n_steps: int,
    background_xy: np.ndarray,
    target_xy: np.ndarray,
    state_xy: List[np.ndarray],
    step_metrics: List[Dict],
    n_cells: int,
    max_context_cells: int,
):
    n_panels = len(display_steps)
    n_cols = n_panels
    n_rows = 1
    fig_width = min(5.45, 1.28 * n_panels + 0.28)
    fig_height = 1.58
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), constrained_layout=False)
    axes = np.atleast_1d(axes).ravel()

    step_color_map = {
        step: ROLLOUT_STEP_COLORS[idx]
        for idx, step in enumerate(display_steps)
    }

    for ax, step in zip(axes, display_steps):
        _scatter_panel(
            ax,
            background_xy=background_xy,
            target_xy=target_xy,
            rollout_xy=state_xy[step],
            label_text=_panel_label(step, n_steps),
            step_color=step_color_map[step],
        )
    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.998, top=0.985, bottom=0.04, wspace=0.045)
    fig.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "source_time": source_time,
        "target_time": target_time,
        "n_steps": int(n_steps),
        "display_steps": display_steps,
        "embedding": embedding,
        "step_metrics": step_metrics,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_cells": int(n_cells),
        "max_context_cells": int(max_context_cells),
    }
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    np.savez_compressed(
        output_dir / f"{stem}_coords.npz",
        background_xy=background_xy.astype(np.float32),
        target_xy=target_xy.astype(np.float32),
        **{f"rollout_xy_step_{step}": state_xy[step].astype(np.float32) for step in range(len(state_xy))},
    )


def main():
    _configure_cpu_thread_env()
    parser = argparse.ArgumentParser(description="Plot temporal rollout trajectories for mouse near extrapolation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Route A config yaml")
    parser.add_argument(
        "--checkpoint-mode",
        choices=["best_pca50", "final", "path"],
        default="best_pca50",
        help="Which checkpoint to use",
    )
    parser.add_argument("--checkpoint-path", default=None, help="Used only when checkpoint-mode=path")
    parser.add_argument("--target-time", default=None, help="Single held-out target time bin; deprecated in favor of --target-times")
    parser.add_argument("--target-times", default="", help="Comma-separated held-out target time bins; default is earliest eval bin")
    parser.add_argument("--source-time", default=None, help="Optional source time override")
    parser.add_argument("--n-steps", type=int, default=None, help="Euler rollout steps; default uses config inference.n_steps")
    parser.add_argument("--n-cells", type=int, default=1500, help="Number of cells to visualize")
    parser.add_argument("--max-context-cells", type=int, default=20000, help="Background true cells for 2D fit")
    parser.add_argument("--embedding", choices=["umap", "pca"], default="pca", help="2D embedding on PCA-50 space")
    parser.add_argument(
        "--display-steps",
        default="",
        help="Comma-separated rollout steps to show as panels, e.g. 0,2,4,6,8,10; default auto-selects up to 6 panels",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--device", default=None, help="cuda:0 or cpu; default auto")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", help="Reuse cached coords/metrics when available")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Ignore cached coords/metrics and recompute")
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    config_path = Path(args.config).resolve()
    cfg = _load_cfg(config_path)
    protocol_path = str((cfg.get("protocol", {}) or {}).get("path", "") or "").strip()
    if not protocol_path:
        raise ValueError("config is missing protocol.path")
    bundle = load_protocol_bundle(protocol_path)

    run_name = cfg["experiment"].get("run_name") or Path(protocol_path).stem
    exp_dir = Path(cfg["experiment"].get("output_dir", ROOT / "results" / "experiments" / "mouse")) / run_name
    ckpt_path = _resolve_checkpoint(exp_dir, cfg, mode=args.checkpoint_mode, explicit_path=args.checkpoint_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(args.n_steps or cfg.get("inference", {}).get("n_steps", 10))
    display_steps = _parse_display_steps(args.display_steps, n_steps=n_steps)
    if not display_steps:
        raise ValueError("no display steps selected")
    target_times_arg = args.target_times or (str(args.target_time) if args.target_time else "")
    target_times = _parse_target_times(target_times_arg, bundle)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = None
    pca50 = None
    for target_time in tqdm(target_times, desc="targets", leave=False):
        resolved_sources = resolve_generation_sources(bundle, target_time)
        default_source = str(resolved_sources[0]["time_bin"])
        source_time = str(args.source_time or default_source)
        desired_n_cells = int(min(args.n_cells, bundle["time_to_indices"][target_time].shape[0]))
        stem = f"near_rollout_{source_time}_to_{target_time}_{args.embedding}_{args.checkpoint_mode}"
        coords_path = output_dir / f"{stem}_coords.npz"
        metrics_path = output_dir / f"{stem}_metrics.json"
        can_use_cache = False
        if args.use_cache and coords_path.exists() and metrics_path.exists():
            try:
                cached_payload = json.loads(metrics_path.read_text())
                cached_coords = np.load(coords_path)
                cached_n_steps = int(cached_payload.get("n_steps", -1))
                has_final_step = f"rollout_xy_step_{cached_n_steps}" in cached_coords.files
                can_use_cache = (
                    cached_payload.get("source_time") == source_time
                    and cached_payload.get("target_time") == target_time
                    and str(cached_payload.get("embedding")) == str(args.embedding)
                    and cached_n_steps == n_steps
                    and int(cached_payload.get("n_cells", desired_n_cells)) == desired_n_cells
                    and has_final_step
                )
            except Exception:
                can_use_cache = False

        if can_use_cache:
            print(f"[rollout_vis] loading cached rollout coordinates for {source_time} -> {target_time}")
            background_xy = np.asarray(cached_coords["background_xy"], dtype=np.float32)
            target_xy = np.asarray(cached_coords["target_xy"], dtype=np.float32)
            state_xy = [
                np.asarray(cached_coords[f"rollout_xy_step_{step_idx}"], dtype=np.float32)
                for step_idx in range(n_steps + 1)
            ]
            step_metrics = list(cached_payload["step_metrics"])
            display_steps = _parse_display_steps(args.display_steps, n_steps=n_steps)
            _render_single_target_figure(
                output_dir=output_dir,
                stem=stem,
                source_time=source_time,
                target_time=target_time,
                embedding=args.embedding,
                display_steps=display_steps,
                n_steps=n_steps,
                background_xy=background_xy,
                target_xy=target_xy,
                state_xy=state_xy,
                step_metrics=step_metrics,
                n_cells=int(cached_payload.get("n_cells", state_xy[0].shape[0])),
                max_context_cells=int(cached_payload.get("max_context_cells", background_xy.shape[0])),
            )
            payload = {
                "config_path": str(config_path),
                "experiment_dir": str(exp_dir),
                "checkpoint_path": str(ckpt_path),
                "assets_dir": None,
                "source_time": source_time,
                "target_time": target_time,
                "n_steps": int(n_steps),
                "display_steps": display_steps,
                "embedding": args.embedding,
                "cache_hit": True,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            continue

        if trainer is None:
            model = _build_model(cfg, bundle)
            trainer = RouteATrainer(model=model, device=device, exp_dir=str(exp_dir), cfg=cfg, bundle=bundle)
            ckpt = torch.load(ckpt_path, map_location=device)
            trainer.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            trainer.model.eval()
            assets = _load_assets(Path(trainer.assets_dir))
            pca50 = assets["pca"]

        src_idx_all = bundle["time_to_indices"][source_time]
        tgt_idx_all = bundle["time_to_indices"][target_time]
        n_cells = desired_n_cells
        tgt_idx = _sample_indices(rng, tgt_idx_all, n=n_cells, replace=False)
        src_idx = _sample_indices(rng, src_idx_all, n=n_cells, replace=(src_idx_all.shape[0] < n_cells))
        src_expr = np.asarray(bundle["expression"][src_idx], dtype=np.float32)
        tgt_expr = np.asarray(bundle["expression"][tgt_idx], dtype=np.float32)

        t_src = float(parse_time_from_bundle(bundle, source_time))
        t_tgt = float(parse_time_from_bundle(bundle, target_time))
        print(f"[rollout_vis] computing rollout states for {source_time} -> {target_time}")
        rollout_states = _build_rollout_states(trainer, src_expr, t_src=t_src, t_tgt=t_tgt, n_steps=n_steps)

        tgt_pca50 = pca50.transform(tgt_expr)
        state_pca50 = []
        for state in tqdm(rollout_states, desc=f"pca50-{target_time}", leave=False):
            state_pca50.append(pca50.transform(state))

        context_pool = bundle["time_to_indices"][source_time]
        if target_time != source_time:
            context_pool = np.concatenate([context_pool, bundle["time_to_indices"][target_time]], axis=0)
        all_pool = np.arange(bundle["expression"].shape[0], dtype=np.int64)
        extra_n = max(0, int(args.max_context_cells) - int(context_pool.shape[0]))
        if extra_n > 0 and all_pool.shape[0] > 0:
            extra_idx = _sample_indices(rng, all_pool, n=min(extra_n, all_pool.shape[0]), replace=False)
            context_idx = np.unique(np.concatenate([context_pool, extra_idx], axis=0))
        else:
            context_idx = np.unique(context_pool)
        context_expr = np.asarray(bundle["expression"][context_idx], dtype=np.float32)
        context_pca50 = pca50.transform(context_expr)

        print(f"[rollout_vis] fitting 2D embedding for {source_time} -> {target_time}")
        embedder = _fit_embedder(args.embedding, context_pca50=context_pca50, random_seed=args.seed)
        background_xy = _transform_2d(embedder, context_pca50)
        target_xy = _transform_2d(embedder, tgt_pca50)
        state_xy = []
        for state in tqdm(state_pca50, desc=f"embed-{target_time}", leave=False):
            state_xy.append(_transform_2d(embedder, state))

        step_metrics = []
        for step_idx, state in enumerate(tqdm(rollout_states, desc=f"metrics-{target_time}", leave=False)):
            state_pca = state_pca50[step_idx]
            metric_row = {
                "step": int(step_idx),
                "wasserstein_pca50_to_target": _sinkhorn_np(
                    state_pca, tgt_pca50, epsilon=trainer.ot_epsilon, max_iter=trainer.ot_max_iter
                ),
                "distribution_mean_r2_to_target": _mean_expr_r2(state, tgt_expr),
            }
            step_metrics.append(metric_row)

        _render_single_target_figure(
            output_dir=output_dir,
            stem=stem,
            source_time=source_time,
            target_time=target_time,
            embedding=args.embedding,
            display_steps=display_steps,
            n_steps=n_steps,
            background_xy=background_xy,
            target_xy=target_xy,
            state_xy=state_xy,
            step_metrics=step_metrics,
            n_cells=int(n_cells),
            max_context_cells=int(args.max_context_cells),
        )

        payload = {
            "config_path": str(config_path),
            "experiment_dir": str(exp_dir),
            "checkpoint_path": str(ckpt_path),
            "assets_dir": str(trainer.assets_dir),
            "source_time": source_time,
            "target_time": target_time,
            "n_steps": int(n_steps),
            "display_steps": display_steps,
            "n_cells": int(n_cells),
            "embedding": args.embedding,
            "step_metrics": step_metrics,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
