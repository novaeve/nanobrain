import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scfit.model import scFiTModel  # noqa: E402
from scfit.paths import as_posix_path  # noqa: E402
from scfit.route_a.eval import evaluate_artifact_dir  # noqa: E402
from scfit.route_a.protocol import load_protocol_bundle  # noqa: E402
from scfit.route_a.train import RouteATrainer  # noqa: E402


def _configure_cpu_thread_env():
    defaults = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _load_cfg(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(cfg: dict, bundle: dict):
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


def main():
    _configure_cpu_thread_env()
    parser = argparse.ArgumentParser(description="Evaluate best scFiT checkpoint from train_log validation metrics.")
    parser.add_argument("--config", required=True, help="Route A config yaml")
    parser.add_argument("--metric", default="val_wasserstein_pca50_mean", help="Validation metric used to select best checkpoint")
    parser.add_argument("--mode", choices=["min", "max"], default="min", help="Whether lower or higher metric is better")
    parser.add_argument("--tag", default="best_val", help="Subdirectory under eval_artifacts/")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    protocol_path = str((cfg.get("protocol", {}) or {}).get("path", "") or "").strip()
    if not protocol_path:
        raise ValueError("config is missing protocol.path")
    protocol_path = as_posix_path(protocol_path)
    cfg["protocol"]["path"] = protocol_path
    if cfg.get("evaluation", {}).get("assets_dir"):
        cfg["evaluation"]["assets_dir"] = as_posix_path(cfg["evaluation"]["assets_dir"])

    exp_cfg = cfg["experiment"]
    run_name = exp_cfg.get("run_name") or Path(protocol_path).stem
    exp_dir = Path(as_posix_path(exp_cfg.get("output_dir", "results/experiments/mouse"))) / run_name
    log_path = exp_dir / "train_log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"train log not found: {log_path}")

    best = _select_best_log_entry(log_path, metric=args.metric, mode=args.mode)
    global_step = int(best["global_step"])
    steps_per_epoch = int(cfg["training"].get("iters_per_epoch", cfg["training"].get("steps_per_epoch", 200)))
    if global_step % steps_per_epoch != 0:
        raise ValueError(
            f"best validation step={global_step} does not align with checkpoint epochs "
            f"(steps_per_epoch={steps_per_epoch})"
        )
    epoch = global_step // steps_per_epoch
    ckpt_path = exp_dir / "checkpoints" / f"epoch_{epoch}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    bundle = load_protocol_bundle(protocol_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(cfg, bundle)
    trainer = RouteATrainer(model=model, device=device, exp_dir=str(exp_dir), cfg=cfg, bundle=bundle)

    ckpt = torch.load(ckpt_path, map_location=device)
    trainer.model.load_state_dict(ckpt["model_state_dict"], strict=True)
    artifact_dir = exp_dir / "eval_artifacts" / args.tag
    trainer.build_prediction_artifact(
        method_name="scFiT",
        output_dir=str(artifact_dir),
        max_cells_per_time=trainer.final_eval_max_cells_per_time,
    )
    metrics = evaluate_artifact_dir(
        artifact_dir=str(artifact_dir),
        assets_dir=trainer.assets_dir,
        metadata_df=bundle["metadata_df"],
        hvg_gene_names=bundle["bundle_meta"]["hvg_gene_names"],
        epsilon=trainer.ot_epsilon,
        max_iter=trainer.ot_max_iter,
    )
    selection = {
        "selection_metric": args.metric,
        "selection_mode": args.mode,
        "selected_global_step": global_step,
        "selected_epoch": epoch,
        "selected_checkpoint": str(ckpt_path.resolve()),
        "selected_metric_value": best[args.metric],
    }
    selection_path = artifact_dir / "selection_summary.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2))
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
