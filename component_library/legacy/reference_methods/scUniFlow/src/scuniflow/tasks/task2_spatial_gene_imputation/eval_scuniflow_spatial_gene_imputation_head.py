import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import (
    DEFAULT_REF_H5AD,
    DEFAULT_VISIUM_ROOT,
    build_target_prediction_adata,
    load_reference_metadata,
    load_visium_mouse_brain,
    save_json,
    summarize_target_metrics,
    transform_counts_array,
)
from spatial_head_common import (
    ScUniFlowSpatialImputer,
    SpatialSpotTargetDataset,
    normalize_spatial_coords,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned scUniFlow spatial gene imputation head"
    )
    parser.add_argument("--task_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ref_h5ad", default="", help="override reference h5ad; default from checkpoint metadata")
    parser.add_argument("--visium_root", default="", help="override visium root; default from checkpoint metadata")
    parser.add_argument("--sample_ids", default="", help="override sample ids, comma-separated")
    parser.add_argument("--save_h5ad", action="store_true")
    parser.add_argument("--save_arrays", action="store_true")
    return parser.parse_args()


def _resolve_indices(ref, ids: list[str]) -> np.ndarray:
    index = {str(g): i for i, g in enumerate(ref.ensembl_ids.tolist())}
    missing = [g for g in ids if str(g) not in index]
    if missing:
        raise ValueError(f"Missing genes while restoring saved panel: {missing[:10]}")
    return np.asarray([index[str(g)] for g in ids], dtype=np.int64)


def _sample_ids_arg(raw: str, fallback):
    if raw:
        return [x.strip() for x in str(raw).split(",") if x.strip()]
    if fallback is None:
        return []
    if isinstance(fallback, str):
        return [] if fallback == "all" else [fallback]
    return list(fallback)


@torch.no_grad()
def main():
    args = parse_args()
    task_ckpt = Path(args.task_ckpt).resolve()
    run_dir = task_ckpt.parent
    split_info_path = run_dir / "split_info.json"
    panel_info_path = run_dir / "panel_info.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"split info not found: {split_info_path}")
    if not panel_info_path.exists():
        raise FileNotFoundError(f"panel info not found: {panel_info_path}")

    split_info = json_load(split_info_path)
    panel_info = json_load(panel_info_path)
    ckpt = torch.load(task_ckpt, map_location="cpu")

    ref_h5ad = str(Path(args.ref_h5ad).resolve()) if args.ref_h5ad else str(split_info["ref_h5ad"])
    visium_root = str(Path(args.visium_root).resolve()) if args.visium_root else str(split_info["visium_root"])
    sample_ids = _sample_ids_arg(args.sample_ids, split_info.get("sample_ids", []))

    ref = load_reference_metadata(Path(ref_h5ad))
    spatial = load_visium_mouse_brain(
        visium_root=Path(visium_root),
        ref=ref,
        sample_ids=sample_ids or None,
        in_tissue_only=bool(split_info.get("in_tissue_only", True)),
        max_spots=int(split_info.get("max_spots", 0)),
        sample_seed=int(ckpt.get("sample_seed", 42)),
    )
    obs_name_to_idx = {str(name): i for i, name in enumerate(spatial.obs_names.astype(str))}

    if args.split == "train":
        selected_obs = split_info.get("obs_names_train", [])
    elif args.split == "val":
        selected_obs = split_info.get("obs_names_val", [])
    else:
        selected_obs = spatial.obs_names.astype(str).tolist()
    selected_idx = np.asarray([obs_name_to_idx[name] for name in selected_obs if name in obs_name_to_idx], dtype=np.int64)
    if selected_idx.size == 0:
        raise ValueError(f"No spots found for split={args.split}")

    observed_idx = _resolve_indices(ref, panel_info["observed_ensembl_ids"])
    target_idx = _resolve_indices(ref, panel_info["target_ensembl_ids"])

    spatial_xy = np.zeros((spatial.n_obs, 2), dtype=np.float32)
    if "spatial" in spatial.obsm:
        spatial_xy = normalize_spatial_coords(np.asarray(spatial.obsm["spatial"], dtype=np.float32))

    expression_space = str(ckpt.get("expression_space", split_info.get("expression_space", "logcpm")))
    dataset = SpatialSpotTargetDataset(
        count_matrix=spatial.X,
        gene_ids=ref.gene_ids,
        spatial_xy=spatial_xy,
        indices=selected_idx,
        observed_idx=observed_idx,
        target_idx=target_idx,
        expression_space=expression_space,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(torch.cuda.is_available() and str(args.device) == "cuda"),
    )

    cfg_path = Path(ckpt["config_path"]).resolve()
    import yaml

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = ScUniFlowSpatialImputer(
        n_genes=int(model_cfg["n_genes"]),
        d_model=int(model_cfg["d_model"]),
        grid_size=int(model_cfg["grid_size"]),
        n_layers=int(model_cfg.get("n_layers", 8)),
        d_state_row=int(model_cfg.get("d_state_row", 64)),
        d_state_col=int(model_cfg.get("d_state_col", 16)),
        activation_checkpointing=bool(model_cfg.get("activation_checkpointing", True)),
        head_hidden_dim=int(ckpt.get("head_hidden_dim", 256)),
        head_dropout=float(ckpt.get("head_dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    preds = []
    truths = []
    spot_ids = []
    for batch in tqdm(loader, desc="spatial head eval", unit="batch"):
        gene_ids, x_visible, mask_missing, target_idx_b, y_target, log_total_visible, spatial_xy_b, spot_idx = batch
        gene_ids = gene_ids.to(device, non_blocking=True)
        x_visible = x_visible.to(device, non_blocking=True)
        mask_missing = mask_missing.to(device, non_blocking=True)
        target_idx_b = target_idx_b.to(device, non_blocking=True)
        log_total_visible = log_total_visible.to(device, non_blocking=True)
        spatial_xy_b = spatial_xy_b.to(device, non_blocking=True)

        pred = model(
            gene_ids=gene_ids,
            x_visible=x_visible,
            mask_missing=mask_missing,
            target_idx=target_idx_b,
            log_total_visible=log_total_visible,
            spatial_xy=spatial_xy_b,
        )
        preds.append(pred.detach().float().cpu().numpy())
        truths.append(y_target.detach().float().cpu().numpy())
        spot_ids.extend(spatial.obs_names[np.asarray(spot_idx, dtype=np.int64)].astype(str).tolist())

    pred_np = np.concatenate(preds, axis=0)
    truth_np = np.concatenate(truths, axis=0)
    metrics = summarize_target_metrics(pred_np, truth_np)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **metrics,
        "task_ckpt": str(task_ckpt),
        "exp_dir": str(ckpt.get("exp_dir", "")),
        "base_checkpoint": str(ckpt.get("base_checkpoint", "")),
        "config": str(cfg_path),
        "ref_h5ad": str(ref_h5ad),
        "visium_root": str(visium_root),
        "sample_ids": sample_ids if sample_ids else "all",
        "split": str(args.split),
        "n_spots": int(selected_idx.shape[0]),
        "n_genes_total": int(ref.ensembl_ids.shape[0]),
        "n_observed_genes": int(observed_idx.shape[0]),
        "n_target_genes": int(target_idx.shape[0]),
        "expression_space": expression_space,
        "evaluation_mode": "task_head_spatial_imputation",
    }
    save_json(output_dir / "summary.json", summary)

    if args.save_arrays:
        np.save(output_dir / "pred_targets.npy", pred_np)
        np.save(output_dir / "truth_targets.npy", truth_np)

    if args.save_h5ad:
        selected_spatial = spatial[selected_idx].copy()
        pred_adata = build_target_prediction_adata(
            spatial=selected_spatial,
            ref=ref,
            target_idx=target_idx,
            pred_targets=pred_np,
            truth_targets=truth_np,
            expression_space=expression_space,
        )
        pred_adata.write_h5ad(output_dir / "scuniflow_spatial_head_predicted_target_genes.h5ad")


def json_load(path: Path):
    with path.open() as f:
        return json.load(f)


if __name__ == "__main__":
    main()
