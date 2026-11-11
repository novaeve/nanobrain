import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .eval import evaluate_artifact_dir, save_prediction_artifact
from .protocol import build_scnode_inputs, load_protocol_bundle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _resolve_device(device: str):
    requested_device = str(device)
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        resolved = torch.device(requested_device)
        torch.cuda.set_device(resolved)
        return resolved
    return torch.device("cpu")


def _import_prescient_modules(scnode_root: str):
    scnode_path = Path(scnode_root).resolve()
    baseline_root = scnode_path / "baseline"
    prescient_pkg_root = scnode_path / "baseline" / "prescient_model"
    if not scnode_path.exists():
        raise FileNotFoundError(f"scNODE root not found: {scnode_path}")
    if str(scnode_path) not in sys.path:
        sys.path.insert(0, str(scnode_path))
    if not baseline_root.exists():
        raise FileNotFoundError(f"baseline root not found: {baseline_root}")
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    if not prescient_pkg_root.exists():
        raise FileNotFoundError(f"PRESCIENT package root not found: {prescient_pkg_root}")
    if str(prescient_pkg_root) not in sys.path:
        sys.path.insert(0, str(prescient_pkg_root))
    try:
        from baseline.prescient_model.running import create_simulate_parser, create_train_parser, makeSimulation, trainModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PRESCIENT baseline requires its upstream dependencies in the current environment "
            f"(missing: {exc})."
        ) from exc
    return create_train_parser, trainModel, create_simulate_parser, makeSimulation


def _prepare_prescient_data(expr_mat: np.ndarray, tps: np.ndarray, genes, num_pcs: int):
    scaler = StandardScaler()
    pca = PCA(n_components=int(num_pcs))
    x = scaler.fit_transform(expr_mat)
    xp = pca.fit_transform(x)
    y = list(np.sort(np.unique(tps)).astype(int))
    x_by_time = [torch.from_numpy(x[(tps == d), :]).float() for d in y]
    xp_by_time = [torch.from_numpy(xp[(tps == d), :]).float() for d in y]
    xu_by_time = [torch.zeros((xp_part.shape[0], 2), dtype=torch.float32) for xp_part in xp_by_time]
    growth_rate = [np.ones((each.shape[0],), dtype=np.float32) for each in x_by_time]
    data_dict = {
        "data": expr_mat,
        "genes": genes,
        "celltype": np.repeat("NAN", expr_mat.shape[0]),
        "tps": tps,
        "x": x_by_time,
        "xp": xp_by_time,
        "xu": xu_by_time,
        "y": y,
        "pca": pca,
        "um": None,
        "w": growth_rate,
    }
    return data_dict, scaler, pca


def run_prescient_protocol(
    protocol_path: str,
    scnode_root: str,
    output_dir: str,
    assets_dir: str = None,
    device: str = "cuda",
    k_dim: int = 100,
    layers: int = 2,
    train_epochs: int = 2000,
    train_lr: float = 1e-3,
    train_sd: float = 0.5,
    train_tau: float = 1e-6,
    train_clip: float = 0.25,
    pretrain_epochs: int = 5,
    num_steps: int = None,
):
    print(f"[prescient_route_a] protocol={protocol_path}")
    print(f"[prescient_route_a] scnode_root={scnode_root}")
    resolved_device = _resolve_device(device)
    print(f"[prescient_route_a] device={resolved_device}")
    create_train_parser, trainModel, create_simulate_parser, makeSimulation = _import_prescient_modules(scnode_root)

    bundle = load_protocol_bundle(protocol_path)
    inputs = build_scnode_inputs(bundle)
    traj_data = inputs["traj_data"]
    train_tps = list(inputs["train_tps"])
    all_tps = list(inputs["all_tps"])

    train_expr = np.concatenate([traj_data[t] for t in train_tps], axis=0)
    train_tp_labels = np.concatenate(
        [np.repeat(train_tps[idx], traj_data[t].shape[0]) for idx, t in enumerate(train_tps)],
        axis=0,
    ).astype(int)
    genes = np.asarray(bundle["bundle_meta"]["hvg_gene_names"]).astype(str)

    print("[prescient_route_a] preparing scaler/PCA data")
    data_dict, scaler, pca = _prepare_prescient_data(
        train_expr,
        train_tp_labels,
        genes,
        num_pcs=int(k_dim),
    )

    data_dict["y"] = list(all_tps)
    data_dict["x"] = [data_dict["x"][train_tps.index(t)] if t in train_tps else [] for t in all_tps]
    data_dict["xp"] = [data_dict["xp"][train_tps.index(t)] if t in train_tps else [] for t in all_tps]
    data_dict["w"] = [data_dict["w"][train_tps.index(t)] if t in train_tps else [] for t in all_tps]

    print("[prescient_route_a] training PRESCIENT")
    train_args = create_train_parser().parse_args([])
    train_args.data_path = bundle["protocol_name"]
    train_args.out_dir = str(Path(output_dir).resolve().parent / "prescient_train_cache")
    train_args.k_dim = int(k_dim)
    train_args.data_pt = data_dict
    train_args.train_t = train_tps[1:]
    train_args.timestamp = bundle["protocol_name"]
    train_args.layers = int(layers)
    train_args.train_epochs = int(train_epochs)
    train_args.train_lr = float(train_lr)
    train_args.train_sd = float(train_sd)
    train_args.train_tau = float(train_tau)
    train_args.train_clip = float(train_clip)
    train_args.pretrain_epochs = int(pretrain_epochs)
    train_args.gpu = 0 if resolved_device.type == "cuda" else -1
    model, best_state_dict, config, _ = trainModel(train_args)

    n_cells_per_time = max(traj_data[0].shape[0], 512)
    total_steps = int(num_steps) if num_steps is not None else (len(all_tps) - 1) * 10
    print(f"[prescient_route_a] simulating with n_cells={n_cells_per_time} num_steps={total_steps}")
    sim_args = create_simulate_parser().parse_args([])
    sim_args.data_pt = data_dict
    sim_args.data_path = bundle["protocol_name"]
    sim_args.best_model_state = best_state_dict
    sim_args.num_cells = int(n_cells_per_time)
    sim_args.num_steps = int(total_steps)
    sim_args.gpu = 0 if resolved_device.type == "cuda" else None
    sim_data = makeSimulation(sim_args, config)

    pred_by_time = {}
    real_by_time = {}
    obs_names_by_time = {}
    metadata_df = bundle["metadata_df"]
    for time_bin in tqdm(bundle["eval_time_bins"], desc="prescient-export", leave=False):
        t_idx = bundle["ordered_time_bins"].index(time_bin)
        latent = np.asarray(sim_data[int(t_idx * 10)], dtype=np.float32)
        pred_by_time[time_bin] = scaler.inverse_transform(pca.inverse_transform(latent)).astype(np.float32, copy=False)
        real_idx = bundle["time_to_indices"][time_bin]
        real_by_time[time_bin] = np.asarray(bundle["expression"][real_idx], dtype=np.float32)
        obs_names_by_time[time_bin] = metadata_df.iloc[real_idx]["obs_name"].astype(str).to_numpy()

    artifact = {
        "protocol_name": bundle["protocol_name"],
        "method_name": "PRESCIENT",
        "ordered_time_bins": bundle["ordered_time_bins"],
        "heldout_time_bins": bundle["eval_time_bins"],
        "time_display_map": bundle.get("time_display_map", {}),
        "pred_by_time": pred_by_time,
        "real_by_time": real_by_time,
        "obs_names_by_time": obs_names_by_time,
    }
    print(f"[prescient_route_a] saving artifact -> {output_dir}")
    save_prediction_artifact(output_dir, artifact)
    if assets_dir:
        print(f"[prescient_route_a] evaluating artifact with assets -> {assets_dir}")
        evaluate_artifact_dir(
            artifact_dir=output_dir,
            assets_dir=assets_dir,
            metadata_df=bundle["metadata_df"],
            hvg_gene_names=bundle["bundle_meta"]["hvg_gene_names"],
        )
    print("[prescient_route_a] done")
    return output_dir
