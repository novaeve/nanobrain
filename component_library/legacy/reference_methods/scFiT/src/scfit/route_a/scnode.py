import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .eval import evaluate_artifact_dir, save_prediction_artifact
from .protocol import build_scnode_inputs, load_protocol_bundle


def _import_scnode_modules(scnode_root: str):
    scnode_path = Path(scnode_root).resolve()
    if not scnode_path.exists():
        raise FileNotFoundError(f"scNODE root not found: {scnode_path}")
    if str(scnode_path) not in sys.path:
        sys.path.insert(0, str(scnode_path))
    existing_model = sys.modules.get("model")
    removed_model = None
    if existing_model is not None:
        model_file = getattr(existing_model, "__file__", "") or ""
        if Path(model_file).resolve() == (Path(__file__).resolve().parents[1] / "model.py"):
            removed_model = existing_model
            del sys.modules["model"]
    try:
        from optim.running import constructscNODEModel, scNODEPredict, scNODETrainWithPreTrain
    except ModuleNotFoundError as exc:
        if removed_model is not None:
            sys.modules["model"] = removed_model
        if "torchdiffeq" in str(exc):
            raise ModuleNotFoundError(
                "scNODE requires `torchdiffeq`, which is not currently installed in this environment."
            ) from exc
        raise
    if removed_model is not None:
        sys.modules["model"] = removed_model
    return constructscNODEModel, scNODEPredict, scNODETrainWithPreTrain


def run_scnode_protocol(
    protocol_path: str,
    scnode_root: str,
    output_dir: str,
    assets_dir: str = None,
    device: str = "cuda",
    latent_dim: int = 50,
    drift_latent_size=None,
    enc_latent_list=None,
    dec_latent_list=None,
    latent_coeff: float = 1.0,
    epochs: int = 10,
    iters: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    pretrain_iters: int = 200,
    pretrain_lr: float = 1e-3,
):
    print(f"[scnode_route_a] protocol={protocol_path}")
    print(f"[scnode_route_a] scnode_root={scnode_root}")
    requested_device = str(device)
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        resolved_device = torch.device(requested_device)
    else:
        resolved_device = torch.device("cpu")
    print(f"[scnode_route_a] device={resolved_device}")
    constructscNODEModel, scNODEPredict, scNODETrainWithPreTrain = _import_scnode_modules(scnode_root)
    bundle = load_protocol_bundle(protocol_path)
    scnode_inputs = build_scnode_inputs(bundle)
    print(
        f"[scnode_route_a] timepoints total={len(scnode_inputs['all_tps'])} "
        f"train={len(scnode_inputs['train_tps'])} test={len(scnode_inputs['test_tps'])}"
    )
    traj_data = [torch.as_tensor(x, dtype=torch.float32, device=resolved_device) for x in scnode_inputs["traj_data"]]
    train_tps = torch.as_tensor(scnode_inputs["train_tps"], dtype=torch.float32, device=resolved_device)
    all_tps = torch.as_tensor(scnode_inputs["all_tps"], dtype=torch.float32, device=resolved_device)

    print("[scnode_route_a] constructing scNODE model")
    model = constructscNODEModel(
        n_genes=traj_data[0].shape[1],
        latent_dim=int(latent_dim),
        enc_latent_list=enc_latent_list,
        dec_latent_list=dec_latent_list,
        drift_latent_size=[50, 50] if drift_latent_size is None else drift_latent_size,
        latent_enc_act="none",
        latent_dec_act="relu",
        drift_act="relu",
        ode_method="euler",
    )
    model = model.to(resolved_device)
    print("[scnode_route_a] training scNODE")
    model, _, _, _, _ = scNODETrainWithPreTrain(
        train_data=[traj_data[t] for t in scnode_inputs["train_tps"]],
        train_tps=train_tps,
        latent_ode_model=model,
        latent_coeff=float(latent_coeff),
        epochs=int(epochs),
        iters=int(iters),
        batch_size=int(batch_size),
        lr=float(lr),
        pretrain_iters=int(pretrain_iters),
        pretrain_lr=float(pretrain_lr),
    )
    n_cells_per_time = max(traj_data[0].shape[0], 512)
    print(f"[scnode_route_a] predicting all timepoints with n_cells={n_cells_per_time}")
    all_pred = scNODEPredict(model, traj_data[0], all_tps, n_cells=n_cells_per_time)

    pred_by_time = {}
    real_by_time = {}
    obs_names_by_time = {}
    metadata_df = bundle["metadata_df"]
    for time_bin in tqdm(bundle["eval_time_bins"], desc="scnode-export", leave=False):
        t_idx = bundle["ordered_time_bins"].index(time_bin)
        pred_by_time[time_bin] = np.asarray(all_pred[:, t_idx, :], dtype=np.float32)
        real_idx = bundle["time_to_indices"][time_bin]
        real_by_time[time_bin] = np.asarray(bundle["expression"][real_idx], dtype=np.float32)
        obs_names_by_time[time_bin] = metadata_df.iloc[real_idx]["obs_name"].astype(str).to_numpy()

    artifact = {
        "protocol_name": bundle["protocol_name"],
        "method_name": "scNODE",
        "ordered_time_bins": bundle["ordered_time_bins"],
        "heldout_time_bins": bundle["eval_time_bins"],
        "time_display_map": bundle.get("time_display_map", {}),
        "pred_by_time": pred_by_time,
        "real_by_time": real_by_time,
        "obs_names_by_time": obs_names_by_time,
    }
    print(f"[scnode_route_a] saving artifact -> {output_dir}")
    save_prediction_artifact(output_dir, artifact)
    if assets_dir:
        print(f"[scnode_route_a] evaluating artifact with assets -> {assets_dir}")
        evaluate_artifact_dir(
            artifact_dir=output_dir,
            assets_dir=assets_dir,
            metadata_df=bundle["metadata_df"],
            hvg_gene_names=bundle["bundle_meta"]["hvg_gene_names"],
        )
    print("[scnode_route_a] done")
    return output_dir
