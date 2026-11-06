import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from .eval import evaluate_artifact_dir, save_prediction_artifact
from .protocol import build_scnode_inputs, load_protocol_bundle


def _resolve_device(device: str):
    requested_device = str(device)
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        resolved = torch.device(requested_device)
        torch.cuda.set_device(resolved)
        return resolved
    return torch.device("cpu")


def _import_mioflow_modules(scnode_root: str):
    scnode_path = Path(scnode_root).resolve()
    revised_root = scnode_path / "baseline" / "MIOFlow_model_revised"
    if not scnode_path.exists():
        raise FileNotFoundError(f"scNODE root not found: {scnode_path}")
    if str(scnode_path) not in sys.path:
        sys.path.insert(0, str(scnode_path))
    if str(revised_root) not in sys.path:
        sys.path.insert(0, str(revised_root))
    try:
        from baseline.MIOFlow_model_revised.running import makeSimulation, trainModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MIOFlow baseline requires its upstream dependencies in the current environment "
            f"(missing: {exc})."
        ) from exc
    return trainModel, makeSimulation


def run_mioflow_protocol(
    protocol_path: str,
    scnode_root: str,
    output_dir: str,
    assets_dir: str = None,
    device: str = "cuda",
    pca_dims: int = 50,
    n_epochs_emb: int = 1000,
    gae_embedded_dim: int = 50,
    encoder_layers=None,
    layers=None,
    batch_size: int = 100,
    n_local_epochs: int = 40,
    n_global_epochs: int = 40,
    n_post_local_epochs: int = 0,
    lambda_density: float = 35.0,
):
    print(f"[mioflow_route_a] protocol={protocol_path}")
    print(f"[mioflow_route_a] scnode_root={scnode_root}")
    resolved_device = _resolve_device(device)
    print(f"[mioflow_route_a] device={resolved_device}")
    trainModel, makeSimulation = _import_mioflow_modules(scnode_root)

    bundle = load_protocol_bundle(protocol_path)
    inputs = build_scnode_inputs(bundle)
    traj_data = inputs["traj_data"]
    train_tps = list(inputs["train_tps"])
    test_tps = list(inputs["test_tps"])
    all_tps = list(inputs["all_tps"])

    train_data = [traj_data[t] for t in train_tps]
    pca_model = PCA(n_components=int(pca_dims), svd_solver="arpack")
    data_pca = pca_model.fit_transform(np.concatenate(train_data, axis=0))
    time_df = pd.DataFrame(
        data=np.concatenate([np.repeat(train_tps[idx], x.shape[0]) for idx, x in enumerate(train_data)])[:, np.newaxis],
        columns=["samples"],
    )
    data_df = pd.DataFrame(data=data_pca)
    train_df = pd.concat([time_df, data_df], axis=1).reset_index(drop=True)

    encoder_layers = [50, 50, 50] if encoder_layers is None else list(encoder_layers)
    layers = [50, 50, 50] if layers is None else list(layers)

    print("[mioflow_route_a] training MIOFlow")
    model, *_rest, opts = trainModel(
        train_df,
        train_tps,
        test_tps,
        n_genes=int(pca_dims),
        n_epochs_emb=int(n_epochs_emb),
        samples_size_emb=(50,),
        gae_embedded_dim=int(gae_embedded_dim),
        encoder_layers=encoder_layers,
        layers=layers,
        batch_size=int(batch_size),
        n_local_epochs=int(n_local_epochs),
        n_global_epochs=int(n_global_epochs),
        n_post_local_epochs=int(n_post_local_epochs),
        lambda_density=float(lambda_density),
        pca_dims=int(pca_dims),
        use_cuda=(resolved_device.type == "cuda"),
    )

    n_cells_per_time = max(traj_data[0].shape[0], 512)
    print(f"[mioflow_route_a] predicting all timepoints with n_cells={n_cells_per_time}")
    generated = makeSimulation(
        train_df,
        model,
        all_tps,
        opts,
        n_sim_cells=int(n_cells_per_time),
        n_trajectories=100,
        n_bins=100,
    )
    forward_recon_traj = [pca_model.inverse_transform(generated[i, :, :]).astype(np.float32, copy=False) for i in range(generated.shape[0])]

    pred_by_time = {}
    real_by_time = {}
    obs_names_by_time = {}
    metadata_df = bundle["metadata_df"]
    for time_bin in tqdm(bundle["eval_time_bins"], desc="mioflow-export", leave=False):
        t_idx = bundle["ordered_time_bins"].index(time_bin)
        pred_by_time[time_bin] = np.asarray(forward_recon_traj[t_idx], dtype=np.float32)
        real_idx = bundle["time_to_indices"][time_bin]
        real_by_time[time_bin] = np.asarray(bundle["expression"][real_idx], dtype=np.float32)
        obs_names_by_time[time_bin] = metadata_df.iloc[real_idx]["obs_name"].astype(str).to_numpy()

    artifact = {
        "protocol_name": bundle["protocol_name"],
        "method_name": "MIOFlow",
        "ordered_time_bins": bundle["ordered_time_bins"],
        "heldout_time_bins": bundle["eval_time_bins"],
        "time_display_map": bundle.get("time_display_map", {}),
        "pred_by_time": pred_by_time,
        "real_by_time": real_by_time,
        "obs_names_by_time": obs_names_by_time,
    }
    print(f"[mioflow_route_a] saving artifact -> {output_dir}")
    save_prediction_artifact(output_dir, artifact)
    if assets_dir:
        print(f"[mioflow_route_a] evaluating artifact with assets -> {assets_dir}")
        evaluate_artifact_dir(
            artifact_dir=output_dir,
            assets_dir=assets_dir,
            metadata_df=bundle["metadata_df"],
            hvg_gene_names=bundle["bundle_meta"]["hvg_gene_names"],
        )
    print("[mioflow_route_a] done")
    return output_dir
