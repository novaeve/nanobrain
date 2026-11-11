#!/usr/bin/env python3
"""Run one real CUDA update with every BrainWM joint objective active."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainwm.development_numpy_loader import DevelopmentPairLoader
from brainwm.multitask_objective import BRAINWM_LOSS_NAMES, MultiObjectiveBalancer, brainwm_task_losses
from brainwm.numpy_shard_loader import NumpyShardDataset
from brainwm.sparse_objective import mask_expression
from brainwm_foundation import BrainWMFoundationMoE
from brainwm_foundation.backends import backend_spec, create_backend
from brainwm.local_assets import gene_vocabulary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--spatial-shard", type=int, default=111)
    parser.add_argument("--max-padded-tokens", type=int, default=60000)
    parser.add_argument("--moe-backend", choices=("easy_deepspeed", "mage_beta"), default="easy_deepspeed")
    parser.add_argument("--optimizer-backend", default="adamw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.optimizer_backend = backend_spec("optimizer", args.optimizer_backend).name
    if not torch.cuda.is_available():
        raise RuntimeError("joint objective smoke requires CUDA")
    torch.manual_seed(20260911)
    device = torch.device("cuda")
    manifest = json.loads((args.benchmark_dir / "manifest.json").read_text())
    vocab_size = len(pd.read_parquet(gene_vocabulary_path(args.benchmark_dir)))
    vocab = json.loads((args.shard_dir / "joint_metadata_vocab.json").read_text())
    categorical = (
        "dataset", "assay", "technology", "donor", "life_stage", "sex",
        "disease_status", "time_status", "split_group", "split_unit", "section",
        "sample", "cl_id", "cl_term", "original_cell_type", "original_subclass",
        "uberon_id", "uberon_term", "original_anatomical_region", "cell_type", "anatomy",
    )
    dimensions = {name: len(vocab[name]) for name in categorical}
    dimensions["development"] = 4
    model = BrainWMFoundationMoE(
        vocab_size, 64, 256, 8, 12, top_k=1, attention_heads=4, modality_dims=dimensions,
        annotation_classes=len(vocab["cell_type"]),
        anatomy_classes=len(vocab["anatomy"]), categorical_modalities=set(categorical),
        moe_backend=args.moe_backend,
    ).to(device).train()
    dataset = NumpyShardDataset(
        args.shard_dir, "train", batch_size=32, max_padded_tokens=args.max_padded_tokens,
    )
    gene_path = args.shard_dir / f"train-shard-{args.spatial_shard:05d}-gene_index.npy"
    if not gene_path.exists():
        raise FileNotFoundError(gene_path)
    dataset.files = [gene_path]
    batch = next(item for item in dataset if item.neighbor_mask is not None and item.neighbor_mask.any())
    batch = mask_expression(batch).to(device)
    batch.modality_masked = {
        name: observed.bool().clone() for name, observed in (batch.modality_observed or {}).items()
    }
    source, target, t0, t1 = next(iter(DevelopmentPairLoader(
        args.shard_dir, max_padded_tokens=args.max_padded_tokens,
    )))
    source, target = source.to(device), target.to(device)
    output = model({"joint_batch": batch, "development": (source, target, t0, t1)})
    losses = brainwm_task_losses(output, batch)
    velocity_loss = torch.nn.functional.mse_loss(output["flow_velocity"], output["flow_target"])
    endpoint_loss = torch.nn.functional.mse_loss(output["flow_endpoint"], output["flow_endpoint_target"])
    losses["flow"] = 0.5 * (velocity_loss + endpoint_loss)
    missing = [name for name in BRAINWM_LOSS_NAMES if losses.get(name) is None]
    if missing:
        raise RuntimeError("inactive joint losses: " + ", ".join(missing))
    balancer = MultiObjectiveBalancer(BRAINWM_LOSS_NAMES, mode="ema_equal").to(device)
    total, telemetry = balancer(losses)
    optimizer = create_backend(
        "optimizer", args.optimizer_backend, model.parameters(), lr=2e-4,
    )
    tracked_name, tracked_parameter = next(
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    before = tracked_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradient_parameters = sum(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    changed = not torch.equal(before, tracked_parameter.detach())
    with torch.no_grad():
        source_state = model.forward_sparse(source)["cell_embedding"]
        rollout, trajectory = model.predict_development(
            source_state, source_state, t0, t1, steps=4, solver="heun",
            return_trajectory=True,
        )
        rollout_finite = bool(torch.isfinite(rollout).all())
    receipt = {
        "format": "brainwm_joint_objective_smoke_v1",
        "split_id": manifest["split_id"],
        "moe_backend": args.moe_backend,
        "optimizer_backend": args.optimizer_backend,
        "spatial_shard": args.spatial_shard,
        "cells": batch.batch_size,
        "tokens": int(batch.token_mask.sum()),
        "development_cells_per_side": len(source.row_ptr) - 1,
        "losses": {name: float(losses[name].detach()) for name in BRAINWM_LOSS_NAMES},
        "flow_velocity_loss": float(velocity_loss.detach()),
        "flow_endpoint_loss": float(endpoint_loss.detach()),
        "balanced_total": float(total.detach()),
        "parameters_with_finite_gradients": int(gradient_parameters),
        "tracked_parameter": tracked_name,
        "adamw_changed_tracked_parameter": changed,
        "development_rollout": {
            "solver": "heun", "steps": 4,
            "trajectory_states": len(trajectory), "finite": rollout_finite,
        },
        "passed": bool(torch.isfinite(total) and changed and gradient_parameters > 0 and rollout_finite),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
