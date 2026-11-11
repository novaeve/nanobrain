#!/usr/bin/env python3
"""Train and evaluate a shared MLP four-task deep-learning baseline.

This is a deliberately simple control for the nanoBrain pilot: one shared
feed-forward representation and independent annotation, integration, spatial
niche, development, and masked-completion heads.  It uses the same fixed
NumPy subset and validation evaluator as the classical and nanoBrain runners.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.data import infer_label_cardinalities, load_v3_subset, verify_subset_manifest  # noqa: E402
from nanobrain.model import NanoBrainModelConfig  # noqa: E402
from scripts.run_nanobrain_baselines import (  # noqa: E402
    _append_group_classification_rows,
    _array,
    _coords,
    _labels,
    _metric_rows,
    _safe_name,
    _write_npz,
)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(max(1, int(depth)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.input(x)
        for block in self.blocks:
            state = state + block(state)
        return state


class DeepBaseline(nn.Module):
    def __init__(self, input_dim: int, annotation_classes: int, niche_classes: int, integration_classes: int,
                 hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        self.encoder = ResidualMLP(input_dim, hidden_dim, depth, dropout)
        self.annotation = nn.Linear(hidden_dim, annotation_classes)
        self.niche = nn.Linear(hidden_dim, niche_classes)
        self.integration = nn.Linear(hidden_dim, integration_classes)
        self.development = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.completion = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encoder(x)
        return {
            "embedding": state,
            "annotation": self.annotation(state),
            "niche": self.niche(state),
            "integration": self.integration(state),
            "development": self.development(state),
            "completion": self.completion(state),
        }


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & target.ge(0)
    return F.cross_entropy(logits[valid], target[valid].long()) if bool(valid.any()) else logits.sum() * 0.0


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    while valid.ndim < pred.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid & torch.isfinite(target)
    return F.mse_loss(pred[valid], target[valid]) if bool(valid.any()) else pred.sum() * 0.0


def _batch(dataset, indices: np.ndarray, device: torch.device, *, completion_input: bool = False):
    index = torch.from_numpy(indices.astype(np.int64, copy=False))
    x = dataset.features[index].to(device=device, dtype=torch.float32)
    completion_mask = dataset.completion_mask[index].to(device=device) if dataset.completion_mask is not None else None
    if completion_input and completion_mask is not None:
        x = x.masked_fill(completion_mask, 0.0)
    labels = {key: value[index].to(device=device) for key, value in dataset.labels.items()}
    masks = {key: value[index].to(device=device).bool() for key, value in dataset.masks.items()}
    target = dataset.completion_target[index].to(device=device, dtype=torch.float32) if dataset.completion_target is not None else x
    return x, labels, masks, target, completion_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "assets" / "nanobrain_v3_100k")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "nanobrain" / "deep-baseline-20260915")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--validation-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260940)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.validation_batch_size, args.hidden_dim, args.depth) < 1:
        parser.error("steps, batch sizes, hidden dimension and depth must be positive")
    _seed(args.seed)
    manifest = verify_subset_manifest(
        args.data,
        verify_hashes=False,
        data_splits=("train", "validation"),
    )
    splits = load_v3_subset(args.data, verify=False, include_test=False)
    cardinalities = infer_label_cardinalities(splits)
    requested_device = str(args.device).lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = DeepBaseline(splits.train.input_dim, cardinalities["annotation"], cardinalities["niche"], cardinalities["integration"], args.hidden_dim, args.depth, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    order_rng = np.random.default_rng(args.seed)
    losses_log: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for step in range(args.steps):
        indices = order_rng.integers(0, splits.train.size, size=args.batch_size, dtype=np.int64)
        x, labels, masks, completion_target, completion_mask = _batch(splits.train, indices, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        terms = {
            "annotation": _masked_ce(output["annotation"], labels["annotation"], masks["annotation"]),
            "niche": _masked_ce(output["niche"], labels["niche"], masks["niche"]),
            "integration": _masked_ce(output["integration"], labels["integration"], masks["integration"]),
            "development": _masked_mse(output["development"], labels["development"].float(), masks["development"]),
        }
        # The completion branch receives an explicitly masked input, so the
        # target expression value cannot be copied through the input.
        if completion_mask is not None:
            completion_x, _, _, _, _ = _batch(splits.train, indices, device, completion_input=True)
            completion_output = model(completion_x)["completion"]
            terms["completion"] = _masked_mse(completion_output, completion_target, completion_mask)
        else:
            terms["completion"] = output["completion"].sum() * 0.0
        total = sum(terms.values()) / len(terms)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step + 1 == args.steps or (step + 1) % max(1, args.steps // 10) == 0:
            losses_log.append({"step": step + 1, **{key: float(value.detach().cpu()) for key, value in terms.items()}, "total": float(total.detach().cpu())})
    train_runtime = time.perf_counter() - started

    # Gather validation predictions in bounded batches.
    model.eval()
    collected = {key: [] for key in ("embedding", "annotation", "niche", "integration", "development", "completion")}
    with torch.no_grad():
        for start in range(0, splits.validation.size, args.validation_batch_size):
            stop = min(splits.validation.size, start + args.validation_batch_size)
            indices = np.arange(start, stop, dtype=np.int64)
            x, _, _, _, completion_mask = _batch(splits.validation, indices, device)
            output = model(x)
            for key in collected:
                collected[key].append(output[key].detach().cpu().numpy())
    outputs = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    output_dir = args.output.resolve()
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    method = "deep_mlp_shared"
    notes = f"shared residual MLP hidden={args.hidden_dim}, depth={args.depth}, steps={args.steps}"
    eval_set = splits.validation
    eval_size = eval_set.size

    ann, ann_valid = _labels(eval_set, "annotation")
    ann_indices = np.flatnonzero(ann_valid)
    ann_logits = outputs["annotation"][ann_indices]
    ann_logits = ann_logits - ann_logits.max(axis=1, keepdims=True)
    ann_prob = np.exp(ann_logits); ann_prob /= np.maximum(ann_prob.sum(axis=1, keepdims=True), 1e-12)
    artifact = pred_dir / f"{method}__annotation.npz"
    _write_npz(artifact, y_true=ann[ann_indices].astype(np.int64), y_pred=ann_prob.argmax(axis=1), y_prob=ann_prob.astype(np.float32), classes=np.arange(ann_prob.shape[1]), eval_indices=ann_indices)
    rows.extend(_metric_rows(task="annotation", artifact=artifact, method=method, family="deep_learning", regime="scratch",
                             fit_rows=splits.train.size, eval_rows=len(ann_indices), runtime=train_runtime, notes=notes))
    _append_group_classification_rows(rows, artifact=artifact, method=method, family="deep_learning", regime="scratch", fit_rows=splits.train.size, runtime=train_runtime, notes=notes, dataset=eval_set)

    niche, niche_valid = _labels(eval_set, "niche")
    niche_indices = np.flatnonzero(niche_valid & np.isfinite(_coords(eval_set)[:, :2]).all(axis=1))
    artifact = pred_dir / f"{method}__spatial_niche.npz"
    _write_npz(artifact, embedding=outputs["embedding"][niche_indices], coordinates=_coords(eval_set)[niche_indices, :2], labels=niche[niche_indices].astype(np.int64), clusters=outputs["niche"][niche_indices].argmax(axis=1).astype(np.int64), eval_indices=niche_indices)
    rows.extend(_metric_rows(task="spatial_niche", artifact=artifact, method=method, family="deep_learning", regime="scratch", fit_rows=splits.train.size, eval_rows=len(niche_indices), runtime=train_runtime, notes=notes))

    integration, integration_valid = _labels(eval_set, "integration")
    integration_indices = np.flatnonzero(integration_valid & ann_valid)
    artifact = pred_dir / f"{method}__integration.npz"
    _write_npz(artifact, embedding=outputs["embedding"][integration_indices], biology_label=ann[integration_indices].astype(np.int64), batch_label=integration[integration_indices].astype(np.int64), eval_indices=integration_indices)
    rows.extend(_metric_rows(task="integration", artifact=artifact, method=method, family="deep_learning", regime="scratch", fit_rows=splits.train.size, eval_rows=len(integration_indices), runtime=train_runtime, notes=notes))

    target = _array(eval_set.completion_target if eval_set.completion_target is not None else eval_set.features, np.float32)
    completion_mask = _array(eval_set.completion_mask).astype(bool, copy=False)
    completion_prediction: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, eval_size, args.validation_batch_size):
            stop = min(eval_size, start + args.validation_batch_size)
            indices = np.arange(start, stop, dtype=np.int64)
            completion_x, _, _, _, _ = _batch(eval_set, indices, device, completion_input=True)
            completion_prediction.append(model(completion_x)["completion"].detach().cpu().numpy())
    completion_prediction_array = np.concatenate(completion_prediction, axis=0)
    artifact = pred_dir / f"{method}__spatial_completion.npz"
    _write_npz(artifact, target=target, prediction=completion_prediction_array.astype(np.float32), mask=completion_mask)
    rows.extend(_metric_rows(task="spatial_completion", artifact=artifact, method=method, family="deep_learning", regime="scratch", fit_rows=splits.train.size, eval_rows=eval_size, runtime=train_runtime, notes=notes))

    development, development_valid = _labels(eval_set, "development")
    development_indices = np.flatnonzero(development_valid)
    artifact = pred_dir / f"{method}__development.npz"
    _write_npz(artifact, target=development[development_indices].astype(np.float32), prediction=outputs["development"][development_indices].astype(np.float32), eval_indices=development_indices)
    rows.extend(_metric_rows(task="development", artifact=artifact, method=method, family="deep_learning", regime="scratch", fit_rows=splits.train.size, eval_rows=len(development_indices), runtime=train_runtime, notes=notes))

    import csv
    fields = ["split_id", "subset", "task", "method", "method_family", "training_regime", "fit_rows", "eval_rows", "status", "runtime_seconds", "prediction_artifact", "metric", "value", "direction", "stratum_type", "stratum_value", "notes", "error"]
    with (output_dir / "benchmark_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": {"input_dim": splits.train.input_dim, **cardinalities, "hidden_dim": args.hidden_dim, "depth": args.depth, "dropout": args.dropout}, "train": vars(args), "losses": losses_log}, checkpoint)
    receipt = {
        "format": "nanobrain_deep_baseline_receipt_v1", "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3", "subset": "nanobrain_v3_100k",
        "data": str(args.data.resolve()), "fit_split": "train", "evaluation_split": "validation", "test_opened": False,
        "method": method, "family": "deep_learning", "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "train": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed, "runtime_seconds": train_runtime, "losses": losses_log},
        "validation_rows": {"annotation": int(len(ann_indices)), "integration": int(len(integration_indices)), "spatial_niche": int(len(niche_indices)), "spatial_completion": int(eval_size), "development": int(len(development_indices))},
        "metrics": str(output_dir / "benchmark_table.csv"), "checkpoint": str(checkpoint), "manifest_sha256": __import__("hashlib").sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "parameters": receipt["parameters"], "train_runtime_seconds": train_runtime, "metric_rows": len(rows), "tasks": sorted({row["task"] for row in rows})}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
