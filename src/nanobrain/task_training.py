"""Independent task objectives for the existing nanoBrain trainer.

Each invocation owns a fresh model and optimizer. Reconstruction inside the
integration objective preserves expression without using cell-type labels.
"""
from __future__ import annotations

from dataclasses import replace
import torch
import torch.nn.functional as F

from .contracts import LossBundle
from .model import _GradientReverse

TASKS = ("annotation", "integration", "spatial_niche", "development", "spatial_completion")
CORE_TASKS = TASKS[:4]


def task_input(batch, task):
    """Hide completion targets at both training and inference boundaries."""
    if task != "spatial_completion":
        return batch
    if batch.completion_mask is None:
        raise ValueError("completion requires a fixed held-out mask")
    visible = ~batch.completion_mask
    if batch.feature_observed is not None:
        visible = visible & batch.feature_observed
    context = batch.neighbor_features
    if context is not None and batch.neighbor_completion_mask is not None:
        context = context.masked_fill(batch.neighbor_completion_mask, 0)
    return replace(batch, features=batch.features.masked_fill(~visible, 0),
                   feature_observed=visible, neighbor_features=context)


def configure_task(model, task: str) -> list[str]:
    if task == "joint":
        if model.genome.architecture is not None:
            raise ValueError("graph protocol requires independent task training")
        return [name for name, _ in model.named_parameters()]
    if task not in TASKS:
        raise ValueError(f"unsupported training task: {task}")
    if task == "development" and model.genome.flow == "none":
        raise ValueError("independent development requires an active flow model")
    active = {"encoder", "attention", "spatial", "residual", "backbone"}
    active.update({
        "annotation": {"annotation_head"},
        "integration": {"integration_head", "completion_head"},
        "spatial_niche": {"niche_head"},
        "development": {"flow", "completion_head"},
        "spatial_completion": {"completion_head"},
    }[task])
    if model.genome.architecture is not None and task in {"integration", "development"}:
        active.discard("completion_head")
        active.add("integration_reconstruction_head" if task == "integration" else "development_head")
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split(".")[0] in active)
        if parameter.requires_grad:
            names.append(name)
    return names


def row_task_loss(model, batch, task: str) -> tuple[LossBundle, dict]:
    from .objectives import row_objective
    return row_objective(model, batch, task)


def evaluate_task(model, splits, task, batch_size, device, *, development_groups=32):
    """Training diagnostics only; promotion uses the canonical evaluators."""
    import numpy as np

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        if task == "development":
            pairs = splits.development_validation
            if pairs is None:
                raise ValueError("independent development requires population pairs")
            selected = np.unique(np.linspace(0, len(pairs) - 1, min(len(pairs), max(1, development_groups)), dtype=np.int64))
            for index in selected:
                batch = pairs.batch(int(index), max_cells=64, seed=20260915).to(device)
                predicted, _ = model.predict_development_population(batch.source, batch.source_time_days, batch.target_time_days)
                total += float(F.mse_loss(predicted.float().mean(0), batch.target.features.float().mean(0)).cpu())
                count += 1
            key = "development_population_mse"
        else:
            for batch in splits.validation.iter_batches(batch_size, shuffle=False):
                batch = batch.to(device)
                if task in {"annotation", "spatial_niche"}:
                    label = "annotation" if task == "annotation" else "niche"
                    head = model.annotation_head if task == "annotation" else model.niche_head
                    logits = head(model.encode(batch))
                    valid = batch.labels[label].ge(0) & batch.label_masks[label]
                    total += float((logits.argmax(-1)[valid] == batch.labels[label][valid]).sum().cpu())
                    count += int(valid.sum())
                else:
                    losses, _ = row_task_loss(model, batch, task)
                    total += float(losses.total.cpu()) * len(batch.features)
                    count += len(batch.features)
            key = {"annotation": "annotation_accuracy", "spatial_niche": "niche_accuracy",
                   "integration": "integration_objective", "spatial_completion": "completion_mse"}[task]
    if not count:
        raise ValueError(f"no valid validation observations for {task}")
    return {key: total / count}
