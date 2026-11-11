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
    if task == "spatial_niche" and batch.labels["niche"].ndim == 3:
        # Nicheformer-style benchmark: center expression predicts unseen
        # neighbor composition. No neighbor expression or target labels enter.
        return replace(batch, neighbors=None, neighbor_mask=None, neighbor_features=None,
                       neighbor_completion_mask=None, coordinates=None)
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
        "spatial_niche": {"niche_head", "niche_density_head"},
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
    """Select checkpoints with the same canonical metric used for final scoring."""
    import numpy as np
    from brainwm_benchmark.metrics import annotation, development, integration, niche_composition
    from brainwm_benchmark.protocol import DEVELOPMENT_EVAL_CELLS, EVALUATION_SEED

    def numpy(value):
        return value.detach().float().cpu().numpy()

    model.eval()
    total, count = 0.0, 0
    truths, predictions, embeddings, batch_labels = [], [], [], []
    pointer = [0]
    niche_values = {k: [] for k in ("target_composition", "prediction_composition", "composition_mask",
                                   "target_density", "prediction_density", "density_mask")}
    with torch.no_grad():
        if task == "development":
            pairs = splits.development_validation
            if pairs is None:
                raise ValueError("independent development requires population pairs")
            selected = (np.arange(len(pairs)) if development_groups <= 0 else
                        np.unique(np.linspace(0, len(pairs) - 1, min(len(pairs), development_groups), dtype=np.int64)))
            for index in selected:
                batch = pairs.batch(int(index), max_cells=DEVELOPMENT_EVAL_CELLS, seed=EVALUATION_SEED).to(device)
                predicted, _ = model.predict_development_population(batch.source, batch.source_time_days, batch.target_time_days)
                truths.append(numpy(batch.target.features))
                predictions.append(numpy(predicted))
                pointer.append(pointer[-1] + len(truths[-1]))
            if not truths:
                raise ValueError("no validation development populations")
            scores = development({"target": np.concatenate(truths), "prediction": np.concatenate(predictions),
                                  "group_ptr": np.asarray(pointer)})
            return {f"development.{key}": value for key, value in scores.items()}
        else:
            for batch in splits.validation.iter_batches(batch_size, shuffle=False):
                batch = batch.to(device)
                if task == "spatial_niche" and model.config.niche_mode == "composition":
                    composition, density = model.predict_niche(model.encode(task_input(batch, task)))
                    for key, value in {"target_composition": batch.labels["niche"], "prediction_composition": composition,
                                       "composition_mask": batch.label_masks["niche"], "target_density": batch.labels["niche_density"],
                                       "prediction_density": density, "density_mask": batch.label_masks["niche_density"]}.items():
                        niche_values[key].append(numpy(value))
                    continue
                if task in {"annotation", "spatial_niche"}:
                    label = "annotation" if task == "annotation" else "niche"
                    head = model.annotation_head if task == "annotation" else model.niche_head
                    logits = head(model.encode(batch))
                    valid = batch.labels[label].ge(0) & batch.label_masks[label]
                    if task == "spatial_niche":
                        valid &= torch.isfinite(batch.coordinates[:, :2]).all(dim=1)
                    truths.append(batch.labels[label][valid].detach().cpu().numpy())
                    predictions.append(logits.argmax(-1)[valid].detach().cpu().numpy())
                elif task == "integration":
                    valid = (batch.label_masks["annotation"] & batch.labels["annotation"].ge(0)
                             & batch.label_masks["integration"] & batch.labels["integration"].ge(0))
                    embeddings.append(numpy(model.encode(batch))[valid.cpu().numpy()])
                    truths.append(batch.labels["annotation"][valid].detach().cpu().numpy())
                    batch_labels.append(batch.labels["integration"][valid].detach().cpu().numpy())
                else:
                    losses, _ = row_task_loss(model, batch, task)
                    total += float(losses.total.cpu()) * len(batch.features)
                    count += len(batch.features)
            if task in {"annotation", "spatial_niche"}:
                if niche_values["target_composition"]:
                    payload = {key: np.concatenate(value) for key, value in niche_values.items()}
                    if "sample" in (splits.validation.metadata or {}):
                        payload["section_id"] = splits.validation.metadata["sample"]
                    scores = niche_composition(payload)
                    return {f"spatial_niche.{key}": value for key, value in scores.items()}
                y = np.concatenate(truths)
                if not len(y):
                    raise ValueError(f"no valid validation observations for {task}")
                scores = annotation({"y_true": y, "y_pred": np.concatenate(predictions)})
                return {f"{task}.{key}": value for key, value in scores.items()}
            if task == "integration":
                scores = integration({"embedding": np.concatenate(embeddings),
                                      "biology_label": np.concatenate(truths),
                                      "batch_label": np.concatenate(batch_labels)}, diagnostics=False)
                return {f"integration.{key}": value for key, value in scores.items()}
            key = "completion_mse"
    if not count:
        raise ValueError(f"no valid validation observations for {task}")
    return {key: total / count}
