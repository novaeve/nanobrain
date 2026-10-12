"""Source-fitted late-fusion residual for cross-species annotation.

This module deliberately keeps the Stage-0 checkpoint untouched.  A source
fold fits one prototype bank for a transparent expression view and one for
the frozen Stage-0 representation.  Their standardized class-score vectors
are combined as a residual score.  The operation is therefore label-aware at
the evaluator boundary, rather than an in-place mutation of gene embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from .crossspecies_annotation import (
    CENTROID_MODES,
    fit_grouped_centroids,
    predict_grouped_centroids,
    torch_log_cpm_l2,
)

RESIDUAL_ALPHAS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3)


def _check_scores(values: Tensor, name: str) -> Tensor:
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError(f"{name} must be a floating-point matrix")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def standardize_class_scores(scores: Tensor) -> Tensor:
    """Standardize class scores independently for every query cell."""
    scores = _check_scores(scores, "scores")
    if scores.shape[1] < 2:
        return scores - scores.mean(dim=1, keepdim=True)
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1.0e-5)


def fit_residual_prototypes(
    expression_train: Tensor,
    stage0_train: Tensor,
    expression_query: Tensor,
    stage0_query: Tensor,
    label_codes: Tensor,
    group_codes: Tensor,
    n_classes: int,
    mode: str = "cell_mean",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fit expression/Stage0 prototype banks and return train/query scores.

    The returned score matrices have columns in the same integer class order.
    Prototype fitting is entirely on the input CUDA device.
    """
    if mode not in CENTROID_MODES:
        raise ValueError(f"unknown centroid mode: {mode}")
    for value, name in (
        (expression_train, "expression_train"),
        (stage0_train, "stage0_train"),
        (expression_query, "expression_query"),
        (stage0_query, "stage0_query"),
    ):
        _check_scores(value, name)
        if not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    if expression_train.device != stage0_train.device:
        raise ValueError("expression and Stage0 train devices differ")
    if expression_query.device != stage0_query.device:
        raise ValueError("expression and Stage0 query devices differ")
    if expression_train.shape[0] != stage0_train.shape[0]:
        raise ValueError("train rows do not align")
    if expression_query.shape[0] != stage0_query.shape[0]:
        raise ValueError("query rows do not align")
    expression_centroids = fit_grouped_centroids(
        expression_train, label_codes, group_codes, n_classes, mode
    )
    stage_centroids = fit_grouped_centroids(
        stage0_train, label_codes, group_codes, n_classes, mode
    )
    expression_train_scores = torch.nn.functional.normalize(
        expression_train, dim=1, eps=1.0e-8
    ) @ expression_centroids.T
    expression_query_scores = torch.nn.functional.normalize(
        expression_query, dim=1, eps=1.0e-8
    ) @ expression_centroids.T
    stage_train_scores = torch.nn.functional.normalize(
        stage0_train, dim=1, eps=1.0e-8
    ) @ stage_centroids.T
    stage_query_scores = torch.nn.functional.normalize(
        stage0_query, dim=1, eps=1.0e-8
    ) @ stage_centroids.T
    return (
        standardize_class_scores(expression_train_scores),
        standardize_class_scores(expression_query_scores),
        standardize_class_scores(stage_train_scores),
        standardize_class_scores(stage_query_scores),
    )


def residual_predict(
    expression_scores: Tensor,
    stage0_scores: Tensor,
    alpha: float,
) -> Tensor:
    """Predict from an expression score plus a frozen-Stage0 residual."""
    expression_scores = _check_scores(expression_scores, "expression_scores")
    stage0_scores = _check_scores(stage0_scores, "stage0_scores")
    if expression_scores.shape != stage0_scores.shape:
        raise ValueError("expression and Stage0 score shapes differ")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    return (expression_scores + float(alpha) * stage0_scores).argmax(dim=1)


def residual_grid(
    expression_train: Tensor,
    stage0_train: Tensor,
    expression_query: Tensor,
    stage0_query: Tensor,
    label_codes: Tensor,
    group_codes: Tensor,
    n_classes: int,
    mode: str = "cell_mean",
    alphas: Sequence[float] = RESIDUAL_ALPHAS,
) -> dict[float, Tensor]:
    """Return CUDA predictions for every source-selected residual weight."""
    _, expression_query_scores, _, stage_query_scores = fit_residual_prototypes(
        expression_train,
        stage0_train,
        expression_query,
        stage0_query,
        label_codes,
        group_codes,
        n_classes,
        mode,
    )
    return {
        float(alpha): residual_predict(
            expression_query_scores, stage_query_scores, float(alpha)
        )
        for alpha in alphas
    }


__all__ = [
    "RESIDUAL_ALPHAS",
    "fit_residual_prototypes",
    "residual_grid",
    "residual_predict",
    "standardize_class_scores",
]
