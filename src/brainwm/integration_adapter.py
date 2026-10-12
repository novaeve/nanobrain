"""Small, label-free operators for raw-preserving integration adapters."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all():
        raise ValueError("values must be a finite 2-D matrix with at least two rows")
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    return mean, std


def normalize_for_clustering(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    normalized = (matrix - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    return (normalized / np.clip(norms, 1.0e-6, None)).astype(np.float32, copy=False)


@torch.inference_mode()
def assign_centroids(
    values: np.ndarray,
    centroids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    centers = torch.as_tensor(centroids, dtype=torch.float32, device=device)
    if matrix.ndim != 2 or centers.ndim != 2 or matrix.shape[1] != centers.shape[1]:
        raise ValueError("values and centroids must be compatible 2-D matrices")
    assignments = np.empty(len(matrix), dtype=np.int32)
    center_norm = (centers * centers).sum(dim=1).unsqueeze(0)
    for start in range(0, len(matrix), int(batch_size)):
        stop = min(len(matrix), start + int(batch_size))
        part = torch.from_numpy(np.array(matrix[start:stop], copy=True)).to(device)
        distances = (
            (part * part).sum(dim=1, keepdim=True)
            + center_norm
            - 2.0 * part @ centers.T
        )
        assignments[start:stop] = torch.argmin(distances, dim=1).cpu().numpy()
    return assignments


@torch.inference_mode()
def fit_deterministic_kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    device: torch.device,
    seed: int,
    iterations: int = 12,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < int(clusters) or int(clusters) < 2:
        raise ValueError("invalid matrix or cluster count")
    rng = np.random.default_rng(int(seed))
    initial = rng.choice(len(matrix), size=int(clusters), replace=False)
    centers = torch.from_numpy(np.array(matrix[initial], copy=True)).to(device)
    assignments = np.zeros(len(matrix), dtype=np.int32)
    completed = 0
    for iteration in range(int(iterations)):
        previous = assignments.copy()
        assignments = assign_centroids(
            matrix,
            centers.detach().cpu().numpy(),
            device=device,
            batch_size=int(batch_size),
        )
        sums = torch.zeros_like(centers)
        counts = torch.zeros(int(clusters), dtype=torch.float32, device=device)
        for start in range(0, len(matrix), int(batch_size)):
            stop = min(len(matrix), start + int(batch_size))
            part = torch.from_numpy(np.array(matrix[start:stop], copy=True)).to(device)
            labels = torch.from_numpy(assignments[start:stop].astype(np.int64)).to(device)
            sums.index_add_(0, labels, part)
            counts.index_add_(0, labels, torch.ones(len(labels), device=device))
        empty = torch.nonzero(counts.eq(0), as_tuple=False).flatten().cpu().numpy()
        if len(empty):
            replacements = rng.choice(len(matrix), size=len(empty), replace=False)
            for cluster, row in zip(empty, replacements, strict=True):
                sums[int(cluster)] = torch.from_numpy(np.array(matrix[int(row)], copy=True)).to(device)
                counts[int(cluster)] = 1.0
        centers = sums / counts.clamp_min(1.0).unsqueeze(1)
        completed = iteration + 1
        if iteration > 0 and np.array_equal(assignments, previous):
            break
    final_centers = centers.detach().cpu().numpy().astype(np.float32, copy=False)
    assignments = assign_centroids(
        matrix, final_centers, device=device, batch_size=int(batch_size)
    )
    diagnostics = {
        "clusters": int(clusters),
        "iterations": int(completed),
        "minimum_cluster_size": int(np.bincount(assignments, minlength=int(clusters)).min()),
        "maximum_cluster_size": int(np.bincount(assignments, minlength=int(clusters)).max()),
        "seed": int(seed),
    }
    return assignments, final_centers, diagnostics


def cluster_conditional_residual(
    values: np.ndarray,
    clusters: np.ndarray,
    batches: np.ndarray,
    *,
    alpha: float,
    shrinkage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float32)
    cluster_ids = np.asarray(clusters)
    batch_ids = np.asarray(batches, dtype=object)
    if matrix.ndim != 2 or len(matrix) != len(cluster_ids) or len(matrix) != len(batch_ids):
        raise ValueError("values, clusters, and batches must have aligned rows")
    if not np.isfinite(matrix).all():
        raise ValueError("values contain non-finite entries")
    if float(alpha) < 0.0 or float(alpha) > 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if float(shrinkage) < 0.0:
        raise ValueError("shrinkage must be non-negative")
    if float(alpha) == 0.0:
        return matrix.copy(), {
            "alpha": 0.0,
            "shrinkage": float(shrinkage),
            "corrected_groups": 0,
            "mean_applied_residual_l2": 0.0,
        }

    result = matrix.copy()
    residual_norms: list[float] = []
    corrected_groups = 0
    for cluster in np.unique(cluster_ids):
        cluster_mask = cluster_ids == cluster
        present_batches = np.unique(batch_ids[cluster_mask])
        if len(present_batches) < 2:
            continue
        means: dict[object, np.ndarray] = {}
        counts: dict[object, int] = {}
        for batch in present_batches:
            mask = cluster_mask & (batch_ids == batch)
            counts[batch] = int(mask.sum())
            means[batch] = matrix[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        equal_batch_center = np.mean(
            np.stack([means[batch] for batch in present_batches]),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        for batch in present_batches:
            mask = cluster_mask & (batch_ids == batch)
            weight = counts[batch] / (counts[batch] + float(shrinkage))
            residual = float(alpha) * float(weight) * (
                means[batch] - equal_batch_center
            )
            result[mask] -= residual
            residual_norms.append(float(np.linalg.norm(residual)))
            corrected_groups += 1
    return result, {
        "alpha": float(alpha),
        "shrinkage": float(shrinkage),
        "corrected_groups": int(corrected_groups),
        "mean_applied_residual_l2": float(np.mean(residual_norms))
        if residual_norms
        else 0.0,
    }


def choose_source_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if float(row["alpha"]) > 0.0]
    if not candidates:
        raise ValueError("candidate grid has no positive-alpha row")
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["score"]),
            float(row["alpha"]),
            int(row["clusters"]),
            float(row["shrinkage"]),
        ),
    )[0]


__all__ = [
    "assign_centroids",
    "choose_source_candidate",
    "cluster_conditional_residual",
    "fit_deterministic_kmeans",
    "normalize_for_clustering",
    "standardizer",
]
