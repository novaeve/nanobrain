"""CUDA-first adapters for strict human-to-mouse annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

VARIANTS = ("ortholog_tied", "homology_only")
CENTROID_MODES = ("cell_mean", "donor_balanced")


def _as_finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix")
    return result


def clr_expression(counts: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Reference NumPy CLR used by unit tests and compact audits."""
    values = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    values = _as_finite_matrix(values, name="counts")
    if np.any(values < 0):
        raise ValueError("counts must be non-negative")
    logged = np.log1p(values)
    return (logged - logged.mean(axis=1, keepdims=True)).astype(np.float32, copy=False)


def _validate_cuda_matrix(values: Tensor, *, name: str) -> Tensor:
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError(f"{name} must be a floating-point matrix")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def torch_clr_expression(counts: Tensor) -> Tensor:
    counts = _validate_cuda_matrix(counts, name="counts")
    if torch.any(counts < 0):
        raise ValueError("counts must be non-negative")
    logged = torch.log1p(counts)
    return logged - logged.mean(dim=1, keepdim=True)


def torch_log_cpm_l2(counts: Tensor) -> Tensor:
    counts = _validate_cuda_matrix(counts, name="counts")
    totals = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    values = torch.log1p(counts * (1.0e4 / totals))
    return torch.nn.functional.normalize(values, dim=1, eps=1.0e-8)


def torch_tfidf_l2(fit_counts: Tensor, query_counts: Tensor) -> tuple[Tensor, Tensor]:
    fit_counts = _validate_cuda_matrix(fit_counts, name="fit_counts")
    query_counts = _validate_cuda_matrix(query_counts, name="query_counts")
    if fit_counts.shape[1] != query_counts.shape[1]:
        raise ValueError("TF-IDF fit and query panels differ")
    document_frequency = (fit_counts > 0).sum(dim=0)
    idf = (
        torch.log(
            (1.0 + float(fit_counts.shape[0])) / (1.0 + document_frequency.float())
        )
        + 1.0
    )

    def transform(values: Tensor) -> Tensor:
        totals = values.sum(dim=1, keepdim=True).clamp_min(1.0)
        result = torch.log1p(values * (1.0e4 / totals)) * idf
        return torch.nn.functional.normalize(result, dim=1, eps=1.0e-8)

    return transform(fit_counts), transform(query_counts)


def _mean_scale(values: Tensor) -> tuple[Tensor, Tensor]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))
    return mean, scale


@dataclass(frozen=True)
class FusionProjection:
    """Source-fit CUDA CLR/PCA and orthogonal Stage0 alignment."""

    clr_mean: Tensor
    components: Tensor
    raw_mean: Tensor
    raw_scale: Tensor
    stage_mean: Tensor
    stage_scale: Tensor
    rotation: Tensor
    explained_variance_ratio_sum: float
    fit_rows: int
    dimension: int
    seed: int

    def transform(self, counts: Tensor, stage0: Tensor, *, alpha: float) -> Tensor:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        counts = _validate_cuda_matrix(counts, name="counts")
        stage0 = _validate_cuda_matrix(stage0, name="stage0")
        if (
            counts.device != self.components.device
            or stage0.device != self.components.device
        ):
            raise ValueError("fusion tensors must share the fitted CUDA device")
        if counts.shape[0] != stage0.shape[0]:
            raise ValueError("counts and Stage0 rows must align")
        raw = (torch_clr_expression(counts) - self.clr_mean) @ self.components
        raw = (raw - self.raw_mean) / self.raw_scale
        stage = (stage0 - self.stage_mean) / self.stage_scale
        aligned = stage @ self.rotation
        return raw + float(alpha) * aligned

    def audit(self) -> dict[str, Any]:
        return {
            "method": "source_fit_cuda_clr_pca_then_orthogonal_procrustes",
            "dimension": self.dimension,
            "seed": self.seed,
            "pca_fit_rows": self.fit_rows,
            "pca_input_features": int(self.components.shape[0]),
            "pca_explained_variance_ratio_sum": self.explained_variance_ratio_sum,
            "constant_raw_dimensions": int((self.raw_scale == 1.0).sum().item()),
            "constant_stage0_dimensions": int((self.stage_scale == 1.0).sum().item()),
            "tensor_device": str(self.components.device),
        }


def fit_fusion_projection(
    source_counts: Tensor,
    source_stage0: Tensor,
    *,
    dimension: int = 256,
    seed: int = 20260902,
) -> FusionProjection:
    """Fit PCA and Procrustes entirely on the tensors' current device."""
    source_counts = _validate_cuda_matrix(source_counts, name="source_counts")
    source_stage0 = _validate_cuda_matrix(source_stage0, name="source_stage0")
    if source_counts.device != source_stage0.device:
        raise ValueError("source counts and Stage0 tensors must share a device")
    if source_counts.shape[0] != source_stage0.shape[0]:
        raise ValueError("source counts and Stage0 rows must align")
    maximum = min(
        source_counts.shape[0] - 1, source_counts.shape[1], source_stage0.shape[1]
    )
    dimension = int(dimension)
    if not 1 <= dimension <= maximum:
        raise ValueError(f"dimension must be in [1, {maximum}]")
    clr = torch_clr_expression(source_counts)
    clr_mean = clr.mean(dim=0)
    centered = clr - clr_mean
    fork_devices = [source_counts.device.index or 0] if source_counts.is_cuda else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(seed))
        _, singular_values, components = torch.pca_lowrank(
            centered, q=dimension, center=False, niter=2
        )
    raw = centered @ components
    raw_mean, raw_scale = _mean_scale(raw)
    stage_mean, stage_scale = _mean_scale(source_stage0)
    raw_standard = (raw - raw_mean) / raw_scale
    stage_standard = (source_stage0 - stage_mean) / stage_scale
    covariance = stage_standard.T @ raw_standard
    left, _, right_h = torch.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_h
    total_variance = centered.square().sum().clamp_min(1.0e-12)
    explained = float((singular_values.square().sum() / total_variance).item())
    return FusionProjection(
        clr_mean=clr_mean,
        components=components,
        raw_mean=raw_mean,
        raw_scale=raw_scale,
        stage_mean=stage_mean,
        stage_scale=stage_scale,
        rotation=rotation,
        explained_variance_ratio_sum=explained,
        fit_rows=source_counts.shape[0],
        dimension=dimension,
        seed=int(seed),
    )


def fit_grouped_centroids(
    values: Tensor,
    label_codes: Tensor,
    group_codes: Tensor,
    n_classes: int,
    mode: str,
) -> Tensor:
    """Fit cell- or source-group-balanced prototypes on CUDA."""
    values = _validate_cuda_matrix(values, name="values")
    if mode not in CENTROID_MODES:
        raise ValueError(f"unknown centroid mode: {mode}")
    if label_codes.device != values.device or group_codes.device != values.device:
        raise ValueError("prototype codes and values must share a device")
    if label_codes.ndim != 1 or group_codes.ndim != 1:
        raise ValueError("prototype codes must be vectors")
    if len(label_codes) != len(values) or len(group_codes) != len(values):
        raise ValueError("prototype codes and values must align")
    n_classes = int(n_classes)
    if (
        n_classes < 1
        or torch.any(label_codes < 0)
        or torch.any(label_codes >= n_classes)
    ):
        raise ValueError("label code is outside the class vocabulary")
    if mode == "cell_mean":
        sums = values.new_zeros((n_classes, values.shape[1]))
        sums.index_add_(0, label_codes, values)
        counts = torch.bincount(label_codes, minlength=n_classes).clamp_min(1)
        centroids = sums / counts[:, None]
    else:
        _, compact_groups = torch.unique(group_codes, sorted=True, return_inverse=True)
        n_groups = int(compact_groups.max().item()) + 1
        pair_key = label_codes * n_groups + compact_groups
        unique_pairs, pair_codes = torch.unique(
            pair_key, sorted=True, return_inverse=True
        )
        pair_sums = values.new_zeros((len(unique_pairs), values.shape[1]))
        pair_sums.index_add_(0, pair_codes, values)
        pair_counts = torch.bincount(pair_codes, minlength=len(unique_pairs)).clamp_min(
            1
        )
        pair_means = pair_sums / pair_counts[:, None]
        pair_labels = torch.div(unique_pairs, n_groups, rounding_mode="floor")
        class_sums = values.new_zeros((n_classes, values.shape[1]))
        class_sums.index_add_(0, pair_labels, pair_means)
        class_counts = torch.bincount(pair_labels, minlength=n_classes).clamp_min(1)
        centroids = class_sums / class_counts[:, None]
    return torch.nn.functional.normalize(centroids, dim=1, eps=1.0e-8)


def predict_grouped_centroids(
    values: Tensor, centroids: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    values = _validate_cuda_matrix(values, name="values")
    centroids = _validate_cuda_matrix(centroids, name="centroids")
    if values.device != centroids.device or values.shape[1] != centroids.shape[1]:
        raise ValueError("query values and centroids are incompatible")
    scores = torch.nn.functional.normalize(values, dim=1, eps=1.0e-8) @ centroids.T
    top = torch.topk(scores, k=min(2, scores.shape[1]), dim=1)
    prediction = top.indices[:, 0]
    confidence = top.values[:, 0]
    margin = (
        top.values[:, 0] - top.values[:, 1]
        if scores.shape[1] > 1
        else torch.ones_like(confidence)
    )
    return prediction, confidence, margin


def normalize_ortholog_pairs(pairs: np.ndarray, n_genes: int) -> np.ndarray:
    result = np.asarray(pairs, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 2 or not len(result):
        raise ValueError("ortholog pairs must have shape [N, 2]")
    if np.any(result < 0) or np.any(result >= int(n_genes)):
        raise ValueError("ortholog pair index is outside the gene vocabulary")
    if len(np.unique(result[:, 0])) != len(result):
        raise ValueError("human ortholog endpoints must be one-to-one")
    if len(np.unique(result[:, 1])) != len(result):
        raise ValueError("mouse ortholog endpoints must be one-to-one")
    if np.intersect1d(result[:, 0], result[:, 1]).size:
        raise ValueError("human and mouse endpoint namespaces overlap")
    return result


@torch.no_grad()
def apply_gene_embedding_variant_(
    embedding: torch.nn.Embedding,
    pairs: np.ndarray,
    variant: str,
) -> dict[str, Any]:
    """Apply an evaluation-only ortholog sharing rule in memory."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown cross-species variant: {variant}")
    normalized = normalize_ortholog_pairs(pairs, embedding.num_embeddings)
    human = torch.as_tensor(
        normalized[:, 0], dtype=torch.long, device=embedding.weight.device
    )
    mouse = torch.as_tensor(
        normalized[:, 1], dtype=torch.long, device=embedding.weight.device
    )
    if variant == "ortholog_tied":
        shared = 0.5 * (embedding.weight[human] + embedding.weight[mouse])
        embedding.weight[human] = shared
        embedding.weight[mouse] = shared
    else:
        embedding.weight[human] = 0.0
        embedding.weight[mouse] = 0.0
    maximum_pair_difference = float(
        (embedding.weight[human] - embedding.weight[mouse]).abs().max().item()
    )
    return {
        "variant": variant,
        "pairs": len(normalized),
        "checkpoint_modified_on_disk": False,
        "maximum_pair_embedding_difference_after_transform": maximum_pair_difference,
    }
