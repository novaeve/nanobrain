"""Strict CUDA primitives for source-only cross-species evaluation.

The fresh cross-species evaluator has a deliberately small numerical core.  This
module keeps that core independent from the data loader and from the legacy
``crossspecies_annotation`` helpers:

* CPU dense, NumPy-like, and SciPy sparse inputs can be *unpacked* with
  :func:`to_cuda_matrix`; all fitting and prediction functions then require
  CUDA tensors.
* Normalization and PCA statistics are fitted on the source rows only.
* Prototype fitting and cosine prediction never move feature tensors back to
  the CPU.  Only the final scalar balanced-accuracy value is serialized.

The public names intentionally mirror the evaluator's existing vocabulary
(``fit_fusion_projection``, ``fit_grouped_centroids``, and
``predict_grouped_centroids``) so the runner can opt into this implementation
without changing its experiment contract.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

try:  # SciPy is an input adapter, never part of the numerical core.
    import scipy.sparse as _scipy_sparse
except ImportError:  # pragma: no cover - optional dependency at import time
    _scipy_sparse = None


NormalizationMethod = Literal["clr", "log_cpm"]
CentroidMode = Literal["cell_mean", "donor_balanced"]


@dataclass(frozen=True)
class CudaTelemetry:
    """Device and peak-memory information for one CUDA operation."""

    operation: str
    device: str
    device_type: str
    device_index: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "device": self.device,
            "device_type": self.device_type,
            "device_index": self.device_index,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _cuda_device(device: torch.device | str | int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("cross-species GPU operations require CUDA")
    if isinstance(device, int):
        resolved = torch.device("cuda", device)
    elif device is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"device must be CUDA, got {resolved}")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _begin_cuda_operation(device: torch.device) -> float:
    device = _cuda_device(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _finish_cuda_operation(
    operation: str, device: torch.device, started: float
) -> CudaTelemetry:
    device = _cuda_device(device)
    torch.cuda.synchronize(device)
    return CudaTelemetry(
        operation=operation,
        device=str(device),
        device_type=device.type,
        device_index=int(device.index),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def _require_cuda_matrix(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA; call to_cuda_matrix first")
    if not value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_cuda_vector(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA; call to_cuda_vector first")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must be an integer CUDA tensor")
    if value.numel() < 1:
        raise ValueError(f"{name} must be non-empty")
    return value


def to_cuda_matrix(
    value: Any,
    *,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Unpack a dense/sparse matrix and materialize it as a CUDA tensor.

    This is the only public function in this module that intentionally accepts
    CPU data.  SciPy CSR/CSC/COO inputs are transferred as a CUDA CSR tensor
    before densification; no CPU ``toarray`` copy is made.
    """

    target = _cuda_device(device)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point torch dtype")

    if isinstance(value, Tensor):
        if value.ndim != 2:
            raise ValueError("matrix input must be rank-2")
        if value.layout != torch.strided:
            # Moving first keeps sparse data on the GPU during densification.
            result = value.to(device=target, dtype=dtype).to_dense()
        else:
            result = value.to(device=target, dtype=dtype)
    elif _scipy_sparse is not None and _scipy_sparse.issparse(value):
        csr = value.tocsr()
        if len(csr.shape) != 2:
            raise ValueError("sparse matrix input must be rank-2")
        crow = torch.as_tensor(csr.indptr, dtype=torch.int64, device=target)
        col = torch.as_tensor(csr.indices, dtype=torch.int64, device=target)
        data = torch.as_tensor(csr.data, dtype=dtype, device=target)
        sparse = torch.sparse_csr_tensor(
            crow, col, data, size=tuple(int(v) for v in csr.shape), device=target
        )
        result = sparse.to_dense()
    else:
        # ``torch.as_tensor`` is an input-unpacking operation; all subsequent
        # numerical work is performed by CUDA kernels.
        result = torch.as_tensor(value, dtype=dtype, device=target)
        if result.ndim != 2:
            raise ValueError("matrix input must be rank-2")

    result = result.contiguous()
    return _require_cuda_matrix(result, name="matrix")


def to_cuda_vector(
    value: Any,
    *,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype = torch.int64,
) -> Tensor:
    """Unpack integer labels/donor IDs onto CUDA for prototype operations."""

    target = _cuda_device(device)
    if dtype == torch.bool or dtype.is_floating_point or dtype.is_complex:
        raise ValueError("vector dtype must be an integer torch dtype")
    result = torch.as_tensor(value, dtype=dtype, device=target)
    return _require_cuda_vector(result, name="vector")


def _check_same_device(*values: Tensor) -> torch.device:
    if not values:
        raise ValueError("at least one tensor is required")
    devices = [value.device for value in values]
    if any(device.type != "cuda" for device in devices):
        raise ValueError("all core tensors must be on CUDA")
    if any(device != devices[0] for device in devices[1:]):
        raise ValueError("core tensors must share one CUDA device")
    return devices[0]


def clr_cuda(counts: Tensor) -> Tensor:
    """Per-cell centered log expression (CLR) on CUDA."""

    counts = _require_cuda_matrix(counts, name="counts")
    if bool(torch.any(counts < 0).item()):
        raise ValueError("counts must be non-negative")
    logged = torch.log1p(counts)
    return logged - logged.mean(dim=1, keepdim=True)


def log_cpm_cuda(counts: Tensor, *, library_scale: float = 1.0e4) -> Tensor:
    """Library-size normalized log expression on CUDA."""

    counts = _require_cuda_matrix(counts, name="counts")
    if bool(torch.any(counts < 0).item()):
        raise ValueError("counts must be non-negative")
    scale = float(library_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("library_scale must be positive and finite")
    totals = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.log1p(counts * (scale / totals))


def log_cpm_l2_cuda(counts: Tensor, *, library_scale: float = 1.0e4) -> Tensor:
    return F.normalize(log_cpm_cuda(counts, library_scale=library_scale), dim=1, eps=1.0e-8)


# Compatibility names used by the existing evaluator vocabulary.
torch_clr_expression = clr_cuda
torch_log_cpm = log_cpm_cuda
torch_log_cpm_l2 = log_cpm_l2_cuda


@dataclass(frozen=True)
class SourceFitNormalizer:
    """Source-fitted feature centering/scaling for CLR or log-CPM views."""

    method: NormalizationMethod
    feature_mean: Tensor
    feature_scale: Tensor
    library_scale: float
    standardize: bool
    fit_rows: int
    telemetry: CudaTelemetry

    @property
    def device(self) -> torch.device:
        return self.feature_mean.device

    def _base(self, counts: Tensor) -> Tensor:
        counts = _require_cuda_matrix(counts, name="counts")
        if counts.device != self.device:
            raise ValueError("counts and source-fit normalizer use different devices")
        if self.method == "clr":
            return clr_cuda(counts)
        if self.method == "log_cpm":
            return log_cpm_cuda(counts, library_scale=self.library_scale)
        raise RuntimeError(f"unsupported normalization method: {self.method}")

    def transform(self, counts: Tensor) -> Tensor:
        base = self._base(counts)
        if base.shape[1] != self.feature_mean.numel():
            raise ValueError("counts feature width differs from source fit")
        transformed = (base - self.feature_mean) / self.feature_scale
        if not bool(torch.isfinite(transformed).all().item()):
            raise FloatingPointError("source-fit normalization produced non-finite values")
        return transformed

    def audit(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "fit_rows": self.fit_rows,
            "features": int(self.feature_mean.numel()),
            "standardize": self.standardize,
            "library_scale": self.library_scale,
            "device": str(self.device),
            "telemetry": self.telemetry.as_dict(),
        }


def fit_source_normalizer(
    source_counts: Tensor,
    *,
    method: NormalizationMethod = "clr",
    library_scale: float = 1.0e4,
    standardize: bool = True,
) -> SourceFitNormalizer:
    """Fit normalization statistics using source rows only."""

    source_counts = _require_cuda_matrix(source_counts, name="source_counts")
    device = source_counts.device
    _check_same_device(source_counts)
    if method not in ("clr", "log_cpm"):
        raise ValueError("method must be 'clr' or 'log_cpm'")
    started = _begin_cuda_operation(device)
    base = clr_cuda(source_counts) if method == "clr" else log_cpm_cuda(
        source_counts, library_scale=library_scale
    )
    if standardize:
        mean = base.mean(dim=0)
        scale = base.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))
    else:
        mean = torch.zeros(base.shape[1], dtype=base.dtype, device=device)
        scale = torch.ones(base.shape[1], dtype=base.dtype, device=device)
    telemetry = _finish_cuda_operation("fit_source_normalizer", device, started)
    return SourceFitNormalizer(
        method=method,
        feature_mean=mean,
        feature_scale=scale,
        library_scale=float(library_scale),
        standardize=bool(standardize),
        fit_rows=int(source_counts.shape[0]),
        telemetry=telemetry,
    )


def fit_source_clr(
    source_counts: Tensor, *, standardize: bool = True
) -> SourceFitNormalizer:
    return fit_source_normalizer(source_counts, method="clr", standardize=standardize)


def fit_source_log_normalizer(
    source_counts: Tensor, *, library_scale: float = 1.0e4, standardize: bool = True
) -> SourceFitNormalizer:
    return fit_source_normalizer(
        source_counts,
        method="log_cpm",
        library_scale=library_scale,
        standardize=standardize,
    )


def source_fit_transform(
    source_counts: Tensor,
    query_counts: Tensor,
    *,
    method: NormalizationMethod = "clr",
    library_scale: float = 1.0e4,
    standardize: bool = True,
) -> tuple[Tensor, Tensor, SourceFitNormalizer]:
    """Fit on source and transform source/query with the same parameters."""

    _check_same_device(source_counts, query_counts)
    normalizer = fit_source_normalizer(
        source_counts,
        method=method,
        library_scale=library_scale,
        standardize=standardize,
    )
    return normalizer.transform(source_counts), normalizer.transform(query_counts), normalizer


@dataclass(frozen=True)
class GPUPrincipalComponents:
    """Exact source-fit PCA represented by a CUDA SVD."""

    mean: Tensor
    components: Tensor
    singular_values: Tensor
    explained_variance_ratio: Tensor
    fit_rows: int
    input_features: int
    seed: int
    telemetry: CudaTelemetry

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @property
    def dimension(self) -> int:
        return int(self.components.shape[1])

    def transform(self, values: Tensor) -> Tensor:
        values = _require_cuda_matrix(values, name="values")
        if values.device != self.device:
            raise ValueError("values and PCA use different devices")
        if values.shape[1] != self.input_features:
            raise ValueError("values feature width differs from PCA fit")
        result = (values - self.mean) @ self.components
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("PCA transform produced non-finite values")
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "method": "cuda_svd_pca",
            "fit_rows": self.fit_rows,
            "input_features": self.input_features,
            "dimension": self.dimension,
            "seed": self.seed,
            "explained_variance_ratio_sum": float(self.explained_variance_ratio.sum().item()),
            "device": str(self.device),
            "telemetry": self.telemetry.as_dict(),
        }


def _fit_gpu_pca_impl(
    values: Tensor,
    *,
    n_components: int,
    seed: int,
    operation_started: float | None = None,
) -> GPUPrincipalComponents:
    values = _require_cuda_matrix(values, name="values")
    if values.shape[0] < 2:
        raise ValueError("PCA requires at least two source rows")
    maximum = min(int(values.shape[0] - 1), int(values.shape[1]))
    k = int(n_components)
    if k < 1 or k > maximum:
        raise ValueError(f"n_components must be in [1, {maximum}]")
    device = values.device
    started = _begin_cuda_operation(device) if operation_started is None else operation_started
    mean = values.mean(dim=0)
    centered = values - mean
    # Exact SVD is deterministic and keeps all numerical work on the GPU.
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:k].transpose(0, 1).contiguous()
    selected = singular_values[:k]
    variance = singular_values.square()
    denominator = variance.sum().clamp_min(1.0e-12)
    ratio = variance[:k] / denominator
    telemetry = _finish_cuda_operation("fit_gpu_pca", device, started)
    return GPUPrincipalComponents(
        mean=mean,
        components=components,
        singular_values=selected,
        explained_variance_ratio=ratio,
        fit_rows=int(values.shape[0]),
        input_features=int(values.shape[1]),
        seed=int(seed),
        telemetry=telemetry,
    )


def fit_gpu_pca(
    values: Tensor, *, n_components: int = 256, seed: int = 20260902
) -> GPUPrincipalComponents:
    """Fit an exact PCA/SVD projection on CUDA."""

    return _fit_gpu_pca_impl(values, n_components=n_components, seed=seed)


@dataclass(frozen=True)
class FusionProjection:
    """CLR/log-PCA plus source-fit standardized Stage0 Procrustes fusion."""

    pca: GPUPrincipalComponents
    raw_mean: Tensor
    raw_scale: Tensor
    stage_mean: Tensor
    stage_scale: Tensor
    rotation: Tensor
    normalization: NormalizationMethod
    alpha_default: float
    fit_rows: int
    seed: int
    telemetry: CudaTelemetry

    @property
    def device(self) -> torch.device:
        return self.pca.device

    @property
    def dimension(self) -> int:
        return self.pca.dimension

    def _transform_impl(self, counts: Tensor, stage0: Tensor, alpha: float) -> Tensor:
        counts = _require_cuda_matrix(counts, name="counts")
        stage0 = _require_cuda_matrix(stage0, name="stage0")
        _check_same_device(counts, stage0)
        if counts.device != self.device:
            raise ValueError("fusion inputs and fit use different devices")
        if counts.shape[0] != stage0.shape[0]:
            raise ValueError("counts and Stage0 rows must align")
        if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        base = clr_cuda(counts) if self.normalization == "clr" else log_cpm_cuda(counts)
        raw = self.pca.transform(base)
        raw = (raw - self.raw_mean) / self.raw_scale
        stage = (stage0 - self.stage_mean) / self.stage_scale
        aligned = stage @ self.rotation
        result = raw + float(alpha) * aligned
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("fusion transform produced non-finite values")
        return result

    def transform(self, counts: Tensor, stage0: Tensor, *, alpha: float | None = None) -> Tensor:
        selected = self.alpha_default if alpha is None else float(alpha)
        return self._transform_impl(counts, stage0, selected)

    def transform_with_telemetry(
        self, counts: Tensor, stage0: Tensor, *, alpha: float | None = None
    ) -> tuple[Tensor, CudaTelemetry]:
        started = _begin_cuda_operation(self.device)
        result = self.transform(counts, stage0, alpha=alpha)
        return result, _finish_cuda_operation("fusion_transform", self.device, started)

    def audit(self) -> dict[str, Any]:
        return {
            "method": "source_fit_pca_procrustes_fusion",
            "normalization": self.normalization,
            "fit_rows": self.fit_rows,
            "dimension": self.dimension,
            "stage_dimension": int(self.rotation.shape[0]),
            "alpha_default": self.alpha_default,
            "seed": self.seed,
            "device": str(self.device),
            "pca": self.pca.audit(),
            "telemetry": self.telemetry.as_dict(),
        }


def _mean_scale(values: Tensor) -> tuple[Tensor, Tensor]:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1.0e-6, scale, torch.ones_like(scale))
    return mean, scale


def fit_fusion_projection(
    source_counts: Tensor,
    source_stage0: Tensor,
    *,
    dimension: int = 256,
    seed: int = 20260902,
    normalization: NormalizationMethod = "clr",
    alpha_default: float = 1.0,
) -> FusionProjection:
    """Fit source-only PCA, standardization, and orthogonal Procrustes."""

    source_counts = _require_cuda_matrix(source_counts, name="source_counts")
    source_stage0 = _require_cuda_matrix(source_stage0, name="source_stage0")
    device = _check_same_device(source_counts, source_stage0)
    if source_counts.shape[0] != source_stage0.shape[0]:
        raise ValueError("source counts and Stage0 rows must align")
    if normalization not in ("clr", "log_cpm"):
        raise ValueError("normalization must be 'clr' or 'log_cpm'")
    if not math.isfinite(float(alpha_default)) or float(alpha_default) < 0.0:
        raise ValueError("alpha_default must be finite and non-negative")
    started = _begin_cuda_operation(device)
    normalized = clr_cuda(source_counts) if normalization == "clr" else log_cpm_cuda(source_counts)
    pca = _fit_gpu_pca_impl(
        normalized,
        n_components=int(dimension),
        seed=int(seed),
        operation_started=started,
    )
    raw = pca.transform(normalized)
    raw_mean, raw_scale = _mean_scale(raw)
    stage_mean, stage_scale = _mean_scale(source_stage0)
    raw_standard = (raw - raw_mean) / raw_scale
    stage_standard = (source_stage0 - stage_mean) / stage_scale
    covariance = stage_standard.transpose(0, 1) @ raw_standard
    left, _, right_h = torch.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_h
    telemetry = _finish_cuda_operation("fit_fusion_projection", device, started)
    return FusionProjection(
        pca=pca,
        raw_mean=raw_mean,
        raw_scale=raw_scale,
        stage_mean=stage_mean,
        stage_scale=stage_scale,
        rotation=rotation,
        normalization=normalization,
        alpha_default=float(alpha_default),
        fit_rows=int(source_counts.shape[0]),
        seed=int(seed),
        telemetry=telemetry,
    )


def fit_fusion_projection_with_telemetry(
    source_counts: Tensor,
    source_stage0: Tensor,
    **kwargs: Any,
) -> tuple[FusionProjection, CudaTelemetry]:
    projection = fit_fusion_projection(source_counts, source_stage0, **kwargs)
    return projection, projection.telemetry


def fit_grouped_centroids(
    values: Tensor,
    label_codes: Tensor,
    group_codes: Tensor,
    n_classes: int,
    mode: CentroidMode = "cell_mean",
) -> Tensor:
    """Fit normalized cell-mean or donor-balanced class prototypes on CUDA."""

    values = _require_cuda_matrix(values, name="values")
    label_codes = _require_cuda_vector(label_codes, name="label_codes")
    group_codes = _require_cuda_vector(group_codes, name="group_codes")
    _check_same_device(values, label_codes, group_codes)
    if mode not in ("cell_mean", "donor_balanced"):
        raise ValueError("mode must be 'cell_mean' or 'donor_balanced'")
    classes = int(n_classes)
    if classes < 1:
        raise ValueError("n_classes must be positive")
    if label_codes.numel() != values.shape[0] or group_codes.numel() != values.shape[0]:
        raise ValueError("prototype codes and values must have matching rows")
    if bool(torch.any(label_codes < 0).item()) or bool(torch.any(label_codes >= classes).item()):
        raise ValueError("label code is outside the class vocabulary")
    if bool(torch.any(group_codes < 0).item()):
        raise ValueError("group codes must be non-negative")

    sums = values.new_zeros((classes, values.shape[1]))
    if mode == "cell_mean":
        sums.index_add_(0, label_codes, values)
        counts = torch.bincount(label_codes, minlength=classes).to(values.dtype).clamp_min(1.0)
        centroids = sums / counts[:, None]
    else:
        _, compact_groups = torch.unique(group_codes, sorted=True, return_inverse=True)
        n_groups = int(compact_groups.max().item()) + 1
        pair_key = label_codes * n_groups + compact_groups
        unique_pairs, pair_codes = torch.unique(pair_key, sorted=True, return_inverse=True)
        pair_sums = values.new_zeros((unique_pairs.numel(), values.shape[1]))
        pair_sums.index_add_(0, pair_codes, values)
        pair_counts = torch.bincount(pair_codes, minlength=unique_pairs.numel()).to(values.dtype)
        pair_means = pair_sums / pair_counts.clamp_min(1.0)[:, None]
        pair_labels = torch.div(unique_pairs, n_groups, rounding_mode="floor")
        sums.index_add_(0, pair_labels, pair_means)
        donor_counts = torch.bincount(pair_labels, minlength=classes).to(values.dtype).clamp_min(1.0)
        centroids = sums / donor_counts[:, None]
    return F.normalize(centroids, dim=1, eps=1.0e-8)


def fit_grouped_centroids_with_telemetry(
    values: Tensor,
    label_codes: Tensor,
    group_codes: Tensor,
    n_classes: int,
    mode: CentroidMode = "cell_mean",
) -> tuple[Tensor, CudaTelemetry]:
    values = _require_cuda_matrix(values, name="values")
    started = _begin_cuda_operation(values.device)
    result = fit_grouped_centroids(values, label_codes, group_codes, n_classes, mode)
    return result, _finish_cuda_operation("fit_grouped_centroids", values.device, started)


def predict_grouped_centroids(
    values: Tensor, centroids: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Cosine nearest-prototype prediction, confidence, and top-2 margin."""

    values = _require_cuda_matrix(values, name="values")
    centroids = _require_cuda_matrix(centroids, name="centroids")
    _check_same_device(values, centroids)
    if values.shape[1] != centroids.shape[1]:
        raise ValueError("query values and centroids have incompatible widths")
    if centroids.shape[0] < 1:
        raise ValueError("at least one centroid is required")
    scores = F.normalize(values, dim=1, eps=1.0e-8) @ F.normalize(centroids, dim=1, eps=1.0e-8).transpose(0, 1)
    top_values, top_indices = torch.topk(scores, k=min(2, scores.shape[1]), dim=1)
    prediction = top_indices[:, 0]
    confidence = top_values[:, 0]
    margin = (
        top_values[:, 0] - top_values[:, 1]
        if top_values.shape[1] > 1
        else torch.ones_like(confidence)
    )
    return prediction, confidence, margin


cosine_nearest_prototype = predict_grouped_centroids


def balanced_accuracy_tensor(
    truth: Tensor,
    prediction: Tensor,
    *,
    n_classes: int | None = None,
) -> Tensor:
    """Return macro recall as a scalar CUDA tensor without sklearn/numpy."""

    truth = _require_cuda_vector(truth, name="truth")
    prediction = _require_cuda_vector(prediction, name="prediction")
    _check_same_device(truth, prediction)
    if truth.numel() != prediction.numel():
        raise ValueError("truth and prediction lengths differ")
    if bool(torch.any(truth < 0).item()) or bool(torch.any(prediction < 0).item()):
        raise ValueError("labels must be non-negative")
    observed_max = torch.maximum(truth.max(), prediction.max())
    classes = int(observed_max.item()) + 1 if n_classes is None else int(n_classes)
    if classes < 1:
        raise ValueError("n_classes must be positive")
    if bool(torch.any(truth >= classes).item()) or bool(torch.any(prediction >= classes).item()):
        raise ValueError("label exceeds n_classes")
    support = torch.bincount(truth, minlength=classes)
    correct = truth[prediction == truth]
    hits = torch.bincount(correct, minlength=classes)
    recalls = hits.to(torch.float32) / support.to(torch.float32).clamp_min(1.0)
    present = support > 0
    return recalls[present].mean()


def balanced_accuracy(
    truth: Tensor, prediction: Tensor, *, n_classes: int | None = None
) -> float:
    """Serialize CUDA macro recall to a Python scalar at the API boundary."""

    return float(balanced_accuracy_tensor(truth, prediction, n_classes=n_classes).item())


def balanced_accuracy_with_telemetry(
    truth: Tensor,
    prediction: Tensor,
    *,
    n_classes: int | None = None,
) -> tuple[float, CudaTelemetry]:
    truth = _require_cuda_vector(truth, name="truth")
    started = _begin_cuda_operation(truth.device)
    score = balanced_accuracy(truth, prediction, n_classes=n_classes)
    return score, _finish_cuda_operation("balanced_accuracy", truth.device, started)


__all__ = [
    "CudaTelemetry",
    "FusionProjection",
    "GPUPrincipalComponents",
    "SourceFitNormalizer",
    "balanced_accuracy",
    "balanced_accuracy_tensor",
    "balanced_accuracy_with_telemetry",
    "clr_cuda",
    "cosine_nearest_prototype",
    "fit_fusion_projection",
    "fit_fusion_projection_with_telemetry",
    "fit_gpu_pca",
    "fit_grouped_centroids",
    "fit_grouped_centroids_with_telemetry",
    "fit_source_clr",
    "fit_source_log_normalizer",
    "fit_source_normalizer",
    "log_cpm_cuda",
    "log_cpm_l2_cuda",
    "predict_grouped_centroids",
    "source_fit_transform",
    "to_cuda_matrix",
    "to_cuda_vector",
    "torch_clr_expression",
    "torch_log_cpm",
    "torch_log_cpm_l2",
]
