"""Training objectives from Section 5 of `brain_world_model.pdf`."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import LossConfig
from .decoder import dense_counts_for_genes
from .endpoint import latent_endpoint_loss
from .stage_residual import (
    representation_anchor_loss as stage0_representation_anchor_loss,
)
from .types import BrainWorldModelOutput, WorldModelBatch
from .backends import compile_callable


def negative_binomial_nll(count: Tensor, mean: Tensor, dispersion: Tensor) -> Tensor:
    count = count.float()
    mean = mean.float().clamp_min(1.0e-8)
    theta = dispersion.float().clamp_min(1.0e-8)[None, :]
    log_prob = (
        torch.lgamma(count + theta)
        - torch.lgamma(theta)
        - torch.lgamma(count + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mean))
        + count * (torch.log(mean) - torch.log(theta + mean))
    )
    return -log_prob.mean()


def masked_cell_reconstruction_loss(count: Tensor, mean: Tensor,
                                    dispersion: Tensor,
                                    cell_mask: Tensor) -> Tensor:
    """NB reconstruction restricted to cells hidden from the encoder."""
    if cell_mask.ndim != 1 or cell_mask.shape[0] != count.shape[0]:
        raise ValueError("cell_mask must have shape [B]")
    if not torch.any(cell_mask):
        return mean.new_zeros(())
    return negative_binomial_nll(
        count[cell_mask], mean[cell_mask], dispersion
    )


def reconstruction_loss(output: BrainWorldModelOutput, batch: WorldModelBatch) -> Tensor:
    target = dense_counts_for_genes(
        batch.expression.row_ptr,
        batch.expression.gene_index,
        batch.expression.count,
        output.decoder.gene_index,
    )
    return negative_binomial_nll(
        target, output.decoder.mean, output.decoder.dispersion
    )


def variational_kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
    """Dimension-normalized KL(q(z|x) || N(0, I))."""
    return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()


def latent_variance_loss(latent: Tensor, target_std: float = 1.0) -> Tensor:
    """VICReg-style per-dimension spread floor to prevent latent collapse."""
    if latent.ndim != 2 or latent.shape[0] < 2:
        return latent.new_zeros(())
    if target_std <= 0:
        raise ValueError("target_std must be positive")
    std = latent.float().std(dim=0, unbiased=False)
    return torch.relu(target_std - std).mean()


def latent_covariance_loss(latent: Tensor) -> Tensor:
    """Penalize off-diagonal covariance after centering latent dimensions."""
    if latent.ndim != 2 or latent.shape[0] < 2:
        return latent.new_zeros(())
    centered = latent.float() - latent.float().mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / latent.shape[0]
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag(diagonal)
    return off_diagonal.square().mean()


def expression_geometry_loss(latent: Tensor, count: Tensor,
                             scale_factor: float = 1.0e4,
                             gene_weight: Tensor | None = None) -> Tensor:
    """Match latent cosine geometry to log-normalized expression.

    ``gene_weight`` is an optional fixed, source-defined feature mask. It is
    applied before per-cell library normalization, so unmeasured genes remain
    absent from ``count`` and are never silently converted to zeros.
    """
    if latent.ndim != 2 or count.ndim != 2:
        raise ValueError("latent and count must have shapes [B, D] and [B, G]")
    if latent.shape[0] != count.shape[0]:
        raise ValueError("latent and count batch sizes differ")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if gene_weight is not None:
        gene_weight = gene_weight.float()
        if gene_weight.ndim != 1 or gene_weight.shape[0] != count.shape[1]:
            raise ValueError("gene_weight must have shape [G]")
        if not torch.isfinite(gene_weight).all() or torch.any(gene_weight < 0):
            raise ValueError("gene_weight must be finite and nonnegative")
        if not torch.any(gene_weight > 0):
            raise ValueError("gene_weight must contain at least one positive value")
        count = count * gene_weight[None, :]
    if latent.shape[0] < 2:
        return latent.new_zeros(())
    count = count.float().clamp_min(0.0)
    library = count.sum(dim=1, keepdim=True).clamp_min(1.0)
    expression = torch.log1p(count * (scale_factor / library))
    target = F.normalize(expression, dim=-1)
    prediction = F.normalize(latent.float(), dim=-1)
    target_similarity = (target @ target.transpose(0, 1)).detach()
    predicted_similarity = prediction @ prediction.transpose(0, 1)
    off_diagonal = ~torch.eye(
        latent.shape[0], dtype=torch.bool, device=latent.device
    )
    return F.mse_loss(
        predicted_similarity[off_diagonal], target_similarity[off_diagonal]
    )


def contrastive_loss(first: Tensor, second: Tensor, temperature: float) -> Tensor:
    """Symmetric in-batch InfoNCE between two masked views of each cell."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("contrastive views must have equal [B, D] shapes")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    first = F.normalize(first.float(), dim=-1)
    second = F.normalize(second.float(), dim=-1)
    logits = first @ second.transpose(0, 1) / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def barlow_twins_loss(first: Tensor, second: Tensor,
                      redundancy_weight: float = 0.005,
                      eps: float = 1.0e-8) -> Tensor:
    """Barlow Twins cross-correlation objective for two cell views.

    This is intentionally an isolated objective.  It is not enabled by any
    default configuration; callers can use it for a matched ablation without
    changing the production Stage-0 objective.
    """
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("Barlow Twins views must have equal [B, D] shapes")
    if (
        not math.isfinite(float(redundancy_weight))
        or redundancy_weight < 0
        or not math.isfinite(float(eps))
        or eps <= 0
        or not torch.isfinite(first).all()
        or not torch.isfinite(second).all()
    ):
        raise ValueError("invalid Barlow Twins inputs or redundancy weight")
    if first.shape[0] < 2:
        return first.new_zeros(())
    first = (first.float() - first.float().mean(dim=0, keepdim=True))
    second = (second.float() - second.float().mean(dim=0, keepdim=True))
    first = first / first.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    second = second / second.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    correlation = first.transpose(0, 1) @ second / first.shape[0]
    diagonal = torch.diagonal(correlation)
    invariance = (diagonal - 1.0).square().sum()
    off_diagonal = correlation - torch.diag_embed(diagonal)
    redundancy = off_diagonal.square().sum()
    return invariance + float(redundancy_weight) * redundancy


def assay_adversarial_loss(logits: Tensor, assay_id: Tensor) -> Tensor:
    if logits.ndim != 2 or assay_id.shape != (logits.shape[0],):
        raise ValueError("assay logits/labels must have shapes [B, A] and [B]")
    return F.cross_entropy(logits.float(), assay_id.long())


def flow_matching_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("flow prediction and target shapes differ")
    return F.mse_loss(prediction.float(), target.float())


def _developmental_batch_scalar(
    value: Tensor | float,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Normalize a scalar or per-cell value to a finite ``[B]`` tensor."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if not isinstance(value, Tensor):
        try:
            value = torch.as_tensor(value, device=device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
    else:
        value = value.to(device=device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.ndim == 1 and value.shape[0] == batch_size:
        pass
    elif value.ndim == 2 and value.shape == (batch_size, 1):
        value = value[:, 0]
    else:
        raise ValueError(
            f"{name} must be a scalar, [B], or [B, 1] tensor; "
            f"got {tuple(value.shape)} for B={batch_size}"
        )
    if value.dtype == torch.bool or value.is_complex():
        raise ValueError(f"{name} must be a real numeric value")
    value = value.to(dtype=dtype)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_developmental_latents(source_latent: Tensor, target_latent: Tensor) -> None:
    if not isinstance(source_latent, Tensor) or not isinstance(target_latent, Tensor):
        raise ValueError("source_latent and target_latent must be tensors")
    if source_latent.ndim != 2 or target_latent.ndim != 2:
        raise ValueError("source_latent and target_latent must have shape [B, D]")
    if source_latent.shape != target_latent.shape:
        raise ValueError("source_latent and target_latent must have equal shapes")
    if source_latent.shape[0] < 1 or source_latent.shape[1] < 1:
        raise ValueError("source_latent and target_latent must be non-empty")
    if source_latent.device != target_latent.device:
        raise ValueError("source_latent and target_latent must share a device")
    if not source_latent.is_floating_point() or not target_latent.is_floating_point():
        raise ValueError("source_latent and target_latent must be floating tensors")
    if not torch.isfinite(source_latent).all() or not torch.isfinite(target_latent).all():
        raise ValueError("source_latent and target_latent must be finite")


def developmental_flow_matching_pair(
    source_latent: Tensor,
    target_latent: Tensor,
    source_time_days: Tensor | float,
    target_time_days: Tensor | float,
    *,
    flow_time: Tensor | float | None = None,
    time_scale_days: float = 365.2425,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Construct a conditional-flow-matching training pair for development.

    The source and target are *cross-sectional populations*.  ``target_latent``
    is normally the detached Sinkhorn barycentric target, so this helper never
    sends gradients into the observed target population.  For an interpolation
    coefficient :math:`u \in [0,1]`, it returns

    .. math::

       z_u=(1-u)z_s+u\bar z_t, \qquad
       v^*=\frac{\bar z_t-z_s}{(t_t-t_s)/T}.

    The tuple is ``(interpolated_latent, target_velocity, flow_time,
    interpolated_time_days)``.  ``T`` is ``time_scale_days`` (one biological
    year by default), matching :class:`ControlledLatentDynamics`' integration
    convention.
    """
    _validate_developmental_latents(source_latent, target_latent)
    if (
        isinstance(time_scale_days, bool)
        or not math.isfinite(float(time_scale_days))
        or float(time_scale_days) <= 0.0
    ):
        raise ValueError("time_scale_days must be finite and positive")
    batch_size = int(source_latent.shape[0])
    source_time = _developmental_batch_scalar(
        source_time_days,
        name="source_time_days",
        batch_size=batch_size,
        device=source_latent.device,
        dtype=source_latent.dtype,
    )
    target_time = _developmental_batch_scalar(
        target_time_days,
        name="target_time_days",
        batch_size=batch_size,
        device=source_latent.device,
        dtype=source_latent.dtype,
    )
    if torch.any(target_time <= source_time):
        raise ValueError("target_time_days must be strictly later than source_time_days")
    if flow_time is None:
        flow = torch.rand(
            (batch_size,), device=source_latent.device, dtype=source_latent.dtype
        )
    else:
        flow = _developmental_batch_scalar(
            flow_time,
            name="flow_time",
            batch_size=batch_size,
            device=source_latent.device,
            dtype=source_latent.dtype,
        )
    if torch.any((flow < 0.0) | (flow > 1.0)):
        raise ValueError("flow_time must lie in [0, 1]")

    # The endpoint is an observed/OT-derived target.  Detaching here makes the
    # supervision direction explicit even when a caller supplies a target with
    # ``requires_grad=True``; source gradients remain available for optional
    # joint fine-tuning of the encoder.
    endpoint = target_latent.detach()
    flow_column = flow[:, None]
    interpolated = (1.0 - flow_column) * source_latent + flow_column * endpoint
    elapsed = (target_time - source_time) / float(time_scale_days)
    target_velocity = (endpoint - source_latent) / elapsed[:, None]
    interpolated_time = (1.0 - flow) * source_time + flow * target_time
    if not torch.isfinite(interpolated).all() or not torch.isfinite(target_velocity).all():
        raise FloatingPointError("developmental flow-matching pair is non-finite")
    return interpolated, target_velocity, flow, interpolated_time


def developmental_flow_matching_target(
    source_latent: Tensor,
    target_latent: Tensor,
    source_time_days: Tensor | float,
    target_time_days: Tensor | float,
    *,
    time_scale_days: float = 365.2425,
) -> Tensor:
    """Return only the detached-OT developmental velocity target.

    This convenience wrapper is useful when the caller has already sampled an
    interpolation state and only needs the analytically defined target vector.
    ``flow_time`` does not affect the constant straight-line target velocity.
    """
    _, target_velocity, _, _ = developmental_flow_matching_pair(
        source_latent,
        target_latent,
        source_time_days,
        target_time_days,
        flow_time=0.0,
        time_scale_days=time_scale_days,
    )
    return target_velocity


def developmental_flow_matching_loss(
    prediction: Tensor,
    source_latent: Tensor,
    target_latent: Tensor,
    source_time_days: Tensor | float,
    target_time_days: Tensor | float,
    *,
    time_scale_days: float = 365.2425,
) -> Tensor:
    r"""Compare a predicted developmental drift with the OT target velocity.

    ``prediction`` must be the existing dynamics vector field evaluated at an
    interpolated state.  Keeping this loss separate from the Stage-0 prior
    ``flow_matching`` objective prevents an age-conditioned aggregate prior
    from being mistaken for a cross-sectional developmental transition.
    """
    _, target_velocity, _, _ = developmental_flow_matching_pair(
        source_latent,
        target_latent,
        source_time_days,
        target_time_days,
        flow_time=0.0,
        time_scale_days=time_scale_days,
    )
    return flow_matching_loss(prediction, target_velocity)


def partition_independence_loss(latent: Tensor,
                                partitions: dict[str, int]) -> Tensor:
    """Penalize cross-covariance while leaving within-partition structure free."""
    if latent.ndim != 2 or sum(partitions.values()) != latent.shape[1]:
        raise ValueError("partition widths must cover latent dimension")
    if latent.shape[0] < 2:
        return latent.new_zeros(())
    normalized = latent.float() - latent.float().mean(dim=0, keepdim=True)
    normalized = normalized / normalized.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    covariance = normalized.transpose(0, 1) @ normalized / latent.shape[0]
    cross_partition = torch.ones_like(covariance, dtype=torch.bool)
    start = 0
    for width in partitions.values():
        cross_partition[start:start + width, start:start + width] = False
        start += width
    return covariance[cross_partition].square().mean()


def dynamics_loss(predicted_z: Tensor, target_z: Tensor) -> Tensor:
    return torch.mean((predicted_z - target_z) ** 2)


def temporal_endpoint_loss(
    predicted_endpoint: Tensor,
    target_endpoint: Tensor,
    *,
    predicted_velocity: Tensor | None = None,
    target_velocity: Tensor | None = None,
    endpoint_weight: float = 1.0,
    velocity_weight: float = 0.0,
) -> Tensor:
    """scFiT-compatible endpoint/velocity objective for Stage-1.

    The wrapper keeps the public loss namespace aligned with the BrainWM stage
    vocabulary while delegating shape and finite-value checks to the maintained
    latent endpoint implementation.
    """
    return latent_endpoint_loss(
        predicted_endpoint,
        target_endpoint,
        predicted_velocity=predicted_velocity,
        target_velocity=target_velocity,
        endpoint_weight=endpoint_weight,
        velocity_weight=velocity_weight,
    )


def source7_population_expression_loss(
    prediction: Tensor,
    source: Tensor,
    target: Tensor,
    *,
    metric_mean: Tensor | None = None,
    metric_scale: Tensor | None = None,
    epsilon: float = 0.05,
    iterations: int = 30,
) -> Tensor:
    """Match the source-7 population-MSE target in expression space.

    The frozen source-7 evaluator first computes a balanced Sinkhorn plan from
    the observed source population to the observed target population, then
    scores a prediction against the source-indexed target barycenter in the
    fixed standardized panel.  The plan is deliberately detached here: target
    populations define supervision, while gradients flow only through the
    predicted expression.  Keeping this helper in the shared loss module makes
    the training objective auditable against ``EVAL-S7-POPULATION-V1``.
    """
    if prediction.ndim != 2 or source.ndim != 2 or target.ndim != 2:
        raise ValueError("source-7 expression tensors must have shape [cells, genes]")
    if prediction.shape != source.shape or target.shape[1] != source.shape[1]:
        raise ValueError("source-7 expression tensor shapes are incompatible")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("source-7 Sinkhorn epsilon/iterations must be positive")
    feature_count = int(source.shape[1])
    if metric_mean is None and metric_scale is None:
        mean = source.new_zeros(feature_count)
        scale = source.new_ones(feature_count)
    elif metric_mean is None or metric_scale is None:
        raise ValueError("metric_mean and metric_scale must be provided together")
    else:
        mean = metric_mean.to(device=source.device, dtype=torch.float32)
        scale = metric_scale.to(device=source.device, dtype=torch.float32)
        if mean.shape != (feature_count,) or scale.shape != (feature_count,):
            raise ValueError("source-7 metric assets must match the panel width")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("source-7 metric assets must be finite")
        if torch.any(scale <= 0):
            raise ValueError("source-7 metric scale must be positive")

    source_metric = (source.float() - mean) / scale
    target_metric = (target.float() - mean) / scale
    prediction_metric = (prediction.float() - mean) / scale
    # ``sinkhorn_transport_plan`` is already no-grad and uses the evaluator's
    # normalized cosine geometry.  Explicitly detach the inputs to document
    # that observed target values never receive gradients.
    plan = sinkhorn_transport_plan(
        source_metric.detach(),
        target_metric.detach(),
        epsilon=float(epsilon),
        iterations=int(iterations),
    )
    barycentric = barycentric_transport_target(plan, target_metric.detach())
    return F.mse_loss(prediction_metric, barycentric)


def spatial_adapter_consistency_loss(
    adapted_state: Tensor,
    reference_state: Tensor,
    *,
    weight: float = 1.0,
) -> Tensor:
    """Optional regularizer for the Stage-2 spatial residual branch."""
    if adapted_state.shape != reference_state.shape:
        raise ValueError("adapted_state and reference_state must have equal shape")
    if adapted_state.ndim != 2:
        raise ValueError("spatial adapter states must have shape [B, D]")
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    return float(weight) * F.mse_loss(adapted_state.float(), reference_state.float())


def spatial_adapter_smoothness_loss(
    delta: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
) -> Tensor:
    """Penalize abrupt adapter residuals across section-local neighbors.

    This is a distinct Stage-2 regularizer: it constrains the *new* spatial
    residual field, while the existing ``spatial_consistency_loss`` supervises
    the biological state transition itself.  An empty graph yields a
    connected zero so batches without enough neighbors remain trainable.
    """
    if delta.ndim != 2:
        raise ValueError("delta must have shape [B, D]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.numel() == 0:
        return delta.new_zeros(())
    src, dst = edge_index.to(device=delta.device, dtype=torch.long)
    if torch.any(src < 0) or torch.any(src >= delta.shape[0]) or torch.any(
        dst < 0
    ) or torch.any(dst >= delta.shape[0]):
        raise ValueError("edge_index contains an out-of-range node")
    error = (delta[src] - delta[dst]).float().square().mean(dim=-1)
    if edge_weight is None:
        return error.mean()
    if edge_weight.ndim != 1 or edge_weight.shape[0] != error.shape[0]:
        raise ValueError("edge_weight must have shape [E]")
    weights = edge_weight.to(device=error.device, dtype=error.dtype)
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("edge_weight must be finite and nonnegative")
    return torch.sum(error * weights) / weights.sum().clamp_min(1.0e-8)


def representation_anchor_loss(
    current_stage0: Tensor,
    frozen_stage0: Tensor,
    *,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.0,
) -> Tensor:
    """Expose the Stage-0 replay anchor through the unified loss module."""
    return stage0_representation_anchor_loss(
        current_stage0,
        frozen_stage0,
        mse_weight=mse_weight,
        cosine_weight=cosine_weight,
    )


def masked_gene_modeling_loss(logits: Tensor, target_index: Tensor) -> Tensor:
    """Cross-entropy over sampled gene candidates for masked tokens."""
    if logits.ndim != 2 or target_index.ndim != 1 or logits.shape[0] != target_index.shape[0]:
        raise ValueError("MGM logits/targets must have shapes [T, C] and [T]")
    if logits.shape[0] == 0:
        return logits.float().sum()
    return torch.nn.functional.cross_entropy(logits.float(), target_index.long())


def sampled_gene_candidates(target_gene: Tensor, n_genes: int,
                            negative_samples: int) -> tuple[Tensor, Tensor]:
    """Return candidate IDs and class labels for masked target gene IDs."""
    if target_gene.ndim != 1:
        raise ValueError("target_gene must have shape [T]")
    if n_genes < 1 or negative_samples < 1:
        raise ValueError("n_genes and negative_samples must be positive")
    if target_gene.numel() == 0:
        candidate = torch.arange(min(n_genes, negative_samples), device=target_gene.device)
        return candidate, target_gene.new_empty((0,), dtype=torch.long)
    target_gene = target_gene.long()
    if torch.any(target_gene < 0) or torch.any(target_gene >= n_genes):
        raise ValueError("target gene is outside vocabulary")
    candidate = torch.unique(target_gene, sorted=True)
    count = min(negative_samples, n_genes - int(candidate.numel()))
    if count:
        # A deterministic candidate set is unnecessary for training, but this
        # local generator makes smoke tests reproducible without changing the
        # caller's global RNG stream.
        generator = torch.Generator(device=target_gene.device)
        generator.manual_seed(17 + int(target_gene.numel()))
        negative = torch.randperm(n_genes, generator=generator,
                                  device=target_gene.device)[:count]
        negative = negative[~torch.isin(negative, candidate)]
        while negative.numel() < count:
            extra = torch.randint(n_genes, (count - negative.numel(),),
                                  generator=generator, device=target_gene.device)
            extra = extra[~torch.isin(extra, candidate)]
            negative = torch.unique(torch.cat([negative, extra]), sorted=True)
        candidate = torch.unique(torch.cat([candidate, negative[:count]]), sorted=True)
    labels = torch.searchsorted(candidate, target_gene)
    return candidate, labels


@torch.no_grad()
def sinkhorn_transport_plan(source: Tensor, target: Tensor,
                            epsilon: float = 0.05,
                            iterations: int = 30) -> Tensor:
    """Uniform entropic OT plan in normalized Stage-0 latent space."""
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("Sinkhorn inputs must have equal [B, D] shapes")
    if source.shape[0] < 1:
        raise ValueError("Sinkhorn inputs cannot be empty")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("Sinkhorn epsilon/iterations must be positive")
    first = F.normalize(source.float(), dim=-1)
    second = F.normalize(target.float(), dim=-1)
    cost = torch.cdist(first, second).square()
    log_kernel = -cost / epsilon
    batch = source.shape[0]
    log_mass = -torch.log(source.new_tensor(float(batch), dtype=torch.float32))
    log_u = torch.zeros(batch, device=source.device, dtype=torch.float32)
    log_v = torch.zeros_like(log_u)
    for _ in range(iterations):
        log_u = log_mass - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_mass - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    return torch.exp(log_kernel + log_u[:, None] + log_v[None, :])


def _padded_sinkhorn_transport_plan(
    source: Tensor,
    target: Tensor,
    active: Tensor,
    epsilon: float,
    iterations: int,
) -> Tensor:
    """Run the uniform OT updates on a fixed-size, masked population.

    ``source`` and ``target`` have a compile-time bucket size. ``active`` marks
    the real prefix; inactive rows/columns carry zero mass and therefore cannot
    affect any log-sum-exp update. This lets TorchInductor compile one graph per
    population bucket while preserving the unpadded plan exactly for the active
    submatrix.
    """
    first = F.normalize(source.float(), dim=-1)
    second = F.normalize(target.float(), dim=-1)
    cost = torch.cdist(first, second).square()
    log_kernel = -cost / epsilon
    negative_infinity = torch.full(
        (source.shape[0],), -torch.inf, device=source.device,
        dtype=torch.float32,
    )
    active = active.bool()
    log_mass = torch.where(
        active,
        -torch.log(active.sum().to(dtype=torch.float32)),
        negative_infinity,
    )
    log_u = torch.where(
        active, torch.zeros_like(log_mass), negative_infinity,
    )
    log_v = log_u
    for _ in range(iterations):
        log_u = log_mass - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0), dim=1,
        )
        log_u = torch.where(active, log_u, negative_infinity)
        log_v = log_mass - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1), dim=0,
        )
        log_v = torch.where(active, log_v, negative_infinity)
    return torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))


class SinkhornRuntime:
    """Optional compiled Sinkhorn executor with bounded shape-specialization.

    The ordinary helper above remains the reference implementation.  This
    runtime only compiles CUDA calls and keeps a small LRU of fixed ``(bucket,
    width, dtype, device)`` graphs, avoiding the unbounded recompilation that a
    fully dynamic ``torch.compile`` function causes for every population size.
    It is deliberately a plain Python object, so no compiled callable or cache
    entry is included in model parameters or checkpoints.
    """

    DEFAULT_BUCKETS = (8, 16, 32, 64)

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        iterations: int = 50,
        enabled: bool = False,
        mode: str = "reduce-overhead",
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
        max_cached_graphs: int = 8,
    ) -> None:
        if epsilon <= 0 or iterations < 1:
            raise ValueError("Sinkhorn epsilon and iterations must be positive")
        if mode not in {
            "default", "reduce-overhead", "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(f"unknown Sinkhorn compile mode: {mode!r}")
        normalized_buckets = tuple(sorted({int(value) for value in buckets}))
        if not normalized_buckets or normalized_buckets[0] < 1:
            raise ValueError("Sinkhorn compile buckets must be positive")
        if max_cached_graphs < 1:
            raise ValueError("max_cached_graphs must be positive")
        self.epsilon = float(epsilon)
        self.iterations = int(iterations)
        self.enabled = bool(enabled)
        self.mode = mode
        self.buckets = normalized_buckets
        self.max_cached_graphs = int(max_cached_graphs)
        self._compiled: OrderedDict[tuple, object] = OrderedDict()
        self.compile_count = 0
        self.fallback_count = 0
        self.last_bucket: int | None = None
        self.last_fallback: str | None = None

    def _bucket_for(self, population_size: int) -> int | None:
        for bucket in self.buckets:
            if population_size <= bucket:
                return bucket
        return None

    def _compiled_graph(
        self, bucket: int, width: int, source: Tensor,
    ) -> object:
        device = source.device
        key = (
            device.type, device.index, bucket, width, source.dtype,
            self.epsilon, self.iterations, self.mode,
        )
        cached = self._compiled.get(key)
        if cached is not None:
            self._compiled.move_to_end(key)
            return cached
        if getattr(torch, "compile", None) is None:
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")

        # Capture scalar hyperparameters and bucket shape in the closure. The
        # compiled graph then sees only tensors with stable dimensions.
        epsilon = self.epsilon
        iterations = self.iterations

        def fixed_shape(source_value, target_value, active_value):
            return _padded_sinkhorn_transport_plan(
                source_value, target_value, active_value, epsilon, iterations,
            )

        graph = compile_callable(
            "torch_compile",
            fixed_shape, mode=self.mode, fullgraph=True, dynamic=False,
        )
        self._compiled[key] = graph
        self._compiled.move_to_end(key)
        self.compile_count += 1
        while len(self._compiled) > self.max_cached_graphs:
            self._compiled.popitem(last=False)
        return graph

    def plan(self, source: Tensor, target: Tensor) -> Tensor:
        """Return an OT plan, falling back to the reference path on failure."""
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError("Sinkhorn inputs must have equal [B, D] shapes")
        if source.shape[0] < 1:
            raise ValueError("Sinkhorn inputs cannot be empty")
        if not self.enabled or not source.is_cuda:
            return sinkhorn_transport_plan(
                source, target, epsilon=self.epsilon, iterations=self.iterations,
            )
        population_size, width = int(source.shape[0]), int(source.shape[1])
        bucket = self._bucket_for(population_size)
        if bucket is None:
            self.fallback_count += 1
            self.last_fallback = "population_exceeds_compile_buckets"
            return sinkhorn_transport_plan(
                source, target, epsilon=self.epsilon, iterations=self.iterations,
            )
        try:
            graph = self._compiled_graph(bucket, width, source)
            if population_size == bucket:
                padded_source, padded_target = source, target
            else:
                padded_source = F.pad(source, (0, 0, 0, bucket - population_size))
                padded_target = F.pad(target, (0, 0, 0, bucket - population_size))
            active = torch.arange(bucket, device=source.device) < population_size
            result = graph(padded_source, padded_target, active)
            self.last_bucket = bucket
            self.last_fallback = None
            return result[:population_size, :population_size]
        except Exception as exc:
            self.fallback_count += 1
            self.last_fallback = type(exc).__name__
            import os
            if os.environ.get("BRAINWM_COMPILE_STRICT", "0") == "1":
                raise
            return sinkhorn_transport_plan(
                source, target, epsilon=self.epsilon, iterations=self.iterations,
            )

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "buckets": list(self.buckets),
            "cached_graphs": len(self._compiled),
            "compile_count": self.compile_count,
            "fallback_count": self.fallback_count,
            "last_bucket": self.last_bucket,
            "last_fallback": self.last_fallback,
        }


def barycentric_transport_target(plan: Tensor, target: Tensor) -> Tensor:
    if plan.ndim != 2 or target.ndim != 2 or plan.shape[1] != target.shape[0]:
        raise ValueError("transport plan and target shapes are incompatible")
    mass = plan.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (plan.to(target.dtype) @ target) / mass.to(target.dtype)


def multi_step_loss(predicted: list[Tensor], target: list[Tensor],
                    weights: Tensor | None = None) -> Tensor:
    if len(predicted) != len(target) or not predicted:
        raise ValueError("predicted/target trajectories must have equal nonzero length")
    losses = torch.stack([torch.mean((p - t) ** 2) for p, t in zip(predicted, target)])
    if weights is None:
        return losses.mean()
    return torch.sum(losses * weights / weights.sum().clamp_min(1.0e-8))


def semigroup_loss(direct: Tensor, composed: Tensor) -> Tensor:
    return torch.mean((direct - composed) ** 2)


def spatial_consistency_loss(predicted_z: Tensor, target_z: Tensor,
                             source_z: Tensor, edge_index: Tensor,
                             edge_weight: Tensor | None = None) -> Tensor:
    if edge_index.shape[1] == 0:
        return predicted_z.new_zeros(())
    src, dst = edge_index
    error = ((predicted_z[dst] - predicted_z[src])
             - (target_z[dst] - target_z[src]))
    error = torch.mean(error ** 2, dim=-1)
    if edge_weight is not None:
        return torch.sum(error * edge_weight) / edge_weight.sum().clamp_min(1.0e-8)
    # source_z is explicit in the signature to keep the paper's source graph
    # contract visible even though the loss compares predicted/target deltas.
    _ = source_z
    return error.mean()


def mmd_rbf(x: Tensor, y: Tensor,
            bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)) -> Tensor:
    xx = torch.cdist(x, x).square()
    yy = torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()
    result = x.new_zeros(())
    for bandwidth in bandwidths:
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        result = result + torch.exp(-gamma * xx).mean()
        result = result + torch.exp(-gamma * yy).mean()
        result = result - 2.0 * torch.exp(-gamma * xy).mean()
    return result / len(bandwidths)


def cross_species_alignment_loss(mapped_source: Tensor, target: Tensor) -> Tensor:
    return torch.mean((mapped_source - target) ** 2)


def cycle_consistency_loss(cycled: Tensor, source: Tensor) -> Tensor:
    return torch.mean((cycled - source) ** 2)


def intervention_effect_loss(predicted_treated: Tensor, predicted_control: Tensor,
                             observed_treated: Tensor, observed_control: Tensor) -> Tensor:
    predicted_effect = predicted_treated - predicted_control
    observed_effect = observed_treated - observed_control
    return torch.mean((predicted_effect - observed_effect) ** 2)


def regulatory_loss(predicted_activity: Tensor, target_activity: Tensor,
                    drift_jacobian: Tensor | None = None,
                    sparsity: float = 0.0) -> Tensor:
    loss = torch.mean((predicted_activity - target_activity) ** 2)
    if drift_jacobian is not None and sparsity:
        loss = loss + sparsity * drift_jacobian.abs().mean()
    return loss


@dataclass
class ObjectiveOutput:
    loss: Tensor
    terms: dict[str, Tensor]


class BrainWorldObjective:
    """Weighted full objective; unavailable supervision is omitted explicitly."""

    def __init__(self, config: LossConfig):
        self.config = config

    def __call__(self, terms: dict[str, Tensor]) -> ObjectiveOutput:
        weights = {
            "reconstruction": self.config.reconstruction,
            "context_neutral_reconstruction": (
                self.config.context_neutral_reconstruction
            ),
            "kl": self.config.kl,
            "contrastive": self.config.contrastive,
            "assay_adversarial": self.config.assay_adversarial,
            "flow_matching": self.config.flow_matching,
            "developmental_flow_matching": getattr(
                self.config, "developmental_flow_matching", 0.0
            ),
            "partition_independence": self.config.partition_independence,
            "dynamics": self.config.dynamics,
            "multi_step": self.config.multi_step,
            "semigroup": self.config.semigroup,
            "spatial": self.config.spatial,
            "population": self.config.population,
            "alignment": self.config.alignment,
            "cycle": self.config.cycle,
            "intervention_effect": self.config.intervention_effect,
            "regulatory": self.config.regulatory,
            "masked_gene_modeling": getattr(self.config, "masked_gene_modeling", 0.0),
            "latent_variance": getattr(self.config, "latent_variance", 0.0),
            "latent_covariance": getattr(self.config, "latent_covariance", 0.0),
            "expression_geometry": getattr(self.config, "expression_geometry", 0.0),
            "marker_expression_geometry": getattr(
                self.config, "marker_expression_geometry", 0.0
            ),
            "hierarchical_annotation": getattr(
                self.config, "hierarchical_annotation", 0.0
            ),
            "temporal_endpoint": getattr(self.config, "temporal_endpoint", 0.0),
            "temporal_velocity": getattr(self.config, "temporal_velocity", 0.0),
            "spatial_adapter": getattr(self.config, "spatial_adapter", 0.0),
            "representation_anchor": getattr(
                self.config, "representation_anchor", 0.0
            ),
            "expression_population": getattr(
                self.config, "expression_population", 0.0
            ),
        }
        unknown = set(terms) - set(weights)
        if unknown:
            raise KeyError(f"unknown objective terms: {sorted(unknown)}")
        if not terms:
            raise ValueError("at least one objective term is required")
        total = next(iter(terms.values())).new_zeros(())
        for name, value in terms.items():
            total = total + weights[name] * value
        return ObjectiveOutput(loss=total, terms=terms)
