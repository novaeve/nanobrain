"""Distributed Stage-1 training utilities for the indexed H5AD corpus."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn

from .blocks import row_index_from_ptr
from .config import ExperimentConfig
from .data import MetadataTensorizer
from .decoder import dense_counts_for_genes
from .losses import (
    BrainWorldObjective,
    assay_adversarial_loss,
    barlow_twins_loss,
    contrastive_loss,
    expression_geometry_loss,
    flow_matching_loss,
    latent_covariance_loss,
    latent_variance_loss,
    masked_gene_modeling_loss,
    negative_binomial_nll,
    partition_independence_loss,
    reconstruction_loss,
    variational_kl_loss,
)
from .modalities import mask_named_modalities
from .types import WorldModelBatch

# CUDA 0-1 remain reserved for other projects.  The current BrainWM
# enablement campaign is authorized to use the six physical devices 2-7.
ALLOWED_TRAINING_PHYSICAL_CUDA_IDS = frozenset(range(2, 8))


@dataclass(frozen=True)
class BatchDescriptor:
    source_id: int
    start: int
    stop: int


class IndexedBatchPlan:
    """Deterministic source-local batches that can be evenly sharded by rank."""

    def __init__(self, reader, split: str, batch_size: int, drop_last: bool,
                 sample_keys: set[str] | None = None,
                 cells: pd.DataFrame | None = None):
        """Build a plan over the reader index or an explicit cell cohort.

        ``cells`` is an opt-in source-local selection frame.  It is kept
        separate from ``reader.cells`` so a candidate cohort can reuse the
        backed H5AD reader while validation and historical callers continue to
        use the complete index unchanged.
        """
        cells = reader.cells if cells is None else cells
        cells = cells[cells["split"] == split]
        if sample_keys is not None:
            cells = cells[cells["sample_key"].isin(sample_keys)]
        self.frames: dict[int, pd.DataFrame] = {
            int(source_id): frame.sort_values("row_index").reset_index(drop=True)
            for source_id, frame in cells.groupby("source_id", sort=True)
        }
        descriptors: list[BatchDescriptor] = []
        for source_id, frame in self.frames.items():
            limit = len(frame)
            for start in range(0, limit, batch_size):
                stop = min(start + batch_size, limit)
                if drop_last and stop - start < batch_size:
                    continue
                descriptors.append(BatchDescriptor(source_id, start, stop))
        self.descriptors = descriptors
        self.source_cost = {
            int(source_id): int(reader.sources.loc[int(source_id), "n_vars"])
            for source_id in self.frames
        }

    def rank_descriptors(self, rank: int, world_size: int, seed: int,
                         shuffle: bool, maximum_per_rank: int = 0,
                         balance_by_cost: bool = True,
                         descriptors: list[BatchDescriptor] | None = None
                         ) -> list[BatchDescriptor]:
        """Return one deterministic rank shard.

        ``descriptors`` is an optional pre-built schedule.  It is used by
        exposure-balanced assay training to shard a globally balanced list
        without changing the historical default schedule.
        """
        descriptors = list(self.descriptors if descriptors is None else descriptors)
        usable = len(descriptors) - (len(descriptors) % world_size)
        if balance_by_cost and world_size > 1:
            descriptors.sort(
                key=lambda item: self.source_cost[item.source_id], reverse=True
            )
            descriptors = descriptors[:usable]
            rounds = [
                descriptors[start:start + world_size]
                for start in range(0, usable, world_size)
            ]
            if shuffle:
                rng = random.Random(seed)
                for group in rounds:
                    rng.shuffle(group)
                rng.shuffle(rounds)
            local = [group[rank] for group in rounds]
        else:
            if shuffle:
                random.Random(seed).shuffle(descriptors)
            descriptors = descriptors[:usable]
            local = descriptors[rank:usable:world_size]
        return local[:maximum_per_rank] if maximum_per_rank else local

    def frame(self, descriptor: BatchDescriptor) -> pd.DataFrame:
        return self.frames[descriptor.source_id].iloc[descriptor.start:descriptor.stop]


@lru_cache(maxsize=32)
def _load_marker_gene_indices(path: str) -> tuple[int, ...]:
    """Load and validate a source-defined marker gene set once per process."""
    marker_path = Path(path)
    if not marker_path.is_file():
        raise FileNotFoundError(f"marker gene set not found: {marker_path}")
    raw = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") != "brainwm_marker_gene_set_v1":
        raise ValueError("unsupported marker gene set format")
    values = raw.get("gene_index")
    if not isinstance(values, list) or not values:
        raise ValueError("marker gene set must contain a non-empty gene_index list")
    indices = tuple(sorted({int(value) for value in values}))
    if any(value < 0 for value in indices):
        raise ValueError("marker gene indices must be nonnegative")
    return indices


def _marker_gene_weight(config: ExperimentConfig, gene_index: Tensor) -> Tensor:
    """Return a measured-gene mask, falling back to all genes if needed."""
    marker_index = torch.as_tensor(
        _load_marker_gene_indices(config.data.marker_gene_index_path),
        dtype=torch.long,
        device=gene_index.device,
    )
    weight = torch.isin(gene_index, marker_index).float()
    # Some source assays measure only a tiny subset of the shared marker panel.
    # In that case the unweighted geometry objective is the better-defined
    # signal and avoids a one-gene cosine target.
    if int(torch.count_nonzero(weight)) < 2:
        return torch.ones_like(weight)
    return weight


def descriptor_group_counts(
    plan: IndexedBatchPlan, sample_groups: pd.Series
) -> dict[BatchDescriptor, dict[str, int]]:
    """Count sample-level groups in every descriptor without changing the plan."""
    lookup = {str(key): str(value) for key, value in sample_groups.items()}
    by_source: dict[int, list[BatchDescriptor]] = {}
    for descriptor in plan.descriptors:
        by_source.setdefault(descriptor.source_id, []).append(descriptor)

    profiles: dict[BatchDescriptor, dict[str, int]] = {}
    for source_id, descriptors in by_source.items():
        frame = plan.frames[source_id]
        groups = frame["sample_key"].astype(str).map(lookup)
        if groups.isna().any():
            missing = sorted(set(frame.loc[groups.isna(), "sample_key"].astype(str)))
            raise ValueError(
                f"missing descriptor group for sample keys: {missing[:5]}"
            )
        unique = groups.unique().tolist()
        if len(unique) == 1:
            group = str(unique[0])
            for descriptor in descriptors:
                profiles[descriptor] = {group: descriptor.stop - descriptor.start}
            continue
        for descriptor in descriptors:
            counts = groups.iloc[descriptor.start:descriptor.stop].value_counts()
            profiles[descriptor] = {
                str(group): int(count) for group, count in counts.items()
            }
    return profiles


def round_robin_descriptors(
    descriptors: list[BatchDescriptor],
    group_counts: dict[BatchDescriptor, dict[str, int]],
    group_order: list[str],
) -> list[BatchDescriptor]:
    """Interleave descriptor groups while preserving their shuffled order."""
    if len(set(group_order)) != len(group_order) or not group_order:
        raise ValueError("group_order must contain unique group names")
    order_index = {group: index for index, group in enumerate(group_order)}
    buckets: dict[str, list[BatchDescriptor]] = {group: [] for group in group_order}
    for descriptor in descriptors:
        counts = group_counts.get(descriptor)
        if not counts:
            raise ValueError(f"missing group counts for descriptor {descriptor}")
        unknown = set(counts) - set(order_index)
        if unknown:
            raise ValueError(f"descriptor contains unknown groups: {sorted(unknown)}")
        dominant = max(
            group_order,
            key=lambda group: (counts.get(group, 0), -order_index[group]),
        )
        buckets[dominant].append(descriptor)

    result: list[BatchDescriptor] = []
    positions = {group: 0 for group in group_order}
    while len(result) < len(descriptors):
        for group in group_order:
            position = positions[group]
            if position < len(buckets[group]):
                result.append(buckets[group][position])
                positions[group] = position + 1
    return result


def balanced_group_descriptors(
    descriptors: list[BatchDescriptor],
    group_counts: dict[BatchDescriptor, dict[str, int]],
    group_order: list[str],
    *,
    total_descriptors: int | None = None,
    seed: int = 0,
) -> list[BatchDescriptor]:
    """Build a deterministic equal-exposure schedule from source-local batches.

    Each descriptor is assigned to its dominant group (ties use
    ``group_order``).  The returned schedule contains the same number of
    descriptors for every requested group.  Groups with fewer batches are
    cycled (oversampled); groups with more batches are sampled without
    replacement for the current epoch.  Thus the full corpus remains
    available across epochs while one large assay cannot dominate a single
    epoch.  The default schedule length is the largest value not exceeding
    the input length that is divisible by the number of groups; callers can
    request another length explicitly.

    This helper balances *training exposure*, not biological labels or
    expression values.  It never moves rows between sources and never reads
    target annotations.
    """
    if not descriptors:
        raise ValueError("cannot balance an empty descriptor list")
    if not group_order or len(set(group_order)) != len(group_order):
        raise ValueError("group_order must contain unique group names")
    if total_descriptors is None:
        total_descriptors = len(descriptors)
    if isinstance(total_descriptors, bool) or total_descriptors < 1:
        raise ValueError("total_descriptors must be positive")
    n_groups = len(group_order)
    usable = (int(total_descriptors) // n_groups) * n_groups
    if usable < n_groups:
        raise ValueError(
            "total_descriptors is too small for one batch per requested group"
        )
    order_index = {group: index for index, group in enumerate(group_order)}
    buckets: dict[str, list[BatchDescriptor]] = {
        group: [] for group in group_order
    }
    for descriptor in descriptors:
        counts = group_counts.get(descriptor)
        if not counts:
            raise ValueError(f"missing group counts for descriptor {descriptor}")
        unknown = set(counts) - set(order_index)
        if unknown:
            raise ValueError(f"descriptor contains unknown groups: {sorted(unknown)}")
        dominant = max(
            group_order,
            key=lambda group: (counts.get(group, 0), -order_index[group]),
        )
        if counts.get(dominant, 0) <= 0:
            raise ValueError(f"descriptor has no requested group: {descriptor}")
        buckets[dominant].append(descriptor)
    empty = [group for group, bucket in buckets.items() if not bucket]
    if empty:
        raise ValueError(f"requested groups have no descriptors: {empty}")

    per_group = usable // n_groups
    selected: dict[str, list[BatchDescriptor]] = {}
    for group_index, group in enumerate(group_order):
        bucket = list(buckets[group])
        random.Random(int(seed) + group_index).shuffle(bucket)
        if len(bucket) >= per_group:
            chosen = bucket[:per_group]
        else:
            # Cycling is intentional: minority assays receive repeated
            # exposure, while the next epoch's seed rotates the order.
            chosen = [bucket[index % len(bucket)] for index in range(per_group)]
            random.Random(int(seed) + 1009 + group_index).shuffle(chosen)
        selected[group] = chosen

    result: list[BatchDescriptor] = []
    for index in range(per_group):
        for group in group_order:
            result.append(selected[group][index])
    return result


def validation_rank_descriptors(
    descriptors: list[BatchDescriptor],
    *,
    rank: int,
    world_size: int,
    seed: int,
    maximum_per_rank: int,
    group_counts: dict[BatchDescriptor, dict[str, int]] | None = None,
    requested_groups: list[str] | None = None,
) -> list[BatchDescriptor]:
    """Build one globally balanced validation window, then shard it by rank.

    Shuffling and optional round-robin happen before the global window is
    truncated. This prevents every rank from independently selecting the same
    leading groups. Requested groups fail closed when the complete plan or the
    selected window cannot represent them with equal descriptor counts.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    if maximum_per_rank < 0:
        raise ValueError("maximum_per_rank must be >= 0")
    groups = list(requested_groups or [])
    if bool(groups) != (group_counts is not None):
        raise ValueError(
            "group_counts and requested_groups must be provided together"
        )

    ordered = list(descriptors)
    random.Random(seed).shuffle(ordered)
    if groups:
        assert group_counts is not None
        present = {
            group
            for descriptor in ordered
            for group, count in group_counts.get(descriptor, {}).items()
            if count > 0
        }
        missing = [group for group in groups if group not in present]
        if missing:
            raise ValueError(
                "requested validation groups have no descriptors: "
                f"{missing}"
            )
        ordered = round_robin_descriptors(ordered, group_counts, groups)

    usable = len(ordered) - len(ordered) % world_size
    requested_window = (
        maximum_per_rank * world_size if maximum_per_rank else usable
    )
    window_size = min(requested_window, usable)
    if window_size < world_size:
        raise ValueError(
            "validation plan has fewer usable descriptors than DDP ranks"
        )
    window = ordered[:window_size]

    if groups:
        assert group_counts is not None
        group_order = {group: index for index, group in enumerate(groups)}
        selected = {group: 0 for group in groups}
        for descriptor in window:
            counts = group_counts[descriptor]
            dominant = max(
                groups,
                key=lambda group: (
                    counts.get(group, 0), -group_order[group]
                ),
            )
            selected[dominant] += 1
        if len(set(selected.values())) != 1:
            raise ValueError(
                "validation window cannot exactly balance requested groups: "
                f"{selected}"
            )

    return window[rank:window_size:world_size]


def validate_training_cuda_visible_devices(
    requested_device: str | torch.device = "cuda",
) -> tuple[int, ...]:
    """Require an auditable physical GPU mapping before CUDA training."""
    device = torch.device(requested_device)
    if device.type != "cuda":
        return ()

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        raise RuntimeError(
            "CUDA training requires CUDA_VISIBLE_DEVICES to explicitly list "
            "allowed physical GPU IDs 2-7; the variable is unset"
        )
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise RuntimeError(
            "CUDA training requires a non-empty CUDA_VISIBLE_DEVICES list of "
            "physical GPU IDs 2-7"
        )
    try:
        physical_ids = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain only comma-separated physical "
            f"integer GPU IDs 2-7; got {raw!r}"
        ) from exc
    if len(set(physical_ids)) != len(physical_ids):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must not repeat physical GPU IDs; got {raw!r}"
        )
    disallowed = sorted(set(physical_ids) - ALLOWED_TRAINING_PHYSICAL_CUDA_IDS)
    if disallowed:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES contains disallowed physical GPU IDs "
            f"{disallowed}; allowed IDs are 2-7"
        )
    return physical_ids


def resolve_training_device(
    requested_device: str | torch.device = "cuda",
    *,
    allow_cuda_fallback: bool = False,
) -> torch.device:
    """Resolve a training device while enforcing the physical CUDA allowlist."""
    device = torch.device(requested_device)
    if device.type != "cuda":
        return device

    validate_training_cuda_visible_devices(device)
    cuda_available = torch.cuda.is_available()
    if allow_cuda_fallback and not cuda_available:
        return torch.device("cpu")
    if not cuda_available:
        raise RuntimeError("training device is CUDA but CUDA is unavailable")
    return device


def distributed_context(requested_device: str = "cuda") -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = torch.device(requested_device)
    if world_size > 1:
        if requested.type != "cuda":
            raise ValueError("multi-process DDP training requires training.device=cuda")
        resolve_training_device(requested)
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_training_device(requested)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", 0)
    return rank, world_size, local_rank, device


def freeze_for_stage(model: nn.Module, components: list[str]) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module = model.module if hasattr(model, "module") else model
    aliases = {
        # The gene adapter is registered below ``encoder`` to avoid duplicate
        # state-dict keys, but it is still an independently trainable stage
        # component from the caller's perspective.
        "scuniflow_gene_adapter": "encoder.scuniflow_gene_adapter",
        "scbrain_depth_head": "annotation_head",
        "scfit_temporal_adapter": "scfit_temporal_adapter",
        "scuniflow_spatial_adapter": "scuniflow_spatial_adapter",
        "modality_adapters": "encoder.modality_router",
    }
    for name in components:
        path = aliases.get(name, name)
        target = module
        for part in path.split("."):
            target = getattr(target, part, None)
            if target is None:
                raise ValueError(
                    f"requested training component {name!r} is not enabled"
                )
        if not isinstance(target, nn.Module):
            raise ValueError(
                f"requested training component {name!r} is not a module"
            )
        for parameter in target.parameters():
            parameter.requires_grad_(True)
        if name == "modality_adapters":
            # The scalar gate is intentionally kept outside the router module
            # so enabling the branch does not alter legacy state-dict names.
            gate = getattr(getattr(module, "encoder", None), "modality_gate", None)
            if gate is None:
                raise ValueError(
                    "requested training component 'modality_adapters' is not enabled"
                )
            gate.requires_grad_(True)
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


def cosine_warmup_scheduler(optimizer: torch.optim.Optimizer,
                            warmup_steps: int, total_steps: int,
                            minimum_ratio: float):
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def precision_settings(config: ExperimentConfig, device: torch.device) -> tuple[bool, torch.dtype, bool]:
    precision = config.training.precision
    enabled = bool(config.training.amp and device.type == "cuda" and precision != "fp32")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_scaler = enabled and precision == "fp16"
    return enabled, dtype, use_scaler


def reduce_sum(value: Tensor, world_size: int) -> Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _budget_preserving_probabilities(
    weights: Tensor,
    probability: float,
    iterations: int = 32,
) -> Tensor:
    """Scale positive weights to a fixed mean probability after clipping."""
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if not 0.0 <= probability < 1.0:
        raise ValueError("mask probability must be in [0, 1)")
    if weights.numel() == 0 or probability == 0.0:
        return torch.zeros_like(weights)
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("weights must be finite and positive")
    target = weights.new_tensor(probability)
    lower = weights.new_zeros(())
    upper = target / weights.mean().clamp_min(torch.finfo(weights.dtype).tiny)
    upper = upper.clamp_min(torch.finfo(weights.dtype).eps)
    while bool(torch.clamp(upper * weights, max=1.0).mean() < target):
        upper = upper * 2.0
    for _ in range(iterations):
        midpoint = (lower + upper) * 0.5
        if bool(torch.clamp(midpoint * weights, max=1.0).mean() < target):
            lower = midpoint
        else:
            upper = midpoint
    return torch.clamp(((lower + upper) * 0.5) * weights, max=1.0)


def masked_expression_view(batch: WorldModelBatch,
                           probability: float,
                           mask_gene_ids: bool = False,
                           depth_aware_alpha: float = 0.0,
                           depth_aware_preserve_budget: bool = False,
                           ) -> tuple[WorldModelBatch, Tensor]:
    """Zero a random subset of observed token values without changing targets."""
    if not 0.0 <= probability < 1.0:
        raise ValueError("mask probability must be in [0, 1)")
    if not math.isfinite(depth_aware_alpha) or depth_aware_alpha < 0.0:
        raise ValueError("depth_aware_alpha must be finite and nonnegative")
    count = batch.expression.count.clone()
    eligible = torch.ones(count.shape, dtype=torch.bool, device=count.device)
    if batch.expression.missing_token_mask is not None:
        # An unmeasured placeholder is already a null query.  It must not be
        # relabeled as an active MGM mask, otherwise the two semantics become
        # indistinguishable at the encoder boundary.
        eligible &= ~batch.expression.missing_token_mask
    if depth_aware_alpha == 0.0:
        mask = (torch.rand(count.shape, device=count.device) < probability) & eligible
    else:
        row_index = row_index_from_ptr(batch.expression.row_ptr)
        library = batch.expression.library_size.float().clamp_min(1.0)
        observed = library[library > 0]
        median = observed.median().clamp_min(1.0) if observed.numel() else library.new_tensor(1.0)
        scale = (library[row_index] / median).pow(depth_aware_alpha)
        if depth_aware_preserve_budget:
            token_probability = _budget_preserving_probabilities(
                scale, probability
            )
        else:
            token_probability = (probability * scale).clamp_(0.0, 1.0)
        mask = (torch.rand(count.shape, device=count.device) < token_probability) & eligible

        # Keep at least one observed positive token per cell so depth-aware
        # masking cannot turn a low-depth cell into an all-zero input.
        positive = count > 0
        for row in range(batch.expression.batch_size):
            start = int(batch.expression.row_ptr[row])
            stop = int(batch.expression.row_ptr[row + 1])
            positive_indices = torch.nonzero(
                positive[start:stop], as_tuple=False
            ).flatten()
            if positive_indices.numel() and bool(mask[start:stop][positive_indices].all()):
                mask[start + int(positive_indices[0])] = False
    count.masked_fill_(mask, 0.0)
    token_mask = batch.expression.token_mask
    if mask_gene_ids:
        token_mask = mask
    expression = replace(batch.expression, count=count, token_mask=token_mask)
    return replace(batch, expression=expression), mask


def restrict_expression_to_genes(batch: WorldModelBatch,
                                 allowed_gene_index: Tensor) -> WorldModelBatch:
    """Physically retain only tokens from one batch-shared measured panel."""
    expression = batch.expression
    if allowed_gene_index.ndim != 1:
        raise ValueError("allowed_gene_index must be one-dimensional")
    allowed = torch.unique(
        allowed_gene_index.to(
            device=expression.gene_index.device,
            dtype=expression.gene_index.dtype,
        )
    )
    keep = torch.isin(expression.gene_index, allowed)
    row_index = row_index_from_ptr(expression.row_ptr)
    kept_rows = row_index[keep]
    row_lengths = torch.bincount(
        kept_rows, minlength=expression.batch_size
    )
    row_ptr = torch.zeros_like(expression.row_ptr)
    row_ptr[1:] = torch.cumsum(row_lengths, dim=0)
    count = expression.count[keep]
    library_size = torch.zeros_like(expression.library_size)
    library_size.scatter_add_(0, kept_rows, count)
    measured_keep = torch.isin(expression.measured_gene_index, allowed)
    token_mask = (
        expression.token_mask[keep]
        if expression.token_mask is not None else None
    )
    missing_token_mask = (
        expression.missing_token_mask[keep]
        if expression.missing_token_mask is not None else None
    )
    measured_zero_mask = (
        expression.measured_zero_mask[keep]
        if expression.measured_zero_mask is not None else None
    )
    restricted = replace(
        expression,
        row_ptr=row_ptr,
        gene_index=expression.gene_index[keep],
        count=count,
        library_size=library_size,
        measured_gene_index=expression.measured_gene_index[measured_keep],
        token_mask=token_mask,
        missing_token_mask=missing_token_mask,
        measured_zero_mask=measured_zero_mask,
        panel_missing_summary=None,
    )
    return replace(batch, expression=restricted)


def panel_expression_view(batch: WorldModelBatch, minimum_genes: int,
                          maximum_genes: int,
                          generator: torch.Generator | None = None,
                          include_unobserved: bool = False,
                          ) -> tuple[WorldModelBatch, Tensor]:
    """Sample one measured gene panel and apply it to every cell in the batch.

    The historical contrastive view samples from the observed nonzero support.
    ``include_unobserved=True`` instead samples from the source-measured panel;
    a separate zero-token view can then make measured zeros explicit while
    leaving genuinely unmeasured genes absent.
    """
    if minimum_genes < 1 or maximum_genes < minimum_genes:
        raise ValueError("panel gene limits must satisfy 1 <= minimum <= maximum")
    candidates = (
        torch.unique(batch.expression.measured_gene_index)
        if include_unobserved
        else torch.unique(batch.expression.gene_index)
    )
    if not candidates.numel():
        return restrict_expression_to_genes(batch, candidates), candidates
    upper = min(maximum_genes, int(candidates.numel()))
    lower = min(minimum_genes, upper)
    if lower == upper:
        panel_size = upper
    else:
        panel_size = int(torch.randint(
            lower,
            upper + 1,
            (),
            device=candidates.device,
            generator=generator,
        ))
    permutation = torch.randperm(
            candidates.numel(), device=candidates.device, generator=generator
        )
    panel = candidates[permutation[:panel_size]].sort().values
    return restrict_expression_to_genes(batch, panel), panel


def measured_zero_token_view(
    batch: WorldModelBatch,
    maximum_tokens_per_cell: int,
    *,
    generator: torch.Generator | None = None,
) -> WorldModelBatch:
    """Append sampled measured-but-undetected genes as zero-valued tokens.

    The source-level ``measured_gene_index`` is authoritative for the panel.
    Only genes absent from a cell's nonzero support are added, so this view
    distinguishes a measured zero from an unmeasured gene without materializing
    a dense vocabulary-sized matrix.
    """
    if maximum_tokens_per_cell < 0:
        raise ValueError("maximum_tokens_per_cell must be nonnegative")
    if maximum_tokens_per_cell == 0:
        return batch
    expression = batch.expression
    measured = torch.unique(
        expression.measured_gene_index.to(
            device=expression.gene_index.device,
            dtype=expression.gene_index.dtype,
        )
    ).sort().values
    if not measured.numel():
        return batch

    row_index = row_index_from_ptr(expression.row_ptr)
    additions: list[Tensor] = []
    for row in range(expression.batch_size):
        start = int(expression.row_ptr[row])
        stop = int(expression.row_ptr[row + 1])
        observed = expression.gene_index[start:stop]
        if observed.numel():
            candidates = measured[~torch.isin(measured, observed)]
        else:
            candidates = measured
        if not candidates.numel():
            additions.append(candidates)
            continue
        take = min(maximum_tokens_per_cell, int(candidates.numel()))
        if take < candidates.numel():
            order = torch.randperm(
                candidates.numel(), device=candidates.device, generator=generator
            )[:take]
            candidates = candidates[order].sort().values
        additions.append(candidates)

    lengths = torch.tensor(
        [int(value.numel()) for value in additions],
        dtype=expression.row_ptr.dtype,
        device=expression.row_ptr.device,
    )
    if not torch.any(lengths):
        return batch
    add_genes = torch.cat(additions)
    add_counts = torch.zeros(
        add_genes.shape, dtype=expression.count.dtype, device=expression.count.device
    )
    add_rows = torch.repeat_interleave(
        torch.arange(expression.batch_size, device=expression.row_ptr.device), lengths
    )
    new_row_index = torch.cat([row_index, add_rows])
    new_gene_index = torch.cat([expression.gene_index, add_genes])
    new_count = torch.cat([expression.count, add_counts])
    order = torch.argsort(new_row_index, stable=True)
    new_row_index = new_row_index[order]
    new_gene_index = new_gene_index[order]
    new_count = new_count[order]
    row_lengths = torch.bincount(
        new_row_index, minlength=expression.batch_size
    )
    row_ptr = torch.zeros_like(expression.row_ptr)
    row_ptr[1:] = torch.cumsum(row_lengths, dim=0)
    token_mask = None
    if expression.token_mask is not None:
        token_mask = torch.cat([
            expression.token_mask.to(device=add_genes.device),
            torch.zeros(add_genes.shape, dtype=torch.bool, device=add_genes.device),
        ])[order]
    old_missing = (
        expression.missing_token_mask.to(device=add_genes.device)
        if expression.missing_token_mask is not None
        else torch.zeros(
            expression.gene_index.shape, dtype=torch.bool, device=add_genes.device
        )
    )
    old_zero = (
        expression.measured_zero_mask.to(device=add_genes.device)
        if expression.measured_zero_mask is not None
        else torch.zeros(
            expression.gene_index.shape, dtype=torch.bool, device=add_genes.device
        )
    )
    missing_token_mask = torch.cat([
        old_missing,
        torch.zeros(add_genes.shape, dtype=torch.bool, device=add_genes.device),
    ])[order]
    measured_zero_mask = torch.cat([
        old_zero,
        torch.ones(add_genes.shape, dtype=torch.bool, device=add_genes.device),
    ])[order]
    augmented = replace(
        expression,
        row_ptr=row_ptr,
        gene_index=new_gene_index,
        count=new_count,
        token_mask=token_mask,
        missing_token_mask=missing_token_mask,
        measured_zero_mask=measured_zero_mask,
        panel_missing_summary=None,
    )
    return replace(batch, expression=augmented)


def unmeasured_gene_token_view(
    batch: WorldModelBatch,
    maximum_tokens_per_cell: int,
    *,
    n_genes: int | None = None,
    candidate_gene_index: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> WorldModelBatch:
    """Append bounded placeholders for genes absent from the input panel.

    The sparse expression contract stores only nonzero measured genes and a
    batch-level ``measured_gene_index`` panel.  This helper materializes a
    small, per-cell query set from the complement of that panel.  Each added
    token has zero count but is marked by ``missing_token_mask``; it therefore
    cannot be confused with a measured zero or with an active MGM mask.  The
    budget is deliberately bounded so a 60k-gene vocabulary never becomes a
    dense per-cell expansion.

    ``candidate_gene_index`` can provide an explicit global vocabulary (or a
    held-out target panel) and takes precedence over ``n_genes``.  When neither
    is supplied, the smallest vocabulary covering the current indices is used;
    callers training a true completion objective should pass ``n_genes``.
    """
    if isinstance(maximum_tokens_per_cell, bool) or maximum_tokens_per_cell < 0:
        raise ValueError("maximum_tokens_per_cell must be a nonnegative integer")
    if maximum_tokens_per_cell == 0:
        return batch
    expression = batch.expression
    device = expression.gene_index.device
    dtype = expression.gene_index.dtype
    measured = torch.unique(expression.measured_gene_index.to(device=device, dtype=dtype))
    if candidate_gene_index is None:
        if n_genes is None:
            max_index = -1
            if expression.gene_index.numel():
                max_index = max(max_index, int(expression.gene_index.max()))
            if expression.measured_gene_index.numel():
                max_index = max(max_index, int(expression.measured_gene_index.max()))
            if max_index < 0:
                # There is no way to infer a global complement from a wholly
                # empty sparse batch; leave it unchanged unless the caller
                # supplies an explicit vocabulary.
                return batch
            n_genes = max_index + 1
        if isinstance(n_genes, bool) or int(n_genes) < 1:
            raise ValueError("n_genes must be a positive integer")
        candidates_global = torch.arange(int(n_genes), device=device, dtype=dtype)
    else:
        if candidate_gene_index.ndim != 1:
            raise ValueError("candidate_gene_index must be one-dimensional")
        if (
            candidate_gene_index.is_floating_point()
            or candidate_gene_index.is_complex()
        ) and not torch.all(
            candidate_gene_index == candidate_gene_index.to(dtype=torch.long)
        ):
            raise ValueError("candidate_gene_index must contain integer values")
        candidates_global = torch.unique(
            candidate_gene_index.to(device=device, dtype=dtype)
        )
        if n_genes is not None and (
            isinstance(n_genes, bool) or int(n_genes) < 1
        ):
            raise ValueError("n_genes must be a positive integer")
    if candidates_global.numel() == 0:
        return batch
    if torch.any(candidates_global < 0):
        raise ValueError("candidate_gene_index cannot contain negative values")
    if n_genes is not None and torch.any(candidates_global >= int(n_genes)):
        raise ValueError("candidate_gene_index is outside the configured vocabulary")
    # A gene is unmeasured whenever it is outside the current panel.  Exclude
    # any malformed pre-existing token as well, preventing duplicate CSR genes.
    candidates_global = candidates_global[~torch.isin(candidates_global, measured)]
    if not candidates_global.numel():
        return batch

    row_index = row_index_from_ptr(expression.row_ptr)
    additions: list[Tensor] = []
    for row in range(expression.batch_size):
        start = int(expression.row_ptr[row])
        stop = int(expression.row_ptr[row + 1])
        observed = expression.gene_index[start:stop]
        candidates = candidates_global
        if observed.numel():
            candidates = candidates[~torch.isin(candidates, observed)]
        if not candidates.numel():
            additions.append(candidates)
            continue
        take = min(int(maximum_tokens_per_cell), int(candidates.numel()))
        if take < candidates.numel():
            order = torch.randperm(
                candidates.numel(), device=device, generator=generator
            )[:take]
            candidates = candidates[order]
        additions.append(candidates.sort().values)

    lengths = torch.tensor(
        [int(value.numel()) for value in additions],
        dtype=expression.row_ptr.dtype,
        device=expression.row_ptr.device,
    )
    if not torch.any(lengths):
        return batch
    add_genes = torch.cat(additions)
    add_counts = torch.zeros(
        add_genes.shape, dtype=expression.count.dtype, device=expression.count.device
    )
    add_rows = torch.repeat_interleave(
        torch.arange(expression.batch_size, device=expression.row_ptr.device), lengths
    )
    new_row_index = torch.cat([row_index, add_rows])
    new_gene_index = torch.cat([expression.gene_index, add_genes])
    new_count = torch.cat([expression.count, add_counts])
    order = torch.argsort(new_row_index, stable=True)
    new_row_index = new_row_index[order]
    new_gene_index = new_gene_index[order]
    new_count = new_count[order]
    row_lengths = torch.bincount(
        new_row_index, minlength=expression.batch_size
    )
    row_ptr = torch.zeros_like(expression.row_ptr)
    row_ptr[1:] = torch.cumsum(row_lengths, dim=0)

    # Materialize all masks once an explicit placeholder exists.  False
    # entries for legacy tokens retain their prior semantics.
    old_missing = (
        expression.missing_token_mask.to(device=device)
        if expression.missing_token_mask is not None
        else torch.zeros(
            expression.gene_index.shape, dtype=torch.bool, device=device
        )
    )
    old_zero = (
        expression.measured_zero_mask.to(device=device)
        if expression.measured_zero_mask is not None
        else torch.zeros(
            expression.gene_index.shape, dtype=torch.bool, device=device
        )
    )
    missing_mask = torch.cat([
        old_missing,
        torch.ones(add_genes.shape, dtype=torch.bool, device=device),
    ])[order]
    zero_mask = torch.cat([
        old_zero,
        torch.zeros(add_genes.shape, dtype=torch.bool, device=device),
    ])[order]
    token_mask = None
    if expression.token_mask is not None:
        token_mask = torch.cat([
            expression.token_mask.to(device=device),
            torch.zeros(add_genes.shape, dtype=torch.bool, device=device),
        ])[order]
    augmented = replace(
        expression,
        row_ptr=row_ptr,
        gene_index=new_gene_index,
        count=new_count,
        token_mask=token_mask,
        missing_token_mask=missing_mask,
        measured_zero_mask=zero_mask,
        panel_missing_summary=None,
    )
    return replace(batch, expression=augmented)


def neutralized_context_view(batch: WorldModelBatch) -> WorldModelBatch:
    """Remove categorical and continuous metadata from an encoder view."""
    return replace(
        batch,
        categorical_context=torch.zeros_like(batch.categorical_context),
        continuous_context=torch.zeros_like(batch.continuous_context),
    )


def masked_cell_view(batch: WorldModelBatch, probability: float,
                     seed: int | None = None) -> tuple[WorldModelBatch, Tensor]:
    """Hide complete cells while preserving coordinates and target counts."""
    if not 0.0 < probability < 1.0:
        raise ValueError("cell mask probability must be in (0, 1)")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=batch.expression.count.device)
        generator.manual_seed(seed)
    cells = torch.rand(
        batch.expression.batch_size,
        device=batch.expression.count.device,
        generator=generator,
    ) < probability
    if not torch.any(cells):
        cells[0] = True
    row_index = row_index_from_ptr(batch.expression.row_ptr)
    token_mask = cells[row_index]
    count = batch.expression.count.clone()
    count.masked_fill_(token_mask, 0.0)
    expression = replace(batch.expression, count=count, token_mask=token_mask)
    modality_inputs, modality_presence = mask_named_modalities(
        batch.modality_inputs,
        batch.modality_presence,
        cells,
    )
    return replace(
        batch,
        expression=expression,
        modality_inputs=modality_inputs,
        modality_presence=modality_presence,
    ), cells


def _sample_mgm_mask(
    positive_tokens: Tensor,
    probability: float,
    maximum_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Sample MGM targets from the original positive-token support.

    The returned scalar is the number selected before the deterministic-size
    cap.  Keeping it separate makes an over-active cap visible in run metrics.
    """
    if not torch.any(positive_tokens):
        return (
            torch.zeros_like(positive_tokens, dtype=torch.bool),
            torch.zeros((), device=positive_tokens.device, dtype=torch.long),
        )
    mask = (
        torch.rand(positive_tokens.shape, device=positive_tokens.device)
        < probability
    ) & positive_tokens
    if not torch.any(mask):
        positive = torch.nonzero(positive_tokens, as_tuple=False).flatten()
        mask[positive[0]] = True
    candidates = mask.sum()
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    if selected.numel() > maximum_tokens:
        selected = selected[
            torch.randperm(selected.numel(), device=selected.device)[
                :maximum_tokens
            ]
        ]
        mask = torch.zeros_like(mask)
        mask[selected] = True
    return mask, candidates


def _deterministic_spread_latent_mean(
    model: nn.Module,
    batch: WorldModelBatch,
) -> Tensor:
    """Encode one view without dropout while retaining gradients and RNG state."""
    module = model.module if hasattr(model, "module") else model
    encoder = module.encoder
    training_states = [
        (submodule, submodule.training) for submodule in encoder.modules()
    ]
    device = batch.expression.count.device
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    try:
        encoder.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            encoder_kwargs = {}
            # Keep compatibility with lightweight legacy encoder test doubles
            # that predate named modalities, while ensuring a supplied mapping
            # is never silently dropped from the deterministic auxiliary view.
            if (
                batch.modality_inputs is not None
                or batch.modality_presence is not None
            ):
                encoder_kwargs = {
                    "modality_inputs": batch.modality_inputs,
                    "modality_presence": batch.modality_presence,
                }
            _, latent_mean, _, _, _ = encoder(
                batch.expression,
                batch.categorical_context,
                batch.continuous_context,
                batch.optional_modalities,
                token_mask=batch.expression.token_mask,
                **encoder_kwargs,
            )
    finally:
        # Direct assignment preserves deliberately mixed parent/child states;
        # calling train() recursively here would overwrite them.
        for submodule, training in training_states:
            submodule.training = training
    return latent_mean


def pretraining_terms(
    model: nn.Module,
    batch: WorldModelBatch,
    config: ExperimentConfig,
    global_step: int,
    masking_audit: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Build all enabled Stage-0 objectives with independent input/MGM masks."""
    first_view = batch
    if (
        config.training.panel_corruption_probability
        and torch.rand((), device=batch.expression.count.device)
        < config.training.panel_corruption_probability
    ):
        first_view, _ = panel_expression_view(
            batch,
            config.training.panel_corruption_min_genes,
            config.training.panel_corruption_max_genes,
            include_unobserved=True,
        )
        first_view = measured_zero_token_view(
            first_view,
            config.training.panel_zero_token_budget,
        )
    # Keep a bounded complement of the sampled panel as explicit null-token
    # queries.  These placeholders are input-only; the original batch remains
    # the authoritative reconstruction target below.
    missing_token_budget = int(
        getattr(config.training, "panel_missing_token_budget", 0)
    )
    if missing_token_budget:
        first_view = unmeasured_gene_token_view(
            first_view,
            missing_token_budget,
            n_genes=config.model.n_genes,
        )
    # Keep the post-panel, pre-mask token identity/support as the independent
    # MGM sampling population.  This is the original model input for this
    # view, while remaining shape-compatible with panel corruption.
    mgm_source = first_view
    first_view, first_view_mask = masked_expression_view(
        first_view,
        config.training.masked_token_probability,
        mask_gene_ids=True,
        depth_aware_alpha=config.training.depth_aware_masking_alpha,
        depth_aware_preserve_budget=(
            config.training.depth_aware_masking_preserve_budget
        ),
    )
    mgm_mask = None
    mgm_labels = None
    mgm_candidates = torch.zeros(
        (), device=batch.expression.count.device, dtype=torch.long
    )
    if config.model.masked_gene_modeling.enabled:
        # MGM is an independent 15%-style objective over the original input,
        # not 15% of the 85% that survived first-view corruption.  Targets are
        # likewise read from the original gene IDs below.
        mgm_mask, mgm_candidates = _sample_mgm_mask(
            mgm_source.expression.count > 0,
            config.model.masked_gene_modeling.mask_probability,
            config.model.masked_gene_modeling.max_tokens_per_batch,
        )
        input_mask = mgm_mask
        if first_view.expression.token_mask is not None:
            input_mask = input_mask | first_view.expression.token_mask
        first_view = replace(
            first_view,
            expression=replace(
                first_view.expression,
                count=first_view.expression.count.masked_fill(mgm_mask, 0.0),
                token_mask=input_mask,
            ),
        )
        # sampled_forward prepends each target gene at candidate column zero;
        # therefore every sampled MGM cross-entropy target is intentionally 0.
        mgm_labels = torch.zeros(
            int(mgm_mask.sum()), dtype=torch.long,
            device=batch.expression.gene_index.device,
        )

    if masking_audit is not None:
        positive = mgm_source.expression.count > 0
        union_mask = first_view_mask.clone()
        if mgm_mask is not None:
            union_mask |= mgm_mask
        masking_audit.update({
            "positive_tokens": positive.sum(),
            "missing_placeholder_tokens": (
                mgm_source.expression.missing_token_mask.sum()
                if mgm_source.expression.missing_token_mask is not None
                else positive.new_zeros(())
            ),
            "measured_zero_tokens": (
                mgm_source.expression.measured_zero_mask.sum()
                if mgm_source.expression.measured_zero_mask is not None
                else positive.new_zeros(())
            ),
            "first_view_masked_positive_tokens": (
                first_view_mask & positive
            ).sum(),
            "mgm_candidate_tokens_before_cap": mgm_candidates,
            "mgm_masked_positive_tokens": (
                (mgm_mask & positive).sum()
                if mgm_mask is not None else positive.new_zeros(())
            ),
            "union_masked_positive_tokens": (union_mask & positive).sum(),
        })

    second_view = None
    if config.losses.contrastive:
        if config.training.contrastive_panel_max_genes:
            second_view, _ = panel_expression_view(
                batch,
                config.training.contrastive_panel_min_genes,
                config.training.contrastive_panel_max_genes,
            )
        else:
            second_view, _ = masked_expression_view(
                batch,
                config.training.masked_token_probability,
                mask_gene_ids=True,
            )
        if config.training.contrastive_neutralize_context:
            second_view = neutralized_context_view(second_view)
    # Panel corruption must still decode the original measured panel.  Without
    # this override, ``model`` would infer decode_gene_index from the sampled
    # input panel and silently remove the held-out-gene supervision signal.
    target_decode_gene_index = (
        batch.decode_gene_index
        if batch.decode_gene_index is not None
        else batch.expression.measured_gene_index
    )
    first_view = replace(first_view, decode_gene_index=target_decode_gene_index)
    output = model(
        first_view,
        compute_structure=False,
        contrastive_batch=second_view,
        compute_auxiliary=(
            config.model.assay_alignment.enabled
            or config.model.flow_matching.enabled
        ),
        token_mask=first_view.expression.token_mask,
        mgm_token_mask=mgm_mask,
        mgm_target_gene_index=(
            mgm_source.expression.gene_index[mgm_mask]
            if mgm_mask is not None else None
        ),
        annotation_labels=batch.annotation_labels,
    )
    reconstruction = reconstruction_loss(output, batch)
    terms = {"reconstruction": reconstruction}
    if config.losses.context_neutral_reconstruction:
        module = model.module if hasattr(model, "module") else model
        neutral_decoder = module.decoder(
            output.z_initial,
            torch.zeros_like(output.context),
            batch.expression.library_size,
            output.decoder.gene_index,
            control=batch.control,
            sample=False,
        )
        target = dense_counts_for_genes(
            batch.expression.row_ptr,
            batch.expression.gene_index,
            batch.expression.count,
            neutral_decoder.gene_index,
        )
        terms["context_neutral_reconstruction"] = negative_binomial_nll(
            target,
            neutral_decoder.mean,
            neutral_decoder.dispersion,
        )
    if config.losses.latent_variance or config.losses.latent_covariance:
        spread_latent_mean = output.latent_mean
        module = model.module if hasattr(model, "module") else model
        if module.training and config.training.deterministic_spread_view:
            spread_latent_mean = _deterministic_spread_latent_mean(model, first_view)
        if config.losses.latent_variance:
            terms["latent_variance"] = latent_variance_loss(spread_latent_mean)
        if config.losses.latent_covariance:
            terms["latent_covariance"] = latent_covariance_loss(spread_latent_mean)
    if config.losses.expression_geometry:
        expression_target = dense_counts_for_genes(
            batch.expression.row_ptr,
            batch.expression.gene_index,
            batch.expression.count,
            output.decoder.gene_index,
        )
        terms["expression_geometry"] = expression_geometry_loss(
            output.latent_mean, expression_target
        )
    if config.losses.marker_expression_geometry:
        marker_target = dense_counts_for_genes(
            batch.expression.row_ptr,
            batch.expression.gene_index,
            batch.expression.count,
            output.decoder.gene_index,
        )
        terms["marker_expression_geometry"] = expression_geometry_loss(
            output.latent_mean,
            marker_target,
            gene_weight=_marker_gene_weight(
                config, output.decoder.gene_index
            ),
        )
    if config.model.encoder.variational and config.losses.kl:
        kl_scale = (
            min(
                (global_step + 1) / max(config.training.kl_warmup_steps, 1),
                1.0,
            )
            if config.training.kl_warmup_steps else 1.0
        )
        terms["kl"] = variational_kl_loss(
            output.latent_mean, output.latent_logvar
        ) * kl_scale
    if config.losses.contrastive:
        assert output.contrastive_z is not None
        if config.training.contrastive_objective == "barlow_twins":
            terms["contrastive"] = barlow_twins_loss(
                output.latent_mean,
                output.contrastive_z,
                redundancy_weight=config.training.barlow_redundancy_weight,
            )
        else:
            terms["contrastive"] = contrastive_loss(
                output.latent_mean,
                output.contrastive_z,
                config.training.contrastive_temperature,
            )
            reference_batch = config.training.contrastive_reference_batch_size
            if reference_batch:
                terms["contrastive"] = terms["contrastive"] + math.log(
                    reference_batch / batch.expression.batch_size
                )
    if config.model.assay_alignment.enabled:
        assert output.assay_logits is not None
        assay_index = list(
            config.model.context.categorical_vocab_sizes
        ).index("assay_type")
        terms["assay_adversarial"] = assay_adversarial_loss(
            output.assay_logits,
            batch.categorical_context[:, assay_index],
        )
    if config.model.flow_matching.enabled:
        assert output.flow_prediction is not None
        assert output.flow_target is not None
        terms["flow_matching"] = flow_matching_loss(
            output.flow_prediction, output.flow_target
        )
    if config.losses.partition_independence:
        terms["partition_independence"] = partition_independence_loss(
            output.latent_mean,
            config.model.encoder.latent_partitions,
        )
    if config.model.masked_gene_modeling.enabled:
        assert output.masked_gene_logits is not None
        assert mgm_labels is not None
        terms["masked_gene_modeling"] = masked_gene_modeling_loss(
            output.masked_gene_logits, mgm_labels
        )
    if config.losses.hierarchical_annotation:
        if output.annotation_loss is None:
            raise ValueError(
                "losses.hierarchical_annotation is enabled but no annotation "
                "head/labels were provided on the batch"
            )
        terms["hierarchical_annotation"] = output.annotation_loss
    return terms


def atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


RANK_CHECKPOINT_STATE_FORMAT = "brainwm_rank_checkpoint_state_v1"


def gather_rank_checkpoint_states(
    rank: int,
    world_size: int,
    device: torch.device,
    progress: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect reproducibility state from every rank at a checkpoint boundary."""
    local_state: dict[str, Any] = {
        "format": RANK_CHECKPOINT_STATE_FORMAT,
        "rank": rank,
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device).cpu()
            if device.type == "cuda" else None
        ),
        "progress": dict(progress or {}),
    }
    if world_size == 1:
        return [local_state]
    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise RuntimeError(
            "per-rank checkpoint gathering requires an initialized process "
            "group matching world_size"
        )
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    result = [state for state in gathered if state is not None]
    _rank_checkpoint_state_by_rank(result, world_size)
    return result


def _rank_checkpoint_state_by_rank(
    rank_states: object,
    world_size: int,
) -> dict[int, dict[str, Any]]:
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError(
            "checkpoint per-rank state count does not match the current "
            f"world size ({world_size})"
        )
    by_rank: dict[int, dict[str, Any]] = {}
    for raw in rank_states:
        if not isinstance(raw, dict):
            raise TypeError("checkpoint per-rank state entries must be mappings")
        if raw.get("format") != RANK_CHECKPOINT_STATE_FORMAT:
            raise ValueError("unsupported checkpoint per-rank state format")
        saved_rank = int(raw.get("rank", -1))
        if saved_rank in by_rank:
            raise ValueError(f"duplicate checkpoint state for rank {saved_rank}")
        if not isinstance(raw.get("torch_rng_state"), Tensor):
            raise TypeError(
                f"checkpoint state for rank {saved_rank} lacks CPU RNG state"
            )
        cuda_rng_state = raw.get("cuda_rng_state")
        if cuda_rng_state is not None and not isinstance(cuda_rng_state, Tensor):
            raise ValueError(
                f"checkpoint state for rank {saved_rank} has invalid CUDA RNG state"
            )
        if not isinstance(raw.get("progress"), dict):
            raise TypeError(
                f"checkpoint state for rank {saved_rank} lacks progress state"
            )
        by_rank[saved_rank] = raw
    expected_ranks = set(range(world_size))
    if set(by_rank) != expected_ranks:
        raise ValueError(
            "checkpoint rank IDs do not match the current process group: "
            f"saved={sorted(by_rank)}, expected={sorted(expected_ranks)}"
        )
    return by_rank


def validate_rank_checkpoint_states(
    rank_states: object,
    world_size: int,
) -> None:
    """Validate that a checkpoint contains one reproducible state per rank."""
    _rank_checkpoint_state_by_rank(rank_states, world_size)


def checkpoint_rank_progress(
    state: dict[str, Any],
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Return the saved cumulative progress belonging to one DDP rank."""
    rank_state = _rank_checkpoint_state_by_rank(
        state.get("rank_states"), world_size
    )[rank]
    return dict(rank_state["progress"])


def checkpoint_state(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    config: ExperimentConfig,
    epoch: int,
    next_batch_index: int,
    global_step: int,
    best_validation: float,
    *,
    rank_states: list[dict[str, Any]] | None = None,
    resume_identity: dict[str, Any] | None = None,
    stale_validations: int | None = None,
) -> dict[str, Any]:
    module = model.module if hasattr(model, "module") else model
    state = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "global_step": global_step,
        "best_validation": best_validation,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if rank_states is not None:
        _rank_checkpoint_state_by_rank(rank_states, len(rank_states))
        state["rank_states"] = rank_states
    if resume_identity is not None:
        state["resume_identity"] = resume_identity
    if stale_validations is not None:
        state["stale_validations"] = stale_validations
    return state


def rotate_checkpoints(
    output_dir: Path,
    keep: int,
    protected_paths: set[Path] | tuple[Path, ...] | list[Path] = (),
) -> None:
    checkpoints = sorted(
        output_dir.glob("checkpoint_step_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    protected = {Path(path).resolve() for path in protected_paths}
    remove_count = (
        max(len(checkpoints) - keep, 0) if keep > 0 else len(checkpoints)
    )
    removable = [path for path in checkpoints if path.resolve() not in protected]
    for path in removable[:remove_count]:
        path.unlink()


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
    require_per_rank_rng: bool = False,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    rank_states = state.get("rank_states")
    if rank_states is None and require_per_rank_rng:
        raise ValueError(
            "legacy checkpoint lacks per-rank RNG/progress state and cannot "
            "be resumed reproducibly; start a fresh run or use a checkpoint "
            "written by the current Stage-0 trainer"
        )
    selected_rank_state = None
    # Stage-0 supplies rank/world-size explicitly and requires this path.
    # Other trainers retain their legacy top-level RNG behavior even if a
    # multi-rank Stage-0 checkpoint is inspected through this shared API.
    if rank_states is not None and (
        require_per_rank_rng or len(rank_states) == world_size
    ):
        selected_rank_state = _rank_checkpoint_state_by_rank(
            rank_states, world_size
        )[rank]
    module = model.module if hasattr(model, "module") else model
    compatible_loader = getattr(module, "load_compatible_state_dict", None)
    if compatible_loader is not None:
        compatible_loader(state["model"], strict=True)
    else:
        module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state.get("scaler", {}))
    if selected_rank_state is not None:
        torch.set_rng_state(selected_rank_state["torch_rng_state"].cpu())
        cuda_rng_state = selected_rank_state.get("cuda_rng_state")
        if device.type == "cuda" and cuda_rng_state is not None:
            torch.cuda.set_rng_state(
                cuda_rng_state.detach().to(device="cpu", dtype=torch.uint8),
                device=device,
            )
    else:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if (
        selected_rank_state is None
        and device.type == "cuda"
        and state.get("cuda_rng_state")
    ):
        # Checkpoints may be loaded with map_location=device, which places
        # RNG tensors on CUDA, and may have been created with a different
        # world size. CUDA expects CPU ByteTensors here; restore only the
        # states corresponding to the currently visible devices.
        visible_devices = torch.cuda.device_count()
        rng_states = [
            rng_state.detach().to(device="cpu", dtype=torch.uint8)
            for rng_state in state["cuda_rng_state"][:visible_devices]
        ]
        if rng_states:
            torch.cuda.set_rng_state_all(rng_states)
    return state


def build_batch(reader, plan: IndexedBatchPlan, descriptor: BatchDescriptor,
                sample_metadata: pd.DataFrame, tensorizer: MetadataTensorizer,
                device: torch.device):
    frame = plan.frame(descriptor)
    # H5AD indexed reads require increasing row indices. Spatial plans may
    # intentionally order a batch by coordinate locality; the selected cell
    # set remains local after sorting for I/O.
    frame = frame.sort_values("row_index").reset_index(drop=True)
    indexed = reader.read_batch(
        descriptor.source_id,
        frame["row_index"].to_numpy(),
        frame,
    )
    metadata = indexed["cells"].join(
        sample_metadata, on="sample_key", how="left", rsuffix="_sample"
    )
    # Composite readers expose a stable cell schema, so sample-level columns
    # arrive with the ``_sample`` suffix when a cell column already exists.
    # Fill only missing cell values; explicit cell annotations remain primary.
    for column in (
        "assay_type", "technology", "dataset", "species", "species_id",
        "time_point_days", "time_lower_days", "time_upper_days", "time_basis",
        "time_status", "life_stage", "age_raw", "timepoint_raw",
        "developmental_stage_raw", "uberon_id", "uberon_term",
        "original_anatomical_region", "anatomical_region", "anatomical_layer",
        "brain_region", "brain_area", "layer",
    ):
        sample_column = f"{column}_sample"
        if sample_column not in metadata:
            continue
        if column not in metadata:
            metadata[column] = metadata[sample_column]
        else:
            metadata[column] = metadata[column].where(
                metadata[column].notna(), metadata[sample_column]
            )
    return tensorizer.tensorize(indexed, metadata, device), indexed


@torch.no_grad()
def validate(model: nn.Module, reader, plan: IndexedBatchPlan,
             descriptors: list[BatchDescriptor], sample_metadata: pd.DataFrame,
             tensorizer: MetadataTensorizer, objective: BrainWorldObjective,
             device: torch.device, autocast_enabled: bool,
             autocast_dtype: torch.dtype, world_size: int,
             config: ExperimentConfig | None = None,
             global_step: int = 0) -> dict[str, Any]:
    model.eval()
    objective_sum = torch.zeros((), device=device, dtype=torch.float64)
    term_sums: dict[str, Tensor] = {}
    cell_sum = torch.zeros((), device=device, dtype=torch.float64)
    source_groups: dict[str, dict[int, str]] = {}
    group_names: dict[str, list[str]] = {}
    sources = getattr(reader, "sources", None)
    if isinstance(sources, pd.DataFrame):
        for dimension in ("species", "corpus"):
            if dimension not in sources:
                continue
            lookup = {
                int(source_id): str(value)
                for source_id, value in zip(sources.index, sources[dimension])
                if pd.notna(value)
            }
            names = sorted(set(lookup.values()))
            if lookup and names:
                source_groups[dimension] = lookup
                group_names[dimension] = names
    group_objective_sums = {
        dimension: {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in names
        }
        for dimension, names in group_names.items()
    }
    group_cell_sums = {
        dimension: {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in names
        }
        for dimension, names in group_names.items()
    }
    group_term_sums: dict[str, dict[str, dict[str, Tensor]]] = {
        dimension: {name: {} for name in names}
        for dimension, names in group_names.items()
    }
    masking_sums: dict[str, Tensor] = {}
    for descriptor in descriptors:
        batch, _ = build_batch(
            reader, plan, descriptor, sample_metadata, tensorizer, device
        )
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(
                31 * int(descriptor.source_id) + int(descriptor.start)
            )
            with torch.autocast(
                device_type=device.type,
                enabled=autocast_enabled,
                dtype=autocast_dtype,
            ):
                if config is None:
                    output = model(batch, compute_structure=False)
                    terms = {"reconstruction": reconstruction_loss(output, batch)}
                else:
                    masking_audit: dict[str, Tensor] = {}
                    terms = pretraining_terms(
                        model,
                        batch,
                        config,
                        global_step,
                        masking_audit=masking_audit,
                    )
                loss = objective(terms).loss
        cells = batch.expression.batch_size
        objective_sum += loss.double() * cells
        for name, value in terms.items():
            term_sums.setdefault(
                name, torch.zeros((), device=device, dtype=torch.float64)
            )
            term_sums[name] += value.double() * cells
        if config is not None:
            for name, value in masking_audit.items():
                masking_sums.setdefault(
                    name, torch.zeros((), device=device, dtype=torch.float64)
                )
                masking_sums[name] += value.double()
        for dimension, lookup in source_groups.items():
            group = lookup.get(int(descriptor.source_id))
            if group is None:
                continue
            group_objective_sums[dimension][group] += loss.double() * cells
            group_cell_sums[dimension][group] += cells
            for name, value in terms.items():
                group_term_sums[dimension][group].setdefault(
                    name, torch.zeros((), device=device, dtype=torch.float64)
                )
                group_term_sums[dimension][group][name] += value.double() * cells
        cell_sum += cells
    reduce_sum(objective_sum, world_size)
    for value in term_sums.values():
        reduce_sum(value, world_size)
    for value in masking_sums.values():
        reduce_sum(value, world_size)
    for dimension, names in group_names.items():
        for group in names:
            reduce_sum(group_objective_sums[dimension][group], world_size)
            reduce_sum(group_cell_sums[dimension][group], world_size)
            for name in term_sums:
                group_term_sums[dimension][group].setdefault(
                    name, torch.zeros((), device=device, dtype=torch.float64)
                )
                reduce_sum(
                    group_term_sums[dimension][group][name], world_size
                )
    reduce_sum(cell_sum, world_size)
    model.train()
    denominator = cell_sum.clamp_min(1)
    validation_objective = float((objective_sum / denominator).cpu())
    result: dict[str, Any] = {
        "validation_loss": validation_objective,
        "validation_objective": validation_objective,
        "validation_terms": {
            name: float((value / denominator).cpu())
            for name, value in term_sums.items()
        },
        "validation_cells": int(cell_sum.cpu()),
    }
    positive_tokens = masking_sums.get("positive_tokens")
    if positive_tokens is not None:
        positive_denominator = positive_tokens.clamp_min(1)
        result["validation_masking"] = {
            name: int(value.cpu())
            for name, value in masking_sums.items()
        }
        result["validation_masking_rates"] = {
            name.removesuffix("_tokens") + "_rate": float(
                (value / positive_denominator).cpu()
            )
            for name, value in masking_sums.items()
            if name != "positive_tokens"
        }
        candidates = masking_sums.get("mgm_candidate_tokens_before_cap")
        retained = masking_sums.get("mgm_masked_positive_tokens")
        if candidates is not None and retained is not None:
            result["validation_masking_rates"][
                "mgm_cap_retention_rate"
            ] = float((retained / candidates.clamp_min(1)).cpu())
    for dimension, names in group_names.items():
        breakdown: dict[str, Any] = {}
        objectives: list[float] = []
        for group in names:
            cells = group_cell_sums[dimension][group]
            if int(cells.cpu()) == 0:
                continue
            denominator = cells.clamp_min(1)
            group_objective = float(
                (group_objective_sums[dimension][group] / denominator).cpu()
            )
            objectives.append(group_objective)
            breakdown[group] = {
                "objective": group_objective,
                "terms": {
                    name: float(
                        (
                            group_term_sums[dimension][group][name]
                            / denominator
                        ).cpu()
                    )
                    for name in term_sums
                },
                "cells": int(cells.cpu()),
            }
        if breakdown:
            result[f"validation_by_{dimension}"] = breakdown
            result[f"validation_macro_{dimension}_objective"] = (
                sum(objectives) / len(objectives)
            )
    return result
