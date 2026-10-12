"""Stable balancing for partially observed BrainWM training objectives."""
from __future__ import annotations
from collections.abc import Mapping
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor, nn


CELL_TYPE_MODALITIES = frozenset({
    "cell_type", "cl_id", "cl_term", "original_cell_type",
    "original_subclass",
})
ANATOMY_MODALITIES = frozenset({
    "anatomy", "uberon_id", "uberon_term", "original_anatomical_region",
})


def structured_modality_mask(
    observed: Mapping[str, Tensor] | None,
    *,
    task_probability: float = 0.5,
    all_context_probability: float = 0.15,
    context_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Sample leakage-safe task masks while retaining arbitrary missingness.

    Ontology IDs, terms, and original labels describe the same biological
    target. They must be hidden together whenever the corresponding prediction
    head is supervised. A separate all-context mask trains the exact null-input
    condition used for expression-only validation and downstream inference.
    """
    if not observed:
        return {}
    probabilities = {
        "task_probability": task_probability,
        "all_context_probability": all_context_probability,
        "context_probability": context_probability,
    }
    for name, probability in probabilities.items():
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    first = next(iter(observed.values()))
    if first.ndim != 1:
        raise ValueError("modality observation masks must be vectors")
    cells = len(first)
    device = first.device
    availability: dict[str, Tensor] = {}
    for name, value in observed.items():
        if value.ndim != 1 or len(value) != cells:
            raise ValueError(f"observation mask for {name!r} must have shape [{cells}]")
        if value.device != device:
            raise ValueError("all modality observation masks must share a device")
        availability[name] = value.bool()

    def sample(probability: float) -> Tensor:
        if probability == 0:
            return torch.zeros(cells, dtype=torch.bool, device=device)
        if probability == 1:
            return torch.ones(cells, dtype=torch.bool, device=device)
        return torch.rand(cells, device=device, generator=generator) < probability

    all_context = sample(float(all_context_probability))

    def task_rows(group: frozenset[str], anchor: str) -> Tensor:
        selected = sample(float(task_probability)) | all_context
        eligible = availability.get(anchor)
        if eligible is None:
            eligible = torch.zeros(cells, dtype=torch.bool, device=device)
            for name in group:
                if name in availability:
                    eligible |= availability[name]
        # Sparse labels can otherwise receive no supervised row in a small
        # batch. Selecting one observed row preserves stochastic masking while
        # guaranteeing an active task whenever its target is available.
        first=eligible.to(torch.int64).argmax()
        force=eligible.any() & ~(selected & eligible).any()
        forced=torch.arange(cells,device=device).eq(first) & force
        return selected | forced

    cell_rows = task_rows(CELL_TYPE_MODALITIES, "cell_type")
    anatomy_rows = task_rows(ANATOMY_MODALITIES, "anatomy")
    result: dict[str, Tensor] = {}
    for name, available in availability.items():
        if name in CELL_TYPE_MODALITIES:
            selected = cell_rows
        elif name in ANATOMY_MODALITIES:
            selected = anatomy_rows
        else:
            selected = all_context | sample(float(context_probability))
        result[name] = selected & available
    return result


class MultiObjectiveBalancer(nn.Module):
    """Normalize active losses without backpropagating through their scales."""

    def __init__(self, names, mode: str = "ema_equal", decay: float = 0.99,
                 eps: float = 1e-8):
        super().__init__()
        if mode not in {"ema_equal", "detached_max_equal", "fixed"}:
            raise ValueError(f"unknown loss balancing mode: {mode}")
        self.names = tuple(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("loss names must be unique")
        self.mode, self.decay, self.eps = mode, float(decay), float(eps)
        self.register_buffer("ema", torch.ones(len(self.names)))
        self.register_buffer("seen", torch.zeros(len(self.names), dtype=torch.bool))

    def forward(self, losses: Mapping[str, Tensor | None],
                weights: Mapping[str, float] | None = None,
                materialize_telemetry: bool = True,
                active: Mapping[str, Tensor | bool] | Tensor | None = None):
        available_flags=[losses.get(name) is not None and losses[name].requires_grad
                         for name in self.names]
        if not any(available_flags):
            raise ValueError("joint objective has no active differentiable loss")
        if active is None:
            local_active=self.ema.new_tensor(available_flags)
        elif isinstance(active,Tensor):
            if active.ndim!=1 or active.numel()!=len(self.names):
                raise ValueError("objective activity tensor has the wrong shape")
            local_active=active.to(device=self.ema.device,dtype=self.ema.dtype)
            local_active=local_active*self.ema.new_tensor(available_flags)
        else:
            flags=[]
            for name,available in zip(self.names,available_flags):
                value=active.get(name,False)
                if isinstance(value,Tensor):
                    if value.numel()!=1:
                        raise ValueError(f"activity for {name!r} must be scalar")
                    value=value.to(device=self.ema.device,dtype=self.ema.dtype).reshape(())
                else:
                    value=self.ema.new_tensor(bool(value))
                flags.append(value*available)
            local_active=torch.stack(flags)
        local_magnitude=torch.stack([
            (losses[name].detach().abs().clamp_min(self.eps)
             if available else self.ema.new_zeros(()))
            for name,available in zip(self.names,available_flags)
        ])*local_active
        counts=local_active.clone(); magnitude_sum=local_magnitude.clone()
        if dist.is_initialized():
            # One packed collective keeps objective statistics consistent
            # across ranks without paying two synchronization latencies.
            statistics=torch.cat((counts,magnitude_sum))
            dist.all_reduce(statistics)
            counts,magnitude_sum=statistics.chunk(2)
        global_active=counts.gt(0); global_magnitude=magnitude_sum/counts.clamp_min(1)
        if self.training:
            updated=torch.where(
                self.seen,
                self.decay*self.ema+(1-self.decay)*global_magnitude,
                global_magnitude,
            )
            self.ema.copy_(torch.where(global_active,updated,self.ema))
            self.seen.logical_or_(global_active)
        if self.mode == "fixed":
            scales=torch.ones_like(global_magnitude)
        elif self.mode == "detached_max_equal":
            maximum=global_magnitude.masked_fill(~global_active,float("-inf")).max()
            scales=torch.where(
                global_active,maximum/global_magnitude.clamp_min(self.eps),
                torch.zeros_like(global_magnitude),
            )
        else:
            reference=self.ema.detach().clamp_min(self.eps)
            active_count=global_active.sum().clamp_min(1)
            mean=(reference*global_active).sum()/active_count
            scales=torch.where(
                global_active,mean/reference,torch.zeros_like(reference),
            )
        terms={};term_indices={}
        for index,(name,available) in enumerate(zip(self.names,available_flags)):
            if not available:continue
            value=losses[name]
            coefficient = float((weights or {}).get(name, 1.0))
            terms[name] = value * scales[index] * coefficient
            term_indices[name]=index
        total=torch.stack(list(terms.values())).sum()/global_active.sum()
        telemetry={
            name:{"raw":losses[name].detach(),
                  "scale":scales[term_indices[name]].detach(),
                  "weighted":terms[name].detach()}
            for name in terms
        }
        if materialize_telemetry:
            telemetry=self.materialize_telemetry(telemetry)
        return total, telemetry

    @staticmethod
    def materialize_telemetry(telemetry):
        """Transfer every objective scalar in one device-to-host operation."""
        names=list(telemetry)
        if not names:
            return {}
        first=telemetry[names[0]]["raw"]
        if not isinstance(first,Tensor):
            return telemetry
        packed=torch.stack([
            torch.stack((telemetry[name]["raw"].float(),
                         telemetry[name]["scale"].float(),
                         telemetry[name]["weighted"].float()))
            for name in names
        ]).cpu().tolist()
        return {
            name:{"raw":values[0],"scale":values[1],"weighted":values[2]}
            for name,values in zip(names,packed)
        }


BRAINWM_LOSS_NAMES = (
    "mgm", "zero", "cell", "anatomy", "spatial", "flow",
    "completion", "expert",
)


def _masked_mean(values: Tensor, mask: Tensor,
                 denominator_weight: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Reduce a fixed-shape loss without a GPU-to-CPU activity branch."""
    mask=mask.to(values.dtype)
    denominator=(mask.sum() if denominator_weight is None else
                 (mask*denominator_weight.to(values.dtype)).sum())
    return (values*mask).sum()/denominator.clamp_min(1),denominator.gt(0)


def brainwm_task_losses(output, batch,
                        class_weights: Mapping[str, Tensor] | None = None,
                        *,return_activity: bool = False):
    """Return task losses and optionally their device-resident activity mask.

    ``return_activity=True`` keeps reductions at fixed shape and lets the loss
    balancer combine activity across DDP ranks without synchronizing Python.
    The default retains the historical ``None`` values for external callers.
    """
    flat_target=batch.values.reshape(-1); valid=batch.token_mask.bool().reshape(-1)
    masked=batch.expression_masked.bool().reshape(-1)
    prediction=output['expression_reconstruction'].reshape(-1)
    nonzero=valid & flat_target.ne(0) & masked
    zero=valid & flat_target.eq(0) & masked
    losses={name:None for name in BRAINWM_LOSS_NAMES}
    activity={name:prediction.new_zeros((),dtype=torch.bool)
              for name in BRAINWM_LOSS_NAMES}
    losses['mgm'],activity['mgm']=_masked_mean(
        (prediction-flat_target).square(),nonzero,
    )
    zero_error=prediction.square()
    if batch.token_sampling_weight is not None:
        zero_error=zero_error*batch.token_sampling_weight.reshape(-1)
    losses['zero'],activity['zero']=_masked_mean(zero_error,zero)
    completion=output.get('completed_expression')
    if completion is not None:
        completion_error=F.smooth_l1_loss(
            completion.reshape(-1),flat_target,reduction='none',
        )
        losses['completion'],activity['completion']=_masked_mean(
            completion_error,masked&valid,
        )
    if batch.cell_type_target is not None and 'canonical_annotation_logits' in output:
        target=batch.cell_type_target.long(); keep=target.ge(0)
        if (batch.modality_masked or {}).get('cell_type') is not None:keep=keep&(batch.modality_masked['cell_type'].bool())
        weight = None if class_weights is None else class_weights.get('cell')
        safe_target=target.masked_fill(~keep,0)
        cell_error=F.cross_entropy(
            output['canonical_annotation_logits'],safe_target,
            weight=weight,reduction='none',
        )
        denominator_weight=None if weight is None else weight[safe_target]
        losses['cell'],activity['cell']=_masked_mean(
            cell_error,keep,denominator_weight,
        )
    if batch.anatomy_target is not None and 'anatomy_logits' in output:
        target=batch.anatomy_target.long(); keep=target.ge(0)
        if (batch.modality_masked or {}).get('anatomy') is not None:keep=keep&(batch.modality_masked['anatomy'].bool())
        weight = None if class_weights is None else class_weights.get('anatomy')
        safe_target=target.masked_fill(~keep,0)
        anatomy_error=F.cross_entropy(
            output['anatomy_logits'],safe_target,
            weight=weight,reduction='none',
        )
        denominator_weight=None if weight is None else weight[safe_target]
        losses['anatomy'],activity['anatomy']=_masked_mean(
            anatomy_error,keep,denominator_weight,
        )
    if batch.neighbor_indices is not None and batch.neighbor_mask is not None:
        has=batch.neighbor_mask.any(1)
        safe=batch.neighbor_indices.clamp_min(0); source=output['pre_spatial_embedding']; gathered=source[safe]
        weights=batch.neighbor_mask.unsqueeze(-1); target=(gathered*weights).sum(1)/weights.sum(1).clamp_min(1)
        spatial_error=(output['cell_embedding']-target.detach()).square().mean(-1)
        losses['spatial'],activity['spatial']=_masked_mean(spatial_error,has)
    observed=batch.developmental_observed
    if observed is not None:
        keep=observed.bool()
        flow_error=.5*((output['flow_velocity']-output['flow_target']).square().mean(-1)+
                       (output['flow_endpoint']-output['flow_endpoint_target']).square().mean(-1))
        losses['flow'],activity['flow']=_masked_mean(flow_error,keep)
    losses['expert']=output.get('expert_balance_loss')
    if losses['expert'] is not None:
        activity['expert']=prediction.new_ones((),dtype=torch.bool)
    if return_activity:
        return losses,activity
    for name in BRAINWM_LOSS_NAMES:
        if losses[name] is not None and not bool(activity[name]):
            losses[name]=None
    return losses
