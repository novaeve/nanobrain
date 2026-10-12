"""Joint objective for the unified Human BrainWM experiment."""
from __future__ import annotations
import torch
from torch import Tensor
from .foundation_moe import BrainWMInput


def masked_mean(value: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        return value.mean()
    mask = mask.bool()
    return value[mask].mean() if mask.any() else value.new_zeros(())


def foundation_loss(output: dict[str, Tensor], batch: BrainWMInput,
                    target_expression: Tensor | None = None,
                    developmental_weight: float = 1.0) -> dict[str, Tensor]:
    """Compute only losses whose supervision is present in the batch."""
    target = batch.expression if target_expression is None else target_expression
    observed = batch.expression_observed.bool()
    reconstruction = masked_mean((output['expression_reconstruction'] - target).square(), observed)
    # Development is population-level Flow Matching.  When age/time is absent,
    # the model's identity/zero-flow fallback is valid and its loss is skipped.
    dev_ok = batch.developmental_observed
    if dev_ok is None:
        dev_ok = torch.zeros(len(target), dtype=torch.bool, device=target.device)
    dev_ok=dev_ok.to(target.device,dtype=torch.bool)
    if dev_ok.ndim==0:
        dev_ok=dev_ok.expand(len(target))
    if dev_ok.shape!=(len(target),):
        raise ValueError('developmental_observed must have one flag per cell')
    if dev_ok.any():
        prediction,truth=output['flow_velocity'],output['flow_target']
        if prediction.shape!=truth.shape or prediction.shape[0]!=len(target):
            raise ValueError('flow predictions and targets must have matching per-cell shapes')
        # Slice first: unknown targets may contain NaNs and must not poison
        # the supervised velocity or its gradient.
        flow=(prediction[dev_ok]-truth[dev_ok]).square().mean()
    else:
        flow = target.new_zeros(())
    total = reconstruction + developmental_weight * flow
    result = {'total': total, 'expression_reconstruction': reconstruction, 'flow_matching': flow}
    if 'annotation_logits' in output:
        result['annotation_available'] = total.new_tensor(1.)
    return result
