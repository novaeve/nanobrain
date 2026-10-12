"""Masked observed-token expression supervision for the sparse input path.

This objective covers supplied observed tokens. It does not by itself provide
measured-zero sampling or ordered whole-panel completion supervision.
"""
from dataclasses import replace
import torch


def mask_expression(batch, probability=.15, generator=None):
    if not 0<probability<=1:raise ValueError('mask probability must be in (0,1]')
    valid=batch.token_mask
    if valid is None:valid=torch.ones_like(batch.values,dtype=torch.bool)
    masked=(torch.rand(batch.values.shape,device=batch.values.device,generator=generator)<probability)&valid
    row_ids=batch.token_row_ids()
    eligible_count=torch.zeros(batch.batch_size,device=batch.values.device,dtype=torch.long)
    eligible_count.scatter_add_(0,row_ids,valid.long())
    probability_by_row=(probability+(1-probability)**eligible_count.float()
                        / eligible_count.clamp_min(1))
    inclusion=torch.where(valid,probability_by_row[row_ids].to(batch.values),torch.zeros_like(batch.values))
    selected_count=torch.zeros_like(eligible_count);selected_count.scatter_add_(0,row_ids,masked.long())
    needs=(selected_count==0)&(eligible_count>0)
    scores=torch.rand(batch.values.shape,device=batch.values.device,generator=generator)
    scores=scores.masked_fill(~valid|~needs[row_ids],-1)
    maxima=torch.full((batch.batch_size,),-1.,device=batch.values.device)
    maxima.scatter_reduce_(0,row_ids,scores,reduce='amax',include_self=True)
    masked=masked|((scores==maxima[row_ids])&valid&needs[row_ids])
    return replace(batch,expression_masked=masked,expression_mask_probability=inclusion)


def masked_expression_loss(output,batch):
    if batch.expression_masked is None:raise ValueError('expression training mask is required')
    selected=batch.expression_masked.bool()
    if batch.token_mask is not None:selected=selected & batch.token_mask.bool()
    prediction=output['expression_reconstruction'].reshape(-1)
    if prediction.shape!=batch.values.shape:raise ValueError('expression output shape differs from targets')
    if not selected.any():return prediction.sum()*0
    if batch.token_sampling_weight is not None:
        if batch.expression_mask_probability is None:
            raise ValueError('weighted expression loss requires mask inclusion probabilities')
        weights=batch.token_sampling_weight[selected]/batch.expression_mask_probability[selected]
        denominator=batch.batch_size*batch.measured_gene_index.numel()
        if denominator==0:raise ValueError('weighted loss requires a measured panel')
        return ((prediction[selected]-batch.values[selected]).square()*weights).sum()/denominator
    return (prediction[selected]-batch.values[selected]).square().mean()
