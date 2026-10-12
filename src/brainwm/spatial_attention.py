"""Bounded-neighborhood cell attention; absent neighborhoods are exact identity."""
import torch
from torch import nn
from .profiling import profile_range


class SpatialCellAttention(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.output=nn.Linear(width,width,bias=False)
        self.scale=width**-.5

    def forward(self,cells,neighbors=None,neighbor_mask=None,*,indices_validated=False,
                edges_present=None):
        if neighbors is None:return cells
        # NumPy-backed batches compute this fact while constructing sidecars.
        # This avoids both a device synchronization and projections for an
        # empty/self-loop graph.
        if edges_present is False:
            return cells
        with profile_range("brainwm.spatial/validate_and_index"):
            if neighbors.ndim!=2 or neighbors.shape[0]!=len(cells):
                raise ValueError('neighbors must be [cells, bounded_neighbors]')
            if neighbor_mask is None or neighbor_mask.shape!=neighbors.shape:
                raise ValueError('neighbor mask must match neighbor indices')
            valid=neighbor_mask.bool()
            if (not indices_validated and
                    bool(((neighbors[valid]<0)|(neighbors[valid]>=len(cells))).any())):
                raise ValueError('neighbor index out of bounds')
            safe=neighbors.masked_fill(~valid,0).long()
        # Self-loop-only rows are exactly identity, including when explicitly
        # supplied as a one-neighbor row.
        with profile_range("brainwm.spatial/gather"):
            valid=valid & (safe!=torch.arange(len(cells),device=cells.device)[:,None])
            has_neighbors=valid.any(dim=1)
            gathered=cells[safe]
        with profile_range("brainwm.spatial/qk_score_softmax"):
            score=(self.query(cells)[:,None,:]*self.key(gathered)).sum(-1)*self.scale
            score=score.masked_fill(~valid,-torch.inf)
            score=torch.where(has_neighbors[:,None],score,torch.zeros_like(score))
            weights=score.softmax(-1)*valid
        with profile_range("brainwm.spatial/value_message"):
            message=(weights[...,None]*self.value(gathered-cells[:,None,:])).sum(1)
        with profile_range("brainwm.spatial/output_residual"):
            correction=self.output(message)
            return cells+torch.where(
                has_neighbors[:,None],correction,torch.zeros_like(correction)
            )
