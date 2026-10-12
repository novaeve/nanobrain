"""Bounded neighbors among supplied cells, never across samples or splits.

This is a batch-local operator. A whole-section sampler must supply complete
candidate neighborhoods before claiming section-wide nearest neighbors.
"""
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree


def batch_spatial_neighbors(cells, k=8, device='cpu'):
    if not isinstance(k,int) or k<1:raise ValueError('k must be positive')
    ids=np.zeros((len(cells),k),dtype=np.int64)
    mask=np.zeros((len(cells),k),dtype=bool)
    required={'sample_key','global_split','spatial_x','spatial_y'}
    if not required.issubset(cells):
        return torch.as_tensor(ids,device=device),torch.as_tensor(mask,device=device)
    frame=cells.reset_index(drop=True)
    group_columns=['global_split','sample_key']
    # sample_key itself remains in the scope: section names may be reused by
    # different studies or individuals.
    if 'section_id' in frame:group_columns.append('section_id')
    xy=frame[['spatial_x','spatial_y']].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    valid=np.isfinite(xy).all(1) & frame.sample_key.notna().to_numpy() & frame.global_split.notna().to_numpy()
    if 'assay_type' in frame:
        valid &= frame.assay_type.eq('spatialRNA').to_numpy()
    for _,group in frame[valid].groupby(group_columns,dropna=False,sort=False):
        indices=group.index.to_numpy()
        if len(indices)<2:continue
        neighbors=cKDTree(xy[indices]).query(xy[indices],k=min(k+1,len(indices)))[1]
        for position,local in enumerate(neighbors):
            chosen=indices[np.atleast_1d(local)]
            chosen=chosen[chosen!=indices[position]][:k]
            ids[indices[position],:len(chosen)]=chosen
            mask[indices[position],:len(chosen)]=True
    return torch.as_tensor(ids,device=device),torch.as_tensor(mask,device=device)
