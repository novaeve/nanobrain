"""Append sampled measured-zero genes to complete, untruncated reader rows.

Never apply after token truncation: a discarded nonzero cannot be inferred to
be zero. All original nonzero tokens are retained. This samples zero features,
not cells; callers must record the sampling policy in their training config.
"""
import numpy as np


def append_measured_zeros(raw, zero_tokens_per_cell, rng, row_seeds=None):
    if not isinstance(zero_tokens_per_cell,int) or zero_tokens_per_cell<0:
        raise ValueError('zero_tokens_per_cell must be a nonnegative integer')
    panel=np.unique(np.asarray(raw['measured_gene_index'],dtype=np.int64))
    genes=np.asarray(raw['gene_index'],dtype=np.int64)
    values=np.asarray(raw['count'])
    ptr=np.asarray(raw['row_ptr'],dtype=np.int64)
    if len(ptr)<2 or ptr[0]!=0 or ptr[-1]!=len(genes) or len(values)!=len(genes) or (np.diff(ptr)<0).any():
        raise ValueError('invalid sparse row structure')
    if not np.isin(genes,panel).all():raise ValueError('expression genes must belong to measured panel')
    parts_g=[];parts_v=[];parts_w=[];new_ptr=[0];zero_counts=[]
    if row_seeds is not None and len(row_seeds)!=len(ptr)-1:
        raise ValueError('row seeds must match sparse rows')
    for row,(lo,hi) in enumerate(zip(ptr[:-1],ptr[1:])):
        candidates=np.setdiff1d(panel,genes[lo:hi],assume_unique=False)
        n=min(zero_tokens_per_cell,len(candidates))
        local_rng=rng if row_seeds is None else np.random.default_rng(row_seeds[row])
        extra=local_rng.choice(candidates,size=n,replace=False)
        parts_g.append(np.concatenate([genes[lo:hi],extra]))
        parts_v.append(np.concatenate([values[lo:hi],np.zeros(n,dtype=values.dtype)]))
        parts_w.append(np.concatenate([np.ones(hi-lo),np.full(n,len(candidates)/n if n else 0.)]))
        new_ptr.append(new_ptr[-1]+hi-lo+n)
        zero_counts.append(dict(population=int(len(candidates)),sampled=n))
    return dict(raw,gene_index=np.concatenate(parts_g),count=np.concatenate(parts_v),
                row_ptr=np.asarray(new_ptr,dtype=np.int64),zero_sampling=zero_counts,
                token_sampling_weight=np.concatenate(parts_w))
