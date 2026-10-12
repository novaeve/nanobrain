"""Random-access developmental populations from V3 NumPy row references."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from .sparse_tokens import SparseTokenBatch


def rank_group_positions(group_count,rank,world,start_step=0):
    """Return one rank's infinite-cycle origin after an optimizer-step resume."""
    positions=list(range(int(rank),int(group_count),int(world)))
    if not positions:return positions
    offset=int(start_step)%len(positions)
    return positions[offset:]+positions[:offset]


def _identity(sample):
    return sample

def load_referenced_rows(root, references, split='train', packed=False):
    temporal=('time_point_days','time_lower_days','time_upper_days','chronological_time_proxy_days')
    root=Path(root); rows=[]; metadata={k:[] for k in ('assay','technology','donor','life_stage','cell_type','anatomy',*temporal)}
    for reference in np.asarray(references,dtype=np.int64):
        shard=int(reference>>32); row=int(reference&0xffffffff); stem=root/f'{split}-shard-{shard:05d}'
        ptr=np.load(str(stem)+'-row_ptr.npy',mmap_mode='r'); lo,hi=int(ptr[row]),int(ptr[row+1]); genes=np.load(str(stem)+'-gene_index.npy',mmap_mode='r');values=np.load(str(stem)+'-values.npy',mmap_mode='r')
        rows.append((np.array(genes[lo:hi],copy=True),np.array(values[lo:hi],copy=True)))
        for name in metadata:metadata[name].append(np.load(str(stem)+f'-{name}.npy',mmap_mode='r')[row])
    lengths=np.fromiter((len(g) for g,_ in rows),dtype=np.int64,count=len(rows));n=len(rows)
    if packed:
        row_ptr=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(lengths,dtype=np.int64)))
        row_ids=np.repeat(np.arange(n,dtype=np.int64),lengths)
        gene=np.concatenate([g for g,_ in rows]) if row_ptr[-1] else np.empty(0,np.int64)
        value=np.concatenate([v for _,v in rows]) if row_ptr[-1] else np.empty(0,np.float32)
        mask=np.ones(len(gene),dtype=bool)
    else:
        width=max(int(lengths.max()),1);gene=np.zeros((n,width),np.int64);value=np.zeros((n,width),np.float32);mask=np.zeros((n,width),bool)
        row_ids=np.repeat(np.arange(n,dtype=np.int64),width)
        for i,(g,v) in enumerate(rows):gene[i,:len(g)]=g;value[i,:len(v)]=v;mask[i,:len(g)]=True
        row_ptr=np.arange(0,(n+1)*width,width,dtype=np.int64)
    tensor={k:torch.as_tensor(np.asarray(v)) for k,v in metadata.items()};modality={k:tensor[k].long() for k in ('assay','technology','donor','life_stage','cell_type','anatomy')};modality['development']=torch.nan_to_num(torch.stack([tensor[k].float() for k in temporal],dim=-1))/365.25
    observed={k:v.ge(0) for k,v in modality.items() if k!='development'};observed['development']=torch.isfinite(tensor['time_point_days'])
    packed_min_seqlen=int(lengths.min()) if packed and len(lengths) else None
    packed_max_seqlen=int(lengths.max()) if packed and len(lengths) else None
    packed_layout_validated=bool(
        packed and row_ptr[0]==0 and row_ptr[-1]==len(gene) and mask.all()
    )
    token_length_source=(
        lengths if packed else np.repeat(lengths, width)
    )
    return SparseTokenBatch(
        torch.from_numpy(gene.reshape(-1)), torch.from_numpy(value.reshape(-1)),
        torch.from_numpy(row_ptr), torch.empty(0,dtype=torch.long),
        torch.from_numpy(mask.reshape(-1)), modality=modality,
        modality_observed=observed, cell_type_target=tensor['cell_type'].long(),
        anatomy_target=tensor['anatomy'].long(), packed=bool(packed),
        packed_layout_validated=packed_layout_validated,
        packed_min_seqlen=packed_min_seqlen,
        packed_max_seqlen=packed_max_seqlen, row_ids=torch.from_numpy(row_ids),
        row_lengths=torch.from_numpy(np.asarray(lengths,dtype=np.int64).copy()),
        token_lengths=torch.from_numpy(
            np.asarray(token_length_source,dtype=np.int64).copy()
        ),
        cu_seqlens=torch.from_numpy(np.asarray(row_ptr,dtype=np.int32).copy()),
    )

class DevelopmentPairLoader(torch.utils.data.IterableDataset):
    def __init__(self,root,max_padded_tokens=60000,split='train',group_samples=None,seed=20260912,
                 distributed=True,packed=False,start_step=0):
        super().__init__()
        self.root=Path(root);self.split=str(split);prefix='development' if split=='train' else f'development_{split}'
        self.source=np.load(self.root/f'{prefix}_source_ref.npy',mmap_mode='r');self.target=np.load(self.root/f'{prefix}_target_ref.npy',mmap_mode='r');self.ptr=np.load(self.root/f'{prefix}_pair_ptr.npy',mmap_mode='r');self.info=json.loads((self.root/f'{prefix}_pairs.json').read_text())['groups_metadata'];self.max_padded_tokens=int(max_padded_tokens);self._row_ptr={};self._batch_cache={};self.seed=int(seed);self.distributed=bool(distributed)
        self.packed=bool(packed);self.start_step=int(start_step);count=len(self.ptr)-1
        if group_samples is None or int(group_samples)>=count:self.indices=np.arange(count,dtype=np.int64)
        else:self.indices=np.unique(np.linspace(0,count-1,max(1,int(group_samples)),dtype=np.int64))
        np.random.default_rng(self.seed).shuffle(self.indices)
        if self.max_padded_tokens<1:raise ValueError('max_padded_tokens must be positive')
    def __len__(self):return len(self.indices)
    def _lengths(self,references):
        result=[]
        for reference in np.asarray(references,dtype=np.int64):
            shard=int(reference>>32);row=int(reference&0xffffffff)
            ptr=self._row_ptr.get(shard)
            if ptr is None:
                ptr=np.load(self.root/f'{self.split}-shard-{shard:05d}-row_ptr.npy',mmap_mode='r');self._row_ptr[shard]=ptr
            result.append(int(ptr[row+1]-ptr[row]))
        return np.asarray(result,dtype=np.int64)
    def __iter__(self):
        rank=int(os.environ.get('RANK','0')) if self.distributed else 0
        world=int(os.environ.get('WORLD_SIZE','1')) if self.distributed else 1
        for position in rank_group_positions(len(self),rank,world,self.start_step):
            i=int(self.indices[position])
            cached=self._batch_cache.get(i)
            if cached is not None:
                yield cached
                continue
            lo,hi=int(self.ptr[i]),int(self.ptr[i+1]);meta=self.info[i]
            source=self.source[lo:hi];target=self.target[lo:hi]
            source_width=np.maximum.accumulate(self._lengths(source));target_width=np.maximum.accumulate(self._lengths(target))
            padded=np.arange(1,len(source)+1)*np.maximum(source_width,target_width)
            allowed=np.nonzero(padded<=self.max_padded_tokens)[0]
            count=int(allowed[-1])+1 if len(allowed) else 1
            batch=(load_referenced_rows(self.root,source[:count],self.split,self.packed),load_referenced_rows(self.root,target[:count],self.split,self.packed),float(meta['source_time_days']),float(meta['target_time_days']))
            self._batch_cache[i]=batch
            yield batch


def prefetched_development_loader(dataset,pin_memory=True,prefetch_factor=4):
    """Use one PyTorch worker without changing rank-level pair assignment."""
    return torch.utils.data.DataLoader(
        dataset,batch_size=None,num_workers=1,pin_memory=pin_memory,
        persistent_workers=True,prefetch_factor=int(prefetch_factor),
        collate_fn=_identity,
    )
