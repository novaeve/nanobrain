"""Zero-copy NumPy-memmap -> pinned Torch sparse batches for V3 training."""
from pathlib import Path
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from .sparse_tokens import SparseTokenBatch


CATEGORICAL_SIDECARS = (
    'dataset','assay','technology','donor','life_stage','sex',
    'disease_status','time_status','split_group','split_unit','section',
    'sample','cl_id','cl_term','original_cell_type','original_subclass',
    'uberon_id','uberon_term','original_anatomical_region',
)

OPTIONAL_SIDECARS = tuple(dict.fromkeys((*CATEGORICAL_SIDECARS,
    'cell_type','anatomy','spatial','time_point_days','time_lower_days',
    'time_upper_days','chronological_time_proxy_days','integration_batch',
    'integration_biology','annotation_eligible','development_eligible',
    'spatial_eligible','integration_eligible',
)))


def sample_measured_zeros(panel, row_genes, count, rng):
    """Match ``rng.choice(setdiff1d(...))`` without materializing the complement.

    Frozen V3 panels are sorted and unique. Sampling indices from the implicit
    complement uses NumPy's identical ``Generator.choice`` path, then maps each
    complement rank around the expressed positions. This preserves every
    sampled gene for an existing seed while avoiding an O(panel) allocation per
    cell.
    """
    panel=np.asarray(panel)
    row_genes=np.asarray(row_genes)
    if panel.ndim != 1 or row_genes.ndim != 1:
        raise ValueError('panel and row genes must be vectors')
    if count < 0:
        raise ValueError('zero sample count must be nonnegative')
    if not len(panel):
        return panel[:0],0

    positions=np.searchsorted(panel,row_genes)
    clipped=np.minimum(positions,len(panel)-1)
    present=(positions<len(panel)) & (panel[clipped]==row_genes)
    # Materialized V3 rows are sorted and unique.  Keep a defensive fallback
    # for hand-built/legacy rows with duplicates, but avoid ``np.unique`` on
    # every production cell (it is a measurable part of the CPU loader).
    excluded=(
        positions[present]
        if len(row_genes) < 2 or bool(np.all(row_genes[1:] > row_genes[:-1]))
        else np.unique(positions[present])
    )
    candidate_count=len(panel)-len(excluded)
    count=min(int(count),candidate_count)
    if count == 0:
        return panel[:0],candidate_count
    selected=rng.choice(candidate_count,size=count,replace=False)

    # Map a rank in the implicit complement back to a panel position without
    # the old fixed-point loop.  For sorted excluded positions ``e_j``, the
    # number of excluded slots skipped by complement rank ``s`` is the count
    # of ``e_j - j <= s``.  One search therefore gives the exact fixed point:
    # ``mapped = s + searchsorted(e - arange(len(e)), s, right)``.
    # This is still the same NumPy Generator.choice stream and preserves the
    # historical sampled genes bit-for-bit, while removing several small
    # searchsorted/array_equal kernels per cell.
    thresholds=excluded-np.arange(
        len(excluded),dtype=excluded.dtype,
    )
    mapped=selected+np.searchsorted(thresholds,selected,side='right')
    return panel[mapped],candidate_count


def bucketed_attention_token_cost(lengths, group_cells):
    """Allocated token slots for the model's sorted-length micro-buckets."""
    if group_cells < 1:
        raise ValueError('packed attention group size must be positive')
    ordered=np.sort(np.asarray(lengths,dtype=np.int64))
    if not len(ordered):return 0
    ends=np.arange(group_cells,len(ordered)+1,group_cells,dtype=np.int64)
    if not len(ends) or ends[-1]!=len(ordered):ends=np.append(ends,len(ordered))
    begins=np.concatenate((np.zeros(1,dtype=np.int64),ends[:-1]))
    return int(np.sum(ordered[ends-1]*(ends-begins)))

def rank_offsets_for_global_prefix(root, split, global_rows, world_size):
    """Map a legacy single-rank prefix to deterministic per-rank shard cursors."""
    files=sorted(Path(root).glob(f'{split}-shard-*-row_ptr.npy'))
    offsets=[0]*int(world_size);remaining=int(global_rows)
    for file_i,path in enumerate(files):
        if remaining<=0:break
        rows=len(np.load(path,mmap_mode='r'))-1
        consumed=min(remaining,rows)
        offsets[file_i%world_size]+=consumed;remaining-=consumed
    if remaining:raise ValueError(f'global resume row {global_rows} exceeds {split} corpus')
    return offsets


def balanced_shard_order(files, consumers, seed, balance_waves=8):
    """Shuffle biological sources while keeping DDP ranks near equal density."""
    if consumers < 1 or balance_waves < 1:
        raise ValueError('consumers and balance_waves must be positive')
    entries=[]
    for file_i,gene_file in enumerate(files):
        stem=str(gene_file)[:-len('-gene_index.npy')]
        tokens=len(np.load(gene_file,mmap_mode='r'))
        rows=len(np.load(stem+'-row_ptr.npy',mmap_mode='r'))-1
        entries.append((file_i,gene_file,tokens/max(rows,1)))
    entries.sort(key=lambda item:item[2])
    band_size=consumers*balance_waves
    bands=[entries[start:start+band_size] for start in range(0,len(entries),band_size)]
    rng=np.random.default_rng(int(seed));rng.shuffle(bands)
    ordered=[]
    for band in bands:
        rng.shuffle(band);ordered.extend((file_i,path) for file_i,path,_ in band)
    return ordered

class NumpyShardDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, split='train', batch_size=256, spatial_neighbors=8,zero_tokens_per_cell=16,
                 max_padded_tokens=60000, start_row=0, distributed=True, packed=False,
                 packed_group_cells=32, shuffle=False, seed=0, balance_waves=8):
        self.root=Path(root); self.files=sorted(self.root.glob(f'{split}-shard-*-gene_index.npy'))
        if not self.files: raise FileNotFoundError(f'no NumPy shards for {split} in {root}')
        self.batch_size=int(batch_size)
        self.spatial_neighbors=int(spatial_neighbors)
        self.zero_tokens_per_cell=int(zero_tokens_per_cell)
        self.max_padded_tokens=int(max_padded_tokens)
        self.start_row=int(start_row)
        self.distributed=bool(distributed)
        self.packed=bool(packed)
        self.packed_group_cells=int(packed_group_cells)
        self.shuffle=bool(shuffle)
        self.seed=int(seed)
        self.balance_waves=int(balance_waves)
        if self.start_row < 0: raise ValueError('start_row must be nonnegative')
        if self.max_padded_tokens < 1: raise ValueError('max_padded_tokens must be positive')
        if self.packed_group_cells < 1: raise ValueError('packed_group_cells must be positive')
        if self.balance_waves < 1: raise ValueError('balance_waves must be positive')
        source_file=self.root/'panel_source_id.npy'; ptr_file=self.root/'panel_ptr.npy'; gene_file=self.root/'panel_gene_index.npy'
        self.has_panel_catalog=source_file.exists() and ptr_file.exists() and gene_file.exists()
        if self.has_panel_catalog:
            source=np.load(source_file,mmap_mode='r'); ptr=np.load(ptr_file,mmap_mode='r'); genes=np.load(gene_file,mmap_mode='r')
            self.panels={int(source[i]):np.asarray(genes[int(ptr[i]):int(ptr[i+1])]) for i in range(len(source))}
            if any(len(panel) and np.any(panel[1:]<=panel[:-1])
                   for panel in self.panels.values()):
                raise ValueError('measured gene panels must be sorted and unique')
    def __len__(self): return len(self.files)
    def _batch_end(self, ptr, row_source, start, nrows):
        end=min(start+self.batch_size,nrows)
        if row_source is not None:
            source_id=int(row_source[start]); changes=np.flatnonzero(row_source[start+1:end]!=source_id)
            if len(changes):end=start+1+int(changes[0])
        candidate_lengths=np.diff(ptr[start:end+1]).astype(np.int64)
        if self.packed:
            candidate_lengths=candidate_lengths+self.zero_tokens_per_cell
            real_cost=np.cumsum(candidate_lengths,dtype=np.int64)
            allowed=np.nonzero(real_cost<=self.max_padded_tokens)[0]
            upper=int(allowed[-1])+1 if len(allowed) else 1
            best=0
            for count in range(upper,0,-1):
                if bucketed_attention_token_cost(candidate_lengths[:count],self.packed_group_cells)<=self.max_padded_tokens:
                    best=count;break
            return start+max(best,1)
        running_max=np.maximum.accumulate(candidate_lengths)+self.zero_tokens_per_cell
        batch_cost=np.arange(1,len(candidate_lengths)+1)*running_max
        allowed=np.nonzero(batch_cost<=self.max_padded_tokens)[0]
        return start+(int(allowed[-1])+1 if len(allowed) else 1)

    def __iter__(self):
        worker=torch.utils.data.get_worker_info(); worker_id=worker.id if worker else 0; workers=worker.num_workers if worker else 1
        rank=int(os.environ.get('RANK','0')) if self.distributed else 0
        world=int(os.environ.get('WORLD_SIZE','1')) if self.distributed else 1
        remaining_skip=self.start_row
        # Every worker reconstructs the same rank-local descriptor stream and
        # handles a round-robin subset. PyTorch DataLoader merges worker tasks
        # in order, so worker count changes prefetch parallelism, not examples.
        file_entries=(balanced_shard_order(self.files,world,self.seed,self.balance_waves)
                      if self.shuffle else list(enumerate(self.files)))
        batch_index=0
        for file_position,(file_i,gene_file) in enumerate(file_entries):
            if file_position%world!=rank: continue
            stem=str(gene_file)[:-len('-gene_index.npy')]
            genes=np.load(gene_file,mmap_mode='r'); vals=np.load(stem+'-values.npy',mmap_mode='r'); ptr=np.load(stem+'-row_ptr.npy',mmap_mode='r')
            row_source=np.load(stem+'-row_source_id.npy',mmap_mode='r') if self.has_panel_catalog else None
            legacy_panel=None if self.has_panel_catalog else np.load(stem+'-measured_gene_index.npy',mmap_mode='r')
            if legacy_panel is not None and len(legacy_panel) and np.any(
                legacy_panel[1:]<=legacy_panel[:-1]
            ):
                raise ValueError('measured gene panels must be sorted and unique')
            sidecar_arrays={}
            for name in OPTIONAL_SIDECARS:
                path=Path(stem+f'-{name}.npy')
                if path.exists():
                    sidecar_arrays[name]=np.load(path,mmap_mode='r')
            nrows=len(ptr)-1
            if remaining_skip >= nrows:
                remaining_skip -= nrows
                continue
            ranges=[];cursor=0
            while cursor<nrows:
                end=self._batch_end(ptr,row_source,cursor,nrows)
                ranges.append((cursor,end));cursor=end
            if self.shuffle:
                # Keep the historical single-worker seed independent of the
                # worker that happens to execute this batch.
                sequence=np.random.SeedSequence([self.seed,file_i,0])
                np.random.default_rng(sequence).shuffle(ranges)
            for range_start,range_end in ranges:
                range_rows=range_end-range_start
                if remaining_skip>=range_rows:
                    remaining_skip-=range_rows
                    continue
                start=range_start+remaining_skip;end=range_end;remaining_skip=0
                assigned=batch_index%workers==worker_id
                batch_index+=1
                if not assigned:
                    continue
                if row_source is not None:
                    source_id=int(row_source[start]);panel=self.panels[source_id]
                else:panel=legacy_panel
                lo=int(ptr[start]); hi=int(ptr[end])
                local_ptr=np.asarray(ptr[start:end+1],dtype=np.int64)-lo
                lengths=np.diff(local_ptr)
                row_lengths=np.asarray(lengths,dtype=np.int64)
                gene_rows=[];value_rows=[];weight_rows=[]
                for row,(row_lo,row_hi) in enumerate(zip(local_ptr[:-1],local_ptr[1:])):
                    length=int(row_hi-row_lo)
                    row_gene=np.asarray(genes[lo+row_lo:lo+row_hi],dtype=np.int64)
                    row_value=np.asarray(vals[lo+row_lo:lo+row_hi],dtype=np.float32)
                    row_weight=np.ones(length,dtype=np.float32)
                    if self.zero_tokens_per_cell:
                        rng=np.random.default_rng((self.seed*104729+file_i*65537+start+row)&0xffffffff)
                        chosen,candidate_count=sample_measured_zeros(
                            panel,row_gene,self.zero_tokens_per_cell,rng,
                        )
                        count=len(chosen)
                        if count:
                            row_gene=np.concatenate((row_gene,chosen.astype(np.int64,copy=False)))
                            row_value=np.concatenate((row_value,np.zeros(count,dtype=np.float32)))
                            row_weight=np.concatenate((row_weight,np.full(count,candidate_count/count,dtype=np.float32)))
                    gene_rows.append(row_gene);value_rows.append(row_value);weight_rows.append(row_weight)
                if self.packed:
                    row_lengths=np.fromiter((len(row) for row in gene_rows),dtype=np.int64,count=len(gene_rows))
                    token_row_ids=np.repeat(
                        np.arange(len(row_lengths),dtype=np.int64),row_lengths,
                    )
                    output_ptr=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(row_lengths,dtype=np.int64)))
                    output_gene=np.concatenate(gene_rows) if output_ptr[-1] else np.empty(0,dtype=np.int64)
                    output_value=np.concatenate(value_rows) if output_ptr[-1] else np.empty(0,dtype=np.float32)
                    output_weight=np.concatenate(weight_rows) if output_ptr[-1] else np.empty(0,dtype=np.float32)
                    output_mask=np.ones(len(output_gene),dtype=np.bool_)
                else:
                    width=max(1,max((len(row) for row in gene_rows),default=0))
                    token_row_ids=np.repeat(
                        np.arange(end-start,dtype=np.int64),width,
                    )
                    output_gene=np.zeros((end-start,width),dtype=np.int64)
                    output_value=np.zeros((end-start,width),dtype=np.float32)
                    output_mask=np.zeros((end-start,width),dtype=np.bool_)
                    output_weight=np.zeros((end-start,width),dtype=np.float32)
                    for row,(row_gene,row_value,row_weight) in enumerate(zip(gene_rows,value_rows,weight_rows)):
                        row_length=len(row_gene)
                        output_gene[row,:row_length]=row_gene
                        output_value[row,:row_length]=row_value
                        output_mask[row,:row_length]=True
                        output_weight[row,:row_length]=row_weight
                    output_ptr=np.arange(0,(end-start+1)*width,width,dtype=np.int64)
                def sidecar(name,dtype=None):
                    array=sidecar_arrays.get(name)
                    if array is None:return None
                    value=np.array(array[start:end],copy=True)
                    return torch.from_numpy(value).to(dtype) if dtype else torch.from_numpy(value)
                context={name:value for name in CATEGORICAL_SIDECARS if (value:=sidecar(name,torch.long)) is not None}
                cell_type=sidecar('cell_type',torch.long); anatomy=sidecar('anatomy',torch.long); coordinates=sidecar('spatial',torch.float32); section=sidecar('section',torch.long)
                time_days=sidecar('time_point_days',torch.float32); time_lower=sidecar('time_lower_days',torch.float32); time_upper=sidecar('time_upper_days',torch.float32); time_proxy=sidecar('chronological_time_proxy_days',torch.float32)
                modality=dict(context)
                if cell_type is not None:modality['cell_type']=cell_type
                if anatomy is not None:modality['anatomy']=anatomy
                if time_days is not None:
                    temporal=torch.stack((time_days,time_lower,time_upper,time_proxy),dim=-1)
                    modality['development']=torch.nan_to_num(temporal)/365.25
                observed={name:value.ge(0) for name,value in context.items()}
                if cell_type is not None:observed['cell_type']=cell_type.ge(0)
                if anatomy is not None:observed['anatomy']=anatomy.ge(0)
                if time_days is not None:observed['development']=torch.isfinite(time_days)
                targets=dict(context)
                for name in ('integration_batch','integration_biology','annotation_eligible','development_eligible','spatial_eligible','integration_eligible'):
                    value=sidecar(name,torch.long)
                    if value is not None:targets[name]=value
                neighbors=neighbor_mask=None
                neighbor_indices_validated=False
                neighbor_edges_present=None
                technical_group=(torch.from_numpy(np.array(row_source[start:end],copy=True)).long()
                                 if row_source is not None else section)
                if coordinates is not None and technical_group is not None:
                    valid=torch.isfinite(coordinates[:,:2]).all(1)&technical_group.ge(0); n=end-start; k=max(1,min(self.spatial_neighbors,n))
                    neighbors=torch.arange(n).view(n,1).expand(n,k).clone(); neighbor_mask=torch.zeros((n,k),dtype=torch.bool)
                    valid_np=valid.numpy(); group_np=technical_group.numpy(); xy=coordinates[:,:2].numpy()
                    for group_id in np.unique(group_np[valid_np]):
                        members=np.flatnonzero(valid_np & (group_np==group_id))
                        if len(members)<2:continue
                        query_k=min(k+1,len(members))
                        _,local_neighbors=cKDTree(xy[members]).query(xy[members],k=query_k)
                        local_neighbors=np.asarray(local_neighbors).reshape(len(members),query_k)
                        for local_row,global_row in enumerate(members):
                            chosen=members[local_neighbors[local_row]]
                            chosen=chosen[chosen!=global_row][:k]
                            neighbors[global_row,:len(chosen)]=torch.from_numpy(chosen.copy())
                            neighbor_mask[global_row,:len(chosen)]=True
                    neighbor_indices_validated=True
                    # This reduction is CPU-side because the neighbor tensors
                    # are built on CPU. The model uses it to skip an otherwise
                    # unnecessary GPU spatial-attention launch for empty
                    # neighborhoods.
                    neighbor_edges_present=bool(neighbor_mask.any())
                packed_min_seqlen=(int(row_lengths.min()) if self.packed and len(row_lengths)
                                   else None)
                packed_max_seqlen=(int(row_lengths.max()) if self.packed and len(row_lengths)
                                   else None)
                packed_layout_validated=bool(
                    self.packed and output_ptr[0]==0 and
                    output_ptr[-1]==len(output_gene) and output_mask.all()
                )
                # These three sidecars are immutable for the lifetime of the
                # batch.  Moving them with the data lets the Flash-BLA path
                # consume precomputed CSR metadata instead of launching a
                # subtraction, int32 cast, and token-length gather in every
                # block.
                row_lengths_tensor=torch.from_numpy(
                    np.asarray(row_lengths, dtype=np.int64).copy()
                )
                token_length_source=(
                    row_lengths if self.packed
                    else np.repeat(row_lengths, width)
                )
                token_lengths_tensor=torch.from_numpy(
                    np.asarray(token_length_source, dtype=np.int64).copy()
                )
                cu_seqlens_tensor=torch.from_numpy(
                    np.asarray(output_ptr, dtype=np.int32).copy()
                )
                yield SparseTokenBatch(
                    torch.from_numpy(output_gene.reshape(-1)),
                    torch.from_numpy(output_value.reshape(-1)),
                    torch.from_numpy(output_ptr),
                    torch.from_numpy(np.array(panel,copy=True)).long(),
                    torch.from_numpy(output_mask.reshape(-1)),
                    modality=modality, modality_observed=observed,
                    neighbor_indices=neighbors, neighbor_mask=neighbor_mask,
                    token_sampling_weight=torch.from_numpy(output_weight.reshape(-1)),
                    cell_type_target=cell_type, anatomy_target=anatomy,
                    context_target=targets, spatial_coordinates=coordinates,
                    packed=self.packed,
                    packed_layout_validated=packed_layout_validated,
                    packed_min_seqlen=packed_min_seqlen,
                    packed_max_seqlen=packed_max_seqlen,
                    neighbor_indices_validated=neighbor_indices_validated,
                    neighbor_edges_present=neighbor_edges_present,
                    row_ids=torch.from_numpy(token_row_ids),
                    row_lengths=row_lengths_tensor,
                    token_lengths=token_lengths_tensor,
                    cu_seqlens=cu_seqlens_tensor,
                )
        if remaining_skip:
            raise ValueError(f'start_row exceeds assigned rows by {remaining_skip}')

def shard_loader(root, split='train', workers=0, pin_memory=True, batch_size=256, prefetch_factor=4,
                 spatial_neighbors=8, zero_tokens_per_cell=16, max_padded_tokens=60000,
                 start_row=0, distributed=True, packed=False, packed_group_cells=32,
                 shuffle=False, seed=0, balance_waves=8):
    dataset=NumpyShardDataset(root,split,batch_size,spatial_neighbors,zero_tokens_per_cell,
                              max_padded_tokens,start_row,distributed,packed,packed_group_cells,
                              shuffle,seed,balance_waves)
    return torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=workers,pin_memory=pin_memory,
                                       persistent_workers=workers>0,
                                       prefetch_factor=prefetch_factor if workers else None)


def representative_shard_loader(root, split='validation', shard_samples=96, seed=20260912,
                                **loader_kwargs):
    """Yield one deterministic batch from shards spread over the full split."""
    root=Path(root);files=sorted(root.glob(f'{split}-shard-*-gene_index.npy'))
    if not files:raise FileNotFoundError(f'no NumPy shards for {split} in {root}')
    count=max(1,min(int(shard_samples),len(files)))
    selected=[files[i] for i in np.unique(np.linspace(0,len(files)-1,count,dtype=np.int64))]
    np.random.default_rng(int(seed)).shuffle(selected)
    dataset=NumpyShardDataset(root,split=split,distributed=False,shuffle=False,
                              **loader_kwargs)
    for gene_file in selected:
        dataset.files=[gene_file]
        yield next(iter(dataset))
