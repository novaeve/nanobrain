"""Streaming adapter from the frozen human global row-plan reader."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .foundation_moe import BrainWMInput
from .sparse_tokens import reader_batch_to_sparse, collate_sparse_batches


def sparse_batch_to_input(batch: dict, vocab_size: int, device='cpu') -> BrainWMInput:
    """Project a reader sparse batch into dense model input without confusing zeros."""
    counts = np.zeros((len(batch['row_ptr']) - 1, vocab_size), dtype=np.float32)
    observed = np.zeros_like(counts, dtype=bool)
    genes = np.asarray(batch['gene_index']); values = np.asarray(batch['count'])
    for row in range(len(counts)):
        lo, hi = int(batch['row_ptr'][row]), int(batch['row_ptr'][row + 1])
        valid = genes[lo:hi]
        counts[row, valid] = values[lo:hi]
        observed[row, np.asarray(batch['measured_gene_index'], dtype=int)] = True
    frame = batch.get('cells')
    spatial_obs = None
    relation = None
    if frame is not None and {'spatial_x', 'spatial_y'}.issubset(frame.columns):
        coords = frame[['spatial_x', 'spatial_y']].notna().all(axis=1).to_numpy()
        spatial_obs = torch.from_numpy(coords).to(device)
        # Coordinates alone do not establish neighbors. Keep identity until
        # a section-local neighborhood sampler supplies explicit indices.
    return BrainWMInput(torch.from_numpy(counts).to(device),
                        torch.from_numpy(observed).to(device),
                        spatial_relation=relation, spatial_observed=spatial_obs)


def iter_human_batches(benchmark_dir: str | Path, batch_size=256,
                       split='train', task='integration', vocab_size=30000,
                       device='cpu'):
    """Yield every selected human cell exactly once, source-locally and lazily."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pretrain_dataset'))
    from brainwm_data import GlobalBenchmarkIndex
    index = GlobalBenchmarkIndex(benchmark_dir)
    # Larger row-group batches make checkpoint resume practical while the
    # exact row accounting below still enforces the frozen population.
    reader = index.reader(split, task=task, row_group_batch_size=32)
    try:
        for raw in reader.iter_local_batches(batch_size=batch_size):
            yield sparse_batch_to_input(raw, vocab_size, device)
    finally:
        reader.close()


def iter_human_sparse_batches(benchmark_dir: str | Path, batch_size=256,
                              split='train', task='integration', device='cpu', start_row=0,
                              zero_tokens_per_cell=0, sampling_seed=0, max_rows=None):
    """Yield reader batches in bounded sparse token form."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pretrain_dataset'))
    from brainwm_data import GlobalBenchmarkIndex
    index = GlobalBenchmarkIndex(benchmark_dir)
    if not isinstance(start_row,int) or start_row<0:
        raise ValueError('start_row must be a nonnegative integer')
    remaining_skip=start_row
    skipped_rows = 0
    context=index.samples.set_index('sample_key')
    reader = index.reader(split, task=task, row_group_batch_size=32)
    try:
        for raw in reader.iter_local_batches(batch_size=batch_size):
            rows=len(raw['row_ptr'])-1
            if remaining_skip>=rows:
                remaining_skip-=rows
                skipped_rows += rows
                if skipped_rows and skipped_rows % 1000000 < rows:
                    print(f'resume-skip rows={skipped_rows}/{start_row}', flush=True)
                continue
            if remaining_skip:
                # Slice CSR storage before collation or any stochastic transform.
                first=int(raw['row_ptr'][remaining_skip])
                raw=dict(raw)
                raw['gene_index']=raw['gene_index'][first:]
                raw['count']=raw['count'][first:]
                raw['row_ptr']=np.asarray(raw['row_ptr'][remaining_skip:])-first
                if raw.get('cells') is not None:
                    raw['cells']=raw['cells'].iloc[remaining_skip:].reset_index(drop=True)
                remaining_skip=0
            if max_rows is not None:
                available = max_rows - start_row
                if available <= 0:
                    break
                if rows > available:
                    cut = int(raw['row_ptr'][available])
                    raw = dict(raw)
                    raw['gene_index'] = raw['gene_index'][:cut]
                    raw['count'] = raw['count'][:cut]
                    raw['row_ptr'] = np.asarray(raw['row_ptr'][:available + 1])
                    if raw.get('cells') is not None:
                        raw['cells'] = raw['cells'].iloc[:available].reset_index(drop=True)
                    rows = available
            if raw.get('cells') is not None:
                frame=raw['cells'].copy()
                for column in ['donor_id','dataset','assay_type','life_stage','time_point_days',
                               'time_status','time_basis','split_group','section_id']:
                    if column in context:
                        frame[column]=frame.sample_key.map(context[column])
                frame['global_split']=frame.sample_key.map(context['split'])
                if not frame.global_split.eq(split).all():
                    raise ValueError('batch metadata escaped the requested global split')
                raw=dict(raw,cells=frame)
            if zero_tokens_per_cell:
                from .measured_zero_tokens import append_measured_zeros
                frame=raw['cells']
                seeds=[np.random.SeedSequence([sampling_seed,int(s),int(r)]).generate_state(1)[0]
                       for s,r in zip(frame.source_id,frame.row_index)]
                raw=append_measured_zeros(raw,zero_tokens_per_cell,
                    np.random.default_rng(sampling_seed),row_seeds=seeds)
            yield collate_sparse_batches([raw], device=device)
            start_row += rows
        if remaining_skip:
            raise ValueError('resume offset exceeds selected corpus')
    finally:
        reader.close()
