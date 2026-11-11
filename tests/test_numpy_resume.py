import numpy as np

from brainwm.numpy_shard_loader import (NumpyShardDataset, balanced_shard_order,
    rank_offsets_for_global_prefix, sample_measured_zeros, shard_loader)


def test_fast_measured_zero_sampler_matches_materialized_complement():
    panel=np.array([2,4,7,9,11,15,20,31],dtype=np.int64)
    rows=(
        np.array([],dtype=np.int64),
        np.array([2,7,15]),
        np.array([7,7,8,31]),
        panel.copy(),
    )
    for row in rows:
        candidates=np.setdiff1d(panel,row,assume_unique=False)
        for count in (0,1,3,16):
            size=min(count,len(candidates))
            for seed in range(20):
                expected=np.random.default_rng(seed).choice(
                    candidates,size=size,replace=False,
                )
                actual,candidate_count=sample_measured_zeros(
                    panel,row,count,np.random.default_rng(seed),
                )
                assert candidate_count==len(candidates)
                np.testing.assert_array_equal(actual,expected)


def test_fast_measured_zero_sampler_handles_duplicate_row_indices():
    """The rank mapping remains exact when a hand-built row repeats a gene."""
    panel=np.arange(16,dtype=np.int64)
    row=np.array([1,1,2,9,9],dtype=np.int64)
    candidates=np.setdiff1d(panel,row,assume_unique=False)
    for count in (0,1,3,16):
        for seed in range(20):
            expected=np.random.default_rng(seed).choice(
                candidates,size=min(count,len(candidates)),replace=False,
            )
            actual,candidate_count=sample_measured_zeros(
                panel,row,count,np.random.default_rng(seed),
            )
            assert candidate_count==len(candidates)
            np.testing.assert_array_equal(actual,expected)
def test_numpy_loader_resumes_at_exact_global_row(tmp_path):
    np.save(tmp_path / 'panel_source_id.npy', np.array([0]))
    np.save(tmp_path / 'panel_ptr.npy', np.array([0, 3]))
    np.save(tmp_path / 'panel_gene_index.npy', np.array([0, 1, 2]))
    for shard, values in enumerate(([10, 11], [20, 21, 22])):
        stem = tmp_path / f'train-shard-{shard:05d}'
        np.save(f'{stem}-gene_index.npy', np.arange(len(values)))
        np.save(f'{stem}-values.npy', np.asarray(values, dtype=np.float32))
        np.save(f'{stem}-row_ptr.npy', np.arange(len(values) + 1))
        np.save(f'{stem}-row_source_id.npy', np.zeros(len(values), dtype=np.int64))
    dataset = NumpyShardDataset(tmp_path, batch_size=8, zero_tokens_per_cell=0,
                                start_row=3)
    batch = next(iter(dataset))
    assert batch.values.tolist() == [21.0, 22.0]


def test_single_rank_prefix_maps_to_rank_local_shard_offsets(tmp_path):
    for shard,rows in enumerate((2,3,4,5)):
        np.save(tmp_path/f'train-shard-{shard:05d}-row_ptr.npy',np.arange(rows+1))
    assert rank_offsets_for_global_prefix(tmp_path,'train',8,2)==[5,3]


def test_shuffled_loader_is_deterministic_complete_and_resumable(tmp_path):
    np.save(tmp_path / 'panel_source_id.npy', np.array([0]))
    np.save(tmp_path / 'panel_ptr.npy', np.array([0, 1]))
    np.save(tmp_path / 'panel_gene_index.npy', np.array([0]))
    for shard in range(5):
        values=np.arange(shard*10,shard*10+6,dtype=np.float32)
        stem=tmp_path/f'train-shard-{shard:05d}'
        np.save(f'{stem}-gene_index.npy',np.zeros(len(values),dtype=np.int64))
        np.save(f'{stem}-values.npy',values)
        np.save(f'{stem}-row_ptr.npy',np.arange(len(values)+1))
        np.save(f'{stem}-row_source_id.npy',np.zeros(len(values),dtype=np.int64))

    def batches(start=0,seed=13):
        dataset=NumpyShardDataset(tmp_path,batch_size=2,zero_tokens_per_cell=0,
                                 start_row=start,shuffle=True,seed=seed)
        return [batch.values.tolist() for batch in dataset]

    full=batches()
    assert full==batches()
    assert full!=batches(seed=14)
    assert sorted(value for batch in full for value in batch)==sorted(
        [float(shard*10+row) for shard in range(5) for row in range(6)])
    consumed=sum(map(len,full[:4]))
    assert batches(start=consumed)==full[4:]


def test_main_loader_worker_count_preserves_batch_sequence(tmp_path):
    np.save(tmp_path/'panel_source_id.npy',np.array([0]))
    np.save(tmp_path/'panel_ptr.npy',np.array([0,10]))
    np.save(tmp_path/'panel_gene_index.npy',np.arange(10))
    for shard in range(4):
        stem=tmp_path/f'train-shard-{shard:05d}'
        genes=np.tile(np.arange(6)%10,1)
        np.save(f'{stem}-gene_index.npy',genes)
        np.save(f'{stem}-values.npy',np.arange(shard*10,shard*10+6,dtype=np.float32))
        np.save(f'{stem}-row_ptr.npy',np.arange(7))
        np.save(f'{stem}-row_source_id.npy',np.zeros(6,dtype=np.int64))

    def signature(workers,start_row=0):
        loader=shard_loader(
            tmp_path,workers=workers,pin_memory=False,batch_size=2,
            zero_tokens_per_cell=2,max_padded_tokens=100,
            start_row=start_row,distributed=False,packed=True,
            shuffle=True,seed=19,balance_waves=2,
        )
        return [(
            batch.gene_index.tolist(),batch.values.tolist(),batch.row_ptr.tolist(),
            batch.token_sampling_weight.tolist(),
        ) for batch in loader]

    direct=signature(0)
    assert signature(1)==direct
    assert signature(2)==direct
    assert signature(3)==direct
    assert signature(2,start_row=5)==signature(0,start_row=5)


def test_spatial_graph_uses_private_source_group_when_section_is_oov(tmp_path):
    np.save(tmp_path/'panel_source_id.npy',np.array([7]))
    np.save(tmp_path/'panel_ptr.npy',np.array([0,1]))
    np.save(tmp_path/'panel_gene_index.npy',np.array([0]))
    stem=tmp_path/'train-shard-00000'
    np.save(f'{stem}-gene_index.npy',np.zeros(3,dtype=np.int64))
    np.save(f'{stem}-values.npy',np.ones(3,dtype=np.float32))
    np.save(f'{stem}-row_ptr.npy',np.arange(4))
    np.save(f'{stem}-row_source_id.npy',np.full(3,7,dtype=np.int64))
    np.save(f'{stem}-section.npy',np.full(3,-1,dtype=np.int32))
    np.save(f'{stem}-spatial.npy',np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],dtype=np.float32))
    batch=next(iter(NumpyShardDataset(tmp_path,batch_size=3,zero_tokens_per_cell=0)))
    assert batch.neighbor_mask.any(1).tolist()==[True,True,True]


def test_balanced_shard_order_is_complete_reproducible_and_density_local(tmp_path):
    files=[]
    for shard,density in enumerate(range(1,17)):
        stem=tmp_path/f'train-shard-{shard:05d}'
        gene=tmp_path/f'train-shard-{shard:05d}-gene_index.npy'
        np.save(gene,np.zeros(density*4,dtype=np.int32))
        np.save(f'{stem}-row_ptr.npy',np.arange(0,density*4+1,density))
        files.append(gene)
    first=balanced_shard_order(files,consumers=4,seed=11,balance_waves=2)
    second=balanced_shard_order(files,consumers=4,seed=11,balance_waves=2)
    assert [path for _,path in first]==[path for _,path in second]
    assert sorted(index for index,_ in first)==list(range(16))
    for start in range(0,16,8):
        densities=[]
        for _,path in first[start:start+8]:
            shard=int(path.name.split('-')[2]);densities.append(shard+1)
        assert max(densities)-min(densities)<=7
