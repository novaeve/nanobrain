import numpy as np

from brainwm.development_numpy_loader import (
    load_referenced_rows,rank_group_positions,
)


def test_rank_group_positions_resume_the_infinite_cycle_exactly():
    group_count=17;world=4;step=23
    for rank in range(world):
        base=list(range(rank,group_count,world))
        resumed=rank_group_positions(group_count,rank,world,step)
        expected=(base*20)[step%len(base):step%len(base)+len(base)]
        assert resumed==expected


def test_referenced_rows_packed_matches_padded_valid_tokens(tmp_path):
    stem=tmp_path/"train-shard-00000"
    np.save(f"{stem}-row_ptr.npy",np.array([0,2,5],dtype=np.int64))
    np.save(f"{stem}-gene_index.npy",np.array([1,2,3,4,5],dtype=np.int64))
    np.save(f"{stem}-values.npy",np.array([.1,.2,.3,.4,.5],dtype=np.float32))
    categorical=("assay","technology","donor","life_stage","cell_type","anatomy")
    temporal=("time_point_days","time_lower_days","time_upper_days",
              "chronological_time_proxy_days")
    for index,name in enumerate(categorical):
        np.save(f"{stem}-{name}.npy",np.array([index,index+1],dtype=np.int32))
    for index,name in enumerate(temporal):
        np.save(f"{stem}-{name}.npy",np.array([10+index,20+index],dtype=np.float32))

    references=np.array([0,1],dtype=np.int64)
    padded=load_referenced_rows(tmp_path,references,packed=False)
    packed=load_referenced_rows(tmp_path,references,packed=True)

    assert not padded.packed
    assert packed.packed
    assert packed.row_ptr.tolist()==[0,2,5]
    assert packed.token_mask.all()
    assert packed.gene_index.tolist()==[1,2,3,4,5]
    assert np.allclose(packed.values.numpy(),[.1,.2,.3,.4,.5])
    assert padded.gene_index[padded.token_mask].tolist()==packed.gene_index.tolist()
    assert np.allclose(
        padded.values[padded.token_mask].numpy(),packed.values.numpy(),
    )
    assert padded.modality.keys()==packed.modality.keys()
    expected_padded=np.repeat(np.arange(padded.batch_size),3)
    expected_packed=np.repeat(np.arange(packed.batch_size),[2,3])
    np.testing.assert_array_equal(padded.token_row_ids().numpy(),expected_padded)
    np.testing.assert_array_equal(packed.token_row_ids().numpy(),expected_packed)
