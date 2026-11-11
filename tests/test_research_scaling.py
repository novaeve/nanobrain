"""Exercise the real streaming scale builder on a tiny invented CSR corpus."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nanobrain.research.scaling import build_scale,stable_id,verify_scale
from nanobrain.research.workflow import save,file_sha
from nanobrain.research.csr import _discover,_dense_features,_panel_catalog,_neighbors
from nanobrain.niche_data import neighborhood_targets


def test_streaming_csr_scale_keeps_parent_rows_targets_and_holdout(tmp_path):
    source=tmp_path/'source';source.mkdir();base=tmp_path/'base';base.mkdir();contract=tmp_path/'contract';contract.mkdir();cache=tmp_path/'cache';cache.mkdir()
    n=24;panel=np.arange(4,dtype=np.int64);stem=source/'train-shard-00000'
    sample=np.array([stable_id('A')]*12+[stable_id('B')]*12,dtype=np.int64)
    xy=np.c_[np.tile(np.arange(12),2),np.repeat([0.,100.],12),np.zeros(n)].astype(np.float32)
    labels=np.arange(n,dtype=np.int32)%2
    payload=dict(row_ptr=np.arange(n+1,dtype=np.int64)*4,gene_index=np.tile(panel,n),values=np.arange(n*4,dtype=np.float32)%7+1,
        row_source_id=np.ones(n,np.int32),evaluation_sample=sample,cell_type=labels,anatomy=np.ones(n,np.int32),
        spatial=xy,integration_batch=np.arange(n,dtype=np.int32)%2,annotation_eligible=np.ones(n,bool),integration_eligible=np.ones(n,bool),
        development_eligible=np.ones(n,bool),spatial_eligible=np.ones(n,bool),life_stage=np.ones(n,np.int16),time_point_days=np.repeat([1.,2.],12).astype(np.float32))
    for name,array in payload.items():np.save(str(stem)+'-'+name+'.npy',array)
    for name,value in [('source_id',np.array([1])),('ptr',np.array([0,4])),('gene_index',panel)]:np.save(source/('panel_'+name+'.npy'),value)
    save(source/'train-manifest.json',dict(rows=n,complete=True))
    save(source/'joint_metadata_vocab.json',dict(anatomy={'UBERON:1':1}))
    pd.DataFrame([dict(sample_key=s,split='train',source_id=1,dataset='toy',donor_id=d,split_group='g'+s) for s,d in [('A','a'),('B','b')]]).to_parquet(contract/'samples.parquet',index=False)
    pd.DataFrame([dict(interval_id='i',split='train',source_sample_keys=json.dumps(['A']),target_sample_keys=json.dumps(['B']),source_time_days=1.,target_time_days=2.)]).to_parquet(contract/'task_development_labels.parquet',index=False)
    pd.DataFrame([dict(interval_id='i',split='train',uberon_id='UBERON:1')]).to_parquet(contract/'development_interval_anatomy.parquet',index=False)
    pd.DataFrame([dict(sample_key=s,niche_geometry_ready=True,independent_section_key_verified=True) for s in ('A','B')]).to_parquet(base/'verified_spatial_units.parquet',index=False)
    metadata=dict(fingerprint=dict(toy='fixed'),sections=[])
    for s in ('A','B'):
        refs=np.flatnonzero(sample==stable_id(s));path='train-'+s+'.npz';np.savez(cache/path,coordinates=xy[refs,:2],labels=labels[refs],row_refs=refs)
        metadata['sections'].append(dict(split='train',sample_id=stable_id(s),path=path,source_id=1,rows=len(refs),known_fraction=1.,sample_key=s))
    save(cache/'metadata.json',metadata);save(base/'niche_target_receipt.json',dict(source_metadata=metadata['fingerprint']))
    save(base/'niche_vocabulary.json',[dict(index=i,source_code=c) for i,c in enumerate([0,1,-1])])
    np.save(base/'panel_gene_index.npy',panel);np.save(base/'normalization_mean.npy',np.zeros(4,np.float32));np.save(base/'normalization_std.npy',np.ones(4,np.float32))
    nc=dict(mode='composition',input='expression_only',class_count=3,unknown_class_index=2,scales=[10,20,50,100],radii_by_source={'1':[1.5,2.5,3.5,5.5]},min_known_fraction=.8,artifacts=[])
    initial=np.r_[np.arange(6),np.arange(12,18)].astype(np.int64);train=base/'train';train.mkdir()
    features,observed,library=_dense_features(_discover(source,'train'),initial,panel,_panel_catalog(source))
    data=dict(features=features,feature_observed=observed,source_row_ref=initial,row_id=initial,sample=sample[initial],label_annotation=labels[initial],
        mask_annotation=np.ones(12,bool),mask_integration=np.ones(12,bool),label_integration=payload['integration_batch'][initial],coordinates=xy[initial],
        development_source_index=np.arange(6,dtype=np.int32),development_target_index=np.arange(6,12,dtype=np.int32),development_pair_ptr=np.array([0,6],np.int64))
    target=np.zeros((12,4,3),np.float32);density=np.zeros((12,4),np.float32);mask=np.zeros((12,4),bool)
    for s in ('A','B'):
        allrefs=np.flatnonzero(sample==stable_id(s));idx=np.flatnonzero(sample[initial]==stable_id(s))
        p,d,m=neighborhood_targets(xy[allrefs,:2],labels[allrefs],allrefs,xy[initial[idx],:2],initial[idx],nc['radii_by_source']['1'],2)
        target[idx],density[idx],mask[idx]=p,d,m
    data.update(label_niche=target,label_niche_density=density,mask_niche=mask,mask_niche_density=np.ones_like(mask))
    sections=np.array([stable_id('section::'+('A' if i<12 else 'B')) for i in initial]);neighbors,neighbor_mask=_neighbors(xy[initial],np.ones(12),sections,8)
    data.update(neighbors=neighbors,neighbor_mask=neighbor_mask)
    for key,array in data.items():np.save(train/(key+'.npy'),array)
    save(train/'development_pairs.json',dict(format='nanobrain_population_pairs_v1',split='train',groups_metadata=[dict(source_time_days=1.,target_time_days=2.,interval_id='i',uberon_id='UBERON:1')]))
    manifest=dict(format='nanobrain_v3_subset_v1',split_id='BRAINWM-HUMAN-GLOBAL-6-2-2-V3',panel=dict(size=4),normalization={},niche_contract=nc,splits={})
    for split in ('train','validation','test'):
        directory=base/split;directory.mkdir(exist_ok=True)
        if split!='train':
            np.save(directory/'features.npy',np.zeros((3,4),np.float32));np.save(directory/'source_row_ref.npy',np.arange(3,dtype=np.int64))
        files={p.stem:dict(path=split+'/'+p.name,sha256=file_sha(p)) for p in directory.iterdir()}
        manifest['splits'][split]=dict(rows=12 if split=='train' else 3,files=files)
    save(base/'manifest.json',manifest)
    before={p.name:file_sha(p) for p in train.iterdir()}
    output=build_scale(base,source,contract,cache,tmp_path/'scaled',np.arange(n,dtype=np.int64))
    assert verify_scale(base,output)['train_rows']==24
    assert {p.name:file_sha(p) for p in train.iterdir()}==before
    assert np.array_equal(np.load(output/'train/features.npy')[initial],features)
    assert np.array_equal(np.load(output/'train/label_niche.npy')[initial],target)
    assert np.array_equal(np.load(output/'train/mask_niche.npy')[initial],mask)
    assert not (output/'train/label_niche_anatomy_proxy.npy').exists()
