"""Frozen-champion action: nested train sizes, unchanged V3 heldout/protocol."""
from __future__ import annotations
import csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
from .workflow import file_sha,save


def stable_id(value):
    if value is None or str(value).strip().lower() in {'','unknown','nan','none','<na>'}:return -1
    return int.from_bytes(hashlib.blake2b(str(value).strip().encode(),digest_size=8,person=b'brainwm3').digest(),'little') & ((1<<63)-1)


def nested_order(total,base_refs,seed):
    """The initial 60k is compulsory; every larger size contains the preceding set."""
    from .csr import _splitmix64
    refs=np.arange(total,dtype=np.int64);mask=np.ones(total,dtype=bool);mask[base_refs]=False
    rest=refs[mask];priorities=_splitmix64(rest,seed)
    return np.concatenate([np.asarray(base_refs),rest[np.lexsort((rest,priorities))]])


def verify_scale(base,scale,previous=None):
    base,scale=Path(base),Path(scale)
    bm=json.loads((base/'manifest.json').read_text());sm=json.loads((scale/'manifest.json').read_text())
    for name in ('panel_gene_index.npy','normalization_mean.npy','normalization_std.npy','niche_vocabulary.json'):
        if file_sha(base/name)!=file_sha(scale/name):raise ValueError('scale changed fixed preprocessing/vocabulary')
    if sm['niche_contract']['radii_by_source']!=bm['niche_contract']['radii_by_source']:raise ValueError('scale recalibrated radii')
    for split in ('validation','test'):
        if sm['splits'][split]!=bm['splits'][split]:raise ValueError('scale changed heldout manifest')
        for info in bm['splits'][split]['files'].values():
            if file_sha(scale/info['path'])!=info['sha256']:raise ValueError('heldout file changed')
    refs=np.load(scale/'train/source_row_ref.npy',mmap_mode='r');small=np.load((Path(previous) if previous else base)/'train/source_row_ref.npy')
    if len(np.unique(refs))!=len(refs) or not np.isin(small,refs).all():raise ValueError('non-nested/duplicate training rows')
    return dict(status='verified',train_rows=len(refs),heldout_rows={s:sm['splits'][s]['rows'] for s in ('validation','test')})


def build_scale(base,source,contract,cache,output,refs):
    """Streaming CSR -> fixed-panel memmaps; only frozen verified sections get niche labels.

    No H5AD access, new normalization fitting, anatomy proxy or holdout resampling.
    The 98 verified full-section cache units are held fixed across scale points;
    effective niche counts are reported and may saturate before corpus size.
    """
    import pandas as pd
    from .csr import _discover,_gather,_dense_features,_panel_catalog,_neighbors
    from nanobrain.niche_data import neighborhood_targets
    base,source,contract,cache,output=map(Path,(base,source,contract,cache,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);train=output/'train';train.mkdir()
    manifest=json.loads((base/'manifest.json').read_text());n=len(refs);
    shards=_discover(source,'train');source_receipt=json.loads((source/'train-manifest.json').read_text())
    if not source_receipt.get('complete') or sum(s.rows for s in shards)!=source_receipt['rows']:
        raise ValueError('incomplete frozen training shards; refusing a false full-data result')
    panel=np.load(base/'panel_gene_index.npy');panels=_panel_catalog(source)
    mean=np.load(base/'normalization_mean.npy');std=np.load(base/'normalization_std.npy')
    for f in base.iterdir():
        if f.is_file() and f.name!='manifest.json':shutil.copy2(f,output/f.name)
    for split in ('validation','test'):(output/split).symlink_to((base/split).resolve(),target_is_directory=True)
    refs=np.sort(refs);np.save(train/'source_row_ref.npy',refs);np.save(train/'row_id.npy',refs)
    arrays={}
    def put(name,value,start,end):
        if name not in arrays:arrays[name]=np.lib.format.open_memmap(train/(name+'.npy'),mode='w+',dtype=value.dtype,shape=(n,*value.shape[1:]))
        arrays[name][start:end]=value
    samples=pd.read_parquet(contract/'samples.parquet');samples['sample_hash']=samples.sample_key.map(stable_id)
    if samples.sample_hash.duplicated().any():raise ValueError('sample hash collision')
    lookup=samples.set_index('sample_hash',drop=False)
    proof=pd.read_parquet(base/'verified_spatial_units.parquet')
    verified=set(proof.loc[proof.niche_geometry_ready & proof.independent_section_key_verified,'sample_key'])
    full_metadata=json.loads((cache/'metadata.json').read_text())
    niche_contract=manifest['niche_contract'];classes=niche_contract['class_count']-1
    vocabulary=np.array([r['source_code'] for r in json.loads((base/'niche_vocabulary.json').read_text())[:-1]],dtype=np.int64)
    # Verify the full-section cache is the one used by V3, including target labels.
    base_receipt=json.loads((base/'niche_target_receipt.json').read_text())
    if full_metadata['fingerprint']!=base_receipt['source_metadata']:raise ValueError('full-section cache provenance mismatch')
    written=0
    for shard in shards:
        lo,hi=np.searchsorted(refs,[shard.offset,shard.offset+shard.rows]);selected=refs[lo:hi]
        if not len(selected):continue
        if lo!=written:raise ValueError('unmaterialized training row gap')
        written=hi
        features,observed,library=_dense_features([shard],selected,panel,panels)
        features=((features-mean)/std).astype(np.float32)
        gather=lambda field,dtype=None:_gather([shard],selected,field,dtype)
        ids=gather('evaluation_sample',np.int64)
        if not np.isin(ids,lookup.index).all():raise ValueError('unknown canonical sample')
        rows=lookup.loc[ids]
        source_ids=gather('row_source_id',np.int32)
        if not rows.split.eq('train').all() or not np.array_equal(rows.source_id.to_numpy(),source_ids):raise ValueError('frozen sample split/source mismatch')
        annotation=gather('cell_type',np.int32);batch=gather('integration_batch',np.int32)
        if np.all(batch<0):batch=gather('dataset',np.int32)
        time=gather('time_point_days',np.float32);life=gather('life_stage',np.int16);development=(time/np.float32(365.25)).reshape(-1,1)
        section=np.array([stable_id('section::'+s) if s in verified else -1 for s in rows.sample_key],dtype=np.int64)
        donors=np.array([stable_id(str(r.dataset)+'::'+str(r.donor_id)) if pd.notna(r.donor_id) else -1 for r in rows.itertuples()],dtype=np.int64)
        chunk=dict(features=features,feature_observed=observed,library_size=library,completion_target=features,
            completion_mask=np.zeros_like(observed),source_split=np.zeros(len(selected),np.int8),source_id=source_ids,sample=ids,
            section=section,donor=donors,split_group=rows.split_group.map(stable_id).to_numpy(np.int64),
            coordinates=gather('spatial',np.float32),label_annotation=annotation,label_integration=batch,
            mask_annotation=gather('annotation_eligible',bool)&(annotation>=0),mask_integration=gather('integration_eligible',bool)&(batch>=0),
            label_development=development,mask_development=gather('development_eligible',bool)[:,None]&np.isfinite(development),
            development_condition=np.where(life[:,None]>=0,(life[:,None].astype(np.float32)+1)/5,0).astype(np.float32),development_condition_observed=life>=0)
        for name in ('assay','dataset','technology','life_stage','sex','time_point_days','time_lower_days','time_upper_days','chronological_time_proxy_days',
                     'annotation_eligible','development_eligible','spatial_eligible','integration_eligible','anatomy'):chunk[name]=gather(name)
        for name,value in chunk.items():put(name,value,lo,hi)
    if written!=n:raise ValueError('not all requested training rows were materialized')
    for a in arrays.values():a.flush()
    # kNN is restricted to previously verified section identities; unknown sections remain masked.
    neighbors,neighbor_mask=_neighbors(arrays['coordinates'],arrays['source_id'],arrays['section'],8)
    np.save(train/'neighbors.npy',neighbors);np.save(train/'neighbor_mask.npy',neighbor_mask)
    target=np.lib.format.open_memmap(train/'label_niche.npy',mode='w+',dtype='float32',shape=(n,4,classes+1));target[:]=0
    density=np.lib.format.open_memmap(train/'label_niche_density.npy',mode='w+',dtype='float32',shape=(n,4));density[:]=0
    masks=np.zeros((n,4),bool);density_mask=np.zeros((n,4),bool)
    finite_centers=np.isfinite(arrays['coordinates'][:,:2]).all(axis=1)
    for record in full_metadata['sections']:
        if record['split']!='train' or str(record['source_id']) not in niche_contract['radii_by_source']:continue
        take=np.flatnonzero((arrays['sample']==record['sample_id']) & finite_centers)
        if not len(take):continue
        with np.load(cache/record['path']) as full:
            pos=np.searchsorted(full['row_refs'],refs[take])
            if np.any(pos>=len(full['row_refs'])) or not np.array_equal(full['row_refs'][pos],refs[take]):raise ValueError('section row identity mismatch')
            if not np.allclose(full['coordinates'][pos],arrays['coordinates'][take,:2],atol=1e-5):raise ValueError('section coordinate mismatch')
            raw=full['labels'];encoded=np.searchsorted(vocabulary,raw);known=(encoded<classes)&(vocabulary[encoded.clip(0,classes-1)]==raw)
            encoded=np.where(known,encoded,-1)
            p,d,m=neighborhood_targets(full['coordinates'],encoded,full['row_refs'],arrays['coordinates'][take,:2],refs[take],
                niche_contract['radii_by_source'][str(record['source_id'])],classes,min_known_fraction=niche_contract['min_known_fraction'])
            target[take],density[take],masks[take]=p,d,m;density_mask[take]=True
    target.flush();density.flush();np.save(train/'mask_niche.npy',masks);np.save(train/'mask_niche_density.npy',density_mask)
    # Rebuild unpaired endpoint groups using canonical sample hashes for BOTH endpoints.
    _development_groups(train,contract,source)
    # Preserve the exact initial rows (targets/masks and selected neighbors included), remapping indices.
    initial=np.load(base/'train/source_row_ref.npy');positions=np.searchsorted(refs,initial)
    for path in (base/'train').glob('*.npy'):
        name=path.stem
        if name.startswith('development_') and name in {'development_source_index','development_target_index','development_pair_ptr'}:continue
        destination=train/path.name
        if not destination.exists():continue
        old=np.load(path,mmap_mode='r');new=np.load(destination,mmap_mode='r+')
        if old.shape[0]!=len(initial) or old.shape[1:]!=new.shape[1:]:continue
        values=np.asarray(old)
        if name=='neighbors':values=positions[values]
        new[positions]=values;new.flush()
    files={}
    for p in train.iterdir():
        info=dict(path='train/'+p.name,sha256=file_sha(p))
        if p.suffix=='.npy':a=np.load(p,mmap_mode='r');info.update(dtype=str(a.dtype),shape=list(a.shape))
        files[p.stem]=info
    manifest['splits']['train']=dict(rows=n,files=files,task_rows=dict(annotation=int(arrays['mask_annotation'].sum()),integration=int(arrays['mask_integration'].sum()),
        spatial_niche=int(masks.any(1).sum()),development=int(np.load(train/'development_pair_ptr.npy')[-1])),spatial_edges=int(neighbor_mask.sum()))
    manifest['selection']=dict(mode='nested_training_only',train_rows=n,train_refs_sha256=file_sha(train/'source_row_ref.npy'),
        base_manifest_sha256=file_sha(base/'manifest.json'),fixed_validation_rows=manifest['splits']['validation']['rows'],fixed_test_rows=manifest['splits']['test']['rows'])
    receipt=dict(base_receipt);receipt['same_rows_and_splits']=False;receipt['fixed_holdout']=True
    receipt['parent_manifest_sha256']=file_sha(base/'manifest.json');receipt['splits']=dict(base_receipt.get('splits',{}))
    receipt['splits']['train']=dict(rows=n,valid_centers=int(masks.any(1).sum()),valid_per_scale=masks.sum(0).tolist(),density_centers=int(density_mask.any(1).sum()))
    save(output/'niche_target_receipt.json',receipt)
    for artifact in manifest['niche_contract'].get('artifacts',[]):artifact['sha256']=file_sha(output/artifact['path'])
    manifest['scaling']=dict(base_manifest_sha256=file_sha(base/'manifest.json'),nested_train=True,train_rows=n,
        fixed_holdout=True,fixed_panel=True,fixed_normalization=True,fixed_niche_sections=True,all_training_rows_materialized=True)
    save(output/'manifest.json',manifest);verify_scale(base,output)
    return output


def _development_groups(train,contract,source):
    import pandas as pd
    vocab=json.loads((source/'joint_metadata_vocab.json').read_text())['anatomy']
    intervals=pd.read_parquet(contract/'task_development_labels.parquet');intervals=intervals[intervals.split.eq('train')].set_index('interval_id')
    groups=pd.read_parquet(contract/'development_interval_anatomy.parquet');groups=groups[groups.split.eq('train')]
    sample=np.load(train/'sample.npy');anatomy=np.load(train/'anatomy.npy');lookup={}
    for i,(s,a) in enumerate(zip(sample,anatomy)):lookup.setdefault((int(s),int(a)),[]).append(i)
    src=[];tgt=[];ptr=[0];metadata=[];rng=np.random.default_rng(20260915)
    for g in groups.itertuples():
        if g.interval_id not in intervals.index or str(g.uberon_id) not in vocab:continue
        interval=intervals.loc[g.interval_id];code=vocab[str(g.uberon_id)]
        def members(keys):
            if isinstance(keys,str):keys=json.loads(keys)
            return np.array([i for key in keys for i in lookup.get((stable_id(key),code),[])],dtype=np.int32)
        a=members(interval.source_sample_keys);b=members(interval.target_sample_keys);n=min(len(a),len(b))
        if n<2:continue
        src.append(rng.choice(a,n,replace=False));tgt.append(rng.choice(b,n,replace=False));ptr.append(ptr[-1]+n)
        metadata.append(dict(interval_id=str(g.interval_id),uberon_id=str(g.uberon_id),source_time_days=float(interval.source_time_days),target_time_days=float(interval.target_time_days),cells=n))
    if not metadata:raise ValueError('no valid developmental intervals in scale data')
    np.save(train/'development_source_index.npy',np.concatenate(src));np.save(train/'development_target_index.npy',np.concatenate(tgt));np.save(train/'development_pair_ptr.npy',np.array(ptr,np.int64))
    save(train/'development_pairs.json',dict(format='nanobrain_population_pairs_v1',split='train',groups_metadata=metadata,groups=len(metadata),pairs=ptr[-1]))


def run_scaling_action(workflow,frozen):
    from .workflow import source_fingerprint
    if source_fingerprint(workflow.root)!=frozen['source_sha256']:raise ValueError('trusted/model sources changed after champion freeze')
    cfg=workflow.cfg['scaling'];base=workflow.data;root=workflow.output/'scaling';root.mkdir(exist_ok=True)
    total=json.loads((Path(cfg['source'])/'train-manifest.json').read_text())['rows']
    initial=np.load(base/'train/source_row_ref.npy');order=nested_order(total,initial,cfg['sampling_seed'])
    rows=[];previous=base
    holdout=sum(json.loads((base/'manifest.json').read_text())['splits'][s]['rows'] for s in ('validation','test'))
    points=scale_points(cfg,total,holdout)
    for point in points:
        size=point['train_rows']
        if size>total:raise ValueError('requested training scale exceeds frozen training corpus')
        data=base if size==len(initial) else root/'data'/str(size)
        if not (data/'manifest.json').exists():build_scale(base,cfg['source'],cfg['contract'],cfg['niche_cache'],data,order[:size])
        verify_scale(base,data,previous);previous=data
        steps=math.ceil(cfg['epochs']*size/workflow.cfg['train']['batch_size'])
        # Architecture, plugins, loss and optimizer are immutable; all weights restart from scratch.
        champion=workflow.jobs(workflow.candidate_jobs(frozen['bundle'],'frozen_champion',data,steps))
        baselines=workflow.baseline_phase(data,steps);runs=champion+baselines
        if any(r['status']!='complete' for r in runs):raise RuntimeError('scale point is incomplete; no partial success reported')
        save(root/(str(size)+'-results.json'),runs)
        for run in runs:
            for record in run.get('test',[run]):
                for metric,value in record.get('metrics',{}).items():
                    rows.append(dict(dataset_rows=point['dataset_rows'],scale_label=point['label'],train_rows=size,method=run['method'],seed=run['seed'],task=record['task'],metric=metric,
                        value=value,steps=run.get('steps'),effective_train_rows=run.get('effective_train_rows',record.get('fit_rows')),row_samples_seen=run.get('row_samples_seen'),population_source_cells_seen=run.get('population_source_cells_seen'),runtime_seconds=run['runtime_seconds'],regime=record.get('regime','backbone_and_task_head')))
        write_curves(root,rows)
    result=dict(status='complete',frozen_champion=frozen['candidate_id'],curve=str(root/'learning-curves.csv'),
        fixed_validation_test=True,test_used_during_search=True)
    save(root/'action-receipt.json',result);return result


def scale_points(cfg,total_train,holdout_rows):
    """Named corpus sizes include the unchanged validation/test cells.

    Thus 100k retains the original 60k/20k/20k benchmark; larger points
    increase training rows without resampling or consuming either holdout.
    """
    if 'dataset_sizes' not in cfg:
        return [dict(train_rows=total_train if n=='full_train' else n,
                     dataset_rows=total_train+holdout_rows if n=='full_train' else n+holdout_rows,
                     label='full' if n=='full_train' else str(n+holdout_rows)) for n in cfg['train_sizes']]
    result=[]
    for requested in cfg['dataset_sizes']:
        full=requested in ('full','full_train')
        size=total_train if full else int(requested)-holdout_rows
        if size<=0 or size>total_train:raise ValueError('invalid dataset scale or scale exceeds frozen training corpus')
        result.append(dict(train_rows=size,dataset_rows=size+holdout_rows,label='full' if full else str(requested)))
    if any(a['train_rows']>=b['train_rows'] for a,b in zip(result,result[1:])):raise ValueError('scale points must strictly increase')
    return result


def write_curves(root,rows):
    from brainwm_benchmark.protocol import BENCHMARK_METRICS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);allowed={t+'.'+m for t,m,_ in BENCHMARK_METRICS};rows=[r for r in rows if r['metric'] in allowed]
    if not rows:return
    with (root/'learning-curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for metric in sorted({r['metric'] for r in rows}):
        fig,ax=plt.subplots(figsize=(8,5))
        for method in sorted({r['method'] for r in rows if r['metric']==metric}):
            points={}
            for r in rows:
                if r['metric']==metric and r['method']==method:points.setdefault(r.get('dataset_rows',r['train_rows']),[]).append(r['value'])
            x=sorted(points);means=[np.mean(points[n]) for n in x];sd=[np.std(points[n],ddof=1) if len(points[n])>1 else 0 for n in x]
            ax.errorbar(x,means,yerr=sd,marker='o',label=method)
        ax.set_xscale('log');ax.set_xlabel('Dataset cells (fixed validation/test; training rows recorded separately)');ax.set_ylabel(metric);ax.legend(fontsize=6);fig.tight_layout()
        fig.savefig(root/(metric+'.png'),dpi=160);fig.savefig(root/(metric+'.pdf'));plt.close(fig)
