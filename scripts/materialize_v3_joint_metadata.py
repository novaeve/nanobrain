#!/usr/bin/env python3
"""Materialize train-fitted joint-task metadata aligned to V3 shards."""
from __future__ import annotations
import argparse,functools,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brainwm_data import GlobalBenchmarkIndex

MISSING={"","nan","none","null","<na>"}
def text(value):
    if pd.isna(value): return None
    value=str(value).strip(); return None if value.lower() in MISSING else value
def atomic_save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as h: np.save(h,value,allow_pickle=False)
    os.replace(tmp,path)
def vocab(values): return {value:i for i,value in enumerate(sorted(set(values)))}

CATEGORICAL_SAMPLE_FIELDS={
  'dataset':'dataset','assay':'assay_type','technology':'technology','donor':'donor_id',
  'life_stage':'life_stage','sex':'sex','disease_status':'disease_status',
  'time_status':'time_status','split_group':'split_group','split_unit':'split_unit',
}
CELL_LABEL_FIELDS=('cl_id','cl_term','original_cell_type','original_subclass')
ANATOMY_LABEL_FIELDS=('uberon_id','uberon_term','original_anatomical_region')
NUMERIC_SAMPLE_FIELDS=('time_point_days','time_lower_days','time_upper_days')
ELIGIBILITY_FIELDS=('annotation_eligible','development_eligible','spatial_eligible','integration_eligible')
EVALUATION_ID_FIELDS=('evaluation_dataset','evaluation_donor','evaluation_sample','evaluation_batch','evaluation_cell_type','evaluation_anatomy')

@functools.lru_cache(maxsize=None)
def stable_id(value):
    value=text(value)
    if value is None:return -1
    return int.from_bytes(hashlib.blake2b(value.encode(),digest_size=8,person=b'brainwm3').digest(),'little') & ((1<<63)-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark-dir',type=Path,required=True); ap.add_argument('--shard-dir',type=Path,required=True)
    ap.add_argument('--split',default='train',choices=['train','validation','test']); ap.add_argument('--task',default='integration')
    ap.add_argument('--row-group-batch-size',type=int,default=1)
    ap.add_argument('--evaluation-only',action='store_true',help='backfill evaluator IDs without rewriting model-input sidecars')
    args=ap.parse_args()
    idx=GlobalBenchmarkIndex(args.benchmark_dir); all_samples=pd.read_parquet(args.benchmark_dir/'task_context_labels.parquet')
    train_samples=all_samples[all_samples.split.astype(str).eq('train')].copy()
    samples=all_samples[all_samples.split.astype(str).eq(args.split)].copy(); by_sample=samples.set_index('sample_key',drop=False)
    vocab_path=args.shard_dir/'joint_metadata_vocab.json'
    if args.split!='train' or args.evaluation_only:
      if not vocab_path.exists():raise RuntimeError('materialize train vocabulary before validation/test metadata')
      vocabs=json.loads(vocab_path.read_text())
    else:
      annotation=pd.read_parquet(args.benchmark_dir/'annotation_vocab_train.parquet')
      anatomy=pd.read_parquet(args.benchmark_dir/'anatomy_vocab_train.parquet')
      effective_sections=[text(row.section_id) or (f'sample::{row.sample_key}' if str(row.assay_type)=='spatialRNA' else None) for row in train_samples.itertuples()]
      vocabs={name:vocab(filter(None,map(text,train_samples[column]))) for name,column in CATEGORICAL_SAMPLE_FIELDS.items()}
      vocabs.update({'section':vocab(filter(None,effective_sections)),'sample':vocab(map(str,train_samples.sample_key)),
        'cell_type':{str(v):i for i,v in enumerate(annotation.annotation_label.astype(str))},
        'anatomy':{str(v):i for i,v in enumerate(anatomy.uberon_id.astype(str))}})
      source_cells=idx.source_artifact/'cells.parquet'
      import pyarrow.parquet as pq
      cell_schema=set(pq.read_schema(source_cells).names)
      raw_label_fields=[field for field in (*CELL_LABEL_FIELDS,*ANATOMY_LABEL_FIELDS) if field in cell_schema]
      raw_labels=pd.read_parquet(source_cells,columns=['sample_key',*raw_label_fields])
      raw_labels=raw_labels[raw_labels.sample_key.astype(str).isin(set(train_samples.sample_key.astype(str)))]
      for field in raw_label_fields:vocabs[field]=vocab(filter(None,map(text,raw_labels[field])))
      # The contract columns named integration_*_labels describe which factors
      # evaluators must use; their values are schema strings, not row labels.
      # Keep a stable primary target while retaining every component sidecar.
      vocabs['integration_batch']=dict(vocabs['dataset'])
      vocabs['integration_biology']=dict(vocabs['cell_type'])
      vocab_path.write_text(json.dumps(vocabs,indent=2,ensure_ascii=False)+'\n')
    shard_ptrs=sorted(args.shard_dir.glob(f'{args.split}-shard-*-row_ptr.npy')); shard_i=0; buffered=[]; buffered_n=0; total=0
    columns=('source_id','row_index','sample_key',*CELL_LABEL_FIELDS,*ANATOMY_LABEL_FIELDS,'spatial_x','spatial_y','spatial_z')
    def encode(frame):
      context=by_sample.loc[frame.sample_key.astype(str)].reset_index(drop=True)
      n=len(frame); out={}
      raw_cell_labels=[]
      for row in frame.itertuples(index=False):
        raw_cell_labels.append(next((text(getattr(row,c)) for c in ('cl_id','cl_term','original_cell_type','original_subclass') if text(getattr(row,c)) is not None),None))
      batch_keys=["|".join(str(text(x) or '<missing>') for x in values) for values in zip(context.assay_type,context.technology,context.dataset,context.donor_id,context.sample_key)]
      out.update({
        'evaluation_dataset':np.asarray([stable_id(x) for x in context.dataset],dtype=np.int64),
        'evaluation_donor':np.asarray([stable_id(f'{text(d) or "<missing>"}|{text(x) or "<missing>"}') for d,x in zip(context.dataset,context.donor_id)],dtype=np.int64),
        'evaluation_sample':np.asarray([stable_id(x) for x in context.sample_key],dtype=np.int64),
        'evaluation_batch':np.asarray([stable_id(x) for x in batch_keys],dtype=np.int64),
        'evaluation_cell_type':np.full(n,-1,dtype=np.int64) if args.split=='test' else np.asarray([stable_id(x) for x in raw_cell_labels],dtype=np.int64),
        'evaluation_anatomy':np.asarray([stable_id(x) for x in frame.uberon_id],dtype=np.int64),
      })
      if args.evaluation_only:return out
      for field,column in CATEGORICAL_SAMPLE_FIELDS.items():
        out[field]=np.asarray([vocabs[field].get(text(x),-1) for x in context[column]],dtype=np.int32)
      sections=[text(row.section_id) or (f'sample::{row.sample_key}' if str(row.assay_type)=='spatialRNA' else None) for row in context.itertuples()]
      out['section']=np.asarray([vocabs['section'].get(x,-1) for x in sections],dtype=np.int32)
      out['sample']=np.asarray([vocabs['sample'].get(str(x),-1) for x in context.sample_key],dtype=np.int32)
      for field in CELL_LABEL_FIELDS:
        values=np.asarray([vocabs[field].get(text(x),-1) for x in frame[field]],dtype=np.int32)
        out[field]=np.full(n,-1,dtype=np.int32) if args.split=='test' else values
      for field in ANATOMY_LABEL_FIELDS:
        out[field]=np.asarray([vocabs[field].get(text(x),-1) for x in frame[field]],dtype=np.int32)
      labels=raw_cell_labels
      out['cell_type']=np.full(n,-1,dtype=np.int32) if args.split=='test' else np.asarray(labels,dtype=np.int32)
      out['integration_batch']=out['dataset'].copy()
      out['integration_biology']=out['cell_type'].copy()
      out['anatomy']=np.asarray([vocabs['anatomy'].get(text(x),-1) for x in frame.uberon_id],dtype=np.int32)
      for field in NUMERIC_SAMPLE_FIELDS:
        out[field]=pd.to_numeric(context[field],errors='coerce').to_numpy(np.float32)
      out['chronological_time_proxy_days']=out['time_point_days'].copy()
      for field in ELIGIBILITY_FIELDS:
        out[field]=context[field].fillna(False).to_numpy(np.bool_)
      out['spatial']=frame[['spatial_x','spatial_y','spatial_z']].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
      return out
    def emit(force=False):
      nonlocal shard_i,buffered,buffered_n,total
      while shard_i<len(shard_ptrs):
        need=len(np.load(shard_ptrs[shard_i],mmap_mode='r'))-1
        if buffered_n<need and not force:return
        if buffered_n<need:raise RuntimeError('metadata ended before expression')
        joined={k:np.concatenate([x[k] for x in buffered]) for k in buffered[0]}
        stem=str(shard_ptrs[shard_i])[:-len('-row_ptr.npy')]
        for k,value in joined.items():atomic_save(Path(stem+f'-{k}.npy'),value[:need])
        remainder={k:value[need:] for k,value in joined.items()}; buffered=[remainder] if len(next(iter(remainder.values()))) else []
        buffered_n-=need; total+=need; shard_i+=1
    for frame in idx.iter_cell_index(args.split,task=args.task,columns=columns,row_group_batch_size=args.row_group_batch_size):
      for _,part in frame.groupby('source_id',sort=False):
        encoded=encode(part.sort_values('row_index',kind='stable').reset_index(drop=True)); buffered.append(encoded); buffered_n+=len(part); emit()
    emit(True)
    if shard_i!=len(shard_ptrs) or buffered_n:raise RuntimeError(f'alignment mismatch {shard_i}/{len(shard_ptrs)} remainder={buffered_n}')
    split_manifest=json.loads((args.shard_dir/f'{args.split}-manifest.json').read_text())
    receipt={'format':'brainwm_v3_evaluation_labels_v1' if args.evaluation_only else 'brainwm_v3_joint_metadata_v2','split_id':split_manifest['split_id'],'split':args.split,'task':args.task,'rows':total,'shards':shard_i,'vocabulary_fit_split':'train','vocab_sizes':{k:len(v) for k,v in vocabs.items()},'evaluation_id_semantics':'stable_blake2b_63bit_not_model_input','evaluation_id_fields':list(EVALUATION_ID_FIELDS),'integration_targets':{'primary_batch':'dataset','primary_biology':'cell_type','batch_factors':['assay','technology','dataset','donor','sample'],'biology_factors':['anatomy','development','sex','cell_type']},'missing_value':-1,'test_cell_type_sealed':args.split=='test','test_labels_opened':False}
    suffix='evaluation-label-manifest' if args.evaluation_only else 'joint-metadata-manifest'
    (args.shard_dir/f'{args.split}-{suffix}.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt))
if __name__=='__main__':main()
