#!/usr/bin/env python3
"""Materialize the frozen V3 sparse row plan into Torch CPU shards.

This is the only stage allowed to read H5AD/Parquet. Training consumes the
resulting ``.pt`` shards and records the exact parallelism/batch contract.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import torch
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brainwm_data import GlobalBenchmarkIndex

def main():
    p=argparse.ArgumentParser(); p.add_argument('--benchmark-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--split',default='train')
    p.add_argument('--task',default='integration'); p.add_argument('--rows-per-shard',type=int,default=65536)
    p.add_argument('--batch-size',type=int,default=256); p.add_argument('--workers',type=int,default=1)
    p.add_argument('--row-group-batch-size',type=int,default=1)
    p.add_argument('--resume',action='store_true',help='resume after complete atomic shards')
    p.add_argument('--max-rows',type=int,default=None,help='diagnostic only; incomplete output is never training-ready')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.benchmark_dir/'manifest.json').read_text())
    if manifest['split_id']!='BRAINWM-HUMAN-GLOBAL-6-2-2-V3': raise SystemExit('unexpected split')
    expected=int(manifest['counts']['task_cells_by_split'][args.task][args.split])
    existing=sorted(args.output.glob(f'{args.split}-shard-*-row_ptr.npy'))
    if existing and not args.resume:
        raise SystemExit(f'{len(existing)} existing {args.split} shards; pass --resume or use an empty output')
    completed_rows=0
    for path in existing:
        completed_rows += len(np.load(path,mmap_mode='r'))-1
    idx=GlobalBenchmarkIndex(args.benchmark_dir); reader=idx.reader(args.split,task=args.task,row_group_batch_size=args.row_group_batch_size)
    buf=[]; rows=completed_rows; shard=len(existing); started=time.time()
    # Build the complete immutable panel catalog from the compact gene map.
    # It must not depend on which sources happen to be visited after resume.
    panel_by_source={int(source_id):np.unique(part.loc[part.gene_index.ge(0),'gene_index'].to_numpy(np.int32))
                     for source_id,part in reader.gene_map.groupby('source_id',sort=True)}
    catalog_source=args.output/'panel_source_id.npy'; catalog_ptr=args.output/'panel_ptr.npy'; catalog_gene=args.output/'panel_gene_index.npy'
    if args.resume and catalog_source.exists() and catalog_ptr.exists() and catalog_gene.exists():
        old_source=np.load(catalog_source,mmap_mode='r'); old_ptr=np.load(catalog_ptr,mmap_mode='r'); old_gene=np.load(catalog_gene,mmap_mode='r')
        restored={int(source_id):np.asarray(old_gene[int(old_ptr[i]):int(old_ptr[i+1])]) for i,source_id in enumerate(old_source)}
        for source_id,panel in restored.items():
            if source_id in panel_by_source and not np.array_equal(panel_by_source[source_id],panel):
                raise RuntimeError(f'existing panel catalog mismatch for source {source_id}')
    def atomic_save(path, value):
        tmp=Path(str(path)+'.tmp')
        with tmp.open('wb') as handle: np.save(handle,value,allow_pickle=False)
        os.replace(tmp,path)
    def save_catalog():
        source_ids=np.asarray(sorted(panel_by_source),dtype=np.int32); ptr=[0]; genes=[]
        for source_id in source_ids:
            genes.append(panel_by_source[int(source_id)]); ptr.append(ptr[-1]+len(genes[-1]))
        if genes:
            atomic_save(args.output/'panel_source_id.npy',source_ids)
            atomic_save(args.output/'panel_ptr.npy',np.asarray(ptr,dtype=np.int64))
            atomic_save(args.output/'panel_gene_index.npy',np.concatenate(genes))
    def save_progress():
        payload={'format':'brainwm_v3_numpy_materialization_progress_v1','split_id':manifest['split_id'],
                 'task':args.task,'split':args.split,'rows':rows,'expected_rows':expected,
                 'complete':False,'shards':shard,'updated_at':time.time()}
        target=args.output/f'{args.split}-progress.json'; tmp=Path(str(target)+'.tmp')
        tmp.write_text(json.dumps(payload,indent=2)+'\n'); os.replace(tmp,target)
    def flush():
        nonlocal buf,shard
        if not buf:return
        # Persist raw NumPy arrays. Training never opens H5AD/Parquet again.
        gi=np.concatenate([x['gene_index'].numpy() for x in buf]); vv=np.concatenate([x['values'].numpy() for x in buf])
        rp=[]; base=0
        for x in buf:
            q=x['row_ptr'].numpy(); rp.extend((q[1:]+base).tolist()); base += int(q[-1])
        stem=args.output/f'{args.split}-shard-{shard:05d}'
        atomic_save(Path(str(stem)+'-gene_index.npy'),gi); atomic_save(Path(str(stem)+'-values.npy'),vv); atomic_save(Path(str(stem)+'-row_ptr.npy'),np.asarray([0]+rp,dtype=np.int64))
        row_source=np.concatenate([np.full(x['row_ptr'].numel()-1,x['source_id'],dtype=np.int32) for x in buf])
        atomic_save(Path(str(stem)+'-row_source_id.npy'),row_source)
        shard+=1; buf=[]; save_catalog(); save_progress()
    try:
      for raw in reader.iter_local_batches(batch_size=args.batch_size,start_row=completed_rows):
        n=len(raw['row_ptr'])-1
        if args.max_rows is not None and rows-completed_rows >= args.max_rows: break
        source_id=int(raw['source_id']); panel=np.asarray(raw['measured_gene_index'],dtype=np.int32)
        previous=panel_by_source.setdefault(source_id,panel)
        if not np.array_equal(previous,panel): raise RuntimeError(f'measured panel changed within source {source_id}')
        buf.append({'gene_index':torch.as_tensor(raw['gene_index'],dtype=torch.int32),'values':torch.as_tensor(raw['count'],dtype=torch.float32),'row_ptr':torch.as_tensor(raw['row_ptr'],dtype=torch.int64),'source_id':source_id})
        rows+=n
        if sum(int(x['row_ptr'].numel()-1) for x in buf)>=args.rows_per_shard: flush()
        if rows and rows%(args.rows_per_shard*10)<n: print(f'materialize split={args.split} rows={rows} shards={shard}',flush=True)
    finally: reader.close(); flush()
    source_ids=np.asarray(sorted(panel_by_source),dtype=np.int32); panel_ptr=[0]; panel_genes=[]
    for source_id in source_ids:
        panel_genes.append(panel_by_source[int(source_id)]); panel_ptr.append(panel_ptr[-1]+len(panel_genes[-1]))
    if panel_genes: save_catalog()
    complete=rows==expected and args.max_rows is None
    if rows>expected: raise RuntimeError(f'materialized too many rows: {rows}>{expected}')
    receipt={'format':'brainwm_v3_numpy_sparse_shards_v3','split_id':manifest['split_id'],'task':args.task,'split':args.split,'rows':rows,'expected_rows':expected,'complete':complete,'shards':shard,'rows_per_shard':args.rows_per_shard,'panel_semantics':'row_source_id_to_source_panel_catalog','source':'one_time_GlobalBenchmarkIndex_materialization','training_contract':{'workers':args.workers,'local_batch_size':args.batch_size,'global_batch_size':args.batch_size*args.workers,'row_group_batch_size':args.row_group_batch_size,'prefetch_factor':4,'pin_memory':True,'load':'numpy_mmap_then_torch'},'elapsed_seconds':time.time()-started}
    (args.output/f'{args.split}-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt))
if __name__=='__main__': main()
