"""Trusted worker entrypoint. No CLI is invoked during preparation except probes."""
from __future__ import annotations
import argparse,json,resource,time,math
from pathlib import Path


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value


def write(value):
    value=json_safe(value)
    p=Path('/work/receipt.json');temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)


def candidate_job(job):
    import torch
    from nanobrain.data import load_v3_subset,infer_label_cardinalities,niche_model_options,DatasetSplits
    from nanobrain.genome import CandidateGenome
    from nanobrain.model import NanoBrainModel,NanoBrainModelConfig
    from nanobrain.training import train_candidate,TrainConfig
    from nanobrain.task_training import evaluate_task
    from .plugins import install_plugins
    bundle=job['bundle'];install_plugins(bundle);torch.set_num_threads(job.get('threads',2))
    # Test is not opened until the best validation checkpoint has been persisted.
    data=load_v3_subset('/dataset',include_test=False)
    counts=infer_label_cardinalities(data);width=data.train.features.shape[1]
    config=NanoBrainModelConfig(width,counts['annotation'],counts['niche'],counts['integration'],width,**niche_model_options(data))
    genome=CandidateGenome.from_dict({**bundle['genome'],'seed':job['seed']})
    opts=dict(job['train']);opts.update(seed=job['seed'],task=job['task'],optimizer_spec=bundle['optimizer'],
        grad_clip=bundle['optimizer']['grad_clip'],weight_decay=bundle['optimizer']['weight_decay'],checkpoint_policy='best_weights')
    if NanoBrainModel(config,genome).parameter_count>20_000_000:raise ValueError('real-task parameter budget exceeded')
    started=time.monotonic();receipt=train_candidate(data,config,genome,TrainConfig(**opts),output_dir='/work/training')
    if not receipt['stability']['passed'] or not receipt['optimizer_changed_parameter']:raise ValueError('unstable/ineffective candidate')
    saved=torch.load(receipt['best_checkpoint'],map_location='cpu',weights_only=False)
    model=NanoBrainModel(config,genome);model.load_state_dict(saved['model']);device=torch.device(opts['device']);model.to(device)
    full=load_v3_subset('/dataset',include_test=True)
    from nanobrain.integration_components import prepare_component_views
    heldout=DatasetSplits(full.train,full.test,None,full.development_train,full.development_test)
    prepare_component_views(heldout,genome)
    metrics=evaluate_task(model,heldout,job['task'],job.get('evaluation_batch_size',2048),device,development_groups=0)
    return dict(status='complete',method=job['method'],seed=job['seed'],task=job['task'],
        metrics=metrics,validation=receipt.get('best_validation',receipt.get('validation',{})),checkpoint=receipt['best_checkpoint'],
        checkpoint_selection='validation',test_used_for_architecture_search=True,
        rows=full.train.size,steps=opts['steps'],effective_train_rows=(len(data.development_train.source_index) if job['task']=='development' else int(data.train.masks['niche'].any(1).sum()) if job['task']=='spatial_niche' else int(data.train.masks['annotation' if job['task']=='annotation' else 'integration'].sum())),row_samples_seen=receipt['row_samples_seen'],population_source_cells_seen=receipt['population_source_cells_seen'],samples_seen_upper_bound=opts['steps']*(opts.get('development_cells',64)*2 if job['task']=='development' else opts['batch_size']),
        runtime_seconds=time.monotonic()-started,training=receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['validate','candidate','native','transfer']);p.add_argument('input');args=p.parse_args()
    import os
    cpus=sorted(os.sched_getaffinity(0));index=int(os.environ.get('NANOBRAIN_CPU_INDEX','0'))%len(cpus)
    os.sched_setaffinity(0,{cpus[index],cpus[(index+1)%len(cpus)]})
    resource.setrlimit(resource.RLIMIT_NOFILE,(4096,4096));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if args.action=='validate':
        resource.setrlimit(resource.RLIMIT_AS,(24<<30,24<<30));resource.setrlimit(resource.RLIMIT_CPU,(180,180));resource.setrlimit(resource.RLIMIT_FSIZE,(1<<30,1<<30))
        from .validation import validate_in_worker
        result=validate_in_worker(json.loads(Path(args.input).read_text()),'/work')
    elif args.action=='candidate':result=candidate_job(json.loads(Path(args.input).read_text()))
    elif args.action=='transfer':
        from .transfer import transfer_job
        result=transfer_job(json.loads(Path(args.input).read_text()))
    else:
        from .benchmark import native_job
        result=native_job(json.loads(Path(args.input).read_text()))
    write(result)

if __name__=='__main__':main()
