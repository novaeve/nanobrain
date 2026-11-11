"""Complete-method baselines; component bundles are search/probe utilities only."""
from __future__ import annotations
import dataclasses,json,time
from pathlib import Path
import numpy as np
from .contracts import VERSION,TASKS,digest

NATIVE=['xgboost','celltypist','harmony','scanorama','bbknn','scvi_scanvi',
        'geneformer_v1_10m','scgpt_brain','scgpt_human','scfit','scnode','trajectorynet',
        'geneformer_v1_10m_finetuned','scgpt_brain_finetuned','scgpt_human_finetuned',
        'scbrain_original','graphst_composition','spagcn_composition','stagate_composition','scuniflow_pretrain_readout']
NOT_APPLICABLE={
 'uce':'not a current library component; optional external assets/adapter not included in required coverage'}
CONTROLS={
 'control_mlp':{},
 'component_scbrain':dict(encoder='scbrain_swiglu',classification_head='scbrain',regression_head='scbrain'),
 'component_scbrain_gcn':dict(encoder='scbrain_swiglu',spatial='scbrain_gcn'),
 'component_scnode':dict(encoder='scnode_mlp',flow='scnode_time_mlp'),
 'component_scvi':dict(encoder='scvi_fc'),
 'component_geneformer_bert_random':dict(encoder='panel_patches',attention='bert'),
 'component_scgpt_transformer_random':dict(encoder='panel_patches',attention='torch_transformer'),
 'component_scfit':dict(encoder='panel_patches',attention='scfit_block',flow='scfit_adaln'),
 'component_trajectorynet':dict(flow='trajectorynet'),
 'component_scuniflow_field':dict(flow='scuniflow_field'),
 'component_graphst':dict(spatial='graphst_sparse'),
 'component_celltypist_head':dict(classification_head='celltypist_linear'),
 'component_harmony':dict(input_component='harmony'),
 'component_scanorama':dict(input_component='scanorama'),
 'component_bbknn':dict(input_component='bbknn'),
 'component_scfit_head':dict(classification_head='scfit',regression_head='scfit'),
 'component_nbeats':dict(residual='nbeats')}

# Registration is local to this checkout; historical experiment adapters are untouched.
from . import original_baselines,spatial_originals,scuniflow_original


def default_optimizer():
    return dict(name='adamw',params={},weight_decay=.0001,grad_clip=1.,schedule=dict(name='constant',warmup_steps=0,min_lr_ratio=0.))


def control_bundle(name,input_dim=256):
    from nanobrain.genome import CandidateGenome
    from nanobrain.architecture_migration import legacy_to_graph
    opts=dict(CONTROLS[name]);component=opts.pop('input_component',None)
    g=CandidateGenome(hidden_dim=96,dropout=.05,**(dict(flow='scnode_time_mlp') | opts))
    # Controls share a compact adaptation recipe; none are advertised as pretrained full methods.
    a=legacy_to_graph(g,input_dim);a['objectives']['spatial_niche']=dict(name='niche_composition',params={'density_weight':.1})
    if component:
        for n in a['backbone']['nodes']:
            n['inputs']=[component if ref=='features' else ref for ref in n['inputs']]
        a['backbone']['nodes'].insert(0,dict(id=component,kind='operation',name=component,inputs=['features'],params={}))
    return dict(schema_version=VERSION,parent_ids=['0'*64],target_tasks=list(TASKS),hypothesis='Frozen source-component adaptation control: '+name,
        genome=dict(architecture=a,learning_rate=.0003),optimizer=default_optimizer(),plugins=[],files={})


def native_job(job):
    from nanobrain.data import load_v3_subset
    from nanobrain.baselines import build
    from nanobrain.baselines.base import AdapterContext
    from brainwm_benchmark import evaluate
    from brainwm_benchmark.protocol import BENCHMARK_METRICS,PRIMARY_METRICS
    import torch
    torch.set_num_threads(job.get('threads',2));data=load_v3_subset(job['data'],include_test=False)
    output=Path('/work/native');output.mkdir(exist_ok=True)
    context=AdapterContext(data.train,data.validation,Path(job['data']),output,job['seed'],job['device'],job.get('threads',2),
        data.development_train,data.development_validation,job.get('options',{}))
    adapter=build(job['method']);available=adapter.availability(context)
    if not available.runnable:raise RuntimeError(f'{job["method"]}: {available.status}: {available.reason}')
    started=time.monotonic();validation=adapter.run(context)
    def scores(predictions,split):
        rows=[]
        for prediction in predictions:
            if prediction.task not in TASKS:continue
            metrics=evaluate(prediction.task,dict(prediction.arrays))
            if not isinstance(metrics,dict):metrics=dataclasses.asdict(metrics)
            # evaluator API returns a plain metric mapping in V3.
            if 'metrics' in metrics:metrics=metrics['metrics']
            np.savez_compressed(output/(split+'-'+prediction.task+'.npz'),**prediction.arrays)
            rows.append(dict(task=prediction.task,metrics={prediction.task+'.'+k:float(v) for k,v in metrics.items()},
                regime=prediction.training_regime,fit_rows=prediction.fit_rows,eval_rows=prediction.eval_rows,notes=prediction.notes))
        return rows
    val=scores(validation,'validation');options=dict(context.options)
    paths={'xgboost':output/'xgboost.joblib','celltypist':output/'celltypist.pkl',
        'geneformer_v1_10m':output/'foundation-probe.joblib','scgpt_brain':output/'foundation-probe.joblib','scgpt_human':output/'foundation-probe.joblib'}
    if job['method'] in paths:
        path=paths[job['method']]
        if not path.is_file():raise ValueError('missing frozen baseline fit artifact')
        options['foundation_probe_checkpoint' if 'probe' in path.name else job['method']+'_checkpoint']=str(path)
    elif job['method'] in {'scfit','scnode','trajectorynet'}:
        paths=list(output.rglob(job['method']+'.pt'))
        if len(paths)!=1:raise ValueError('baseline checkpoint is ambiguous or missing')
        options[job['method']+'_checkpoint']=str(paths[0])
    elif getattr(adapter,'frozen_artifact',None):
        path=output/adapter.frozen_artifact
        if not path.is_file():raise ValueError('missing validation-best original-method checkpoint')
        options[job['method']+'_checkpoint']=str(path)
        if (output/'foundation-probe.joblib').exists():options['foundation_probe_checkpoint']=str(output/'foundation-probe.joblib')
    # Integration algorithms with no inductive API independently fit unlabeled eval expression,
    # under the same frozen recipe. They are explicitly marked transductive.
    full=load_v3_subset(job['data'],include_test=True)
    test_context=dataclasses.replace(context,validation=full.test,development_validation=full.development_test,options=options)
    test=scores(adapter.run(test_context),'test')
    import hashlib
    artifacts=[]
    for artifact in output.rglob('*'):
        if artifact.is_file() and artifact.suffix in {'.pt','.pkl','.joblib'}:
            checksum=hashlib.sha256()
            with artifact.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''):checksum.update(chunk)
            artifacts.append(dict(path=str(artifact.relative_to(Path('/work'))),sha256=checksum.hexdigest()))
    return dict(status='complete',method=job['method'],seed=job['seed'],rows=data.train.size,
        validation=val,test=test,configuration_frozen_before_test=True,options=job.get('options',{}),
        adapter_metadata=adapter.metadata(),model_artifacts=artifacts,runtime_seconds=time.monotonic()-started)


def summary(records):
    from brainwm_benchmark.protocol import BENCHMARK_METRICS,PRIMARY_METRICS
    allowed={t+'.'+m for t,m,_ in (*BENCHMARK_METRICS,*PRIMARY_METRICS) if t in TASKS}
    result=[]
    for method in sorted({r['method'] for r in records}):
        runs=[r for r in records if r['method']==method and r.get('status')=='complete']
        values={}
        for r in runs:
            rows=r.get('test',[r])
            for row in rows:
                for key,value in row.get('metrics',{}).items():
                    if key in allowed and isinstance(value,(int,float)) and np.isfinite(value):values.setdefault(key,[]).append(value)
        regimes={row.get('task'):('transfer_adaptation' if row.get('adaptation') else row.get('regime','native_or_documented_adaptation')) for r in runs for row in r.get('test',[r]) if row.get('task')}
        result.append(dict(method=method,completed_jobs=len(runs),task_regimes=regimes,metrics={k:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)) if len(v)>1 else 0.,n=len(v)) for k,v in values.items()}))
    return result


def coverage_catalog():
    """Native tasks plus explicit representation transfer; no component controls."""
    from nanobrain.baselines import build
    from .transfer import missing_tasks,NON_TRANSFERABLE,POLICY
    sources={
        'scBrain':['scbrain_original'],'scFiT':['scfit'],'scNODE':['scnode'],'scVI/scANVI':['scvi_scanvi'],
        'Geneformer/BERT':['geneformer_v1_10m','geneformer_v1_10m_finetuned'],
        'scGPT':['scgpt_brain','scgpt_human','scgpt_brain_finetuned','scgpt_human_finetuned'],
        'TrajectoryNet':['trajectorynet'],'scUniFlow':['scuniflow_pretrain_readout'],'GraphST':['graphst_composition'],
        'SpaGCN':['spagcn_composition'],'STAGATE':['stagate_composition'],
        'Harmony':['harmony'],'Scanorama':['scanorama'],'BBKNN':['bbknn'],'CellTypist':['celltypist'],'XGBoost':['xgboost']}
    methods={}
    for method in NATIVE:
        native=[t for t in TASKS if t in build(method).tasks];transfer=missing_tasks(method)
        methods[method]=dict(tasks=[t for t in TASKS if t in native+transfer],native_tasks=native,transfer_tasks=transfer,
            category='full_method_with_explicit_downstream_adaptations',not_applicable_reason=NON_TRANSFERABLE.get(method,''))
    missing={source:sorted(set(required)-set(methods)) for source,required in sources.items() if set(required)-set(methods)}
    if missing:raise ValueError('component source baseline coverage is incomplete: '+str(missing))
    return dict(sources=sources,methods=methods,paired_tasks=list(TASKS),
        transfer_policy=POLICY,component_controls=[],
        excluded_component_only_source={'N-BEATS-inspired residual':'library building block only; no independent cell representation baseline'},
        policy='Each native or transferable method/task needs every seed and standard metric. Frozen full-method encoders receive fixed train-only, validation-selected task heads. Original tasks and adaptations are separately labelled; component controls are excluded.')


def merge_baselines(records):
    """Keep native scores, append only genuinely missing tasks from the same method."""
    merged=[]
    for method,seed in sorted({(r['method'],r['seed']) for r in records}):
        runs=[r for r in records if r['method']==method and r['seed']==seed]
        item=dict(status='complete' if all(r['status']=='complete' for r in runs) else 'failed',method=method,seed=seed,
            validation=[],test=[],source_job_ids=[r.get('job_id') for r in runs],runtime_seconds=sum(r.get('runtime_seconds',0) for r in runs))
        for split in ('validation','test'):
            for run in runs:
                for row in run.get(split,[]):
                    if any(old['task']==row['task'] for old in item[split]):raise ValueError('duplicate native/transfer task')
                    item[split].append(row)
        merged.append(item)
    return merged


def export_benchmark(output,records,seeds):
    import csv
    from brainwm_benchmark.protocol import BENCHMARK_METRICS
    output=Path(output);catalog=coverage_catalog();matrix=[];long=[]
    for method,definition in catalog['methods'].items():
        for task in TASKS:
            for seed in seeds:
                applicable=task in definition['tasks']
                matches=[]
                for run in records:
                    if run['method']!=method or run['seed']!=seed:continue
                    for row in run.get('test',[run]):
                        if row.get('task')==task:matches.append((run,row))
                state='not_applicable' if not applicable else 'complete' if len(matches)==1 and matches[0][0]['status']=='complete' else 'missing_or_failed'
                matrix.append(dict(method=method,task=task,seed=seed,status=state,category=definition['category'],
                    regime='transfer_adaptation' if task in definition['transfer_tasks'] else 'native_or_existing_documented_adaptation',
                    reason='' if applicable else definition['not_applicable_reason']))
                for run,row in matches:
                    for t,metric,direction in BENCHMARK_METRICS:
                        if t==task:
                            long.append(dict(method=method,task=task,seed=seed,metric=metric,value=row.get('metrics',{}).get(t+'.'+metric),direction=direction,
                                status=run['status'],regime=row.get('regime','component_backbone_and_task_head'),job_id=run.get('job_id')))
    for name,rows in [('baseline-coverage-matrix.csv',matrix),('baseline-standard-metrics.csv',long)]:
        if rows:
            with (output/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Full-method benchmark','', 'Three paired seeds. Validation chooses weights; test results are used during architecture search. Downstream adaptations are labelled explicitly. No standalone component controls.','',
           '| Method | Task metric | Regime | Mean | SD | Seeds |','|---|---|---|---:|---:|---:|']
    for method in summary(records):
        for metric,item in method['metrics'].items():
            if metric!='integration.scib_composite':lines.append(f"| {method['method']} | {metric} | {method['task_regimes'].get(metric.split('.')[0],'')} | {item['mean']:.7g} | {item['sd']:.5g} | {item['n']} |")
    (output/'BASELINE_BENCHMARK.md').write_text('\n'.join(lines)+'\n')
    return catalog
