"""Resumable baseline -> ReAct evolution -> freeze -> scale workflow."""
from __future__ import annotations
import concurrent.futures,contextlib,json,math,queue,statistics,sys,threading,time
from pathlib import Path
from .contracts import digest,TASKS,VERSION,semantic_id
from .store import Store
from .sandbox import launch
from .benchmark import NATIVE,CONTROLS,NOT_APPLICABLE,control_bundle,default_optimizer,summary
from .selection import select_parents
from .agent import ToolSession,propose


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def file_sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    if cfg['gpu_ids']!=[0,1,2,3,4]:raise ValueError('this experiment is restricted to physical GPUs 0..4')
    if cfg['checkpoint_selection']!='validation' or cfg['architecture_selection']!='test':raise ValueError('test-driven protocol mismatch')
    if cfg['model']!='gpt-6.1-sol' or cfg['reasoning_effort']!='max':raise ValueError('model/reasoning configuration mismatch')
    if len(set(cfg['seeds']))!=len(cfg['seeds']) or len(cfg['seeds'])<3:raise ValueError('three distinct paired seeds required')
    if not 1<=cfg['per_gpu_workers']<=8:raise ValueError('invalid per-GPU concurrency')
    return cfg


class Workflow:
    def __init__(self,root,cfg,output):
        self.root=Path(root).resolve();self.cfg=cfg;self.output=Path(output).resolve();self.store=Store(self.output/'lineage')
        self.data=(self.root/cfg['data']).absolute();self.baselines=[]
        self.tokens=queue.Queue()
        for gpu in cfg['gpu_ids']:
            for _ in range(cfg['per_gpu_workers']):self.tokens.put(gpu)
        self.python=self.root/'.venv/bin/python'
        from .resources import ResourcePool
        self.resources=ResourcePool(cfg['gpu_ids'],cfg['per_gpu_workers'],dict(cfg.get('resources',{})))
        self.maximum_workers=len(cfg['gpu_ids'])*cfg['per_gpu_workers']
        self.activity_lock=threading.RLock();self.activity={};self.status=dict(status='initialized',generations=cfg.get('generations'))
        self.cpu_slots=threading.Semaphore(5)
        from .revision import adoption_plan
        self.adoptions,_=adoption_plan(self.output)
        for identity,worker in self.adoptions.items():
            receipt=self.output/'jobs'/identity/'receipt.json'
            if receipt.exists() and json.loads(receipt.read_text()).get('job_id')==identity:continue
            self.resources.leases[identity]=dict(gpu=worker['gpu'],reserve_mib=self.cfg.get('resources',{}).get('foundation_reservation_mib',12288),
                host_reserve_mib=4096,method=worker['method'],pid=worker['pid'])

    def progress(self,**fields):
        from datetime import datetime,timezone
        with self.activity_lock:
            self.status.update(fields);self.status['updated_at']=datetime.now(timezone.utc).isoformat()
            self.status['active_jobs']=list(self.activity.values());self.status['active_processes']=len(self.activity)
            save(self.output/'status.json',self.status)

    def plan(self):
        return dict(status='prepared_not_started',data=str(self.data),baseline_native=NATIVE,
            component_controls=[],excluded_methods=NOT_APPLICABLE,
            generations=self.cfg['generations'],candidates_per_generation=self.cfg['candidates_per_generation'],seeds=self.cfg['seeds'],
            task_jobs_per_generation=self.cfg['candidates_per_generation']*len(self.cfg['seeds'])*4,
            gpu_ids=self.cfg['gpu_ids'],maximum_workers=self.tokens.qsize(),
            model=self.cfg['model'],reasoning=self.cfg['reasoning_effort'],
            after_evolution='freeze then nested training-scale comparison with fixed validation/test',
            scaling=self.cfg['scaling'],formal_training_started=False,llm_called=False)

    def validate(self,identifier,bundle):
        directory=self.store.candidates/identifier/'probe';save(directory/'input.json',bundle)
        try:return launch(self.python,self.root,directory,['validate','/work/input.json'],timeout=240)
        except Exception as exc:return dict(status='rejected',error=str(exc),candidate_id=identifier)

    def jobs(self,jobs):
        self.progress(queue_total=len(jobs),queue_finished=0,queue_successful=0,queue_failed=0)
        finished=[]
        def run(job):
            identity=digest(job);directory=self.output/'jobs'/identity;receipt=directory/'receipt.json'
            if receipt.exists():
                result=json.loads(receipt.read_text())
                if result.get('status')=='complete' and result.get('job_id')==identity and result.get('data_manifest_sha256')==job['data_manifest_sha256']:return result
            if identity in self.adoptions:
                from .revision import adopt_receipt,retire_old_controller
                worker=self.adoptions[identity]
                with self.activity_lock:self.activity[identity]=dict(worker,phase='adopted_native')
                self.progress()
                try:
                    result=adopt_receipt(self.root,self.output,job,worker,self.cfg['job_timeout_seconds']);save(receipt,result)
                    retire_old_controller(self.output);return result
                finally:
                    self.resources.release(identity)
                    with self.activity_lock:self.activity.pop(identity,None)
                    self.progress()
            if job['action']=='transfer':
                dependency=self.output/'jobs'/job['baseline_job_id'];deadline=time.monotonic()+self.cfg['job_timeout_seconds']
                while True:
                    path=dependency/'receipt.json'
                    if path.exists() and json.loads(path.read_text()).get('job_id')==job['baseline_job_id']:break
                    if (dependency/'failure.json').exists() or time.monotonic()>deadline:
                        return dict(status='failed',method=job['method'],seed=job['seed'],job_id=identity,error='required native baseline did not complete')
                    time.sleep(5)
            cpu=job['action']=='native' and job['method'] in {'xgboost','celltypist','harmony','scanorama','bbknn'}
            retries=self.cfg.get('resources',{}).get('job_retries',3)
            for attempt in range(retries+1):
                if cpu:self.cpu_slots.acquire()
                gpu=self.resources.acquire(identity,job,needs_gpu=not cpu)
                with self.activity_lock:self.activity[identity]=dict(job_id=identity,method=job['method'],seed=job['seed'],task=job.get('task'),gpu=gpu,attempt=attempt+1,phase=job['action'])
                try:
                    effective=dict(job)
                    if job['action']=='candidate':effective['train']=dict(job['train'],device='cuda:0')
                    else:effective['device']='cpu' if cpu else 'cuda:0'
                    save(directory/'input.json',effective)
                    def started(pid):
                        self.resources.started(identity,pid)
                        with self.activity_lock:self.activity[identity]['pid']=pid
                        self.progress()
                    timeout=self.cfg.get('scaling_job_timeout_seconds',self.cfg['job_timeout_seconds']) if self.status.get('status')=='scaling' else self.cfg['job_timeout_seconds']
                    result=launch(self.python,self.root,directory,[job['action'],'/work/input.json'],data=job['data'],gpu=gpu,cache=job.get('input_cache'),
                        baseline=(self.output/'jobs'/job['baseline_job_id']) if job['action']=='transfer' else None,timeout=timeout,on_started=started)
                    result.update(job_id=identity,physical_gpu=gpu,data_manifest_sha256=job['data_manifest_sha256'])
                    save(receipt,result);return result
                except Exception as exc:
                    result=dict(status='failed',method=job['method'],seed=job['seed'],task=job.get('task'),job_id=identity,error=str(exc),attempt=attempt+1)
                    save(directory/f'failure-{attempt+1}.json',result)
                    log=directory/'worker.log';detail=log.read_text(errors='replace')[-20000:] if log.exists() else ''
                    infrastructure=any(t in detail.lower() for t in ('out of memory','outofmemoryerror','cuda error:','resource temporarily unavailable'))
                    if not infrastructure or attempt==retries:
                        save(directory/'failure.json',result);return result
                    if not cpu:self.resources.grow_reservation(job)
                finally:
                    if cpu:self.cpu_slots.release()
                    self.resources.release(identity)
                    with self.activity_lock:self.activity.pop(identity,None)
                    self.progress()
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.maximum_workers) as pool:
            pending=[pool.submit(run,job) for job in jobs]
            for future in concurrent.futures.as_completed(pending):
                result=future.result();results.append(result)
                self.progress(queue_finished=len(results),queue_successful=sum(r['status']=='complete' for r in results),queue_failed=sum(r['status']!='complete' for r in results))
        return results

    def candidate_jobs(self,bundle,method,data=None,steps=None):
        data=Path(data or self.data);fingerprint=file_sha(data/'manifest.json');jobs=[]
        for seed in self.cfg['seeds']:
            for task in TASKS:
                budget=steps or self.cfg['train']['steps']
                if steps is not None and task=='development':
                    import numpy as np
                    pairs=int(np.load(data/'train/development_pair_ptr.npy',mmap_mode='r')[-1])
                    budget=math.ceil(self.cfg['scaling']['epochs']*pairs/self.cfg['train']['development_cells'])
                jobs.append(dict(action='candidate',method=method,bundle=bundle,task=task,seed=seed,data=str(data),
                    data_manifest_sha256=fingerprint,input_cache=str(self.output/'trusted-input-cache'),
                    train=dict(self.cfg['train'],steps=budget),threads=2,evaluation_batch_size=2048,estimated_host_mib=2048))
        return jobs

    def prepare_inputs(self,data):
        import os
        from nanobrain.data import load_v3_subset,DatasetSplits
        from nanobrain.genome import CandidateGenome
        from nanobrain.integration_components import prepare_component_views
        cache=self.output/'trusted-input-cache';cache.mkdir(exist_ok=True)
        old={k:os.environ.get(k) for k in ('NANOBRAIN_COMPONENT_CACHE','NANOBRAIN_COMPONENT_CACHE_READONLY')}
        affinity=os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else None
        if affinity:os.sched_setaffinity(0,set(sorted(affinity)[:4]))
        os.environ['NANOBRAIN_COMPONENT_CACHE']=str(cache);os.environ['NANOBRAIN_COMPONENT_CACHE_READONLY']='0'
        try:
            loaded=load_v3_subset(data,include_test=True)
            for name in ('component_harmony','component_scanorama','component_bbknn'):
                genome=CandidateGenome.from_dict({**control_bundle(name)['genome'],'seed':self.cfg['seeds'][0]})
                prepare_component_views(loaded,genome)
                prepare_component_views(DatasetSplits(loaded.train,loaded.test,None),genome)
        finally:
            if affinity:os.sched_setaffinity(0,affinity)
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v

    def check_metrics(self,metrics):
        from brainwm_benchmark.protocol import BENCHMARK_METRICS,PRIMARY_METRICS
        for task,key,_ in (*BENCHMARK_METRICS,*PRIMARY_METRICS):
            if task not in TASKS:continue
            value=metrics.get(task+'.'+key)
            if value is None or not math.isfinite(value):raise ValueError('missing/nonfinite standard metric '+task+'.'+key)

    def evaluate_candidates(self,identifiers,generation):
        jobs=[]
        for identifier in identifiers:jobs+=self.candidate_jobs(self.store.bundle(identifier),identifier)
        results=self.jobs(jobs);self.assert_source();output=[]
        for identifier in identifiers:
            runs=[r for r in results if r['method']==identifier];expected=len(self.cfg['seeds'])*4
            if len(runs)!=expected or any(r['status']!='complete' for r in runs):
                self.store.put_result(identifier,dict(candidate_id=identifier,generation=generation,status='rejected_runtime',all_seeds_passed=False,
                    seed_results=runs,semantic_id=semantic_id(self.store.bundle(identifier)),reason='incomplete/failed seed; excluded from parent and champion selection'))
                continue
            combined={}
            for r in runs:
                for k,v in r['metrics'].items():
                    if isinstance(v,(int,float)) and math.isfinite(v):combined.setdefault(k,[]).append(v)
            means={k:statistics.mean(v) for k,v in combined.items()}
            try:self.check_metrics(means)
            except Exception as exc:
                self.store.put_result(identifier,dict(candidate_id=identifier,generation=generation,status='rejected_runtime',all_seeds_passed=False,
                    semantic_id=semantic_id(self.store.bundle(identifier)),reason=str(exc)))
                continue
            from nanobrain.evolution import joint_score
            node=dict(candidate_id=identifier,generation=generation,status='complete',all_seeds_passed=True,
                selection_metrics=means,search_score=joint_score(means),seed_results=runs,
                semantic_id=semantic_id(self.store.bundle(identifier)))
            self.store.put_result(identifier,node);output.append(node)
        return output

    def native_options(self):
        options=dict(self.cfg['native_options'])
        for key,value in dict(gene_catalog=self.root/'assets/human_v3_genes.parquet',
            geneformer_root=self.root/'assets/foundation_models/geneformer',scgpt_brain_root=self.root/'assets/foundation_models/scgpt-brain',
            scgpt_human_root=self.root/'assets/foundation_models/scgpt-human',scgpt_reference_root=self.root/'reference_methods/scBrain').items():options.setdefault(key,str(value))
        return options

    def baseline_phase(self,data=None,steps=None):
        data=Path(data or self.data);fingerprint=file_sha(data/'manifest.json');jobs=[]
        self.progress(queue_phase='prepare_integration_views',dataset=str(data))
        self.prepare_inputs(data)
        options=self.native_options()
        if steps is not None:
            import numpy as np
            pairs=int(np.load(data/'train/development_pair_ptr.npy',mmap_mode='r')[-1])
            for method in ('scfit','scnode','trajectorynet'):
                cells=options.get(method+'_cells',24)
                options[method+'_steps']=math.ceil(self.cfg['scaling']['native_development_epochs']*pairs/cells)
            # Original epoch-based fine-tuning must not remain capped at the
            # 100k search budget when the training corpus grows.
            train_rows=json.loads((data/'manifest.json').read_text())['splits']['train']['rows']
            options['native_max_steps']=max(options.get('native_max_steps',10000),math.ceil(10*train_rows/16))
            options['scuniflow_pretrain_steps']=math.ceil(options.get('scuniflow_pretrain_epochs',80)*train_rows/options.get('scuniflow_batch_size',25))
            niche_centers=int(np.load(data/'train/mask_niche.npy',mmap_mode='r').any(1).sum())
            options['niche_readout_steps']=math.ceil(self.cfg['scaling']['epochs']*niche_centers/256)
        train_rows=json.loads((data/'manifest.json').read_text())['splits']['train']['rows']
        for method in NATIVE:
            for seed in self.cfg['seeds']:jobs.append(dict(action='native',method=method,seed=seed,data=str(data),
                data_manifest_sha256=fingerprint,options=options,threads=2,
                estimated_host_mib=4096+math.ceil(train_rows*(7168 if method in {'geneformer_v1_10m','scgpt_brain','scgpt_human'} else 2048)/1048576)))
        from .transfer import missing_tasks,POLICY
        policy=dict(hidden=128,lr=.001,weight_decay=.0001,batch_size=256,steps=self.cfg['train']['steps'],eval_interval=500,
            development_steps=self.cfg['train']['steps'],development_cells=32,rollout_steps=10) | self.cfg.get('transfer',{})
        if steps is not None:
            policy['steps']=steps;policy['development_steps']=math.ceil(self.cfg['scaling']['epochs']*pairs/policy['development_cells'])
        native_jobs=list(jobs)
        for job in native_jobs:
            tasks=missing_tasks(job['method'])
            if tasks:jobs.append(dict(action='transfer',method=job['method'],seed=job['seed'],data=str(data),data_manifest_sha256=fingerprint,
                options=options,tasks=tasks,threads=2,baseline_job_id=digest(job),transfer_policy=policy,policy=POLICY,
                estimated_host_mib=8192+math.ceil(train_rows*6144/1048576)))
        self.progress(queue_phase='baseline_jobs')
        raw=self.jobs(jobs);save(self.output/('baseline-jobs-'+fingerprint+'.json'),raw)
        from .benchmark import export_benchmark,merge_baselines
        results=merge_baselines(raw);save(self.output/('baseline-'+fingerprint+'.json'),results)
        export_benchmark(self.output,results,self.cfg['seeds'])
        self.coverage(results);return results

    def coverage(self,results):
        from brainwm_benchmark.protocol import BENCHMARK_METRICS,PRIMARY_METRICS
        for run in results:
            if run.get('status')!='complete':continue
            for record in run.get('test',[run]):
                task=record['task'];metrics=record['metrics']
                for t,key,_ in (*BENCHMARK_METRICS,*PRIMARY_METRICS):
                    if t==task and (metrics.get(t+'.'+key) is None or not math.isfinite(metrics[t+'.'+key])):
                        raise ValueError('baseline has undefined/missing metric '+t+'.'+key)
        failures=[r for r in results if r.get('status')!='complete']
        if failures:raise RuntimeError(f'baseline coverage blocked by {len(failures)} failed/missing jobs; no evolution started')
        from nanobrain.baselines import build
        for method in NATIVE:
            runs=[r for r in results if r['method']==method]
            if len(runs)!=len(self.cfg['seeds']) or {r['seed'] for r in runs}!=set(self.cfg['seeds']):raise ValueError('incomplete native baseline coverage')
            from .benchmark import coverage_catalog
            expected=set(coverage_catalog()['methods'][method]['tasks'])
            for run in runs:
                if {r['task'] for r in run.get('test',[])}!=expected or {r['task'] for r in run.get('validation',[])}!=expected:
                    raise ValueError('missing applicable native baseline task')
        from .benchmark import coverage_catalog
        save(self.output/'baseline-coverage.json',dict(status='complete',results=len(results),excluded=NOT_APPLICABLE,catalog=coverage_catalog()))

    def assert_source(self):
        if getattr(self,'contract_source',None) is not None and source_fingerprint(self.root)!=self.contract_source:
            raise ValueError('trusted sources changed during the run; mixed-source results cannot be promoted')

    def execute(self,client):
        # Config, data, baseline recipes and trusted evaluator are frozen before the first job.
        from .audit import audit
        self.progress(status='preflight')
        available=audit(self.root,self.cfg)
        if any(r['status']!='ready' for r in available['native']):raise RuntimeError('baseline dependencies/assets not ready; see preparation audit')
        from nanobrain.data import verify_subset_manifest
        verify_subset_manifest(self.data,verify_hashes=True)
        from .benchmark import coverage_catalog
        contract=dict(config=self.cfg,data_manifest_sha256=file_sha(self.data/'manifest.json'),source_sha256=source_fingerprint(self.root),runtime_versions=runtime_versions(),baseline_catalog=coverage_catalog())
        path=self.output/'run-contract.json'
        if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('resume requires identical config/data contract')
        save(path,contract);self.contract_source=contract['source_sha256']
        self.progress(status='baseline',completed_generations=0)
        self.baselines=self.baseline_phase();self.assert_source();save(self.output/'baseline-summary.json',summary(self.baselines))
        root_bundle=dict(schema_version=VERSION,parent_ids=['0'*64],target_tasks=list(TASKS),hypothesis='V3 reference architecture, retrained at the common research budget.',
            genome=self.cfg['reference_genome'],optimizer=default_optimizer(),plugins=[],files={})
        identifier=self.store.stage(root_bundle,origin=dict(generation=0,proposal=0))['candidate_id']
        if self.store.node(identifier).get('status')!='complete':self.evaluate_candidates([identifier],0)
        if self.store.node(identifier).get('status')!='complete':raise RuntimeError('reference evaluation failed; cannot initialize evolution')
        for generation in range(1,self.cfg['generations']+1):
            self.assert_source()
            generation_path=self.output/'generations'/f'{generation:03d}.json'
            if generation_path.exists():continue
            self.progress(status='evolution_proposals',current_generation=generation,completed_generations=generation-1)
            nodes=self.store.nodes();parents=select_parents(nodes,self.store);seen={n.get('semantic_id') for n in nodes if n.get('semantic_id')}
            proposals=[]
            for index in range(self.cfg['candidates_per_generation']):
                proposal_path=self.output/'proposals'/f'{generation:03d}-{index}.json'
                if proposal_path.exists():
                    cached=json.loads(proposal_path.read_text());candidate=cached['candidate_id'];bundle=self.store.bundle(candidate)
                    validation=json.loads((self.store.candidates/candidate/'validation.json').read_text())
                    if validation.get('status')!='accepted':raise ValueError('cached proposal lost its accepted validation')
                    proposals.append(candidate);seen.add(semantic_id(bundle));continue
                available_parents=[n['candidate_id'] for n in self.store.nodes()
                                   if n.get('status')=='complete' and n.get('all_seeds_passed')]
                for attempt in range(self.cfg.get('proposal_attempts',4)):
                    session=ToolSession(self.store,self.baselines,self.validate,available_parents,seen,
                                        origin=dict(generation=generation,proposal=index,attempt=attempt+1))
                    try:
                        proposal=propose(client,session,dict(generation=generation,proposal=index,
                            parent_ids=parents,tree=self.store.tree_summary(),baseline_summary=summary(self.baselines)),reasoning=self.cfg['reasoning_effort'])
                        save(proposal_path,proposal);proposals.append(proposal['candidate_id'])
                        seen.add(semantic_id(self.store.bundle(proposal['candidate_id'])));break
                    except Exception as exc:
                        # No malformed/unvalidated candidate enters training.
                        key=getattr(getattr(client,'config',None),'api_key','')
                        reason=str(exc).replace(key,'[redacted]') if key else str(exc)
                        save(self.output/'proposal-failures'/f'{generation:03d}-{index}-{attempt+1}.json',dict(status='rejected',reason=reason[:2000]))
                        if attempt+1==self.cfg.get('proposal_attempts',4):
                            self.progress(last_proposal_rejection=dict(generation=generation,proposal=index,reason=reason[:500]))
            self.progress(status='evolution_evaluation')
            evaluated=self.evaluate_candidates(proposals,generation)
            save(generation_path,dict(status='complete',candidate_ids=proposals,valid_candidates=len(evaluated),rejected_candidates=len(proposals)-len(evaluated)))
            self.progress(completed_generations=generation)
        self.assert_source()
        champion=max((n for n in self.store.nodes() if n.get('status')=='complete'),key=lambda n:n['search_score'])
        frozen=dict(candidate_id=champion['candidate_id'],bundle=self.store.bundle(champion['candidate_id']),
            baseline_recipes=dict(native=NATIVE,controls=CONTROLS,options=self.native_options()),
            metric_protocol='NANOBRAIN-METRICS-V3-20261102',search_used_test=True,config_sha256=digest(self.cfg),source_sha256=source_fingerprint(self.root),runtime_versions=runtime_versions())
        save(self.output/'frozen-champion.json',frozen)
        self.progress(status='scaling',champion=champion['candidate_id'])
        from .scaling import run_scaling_action
        result=run_scaling_action(self,frozen);self.progress(status='complete');return result


def source_fingerprint(root):
    root=Path(root);files={}
    for folder in ('src','scripts','reference_methods'):
        for p in sorted((root/folder).rglob('*.py')):
            if '__pycache__' not in p.parts:files[str(p.relative_to(root))]=file_sha(p)
    return digest(files)


def runtime_versions():
    import importlib.metadata
    result={}
    for package in ('torch','numpy','scipy','scikit-learn','scib-metrics','xgboost','xgboost-cpu','jsonschema','geomloss','POT','matplotlib',
                    'transformers','scvi-tools','celltypist','harmonypy','scanorama','bbknn','GraphST','SpaGCN','torch-geometric','torchdiffeq','einops'):
        try:result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:result[package]=None
    return result
