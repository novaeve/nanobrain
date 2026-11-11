"""Acceptance coverage for executable research search. All data are synthetic."""
import copy,json
from pathlib import Path
import pytest,torch
from nanobrain.research.contracts import validate_bundle,semantic_id,digest,candidate_schema
from nanobrain.research.benchmark import control_bundle,CONTROLS
from nanobrain.research.code_policy import inspect_source
from nanobrain.research.store import Store
from nanobrain.research.agent import ToolSession,propose,loads
from nanobrain.research.scaling import nested_order,write_curves

ROOT=Path(__file__).resolve().parents[1]


def custom_bundle():
    b=control_bundle('control_mlp');b['hypothesis']='Test a gated residual module, robust loss and normalized-gradient optimizer.'
    b['plugins']=[
      dict(name='custom_gate',kind='operation',file='modules/gated.py',entrypoint='build',parameters={'expansion':dict(type='integer',minimum=1,maximum=4,default=2)},settings={}),
      dict(name='custom_robust',kind='regression_loss',file='losses/robust.py',entrypoint='compute',parameters={'delta':dict(type='number',minimum=.01,maximum=4,default=1.)},settings={}),
      dict(name='custom_normalized',kind='optimizer',file='optimizers/normalized.py',entrypoint='build',parameters={'epsilon':dict(type='number',minimum=1e-6,maximum=.1,default=.0001)},settings={})]
    b['files']={n:(ROOT/'examples/research'/f).read_text() for n,f in [('modules/gated.py','gated_module.py'),('losses/robust.py','robust_loss.py'),('optimizers/normalized.py','custom_optimizer.py')]}
    graph=b['genome']['architecture']['backbone'];previous=graph['output'];graph['nodes'].append(dict(id='gate',kind='operation',name='custom_gate',inputs=[previous],params={'expansion':2}));graph['output']='gate'
    b['genome']['architecture']['objectives']['development']['params']['velocity_regression']='custom_robust'
    b['optimizer']['name']='custom_normalized';b['optimizer']['params']={'epsilon':.0001}
    return b


def test_strict_contract_and_duplicate_detection():
    b=control_bundle('control_mlp');validate_bundle(b);other=copy.deepcopy(b)
    other['hypothesis']='Different prose and ancestry';other['parent_ids']=['f'*64]
    assert semantic_id(b)==semantic_id(other)
    for key in ('score','metrics','seed','data','evaluation'):
        changed=copy.deepcopy(b);changed[key]=0
        with pytest.raises(Exception):validate_bundle(changed)
    with pytest.raises(ValueError):loads('{"x":1,"x":2}')
    with pytest.raises(ValueError):loads('{"x":NaN}')
    b=custom_bundle();validate_bundle(b)
    b['plugins'][0]['settings']['expansion']=3
    with pytest.raises(ValueError):validate_bundle(b)


@pytest.mark.parametrize('source',[
 'import os\ndef build(hidden_dim):\n return os.system("true")',
 'import torch\ndef build(hidden_dim):\n return torch.load("/tmp/x")',
 'import torch\ndef build(hidden_dim):\n return open("/tmp/x")',
 'import torch\ndef build(hidden_dim):\n return torch.__dict__',
 'import torch\nx=torch.zeros(10)\ndef build(hidden_dim):\n return x',
])
def test_io_introspection_and_import_time_work_rejected(source):
    with pytest.raises(ValueError):inspect_source(source,'build')


def test_store_rejects_mutation_and_path_escape(tmp_path):
    store=Store(tmp_path);b=control_bundle('control_mlp');identifier=store.stage(b)['candidate_id'];assert store.bundle(identifier)==b
    with pytest.raises(ValueError):store.read_text('../secret')
    (store.candidates/identifier/'bundle.json').write_text('{}')
    with pytest.raises(ValueError):store.bundle(identifier)


def test_react_mock_repair_inspect_submit_without_network(tmp_path):
    store=Store(tmp_path);parent=store.stage(control_bundle('control_mlp'))['candidate_id'];b=control_bundle('control_mlp');b['parent_ids']=[parent];b['genome']['learning_rate']=.0005
    identifier=digest(b);calls=[('inspect_contract',{}),('list_components',dict(kind='encoder',offset=0)),
      ('submit_candidate',dict(candidate_id=identifier)),('write_candidate',dict(bundle=b)),('validate_candidate',dict(candidate_id=identifier)),('submit_candidate',dict(candidate_id=identifier))]
    class Client:
        def chat(self,messages,**kwargs):
            name,args=calls.pop(0)
            return dict(choices=[dict(message=dict(content=None,tool_calls=[dict(id=str(len(messages)),type='function',function=dict(name=name,arguments=json.dumps(args)))]))],usage={'total_tokens':10})
    session=ToolSession(store,[],lambda i,b:dict(status='accepted',candidate_id=i,semantic_id=semantic_id(b)),[parent],[])
    result=propose(Client(),session,dict(tree=store.tree_summary()))
    assert result['candidate_id']==identifier and result['usage']['total_tokens']==60
    assert result['trace'][2]['response']['status']=='rejected'


def test_custom_factories_loss_optimizer_train_reload(tmp_path):
    from nanobrain.research.validation import validate_in_worker
    result=validate_in_worker(custom_bundle(),tmp_path)
    assert result['status']=='accepted' and len(result['checks'])==4 and result['synthetic_only']


def test_nested_training_order_and_standard_metric_curves(tmp_path):
    import numpy as np
    order=nested_order(100,np.array([0,10,20]),19)
    assert set(order[:3])=={0,10,20} and len(set(order))==100 and set(order[:10])<=set(order[:20])
    rows=[dict(train_rows=n,method='toy',seed=s,task='annotation',metric='annotation.f1_macro',value=.5+n/1000+s/10000) for n in (10,20) for s in (1,2,3)]
    rows+=[dict(train_rows=10,method='toy',seed=1,task='integration',metric='integration.scib_composite',value=.9)]
    write_curves(tmp_path,rows)
    assert 'scib_composite' not in (tmp_path/'learning-curves.csv').read_text()
    assert (tmp_path/'annotation.f1_macro.pdf').exists()


def test_workflow_mock_freezes_then_invokes_final_action(tmp_path,monkeypatch):
    from nanobrain.research.workflow import Workflow,save
    import nanobrain.research.audit as audit_module
    import nanobrain.research.scaling as scale_module
    import nanobrain.data as data_module
    root=tmp_path/'project';root.mkdir();data=root/'assets/toy';data.mkdir(parents=True)
    (data/'manifest.json').write_text('{}')
    cfg=dict(data='assets/toy',gpu_ids=[0,1,2,3,4],per_gpu_workers=1,seeds=[1,2,3],generations=1,candidates_per_generation=1,
        reference_genome=control_bundle('control_mlp')['genome'],train=dict(steps=2),native_options={},reasoning_effort='max')
    workflow=Workflow(root,cfg,tmp_path/'run')
    monkeypatch.setattr(audit_module,'audit',lambda r,c:dict(native=[dict(status='ready')]))
    monkeypatch.setattr(data_module,'verify_subset_manifest',lambda *a,**k:{})
    monkeypatch.setattr(workflow,'baseline_phase',lambda *a,**k:[])
    monkeypatch.setattr(workflow,'validate',lambda i,b:dict(status='accepted',candidate_id=i,semantic_id=semantic_id(b)))
    from brainwm_benchmark.protocol import BENCHMARK_METRICS,PRIMARY_METRICS
    def jobs(items):
        return [dict(status='complete',method=j['method'],seed=j['seed'],task=j['task'],metrics={t+'.'+m:(.7 if direction=='max' else .2) for t,m,direction in (*BENCHMARK_METRICS,*PRIMARY_METRICS) if t==j['task']}) for j in items]
    monkeypatch.setattr(workflow,'jobs',jobs)
    invoked=[]
    monkeypatch.setattr(scale_module,'run_scaling_action',lambda w,f:invoked.append(f) or dict(status='mock_scaling_complete'))
    class Client:
        turn=0
        def chat(self,messages,**kwargs):
            context=json.loads(messages[1]['content'])
            b=copy.deepcopy(cfg['reference_genome']);b['learning_rate']=.0007
            candidate=control_bundle('control_mlp');candidate['genome']=b;candidate['parent_ids']=context['parent_ids'][:1]
            identifier=digest(candidate)
            call=[('write_candidate',dict(bundle=candidate)),('validate_candidate',dict(candidate_id=identifier)),('submit_candidate',dict(candidate_id=identifier))][self.turn]
            self.turn+=1
            return dict(choices=[dict(message=dict(content=None,tool_calls=[dict(id=str(self.turn),type='function',function=dict(name=call[0],arguments=json.dumps(call[1])))]))])
    assert workflow.execute(Client())['status']=='mock_scaling_complete'
    assert len(invoked)==1 and invoked[0]['search_used_test']
    assert (workflow.output/'frozen-champion.json').exists()


def test_shared_input_views_are_readonly_in_candidate_workers(tmp_path,monkeypatch):
    from nanobrain.data import make_synthetic_splits
    from nanobrain.genome import CandidateGenome
    import nanobrain.integration_components as components
    data=make_synthetic_splits(train_rows=12,validation_rows=8,test_rows=4,input_dim=8)
    genome=CandidateGenome.from_dict({**control_bundle('component_harmony',8)['genome'],'seed':7})
    calls=[]
    monkeypatch.setenv('NANOBRAIN_COMPONENT_CACHE',str(tmp_path))
    monkeypatch.setenv('NANOBRAIN_COMPONENT_CACHE_READONLY','0')
    monkeypatch.setattr(components,'compute_component',lambda name,x,b:calls.append(name) or (x+1).astype('float32'))
    components.prepare_component_views(data,genome);assert len(calls)==2
    for p in tmp_path.glob('*.lock'):p.unlink()
    monkeypatch.setenv('NANOBRAIN_COMPONENT_CACHE_READONLY','1')
    components.prepare_component_views(data,genome)
    assert len(calls)==2 and not list(tmp_path.glob('*.lock'))


def test_worker_opens_test_only_after_loading_validation_best(tmp_path,monkeypatch):
    from nanobrain.research.worker import candidate_job
    from nanobrain.research.validation import synthetic_splits
    import nanobrain.data as data_module
    import nanobrain.training as training_module
    import nanobrain.task_training as task_module
    from nanobrain.data import DatasetSplits
    from nanobrain.model import NanoBrainModel
    data=synthetic_splits();events=[]
    def load(root,include_test):
        events.append('test_opened' if include_test else 'train_validation_opened')
        if include_test:
            assert events[-2]=='best_checkpoint_saved'
            return DatasetSplits(data.train,data.validation,data.validation,data.development_train,data.development_validation,data.development_validation)
        return data
    monkeypatch.setattr(data_module,'load_v3_subset',load)
    def train(data,config,genome,opts,**kwargs):
        model=NanoBrainModel(config,genome)
        with torch.no_grad():
            for p in model.parameters():p.fill_(.123)
        checkpoint=tmp_path/'best.pt';torch.save(dict(model=model.state_dict()),checkpoint);events.append('best_checkpoint_saved')
        return dict(stability={'passed':True},optimizer_changed_parameter=True,best_checkpoint=str(checkpoint),validation={'annotation.f1_macro':.3},
            row_samples_seen=16,population_source_cells_seen=0)
    monkeypatch.setattr(training_module,'train_candidate',train)
    def evaluate(model,*args,**kwargs):
        assert torch.allclose(next(model.parameters()),torch.full_like(next(model.parameters()),.123))
        events.append('test_scored');return {'annotation.f1_macro':.4}
    monkeypatch.setattr(task_module,'evaluate_task',evaluate)
    result=candidate_job(dict(bundle=control_bundle('control_mlp'),seed=7,task='annotation',method='toy',train=dict(device='cpu',steps=2,batch_size=8)))
    assert events==['train_validation_opened','best_checkpoint_saved','test_opened','test_scored']
    assert result['checkpoint_selection']=='validation'
