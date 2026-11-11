"""Regression checks for the requested unrestricted plugins and complete tree."""
import copy,json
from pathlib import Path
import numpy as np
import pytest,torch
from nanobrain.research.benchmark import control_bundle,coverage_catalog,summary,NATIVE
from nanobrain.research.contracts import validate_bundle,candidate_schema,digest
from nanobrain.research.store import Store
from nanobrain.research.agent import ToolSession
from nanobrain.research.scaling import scale_points


def many_plugins(count=12):
    bundle=control_bundle('control_mlp');graph=bundle['genome']['architecture']['backbone']
    for i in range(count):
        name=f'custom_layer_{i}';file=f'modules/layer_{i}.py'
        bundle['plugins'].append(dict(name=name,kind='operation',file=file,entrypoint='build',parameters={},settings={}))
        bundle['files'][file]='import torch\ndef build(hidden_dim):\n    return torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU())\n'
        graph['nodes'].append(dict(id=f'layer_{i}',kind='operation',name=name,inputs=[graph['output']],params={}))
        graph['output']=f'layer_{i}'
    return bundle


def test_more_than_eight_plugins_and_files_validate_and_update(tmp_path):
    from nanobrain.research.validation import validate_in_worker
    bundle=many_plugins();validate_bundle(bundle)
    assert 'maxItems' not in candidate_schema()['properties']['plugins']
    assert 'maxProperties' not in candidate_schema()['properties']['files']
    assert validate_in_worker(bundle,tmp_path)['status']=='accepted'


def test_tree_preserves_all_branches_low_scores_and_rejections(tmp_path):
    store=Store(tmp_path);root=store.stage(control_bundle('control_mlp'),origin={'generation':0})['candidate_id']
    all_ids={root}
    for i in range(30):
        b=control_bundle('control_mlp');b['genome']['learning_rate']=.000101+i*.000001;b['parent_ids']=[root]
        identifier=store.stage(b,origin={'generation':i//6+1})['candidate_id'];all_ids.add(identifier)
        store.put_result(identifier,dict(candidate_id=identifier,status='rejected_runtime' if i==29 else 'complete',generation=i//6+1,
            selection_metrics={'annotation.f1_macro':i/100},search_score=i/100))
    tree=store.tree_summary(limit=1)
    assert tree['complete'] and not tree['truncated'] and len(tree['nodes'])==31
    assert len(tree['edges'])==30
    assert {store.resolve_id(n['id']) for n in tree['nodes']}==all_ids
    assert any(n['s']=='rejected_runtime' for n in tree['nodes'])
    reference=tree['nodes'][0]['id'];session=ToolSession(store,[],None,[],[])
    detail=session.call('lineage',{'candidate_id':reference});assert detail['candidate_id']==root
    assert session.call('read_artifact',dict(candidate_id=reference,kind='bundle',offset=0))['candidate_id']==root
    malformed=control_bundle('control_mlp');malformed['parent_ids']=[root]
    malformed['genome']['architecture']['backbone']['nodes'][0]['repeat']='invalid'
    store.stage(malformed,origin={'generation':6})
    assert len(store.tree_summary()['nodes'])==32


def test_requested_scales_retain_original_100k_and_fixed_holdout():
    points=scale_points({'dataset_sizes':[100000,1000000,5000000,10000000,20000000,'full_train']},25702876,40000)
    assert [p['train_rows'] for p in points]==[60000,960000,4960000,9960000,19960000,25702876]
    assert points[-1]['label']=='full' and points[-1]['dataset_rows']==25742876
    with pytest.raises(ValueError):scale_points({'dataset_sizes':[100000,99999]},25702876,40000)


def test_every_source_and_method_is_in_coverage_catalog():
    catalog=coverage_catalog();assert set(NATIVE)<=set(catalog['methods'])
    assert {'scBrain','GraphST','SpaGCN','STAGATE','scUniFlow'}<=set(catalog['sources'])
    assert all(set(v)<=set(catalog['methods']) for v in catalog['sources'].values())
    assert 'geneformer_v1_10m_finetuned' in NATIVE and 'scgpt_human_finetuned' in NATIVE


def test_initial_baseline_summary_omits_hundreds_of_diagnostic_columns():
    records=[dict(method='toy',status='complete',metrics={'annotation.f1_macro':.5,'spatial_niche.composition_mae_scale_0_type_62':.2})]
    result=summary(records)
    assert set(result[0]['metrics'])=={'annotation.f1_macro'}


def test_mamba_ssd_matches_independent_selective_state_recurrence():
    from nanobrain.research.mamba2_compat import ssd_function
    torch.manual_seed(5);x=torch.randn(2,8,3,4);a=-torch.rand(2,8,3);b=torch.randn(2,8,3,5);c=torch.randn_like(b)
    output,_=ssd_function()(x,a,b,c,4);state=torch.zeros(2,3,4,5);expected=[]
    for t in range(8):
        state=state*a[:,t].exp()[:,:,None,None]+torch.einsum('bhp,bhn->bhpn',x[:,t],b[:,t])
        expected.append(torch.einsum('bhpn,bhn->bhp',state,c[:,t]))
    assert torch.allclose(output,torch.stack(expected,dim=1),rtol=2e-5,atol=2e-5)


def test_resource_pool_enforces_40_slots_and_cpu_admission_when_gpu_busy(monkeypatch):
    import time
    import nanobrain.research.resources as resources
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'MemAvailable: 1000000000 kB\n' if str(p)=='/proc/meminfo' else original(p,*a,**k))
    pool=resources.ResourcePool([0,1,2,3,4],8,{})
    def refresh():
        pool.snapshot={i:dict(free_mib=100000,own_mib=0) for i in range(5)};pool.snapshot_at=time.monotonic()
    monkeypatch.setattr(pool,'refresh',refresh)
    job=dict(action='candidate',method='toy',estimated_host_mib=1024)
    choices=[pool.acquire(str(i),job) for i in range(40)]
    from collections import Counter
    assert Counter(choices)=={i:8 for i in range(5)}
    pool.snapshot={i:dict(free_mib=0,own_mib=0) for i in range(5)};pool.snapshot_at=time.monotonic()
    assert pool.acquire('cpu-job',dict(action='native',method='xgboost',estimated_host_mib=1024),needs_gpu=False) is None
    assert pool.leases['cpu-job']['gpu'] is None
    with pytest.raises(ValueError):resources.ResourcePool([5],8,{})
