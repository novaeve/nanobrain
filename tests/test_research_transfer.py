"""Scientific/task-boundary checks for full-method transfer, never component controls."""
from dataclasses import replace
from pathlib import Path
import copy,json
import numpy as np
import pytest,torch
from nanobrain.research.benchmark import NATIVE,coverage_catalog,merge_baselines
from nanobrain.research.transfer import missing_tasks,transfer_predictions,source_indices,normalize
from nanobrain.research.representations import Encoder
from nanobrain.research.validation import synthetic_splits
from nanobrain.baselines.base import AdapterContext


def test_fifteen_full_method_encoders_cover_four_tasks_without_component_controls():
    catalog=coverage_catalog();assert set(catalog['methods'])==set(NATIVE) and catalog['component_controls']==[]
    transferable=[m for m in NATIVE if missing_tasks(m)];assert len(transferable)==15
    assert all(set(catalog['methods'][m]['tasks'])=={'annotation','integration','spatial_niche','development'} for m in transferable)
    for m in ['harmony','scanorama','bbknn']:assert catalog['methods'][m]['tasks']==['integration'] and not missing_tasks(m)
    assert all(not m.startswith('component_') for m in catalog['methods'])


def test_original_scores_are_preserved_and_duplicate_tasks_rejected():
    native=dict(method='toy',seed=1,status='complete',job_id='native',test=[dict(task='annotation',metrics={'annotation.f1_macro':.6})],validation=[],runtime_seconds=1)
    transfer=dict(method='toy',seed=1,status='complete',job_id='transfer',test=[dict(task='development',metrics={'development.wasserstein2':3.})],validation=[],runtime_seconds=2)
    merged=merge_baselines([native,transfer])[0]
    assert merged['test'][0]==native['test'][0] and merged['source_job_ids']==['native','transfer']
    transfer['test']=native['test']
    with pytest.raises(ValueError):merge_baselines([native,transfer])


def test_train_only_normalization_and_population_sampling_match_trusted_protocol():
    train=np.array([[0.,1.],[2.,3.]],np.float32);held=np.array([[100.,200.]],np.float32)
    mean,std=normalize(train,(held,));np.testing.assert_allclose(mean,[1,2]);np.testing.assert_allclose(std,[1,1]);np.testing.assert_allclose(held,[[99,198]])
    data=synthetic_splits(8);p=data.development_validation
    src,tgt=source_indices(p,0,4,123);b=p.batch(0,max_cells=4,seed=123)
    np.testing.assert_allclose(p.dataset.features[src],b.source.features);np.testing.assert_allclose(p.dataset.features[tgt],b.target.features)


def test_all_heads_freeze_before_test_and_test_never_refits(tmp_path,monkeypatch):
    torch.set_num_threads(2);data=synthetic_splits(8)
    model=torch.nn.Linear(8,6);encoder=Encoder(model,lambda x:model(torch.tensor(x)),notes='synthetic frozen full encoder',batch_size=8)
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    ctx=AdapterContext(data.train,data.validation,tmp_path,tmp_path/'fit',17,'cpu',2,data.development_train,data.development_validation,{})
    policy=dict(hidden=16,lr=.001,weight_decay=.0001,batch_size=8,steps=3,eval_interval=2,development_steps=3,development_cells=8,rollout_steps=2)
    called=[]
    def open_test():
        for task in ['annotation','spatial_niche','development']:
            saved=torch.load(ctx.output_root/(task+'-head.pt'),map_location='cpu',weights_only=False)
            assert saved['test_opened'] is False and saved['representation_frozen']
        monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('test refit')))
        called.append(True);return replace(ctx,validation=data.validation,development_validation=data.development_validation)
    validation,test,heads=transfer_predictions(ctx,encoder,['annotation','spatial_niche','development'],policy,open_test)
    assert called==[True] and set(heads)=={'annotation','spatial_niche','development'}
    for a,b in zip(validation,test):
        for key,value in a['metrics'].items():
            other=b['metrics'][key]
            if np.isfinite(value):assert other==pytest.approx(value)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,frozen[k])
    assert all(not p.requires_grad for p in model.parameters())


def test_sandbox_mounts_native_checkpoint_readonly(tmp_path,monkeypatch):
    from nanobrain.research.sandbox import sandbox_command
    import nanobrain.research.sandbox as sandbox
    monkeypatch.setattr(sandbox.subprocess,'check_output',lambda *a,**k:json.dumps({'prefix':'/usr','paths':[]}).encode() if not k.get('text') else json.dumps({'prefix':'/usr','paths':[]}))
    command=sandbox_command('/usr/bin/python3',tmp_path,tmp_path/'work',['transfer','/work/input.json'],baseline=tmp_path/'native')
    i=command.index('/baseline');assert command[i-2]=='--ro-bind'


def test_controller_can_adopt_atomic_receipt_but_rejects_recipe_change(tmp_path):
    from nanobrain.research.revision import adopt_receipt
    from nanobrain.research.contracts import digest
    job=dict(action='native',method='toy',seed=1,data_manifest_sha256='data')
    identity=digest(job);directory=tmp_path/'jobs'/identity;directory.mkdir(parents=True)
    (directory/'input.json').write_text(json.dumps(dict(job,device='cuda:0')))
    (directory/'receipt.json').write_text(json.dumps(dict(status='complete',method='toy',seed=1,configuration_frozen_before_test=True,model_artifacts=[])))
    worker=dict(job_id=identity,gpu=2,pid=-1,process_start='none')
    result=adopt_receipt(tmp_path,tmp_path,job,worker,1)
    assert result['job_id']==identity and result['physical_gpu']==2 and result['adopted_from_previous_baseline_contract']
    (directory/'input.json').write_text(json.dumps(dict(job,device='cpu')))
    with pytest.raises(ValueError):adopt_receipt(tmp_path,tmp_path,job,worker,1)
