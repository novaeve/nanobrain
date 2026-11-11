"""Baseline test inference must reuse a train-only fit. Synthetic data only."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from nanobrain.data import make_synthetic_splits
from nanobrain.baselines.base import AdapterContext


def context(tmp_path):
    data=make_synthetic_splits(train_rows=64,validation_rows=32,test_rows=16,input_dim=8,annotation_classes=3)
    return data,AdapterContext(data.train,data.validation,tmp_path/'assets/data',tmp_path/'fit',7,'cpu',2,options={'xgboost_estimators':3,'foundation_probe_epochs':3})


def test_xgboost_reloads_train_fit_without_fitting_on_test(tmp_path,monkeypatch):
    pytest.importorskip('xgboost')
    from nanobrain.baselines.annotation import XGBoostAdapter
    from xgboost import XGBClassifier
    data,ctx=context(tmp_path);adapter=XGBoostAdapter();prediction=adapter.run(ctx)[0]
    path=ctx.output_root/'xgboost.joblib';assert path.exists()
    monkeypatch.setattr(XGBClassifier,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('refit after freeze')))
    result=adapter.run(replace(ctx,validation=data.test,options={'xgboost_checkpoint':str(path)}))[0]
    assert result.task=='annotation' and len(result.arrays['y_pred'])==int(data.test.masks['annotation'].sum())


def test_foundation_task_probe_reloads_train_fit(tmp_path,monkeypatch):
    from nanobrain.baselines.foundation import _linear_probe_predictions
    from sklearn.linear_model import SGDClassifier
    data,ctx=context(tmp_path)
    _linear_probe_predictions(ctx,data.train.features.numpy(),data.validation.features.numpy(),method='toy',notes='frozen synthetic encoder')
    path=ctx.output_root/'foundation-probe.joblib';assert path.exists()
    monkeypatch.setattr(SGDClassifier,'fit',lambda *a,**k:(_ for _ in ()).throw(AssertionError('refit after freeze')))
    result=_linear_probe_predictions(replace(ctx,validation=data.test,options={'foundation_probe_checkpoint':str(path)}),data.train.features.numpy(),data.test.features.numpy(),method='toy',notes='frozen synthetic encoder')
    assert {r.task for r in result}=={'annotation','integration'}


def test_coverage_gate_requires_all_seeds_tasks_and_metrics(tmp_path):
    from nanobrain.research.workflow import Workflow
    from nanobrain.research.benchmark import NATIVE,coverage_catalog
    from nanobrain.baselines import build
    from nanobrain.research.contracts import TASKS
    from brainwm_benchmark.protocol import BENCHMARK_METRICS,PRIMARY_METRICS
    cfg=dict(data='unused',gpu_ids=[0,1,2,3,4],per_gpu_workers=1,seeds=[1,2,3])
    workflow=Workflow(tmp_path,cfg,tmp_path/'run')
    def record(task):
        return dict(task=task,metrics={t+'.'+m:(.7 if d=='max' else .2) for t,m,d in (*BENCHMARK_METRICS,*PRIMARY_METRICS) if t==task})
    results=[]
    for method in NATIVE:
        for seed in cfg['seeds']:
            tasks=[record(t) for t in coverage_catalog()['methods'][method]['tasks']]
            results.append(dict(status='complete',method=method,seed=seed,validation=tasks,test=tasks))
    workflow.coverage(results)
    results[0]['test']=[]
    with pytest.raises(ValueError):workflow.coverage(results)
    results[0]['test']=results[0]['validation']
    with pytest.raises(ValueError):workflow.coverage(results[:-1])


def test_gene_catalog_and_foundation_paths_are_independent_of_scale_directory(tmp_path):
    import pandas as pd
    from nanobrain.baselines.foundation import _catalog,ScGPTBrainAdapter
    from nanobrain.baselines.annotation import _panel_gene_symbols
    from nanobrain.research.workflow import Workflow
    root=tmp_path/'project';assets=root/'assets';assets.mkdir(parents=True)
    catalog=assets/'human_v3_genes.parquet';pd.DataFrame(dict(gene_index=[0,1],ensembl_id=['ENSG1','ENSG2'],gene_symbol=['A','B'])).to_parquet(catalog,index=False)
    scale=root/'results/scaling/data/1000000';scale.mkdir(parents=True);np.save(scale/'panel_gene_index.npy',np.array([0,1]))
    assert _catalog(scale,catalog)==(['ENSG1','ENSG2'],['A','B'])
    assert _panel_gene_symbols(scale,catalog)==['A','B']
    cfg=dict(data='unused',gpu_ids=[0,1,2,3,4],per_gpu_workers=1,native_options={})
    options=Workflow(root,cfg,tmp_path/'run').native_options()
    model_root=Path(options['scgpt_brain_root']);model_root.mkdir(parents=True)
    for name in ('args.json','vocab.json','best_model.pt'):(model_root/name).write_text('{}')
    reference=Path(options['scgpt_reference_root'])/'src/models/scGPT/scGPT_annotation.py';reference.parent.mkdir(parents=True);reference.write_text('# synthetic availability fixture')
    data,ctx=context(tmp_path)
    assert ScGPTBrainAdapter().availability(replace(ctx,data_root=scale,options=options)).runnable
