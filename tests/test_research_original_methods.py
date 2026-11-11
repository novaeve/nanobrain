"""Tiny synthetic original-model training and frozen test reload checks."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest,torch
from nanobrain.research.validation import synthetic_splits
from nanobrain.research import benchmark
from nanobrain.baselines import build
from nanobrain.baselines.base import AdapterContext

ROOT=Path(__file__).resolve().parents[1]


def context(tmp_path,dimension=16):
    data=synthetic_splits(dimension);assets=tmp_path/'assets';assets.mkdir()
    np.save(assets/'normalization_mean.npy',np.ones(dimension,np.float32))
    np.save(assets/'normalization_std.npy',np.ones(dimension,np.float32)*.1)
    np.save(assets/'panel_gene_index.npy',np.load(ROOT/'assets/nanobrain_v3_100k_niche_v3/panel_gene_index.npy')[:dimension])
    options=dict(native_max_steps=2,native_patience=1,scbrain_epochs=1,graphst_steps=1,spagcn_steps=1,stagate_steps=1,
        spatial_pretrain_batch=16,niche_readout_steps=2,scuniflow_width=64,scuniflow_layers=1,scuniflow_pretrain_steps=1,scuniflow_batch_size=2,
        foundation_probe_epochs=3,geneformer_finetune_epochs=1,geneformer_finetune_batch_size=2,
        scgpt_finetune_epochs=1,scgpt_finetune_batch_size=2,gene_catalog=str(ROOT/'assets/human_v3_genes.parquet'),
        geneformer_root=str(ROOT/'assets/foundation_models/geneformer'),scgpt_brain_root=str(ROOT/'assets/foundation_models/scgpt-brain'),
        scgpt_human_root=str(ROOT/'assets/foundation_models/scgpt-human'),scgpt_reference_root=str(ROOT/'reference_methods/scBrain'))
    output=tmp_path/'fit';output.mkdir()
    return data,AdapterContext(data.train,data.validation,assets,output,7,'cpu',2,data.development_train,data.development_validation,options)


@pytest.mark.parametrize('method',['scbrain_original','graphst_composition','spagcn_composition','stagate_composition','scuniflow_pretrain_readout'])
def test_original_networks_train_then_reload_without_optimizing(tmp_path,monkeypatch,method):
    torch.set_num_threads(2);data,ctx=context(tmp_path);adapter=build(method)
    assert adapter.availability(ctx).runnable
    predictions=adapter.run(ctx)
    assert {r.task for r in predictions}==set(adapter.tasks)
    checkpoint=ctx.output_root/adapter.frozen_artifact;assert checkpoint.is_file()
    assert torch.load(checkpoint,map_location='cpu',weights_only=False)['test_opened'] is False
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('test refit')))
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('test refit')))
    options=dict(ctx.options,**{method+'_checkpoint':str(checkpoint)})
    if (ctx.output_root/'foundation-probe.joblib').exists():options['foundation_probe_checkpoint']=str(ctx.output_root/'foundation-probe.joblib')
    result=adapter.run(replace(ctx,options=options))
    for first,second in zip(predictions,result):
        for key in first.arrays:
            if np.issubdtype(first.arrays[key].dtype,np.number):np.testing.assert_allclose(first.arrays[key],second.arrays[key],rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize('method',['geneformer_v1_10m_finetuned','scgpt_brain_finetuned','scgpt_human_finetuned'])
def test_real_pretrained_backbones_finetune_and_freeze(tmp_path,monkeypatch,method):
    torch.set_num_threads(2);data,ctx=context(tmp_path,256);adapter=build(method)
    assert adapter.availability(ctx).runnable
    result=adapter.run(ctx);assert len(result)==1 and result[0].task=='annotation'
    checkpoint=ctx.output_root/adapter.frozen_artifact;payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['test_opened'] is False and payload['steps']>0
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**k:(_ for _ in ()).throw(AssertionError('test refit')))
    repeated=adapter.run(replace(ctx,options={**ctx.options,method+'_checkpoint':str(checkpoint)}))
    np.testing.assert_allclose(result[0].arrays['y_prob'],repeated[0].arrays['y_prob'],rtol=1e-5,atol=1e-5)
