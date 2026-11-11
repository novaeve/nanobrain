"""CPU synthetic tensor/gradient/reload probes, never scientific scores."""
from __future__ import annotations
import copy
from pathlib import Path
import torch
from .contracts import validate_bundle, semantic_id, digest
from .plugins import install_plugins, check_custom_parameters, checked_loss
from .optimizers import validate_optimizer, build_optimizer


def synthetic_splits(input_dim=256):
    import numpy as np
    from nanobrain.data import make_synthetic_splits,DatasetSplits
    from nanobrain.development import DevelopmentPairDataset
    from nanobrain.niche_data import neighborhood_targets
    data=make_synthetic_splits(train_rows=32,validation_rows=16,test_rows=8,input_dim=input_dim,
        annotation_classes=5,niche_classes=3,integration_classes=3,development_dim=input_dim,seed=19)
    for split in (data.train,data.validation):
        xy=split.coordinates[:,:2].numpy();p,d,mask=neighborhood_targets(xy,split.labels['annotation'].numpy(),np.arange(split.size),
            xy,np.arange(split.size),[2.,4.,8.,100.],5,min_known_fraction=0)
        split.labels['niche']=torch.from_numpy(p);split.labels['niche_density']=torch.from_numpy(d)
        split.masks['niche']=torch.from_numpy(mask);split.masks['niche_density']=torch.ones_like(split.masks['niche'])
    def pairs(ds,split):
        n=ds.size//2
        return DevelopmentPairDataset(ds,np.arange(n),np.arange(n,2*n),np.array([0,n]),
            [dict(source_time_days=1.,target_time_days=2.,interval_id='probe',uberon_id='synthetic')],split=split)
    return DatasetSplits(data.train,data.validation,None,pairs(data.train,'train'),pairs(data.validation,'validation'))


def validate_in_worker(bundle,output,*,probe_steps=2,max_parameters=20_000_000):
    torch.set_num_threads(2);validate_bundle(bundle)
    install_plugins(bundle);check_custom_parameters(bundle);validate_optimizer(bundle['optimizer'])
    from nanobrain.genome import CandidateGenome
    from nanobrain.model import NanoBrainModel,NanoBrainModelConfig
    from nanobrain.data import infer_label_cardinalities,niche_model_options
    from nanobrain.task_training import configure_task,row_task_loss,task_input
    from nanobrain.objectives import COMPOSITION_OBJECTIVES
    genome=CandidateGenome.from_dict({**bundle['genome'],'seed':19})
    if genome.architecture is None:raise ValueError('explicit backbone and task-head graphs required')
    if genome.architecture['objectives']['spatial_niche']['name'] not in COMPOSITION_OBJECTIVES:raise ValueError('niche must remain composition regression')
    data=synthetic_splits(genome.architecture['input_dim']);counts=infer_label_cardinalities(data)
    config=NanoBrainModelConfig(genome.architecture['input_dim'],counts['annotation'],counts['niche'],counts['integration'],genome.architecture['input_dim'],**niche_model_options(data))
    # Dataset-level correction dependencies are validated separately by the trusted trainer.
    from nanobrain.integration_components import prepare_component_views
    prepare_component_views(data,genome)
    from .code_policy import load_tensor_code
    from .contracts import parameter_values
    for plugin in bundle['plugins']:
        if plugin['kind'] not in {'regression_loss','niche_loss','classification_loss'}:continue
        f=load_tensor_code(bundle['files'][plugin['file']],plugin['entrypoint'],'loss_probe')
        settings=parameter_values(plugin['parameters'],plugin['settings'])
        leaf=torch.randn(9,5,requires_grad=True);x=leaf
        target=torch.arange(9)%5 if plugin['kind']=='classification_loss' else torch.randn(9,5)
        if plugin['kind']=='niche_loss':x=x.softmax(-1);target=target.softmax(-1)
        loss=checked_loss(lambda a,b:f(a,b,**settings),x,target)
        alternative=f(x,(target+1)%5 if target.ndim==1 else target.roll(1,-1),**settings)
        if torch.allclose(loss,alternative):raise ValueError('custom loss ignores target perturbations')
        grad=torch.autograd.grad(loss,leaf)[0]
        if not torch.isfinite(grad).all() or not bool(grad.abs().sum()>0):raise ValueError('loss has no useful finite gradient')
    records=[]
    for task in ('annotation','spatial_niche','integration','development'):
        torch.manual_seed(19);model=NanoBrainModel(config,genome)
        if model.parameter_count>max_parameters:raise ValueError('parameter budget exceeded')
        configure_task(model,task);params=[p for p in model.parameters() if p.requires_grad]
        optimizer=build_optimizer(params,genome.learning_rate,bundle['optimizer'])
        before=[p.detach().clone() for p in params]
        batch=data.train.batch(list(range(8)))
        for step in range(probe_steps):
            model.train();optimizer.zero_grad()
            if task=='development':loss=model.population_flow_losses(data.development_train.batch(0,max_cells=8))[0]
            else:loss=row_task_loss(model,batch,task)[0].total
            if not torch.isfinite(loss) or not loss.requires_grad:raise ValueError('nonfinite/detached task loss')
            loss.backward()
            grads=[p.grad for p in params if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise ValueError('invalid gradients')
            from nanobrain.architecture import ModuleGraph
            for graph in model.modules():
                if not isinstance(graph,ModuleGraph):continue
                for node in graph.plan:
                    if not node['name'].startswith('custom_'):continue
                    parameters=[p for p in graph.layers[node['id']].parameters() if p.requires_grad]
                    if parameters and not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in parameters):
                        raise ValueError('connected custom module has no effective gradient')
            torch.nn.utils.clip_grad_norm_(params,bundle['optimizer']['grad_clip']);optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in params):raise ValueError('nonfinite weights')
        if not any(not torch.equal(a,b) for a,b in zip(before,params)):raise ValueError('optimizer made no parameter update')
        model.eval()
        with torch.no_grad():
            if task=='development':
                state=torch.randn(7,genome.hidden_dim);context=torch.randn_like(state);time=torch.rand(7,3)
                velocity=model.flow(state,context,time)
                for value in (model.flow(state+1,context,time),model.flow(state,context+1,time),model.flow(state,context,time+1)):
                    if torch.allclose(velocity,value):raise ValueError('flow must numerically depend on state, context and time')
            a=model.encode(task_input(batch,task));b=model.encode(task_input(batch,task))
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            if not torch.isfinite(a).all():raise ValueError('nonfinite inference')
        path=Path(output)/(task+'.pt');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        restored=NanoBrainModel(config,genome);restored.load_state_dict(saved['model']);restored.eval()
        configure_task(restored,task)
        build_optimizer([p for p in restored.parameters() if p.requires_grad],genome.learning_rate,bundle['optimizer']).load_state_dict(saved['optimizer'])
        with torch.no_grad():torch.testing.assert_close(a,restored.encode(task_input(batch,task)))
        records.append(dict(task=task,updated=True,finite=True,reload=True,parameters=model.parameter_count))
    return dict(status='accepted',candidate_id=digest(bundle),semantic_id=semantic_id(bundle),checks=records,
        synthetic_only=True,benchmark_training_started=False)
