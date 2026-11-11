"""Official graph networks/objectives plus a center-only niche readout.

Graphs are used ONLY for label-free training-split representation pretraining.
The frozen encoder sees self loops for both readout fitting and validation/test,
so hidden neighbor features, test geometry and composition labels are never inputs.
"""
from __future__ import annotations
import importlib.util,math,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nanobrain.baselines.base import BaselineAdapter,Availability,TaskPrediction
from nanobrain.baselines.registry import register
from nanobrain.baselines.common import array,log_expression
from .original_baselines import torch_seed,ROOT


def _niche_head(context,train_z,eval_z,payload,path,notes):
    from brainwm_benchmark.metrics import niche_composition
    y=array(context.train.labels['niche'],np.float32);mask=array(context.train.masks['niche'],bool)
    vy=array(context.validation.labels['niche'],np.float32);vmask=array(context.validation.masks['niche'],bool)
    shape=y.shape[1:];device=torch.device(context.device);head=nn.Linear(train_z.shape[1],int(np.prod(shape))).to(device)
    def predict(z):
        head.eval();out=[]
        with torch.no_grad():
            for start in range(0,len(z),1024):out.append(head(torch.tensor(z[start:start+1024],device=device)).view(-1,*shape).softmax(-1).cpu().numpy())
        return np.concatenate(out)
    if 'niche_head' in payload:head.load_state_dict(payload['niche_head'])
    else:
        valid=np.flatnonzero(mask.any(1));rng=np.random.default_rng(context.seed);optimizer=torch.optim.Adam(head.parameters(),lr=.001)
        best=math.inf;budget=int(context.options.get('niche_readout_steps',10000))
        for step in range(budget):
            head.train();idx=rng.choice(valid,min(256,len(valid)),replace=len(valid)<256)
            z=torch.tensor(train_z[idx],device=device);target=torch.tensor(y[idx],device=device);m=torch.tensor(mask[idx],device=device)
            logits=head(z).view(-1,*shape);per=-(target*logits.log_softmax(-1)).sum(-1)
            # Equal weighting of the fixed radii, matching the trusted niche evaluator.
            loss=torch.stack([per[:,s][m[:,s]].mean() for s in range(shape[0]) if m[:,s].any()]).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if (step+1)%min(500,budget)==0 or step+1==budget:
                quality=niche_composition(dict(target_composition=vy,prediction_composition=predict(eval_z),composition_mask=vmask))['composition_mae']
                if quality<best:
                    best=quality;payload['niche_head']={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                    payload['readout_validation_mae']=quality;payload['test_opened']=False;torch.save(payload,path)
        head.load_state_dict(payload['niche_head'])
    pred=predict(eval_z)
    arrays=dict(target_composition=vy,prediction_composition=pred,composition_mask=vmask)
    if 'sample' in (context.validation.metadata or {}):arrays['section_id']=array(context.validation.metadata['sample'])
    return TaskPrediction('spatial_niche',arrays,int(mask.any(1).sum()),int(vmask.any(1).sum()),notes,
                          'original_unsupervised_backbone_plus_center_only_composition_readout')


def graph_input(context,rng,batch_size=512):
    data=context.train;xy=array(data.coordinates,np.float32);mask=array(data.masks['niche'],bool).any(1)
    valid=np.flatnonzero(mask & np.isfinite(xy[:,:2]).all(1))
    if not len(valid):raise ValueError('no verified spatial training centers')
    chosen=rng.choice(valid,min(batch_size,len(valid)),replace=False)
    # Recorded split-local spatial neighbors cannot connect a heldout row.
    neighbors=array(data.neighbors,np.int64);neighbor_mask=array(data.neighbor_mask,bool)
    ids=np.unique(np.concatenate([chosen,neighbors[chosen][neighbor_mask[chosen]]]))
    lookup={int(v):i for i,v in enumerate(ids)};edges={(i,i) for i in range(len(ids))}
    for original in ids:
        for other in neighbors[original][neighbor_mask[original]]:
            if int(other) in lookup:
                i,j=lookup[int(original)],lookup[int(other)];edges.add((i,j));edges.add((j,i))
    index=np.asarray(sorted(edges),np.int64).T
    return ids,index


class _GraphOriginal(BaselineAdapter):
    family='original_spatial_model_adapted';tasks=('spatial_niche',);source=''
    @property
    def frozen_artifact(self):return self.method_id+'.pt'
    def availability(self,context):
        if self.source=='stagate':return Availability('ready') if (ROOT/'reference_methods/STAGATE_pyG/STAGATE_pyG/STAGATE.py').exists() else Availability('source_missing','STAGATE official tensor source required')
        module='GraphST' if self.source=='graphst' else 'SpaGCN'
        return Availability('ready') if importlib.util.find_spec(module) else Availability('dependency_missing',module)
    def run(self,context):
        torch_seed(context.seed);device=torch.device(context.device);dim=context.train.input_dim
        train_x=log_expression(context.train,context.data_root);eval_x=log_expression(context.validation,context.data_root)
        rng=np.random.default_rng(context.seed);width=int(context.options.get('spatial_latent',32))
        if self.source=='graphst':
            from GraphST.model import Encoder_sparse
            model=Encoder_sparse(dim,width,graph_neigh=None).to(device)
        elif self.source=='spagcn':
            from SpaGCN.models import simple_GC_DEC
            model=simple_GC_DEC(dim,width).to(device);model.mu=nn.Parameter(torch.randn(10,width,device=device)*.05)
        else:
            sys.path.insert(0,str(ROOT/'reference_methods/STAGATE_pyG'));from STAGATE_pyG import STAGATE
            model=STAGATE([dim,512,width]).to(device)
        checkpoint=context.options.get(self.method_id+'_checkpoint');path=context.output_root/self.frozen_artifact
        context.output_root.mkdir(exist_ok=True,parents=True)
        if checkpoint:
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if payload.get('test_opened') is not False:raise ValueError('unsealed spatial checkpoint')
            model.load_state_dict(payload['model'])
        else:
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
            eligible=int(array(context.train.masks['niche'],bool).any(1).sum())
            epochs=context.options.get(self.source+'_epochs',600 if self.source=='graphst' else 1000)
            default_steps=5000 if self.source=='spagcn' else math.ceil(epochs*eligible/context.options.get('spatial_pretrain_batch',512))
            budget=int(context.options.get(self.source+'_steps',default_steps))
            for step in range(budget):
                ids,edges=graph_input(context,rng,int(context.options.get('spatial_pretrain_batch',512)))
                x=torch.tensor(train_x[ids],device=device);index=torch.tensor(edges,device=device);n=len(ids)
                degree=torch.bincount(index[0],minlength=n).float();weights=(degree[index[0]]*degree[index[1]]).clamp_min(1).rsqrt()
                adj=torch.sparse_coo_tensor(index,weights,(n,n)).coalesce()
                if self.source=='spagcn':
                    xy=torch.tensor(array(context.train.coordinates,np.float32)[ids,:2],device=device)
                    squared=(xy[index[0]]-xy[index[1]]).square().sum(1)
                    positive=squared[squared>0];scale=positive.median() if len(positive) else torch.tensor(1.,device=device)
                    gaussian=torch.exp(-squared/(2*scale.clamp_min(1e-8)))
                    adj=torch.sparse_coo_tensor(index,gaussian,(n,n)).coalesce()
                model.train();optimizer.zero_grad(set_to_none=True)
                if self.source=='graphst':
                    # Original GraphST reconstruction + two spatial contrastive BCE terms.
                    neigh=torch.sparse_coo_tensor(index,torch.ones(len(weights),device=device),(n,n)).to_dense()
                    model.graph_neigh=neigh
                    latent,recon,positive,negative=model(x,x[torch.randperm(n,device=device)],adj)
                    target=torch.cat([torch.ones(n,1,device=device),torch.zeros(n,1,device=device)],1)
                    loss=10*F.mse_loss(recon,x)+F.binary_cross_entropy_with_logits(positive,target)+F.binary_cross_entropy_with_logits(negative,target)
                elif self.source=='spagcn':
                    # Original SpaGCN DEC objective, KMeans initialization and sparse distance weights.
                    if step==0:
                        from sklearn.cluster import KMeans
                        with torch.no_grad():initial=model.gc(x,adj).cpu().numpy()
                        centers=KMeans(10,n_init=20,random_state=context.seed).fit(initial).cluster_centers_
                        model.mu.data.copy_(torch.tensor(centers,device=device))
                    latent,q=model(x,adj);target=model.target_distribution(q).detach();loss=model.loss_function(target,q)
                else:
                    latent,recon=model(x,index);loss=F.mse_loss(recon,x)
                if not torch.isfinite(loss):raise ValueError('nonfinite original spatial pretraining')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},test_opened=False,pretraining_steps=budget)
            torch.save(payload,path)
        def embed(values):
            output=[];model.eval()
            with torch.no_grad():
                for start in range(0,len(values),1024):
                    x=torch.tensor(values[start:start+1024],device=device);n=len(x);idx=torch.arange(n,device=device)
                    if self.source=='graphst':z=x@model.weight1
                    elif self.source=='spagcn':z=model.gc(x,torch.sparse_coo_tensor(torch.stack([idx,idx]),torch.ones(n,device=device),(n,n)))
                    else:z,_=model(x,torch.stack([idx,idx]))
                    output.append(z.cpu().numpy())
            return np.concatenate(output)
        return [_niche_head(context,embed(train_x),embed(eval_x),payload,path,
            self.source+' original full network and unsupervised objective; bounded split-local graph minibatch adaptation; pretraining uses train geometry only; frozen self-loop encoder + train-only niche composition head; no test neighbors/coordinates')]

@register('graphst_composition')
class GraphSTComposition(_GraphOriginal):
    method_id='graphst_composition';display_name='GraphST + center-only composition head';source='graphst'

@register('spagcn_composition')
class SpaGCNComposition(_GraphOriginal):
    method_id='spagcn_composition';display_name='SpaGCN + center-only composition head';source='spagcn'

@register('stagate_composition')
class STAGATEComposition(_GraphOriginal):
    method_id='stagate_composition';display_name='STAGATE + center-only composition head';source='stagate'
