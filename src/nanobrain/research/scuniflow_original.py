"""Original scUniFlow router/Mamba/field pretraining plus V3 readout adaptation."""
from __future__ import annotations
import ast,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from nanobrain.baselines.base import Availability,BaselineAdapter
from nanobrain.baselines.registry import register
from nanobrain.baselines.common import log_expression
from .original_baselines import ROOT,torch_seed
from .spatial_originals import _niche_head


def original_losses():
    source=(ROOT/'reference_methods/scUniFlow/src/scuniflow/train.py').read_text();tree=ast.parse(source)
    names={'pearson_loss','cell_pearson_loss','weighted_recon_loss','sample_t_stratified'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(functions)!=len(names):raise ValueError('scUniFlow original loss snapshot mismatch')
    scope=dict(torch=torch,F=F,np=np);exec(compile(ast.Module(body=functions,type_ignores=[]),'scuniflow_original_losses','exec'),scope)
    return scope


@register('scuniflow_pretrain_readout')
class ScUniFlowOriginal(BaselineAdapter):
    method_id='scuniflow_pretrain_readout';display_name='scUniFlow original pretrained backbone + V3 readouts'
    family='original_model_adapted';tasks=('annotation','integration','spatial_niche');frozen_artifact='scuniflow_pretrain_readout.pt'
    def availability(self,context):
        required=[ROOT/'reference_methods/scUniFlow/src/scuniflow/model.py',ROOT/'reference_methods/mamba_ssd/ssd_minimal.py']
        missing=[str(p) for p in required if not p.is_file()]
        return Availability('source_missing','; '.join(missing)) if missing else Availability('ready')
    def run(self,context):
        from nanobrain.baseline_components import reference_module
        from nanobrain.baselines.foundation import _linear_probe_predictions
        from .mamba2_compat import install_if_missing
        implementation=install_if_missing();sys.path.insert(0,str(ROOT/'reference_methods/scUniFlow/src/scuniflow'))
        source=reference_module('scuniflow');torch_seed(context.seed);device=torch.device(context.device)
        width=context.train.input_dim;grid=math.ceil(math.sqrt(width))
        model=source.scRNAModel(n_genes=width,grid_size=grid,d_model=int(context.options.get('scuniflow_width',512)),
            n_layers=int(context.options.get('scuniflow_layers',8)),d_state_row=64,d_state_col=16,activation_checkpointing=True).to(device)
        model.init_from_spectral(source.build_ordered_gene_coords(width,grid))
        train=log_expression(context.train,context.data_root);evaluation=log_expression(context.validation,context.data_root)
        checkpoint=context.options.get(self.method_id+'_checkpoint');path=context.output_root/self.frozen_artifact;context.output_root.mkdir(parents=True,exist_ok=True)
        genes=torch.arange(width,device=device);batch_size=int(context.options.get('scuniflow_batch_size',4))
        notes='Full original scUniFlow router + 8-layer row/column bidirectional GeneMamba + VectorFieldNet; original weighted Huber and Pearson masked reconstruction; pretrained on this training split; original deterministic Hilbert fallback for unavailable external spectral coordinates; '+implementation
        if checkpoint:
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if payload.get('test_opened') is not False:raise ValueError('unsealed scUniFlow pretraining checkpoint')
            model.load_state_dict(payload['model'])
        else:
            losses=original_losses();rng=np.random.default_rng(context.seed);valid=np.flatnonzero((train>0).any(1))
            mean=torch.tensor(train.mean(0),device=device);std=torch.tensor(train.std(0).clip(1e-4),device=device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
            budget=int(context.options.get('scuniflow_pretrain_steps',math.ceil(context.options.get('scuniflow_pretrain_epochs',80)*len(train)/batch_size)))
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=budget,eta_min=1e-6)
            best=math.inf
            def loss_for(x,seed=None):
                if seed is not None:torch.manual_seed(seed)
                mask=torch.zeros_like(x)
                for i,row in enumerate(x):
                    nz=torch.nonzero(row>0).flatten();take=nz[torch.randperm(len(nz),device=device)[:max(1,int(len(nz)*.2))]];mask[i,take]=1
                x0=torch.randn_like(x)*std+mean;t=losses['sample_t_stratified'](len(x),device)
                pred,target,hats,targets,_=model.forward_train(genes[None,:].expand(len(x),-1),x,mask,t,x0)
                return losses['weighted_recon_loss'](hats,targets)+.1*losses['pearson_loss'](torch.cat(pred),torch.cat(target))+.1*losses['cell_pearson_loss'](pred,target)
            for step in range(budget):
                model.train();idx=rng.choice(valid,batch_size,replace=len(valid)<batch_size);x=torch.tensor(train[idx],device=device)
                optimizer.zero_grad(set_to_none=True);loss=loss_for(x)
                if not torch.isfinite(loss):raise ValueError('nonfinite original scUniFlow pretraining loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step()
                if (step+1)%min(1000,budget)==0 or step+1==budget:
                    model.eval();values=[]
                    with torch.no_grad():
                        for offset in range(0,min(256,len(evaluation)),batch_size):
                            x=torch.tensor(evaluation[offset:offset+batch_size],device=device);values.append(float(loss_for(x,seed=1234+offset)))
                    quality=float(np.mean(values))
                    if quality<best:
                        best=quality;payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},test_opened=False,
                            pretraining_steps=step+1,validation_reconstruction_loss=quality,recipe=notes);torch.save(payload,path)
                    with (context.output_root/'scuniflow-pretrain.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,validation_loss=quality))+'\n')
            payload=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'])
        def embed(features):
            output=[];model.eval()
            with torch.no_grad():
                for start in range(0,len(features),batch_size):
                    x=torch.tensor(features[start:start+batch_size],device=device)
                    z,_=model.compute_context(genes[None,:].expand(len(x),-1),x,torch.zeros_like(x));output.append(z.mean(1).cpu().numpy())
            return np.concatenate(output)
        train_z,eval_z=embed(train),embed(evaluation)
        predictions=_linear_probe_predictions(context,train_z,eval_z,method=self.method_id,notes=notes+'; original downstream frozen-backbone pattern adapted to annotation/integration')
        predictions.append(_niche_head(context,train_z,eval_z,payload,path,notes+'; center-only neighborhood-composition readout adaptation'))
        return predictions
