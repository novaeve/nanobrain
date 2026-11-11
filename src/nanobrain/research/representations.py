"""Frozen full-method encoders for explicitly labelled downstream adaptations.

No component-library approximation is used here. Native checkpoints are mounted
read-only. The sole exception is scVI, whose historical transductive adapter did
not export weights: its inductive, batch-agnostic VAE is fitted on TRAIN ONLY.
"""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
import json,math,pickle,sys
import numpy as np
import torch
from torch import nn
from nanobrain.baselines.common import array,labels
from .original_baselines import ROOT,geneformer_model,torch_seed


def checkpoint(directory,method):
    paths=list(Path(directory).rglob(method+'.pt'))
    if len(paths)!=1:raise ValueError('exactly one native checkpoint required: '+method)
    payload=torch.load(paths[0],map_location='cpu',weights_only=False)
    if payload.get('test_opened') is not False:raise ValueError('native checkpoint lacks sealed-test attestation')
    return payload,paths[0]


class Encoder:
    def __init__(self,model,forward,notes,batch_size=256):
        self.model=model;self.forward=forward;self.notes=notes;self.batch_size=batch_size
        if model is not None:
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
    def __call__(self,values):
        output=[]
        for start in range(0,len(values),self.batch_size):
            # Some CNFs compute a divergence internally and enable autograd.
            with torch.no_grad():z=self.forward(np.asarray(values[start:start+self.batch_size],np.float32))
            z=array(z,np.float32)
            if z.ndim!=2 or not np.isfinite(z).all():raise ValueError('invalid frozen representation')
            output.append(z)
        return np.concatenate(output)


def load_encoder(context,method,native_directory):
    torch_seed(context.seed);device=torch.device(context.device);options=context.options
    mean=np.load(context.data_root/'normalization_mean.npy');std=np.load(context.data_root/'normalization_std.npy')
    def native(x):return np.maximum(x*std+mean,0).astype(np.float32)
    def tensor(x):return torch.tensor(x,dtype=torch.float32,device=device)
    def counts(x):return np.rint(np.expm1(native(x))).astype(np.float32)
    notes='frozen complete-method encoder; no task labels or test neighbors enter the encoder'
    if method in {'geneformer_v1_10m','scgpt_brain','scgpt_human'}:
        from nanobrain.baselines.foundation import _catalog
        from safetensors.torch import load_file
        ensembl,symbols=_catalog(context.data_root,options.get('gene_catalog'))
        if method=='geneformer_v1_10m':
            from transformers import BertConfig,BertModel
            root=Path(options['geneformer_root']);d=root/'geneformer/gene_dictionaries_30m'
            with (d/'token_dictionary_gc30M.pkl').open('rb') as f:tokens=pickle.load(f)
            with (d/'gene_median_dictionary_gc30M.pkl').open('rb') as f:medians=pickle.load(f)
            mapping=[(i,tokens[g],medians[g]) for i,g in enumerate(ensembl) if g in tokens and float(medians.get(g,0))>0]
            if not mapping:raise ValueError('no Geneformer genes mapped')
            config=BertConfig.from_pretrained(root/'Geneformer-V1-10M',local_files_only=True)
            model=BertModel(config,add_pooling_layer=False)
            weights=load_file(str(root/'Geneformer-V1-10M/model.safetensors'))
            missing,extra=model.load_state_dict({k.removeprefix('bert.'):v for k,v in weights.items() if k.startswith('bert.')},strict=False)
            if any(not k.endswith('position_ids') for k in missing+extra):raise ValueError('Geneformer weight mismatch')
            model.to(device);cols=np.array([v[0] for v in mapping]);ids=torch.tensor([int(v[1]) for v in mapping],dtype=torch.long,device=device)
            med=tensor([v[2] for v in mapping])
            def forward(x):
                x=tensor(counts(x)[:,cols]);rank=x/x.sum(1,keepdim=True).clamp_min(1)*10000/med
                order=rank.masked_fill(x<=0,-torch.inf).argsort(1,descending=True)
                mask=torch.arange(len(mapping),device=device)[None,:]<(x>0).sum(1).clamp_min(1)[:,None]
                hidden=model(input_ids=ids[order],attention_mask=mask).last_hidden_state
                return (hidden*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        else:
            sys.path.insert(0,str(options['scgpt_reference_root']))
            from src.models.scGPT.scGPT_annotation import scGPTForAnnotation
            root=Path(options[method+'_root']);vocab=json.loads((root/'vocab.json').read_text())
            mapping=[(i,vocab[g]) for i,g in enumerate(symbols) if g in vocab]
            if not mapping:raise ValueError('no scGPT genes mapped')
            model=scGPTForAnnotation(str(root),torch.tensor([v[1] for v in mapping]),output_num=[1,1,1],dropout=0.,freeze_backbone=True).to(device)
            cols=np.array([v[0] for v in mapping]);forward=lambda x:model(tensor(counts(x)[:,cols]))[1]
        return Encoder(model,forward,notes+'; original external pretrained weights; original pooling',options.get('transfer_encode_batch',128))
    if method in {'geneformer_v1_10m_finetuned','scgpt_brain_finetuned','scgpt_human_finetuned','scbrain_original'}:
        payload,_=checkpoint(native_directory,method)
        if method.startswith('geneformer'):
            model=geneformer_model(context);convert=counts
        elif method.startswith('scgpt'):
            from nanobrain.baselines.foundation import _catalog
            sys.path.insert(0,str(options['scgpt_reference_root']))
            from src.models.scGPT.scGPT_annotation import scGPTForAnnotation
            base=method.removesuffix('_finetuned');root=Path(options[base+'_root'])
            _,symbols=_catalog(context.data_root,options.get('gene_catalog'));vocab=json.loads((root/'vocab.json').read_text())
            mapping=[(i,vocab[g]) for i,g in enumerate(symbols) if g in vocab]
            y,valid=labels(context.train,'annotation');classes=int(y[valid].max())+1
            original=scGPTForAnnotation(str(root),torch.tensor([v[1] for v in mapping]),output_num=[1,1,classes],freeze_backbone=False)
            class Model(nn.Module):
                def __init__(self):super().__init__();self.original=original;self.register_buffer('columns',torch.tensor([v[0] for v in mapping]))
                def forward(self,x):
                    logits,z=self.original(x[:,self.columns]);return logits[-1],z,None
            model=Model();convert=counts
        else:
            from nanobrain.baseline_components import reference_module
            y,valid=labels(context.train,'annotation');classes=int(y[valid].max())+1
            original=reference_module('scbrain').mjm_1(input_dim=context.train.input_dim,latent_dim=20,e_layers=3,d_layers=1,
                enc_hidden_dim=256,dec_hidden_dim=256,dropout=.3,output_num=[1,1,classes],residual_mode='feature')
            class Model(nn.Module):
                def __init__(self):super().__init__();self.original=original
                def forward(self,x):
                    recon,logits,z=self.original(x);return logits[-1],z,recon
            model=Model();convert=native
        model.load_state_dict(payload['model']);model.to(device)
        return Encoder(model,lambda x:model(tensor(convert(x)))[1],notes+'; native validation-best annotation-trained backbone; label-supervised pretraining explicitly retained',128)
    if method in {'graphst_composition','spagcn_composition','stagate_composition'}:
        payload,_=checkpoint(native_directory,method);width=int(options.get('spatial_latent',32));dim=context.train.input_dim
        if method.startswith('graphst'):
            from GraphST.model import Encoder_sparse
            model=Encoder_sparse(dim,width,graph_neigh=None)
            def forward(x):return tensor(native(x))@model.weight1
        elif method.startswith('spagcn'):
            from SpaGCN.models import simple_GC_DEC
            model=simple_GC_DEC(dim,width);model.mu=nn.Parameter(torch.zeros(10,width))
            def forward(x):
                x=tensor(native(x));i=torch.arange(len(x),device=device)
                return model.gc(x,torch.sparse_coo_tensor(torch.stack([i,i]),torch.ones(len(i),device=device),(len(i),len(i))))
        else:
            sys.path.insert(0,str(ROOT/'reference_methods/STAGATE_pyG'));from STAGATE_pyG import STAGATE
            model=STAGATE([dim,512,width])
            def forward(x):
                x=tensor(native(x));i=torch.arange(len(x),device=device);return model(x,torch.stack([i,i]))[0]
        model.load_state_dict(payload['model']);model.to(device)
        return Encoder(model,forward,notes+'; original train-only spatial pretraining; self-loop induction on every downstream split')
    if method=='scuniflow_pretrain_readout':
        from nanobrain.baseline_components import reference_module
        from .mamba2_compat import install_if_missing
        install_if_missing();sys.path.insert(0,str(ROOT/'reference_methods/scUniFlow/src/scuniflow'));source=reference_module('scuniflow')
        dim=context.train.input_dim;grid=math.ceil(math.sqrt(dim))
        model=source.scRNAModel(n_genes=dim,grid_size=grid,d_model=options.get('scuniflow_width',512),n_layers=options.get('scuniflow_layers',8),
            d_state_row=64,d_state_col=16,activation_checkpointing=True)
        model.init_from_spectral(source.build_ordered_gene_coords(dim,grid));payload,_=checkpoint(native_directory,method)
        model.load_state_dict(payload['model']);model.to(device);genes=torch.arange(dim,device=device)
        def forward(x):
            x=tensor(native(x));return model.compute_context(genes[None,:].expand(len(x),-1),x,torch.zeros_like(x))[0].mean(1)
        return Encoder(model,forward,notes+'; original router and complete GeneMamba encoder',options.get('scuniflow_batch_size',25))
    if method in {'scfit','scnode','trajectorynet'}:
        from nanobrain.baselines.development import _load_scfit,_load_scnode,_load_trajectorynet
        payload,path=checkpoint(native_directory,method);a=payload.get('architecture',{})
        if method=='scfit':
            cls,_=_load_scfit();dim=context.train.input_dim
            model=cls(n_genes=dim,top_k_genes=dim,d_model=a.get('d_model',256),nhead=a.get('heads',8),d_ff=a.get('d_ff',1024),
                nlayers=a.get('layers',4),dropout=options.get('scfit_dropout',.1),value_max=options.get('scfit_value_max',32.),time_conditioning_mode='adaln')
            def forward(x):
                x=tensor(native(x));values=torch.nn.functional.pad(x,(1,0));n=len(x)
                genes=torch.cat([torch.tensor([model.cls_token_id],device=device),torch.arange(dim,device=device)])[None,:].expand(n,-1)
                states=torch.ones_like(genes);states[:,0]=3;zero=torch.zeros(n,1,device=device)
                return model.encode(genes,values,states,torch.zeros_like(genes,dtype=torch.bool),zero,t_src=zero,t_tgt=zero)[:,0]
        elif method=='scnode':
            cls,_=_load_scnode();h=a.get('hidden_dim',128)
            model=cls(n_genes=context.train.input_dim,latent_dim=a.get('latent_dim',32),enc_latent_list=[h],dec_latent_list=[h],drift_latent_size=[h],
                latent_enc_act='relu',latent_dec_act='relu',drift_act='relu',ode_method='rk4')
            forward=lambda x:model.latent_encoder(tensor(native(x)))[0]
        else:
            import joblib
            cls,_=_load_trajectorynet();pre=joblib.load(path.with_name('trajectorynet_preprocessing.joblib'))
            args=Namespace(dims=payload['hidden'],layer_type='concatsquash',nonlinearity='tanh',divergence_fn='brute_force',residual=False,rademacher=False,
                time_scale=1.,train_T=False,solver='rk4',num_blocks=1,batch_norm=False,bn_lag=0.,atol=1e-5,rtol=1e-5,step_size=.25,
                test_solver='rk4',test_atol=1e-5,test_rtol=1e-5)
            model=cls(args,payload['dimensions'],regularization_fns=None)
            def forward(x):
                x=tensor(pre['scaler'].transform(pre['pca'].transform(x)))
                return model(x,torch.zeros(len(x),1,device=device),integration_times=torch.tensor([0.,1.],device=device),reverse=False)[0]
        model.load_state_dict(payload['model']);model.to(device)
        return Encoder(model,forward,notes+'; full native dynamics checkpoint; static extraction uses fixed zero-time scFiT / scNODE latent mean / canonical unit-time CNF map',128)
    if method=='scvi_scanvi':
        import anndata as ad,scvi
        scvi.settings.seed=context.seed;path=context.output_root/'scvi-transfer-backbone.pt'
        configuration=dict(n_hidden=options.get('scvi_hidden',128),n_latent=options.get('scvi_latent',32),n_layers=2,gene_likelihood='nb')
        adata=ad.AnnData(X=counts(array(context.train.features,np.float32)));scvi.model.SCVI.setup_anndata(adata)
        vae=scvi.model.SCVI(adata,**configuration)
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if payload.get('test_opened') is not False:raise ValueError('unsealed scVI transfer fit')
            vae.module.load_state_dict(payload['model'])
        else:
            vae.train(max_epochs=options.get('scvi_epochs',400),batch_size=options.get('scvi_batch_size',2048),accelerator='gpu' if device.type=='cuda' else 'cpu',
                devices=1,early_stopping=True,enable_progress_bar=False,check_val_every_n_epoch=5)
            torch.save(dict(model={k:v.detach().cpu() for k,v in vae.module.state_dict().items()},configuration=configuration,test_opened=False),path)
        model=vae.module.to(device)
        def forward(x):return model.inference(x=tensor(counts(x)),batch_index=torch.zeros(len(x),1,dtype=torch.long,device=device))['qz'].loc
        return Encoder(model,forward,notes+'; explicitly adapted TRAIN-ONLY batch-agnostic scVI VAE; historical transductive native runner did not export weights; new backbone fit is separately recorded')
    raise ValueError('no transferable complete-method representation: '+method)
