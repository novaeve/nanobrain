"""Original model training recipes with explicit V3 input/output adaptations.

Pretrained classification fine-tuning is distinct from frozen linear probes.
The common panel, fixed split and validation-best checkpoint are never changed.
"""
from __future__ import annotations
import importlib.util,json,math,pickle,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nanobrain.baselines.base import BaselineAdapter,Availability,TaskPrediction
from nanobrain.baselines.registry import register,build as base_build
from nanobrain.baselines.common import array,labels,panel_pseudocounts,log_expression

ROOT=Path(__file__).resolve().parents[3]


def torch_seed(seed):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _annotation_prediction(context,logits,indices,notes,regime):
    probability=torch.softmax(torch.as_tensor(logits),-1).numpy()
    y,_=labels(context.validation,'annotation')
    return TaskPrediction('annotation',dict(y_true=y[indices].astype(np.int64),y_pred=probability.argmax(1).astype(np.int64),
        y_prob=probability.astype(np.float32),classes=np.arange(probability.shape[1]),eval_indices=indices),
        int(labels(context.train,'annotation')[1].sum()),len(indices),notes,regime)


def _inference(model,features,indices,batch_size,device):
    values=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            x=torch.tensor(features[indices[start:start+batch_size]],dtype=torch.float32,device=device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits,_,_=model(x)
            values.append(logits.float().cpu().numpy())
    return np.concatenate(values)


def fine_tune(context,method,model,train_features,eval_features,*,epochs,batch_size,lr,reconstruction=False,notes=''):
    from sklearn.metrics import f1_score
    torch_seed(context.seed);device=torch.device(context.device);model.to(device)
    y,valid=labels(context.train,'annotation');vy,vvalid=labels(context.validation,'annotation')
    ti=np.flatnonzero(valid);vi=np.flatnonzero(vvalid);checkpoint=context.options.get(method+'_checkpoint')
    path=context.output_root/(method+'.pt');context.output_root.mkdir(parents=True,exist_ok=True)
    steps=0;best=-math.inf;patience=0
    if checkpoint:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload.get('test_opened') is not False:raise ValueError('missing sealed-test attestation')
        model.load_state_dict(payload['model']);steps=payload['steps']
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        cap=int(context.options.get('native_max_steps',10000));rng=np.random.default_rng(context.seed)
        for epoch in range(epochs):
            model.train();order=rng.permutation(ti)
            for start in range(0,len(order),batch_size):
                idx=order[start:start+batch_size];x=torch.tensor(train_features[idx],dtype=torch.float32,device=device)
                target=torch.tensor(y[idx],dtype=torch.long,device=device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    logits,_,recon=model(x);loss=F.cross_entropy(logits.float(),target)
                    if reconstruction:loss=loss+F.mse_loss(recon.float(),x.float())
                if not torch.isfinite(loss):raise ValueError('nonfinite original-model training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();steps+=1
                if steps>=cap:break
            predicted=_inference(model,eval_features,vi,batch_size,device).argmax(1)
            quality=float(f1_score(vy[vi],predicted,average='macro',zero_division=0))
            with (context.output_root/(method+'-training.jsonl')).open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,steps=steps,validation_macro_f1=quality))+'\n')
            if quality>best:
                best=quality;patience=0
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},steps=steps,epoch=epoch+1,
                    validation_macro_f1=quality,test_opened=False,recipe=notes),path)
            else:patience+=1
            if patience>=int(context.options.get('native_patience',30)) or steps>=cap:break
        payload=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'])
    logits=_inference(model,eval_features,vi,batch_size,device)
    return [_annotation_prediction(context,logits,vi,notes+'; validation Macro-F1 selects checkpoint; steps='+str(steps),
                                   'pretrained_backbone_finetune' if 'pretrained' in notes.lower() else 'original_model_from_scratch')]


def geneformer_model(context):
    from transformers import BertConfig,BertForSequenceClassification
    from safetensors.torch import load_file
    from nanobrain.baselines.foundation import _catalog
    root=Path(context.options.get('geneformer_root',ROOT/'assets/foundation_models/geneformer'))
    dictionaries=root/'geneformer/gene_dictionaries_30m'
    with (dictionaries/'token_dictionary_gc30M.pkl').open('rb') as f:tokens=pickle.load(f)
    with (dictionaries/'gene_median_dictionary_gc30M.pkl').open('rb') as f:medians=pickle.load(f)
    ensembl,_=_catalog(context.data_root,context.options.get('gene_catalog'))
    mapping=[(i,tokens[g],medians[g]) for i,g in enumerate(ensembl) if g in tokens and float(medians.get(g,0))>0]
    if not mapping:raise ValueError('no Geneformer panel genes mapped')
    config=BertConfig.from_pretrained(root/'Geneformer-V1-10M',local_files_only=True)
    y,valid=labels(context.train,'annotation');classes=int(y[valid].max())+1
    config.num_labels=classes
    backbone=BertForSequenceClassification(config)
    weights=load_file(str(root/'Geneformer-V1-10M/model.safetensors'))
    weights={k:v for k,v in weights.items() if k.startswith('bert.')}
    missing,extra=backbone.load_state_dict(weights,strict=False)
    expected_new={'bert.pooler.dense.weight','bert.pooler.dense.bias','classifier.weight','classifier.bias'}
    if any(k not in expected_new and not k.endswith('position_ids') for k in missing+extra):raise ValueError('Geneformer pretrained weight mismatch')
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=backbone
            self.register_buffer('columns',torch.tensor([m[0] for m in mapping],dtype=torch.long));self.register_buffer('tokens',torch.tensor([m[1] for m in mapping],dtype=torch.long))
            self.register_buffer('medians',torch.tensor([m[2] for m in mapping],dtype=torch.float32))
        def forward(self,x):
            x=x[:,self.columns];rank=x/x.sum(1,keepdim=True).clamp_min(1.)*10000/self.medians
            order=rank.masked_fill(x<=0,-torch.inf).argsort(1,descending=True);ids=self.tokens[order]
            mask=torch.arange(len(mapping),device=x.device)[None,:]<(x>0).sum(1).clamp_min(1)[:,None]
            result=self.backbone(input_ids=ids,attention_mask=mask,output_hidden_states=True)
            return result.logits,result.hidden_states[-1][:,0],None
    return Model()


class _FineTunedFoundation(BaselineAdapter):
    tasks=('annotation',);family='foundation_finetuned';base_method=''
    @property
    def frozen_artifact(self):return self.method_id+'.pt'
    def availability(self,context):return base_build(self.base_method).availability(context)
    def run(self,context):
        torch_seed(context.seed)
        if self.base_method.startswith('geneformer'):
            model=geneformer_model(context);epochs=int(context.options.get('geneformer_finetune_epochs',3));batch=int(context.options.get('geneformer_finetune_batch_size',16));lr=5e-5
            notes='Geneformer official V1-10M pretrained BERT + official BertForSequenceClassification cell classifier; median-normalized gene ranks; all backbone layers fine-tuned; original cell-classification recipe adapted to fixed 256-gene panel'
        else:
            from nanobrain.baselines.foundation import _catalog
            reference=Path(context.options.get('scgpt_reference_root',ROOT/'reference_methods/scBrain'));sys.path.insert(0,str(reference))
            from src.models.scGPT.scGPT_annotation import scGPTForAnnotation
            _,symbols=_catalog(context.data_root,context.options.get('gene_catalog'))
            root=Path(context.options.get(self.base_method+'_root',ROOT/'assets/foundation_models'/self.base_method.replace('_','-')))
            vocab=json.loads((root/'vocab.json').read_text());mapped=[(i,vocab[g]) for i,g in enumerate(symbols) if g in vocab]
            y,valid=labels(context.train,'annotation');classes=int(y[valid].max())+1
            original=scGPTForAnnotation(str(root),torch.tensor([x[1] for x in mapped]),output_num=[1,1,classes],freeze_backbone=False)
            class Model(nn.Module):
                def __init__(self):super().__init__();self.original=original;self.register_buffer('columns',torch.tensor([x[0] for x in mapped]))
                def forward(self,x):
                    logits,embedding=self.original(x[:,self.columns]);return logits[-1],embedding,None
            model=Model();epochs=int(context.options.get('scgpt_finetune_epochs',10));batch=int(context.options.get('scgpt_finetune_batch_size',32));lr=1e-4
            notes=self.base_method+' official pretrained checkpoint + original scGPT transformer/CLS decoder; full-backbone cell annotation fine-tuning, quantile-binned panel, 10-epoch recipe'
        return fine_tune(context,self.method_id,model,panel_pseudocounts(context.train,context.data_root),panel_pseudocounts(context.validation,context.data_root),epochs=epochs,batch_size=batch,lr=lr,notes=notes)


@register('geneformer_v1_10m_finetuned')
class GeneformerFineTuned(_FineTunedFoundation):
    method_id='geneformer_v1_10m_finetuned';display_name='Geneformer V1-10M fine-tuned';base_method='geneformer_v1_10m'

@register('scgpt_brain_finetuned')
class ScGPTBrainFineTuned(_FineTunedFoundation):
    method_id='scgpt_brain_finetuned';display_name='scGPT brain fine-tuned';base_method='scgpt_brain'

@register('scgpt_human_finetuned')
class ScGPTHumanFineTuned(_FineTunedFoundation):
    method_id='scgpt_human_finetuned';display_name='scGPT human fine-tuned';base_method='scgpt_human'

@register('scbrain_original')
class ScBrainOriginal(BaselineAdapter):
    method_id='scbrain_original';display_name='scBrain original cascade';family='original_model_adapted';tasks=('annotation',);frozen_artifact='scbrain_original.pt'
    def availability(self,context):return Availability('ready') if (ROOT/'reference_methods/scBrain/src/models/mjm.py').exists() else Availability('source_missing')
    def run(self,context):
        from nanobrain.baseline_components import reference_module
        torch_seed(context.seed);y,valid=labels(context.train,'annotation');classes=int(y[valid].max())+1
        original=reference_module('scbrain').mjm_1(input_dim=context.train.input_dim,latent_dim=20,e_layers=3,d_layers=1,
            enc_hidden_dim=256,dec_hidden_dim=256,dropout=.3,output_num=[1,1,classes],residual_mode='feature')
        class Model(nn.Module):
            def __init__(self):super().__init__();self.original=original
            def forward(self,x):
                recon,logits,embedding=self.original(x);return logits[-1],embedding,recon
        return fine_tune(context,self.method_id,Model(),log_expression(context.train,context.data_root),log_expression(context.validation,context.data_root),
            epochs=int(context.options.get('scbrain_epochs',500)),batch_size=int(context.options.get('scbrain_batch_size',512)),lr=1e-3,reconstruction=True,
            notes='Full original scBrain 3-stage cascade and feature residuals, reconstruction+cross entropy; two unavailable coarse ontology levels collapsed to singleton heads, finest canonical cell labels retained; common native_max_steps budget')
