"""Train-only, validation-selected task heads on complete-method representations."""
from __future__ import annotations
import dataclasses,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nanobrain.baselines.base import AdapterContext,TaskPrediction
from nanobrain.baselines.common import array,labels
from brainwm_benchmark import evaluate
from brainwm_benchmark.protocol import DEVELOPMENT_EVAL_CELLS,EVALUATION_SEED
from .contracts import TASKS
from .original_baselines import torch_seed
from .representations import load_encoder

POLICY='full-method-frozen-representation-transfer-v1'
NON_TRANSFERABLE={
 'xgboost':'supervised decision-tree classifier, no learned cell encoder',
 'celltypist':'task-specific logistic cell classifier, no independent representation backbone',
 'harmony':'transductive batch-correction method; retained on native integration task',
 'scanorama':'transductive integration method; retained on native integration task',
 'bbknn':'batch-balanced neighbor graph; no deployable cell encoder'}


def missing_tasks(method):
    from nanobrain.baselines import build
    return [] if method in NON_TRANSFERABLE else [t for t in TASKS if t not in build(method).tasks]


def encode_file(encoder,features,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);output=None
    for start in range(0,len(features),encoder.batch_size):
        z=encoder(array(features[start:start+encoder.batch_size],np.float32))
        if output is None:output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(features),z.shape[1]))
        output[start:start+len(z)]=z
    if output is None:raise ValueError('empty representation split')
    output.flush();return output


def normalize(train,others=(),statistics=None):
    if statistics is None:
        total=np.zeros(train.shape[1],np.float64);squares=total.copy();n=0
        for start in range(0,len(train),4096):
            x=np.asarray(train[start:start+4096],np.float64);total+=x.sum(0);squares+=(x*x).sum(0);n+=len(x)
        mean=total/n;std=np.sqrt(np.maximum(squares/n-mean*mean,1e-12)).clip(1e-6)
    else:mean,std=statistics
    for z in (train,*others):
        for start in range(0,len(z),4096):z[start:start+4096]=(z[start:start+4096]-mean)/std
        if hasattr(z,'flush'):z.flush()
    return mean,std


def mlp(inputs,outputs,hidden=128):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,outputs))


def numpy_predict(model,z,device,shape=None):
    model.eval();output=[]
    with torch.no_grad():
        for start in range(0,len(z),1024):
            logits=model(torch.tensor(z[start:start+1024],dtype=torch.float32,device=device))
            if shape:logits=logits.view(-1,*shape).softmax(-1)
            output.append(logits.cpu().numpy())
    return np.concatenate(output)


def row_arrays(context,task,model,z):
    device=torch.device(context.device)
    if task=='annotation':
        truth,valid=labels(context.validation,'annotation');idx=np.flatnonzero(valid)
        logits=numpy_predict(model,z[idx],device);prob=torch.softmax(torch.tensor(logits),-1).numpy()
        return dict(y_true=truth[idx].astype(np.int64),y_pred=prob.argmax(1),y_prob=prob,classes=np.arange(prob.shape[1]),eval_indices=idx)
    if task=='spatial_niche':
        truth=array(context.validation.labels['niche'],np.float32);mask=array(context.validation.masks['niche'],bool)
        payload=dict(target_composition=truth,prediction_composition=numpy_predict(model,z,device,truth.shape[1:]),composition_mask=mask)
        if 'sample' in (context.validation.metadata or {}):payload['section_id']=array(context.validation.metadata['sample'])
        return payload
    if task=='integration':
        biology,bv=labels(context.validation,'annotation');batch,bav=labels(context.validation,'integration');valid=bv&bav
        return dict(embedding=np.asarray(z[valid],np.float32),biology_label=biology[valid],batch_label=batch[valid],eval_indices=np.flatnonzero(valid))
    raise ValueError(task)


def metric(task,arrays):
    result=evaluate(task,arrays)
    if dataclasses.is_dataclass(result):result=dataclasses.asdict(result)
    return result.get('metrics',result)


def fit_row_head(context,task,train_z,val_z,policy):
    torch_seed(context.seed+TASKS.index(task));device=torch.device(context.device)
    if task=='annotation':
        y,valid=labels(context.train,'annotation');shape=None;outputs=int(y[valid].max())+1
    else:
        y=array(context.train.labels['niche'],np.float32);mask=array(context.train.masks['niche'],bool);valid=mask.any(1)
        shape=y.shape[1:];outputs=int(np.prod(shape))
    indices=np.flatnonzero(valid)
    if not len(indices):raise ValueError('no train targets for '+task)
    model=mlp(train_z.shape[1],outputs,policy['hidden']).to(device)
    path=context.output_root/(task+'-head.pt');optimizer=torch.optim.AdamW(model.parameters(),lr=policy['lr'],weight_decay=policy['weight_decay'])
    rng=np.random.default_rng(context.seed);best=None;selected_step=0;steps=policy['steps'];interval=policy['eval_interval']
    for step in range(steps):
        idx=rng.choice(indices,min(policy['batch_size'],len(indices)),replace=False);model.train()
        logits=model(torch.tensor(train_z[idx],device=device));target=torch.tensor(y[idx],device=device)
        if task=='annotation':loss=F.cross_entropy(logits,target.long())
        else:
            m=torch.tensor(mask[idx],device=device);per=-(target*logits.view(-1,*shape).log_softmax(-1)).sum(-1)
            loss=torch.stack([per[:,i][m[:,i]].mean() for i in range(shape[0]) if m[:,i].any()]).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite transfer head loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%interval==0 or step+1==steps:
            scores=metric(task,row_arrays(context,task,model,val_z));quality=scores['f1_macro' if task=='annotation' else 'composition_mae']
            if not math.isfinite(quality):raise ValueError('nonfinite validation criterion')
            if best is None or (quality>best if task=='annotation' else quality<best):
                best=quality;selected_step=step+1
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},test_opened=False,selected_step=selected_step,
                    validation_criterion=quality,policy=policy,representation_frozen=True),path)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);return model,dict(checkpoint=path.name,selected_step=selected_step,validation_criterion=best)


def source_indices(pairs,index,max_cells,seed):
    lo,hi=map(int,pairs.pair_ptr[index:index+2]);ids=np.arange(lo,hi)
    if len(ids)>max_cells:ids=ids[np.sort(np.random.default_rng(seed+index*104729).choice(len(ids),max_cells,replace=False))]
    return pairs.source_index[ids],pairs.target_index[ids]


def times(pairs):
    from nanobrain.baselines.development import _TimeScale
    return _TimeScale.from_training(pairs)


def flow_input(z,state,tau,source_time,target_time,scale,device):
    n=len(state);c=torch.tensor([[tau,scale.absolute(source_time),scale.delta(source_time,target_time)]],device=device).expand(n,-1)
    return torch.cat([torch.tensor(np.asarray(z),dtype=torch.float32,device=device),state,c],1)


def rollout(model,z,source,st,tt,scale,device,steps):
    state=torch.tensor(np.asarray(source),dtype=torch.float32,device=device);model.eval();dt=1./steps
    with torch.no_grad():
        for i in range(steps):
            velocity=model(flow_input(z,state,i*dt,st,tt,scale,device))
            other=model(flow_input(z,state+dt*velocity,(i+1)*dt,st,tt,scale,device));state=state+dt*(velocity+other)/2
    return state.cpu().numpy()


def development_arrays(context,model,z,scale,policy):
    pairs=context.development_validation
    if pairs is None:raise ValueError('development populations unavailable')
    truth=[];prediction=[];pointer=[0]
    for i,group in enumerate(pairs.groups):
        src,tgt=source_indices(pairs,i,DEVELOPMENT_EVAL_CELLS,EVALUATION_SEED)
        x=array(pairs.dataset.features[src],np.float32);target=array(pairs.dataset.features[tgt],np.float32)
        pred=rollout(model,z[src],x,group['source_time_days'],group['target_time_days'],scale,torch.device(context.device),policy['rollout_steps'])
        truth.append(target);prediction.append(pred);pointer.append(pointer[-1]+len(target))
    return dict(target=np.concatenate(truth),prediction=np.concatenate(prediction),group_ptr=np.array(pointer),population_unpaired=np.array([True]))


def fit_development_head(context,train_z,val_z,policy):
    from scipy.optimize import linear_sum_assignment
    torch_seed(context.seed+TASKS.index('development'));device=torch.device(context.device);pairs=context.development_train
    if pairs is None:raise ValueError('development train populations unavailable')
    scale=times(pairs);model=nn.Sequential(nn.Linear(train_z.shape[1]+context.train.input_dim+3,policy['hidden']),nn.SiLU(),
        nn.Linear(policy['hidden'],policy['hidden']),nn.SiLU(),nn.Linear(policy['hidden'],context.train.input_dim)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=policy['lr'],weight_decay=policy['weight_decay']);rng=np.random.default_rng(context.seed)
    path=context.output_root/'development-head.pt';best=math.inf;selected=0
    for step in range(policy['development_steps']):
        i=int(rng.integers(len(pairs)));g=pairs.groups[i];src,tgt=source_indices(pairs,i,policy['development_cells'],context.seed+step)
        x=array(pairs.dataset.features[src],np.float32);y=array(pairs.dataset.features[tgt],np.float32)
        # Equal-mass exact OT couples unpaired populations ONLY inside a train minibatch.
        cost=((x[:,None,:]-y[None,:,:])**2).sum(-1);a,b=linear_sum_assignment(cost)
        x=x[a];y=y[b];z=train_z[src[a]];tau=float(rng.random());state=torch.tensor((1-tau)*x+tau*y,device=device)
        model.train();prediction=model(flow_input(z,state,tau,g['source_time_days'],g['target_time_days'],scale,device))
        loss=F.mse_loss(prediction,torch.tensor(y-x,device=device))
        if not torch.isfinite(loss):raise ValueError('nonfinite transfer flow loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%policy['eval_interval']==0 or step+1==policy['development_steps']:
            quality=metric('development',development_arrays(context,model,val_z,scale,policy))['wasserstein2']
            if not math.isfinite(quality):raise ValueError('nonfinite development validation W2')
            if quality<best:
                best=quality;selected=step+1;torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},test_opened=False,
                    validation_wasserstein2=best,selected_step=selected,time_scale=scale.__dict__,policy=policy,representation_frozen=True),path)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    return model,scale,dict(checkpoint=path.name,selected_step=selected,validation_criterion=best)


def transfer_predictions(context,encoder,tasks,policy,open_test):
    """`open_test` is invoked only after every fitted head has a sealed checkpoint."""
    context.output_root.mkdir(parents=True,exist_ok=True)
    train_z=encode_file(encoder,context.train.features,context.output_root/'train-embedding.npy')
    val_z=encode_file(encoder,context.validation.features,context.output_root/'validation-embedding.npy')
    stats=normalize(train_z,(val_z,));np.savez(context.output_root/'train-embedding-scale.npz',mean=stats[0],std=stats[1])
    models={};attestations={};scale=None
    for task in tasks:
        if task=='integration':continue
        if task=='development':models[task],scale,attestations[task]=fit_development_head(context,train_z,val_z,policy)
        else:models[task],attestations[task]=fit_row_head(context,task,train_z,val_z,policy)
    def predict(ctx,z,split_name):
        rows=[]
        for task in tasks:
            arrays=development_arrays(ctx,models[task],z,scale,policy) if task=='development' else row_arrays(ctx,task,models.get(task),z)
            fit_rows=(len(context.development_train.source_index) if task=='development' else int(array(context.train.masks['niche'],bool).any(1).sum()) if task=='spatial_niche' else int(labels(context.train,'annotation')[1].sum()) if task=='annotation' else context.train.size)
            rows.append(dict(task=task,metrics={task+'.'+k:float(v) for k,v in metric(task,arrays).items()},fit_rows=fit_rows,
                eval_rows=len(arrays.get('target',arrays.get('y_true',arrays.get('embedding',arrays.get('target_composition',[]))))),
                regime='frozen_full_method_representation_with_task_head' if task!='integration' else 'frozen_full_method_representation',
                notes=encoder.notes+'; '+POLICY,adaptation=True,head=attestations.get(task,{'type':'identity; standard integration metrics, no label-aware fitted head'})))
            np.savez_compressed(context.output_root/(split_name+'-'+task+'.npz'),**arrays)
        return rows
    validation=predict(context,val_z,'validation')
    for task,a in attestations.items():
        if torch.load(context.output_root/a['checkpoint'],map_location='cpu',weights_only=False).get('test_opened') is not False:raise ValueError('unsealed task head')
    test_context=open_test();test_z=encode_file(encoder,test_context.validation.features,context.output_root/'test-embedding.npy');normalize(test_z,statistics=stats)
    test=predict(test_context,test_z,'test')
    return validation,test,attestations


def transfer_job(job):
    from nanobrain.data import load_v3_subset
    torch.set_num_threads(job.get('threads',2));data=load_v3_subset(job['data'],include_test=False)
    output=Path('/work/transfer');output.mkdir(parents=True,exist_ok=True)
    ctx=AdapterContext(data.train,data.validation,Path(job['data']),output,job['seed'],job['device'],job.get('threads',2),data.development_train,data.development_validation,job['options'])
    encoder=load_encoder(ctx,job['method'],'/baseline/native');started=time.monotonic()
    def open_test():
        full=load_v3_subset(job['data'],include_test=True)
        return dataclasses.replace(ctx,validation=full.test,development_validation=full.development_test)
    validation,test,heads=transfer_predictions(ctx,encoder,job['tasks'],job['transfer_policy'],open_test)
    # Embeddings are regenerable scratch, not required benchmark artifacts.
    for path in output.glob('*-embedding.npy'):path.unlink()
    return dict(status='complete',method=job['method'],seed=job['seed'],validation=validation,test=test,head_checkpoints=heads,
        baseline_job_id=job['baseline_job_id'],rows=data.train.size,configuration_frozen_before_test=True,representation_frozen=True,
        transfer_policy=job['transfer_policy'],policy=POLICY,runtime_seconds=time.monotonic()-started)
