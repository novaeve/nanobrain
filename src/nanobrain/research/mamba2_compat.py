"""Unfused Mamba-2 using the upstream paper's pure-PyTorch SSD algorithm.

Compatibility implementation of Mamba2Simple's non-fused path for environments
without a matching mamba_ssm CUDA wheel. Model topology is unchanged. The
upstream SSD implementation and Apache-2.0 license are vendored under reference_methods.
"""
from __future__ import annotations
import importlib.util,math,sys,types
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def ssd_function():
    path=Path(__file__).resolve().parents[3]/'reference_methods/mamba_ssd/ssd_minimal.py'
    spec=importlib.util.spec_from_file_location('nanobrain_upstream_ssd',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ssd_minimal_discrete


class Mamba2(nn.Module):
    def __init__(self,d_model,d_state=64,d_conv=4,expand=2,headdim=64,ngroups=1,chunk_size=256,**kwargs):
        super().__init__()
        if kwargs:raise ValueError('unsupported Mamba compatibility options: '+str(sorted(kwargs)))
        self.d_inner=d_model*expand;self.d_state=d_state;self.headdim=headdim;self.ngroups=ngroups;self.chunk_size=chunk_size
        self.nheads=self.d_inner//headdim
        if self.d_inner%headdim or self.nheads%ngroups:raise ValueError('invalid Mamba head/group dimensions')
        self.in_proj=nn.Linear(d_model,2*self.d_inner+2*ngroups*d_state+self.nheads,bias=False)
        conv_dim=self.d_inner+2*ngroups*d_state
        self.conv1d=nn.Conv1d(conv_dim,conv_dim,d_conv,groups=conv_dim,padding=d_conv-1)
        dt=torch.exp(torch.rand(self.nheads)*(math.log(.1)-math.log(.001))+math.log(.001)).clamp_min(1e-4)
        self.dt_bias=nn.Parameter(dt+torch.log(-torch.expm1(-dt)))
        self.A_log=nn.Parameter(torch.empty(self.nheads).uniform_(1,16).log());self.D=nn.Parameter(torch.ones(self.nheads))
        self.norm_weight=nn.Parameter(torch.ones(self.d_inner));self.out_proj=nn.Linear(self.d_inner,d_model,bias=False)
        self.ssd=ssd_function()
    def forward(self,u):
        batch,length,_=u.shape
        z,xbc,dt=self.in_proj(u).split([self.d_inner,self.d_inner+2*self.ngroups*self.d_state,self.nheads],dim=-1)
        xbc=F.silu(self.conv1d(xbc.transpose(1,2)).transpose(1,2)[:,:length])
        x,b,c=xbc.split([self.d_inner,self.ngroups*self.d_state,self.ngroups*self.d_state],dim=-1)
        x=x.float().reshape(batch,length,self.nheads,self.headdim)
        b=b.float().reshape(batch,length,self.ngroups,self.d_state).repeat_interleave(self.nheads//self.ngroups,dim=2)
        c=c.float().reshape_as(b) if self.ngroups==self.nheads else c.float().reshape(batch,length,self.ngroups,self.d_state).repeat_interleave(self.nheads//self.ngroups,dim=2)
        dt=F.softplus(dt.float()+self.dt_bias.float());a=-self.A_log.float().exp()*dt
        block=min(self.chunk_size,length)
        while length%block:block-=1
        y,_=self.ssd(x*dt[:,:,:,None],a,b,c,block)
        y=(y+x*self.D.float()[None,None,:,None]).reshape(batch,length,self.d_inner)
        y=y*F.silu(z.float());y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-5)*self.norm_weight
        return self.out_proj(y.to(u.dtype))


def install_if_missing():
    try:
        import mamba_ssm
        return 'installed mamba_ssm CUDA implementation'
    except ImportError:
        module=types.ModuleType('mamba_ssm');module.Mamba2=Mamba2;sys.modules['mamba_ssm']=module
        return 'upstream SSD paper algorithm with unfused PyTorch Mamba-2 operations (CUDA wheel compatibility adaptation)'
