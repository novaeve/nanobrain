#!/usr/bin/env python3
"""Populate persistent Triton autotune results for Flash BLA varlen kernels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import torch
import triton

from flash_bla.ops.linear_attn import varlen_linear_attention


def power_of_two_buckets(minimum:int,maximum:int)->list[int]:
    if minimum<1 or maximum<minimum:
        raise ValueError("expected 1 <= minimum <= maximum")
    value=1<<(minimum-1).bit_length();result=[]
    while value<=maximum:
        result.append(value);value*=2
    return result


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--min-seqlen",type=int,default=32)
    parser.add_argument("--max-seqlen",type=int,default=65536)
    parser.add_argument("--heads",type=int,default=8,
                        help="combined positive and correction heads")
    parser.add_argument("--head-dim",type=int,default=64)
    parser.add_argument("--sequences",type=int,default=8)
    parser.add_argument("--token-budget",type=int,default=65536)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output",type=Path,default=None)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Flash BLA warmup")
    if min(args.heads,args.head_dim,args.sequences,args.token_budget)<1:
        raise SystemExit("heads, head-dim, sequences, and token-budget must be positive")

    device=torch.device(args.device);torch.cuda.set_device(device)
    torch.manual_seed(20260912);torch.cuda.manual_seed(20260912)
    records=[]
    for length in power_of_two_buckets(args.min_seqlen,args.max_seqlen):
        sequences=max(1,min(args.sequences,args.token_budget//length))
        lengths=torch.full((sequences,),length,device=device,dtype=torch.int32)
        cu_seqlens=torch.cat((lengths.new_zeros(1),lengths.cumsum(0,dtype=torch.int32)))
        shape=(int(cu_seqlens[-1]),args.heads,args.head_dim)
        q=torch.rand(shape,device=device,dtype=torch.bfloat16,requires_grad=True)
        k=torch.rand(shape,device=device,dtype=torch.bfloat16,requires_grad=True)
        v=torch.randn(shape,device=device,dtype=torch.bfloat16,requires_grad=True)
        started=time.perf_counter()
        output=varlen_linear_attention(q,k,v,cu_seqlens,length)
        output.float().square().mean().backward()
        torch.cuda.synchronize(device)
        duration=time.perf_counter()-started
        record={"max_seqlen_bucket":length,"sequences":sequences,
                "tokens":shape[0],"duration_s":duration}
        records.append(record);print(json.dumps(record),flush=True)
        del q,k,v,output
    receipt={
        "format":"brainwm_flash_bla_warmup_v1",
        "gpu":torch.cuda.get_device_name(device),
        "torch":torch.__version__,"triton":triton.__version__,
        "triton_cache_dir":os.environ.get("TRITON_CACHE_DIR"),
        "precision":"bf16","combined_heads":args.heads,
        "head_dim":args.head_dim,"buckets":records,
    }
    output_path=args.output
    if output_path is not None:
        output_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=output_path.with_suffix(output_path.suffix+".tmp")
        temporary.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        temporary.replace(output_path)


if __name__=="__main__":main()
