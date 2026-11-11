#!/usr/bin/env python3
"""Profile one real BrainWM training shape with named module ranges."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_runtime_tmp = Path(os.environ.get("BRAINWM_TMPDIR", ROOT / "runtime/tmp"))
_runtime_tmp.mkdir(parents=True, exist_ok=True)
for _name in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_name] = str(_runtime_tmp)
tempfile.tempdir = str(_runtime_tmp)
os.environ.setdefault("TRITON_CACHE_DIR",str(ROOT/"triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(ROOT/"runtime/torchinductor"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR",str(ROOT/"runtime/torch_extensions"))
os.environ.setdefault("DS_BUILD_AIO","0")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True,exist_ok=True)

from brainwm.development_numpy_loader import DevelopmentPairLoader
from brainwm.foundation_moe import BrainWMFoundationMoE
from brainwm.backends import (
    backend_spec, create_backend, load_backend_entry_points, load_backend_plugins,
)
from brainwm.multitask_objective import (
    BRAINWM_LOSS_NAMES,
    MultiObjectiveBalancer,
    brainwm_task_losses,
    structured_modality_mask,
)
from brainwm.numpy_shard_loader import (
    NumpyShardDataset,
    bucketed_attention_token_cost,
)
from brainwm.profiling import enable_profile_ranges, profile_range
from brainwm.sparse_objective import mask_expression
from brainwm.training_observability import atomic_json, utc_now
from brainwm.optimizer_utils import GradientHealthMonitor


def build_dataset(args):
    return NumpyShardDataset(
        args.numpy_shards,
        split="train",
        batch_size=args.batch_size,
        zero_tokens_per_cell=args.zero_tokens_per_cell,
        max_padded_tokens=args.max_padded_tokens,
        start_row=args.rank_start_row,
        distributed=args.rank_local is not None,
        packed=True,
        packed_group_cells=args.packed_group_cells,
        shuffle=args.batch_regime == "production_head",
        seed=args.shuffle_seed,
        balance_waves=args.shard_balance_waves,
    )


def select_dense_location(dataset, candidate_shards, batches_per_shard):
    ranked=[]
    for path in dataset.files:
        stem=str(path)[:-len("-gene_index.npy")]
        tokens=len(np.load(path,mmap_mode="r"))
        rows=len(np.load(stem+"-row_ptr.npy",mmap_mode="r"))-1
        ranked.append((tokens/max(rows,1),path))
    best=None
    for _,path in sorted(ranked,reverse=True)[:candidate_shards]:
        stem=str(path)[:-len("-gene_index.npy")]
        ptr=np.load(stem+"-row_ptr.npy",mmap_mode="r")
        row_source=(np.load(stem+"-row_source_id.npy",mmap_mode="r")
                    if dataset.has_panel_catalog else None)
        cursor=0
        for _ in range(batches_per_shard):
            if cursor>=len(ptr)-1:break
            end=dataset._batch_end(ptr,row_source,cursor,len(ptr)-1)
            lengths=np.diff(ptr[cursor:end+1]).astype(np.int64)
            estimated_lengths=lengths+dataset.zero_tokens_per_cell
            slots=bucketed_attention_token_cost(
                estimated_lengths,dataset.packed_group_cells
            )
            candidate=(slots,int(lengths.sum()),end-cursor,path,cursor)
            if best is None or candidate[:3]>best[:3]:best=candidate
            cursor=end
    if best is None:raise RuntimeError("no dense profiling batch was found")
    slots,real_tokens,cells,path,start=best
    return path,start,{"estimated_attention_slots":slots,
                       "nonzero_tokens_before_zero_sampling":real_tokens,
                       "selected_cells":cells}


def describe_batch(batch,args,**metadata):
    lengths=(batch.row_ptr[1:]-batch.row_ptr[:-1]).numpy()
    maximum=int(lengths.max())
    return {
        **metadata,
        "cells":batch.batch_size,
        "base_attention_buckets":(
            batch.batch_size+args.packed_group_cells-1
        )//args.packed_group_cells,
        "real_tokens":int(batch.token_mask.sum()),
        "minimum_tokens_per_cell":int(lengths.min()),
        "mean_tokens_per_cell":float(lengths.mean()),
        "maximum_tokens_per_cell":maximum,
        "maximum_tokens_bucket":1 << (maximum-1).bit_length(),
        "attention_allocated_tokens":bucketed_attention_token_cost(
            lengths,args.packed_group_cells
        ),
    }


def load_batch(args):
    started=time.perf_counter()
    dataset=build_dataset(args)
    selection={"regime":args.batch_regime}
    if args.batch_regime == "dense":
        path,start,dense=select_dense_location(
            dataset,args.dense_candidate_shards,args.dense_batches_per_shard
        )
        dataset.files=[path];dataset.start_row=start;dataset.shuffle=False
        selection.update(dense)
        selection.update({"shard":path.name,"row_offset":start})
    batches=iter(dataset)
    batch=None
    for _ in range(args.rank_step_offset+1):
        batch=next(batches)
    selection=describe_batch(batch,args,**selection)
    selection.update({
        "batch_selection_and_load_seconds":time.perf_counter()-started,
    })
    return batch,selection


def load_batch_stream(args,count):
    if args.batch_regime != "production_head":
        raise ValueError("dynamic stream profiling requires production_head batches")
    dataset=build_dataset(args)
    batches=iter(dataset)
    for _ in range(args.rank_step_offset):
        next(batches)
    result=[]
    for index in range(count):
        started=time.perf_counter()
        batch=next(batches)
        result.append((batch,describe_batch(
            batch,args,regime="production_head",stream_index=index,
            batch_selection_and_load_seconds=time.perf_counter()-started,
        )))
    return result


def describe_development(source,target):
    def describe(batch):
        lengths=batch.row_ptr[1:]-batch.row_ptr[:-1]
        maximum=int(lengths.max())
        return {
            "cells":batch.batch_size,
            "tokens":int(batch.values.numel()),
            "maximum_tokens_per_cell":maximum,
            "maximum_tokens_bucket":1 << (maximum-1).bit_length(),
        }
    return {"source":describe(source),"target":describe(target)}


def event_record(event, steps):
    record = {
        "name":event.key,
        "calls":int(event.count),
        "calls_per_step":float(event.count/steps),
        "cpu_total_ms_per_step":float(event.cpu_time_total/1000/steps),
        "cpu_self_ms_per_step":float(event.self_cpu_time_total/1000/steps),
        "cuda_total_ms_per_step":float(event.device_time_total/1000/steps),
        "cuda_self_ms_per_step":float(event.self_device_time_total/1000/steps),
        "cpu_memory_bytes":int(event.cpu_memory_usage),
        "cuda_memory_bytes":int(event.device_memory_usage),
    }
    if getattr(event,"input_shapes",None):
        record["input_shapes"]=event.input_shapes
    return record


def markdown_table(rows,limit):
    lines=["| Range | CUDA total ms/step | CPU total ms/step | Calls/step |",
           "| --- | ---: | ---: | ---: |"]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['name']}` | {row['cuda_total_ms_per_step']:.4f} | "
            f"{row['cpu_total_ms_per_step']:.4f} | {row['calls_per_step']:.2f} |"
        )
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,
                        default=ROOT/"checkpoints/light-full-v2.pt")
    parser.add_argument("--numpy-shards",type=Path,
                        default=ROOT/"assets/v3_numpy_shards")
    parser.add_argument("--output-dir",type=Path,
                        default=ROOT/"results/profiling/light-full-v2")
    parser.add_argument("--batch-regime",choices=["production_head","dense"],
                        default="production_head")
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--max-padded-tokens",type=int,default=720000)
    parser.add_argument("--packed-group-cells",type=int,default=32)
    parser.add_argument("--packed-attention-backend",
                        choices=["auto","bucket","flash_bla"],default="auto")
    parser.add_argument("--expert-execution",choices=["grouped","batched","loop"],
                        default="batched")
    parser.add_argument("--router-execution",choices=["eager","compile","auto"],
                        default="eager",
                        help="fixed-shape gate projection execution policy")
    parser.add_argument("--router-min-tokens",type=int,default=256,
                        help="minimum cell rows before router auto compilation")
    parser.add_argument("--router-compile-mode",
                        choices=["default","reduce-overhead","max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        default="max-autotune-no-cudagraphs")
    parser.add_argument("--router-compile-dynamic",
                        action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--optimizer-backend", default="adamw",
                        help="registered optimizer used for the replay step")
    parser.add_argument("--backend-plugin", action="append", default=[],
                        help="import a Python module that registers extra backends")
    parser.add_argument("--load-backend-entry-points", action="store_true",
                        help="load installed brainwm.backends entry points")
    parser.add_argument("--attention-projection-execution",
                        choices=["fla_layernorm_qkv","fused_triplet","fused","separate"],
                        default="fused_triplet")
    parser.add_argument("--residual-norm-execution",
                        choices=["torch","torch_bf16","fla","auto"],
                        default="torch",
                        help="residual+LayerNorm backend used by each backbone block")
    parser.add_argument("--residual-norm-min-tokens",type=int,
                        default=131072,
                        help="token threshold for the automatic FLA residual norm selector")
    parser.add_argument("--feature-map-execution",
                        choices=["torch","compile","bucket","auto"],default="auto",
                        help="Flash-BLA feature map execution backend")
    parser.add_argument("--feature-map-min-tokens",type=int,default=131072,
                        help="minimum tokens before automatic dynamic feature-map compilation")
    parser.add_argument("--feature-map-compile-mode",
                        choices=["default","reduce-overhead","max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        default="max-autotune-no-cudagraphs",
                        help="TorchInductor mode for the compiled feature map")
    parser.add_argument("--feature-map-bucket-size",type=int,default=65536,
                        help="static feature-map token bucket granularity")
    parser.add_argument("--feature-map-max-padding-ratio",type=float,default=1.10,
                        help="maximum padding ratio for explicit static bucket mode")
    parser.add_argument("--differential-merge-execution",
                        choices=["torch","compile","auto"],default="auto",
                        help="positive-minus-correction merge execution backend")
    parser.add_argument("--differential-merge-min-tokens",type=int,default=131072,
                        help="minimum real tokens before automatic merge compilation")
    parser.add_argument("--differential-merge-compile-mode",
                        choices=["default","reduce-overhead","max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        default="max-autotune-no-cudagraphs")
    parser.add_argument("--population-balance-mode",
                        choices=["exact", "combined"], default="combined",
                        help="expert balance statistic for fused populations (production default: combined)")
    parser.add_argument("--flow-sdpa-backend",
                        choices=["auto","flash","efficient","cudnn","math"],
                        default="flash")
    parser.add_argument("--compile-flow",action=argparse.BooleanOptionalAction,
                        default=False,
                        help="compile only the Flow Matching Transformer")
    parser.add_argument("--compile-mode",
                        choices=["default","reduce-overhead","max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        default="reduce-overhead")
    parser.add_argument("--compile-dynamic",action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--compile-sinkhorn",action=argparse.BooleanOptionalAction,
                        default=False,
                        help="compile developmental Sinkhorn on fixed population buckets")
    parser.add_argument("--sinkhorn-compile-mode",
                        choices=["default","reduce-overhead","max-autotune",
                                 "max-autotune-no-cudagraphs"],
                        default="reduce-overhead")
    parser.add_argument("--sinkhorn-compile-buckets",type=int,nargs="+",
                        default=[8,16,32,64])
    parser.add_argument("--fuse-development-attention",
                        action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--fuse-development-target-attention",
                        action=argparse.BooleanOptionalAction,default=True,
                        help="join the inference-only target stream to the shared Flash-BLA call")
    parser.add_argument("--rank-local",type=int,default=None,
                        help="profile one rank's checkpoint cursor without initializing distributed communication")
    parser.add_argument("--rank-world-size",type=int,default=4)
    parser.add_argument("--rank-step-offset",type=int,default=0,
                        help="advance this many main and development batches from the saved rank cursor")
    parser.add_argument("--dynamic-stream",action="store_true",
                        help="profile consecutive real batches once each, including GPU input preparation")
    parser.add_argument("--adaptive-buckets",action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--zero-tokens-per-cell",type=int,default=16)
    parser.add_argument("--development-max-padded-tokens",type=int,default=60000)
    parser.add_argument("--shuffle-seed",type=int,default=20260912)
    parser.add_argument("--shard-balance-waves",type=int,default=8)
    parser.add_argument("--dense-candidate-shards",type=int,default=8)
    parser.add_argument("--dense-batches-per-shard",type=int,default=128)
    parser.add_argument("--warmup-steps",type=int,default=2)
    parser.add_argument("--baseline-steps",type=int,default=3)
    parser.add_argument("--profile-steps",type=int,default=3)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--record-shapes",action="store_true")
    parser.add_argument("--profile-memory",action="store_true")
    parser.add_argument("--with-stack",action="store_true",
                        help="aggregate Python stacks for CUDA stream synchronizations")
    parser.add_argument("--export-trace",action="store_true")
    parser.add_argument("--gradient-norm-interval",type=int,default=10,
                        help="sample full L2 gradient norms every N steps")
    parser.add_argument("--gradient-health-execution",
                        choices=["auto","foreach","amp_foreach"],default="auto",
                        help="finite-gradient check and fused-optimizer skip policy")
    args=parser.parse_args()
    plugin_names = list(args.backend_plugin)
    plugin_names.extend(
        value.strip() for value in os.environ.get("BRAINWM_BACKEND_PLUGINS", "").split(",")
        if value.strip()
    )
    args.backend_plugins_loaded = list(load_backend_plugins(plugin_names))
    args.backend_entry_points_loaded = []
    if args.load_backend_entry_points or os.environ.get("BRAINWM_LOAD_BACKEND_ENTRY_POINTS", "0") == "1":
        args.backend_entry_points_loaded = list(load_backend_entry_points())
    args.optimizer_backend = backend_spec("optimizer", args.optimizer_backend).name
    if not torch.cuda.is_available():raise SystemExit("CUDA is required")
    if min(args.warmup_steps,args.baseline_steps,args.profile_steps)<1:
        raise SystemExit("warmup, baseline, and profile steps must be positive")
    if args.feature_map_bucket_size < 1:
        raise SystemExit("--feature-map-bucket-size must be positive")
    if args.feature_map_max_padding_ratio < 1.0:
        raise SystemExit("--feature-map-max-padding-ratio must be at least 1")
    if args.differential_merge_min_tokens < 1:
        raise SystemExit("--differential-merge-min-tokens must be positive")
    if args.rank_step_offset < 0:
        raise SystemExit("--rank-step-offset must be nonnegative")
    args.rank_start_row=0
    args.development_start_step=0
    if args.rank_local is not None:
        if not 0 <= args.rank_local < args.rank_world_size:
            raise SystemExit("--rank-local must lie inside --rank-world-size")
        rank_payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        rows_by_rank=rank_payload.get("rows_by_rank")
        if rows_by_rank is None or len(rows_by_rank) != args.rank_world_size:
            raise SystemExit("checkpoint has no matching rank-local cursors")
        args.rank_start_row=int(rows_by_rank[args.rank_local])
        args.development_start_step=int(rank_payload.get("step",0))
        os.environ["RANK"]=str(args.rank_local)
        os.environ["WORLD_SIZE"]=str(args.rank_world_size)

    torch.manual_seed(args.shuffle_seed)
    torch.cuda.manual_seed(args.shuffle_seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device(args.device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    total_steps=args.warmup_steps+args.baseline_steps+args.profile_steps
    if args.dynamic_stream:
        loaded_batches=load_batch_stream(args,total_steps)
    else:
        batch,selection=load_batch(args)
        loaded_batches=[(batch,selection)]*total_steps
    development_loader=DevelopmentPairLoader(
        args.numpy_shards,args.development_max_padded_tokens,
        split="train",distributed=args.rank_local is not None,
        seed=args.shuffle_seed,packed=True,
        start_step=args.development_start_step,
    )
    development_batches=iter(development_loader)
    for _ in range(args.rank_step_offset):
        next(development_batches)
    if args.dynamic_stream:
        loaded_development=[next(development_batches) for _ in range(total_steps)]
    else:
        development=next(development_batches)
        loaded_development=[development]*total_steps
    stream_metadata=[]
    for (_,batch_selection),development in zip(
        loaded_batches,loaded_development,strict=True,
    ):
        source,target,_,_=development
        stream_metadata.append({
            "batch":batch_selection,
            "development":describe_development(source,target),
        })
    selection=stream_metadata[-1]["batch"]
    source,target,_,_=loaded_development[-1]

    # The production DataLoaders pin SparseTokenBatch recursively. The profiler
    # iterates datasets directly to select exact checkpoint batches, so mirror
    # that transport contract before entering any timed region.
    if device.type=="cuda":
        loaded_batches=[(batch.pin_memory(),metadata)
                        for batch,metadata in loaded_batches]
        loaded_development=[(source.pin_memory(),target.pin_memory(),t0,t1)
                            for source,target,t0,t1 in loaded_development]

    model,payload=BrainWMFoundationMoE.from_checkpoint(
        args.checkpoint,map_location="cpu"
    )
    moe_capabilities = backend_spec("moe", model.moe_backend).capabilities
    model.packed_group_cells=args.packed_group_cells
    model.packed_attention_backend=args.packed_attention_backend
    if moe_capabilities.get("expert_execution", False):
        model.set_expert_execution(args.expert_execution)
        model.set_router_execution(
            args.router_execution,
            min_tokens=args.router_min_tokens,
            mode=args.router_compile_mode,
            dynamic=args.router_compile_dynamic,
        )
    if moe_capabilities.get("attention_projection_execution", False):
        model.set_attention_projection_execution(args.attention_projection_execution)
    if moe_capabilities.get("residual_norm_execution", False):
        model.set_residual_norm_execution(args.residual_norm_execution)
        model.set_residual_norm_min_tokens(args.residual_norm_min_tokens)
    if moe_capabilities.get("feature_map_execution", False):
        model.set_feature_map_execution(args.feature_map_execution)
        model.set_feature_map_min_tokens(args.feature_map_min_tokens)
        model.set_feature_map_compile_mode(args.feature_map_compile_mode)
        model.set_feature_map_bucket_size(args.feature_map_bucket_size)
        model.set_feature_map_max_padding_ratio(args.feature_map_max_padding_ratio)
    if moe_capabilities.get("differential_merge_execution", False):
        model.set_differential_merge_execution(args.differential_merge_execution)
        model.set_differential_merge_min_tokens(
            args.differential_merge_min_tokens
        )
        model.set_differential_merge_compile_mode(
            args.differential_merge_compile_mode
        )
    if moe_capabilities.get("population_balance", True):
        model.set_population_balance_mode(args.population_balance_mode)
    flow_capabilities = backend_spec("flow", model.flow_backend).capabilities
    if flow_capabilities.get("sdpa_backend_selection", False):
        model.set_flow_sdpa_backend(args.flow_sdpa_backend)
    model.set_fuse_development_attention(args.fuse_development_attention)
    model.set_fuse_development_target_attention(
        args.fuse_development_target_attention
    )
    model.packed_attention_budget=(
        args.max_padded_tokens if args.adaptive_buckets else None
    )
    model=model.to(device).train()
    model.set_flow_compile(
        args.compile_flow,
        mode=args.compile_mode,
        dynamic=args.compile_dynamic,
    )
    model.set_sinkhorn_compile(
        args.compile_sinkhorn,
        mode=args.sinkhorn_compile_mode,
        buckets=tuple(args.sinkhorn_compile_buckets),
    )
    optimizer=create_backend(
        "optimizer", args.optimizer_backend, model.parameters(), lr=2e-4, fused=True,
    )
    if args.gradient_norm_interval < 1:
        raise SystemExit("--gradient-norm-interval must be positive")
    parameters=tuple(model.parameters())
    gradient_monitor=GradientHealthMonitor(
        parameters, optimizer, execution=args.gradient_health_execution,
    )
    if payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    balancer=MultiObjectiveBalancer(BRAINWM_LOSS_NAMES,mode="ema_equal").to(device)
    if payload.get("loss_balancer") is not None:
        balancer.load_state_dict(payload["loss_balancer"])

    last={}
    def prepare_step(batch,development):
        source,target,t0,t1=development
        non_blocking=device.type=="cuda"
        batch=mask_expression(batch.to(device,non_blocking=non_blocking))
        batch.modality_masked=structured_modality_mask(
            batch.modality_observed,task_probability=.5,
            all_context_probability=.15,context_probability=.15,
        )
        return batch,(source.to(device,non_blocking=non_blocking),
                      target.to(device,non_blocking=non_blocking),t0,t1)

    if not args.dynamic_stream:
        prepared=prepare_step(loaded_batches[0][0],loaded_development[0])
        step_inputs=[prepared]*total_steps
    else:
        step_inputs=[(batch,development) for (batch,_),development in zip(
            loaded_batches,loaded_development,strict=True,
        )]

    def training_step(step_input):
        if args.dynamic_stream:
            with profile_range("brainwm.step/input_prepare"):
                batch,development=prepare_step(*step_input)
        else:
            batch,development=step_input
        with profile_range("brainwm.step/zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            with profile_range("brainwm.step/forward"):
                output=model({"joint_batch":batch,"development":development})
            with profile_range("brainwm.step/task_losses"):
                losses,activity=brainwm_task_losses(
                    output,batch,return_activity=True,
                )
                velocity=torch.nn.functional.mse_loss(
                    output["flow_velocity"],output["flow_target"]
                )
                endpoint=torch.nn.functional.mse_loss(
                    output["flow_endpoint"],output["flow_endpoint_target"]
                )
                losses["flow"]=.5*(velocity+endpoint)
                activity["flow"]=losses["flow"].new_ones((),dtype=torch.bool)
            with profile_range("brainwm.step/loss_balance"):
                loss,_=balancer(
                    losses,materialize_telemetry=False,active=activity,
                )
        with profile_range("brainwm.step/backward"):
            loss.backward()
        gradient_norm_sampled = (
            gradient_monitor.last_norm is None
            or args.gradient_norm_interval == 1
            or (cursor + 1) % args.gradient_norm_interval == 0
        )
        with profile_range("brainwm.step/gradient_clip"):
            grad_norm, finite_grad = gradient_monitor.check(
                compute_norm=gradient_norm_sampled,
            )
        with profile_range("brainwm.step/optimizer"):
            finite_step=gradient_monitor.optimizer_step(
                optimizer, loss=loss, finite_gradients=finite_grad,
            )
        if not bool(finite_step):
            raise FloatingPointError("nonfinite loss or gradient in profile step")
        last.update(loss=loss.detach(),grad_norm=grad_norm.detach(),
                    gradient_norm_sampled=gradient_norm_sampled,
                    attention_allocated_tokens=output["attention_allocated_tokens"],
                    attention_bucket_count=output["attention_bucket_count"],
                    attention_backend=output.get("attention_backend","padded"),
                    development_attention_fused=bool(
                        output.get("development_attention_fused", False)
                    ),
                    development_attention_streams=int(
                        output.get("development_attention_streams", 1)
                    ),
                    development_target_attention_fused=bool(
                        output.get("development_target_attention_fused", False)
                    ),
                    feature_map_backend=output.get("feature_map_backend", "torch"),
                    differential_merge_backend=output.get(
                        "differential_merge_backend", "torch"
                    ),
                )

    cursor=0
    for _ in range(args.warmup_steps):
        training_step(step_inputs[cursor]);cursor+=1
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_step_ms=[]
    for _ in range(args.baseline_steps):
        start_event=torch.cuda.Event(enable_timing=True)
        end_event=torch.cuda.Event(enable_timing=True)
        start_event.record();training_step(step_inputs[cursor]);end_event.record()
        end_event.synchronize()
        baseline_step_ms.append(start_event.elapsed_time(end_event))
        cursor+=1
    baseline_cuda_step_ms=sum(baseline_step_ms)/len(baseline_step_ms)
    baseline_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30
    baseline_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30
    torch.cuda.reset_peak_memory_stats(device)
    profiled_step_ms=[]
    with profile(
        activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
        record_shapes=args.record_shapes,profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=args.with_stack,
        ),
        acc_events=True,
    ) as profiler:
        with enable_profile_ranges():
            for profile_index in range(args.profile_steps):
                start_event=torch.cuda.Event(enable_timing=True)
                end_event=torch.cuda.Event(enable_timing=True)
                start_event.record()
                range_name=(f"brainwm.dynamic_step/{profile_index:03d}"
                            if args.dynamic_stream else "brainwm.profile_step")
                with profile_range(range_name):
                    training_step(step_inputs[cursor])
                end_event.record()
                end_event.synchronize()
                profiled_step_ms.append(start_event.elapsed_time(end_event))
                cursor+=1
                profiler.step()
    cuda_step_ms=sum(profiled_step_ms)/len(profiled_step_ms)
    selection["attention_buckets"]=int(last["attention_bucket_count"])
    selection["attention_allocated_tokens"]=int(
        last["attention_allocated_tokens"]
    )
    selection["attention_backend"]=last["attention_backend"]
    effective_backend=selection["attention_backend"]

    averages=[event for event in profiler.key_averages()
              if event.device_type==torch.autograd.DeviceType.CPU]
    ranges=sorted(
        (event_record(event,args.profile_steps) for event in averages
         if event.key.startswith("brainwm.")),
        key=lambda row:row["cuda_total_ms_per_step"],reverse=True,
    )
    operators=sorted(
        (event_record(event,args.profile_steps) for event in averages
         if not event.key.startswith("brainwm.")),
        key=lambda row:row["cuda_self_ms_per_step"],reverse=True,
    )
    operator_shape_groups=[]
    if args.record_shapes:
        shape_averages=profiler.key_averages(group_by_input_shape=True)
        operator_shape_groups=sorted(
            (event_record(event,args.profile_steps) for event in shape_averages
             if (event.device_type==torch.autograd.DeviceType.CPU and
                 not event.key.startswith("brainwm."))),
            key=lambda row:row["cuda_self_ms_per_step"],reverse=True,
        )[:200]
    synchronization_sites=[]
    operator_stacks=[]
    if args.with_stack:
        site_counts=Counter(
            tuple(event.stack)
            for event in profiler.events()
            if event.key=="cudaStreamSynchronize"
        )
        synchronization_sites=[
            {"calls":count,"calls_per_step":count/args.profile_steps,
             "stack":list(stack)}
            for stack,count in site_counts.most_common()
        ]
        # Operator tables tell us which kernel is expensive, but not which
        # Python path created it.  Keep a small, bounded attribution sample
        # for the launch-heavy operators that dominate CSR training.  This is
        # diagnostic-only and never runs unless --with-stack is requested.
        traced_operator_names={
            "aten::_index_put_impl_", "aten::cat", "aten::copy_",
            "aten::to", "aten::_to_copy", "aten::mul",
        }
        for name in sorted(traced_operator_names):
            events=[event for event in profiler.events() if event.key==name]
            stacks=Counter(tuple(event.stack) for event in events if event.stack)
            if stacks:
                operator_stacks.append({
                    "name":name,
                    "calls":len(events),
                    "calls_per_step":len(events)/args.profile_steps,
                    "top_stacks":[
                        {"calls":count,"stack":list(stack)}
                        for stack,count in stacks.most_common(8)
                    ],
                })
    phases=[row for row in ranges if row["name"].startswith("brainwm.step/")]
    dynamic_ranges=sorted(
        (event_record(event,1) for event in averages
         if event.key.startswith("brainwm.dynamic_step/")),
        key=lambda row:row["name"],
    )
    warmup_end=args.warmup_steps
    baseline_end=warmup_end+args.baseline_steps
    baseline_metadata=stream_metadata[warmup_end:baseline_end]
    profiled_metadata=stream_metadata[baseline_end:]
    baseline_stream=[{**metadata,"cuda_event_ms":milliseconds}
                     for metadata,milliseconds in zip(
                         baseline_metadata,baseline_step_ms,strict=True,
                     )]
    profiled_stream=[{**metadata,"cuda_event_ms":milliseconds}
                     for metadata,milliseconds in zip(
                         profiled_metadata,profiled_step_ms,strict=True,
                     )]
    baseline_cells=sum(item["batch"]["cells"] for item in baseline_metadata)
    baseline_tokens=sum(item["batch"]["real_tokens"] for item in baseline_metadata)
    profiled_cells=sum(item["batch"]["cells"] for item in profiled_metadata)
    profiled_tokens=sum(item["batch"]["real_tokens"] for item in profiled_metadata)
    report={
        "format":"brainwm_training_step_profile_v3",
        "timestamp_utc":utc_now(),
        "checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_step":int(payload.get("step",-1)),
        "checkpoint_model_config":payload.get("model_config"),
        "backends":(model.backend_manifest()
                     if hasattr(model,"backend_manifest") else None),
        "optimizer_backend":args.optimizer_backend,
        "backend_plugins_loaded":args.backend_plugins_loaded,
        "backend_entry_points_loaded":args.backend_entry_points_loaded,
        "configured_batch_size":args.batch_size,
        "configured_max_padded_tokens":args.max_padded_tokens,
        "effective_packed_group_cells":args.packed_group_cells,
        "packed_attention_backend":effective_backend,
        "expert_execution":args.expert_execution,
        "router_execution":args.router_execution,
        "router_min_tokens":args.router_min_tokens,
        "router_compile_mode":args.router_compile_mode,
        "router_compile_dynamic":args.router_compile_dynamic,
        "router_runtime":(
            model.router_runtime_status()
            if hasattr(model,"router_runtime_status") else None
        ),
        "attention_projection_execution":args.attention_projection_execution,
        "residual_norm_execution":args.residual_norm_execution,
        "residual_norm_min_tokens":args.residual_norm_min_tokens,
        "feature_map_execution":args.feature_map_execution,
        "feature_map_min_tokens":args.feature_map_min_tokens,
        "feature_map_compile_mode":args.feature_map_compile_mode,
        "feature_map_bucket_size":args.feature_map_bucket_size,
        "feature_map_max_padding_ratio":args.feature_map_max_padding_ratio,
        "differential_merge_execution":args.differential_merge_execution,
        "differential_merge_min_tokens":args.differential_merge_min_tokens,
        "differential_merge_compile_mode":args.differential_merge_compile_mode,
        "differential_merge_backend":last["differential_merge_backend"],
        "population_balance_mode":args.population_balance_mode,
        "flow_sdpa_backend":args.flow_sdpa_backend,
        "compile_flow":args.compile_flow,
        "compile_mode":args.compile_mode,
        "compile_dynamic":args.compile_dynamic,
        "compile_sinkhorn":args.compile_sinkhorn,
        "sinkhorn_compile_mode":args.sinkhorn_compile_mode,
        "sinkhorn_compile_buckets":list(args.sinkhorn_compile_buckets),
        "sinkhorn_runtime":model.sinkhorn_runtime_status(),
        "gradient_norm_interval":args.gradient_norm_interval,
        "gradient_health_execution":args.gradient_health_execution,
        "gradient_health_backend":gradient_monitor.execution,
        "gradient_norm_sampled":bool(last["gradient_norm_sampled"]),
        "requested_fuse_development_attention":args.fuse_development_attention,
        "requested_fuse_development_target_attention":args.fuse_development_target_attention,
        "fuse_development_attention":last["development_attention_fused"],
        "fuse_development_target_attention":last["development_target_attention_fused"],
        "feature_map_backend":last["feature_map_backend"],
        "development_attention_streams":last["development_attention_streams"],
        "rank_local":args.rank_local,
        "rank_world_size":args.rank_world_size if args.rank_local is not None else None,
        "rank_start_row":args.rank_start_row,
        "rank_step_offset":args.rank_step_offset,
        "dynamic_stream":args.dynamic_stream,
        "adaptive_buckets":args.adaptive_buckets,
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES",""),
        "gpu":torch.cuda.get_device_name(device),
        "torch":torch.__version__,
        "precision":"bf16",
        "batch":selection,
        "development":{"source_cells":source.batch_size,
                       "target_cells":target.batch_size,
                       "source_attention_buckets":(0 if effective_backend=="flash_bla" else
                           (source.batch_size+args.packed_group_cells-1)//args.packed_group_cells),
                       "target_attention_buckets":(0 if effective_backend=="flash_bla" else
                           (target.batch_size+args.packed_group_cells-1)//args.packed_group_cells)},
        "warmup_steps":args.warmup_steps,
        "baseline_steps":args.baseline_steps,
        "baseline_step_ms":baseline_step_ms,
        "baseline_cuda_event_ms_per_step":baseline_cuda_step_ms,
        "baseline_peak_allocated_gib":baseline_peak_allocated_gib,
        "baseline_peak_reserved_gib":baseline_peak_reserved_gib,
        "profile_steps":args.profile_steps,
        "profiled_step_ms":profiled_step_ms,
        "baseline_stream":baseline_stream,
        "profiled_stream":profiled_stream,
        "dynamic_step_ranges":dynamic_ranges,
        "cuda_event_ms_per_step":cuda_step_ms,
        "profiler_slowdown_ratio":cuda_step_ms/baseline_cuda_step_ms,
        "main_cells_per_second":1000*baseline_cells/sum(baseline_step_ms),
        "main_tokens_per_second":1000*baseline_tokens/sum(baseline_step_ms),
        "profiled_main_cells_per_second":1000*profiled_cells/sum(profiled_step_ms),
        "profiled_main_tokens_per_second":1000*profiled_tokens/sum(profiled_step_ms),
        "peak_allocated_gib":torch.cuda.max_memory_allocated(device)/2**30,
        "peak_reserved_gib":torch.cuda.max_memory_reserved(device)/2**30,
        "final_loss":float(last["loss"]),
        "final_gradient_norm":float(last["grad_norm"]),
        "phases":phases,
        "module_ranges":ranges,
        "operators":operators[:100],
        "operator_shape_groups":operator_shape_groups,
        "synchronization_sites":synchronization_sites,
        "operator_stacks":operator_stacks,
    }
    bucket_policy="adaptive" if args.adaptive_buckets else "fixed"
    execution_policy=(effective_backend if effective_backend!="bucket" else
                      f"bucket-{bucket_policy}")
    stream_suffix="-dynamic" if args.dynamic_stream else ""
    rank_suffix=("" if args.rank_local is None else
                 f"-rank-{args.rank_local:02d}-offset-{args.rank_step_offset:02d}")
    stem=(f"{args.batch_regime}-bs-{args.batch_size}-{execution_policy}-"
          f"{args.expert_execution}-experts-"
          f"{args.attention_projection_execution}-qkv-"
          f"norm-{args.residual_norm_execution}-{args.residual_norm_min_tokens}-"
          f"feature-{args.feature_map_execution}-{args.feature_map_min_tokens}-"
          f"{args.feature_map_compile_mode}-"
          f"merge-{args.differential_merge_execution}-"
          f"balance-{args.population_balance_mode}-"
          f"flow-{args.flow_sdpa_backend}-"
          f"sinkhorn-{'compiled' if args.compile_sinkhorn else 'eager'}-"
          f"joint-attn-{'on' if report['fuse_development_attention'] else 'off'}-"
          f"target-attn-{'on' if report['fuse_development_target_attention'] else 'off'}-"
          f"step-{report['checkpoint_step']:08d}{stream_suffix}{rank_suffix}")
    json_path=args.output_dir/f"{stem}.json"
    trace_path=args.output_dir/f"{stem}.trace.json"
    table_path=args.output_dir/f"{stem}.operators.txt"
    markdown_path=args.output_dir/f"{stem}.md"
    atomic_json(json_path,report)
    if args.export_trace:profiler.export_chrome_trace(str(trace_path))
    table_path.write_text(
        profiler.key_averages().table(
            sort_by="self_device_time_total",row_limit=100
        )+"\n",encoding="utf-8",
    )
    markdown_path.write_text(
        "# BrainWM Training-Step Profile\n\n"
        f"Checkpoint step: {report['checkpoint_step']}  \n"
        f"Registered backends: `{report['backends']}`  \n"
        f"Optimizer backend: `{report['optimizer_backend']}`  \n"
        f"Batch regime: `{args.batch_regime}`  \n"
        f"Packed attention backend: `{effective_backend}`  \n"
        f"Expert execution: `{args.expert_execution}`  \n"
        f"Router execution: `{args.router_execution}` (min cells `{args.router_min_tokens}`)  \n"
        f"Router runtime: `{report['router_runtime']}`  \n"
        f"Attention projection execution: `{args.attention_projection_execution}`  \n"
        f"Residual norm execution: `{args.residual_norm_execution}`  \n"
        f"Residual norm minimum tokens: `{args.residual_norm_min_tokens}`  \n"
        f"Feature-map execution: `{args.feature_map_execution}`  \n"
        f"Feature-map minimum tokens: `{args.feature_map_min_tokens}`  \n"
        f"Feature-map compile mode: `{args.feature_map_compile_mode}`  \n"
        f"Feature-map bucket size: `{args.feature_map_bucket_size}`; max padding ratio: `{args.feature_map_max_padding_ratio}`  \n"
        f"Differential-merge execution: `{args.differential_merge_execution}`; runtime: `{report['differential_merge_backend']}`  \n"
        f"Differential-merge minimum tokens: `{args.differential_merge_min_tokens}`; compile mode: `{args.differential_merge_compile_mode}`  \n"
        f"Gradient health: `{args.gradient_health_execution}` (runtime `{gradient_monitor.execution}`)  \n"
        f"Population balance mode: `{args.population_balance_mode}`  \n"
        f"Flow SDPA backend: `{args.flow_sdpa_backend}`  \n"
        f"Fused main/source attention: `{report['fuse_development_attention']}`  \n"
        f"Fused developmental target attention: `{report['fuse_development_target_attention']}`  \n"
        f"Development attention streams: `{report['development_attention_streams']}`  \n"
        f"Rank-local replay: `{args.rank_local}` at offset `{args.rank_step_offset}`  \n"
        f"Dynamic stream: `{args.dynamic_stream}`  \n"
        f"Main batch: {selection['cells']} cells, {selection['real_tokens']} real tokens  \n"
        f"Attention buckets per backbone block: "
        f"{selection['attention_buckets']} main + "
        f"{report['development']['source_attention_buckets']} source + "
        f"{report['development']['target_attention_buckets']} target  \n"
        f"Unprofiled CUDA-event latency: {baseline_cuda_step_ms:.3f} ms/step  \n"
        f"Profiled CUDA-event latency: {cuda_step_ms:.3f} ms/step "
        f"({report['profiler_slowdown_ratio']:.2f}x profiler slowdown)  \n"
        f"Peak allocated: {report['peak_allocated_gib']:.3f} GiB  \n\n"
        "Named CUDA times are measured under the profiler and are inclusive, "
        "so nested rows must not be summed. Use the unprofiled latency for "
        "production throughput.\n\n"
        "## Training Phases\n\n"+markdown_table(phases,30)+"\n\n"
        "## Module Ranges\n\n"+markdown_table(ranges,80)+"\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report":str(json_path),
        "trace":str(trace_path) if args.export_trace else None,
        "operator_table":str(table_path),"markdown":str(markdown_path),
        "checkpoint_step":report["checkpoint_step"],
        "batch":selection,"cuda_event_ms_per_step":cuda_step_ms,
        "baseline_cuda_event_ms_per_step":baseline_cuda_step_ms,
        "profiler_slowdown_ratio":report["profiler_slowdown_ratio"],
        "peak_allocated_gib":report["peak_allocated_gib"],
    },indent=2))


if __name__=="__main__":main()
