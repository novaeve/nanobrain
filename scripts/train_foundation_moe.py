#!/usr/bin/env python3
"""nanoGPT-style full-corpus trainer for the unified BrainWM MoE backbone.

The dataset adapter is deliberately a small protocol: an iterable yielding
BrainWMInput objects. Existing H5AD readers can be wrapped without changing the
model or training loop.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import time
import sys

# Keep Python multiprocessing, torchrun rendezvous files, and compiler
# artifacts on the local ESSD workspace even when the parent shell inherited a
# NAS-backed TMPDIR. BRAINWM_TMPDIR remains an explicit override.
_runtime_root = Path(__file__).resolve().parents[1]
_runtime_tmp = Path(os.environ.get("BRAINWM_TMPDIR", _runtime_root / "runtime/tmp"))
_runtime_tmp.mkdir(parents=True, exist_ok=True)
for _name in ("TMPDIR", "TMP", "TEMP"):
    os.environ[_name] = str(_runtime_tmp)
tempfile.tempdir = str(_runtime_tmp)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_runtime_root / "runtime/torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_runtime_root / "triton"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_runtime_root / "runtime/torch_extensions"))
os.environ.setdefault("DS_BUILD_AIO", "0")
Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(_runtime_root / 'src'))
from brainwm_foundation import BrainWMFoundationMoE, BrainWMInput, parameter_count
from brainwm_foundation.backends import (
    backend_spec, create_backend, load_backend_entry_points, load_backend_plugins,
)
from brainwm.human_data_adapter import iter_human_batches, iter_human_sparse_batches
from brainwm.numpy_shard_loader import (representative_shard_loader, shard_loader,
    rank_offsets_for_global_prefix)
from brainwm.sparse_objective import mask_expression, masked_expression_loss
from brainwm.multitask_objective import (BRAINWM_LOSS_NAMES, MultiObjectiveBalancer,
    brainwm_task_losses, structured_modality_mask)
from brainwm.development_numpy_loader import (
    DevelopmentPairLoader,prefetched_development_loader,
)
from brainwm.local_assets import gene_vocabulary_path
from brainwm.training_observability import (ValidationAccumulator, append_jsonl,
    atomic_json, atomic_torch_save, finite_float, runtime_record, utc_now)
from brainwm.cuda_prefetch import CUDAPrefetcher
from brainwm.optimizer_utils import GradientHealthMonitor


def learning_rate_at_progress(base, minimum, progress, warmup_fraction):
    """Linear warmup followed by cosine decay over complete-corpus progress."""
    base=float(base);minimum=float(minimum);progress=min(max(float(progress),0.0),1.0)
    warmup_fraction=float(warmup_fraction)
    if not 0.0<=minimum<=base:raise ValueError('minimum learning rate must lie in [0, base]')
    if not 0.0<=warmup_fraction<1.0:raise ValueError('warmup fraction must lie in [0, 1)')
    if warmup_fraction and progress<warmup_fraction:
        return base*(0.01+0.99*progress/warmup_fraction)
    decay_progress=((progress-warmup_fraction)/(1.0-warmup_fraction)
                    if warmup_fraction<1.0 else 1.0)
    return minimum+0.5*(base-minimum)*(1.0+math.cos(math.pi*decay_progress))


def load_class_weights(path, metadata_vocab, device):
    """Bind frozen label-balance weights to model class indices."""
    import pandas as pd
    table=pd.read_csv(path)
    result={}
    for task,dimension,vocab_name in (('cell','cl_id','cell_type'),
                                      ('anatomy','uberon_id','anatomy')):
        lookup=dict(zip(table.loc[table.dimension.eq(dimension),'label'].astype(str),
                        table.loc[table.dimension.eq(dimension),'suggested_loss_weight'].astype(float)))
        weights=torch.ones(len(metadata_vocab[vocab_name]),dtype=torch.float32,device=device)
        for label,index in metadata_vocab[vocab_name].items():
            if str(label) in lookup:weights[int(index)]=float(lookup[str(label)])
        result[task]=weights
    return result


def train(model, batch_factory, optimizer, *, train_rows: int, full_passes: int,
          device='cuda', checkpoint: Path | None = None, resume: Path | None = None,
          state_path: Path | None = None, max_steps: int | None = None,
          development_factory=None, precision: str = 'bf16', metrics_dir: Path | None = None,
          validation_factory=None, validation_every: int = 1000,
          validation_rows: int = 8192, checkpoint_every: int = 25,
          version_checkpoint_every: int = 1000, run_config: dict | None = None,
          resume_rank_offsets=None, console_every: int = 10,
          class_weights=None, task_mask_probability: float = 0.5,
          all_context_mask_probability: float = 0.15,
          context_mask_probability: float = 0.15,
          base_learning_rate: float = 2e-4,
          minimum_learning_rate: float = 2e-5,
          warmup_fraction: float = 0.01,
          validation_development_factory=None, training_seed: int = 20260912,
          validation_seed: int = 20260912,
          optimizer_backend: str = 'adamw',
          cuda_phase_timing: bool = False,
          cuda_prefetch: bool | None = None,
          cuda_timing_interval: int = 0,
          telemetry_interval: int = 1,
          gradient_norm_interval: int = 10,
          gradient_health_execution: str = 'auto'):
    model.to(device)
    raw_model=model.module if hasattr(model,'module') else model
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    start_pass = 0;start_rows = 0;start_global_rows=0
    if resume:
        state = torch.load(resume, map_location=device)
        previous_config=state.get('run_config') or {}
        execution_only_keys={
            'workers','packed_attention_backend','expert_execution',
            'router_execution','router_min_tokens','router_compile_mode',
            'router_compile_dynamic',
            'attention_projection_execution','flow_sdpa_backend',
            'fuse_development_attention','fuse_development_target_attention',
            'cuda_prefetch','cuda_timing_interval','fuse_nbeats_heads',
            'residual_norm_execution','residual_norm_min_tokens',
            'feature_map_execution','feature_map_min_tokens',
            'feature_map_compile_mode','feature_map_bucket_size',
            'feature_map_max_padding_ratio',
            'differential_merge_execution','differential_merge_min_tokens',
            'differential_merge_compile_mode',
            'telemetry_interval','compile_flow','compile_mode','compile_dynamic',
            'compile_sinkhorn','sinkhorn_compile_mode','sinkhorn_compile_buckets',
            'gradient_norm_interval','gradient_health_execution',
            'ddp_find_unused_parameters','ddp_static_graph','ddp_bucket_cap_mb',
        }
        for key in ('seed','shuffle_training','shuffle_seed','shard_order',
                    'shard_balance_waves','workers','packed_tokens',
                    'packed_group_cells','packed_attention_backend',
                    'expert_execution','attention_projection_execution',
                    'router_execution','router_min_tokens','router_compile_mode',
                    'router_compile_dynamic',
                    'flow_sdpa_backend','fuse_development_attention',
                    'fuse_development_target_attention',
                    'fuse_nbeats_heads','residual_norm_execution',
                    'residual_norm_min_tokens',
                    'feature_map_execution','feature_map_min_tokens',
                    'feature_map_compile_mode','feature_map_bucket_size',
                    'feature_map_max_padding_ratio',
                    'differential_merge_execution','differential_merge_min_tokens',
                    'differential_merge_compile_mode',
                    'population_balance_mode',
                    'optimizer_backend', 'attention_backend',
                    'expert_attention_backend', 'flow_backend',
                    'compile_backend',
                    'cuda_prefetch','cuda_timing_interval',
                    'telemetry_interval','compile_flow','compile_mode','compile_dynamic',
                    'compile_sinkhorn','sinkhorn_compile_mode','sinkhorn_compile_buckets',
                    'gradient_norm_interval','gradient_health_execution',
                    'ddp_static_graph','ddp_bucket_cap_mb',
                    'batch_size','max_padded_tokens',
                    'task_mask_probability','all_context_mask_probability',
                    'context_mask_probability','ddp_find_unused_parameters'):
            if key not in previous_config or run_config is None:
                continue
            previous_value=previous_config[key];current_value=run_config.get(key)
            # These backends preserve the model/optimizer parameter contract.
            # Their numerical and worker-order equivalence is covered by the
            # benchmark regression suite and each change is logged at resume.
            if key in execution_only_keys:
                continue
            if (key=='batch_size' and
                    bool(run_config.get('allow_resume_batch_size_change'))):
                continue
            if previous_value!=current_value:
                raise ValueError(f'resume configuration changed {key}: {previous_config[key]!r} != {run_config.get(key)!r}')
        raw_model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start_pass = state.get('full_pass', 0)
        saved_offsets=state.get('rows_by_rank')
        if world>1:
            offsets=saved_offsets if saved_offsets is not None and len(saved_offsets)==world else resume_rank_offsets
            if offsets is None:raise ValueError('distributed resume requires rank-local row offsets')
            start_rows=int(offsets[rank]);start_global_rows=sum(map(int,offsets))
        else:
            start_rows = state.get('rows_in_pass', 0);start_global_rows=start_rows
    model.train()
    balancer=MultiObjectiveBalancer(BRAINWM_LOSS_NAMES,mode='ema_equal').to(device)
    if resume and state.get('loss_balancer') is not None:
        balancer.load_state_dict(state['loss_balancer'])
    prefetch_enabled=(
        device.startswith('cuda') if cuda_prefetch is None else bool(cuda_prefetch)
    )
    if cuda_timing_interval < 0:
        raise ValueError('cuda_timing_interval must be nonnegative')
    if telemetry_interval < 0:
        raise ValueError('telemetry_interval must be nonnegative')
    if gradient_norm_interval < 1:
        raise ValueError('gradient_norm_interval must be positive')
    development_iter=(
        CUDAPrefetcher(development_factory(),device,enabled=prefetch_enabled)
        if development_factory else None
    )
    completed_steps = int(state.get('step', 0)) if resume else 0
    parameters = tuple(model.parameters())
    gradient_monitor = GradientHealthMonitor(
        parameters, optimizer, execution=gradient_health_execution,
    )
    train_log=metrics_dir/'train_steps.jsonl' if metrics_dir else None
    validation_log=metrics_dir/'validation.jsonl' if metrics_dir else None
    checkpoint_index=metrics_dir/'checkpoint_index.jsonl' if metrics_dir else None
    best_pointer=metrics_dir/'best_checkpoint.json' if metrics_dir else None
    last_validation=None
    phase_names=(
        'input_prepare','model_forward','task_losses','loss_balance',
        'backward','gradient_clip','optimizer',
    )
    allocator_counter_names=(
        'allocation.all.allocated','allocation.all.freed',
        'segment.all.allocated','segment.all.freed',
        'num_alloc_retries','num_sync_all_streams',
    )
    allocator_gauge_names=(
        'allocation.all.current','segment.all.current',
        'active_bytes.all.current','reserved_bytes.all.current',
    )
    previous_allocator_counters=None
    if cuda_phase_timing and device.startswith('cuda'):
        memory_stats=torch.cuda.memory_stats(device)
        previous_allocator_counters={
            name:int(memory_stats.get(name,0)) for name in allocator_counter_names
        }

    def gather_row_offsets(rows_seen):
        if world==1:return [int(rows_seen)]
        value=torch.tensor(int(rows_seen),device=device,dtype=torch.long)
        gathered=[torch.zeros_like(value) for _ in range(world)];dist.all_gather(gathered,value)
        return [int(item) for item in gathered]

    def save_checkpoint(path, full_pass, rows_seen, rows_by_rank, event='checkpoint'):
        payload={'step':completed_steps,'full_pass':full_pass,'rows_in_pass':sum(rows_by_rank),
                 'rows_by_rank':rows_by_rank,'world_size':world,
                 'model_config':raw_model.model_config,'model':raw_model.state_dict(),
                 'backends':(raw_model.backend_manifest()
                             if hasattr(raw_model,'backend_manifest') else None),
                 'optimizer':optimizer.state_dict(),'optimizer_backend':optimizer_backend,
                 'loss_balancer':balancer.state_dict(),
                 'last_validation':last_validation,'run_config':run_config or {}}
        atomic_torch_save(path,payload)
        if checkpoint_index:
            record={'event':event,'timestamp_utc':utc_now(),'step':completed_steps,
                    'full_pass':full_pass,'rows_in_pass':sum(rows_by_rank),
                    'rows_by_rank':rows_by_rank,'world_size':world,'path':str(path),
                    'validation':last_validation,'size_bytes':path.stat().st_size}
            if hasattr(raw_model, 'backend_manifest'):
                record['backends'] = raw_model.backend_manifest()
            append_jsonl(checkpoint_index,record,durable=True)
            manifest=path.parent/f'{path.stem}.manifest.json'
            atomic_json(manifest,record)
            if event=='version_checkpoint' and last_validation is not None:
                score=last_validation['metrics'].get('loss/mean_active_objectives')
                previous=json.loads(best_pointer.read_text()).get('score') if best_pointer.exists() else None
                if score is not None and (previous is None or score<previous):
                    atomic_json(best_pointer,{'step':completed_steps,'score':score,
                        'metric':'loss/mean_active_objectives','direction':'minimize',
                        'checkpoint':str(path),'checkpoint_manifest':str(manifest),
                        'updated_at_utc':utc_now()})

    def validate_now(full_pass, rows_seen):
        nonlocal last_validation
        if validation_factory is None:return None
        if world>1:dist.barrier()
        if rank!=0:
            dist.barrier();return None
        rng_cpu=torch.random.get_rng_state(); rng_cuda=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None
        torch.random.default_generator.manual_seed(int(validation_seed))
        if device.startswith('cuda'):torch.cuda.manual_seed(int(validation_seed))
        eval_model=raw_model if world>1 else model
        eval_model.eval(); accumulator=ValidationAccumulator(); started=time.perf_counter()
        try:
          with torch.no_grad():
            for batch in validation_factory():
                batch=batch.to(device,non_blocking=device.startswith('cuda'))
                batch=mask_expression(batch)
                batch.modality_masked={name:observed.bool() for name,observed in (batch.modality_observed or {}).items()}
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda') and precision=='bf16'):
                    out=eval_model({'joint_batch':batch,'development':None}); losses=brainwm_task_losses(out,batch)
                accumulator.add(batch,out,losses)
                if accumulator.rows>=validation_rows:break
            if validation_development_factory is not None:
                for source,target,t0,t1 in validation_development_factory():
                    source,target=(
                        source.to(device,non_blocking=device.startswith('cuda')),
                        target.to(device,non_blocking=device.startswith('cuda')),
                    )
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,
                                        enabled=device.startswith('cuda') and precision=='bf16'):
                        flow_out=eval_model({'joint_batch':source,
                                             'development':(source,target,t0,t1)})
                        velocity=torch.nn.functional.mse_loss(
                            flow_out['flow_velocity'],flow_out['flow_target'])
                        endpoint=torch.nn.functional.mse_loss(
                            flow_out['flow_endpoint'],flow_out['flow_endpoint_target'])
                    accumulator.add_development(source.batch_size,velocity,endpoint)
        finally:
            eval_model.train();torch.random.set_rng_state(rng_cpu)
            if rng_cuda is not None:torch.cuda.set_rng_state(rng_cuda,device)
        last_validation={'event':'validation','timestamp_utc':utc_now(),'step':completed_steps,
                         'full_pass':full_pass,'train_rows_in_pass':rows_seen,
                         'validation_rows_requested':validation_rows,'duration_s':time.perf_counter()-started,
                         'metrics':accumulator.metrics()}
        append_jsonl(validation_log,last_validation,durable=True)
        if world>1:dist.barrier()
        return last_validation
    for full_pass in range(start_pass, full_passes):
      rows_seen = start_rows if full_pass == start_pass else 0
      global_rows_seen=start_global_rows if full_pass==start_pass else 0
      batches = CUDAPrefetcher(
          batch_factory(start_rows if full_pass == start_pass else 0,full_pass),
          device, enabled=prefetch_enabled,
      )
      last_batch=None
      while True:
        load_started=time.perf_counter()
        try: batch=next(batches);active=True;last_batch=batch
        except StopIteration:
            if world==1:break
            active=False;batch=last_batch
        if world>1:
            active_count=torch.tensor(int(active),device=device,dtype=torch.long);dist.all_reduce(active_count)
            if int(active_count)==0:break
            if batch is None:raise RuntimeError('rank has no batch available for synchronized DDP tail')
        else:active_count=torch.tensor(1,device=device)
        data_seconds=time.perf_counter()-load_started
        step_started=time.perf_counter()
        phase_events=None
        if cuda_phase_timing and device.startswith('cuda'):
            phase_events={'start':torch.cuda.Event(enable_timing=True)}
            phase_events['start'].record()
        def mark_phase(name):
            if phase_events is not None:
                phase_events[name]=torch.cuda.Event(enable_timing=True)
                phase_events[name].record()
        cuda_step_start = None
        cuda_step_end = None
        timing_sampled = bool(
            device.startswith('cuda') and (
                cuda_phase_timing or (
                    cuda_timing_interval > 0
                    and (completed_steps + 1) % cuda_timing_interval == 0
                )
            )
        )
        if timing_sampled:
            cuda_step_start = torch.cuda.Event(enable_timing=True)
            cuda_step_start.record()
        # Cross-rank telemetry is diagnostic data.  Gathering it every step
        # forces a device-to-host copy and a collective into the critical path;
        # the per-step loss log remains intact while rank detail is sampled.
        next_step = completed_steps + 1
        telemetry_sampled = bool(
            world == 1 or telemetry_interval == 1 or
            (telemetry_interval > 0 and next_step % telemetry_interval == 0)
        )
        schedule_progress=(full_pass+global_rows_seen/train_rows)/full_passes
        learning_rate=learning_rate_at_progress(base_learning_rate,minimum_learning_rate,
                                                schedule_progress,warmup_fraction)
        for group in optimizer.param_groups:group['lr']=learning_rate
        physical_rows=int(batch.batch_size) if hasattr(batch,'gene_index') else int(batch.expression.shape[0])
        rows=physical_rows if active else 0
        if active and (rows <= 0 or rows_seen + rows > train_rows):
            raise ValueError(f'full-pass coverage violation: {rows_seen}+{rows}>{train_rows}')
        development=None
        if hasattr(batch, 'gene_index'):
            step_seed=int(training_seed)+completed_steps*world+rank
            torch.random.default_generator.manual_seed(step_seed)
            if device.startswith('cuda'):torch.cuda.manual_seed(step_seed)
            if batch.gene_index.device != torch.device(device):
                batch = batch.to(device,non_blocking=device.startswith('cuda'))
            batch = mask_expression(batch)
            batch.modality_masked=structured_modality_mask(
                batch.modality_observed,
                task_probability=task_mask_probability,
                all_context_probability=all_context_mask_probability,
                context_probability=context_mask_probability,
            )
            if development_factory:
                try:source,target,t0,t1=next(development_iter)
                except StopIteration:
                    development_iter=CUDAPrefetcher(
                        development_factory(),device,enabled=prefetch_enabled,
                    )
                    source,target,t0,t1=next(development_iter)
                if source.gene_index.device != torch.device(device):
                    source,target=(
                        source.to(device,non_blocking=device.startswith('cuda')),
                        target.to(device,non_blocking=device.startswith('cuda')),
                    )
                development=(source,target,t0,t1)
            mark_phase('input_prepare')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device.startswith('cuda') and precision == 'bf16'):
                out=model({'joint_batch':batch,'development':development})
                mark_phase('model_forward')
                task_losses,task_activity=brainwm_task_losses(
                    out,batch,class_weights=class_weights,return_activity=True,
                )
                if development is not None:
                    velocity=torch.nn.functional.mse_loss(out['flow_velocity'],out['flow_target'])
                    endpoint=torch.nn.functional.mse_loss(out['flow_endpoint'],out['flow_endpoint_target'])
                    task_losses['flow']=0.5*(velocity+endpoint)
                    task_activity['flow']=task_losses['flow'].new_ones(
                        (),dtype=torch.bool,
                    )
                mark_phase('task_losses')
                loss,loss_info=balancer(
                    task_losses,materialize_telemetry=False,
                    active=task_activity,
                )
                mark_phase('loss_balance')
                if world>1:loss=loss*(world/float(active_count) if active else 0.0)
        else:
            def move(value):
                if isinstance(value, torch.Tensor): return value.to(device)
                if isinstance(value, dict): return {k:move(v) for k,v in value.items()}
                return value
            batch = BrainWMInput(**{k:move(v) for k,v in vars(batch).items()})
            mark_phase('input_prepare')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device.startswith('cuda') and precision == 'bf16'):
                out = model(batch)
                mark_phase('model_forward')
                valid = batch.expression_observed.bool()
                prediction = out['expression_reconstruction'].reshape(-1)
                target = batch.expression.reshape(-1)
                if valid.numel() != prediction.numel():
                    raise ValueError(f'expression mask shape mismatch: {valid.numel()} != {prediction.numel()}')
                loss = (prediction[valid.reshape(-1)] - target[valid.reshape(-1)]).square().mean()
                if 'flow_velocity' in out and 'flow_target' in out:
                    loss = loss + 0.1 * (out['flow_velocity'] - out['flow_target']).square().mean()
                mark_phase('task_losses')
                mark_phase('loss_balance')
        optimizer.zero_grad(set_to_none=True); loss.backward();mark_phase('backward')
        gradient_norm_sampled = (
            gradient_monitor.last_norm is None
            or gradient_norm_interval == 1
            or (completed_steps + 1) % gradient_norm_interval == 0
        )
        grad_norm, finite_grad = gradient_monitor.check(
            compute_norm=gradient_norm_sampled,
        )
        mark_phase('gradient_clip')
        finite_step = gradient_monitor.optimizer_step(
            optimizer, loss=loss, finite_gradients=finite_grad,
        );mark_phase('optimizer')
        if cuda_step_start is not None:
            cuda_step_end = torch.cuda.Event(enable_timing=True)
            cuda_step_end.record()
            # Synchronize only sampled timing steps.  The previous
            # implementation synchronized every CUDA step, preventing the
            # prefetch stream and the next launch from overlapping.
            cuda_step_end.synchronize()
        phase_milliseconds={}
        if phase_events is not None:
            previous='start'
            for name in phase_names:
                phase_milliseconds[name]=phase_events[previous].elapsed_time(phase_events[name])
                previous=name
        step_seconds=(
            cuda_step_start.elapsed_time(cuda_step_end) / 1000.0
            if cuda_step_start is not None and cuda_step_end is not None
            else time.perf_counter()-step_started
        )
        for block in raw_model.blocks: block.update_beta_balance()
        rows_seen += rows
        tokens=(int(batch.token_mask.sum()) if active and hasattr(batch,"token_mask") and batch.token_mask is not None else 0)
        total_seconds=data_seconds+step_seconds
        allocated_tokens=(int(batch.values.numel())
                          if active and hasattr(batch,'values') else 0)
        attention_tokens=(int(out.get('attention_allocated_tokens',allocated_tokens))
                          if active else 0)
        attention_buckets=(int(out.get('attention_bucket_count',1))
                           if active else 0)
        attention_backend=(str(out.get('attention_backend','padded'))
                           if active else 'inactive')
        feature_map_backend=(str(out.get('feature_map_backend','torch'))
                             if active else 'inactive')
        differential_merge_backend=(
            str(out.get('differential_merge_backend','torch'))
            if active else 'inactive'
        )
        development_source_cells=(int(development[0].batch_size)
                                  if active and development is not None else 0)
        development_target_cells=(int(development[1].batch_size)
                                  if active and development is not None else 0)
        development_source_tokens=(int(development[0].values.numel())
                                   if development_source_cells else 0)
        development_target_tokens=(int(development[1].values.numel())
                                   if development_target_cells else 0)
        attention_sequences=(int(out.get('attention_sequence_count',rows))
                             if active else 0)
        attention_max_seqlen=(int(out.get('attention_max_seqlen',0))
                              if active else 0)
        attention_max_seqlen_bucket=(
            int(out.get('attention_max_seqlen_bucket',0)) if active else 0
        )
        development_source_max_seqlen=(
            int(out.get('development_source_attention_max_seqlen',0))
            if active and development is not None else 0
        )
        development_source_max_seqlen_bucket=(
            int(out.get('development_source_attention_max_seqlen_bucket',0))
            if active and development is not None else 0
        )
        development_target_max_seqlen=(
            int(out.get('development_target_attention_max_seqlen',0))
            if active and development is not None else 0
        )
        development_target_max_seqlen_bucket=(
            int(out.get('development_target_attention_max_seqlen_bucket',0))
            if active and development is not None else 0
        )
        local_rank_values=[
            rows,tokens,data_seconds,step_seconds,total_seconds,
            allocated_tokens,attention_tokens,attention_buckets,
            development_source_cells,development_target_cells,
            development_source_tokens,development_target_tokens,
            attention_sequences,attention_max_seqlen,
            attention_max_seqlen_bucket,
            development_source_max_seqlen,
            development_source_max_seqlen_bucket,
            development_target_max_seqlen,
            development_target_max_seqlen_bucket,
        ]
        if world>1 and telemetry_sampled:
            local_rank_stats=torch.tensor(
                local_rank_values,device=device,dtype=torch.float64,
            )
            gathered_rank_stats=[torch.zeros_like(local_rank_stats) for _ in range(world)]
            dist.all_gather(gathered_rank_stats,local_rank_stats)
            rank_stats=torch.stack(gathered_rank_stats).cpu()
        else:
            # Keep unsampled telemetry off the accelerator entirely.  This
            # avoids a hidden synchronization caused by ``tensor.cpu()`` on
            # every rank between optimizer steps.
            rank_stats=torch.tensor([local_rank_values],dtype=torch.float64)
        gathered_allocator_stats=None
        if previous_allocator_counters is not None:
            current_memory_stats=torch.cuda.memory_stats(device)
            current_counters={
                name:int(current_memory_stats.get(name,0))
                for name in allocator_counter_names
            }
            allocator_values=[
                current_counters[name]-previous_allocator_counters[name]
                for name in allocator_counter_names
            ]+[
                int(current_memory_stats.get(name,0))
                for name in allocator_gauge_names
            ]
            previous_allocator_counters=current_counters
            if world>1 and telemetry_sampled:
                allocator_tensor=torch.tensor(
                    allocator_values,device=device,dtype=torch.float64,
                )
                gathered_allocator_stats=[torch.zeros_like(allocator_tensor)
                                            for _ in range(world)]
                dist.all_gather(gathered_allocator_stats,allocator_tensor)
                allocator_matrix=torch.stack(gathered_allocator_stats).cpu()
            else:
                allocator_matrix=torch.tensor(
                    [allocator_values],dtype=torch.float64,
                )
        gathered_phase_stats=None
        if phase_events is not None:
            phase_values=[phase_milliseconds[name] for name in phase_names]
            if world>1 and telemetry_sampled:
                phase_tensor=torch.tensor(
                    phase_values,device=device,dtype=torch.float64,
                )
                gathered_phase_stats=[torch.zeros_like(phase_tensor) for _ in range(world)]
                dist.all_gather(gathered_phase_stats,phase_tensor)
                phase_matrix=torch.stack(gathered_phase_stats).cpu()
            else:
                phase_matrix=torch.tensor([phase_values],dtype=torch.float64)
        rank_cells=[int(value) for value in rank_stats[:,0].tolist()]
        rank_tokens=[int(value) for value in rank_stats[:,1].tolist()]
        if world>1 and not telemetry_sampled:
            # Corpus progress must stay exact even when rank detail is
            # sampled.  A single scalar reduction is much cheaper than
            # gathering the full 19-field rank record and copying it back.
            progress_tensor=torch.tensor(
                [rows,tokens],device=device,dtype=torch.long,
            )
            dist.all_reduce(progress_tensor,op=dist.ReduceOp.SUM)
            global_cells=int(progress_tensor[0])
            global_tokens=int(progress_tensor[1])
            global_seconds=float(total_seconds)
        else:
            global_cells=sum(rank_cells);global_tokens=sum(rank_tokens)
            global_seconds=float(rank_stats[:,4].max())
        global_rows_seen+=global_cells;completed_steps += 1
        if rank==0 and hasattr(batch,'gene_index'):
            loss_info=balancer.materialize_telemetry(loss_info)
            if not bool(finite_step):
                raise FloatingPointError(
                    f'nonfinite loss or gradient at step {completed_steps}'
                )
            detail=' '.join(
                f'{name}={item["raw"]:.4g}' for name,item in loss_info.items()
            )
        else:
            detail=''
        if train_log and rank==0:
            allocated=torch.cuda.memory_allocated() if device.startswith('cuda') else 0
            reserved=torch.cuda.memory_reserved() if device.startswith('cuda') else 0
            peak=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0
            batch_record={'cells':rows,'tokens':tokens,
                          'layout':'packed' if getattr(batch,'packed',False) else 'padded',
                          'allocated_tokens':allocated_tokens,
                          'padding_tokens':allocated_tokens-tokens,
                          'token_utilization':tokens/max(allocated_tokens,1),
                          'attention_allocated_tokens':attention_tokens,
                          'attention_buckets':attention_buckets,
                          'attention_backend':attention_backend,
                          'feature_map_backend':feature_map_backend,
                          'differential_merge_backend':differential_merge_backend,
                          'attention_sequences':attention_sequences,
                          'attention_max_seqlen':attention_max_seqlen,
                          'attention_max_seqlen_bucket':attention_max_seqlen_bucket,
                          'attention_padding_tokens':attention_tokens-tokens,
                          'attention_token_utilization':tokens/max(attention_tokens,1)}
            if development is not None:
                batch_record['development_attention_fused']=bool(
                    out.get('development_attention_fused', False)
                )
                batch_record['development_attention_streams']=int(
                    out.get('development_attention_streams', 1)
                )
                batch_record['development_target_attention_fused']=bool(
                    out.get('development_target_attention_fused', False)
                )
            if not getattr(batch,'packed',False):
                batch_record['padded_width']=allocated_tokens//max(physical_rows,1)
            def supervised_rows(target,name):
                if target is None:return 0
                masked=(batch.modality_masked or {}).get(name)
                keep=target.ge(0)
                if masked is not None:keep=keep&masked.bool()
                return int(keep.sum())
            fully_masked=torch.ones(batch.batch_size,dtype=torch.bool,device=device)
            for name,observed in (batch.modality_observed or {}).items():
                hidden=(batch.modality_masked or {}).get(name,torch.zeros_like(observed,dtype=torch.bool))
                fully_masked &= (~observed.bool())|hidden.bool()
            record={'event':'train_step','timestamp_utc':utc_now(),'step':completed_steps,
                    'full_pass':full_pass,'rows_in_pass':global_rows_seen,'train_rows':train_rows,
                    'progress_fraction':(full_pass+global_rows_seen/train_rows)/full_passes,
                    'batch':batch_record,
                    'distributed':{'world_size':world,'active_ranks':int(active_count),
                                   'telemetry_sampled':telemetry_sampled,
                                   'global_cells':global_cells,'global_tokens':global_tokens,
                                   'global_cells_s':global_cells/global_seconds,
                                   'global_tokens_s':global_tokens/global_seconds,
                                   'rank_cells':rank_cells,'rank_tokens':rank_tokens,
                                   'rank_token_density':[token/max(cell,1) for cell,token in zip(rank_cells,rank_tokens)],
                                   'rank_data_s':rank_stats[:,2].tolist(),
                                   'rank_compute_s':rank_stats[:,3].tolist(),
                                   'rank_total_s':rank_stats[:,4].tolist(),
                                   'rank_allocated_tokens':[int(value) for value in rank_stats[:,5].tolist()],
                                   'rank_attention_allocated_tokens':[int(value) for value in rank_stats[:,6].tolist()],
                                   'rank_attention_buckets':[int(value) for value in rank_stats[:,7].tolist()]},
                    'timing':{'data_s':data_seconds,'compute_s':step_seconds,'cells_s':rows/total_seconds,'tokens_s':tokens/total_seconds},
                    'optimization':{'total_loss':finite_float(loss.detach()),'learning_rate':learning_rate,
                                    'gradient_norm':finite_float(grad_norm),
                                    'gradient_norm_sampled':gradient_norm_sampled,
                                    'gradient_health_backend':gradient_monitor.execution,
                                    'nonfinite':False,
                                    'router_runtime':raw_model.router_runtime_status()
                                    if hasattr(raw_model,'router_runtime_status') else None,
                                    'sinkhorn_runtime':raw_model.sinkhorn_runtime_status()
                                    if hasattr(raw_model,'sinkhorn_runtime_status') else None},
                    'losses':loss_info,'supervision':{
                        'cell_labeled':int(batch.cell_type_target.ge(0).sum()) if batch.cell_type_target is not None else 0,
                        'anatomy_labeled':int(batch.anatomy_target.ge(0).sum()) if batch.anatomy_target is not None else 0,
                        'cell_supervised':supervised_rows(batch.cell_type_target,'cell_type'),
                        'anatomy_supervised':supervised_rows(batch.anatomy_target,'anatomy'),
                        'all_context_masked':int(fully_masked.sum()),
                        'spatial_rows':int(batch.neighbor_mask.any(1).sum()) if batch.neighbor_mask is not None else 0,
                        'development_pair_rows':int(development[0].batch_size) if development is not None else 0},
                    'gpu_memory_bytes':{'allocated':allocated,'reserved':reserved,'peak_allocated':peak}}
            record['timing']['cuda_timing_sampled'] = timing_sampled
            record['timing']['compute_clock'] = (
                'cuda_event' if timing_sampled else 'cpu_enqueue'
            )
            if gathered_phase_stats is not None:
                record['timing']['cuda_phase_ms']={
                    name:phase_milliseconds[name] for name in phase_names
                }
                record['distributed']['rank_cuda_phase_ms']={
                    name:phase_matrix[:,index].tolist()
                    for index,name in enumerate(phase_names)
                }
            record['distributed'].update({
                'rank_development_source_cells':[int(value) for value in rank_stats[:,8].tolist()],
                'rank_development_target_cells':[int(value) for value in rank_stats[:,9].tolist()],
                'rank_development_source_tokens':[int(value) for value in rank_stats[:,10].tolist()],
                'rank_development_target_tokens':[int(value) for value in rank_stats[:,11].tolist()],
                'rank_encoder_tokens':[int(rank_stats[index,1]+rank_stats[index,10]+rank_stats[index,11])
                                       for index in range(len(rank_stats))],
                'rank_attention_sequences':[int(value) for value in rank_stats[:,12].tolist()],
                'rank_attention_max_seqlen':[int(value) for value in rank_stats[:,13].tolist()],
                'rank_attention_max_seqlen_bucket':[int(value) for value in rank_stats[:,14].tolist()],
                'rank_development_source_max_seqlen':[int(value) for value in rank_stats[:,15].tolist()],
                'rank_development_source_max_seqlen_bucket':[int(value) for value in rank_stats[:,16].tolist()],
                'rank_development_target_max_seqlen':[int(value) for value in rank_stats[:,17].tolist()],
                'rank_development_target_max_seqlen_bucket':[int(value) for value in rank_stats[:,18].tolist()],
            })
            if gathered_allocator_stats is not None:
                split=len(allocator_counter_names)
                record['gpu_allocator']={
                    'rank_counter_delta':{
                        name:[int(value) for value in allocator_matrix[:,index].tolist()]
                        for index,name in enumerate(allocator_counter_names)
                    },
                    'rank_gauge':{
                        name:[int(value) for value in allocator_matrix[:,split+index].tolist()]
                        for index,name in enumerate(allocator_gauge_names)
                    },
                }
            if development is not None:
                record['losses']['flow_velocity']={'raw':finite_float(velocity.detach())}
                record['losses']['flow_endpoint']={'raw':finite_float(endpoint.detach())}
            append_jsonl(train_log,record)
        if rank==0 and console_every>0 and completed_steps%console_every==0:
            density_ratio=max(rank_tokens[i]/max(rank_cells[i],1) for i in range(world))/max(min(rank_tokens[i]/max(rank_cells[i],1) for i in range(world)),1e-12)
            print(f'rank={rank} step={completed_steps} cells={rows} tokens={tokens} attention_buckets={attention_buckets} feature_map={feature_map_backend} differential_merge={differential_merge_backend} loss={float(loss.detach()):.6g} data_s={data_seconds:.3f} compute_s={step_seconds:.3f} cells_s={rows/total_seconds:.3f} tokens_s={tokens/total_seconds:.1f} global_cells_s={global_cells/global_seconds:.3f} global_tokens_s={global_tokens/global_seconds:.1f} rank_density_ratio={density_ratio:.3f} {detail}',flush=True)
        if max_steps is not None and completed_steps>=max_steps:
            if validation_every>0 and completed_steps%validation_every==0:
                validate_now(full_pass,rows_seen)
            offsets=gather_row_offsets(rows_seen)
            if checkpoint and rank==0:
                checkpoint.parent.mkdir(parents=True,exist_ok=True)
                save_checkpoint(checkpoint,full_pass,rows_seen,offsets,'smoke_checkpoint')
            if state_path and rank==0: atomic_json(state_path,{'status':'smoke_complete','rows_seen':sum(offsets),'rows_by_rank':offsets,'optimizer_steps':completed_steps,'world_size':world,'checkpoint':str(checkpoint)})
            return model
        if completed_steps % 10 == 0:
            print(f'progress pass={full_pass+1}/{full_passes} rows={rows_seen}/{train_rows} step={completed_steps} loss={float(loss.detach()):.6g}', flush=True)
        if validation_every>0 and completed_steps % validation_every==0:
            validate_now(full_pass,rows_seen)
        if checkpoint and completed_steps % checkpoint_every == 0:
            offsets=gather_row_offsets(rows_seen)
            if rank==0:
                save_checkpoint(checkpoint,full_pass,rows_seen,offsets,'rolling_checkpoint')
                if state_path: atomic_json(state_path,{'status':'running','full_pass':full_pass,'rows_seen':sum(offsets),'rows_by_rank':offsets,'train_rows':train_rows,'optimizer_steps':completed_steps,'world_size':world})
        if checkpoint and version_checkpoint_every>0 and completed_steps%version_checkpoint_every==0:
            offsets=gather_row_offsets(rows_seen)
            if rank==0:save_checkpoint(checkpoint.parent/f'{checkpoint.stem}-step-{completed_steps:08d}.pt',full_pass,rows_seen,offsets,'version_checkpoint')
      covered=torch.tensor(rows_seen,device=device,dtype=torch.long)
      if dist.is_initialized():dist.all_reduce(covered)
      if int(covered) != train_rows:
          raise ValueError(f'incomplete full pass: {int(covered)}!={train_rows}')
      start_rows = 0
      if state_path and rank==0: atomic_json(state_path,{'status':'pass_complete','full_pass':full_pass+1,'rows_seen':train_rows,'train_rows':train_rows,'optimizer_steps':completed_steps})
    if checkpoint and rank==0:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(checkpoint,full_passes,0,[0]*world,'final_checkpoint')
    if state_path and rank==0: atomic_json(state_path,{'status':'fit_complete_pending_validation','full_pass':full_passes,'rows_seen':train_rows,'train_rows':train_rows,'optimizer_steps':completed_steps,'checkpoint':str(checkpoint) if checkpoint else ''})
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=['light','normal','ultra'], default='normal')
    p.add_argument('--moe-backend',default='easy_deepspeed')
    p.add_argument('--attention-backend',
                   default='flash_bla_differential',
                   help='registered token-attention implementation')
    p.add_argument('--expert-attention-backend',
                   default='mage_differential',
                   help='registered attention implementation for internal MAGE experts')
    p.add_argument('--flow-backend',default='fmt',
                   help='registered developmental Flow Matching/DiT implementation')
    p.add_argument('--compile-backend',
                   default='torch_compile',
                   help='registered compiler used by optional flow compilation')
    p.add_argument('--optimizer-backend',
                   default='adamw',
                   help='registered optimizer implementation')
    p.add_argument('--backend-plugin',action='append',default=[],
                   help='import a Python module that registers extra backends (repeatable)')
    p.add_argument('--load-backend-entry-points',action='store_true',
                   help='load installed packages exposing the brainwm.backends entry-point group')
    p.add_argument('--vocab-size', type=int, default=None)
    p.add_argument('--full-passes', type=int, default=5,
                   help='minimum complete corpus passes; validation plateau is handled by the campaign runner')
    p.add_argument('--state', type=Path, default=None)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/foundation_moe.pt'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--benchmark-dir', type=Path)
    p.add_argument('--numpy-shards',type=Path)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--precision', choices=['bf16','fp32'], default='bf16')
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--minimum-learning-rate',type=float,default=2e-5)
    p.add_argument('--warmup-fraction',type=float,default=0.01)
    p.add_argument('--workers',type=int,default=2,
                   help='ordered NumPy batch-construction workers; worker count does not change samples or resume cursors')
    p.add_argument('--allow-resume-batch-size-change',action='store_true',
                   help='resume the exact row cursor with a new batch size; LR remains corpus-progress based')
    p.add_argument('--max-padded-tokens',type=int,default=60000,
                   help='GPU attention-slot budget for either padded or packed execution')
    p.add_argument('--packed-tokens',action='store_true',
                   help='use CSR-packed gene tokens and segmented linear attention without padding')
    p.add_argument('--packed-group-cells',type=int,default=32,
                   help='cells per sorted-length attention micro-bucket in packed mode')
    p.add_argument('--packed-attention-backend',
                   choices=['auto','bucket','flash_bla'],default='auto',
                   help='CUDA defaults to the vendored Flash BLA varlen kernel; bucket is the compatibility fallback')
    p.add_argument('--expert-execution',choices=['grouped','batched','loop'],default='batched',
                   help='execute fixed-capacity local SwiGLU experts with PyTorch batched GEMM')
    p.add_argument('--router-execution',choices=['eager','compile','auto'],default='eager',
                   help='fixed-shape gate projection: eager, TorchInductor compile, or CUDA-only auto')
    p.add_argument('--router-min-tokens',type=int,default=256,
                   help='minimum cell rows before --router-execution auto compiles')
    p.add_argument('--router-compile-mode',
                   choices=['default','reduce-overhead','max-autotune',
                            'max-autotune-no-cudagraphs'],
                   default='max-autotune-no-cudagraphs',
                   help='TorchInductor mode for the optional router projection')
    p.add_argument('--router-compile-dynamic',action=argparse.BooleanOptionalAction,
                   default=False,
                   help='allow dynamic row counts for compiled router projections')
    p.add_argument('--attention-projection-execution',
                   choices=['fla_layernorm_qkv','fused_triplet','fused','separate'],
                   default='fused_triplet',
                   help='execute differential-attention projections with the reviewed FLA LayerNorm+QKV kernel, three contiguous Q/K/V GEMMs, one GEMM plus copies, or six reference GEMMs')
    p.add_argument('--flow-sdpa-backend',
                   choices=['auto','flash','efficient','cudnn','math'],default='flash',
                   help='PyTorch SDPA kernel used by short-population FMT blocks')
    p.add_argument('--fuse-development-attention',
                   action=argparse.BooleanOptionalAction,default=True,
                   help='fuse main/source CSR attention while retaining independent MoE routing and target inference')
    p.add_argument('--fuse-development-target-attention',
                   action=argparse.BooleanOptionalAction,default=True,
                   help='join the inference-only developmental target stream to the shared Flash-BLA call')
    p.add_argument('--fuse-nbeats-heads',
                   action=argparse.BooleanOptionalAction,default=True,
                   help='run N-BEATS backcast and forecast heads as one wide GEMM')
    p.add_argument('--residual-norm-execution',
                   choices=['torch','torch_bf16','fla','auto'],default='auto',
                   help='residual+LayerNorm backend: legacy PyTorch, BF16 state, reviewed FLA fused CUDA kernel, or token-size auto selection')
    p.add_argument('--residual-norm-min-tokens',type=int,default=131072,
                   help='token threshold for the automatic FLA residual norm selector')
    p.add_argument('--feature-map-execution',
                   choices=['torch','compile','bucket','auto'],default='auto',
                   help='Flash-BLA exp/length feature map: eager PyTorch, dynamic Inductor, static bucket, or conservative auto selection')
    p.add_argument('--feature-map-min-tokens',type=int,default=131072,
                   help='minimum tokens before automatic dynamic feature-map compilation')
    p.add_argument('--feature-map-compile-mode',
                   choices=['default','reduce-overhead','max-autotune',
                            'max-autotune-no-cudagraphs'],
                   default='max-autotune-no-cudagraphs',
                   help='TorchInductor mode for the compiled Flash-BLA feature map')
    p.add_argument('--feature-map-bucket-size',type=int,default=65536,
                   help='static feature-map token bucket granularity')
    p.add_argument('--feature-map-max-padding-ratio',type=float,default=1.10,
                   help='maximum static bucket/real-token ratio for explicit bucket mode')
    p.add_argument('--differential-merge-execution',
                   choices=['torch','compile','auto'],default='auto',
                   help='positive-minus-correction merge: eager PyTorch, TorchInductor, or token-threshold auto')
    p.add_argument('--differential-merge-min-tokens',type=int,default=131072,
                   help='minimum real tokens before automatic differential-merge compilation')
    p.add_argument('--differential-merge-compile-mode',
                   choices=['default','reduce-overhead','max-autotune',
                            'max-autotune-no-cudagraphs'],
                   default='max-autotune-no-cudagraphs',
                   help='TorchInductor policy for the tensor-only differential merge')
    p.add_argument('--population-balance-mode',
                   choices=['exact','combined'],default='combined',
                   help='expert balance statistic for fused main/source/target populations')
    p.add_argument('--cuda-prefetch',
                   action=argparse.BooleanOptionalAction,default=True,
                   help='stage the next pinned batch on a dedicated CUDA stream')
    p.add_argument('--cuda-timing-interval',type=int,default=0,
                   help='sample an exact CUDA step time every N steps; 0 disables forced timing synchronizations')
    p.add_argument('--telemetry-interval',type=int,default=1,
                   help='cross-rank throughput/allocator telemetry interval; 1 keeps per-step detail, 0 disables rank gathers')
    p.add_argument('--gradient-norm-interval',type=int,default=10,
                   help='compute and log the full L2 gradient norm every N steps; finite-gradient checks still run every step')
    p.add_argument('--gradient-health-execution',
                   choices=('auto','foreach','amp_foreach'),default='auto',
                   help='auto uses PyTorch AMP foreach checking with fused AdamW and otherwise falls back to foreach norms')
    p.add_argument('--compile-flow',action=argparse.BooleanOptionalAction,default=False,
                   help='compile only the stable Flow Matching Transformer with torch.compile')
    p.add_argument('--compile-mode',choices=['default','reduce-overhead','max-autotune','max-autotune-no-cudagraphs'],default='reduce-overhead',
                   help='torch.compile mode used when --compile-flow is enabled')
    p.add_argument('--compile-dynamic',action=argparse.BooleanOptionalAction,default=True,
                   help='allow compiled FMT graphs to accept changing population sizes')
    p.add_argument('--compile-sinkhorn',action=argparse.BooleanOptionalAction,default=False,
                   help='compile developmental Sinkhorn on fixed 8/16/32/64-cell buckets')
    p.add_argument('--sinkhorn-compile-mode',
                   choices=['default','reduce-overhead','max-autotune',
                            'max-autotune-no-cudagraphs'],default='reduce-overhead',
                   help='TorchInductor mode used when --compile-sinkhorn is enabled')
    p.add_argument('--sinkhorn-compile-buckets',type=int,nargs='+',
                   default=[8,16,32,64],
                   help='ascending fixed population sizes for compiled Sinkhorn')
    p.add_argument('--development-max-padded-tokens',type=int,default=60000,
                   help='independent cap for quadratic OT/FMT population batches')
    p.add_argument('--max-steps',type=int,default=None,help='diagnostic only; never a final training result')
    p.add_argument('--metrics-dir',type=Path,default=None)
    p.add_argument('--validation-every',type=int,default=1000)
    p.add_argument('--validation-rows',type=int,default=8192)
    p.add_argument('--validation-shards',type=int,default=96)
    p.add_argument('--validation-seed',type=int,default=20260912)
    p.add_argument('--validation-development-groups',type=int,default=32)
    p.add_argument('--checkpoint-every',type=int,default=25)
    p.add_argument('--version-checkpoint-every',type=int,default=1000)
    p.add_argument('--console-every',type=int,default=10)
    p.add_argument('--split', choices=['train', 'validation'], default='train')
    p.add_argument('--task', default='integration')
    p.add_argument('--shuffle-training',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--shuffle-seed',type=int,default=20260912)
    p.add_argument('--shard-balance-waves',type=int,default=8,
                   help='number of randomized shard waves per density band')
    p.add_argument('--task-mask-probability',type=float,default=0.5)
    p.add_argument('--all-context-mask-probability',type=float,default=0.15)
    p.add_argument('--context-mask-probability',type=float,default=0.15)
    p.add_argument('--ddp-find-unused-parameters',
                   action=argparse.BooleanOptionalAction,default=None,
                   help='auto enables unused-parameter discovery only when the execution graph requires it')
    p.add_argument('--ddp-static-graph',
                   action=argparse.BooleanOptionalAction,default=False,
                   help='opt in to PyTorch DDP static-graph reducer scheduling; validate with a multi-GPU smoke first')
    p.add_argument('--ddp-bucket-cap-mb',type=float,default=None,
                   help='optional PyTorch DDP gradient bucket size in MiB for communication A/B tests')
    p.add_argument('--cuda-phase-timing',action='store_true',
                   help='record synchronized CUDA phase timings for diagnostics')
    p.add_argument('--diagnostic-ddp-noop-hook',action='store_true',
                   help='diagnostic only: replace DDP all-reduce with PyTorch noop hook and disable checkpoints')
    p.add_argument('--diagnostic-no-ddp-wrapper',action='store_true',
                   help='diagnostic only: keep distributed data/stats but run independent unwrapped models')
    args = p.parse_args()
    plugin_names = list(args.backend_plugin)
    plugin_names.extend(
        value.strip() for value in os.environ.get('BRAINWM_BACKEND_PLUGINS','').split(',')
        if value.strip()
    )
    args.backend_plugins_loaded = list(load_backend_plugins(plugin_names))
    args.backend_entry_points_loaded = []
    if args.load_backend_entry_points or os.environ.get('BRAINWM_LOAD_BACKEND_ENTRY_POINTS','0') == '1':
        args.backend_entry_points_loaded = list(load_backend_entry_points())
    # Resolve names after plugin import so externally registered factories are
    # accepted while misspellings still fail before any GPU/data setup.
    for attribute, kind in (
        ('moe_backend', 'moe'), ('attention_backend', 'attention'),
        ('expert_attention_backend', 'attention'),
        ('flow_backend', 'flow'), ('compile_backend', 'compile'),
        ('optimizer_backend', 'optimizer'),
    ):
        setattr(args, attribute, backend_spec(kind, getattr(args, attribute)).name)
    # The balance statistic is part of the mathematical objective. A resume
    # must use the checkpoint value even when a launcher now has a different
    # default. Checkpoints predating this field used the historical exact mode.
    if args.resume is not None:
        resume_header = torch.load(
            args.resume, map_location='cpu', weights_only=False,
        )
        resume_model_config = resume_header.get('model_config') or {}
        args.population_balance_mode = resume_model_config.get(
            'population_balance_mode', 'exact',
        )
    world_size=int(__import__('os').environ.get('WORLD_SIZE','1'))
    if args.diagnostic_ddp_noop_hook and args.diagnostic_no_ddp_wrapper:
        raise SystemExit('select only one distributed diagnostic mode')
    if args.diagnostic_ddp_noop_hook or args.diagnostic_no_ddp_wrapper:
        if world_size < 2 or args.max_steps is None or args.metrics_dir is None:
            raise SystemExit(
                'distributed diagnostic modes require distributed execution, '
                '--max-steps, and an explicit --metrics-dir'
            )
        args.checkpoint=None
        args.state=None
    if args.ddp_bucket_cap_mb is not None and args.ddp_bucket_cap_mb <= 0:
        raise SystemExit('--ddp-bucket-cap-mb must be positive')
    if args.feature_map_bucket_size < 1:
        raise SystemExit('--feature-map-bucket-size must be positive')
    if args.feature_map_max_padding_ratio < 1.0:
        raise SystemExit('--feature-map-max-padding-ratio must be at least 1')
    if args.differential_merge_min_tokens < 1:
        raise SystemExit('--differential-merge-min-tokens must be positive')
    if world_size>1:
        local_rank=int(__import__('os').environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl',device_id=torch.device('cuda',local_rank))
    global_rank=int(__import__('os').environ.get('RANK','0'))
    torch.random.default_generator.manual_seed(args.seed+global_rank)
    if torch.cuda.is_available():torch.cuda.manual_seed(args.seed+global_rank)
    if args.benchmark_dir is None:
        raise SystemExit('--benchmark-dir is required for the human corpus')
    import json
    import pandas as pd
    manifest = json.loads((args.benchmark_dir/'manifest.json').read_text())
    actual_vocab = len(pd.read_parquet(gene_vocabulary_path(args.benchmark_dir)))
    if args.vocab_size is not None and args.vocab_size != actual_vocab:
        raise ValueError('--vocab-size differs from the frozen source vocabulary')
    args.vocab_size = actual_vocab
    # Easy-MoE keeps sparse active compute fixed while capacity scales by preset.
    # The research mage_beta backend retains its historical 16 experts.
    easy_presets = {'light': (64,256,8,12,1,4), 'normal': (256,512,8,24,1,8), 'ultra': (512,1024,16,32,2,16)}
    mage_presets = {'light': (64,128,4,16), 'normal': (256,512,8,16), 'ultra': (512,1024,12,16)}
    presets = easy_presets if args.moe_backend == 'easy_deepspeed' else mage_presets
    if args.moe_backend == 'easy_deepspeed': d,h,b,e,k,heads = presets[args.preset]
    else: d,h,b,e = presets[args.preset]; k,heads = 1,1
    model = BrainWMFoundationMoE(args.vocab_size,d,h,b,e,top_k=k,attention_heads=heads,
                                 moe_backend=args.moe_backend,
                                 attention_backend=args.attention_backend,
                                 expert_attention_backend=args.expert_attention_backend,
                                 flow_backend=args.flow_backend,
                                 compile_backend=args.compile_backend,
                                 packed_group_cells=args.packed_group_cells,
                                 packed_attention_budget=(
                                     args.max_padded_tokens if args.packed_tokens else None
                                 ),
                                 packed_attention_backend=args.packed_attention_backend,
                                 expert_execution=args.expert_execution,
                                 router_execution=args.router_execution,
                                 router_min_tokens=args.router_min_tokens,
                                 router_compile_mode=args.router_compile_mode,
                                 router_compile_dynamic=args.router_compile_dynamic,
                                 attention_projection_execution=args.attention_projection_execution,
                                 flow_sdpa_backend=args.flow_sdpa_backend,
                                 fuse_development_attention=args.fuse_development_attention,
                                 fuse_development_target_attention=args.fuse_development_target_attention,
                                 fuse_nbeats_heads=args.fuse_nbeats_heads,
                                 residual_norm_execution=args.residual_norm_execution,
                                 residual_norm_min_tokens=args.residual_norm_min_tokens,
                                 feature_map_execution=args.feature_map_execution,
                                 feature_map_min_tokens=args.feature_map_min_tokens,
                                 feature_map_compile_mode=args.feature_map_compile_mode,
                                 feature_map_bucket_size=args.feature_map_bucket_size,
                                 feature_map_max_padding_ratio=args.feature_map_max_padding_ratio,
                                 differential_merge_execution=args.differential_merge_execution,
                                 differential_merge_min_tokens=args.differential_merge_min_tokens,
                                 differential_merge_compile_mode=args.differential_merge_compile_mode,
                                 population_balance_mode=args.population_balance_mode,
                                 sinkhorn_compile=args.compile_sinkhorn,
                                 sinkhorn_compile_mode=args.sinkhorn_compile_mode,
                                 sinkhorn_compile_buckets=tuple(args.sinkhorn_compile_buckets))
    metadata_vocab={}
    if args.numpy_shards and (args.numpy_shards/'joint_metadata_vocab.json').exists():
        metadata_vocab=json.loads((args.numpy_shards/'joint_metadata_vocab.json').read_text())
        categorical_names=('dataset','assay','technology','donor','life_stage','sex','disease_status','time_status',
                           'split_group','split_unit','section','sample','cl_id','cl_term','original_cell_type',
                           'original_subclass','uberon_id','uberon_term','original_anatomical_region','cell_type','anatomy')
        modality_dims={name:len(metadata_vocab[name]) for name in categorical_names}
        modality_dims['development']=4
        model=BrainWMFoundationMoE(args.vocab_size,d,h,b,e,top_k=k,attention_heads=heads,modality_dims=modality_dims,
                                   annotation_classes=len(metadata_vocab['cell_type']),anatomy_classes=len(metadata_vocab['anatomy']),
                                   categorical_modalities=set(categorical_names),moe_backend=args.moe_backend,
                                   attention_backend=args.attention_backend,
                                   expert_attention_backend=args.expert_attention_backend,
                                   flow_backend=args.flow_backend,
                                   compile_backend=args.compile_backend,
                                   packed_group_cells=args.packed_group_cells,
                                   packed_attention_budget=(
                                       args.max_padded_tokens if args.packed_tokens else None
                                   ),
                                   packed_attention_backend=args.packed_attention_backend,
                                   expert_execution=args.expert_execution,
                                   router_execution=args.router_execution,
                                   router_min_tokens=args.router_min_tokens,
                                   router_compile_mode=args.router_compile_mode,
                                   router_compile_dynamic=args.router_compile_dynamic,
                                   attention_projection_execution=args.attention_projection_execution,
                                   flow_sdpa_backend=args.flow_sdpa_backend,
                                   fuse_development_attention=args.fuse_development_attention,
                                   fuse_development_target_attention=args.fuse_development_target_attention,
                                   fuse_nbeats_heads=args.fuse_nbeats_heads,
                                   residual_norm_execution=args.residual_norm_execution,
                                   residual_norm_min_tokens=args.residual_norm_min_tokens,
                                   feature_map_execution=args.feature_map_execution,
                                   feature_map_min_tokens=args.feature_map_min_tokens,
                                   feature_map_compile_mode=args.feature_map_compile_mode,
                                   feature_map_bucket_size=args.feature_map_bucket_size,
                                   feature_map_max_padding_ratio=args.feature_map_max_padding_ratio,
                                   differential_merge_execution=args.differential_merge_execution,
                                   differential_merge_min_tokens=args.differential_merge_min_tokens,
                                   differential_merge_compile_mode=args.differential_merge_compile_mode,
                                   population_balance_mode=args.population_balance_mode,
                                   sinkhorn_compile=args.compile_sinkhorn,
                                   sinkhorn_compile_mode=args.sinkhorn_compile_mode,
                                   sinkhorn_compile_buckets=tuple(args.sinkhorn_compile_buckets))
    print(f'{args.preset}/{args.moe_backend}: {parameter_count(model)/1e6:.2f}M parameters')
    if args.benchmark_dir is None:
        raise SystemExit('--benchmark-dir is required for the human corpus')
    device = f'cuda:{int(__import__("os").environ.get("LOCAL_RANK","0"))}' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
    model=model.to(device)
    model.set_flow_compile(
        args.compile_flow,
        mode=args.compile_mode,
        dynamic=args.compile_dynamic,
    )
    if world_size>1 and not args.diagnostic_no_ddp_wrapper:
        has_development=bool(
            args.numpy_shards and (args.numpy_shards/'development_pairs.json').exists()
        )
        if args.ddp_find_unused_parameters is None:
            args.ddp_find_unused_parameters=(
                args.moe_backend!='easy_deepspeed' or not has_development
            )
        if args.ddp_static_graph and args.ddp_find_unused_parameters:
            raise SystemExit(
                '--ddp-static-graph requires --no-ddp-find-unused-parameters; '
                'the reducer cannot safely combine both modes'
            )
        ddp_kwargs = {
            'device_ids': [torch.cuda.current_device()],
            'find_unused_parameters': args.ddp_find_unused_parameters,
            'gradient_as_bucket_view': True,
            'static_graph': args.ddp_static_graph,
        }
        if args.ddp_bucket_cap_mb is not None:
            ddp_kwargs['bucket_cap_mb'] = args.ddp_bucket_cap_mb
        model=torch.nn.parallel.DistributedDataParallel(model,**ddp_kwargs)
        if args.diagnostic_ddp_noop_hook:
            from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
            model.register_comm_hook(None,noop_hook)
    import json as _json
    train_rows = _json.loads((args.benchmark_dir/'manifest.json').read_text())['counts']['task_cells_by_split'][args.task][args.split]
    resume_state=None
    if args.numpy_shards:
        resume_rank_offsets=None
        resume_state=(torch.load(args.resume,map_location='cpu',weights_only=False)
                      if args.resume else None)
        resume_step=int(resume_state.get('step',0)) if resume_state else 0
        if resume_state is not None and world_size>1:
            saved=resume_state.get('rows_by_rank')
            if saved is not None and len(saved)==world_size:
                resume_rank_offsets=list(map(int,saved))
            elif not args.shuffle_training:
                resume_rank_offsets=rank_offsets_for_global_prefix(
                    args.numpy_shards,args.split,int(resume_state.get('rows_in_pass',0)),world_size)
            else:
                raise ValueError('shuffled distributed resume requires saved rank-local offsets')
        factory=lambda offset,full_pass: shard_loader(args.numpy_shards,split=args.split,workers=args.workers,batch_size=args.batch_size,pin_memory=device.startswith('cuda'),max_padded_tokens=args.max_padded_tokens,start_row=offset,packed=args.packed_tokens,packed_group_cells=args.packed_group_cells,shuffle=args.shuffle_training,seed=args.shuffle_seed+full_pass,balance_waves=args.shard_balance_waves)
        development_loader=(DevelopmentPairLoader(
            args.numpy_shards,args.development_max_padded_tokens,
            packed=args.packed_tokens,start_step=resume_step,
        ) if (args.numpy_shards/'development_pairs.json').exists() else None)
        if development_loader is not None and args.workers>0:
            development_loader=prefetched_development_loader(
                development_loader,pin_memory=device.startswith('cuda'),
            )
        development_factory=(lambda:iter(development_loader)) if development_loader is not None else None
        validation_development_loader=(DevelopmentPairLoader(
            args.numpy_shards,args.development_max_padded_tokens,split='validation',
            group_samples=args.validation_development_groups,seed=args.validation_seed,
            distributed=False,packed=args.packed_tokens)
            if (args.numpy_shards/'development_validation_pairs.json').exists() else None)
        validation_development_factory=((lambda:iter(validation_development_loader))
                                        if validation_development_loader is not None else None)
        validation_factory=lambda:representative_shard_loader(args.numpy_shards,split='validation',shard_samples=args.validation_shards,seed=args.validation_seed,batch_size=min(args.batch_size,128),max_padded_tokens=args.max_padded_tokens,start_row=0,packed=args.packed_tokens,packed_group_cells=args.packed_group_cells)
    else:
        development_factory=None
        validation_development_factory=None
        validation_factory=None
        resume_rank_offsets=None
        factory = lambda offset,full_pass: iter_human_sparse_batches(args.benchmark_dir, args.batch_size, split=args.split,
                                            task=args.task, device=device, start_row=offset,
                                            max_rows=train_rows)
    try:
        optimizer_kwargs={'lr':args.learning_rate}
        if device.startswith('cuda'): optimizer_kwargs['fused']=True
        metrics_dir=args.metrics_dir or (args.checkpoint.parent.parent/'results'/'training'/args.checkpoint.stem)
        serializable_args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
        serializable_args['shard_order']=('density_balanced_v1' if args.shuffle_training
                                          else 'sequential_v1')
        class_weights=(load_class_weights(args.benchmark_dir/'train_loss_weights.csv',metadata_vocab,device)
                       if metadata_vocab else None)
        if int(__import__('os').environ.get('RANK','0'))==0:
            run_event=runtime_record(argparse.Namespace(**serializable_args),model)
            run_event['event']='resume' if args.resume else 'run_start'
            run_event['resume_checkpoint']=str(args.resume) if args.resume else None
            if resume_state is not None:
                previous=resume_state.get('run_config') or {}
                run_event['resume_configuration_changes']={
                    key:{'previous':previous.get(key),'current':value}
                    for key,value in serializable_args.items()
                    if key in previous and previous.get(key)!=value
                }
            append_jsonl(metrics_dir/'events.jsonl',run_event)
        optimizer = create_backend(
            'optimizer', args.optimizer_backend, model.parameters(), **optimizer_kwargs,
        )
        train(model, factory, optimizer,
              train_rows=train_rows, full_passes=args.full_passes, device=device,
              checkpoint=args.checkpoint, resume=args.resume, state_path=args.state,max_steps=args.max_steps,
              development_factory=development_factory, precision=args.precision,
              metrics_dir=metrics_dir,validation_factory=validation_factory,
              validation_every=args.validation_every,validation_rows=args.validation_rows,
              checkpoint_every=args.checkpoint_every,version_checkpoint_every=args.version_checkpoint_every,
              run_config=serializable_args,resume_rank_offsets=resume_rank_offsets,
              console_every=args.console_every,class_weights=class_weights,
              task_mask_probability=args.task_mask_probability,
              all_context_mask_probability=args.all_context_mask_probability,
              context_mask_probability=args.context_mask_probability,
              base_learning_rate=args.learning_rate,
              minimum_learning_rate=args.minimum_learning_rate,
              warmup_fraction=args.warmup_fraction,
              validation_development_factory=validation_development_factory,
              training_seed=args.seed,validation_seed=args.validation_seed,
              optimizer_backend=args.optimizer_backend,
              cuda_phase_timing=args.cuda_phase_timing,
              cuda_prefetch=args.cuda_prefetch,
              cuda_timing_interval=args.cuda_timing_interval,
              telemetry_interval=args.telemetry_interval,
              gradient_norm_interval=args.gradient_norm_interval,
              gradient_health_execution=args.gradient_health_execution)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
