"""Structured, checkpoint-scoped telemetry for full BrainWM training."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, f1_score, matthews_corrcoef, precision_score, recall_score)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def append_jsonl(path: Path, record: dict, *, durable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        if durable:
            os.fsync(stream.fileno())


def atomic_json(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_torch_save(path: Path, payload):
    """Write a Torch checkpoint without exposing a partially written target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        # Windows fsync requires a writable descriptor; opening for update is
        # also valid on POSIX and does not alter the serialized bytes.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def classification_metrics(true, predicted, prefix):
    true=np.asarray(true); predicted=np.asarray(predicted)
    if true.size == 0:return {f"{prefix}/count": 0}
    result={f"{prefix}/count":int(true.size),f"{prefix}/accuracy":accuracy_score(true,predicted),
            f"{prefix}/balanced_accuracy":balanced_accuracy_score(true,predicted),
            f"{prefix}/mcc":matthews_corrcoef(true,predicted),
            f"{prefix}/cohen_kappa":cohen_kappa_score(true,predicted)}
    for average in ("macro","micro","weighted"):
        result[f"{prefix}/precision_{average}"]=precision_score(true,predicted,average=average,zero_division=0)
        result[f"{prefix}/recall_{average}"]=recall_score(true,predicted,average=average,zero_division=0)
        result[f"{prefix}/f1_{average}"]=f1_score(true,predicted,average=average,zero_division=0)
    return {key:finite_float(value) if not key.endswith("/count") else value for key,value in result.items()}


class ValidationAccumulator:
    def __init__(self):
        self.loss_sum=defaultdict(float); self.loss_count=defaultdict(int)
        self.targets=defaultdict(list); self.predictions=defaultdict(list)
        self.rows=0; self.tokens=0; self.spatial_rows=0
        self.development_rows=0;self.development_groups=0

    def add(self, batch, output, losses, velocity=None, endpoint=None):
        rows=int(batch.batch_size); self.rows+=rows; self.tokens+=int(batch.token_mask.sum())
        for name,value in losses.items():
            if value is not None:
                self.loss_sum[name]+=float(value.detach())*rows; self.loss_count[name]+=rows
        if velocity is not None:self.loss_sum["flow_velocity"]+=float(velocity.detach())*rows;self.loss_count["flow_velocity"]+=rows
        if endpoint is not None:self.loss_sum["flow_endpoint"]+=float(endpoint.detach())*rows;self.loss_count["flow_endpoint"]+=rows
        for name,target_key,logit_key in (("cell","cell_type_target","canonical_annotation_logits"),("anatomy","anatomy_target","anatomy_logits")):
            target=getattr(batch,target_key)
            if target is not None and logit_key in output:
                keep=target.ge(0)
                self.targets[name].extend(target[keep].detach().cpu().tolist())
                self.predictions[name].extend(output[logit_key][keep].argmax(-1).detach().cpu().tolist())
        if batch.neighbor_mask is not None:self.spatial_rows+=int(batch.neighbor_mask.any(1).sum())

    def add_development(self, rows, velocity, endpoint):
        rows=int(rows);self.development_rows+=rows;self.development_groups+=1
        self.loss_sum['flow_velocity']+=float(velocity.detach())*rows
        self.loss_count['flow_velocity']+=rows
        self.loss_sum['flow_endpoint']+=float(endpoint.detach())*rows
        self.loss_count['flow_endpoint']+=rows

    def metrics(self):
        result={f"loss/{name}":finite_float(total/self.loss_count[name]) for name,total in self.loss_sum.items()}
        result.update(classification_metrics(self.targets["cell"],self.predictions["cell"],"annotation"))
        result.update(classification_metrics(self.targets["anatomy"],self.predictions["anatomy"],"anatomy"))
        result.update({"coverage/rows":self.rows,"coverage/tokens":self.tokens,
                       "coverage/spatial_rows":self.spatial_rows,
                       "coverage/development_rows":self.development_rows,
                       "coverage/development_groups":self.development_groups})
        active=[v for k,v in result.items() if k.startswith("loss/") and v is not None]
        result["loss/mean_active_objectives"]=finite_float(sum(active)/len(active)) if active else None
        return result


def runtime_record(args, model):
    record = {
        "event":"run_start","timestamp_utc":utc_now(),"argv":vars(args),
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "torch":torch.__version__, "python":platform.python_version(),
        "hostname":platform.node(),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES",""),
        "gpu_count":torch.cuda.device_count(),
        "gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    manifest = getattr(model, "backend_manifest", None)
    if callable(manifest):
        record["backends"] = manifest()
    model_config = getattr(model, "model_config", None)
    if model_config is not None:
        record["model_config"] = dict(model_config)
    return record
