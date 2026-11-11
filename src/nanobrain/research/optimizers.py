"""Bounded standard/custom optimizers; evaluation never uses this search API."""
from __future__ import annotations
import math
import torch

CUSTOM = {}
BOUNDS = {"eps": (1e-10, 1e-3), "momentum": (0, .99), "alpha": (.8, .9999),
          "dampening": (0, .9), "lr_decay": (0, .1), "initial_accumulator_value": (0, .1)}
ALLOWED = {"adamw": {"betas", "eps", "amsgrad"}, "adam": {"betas", "eps", "amsgrad"},
           "sgd": {"momentum", "dampening", "nesterov"}, "rmsprop": {"alpha", "eps", "momentum", "centered"},
           "adagrad": {"lr_decay", "eps", "initial_accumulator_value"}}


def validate_optimizer(spec):
    from .contracts import strict, finite
    strict(spec, {"name", "params", "weight_decay", "grad_clip", "schedule"}); finite(spec)
    name, params = spec["name"], spec["params"]
    if name not in ALLOWED and name not in CUSTOM: raise ValueError("optimizer is not registered")
    if name in ALLOWED and set(params) - ALLOWED[name]: raise ValueError("unsupported optimizer parameters")
    for key, value in params.items():
        if key == "betas":
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(isinstance(x, bool) or not isinstance(x, (int,float)) or not 0 <= x < 1 for x in value): raise ValueError("invalid optimizer betas")
        elif key in {"amsgrad", "nesterov", "centered"}:
            if type(value) is not bool: raise ValueError("optimizer flag must be boolean")
        elif name in ALLOWED:
            if key not in BOUNDS or isinstance(value,bool) or not isinstance(value,(int,float)) or not BOUNDS[key][0] <= value <= BOUNDS[key][1]: raise ValueError("optimizer parameter out of range")
    if not 0 <= spec["weight_decay"] <= .2 or not .1 <= spec["grad_clip"] <= 10: raise ValueError("weight decay/gradient clip out of range")
    if name == "sgd" and params.get("nesterov") and (params.get("momentum",0) <= 0 or params.get("dampening",0) != 0): raise ValueError("invalid Nesterov configuration")
    schedule = spec["schedule"];strict(schedule,{"name","warmup_steps","min_lr_ratio"})
    if schedule["name"] not in {"constant","linear","cosine"} or type(schedule["warmup_steps"]) is not int or not 0 <= schedule["warmup_steps"] <= 2000 or not 0 <= schedule["min_lr_ratio"] <= 1: raise ValueError("invalid learning-rate schedule")
    return spec


def build_optimizer(parameters, lr, spec):
    validate_optimizer(spec)
    parameters = list(parameters)
    factory = CUSTOM.get(spec["name"]) or {"adamw":torch.optim.AdamW,"adam":torch.optim.Adam,
        "sgd":torch.optim.SGD,"rmsprop":torch.optim.RMSprop,"adagrad":torch.optim.Adagrad}[spec["name"]]
    optimizer = factory(parameters, lr=float(lr), weight_decay=spec["weight_decay"], **spec["params"])
    if not isinstance(optimizer, torch.optim.Optimizer): raise ValueError("optimizer factory must return torch.optim.Optimizer")
    found = [p for group in optimizer.param_groups for p in group["params"]]
    if len(found) != len(parameters) or {id(p) for p in found} != {id(p) for p in parameters}: raise ValueError("optimizer must cover each trainable parameter exactly once")
    return optimizer


def learning_rate_at(base, step, total, spec):
    schedule = spec["schedule"];warmup=min(schedule["warmup_steps"],max(0,total-1))
    if step < warmup: return float(base)*(step+1)/max(1,warmup)
    ratio = min(1.,max(0.,(step-warmup)/max(1,total-warmup-1)))
    factor = 1. if schedule["name"]=="constant" else 1-ratio if schedule["name"]=="linear" else .5*(1+math.cos(math.pi*ratio))
    return float(base)*(schedule["min_lr_ratio"]+(1-schedule["min_lr_ratio"])*factor)
