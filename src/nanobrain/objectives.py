"""Independent task objectives, separate from graph construction/evaluation.

Loss families reuse torch.nn.functional. Validation metrics and label access
remain task-specific; candidates cannot change evaluator definitions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contracts import LossBundle


@dataclass(frozen=True)
class ObjectiveAPI:
    tasks: tuple[str, ...]
    compute: object
    defaults: dict
    population: bool = False


OBJECTIVES: dict[str, ObjectiveAPI] = {}


def register_objective(name, *, tasks, compute, defaults, population=False):
    """Register a trusted implementation; LLM JSON can only select these APIs."""
    if name in OBJECTIVES:
        raise ValueError(f"objective already registered: {name}")
    OBJECTIVES[name] = ObjectiveAPI(tuple(tasks), compute, dict(defaults), population)


def default_objectives(endpoint=1., expression=.5, adversarial=.5):
    return {
        "annotation": {"name": "classification", "params": {"label_smoothing": 0.}},
        "spatial_niche": {"name": "classification", "params": {"label_smoothing": 0.}},
        "integration": {"name": "adversarial_reconstruction", "params": {
            "regression": "mse", "reconstruction_weight": 1., "batch_weight": 1., "adversarial_strength": adversarial}},
        "spatial_completion": {"name": "masked_regression", "params": {"regression": "mse"}},
        "development": {"name": "conditional_flow_matching", "params": {
            "velocity_regression": "mse", "endpoint_regression": "mse", "expression_regression": "mse",
            "velocity_weight": 1., "endpoint_weight": endpoint, "expression_weight": expression}},
    }


def validate_objectives(spec):
    defaults = default_objectives()
    if not isinstance(spec, dict) or set(spec) != set(defaults):
        raise ValueError("objectives must specify all five independent tasks")
    for task, item in spec.items():
        if not isinstance(item, dict) or set(item) != {"name", "params"}:
            raise ValueError("objective requires name and params")
        api = OBJECTIVES.get(item["name"])
        if api is None or task not in api.tasks or api.population != (task == "development"):
            raise ValueError(f"objective {item['name']} is incompatible with task {task}")
        if not isinstance(item["params"], dict) or set(item["params"]) != set(api.defaults):
            raise ValueError(f"objective {task} parameter keys mismatch")
        for k, v in item["params"].items():
            if "regression" in k:
                if v not in {"mse", "smooth_l1", "l1"}:
                    raise ValueError("unknown regression loss")
            elif isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError("objective parameters must be finite numbers")
            elif k == "label_smoothing":
                if not 0 <= v <= .2:
                    raise ValueError("label_smoothing must be in [0,.2]")
            elif not 0 < v <= 4:
                raise ValueError("objective weights/strength must be in (0,4]; required losses cannot be disabled")
    return spec


def masked_classification(logits, target, mask=None, label_smoothing=0.):
    valid = target.ge(0)
    if mask is not None:
        valid = valid & mask
    if not bool(valid.any()):
        return logits.sum() * 0.
    return F.cross_entropy(logits[valid], target[valid].long(), label_smoothing=label_smoothing)


REGRESSION_LOSSES = {"mse": F.mse_loss, "smooth_l1": F.smooth_l1_loss, "l1": F.l1_loss}


def masked_regression(prediction, target, mask=None, regression="mse"):
    valid = torch.isfinite(target)
    if mask is not None:
        while mask.ndim < valid.ndim:
            mask = mask.unsqueeze(-1)
        valid = valid & mask
    if not bool(valid.any()):
        return prediction.sum() * 0.
    return REGRESSION_LOSSES[regression](
        prediction[valid].float(), target[valid].float())


def classification_objective(model, batch, embedding, params, task):
    label = "annotation" if task == "annotation" else "niche"
    head = model.annotation_head if task == "annotation" else model.niche_head
    return masked_classification(head(embedding), batch.labels[label], batch.label_masks.get(label), **params), {}


def completion_objective(model, batch, embedding, params, task):
    if batch.completion_target is None or batch.completion_mask is None:
        raise ValueError("completion requires held-out targets and masks")
    return masked_regression(model.completion_head(embedding), batch.completion_target,
                             batch.completion_mask, **params), {}


def integration_objective(model, batch, embedding, params, task):
    observed = torch.isfinite(batch.features)
    if batch.feature_observed is not None:
        observed = observed & batch.feature_observed
    if batch.completion_mask is not None:
        observed = observed & ~batch.completion_mask
    decoder = model.integration_reconstruction_head if model.genome.architecture is not None else model.completion_head
    reconstruction = masked_regression(decoder(embedding), batch.features, observed, params["regression"])
    batch_logits, stability = model.integration_adversarial_logits(
        embedding, adversarial_strength=params["adversarial_strength"]
    )
    batch_loss = masked_classification(
        batch_logits, batch.labels["integration"], batch.label_masks.get("integration")
    )
    loss = params["reconstruction_weight"] * reconstruction + params["batch_weight"] * batch_loss
    return loss, {
        "integration_reconstruction": reconstruction,
        "integration_batch_adversarial": batch_loss,
        **stability,
    }


def row_objective(model, batch, task):
    from .task_training import task_input
    spec = model.objective_specs[task]
    api = OBJECTIVES[spec["name"]]
    if api.population:
        raise ValueError(f"task {task} requires a population batch")
    embedding = model.encode(task_input(batch, task))
    loss, diagnostics = api.compute(model, batch, embedding, spec["params"], task)
    return LossBundle(terms={task: loss}, weights={task: 1.}), diagnostics


def population_flow_objective(model, batch, *, tau=None):
    from brainwm.losses import barycentric_transport_target
    p = model.objective_specs["development"]["params"]
    source, target = model.encode(batch.source), model.encode(batch.target)
    matched_target, plan = model.pairing(source, target)
    tau = (torch.rand((len(source), 1), device=source.device, dtype=source.dtype) if tau is None
           else tau.to(device=source.device, dtype=source.dtype))
    state = (1 - tau) * source + tau * matched_target
    time = model._time_features(len(source), tau, batch.source_time_days, batch.target_time_days)
    velocity = model.flow(state, source, time)
    endpoint = state + (1 - tau) * velocity
    target_features = barycentric_transport_target(plan, batch.target.features.detach().float()).to(endpoint.dtype)
    observed = None
    if batch.target.feature_observed is not None:
        observed = barycentric_transport_target(plan, batch.target.feature_observed.detach().float()) >= .5
    terms = {
        "development_velocity": masked_regression(velocity, matched_target - source, regression=p["velocity_regression"]),
        "development_endpoint": masked_regression(endpoint, matched_target, regression=p["endpoint_regression"]),
        "development_expression": masked_regression(model.population_decoder(endpoint), target_features, observed, p["expression_regression"]),
    }
    total = sum(p[key + "_weight"] * terms["development_" + key] for key in ("velocity", "endpoint", "expression"))
    return total, terms


def objective_catalog():
    return {"defaults": default_objectives(), "regression_choices": ["mse", "smooth_l1", "l1"],
            "registry": {name: {"tasks": api.tasks, "params": api.defaults, "population": api.population}
                         for name, api in OBJECTIVES.items()},
            "label_smoothing_range": [0, .2], "positive_weight_range": [0.000001, 4],
            "contract": "Independent task label access. Required losses stay positive. Evaluators never change."}


def population_objective(model, batch, *, tau=None):
    spec = model.objective_specs["development"]
    api = OBJECTIVES[spec["name"]]
    if not api.population:
        raise ValueError("development requires a population objective")
    return api.compute(model, batch, tau=tau)


for _task, _function, _population in (
    ("annotation", classification_objective, False),
    ("integration", integration_objective, False),
    ("spatial_completion", completion_objective, False),
    ("development", population_flow_objective, True),
):
    _spec = default_objectives()[_task]
    register_objective(_spec["name"], tasks=("annotation", "spatial_niche") if _task == "annotation" else (_task,),
                       compute=_function, defaults=_spec["params"], population=_population)
