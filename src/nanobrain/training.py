"""Short-budget candidate training and validation receipts."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .contracts import BrainBatch, LossBundle
from .data import DatasetSplits, NanoDataset
from .genome import CandidateGenome
from .model import NanoBrainModel, NanoBrainModelConfig
from .task_training import TASKS, configure_task, row_task_loss, evaluate_task
from brainwm.multitask_objective import MultiObjectiveBalancer
from brainwm.training_observability import (
    append_jsonl,
    atomic_torch_save,
    finite_float,
    utc_now,
)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 25
    batch_size: int = 128
    validation_batch_size: int = 256
    seed: int = 7
    device: str = "auto"
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    development_cells: int = 64
    validation_development_groups: int = 32
    eval_interval: int = 0
    precision: str = "bf16"
    loss_balance_mode: str = "ema_equal"
    loss_balance_decay: float = 0.99
    task: str = "joint"
    checkpoint_policy: str = "all"
    integration_adversarial_warmup_steps: int = 500
    integration_logit_clip: float = 20.0
    stability_max_loss: float = 1.0e8
    stability_max_gradient_norm: float = 1.0e6

    def __post_init__(self):
        if self.task not in ("joint", *TASKS):
            raise ValueError(f"unsupported training task: {self.task}")
        if self.checkpoint_policy not in {"all", "final_weights"}:
            raise ValueError("unsupported checkpoint_policy")
        if int(self.integration_adversarial_warmup_steps) < 1:
            raise ValueError("integration_adversarial_warmup_steps must be positive")
        if not np.isfinite(float(self.integration_logit_clip)) or float(self.integration_logit_clip) <= 0:
            raise ValueError("integration_logit_clip must be positive and finite")
        for name in ("stability_max_loss", "stability_max_gradient_norm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def _classification_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask & target.ge(0)
    if not bool(valid.any()):
        return {"accuracy": float("nan")}
    prediction = logits.argmax(-1)
    return {"accuracy": float((prediction[valid] == target[valid]).float().mean().detach().cpu())}


def _evaluate(
    model: NanoBrainModel,
    splits: DatasetSplits,
    batch_size: int,
    device: torch.device,
    *,
    development_groups: int = 32,
    task: str = "joint",
) -> dict[str, float]:
    if task != "joint":
        return evaluate_task(model, splits, task, batch_size, device, development_groups=development_groups)
    dataset = splits.validation
    model.eval()
    totals: dict[str, list[float]] = {
        "annotation_accuracy": [],
        "niche_accuracy": [],
        "integration_accuracy": [],
        "development_row_mse": [],
        "development_population_mse": [],
        "completion_mse": [],
    }
    with torch.no_grad():
        for batch in dataset.iter_batches(batch_size, shuffle=False):
            batch = batch.to(device)
            prediction = model(batch)
            for name, logits_name in (
                ("annotation", "annotation_logits"),
                ("niche", "niche_logits"),
                ("integration", "integration_prediction"),
            ):
                metrics = _classification_metrics(
                    getattr(prediction, logits_name),
                    batch.labels[name],
                    batch.label_masks.get(name, torch.ones_like(batch.labels[name], dtype=torch.bool)),
                )
                if np.isfinite(metrics["accuracy"]):
                    totals[f"{name}_accuracy"].append(metrics["accuracy"])
            valid = batch.label_masks["development"]
            if splits.development_validation is None and bool(valid.any()):
                error = prediction.development_prediction[valid] - batch.labels["development"][valid]
                totals["development_row_mse"].append(float(error.square().mean().detach().cpu()))
            if batch.completion_target is not None and batch.completion_mask is not None:
                valid_completion = batch.completion_mask & torch.isfinite(batch.completion_target)
                if bool(valid_completion.any()):
                    error = prediction.completion_prediction[valid_completion] - batch.completion_target[valid_completion]
                    totals["completion_mse"].append(float(error.square().mean().detach().cpu()))
        pairs = splits.development_validation
        if pairs is not None:
            limit = min(len(pairs), max(1, int(development_groups)))
            selected = np.unique(np.linspace(0, len(pairs) - 1, limit, dtype=np.int64))
            for index in selected:
                population = pairs.batch(
                    int(index), max_cells=model.genome.rollout_steps * 8,
                    seed=model.genome.seed,
                ).to(device)
                predicted, _ = model.predict_development_population(
                    population.source,
                    population.source_time_days,
                    population.target_time_days,
                )
                target = population.target.features
                mean_error = (predicted.float().mean(0) - target.float().mean(0)).square().mean()
                totals["development_population_mse"].append(float(mean_error.cpu()))
    return {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in totals.items()
    }


def _quality(validation: dict[str, float]) -> float:
    values: list[float] = []
    for name in ("annotation_accuracy", "niche_accuracy"):
        value = validation.get(name, float("nan"))
        if np.isfinite(value):
            values.append(float(value))
    integration_accuracy = validation.get("integration_accuracy", float("nan"))
    if np.isfinite(integration_accuracy):
        values.append(float(1.0 - integration_accuracy))
    for name in ("completion_mse", "development_population_mse", "integration_objective"):
        value = validation.get(name, float("nan"))
        if np.isfinite(value):
            values.append(float(1.0 / (1.0 + max(0.0, value))))
    return float(np.mean(values)) if values else 0.0


def train_candidate(
    splits: DatasetSplits,
    model_config: NanoBrainModelConfig,
    genome: CandidateGenome,
    train_config: TrainConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train one candidate and return an immutable, JSON-safe receipt."""
    genome.validate()
    from .integration_components import prepare_component_views, selected_components
    if train_config.task == "spatial_completion" and selected_components(genome):
        raise ValueError("dataset-level input components are unavailable for masked completion")
    component_receipts = prepare_component_views(splits, genome)
    _seed_everything(int(genome.seed) if genome.seed is not None else int(train_config.seed))
    device = resolve_device(train_config.device)
    model = NanoBrainModel(model_config, genome).to(device)
    model._integration_warmup_steps = int(train_config.integration_adversarial_warmup_steps)
    model._integration_logit_clip = float(train_config.integration_logit_clip)
    task = train_config.task
    if task == "development" and (splits.development_train is None or splits.development_validation is None):
        raise ValueError("independent development requires train/validation population pairs")
    trainable_names = configure_task(model, task)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(genome.learning_rate),
        weight_decay=float(train_config.weight_decay),
    )
    balancer = MultiObjectiveBalancer(
        tuple(genome.loss_weights) if task == "joint" else (task,),
        mode=train_config.loss_balance_mode if task == "joint" else "fixed",
        decay=train_config.loss_balance_decay,
    ).to(device)
    balancer.train()
    model.train()
    start_time = time.perf_counter()
    step_records: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    gradient_finite_parameters = 0
    gradient_nonzero_parameters = 0
    tracked_parameter = None
    tracked_before = None
    tracked_after = None
    iterator = None
    development_order = None
    development_position = 0
    best_quality = float("-inf")
    best_state = None
    best_validation: dict[str, float] | None = None
    best_step = 0
    completed_steps = 0
    stability_failures: list[dict[str, Any]] = []
    use_amp = device.type == "cuda" and train_config.precision == "bf16"
    target = Path(output_dir) if output_dir is not None else None
    history_path = target / "training_history.jsonl" if target is not None else None
    test_opened = splits.test is not None
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        append_jsonl(
            history_path,
            {
                "event": "run_start",
                "timestamp_utc": utc_now(),
                "candidate_id": genome.genome_id,
                "genome": genome.to_dict(),
                "model": asdict(model_config),
                "train": asdict(train_config),
                "test_opened": test_opened,
            },
            durable=True,
        )
    train_log_interval = max(1, min(100, int(train_config.steps) // 20))

    def checkpoint_payload(
        *, step: int, validation: dict[str, float], include_optimizer: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "genome": genome.to_dict(),
            "config": asdict(model_config),
            "train": asdict(train_config),
            "step": int(step),
            "validation": validation,
            "best_validation_quality": best_quality,
            "test_opened": test_opened,
            "integration_component_receipts": component_receipts,
        }
        if include_optimizer:
            payload["optimizer"] = optimizer.state_dict()
            payload["loss_balancer"] = balancer.state_dict()
        return payload

    for step in range(int(train_config.steps)):
        model._objective_step = int(step)
        if iterator is None and task != "development":
            iterator = iter(splits.train.iter_batches(train_config.batch_size, shuffle=True, seed=genome.seed + step))
        if task != "development":
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(splits.train.iter_batches(train_config.batch_size, shuffle=True, seed=genome.seed + step))
                batch = next(iterator)
            batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            flow_diagnostics: dict[str, torch.Tensor] = {}
            if task == "joint":
                prediction = model(batch)
                losses = model.loss(batch, prediction, include_row_development=splits.development_train is None)
            elif task != "development":
                losses, flow_diagnostics = row_task_loss(model, batch, task)
            else:
                losses = LossBundle(terms={}, weights={"development": 1.0})
            if task in {"joint", "development"} and splits.development_train is not None:
                if development_order is None or development_position >= len(development_order):
                    development_order = np.arange(len(splits.development_train), dtype=np.int64)
                    np.random.default_rng(genome.seed + step).shuffle(development_order)
                    development_position = 0
                population_index = int(development_order[development_position])
                development_position += 1
                population = splits.development_train.batch(
                    population_index,
                    max_cells=train_config.development_cells,
                    seed=genome.seed + step,
                ).to(device)
                development_loss, flow_diagnostics = model.population_flow_losses(population)
                losses.terms["development"] = development_loss
        total, balance_info = balancer(
            losses.terms,
            weights=losses.weights,
            materialize_telemetry=False,
        )
        term_finite = all(bool(torch.isfinite(value).all()) for value in losses.terms.values())
        total_finite = bool(torch.isfinite(total).all())
        total_value = float(total.detach().float().abs().cpu()) if total_finite else float("inf")
        if not term_finite or not total_finite or total_value > float(train_config.stability_max_loss):
            failure = {
                "step": int(step + 1),
                "reason": "nonfinite_loss" if not term_finite or not total_finite else "loss_threshold_exceeded",
                "total": total_value,
                "terms": {
                    name: (float(value.detach().float().cpu()) if bool(torch.isfinite(value).all()) else float("nan"))
                    for name, value in losses.terms.items()
                },
            }
            stability_failures.append(failure)
            if history_path is not None:
                append_jsonl(history_path, {"event": "stability_failure", "timestamp_utc": utc_now(), **failure}, durable=True)
            break
        total.backward()
        squared_grad_norm = total.new_zeros(())
        gradients_finite = True
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            finite = bool(torch.isfinite(parameter.grad).all())
            gradients_finite = gradients_finite and finite
            if finite:
                gradient_finite_parameters += 1
            if bool(parameter.grad.detach().abs().sum() > 0):
                gradient_nonzero_parameters += 1
                squared_grad_norm = squared_grad_norm + parameter.grad.detach().float().square().sum()
                if step == 0 and tracked_parameter is None:
                    tracked_parameter = parameter
                    tracked_before = parameter.detach().clone()
        gradient_norm = float(squared_grad_norm.sqrt().detach().cpu()) if gradients_finite else float("inf")
        if (not gradients_finite or not np.isfinite(gradient_norm)
                or gradient_norm > float(train_config.stability_max_gradient_norm)):
            failure = {
                "step": int(step + 1),
                "reason": "nonfinite_gradient" if not gradients_finite or not np.isfinite(gradient_norm)
                else "gradient_threshold_exceeded",
                "gradient_norm": gradient_norm,
            }
            stability_failures.append(failure)
            if history_path is not None:
                append_jsonl(history_path, {"event": "stability_failure", "timestamp_utc": utc_now(), **failure}, durable=True)
            break
        if train_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip))
        optimizer.step()
        completed_steps = step + 1
        if step == 0 and tracked_parameter is not None:
            # Clone after the update so the receipt proves an actual optimizer
            # step occurred, rather than only a successful backward pass.
            tracked_after = tracked_parameter.detach().clone()
        if step == 0 or step + 1 == train_config.steps or (step + 1) % train_log_interval == 0:
            balance_values = balancer.materialize_telemetry(balance_info)
            record = {
                "step": float(step + 1),
                **{name: float(value.detach().cpu()) for name, value in losses.terms.items()},
                "total": float(total.detach().cpu()),
                "unbalanced_total": float(losses.total.detach().cpu()),
                "gradient_norm": gradient_norm,
            }
            for name, values in balance_values.items():
                record[f"balance_{name}_scale"] = float(values["scale"])
                record[f"balance_{name}_weighted"] = float(values["weighted"])
            record.update({name: float(value.detach().cpu()) for name, value in flow_diagnostics.items()})
            step_records.append(record)
            if history_path is not None:
                append_jsonl(
                    history_path,
                    {
                        "event": "train",
                        "timestamp_utc": utc_now(),
                        "elapsed_seconds": float(time.perf_counter() - start_time),
                        **record,
                    },
                )
        should_evaluate = (
            train_config.eval_interval > 0
            and ((step + 1) % train_config.eval_interval == 0 or step + 1 == train_config.steps)
        )
        if should_evaluate:
            validation = _evaluate(
                model,
                splits,
                train_config.validation_batch_size,
                device,
                development_groups=train_config.validation_development_groups,
                task=task,
            )
            current_quality = _quality(validation)
            validation_history.append({"step": float(step + 1), "quality": current_quality, **validation})
            improved = current_quality > best_quality
            if improved:
                best_quality = current_quality
                best_state = copy.deepcopy(model.state_dict()) if train_config.checkpoint_policy == "all" else None
                best_validation = dict(validation)
                best_step = step + 1
            if target is not None:
                version_path = target / f"candidate-{genome.genome_id}-step-{step + 1:08d}.pt"
                latest_path = target / f"candidate-{genome.genome_id}-latest.pt"
                payload = checkpoint_payload(
                    step=step + 1,
                    validation=validation,
                    include_optimizer=True,
                )
                if train_config.checkpoint_policy == "all":
                    atomic_torch_save(version_path, payload)
                    atomic_torch_save(latest_path, payload)
                if improved and train_config.checkpoint_policy == "all":
                    atomic_torch_save(
                        target / f"candidate-{genome.genome_id}-best.pt",
                        {
                            **checkpoint_payload(
                                step=step + 1,
                                validation=validation,
                                include_optimizer=False,
                            ),
                            "model": best_state,
                            "validation_quality": best_quality,
                        },
                    )
                append_jsonl(
                    history_path,
                    {
                        "event": "validation",
                        "timestamp_utc": utc_now(),
                        "elapsed_seconds": float(time.perf_counter() - start_time),
                        "step": int(step + 1),
                        "quality": finite_float(current_quality),
                        "best_quality": finite_float(best_quality),
                        "improved": improved,
                        "checkpoint": str(version_path) if train_config.checkpoint_policy == "all" else None,
                        "metrics": {
                            name: finite_float(value) for name, value in validation.items()
                        },
                        "test_opened": test_opened,
                    },
                    durable=True,
                )
            model.train()
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    if validation_history and int(validation_history[-1]["step"]) == int(train_config.steps):
        final_record = validation_history[-1]
        validation = {
            name: float(value)
            for name, value in final_record.items()
            if name not in {"step", "quality"}
        }
        quality = float(final_record["quality"])
    elif not stability_failures:
        validation = _evaluate(
            model,
            splits,
            train_config.validation_batch_size,
            device,
            development_groups=train_config.validation_development_groups,
            task=task,
        )
        quality = _quality(validation)
        if quality > best_quality:
            best_quality = quality
            best_state = copy.deepcopy(model.state_dict()) if train_config.checkpoint_policy == "all" else None
            best_validation = dict(validation)
            best_step = int(train_config.steps)
        validation_history.append({"step": float(train_config.steps), "quality": quality, **validation})
    else:
        validation = {}
        quality = float("nan")
    checkpoint_path = None
    best_checkpoint_path = None
    if target is not None:
        checkpoint_path = target / f"candidate-{genome.genome_id}.pt"
        atomic_torch_save(
            checkpoint_path,
            checkpoint_payload(
                step=int(completed_steps),
                validation=validation,
                include_optimizer=train_config.checkpoint_policy == "all",
            ),
        )
        if train_config.checkpoint_policy == "all":
            best_checkpoint_path = target / f"candidate-{genome.genome_id}-best.pt"
            atomic_torch_save(
                best_checkpoint_path,
                {
                    **checkpoint_payload(
                        step=best_step or int(completed_steps),
                        validation=best_validation or validation,
                        include_optimizer=False,
                    ),
                    "model": best_state if best_state is not None else model.state_dict(),
                    "validation_quality": best_quality,
                },
            )
        append_jsonl(
            history_path,
            {
                "event": "run_complete",
                "timestamp_utc": utc_now(),
                "elapsed_seconds": elapsed,
                "step": int(completed_steps),
                "quality": finite_float(quality),
                "best_quality": finite_float(best_quality),
                "checkpoint": str(checkpoint_path),
                "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
                "status": "failed_stability" if stability_failures else "complete",
                "test_opened": test_opened,
            },
            durable=True,
        )
    subset_manifest = (splits.train.metadata or {}).get("subset_manifest", {})
    manifest_test_rows = subset_manifest.get("splits", {}).get("test", {}).get("rows", 0)
    test_rows = splits.test.size if splits.test is not None else int(manifest_test_rows)
    receipt: dict[str, Any] = {
        "format": "nanobrain_experiment_receipt_v1",
        "candidate_id": genome.genome_id,
        "genome": genome.to_dict(),
        "model": asdict(model_config),
        "train": asdict(train_config),
        "device": str(device),
        "parameter_count": model.parameter_count,
        "training_task": task,
        "integration_component_receipts": component_receipts,
        "training_regime": "joint_from_scratch" if task == "joint" else "independent_task_from_scratch",
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_parameters": trainable_names,
        "effective_flow_depth": (None if genome.architecture is not None else
                                 max(2, genome.flow_depth + genome.flow_depth % 2)
                                 if genome.flow == "fmt" else genome.flow_depth),
        "architecture_version": 2 if genome.architecture is not None else 1,
        "module_manifest": ([dict(graph=part, id=n["id"], kind=n["kind"], name=n["name"],
                                  params=n["params"], repeat=n["repeat"],
                                  output_rank=n["shape"].rank, output_width=n["shape"].width,
                                  parameters=sum(p.numel() for p in graph.layers[n["id"]].parameters()))
                             for part, graph in (("backbone", model.backbone), ("flow", model.flow))
                             for n in graph.plan] if genome.architecture is not None else None),
        "runtime_seconds": elapsed,
        "train_rows": splits.train.size,
        "validation_rows": splits.validation.size,
        "test_rows": test_rows,
        "fit_split": "train",
        "evaluation_split": "validation",
        "test_opened": test_opened,
        "steps": int(train_config.steps),
        "completed_steps": int(completed_steps),
        "train_steps": step_records,
        "validation_history": validation_history,
        "gradient_finite_parameters": int(gradient_finite_parameters),
        "gradient_nonzero_parameters": int(gradient_nonzero_parameters),
        "optimizer_changed_parameter": bool(
            tracked_before is not None
            and tracked_after is not None
            and not torch.equal(tracked_before, tracked_after)
        ),
        "validation": validation,
        "quality": finite_float(quality),
        "best_validation_quality": finite_float(best_quality),
        "loss_balance": {
            "mode": balancer.mode,
            "decay": balancer.decay,
            "ema": [float(value) for value in balancer.ema.detach().cpu()],
            "seen": [bool(value) for value in balancer.seen.detach().cpu()],
        },
        "status": "complete",
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "development_contract": (
            "unpaired_cross_sectional_population_flow_matching"
            if splits.development_train is not None
            else "legacy_row_aligned_regression"
        ),
        "stability": {
            "passed": not stability_failures and completed_steps == int(train_config.steps),
            "failures": stability_failures,
            "max_loss": float(max(
                [abs(float(x.get("total", 0.0))) for x in step_records]
                + [abs(float(x.get("total", 0.0))) for x in stability_failures],
                default=0.0,
            )),
            "max_gradient_norm": float(max(
                [float(x.get("gradient_norm", 0.0)) for x in step_records]
                + [float(x.get("gradient_norm", 0.0)) for x in stability_failures],
                default=0.0,
            )),
        },
        "status": "failed_stability" if stability_failures else "complete",
    }
    return receipt


def append_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True, ensure_ascii=True) + "\n")
