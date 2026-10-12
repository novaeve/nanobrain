"""Official-code adapters for population developmental forecasting baselines."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base import AdapterContext, Availability, BaselineAdapter, TaskPrediction
from .common import array
from .registry import register


ROOT = Path(__file__).resolve().parents[3]
SCFIT_ROOT = ROOT / "reference_methods" / "scFiT"
SCNODE_ROOT = SCFIT_ROOT / "third_party" / "scNODE-main"
TRAJECTORYNET_ROOT = ROOT / "reference_methods" / "TrajectoryNet"


def _option(context: AdapterContext, name: str, default: Any, cast: Callable[[Any], Any]) -> Any:
    value = context.options.get(name, default)
    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid development adapter option {name}={value!r}") from exc


def _positive_option(context: AdapterContext, name: str, default: Any, cast: Callable[[Any], Any]) -> Any:
    value = _option(context, name, default, cast)
    if value <= 0:
        raise ValueError(f"development adapter option {name} must be positive")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(context: AdapterContext) -> torch.device:
    device = torch.device(context.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    return device


@dataclass(frozen=True)
class _ExpressionScale:
    mean: Tensor
    std: Tensor

    @classmethod
    def load(cls, context: AdapterContext, device: torch.device) -> "_ExpressionScale":
        mean = torch.from_numpy(
            np.load(context.data_root / "normalization_mean.npy", allow_pickle=False).astype(np.float32)
        ).to(device)
        std = torch.from_numpy(
            np.load(context.data_root / "normalization_std.npy", allow_pickle=False).astype(np.float32)
        ).to(device)
        if mean.shape != (context.train.input_dim,) or std.shape != mean.shape:
            raise ValueError("expression normalization sidecars do not match the feature panel")
        return cls(mean=mean, std=std.clamp_min(1.0e-6))

    def native(self, value: Tensor) -> Tensor:
        return (value.float() * self.std + self.mean).clamp_min(0.0)

    def benchmark(self, value: Tensor) -> Tensor:
        return (value.float() - self.mean) / self.std


@dataclass(frozen=True)
class _TimeScale:
    denominator: float
    maximum_delta: float

    @classmethod
    def from_training(cls, dataset: Any) -> "_TimeScale":
        values = [
            float(group[key])
            for group in dataset.groups
            for key in ("source_time_days", "target_time_days")
        ]
        deltas = [
            float(group["target_time_days"]) - float(group["source_time_days"])
            for group in dataset.groups
        ]
        denominator = max(max(abs(value) for value in values), 1.0)
        maximum_delta = max(max(deltas), 1.0)
        return cls(denominator=denominator, maximum_delta=maximum_delta)

    def absolute(self, value: float) -> float:
        return float(value) / self.denominator

    def delta(self, source: float, target: float) -> float:
        return max(float(target) - float(source), 1.0e-6) / self.maximum_delta


class _TrainingLog:
    def __init__(self, context: AdapterContext, method: str) -> None:
        self.path = context.output_root / f"{method}_training_log.jsonl"

    def write(self, **payload: Any) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _population_prediction(
    context: AdapterContext,
    *,
    method: str,
    predictor: Callable[[Any, int], Tensor],
    fit_rows: int,
    notes: str,
    training_regime: str,
) -> TaskPrediction:
    development = context.development_validation
    if development is None:
        raise RuntimeError("validation development populations are unavailable")
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    pointer = [0]
    for group_index in range(len(development)):
        batch = development.batch(group_index)
        predicted = predictor(batch, group_index)
        expected_shape = batch.target.features.shape
        if tuple(predicted.shape) != tuple(expected_shape):
            raise RuntimeError(
                f"{method} returned {tuple(predicted.shape)} for development group "
                f"{group_index}, expected {tuple(expected_shape)}"
            )
        truth.append(array(batch.target.features, np.float32))
        prediction.append(array(predicted, np.float32))
        pointer.append(pointer[-1] + batch.cells)
    target = np.concatenate(truth, axis=0)
    predicted = np.concatenate(prediction, axis=0)
    if not np.isfinite(predicted).all():
        raise FloatingPointError(f"{method} produced non-finite validation predictions")
    return TaskPrediction(
        task="development",
        arrays={
            "target": target,
            "prediction": predicted,
            "group_ptr": np.asarray(pointer, dtype=np.int64),
            "population_unpaired": np.asarray([True]),
        },
        fit_rows=int(fit_rows),
        eval_rows=int(len(target)),
        notes=notes,
        training_regime=training_regime,
    )


def _checkpoint_directory(context: AdapterContext) -> Path:
    directory = context.output_root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _frozen_checkpoint(
    context: AdapterContext, method: str
) -> tuple[Path, dict[str, Any]] | None:
    value = context.options.get(f"{method}_checkpoint")
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen {method} checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("test_opened") is not False:
        raise ValueError(f"frozen {method} checkpoint does not attest sealed test: {path}")
    return path, payload


def _load_scfit() -> tuple[type[nn.Module], Any]:
    source = str(SCFIT_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from scfit.model import scFiTModel
    from scfit import utils

    return scFiTModel, utils


def _load_scnode() -> tuple[Callable[..., nn.Module], Callable[..., Tensor]]:
    source = str(SCNODE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    from optim.loss_func import SinkhornLoss
    from optim.running import constructscNODEModel

    return constructscNODEModel, SinkhornLoss


def _load_trajectorynet() -> tuple[Callable[..., nn.Module], Callable[[Tensor], Tensor]]:
    source = str(TRAJECTORYNET_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    from TrajectoryNet.train_misc import build_model_tabular, standard_normal_logprob

    return build_model_tabular, standard_normal_logprob


class _PopulationForecastAdapter(BaselineAdapter):
    family = "development"
    tasks = ("development",)
    required_path: Path
    required_modules: tuple[str, ...] = ()

    def availability(self, context: AdapterContext) -> Availability:
        if context.development_train is None or context.development_validation is None:
            return Availability(
                "data_missing",
                "materialize split-local development population sidecars before running this method",
            )
        if not self.required_path.exists():
            return Availability("source_missing", f"official source is missing: {self.required_path}")
        missing = [name for name in self.required_modules if importlib.util.find_spec(name) is None]
        if missing:
            return Availability("dependency_missing", f"install {', '.join(missing)}")
        return Availability("ready")


@register("scfit")
class ScFiTAdapter(_PopulationForecastAdapter):
    method_id = "scfit"
    display_name = "scFiT"
    required_path = SCFIT_ROOT / "src" / "scfit" / "model.py"
    required_modules = ("geomloss",)

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "library": "official scFiT source", "library_version": "local snapshot"}

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        from geomloss import SamplesLoss

        development = context.development_train
        assert development is not None
        _seed_everything(context.seed)
        device = _device(context)
        scale = _ExpressionScale.load(context, device)
        time_scale = _TimeScale.from_training(development)
        model_class, utils = _load_scfit()
        frozen = _frozen_checkpoint(context, self.method_id)

        d_model = _positive_option(context, "scfit_d_model", 256, int)
        heads = _positive_option(context, "scfit_heads", 8, int)
        layers = _positive_option(context, "scfit_layers", 4, int)
        d_ff = _positive_option(context, "scfit_d_ff", d_model * 4, int)
        if frozen is not None:
            architecture = frozen[1].get("architecture", {})
            d_model = int(architecture.get("d_model", d_model))
            heads = int(architecture.get("heads", heads))
            layers = int(architecture.get("layers", layers))
            d_ff = int(architecture.get("d_ff", d_ff))
        if d_model % heads:
            raise ValueError("scfit_d_model must be divisible by scfit_heads")
        steps = _positive_option(context, "scfit_steps", 2000, int)
        if frozen is not None:
            steps = int(frozen[1]["steps"])
        cells = _positive_option(context, "scfit_cells", 24, int)
        learning_rate = _positive_option(context, "scfit_lr", 2.0e-4, float)
        rollout_steps = _positive_option(context, "scfit_rollout_steps", 10, int)
        epsilon = _positive_option(context, "scfit_sinkhorn_epsilon", 0.05, float)
        sinkhorn_iterations = _positive_option(context, "scfit_sinkhorn_iterations", 100, int)
        use_amp = bool(context.options.get("development_bf16", True)) and device.type == "cuda"

        model = model_class(
            n_genes=context.train.input_dim,
            top_k_genes=context.train.input_dim,
            d_model=d_model,
            nhead=heads,
            d_ff=d_ff,
            nlayers=layers,
            dropout=float(context.options.get("scfit_dropout", 0.1)),
            value_max=float(context.options.get("scfit_value_max", 32.0)),
            time_conditioning_mode="adaln",
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=float(context.options.get("scfit_weight_decay", 1.0e-4))
        )
        if frozen is not None:
            model.load_state_dict(frozen[1]["model"])
        population_loss = SamplesLoss(
            "sinkhorn", p=2, blur=epsilon, scaling=0.5, debias=True, backend="tensorized"
        )
        gene_ids_template = torch.cat(
            (
                torch.tensor([model.cls_token_id], device=device, dtype=torch.long),
                torch.arange(context.train.input_dim, device=device, dtype=torch.long),
            )
        )
        mask_states_template = torch.ones(context.train.input_dim + 1, device=device, dtype=torch.long)
        mask_states_template[0] = 3
        padding_template = torch.zeros(context.train.input_dim + 1, device=device, dtype=torch.bool)
        log = _TrainingLog(context, self.method_id)
        rng = np.random.default_rng(context.seed)

        def raw_prediction(value: Tensor, tau: Tensor, source_time: float, target_time: float) -> Tensor:
            count = len(value)
            sequence = F.pad(value, (1, 0))
            gene_ids = gene_ids_template.unsqueeze(0).expand(count, -1)
            mask_states = mask_states_template.unsqueeze(0).expand(count, -1)
            padding = padding_template.unsqueeze(0).expand(count, -1)
            source_condition = tau.new_full((count, 1), time_scale.absolute(source_time))
            target_condition = tau.new_full((count, 1), time_scale.absolute(target_time))
            output = model(
                gene_ids,
                sequence,
                mask_states,
                padding,
                tau,
                t_src=source_condition,
                t_tgt=target_condition,
            )
            return output[:, 1:]

        if frozen is None:
            model.train()
            for step in range(steps):
                group_index = int(rng.integers(len(development)))
                batch = development.batch(group_index, max_cells=cells, seed=context.seed + step).to(device)
                source = scale.native(batch.source.features)
                target = scale.native(batch.target.features)
                with torch.no_grad():
                    cost = utils.pairwise_mean_sq_dist(source, target)
                    plan = utils.sinkhorn_uniform_transport(
                        cost, epsilon=epsilon, max_iter=sinkhorn_iterations
                    )
                    source_index, target_index = utils.sample_transport_pairs(plan, len(source))
                    paired_source = source[source_index]
                    paired_target = target[target_index]
                tau = utils.sample_t_stratified(len(source), device)
                state = (1.0 - tau) * paired_source + tau * paired_target
                target_velocity = paired_target - paired_source
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    velocity = raw_prediction(
                        state, tau, batch.source_time_days, batch.target_time_days
                    )
                    endpoint_prediction = (state + (1.0 - tau) * velocity).clamp_min(0.0)
                    cfm = F.mse_loss(velocity.float(), target_velocity.float())
                    wasserstein = population_loss(endpoint_prediction.float(), target.float())
                    mean_loss = F.mse_loss(endpoint_prediction.float().mean(0), target.float().mean(0))
                    loss = cfm + 0.5 * wasserstein + 0.1 * mean_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"scFiT loss became non-finite at step {step + 1}")
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
                if step == 0 or step + 1 == steps or (step + 1) % max(1, steps // 20) == 0:
                    log.write(
                        step=step + 1,
                        loss=float(loss.detach()),
                        cfm=float(cfm.detach()),
                        sinkhorn=float(wasserstein.detach()),
                        mean=float(mean_loss.detach()),
                        gradient_norm=gradient_norm,
                        interval_id=batch.interval_id,
                    )

            checkpoint = _checkpoint_directory(context) / "scfit.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": steps,
                    "seed": context.seed,
                    "architecture": {"d_model": d_model, "heads": heads, "layers": layers, "d_ff": d_ff},
                    "time_scale": time_scale.__dict__,
                    "test_opened": False,
                },
                checkpoint,
            )
        else:
            checkpoint = frozen[0]

        @torch.no_grad()
        def predict(batch: Any, _: int) -> Tensor:
            model.eval()
            state = scale.native(batch.source.features.to(device))
            step_size = 1.0 / rollout_steps
            for index in range(rollout_steps):
                tau = state.new_full((len(state), 1), index / rollout_steps)
                velocity = raw_prediction(
                    state, tau, batch.source_time_days, batch.target_time_days
                )
                state = (state + step_size * velocity).clamp_min(0.0)
            return scale.benchmark(state).cpu()

        checkpoint_note = str(checkpoint)
        notes = (
            f"official scFiT AdaLN Transformer; direct-velocity OT conditional flow matching; "
            f"{steps} train-only interval steps; official Euler-{rollout_steps} rollout; "
            f"frozen_checkpoint={frozen is not None}; checkpoint={checkpoint_note}"
        )
        return [
            _population_prediction(
                context,
                method=self.method_id,
                predictor=predict,
                fit_rows=len(development.source_index),
                notes=notes,
                training_regime=(
                    "frozen_checkpoint_population_cfm"
                    if frozen is not None
                    else "train_from_scratch_population_cfm"
                ),
            )
        ]


@register("scnode")
class ScNODEAdapter(_PopulationForecastAdapter):
    method_id = "scnode"
    display_name = "scNODE"
    required_path = SCNODE_ROOT / "model" / "dynamic_model.py"
    required_modules = ("geomloss", "torchdiffeq")

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "library": "official scNODE source", "library_version": "local snapshot"}

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        development = context.development_train
        assert development is not None
        _seed_everything(context.seed)
        device = _device(context)
        scale = _ExpressionScale.load(context, device)
        time_scale = _TimeScale.from_training(development)
        construct_model, sinkhorn_loss = _load_scnode()
        frozen = _frozen_checkpoint(context, self.method_id)

        latent_dim = _positive_option(context, "scnode_latent_dim", 32, int)
        hidden_dim = _positive_option(context, "scnode_hidden_dim", 128, int)
        if frozen is not None:
            architecture = frozen[1].get("architecture", {})
            latent_dim = int(architecture.get("latent_dim", latent_dim))
            hidden_dim = int(architecture.get("hidden_dim", hidden_dim))
        cells = _positive_option(context, "scnode_cells", 64, int)
        pretrain_steps = _positive_option(context, "scnode_pretrain_steps", 500, int)
        steps = _positive_option(context, "scnode_steps", 2000, int)
        if frozen is not None:
            pretrain_steps = int(frozen[1]["pretrain_steps"])
            steps = int(frozen[1]["steps"])
        learning_rate = _positive_option(context, "scnode_lr", 1.0e-3, float)
        blur = _positive_option(context, "scnode_sinkhorn_blur", 0.05, float)
        model = construct_model(
            n_genes=context.train.input_dim,
            latent_dim=latent_dim,
            enc_latent_list=[hidden_dim],
            dec_latent_list=[hidden_dim],
            drift_latent_size=[hidden_dim],
            latent_enc_act="relu",
            latent_dec_act="relu",
            drift_act="relu",
            ode_method="rk4",
        ).to(device)
        if frozen is not None:
            model.load_state_dict(frozen[1]["model"])
        log = _TrainingLog(context, self.method_id)
        rng = np.random.default_rng(context.seed)

        endpoints = np.unique(
            np.concatenate((development.source_index, development.target_index)).astype(np.int64)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.95, 0.99))
        if frozen is None:
            pretrain_optimizer = torch.optim.Adam(
                list(model.latent_encoder.parameters()) + list(model.obs_decoder.parameters()),
                lr=learning_rate,
                betas=(0.95, 0.99),
            )
            model.train()
            for step in range(pretrain_steps):
                chosen = rng.choice(endpoints, size=min(cells, len(endpoints)), replace=len(endpoints) < cells)
                value = scale.native(development.dataset.features[torch.from_numpy(chosen)].to(device))
                pretrain_optimizer.zero_grad(set_to_none=True)
                mean, std = model.latent_encoder(value)
                sample = model._sampleGaussian(mean, std)
                reconstruction = model.obs_decoder(sample)
                mse = F.mse_loss(reconstruction, value)
                kl = (std.square() + mean.square() - 2.0 * torch.log(std + 1.0e-5)).mean()
                loss = mse + 1.0e-4 * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"scNODE VAE loss became non-finite at step {step + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                pretrain_optimizer.step()
                if step == 0 or step + 1 == pretrain_steps or (step + 1) % max(1, pretrain_steps // 10) == 0:
                    log.write(phase="vae", step=step + 1, loss=float(loss.detach()), mse=float(mse.detach()), kl=float(kl.detach()))

            for step in range(steps):
                group_index = int(rng.integers(len(development)))
                batch = development.batch(group_index, max_cells=cells, seed=context.seed + step).to(device)
                source = scale.native(batch.source.features)
                target = scale.native(batch.target.features)
                times = torch.tensor(
                    [0.0, time_scale.delta(batch.source_time_days, batch.target_time_days)],
                    dtype=torch.float32,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                reconstructed, _, _, latent_sequence = model([source, target], times, batch_size=None)
                distribution = sinkhorn_loss(
                    [source, target], reconstructed, blur=blur, scaling=0.5, batch_size=None
                )
                source_mean, _ = model.latent_encoder(source)
                target_mean, _ = model.latent_encoder(target)
                dynamic = sinkhorn_loss(
                    [source_mean, target_mean], latent_sequence, blur=blur, scaling=0.5, batch_size=None
                )
                loss = distribution + 0.1 * dynamic
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"scNODE loss became non-finite at step {step + 1}")
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                optimizer.step()
                if step == 0 or step + 1 == steps or (step + 1) % max(1, steps // 20) == 0:
                    log.write(
                        phase="ode",
                        step=step + 1,
                        loss=float(loss.detach()),
                        sinkhorn=float(distribution.detach()),
                        dynamic=float(dynamic.detach()),
                        gradient_norm=gradient_norm,
                        interval_id=batch.interval_id,
                    )

            checkpoint = _checkpoint_directory(context) / "scnode.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pretrain_steps": pretrain_steps,
                    "steps": steps,
                    "seed": context.seed,
                    "architecture": {"latent_dim": latent_dim, "hidden_dim": hidden_dim},
                    "time_scale": time_scale.__dict__,
                    "test_opened": False,
                },
                checkpoint,
            )
        else:
            checkpoint = frozen[0]

        @torch.no_grad()
        def predict(batch: Any, group_index: int) -> Tensor:
            model.eval()
            torch.manual_seed(context.seed + group_index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(context.seed + group_index)
            source = scale.native(batch.source.features.to(device))
            times = torch.tensor(
                [0.0, time_scale.delta(batch.source_time_days, batch.target_time_days)],
                dtype=torch.float32,
                device=device,
            )
            _, _, output = model.predict(source, times, n_cells=batch.cells)
            return scale.benchmark(output[:, -1]).cpu()

        notes = (
            f"official scNODE VAE plus latent neural ODE; {pretrain_steps} VAE and {steps} train-only population steps; "
            f"frozen_checkpoint={frozen is not None}; checkpoint={checkpoint}"
        )
        return [
            _population_prediction(
                context,
                method=self.method_id,
                predictor=predict,
                fit_rows=len(development.source_index),
                notes=notes,
                training_regime=(
                    "frozen_checkpoint_vae_latent_ode"
                    if frozen is not None
                    else "train_from_scratch_vae_latent_ode"
                ),
            )
        ]


@register("trajectorynet")
class TrajectoryNetAdapter(_PopulationForecastAdapter):
    method_id = "trajectorynet"
    display_name = "TrajectoryNet"
    required_path = TRAJECTORYNET_ROOT / "TrajectoryNet" / "train_misc.py"
    required_modules = ("torchdiffeq",)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "library": "official TrajectoryNet source",
            "library_version": "810c89b081f95405bc0ad42bfb3cb41038e13700",
        }

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import joblib
        from geomloss import SamplesLoss
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        development = context.development_train
        assert development is not None
        _seed_everything(context.seed)
        device = _device(context)
        build_model, standard_normal_logprob = _load_trajectorynet()
        frozen = _frozen_checkpoint(context, self.method_id)
        dimensions = _positive_option(context, "trajectorynet_dimensions", 10, int)
        dimensions = min(dimensions, context.train.input_dim)
        hidden = str(context.options.get("trajectorynet_hidden", "64-64-64"))
        steps = _positive_option(context, "trajectorynet_steps", 2000, int)
        if frozen is not None:
            dimensions = int(frozen[1]["dimensions"])
            hidden = str(frozen[1]["hidden"])
            steps = int(frozen[1]["steps"])
        cells = _positive_option(context, "trajectorynet_cells", 64, int)
        learning_rate = _positive_option(context, "trajectorynet_lr", 1.0e-3, float)
        ot_weight = _positive_option(context, "trajectorynet_ot_weight", 1.0, float)
        time_scale = _TimeScale.from_training(development)

        fit_index = np.unique(
            np.concatenate((development.source_index, development.target_index)).astype(np.int64)
        )
        if frozen is None:
            fit_values = array(development.dataset.features[torch.from_numpy(fit_index)], np.float32)
            pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=context.seed)
            scaler = StandardScaler()
            scaler.fit(pca.fit_transform(fit_values))
        else:
            preprocessing_value = context.options.get("trajectorynet_preprocessing")
            preprocessing = (
                Path(str(preprocessing_value)).expanduser().resolve()
                if preprocessing_value not in (None, "")
                else frozen[0].with_name("trajectorynet_preprocessing.joblib")
            )
            if not preprocessing.is_file():
                raise FileNotFoundError(
                    f"frozen TrajectoryNet preprocessing does not exist: {preprocessing}"
                )
            preprocessing_payload = joblib.load(preprocessing)
            pca = preprocessing_payload["pca"]
            scaler = preprocessing_payload["scaler"]
            fit_index = np.asarray(preprocessing_payload["fit_index"], dtype=np.int64)

        args = Namespace(
            dims=hidden,
            layer_type="concatsquash",
            nonlinearity="tanh",
            divergence_fn="brute_force",
            residual=False,
            rademacher=False,
            time_scale=1.0,
            train_T=False,
            solver="rk4",
            num_blocks=1,
            batch_norm=False,
            bn_lag=0.0,
            atol=1.0e-5,
            rtol=1.0e-5,
            step_size=0.25,
            test_solver="rk4",
            test_atol=1.0e-5,
            test_rtol=1.0e-5,
        )
        model = build_model(args, dimensions, regularization_fns=None).to(device)
        if frozen is not None:
            model.load_state_dict(frozen[1]["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-5)
        distribution_loss = SamplesLoss(
            "sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized"
        )
        log = _TrainingLog(context, self.method_id)
        rng = np.random.default_rng(context.seed)

        def transform(value: Tensor) -> Tensor:
            values = array(value, np.float32)
            return torch.from_numpy(scaler.transform(pca.transform(values)).astype(np.float32)).to(device)

        if frozen is None:
            model.train()
            for step in range(steps):
                group_index = int(rng.integers(len(development)))
                batch = development.batch(group_index, max_cells=cells, seed=context.seed + step)
                source = transform(batch.source.features)
                target = transform(batch.target.features)
                interval = torch.tensor(
                    [0.0, time_scale.delta(batch.source_time_days, batch.target_time_days)],
                    dtype=torch.float32,
                    device=device,
                )
                zero = torch.zeros((len(target), 1), dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                latent, delta_log_probability = model(
                    target, zero, integration_times=interval, reverse=False
                )
                log_probability = standard_normal_logprob(latent).sum(1, keepdim=True) - delta_log_probability
                negative_log_likelihood = -log_probability.mean()
                predicted, _ = model(
                    source,
                    torch.zeros((len(source), 1), dtype=torch.float32, device=device),
                    integration_times=interval,
                    reverse=True,
                )
                sinkhorn = distribution_loss(predicted, target)
                loss = negative_log_likelihood + ot_weight * sinkhorn
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"TrajectoryNet loss became non-finite at step {step + 1}")
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                optimizer.step()
                if step == 0 or step + 1 == steps or (step + 1) % max(1, steps // 20) == 0:
                    log.write(
                        step=step + 1,
                        loss=float(loss.detach()),
                        negative_log_likelihood=float(negative_log_likelihood.detach()),
                        sinkhorn=float(sinkhorn.detach()),
                        gradient_norm=gradient_norm,
                        interval_id=batch.interval_id,
                    )

            checkpoint = _checkpoint_directory(context) / "trajectorynet.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": steps,
                    "seed": context.seed,
                    "dimensions": dimensions,
                    "hidden": hidden,
                    "time_scale": time_scale.__dict__,
                    "test_opened": False,
                },
                checkpoint,
            )
            preprocessing = _checkpoint_directory(context) / "trajectorynet_preprocessing.joblib"
            joblib.dump({"pca": pca, "scaler": scaler, "fit_index": fit_index}, preprocessing)
        else:
            checkpoint = frozen[0]

        @torch.no_grad()
        def predict(batch: Any, _: int) -> Tensor:
            model.eval()
            source = transform(batch.source.features)
            interval = torch.tensor(
                [0.0, time_scale.delta(batch.source_time_days, batch.target_time_days)],
                dtype=torch.float32,
                device=device,
            )
            output, _ = model(
                source,
                torch.zeros((len(source), 1), dtype=torch.float32, device=device),
                integration_times=interval,
                reverse=True,
            )
            restored = pca.inverse_transform(scaler.inverse_transform(array(output, np.float32)))
            return torch.from_numpy(restored.astype(np.float32, copy=False))

        notes = (
            f"official TrajectoryNet continuous normalizing flow; train-only PCA({dimensions}) plus whitening; "
            f"{steps} interval-density steps; frozen_checkpoint={frozen is not None}; checkpoint={checkpoint}"
        )
        return [
            _population_prediction(
                context,
                method=self.method_id,
                predictor=predict,
                fit_rows=len(fit_index),
                notes=notes,
                training_regime=(
                    "frozen_checkpoint_cnf_population"
                    if frozen is not None
                    else "train_from_scratch_cnf_population"
                ),
            )
        ]


__all__ = ["ScFiTAdapter", "ScNODEAdapter", "TrajectoryNetAdapter"]
