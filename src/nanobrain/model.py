"""A small four-task model assembled entirely through module APIs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import BrainBatch, LossBundle, PredictionBundle
from .development import DevelopmentPopulationBatch
from .genome import CandidateGenome
from .registry import registry
from . import modules as _modules  # noqa: F401 - installs built-in registrations


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


@dataclass(frozen=True)
class NanoBrainModelConfig:
    input_dim: int
    annotation_classes: int
    niche_classes: int
    integration_classes: int
    development_dim: int | None = None
    development_condition_dim: int = 1

    def __post_init__(self) -> None:
        for name in ("input_dim", "annotation_classes", "niche_classes", "integration_classes"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.development_dim is not None and int(self.development_dim) < 1:
            raise ValueError("development_dim must be positive")
        if int(self.development_condition_dim) < 1:
            raise ValueError("development_condition_dim must be positive")


class NanoBrainModel(nn.Module):
    def __init__(self, config: NanoBrainModelConfig, genome: CandidateGenome):
        super().__init__()
        self.config = config
        self.genome = genome.validate()
        from .objectives import default_objectives
        self.objective_specs = default_objectives(genome.flow_endpoint_weight, genome.flow_expression_weight,
                                                 genome.integration_adversarial_strength)
        if genome.architecture is not None:
            from .architecture import ModuleGraph, FlowGraph, GraphHead, Shape, checked_parameters, seeded_module, head_parameters
            spec = genome.to_dict()["architecture"]
            if config.input_dim != spec["input_dim"]:
                raise ValueError("architecture input_dim differs from dataset")
            self.backbone = ModuleGraph(spec["backbone"], {"features": Shape(config.input_dim)}, allow_spatial=True,
                                        seed=genome.seed, scope="backbone")
            self.flow = FlowGraph(spec["flow"], spec["width"], seed=genome.seed)
            pair = spec["pairing"]
            self.pairing = registry("pairing").create(pair["name"], **checked_parameters("pairing", pair["name"], pair["params"], {}))
            for name, output_dim in {"annotation": config.annotation_classes, "integration": config.integration_classes,
                                     "niche": config.niche_classes, "development": config.input_dim,
                                     "integration_reconstruction": config.input_dim,
                                     "completion": config.input_dim}.items():
                head = spec["heads"][name]
                params = head_parameters(head, spec["width"], output_dim)
                module = (GraphHead(head, spec["width"], output_dim, genome.seed, name) if "pre" in head else
                          seeded_module(genome.seed, "head/" + name,
                                        lambda head=head, params=params: registry("head").create(head["name"], **params)))
                setattr(self, name + "_head", module)
            self.objective_specs = spec["objectives"]
            return
        common = {
            "hidden_dim": genome.hidden_dim,
            "depth": genome.encoder_depth,
            "heads": genome.attention_heads,
            "dropout": genome.dropout,
        }
        self.encoder = registry("encoder").create(
            genome.encoder, input_dim=config.input_dim, **common
        )
        self.attention = registry("attention").create(genome.attention, **common)
        self.spatial = registry("spatial").create(genome.spatial, **common)
        self.residual = registry("residual").create(genome.residual, **common)
        self.pairing = registry("pairing").create(
            genome.pairing,
            epsilon=genome.sinkhorn_epsilon,
            iterations=genome.sinkhorn_iterations,
        )
        self.flow = registry("flow").create(
            genome.flow,
            hidden_dim=genome.hidden_dim,
            depth=genome.flow_depth,
            heads=genome.attention_heads,
            dropout=genome.dropout,
        )
        heads = registry("head")
        self.annotation_head = heads.create(
            genome.classification_head, hidden_dim=genome.hidden_dim,
            output_dim=config.annotation_classes, dropout=genome.dropout,
        )
        self.niche_head = heads.create(
            genome.classification_head, hidden_dim=genome.hidden_dim,
            output_dim=config.niche_classes, dropout=genome.dropout,
        )
        self.integration_head = heads.create(
            genome.classification_head, hidden_dim=genome.hidden_dim,
            output_dim=config.integration_classes, dropout=genome.dropout,
        )
        development_dim = config.development_dim or config.input_dim
        self.feature_to_development = nn.Linear(config.input_dim, genome.hidden_dim)
        self.development_condition_input = nn.Linear(
            config.development_condition_dim, genome.hidden_dim
        )
        self.development_null = nn.Parameter(torch.zeros(genome.hidden_dim))
        self.development_head = heads.create(
            genome.regression_head, hidden_dim=genome.hidden_dim,
            output_dim=development_dim, dropout=genome.dropout,
        )
        self.completion_head = heads.create(
            genome.regression_head, hidden_dim=genome.hidden_dim,
            output_dim=config.input_dim, dropout=genome.dropout,
        )

    def integration_adversarial_logits(
        self, embedding: Tensor, *, adversarial_strength: float | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the numerically bounded batch-adversarial branch.

        Reconstruction keeps the original embedding, while the discriminator
        receives a normalized copy.  A short linear warmup prevents the
        gradient-reversal signal from destabilizing a freshly initialized
        encoder.  The trainer sets ``_objective_step`` and the warmup length;
        direct objective/model calls intentionally start at warmup step zero.
        """
        if self.genome.architecture is not None:
            params = self.objective_specs["integration"]["params"]
            base_strength = float(
                params["adversarial_strength"] if adversarial_strength is None else adversarial_strength
            )
        else:
            base_strength = float(
                self.genome.integration_adversarial_strength
                if adversarial_strength is None else adversarial_strength
            )
        warmup_steps = max(1, int(getattr(self, "_integration_warmup_steps", 500)))
        step = max(0, int(getattr(self, "_objective_step", 0)))
        warmup_fraction = min(1.0, float(step + 1) / float(warmup_steps))
        strength = base_strength * warmup_fraction
        normalized = F.normalize(embedding.float(), dim=-1, eps=1.0e-6)
        logits = self.integration_head(_GradientReverse.apply(normalized, strength))
        logit_clip = float(getattr(self, "_integration_logit_clip", 20.0))
        logits = logits.clamp(min=-logit_clip, max=logit_clip)
        return logits, {
            "integration_embedding_norm": embedding.detach().float().norm(dim=-1).mean(),
            "integration_adversarial_embedding_norm": normalized.detach().norm(dim=-1).mean(),
            "integration_batch_logit_norm": logits.detach().float().norm(dim=-1).mean(),
            "integration_adversarial_strength": embedding.new_tensor(strength),
            "integration_adversarial_warmup_fraction": embedding.new_tensor(warmup_fraction),
        }

    def encode(self, batch: BrainBatch) -> Tensor:
        batch.validate()
        features = batch.features
        neighbors, neighbor_mask = batch.neighbors, batch.neighbor_mask
        context = batch.neighbor_features
        uses_spatial = self.backbone.uses_spatial if self.genome.architecture is not None else self.genome.spatial != "identity"
        if context is not None and uses_spatial:
            features = torch.cat((features, context), dim=0)
            if neighbors is None or neighbor_mask is None:
                raise ValueError("neighbor context requires neighbor indices and mask")
            neighbors = F.pad(neighbors, (0, 0, 0, len(context)), value=0)
            neighbor_mask = F.pad(neighbor_mask, (0, 0, 0, len(context)), value=False)
        if self.genome.architecture is not None:
            from .integration_components import METHODS, PREFIX
            views = {name: torch.as_tensor(batch.metadata[PREFIX + name][:len(features)],
                         device=features.device, dtype=features.dtype)
                     for name in METHODS if PREFIX + name in batch.metadata}
            return self.backbone({"features": features}, neighbors=neighbors, neighbor_mask=neighbor_mask,
                                 coordinates=batch.coordinates, integration_views=views)[:batch.batch_size]
        state = self.encoder(features)
        if state.ndim not in (2, 3) or state.shape[0] != len(features) or state.shape[-1] != self.genome.hidden_dim:
            raise ValueError("encoder must return [B,H] or [B,T,H] with configured H")
        expected_shape = state.shape
        state = self.attention(state)
        if state.shape != expected_shape:
            raise ValueError("attention must preserve the encoder output shape")
        if state.ndim == 3:
            state = state.mean(dim=1)
        state = self.spatial(
            state,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
            coordinates=batch.coordinates,
        )
        return self.residual(state[:batch.batch_size])

    def forward(self, batch: BrainBatch) -> PredictionBundle:
        embedding = self.encode(batch)
        if self.genome.architecture is not None:
            return PredictionBundle(embedding=embedding, annotation_logits=self.annotation_head(embedding),
                                    niche_logits=self.niche_head(embedding),
                                    integration_prediction=self.integration_adversarial_logits(embedding)[0],
                                    completion_prediction=self.completion_head(embedding))
        # Development labels are supervision only.  The state is conditioned
        # on observed metadata (or a learned null condition), never on the
        # target stored in ``batch.labels``.
        development_state = self.feature_to_development(batch.features)
        condition = batch.development_condition
        if condition is None:
            condition = embedding.new_zeros(
                (len(embedding), self.config.development_condition_dim)
            )
            condition_observed = torch.zeros(len(embedding), dtype=torch.bool, device=embedding.device)
        else:
            if condition.ndim == 1:
                condition = condition[:, None]
            if condition.shape[-1] != self.config.development_condition_dim:
                raise ValueError(
                    "development_condition width must be "
                    f"{self.config.development_condition_dim}, got {condition.shape[-1]}"
                )
            condition_observed = batch.development_condition_observed
            if condition_observed is None:
                condition_observed = torch.ones(len(embedding), dtype=torch.bool, device=embedding.device)
            else:
                condition_observed = condition_observed.bool()
        conditioned = self.development_condition_input(condition.to(embedding.dtype))
        null = self.development_null.to(dtype=conditioned.dtype, device=conditioned.device)
        development_state = development_state + torch.where(
            condition_observed[:, None], conditioned, null[None, :]
        )
        time = batch.development_time
        if time is None:
            time = embedding.new_zeros((len(embedding), 1))
        if time.ndim == 1:
            time = time[:, None]
        # Population FMT is trained and evaluated through explicit endpoint
        # batches below.  Do not run population attention on an arbitrary
        # row minibatch merely to populate the legacy age-regression field.
        velocity = (
            torch.zeros_like(development_state)
            if self.genome.flow == "fmt"
            else self.flow(development_state, embedding, time)
        )
        development = self.development_head(development_state + velocity)
        return PredictionBundle(
            embedding=embedding,
            annotation_logits=self.annotation_head(embedding),
            niche_logits=self.niche_head(embedding),
            integration_prediction=self.integration_adversarial_logits(embedding)[0],
            development_prediction=development,
            completion_prediction=self.completion_head(embedding),
            auxiliary={"development_velocity": velocity},
        )

    @staticmethod
    def _masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
        valid = target.ge(0)
        if mask is not None:
            valid = valid & mask
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], target[valid].long())

    @staticmethod
    def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
        valid = torch.isfinite(target)
        if mask is not None:
            while mask.ndim < valid.ndim:
                mask = mask.unsqueeze(-1)
            valid = valid & mask
        if not bool(valid.any()):
            return prediction.sum() * 0.0
        return F.mse_loss(prediction[valid], target[valid])

    def loss(
        self,
        batch: BrainBatch,
        prediction: PredictionBundle,
        *,
        include_row_development: bool = True,
    ) -> LossBundle:
        required = {"annotation", "niche", "integration", "development"}
        missing = required - set(batch.labels)
        if missing:
            raise ValueError(f"batch is missing four-task labels: {sorted(missing)}")
        terms = {
            "annotation": self._masked_cross_entropy(
                prediction.annotation_logits,
                batch.labels["annotation"],
                batch.label_masks.get("annotation"),
            ),
            "niche": self._masked_cross_entropy(
                prediction.niche_logits,
                batch.labels["niche"],
                batch.label_masks.get("niche"),
            ),
            "integration": self._masked_cross_entropy(
                prediction.integration_prediction,
                batch.labels["integration"],
                batch.label_masks.get("integration"),
            ),
        }
        if include_row_development:
            terms["development"] = self._masked_mse(
                prediction.development_prediction,
                batch.labels["development"],
                batch.label_masks.get("development"),
            )
        if batch.completion_target is not None and batch.completion_mask is not None:
            terms["completion"] = self._masked_mse(
                prediction.completion_prediction,
                batch.completion_target,
                batch.completion_mask,
            )
        return LossBundle(terms=terms, weights=self.genome.loss_weights)

    @staticmethod
    def _time_features(
        cells: int,
        tau: Tensor,
        source_time_days: float,
        target_time_days: float,
    ) -> Tensor:
        if tau.shape != (cells, 1):
            raise ValueError("flow time must have shape [cells,1]")
        source = tau.new_full((cells, 1), float(source_time_days) / 365.25)
        delta = tau.new_full(
            (cells, 1), float(target_time_days - source_time_days) / 365.25
        )
        return torch.cat((tau, source, delta), dim=-1)

    def population_flow_losses(
        self,
        batch: DevelopmentPopulationBatch,
        *,
        tau: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Conditional Flow Matching with detached population OT targets."""

        from .objectives import population_objective
        return population_objective(self, batch, tau=tau)

    def predict_development_population(
        self,
        source_batch: BrainBatch,
        source_time_days: float,
        target_time_days: float,
        *,
        steps: int | None = None,
        solver: str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Roll a source population to a future endpoint without target input."""

        steps = int(self.genome.rollout_steps if steps is None else steps)
        if solver is None:
            solver = self.genome.architecture["solver"]["name"] if self.genome.architecture is not None else "heun"
        if steps < 1 or solver not in {"euler", "heun"}:
            raise ValueError("rollout requires positive steps and euler/heun solver")
        if target_time_days <= source_time_days:
            raise ValueError("development rollout requires target_time > source_time")
        context = self.encode(source_batch)
        state = context
        step_size = 1.0 / steps
        for index in range(steps):
            tau = state.new_full((len(state), 1), index / steps)
            time = self._time_features(
                len(state), tau, source_time_days, target_time_days
            )
            first = self.flow(state, context, time)
            if solver == "euler":
                proposal = state + step_size * first
            else:
                predictor = state + step_size * first
                next_tau = state.new_full((len(state), 1), (index + 1) / steps)
                next_time = self._time_features(
                    len(state), next_tau, source_time_days, target_time_days
                )
                second = self.flow(predictor, context, next_time)
                proposal = state + 0.5 * step_size * (first + second)
            if not torch.isfinite(proposal).all():
                raise FloatingPointError("development rollout produced non-finite state")
            state = proposal
        return self.population_decoder(state), state

    @property
    def population_decoder(self):
        return self.development_head if self.genome.architecture is not None else self.completion_head

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
