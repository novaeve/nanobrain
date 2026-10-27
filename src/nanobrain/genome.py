"""Serializable architecture and hyperparameter genome for nanoBrain."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import random
from typing import Any


SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "encoder": ("identity", "mlp", "residual_mlp", "scbrain_swiglu", "scnode_mlp", "scvi_fc", "panel_patches"),
    "attention": ("identity", "sdpa", "differential_linear", "scfit_block", "bert", "torch_transformer"),
    "spatial": ("identity", "neighbor_mean", "scbrain_gcn", "graphst_sparse"),
    "residual": ("none", "nbeats"),
    "pairing": ("identity", "sinkhorn"),
    "flow": ("none", "mlp", "fmt", "scfit_adaln", "scnode_time_mlp", "trajectorynet", "scuniflow_field"),
    "classification_head": ("classification", "scbrain", "scfit"),
    "regression_head": ("regression", "scfit"),
    "hidden_dim": (32, 64, 96, 128),
    "encoder_depth": (1, 2, 3),
    "attention_heads": (1, 2, 4),
    "flow_depth": (1, 2, 3, 4),
    "dropout": (0.0, 0.05, 0.1),
    "learning_rate": (1e-3, 5e-4, 2e-4, 1e-4),
    "sinkhorn_epsilon": (0.03, 0.05, 0.1),
    "sinkhorn_iterations": (20, 30, 50),
    "rollout_steps": (4, 8, 12),
    "flow_endpoint_weight": (0.5, 1.0, 2.0),
    "flow_expression_weight": (0.25, 0.5, 1.0),
    "integration_adversarial_strength": (0.25, 0.5, 1.0),
}


@dataclass
class CandidateGenome:
    encoder: str = "mlp"
    attention: str = "differential_linear"
    spatial: str = "neighbor_mean"
    residual: str = "nbeats"
    pairing: str = "sinkhorn"
    flow: str = "fmt"
    classification_head: str = "classification"
    regression_head: str = "regression"
    hidden_dim: int = 64
    encoder_depth: int = 2
    attention_heads: int = 4
    flow_depth: int = 2
    dropout: float = 0.05
    learning_rate: float = 2e-4
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 30
    rollout_steps: int = 8
    flow_endpoint_weight: float = 1.0
    flow_expression_weight: float = 0.5
    integration_adversarial_strength: float = 0.5
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "integration": 1.0,
            "annotation": 1.0,
            "niche": 1.0,
            "development": 1.0,
            "completion": 1.0,
        }
    )
    seed: int = 7
    architecture: dict[str, Any] | None = None

    def validate(self) -> "CandidateGenome":
        if self.architecture is not None:
            from .architecture import validate_architecture
            validate_architecture(self.architecture)
            if self.learning_rate not in SEARCH_SPACE["learning_rate"]:
                raise ValueError("unsupported learning_rate")
            self.hidden_dim = self.architecture["width"]
            self.rollout_steps = self.architecture["solver"]["steps"]
            self.flow = "graph"
            return self
        plugin_kinds = {"encoder": "encoder", "attention": "attention", "spatial": "spatial", "residual": "residual", "pairing": "pairing", "flow": "flow", "classification_head": "head", "regression_head": "head"}
        from .registry import registry
        for key, kind in plugin_kinds.items():
            value = getattr(self, key)
            if value not in SEARCH_SPACE[key] and value not in registry(kind).names():
                raise ValueError(f"unsupported {key}: {value!r}")
        for key in (
            "hidden_dim", "encoder_depth", "attention_heads", "flow_depth",
            "sinkhorn_iterations", "rollout_steps",
        ):
            value = int(getattr(self, key))
            if value not in SEARCH_SPACE[key]:
                raise ValueError(f"unsupported {key}: {value}")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if float(self.dropout) not in SEARCH_SPACE["dropout"]:
            raise ValueError(f"unsupported dropout: {self.dropout}")
        if float(self.learning_rate) not in SEARCH_SPACE["learning_rate"]:
            raise ValueError(f"unsupported learning_rate: {self.learning_rate}")
        for key in (
            "sinkhorn_epsilon", "flow_endpoint_weight", "flow_expression_weight",
            "integration_adversarial_strength",
        ):
            if float(getattr(self, key)) not in SEARCH_SPACE[key]:
                raise ValueError(f"unsupported {key}: {getattr(self, key)}")
        required = {"integration", "annotation", "niche", "development"}
        allowed = required | {"completion"}
        if not required <= set(self.loss_weights) <= allowed:
            raise ValueError("loss_weights must contain the four task names and may contain completion")
        if any(float(value) < 0 for value in self.loss_weights.values()):
            raise ValueError("loss weights must be non-negative")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        if self.architecture is not None:
            from .architecture import canonical_architecture
            return {"architecture": canonical_architecture(self.architecture),
                    "learning_rate": self.learning_rate, "seed": self.seed}
        payload = asdict(self)
        payload.pop("architecture")  # preserve historical candidate/checkpoint hashes
        payload["loss_weights"] = {key: float(value) for key, value in sorted(self.loss_weights.items())}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateGenome":
        if payload.get("architecture") is not None:
            unknown = set(payload) - {"architecture", "learning_rate", "seed"}
            if unknown:
                raise ValueError(f"graph genome rejects unused legacy fields: {sorted(unknown)}")
        allowed = {field for field in cls.__dataclass_fields__}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown genome fields: {sorted(unknown)}")
        values = {key: value for key, value in payload.items() if key in allowed}
        genome = cls(**values)
        return genome.validate()

    @property
    def genome_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def mutate(self, rng: random.Random, rate: float = 0.25) -> "CandidateGenome":
        if self.architecture is not None:
            import copy
            for _ in range(32):
                child = copy.deepcopy(self.to_dict())
                graph = child["architecture"][rng.choice(("backbone", "flow"))]
                node = rng.choice(graph["nodes"])
                if rng.random() < .5:
                    node["repeat"] = rng.choice((1, 2, 3))
                else:
                    from .registry import registry
                    node["name"] = rng.choice(registry(node["kind"]).names())
                    node["params"] = {}
                child["seed"] = rng.randrange(1, 2**31 - 1)
                try:
                    return CandidateGenome.from_dict(child)
                except (ValueError, KeyError, TypeError):
                    continue
            raise ValueError("could not construct a valid graph mutation")
        child = self.to_dict()
        for key, choices in SEARCH_SPACE.items():
            if rng.random() >= float(rate):
                continue
            current = child[key]
            alternatives = [value for value in choices if value != current]
            if alternatives:
                child[key] = rng.choice(alternatives)
        child["seed"] = int(rng.randrange(1, 2**31 - 1))
        return CandidateGenome.from_dict(child)

    def crossover(self, other: "CandidateGenome", rng: random.Random) -> "CandidateGenome":
        if self.architecture is not None or other.architecture is not None:
            if self.architecture is None or other.architecture is None:
                raise ValueError("migrate both parents before graph crossover")
            import copy
            child = copy.deepcopy(self.to_dict())
            for part in ("backbone", "flow", "heads", "objectives", "pairing", "solver"):
                if rng.random() < .5:
                    child["architecture"][part] = copy.deepcopy(other.to_dict()["architecture"][part])
            child["seed"] = rng.randrange(1, 2**31 - 1)
            return CandidateGenome.from_dict(child)
        left, right = self.to_dict(), other.to_dict()
        child = {
            key: (left[key] if rng.random() < 0.5 else right[key])
            for key in SEARCH_SPACE
        }
        child["loss_weights"] = {
            key: (left["loss_weights"][key] if rng.random() < 0.5 else right["loss_weights"][key])
            for key in left["loss_weights"]
        }
        child["seed"] = int(rng.randrange(1, 2**31 - 1))
        return CandidateGenome.from_dict(child)


def default_genome(seed: int = 7) -> CandidateGenome:
    return CandidateGenome(seed=int(seed)).validate()
