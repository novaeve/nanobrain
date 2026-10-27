"""Compact virtual-cell models, data contracts, and task components."""

from .campaign import (
    PRIMARY_METRICS,
    append_generation_baseline,
    baseline_regression_gate,
    generation_baseline_record,
    load_generation_baseline,
    strongest_baselines,
)
from .contracts import BrainBatch, LossBundle, PredictionBundle
from .data import DatasetSplits, NanoDataset, infer_label_cardinalities, load_v3_subset
from .development import DevelopmentPairDataset, DevelopmentPopulationBatch
from .evolution import architecture_id, improvement_gate, joint_score
from .genome import CandidateGenome, default_genome
from .model import NanoBrainModel, NanoBrainModelConfig
from .paths import DatasetLocation, describe_v3_shards, locate_v3_shards
from .registry import REGISTRIES, ModuleRegistry, backend_catalog

__all__ = [
    "PRIMARY_METRICS",
    "REGISTRIES",
    "BrainBatch",
    "CandidateGenome",
    "DatasetLocation",
    "DatasetSplits",
    "DevelopmentPairDataset",
    "DevelopmentPopulationBatch",
    "LossBundle",
    "ModuleRegistry",
    "NanoBrainModel",
    "NanoBrainModelConfig",
    "NanoDataset",
    "PredictionBundle",
    "append_generation_baseline",
    "architecture_id",
    "backend_catalog",
    "baseline_regression_gate",
    "default_genome",
    "describe_v3_shards",
    "generation_baseline_record",
    "improvement_gate",
    "infer_label_cardinalities",
    "joint_score",
    "load_generation_baseline",
    "load_v3_subset",
    "locate_v3_shards",
    "strongest_baselines",
]
