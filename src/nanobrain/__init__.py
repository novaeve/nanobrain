"""Compact virtual-cell models, data contracts, and task components."""

from .contracts import BrainBatch, LossBundle, PredictionBundle
from .genome import CandidateGenome, default_genome
from .model import NanoBrainModel, NanoBrainModelConfig
from .registry import REGISTRIES, ModuleRegistry, backend_catalog
from .paths import DatasetLocation, describe_v3_shards, locate_v3_shards
from .data import DatasetSplits, NanoDataset, infer_label_cardinalities, load_v3_subset
from .development import DevelopmentPairDataset, DevelopmentPopulationBatch

__all__ = [
    "BrainBatch",
    "CandidateGenome",
    "LossBundle",
    "ModuleRegistry",
    "NanoBrainModel",
    "NanoBrainModelConfig",
    "PredictionBundle",
    "REGISTRIES",
    "backend_catalog",
    "default_genome",
    "DatasetLocation",
    "describe_v3_shards",
    "locate_v3_shards",
    "DatasetSplits",
    "DevelopmentPairDataset",
    "DevelopmentPopulationBatch",
    "NanoDataset",
    "infer_label_cardinalities",
    "load_v3_subset",
]
