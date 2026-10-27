"""Small, contract-first research loop for BrainWM module evolution."""

from .contracts import BrainBatch, LossBundle, PredictionBundle
from .genome import CandidateGenome, default_genome
from .model import NanoBrainModel, NanoBrainModelConfig
from .registry import REGISTRIES, ModuleRegistry, backend_catalog
from .paths import DatasetLocation, describe_v3_shards, locate_v3_shards
from .data import DatasetSplits, NanoDataset, infer_label_cardinalities, load_v3_subset
from .development import DevelopmentPairDataset, DevelopmentPopulationBatch
from .api_config import (
    APIConfig,
    APIConfigError,
    ThirdPartyAPIConfig,
    merged_environment,
    read_env_file,
)
from .llm_client import APIRequestError, OpenAICompatibleClient

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
    "APIConfig",
    "APIConfigError",
    "ThirdPartyAPIConfig",
    "merged_environment",
    "read_env_file",
    "APIRequestError",
    "OpenAICompatibleClient",
]
