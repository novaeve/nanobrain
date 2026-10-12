import json

import numpy as np
import pytest
import torch

from nanobrain.data import NanoDataset
from nanobrain.development import DevelopmentPairDataset, DevelopmentPopulationBatch
from nanobrain.genome import default_genome
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig


def _dataset(rows=8):
    labels = {
        "annotation": torch.zeros(rows, dtype=torch.long),
        "niche": torch.zeros(rows, dtype=torch.long),
        "integration": torch.zeros(rows, dtype=torch.long),
        "development": torch.zeros(rows, 1),
    }
    masks = {name: torch.ones(rows, dtype=torch.bool) for name in labels}
    return NanoDataset(torch.randn(rows, 4), labels, masks)


def test_population_pair_loader_builds_equal_endpoint_batches(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "development_source_index.npy", np.array([0, 1, 2], np.int32))
    np.save(split / "development_target_index.npy", np.array([4, 5, 6], np.int32))
    np.save(split / "development_pair_ptr.npy", np.array([0, 3], np.int64))
    (split / "development_pairs.json").write_text(json.dumps({
        "format": "nanobrain_population_pairs_v1",
        "split": "train",
        "groups_metadata": [{
            "interval_id": "DEV-1", "uberon_id": "UBERON:1",
            "source_time_days": 1.0, "target_time_days": 2.0,
        }],
    }))

    pairs = DevelopmentPairDataset.load(tmp_path, "train", _dataset())
    batch = pairs.batch(0)

    assert len(pairs) == 1
    assert batch.cells == 3
    assert batch.source_time_days == 1.0
    assert batch.target_time_days == 2.0


def test_population_pair_loader_rejects_nonpositive_interval(tmp_path):
    dataset = _dataset(4)
    with pytest.raises(ValueError, match="non-positive"):
        DevelopmentPairDataset(
            dataset,
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([0, 2]),
            [{"source_time_days": 2.0, "target_time_days": 2.0}],
            split="train",
        )


def test_population_flow_loss_backpropagates_and_rolls_out():
    torch.manual_seed(4)
    dataset = _dataset(8)
    population = DevelopmentPopulationBatch(
        source=dataset.batch([0, 1, 2, 3]),
        target=dataset.batch([4, 5, 6, 7]),
        source_time_days=10.0,
        target_time_days=20.0,
        interval_id="DEV-1",
        anatomy="UBERON:1",
    )
    genome = default_genome(seed=4)
    genome.hidden_dim = 32
    genome.attention_heads = 2
    genome.sinkhorn_iterations = 20
    genome.rollout_steps = 4
    model = NanoBrainModel(
        NanoBrainModelConfig(
            input_dim=4,
            annotation_classes=2,
            niche_classes=2,
            integration_classes=2,
            development_dim=4,
        ),
        genome,
    )

    total, terms = model.population_flow_losses(
        population, tau=torch.full((4, 1), 0.5)
    )
    total.backward()
    prediction, latent = model.predict_development_population(
        population.source, 10.0, 20.0, steps=2
    )

    assert torch.isfinite(total)
    assert set(terms) == {
        "development_velocity", "development_endpoint", "development_expression"
    }
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.flow.parameters()
    )
    assert prediction.shape == population.source.features.shape
    assert latent.shape == (4, genome.hidden_dim)
