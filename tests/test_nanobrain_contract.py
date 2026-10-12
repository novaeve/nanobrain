import torch

from nanobrain.contracts import BrainBatch
from nanobrain.data import NanoDataset
from nanobrain.genome import default_genome
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig


def _dataset() -> NanoDataset:
    features = torch.randn(5, 4)
    # Row 0 has row 4 as a neighbor; row 1 has only its self slot.
    neighbors = torch.tensor([[0, 4], [1, 1], [2, 3], [3, 2], [4, 0]])
    neighbor_mask = torch.tensor([[False, True], [False, False], [False, True], [False, True], [False, True]])
    labels = {
        "annotation": torch.tensor([0, 1, 0, 1, 0]),
        "niche": torch.tensor([0, 0, 1, 1, 0]),
        "integration": torch.tensor([0, 1, 0, 1, 0]),
        "development": torch.randn(5, 1),
    }
    masks = {name: torch.ones(5, dtype=torch.bool) for name in labels}
    return NanoDataset(
        features=features,
        labels=labels,
        masks=masks,
        neighbors=neighbors,
        neighbor_mask=neighbor_mask,
        feature_observed=torch.ones_like(features, dtype=torch.bool),
        development_condition=torch.ones(5, 1),
        development_condition_observed=torch.ones(5, dtype=torch.bool),
        completion_target=features.clone(),
        completion_mask=torch.ones_like(features, dtype=torch.bool),
    )


def test_minibatch_neighbor_indices_are_remapped_and_masked():
    dataset = _dataset()
    batch = dataset.batch(torch.tensor([4, 0]))
    assert batch.neighbors.max().item() < batch.batch_size
    # Source row 4 -> row 0 is present at local index 1.
    assert batch.neighbors[0, 1].item() == 1
    assert bool(batch.neighbor_mask[0, 1])
    # Source row 0 -> row 4 is present at local index 0.
    assert batch.neighbors[1, 1].item() == 0
    assert bool(batch.neighbor_mask[1, 1])


def test_minibatch_accepts_plain_index_sequences():
    batch = _dataset().batch([4, 0])
    assert batch.batch_size == 2
    assert batch.neighbors.max().item() < batch.batch_size


def test_batch_contract_keeps_legacy_positional_order():
    batch = BrainBatch(torch.ones(2, 3), {"x": torch.zeros(2)}, {"x": torch.ones(2, dtype=torch.bool)})
    assert batch.coordinates is None
    assert batch.feature_observed is None


def test_development_target_is_not_read_by_forward():
    torch.manual_seed(3)
    model = NanoBrainModel(
        NanoBrainModelConfig(4, annotation_classes=2, niche_classes=2, integration_classes=2, development_dim=1),
        default_genome(seed=3),
    )
    model.eval()
    dataset = _dataset()
    first = dataset.batch(torch.tensor([0, 1]))
    second = dataset.batch(torch.tensor([0, 1]))
    second.labels["development"] = second.labels["development"] + 1000
    first_output = model(first)
    second_output = model(second)
    torch.testing.assert_close(first_output.embedding, second_output.embedding)
    torch.testing.assert_close(first_output.development_prediction, second_output.development_prediction)
