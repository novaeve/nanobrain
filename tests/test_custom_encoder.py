from __future__ import annotations

import torch

from nanobrain.genome import CandidateGenome
from nanobrain.registry import registry


def test_registered_encoder_plugin_can_be_selected(monkeypatch) -> None:
    encoders = registry("encoder")
    monkeypatch.setattr(encoders, "_items", dict(encoders._items))
    monkeypatch.setattr(encoders, "_aliases", dict(encoders._aliases))
    encoders.register(
        "test_custom_encoder", lambda input_dim, hidden_dim: torch.nn.Linear(input_dim, hidden_dim)
    )

    genome = CandidateGenome.from_dict({"encoder": "test_custom_encoder"})
    module = encoders.create(genome.encoder, input_dim=256, hidden_dim=genome.hidden_dim)

    assert module(torch.zeros(3, 256)).shape == (3, genome.hidden_dim)
