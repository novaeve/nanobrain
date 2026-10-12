"""A small user-owned encoder registered through the public module API."""

from torch import nn

from nanobrain.registry import registry


@registry("encoder").register(
    "tutorial_encoder",
    input_rank=2,
    output_rank=2,
    complexity="O(B * G * H)",
    source="examples.custom_encoder",
    notes="Two-layer expression encoder for the custom-model tutorial",
)
def tutorial_encoder(input_dim: int, hidden_dim: int, dropout: float = 0.05, **_):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
    )
