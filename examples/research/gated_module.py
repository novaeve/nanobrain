import torch
from torch import nn

class GatedResidual(nn.Module):
    def __init__(self, hidden_dim, expansion=2):
        super().__init__()
        self.up = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.down = nn.Linear(hidden_dim * expansion, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state):
        correction = self.down(torch.nn.functional.silu(self.up(state)))
        return state + torch.sigmoid(self.gate(state)) * correction


def build(hidden_dim, expansion=2):
    return GatedResidual(hidden_dim, expansion)
