"""Opt-in Cajal niche composition head and training objective."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from brainwm_unified.regularizers import OBJECTIVES


NICHE_COMPONENT = {"slot": "niche", "name": "cajal.niche_composition", "params": {}}


def install_head(model: nn.Module) -> None:
    """Predict the fine cell-type proportions from one centre cell's latent mean."""
    model.niche_head = nn.Linear(model.cfg.z_dim, model.cfg.cell_type_levels[-1])


def composition_targets(
    labels: torch.Tensor, members: torch.Tensor, n_types: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fine cell-type proportions and eligible centres for groups [centre, neighbours...]."""
    centres = members[:, :1]
    neighbours = members[:, 1:]
    types = labels[neighbours.clamp_min(0)]
    valid = (neighbours >= 0) & (neighbours != centres) & (types >= 0) & (types < n_types)
    counts = (F.one_hot(types.masked_fill(~valid, 0), n_types) * valid.unsqueeze(-1)).sum(1)
    total = counts.sum(1)
    return counts.float() / total.clamp_min(1).unsqueeze(-1), (centres.squeeze(1) >= 0) & (total > 0)


@OBJECTIVES.register("niche_composition")
@dataclass
class NicheComposition:
    """Match each spatial centre's predicted distribution to its neighbours' label frequencies."""

    def __call__(self, model, batch: dict, out: dict) -> torch.Tensor:
        head = getattr(model, "niche_head", None)
        if head is None:
            raise ValueError("niche_composition requires the Cajal niche head")
        members = batch["grp_members"]
        target, eligible = composition_targets(batch["cell_type"], members, head.out_features)
        centres = members[:, 0].clamp_min(0)
        log_prob = head(out["mu"][centres].float()).float().log_softmax(-1)
        per_centre = -(target * log_prob).sum(-1)
        return per_centre.masked_fill(~eligible, 0).sum() / eligible.sum().clamp_min(1)
