"""A frozen-backbone per-gene task head for strict spatial MCR.

The head adapts the lightweight downstream-head pattern from scUniFlow to the
BrainWM masked-cell reconstruction contract.  It predicts only a residual over
the transparent neighbor-only distribution; it does not implement feature
routing, Mamba, flow matching, or a new cell graph.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _standardize_gene_axis(value: Tensor, *, epsilon: float = 1.0e-6) -> Tensor:
    if value.ndim != 2:
        raise ValueError("value must have shape [cells, genes]")
    centered = value - value.mean(dim=1, keepdim=True)
    scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(epsilon)
    return centered / scale


class ScUniFlowMCRGeneTaskHead(nn.Module):
    """Shared per-gene residual head over frozen BrainWM/spatial covariates.

    Zero-initializing the output layer makes the initial prediction exactly the
    local-neighbor distribution.  This gives the learned route a transparent,
    strong starting point without a scalar gate or a capacity-changing control.
    """

    def __init__(self, gene_dim: int, *, hidden_dim: int = 64) -> None:
        super().__init__()
        if not isinstance(gene_dim, int) or isinstance(gene_dim, bool) or gene_dim < 1:
            raise ValueError("gene_dim must be a positive integer")
        if (
            not isinstance(hidden_dim, int)
            or isinstance(hidden_dim, bool)
            or hidden_dim < 1
        ):
            raise ValueError("hidden_dim must be a positive integer")
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.gene_norm = nn.LayerNorm(gene_dim, elementwise_affine=False)
        self.net = nn.Sequential(
            nn.Linear(gene_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        output = self.net[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("task-head output must be linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        gene_embedding: Tensor,
        brainwm_logits: Tensor,
        section_probability: Tensor,
        local_probability: Tensor,
        library_size: Tensor,
    ) -> Tensor:
        """Return measured-gene probabilities for each masked query cell."""

        if gene_embedding.ndim != 2 or gene_embedding.shape[1] != self.gene_dim:
            raise ValueError(f"gene_embedding must have shape [genes,{self.gene_dim}]")
        if brainwm_logits.ndim != 2:
            raise ValueError("brainwm_logits must have shape [cells, genes]")
        cells, genes = brainwm_logits.shape
        if gene_embedding.shape[0] != genes:
            raise ValueError("gene_embedding and brainwm_logits disagree on genes")
        if section_probability.shape not in {(genes,), (cells, genes)}:
            raise ValueError(
                "section_probability must have shape [genes] or [cells, genes]"
            )
        if local_probability.shape != brainwm_logits.shape:
            raise ValueError("local_probability must match brainwm_logits")
        if library_size.shape != (cells,):
            raise ValueError("library_size must have shape [cells]")
        if not all(
            torch.isfinite(value).all()
            for value in (
                gene_embedding,
                brainwm_logits,
                section_probability,
                local_probability,
                library_size,
            )
        ):
            raise FloatingPointError("task-head inputs must be finite")
        if torch.any(section_probability < 0) or torch.any(local_probability < 0):
            raise ValueError("probability inputs must be nonnegative")
        if torch.any(library_size <= 0):
            raise ValueError("library_size must be positive")

        eps = torch.finfo(brainwm_logits.dtype).eps
        local_log = local_probability.clamp_min(eps).log()
        section = section_probability
        if section.ndim == 1:
            section = section.unsqueeze(0).expand(cells, -1)
        section_log = section.clamp_min(eps).log()
        brainwm_log = torch.log_softmax(brainwm_logits, dim=-1)
        scalar_features = torch.stack(
            (
                _standardize_gene_axis(local_log),
                _standardize_gene_axis(brainwm_log - local_log),
                _standardize_gene_axis(section_log - local_log),
                torch.log1p(library_size).unsqueeze(1).expand(-1, genes) / 10.0,
            ),
            dim=-1,
        )
        gene = self.gene_norm(gene_embedding.float())
        gene = gene.unsqueeze(0).expand(cells, -1, -1)
        features = torch.cat((gene, scalar_features.float()), dim=-1)
        residual = self.net(features).squeeze(-1)
        probabilities = torch.softmax(local_log.float() + residual.float(), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("task-head probabilities are non-finite")
        return probabilities

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def candidate_gate(
    candidate: dict[str, float],
    baselines: dict[str, float],
    *,
    baseline_replay_ok: bool,
) -> dict[str, bool]:
    """Apply the predeclared train-internal strict metric inequalities."""

    required_candidate = {"real", "permutation", "zero"}
    required_baselines = {
        "neighbor_only",
        "complete_section_global",
        "no_structure",
        "prior_hierarchical",
    }
    if set(candidate) != required_candidate:
        raise ValueError("candidate metrics do not match the frozen routes")
    if set(baselines) != required_baselines:
        raise ValueError("baseline metrics do not match the frozen controls")
    values = [*candidate.values(), *baselines.values()]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gate metrics must be finite")
    decision = {
        "baseline_replay_matches_frozen_evaluator": bool(baseline_replay_ok),
        "real_beats_permutation": candidate["real"] < candidate["permutation"],
        "real_beats_zero_coordinates": candidate["real"] < candidate["zero"],
        "real_beats_neighbor_only": candidate["real"] < baselines["neighbor_only"],
        "real_beats_complete_section_global": (
            candidate["real"] < baselines["complete_section_global"]
        ),
        "real_beats_no_structure": candidate["real"] < baselines["no_structure"],
        "real_beats_prior_hierarchical": (
            candidate["real"] < baselines["prior_hierarchical"]
        ),
    }
    decision["promotion_gate"] = all(decision.values())
    return decision


__all__ = ["ScUniFlowMCRGeneTaskHead", "candidate_gate"]
