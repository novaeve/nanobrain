"""Self-supervised alignment and latent flow modules for foundation pretraining."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import MLP, FourierTimeEmbedding
from .config import AssayAlignmentConfig, FlowMatchingConfig, MaskedGeneModelingConfig


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return value

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.scale * gradient, None


class AssayDiscriminator(nn.Module):
    """Predict assay while reversing its gradient into the cell encoder."""

    def __init__(self, latent_dim: int, config: AssayAlignmentConfig):
        super().__init__()
        self.scale = config.gradient_reversal_scale
        self.classifier = MLP(
            latent_dim,
            config.hidden_dim,
            config.assay_vocab_size,
            layers=2,
        )

    def forward(self, latent: Tensor) -> Tensor:
        reversed_latent = _GradientReverse.apply(latent, self.scale)
        return self.classifier(reversed_latent)


class MaskedGeneModelingHead(nn.Module):
    """Predict masked gene identities from contextualized gene tokens.

    The head uses sampled candidate genes supplied by the trainer. This keeps
    the MGM objective tractable for the 65k-gene vocabulary while retaining all
    masked targets in the candidate set.
    """

    def __init__(self, token_dim: int, gene_dim: int,
                 gene_embedding: nn.Embedding,
                 config: MaskedGeneModelingConfig):
        super().__init__()
        self.projection = MLP(token_dim, config.hidden_dim, gene_dim, layers=2)
        self.gene_embedding = gene_embedding

    def forward(self, token_features: Tensor, candidate_gene_index: Tensor) -> Tensor:
        if token_features.ndim != 2:
            raise ValueError("token_features must have shape [T, D]")
        if candidate_gene_index.ndim != 1 or candidate_gene_index.numel() == 0:
            raise ValueError("candidate_gene_index must be nonempty [C]")
        query = self.projection(token_features)
        candidates = self.gene_embedding(candidate_gene_index)
        return query @ candidates.transpose(0, 1)

    def sampled_forward(self, token_features: Tensor, target_gene_index: Tensor,
                        n_genes: int, negative_samples: int) -> Tensor:
        """Compute per-token positive-vs-negative logits without a dense vocab."""
        if target_gene_index.ndim != 1 or target_gene_index.shape[0] != token_features.shape[0]:
            raise ValueError("target_gene_index must match token_features rows")
        if n_genes < 2 or negative_samples < 1:
            raise ValueError("sampled MGM requires at least two genes and one negative")
        query = self.projection(token_features)
        if target_gene_index.numel() == 0:
            # Keep the MGM head in the autograd graph on ranks whose panel has
            # no positive support. This yields zero gradients instead of an
            # unused-parameter divergence across DDP ranks.
            candidates = torch.arange(
                negative_samples + 1,
                device=target_gene_index.device,
            ) % n_genes
            gene = self.gene_embedding(candidates)
            return query @ gene.transpose(0, 1)
        if torch.any(target_gene_index < 0) or torch.any(target_gene_index >= n_genes):
            raise ValueError("target gene is outside vocabulary")
        # Draw an offset in [1, n_genes - 1] for each target.  This is
        # uniform over all non-target genes and, unlike unrestricted randint,
        # cannot duplicate the positive class in a negative column.
        offset = torch.randint(
            n_genes - 1, (target_gene_index.numel(), negative_samples),
            device=target_gene_index.device,
        ) + 1
        negative = (target_gene_index[:, None] + offset) % n_genes
        candidates = torch.cat([
            target_gene_index[:, None], negative,
        ], dim=1)
        gene = self.gene_embedding(candidates)
        return torch.einsum("td,tcd->tc", query, gene)


class DisentangledLatentFlow(nn.Module):
    """Conditional flow matching with a separate velocity field per partition."""

    def __init__(self, partitions: dict[str, int], context_dim: int,
                 config: FlowMatchingConfig):
        super().__init__()
        self.partitions = dict(partitions)
        self.flow_time = FourierTimeEmbedding(config.time_dim, scale_days=1.0)
        self.biological_time = FourierTimeEmbedding(config.time_dim)
        condition_dim = context_dim + 2 * config.time_dim
        self.velocity = nn.ModuleDict({
            name: MLP(
                width + condition_dim,
                config.hidden_dim,
                width,
                layers=3,
                dropout=config.dropout,
            )
            for name, width in self.partitions.items()
        })

    def matching_pair(self, latent: Tensor, context: Tensor,
                      time_days: Tensor) -> tuple[Tensor, Tensor]:
        # The flow learns the age-conditioned aggregate latent distribution.
        # Detaching keeps this auxiliary prior from collapsing the VAE encoder.
        target_latent = latent.detach()
        condition_context = context.detach()
        noise = torch.randn_like(target_latent)
        interpolation_time = torch.rand(
            target_latent.shape[0], device=target_latent.device,
            dtype=target_latent.dtype,
        )
        interpolated = (
            (1.0 - interpolation_time[:, None]) * noise
            + interpolation_time[:, None] * target_latent
        )
        condition = torch.cat([
            condition_context,
            self.flow_time(interpolation_time),
            self.biological_time(time_days.to(target_latent.dtype)),
        ], dim=-1)
        predictions = []
        targets = []
        start = 0
        for name, width in self.partitions.items():
            stop = start + width
            predictions.append(self.velocity[name](
                torch.cat([interpolated[:, start:stop], condition], dim=-1)
            ))
            targets.append(target_latent[:, start:stop] - noise[:, start:stop])
            start = stop
        return torch.cat(predictions, dim=-1), torch.cat(targets, dim=-1)
