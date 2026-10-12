"""Intervention-aware negative-binomial expression decoder p_psi(x | z)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import NegativeBinomial

from .blocks import MLP
from .config import DecoderConfig
from .types import DecoderOutput


class NegativeBinomialDecoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, context_dim: int,
                 control_dim: int, config: DecoderConfig,
                 tied_gene_embedding: nn.Embedding | None = None):
        super().__init__()
        self.config = config
        self.control_dim = control_dim
        self.latent = MLP(latent_dim + context_dim + control_dim,
                          config.hidden_dim, config.gene_dim, layers=2)
        if config.tie_gene_embeddings:
            if tied_gene_embedding is None:
                raise ValueError("tie_gene_embeddings requires the encoder gene embedding")
            if tied_gene_embedding.embedding_dim != config.gene_dim:
                raise ValueError("decoder.gene_dim must equal encoder.gene_dim when embeddings are tied")
            self.gene_embedding = tied_gene_embedding
        else:
            self.gene_embedding = nn.Embedding(n_genes, config.gene_dim)
        self.gene_bias = nn.Parameter(torch.zeros(n_genes))
        self.raw_dispersion = nn.Parameter(torch.zeros(n_genes))

    def forward(self, z: Tensor, context: Tensor, library_size: Tensor,
                gene_index: Tensor, control: Tensor | None = None,
                sample: bool | None = None) -> DecoderOutput:
        # Context carries assay/dataset metadata and can otherwise provide a
        # shortcut that lets reconstruction ignore the learned cell state.
        # Keep the default path unchanged, while allowing isolated ablations
        # to attenuate or drop that shortcut without changing tensor shapes.
        context = context * self.config.context_scale
        if self.config.context_dropout:
            context = torch.nn.functional.dropout(
                context, p=self.config.context_dropout, training=self.training
            )
        if self.control_dim:
            if control is None:
                raise ValueError("control tensor is required by configured decoder")
            state = torch.cat([z, context, control], dim=-1)
        else:
            state = torch.cat([z, context], dim=-1)
        hidden = self.latent(state)
        gene = self.gene_embedding(gene_index)
        logits = hidden @ gene.transpose(0, 1) + self.gene_bias[gene_index][None, :]
        proportions = torch.softmax(logits, dim=-1)
        mean = library_size.clamp_min(1.0)[:, None] * proportions
        dispersion = torch.nn.functional.softplus(self.raw_dispersion[gene_index]) \
            + self.config.min_dispersion
        should_sample = self.config.sample_distribution if sample is None else sample
        draw = None
        if should_sample:
            total_count = dispersion[None, :].expand_as(mean)
            probs = mean.clamp_min(1.0e-8) / (total_count + mean.clamp_min(1.0e-8))
            draw = NegativeBinomial(total_count=total_count, probs=probs).sample()
        return DecoderOutput(
            gene_index=gene_index,
            logits=logits,
            mean=mean,
            dispersion=dispersion,
            sample=draw,
        )


def dense_counts_for_genes(row_ptr: Tensor, token_gene: Tensor, token_count: Tensor,
                           gene_index: Tensor) -> Tensor:
    """Materialize exact measured-gene targets for NB likelihood."""
    if gene_index.ndim != 1 or gene_index.numel() == 0:
        raise ValueError("decode gene_index must be nonempty")
    if torch.any(gene_index[1:] <= gene_index[:-1]):
        raise ValueError("decode gene_index must be strictly increasing")
    batch_size = int(row_ptr.numel() - 1)
    target = token_count.new_zeros((batch_size, gene_index.numel()))
    row = torch.repeat_interleave(
        torch.arange(batch_size, device=row_ptr.device),
        row_ptr[1:] - row_ptr[:-1],output_size=token_count.numel(),
    )
    position = torch.searchsorted(gene_index, token_gene)
    valid = position < gene_index.numel()
    valid = valid & (gene_index[position.clamp_max(gene_index.numel() - 1)] == token_gene)
    if torch.any(valid):
        target.index_put_((row[valid], position[valid]), token_count[valid], accumulate=True)
    return target
