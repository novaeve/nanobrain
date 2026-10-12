"""Sparse gene-expression encoder E_phi(x, q, c)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .blocks import MLP, ResidualMLP, row_index_from_ptr, segment_softmax, segment_sum
from .config import ContextConfig, EncoderConfig
from .gene_annotations import GeneTokenFeatureEncoder
from .modalities import NamedModalityRouter
from .types import SparseExpressionBatch


class SegmentContextBlock(nn.Module):
    """Exchange information between every token and its pooled cell context."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.update = MLP(2 * dim, hidden_dim, dim, layers=2, dropout=dropout)

    def forward(self, tokens: Tensor, cell_context: Tensor,
                row_index: Tensor) -> Tensor:
        values = torch.cat([
            self.token_norm(tokens), self.context_norm(cell_context)[row_index]
        ], dim=-1)
        return tokens + self.update(values)


def segment_midrank(count: Tensor, row_index: Tensor,
                    batch_size: int) -> Tensor:
    """Return each token's descending within-cell mid-rank in ``[0, 1]``.

    Ties receive their average one-indexed rank.  Stable two-pass sorting keeps
    the operation permutation-equivariant within each CSR cell while avoiding a
    dense token-by-token comparison matrix.
    """
    if count.ndim != 1 or row_index.ndim != 1:
        raise ValueError("count and row_index must be one-dimensional")
    if count.shape != row_index.shape:
        raise ValueError("count and row_index must have the same shape")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if count.numel() == 0:
        return count.new_empty((0,), dtype=torch.float32)
    if torch.any(row_index < 0) or torch.any(row_index >= batch_size):
        raise ValueError("row_index is outside the requested batch")

    # Sort by count descending within each row using stable passes.  The first
    # pass establishes count order; the second makes row the primary key while
    # retaining that order inside each row.
    by_count = torch.argsort(count, descending=True, stable=True)
    row_order = torch.argsort(row_index[by_count], stable=True)
    order = by_count[row_order]
    sorted_row = row_index[order]
    sorted_count = count[order]
    position = torch.arange(count.numel(), device=count.device)
    row_lengths = torch.bincount(row_index, minlength=batch_size)
    row_offsets = torch.cat([
        row_lengths.new_zeros(1),
        torch.cumsum(row_lengths[:-1], dim=0),
    ])
    position_within_row = position - row_offsets[sorted_row]
    same_previous = torch.zeros(count.numel(), dtype=torch.bool,
                                device=count.device)
    if count.numel() > 1:
        same_previous[1:] = (
            (sorted_row[1:] == sorted_row[:-1])
            & (sorted_count[1:] == sorted_count[:-1])
        )
    group_id = (~same_previous).cumsum(dim=0) - 1
    n_groups = int(group_id[-1].item()) + 1
    group_start = torch.zeros(n_groups, dtype=position.dtype,
                              device=count.device)
    group_end = torch.zeros_like(group_start)
    group_start.scatter_reduce_(0, group_id, position_within_row, reduce="amin",
                               include_self=False)
    group_end.scatter_reduce_(0, group_id, position_within_row, reduce="amax",
                             include_self=False)
    lengths = row_lengths[sorted_row].to(dtype=count.dtype).clamp_min(1.0)
    midrank = (0.5 * (group_start[group_id] + group_end[group_id]) + 1.0)
    midrank = midrank / lengths
    result = torch.empty_like(midrank)
    result[order] = midrank
    return result


class ContextEncoder(nn.Module):
    def __init__(self, config: ContextConfig):
        super().__init__()
        self.config = config
        self.field_names = list(config.categorical_vocab_sizes)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, config.categorical_embedding_dim, padding_idx=0)
            for name, size in config.categorical_vocab_sizes.items()
        })
        input_dim = len(self.field_names) * config.categorical_embedding_dim + config.continuous_dim
        self.net = MLP(max(input_dim, 1), config.hidden_dim, config.output_dim,
                       layers=2, dropout=config.dropout)

    def forward(self, categorical: Tensor, continuous: Tensor) -> Tensor:
        if categorical.shape[1] != len(self.field_names):
            raise ValueError(
                f"expected {len(self.field_names)} categorical fields, got {categorical.shape[1]}"
            )
        if continuous.shape[1] != self.config.continuous_dim:
            raise ValueError(
                f"expected {self.config.continuous_dim} continuous fields, got {continuous.shape[1]}"
            )
        parts = [self.embeddings[name](categorical[:, i])
                 for i, name in enumerate(self.field_names)]
        if self.config.continuous_dim:
            parts.append(continuous)
        if not parts:
            parts.append(continuous.new_zeros((continuous.shape[0], 1)))
        return self.net(torch.cat(parts, dim=-1))

    @property
    def anatomy_field_index(self) -> int | None:
        """Column index of the canonical UBERON context field, if configured.

        ``uberon_id`` is the ontology-backed anatomical location token.  It is
        intentionally kept in the generic context table so old checkpoints
        retain their exact parameter layout, while this explicit accessor
        makes the anatomy pathway available to pretraining diagnostics and
        downstream heads.
        """
        try:
            return self.field_names.index("uberon_id")
        except ValueError:
            return None

    def anatomy_embedding(self, categorical: Tensor) -> Tensor:
        """Return the learned per-cell anatomical-location embedding.

        Index ``0`` is the vocabulary's ``<missing>`` value and uses the
        embedding's padding semantics, so a cell without an anatomical
        annotation contributes an exact zero vector.  ``<unk>`` remains a
        distinct, present-but-unresolved value (normally index ``1``).
        """
        if categorical.ndim != 2:
            raise ValueError("categorical context must have shape [B, C]")
        field_index = self.anatomy_field_index
        if field_index is None:
            return categorical.new_zeros(
                (categorical.shape[0], self.config.categorical_embedding_dim),
                dtype=torch.float32,
            )
        return self.embeddings["uberon_id"](categorical[:, field_index])

    def anatomy_presence(self, categorical: Tensor) -> Tensor:
        """Return ``[B]`` availability for the canonical anatomy token."""
        if categorical.ndim != 2:
            raise ValueError("categorical context must have shape [B, C]")
        field_index = self.anatomy_field_index
        if field_index is None:
            return torch.zeros(
                categorical.shape[0], dtype=torch.bool, device=categorical.device
            )
        return categorical[:, field_index] != 0


class SparseGeneExpressionEncoder(nn.Module):
    """Encode nonzero raw-count gene tokens without a dense 65k input."""

    def __init__(
        self,
        n_genes: int,
        encoder: EncoderConfig,
        context: ContextConfig,
        modality_config=None,
    ):
        super().__init__()
        self.config = encoder
        self.context_config = context
        self.n_genes = int(n_genes)
        # Optional scUniFlow-style ordered token branch.  BrainWorldModel
        # attaches the module only when explicitly enabled; ``None`` keeps the
        # legacy forward path and checkpoint key set unchanged.
        self.scuniflow_gene_adapter: nn.Module | None = None
        self.gene_embedding = nn.Embedding(n_genes, encoder.gene_dim)
        self.gene_features = None
        if encoder.gene_features_enabled:
            # Optional annotation branches must not shift initialization of
            # the common encoder/decoder backbone in matched ablations.
            with torch.random.fork_rng(devices=[]):
                self.gene_features = GeneTokenFeatureEncoder(
                    encoder.gene_feature_dir,
                    encoder.gene_dim,
                    encoder.external_gene_embedding_path,
                    include_gene_type=encoder.gene_type_enabled,
                    include_homology_group=encoder.homology_group_enabled,
                    expected_n_genes=n_genes,
                )
        self.value_encoder = MLP(4, encoder.value_dim, encoder.gene_dim, layers=2,
                                 dropout=encoder.dropout)
        self.rank_value_sidecar = None
        self.rank_value_sidecar_gate = None
        if encoder.rank_value_sidecar_enabled:
            sidecar_hidden = (
                encoder.rank_value_sidecar_hidden_dim
                if encoder.rank_value_sidecar_hidden_dim > 0
                else encoder.value_dim
            )
            # Do not let the optional branch consume the initialization stream
            # used by the legacy encoder and decoder-compatible modules.
            with torch.random.fork_rng(devices=[]):
                self.rank_value_sidecar = MLP(
                    2, sidecar_hidden, encoder.gene_dim, layers=2,
                    # The zero-gated candidate must not advance the dropout
                    # RNG before the legacy token path runs.
                    dropout=0.0,
                )
            self.rank_value_sidecar_gate = nn.Parameter(
                torch.tensor(float(encoder.rank_value_sidecar_gate_init))
            )
        self.token_layers = nn.ModuleList([
            ResidualMLP(encoder.gene_dim, encoder.hidden_dim, encoder.dropout)
            for _ in range(encoder.token_layers)
        ])
        self.token_context_layers = nn.ModuleList([
            SegmentContextBlock(
                encoder.gene_dim, encoder.hidden_dim, encoder.dropout
            )
            for _ in range(encoder.token_context_layers)
        ])
        self.attention_score = nn.Linear(encoder.gene_dim, 1)
        self.context_encoder = ContextEncoder(context)
        self.modality_encoder = (
            MLP(encoder.optional_modality_dim, encoder.hidden_dim,
                context.output_dim, layers=2, dropout=encoder.dropout)
            if encoder.optional_modality_dim else None
        )
        self.modality_router: NamedModalityRouter | None = None
        self.modality_gate: nn.Parameter | None = None
        if modality_config is not None and getattr(modality_config, "enabled", False):
            self.modality_router = NamedModalityRouter(
                modality_config.input_dims,
                context.output_dim,
                modality_config.hidden_dim,
                modality_config.dropout,
            )
            self.modality_gate = nn.Parameter(
                torch.tensor(float(modality_config.gate_init))
            )
        self.fusion = MLP(
            encoder.gene_dim + context.output_dim
            + (context.output_dim if encoder.optional_modality_dim else 0),
            encoder.hidden_dim,
            encoder.latent_dim,
            layers=3,
            dropout=encoder.dropout,
        )
        field_names = list(context.categorical_vocab_sizes)
        self.latent_neutral_indices = [
            field_names.index(name) for name in context.latent_neutral_fields
        ]
        self.latent_mean = (
            nn.Linear(encoder.latent_dim, encoder.latent_dim)
            if encoder.variational else None
        )
        self.latent_logvar = (
            nn.Linear(encoder.latent_dim, encoder.latent_dim)
            if encoder.variational else None
        )
        # Missingness is an opt-in branch.  Keeping the parameters absent in
        # the default configuration preserves the historical state-dict and
        # lets old checkpoints load without synthetic values.  Forking the RNG
        # prevents optional token initialization from perturbing legacy module
        # initialization in matched experiments.
        self.mask_token: nn.Parameter | None = None
        self.missing_token: nn.Parameter | None = None
        if getattr(encoder, "null_token_enabled", False):
            init_std = float(getattr(encoder, "null_token_init_std", 0.02))
            if init_std < 0 or not torch.isfinite(torch.tensor(init_std)):
                raise ValueError("null_token_init_std must be finite and nonnegative")
            with torch.random.fork_rng(devices=[]):
                self.mask_token = nn.Parameter(
                    torch.empty(encoder.gene_dim).normal_(mean=0.0, std=init_std)
                )
                self.missing_token = nn.Parameter(
                    torch.empty(encoder.gene_dim).normal_(mean=0.0, std=init_std)
                )

    @property
    def null_token(self) -> nn.Parameter | None:
        """Alias for the learned token used by unmeasured-gene placeholders."""
        return self.missing_token

    def anatomy_embedding(self, categorical_context: Tensor) -> Tensor:
        """Expose the shared UBERON embedding used by the Stage-0 context path."""
        return self.context_encoder.anatomy_embedding(categorical_context)

    def anatomy_presence(self, categorical_context: Tensor) -> Tensor:
        """Expose whether each cell has a non-missing UBERON token."""
        return self.context_encoder.anatomy_presence(categorical_context)

    def _tokenize_with_depth_states(
        self,
        expression: SparseExpressionBatch,
        categorical_context: Tensor,
        continuous_context: Tensor,
        optional_modalities: Tensor | None = None,
        token_mask: Tensor | None = None,
        gene_order: Tensor | None = None,
        missing_token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        row_index = row_index_from_ptr(expression.row_ptr)
        library = expression.library_size[row_index].clamp_min(1.0)
        count = expression.count.float()
        if token_mask is None:
            token_mask = expression.token_mask
        if missing_token_mask is None:
            missing_token_mask = expression.missing_token_mask
        for name, mask in (
            ("token_mask", token_mask),
            ("missing_token_mask", missing_token_mask),
        ):
            if mask is not None:
                if mask.shape != expression.gene_index.shape:
                    raise ValueError(
                        f"{name} must match expression.gene_index/count shape"
                    )
                if mask.dtype != torch.bool:
                    raise ValueError(f"{name} must be boolean")
        if token_mask is not None and missing_token_mask is not None:
            if torch.any(token_mask & missing_token_mask):
                raise ValueError(
                    "token_mask and missing_token_mask cannot overlap"
                )
        if missing_token_mask is not None and torch.any(missing_token_mask):
            if self.missing_token is None:
                raise ValueError(
                    "missing_token_mask was supplied but encoder.null_token_enabled is false"
                )
        value_features = self._value_features(
            count, library, row_index, expression.batch_size
        )
        gene_tokens = self.gene_embedding(expression.gene_index)
        if token_mask is not None:
            mask_embedding = (
                self.mask_token
                if self.mask_token is not None
                else gene_tokens.new_zeros((gene_tokens.shape[-1],))
            )
            gene_tokens = torch.where(
                token_mask[:, None], mask_embedding[None, :], gene_tokens
            )
        if missing_token_mask is not None and torch.any(missing_token_mask):
            assert self.missing_token is not None
            # Keep the gene identity/annotations available as a query while
            # replacing its unobserved expression value by a shared learned
            # null embedding.  This allows the decoder to predict gene-specific
            # values without conflating missingness with biological zero.
            gene_tokens = torch.where(
                missing_token_mask[:, None],
                gene_tokens + self.missing_token[None, :],
                gene_tokens,
            )
            value_features = torch.where(
                missing_token_mask[:, None],
                torch.zeros_like(value_features),
                value_features,
            )
        tokens = gene_tokens + self.value_encoder(value_features)
        if self.rank_value_sidecar is not None:
            assert self.rank_value_sidecar_gate is not None
            rank_features = torch.stack([
                segment_midrank(count, row_index, expression.batch_size),
                count / library,
            ], dim=-1)
            if missing_token_mask is not None:
                rank_features = torch.where(
                    missing_token_mask[:, None],
                    torch.zeros_like(rank_features),
                    rank_features,
                )
            tokens = tokens + self.rank_value_sidecar_gate * self.rank_value_sidecar(
                rank_features
            )
        if self.gene_features is not None:
            feature_tokens = self.gene_features(expression.gene_index)
            if token_mask is not None:
                feature_tokens = feature_tokens.masked_fill(
                    token_mask[:, None], 0.0
                )
            tokens = tokens + feature_tokens
        for layer in self.token_layers:
            tokens = layer(tokens)

        if self.scuniflow_gene_adapter is not None:
            # The adapter operates on flattened CSR tokens and restores their
            # original order.  It is deliberately applied before pooling so
            # Stage-0's molecular state, depth readouts, and MGM token view
            # all observe the same ordered context.
            tokens = self.scuniflow_gene_adapter(
                tokens,
                expression.row_ptr,
                expression.gene_index,
                gene_order,
            )

        batch_size = expression.batch_size
        def pool(values: Tensor) -> Tensor:
            if self.config.pooling == "attention":
                weights = segment_softmax(
                    self.attention_score(values).squeeze(-1), row_index, batch_size
                )
                return segment_sum(values * weights[:, None], row_index, batch_size)
            result = segment_sum(values, row_index, batch_size)
            denominator = (
                expression.row_ptr[1:] - expression.row_ptr[:-1]
            ).clamp_min(1)
            return result / denominator[:, None]

        shallow_pooled = pool(tokens)
        pooled = shallow_pooled
        for layer in self.token_context_layers:
            tokens = layer(tokens, pooled, row_index)
            pooled = pool(tokens)
        if self.missing_token is not None:
            # A compact panel-level signal lets a direct molecular-only batch
            # communicate assay coverage even when no explicit placeholders
            # were materialized.  The same learned vector is used for explicit
            # null tokens, keeping the semantics aligned without a dense mask.
            panel_summary = expression.panel_missing_summary_for(self.n_genes)
            pooled = pooled + panel_summary[:, 1:2].to(pooled.dtype) * self.missing_token[None, :]
        context = self.context_encoder(categorical_context, continuous_context)
        if self.latent_neutral_indices:
            latent_categorical = categorical_context.clone()
            latent_categorical[:, self.latent_neutral_indices] = 0
            latent_continuous = (
                torch.zeros_like(continuous_context)
                if self.context_config.latent_neutralize_continuous
                else continuous_context
            )
            latent_context = self.context_encoder(
                latent_categorical, latent_continuous
            )
        else:
            latent_context = context
        return (
            row_index,
            tokens,
            shallow_pooled,
            pooled,
            context,
            latent_context,
        )

    def _value_features(
        self,
        count: Tensor,
        library: Tensor,
        row_index: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Build the four scalar features consumed by ``value_encoder``.

        ``count`` is always the raw sparse count.  The normalization switch
        changes only this encoder view; decoder targets and library sizes stay
        untouched.  The default branch is byte-for-byte equivalent to the
        historical feature construction, preserving legacy checkpoint scores.
        """
        mode = str(getattr(self.config, "expression_normalization", "log1p_library"))
        if mode == "log1p_library":
            return torch.stack(
                [
                    torch.log1p(count),
                    torch.sqrt(count.clamp_min(0.0)),
                    count / library,
                    torch.ones_like(count),
                ],
                dim=-1,
            )
        if mode == "log1p_raw":
            return torch.stack(
                [
                    torch.log1p(count),
                    torch.sqrt(count.clamp_min(0.0)),
                    count,
                    torch.ones_like(count),
                ],
                dim=-1,
            )
        if mode == "log1p_cpm":
            cpm = count / library * 1.0e4
            return torch.stack(
                [
                    torch.log1p(cpm),
                    torch.sqrt(cpm.clamp_min(0.0)),
                    cpm / 1.0e4,
                    torch.ones_like(count),
                ],
                dim=-1,
            )
        if mode == "clr_nonzero":
            # Center log-counts over the observed (nonzero) tokens in each
            # cell.  Masked tokens have count=0 and remain neutral rather than
            # shifting the cell's CLR reference.
            positive = count > 0
            log_count = torch.where(
                positive, torch.log(count.clamp_min(1.0)), torch.zeros_like(count)
            )
            totals = segment_sum(log_count, row_index, batch_size)
            n_positive = segment_sum(
                positive.to(dtype=log_count.dtype), row_index, batch_size
            ).clamp_min(1.0)
            centered = log_count - (totals / n_positive)[row_index]
            centered = torch.where(positive, centered, torch.zeros_like(centered))
            relative = count / library
            return torch.stack(
                [centered, torch.sign(centered) * torch.sqrt(centered.abs()), relative, torch.ones_like(count)],
                dim=-1,
            )
        raise ValueError(f"unsupported expression normalization: {mode!r}")

    def _tokenize(self, expression: SparseExpressionBatch,
                  categorical_context: Tensor,
                  continuous_context: Tensor,
                  optional_modalities: Tensor | None = None,
                  token_mask: Tensor | None = None,
                  gene_order: Tensor | None = None,
                  missing_token_mask: Tensor | None = None,
                  ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        row_index, tokens, _, pooled, context, latent_context = (
            self._tokenize_with_depth_states(
                expression,
                categorical_context,
                continuous_context,
                optional_modalities,
                token_mask,
                gene_order,
                missing_token_mask,
            )
        )
        return row_index, tokens, pooled, context, latent_context

    def _fuse_latent(
        self,
        pooled: Tensor,
        latent_context: Tensor,
        optional_modalities: Tensor | None,
        modality_inputs: Mapping[str, Tensor] | None = None,
        modality_presence: Mapping[str, Tensor] | None = None,
        *,
        sample_latent: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        fusion_parts = [pooled, latent_context]
        if self.modality_encoder is not None:
            if optional_modalities is None:
                raise ValueError("configured optional modalities were not provided")
            fusion_parts.append(self.modality_encoder(optional_modalities))
        if self.modality_router is not None:
            if self.modality_gate is None:
                raise RuntimeError("modality router gate is not initialized")
            routed = self.modality_router(
                modality_inputs,
                batch_size=pooled.shape[0],
                presence=modality_presence,
            )
            latent_context = latent_context + torch.tanh(self.modality_gate) * routed
            fusion_parts[1] = latent_context
        fused = self.fusion(torch.cat(fusion_parts, dim=-1))
        if self.config.variational:
            assert self.latent_mean is not None and self.latent_logvar is not None
            mean = self.latent_mean(fused)
            logvar = self.latent_logvar(fused).clamp(
                self.config.logvar_min, self.config.logvar_max
            )
            if self.training and sample_latent:
                latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            else:
                latent = mean
        else:
            latent = fused
            mean = fused
            logvar = torch.zeros_like(fused)
        return latent, mean, logvar

    def forward(self, expression: SparseExpressionBatch,
                categorical_context: Tensor,
                continuous_context: Tensor,
                optional_modalities: Tensor | None = None,
                token_mask: Tensor | None = None,
                gene_order: Tensor | None = None,
                modality_inputs: Mapping[str, Tensor] | None = None,
                modality_presence: Mapping[str, Tensor] | None = None,
                missing_token_mask: Tensor | None = None,
                ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        latent, mean, logvar, context, partitions, _ = self.forward_with_depth_states(
            expression,
            categorical_context,
            continuous_context,
            optional_modalities,
            token_mask,
            gene_order,
            modality_inputs,
            modality_presence,
            missing_token_mask,
        )
        return latent, mean, logvar, context, partitions

    def forward_with_depth_states(
        self,
        expression: SparseExpressionBatch,
        categorical_context: Tensor,
        continuous_context: Tensor,
        optional_modalities: Tensor | None = None,
        token_mask: Tensor | None = None,
        gene_order: Tensor | None = None,
        modality_inputs: Mapping[str, Tensor] | None = None,
        modality_presence: Mapping[str, Tensor] | None = None,
        missing_token_mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, Tensor],
        tuple[Tensor, Tensor, Tensor],
    ]:
        """Encode once and expose shallow, middle, and deep states.

        The regular :meth:`forward` method keeps its historical five-value
        contract.  This companion is used by optional readouts that need
        intermediate states; sharing the tokenization pass avoids a second
        sparse scan and keeps dropout/RNG behavior consistent.
        """
        (
            _,
            _,
            shallow_pooled,
            middle_pooled,
            context,
            latent_context,
        ) = self._tokenize_with_depth_states(
            expression,
            categorical_context,
            continuous_context,
            optional_modalities,
            token_mask,
            gene_order,
            missing_token_mask,
        )
        latent, mean, logvar = self._fuse_latent(
            middle_pooled,
            latent_context,
            optional_modalities,
            modality_inputs,
            modality_presence,
        )
        partitions: dict[str, Tensor] = {}
        start = 0
        for name, width in self.config.latent_partitions.items():
            partitions[name] = latent[:, start:start + width]
            start += width
        return (
            latent,
            mean,
            logvar,
            context,
            partitions,
            (shallow_pooled, middle_pooled, mean),
        )

    def depth_aligned_states(
        self,
        expression: SparseExpressionBatch,
        categorical_context: Tensor,
        continuous_context: Tensor,
        optional_modalities: Tensor | None = None,
        token_mask: Tensor | None = None,
        gene_order: Tensor | None = None,
        modality_inputs: Mapping[str, Tensor] | None = None,
        modality_presence: Mapping[str, Tensor] | None = None,
        missing_token_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return shallow, middle, and deep cell states for hierarchy heads.

        The shallow state pools gene tokens after token-local processing and
        before cell-context exchange.  The middle state pools the same tokens
        after all configured cell-context blocks.  The deep state is the
        deterministic variational posterior mean (or fused latent for a
        non-variational encoder).  This method adds no parameters and leaves
        the legacy :meth:`forward` return contract unchanged.
        """

        _, _, _, _, _, depth_states = self.forward_with_depth_states(
            expression,
            categorical_context,
            continuous_context,
            optional_modalities,
            token_mask,
            gene_order,
            modality_inputs,
            modality_presence,
            missing_token_mask,
        )
        return depth_states

    def token_features(self, expression: SparseExpressionBatch,
                       categorical_context: Tensor,
                       continuous_context: Tensor,
                       optional_modalities: Tensor | None = None,
                       token_mask: Tensor | None = None,
                       gene_order: Tensor | None = None,
                       missing_token_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return contextualized gene tokens and their cell row indices."""
        row_index, tokens, _, _, _ = self._tokenize(
            expression, categorical_context, continuous_context,
            optional_modalities, token_mask, gene_order, missing_token_mask,
        )
        return tokens, row_index
