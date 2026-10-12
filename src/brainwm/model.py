"""Full hierarchical spatiotemporal Brain World Model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

from .annotation import HierarchicalResidualAnnotationHead, hierarchical_annotation_loss
from .auxiliary import (
    AssayDiscriminator,
    DisentangledLatentFlow,
    MaskedGeneModelingHead,
)
from .config import (
    AnnotationConfig,
    ExperimentConfig,
    ModalityAdapterConfig,
    ModelConfig,
    ScFiTTemporalConfig,
    ScUniFlowGeneConfig,
    ScUniFlowSpatialConfig,
)
from .decoder import NegativeBinomialDecoder
from .dynamics import ControlledLatentDynamics, FixedStepIntegrator
from .encoder import SparseGeneExpressionEncoder
from .graph import DynamicSpatialGraph
from .hierarchy import CellNicheTissueHierarchy
from .integration import (
    ScFiTTemporalAdapter,
    ScUniFlowSparseGeneAdapter,
    ScUniFlowSpatialAdapter,
)
from .types import (
    BrainWorldModelOutput,
    HierarchicalAnnotationOutput,
    StructuralContext,
    TemporalTransitionOutput,
    WorldModelBatch,
)


def _load_gene_order(path: str | Path, n_genes: int) -> Tensor | None:
    """Load an optional ordered-position -> global-gene permutation.

    The order is metadata rather than a learned parameter, so it is kept as a
    non-persistent buffer by the adapter.  Supporting both ``.npy`` and Torch
    tensors keeps the loader compatible with the existing routing artifacts.
    """
    value = str(path).strip() if path is not None else ""
    if not value:
        return None
    source = Path(value)
    if not source.is_file():
        raise FileNotFoundError(f"scUniFlow gene_order_path does not exist: {source}")
    if source.suffix.lower() == ".npy":
        import numpy as np

        order = torch.from_numpy(np.load(source, allow_pickle=False))
    else:
        try:
            loaded = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 has no ``weights_only`` argument.
            loaded = torch.load(source, map_location="cpu")
        if isinstance(loaded, dict):
            # A few routing manifests store the tensor under a descriptive
            # key; accept only unambiguous one-key mappings.
            candidates = [
                loaded[key]
                for key in ("gene_order", "biological_order", "order")
                if key in loaded
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Torch gene-order artifact must contain gene_order, "
                    "biological_order, or order"
                )
            loaded = candidates[0]
        if not isinstance(loaded, Tensor):
            loaded = torch.as_tensor(loaded)
        order = loaded
    if order.ndim != 1 or order.numel() != n_genes:
        raise ValueError(
            f"scUniFlow gene order must have shape [{n_genes}], got {tuple(order.shape)}"
        )
    if order.dtype == torch.bool or order.is_floating_point() or order.is_complex():
        raise TypeError("scUniFlow gene order must use an integer dtype")
    order = order.to(dtype=torch.long).contiguous()
    if torch.any(order < 0) or torch.any(order >= n_genes):
        raise ValueError("scUniFlow gene order contains an out-of-range index")
    if torch.unique(order).numel() != n_genes:
        raise ValueError("scUniFlow gene order must be a permutation")
    return order


class BrainWorldModel(nn.Module):
    """Six-component implementation of `brain_world_model.pdf`.

    Reconstruction uses ``forward(batch)``. A developmental transition uses
    ``forward(batch, target_time_days=...)``; graph weights and hierarchical
    context are recomputed inside every solver evaluation.
    """

    def __init__(self, config: ModelConfig | ExperimentConfig):
        super().__init__()
        self.config = config.model if isinstance(config, ExperimentConfig) else config
        cfg = self.config
        # Direct ``ModelConfig`` callers may use the explicit alias fields
        # without running ``ExperimentConfig.validate`` first.  Normalize the
        # aliases locally so construction remains predictable in tests and
        # lightweight inference scripts.
        annotation_cfg = getattr(cfg, "annotation", AnnotationConfig())
        depth_head_cfg = getattr(cfg, "scbrain_depth_head", None)
        if isinstance(annotation_cfg, dict):
            annotation_cfg = AnnotationConfig(**annotation_cfg)
        if isinstance(depth_head_cfg, dict):
            depth_head_cfg = AnnotationConfig(**depth_head_cfg)
        if depth_head_cfg is not None and not getattr(annotation_cfg, "enabled", False):
            annotation_cfg = depth_head_cfg
        # Keep the model-held config canonical as well; direct ``ModelConfig``
        # callers are allowed to provide mappings without going through the
        # full ExperimentConfig constructor.
        cfg.annotation = annotation_cfg
        scfit_cfg = getattr(cfg, "scfit_temporal_adapter", ScFiTTemporalConfig())
        gene_cfg = getattr(cfg, "scuniflow_gene_adapter", ScUniFlowGeneConfig())
        spatial_cfg = getattr(
            cfg, "scuniflow_spatial_adapter", ScUniFlowSpatialConfig()
        )
        modality_cfg = getattr(cfg, "modality_adapters", ModalityAdapterConfig())
        if isinstance(scfit_cfg, dict):
            scfit_cfg = ScFiTTemporalConfig(**scfit_cfg)
        if isinstance(gene_cfg, dict):
            gene_cfg = ScUniFlowGeneConfig(**gene_cfg)
        if isinstance(spatial_cfg, dict):
            spatial_cfg = ScUniFlowSpatialConfig(**spatial_cfg)
        if isinstance(modality_cfg, dict):
            modality_cfg = ModalityAdapterConfig(**modality_cfg)
        cfg.modality_adapters = modality_cfg
        self._annotation_config = annotation_cfg
        self.encoder = SparseGeneExpressionEncoder(
            cfg.n_genes, cfg.encoder, cfg.context, modality_cfg
        )
        # Optional BrainWM capability branches informed by reference methods.
        # They are intentionally created
        # only when enabled so a legacy configuration produces exactly the
        # historical parameter/state-dict graph.
        self.scfit_temporal_adapter = None
        if scfit_cfg.enabled:
            temporal_cfg = scfit_cfg
            self.scfit_temporal_adapter = ScFiTTemporalAdapter(
                cfg.encoder.latent_dim,
                hidden_dim=temporal_cfg.hidden_dim,
                time_dim=temporal_cfg.time_dim,
                layers=temporal_cfg.layers,
                dropout=temporal_cfg.dropout,
                parameterization=temporal_cfg.parameterization,
                flow_eps=temporal_cfg.flow_eps,
                biological_time_scale=temporal_cfg.biological_time_scale,
                mode=temporal_cfg.mode,
                gate_init=temporal_cfg.gate_init,
            )
        if gene_cfg.enabled:
            gene_order = _load_gene_order(gene_cfg.gene_order_path, cfg.n_genes)
            gene_adapter = ScUniFlowSparseGeneAdapter(
                cfg.encoder.gene_dim,
                order_mode=gene_cfg.order_mode,
                random_seed=gene_cfg.random_seed,
                grid_shape=gene_cfg.grid_shape,
                ffn_multiplier=gene_cfg.ffn_multiplier,
                dropout=gene_cfg.dropout,
                gate_init=gene_cfg.gate_init,
                gene_order=gene_order,
            )
            # Register under the encoder exactly once; the property below
            # exposes it through BrainWorldModel without duplicate state-dict
            # keys such as ``model.*`` and ``encoder.*``.
            self.encoder.scuniflow_gene_adapter = gene_adapter
        self.scuniflow_spatial_adapter = None
        if spatial_cfg.enabled:
            self.scuniflow_spatial_adapter = ScUniFlowSpatialAdapter(
                cfg.encoder.latent_dim,
                cfg.graph.coordinate_dim,
                hidden_dim=spatial_cfg.hidden_dim,
                dropout=spatial_cfg.dropout,
                gate_init=spatial_cfg.gate_init,
            )
        self.graph = DynamicSpatialGraph(cfg.encoder.latent_dim, cfg.graph)
        self.hierarchy = CellNicheTissueHierarchy(
            cfg.encoder.latent_dim, cfg.graph.coordinate_dim, cfg.hierarchy
        )
        self.dynamics = ControlledLatentDynamics(
            cfg.encoder.latent_dim,
            cfg.dynamics,
            cfg.cross_species,
            cfg.intervention,
        )
        self.integrator = FixedStepIntegrator(cfg.dynamics)
        self.flow_matching = (
            DisentangledLatentFlow(
                cfg.encoder.latent_partitions,
                cfg.context.output_dim,
                cfg.flow_matching,
            )
            if cfg.flow_matching.enabled
            else None
        )
        self.assay_discriminator = (
            AssayDiscriminator(cfg.encoder.latent_dim, cfg.assay_alignment)
            if cfg.assay_alignment.enabled
            else None
        )
        self.mgm_head = (
            MaskedGeneModelingHead(
                cfg.encoder.gene_dim,
                cfg.decoder.gene_dim,
                self.encoder.gene_embedding,
                cfg.masked_gene_modeling,
            )
            if cfg.masked_gene_modeling.enabled
            else None
        )
        self.decoder = NegativeBinomialDecoder(
            n_genes=cfg.n_genes,
            latent_dim=cfg.encoder.latent_dim,
            context_dim=cfg.context.output_dim,
            control_dim=cfg.intervention.control_dim,
            config=cfg.decoder,
            tied_gene_embedding=self.encoder.gene_embedding,
        )
        self.lineage_head = nn.Linear(cfg.encoder.latent_dim, cfg.decoder.n_lineages)
        # The hierarchical annotation readout is opt-in.  Keeping the module
        # absent when disabled preserves the historical checkpoint key set;
        # callers can also pass an externally managed head to ``annotate`` or
        # ``forward`` when Stage-0 is frozen.
        self.annotation_head = None
        if annotation_cfg.enabled:
            n_levels = len(annotation_cfg.output_dims)
            intermediate_dims = [cfg.encoder.gene_dim] * max(n_levels - 1, 0)
            self.annotation_head = HierarchicalResidualAnnotationHead(
                input_dim=cfg.encoder.latent_dim,
                output_dims=annotation_cfg.output_dims,
                hidden_dim=annotation_cfg.hidden_dim,
                residual_mode=annotation_cfg.residual_mode,
                dropout=annotation_cfg.dropout,
                intermediate_dims=intermediate_dims,
            )

    _OPTIONAL_STATE_PREFIXES = (
        "annotation_head.",
        "scfit_temporal_adapter.",
        "scuniflow_spatial_adapter.",
        "encoder.scuniflow_gene_adapter.",
        "encoder.modality_router.",
        "encoder.modality_gate",
        # Learned MGM/null embeddings are present only when the explicit
        # missingness route is enabled.  Treat them like other optional
        # branches so a legacy checkpoint can initialize the shared backbone.
        "encoder.mask_token",
        "encoder.missing_token",
    )

    def load_compatible_state_dict(
        self,
        state: dict[str, Tensor] | dict[str, object],
        *,
        strict: bool = True,
    ):
        """Load a legacy checkpoint while allowing optional branch keys.

        Existing training/evaluation scripts can continue to use strict
        loading for the default model.  This explicit helper is for controlled
        plugin experiments: a pre-plugin Stage-0/1/2 checkpoint initializes
        the shared backbone, while newly enabled adapter parameters retain
        their constructor initialization.  Conversely, optional keys in a
        plugin checkpoint are ignored when loading into a legacy model.
        Unexpected or missing non-optional keys still fail when ``strict`` is
        true.
        """
        if not isinstance(state, dict):
            raise TypeError("state must be a state-dict mapping")
        # Accept a full training checkpoint for convenience without changing
        # the behavior of nn.Module.load_state_dict itself.
        payload = state.get("model")
        if isinstance(payload, dict):
            state_dict = payload
        else:
            state_dict = state
        if not isinstance(state_dict, dict):
            raise TypeError("state['model'] must be a state-dict mapping")
        filtered = {
            key: value
            for key, value in state_dict.items()
            if not any(str(key).startswith(prefix) for prefix in self._OPTIONAL_STATE_PREFIXES)
        }
        # Keep optional keys that correspond to branches present in this model;
        # filtering only the opposite side avoids silently dropping a trained
        # adapter when the target model has it enabled.
        target_prefixes = tuple(
            prefix
            for prefix in self._OPTIONAL_STATE_PREFIXES
            if any(key.startswith(prefix) for key in self.state_dict())
        )
        source_prefixes = tuple(
            prefix
            for prefix in self._OPTIONAL_STATE_PREFIXES
            if any(str(key).startswith(prefix) for key in state_dict)
        )
        if target_prefixes:
            filtered.update(
                {
                    key: value
                    for key, value in state_dict.items()
                    if str(key).startswith(target_prefixes)
                }
            )
        result = super().load_state_dict(filtered, strict=False)
        if strict:
            # Absence of an entire optional prefix is the intended legacy
            # compatibility case.  Once a checkpoint contains any keys for a
            # branch that also exists in the target, require that branch to be
            # complete so a truncated adapter cannot load silently.
            shared_optional_prefixes = tuple(
                prefix for prefix in target_prefixes if prefix in source_prefixes
            )
            missing = [
                key
                for key in result.missing_keys
                if not any(key.startswith(prefix) for prefix in self._OPTIONAL_STATE_PREFIXES)
                or key.startswith(shared_optional_prefixes)
            ]
            unexpected = [
                key
                for key in result.unexpected_keys
                if not any(key.startswith(prefix) for prefix in self._OPTIONAL_STATE_PREFIXES)
            ]
            if missing or unexpected:
                raise RuntimeError(
                    "incompatible BrainWM checkpoint; missing="
                    f"{missing}, unexpected={unexpected}"
                )
        return result

    @property
    def scuniflow_gene_adapter(self) -> nn.Module | None:
        """Expose the encoder-owned ordered gene branch without re-registering it."""
        return getattr(self.encoder, "scuniflow_gene_adapter", None)

    @property
    def scbrain_depth_head(self) -> nn.Module | None:
        """Explicit integration-contract alias for ``annotation_head``."""
        return self.annotation_head

    def optional_integration_modules(self) -> dict[str, nn.Module]:
        """Return enabled BrainWM capability branches under stable names.

        The branches are informed by reference-paper mechanisms, but are
        trained and judged only on BrainWM-native tasks and metrics.

        The mapping is intentionally explicit.  Training scripts can request
        only an adapter (while freezing the foundation backbone) without
        relying on implementation details such as the gene branch living
        inside ``encoder``.
        """
        modules: dict[str, nn.Module] = {}
        for name in (
            "annotation_head",
            "scfit_temporal_adapter",
            "scuniflow_gene_adapter",
            "scuniflow_spatial_adapter",
        ):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                modules[name] = module
        modality_router = getattr(self.encoder, "modality_router", None)
        if isinstance(modality_router, nn.Module):
            # The gate lives beside (rather than inside) the router to retain
            # the legacy encoder state-dict layout.  ``freeze_for_stage``
            # handles the paired parameter when this stable component name is
            # selected.
            modules["modality_adapters"] = modality_router
        return modules

    def structural_context(
        self,
        z: Tensor,
        batch: WorldModelBatch,
        source_mask: Tensor | None = None,
        neutral_query_mask: Tensor | None = None,
    ) -> StructuralContext:
        message, edge_index, edge_weight = self.graph(
            z,
            batch.spatial_coords,
            batch.batch_index,
            edge_index=batch.edge_index,
            ligand_receptor=batch.ligand_receptor,
            source_mask=source_mask,
            neutral_query_mask=neutral_query_mask,
            spatial_presence=batch.spatial_presence,
        )
        (
            niche_context,
            tissue_context,
            niche_probability,
            niche_state,
            tissue_state,
        ) = self.hierarchy(
            z,
            batch.spatial_coords,
            batch.batch_index,
            source_mask=source_mask,
            neutral_query_mask=neutral_query_mask,
            spatial_presence=batch.spatial_presence,
        )
        return StructuralContext(
            message=message,
            niche_context=niche_context,
            tissue_context=tissue_context,
            edge_index=edge_index,
            edge_weight=edge_weight,
            niche_probability=niche_probability,
            niche_state=niche_state,
            tissue_state=tissue_state,
        )

    @staticmethod
    def _has_spatial_observations(batch: WorldModelBatch) -> bool:
        """Whether a batch contains at least one usable spatial observation."""
        if batch.spatial_coords is None:
            return False
        if batch.spatial_presence is None:
            return True
        return bool(torch.any(batch.spatial_presence).item())

    @staticmethod
    def empty_structural_context(z: Tensor) -> StructuralContext:
        batch, latent = z.shape
        edge = torch.empty((2, 0), dtype=torch.long, device=z.device)
        return StructuralContext(
            message=torch.zeros_like(z),
            niche_context=torch.zeros_like(z),
            tissue_context=torch.zeros_like(z),
            edge_index=edge,
            edge_weight=z.new_empty((0,)),
            niche_probability=z.new_empty((batch, 0)),
            niche_state=z.new_empty((0, 0, latent)),
            tissue_state=z.new_empty((0, latent)),
        )

    def _apply_spatial_adapter(
        self,
        z: Tensor,
        batch: WorldModelBatch,
        structure: StructuralContext,
    ) -> Tensor:
        """Apply the optional Stage-2 spatial residual exactly once per state.

        The existing graph and niche hierarchy remain the source of structural
        supervision.  The scUniFlow-informed branch receives the same
        section-local edge set and is therefore an additive context pathway,
        not a second (potentially leaking) graph implementation.
        """
        adapter = self.scuniflow_spatial_adapter
        if adapter is None or batch.spatial_coords is None:
            return z
        presence = batch.spatial_presence
        coords = batch.spatial_coords.to(device=z.device)
        if presence is not None:
            presence = presence.to(device=z.device)
            if not torch.any(presence):
                return z
            # Keep placeholder/NaN coordinates for unavailable cells out of
            # the adapter.  The final mask below also makes those rows an
            # exact identity in the latent state.
            coords = torch.where(presence[:, None], coords, torch.zeros_like(coords))
            if not torch.isfinite(coords[presence]).all():
                raise ValueError("spatial_coords must be finite for present cells")
            edge_index = structure.edge_index
            if edge_index.numel():
                edge_index = edge_index.to(device=z.device, dtype=torch.long)
                keep = presence[edge_index[0]] & presence[edge_index[1]]
                edge_index = edge_index[:, keep]
            else:
                edge_index = edge_index.to(device=z.device, dtype=torch.long)
        else:
            edge_index = structure.edge_index
        adapted = adapter(
            z,
            coords,
            batch.batch_index.to(device=z.device),
            edge_index,
        )
        if presence is not None:
            adapted = torch.where(presence[:, None], adapted, z)
        return adapted

    def encode(
        self, batch: WorldModelBatch, *, gene_order: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.encoder(
            batch.expression,
            batch.categorical_context,
            batch.continuous_context,
            batch.optional_modalities,
            token_mask=batch.expression.token_mask,
            gene_order=gene_order,
            modality_inputs=batch.modality_inputs,
            modality_presence=batch.modality_presence,
        )

    def encode_anatomy(
        self, batch: WorldModelBatch
    ) -> tuple[Tensor, Tensor]:
        """Return the shared UBERON embedding and its availability mask.

        Anatomical ontology information is a categorical context input and is
        therefore available for scRNA/snRNA as well as spatial assays.  This
        method exposes the exact embedding already consumed by the encoder;
        it does not infer anatomy from coordinates or free-text labels.
        """
        cfg = self.config
        batch.validate(
            cfg.n_genes,
            cfg.graph.coordinate_dim,
            cfg.intervention.control_dim,
            cfg.cross_species.n_species,
            cfg.encoder.optional_modality_dim,
            cfg.modality_adapters.input_dims if cfg.modality_adapters.enabled else None,
        )
        return (
            self.encoder.anatomy_embedding(batch.categorical_context),
            self.encoder.anatomy_presence(batch.categorical_context),
        )

    def encode_depths(
        self,
        batch: WorldModelBatch,
        *,
        token_mask: Tensor | None = None,
        gene_order: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the three Stage-0 states used by hierarchical readouts.

        The states are, in order, token-local pooled, token-context pooled,
        and the deterministic variational posterior mean (or fused latent for
        a deterministic encoder).  This is a readout view only: it does not
        alter the legacy ``encode`` return contract or any Stage-1/Stage-2
        rollout semantics.
        """
        cfg = self.config
        batch.validate(
            cfg.n_genes,
            cfg.graph.coordinate_dim,
            cfg.intervention.control_dim,
            cfg.cross_species.n_species,
            cfg.encoder.optional_modality_dim,
            cfg.modality_adapters.input_dims if cfg.modality_adapters.enabled else None,
        )
        if token_mask is None:
            token_mask = batch.expression.token_mask
        states = self.encoder.depth_aligned_states(
            batch.expression,
            batch.categorical_context,
            batch.continuous_context,
            batch.optional_modalities,
            token_mask=token_mask,
            gene_order=gene_order,
            modality_inputs=batch.modality_inputs,
            modality_presence=batch.modality_presence,
        )
        if len(states) != 3:
            # ``depth_aligned_states`` is a stable three-state interface.  A
            # fail-closed check makes a future encoder change explicit rather
            # than silently routing the wrong state to a taxonomy level.
            raise RuntimeError(
                "encoder.depth_aligned_states must return exactly three states"
            )
        return states

    @staticmethod
    def _route_annotation_depths(
        states: Sequence[Tensor], n_levels: int
    ) -> tuple[Tensor, ...]:
        """Select the deepest ``n_levels`` states for a hierarchy head."""
        if n_levels < 1:
            raise ValueError("annotation head must expose at least one level")
        if n_levels > len(states):
            raise ValueError(
                "annotation head has more levels than the available Stage-0 depth states"
            )
        routed = tuple(states[-n_levels:])
        return routed

    def annotate(
        self,
        batch: WorldModelBatch,
        annotation_head: HierarchicalResidualAnnotationHead | None = None,
        *,
        labels: Sequence[Tensor | None] | Tensor | None = None,
        level_weights: Sequence[float] | None = None,
        token_mask: Tensor | None = None,
        gene_order: Tensor | None = None,
        return_features: bool = False,
    ) -> HierarchicalAnnotationOutput:
        """Run the optional scBrain-inspired hierarchical Stage-0 readout.

        ``annotation_head`` may be supplied externally so a frozen Stage-0
        checkpoint can be evaluated/trained without changing its state dict.
        If omitted, the head configured by ``model.annotation`` is used.  The
        deepest state is passed as ``z`` and preceding states are passed as
        continuous intermediate features; no argmax or detached labels enter
        the residual path.
        """
        head = annotation_head if annotation_head is not None else self.annotation_head
        if head is None:
            raise ValueError(
                "annotation head is disabled; pass annotation_head or enable model.annotation"
            )
        if not isinstance(head, HierarchicalResidualAnnotationHead):
            raise TypeError(
                "annotation_head must be a HierarchicalResidualAnnotationHead"
            )
        states = self.encode_depths(
            batch, token_mask=token_mask, gene_order=gene_order
        )
        routed = self._route_annotation_depths(states, head.n_levels)
        # ``HierarchicalResidualAnnotationHead`` appends z when it receives
        # n_levels - 1 intermediate states.  This keeps the deepest latent as
        # the canonical final state for one-, two-, and three-level taxonomies.
        head_result = head.forward_with_features(
            routed[-1], routed[:-1]
        )
        logits, features = head_result
        if labels is None:
            loss = None
        else:
            if level_weights is None and head is self.annotation_head:
                configured = self._annotation_config.level_weights
                level_weights = configured or None
            loss = hierarchical_annotation_loss(
                logits,
                labels,
                level_weights=level_weights,
            )
        return HierarchicalAnnotationOutput(
            depth_states=routed,
            logits=logits,
            features=features if return_features else None,
            loss=loss,
        )

    def _annotate_from_states(
        self,
        states: Sequence[Tensor],
        head: HierarchicalResidualAnnotationHead,
        *,
        labels: Sequence[Tensor | None] | Tensor | None = None,
        level_weights: Sequence[float] | None = None,
        return_features: bool = False,
    ) -> HierarchicalAnnotationOutput:
        """Apply an annotation head to already-computed Stage-0 states."""
        routed = self._route_annotation_depths(states, head.n_levels)
        logits, features = head.forward_with_features(routed[-1], routed[:-1])
        if labels is None:
            loss = None
        else:
            loss = hierarchical_annotation_loss(
                logits,
                labels,
                level_weights=level_weights,
            )
        return HierarchicalAnnotationOutput(
            depth_states=routed,
            logits=logits,
            features=features if return_features else None,
            loss=loss,
        )

    def forward(
        self,
        batch: WorldModelBatch,
        target_time_days: Tensor | None = None,
        sample_decoder: bool | None = None,
        compute_structure: bool = True,
        contrastive_batch: WorldModelBatch | None = None,
        compute_auxiliary: bool = False,
        transition_target_batch: WorldModelBatch | None = None,
        token_mask: Tensor | None = None,
        mgm_token_mask: Tensor | None = None,
        mgm_candidate_gene_index: Tensor | None = None,
        mgm_target_gene_index: Tensor | None = None,
        return_token_features: bool = False,
        structural_source_mask: Tensor | None = None,
        structural_neutral_query_mask: Tensor | None = None,
        annotation_head: HierarchicalResidualAnnotationHead | None = None,
        annotation_labels: Sequence[Tensor | None] | Tensor | None = None,
        annotation_level_weights: Sequence[float] | None = None,
        return_annotation_features: bool = False,
        gene_order: Tensor | None = None,
        missing_token_mask: Tensor | None = None,
    ) -> BrainWorldModelOutput | TemporalTransitionOutput:
        cfg = self.config
        batch.validate(
            cfg.n_genes,
            cfg.graph.coordinate_dim,
            cfg.intervention.control_dim,
            cfg.cross_species.n_species,
            cfg.encoder.optional_modality_dim,
            cfg.modality_adapters.input_dims if cfg.modality_adapters.enabled else None,
        )
        if token_mask is None:
            token_mask = batch.expression.token_mask
        if missing_token_mask is None:
            missing_token_mask = batch.expression.missing_token_mask
        effective_annotation_labels = (
            annotation_labels
            if annotation_labels is not None
            else batch.annotation_labels
        )
        if transition_target_batch is not None:
            if target_time_days is None:
                raise ValueError("temporal transition requires target_time_days")
            if (
                annotation_head is not None
                or effective_annotation_labels is not None
                or return_annotation_features
            ):
                raise ValueError(
                    "hierarchical annotation is only available on the Stage-0 forward path"
                )
            return self.temporal_transition(
                batch,
                transition_target_batch,
                target_time_days,
                compute_structure=compute_structure,
                structural_source_mask=structural_source_mask,
                structural_neutral_query_mask=structural_neutral_query_mask,
                gene_order=gene_order,
            )
        annotation_output: HierarchicalAnnotationOutput | None = None
        effective_annotation_head = (
            annotation_head if annotation_head is not None else self.annotation_head
        )
        if effective_annotation_head is not None and not isinstance(
            effective_annotation_head, HierarchicalResidualAnnotationHead
        ):
            raise TypeError(
                "annotation_head must be a HierarchicalResidualAnnotationHead"
            )
        if effective_annotation_head is not None:
            (
                z_initial,
                latent_mean,
                latent_logvar,
                context,
                partitions,
                depth_states,
            ) = self.encoder.forward_with_depth_states(
                batch.expression,
                batch.categorical_context,
                batch.continuous_context,
                batch.optional_modalities,
                token_mask=token_mask,
                gene_order=gene_order,
                modality_inputs=batch.modality_inputs,
                modality_presence=batch.modality_presence,
                missing_token_mask=missing_token_mask,
            )
            if annotation_level_weights is None and effective_annotation_head is self.annotation_head:
                configured_weights = self.config.annotation.level_weights
                annotation_level_weights = configured_weights or None
            annotation_output = self._annotate_from_states(
                depth_states,
                effective_annotation_head,
                labels=effective_annotation_labels,
                level_weights=annotation_level_weights,
                return_features=return_annotation_features,
            )
        else:
            z_initial, latent_mean, latent_logvar, context, partitions = self.encoder(
                batch.expression,
                batch.categorical_context,
                batch.continuous_context,
                batch.optional_modalities,
                token_mask=token_mask,
                gene_order=gene_order,
                modality_inputs=batch.modality_inputs,
                modality_presence=batch.modality_presence,
                missing_token_mask=missing_token_mask,
            )
        # Keep the ontology-backed anatomy token available for diagnostics and
        # optional downstream objectives.  The reconstruction path above has
        # already consumed the same embedding through ``latent_context``;
        # this readout adds no alternate source of anatomical information.
        anatomy_embedding = self.encoder.anatomy_embedding(
            batch.categorical_context
        )
        anatomy_presence = self.encoder.anatomy_presence(
            batch.categorical_context
        )
        trajectory: list[Tensor] = [z_initial]
        uncertainty = None

        if target_time_days is not None:
            if not cfg.dynamics.enabled and self.scfit_temporal_adapter is None:
                raise ValueError(
                    "target_time_days requires model.dynamics.enabled=true or "
                    "model.scfit_temporal_adapter.enabled=true"
                )
            if target_time_days.shape != batch.time_days.shape:
                raise ValueError("target_time_days must have shape [B]")
            if torch.any(target_time_days < batch.time_days):
                raise ValueError("target_time_days must not precede source time")

            if self.scfit_temporal_adapter is not None:
                # scFiT's endpoint branch is a population-level update.  It
                # intentionally bypasses the ODE solver when selected; the
                # adapter still receives the same source/target biological
                # times and can be run in residual mode for a warm start.
                temporal = self.scfit_temporal_adapter(
                    z_initial,
                    batch.time_days,
                    target_time_days,
                    flow_time=0.0,
                )
                z_final = temporal.state
                trajectory = [z_initial, z_final]
                uncertainty = torch.zeros_like(z_final)
            else:
                def rhs(z: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
                    structure = (
                        self.structural_context(
                            z,
                            batch,
                            source_mask=structural_source_mask,
                            neutral_query_mask=structural_neutral_query_mask,
                        )
                        if compute_structure
                        else self.empty_structural_context(z)
                    )
                    return self.dynamics(
                        z,
                        structure,
                        batch.control,
                        batch.delta_control,
                        batch.species_id,
                        time,
                    )

                z_final, trajectory, uncertainty = self.integrator(
                    rhs, z_initial, batch.time_days, target_time_days
                )
        else:
            z_final = z_initial

        adapter_endpoint = None
        adapter_velocity = None
        if target_time_days is not None and self.scfit_temporal_adapter is not None:
            # Keep the endpoint/velocity available for diagnostics and for
            # downstream loss wrappers without changing the latent state
            # returned by the established BrainWM API.
            adapter_endpoint = temporal.endpoint
            adapter_velocity = temporal.velocity

        structure = (
            self.structural_context(
                z_final,
                batch,
                source_mask=structural_source_mask,
                neutral_query_mask=structural_neutral_query_mask,
            )
            if compute_structure
            else self.empty_structural_context(z_final)
        )
        spatial_pre_adapter = None
        spatial_adapter_delta = None
        # A spatial adapter is meaningful only when the batch carries actual
        # coordinates.  Keep the branch a true identity for molecular-only
        # batches, including telemetry (``None`` means unavailable rather
        # than an observed zero spatial residual).
        if (
            compute_structure
            and self._has_spatial_observations(batch)
            and self.scuniflow_spatial_adapter is not None
        ):
            spatial_pre_adapter = z_final
            z_final = self._apply_spatial_adapter(z_final, batch, structure)
            spatial_adapter_delta = z_final - spatial_pre_adapter
            # Recompute graph/niche features on the updated Stage-2 state so
            # decoder and downstream losses see a self-consistent structure.
            structure = self.structural_context(
                z_final,
                batch,
                source_mask=structural_source_mask,
                neutral_query_mask=structural_neutral_query_mask,
            )
        decode_genes = (
            batch.expression.measured_gene_index
            if batch.decode_gene_index is None
            else batch.decode_gene_index
        )
        decoder = self.decoder(
            z_final,
            context,
            batch.expression.library_size,
            decode_genes,
            control=batch.control,
            sample=sample_decoder,
        )
        lineage_probabilities = torch.softmax(self.lineage_head(z_final), dim=-1)
        contrastive_z = None
        if contrastive_batch is not None:
            contrastive_batch.validate(
                cfg.n_genes,
                cfg.graph.coordinate_dim,
                cfg.intervention.control_dim,
                cfg.cross_species.n_species,
                cfg.encoder.optional_modality_dim,
                cfg.modality_adapters.input_dims if cfg.modality_adapters.enabled else None,
            )
            _, contrastive_z, _, _, _ = self.encode(
                contrastive_batch, gene_order=gene_order
            )
        assay_logits = None
        flow_prediction = None
        flow_target = None
        if compute_auxiliary:
            if self.assay_discriminator is not None:
                assay_logits = self.assay_discriminator(latent_mean)
            if self.flow_matching is not None:
                flow_prediction, flow_target = self.flow_matching.matching_pair(
                    latent_mean, context, batch.time_days
                )
        token_features = None
        token_row_index = None
        masked_gene_logits = None
        masked_gene_candidates = None
        if (
            return_token_features
            or mgm_candidate_gene_index is not None
            or mgm_target_gene_index is not None
        ):
            if self.mgm_head is None:
                raise ValueError("MGM head is disabled in this configuration")
            token_features, token_row_index = self.encoder.token_features(
                batch.expression,
                batch.categorical_context,
                batch.continuous_context,
                batch.optional_modalities,
                token_mask=token_mask,
                gene_order=gene_order,
                missing_token_mask=missing_token_mask,
            )
            if (
                mgm_candidate_gene_index is not None
                and mgm_target_gene_index is not None
            ):
                raise ValueError(
                    "provide either MGM candidates or MGM targets, not both"
                )
            supervision_mask = (
                mgm_token_mask if mgm_token_mask is not None else token_mask
            )
            if mgm_candidate_gene_index is not None:
                if supervision_mask is None:
                    raise ValueError(
                        "an MGM supervision mask is required for candidate logits"
                    )
                masked_gene_logits = self.mgm_head(
                    token_features[supervision_mask], mgm_candidate_gene_index
                )
                masked_gene_candidates = mgm_candidate_gene_index
            if mgm_target_gene_index is not None:
                if supervision_mask is None:
                    raise ValueError(
                        "an MGM supervision mask is required for target logits"
                    )
                masked_gene_logits = self.mgm_head.sampled_forward(
                    token_features[supervision_mask],
                    mgm_target_gene_index,
                    cfg.n_genes,
                    cfg.masked_gene_modeling.negative_samples,
                )
        return BrainWorldModelOutput(
            z_initial=z_initial,
            z_final=z_final,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            context=context,
            latent_partitions=partitions,
            decoder=decoder,
            structure=structure,
            predicted_positions=batch.spatial_coords,
            lineage_probabilities=lineage_probabilities,
            uncertainty=uncertainty,
            trajectory=trajectory,
            contrastive_z=contrastive_z,
            assay_logits=assay_logits,
            flow_prediction=flow_prediction,
            flow_target=flow_target,
            token_features=token_features,
            token_row_index=token_row_index,
            masked_gene_logits=masked_gene_logits,
            masked_gene_candidates=masked_gene_candidates,
            depth_states=(
                annotation_output.depth_states if annotation_output is not None else None
            ),
            annotation_logits=(
                annotation_output.logits if annotation_output is not None else None
            ),
            annotation_features=(
                annotation_output.features if annotation_output is not None else None
            ),
            annotation_loss=(
                annotation_output.loss if annotation_output is not None else None
            ),
            adapter_endpoint=adapter_endpoint,
            adapter_velocity=adapter_velocity,
            spatial_pre_adapter=spatial_pre_adapter,
            spatial_adapter_delta=spatial_adapter_delta,
            anatomy_embedding=anatomy_embedding,
            anatomy_presence=anatomy_presence,
        )

    def temporal_transition(
        self,
        source_batch: WorldModelBatch,
        target_batch: WorldModelBatch,
        target_time_days: Tensor,
        compute_structure: bool = False,
        structural_source_mask: Tensor | None = None,
        structural_neutral_query_mask: Tensor | None = None,
        gene_order: Tensor | None = None,
    ) -> TemporalTransitionOutput:
        """Roll a frozen Stage-0 mean latent toward a cross-sectional target."""
        cfg = self.config
        target_batch.validate(
            cfg.n_genes,
            cfg.graph.coordinate_dim,
            cfg.intervention.control_dim,
            cfg.cross_species.n_species,
            cfg.encoder.optional_modality_dim,
            cfg.modality_adapters.input_dims if cfg.modality_adapters.enabled else None,
        )
        if source_batch.expression.batch_size != target_batch.expression.batch_size:
            raise ValueError("source and target temporal batches must have equal size")
        if target_time_days.shape != source_batch.time_days.shape:
            raise ValueError("target_time_days must have shape [B]")
        if torch.any(target_time_days <= source_batch.time_days):
            raise ValueError("temporal targets must strictly follow source times")

        with torch.no_grad():
            _, source_mean, _, _, _ = self.encode(
                source_batch, gene_order=gene_order
            )
            _, target_mean, _, target_context, _ = self.encode(
                target_batch, gene_order=gene_order
            )

        temporal_adapter_output = None
        if self.scfit_temporal_adapter is not None:
            temporal_adapter_output = self.scfit_temporal_adapter(
                source_mean,
                source_batch.time_days,
                target_time_days,
                flow_time=0.0,
            )
            predicted = temporal_adapter_output.state
            trajectory = [source_mean, predicted]
            uncertainty = torch.zeros_like(predicted)
        else:
            def rhs(z: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
                structure = (
                    self.structural_context(
                        z,
                        source_batch,
                        source_mask=structural_source_mask,
                        neutral_query_mask=structural_neutral_query_mask,
                    )
                    if compute_structure
                    else self.empty_structural_context(z)
                )
                return self.dynamics(
                    z,
                    structure,
                    source_batch.control,
                    source_batch.delta_control,
                    source_batch.species_id,
                    time,
                )

            predicted, trajectory, uncertainty = self.integrator(
                rhs,
                source_mean,
                source_batch.time_days,
                target_time_days,
            )
        spatial_pre_adapter = None
        spatial_adapter_delta = None
        if (
            compute_structure
            and self._has_spatial_observations(source_batch)
            and self.scuniflow_spatial_adapter is not None
        ):
            spatial_pre_adapter = predicted
            structure = self.structural_context(
                predicted,
                source_batch,
                source_mask=structural_source_mask,
                neutral_query_mask=structural_neutral_query_mask,
            )
            predicted = self._apply_spatial_adapter(predicted, source_batch, structure)
            spatial_adapter_delta = predicted - spatial_pre_adapter
            trajectory[-1] = predicted
        return TemporalTransitionOutput(
            source_latent=source_mean,
            target_latent=target_mean,
            target_context=target_context,
            predicted_latent=predicted,
            trajectory=trajectory,
            uncertainty=uncertainty,
            adapter_endpoint=(
                temporal_adapter_output.endpoint
                if temporal_adapter_output is not None else None
            ),
            adapter_velocity=(
                temporal_adapter_output.velocity
                if temporal_adapter_output is not None else None
            ),
            spatial_pre_adapter=spatial_pre_adapter,
            spatial_adapter_delta=spatial_adapter_delta,
        )

    def counterfactual_effect(
        self,
        batch: WorldModelBatch,
        target_time_days: Tensor,
        treated_control: Tensor,
        baseline_control: Tensor,
    ) -> tuple[Tensor, BrainWorldModelOutput, BrainWorldModelOutput]:
        """Return E[x|do(treated)] - E[x|do(baseline)] on identical initial cells."""
        treated = replace(
            batch,
            control=treated_control,
            delta_control=treated_control - baseline_control,
        )
        baseline = replace(
            batch,
            control=baseline_control,
            delta_control=torch.zeros_like(baseline_control),
        )
        treated_output = self(treated, target_time_days=target_time_days)
        baseline_output = self(baseline, target_time_days=target_time_days)
        effect = treated_output.decoder.mean - baseline_output.decoder.mean
        treated_output.treatment_effect = effect
        return effect, treated_output, baseline_output
