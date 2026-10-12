"""Strict tensor contracts for model input and output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class SparseExpressionBatch:
    """CSR-like nonzero raw counts in the global gene vocabulary."""

    row_ptr: Tensor
    gene_index: Tensor
    count: Tensor
    library_size: Tensor
    measured_gene_index: Tensor
    token_mask: Tensor | None = None
    # ``missing_token_mask`` marks explicit placeholders for genes that were
    # not measured in the input assay/panel.  It is intentionally separate
    # from ``token_mask`` (active MGM masking) and from a measured zero.
    # Appending this field keeps all historical positional constructors valid.
    missing_token_mask: Tensor | None = None
    # Explicitly materialized measured-but-undetected genes.  A zero count on
    # its own is ambiguous in a sparse representation, so this mask lets the
    # encoder distinguish a measured zero from an unmeasured placeholder.
    measured_zero_mask: Tensor | None = None
    # Optional panel-level summary supplied by a reader.  The canonical form
    # is ``[2]`` or ``[B, 2]`` containing measured and missing fractions; the
    # encoder also accepts a scalar/[B] missing-fraction shorthand.
    panel_missing_summary: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.row_ptr.numel() - 1)

    def validate(self, n_genes: int) -> None:
        if self.row_ptr.ndim != 1 or self.row_ptr.numel() < 2:
            raise ValueError("expression.row_ptr must be one-dimensional with B+1 entries")
        if self.gene_index.ndim != 1 or self.count.ndim != 1:
            raise ValueError("expression gene_index/count must be one-dimensional")
        if self.gene_index.numel() != self.count.numel():
            raise ValueError("gene_index and count lengths differ")
        if self.token_mask is not None:
            if self.token_mask.shape != self.gene_index.shape:
                raise ValueError("token_mask must match gene_index/count shape")
            if self.token_mask.dtype != torch.bool:
                raise ValueError("token_mask must be boolean")
        for name in ("missing_token_mask", "measured_zero_mask"):
            value = getattr(self, name)
            if value is not None:
                if value.shape != self.gene_index.shape:
                    raise ValueError(f"{name} must match gene_index/count shape")
                if value.dtype != torch.bool:
                    raise ValueError(f"{name} must be boolean")
        if (
            self.missing_token_mask is not None
            and self.measured_zero_mask is not None
            and torch.any(self.missing_token_mask & self.measured_zero_mask)
        ):
            raise ValueError("a token cannot be both unmeasured and a measured zero")
        if self.missing_token_mask is not None and torch.any(
            self.count[self.missing_token_mask] != 0
        ):
            raise ValueError("unmeasured placeholder tokens must have zero count")
        if self.measured_zero_mask is not None and torch.any(
            self.count[self.measured_zero_mask] != 0
        ):
            raise ValueError("measured-zero tokens must have zero count")
        if int(self.row_ptr[0]) != 0 or int(self.row_ptr[-1]) != self.count.numel():
            raise ValueError("row_ptr does not bound the token arrays")
        if self.library_size.shape != (self.batch_size,):
            raise ValueError("library_size must have shape [B]")
        if self.gene_index.numel() and (int(self.gene_index.min()) < 0
                                       or int(self.gene_index.max()) >= n_genes):
            raise ValueError("gene_index is outside the configured vocabulary")
        if self.measured_gene_index.numel() and (
                int(self.measured_gene_index.min()) < 0
                or int(self.measured_gene_index.max()) >= n_genes):
            raise ValueError("measured_gene_index is outside the configured vocabulary")
        if torch.any(self.count < 0):
            raise ValueError("raw counts cannot be negative")
        if self.panel_missing_summary is not None:
            summary = self.panel_missing_summary
            if summary.ndim not in (0, 1, 2):
                raise ValueError(
                    "panel_missing_summary must have shape [2], [B, 2], or a scalar"
                )
            if summary.ndim == 1 and summary.numel() not in (1, 2, self.batch_size):
                raise ValueError(
                    "panel_missing_summary one-dimensional form must have length 1, 2, or B"
                )
            if summary.ndim == 2 and summary.shape[0] not in (1, self.batch_size):
                raise ValueError("panel_missing_summary must have shape [B, D]")
            if summary.ndim == 2 and summary.shape[1] not in (1, 2):
                raise ValueError("panel_missing_summary second dimension must be 1 or 2")
            if not summary.is_floating_point() and summary.dtype != torch.bool:
                raise ValueError("panel_missing_summary must be numeric")
            if not torch.isfinite(summary.float()).all():
                raise ValueError("panel_missing_summary must be finite")
            if torch.any(summary.float() < 0) or torch.any(summary.float() > 1):
                raise ValueError("panel_missing_summary values must lie in [0, 1]")

    def panel_missing_summary_for(self, n_genes: int) -> Tensor:
        """Return a per-cell ``[measured_fraction, missing_fraction]`` tensor.

        Readers can provide a compact summary when the global vocabulary is
        known.  Otherwise it is derived from ``measured_gene_index`` without
        expanding a dense gene-by-cell matrix.  A supplied scalar or one-column
        summary is interpreted as the missing fraction.
        """
        if n_genes < 1:
            raise ValueError("n_genes must be positive")
        batch = self.batch_size
        summary = self.panel_missing_summary
        if summary is None:
            measured = torch.unique(self.measured_gene_index).numel()
            measured_fraction = min(float(measured) / float(n_genes), 1.0)
            missing_fraction = 1.0 - measured_fraction
            return self.count.new_tensor(
                [measured_fraction, missing_fraction], dtype=torch.float32
            ).expand(batch, 2)

        value = summary.to(device=self.count.device, dtype=torch.float32)
        if value.ndim == 0:
            missing = value.reshape(1).expand(batch)
            return torch.stack([1.0 - missing, missing], dim=-1)
        if value.ndim == 1:
            if value.numel() == 2:
                return value.reshape(1, 2).expand(batch, 2)
            if value.numel() == batch:
                missing = value
            elif value.numel() == 1:
                missing = value.expand(batch)
            else:  # guarded by validate, kept defensive for direct callers
                raise ValueError("invalid panel_missing_summary length")
            return torch.stack([1.0 - missing, missing], dim=-1)
        if value.shape[0] == 1:
            value = value.expand(batch, value.shape[1])
        if value.shape[1] == 2:
            return value
        missing = value[:, 0]
        return torch.stack([1.0 - missing, missing], dim=-1)

    @property
    def unmeasured_gene_mask(self) -> Tensor | None:
        """Backward/terminology alias for ``missing_token_mask``."""
        return self.missing_token_mask


@dataclass
class WorldModelBatch:
    expression: SparseExpressionBatch
    categorical_context: Tensor
    continuous_context: Tensor
    time_days: Tensor
    species_id: Tensor
    batch_index: Tensor
    optional_modalities: Tensor | None = None
    spatial_coords: Tensor | None = None
    edge_index: Tensor | None = None
    ligand_receptor: Tensor | None = None
    control: Tensor | None = None
    delta_control: Tensor | None = None
    decode_gene_index: Tensor | None = None
    # Optional project taxonomy labels for the scBrain-inspired Stage-0
    # readout.  They are kept out of the molecular input contract by default.
    annotation_labels: tuple[Tensor | None, ...] | Tensor | None = None
    # Named dense per-cell modalities (for example ATAC, protein, or
    # morphology features).  These fields intentionally remain at the end of
    # the dataclass so historical positional constructors keep their original
    # spatial/graph argument meanings.
    modality_inputs: Mapping[str, Tensor] | None = None
    modality_presence: Mapping[str, Tensor] | None = None
    # Per-cell availability of spatial observations.  ``None`` retains the
    # historical meaning that every row has coordinates when
    # ``spatial_coords`` is present.  The field is deliberately appended so
    # legacy positional constructors are not shifted.
    spatial_presence: Tensor | None = None

    def validate(self, n_genes: int, coordinate_dim: int,
                 control_dim: int, n_species: int,
                 optional_modality_dim: int = 0,
                 modality_input_dims: Mapping[str, int] | None = None) -> None:
        self.expression.validate(n_genes)
        batch = self.expression.batch_size
        for name in ("categorical_context", "continuous_context"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != batch:
                raise ValueError(f"{name} must have shape [B, D]")
        for name in ("time_days", "species_id", "batch_index"):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")
        if torch.any(self.species_id < 0) or torch.any(self.species_id >= n_species):
            raise ValueError("species_id is outside configured species range")
        if self.spatial_coords is not None and self.spatial_coords.shape != (batch, coordinate_dim):
            raise ValueError(f"spatial_coords must have shape [B, {coordinate_dim}]")
        if self.spatial_presence is not None:
            if not isinstance(self.spatial_presence, Tensor):
                raise TypeError("spatial_presence must be a tensor")
            if self.spatial_presence.shape != (batch,):
                raise ValueError("spatial_presence must have shape [B]")
            if self.spatial_presence.dtype != torch.bool:
                raise ValueError("spatial_presence must be boolean")
            if self.spatial_coords is None and torch.any(self.spatial_presence):
                raise ValueError(
                    "spatial_presence cannot mark cells present when spatial_coords is None"
                )
            if (
                self.spatial_coords is not None
                and torch.any(self.spatial_presence)
                and not torch.isfinite(self.spatial_coords[self.spatial_presence]).all()
            ):
                raise ValueError("spatial_coords must be finite for present cells")
        if optional_modality_dim and (
            self.optional_modalities is None
            or self.optional_modalities.shape != (batch, optional_modality_dim)
        ):
            raise ValueError(
                f"optional_modalities must have shape [B, {optional_modality_dim}]"
            )
        if self.modality_inputs is not None:
            if not isinstance(self.modality_inputs, Mapping):
                raise TypeError("modality_inputs must be a mapping of name -> tensor")
            if modality_input_dims is None:
                raise ValueError(
                    "modality_inputs were provided but named modality adapters are disabled"
                )
            if modality_input_dims is not None:
                unknown = sorted(
                    set(self.modality_inputs) - set(modality_input_dims)
                )
                if unknown:
                    raise ValueError(f"unregistered modality inputs: {unknown}")
            for name, value in self.modality_inputs.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("modality input names must be non-empty strings")
                if not isinstance(value, Tensor):
                    raise TypeError(f"modality input {name!r} must be a tensor")
                expected_dim = (
                    int(modality_input_dims[name])
                    if modality_input_dims is not None and name in modality_input_dims
                    else None
                )
                if value.ndim != 2 or value.shape[0] != batch:
                    raise ValueError(
                        f"modality input {name!r} must have shape [B, D]"
                    )
                if expected_dim is not None and value.shape[1] != expected_dim:
                    raise ValueError(
                        f"modality input {name!r} must have shape "
                        f"[{batch}, {expected_dim}]"
                    )
            if self.modality_presence is not None:
                if not isinstance(self.modality_presence, Mapping):
                    raise TypeError(
                        "modality_presence must be a mapping of name -> tensor"
                    )
                unknown_presence = sorted(
                    set(self.modality_presence)
                    - set(modality_input_dims or self.modality_inputs)
                )
                if unknown_presence:
                    raise ValueError(
                        f"unregistered modality presence masks: {unknown_presence}"
                    )
                for name, value in self.modality_presence.items():
                    if not isinstance(value, Tensor) or value.ndim != 1 or value.shape[0] != batch:
                        raise ValueError(
                            f"modality presence for {name!r} must have shape [{batch}]"
                        )
                    if value.dtype != torch.bool:
                        if not value.is_floating_point() or not torch.isfinite(value).all():
                            raise ValueError(
                                f"modality presence for {name!r} must be finite bool/float"
                            )
                        if torch.any(value < 0) or torch.any(value > 1):
                            raise ValueError(
                                f"modality presence for {name!r} must lie in [0, 1]"
                            )
                    if name not in self.modality_inputs and torch.any(value != 0):
                        raise ValueError(
                            f"presence for missing modality {name!r} must be all zero"
                        )
        elif self.modality_presence is not None:
            if modality_input_dims is None:
                raise ValueError(
                    "modality_presence were provided but named modality adapters are disabled"
                )
            if not isinstance(self.modality_presence, Mapping):
                raise TypeError("modality_presence must be a mapping of name -> tensor")
            unknown_presence = sorted(
                set(self.modality_presence) - set(modality_input_dims)
            )
            if unknown_presence:
                raise ValueError(
                    f"unregistered modality presence masks: {unknown_presence}"
                )
            for name, value in self.modality_presence.items():
                if not isinstance(value, Tensor) or value.ndim != 1 or value.shape[0] != batch:
                    raise ValueError(
                        f"modality presence for {name!r} must have shape [{batch}]"
                    )
                if value.dtype != torch.bool:
                    if not value.is_floating_point() or not torch.isfinite(value).all():
                        raise ValueError(
                            f"modality presence for {name!r} must be finite bool/float"
                        )
                    if torch.any(value < 0) or torch.any(value > 1):
                        raise ValueError(
                            f"modality presence for {name!r} must lie in [0, 1]"
                        )
                if torch.any(value != 0):
                    raise ValueError(
                        f"presence for missing modality {name!r} must be all zero"
                    )
        if self.edge_index is not None and (self.edge_index.ndim != 2
                                            or self.edge_index.shape[0] != 2):
            raise ValueError("edge_index must have shape [2, E]")
        if control_dim:
            if self.control is None or self.control.shape != (batch, control_dim):
                raise ValueError(f"control must have shape [B, {control_dim}]")
            if self.delta_control is not None and self.delta_control.shape != (batch, control_dim):
                raise ValueError(f"delta_control must have shape [B, {control_dim}]")
        if self.annotation_labels is not None:
            labels = self.annotation_labels
            if isinstance(labels, Tensor):
                labels = (labels,)
            elif isinstance(labels, (str, bytes)):
                raise ValueError("annotation_labels must be tensors or a tensor sequence")
            try:
                labels = tuple(labels)
            except TypeError as exc:
                raise ValueError(
                    "annotation_labels must be tensors or a tensor sequence"
                ) from exc
            for level, value in enumerate(labels):
                if value is None:
                    continue
                if not isinstance(value, Tensor) or value.ndim != 1:
                    raise ValueError(
                        f"annotation_labels[{level}] must have shape [B] or be None"
                    )
                if value.numel() not in (0, batch):
                    raise ValueError(
                        f"annotation_labels[{level}] must have shape [{batch}]"
                    )


@dataclass
class StructuralContext:
    message: Tensor
    niche_context: Tensor
    tissue_context: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    niche_probability: Tensor
    niche_state: Tensor
    tissue_state: Tensor


@dataclass
class DecoderOutput:
    gene_index: Tensor
    logits: Tensor
    mean: Tensor
    dispersion: Tensor
    sample: Tensor | None = None


@dataclass
class BrainWorldModelOutput:
    z_initial: Tensor
    z_final: Tensor
    latent_mean: Tensor
    latent_logvar: Tensor
    context: Tensor
    latent_partitions: dict[str, Tensor]
    decoder: DecoderOutput
    structure: StructuralContext
    predicted_positions: Tensor | None
    lineage_probabilities: Tensor
    uncertainty: Tensor | None = None
    treatment_effect: Tensor | None = None
    trajectory: list[Tensor] = field(default_factory=list)
    contrastive_z: Tensor | None = None
    assay_logits: Tensor | None = None
    flow_prediction: Tensor | None = None
    flow_target: Tensor | None = None
    token_features: Tensor | None = None
    token_row_index: Tensor | None = None
    masked_gene_logits: Tensor | None = None
    masked_gene_candidates: Tensor | None = None
    # Optional Stage-0 hierarchical annotation readout.  These fields remain
    # ``None`` for the historical reconstruction/rollout path.
    depth_states: tuple[Tensor, ...] | None = None
    annotation_logits: list[Tensor] | None = None
    annotation_features: list[Tensor] | None = None
    annotation_loss: Tensor | None = None
    # Optional scFiT telemetry for the ordinary rollout path.  These fields
    # are ``None`` for reconstruction-only calls and preserve the historical
    # output contract for callers that ignore the integration branch.
    adapter_endpoint: Tensor | None = None
    adapter_velocity: Tensor | None = None
    spatial_pre_adapter: Tensor | None = None
    spatial_adapter_delta: Tensor | None = None
    # Explicit readout of the ontology-backed anatomical token used by the
    # Stage-0 context encoder.  This is a view of the existing ``uberon_id``
    # embedding, not a second anatomy parameter table.
    anatomy_embedding: Tensor | None = None
    anatomy_presence: Tensor | None = None


@dataclass
class HierarchicalAnnotationOutput:
    """Outputs from the optional scBrain-inspired Stage-0 readout."""

    depth_states: tuple[Tensor, ...]
    logits: list[Tensor]
    features: list[Tensor] | None = None
    loss: Tensor | None = None


@dataclass
class TemporalTransitionOutput:
    source_latent: Tensor
    target_latent: Tensor
    target_context: Tensor
    predicted_latent: Tensor
    trajectory: list[Tensor]
    uncertainty: Tensor
    # Optional scFiT branch telemetry.  Legacy callers can ignore these
    # fields; they remain ``None`` when the adapter is disabled.
    adapter_endpoint: Tensor | None = None
    adapter_velocity: Tensor | None = None
    spatial_pre_adapter: Tensor | None = None
    spatial_adapter_delta: Tensor | None = None
