"""Named optional-modality adapters for the BrainWM encoder.

The expression encoder remains the canonical path for scRNA, snRNA, and
spatial expression.  This module provides a small, explicit bridge for other
per-cell modalities (for example ATAC, protein, or morphology features): each
registered modality has its own input projection and all projections emit the
same context-sized vector.  Missing modalities are represented explicitly so
that inference can use any available subset without changing tensor shapes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .blocks import MLP


def mask_named_modalities(
    inputs: Mapping[str, Tensor] | None,
    presence: Mapping[str, Tensor] | None,
    cell_mask: Tensor,
) -> tuple[dict[str, Tensor] | None, dict[str, Tensor] | None]:
    """Hide named per-cell modalities for an explicit query-cell mask.

    A masked cell must not retain a dense side channel that could identify its
    expression state.  Values are replaced by zeros and an explicit presence
    mask is emitted for every supplied modality, so zero-valued features stay
    distinguishable from an unavailable row.  The input mappings are never
    mutated.
    """
    if not isinstance(cell_mask, Tensor) or cell_mask.ndim != 1:
        raise ValueError("cell_mask must be a boolean vector")
    if cell_mask.dtype != torch.bool:
        raise ValueError("cell_mask must be a boolean vector")
    if inputs is None:
        if presence is not None:
            if not isinstance(presence, Mapping):
                raise TypeError("modality_presence must be a mapping")
            # An all-zero presence map is a useful explicit record for a fully
            # missing modality batch; preserve it without inventing tensors.
            for name, value in presence.items():
                if not isinstance(value, Tensor) or value.ndim != 1:
                    raise ValueError(
                        f"modality presence for {name!r} must be a vector"
                    )
                if value.shape != cell_mask.shape:
                    raise ValueError(
                        f"modality presence for {name!r} must match cell_mask"
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
            return None, {
                str(name): value.clone() for name, value in presence.items()
            }
        return None, None
    if not isinstance(inputs, Mapping):
        raise TypeError("modality_inputs must be a mapping")
    if presence is not None and not isinstance(presence, Mapping):
        raise TypeError("modality_presence must be a mapping")
    presence = {} if presence is None else presence
    masked_inputs: dict[str, Tensor] = {}
    masked_presence: dict[str, Tensor] = {}
    # Preserve explicit all-zero records for registered modalities that are
    # omitted from this batch.  Positive availability without a tensor is
    # rejected; the model-level validator applies the same rule.
    for name in set(presence) - set(inputs):
        available = presence[name]
        if not isinstance(available, Tensor) or available.ndim != 1:
            raise ValueError(f"modality presence for {name!r} must be a vector")
        if available.shape != cell_mask.shape:
            raise ValueError(
                f"modality presence for {name!r} must match cell_mask"
            )
        if available.dtype != torch.bool:
            if not available.is_floating_point() or not torch.isfinite(available).all():
                raise ValueError(
                    f"modality presence for {name!r} must be finite bool/float"
                )
            if torch.any(available < 0) or torch.any(available > 1):
                raise ValueError(
                    f"modality presence for {name!r} must lie in [0, 1]"
                )
        if torch.any(available != 0):
            raise ValueError(
                f"presence for missing modality {name!r} must be all zero"
            )
        masked_presence[name] = available.clone()
    for name, value in inputs.items():
        if not isinstance(value, Tensor) or value.ndim != 2:
            raise ValueError(f"modality input {name!r} must be a matrix")
        if value.shape[0] != cell_mask.shape[0]:
            raise ValueError(
                f"modality input {name!r} row count does not match cell_mask"
            )
        row_mask = cell_mask.to(device=value.device)
        masked_inputs[name] = value.masked_fill(row_mask[:, None], 0)
        if name in presence:
            available = presence[name]
            if not isinstance(available, Tensor) or available.ndim != 1:
                raise ValueError(
                    f"modality presence for {name!r} must be a vector"
                )
            if available.shape != cell_mask.shape:
                raise ValueError(
                    f"modality presence for {name!r} must match cell_mask"
                )
            if available.dtype != torch.bool:
                if not available.is_floating_point() or not torch.isfinite(available).all():
                    raise ValueError(
                        f"modality presence for {name!r} must be finite bool/float"
                    )
                if torch.any(available < 0) or torch.any(available > 1):
                    raise ValueError(
                        f"modality presence for {name!r} must lie in [0, 1]"
                    )
            masked_available = available.clone()
        else:
            # Match the router's implicit missing-row rule without retaining
            # non-finite values as an accidental positive signal.
            masked_available = torch.isfinite(value).all(dim=1)
        masked_available = masked_available.masked_fill(
            cell_mask.to(device=masked_available.device), 0
        )
        masked_presence[name] = masked_available
    return masked_inputs, masked_presence


class NamedModalityRouter(nn.Module):
    """Project a named set of dense modality views into one shared vector.

    ``input_dims`` is part of the model contract.  A caller may provide any
    subset of the registered names.  Values containing non-finite rows are
    treated as unavailable for those rows; callers can provide an explicit
    boolean/float ``presence`` mask when missingness is known independently
    of the feature values.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        normalized_dims = {
            str(name).strip(): int(dimension)
            for name, dimension in input_dims.items()
        }
        names = tuple(sorted(normalized_dims))
        if not names or any(not name for name in names):
            raise ValueError("at least one non-empty modality name is required")
        if len(set(names)) != len(names):
            raise ValueError("modality names must be unique")
        dimensions = tuple(normalized_dims[name] for name in names)
        if any(dimension < 1 for dimension in dimensions):
            raise ValueError("modality input dimensions must be positive")
        if output_dim < 1 or hidden_dim < 1:
            raise ValueError("modality adapter dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("modality adapter dropout must be in [0, 1)")

        self.modality_names = names
        self.input_dims = dict(zip(names, dimensions))
        self.adapters = nn.ModuleList(
            MLP(dimension, hidden_dim, output_dim, layers=2, dropout=dropout)
            for dimension in dimensions
        )
        # Separate learned tokens make an absent modality distinguishable from
        # a present all-zero feature vector.  They are initialized at zero so
        # enabling the branch is a conservative warm start.
        self.present_tokens = nn.Parameter(torch.zeros(len(names), output_dim))
        self.absent_tokens = nn.Parameter(torch.zeros(len(names), output_dim))
        self.output_norm = nn.LayerNorm(output_dim)

    @staticmethod
    def _validate_presence(
        value: Tensor, batch_size: int, name: str
    ) -> Tensor:
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(
                f"modality presence for {name!r} must have shape [{batch_size}]"
            )
        if value.dtype == torch.bool:
            return value.to(dtype=torch.float32)
        if not value.is_floating_point() and not value.is_complex():
            value = value.to(dtype=torch.float32)
        if value.is_complex() or not torch.isfinite(value).all():
            raise ValueError(f"modality presence for {name!r} must be finite")
        if torch.any(value < 0) or torch.any(value > 1):
            raise ValueError(
                f"modality presence for {name!r} must lie in [0, 1]"
            )
        return value.float()

    def forward(
        self,
        inputs: Mapping[str, Tensor] | None,
        *,
        batch_size: int,
        presence: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, Mapping):
            raise TypeError("modality_inputs must be a mapping of name -> tensor")
        unknown = sorted(set(inputs) - set(self.modality_names))
        if unknown:
            raise ValueError(f"unregistered modality inputs: {unknown}")
        if presence is None:
            presence = {}
        if not isinstance(presence, Mapping):
            raise TypeError("modality_presence must be a mapping of name -> tensor")
        unknown_presence = sorted(set(presence) - set(self.modality_names))
        if unknown_presence:
            raise ValueError(f"unregistered modality presence masks: {unknown_presence}")

        outputs: list[Tensor] = []
        for index, name in enumerate(self.modality_names):
            value = inputs.get(name)
            if value is None:
                if name in presence:
                    # An all-zero mask is an explicit, shape-preserving way to
                    # record that the complete modality is unavailable.  A
                    # positive mask without a tensor is ambiguous and fails
                    # closed rather than silently fabricating features.
                    missing_presence = self._validate_presence(
                        presence[name], batch_size, name
                    )
                    if torch.any(missing_presence != 0):
                        raise ValueError(
                            f"presence for missing modality {name!r} must be all zero"
                        )
                mask = value = None
                # The parameter is expanded below after a device/dtype anchor
                # is available from another modality or the module itself.
            else:
                if not isinstance(value, Tensor):
                    raise TypeError(f"modality input {name!r} must be a tensor")
                if value.ndim != 2 or value.shape != (batch_size, self.input_dims[name]):
                    raise ValueError(
                        f"modality input {name!r} must have shape "
                        f"[{batch_size}, {self.input_dims[name]}]"
                    )
                finite_rows = torch.isfinite(value).all(dim=1)
                clean = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
                transformed = self.adapters[index](clean)
                mask = (
                    self._validate_presence(presence[name], batch_size, name)
                    if name in presence
                    else finite_rows.to(dtype=transformed.dtype)
                ).to(device=transformed.device, dtype=transformed.dtype)
                present = self.present_tokens[index].to(
                    device=transformed.device, dtype=transformed.dtype
                )
                absent = self.absent_tokens[index].to(
                    device=transformed.device, dtype=transformed.dtype
                )
                outputs.append(
                    transformed * mask[:, None]
                    + (present * mask[:, None])
                    + absent * (1.0 - mask[:, None])
                )

        if presence and not inputs:
            # Presence maps with all-zero rows are valid explicit records for a
            # fully missing modality set; positive masks were rejected above.
            for name, value in presence.items():
                validated = self._validate_presence(value, batch_size, name)
                if torch.any(validated != 0):
                    raise ValueError(
                        f"presence for missing modality {name!r} must be all zero"
                    )
        if not outputs:
            # No input tensor is still a valid expression-only inference mode.
            # Use the module parameters as an anchor for device and dtype.
            anchor = self.absent_tokens
            outputs = [anchor[index].expand(batch_size, -1) for index in range(len(self.modality_names))]
        else:
            # Add an explicit absent token for registered modalities omitted
            # from this batch.  This keeps the output invariant to dictionary
            # ordering and records which modalities were available.
            anchor = outputs[0]
            for index, name in enumerate(self.modality_names):
                if name in inputs:
                    continue
                absent = self.absent_tokens[index].to(
                    device=anchor.device, dtype=anchor.dtype
                )
                outputs.insert(index, absent.expand(batch_size, -1))

        stacked = torch.stack(outputs, dim=0).mean(dim=0)
        return self.output_norm(stacked)
