"""Memory-bounded sparse expression tokenization."""
from __future__ import annotations
from dataclasses import dataclass,fields
import pandas as pd
import torch
from torch import Tensor


class _CSRExpand(torch.autograd.Function):
    """Gather cell rows and reduce their gradients over contiguous CSR runs."""

    @staticmethod
    def forward(ctx, cells: Tensor, row_ids: Tensor, row_ptr: Tensor) -> Tensor:
        ctx.save_for_backward(row_ptr)
        return cells.index_select(0, row_ids)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (row_ptr,) = ctx.saved_tensors
        cell_gradient = torch.segment_reduce(
            gradient, "sum", offsets=row_ptr,
        )
        return cell_gradient, None, None


def expand_csr_rows(cells: Tensor, row_ids: Tensor, row_ptr: Tensor) -> Tensor:
    """Broadcast cell features to tokens with a segmented-sum backward.

    PyTorch advanced indexing implements the backward as an atomic index-put.
    CSR rows are contiguous, so ``segment_reduce`` computes the same gradient
    with substantially less contention for long expression rows.
    """
    if cells.ndim < 1 or row_ids.ndim != 1 or row_ptr.ndim != 1:
        raise ValueError("CSR expansion expects cells [...], row_ids [tokens], row_ptr [cells+1]")
    if not cells.requires_grad:
        return cells.index_select(0, row_ids)
    return _CSRExpand.apply(cells, row_ids, row_ptr)


class _CSRAddExpanded(torch.autograd.Function):
    """Add cell features to their CSR tokens without retaining a gather.

    The usual ``expand_csr_rows(cells) + tokens`` sequence keeps the gathered
    token tensor alive until backward.  BrainWM only needs the sum, so this
    small autograd boundary retains the CSR offsets and reduces the token
    gradient directly into cells.  It uses PyTorch's indexed gather and
    ``segment_reduce`` primitives; no custom CUDA kernel or new parameter is
    introduced.
    """

    @staticmethod
    def forward(
        ctx, tokens: Tensor, cells: Tensor, row_ids: Tensor, row_ptr: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(row_ptr)
        return tokens + cells.index_select(0, row_ids)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (row_ptr,) = ctx.saved_tensors
        cell_gradient = torch.segment_reduce(
            gradient, "sum", offsets=row_ptr,
        )
        return gradient, cell_gradient, None, None


def add_csr_rows(
    tokens: Tensor, cells: Tensor, row_ids: Tensor, row_ptr: Tensor,
) -> Tensor:
    """Return ``tokens + cells[row_ids]`` with a compact backward graph."""
    if tokens.ndim < 1 or cells.ndim < 1:
        raise ValueError("CSR addition expects token and cell feature tensors")
    if row_ids.ndim != 1 or row_ptr.ndim != 1:
        raise ValueError("CSR addition expects one-dimensional row metadata")
    if row_ids.numel() != tokens.shape[0]:
        raise ValueError("row_ids must have one entry per token")
    if not (tokens.requires_grad or cells.requires_grad):
        return tokens + cells.index_select(0, row_ids)
    return _CSRAddExpanded.apply(tokens, cells, row_ids, row_ptr)


class _ConcatDetachedSuffix(torch.autograd.Function):
    """Concatenate a trainable prefix and an inference-only suffix.

    ``torch.cat((prefix, suffix.detach()))`` is mathematically correct, but
    its generic CatBackward still builds a scatter/index-put node for the
    unused suffix slice.  BrainWM's fused developmental stream repeats this
    boundary once per backbone block.  The suffix is deliberately detached,
    so its backward contribution is known to be empty; returning only the
    prefix view keeps the same derivative while avoiding that scatter kernel.
    """

    @staticmethod
    def forward(ctx, prefix: Tensor, suffix: Tensor) -> Tensor:
        ctx.prefix_length = int(prefix.shape[0])
        return torch.cat((prefix, suffix), dim=0)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return gradient.narrow(0, 0, ctx.prefix_length), None


def concat_detached_suffix(prefix: Tensor, suffix: Tensor) -> Tensor:
    """Join streams while keeping the suffix outside the autograd graph."""
    if suffix.requires_grad:
        raise ValueError("detached suffix must not require gradients")
    if not prefix.requires_grad:
        return torch.cat((prefix, suffix), dim=0)
    return _ConcatDetachedSuffix.apply(prefix, suffix)


@dataclass
class SparseTokenBatch:
    gene_index: Tensor
    values: Tensor
    row_ptr: Tensor
    measured_gene_index: Tensor
    token_mask: Tensor | None = None
    cells: pd.DataFrame | None = None
    modality: dict[str,Tensor] | None = None
    modality_observed: dict[str,Tensor] | None = None
    modality_masked: dict[str,Tensor] | None = None
    neighbor_indices: Tensor | None = None
    neighbor_mask: Tensor | None = None
    expression_masked: Tensor | None = None
    token_sampling_weight: Tensor | None = None
    expression_mask_probability: Tensor | None = None
    cell_type_target: Tensor | None = None
    anatomy_target: Tensor | None = None
    developmental_observed: Tensor | None = None
    source_state: Tensor | None = None
    target_state: Tensor | None = None
    source_time: Tensor | None = None
    target_time: Tensor | None = None
    flow_tau: Tensor | None = None
    context_target: dict[str, Tensor] | None = None
    spatial_coordinates: Tensor | None = None
    packed: bool = False
    packed_layout_validated: bool = False
    packed_min_seqlen: int | None = None
    packed_max_seqlen: int | None = None
    neighbor_indices_validated: bool = False
    # CPU-side fact recorded by the NumPy loader. False means that the batch
    # has no non-self spatial edges, so spatial attention can return identity
    # without launching Q/K/V projections on the accelerator.
    neighbor_edges_present: bool | None = None
    # Optional loader-created token-to-cell map. Keeping it with the batch
    # avoids rebuilding the same repeat_interleave map in every objective and
    # backbone invocation.
    row_ids: Tensor | None = None
    # Optional immutable CSR metadata produced by the NumPy materializer.  The
    # values are redundant with ``row_ptr``/``row_ids`` mathematically, but
    # keeping them beside the batch avoids a length subtraction, dtype cast,
    # and token-wise gather on every accelerator forward.
    row_lengths: Tensor | None = None
    token_lengths: Tensor | None = None
    cu_seqlens: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.row_ptr.numel() - 1)

    def _map_tensors(self,transform) -> 'SparseTokenBatch':
        def apply(value):
            if isinstance(value,Tensor):return transform(value)
            if isinstance(value,dict):return {key:apply(item) for key,item in value.items()}
            return value
        return type(self)(**{
            field.name:apply(getattr(self,field.name)) for field in fields(self)
        })

    def to(self,device: str | torch.device,*,non_blocking: bool = False) -> 'SparseTokenBatch':
        return self._map_tensors(
            lambda value:value.to(device,non_blocking=non_blocking)
        )

    def pin_memory(self) -> 'SparseTokenBatch':
        """Support PyTorch DataLoader's recursive pinned-memory protocol."""
        return self._map_tensors(lambda value:value.pin_memory())

    def token_row_ids(self) -> Tensor:
        """Return one cell index per token, reusing the loader sidecar."""
        if self.row_ids is not None:
            if self.row_ids.ndim != 1 or self.row_ids.numel() != self.values.numel():
                raise ValueError('row_ids must have one entry per token')
            return self.row_ids
        lengths = (self.row_ptr[1:] - self.row_ptr[:-1]).long()
        return torch.repeat_interleave(
            torch.arange(self.batch_size, device=self.row_ptr.device),
            lengths,
            output_size=self.values.numel(),
        )


def reader_batch_to_sparse(batch: dict, device='cpu') -> SparseTokenBatch:
    row_ptr=torch.as_tensor(batch['row_ptr'], dtype=torch.long, device=device)
    lengths=row_ptr[1:]-row_ptr[:-1]
    row_ids=torch.repeat_interleave(
        torch.arange(len(lengths),device=device),lengths,
        output_size=int(torch.as_tensor(batch['gene_index']).numel()),
    )
    return SparseTokenBatch(torch.as_tensor(batch['gene_index'], dtype=torch.long, device=device),
        torch.as_tensor(batch['count'], dtype=torch.float32, device=device),
        row_ptr,
        torch.as_tensor(batch['measured_gene_index'], dtype=torch.long, device=device),
        row_ids=row_ids,cells=batch.get('cells'),row_lengths=lengths,
        token_lengths=lengths[row_ids],cu_seqlens=row_ptr.to(torch.int32))


def collate_sparse_batches(batches: list[dict], device='cpu', max_tokens: int | None = None) -> SparseTokenBatch:
    """Pad variable-length reader rows into a bounded token batch.

    Padding uses gene id 0 and a zero value, with token_mask distinguishing it
    from measured expression. row_ptr addresses padded slots, not nonzero counts.
    Source-local batches must share the measured panel. No dense vocabulary
    tensor is allocated. Truncation is opt-in; full training leaves it disabled.
    """
    rows = []
    row_weights=[]
    weighted=any('token_sampling_weight' in src for src in batches)
    if weighted and max_tokens is not None:
        raise ValueError('cannot truncate tokens after measured-zero sampling')
    metadata = []
    panel = None
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError('max_tokens must be positive or None')
    for src in batches:
        measured = torch.as_tensor(src['measured_gene_index'],dtype=torch.long).cpu()
        if panel is None:
            panel = measured
        elif not torch.equal(panel,measured):
            raise ValueError('collate source-local rows with the same measured panel')
        ptr = torch.as_tensor(src['row_ptr'])
        cells = src.get('cells')
        if cells is not None and len(cells)!=len(ptr)-1:
            raise ValueError('cell metadata does not align with sparse rows')
        metadata.append(cells)
        genes = torch.as_tensor(src['gene_index']).long(); vals = torch.as_tensor(src['count']).float()
        weights=torch.as_tensor(src.get('token_sampling_weight',torch.ones_like(vals))).float()
        for i in range(len(ptr)-1):
            g, v = genes[ptr[i]:ptr[i+1]], vals[ptr[i]:ptr[i+1]]
            if max_tokens and len(v) > max_tokens:
                keep = torch.topk(v.abs(), max_tokens).indices; g, v = g[keep], v[keep]
            rows.append((g, v))
            row_weights.append(weights[ptr[i]:ptr[i+1]])
    if not rows:
        raise ValueError('cannot collate an empty batch')
    width = max(1, max(len(g) for g, _ in rows))
    gi = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    vv = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    for i, (g, v) in enumerate(rows): gi[i, :len(g)] = g.to(device); vv[i, :len(v)] = v.to(device)
    ptr = torch.arange(0, (len(rows)+1)*width, width, device=device)
    mask = torch.zeros_like(vv, dtype=torch.bool)
    for i, (g, _) in enumerate(rows): mask[i, :len(g)] = True
    if any(frame is not None for frame in metadata) and not all(frame is not None for frame in metadata):
        raise ValueError('cannot silently drop metadata from partially annotated batches')
    cells = pd.concat(metadata,ignore_index=True) if all(frame is not None for frame in metadata) else None
    sampling_weight=None
    if weighted:
        sampling_weight=torch.zeros_like(vv)
        for i,w in enumerate(row_weights):sampling_weight[i,:len(w)]=w.to(device)
        sampling_weight=sampling_weight.flatten()
    row_ids=torch.arange(len(rows),device=device).repeat_interleave(width)
    return SparseTokenBatch(gi.reshape(-1), vv.reshape(-1), ptr,
                            panel.to(device), mask.reshape(-1), cells,
                            token_sampling_weight=sampling_weight,row_ids=row_ids,
                            row_lengths=mask.sum(dim=1).to(dtype=torch.long),
                            token_lengths=mask.sum(dim=1).to(dtype=torch.long)
                            .repeat_interleave(width),
                            cu_seqlens=ptr.to(dtype=torch.int32))
