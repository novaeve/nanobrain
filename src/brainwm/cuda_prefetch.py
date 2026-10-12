"""Small CUDA-stream prefetcher for nested BrainWM batches.

The implementation follows the standard PyTorch/NVIDIA prefetch pattern: one
batch is copied from pinned host memory on a side stream while the current
batch executes on the default stream.  It deliberately contains no model
logic, so it can be reused by the main NumPy loader and the developmental pair
loader alike.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor


def move_to_device(value: Any, device: torch.device, *, non_blocking: bool = True):
    """Recursively move tensors in a batch while preserving its container type."""
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=non_blocking)
    # SparseTokenBatch and similar domain batches expose an optimized ``to``
    # method that also preserves non-tensor metadata and sidecars.
    mover = getattr(value, "to", None)
    if callable(mover) and not isinstance(value, (str, bytes)):
        try:
            return mover(device, non_blocking=non_blocking)
        except TypeError:
            # A few legacy containers accept only a positional device.
            return mover(device)
    if isinstance(value, Mapping):
        return type(value)(
            (key, move_to_device(item, device, non_blocking=non_blocking))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return type(value)(
            move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        )
    if isinstance(value, list):
        return [
            move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: move_to_device(
                    getattr(value, field.name), device,
                    non_blocking=non_blocking,
                )
                for field in fields(value)
            }
        )
    return value


def record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    """Keep prefetched storage alive until the consumer stream is finished.

    A side-stream copy and the default-stream wait establish execution order,
    but the caching allocator also needs an explicit consumer-stream record
    before the next prefetch can reuse the allocation.  Walking the same
    containers as ``move_to_device`` keeps this independent of BrainWM's
    batch dataclasses.
    """
    if isinstance(value, Tensor):
        if value.is_cuda:
            value.record_stream(stream)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            record_stream(item, stream)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            record_stream(item, stream)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            record_stream(getattr(value, field.name), stream)


class CUDAPrefetcher(Iterator[Any]):
    """One-stage asynchronous host-to-device prefetch iterator.

    On CPU this is an exact pass-through, which keeps unit tests and fallback
    execution deterministic.  The source iterator is intentionally consumed
    only one item ahead, bounding pinned-memory usage for very large sparse
    batches.
    """

    def __init__(
        self, source, device: str | torch.device, *, enabled: bool = True,
    ) -> None:
        self._source = iter(source)
        self.device = torch.device(device)
        self.enabled = bool(enabled and self.device.type == "cuda")
        self._stream = (
            torch.cuda.Stream(device=self.device) if self.enabled else None
        )
        self._next = None
        if self.enabled:
            self._preload()

    def _preload(self) -> None:
        try:
            value = next(self._source)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self._stream):
            self._next = move_to_device(
                value, self.device, non_blocking=True,
            )

    def __iter__(self) -> "CUDAPrefetcher":
        return self

    def __next__(self):
        if not self.enabled:
            return next(self._source)
        if self._next is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self._stream)
        value = self._next
        record_stream(value, current_stream)
        # Start the next host-to-device copy before returning to the model.
        self._preload()
        return value


__all__ = ["CUDAPrefetcher", "move_to_device", "record_stream"]
