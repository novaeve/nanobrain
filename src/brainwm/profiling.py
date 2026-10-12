"""Opt-in named ranges for BrainWM CPU/CUDA profiling."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

from torch.profiler import record_function


_RANGES_ENABLED: ContextVar[bool] = ContextVar(
    "brainwm_profile_ranges_enabled", default=False
)
_NOOP_RANGE = nullcontext()


def profile_range(name: str):
    """Return a profiler range when enabled and a shared no-op otherwise."""
    if not _RANGES_ENABLED.get():
        return _NOOP_RANGE
    return record_function(name)


@contextmanager
def enable_profile_ranges():
    """Enable BrainWM ranges in the current execution context only."""
    token = _RANGES_ENABLED.set(True)
    try:
        yield
    finally:
        _RANGES_ENABLED.reset(token)
