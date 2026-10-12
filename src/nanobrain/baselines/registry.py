"""Registry for independently replaceable baseline adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .base import BaselineAdapter


AdapterFactory = Callable[[], BaselineAdapter]
_REGISTRY: dict[str, AdapterFactory] = {}


def register(method_id: str) -> Callable[[type[BaselineAdapter]], type[BaselineAdapter]]:
    def decorator(adapter_type: type[BaselineAdapter]) -> type[BaselineAdapter]:
        if method_id in _REGISTRY:
            raise KeyError(f"duplicate baseline adapter: {method_id}")
        if getattr(adapter_type, "method_id", None) != method_id:
            raise ValueError(f"adapter method_id must be {method_id!r}")
        _REGISTRY[method_id] = adapter_type
        return adapter_type

    return decorator


def build(method_id: str) -> BaselineAdapter:
    try:
        factory = _REGISTRY[method_id]
    except KeyError as exc:
        raise KeyError(f"unknown baseline adapter {method_id!r}; available={sorted(_REGISTRY)}") from exc
    return factory()


def names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_many(method_ids: Iterable[str]) -> list[BaselineAdapter]:
    return [build(method_id) for method_id in method_ids]


__all__ = ["build", "build_many", "names", "register"]
