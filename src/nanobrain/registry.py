"""Explicit registry for replaceable nanoBrain components."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Iterable

from .contracts import ModuleContract


Factory = Callable[..., Any]


@dataclass(frozen=True)
class RegisteredModule:
    contract: ModuleContract
    factory: Factory
    aliases: tuple[str, ...] = ()


class ModuleRegistry:
    def __init__(self, kind: str):
        self.kind = str(kind)
        self._items: dict[str, RegisteredModule] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: Factory | None = None,
        *,
        aliases: Iterable[str] = (),
        complexity: str = "unknown",
        input_rank: int | None = None,
        output_rank: int | None = None,
        supports_missing: bool = True,
        source: str = "nanobrain",
        notes: str = "",
        replace: bool = False,
    ):
        canonical = str(name).strip()
        alias_tuple = tuple(dict.fromkeys(str(alias).strip() for alias in aliases))

        def install(fn: Factory):
            occupied = {canonical, *alias_tuple}
            collisions = [
                item for item in occupied
                if item in self._items or item in self._aliases
            ]
            if collisions and not replace:
                raise ValueError(f"{self.kind} module already registered: {collisions}")
            if replace and canonical in self._items:
                old = self._items.pop(canonical)
                for alias in old.aliases:
                    self._aliases.pop(alias, None)
            for item in occupied:
                if replace:
                    self._items.pop(item, None)
                    self._aliases.pop(item, None)
            contract = ModuleContract(
                kind=self.kind,
                name=canonical,
                input_rank=input_rank,
                output_rank=output_rank,
                complexity=complexity,
                supports_missing=supports_missing,
                source=source,
                notes=notes,
            )
            self._items[canonical] = RegisteredModule(contract, fn, alias_tuple)
            for alias in alias_tuple:
                self._aliases[alias] = canonical
            return fn

        return install if factory is None else install(factory)

    def resolve(self, name: str) -> RegisteredModule:
        key = self._aliases.get(str(name), str(name))
        if key not in self._items:
            raise KeyError(f"unknown {self.kind} module {name!r}; choices={self.names()}")
        return self._items[key]

    def create(self, name: str, **kwargs: Any) -> Any:
        factory = self.resolve(name).factory
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return factory(**kwargs)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                **item.contract.as_dict(),
                "aliases": list(item.aliases),
            }
            for name, item in sorted(self._items.items())
        ]


REGISTRIES = {
    kind: ModuleRegistry(kind)
    for kind in (
        "encoder", "attention", "spatial", "residual", "pairing", "flow", "head", "operation"
    )
}


def registry(kind: str) -> ModuleRegistry:
    try:
        return REGISTRIES[str(kind)]
    except KeyError as exc:
        raise KeyError(f"unknown module kind {kind!r}") from exc


def backend_catalog() -> dict[str, list[dict[str, Any]]]:
    return {kind: value.catalog() for kind, value in REGISTRIES.items()}
