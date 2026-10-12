"""Small, explicit registries for replaceable BrainWM execution components.

The model contract is intentionally independent from implementation details.
Attention, MoE blocks, flow transformers, optimizers, and compiler frontends
are selected by a stable string and constructed through this module.  Built-in
registrations are kept deliberately minimal; optional projects can register a
factory at import time without modifying the foundation model.

Factories receive keyword arguments from the corresponding public contract.
Their capability metadata is descriptive and is written to run manifests so a
benchmark can be reproduced even when an optional package is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Callable, Iterable, Mapping


Factory = Callable[..., Any]


def _json_safe(value: Any) -> Any:
    """Keep user-supplied capability metadata safe for atomic JSON manifests."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


@dataclass(frozen=True)
class BackendSpec:
    """A named implementation and the operations it can safely support."""

    kind: str
    name: str
    factory: Factory
    capabilities: Mapping[str, Any]
    aliases: tuple[str, ...] = ()
    source: str = "builtin"

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "aliases": list(self.aliases),
            "source": self.source,
            "capabilities": _json_safe(self.capabilities),
        }


class BackendRegistry:
    """Registry with collision checks and serializable capability metadata."""

    def __init__(self, kind: str):
        self.kind = str(kind)
        self._specs: dict[str, BackendSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: Factory | None = None,
        *,
        aliases: Iterable[str] = (),
        capabilities: Mapping[str, Any] | None = None,
        source: str = "builtin",
        replace: bool = False,
    ):
        """Register a factory, either directly or as a decorator.

        ``replace=True`` is useful for an application-level override during
        startup.  Accidental duplicate names remain errors by default.
        """
        canonical = str(name).strip()
        if not canonical or canonical in {"all", "help"}:
            raise ValueError(f"invalid {self.kind} backend name: {name!r}")
        alias_tuple = tuple(dict.fromkeys(str(value).strip() for value in aliases))
        if any(not value for value in alias_tuple):
            raise ValueError(f"invalid alias for {self.kind} backend {canonical!r}")

        def install(fn: Factory):
            if not callable(fn):
                raise TypeError(f"{self.kind} backend factory must be callable")
            occupied = {canonical, *alias_tuple}
            if not replace:
                collisions = [
                    value for value in occupied
                    if value in self._specs or value in self._aliases
                ]
                if collisions:
                    raise ValueError(
                        f"{self.kind} backend already registered: {sorted(collisions)}"
                    )
            # Remove an overridden canonical name and all aliases that pointed
            # to it before installing the new immutable spec.
            if replace and canonical in self._specs:
                old = self._specs.pop(canonical)
                for old_alias in old.aliases:
                    self._aliases.pop(old_alias, None)
            for value in occupied:
                if replace:
                    self._specs.pop(value, None)
                    self._aliases.pop(value, None)
            spec = BackendSpec(
                kind=self.kind,
                name=canonical,
                factory=fn,
                capabilities=dict(capabilities or {}),
                aliases=alias_tuple,
                source=str(source),
            )
            self._specs[canonical] = spec
            for alias in alias_tuple:
                if alias == canonical:
                    continue
                self._aliases[alias] = canonical
            return fn

        if factory is None:
            return install
        return install(factory)

    def resolve(self, name: str) -> BackendSpec:
        selected = str(name).strip()
        canonical = self._aliases.get(selected, selected)
        try:
            return self._specs[canonical]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise ValueError(
                f"unknown {self.kind} backend {name!r}; available: {available}"
            ) from exc

    def create(self, name: str, *args, **kwargs):
        factory = self.resolve(name).factory
        # Model factories share a superset keyword contract.  A lightweight
        # implementation may intentionally expose only the knobs it supports;
        # filter those optional fields while preserving strict positional and
        # required-argument errors. Factories declaring ``**kwargs`` receive
        # the complete contract unchanged.
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return factory(*args, **kwargs)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {
                key: value for key, value in kwargs.items()
                if key in signature.parameters
            }
        return factory(*args, **kwargs)

    def names(self, *, include_aliases: bool = False) -> tuple[str, ...]:
        names = sorted(self._specs)
        if include_aliases:
            names.extend(sorted(self._aliases))
        return tuple(names)

    def manifests(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._specs[name].manifest() for name in sorted(self._specs))


ATTENTION_BACKENDS = BackendRegistry("attention")
MOE_BACKENDS = BackendRegistry("moe")
FLOW_BACKENDS = BackendRegistry("flow")
OPTIMIZER_BACKENDS = BackendRegistry("optimizer")
COMPILE_BACKENDS = BackendRegistry("compile")


def _registry(kind: str) -> BackendRegistry:
    registries = {
        "attention": ATTENTION_BACKENDS,
        "moe": MOE_BACKENDS,
        "flow": FLOW_BACKENDS,
        "optimizer": OPTIMIZER_BACKENDS,
        "compile": COMPILE_BACKENDS,
    }
    try:
        return registries[str(kind)]
    except KeyError as exc:
        raise ValueError(f"unknown backend kind {kind!r}") from exc


def _ensure_model_builtins(kind: str) -> None:
    """Lazily import model modules when the registry is queried directly.

    Keeping this lazy avoids importing DeepSpeed for data/evaluator tools, but
    makes ``backend_names('attention')`` useful even when callers imported only
    ``brainwm.backends``. The defining module still owns the actual class
    registrations, so there is one source of truth for each parameter layout.
    """
    if kind == "attention" and not ATTENTION_BACKENDS.names():
        importlib.import_module(".foundation_moe", package=__package__)
    elif kind == "moe" and not MOE_BACKENDS.names():
        importlib.import_module(".foundation_moe", package=__package__)
    elif kind == "flow" and not FLOW_BACKENDS.names():
        importlib.import_module(".flow_transformer", package=__package__)


def register_attention_backend(name: str, factory: Factory | None = None, **kwargs):
    return ATTENTION_BACKENDS.register(name, factory, **kwargs)


def register_moe_backend(name: str, factory: Factory | None = None, **kwargs):
    return MOE_BACKENDS.register(name, factory, **kwargs)


def register_flow_backend(name: str, factory: Factory | None = None, **kwargs):
    return FLOW_BACKENDS.register(name, factory, **kwargs)


def register_optimizer_backend(name: str, factory: Factory | None = None, **kwargs):
    return OPTIMIZER_BACKENDS.register(name, factory, **kwargs)


def register_compile_backend(name: str, factory: Factory | None = None, **kwargs):
    return COMPILE_BACKENDS.register(name, factory, **kwargs)


def backend_spec(kind: str, name: str) -> BackendSpec:
    _ensure_model_builtins(kind)
    return _registry(kind).resolve(name)


def create_backend(kind: str, name: str, *args, **kwargs):
    _ensure_model_builtins(kind)
    return _registry(kind).create(name, *args, **kwargs)


def backend_names(kind: str, *, include_aliases: bool = False) -> tuple[str, ...]:
    _ensure_model_builtins(kind)
    return _registry(kind).names(include_aliases=include_aliases)


def backend_catalog() -> dict[str, list[dict[str, Any]]]:
    """Return JSON-safe metadata for run manifests and benchmark tables."""
    for kind in ("attention", "moe", "flow"):
        _ensure_model_builtins(kind)
    return {
        kind: registry.manifests()
        for kind, registry in (
            ("attention", ATTENTION_BACKENDS),
            ("moe", MOE_BACKENDS),
            ("flow", FLOW_BACKENDS),
            ("optimizer", OPTIMIZER_BACKENDS),
            ("compile", COMPILE_BACKENDS),
        )
    }


def compile_callable(backend: str, function: Factory, **kwargs):
    """Compile ``function`` through a registered compiler backend."""
    return create_backend("compile", backend, function, **kwargs)


def load_backend_plugins(modules: Iterable[str] = ()) -> tuple[str, ...]:
    """Import application-supplied backend modules and return loaded names.

    A plugin only needs to import this module and call one of the registration
    helpers.  Keeping imports explicit makes benchmark manifests auditable and
    avoids silently discovering incompatible binary extensions. Empty names
    are ignored so an environment variable can be assembled conveniently.
    """
    loaded = []
    for module_name in modules:
        name = str(module_name).strip()
        if not name:
            continue
        importlib.import_module(name)
        loaded.append(name)
    return tuple(loaded)


def load_backend_entry_points(group: str = "brainwm.backends") -> tuple[str, ...]:
    """Load installed package entry points for zero-copy infra integration.

    Entry-point loading is opt-in from the launcher, matching explicit module
    plugins. A package may expose a module or a no-argument registration
    function; both forms are accepted and the entry-point names are returned
    for telemetry.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python 3.10 compatibility
        return ()
    available = entry_points()
    selected = (
        available.select(group=group)
        if hasattr(available, "select") else available.get(group, ())
    )
    loaded = []
    for entry in selected:
        loaded_object = entry.load()
        if callable(loaded_object) and not isinstance(loaded_object, type):
            # Registration functions conventionally return None; invoking
            # them is harmless for module objects and enables tiny plugins.
            loaded_object()
        loaded.append(str(entry.name))
    return tuple(loaded)


def _adamw_factory(parameters, **kwargs):
    # Keep torch lazy so importing the registry remains cheap for data-only
    # tooling and so an optional backend can be inspected without CUDA setup.
    import torch

    return torch.optim.AdamW(parameters, **kwargs)


def _eager_compile(function, **kwargs):
    del kwargs
    return function


def _torch_compile(function, **kwargs):
    import torch

    compiler = getattr(torch, "compile", None)
    if compiler is None:
        raise RuntimeError("torch_compile backend requires PyTorch 2.x")
    return compiler(function, **kwargs)


# Optimizer and compiler primitives are pure infrastructure and can be
# registered at import time. Model-specific registrations happen at the end
# of their defining modules to avoid circular imports.
register_optimizer_backend(
    "adamw", _adamw_factory, aliases=("torch_adamw",),
    capabilities={"bf16": True, "fused": True, "state_dict_compatible": True},
)
register_compile_backend(
    "eager", _eager_compile, aliases=("none",),
    capabilities={"dynamic": True, "state_dict_compatible": True},
)
register_compile_backend(
    "torch_compile", _torch_compile, aliases=("inductor",),
    capabilities={"dynamic": True, "cuda_graphs": True, "state_dict_compatible": True},
)


__all__ = [
    "ATTENTION_BACKENDS", "MOE_BACKENDS", "FLOW_BACKENDS",
    "OPTIMIZER_BACKENDS", "COMPILE_BACKENDS", "BackendRegistry", "BackendSpec",
    "backend_catalog", "backend_names", "backend_spec", "compile_callable",
    "create_backend", "load_backend_entry_points", "load_backend_plugins",
    "register_attention_backend",
    "register_compile_backend",
    "register_flow_backend", "register_moe_backend", "register_optimizer_backend",
]
