"""Named registries for replaceable components (mask specs, objectives, baselines, metrics, ...).

A component registers under a kind and a name; its parameters are the fields of a dataclass
(the component itself, or a separate params class). `build(name, **params)` rejects unknown
parameters and names the valid ones, so TOML configs fail early and say what to write.

    MASKS = Registry("mask")

    @MASKS.register("random_genes", "Scattered genes: imputation and denoising")
    @dataclass
    class RandomGenes: ...

    MASKS.build("random_genes", ratio=0.2)
"""

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    name: str
    target: type
    summary: str


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Entry] = {}

    def register(self, name: str, summary: str = ""):
        def install(target):
            if name in self._entries:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            if not dataclasses.is_dataclass(target):
                raise TypeError(f"{self.kind} {name!r}: register a dataclass (its fields are the parameters)")
            self._entries[name] = Entry(name, target, summary or (target.__doc__ or "").strip().split("\n")[0])
            return target

        return install

    def names(self) -> list[str]:
        return sorted(self._entries)

    def entry(self, name: str) -> Entry:
        if name not in self._entries:
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {', '.join(self.names())}")
        return self._entries[name]

    def params(self, name: str) -> dict:
        """Parameter name -> default (or None when required)."""
        out = {}
        for f in dataclasses.fields(self.entry(name).target):
            if not f.init:  # internal state, not a parameter
                continue
            if f.default is not dataclasses.MISSING:
                out[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                out[f.name] = f.default_factory()
            else:
                out[f.name] = None
        return out

    def build(self, name: str, **params):
        target = self.entry(name).target
        valid = {f.name for f in dataclasses.fields(target) if f.init}
        unknown = sorted(set(params) - valid)
        if unknown:
            raise TypeError(f"{self.kind} {name!r} has no parameter {unknown}; valid: {sorted(valid)}")
        return target(**params)

    def describe(self) -> list[dict]:
        return [{"name": n, "summary": self._entries[n].summary, "params": self.params(n)} for n in self.names()]
