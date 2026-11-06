"""Experiments as TOML: `extends` + `--set` overrides + `[sweep]` -> concrete configs.

    extends = ["../base.toml", "../model/cajal.toml"]   # merged in order, then this file on top
    [run]   name = "cajal_1m", out = "runs/cajal_1m"     # where the run writes
    [model] / [data] / [train]                           # fields of ModelConfig / DataConfig / TrainConfig
    [sweep] "data.masks" = ["cajal_v0", "table_v1"]      # one run per combination

Defaults live only in the dataclasses; a TOML sets what differs. Unknown sections or keys
fail with the valid choices.
"""

import copy
import dataclasses
import itertools
import json
import tomllib
from pathlib import Path

from .config import DataConfig, ModelConfig, TrainConfig

SECTIONS = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig}
FREE = {"run", "description", "status", "outcome"}  # free-form fields carried along
# Where an experiment stands; `outcome` says what it found, in one line, next to the config that asked.
STATUS = ("main", "candidate", "done", "parked", "dropped", "stopped")


def _merge(base: dict, top: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in top.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(path, _chain=()) -> dict:
    """One TOML with its `extends` chain merged (parents first)."""
    path = Path(path).resolve()
    if path in _chain:
        raise ValueError("extends cycle: " + " -> ".join(p.name for p in (*_chain, path)))
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    merged = {}
    for parent in doc.pop("extends", []):
        merged = _merge(merged, load(path.parent / parent, (*_chain, path)))
    return _merge(merged, doc)


def _parse(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value  # bare strings need no quotes on the command line


def override(doc: dict, items) -> dict:
    """Apply "section.key=value" overrides (value parsed as JSON, else kept as a string)."""
    doc = copy.deepcopy(doc)
    for item in items:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        doc.setdefault(section, {})[name] = _parse(value)
    return doc


def expand(doc: dict) -> list[dict]:
    """The sweep's Cartesian product; each run records its assignment under run.sweep."""
    doc = copy.deepcopy(doc)
    sweep = doc.pop("sweep", {})
    if not sweep:
        return [doc]
    keys = sorted(sweep)
    runs = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        assignment = dict(zip(keys, values))
        run = override(doc, [f"{k}={json.dumps(v)}" for k, v in assignment.items()])
        run.setdefault("run", {})["sweep"] = assignment
        runs.append(run)
    return runs


def pin_sweep(doc: dict, overrides) -> dict:
    """One sweep member: the document without its sweep, once `overrides` assign every sweep key."""
    sweep = doc.get("sweep", {})
    unpinned = sorted(set(sweep) - {item.split("=", 1)[0] for item in overrides})
    if unpinned:
        raise ValueError(f"the sweep over {unpinned} is not pinned by overrides; launch it with `cajal run`")
    return {k: v for k, v in doc.items() if k != "sweep"}


def build(doc: dict):
    """(ModelConfig, DataConfig, TrainConfig) from a resolved document, rejecting unknown keys."""
    unknown = set(doc) - set(SECTIONS) - FREE
    if unknown:
        raise ValueError(f"unknown sections {sorted(unknown)}; valid: {sorted(set(SECTIONS) | FREE)}")
    if doc.get("status", "main") not in STATUS:
        raise ValueError(f"status {doc['status']!r}; valid: {list(STATUS)}")
    out = []
    for section, cls in SECTIONS.items():
        values = doc.get(section, {})
        fields = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(values) - fields)
        if bad:
            raise ValueError(f"[{section}] has no field {bad}; valid: {sorted(fields)}")
        out.append(cls(**values))
    return tuple(out)


def resolve(path, overrides=()) -> list[dict]:
    """All runs an experiment TOML describes, overrides applied before the sweep expands."""
    runs = expand(override(load(path), overrides))
    for run in runs:
        build(run)  # validate every run before anything starts
    return runs
