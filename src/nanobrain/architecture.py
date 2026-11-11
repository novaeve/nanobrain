"""Validated JSON computation graphs; PyTorch owns modules and autograd.

Graphs are topologically ordered lists, not executable Python. Shapes and
effective factory parameters are checked before allocating model weights.
"""
from __future__ import annotations

import copy
import inspect
import math
import re
import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from . import modules as _modules
from . import graph_ops as _graph_ops
from .registry import registry, backend_catalog


@dataclass(frozen=True)
class Shape:
    width: int
    tokens: int | None = None

    @property
    def rank(self):
        return 2 if self.tokens is None else 3


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"expected fields {sorted(required)}; optional {sorted(optional)}")


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def factory_parameters(kind, name):
    factory = registry(kind).resolve(name).factory
    return {k: p for k, p in inspect.signature(factory).parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def checked_parameters(kind, name, params, automatic):
    """Never silently ignore a candidate's unsupported parameter."""
    if not isinstance(params, dict):
        raise ValueError("module params must be an object")
    accepted = factory_parameters(kind, name)
    unknown = set(params) - set(accepted)
    if unknown:
        raise ValueError(f"unsupported params for {kind}/{name}: {sorted(unknown)}")
    if set(params) & {"input_dim", "output_dim"}:
        raise ValueError("input_dim/output_dim are inferred from tensor contracts")
    result = {k: v for k, v in automatic.items() if k in accepted}
    result.update(params)
    for k, p in accepted.items():
        if k not in result and p.default is not inspect.Parameter.empty:
            result[k] = p.default
        if k not in result:
            raise ValueError(f"missing parameter {kind}/{name}.{k}")
    for k, v in result.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"nonfinite parameter {k}")
        if k in {"hidden_dim", "input_dim", "output_dim", "depth", "heads", "iterations"}:
            limit = {"depth": 8, "heads": 16, "iterations": 100}.get(k, 65536)
            _integer(v, 1, limit, k)
        if k in {"dropout", "expansion", "epsilon", "eps", "correction_init"}:
            bounds = {"dropout": (0, .5), "expansion": (1, 8), "epsilon": (1e-4, 1),
                      "eps": (1e-8, .1), "correction_init": (0, 1)}[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not bounds[0] <= v <= bounds[1]:
                raise ValueError(f"invalid {k}: {v}")
            result[k] = float(v)
        if k in {"bias", "inplace"} and type(v) is not bool:
            raise ValueError(f"{k} must be boolean")
        if k == "inplace" and v:
            raise ValueError("in-place operations would mutate inputs shared by graph branches")
        if k == "approximate" and v not in {"none", "tanh"}:
            raise ValueError("GELU approximate must be none/tanh")
    if "heads" in result and result["hidden_dim"] % result["heads"]:
        raise ValueError("width must be divisible by heads")
    return result


def plan_graph(spec, inputs, *, allow_spatial=False):
    _keys(spec, {"nodes", "output"})
    if not isinstance(spec["nodes"], list) or not 1 <= len(spec["nodes"]) <= 32:
        raise ValueError("graph requires 1..32 nodes")
    shapes = dict(inputs); dependencies = {k: {k} for k in inputs}
    spatial_depth = {k: 0 for k in inputs}; plan = []; total = 0
    for node in spec["nodes"]:
        _keys(node, {"id", "kind", "name", "inputs"}, {"params", "repeat"})
        key, kind, name = node["id"], node["kind"], node["name"]
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) or key in shapes:
            raise ValueError("node IDs must be unique lowercase identifiers")
        refs = node["inputs"]
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in shapes for r in refs):
            raise ValueError("node inputs must reference earlier nodes or declared graph inputs (no cycles)")
        values = [shapes[r] for r in refs]; first = values[0]
        repeat = node.get("repeat", 1); _integer(repeat, 1, 8, "repeat"); total += repeat
        if total > 64:
            raise ValueError("graph exceeds 64 expanded nodes")
        if kind not in {"encoder", "attention", "spatial", "residual", "operation", "flow"}:
            raise ValueError(f"unsupported graph kind {kind}")
        if kind == "spatial" and not allow_spatial:
            raise ValueError("spatial nodes require a backbone graph with neighbor metadata")
        if kind == "flow" and (name == "none" or "time" not in inputs):
            raise ValueError("active flow nodes belong in the velocity graph")
        contract = registry(kind).resolve(name).contract
        params = checked_parameters(kind, name, node.get("params", {}),
                                    {"input_dim": first.width, "hidden_dim": first.width})
        if kind == "flow" and name == "fmt":
            params["depth"] = max(2, params["depth"] + params["depth"] % 2)
        width = params.get("hidden_dim", first.width)
        _integer(width, 1, 512, "node width")
        output = Shape(width, first.tokens)
        multi = kind == "flow" or (kind == "operation" and name in {"add", "subtract", "concat", "stack", "film"})
        if not multi and len(refs) != 1:
            raise ValueError(f"{kind}/{name} takes one input")
        if kind == "encoder":
            if first.rank != 2:
                raise ValueError("encoder expects cell features; use operation/linear or FFN on tokens")
            output = Shape(width, (first.width + min(16, first.width) - 1) // min(16, first.width)
                           if name == "panel_patches" else None)
        elif kind == "spatial" and first.rank != 2:
            raise ValueError("pool tokens before spatial aggregation")
        elif kind == "flow":
            if len(values) != 3 or first.rank != 2 or values[1] != first or values[2] != Shape(3):
                raise ValueError("flow inputs must be state[P,H], context[P,H], time[P,3]")
        elif kind == "operation":
            if name in {"mean_pool", "first_pool"}:
                if first.rank != 3:
                    raise ValueError("pool requires tokens; cell axis cannot be pooled")
                output = Shape(first.width)
            elif name == "scfit_time":
                if first != Shape(3):
                    raise ValueError("scfit_time requires tau/source_years/delta_years")
                output = Shape(width)
            elif name in {"add", "subtract", "stack"}:
                if len(values) < 2 or any(s != first for s in values):
                    raise ValueError("merge requires at least two equal shapes")
                if name == "subtract" and len(values) != 2:
                    raise ValueError("subtract takes exactly two inputs")
                if name == "stack":
                    if first.rank != 2:
                        raise ValueError("stack creates token axis from cell states")
                    output = Shape(first.width, len(values))
            elif name == "concat":
                if len(values) < 2 or any(s.tokens != first.tokens for s in values):
                    raise ValueError("concat requires matching cell/token axes")
                output = Shape(sum(s.width for s in values), first.tokens)
            elif name == "film":
                if len(values) != 2 or values[1] != Shape(first.width):
                    raise ValueError("FiLM requires state and rank-2 condition with equal width")
        changes_width = kind == "encoder" or (kind == "operation" and name in {"linear", "scfit_time", "concat"})
        if not changes_width and width != first.width:
            raise ValueError("width changes require an explicit projection")
        if contract.input_rank is not None and first.rank != contract.input_rank:
            raise ValueError(f"{name} input rank violates module contract")
        if contract.output_rank is not None and output.rank != contract.output_rank:
            raise ValueError(f"{name} output rank violates module contract")
        if repeat > 1 and (multi or output != first):
            raise ValueError("repeat requires a unary shape-preserving node")
        depth = max(spatial_depth[r] for r in refs) + (repeat if kind == "spatial" and name != "identity" else 0)
        if depth > 1:
            raise ValueError("current data provides one-hop context; stacked spatial hops require a new data contract")
        dependencies[key] = {key}.union(*(dependencies[r] for r in refs))
        spatial_depth[key] = depth; shapes[key] = output
        plan.append(dict(node, params=params, repeat=repeat, shape=output))
    if spec["output"] not in shapes:
        raise ValueError("unknown graph output")
    if any(n["id"] not in dependencies[spec["output"]] for n in spec["nodes"]):
        raise ValueError("dead/disconnected nodes are forbidden")
    return plan, shapes[spec["output"]], dependencies[spec["output"]] & set(inputs)


def seeded_module(seed, key, factory):
    """Unrelated task/flow edits cannot change a head's random initialization."""
    if seed is None:
        return factory()
    local_seed = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little") % (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(local_seed)
        return factory()


class ModuleGraph(nn.Module):
    def __init__(self, spec, inputs, *, allow_spatial=False, seed=None, scope="graph"):
        super().__init__()
        self.plan, self.output_shape, self.used_inputs = plan_graph(spec, inputs, allow_spatial=allow_spatial)
        self.spec = copy.deepcopy(spec); self.input_shapes = dict(inputs)
        self.layers = nn.ModuleDict({n["id"]: nn.ModuleList(
            seeded_module(seed, f"{scope}/{n['id']}/{i}",
                          lambda n=n: registry(n["kind"]).create(n["name"], **n["params"]))
            for i in range(n["repeat"])) for n in self.plan})
        self.uses_spatial = any(n["kind"] == "spatial" and n["name"] != "identity" for n in self.plan)

    def forward(self, values, *, neighbors=None, neighbor_mask=None, coordinates=None):
        values = dict(values)
        count = len(next(iter(values.values())))
        for key, shape in self.input_shapes.items():
            self._check(values[key], shape, count, key)
        for node in self.plan:
            args = [values[k] for k in node["inputs"]]
            for layer in self.layers[node["id"]]:
                output = (layer(*args, neighbors=neighbors, neighbor_mask=neighbor_mask, coordinates=coordinates)
                          if node["kind"] == "spatial" else layer(*args))
                self._check(output, node["shape"], count, node["id"])
                args = [output]
            values[node["id"]] = output
        return values[self.spec["output"]]

    @staticmethod
    def _check(value, shape, count, name):
        expected = (count, shape.width) if shape.rank == 2 else (count, shape.tokens, shape.width)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must produce {expected}, got {tuple(value.shape)}")


class FlowGraph(ModuleGraph):
    def __init__(self, spec, width, seed=None):
        super().__init__(spec, {"state": Shape(width), "context": Shape(width), "time": Shape(3)}, seed=seed, scope="flow")

    def forward(self, state, context, time):
        return super().forward({"state": state, "context": context, "time": time})


def head_parameters(head, width, output_dim):
    _keys(head, {"name", "params"}, {"pre"})
    if "pre" in head:
        _, shape, _ = plan_graph(head["pre"], {"state": Shape(width)})
        if shape.rank != 2:
            raise ValueError("head pre-graph must end in cell states")
        width = shape.width
    params = checked_parameters("head", head["name"], head["params"], {"hidden_dim": width, "output_dim": output_dim})
    if params.get("hidden_dim") != width:
        raise ValueError("head width must match its input graph")
    return params


class GraphHead(nn.Module):
    def __init__(self, spec, width, output_dim, seed, name):
        super().__init__()
        self.pre = ModuleGraph(spec["pre"], {"state": Shape(width)}, seed=seed, scope="head/" + name)
        params = head_parameters(spec, width, output_dim)
        self.output = seeded_module(seed, "head/" + name + "/output",
                                    lambda: registry("head").create(spec["name"], **params))

    def forward(self, state):
        return self.output(self.pre({"state": state}))


def validate_architecture(spec):
    from .objectives import validate_objectives
    _keys(spec, {"version", "input_dim", "width", "backbone", "flow", "pairing", "heads", "objectives", "solver"})
    if spec["version"] != 2:
        raise ValueError("architecture version must be 2")
    _integer(spec["input_dim"], 1, 65536, "input_dim"); _integer(spec["width"], 16, 512, "width")
    width = spec["width"]
    _, output, _ = plan_graph(spec["backbone"], {"features": Shape(spec["input_dim"])}, allow_spatial=True)
    if output != Shape(width):
        raise ValueError("backbone output must be cell states with declared width")
    _, output, used = plan_graph(spec["flow"], {"state": Shape(width), "context": Shape(width), "time": Shape(3)})
    if output != Shape(width) or used != {"state", "context", "time"}:
        raise ValueError("flow output must have declared width and depend on state, context AND time")
    _keys(spec["pairing"], {"name", "params"})
    checked_parameters("pairing", spec["pairing"]["name"], spec["pairing"]["params"], {})
    _keys(spec["heads"], {"annotation", "integration", "integration_reconstruction", "niche", "development", "completion"})
    for head in spec["heads"].values():
        head_parameters(head, width, 1)
    validate_objectives(spec["objectives"])
    _keys(spec["solver"], {"name", "steps"})
    if spec["solver"]["name"] not in {"euler", "heun"}:
        raise ValueError("unsupported solver")
    _integer(spec["solver"]["steps"], 1, 64, "solver steps")
    return spec


def architecture_signature(spec):
    """Topology/module signature used for the structural-mutation quota."""
    graphs = {k: spec[k] for k in ("backbone", "flow")}
    graphs.update({"head/" + k: h["pre"] for k, h in spec["heads"].items() if "pre" in h})
    return {key: [(n["kind"], n["name"], n["inputs"], n.get("repeat", 1)) for n in graph["nodes"]]
            + [(graph["output"],)] for key, graph in graphs.items()}


def canonical_graph(spec, inputs, *, allow_spatial=False):
    plan, _, _ = plan_graph(spec, inputs, allow_spatial=allow_spatial)
    mapping = {k: k for k in inputs}; nodes = []
    for i, n in enumerate(plan):
        mapping[n["id"]] = f"node_{i}"
        params = {k: v for k, v in n["params"].items() if k not in {"input_dim", "output_dim"}}
        nodes.append(dict(id=mapping[n["id"]], kind=n["kind"], name=registry(n["kind"]).resolve(n["name"]).contract.name,
                          inputs=[mapping[r] for r in n["inputs"]], params=params, repeat=n["repeat"]))
    return dict(nodes=nodes, output=mapping[spec["output"]])


def canonical_architecture(spec):
    """Normalize defaults and IDs so cosmetic renaming is not a mutation."""
    validate_architecture(spec)
    result = copy.deepcopy(spec)
    for key, inputs in (("backbone", {"features": Shape(spec["input_dim"])}),
                        ("flow", {"state": Shape(spec["width"]), "context": Shape(spec["width"]), "time": Shape(3)})):
        result[key] = canonical_graph(spec[key], inputs, allow_spatial=key == "backbone")
    pair = result["pairing"]
    pair["params"] = checked_parameters("pairing", pair["name"], pair["params"], {})
    pair["name"] = registry("pairing").resolve(pair["name"]).contract.name
    for head in result["heads"].values():
        head["params"] = {k: v for k, v in head_parameters(head, result["width"], 1).items()
                          if k not in {"hidden_dim", "output_dim"}}
        if "pre" in head:
            head["pre"] = canonical_graph(head["pre"], {"state": Shape(result["width"])})
        head["name"] = registry("head").resolve(head["name"]).contract.name
    for objective in result["objectives"].values():
        objective["params"] = {k: float(v) if isinstance(v, (float, int)) else v
                               for k, v in objective["params"].items()}
    return result


def graph_catalog():
    return {kind: [{**entry, "parameters": {k: {"default": None if p.default is inspect.Parameter.empty else p.default}
                   for k, p in factory_parameters(kind, entry["name"]).items() if k not in {"input_dim", "output_dim"}}}
                  for entry in entries] for kind, entries in backend_catalog().items()}
