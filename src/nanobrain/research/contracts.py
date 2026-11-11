"""Strict wire contracts for proposals, tools, and immutable code artifacts."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

TASKS = ("annotation", "integration", "spatial_niche", "development")
VERSION = "nanobrain.research.candidate.v1"


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def strict(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"object requires {sorted(required)}; optional {sorted(optional)}; unknown keys forbidden")


def finite(value):
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN/Infinity are forbidden")


def candidate_schema():
    number = {"type": "number"}
    scalar = {"type": ["number", "integer", "boolean", "string"]}
    bound = {"type": "object", "additionalProperties": False, "required": ["type", "default"],
             "properties": {"type": {"enum": ["number", "integer", "boolean", "string"]},
                "minimum": number, "maximum": number, "enum": {"type": "array", "items": scalar, "maxItems": 32}, "default": scalar}}
    plugin = {"type": "object", "additionalProperties": False,
              "required": ["name", "kind", "file", "entrypoint", "parameters", "settings"],
              "properties": {"name": {"type": "string", "pattern": "^custom_[a-z][a-z0-9_]{1,47}$"},
                "kind": {"enum": ["encoder", "attention", "operation", "head", "flow", "regression_loss", "classification_loss", "niche_loss", "optimizer"]},
                "file": {"type": "string", "pattern": r"^(modules|losses|optimizers)/[a-z][a-z0-9_]{0,63}\.py$"},
                "entrypoint": {"enum": ["build", "compute"]},
                "parameters": {"type": "object", "maxProperties": 12, "additionalProperties": bound},
                "settings": {"type": "object", "maxProperties": 12, "additionalProperties": scalar}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
        "required": ["schema_version", "parent_ids", "target_tasks", "hypothesis", "genome", "optimizer", "plugins", "files"],
        "properties": {"schema_version": {"const": VERSION},
            "parent_ids": {"type": "array", "minItems": 1, "maxItems": 2, "uniqueItems": True, "items": {"type": "string", "pattern": "^[a-f0-9]{64}$"}},
            "target_tasks": {"type": "array", "minItems": 1, "maxItems": 4, "uniqueItems": True, "items": {"enum": list(TASKS)}},
            "hypothesis": {"type": "string", "minLength": 8, "maxLength": 1600},
            "genome": {"type": "object", "additionalProperties": False, "required": ["architecture", "learning_rate"],
                "properties": {"architecture": {"type": "object"}, "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": .01}}},
            "optimizer": {"type": "object", "additionalProperties": False,
                "required": ["name", "params", "weight_decay", "grad_clip", "schedule"],
                "properties": {"name": {"type": "string", "maxLength": 64}, "params": {"type": "object"},
                    "weight_decay": {"type": "number", "minimum": 0, "maximum": .2},
                    "grad_clip": {"type": "number", "minimum": .1, "maximum": 10},
                    "schedule": {"type": "object", "additionalProperties": False,
                        "required": ["name", "warmup_steps", "min_lr_ratio"], "properties": {
                            "name": {"enum": ["constant", "cosine", "linear"]},
                            "warmup_steps": {"type": "integer", "minimum": 0, "maximum": 2000},
                            "min_lr_ratio": {"type": "number", "minimum": 0, "maximum": 1}}}}},
            "plugins": {"type": "array", "items": plugin},
            "files": {"type": "object",
                "propertyNames": {"pattern": r"^(modules|losses|optimizers)/[a-z][a-z0-9_]{0,63}\.py$"},
                "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 32768}}}}


def parameter_values(definition, values):
    from jsonschema import Draft202012Validator
    if set(values) - set(definition): raise ValueError("unknown plugin parameter")
    result = {k: v["default"] for k, v in definition.items()}; result.update(values)
    for key, spec in definition.items():
        if spec["type"] in {"number", "integer"}:
            if "minimum" not in spec or "maximum" not in spec or not -1e6 <= spec["minimum"] <= spec["maximum"] <= 1e6:
                raise ValueError("custom numeric parameters need finite bounded ranges within +/-1e6")
        schema = {k: v for k, v in spec.items() if k != "default"}
        if spec['type']=='string' and ('enum' not in spec or any(not isinstance(v,str) or len(v)>64 for v in spec['enum'])):
            raise ValueError('custom string knobs need a bounded string enum')
        Draft202012Validator(schema).validate(spec['default'])
        Draft202012Validator(schema).validate(result[key])
    finite(result)
    return result


def validate_bundle(bundle, *, parent_ids=None):
    from jsonschema import Draft202012Validator
    finite(bundle); Draft202012Validator(candidate_schema()).validate(bundle)
    if parent_ids is not None and not set(bundle["parent_ids"]) <= set(parent_ids):
        raise ValueError("parent IDs must reference the current accepted lineage")
    names = [p["name"] for p in bundle["plugins"]]
    if len(names) != len(set(names)): raise ValueError("duplicate custom plugin name")
    if set(bundle["files"]) != {p["file"] for p in bundle["plugins"]}:
        raise ValueError("all submitted source files must be used by exactly declared plugins")
    if sum(len(s.encode()) for s in bundle["files"].values()) > 131072:
        raise ValueError("candidate source exceeds 128 KiB")
    for plugin in bundle["plugins"]:
        parameter_values(plugin["parameters"], plugin["settings"])
        is_loss = plugin["kind"].endswith("loss")
        if plugin["entrypoint"] != ("compute" if is_loss else "build"):
            raise ValueError("loss entrypoint is compute; module/optimizer entrypoint is build")
        folder = "losses/" if is_loss else "optimizers/" if plugin["kind"] == "optimizer" else "modules/"
        if not plugin["file"].startswith(folder): raise ValueError("source folder does not match plugin kind")
        if set(plugin["parameters"]) & {"input_dim", "hidden_dim", "output_dim", "parameters", "lr", "weight_decay"}:
            raise ValueError("dimension/optimizer arguments are supplied by the trusted runtime")
        if not is_loss and plugin["settings"]:raise ValueError("module settings belong in graph node params; optimizer settings belong in optimizer.params")
    return bundle


def canonical_graph(graph):
    graph = copy.deepcopy(graph); ids = {node["id"]: f"n{i}" for i, node in enumerate(graph["nodes"])}
    for node in graph["nodes"]:
        node.setdefault("repeat",1);node.setdefault("params",{})
        node["id"] = ids[node["id"]]; node["inputs"] = [ids.get(x, x) for x in node["inputs"]]
    graph["output"] = ids.get(graph["output"], graph["output"])
    return graph


def semantic_id(bundle):
    """Normalize simple cosmetic/default changes, not arbitrary Python program equivalence."""
    architecture = copy.deepcopy(bundle["genome"]["architecture"])
    for key in ("backbone", "flow"): architecture[key] = canonical_graph(architecture[key])
    for head in architecture.get("heads", {}).values():
        if "pre" in head: head["pre"] = canonical_graph(head["pre"])
    architecture['heads'].pop('completion',None)
    architecture['objectives'].pop('spatial_completion',None)
    class NoDocstrings(ast.NodeTransformer):
        def strip(self,node):
            self.generic_visit(node)
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
            return node
        visit_Module=strip
        visit_FunctionDef=strip
        visit_ClassDef=strip
    code={};plugins=[]
    for plugin in sorted(bundle['plugins'],key=lambda p:p['name']):
        code[plugin['name']]=ast.dump(NoDocstrings().visit(ast.parse(bundle['files'][plugin['file']])),include_attributes=False)
        spec={k:v for k,v in plugin.items() if k!='file'}
        spec['settings']=parameter_values(plugin['parameters'],plugin['settings']) if plugin['kind'].endswith('loss') else {}
        plugins.append(spec)
    optimizer=copy.deepcopy(bundle['optimizer'])
    defaults={'adamw':dict(betas=[.9,.999],eps=1e-8,amsgrad=False),'adam':dict(betas=[.9,.999],eps=1e-8,amsgrad=False),
        'sgd':dict(momentum=0.,dampening=0.,nesterov=False),'rmsprop':dict(alpha=.99,eps=1e-8,momentum=0.,centered=False),
        'adagrad':dict(lr_decay=0.,eps=1e-10,initial_accumulator_value=0.)}
    optimizer['params']={**defaults.get(optimizer['name'],{}),**optimizer['params']}
    if optimizer['schedule']['name']=='constant':optimizer['schedule']['min_lr_ratio']=0.
    return digest(dict(architecture=architecture, learning_rate=bundle["genome"]["learning_rate"],
                       optimizer=optimizer, plugins=plugins, code=code))


def artifact_path(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError("artifact path escapes its root")
    return path
