"""Immutable experiment boundary; candidate code can only see declared tensors."""

import ast
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from . import campaign

ROOT = Path(__file__).resolve().parents[2]
DEFINITION = campaign.load()
_search = DEFINITION["search"]


def _ranges(mapping):
    return {k: tuple(v) for k, v in mapping.items()}


TASKS = tuple(DEFINITION["objectives"]["names"])
METRICS = tuple(m["name"] for m in DEFINITION["metrics"])
METRIC_GROUPS = {
    g: tuple(m["name"] for m in DEFINITION["metrics"] if m["group"] == g)
    for g in dict.fromkeys(m["group"] for m in DEFINITION["metrics"])
}
METRIC_RANGES = {m["name"]: tuple(m["range"]) for m in DEFINITION["metrics"]}
MODEL_SPACE = {k: list(v) for k, v in _search["model"].items()}
MODEL_RANGES = _ranges(_search["model_ranges"])
TRAIN_RANGES = _ranges(_search["train_ranges"])
TRAIN_INTS = _ranges(_search["train_integers"])
DATA_RANGES = _ranges(_search["data_ranges"])
OPTIMIZER_RANGES = _ranges(_search["optimizer_ranges"])
BALANCER_RANGES = {name: _ranges(mapping) for name, mapping in _search["balancer_ranges"].items()}
KINDS = tuple(_search["plugin_kinds"])
BALANCERS = tuple(_search["balancers"])
OPTIMIZERS = tuple(_search["optimizers"])
REGULARIZERS = tuple(_search["regularizers"])
GPU_WHITELIST = tuple(DEFINITION["resources"]["gpu_whitelist"])
GPU_REASON = f"GPU whitelist is {list(GPU_WHITELIST)}; {DEFINITION['resources']['reason']}"
GATE_FLOOR = DEFINITION["gate"]["regression_floor"]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def sha(path):
    digestor = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digestor.update(chunk)
    return digestor.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_config(path=None):
    cfg = json.loads(Path(path or ROOT / "configs/cajal_react.json").read_text())
    if cfg.get("weight_policy") != DEFINITION["campaign"]["weight_policy"] or cfg.get("trusted_teachers"):
        raise ValueError("candidates must train independently without historical or pretrained teacher weights")
    if cfg["reference_recipe"] != DEFINITION["campaign"]["reference_recipe"]:
        raise ValueError("reference must be the frozen existing Cajal v1 recipe")
    if cfg["reference_evaluation"] != DEFINITION["campaign"]["reference_evaluation"]:
        raise ValueError("reuse the existing Cajal reference result; never silently refit it")
    if cfg["schedule_steps"] != DEFINITION["budget"]["schedule_steps"] or cfg["stop_at"] != DEFINITION["budget"]["stop_at"]:
        raise ValueError("this protocol uses the Cajal 30k schedule stopped at 10k")
    if cfg["seeds"] != DEFINITION["budget"]["seeds"]:
        raise ValueError("this revision uses the existing Cajal seed 0 only")
    if cfg["checkpoint_selection"] != DEFINITION["budget"]["checkpoint_selection"] or bool(cfg["scaling_enabled"]) != DEFINITION["budget"]["scaling_enabled"]:
        raise ValueError("fixed-stop Cajal evaluation; no automatic scaling in this protocol")
    if not cfg["gpu_ids"] or any(type(g) is not int or g not in GPU_WHITELIST for g in cfg["gpu_ids"]):
        raise ValueError(GPU_REASON)
    if len(set(cfg["gpu_ids"])) != len(cfg["gpu_ids"]):
        raise ValueError("GPU IDs must be unique")
    if type(cfg["max_workers"]) is not int or not 1 <= cfg["max_workers"] <= 16:
        raise ValueError("stage concurrency must be between 1 and 16")
    # The global stage budget also includes CPU probes. GPU leases independently
    # enforce per-device slots and available memory.
    if type(cfg["per_gpu_workers"]) is not int or cfg["per_gpu_workers"] not in (1, 2, 3):
        raise ValueError("per-GPU concurrency must be between 1 and 3")
    for key, limit in (("cpu_probe_workers", 8), ("task_evaluation_workers", 4)):
        value = cfg.get(key, 1)
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"{key} must be between 1 and {limit}")
    return cfg


def reference_document(cfg=None):
    from brainwm_unified import experiment
    from brainwm_unified.config import to_dict

    cfg = cfg or load_config()
    doc = experiment.load(ROOT / cfg["reference_recipe"])
    m, d, t = experiment.build(doc)
    result = to_dict(m, d, t)
    dimensions = ROOT / "reference/model_dimensions.json"
    if dimensions.is_file():
        result["ModelConfig"].update(json.loads(dimensions.read_text()))
    # Research-only transport fields are empty and do not change the Cajal model.
    return result


def reference_bundle():
    return {
        "schema_version": "cajal.research.candidate.v2",
        "parent_ids": ["0" * 64],
        "target_metrics": list(METRICS),
        "hypothesis": "Existing Cajal v1 architecture and grad_equal joint recipe, unchanged.",
        "model": {},
        "train": {},
        "data": {},
        "components": [],
        "plugins": [],
        "files": {},
    }


def candidate_schema():
    parameter = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "default"],
        "properties": {
            "type": {"enum": ["integer", "number", "boolean", "string"]},
            "default": {},
            "minimum": {"type": "number"},
            "maximum": {"type": "number"},
            "enum": {"type": "array"},
        },
    }
    plugin = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "kind", "file", "parameters", "settings"],
        "properties": {
            "name": {"type": "string", "pattern": "^custom_[a-z][a-z0-9_]{1,47}$"},
            "kind": {"enum": list(KINDS)},
            "file": {"type": "string", "pattern": "^(modules|losses|optimizers)/[a-z][a-z0-9_]{0,63}\\.py$"},
            "parameters": {"type": "object", "additionalProperties": parameter},
            "settings": {"type": "object"},
        },
    }
    model_properties = {key: {"enum": values} for key, values in MODEL_SPACE.items()}
    model_properties.update(
        {key: {"type": "number", "minimum": low, "maximum": high} for key, (low, high) in MODEL_RANGES.items()}
    )
    train_properties = {
        key: {"type": "number", "minimum": low, "maximum": high} for key, (low, high) in TRAIN_RANGES.items()
    }
    train_properties.update(
        {key: {"type": "integer", "minimum": low, "maximum": high} for key, (low, high) in TRAIN_INTS.items()}
    )
    train_properties.update(
        {
            "balancer": {"enum": list(BALANCERS)},
            "optimizer": {"enum": list(OPTIMIZERS)},
            "lr_schedule": {"enum": ["cosine", "linear", "constant"]},
            "optimizer_params": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    k: {"type": "integer" if k == "muon_ns_steps" else "number", "minimum": v[0], "maximum": v[1]}
                    for k, v in OPTIMIZER_RANGES.items()
                },
            },
            "balancer_params": {"type": "object"},
            "task_weights": {
                "type": "object",
                "additionalProperties": False,
                "required": list(TASKS),
                "properties": {k: {"type": "number", "minimum": 0.01, "maximum": 20.0} for k in TASKS},
            },
            "objectives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "weight"],
                    "properties": {
                        "name": {"enum": list(REGULARIZERS)},
                        "weight": {"type": "number", "minimum": 0.0, "maximum": 20.0},
                        "coeff": {"type": "number"},
                        "label": {"enum": ["cell_type", "anatomy"]},
                        "n_cells": {"type": "integer", "minimum": 4, "maximum": 128},
                        "n_proj": {"type": "integer", "minimum": 16, "maximum": 512},
                        "t_max": {"type": "number", "minimum": 1.0, "maximum": 10.0},
                        "n_t": {"type": "integer", "minimum": 5, "maximum": 65},
                        "start": {"type": "integer", "minimum": 0, "maximum": 5000},
                        "dual_lr": {"type": "number", "minimum": 0.0001, "maximum": 0.1},
                        "budget": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {k: {"type": "number", "minimum": 1e-6, "maximum": 1e6} for k in TASKS},
                        },
                    },
                },
            },
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "parent_ids",
            "target_metrics",
            "hypothesis",
            "model",
            "train",
            "data",
            "components",
            "plugins",
            "files",
        ],
        "properties": {
            "schema_version": {"const": "cajal.research.candidate.v2"},
            "parent_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "uniqueItems": True,
                "items": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            },
            "target_metrics": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": list(METRICS)}},
            "hypothesis": {"type": "string", "minLength": 8, "maxLength": 1600},
            "model": {"type": "object", "properties": model_properties, "additionalProperties": False},
            "train": {"type": "object", "properties": train_properties, "additionalProperties": False},
            "data": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    k: {"type": "integer", "minimum": low, "maximum": high} for k, (low, high) in DATA_RANGES.items()
                },
            },
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["slot", "name", "params"],
                    "properties": {
                        "slot": {
                            "enum": [
                                "value_embedding",
                                "token",
                                "block_attention",
                                "block_mixer",
                                "latent",
                                "classifier",
                                "decoder",
                                "neighborhood",
                            ]
                        },
                        "name": {"type": "string"},
                        "params": {"type": "object"},
                    },
                },
            },
            "plugins": {"type": "array", "items": plugin},
            "files": {"type": "object", "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 32768}},
        },
    }


def _number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"expected finite number in [{low}, {high}]")


def validate_bundle(bundle, parent_ids=None):
    from .code_policy import inspect_source
    from .schema import validate

    validate(bundle, candidate_schema())
    from .components import validate_specs

    validate_specs(bundle["components"])
    if parent_ids is not None and not set(bundle["parent_ids"]) <= set(parent_ids):
        raise ValueError("unknown or unevaluated parent")
    base = reference_document()
    model = {**base["ModelConfig"], **bundle["model"]}
    for key, value in bundle["model"].items():
        if key in MODEL_SPACE:
            # Do not let bool pass for an integer enum.
            if not any(type(value) is type(v) and value == v for v in MODEL_SPACE[key]):
                raise ValueError(f"invalid model.{key}")
        elif key in MODEL_RANGES:
            _number(value, *MODEL_RANGES[key])
        else:
            raise ValueError(f"model.{key} is immutable or unsupported")
    if model["d_model"] % model["n_heads"] or model["d_model"] // model["n_heads"] < 64:
        raise ValueError("FA3 requires integral head dimensions >= 64")
    if not model["min_loops"] <= model["loops"] <= model["max_loops"]:
        raise ValueError("evaluation loops must lie inside the training loop range")
    if model["top_k"] > model["n_experts"] or model["n_hash_layers"] > model["n_prelude"]:
        raise ValueError("invalid routing dimensions")
    for key, value in bundle["train"].items():
        if key in TRAIN_RANGES:
            _number(value, *TRAIN_RANGES[key])
        elif key in TRAIN_INTS:
            if type(value) is not int:
                raise ValueError("training step controls require integers")
            _number(value, *TRAIN_INTS[key])
        elif key in ("lr_schedule", "optimizer_params", "balancer_params"):
            pass  # precise cross-field checks below
        elif key == "balancer":
            if value not in BALANCERS:
                raise ValueError("unknown Cajal balancer")
        elif key == "optimizer":
            if value not in OPTIMIZERS:
                raise ValueError("unsupported optimizer")
        elif key == "task_weights":
            if set(value) != set(TASKS):
                raise ValueError("all five joint objectives must remain active")
            for w in value.values():
                _number(w, 0.01, 20.0)
        elif key == "objectives":
            from brainwm_unified.regularizers import OBJECTIVES

            if not isinstance(value, list):
                raise ValueError("objectives must be a list")
            seen = set()
            for entry in value:
                if not isinstance(entry, dict) or set(entry) - {
                    "name",
                    "weight",
                    "coeff",
                    "label",
                    "n_cells",
                    "n_proj",
                    "t_max",
                    "n_t",
                    "start",
                    "budget",
                    "dual_lr",
                }:
                    raise ValueError("only existing tensor-only regularizers are searchable")
                if entry.get("name") not in REGULARIZERS:
                    raise ValueError("external teacher/data paths are outside the candidate contract")
                if entry["name"] in seen:
                    raise ValueError("duplicate objective")
                seen.add(entry["name"])
                _number(entry.get("weight", 1.0), 0.0, 20.0)
                if "budget" in entry and (not entry["budget"] or entry["weight"] <= 0):
                    raise ValueError(
                        "budgeted objective requires nonempty positive budgets and an initial positive weight"
                    )
                params = {k: v for k, v in entry.items() if k not in ("name", "weight", "start", "budget", "dual_lr")}
                OBJECTIVES.build(entry["name"], **params)
        else:
            raise ValueError(f"train.{key} is immutable or unsupported")
    train = {**base["TrainConfig"], **bundle["train"]}
    allowed_opt = {
        "grouped_muon": {"adam_beta1", "adam_beta2", "adam_eps", "muon_momentum", "muon_ns_steps"},
        "adamw": {"adam_beta1", "adam_beta2", "adam_eps"},
        "adam": {"adam_beta1", "adam_beta2", "adam_eps"},
        "sgd": {"sgd_momentum"},
        "custom": set(),
    }
    if set(train.get("optimizer_params", {})) - allowed_opt[train["optimizer"]]:
        raise ValueError("optimizer parameters do not apply to the selected optimizer")
    opt = train.get("optimizer_params", {})
    if opt.get("adam_beta1", 0.9) >= opt.get("adam_beta2", 0.999):
        raise ValueError("Adam beta1 must be below beta2")
    balancer = train["balancer"]
    if balancer == "famo" and any(e.get("start", 0) for e in train.get("objectives", [])):
        raise ValueError("FAMO requires the same active objective set throughout the run")
    params = bundle["train"].get("balancer_params", {})
    if set(params) - set(BALANCER_RANGES[balancer]):
        raise ValueError("unknown balancer parameter")
    for k, value in params.items():
        _number(value, *BALANCER_RANGES[balancer][k])
        if k == "every" and type(value) is not int:
            raise ValueError("grad_equal.every must be integral")
    for spec in bundle["components"]:
        if spec["name"] == "legacy.encoder.scprint_value" and model["input_norm"] != "log1p":
            raise ValueError("scPRINT value embedding requires Cajal's nonnegative log1p input")
        if spec["slot"] == "decoder":
            decoder = "unified" if spec["name"] == "cajal.unified_decoder" else "separate"
            if "decoder" in bundle["model"] and bundle["model"]["decoder"] != decoder:
                raise ValueError("decoder component conflicts with model.decoder")
    if (
        any(e["name"] == "batch_adversary" for e in bundle["train"].get("objectives", []))
        and not model["batch_adversary"]
    ):
        raise ValueError("batch_adversary objective requires its model head")
    if any(len(s.encode()) > 32768 for s in bundle["files"].values()):
        raise ValueError("source file exceeds 32 KiB")
    if sum(len(s.encode()) for s in bundle["files"].values()) > 131072:
        raise ValueError("total source exceeds 128 KiB")
    names, used, singleton = set(), set(), set()
    for p in bundle["plugins"]:
        if set(p) != {"name", "kind", "file", "parameters", "settings"}:
            raise ValueError("invalid plugin fields")
        if not re.fullmatch(r"custom_[a-z][a-z0-9_]{1,47}", p["name"]) or p["name"] in names:
            raise ValueError("invalid or duplicate plugin name")
        names.add(p["name"])
        if p["kind"] not in KINDS:
            raise ValueError("unsupported Cajal plugin kind")
        if p["kind"] != "token_transform":
            if p["kind"] in singleton:
                raise ValueError("only one plugin per loss/optimizer slot")
            singleton.add(p["kind"])
        file = p["file"]
        if (
            not re.fullmatch(r"(modules|losses|optimizers)/[a-z][a-z0-9_]{0,63}\.py", file)
            or file not in bundle["files"]
        ):
            raise ValueError("invalid plugin source path")
        used.add(file)
        entry = "compute" if p["kind"].endswith("loss") else "build"
        inspect_source(bundle["files"][file], entry)
        if not isinstance(p["parameters"], dict) or set(p["settings"]) - set(p["parameters"]):
            raise ValueError("undeclared plugin parameter")
        for name, spec in p["parameters"].items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) or name in (
                "hidden_dim",
                "parameters",
                "lr",
                "weight_decay",
            ):
                raise ValueError("invalid plugin parameter name")
            if set(spec) - {"type", "default", "minimum", "maximum", "enum"} or not {"type", "default"} <= set(spec):
                raise ValueError("invalid plugin parameter declaration")
            for value in (spec["default"], p["settings"].get(name, spec["default"])):
                typ = {"integer": int, "number": (int, float), "boolean": bool, "string": str}.get(spec["type"])
                if typ is None or type(value) not in (typ if isinstance(typ, tuple) else (typ,)):
                    raise ValueError("plugin parameter type mismatch")
                if isinstance(value, (int, float)) and type(value) is not bool:
                    if "minimum" not in spec or "maximum" not in spec:
                        raise ValueError("numeric plugin parameters require bounds")
                    _number(value, spec["minimum"], spec["maximum"])
                if "enum" in spec and value not in spec["enum"]:
                    raise ValueError("plugin parameter enum mismatch")
    if used != set(bundle["files"]):
        raise ValueError("every source file must be connected to a declared plugin")
    if ("optimizer" in singleton) != (bundle["train"].get("optimizer") == "custom"):
        raise ValueError("custom optimizer must be declared and selected together")
    return bundle


def semantic_id(bundle):
    """Ignore ancestry, comments and explicit defaults; not arbitrary program equivalence."""
    b = copy.deepcopy(bundle)
    base = reference_document()
    for section, cls in (("model", "ModelConfig"), ("train", "TrainConfig"), ("data", "DataConfig")):
        b[section] = {k: v for k, v in b[section].items() if v != base[cls].get(k)}
    b["files"] = {k: ast.dump(ast.parse(v), include_attributes=False) for k, v in b["files"].items()}
    return digest({k: b[k] for k in ("model", "train", "data", "components", "plugins", "files")})


def resolved_document(bundle, cfg, seed):
    validate_bundle(bundle)
    if cfg.get("weight_policy") != DEFINITION["campaign"]["weight_policy"] or cfg.get("trusted_teachers"):
        raise ValueError("historical weight dependencies are outside this research protocol")
    base = reference_document(cfg)
    payload = {"plugins": bundle["plugins"], "files": bundle["files"]} if bundle["plugins"] else {}
    model = {**base["ModelConfig"], **bundle["model"], "research_plugins": payload}
    model["research_components"] = bundle["components"]
    for spec in bundle["components"]:
        if spec["slot"] == "decoder":
            model["decoder"] = "unified" if spec["name"] == "cajal.unified_decoder" else "separate"
    train = {
        **base["TrainConfig"],
        **bundle["train"],
        "steps": cfg["schedule_steps"],
        "stop_at": cfg["stop_at"],
        "research_plugins": payload,
        "keep_every": cfg["stop_at"],
    }
    if bundle["train"].get("balancer", base["TrainConfig"]["balancer"]) != base["TrainConfig"]["balancer"]:
        train["balancer_params"] = bundle["train"].get("balancer_params", {})
    train["objectives"] = copy.deepcopy(train["objectives"])
    data = {**base["DataConfig"], **bundle["data"], "root": cfg["data"]["pretrain"], "seed": seed}
    return {"model": model, "train": train, "data": data}
