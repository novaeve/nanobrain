"""Immutable experiment boundary; candidate code can only see declared tensors."""

import ast
import copy
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("gene", "integration", "cell_type", "anatomy", "spatial")
METRICS = (
    "Cell type fine macro-F1",
    "Region fine macro-F1",
    "Integration bio",
    "Integration batch",
    "Gene imputation Pearson (cell median)",
    "Spatial imputation Pearson (cell median)",
)
MODEL_SPACE = {
    "d_model": [128, 256, 384, 512],
    "n_heads": [2, 4, 6, 8],
    "n_prelude": [1, 2, 3, 4],
    "n_core": [1, 2, 3, 4],
    "n_coda": [1, 2, 3, 4],
    "loops": [2, 3, 4, 5, 6],
    "min_loops": [1, 2, 3, 4],
    "max_loops": [2, 3, 4, 5, 6, 8],
    "n_experts": [8, 16, 32, 64],
    "top_k": [1, 2, 4, 8],
    "n_shared_experts": [1, 2],
    "expert_hidden": [128, 256, 384, 512],
    "router_score": ["sqrtsoftplus", "sigmoid"],
    "n_hash_layers": [0, 1, 2],
    "nbeats": [False, True],
    "loop_injection": [False, True],
    "z_dim": [32, 64, 128],
    "decoder_hidden": [256, 512, 768, 1024],
    "decoder": ["separate", "unified"],
    "latent_reg": ["kl", "mog", "none"],
    "mog_components": [8, 16, 32, 64],
    "batch_adversary": [0, 64, 128, 256],
    "residual_dim": [0, 16, 32, 64],
    "recon_detach_mu": [False, True],
    "spatial_d": [64, 128, 256],
    "spatial_layers": [1, 2, 3, 4],
    "spatial_residual_mean": [False, True],
    "input_norm": ["log1p", "pearson"],
}
MODEL_RANGES = {"router_bias_rate": (0.0, 0.01), "spatial_latent": (0.0, 5.0)}
TRAIN_RANGES = {
    "lr": (1e-6, 0.01),
    "weight_decay": (0.0, 0.2),
    "grad_clip": (0.1, 10.0),
    "kl_weight": (0.01, 10.0),
    "kl_target": (0.0, 200.0),
    "kl_dual_lr": (0.0001, 0.1),
}
KINDS = ("token_transform", "gene_loss", "cell_type_loss", "anatomy_loss", "optimizer")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_config(path=None):
    cfg = json.loads(Path(path or ROOT / "configs/cajal_react.json").read_text())
    if cfg["reference_recipe"] != "configs/experiments/cajal_v1.toml":
        raise ValueError("reference must be the frozen existing Cajal v1 recipe")
    if cfg["schedule_steps"] != 30000 or cfg["stop_at"] != 10000:
        raise ValueError("this protocol uses the Cajal 30k schedule stopped at 10k")
    if len(set(cfg["seeds"])) != 3 or any(type(s) is not int for s in cfg["seeds"]):
        raise ValueError("three distinct paired seeds are required")
    if cfg["checkpoint_selection"] != "fixed_stop" or cfg["scaling_enabled"]:
        raise ValueError("fixed-stop Cajal evaluation; no automatic scaling in this protocol")
    if not cfg["gpu_ids"] or any(type(g) is not int or g not in range(5) for g in cfg["gpu_ids"]):
        raise ValueError("GPU whitelist is 0..4")
    if cfg["max_workers"] != 1:
        raise ValueError("prepared scheduler has one active Cajal worker; edit protocol implementation before widening")
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
        "schema_version": "cajal.research.candidate.v1",
        "parent_ids": ["0" * 64],
        "target_metrics": list(METRICS),
        "hypothesis": "Existing Cajal v1 architecture and grad_equal joint recipe, unchanged.",
        "model": {},
        "train": {},
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
        {
            "balancer": {"enum": ["initial", "grad_equal", "famo"]},
            "optimizer": {"enum": ["grouped_muon", "adamw", "adam", "sgd", "custom"]},
            "task_weights": {
                "type": "object",
                "additionalProperties": False,
                "required": list(TASKS),
                "properties": {k: {"type": "number", "minimum": 0.05, "maximum": 10.0} for k in TASKS},
            },
            "objectives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "weight"],
                    "properties": {
                        "name": {"enum": ["cond_hsic", "sigreg", "batch_swap", "batch_adversary"]},
                        "weight": {"type": "number", "minimum": 0.0, "maximum": 5.0},
                        "coeff": {"type": "number"},
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
            "plugins",
            "files",
        ],
        "properties": {
            "schema_version": {"const": "cajal.research.candidate.v1"},
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
        elif key == "balancer":
            if value not in ("initial", "grad_equal", "famo"):
                raise ValueError("unknown Cajal balancer")
        elif key == "optimizer":
            if value not in ("grouped_muon", "adamw", "adam", "sgd", "custom"):
                raise ValueError("unsupported optimizer")
        elif key == "task_weights":
            if set(value) != set(TASKS):
                raise ValueError("all five joint objectives must remain active")
            for w in value.values():
                _number(w, 0.05, 10.0)
        elif key == "objectives":
            from brainwm_unified.regularizers import OBJECTIVES

            if not isinstance(value, list):
                raise ValueError("objectives must be a list")
            seen = set()
            for entry in value:
                if not isinstance(entry, dict) or set(entry) - {"name", "weight", "coeff"}:
                    raise ValueError("only existing tensor-only regularizers are searchable")
                if entry.get("name") not in ("cond_hsic", "sigreg", "batch_swap", "batch_adversary"):
                    raise ValueError("external teacher/data paths are outside the candidate contract")
                if entry["name"] in seen:
                    raise ValueError("duplicate objective")
                seen.add(entry["name"])
                _number(entry.get("weight", 1.0), 0.0, 5.0)
                OBJECTIVES.build(entry["name"], **{k: v for k, v in entry.items() if k not in ("name", "weight")})
        else:
            raise ValueError(f"train.{key} is immutable or unsupported")
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
    for section, cls in (("model", "ModelConfig"), ("train", "TrainConfig")):
        b[section] = {k: v for k, v in b[section].items() if v != base[cls].get(k)}
    b["files"] = {k: ast.dump(ast.parse(v), include_attributes=False) for k, v in b["files"].items()}
    return digest({k: b[k] for k in ("model", "train", "plugins", "files")})


def resolved_document(bundle, cfg, seed):
    validate_bundle(bundle)
    base = reference_document(cfg)
    payload = {"plugins": bundle["plugins"], "files": bundle["files"]} if bundle["plugins"] else {}
    model = {**base["ModelConfig"], **bundle["model"], "research_plugins": payload}
    train = {
        **base["TrainConfig"],
        **bundle["train"],
        "steps": cfg["schedule_steps"],
        "stop_at": cfg["stop_at"],
        "research_plugins": payload,
        "keep_every": cfg["stop_at"],
    }
    if bundle["train"].get("balancer", base["TrainConfig"]["balancer"]) != base["TrainConfig"]["balancer"]:
        train["balancer_params"] = {}
    data = {**base["DataConfig"], "root": cfg["data"]["pretrain"], "seed": seed}
    return {"model": model, "train": train, "data": data}
