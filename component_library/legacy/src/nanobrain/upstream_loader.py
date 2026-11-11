"""Load pinned upstream model definitions, avoiding unrelated training imports.

AST isolation executes the ORIGINAL selected class/function bodies unchanged.
Every file is checked against the collected source manifest before execution.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from functools import lru_cache
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "upstream_components"


@lru_cache(None)
def source_path(method, relative):
    manifest = json.loads((SOURCES.parent / "baseline_candidates_20261026" / "source_manifest.json").read_text()) \
        if not (ROOT / "upstream_source_manifest.json").exists() else json.loads((ROOT / "upstream_source_manifest.json").read_text())
    record = next(m for m in manifest if m["name"] == method)
    file = next(f for f in record["files"] if f["path"] == relative)
    path = SOURCES / method / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != file["sha256"]:
        raise ValueError(f"upstream source integrity failed: {method}/{relative}")
    return path


@lru_cache(None)
def selected(method, relative, names):
    """Dependency isolation, not a rewritten implementation or silent fallback."""
    path = source_path(method, relative)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    chosen = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in chosen} != set(names):
        raise ValueError("upstream requested definition is missing")
    namespace = dict(torch=torch, nn=nn, F=F, np=np, math=math, OrderedDict=OrderedDict,
        Tensor=Tensor, Callable=Callable, Optional=Optional, Iterable=Iterable, Tuple=Tuple, Union=Union,
        Module=nn.Module, Parameter=nn.Parameter,
        logger=logging.getLogger("nanobrain.upstream"), __name__="nanobrain.upstream_selected")
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@lru_cache(None)
def whole_file(method, relative):
    path = source_path(method, relative)
    package = {"scTab": "cellnet", "scBERT": "performer_pytorch"}.get(method)
    if package:
        # Supply only a package path; do not import the full method/trainer.
        package_root = SOURCES / method / package
        if package not in sys.modules:
            module = ModuleType(package); module.__path__ = [str(package_root)]
            sys.modules[package] = module
    name = "nanobrain.upstream_" + method.lower() + "_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def upstream_fingerprint_files():
    """Include every collected source and license in future experiment freezes."""
    manifest = json.loads((ROOT / "upstream_source_manifest.json").read_text())
    return [ROOT / "upstream_source_manifest.json", *[
        SOURCES / method["name"] / file["path"] for method in manifest for file in method["files"]]]
