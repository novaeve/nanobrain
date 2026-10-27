"""Dataset-level upstream components used as explicit, frozen graph inputs.

These operators are transductive within EACH split, use expression and batch
IDs only, and are never applied to learned minibatch embeddings. Downstream
neural layers remain trainable. No straight-through gradient is invented.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

METHODS = ("harmony", "scanorama", "bbknn")
PREFIX = "integration_component_"
SEED = 20261023


class CellTypistLinearHead(nn.Linear):
    """CellTypist's linear logistic decision function, newly initialized.

    The component is the coefficient matrix + intercept, not CellTypist's
    expression preprocessing, fitted atlas, or original fitting procedure.
    """
    def __init__(self, hidden_dim, output_dim):
        super().__init__(hidden_dim, output_dim, bias=True)


class IntegratedInput(nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= 1:
            raise ValueError("integration strength must be in [0,1]")
        self.strength = float(strength)

    def forward(self, raw, *, integrated):
        if raw.shape != integrated.shape:
            raise ValueError("cached integration component must match raw feature shape")
        return torch.lerp(raw, integrated.to(raw), self.strength)


def register_components():
    from .registry import registry
    registry("head").register("celltypist_linear", CellTypistLinearHead,
        source="CellTypist linear logistic decision function; fresh coefficients/intercept",
        input_rank=2, output_rank=2,
        notes="Classification logits only. No pretrained atlas or original fitting procedure.")
    for name in METHODS:
        registry("operation").register(name, IntegratedInput,
            source={"harmony": "harmonypy.run_harmony.Z_corr",
                    "scanorama": "scanorama.assemble MNN correction on fixed panel features",
                    "bbknn": "bbknn.matrix.bbknn connectivities + one-hop feature diffusion"}[name],
            input_rank=2, output_rank=2,
            notes="Backbone input-only node: inputs=['features'], repeat=1. Frozen split-level "
                  "expression+batch operation, no biological labels; strength in [0,1]. "
                  "Can feed any encoder or merge parallel input branches. Not a spatial geometry hop.")


def selected_components(genome):
    spec = genome.to_dict().get("architecture")
    return sorted({n["name"] for n in spec["backbone"]["nodes"]
                   if n["kind"] == "operation" and n["name"] in METHODS}) if spec else []


def compute_component(name, features, batches):
    """Official algorithm calls with fixed, bounded settings and stable row order."""
    from threadpoolctl import threadpool_limits
    x = np.asarray(features, dtype=np.float32)
    batches = np.asarray(batches)
    unique = np.unique(batches)
    if len(unique) < 2:
        raise ValueError(f"{name} requires at least two batches")
    with threadpool_limits(limits=2):
        if name == "harmony":
            import harmonypy
            import pandas as pd
            result = harmonypy.run_harmony(x, pd.DataFrame({"batch": batches.astype(str)}), "batch",
                nclust=max(2, min(100, round(len(x) / 30))), max_iter_harmony=10,
                max_iter_kmeans=10, random_state=SEED, ncores=2, verbose=False)
            out = np.asarray(result.Z_corr, dtype=np.float32)
            if out.shape != x.shape and out.T.shape == x.shape:
                out = out.T
        elif name == "scanorama":
            import scanorama
            groups = [np.flatnonzero(batches == b) for b in unique]
            # assemble operates directly on supplied embeddings; no fake gene names,
            # renormalization or dimensionality reduction of signed panel values.
            rng = np.random.get_state()
            try:
                np.random.seed(SEED)
                assembled = scanorama.assemble([x[i].copy() for i in groups],
                    knn=max(1, min(20, min(map(len, groups)))), approx=True, verbose=0,
                    batch_size=5000)
            finally:
                np.random.set_state(rng)
            out = np.empty_like(x)
            for indices, values in zip(groups, assembled, strict=True):
                out[indices] = values
        elif name == "bbknn":
            from bbknn.matrix import bbknn
            from scipy import sparse
            from sklearn.decomposition import PCA
            coordinates = PCA(n_components=min(32, x.shape[1], len(x) - 1),
                              random_state=SEED, svd_solver="randomized").fit_transform(x)
            minimum = min(np.count_nonzero(batches == b) for b in unique)
            from .baselines.integration import _bbknn_exact_backend
            within = min(3, minimum)
            _, graph, _ = bbknn(coordinates, batches.astype(str), neighbors_within_batch=within,
                n_pcs=coordinates.shape[1], computation=_bbknn_exact_backend(within), metric="euclidean")
            graph = graph.tocsr()
            degree = np.asarray(graph.sum(1)).ravel()
            out = sparse.diags(1 / np.maximum(degree, 1e-12)) @ graph @ x
            out[degree == 0] = x[degree == 0]
        else:
            raise ValueError(f"unknown integration component {name}")
    out = np.asarray(out, dtype=np.float32)
    if out.shape != x.shape or not np.isfinite(out).all():
        raise ValueError(f"invalid {name} output shape or nonfinite values")
    return np.ascontiguousarray(out)


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def prepare_component_views(splits, genome):
    """Attach cached split-level views without changing inputs or supervision."""
    methods = selected_components(genome)
    if not methods:
        return []
    receipts = []
    for dataset in (splits.train, splits.validation):
        if dataset is None:
            continue
        dataset.metadata = dict(dataset.metadata or {})
        x = dataset.features.detach().cpu().numpy()
        # Integration labels are technical batch IDs, NOT biology labels.
        batch = dataset.labels["integration"].detach().cpu().numpy().copy()
        batch[~dataset.masks["integration"].cpu().numpy()] = -1
        for name in methods:
            package = {"harmony": "harmonypy", "scanorama": "scanorama", "bbknn": "bbknn"}[name]
            identity = dict(method=name, seed=SEED, version=importlib.metadata.version(package),
                implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                features_sha256=_digest(x), batch_sha256=_digest(batch),
                split=dataset.metadata.get("component_split", "synthetic"),
                policy="within-split expression+technical-batch only; frozen input transform")
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            root = Path(os.environ.get("NANOBRAIN_COMPONENT_CACHE",
                str(Path(__file__).resolve().parents[2] / "runtime" / "integration_components")))
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{name}-{key}.npy"
            import fcntl
            with (root / f"{name}-{key}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not path.exists() or not path.with_suffix(".json").exists():
                    values = compute_component(name, x, batch)
                    with tempfile.NamedTemporaryFile(dir=root, suffix=".npy", delete=False) as tmp:
                        np.save(tmp, values, allow_pickle=False)
                        temporary = tmp.name
                    os.replace(temporary, path)
                    receipt = {**identity, "output_sha256": _digest(values), "rows": len(x), "width": x.shape[1]}
                    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
                receipt = json.loads(path.with_suffix(".json").read_text())
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if values.shape != x.shape or _digest(values) != receipt["output_sha256"]:
                    raise ValueError(f"integration cache integrity failed: {path}")
            dataset.metadata[PREFIX + name] = values
            receipts.append({**receipt, "cache_path": str(path)})
    return receipts


register_components()
