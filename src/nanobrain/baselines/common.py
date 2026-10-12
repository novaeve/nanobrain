"""Data conversions shared by official-library baseline adapters."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from nanobrain.data import NanoDataset


def array(value: Any, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    output = np.asarray(value)
    return output.astype(dtype, copy=False) if dtype is not None else output


def labels(dataset: NanoDataset, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = array(dataset.labels[name])
    mask = array(dataset.masks[name]).astype(bool, copy=False)
    valid = mask & (values >= 0) if values.ndim == 1 else mask.reshape(len(values), -1).all(1)
    return values, valid


def metadata_array(dataset: NanoDataset, name: str, default: int = -1) -> np.ndarray:
    value = (dataset.metadata or {}).get(name)
    return np.asarray(value) if isinstance(value, np.ndarray) else np.full(dataset.size, default)


def train_pca(train: NanoDataset, validation: NanoDataset, *, seed: int, dimensions: int = 32) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    x_train = array(train.features, np.float32)
    x_validation = array(validation.features, np.float32)
    dimensions = max(2, min(dimensions, x_train.shape[1], x_train.shape[0] - 1))
    model = PCA(n_components=dimensions, svd_solver="randomized", random_state=int(seed))
    return model.fit_transform(x_train).astype(np.float32), model.transform(x_validation).astype(np.float32)


def log_expression(dataset: NanoDataset, data_root: Path) -> np.ndarray:
    """Undo the train-fitted z score and return non-negative log expression."""
    mean = np.load(data_root / "normalization_mean.npy", allow_pickle=False)
    scale = np.load(data_root / "normalization_std.npy", allow_pickle=False)
    values = array(dataset.features, np.float32) * scale + mean
    return np.maximum(values, 0.0).astype(np.float32, copy=False)


def panel_pseudocounts(dataset: NanoDataset, data_root: Path) -> np.ndarray:
    """Convert stored log expression into integer panel pseudo-counts.

    The compact materialization intentionally stores normalized expression,
    not the original count matrix.  scVI-family likelihoods require counts,
    so the adapter uses the documented inverse log1p conversion and records
    this adaptation in every result receipt.
    """
    values = np.expm1(log_expression(dataset, data_root))
    return np.rint(np.maximum(values, 0.0)).astype(np.float32)


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


__all__ = [
    "array",
    "labels",
    "log_expression",
    "metadata_array",
    "package_version",
    "panel_pseudocounts",
    "train_pca",
]
