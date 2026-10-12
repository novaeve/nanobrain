"""Deterministic data sources for nanoBrain experiments.

The real-data path in this module consumes the already materialized BrainWM V3
NumPy shards.  H5AD files are deliberately outside the training data path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
import warnings

import numpy as np
import torch

from .contracts import BrainBatch

if TYPE_CHECKING:
    from .development import DevelopmentPairDataset


@dataclass
class NanoDataset:
    """In-memory split with a stable batch iterator.

    The object deliberately stores only compact dense features.  The formal
    benchmark's sparse CSR reader can produce the same arrays through the
    subset builder without changing the model or trainer contract.
    """

    features: torch.Tensor
    labels: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    coordinates: torch.Tensor | None = None
    neighbors: torch.Tensor | None = None
    neighbor_mask: torch.Tensor | None = None
    development_time: torch.Tensor | None = None
    metadata: dict[str, np.ndarray | list[str]] | None = None
    # Optional extensions are appended to preserve the original positional
    # constructor order used by early nanoBrain experiments.
    feature_observed: torch.Tensor | None = None
    development_condition: torch.Tensor | None = None
    development_condition_observed: torch.Tensor | None = None
    completion_target: torch.Tensor | None = None
    completion_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.features.ndim != 2 or self.features.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError("features must be a floating rank-2 tensor")
        n = len(self.features)
        if n < 1:
            raise ValueError("dataset cannot be empty")
        if self.feature_observed is not None:
            if self.feature_observed.shape != self.features.shape or self.feature_observed.dtype != torch.bool:
                raise ValueError("feature_observed must be bool with the features shape")
        for name, value in self.labels.items():
            if len(value) != n:
                raise ValueError(f"label {name!r} has {len(value)} rows, expected {n}")
        for name, value in self.masks.items():
            if len(value) != n or value.dtype != torch.bool:
                raise ValueError(f"mask {name!r} must be bool with {n} rows")
        if self.coordinates is not None and (self.coordinates.ndim != 2 or len(self.coordinates) != n):
            raise ValueError("coordinates must have shape [rows, dimensions]")
        if self.neighbors is not None:
            if self.neighbor_mask is None or self.neighbors.shape != self.neighbor_mask.shape:
                raise ValueError("neighbors and neighbor_mask must have the same shape")
            if self.neighbors.ndim != 2 or self.neighbors.shape[0] != n:
                raise ValueError("neighbors must have shape [rows, bounded_neighbors]")
        if self.development_time is not None and len(self.development_time) != n:
            raise ValueError("development_time row count differs")
        for name, value in (
            ("development_condition", self.development_condition),
            ("development_condition_observed", self.development_condition_observed),
            ("completion_target", self.completion_target),
            ("completion_mask", self.completion_mask),
        ):
            if value is not None and len(value) != n:
                raise ValueError(f"{name} row count differs")
        if self.development_condition_observed is not None and self.development_condition_observed.dtype != torch.bool:
            raise ValueError("development_condition_observed must be bool")
        if self.completion_target is not None and self.completion_target.shape != self.features.shape:
            raise ValueError("completion_target must match features shape")
        if self.completion_mask is not None:
            if self.completion_mask.shape != self.features.shape or self.completion_mask.dtype != torch.bool:
                raise ValueError("completion_mask must be bool with the features shape")

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    def batch(self, indices: np.ndarray | torch.Tensor) -> BrainBatch:
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices.astype(np.int64, copy=False))
        elif not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.int64)
        indices = indices.long()
        local_neighbors = None
        local_neighbor_mask = None
        context_indices = None
        if self.neighbors is not None:
            # Neighbor sidecars use split-local row IDs.  A minibatch is a
            # view, so remap only neighbors that are present in this view;
            # outside-batch edges are masked instead of indexing the full
            # dataset (which would leak context and can raise OOB errors).
            selected = indices.detach().cpu().numpy().astype(np.int64, copy=False)
            inverse = np.full(self.size, -1, dtype=np.int64)
            inverse[selected] = np.arange(len(selected), dtype=np.int64)
            global_neighbors = self.neighbors[indices].detach().cpu().numpy().astype(np.int64, copy=False)
            global_mask = self.neighbor_mask[indices].detach().cpu().numpy().astype(bool, copy=False)
            mapped = np.zeros_like(global_neighbors, dtype=np.int64)
            valid = global_mask & (global_neighbors >= 0) & (global_neighbors < self.size)
            # Metadata V2 supplies all one-hop input neighbors within this split.
            # Context has no labels and is never a target row; it only supplies
            # differentiable expression features to the spatial operator.
            if (self.metadata or {}).get("subset_manifest", {}).get("identity_repair"):
                outside = np.unique(global_neighbors[valid & (inverse[global_neighbors.clip(0, self.size-1)] < 0)])
                context_indices = torch.from_numpy(outside)
                inverse[outside] = len(selected) + np.arange(len(outside))
            mapped[valid] = inverse[global_neighbors[valid]]
            local_neighbor_mask_np = valid & (mapped >= 0)
            # Every row has a legal self index even when no in-batch spatial
            # neighbors survive; the spatial module ignores this slot.
            self_slots = np.arange(len(selected), dtype=np.int64)[:, None]
            empty = ~local_neighbor_mask_np.any(axis=1)
            if mapped.shape[1]:
                mapped[empty, 0] = self_slots[empty, 0]
                local_neighbor_mask_np[empty, 0] = False
            local_neighbors = torch.from_numpy(mapped)
            local_neighbor_mask = torch.from_numpy(local_neighbor_mask_np)
        return BrainBatch(
            features=self.features[indices],
            feature_observed=self.feature_observed[indices] if self.feature_observed is not None else None,
            labels={name: value[indices] for name, value in self.labels.items()},
            label_masks={name: value[indices] for name, value in self.masks.items()},
            coordinates=self.coordinates[indices] if self.coordinates is not None else None,
            neighbors=local_neighbors,
            neighbor_mask=local_neighbor_mask,
            neighbor_features=self.features[context_indices] if context_indices is not None and len(context_indices) else None,
            neighbor_completion_mask=(self.completion_mask[context_indices]
                if context_indices is not None and len(context_indices) and self.completion_mask is not None else None),
            development_time=self.development_time[indices] if self.development_time is not None else None,
            development_condition=self.development_condition[indices] if self.development_condition is not None else None,
            development_condition_observed=(
                self.development_condition_observed[indices]
                if self.development_condition_observed is not None else None
            ),
            completion_target=self.completion_target[indices] if self.completion_target is not None else None,
            completion_mask=self.completion_mask[indices] if self.completion_mask is not None else None,
            metadata={
                key: value[indices.numpy()] if isinstance(value, np.ndarray) else value
                for key, value in (self.metadata or {}).items()
            },
        ).validate()

    def iter_batches(self, batch_size: int, *, shuffle: bool = False, seed: int = 0) -> Iterator[BrainBatch]:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        order = np.arange(self.size, dtype=np.int64)
        if shuffle:
            np.random.default_rng(int(seed)).shuffle(order)
        for start in range(0, self.size, batch_size):
            yield self.batch(order[start : start + batch_size])


@dataclass(frozen=True)
class DatasetSplits:
    train: NanoDataset
    validation: NanoDataset
    test: NanoDataset | None
    development_train: "DevelopmentPairDataset | None" = None
    development_validation: "DevelopmentPairDataset | None" = None
    development_test: "DevelopmentPairDataset | None" = None


def _make_neighbors(coordinates: np.ndarray, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(coordinates)
    k = max(1, min(int(k), n))
    distance = ((coordinates[:, None, :2] - coordinates[None, :, :2]) ** 2).sum(-1)
    order = np.argsort(distance, axis=1, kind="stable")[:, :k]
    mask = np.ones(order.shape, dtype=bool)
    # Explicitly retain the self-loop slot.  The spatial module ignores it and
    # therefore has an exact identity fallback when it is the only slot.
    return torch.from_numpy(order.astype(np.int64)), torch.from_numpy(mask)


def make_synthetic_splits(
    *,
    train_rows: int = 1024,
    validation_rows: int = 256,
    test_rows: int = 256,
    input_dim: int = 32,
    annotation_classes: int = 5,
    niche_classes: int = 4,
    integration_classes: int = 3,
    development_dim: int = 4,
    seed: int = 7,
    neighbor_k: int = 4,
) -> DatasetSplits:
    """Generate a deterministic, learnable four-task benchmark.

    This source is intentionally self-contained so architecture iterations do
    not depend on a network download.  The real-subset builder writes the same
    field names and is interchangeable with this generator.
    """

    if min(train_rows, validation_rows, test_rows, input_dim, development_dim) < 1:
        raise ValueError("dataset sizes and dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    total = int(train_rows + validation_rows + test_rows)
    latent_dim = max(8, min(16, input_dim))
    latent = rng.normal(size=(total, latent_dim)).astype(np.float32)
    projection = rng.normal(size=(latent_dim, input_dim)).astype(np.float32) / np.sqrt(latent_dim)
    features = np.tanh(latent @ projection + 0.08 * rng.normal(size=(total, input_dim))).astype(np.float32)
    annotation_w = rng.normal(size=(latent_dim, annotation_classes)).astype(np.float32)
    niche_w = rng.normal(size=(latent_dim, niche_classes)).astype(np.float32)
    integration_w = rng.normal(size=(latent_dim, integration_classes)).astype(np.float32)
    annotation = np.argmax(latent @ annotation_w + 0.15 * rng.normal(size=(total, annotation_classes)), axis=1).astype(np.int64)
    # Coordinates carry a smooth spatial signal and are independent of row order.
    coordinates = np.stack(
        (latent[:, 0] + 0.1 * rng.normal(size=total), latent[:, 1] + 0.1 * rng.normal(size=total)),
        axis=1,
    ).astype(np.float32)
    niche = np.argmax(latent @ niche_w + 0.18 * rng.normal(size=(total, niche_classes)), axis=1).astype(np.int64)
    integration = np.argmax(latent @ integration_w + 0.35 * rng.normal(size=(total, integration_classes)), axis=1).astype(np.int64)
    time = (0.5 + 0.15 * latent[:, 0] + 0.03 * rng.normal(size=total)).astype(np.float32)
    dev_w = rng.normal(size=(latent_dim, development_dim)).astype(np.float32) / np.sqrt(latent_dim)
    development = (latent @ dev_w + time[:, None] * np.linspace(0.2, 0.8, development_dim)).astype(np.float32)
    # A small, deterministic amount of missing supervision exercises the same
    # null/mask path used by real V3 sidecars.
    masks = {
        "annotation": rng.random(total) > 0.05,
        "niche": rng.random(total) > 0.18,
        "integration": rng.random(total) > 0.03,
        "development": rng.random(total) > 0.08,
    }
    offsets = (0, train_rows, train_rows + validation_rows)
    lengths = (train_rows, validation_rows, test_rows)

    def make(start: int, length: int) -> NanoDataset:
        stop = start + length
        local_coordinates = coordinates[start:stop]
        local_neighbors, local_neighbor_mask = _make_neighbors(local_coordinates, neighbor_k)
        return NanoDataset(
            features=torch.from_numpy(features[start:stop]),
            labels={
                "annotation": torch.from_numpy(annotation[start:stop]),
                "niche": torch.from_numpy(niche[start:stop]),
                "integration": torch.from_numpy(integration[start:stop]),
                "development": torch.from_numpy(development[start:stop]),
            },
            masks={name: torch.from_numpy(value[start:stop]) for name, value in masks.items()},
            coordinates=torch.from_numpy(local_coordinates),
            neighbors=local_neighbors,
            neighbor_mask=local_neighbor_mask,
            development_time=torch.from_numpy(time[start:stop]),
            metadata={"split_row": np.arange(length, dtype=np.int64)},
        )

    return DatasetSplits(*(make(start, length) for start, length in zip(offsets, lengths)))


def save_dataset(dataset: NanoDataset, path: str | Path) -> None:
    """Write a compact `.npz` consumed by the CLI and external experiments."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "features": dataset.features.detach().cpu().numpy(),
        "feature_observed": (
            dataset.feature_observed.detach().cpu().numpy()
            if dataset.feature_observed is not None else np.empty((0, 0), bool)
        ),
        "coordinates": dataset.coordinates.detach().cpu().numpy() if dataset.coordinates is not None else np.empty((0, 2), np.float32),
        "neighbors": dataset.neighbors.detach().cpu().numpy() if dataset.neighbors is not None else np.empty((0, 0), np.int64),
        "neighbor_mask": dataset.neighbor_mask.detach().cpu().numpy() if dataset.neighbor_mask is not None else np.empty((0, 0), bool),
        "development_time": dataset.development_time.detach().cpu().numpy() if dataset.development_time is not None else np.empty((0,), np.float32),
        "development_condition": (
            dataset.development_condition.detach().cpu().numpy()
            if dataset.development_condition is not None else np.empty((0, 0), np.float32)
        ),
        "development_condition_observed": (
            dataset.development_condition_observed.detach().cpu().numpy()
            if dataset.development_condition_observed is not None else np.empty((0,), bool)
        ),
        "completion_target": (
            dataset.completion_target.detach().cpu().numpy()
            if dataset.completion_target is not None else np.empty((0, 0), np.float32)
        ),
        "completion_mask": (
            dataset.completion_mask.detach().cpu().numpy()
            if dataset.completion_mask is not None else np.empty((0, 0), bool)
        ),
    }
    for name, value in dataset.labels.items():
        arrays[f"label_{name}"] = value.detach().cpu().numpy()
    for name, value in dataset.masks.items():
        arrays[f"mask_{name}"] = value.detach().cpu().numpy()
    np.savez_compressed(target, **arrays)


def load_dataset(path: str | Path) -> NanoDataset:
    source = np.load(Path(path), allow_pickle=False)
    features = torch.from_numpy(np.asarray(source["features"], dtype=np.float32))
    observed_np = np.asarray(source["feature_observed"])
    labels = {
        name: torch.from_numpy(np.asarray(source[f"label_{name}"]))
        for name in ("annotation", "niche", "integration", "development")
    }
    masks = {
        name: torch.from_numpy(np.asarray(source[f"mask_{name}"]).astype(bool, copy=False))
        for name in ("annotation", "niche", "integration", "development")
    }
    coordinates_np = np.asarray(source["coordinates"])
    neighbors_np = np.asarray(source["neighbors"])
    neighbor_mask_np = np.asarray(source["neighbor_mask"])
    time_np = np.asarray(source["development_time"])
    condition_np = np.asarray(source["development_condition"]) if "development_condition" in source else np.empty((0, 0))
    condition_observed_np = np.asarray(source["development_condition_observed"]) if "development_condition_observed" in source else np.empty((0,))
    completion_target_np = np.asarray(source["completion_target"]) if "completion_target" in source else np.empty((0, 0))
    completion_mask_np = np.asarray(source["completion_mask"]) if "completion_mask" in source else np.empty((0, 0))
    return NanoDataset(
        features=features,
        feature_observed=torch.from_numpy(observed_np.astype(bool, copy=False)) if observed_np.size else None,
        labels=labels,
        masks=masks,
        coordinates=torch.from_numpy(coordinates_np) if coordinates_np.size else None,
        neighbors=torch.from_numpy(neighbors_np) if neighbors_np.size else None,
        neighbor_mask=torch.from_numpy(neighbor_mask_np.astype(bool, copy=False)) if neighbor_mask_np.size else None,
        development_time=torch.from_numpy(time_np) if time_np.size else None,
        development_condition=torch.from_numpy(condition_np.astype(np.float32, copy=False)) if condition_np.size else None,
        development_condition_observed=(
            torch.from_numpy(condition_observed_np.astype(bool, copy=False))
            if condition_observed_np.size else None
        ),
        completion_target=torch.from_numpy(completion_target_np.astype(np.float32, copy=False)) if completion_target_np.size else None,
        completion_mask=torch.from_numpy(completion_mask_np.astype(bool, copy=False)) if completion_mask_np.size else None,
    )


_SUBSET_FORMAT = "nanobrain_v3_subset_v1"
_SUBSET_METADATA = (
    "row_id",
    "source_row_ref",
    "source_split",
    "source_id",
    "assay",
    "dataset",
    "technology",
    "sample",
    "donor",
    "section",
    "split_group",
    "life_stage",
    "sex",
    "time_point_days",
    "time_lower_days",
    "time_upper_days",
    "chronological_time_proxy_days",
    "annotation_eligible",
    "development_eligible",
    "spatial_eligible",
    "integration_eligible",
    "anatomy",
)


def _readonly_tensor(path: Path, *, dtype: np.dtype | None = None) -> torch.Tensor:
    """Open a `.npy` file as a read-only memmap and expose it as Torch."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if dtype is not None and array.dtype != dtype:
        array = np.asarray(array, dtype=dtype)
    # Torch can index a read-only memmap without copying.  Suppress its
    # warning because the dataset contract never writes through this view.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(array)


def _subset_root(root: str | Path | None) -> Path:
    if root is None:
        root = os.environ.get("NANOBRAIN_DATA_DIR")
    if root is None:
        return Path(__file__).resolve().parents[2] / "assets" / "nanobrain_v3_100k_metadata_v2"
    return Path(root).expanduser().resolve()


def verify_subset_manifest(
    root: str | Path,
    *,
    verify_hashes: bool = False,
    data_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Validate selected materialized splits and optionally their checksums.

    Validation-only runs pass ``("train", "validation")`` so the sealed test
    arrays are never opened merely to establish their row count.
    """
    location = _subset_root(root)
    manifest_path = location / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"nanoBrain subset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != _SUBSET_FORMAT:
        raise ValueError(f"unsupported nanoBrain subset format: {manifest.get('format')!r}")
    if manifest.get("split_id") != "BRAINWM-HUMAN-GLOBAL-6-2-2-V3":
        raise ValueError("subset does not use the frozen BrainWM V3 split")
    if verify_hashes:
        preprocessing = []
        panel = manifest.get("panel", {}).get("file")
        if panel:
            preprocessing.append(panel)
        normalization = manifest.get("normalization", {})
        if normalization and normalization.get("fit_split") != "train":
            raise ValueError("normalization must be fitted on train only")
        for name in ("mean", "std"):
            if normalization.get(f"{name}_file"):
                preprocessing.append({"path": normalization[f"{name}_file"],
                                      "sha256": normalization[f"{name}_sha256"]})
        for info in preprocessing:
            path = location / info["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"preprocessing checksum mismatch: {path}")
    unknown = sorted(set(data_splits) - {"train", "validation", "test"})
    if unknown:
        raise ValueError(f"unknown subset splits: {unknown}")
    for split in data_splits:
        receipt = manifest.get("splits", {}).get(split)
        split_dir = location / split
        if not isinstance(receipt, dict) or not split_dir.is_dir():
            raise ValueError(f"incomplete nanoBrain subset split: {split}")
        expected = int(receipt.get("rows", 0))
        feature_path = split_dir / "features.npy"
        if not feature_path.is_file() or len(np.load(feature_path, mmap_mode="r")) != expected:
            raise ValueError(f"feature row count mismatch in {split}")
        if verify_hashes:
            for info in receipt.get("files", {}).values():
                path = location / str(info["path"])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != info.get("sha256"):
                    raise ValueError(f"checksum mismatch: {path}")
    return manifest


def _load_optional_tensor(split_dir: Path, name: str) -> torch.Tensor | None:
    path = split_dir / f"{name}.npy"
    return _readonly_tensor(path) if path.is_file() else None


def load_v3_subset(
    root: str | Path | None = None,
    *,
    verify: bool = False,
    include_test: bool = True,
) -> DatasetSplits:
    """Load the fixed nanoBrain V3 subset without opening H5AD files.

    Arrays stay backed by read-only NumPy memmaps until a batch is indexed.
    ``verify=True`` checks all checksums and is intended for startup/audit, not
    every training epoch.
    """
    location = _subset_root(root)
    loaded_splits = ("train", "validation", "test") if include_test else ("train", "validation")
    manifest = verify_subset_manifest(
        location,
        verify_hashes=verify,
        data_splits=loaded_splits,
    )

    def load_split(split: str) -> NanoDataset:
        directory = location / split
        features = _readonly_tensor(directory / "features.npy", dtype=np.dtype("float32"))
        feature_observed = _load_optional_tensor(directory, "feature_observed")
        labels = {
            "annotation": _readonly_tensor(directory / "label_annotation.npy").long(),
            "niche": _readonly_tensor(directory / "label_niche.npy").long(),
            "integration": _readonly_tensor(directory / "label_integration.npy").long(),
            "development": _readonly_tensor(directory / "label_development.npy", dtype=np.dtype("float32")),
        }
        masks = {
            name: _readonly_tensor(directory / f"mask_{name}.npy").bool()
            for name in ("annotation", "niche", "integration", "development")
        }
        development_condition = _load_optional_tensor(directory, "development_condition")
        development_condition_observed = _load_optional_tensor(
            directory, "development_condition_observed"
        )
        # ``development_time`` is retained as a compatibility alias for the
        # flow module.  It is the independent condition, never the age target.
        development_time = (
            development_condition[:, 0]
            if development_condition is not None and development_condition.ndim == 2
            else development_condition
        )
        metadata: dict[str, np.ndarray] = {}
        for name in _SUBSET_METADATA:
            path = directory / f"{name}.npy"
            if path.is_file():
                metadata[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        return NanoDataset(
            features=features,
            labels=labels,
            masks=masks,
            feature_observed=feature_observed,
            coordinates=_load_optional_tensor(directory, "coordinates"),
            neighbors=_load_optional_tensor(directory, "neighbors"),
            neighbor_mask=_load_optional_tensor(directory, "neighbor_mask"),
            development_time=development_time,
            development_condition=development_condition,
            development_condition_observed=development_condition_observed,
            completion_target=_load_optional_tensor(directory, "completion_target"),
            completion_mask=_load_optional_tensor(directory, "completion_mask"),
            metadata=metadata,
        )

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test") if include_test else None
    from .development import DevelopmentPairDataset
    splits = DatasetSplits(
        train,
        validation,
        test,
        development_train=DevelopmentPairDataset.load(location, "train", train),
        development_validation=DevelopmentPairDataset.load(
            location, "validation", validation
        ),
        development_test=(
            DevelopmentPairDataset.load(location, "test", test)
            if include_test and test is not None
            else None
        ),
    )
    # Keep the manifest available to receipt builders without making it part
    # of the tensor contract.
    for dataset in (splits.train, splits.validation, splits.test):
        if dataset is None:
            continue
        dataset.metadata = dict(dataset.metadata or {})
        dataset.metadata["subset_manifest"] = manifest
    return splits


def infer_label_cardinalities(splits: DatasetSplits) -> dict[str, int]:
    """Infer output widths from train/validation labels only.

    Test labels are intentionally excluded because the V3 test annotation
    vocabulary is sealed.
    """
    result: dict[str, int] = {}
    for name in ("annotation", "niche", "integration"):
        maximum = -1
        for dataset in (splits.train, splits.validation):
            values = dataset.labels[name]
            mask = dataset.masks.get(name)
            valid = values.ge(0) & (mask if mask is not None else torch.ones_like(values, dtype=torch.bool))
            if bool(valid.any()):
                maximum = max(maximum, int(values[valid].max().item()))
        result[name] = maximum + 1
    return result


__all__ = [
    "DatasetSplits",
    "NanoDataset",
    "infer_label_cardinalities",
    "load_dataset",
    "load_v3_subset",
    "make_synthetic_splits",
    "save_dataset",
    "verify_subset_manifest",
]
