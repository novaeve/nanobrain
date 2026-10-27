#!/usr/bin/env python3
"""Open the frozen nanoBrain test annotation sidecar exactly once.

The compact test split intentionally contains sealed placeholder annotation
targets.  This evaluator-only transition verifies the pre-test protocol hash,
replays the authoritative V3 row order through ``GlobalTaskLabelIndex``, maps
labels through the train-only vocabulary, and records every changed byte.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainwm_data.task_labels import GlobalTaskLabelIndex  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _verify_post_freeze_test_pairs(
    path: Path, expected_sha256: str
) -> dict[str, Any] | None:
    """Accept only the audited test population-pair materialization delta."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    test_split = manifest.get("splits", {}).get("test", {})
    files = test_split.get("files", {})
    pair_keys = (
        "development_pair_ptr",
        "development_source_index",
        "development_target_index",
        "development_pairs",
    )
    if not all(key in files for key in pair_keys):
        return None
    population = test_split.get("population_development")
    if not isinstance(population, dict):
        return None

    reconstructed = json.loads(json.dumps(manifest))
    reconstructed_test = reconstructed["splits"]["test"]
    for key in pair_keys:
        reconstructed_test["files"].pop(key)
    reconstructed_test.pop("population_development")
    reconstructed_bytes = (
        json.dumps(reconstructed, indent=2, sort_keys=True) + "\n"
    ).encode()
    reconstructed_sha256 = hashlib.sha256(reconstructed_bytes).hexdigest()
    if reconstructed_sha256 != expected_sha256:
        return None

    verified_files: list[dict[str, Any]] = []
    for key in pair_keys:
        metadata = files[key]
        relative = Path(str(metadata["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "test":
            raise ValueError(f"invalid post-freeze test pair path: {relative}")
        sidecar = path.parent / relative
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        actual = _sha256(sidecar)
        if actual != metadata["sha256"]:
            raise ValueError(f"post-freeze test pair hash mismatch: {relative}")
        verified_files.append(
            {
                "key": key,
                "path": str(sidecar.relative_to(ROOT)),
                "bytes": sidecar.stat().st_size,
                "sha256": actual,
            }
        )
    pair_receipt = json.loads(
        (path.parent / str(files["development_pairs"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if (
        pair_receipt.get("format") != "nanobrain_population_pairs_v1"
        or pair_receipt.get("split_id") != manifest.get("split_id")
        or pair_receipt.get("split") != "test"
        or int(pair_receipt.get("groups", -1)) != int(population.get("groups", -2))
        or int(pair_receipt.get("pairs", -1)) != int(population.get("pairs", -2))
    ):
        raise ValueError("post-freeze test population-pair receipt is inconsistent")
    return {
        "kind": "test_population_pair_materialization",
        "reconstructed_frozen_sha256": reconstructed_sha256,
        "groups": int(population["groups"]),
        "pairs": int(population["pairs"]),
        "files": verified_files,
    }


def _verify_freeze(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    if receipt.get("format") != "nanobrain_protocol_freeze_receipt_v1":
        raise ValueError("unexpected protocol freeze receipt")
    if receipt.get("test_opened") is not False:
        raise ValueError("protocol freeze does not attest a sealed test split")
    verified: list[dict[str, Any]] = []
    for artifact in receipt.get("artifacts", []):
        path = ROOT / str(artifact["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        allowed_delta = None
        if actual != artifact["sha256"]:
            if artifact["path"] == "assets/nanobrain_v3_100k/manifest.json":
                allowed_delta = _verify_post_freeze_test_pairs(
                    path, str(artifact["sha256"])
                )
            if allowed_delta is None:
                raise ValueError(f"post-freeze artifact changed: {artifact['path']}")
        verified.append(
            {
                "path": artifact["path"],
                "bytes": int(path.stat().st_size),
                "frozen_sha256": artifact["sha256"],
                "current_sha256": actual,
                "verification": (
                    "exact_frozen_hash" if allowed_delta is None else "allowed_delta"
                ),
                "allowed_delta": allowed_delta,
            }
        )
    if len(verified) != int(receipt.get("artifact_count", -1)):
        raise ValueError("protocol freeze artifact count mismatch")
    return verified


def _encode_selected_rows(
    frames: Iterable[pd.DataFrame],
    refs: np.ndarray,
    expected_sources: np.ndarray,
    vocabulary: dict[str, int],
) -> tuple[np.ndarray, int]:
    """Encode selected split-local offsets from the authoritative V3 stream."""

    refs = np.asarray(refs, dtype=np.int64)
    expected_sources = np.asarray(expected_sources, dtype=np.int64)
    if refs.ndim != 1 or len(refs) != len(expected_sources):
        raise ValueError("row references and source IDs must be aligned vectors")
    if len(refs) and (np.any(refs < 0) or np.any(np.diff(refs) <= 0)):
        raise ValueError("source_row_ref must be strictly increasing and nonnegative")

    labels = np.full(len(refs), -1, dtype=np.int32)
    found = np.zeros(len(refs), dtype=bool)
    offset = 0
    for frame in frames:
        for _, part in frame.groupby("source_id", sort=False):
            part = part.sort_values("row_index", kind="stable").reset_index(drop=True)
            end = offset + len(part)
            left = int(np.searchsorted(refs, offset, side="left"))
            right = int(np.searchsorted(refs, end, side="left"))
            if right > left:
                local = refs[left:right] - offset
                actual_sources = pd.to_numeric(
                    part.iloc[local]["source_id"], errors="raise"
                ).to_numpy(np.int64)
                if not np.array_equal(actual_sources, expected_sources[left:right]):
                    raise ValueError(
                        f"V3 row identity mismatch for offsets {refs[left]}..{refs[right - 1]}"
                    )
                values = part.iloc[local]["annotation_label"]
                labels[left:right] = np.asarray(
                    [
                        vocabulary.get(str(value), -1)
                        if not pd.isna(value)
                        else -1
                        for value in values
                    ],
                    dtype=np.int32,
                )
                found[left:right] = True
            offset = end
    if not found.all():
        missing = refs[~found]
        raise ValueError(f"authoritative stream missed {len(missing)} selected rows")
    return labels, offset


def _filter_to_compact_train_classes(
    labels: np.ndarray,
    eligible: np.ndarray,
    train_labels: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep only classes learnable from the compact benchmark train split."""

    labels = np.asarray(labels, dtype=np.int32).copy()
    eligible = np.asarray(eligible, dtype=bool)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_mask = np.asarray(train_mask, dtype=bool)
    seen = np.unique(train_labels[train_mask & (train_labels >= 0)])
    known_global = eligible & (labels >= 0)
    seen_mask = known_global & np.isin(labels, seen)
    compact_unseen = known_global & ~seen_mask
    unseen_classes, unseen_counts = np.unique(
        labels[compact_unseen], return_counts=True
    )
    labels[~seen_mask] = -1
    stats = {
        "compact_train_seen_classes": int(len(seen)),
        "known_train_rows": int(seen_mask.sum()),
        "compact_train_unseen_rows": int(compact_unseen.sum()),
        "compact_train_unseen_classes": {
            str(int(label)): int(count)
            for label, count in zip(unseen_classes, unseen_counts)
        },
        "unknown_or_out_of_global_train_vocabulary": int(
            np.sum(eligible & ~known_global)
        ),
    }
    return labels, seen_mask.astype(np.bool_), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "assets" / "nanobrain_v3_100k"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=ROOT / "contracts" / "healthy_global_benchmark_v3",
    )
    parser.add_argument(
        "--v3-shards",
        type=Path,
        default=Path(
            "/data/brainwm/BrainWM-Training-Benchmark/assets/v3_numpy_shards"
        ),
    )
    parser.add_argument(
        "--freeze-receipt",
        type=Path,
        default=(
            ROOT
            / "results/nanobrain/final-100k-20260916/protocol_freeze.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/nanobrain/final-100k-20260916/test_label_open_receipt.json"
        ),
    )
    parser.add_argument("--row-group-batch-size", type=int, default=1)
    args = parser.parse_args()

    data = args.data.resolve()
    output = args.output.resolve()
    intent = output.with_name("test_label_access_intent.json")
    snapshot = output.with_name("manifest.pre_test_open.json")
    manifest_path = data / "manifest.json"
    labels_path = data / "test" / "label_annotation.npy"
    mask_path = data / "test" / "mask_annotation.npy"

    if output.is_file():
        completed = json.loads(output.read_text(encoding="utf-8"))
        if completed.get("status") == "complete":
            for key, path in (("labels", labels_path), ("mask", mask_path)):
                if _sha256(path) != completed["outputs"][key]["sha256"]:
                    raise ValueError(f"opened test {key} no longer matches its receipt")
            print(json.dumps(completed, indent=2, sort_keys=True))
            return
        raise FileExistsError(f"incomplete test-open receipt already exists: {output}")

    freeze_path = args.freeze_receipt.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    verified = _verify_freeze(freeze)
    manifest_before_bytes = manifest_path.read_bytes()
    manifest_before_sha = hashlib.sha256(manifest_before_bytes).hexdigest()
    verified_manifest = next(
        (
            artifact
            for artifact in verified
            if artifact["path"] == str(manifest_path.relative_to(ROOT))
        ),
        None,
    )
    if verified_manifest is None or verified_manifest["current_sha256"] != manifest_before_sha:
        raise ValueError("data manifest did not pass the post-freeze audit")

    old_labels = np.load(labels_path, mmap_mode="r")
    old_mask = np.load(mask_path, mmap_mode="r")
    if np.any(old_labels != -1) or np.any(old_mask):
        raise ValueError("test annotation labels are not in their sealed placeholder state")
    before = {
        "manifest": {"sha256": manifest_before_sha, "bytes": len(manifest_before_bytes)},
        "labels": {"sha256": _sha256(labels_path), "bytes": labels_path.stat().st_size},
        "mask": {"sha256": _sha256(mask_path), "bytes": mask_path.stat().st_size},
    }
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(manifest_before_bytes)
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        intent,
        {
            "format": "nanobrain_test_label_access_intent_v1",
            "status": "opening",
            "protocol_id": freeze["protocol_id"],
            "started_at": started_at,
            "freeze_receipt": str(freeze_path.relative_to(ROOT)),
            "freeze_receipt_sha256": _sha256(freeze_path),
            "verified_artifact_count": len(verified),
            "manifest_before": before["manifest"],
            "allowed_mutations": [
                str(labels_path.relative_to(ROOT)),
                str(mask_path.relative_to(ROOT)),
                str(manifest_path.relative_to(ROOT)),
            ],
        },
    )

    refs = np.load(data / "test" / "source_row_ref.npy", mmap_mode="r")
    sources = np.load(data / "test" / "source_id.npy", mmap_mode="r")
    eligible = np.asarray(
        np.load(data / "test" / "annotation_eligible.npy", mmap_mode="r"),
        dtype=bool,
    )
    vocab_path = args.v3_shards.resolve() / "joint_metadata_vocab.json"
    vocabularies = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocabulary = {str(key): int(value) for key, value in vocabularies["cell_type"].items()}
    label_index = GlobalTaskLabelIndex(args.benchmark_dir.resolve())
    frames = label_index.iter_cell_labels(
        "test",
        task="integration",
        allow_sealed_test_labels=True,
        row_group_batch_size=args.row_group_batch_size,
    )
    labels, streamed_rows = _encode_selected_rows(frames, refs, sources, vocabulary)

    source_manifest = json.loads(
        (args.v3_shards.resolve() / "test-manifest.json").read_text(encoding="utf-8")
    )
    if streamed_rows != int(source_manifest["rows"]):
        raise ValueError(
            f"V3 stream length mismatch: {streamed_rows} != {source_manifest['rows']}"
        )
    train_labels = np.load(data / "train" / "label_annotation.npy", mmap_mode="r")
    train_mask = np.load(data / "train" / "mask_annotation.npy", mmap_mode="r")
    labels, mask, compact_stats = _filter_to_compact_train_classes(
        labels, eligible, train_labels, train_mask
    )
    _atomic_save(labels_path, labels)
    _atomic_save(mask_path, mask.astype(np.bool_))

    manifest = json.loads(manifest_before_bytes)
    split = manifest["splits"]["test"]
    for key, path, array in (
        ("label_annotation", labels_path, labels),
        ("mask_annotation", mask_path, mask),
    ):
        split["files"][key].update(
            {
                "path": str(path.relative_to(data)),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _sha256(path),
            }
        )
    split["task_rows"]["annotation"] = int(mask.sum())
    manifest["test_label_access"] = {
        "state": "opened_after_protocol_freeze",
        "protocol_id": freeze["protocol_id"],
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "mapping": "GlobalTaskLabelIndex canonical annotation -> train-only V3 cell_type vocabulary",
        "unknown_or_out_of_train_vocabulary": int(np.sum(eligible & (labels < 0))),
        **compact_stats,
    }
    _write_json(manifest_path, manifest)

    receipt = {
        "format": "nanobrain_test_label_open_receipt_v1",
        "status": "complete",
        "protocol_id": freeze["protocol_id"],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "freeze_receipt": str(freeze_path.relative_to(ROOT)),
        "freeze_receipt_sha256": _sha256(freeze_path),
        "verified_frozen_artifacts": verified,
        "source": {
            "api": "brainwm_data.task_labels.GlobalTaskLabelIndex.iter_cell_labels",
            "benchmark_dir": str(args.benchmark_dir.resolve()),
            "v3_manifest": str((args.v3_shards.resolve() / "test-manifest.json")),
            "v3_rows_streamed": int(streamed_rows),
            "vocabulary": str(vocab_path),
            "vocabulary_fit_split": "train",
            "vocabulary_size": len(vocabulary),
            "row_identity": ["source_row_ref", "source_id", "row_index"],
        },
        "before": before,
        "outputs": {
            "labels": {
                "path": str(labels_path.relative_to(ROOT)),
                "sha256": _sha256(labels_path),
                "bytes": labels_path.stat().st_size,
                "dtype": str(labels.dtype),
                "shape": list(labels.shape),
            },
            "mask": {
                "path": str(mask_path.relative_to(ROOT)),
                "sha256": _sha256(mask_path),
                "bytes": mask_path.stat().st_size,
                "dtype": str(mask.dtype),
                "shape": list(mask.shape),
            },
            "manifest": {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": _sha256(manifest_path),
                "bytes": manifest_path.stat().st_size,
            },
            "pre_open_manifest_snapshot": str(snapshot.relative_to(ROOT)),
        },
        "coverage": {
            "compact_test_rows": int(len(labels)),
            "annotation_eligible": int(eligible.sum()),
            "known_train_vocabulary": int(mask.sum()),
            "unknown_or_out_of_train_vocabulary": int(np.sum(eligible & (labels < 0))),
            "unlabeled_or_ineligible": int(np.sum(~eligible)),
            "known_classes": int(len(np.unique(labels[mask]))),
            **compact_stats,
        },
        "post_open_policy": "evaluation only; no model, hyperparameter, task, or metric changes",
    }
    _write_json(output, receipt)
    _write_json(intent, {**json.loads(intent.read_text()), "status": "complete"})
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
