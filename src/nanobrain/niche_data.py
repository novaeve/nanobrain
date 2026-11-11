"""Build neighborhood composition targets from complete, verified spatial units.

Only metadata is read from the full corpus. Training centers, features and the
global split stay unchanged. No anatomy labels enter these targets.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

SCALES = (10, 20, 50, 100)
MIN_KNOWN_FRACTION = 0.8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_code(value):
    from brainwm_data.task_labels import _text
    value = _text(value)
    return (-1 if value is None else
            int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8, person=b"brainwm3").digest(), "little") & ((1 << 63) - 1))


def materialize_evaluation_neighbor_labels(metadata, cache, contract):
    """Prepare evaluation targets in this private cache, never in shared shards.

    Legacy test sidecars contain -1 sentinels. Raw source annotations are used
    only to construct the held-out target file; training never loads that file.
    """
    import h5py
    import pandas as pd
    from brainwm_data.task_labels import canonicalize_annotation_frame
    lookup = pd.read_parquet(Path(contract) / "source_lookup.parquet").set_index("source_id")
    receipts = []
    for record in metadata["sections"]:
        if record["split"] != "test":
            continue
        file = Path(cache) / record["path"]
        with np.load(file) as stored:
            arrays = {key: stored[key] for key in stored.files}
        if np.any(arrays["labels"] >= 0):
            continue
        original = Path(lookup.loc[record["source_id"], "local_path"])
        sample_id = record["sample_key"].split("::", 1)[1]
        with h5py.File(original) as source:
            obs = source["obs"]
            item = obs["sample_id"]
            if isinstance(item, h5py.Group):
                categories = item["categories"][:].astype(str)
                wanted = np.flatnonzero(categories == sample_id)
                selected = np.flatnonzero(np.isin(item["codes"][:], wanted))
            else:
                selected = np.flatnonzero(item[:].astype(str) == sample_id)
            xy = source["obsm"]["spatial"][:, :2][selected]
            if xy.shape != arrays["coordinates"].shape or not np.allclose(xy, arrays["coordinates"], atol=1e-5):
                raise ValueError("raw held-out annotation rows do not match full-section coordinate order")
            columns = {}
            for field in ("cl_id", "cl_term", "original_cell_type", "original_subclass"):
                item = obs[field]
                if isinstance(item, h5py.Group):
                    categories = item["categories"][:].astype(str).astype(object)
                    codes = item["codes"][:][selected]
                    values = categories[codes.clip(0)].copy()
                    values[codes < 0] = None
                else:
                    values = item[:][selected].astype(str)
                columns[field] = values
        labels = canonicalize_annotation_frame(pd.DataFrame(columns))["annotation_label"]
        mapping = {str(value): label_code(value) for value in labels.dropna().unique()}
        encoded = labels.map(mapping).fillna(-1).to_numpy(np.int64)
        arrays["labels"] = encoded
        np.savez(file, **arrays)
        record["known_fraction"] = float((encoded >= 0).mean())
        receipts.append(dict(source_file=str(original), sample_key=record["sample_key"],
            coordinate_rows_matched=len(encoded), label_sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),
            purpose="held-out target materialization only; excluded from train/validation fitting"))
    if receipts:
        metadata["heldout_target_materialization"] = receipts
        (Path(cache) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def neighborhood_targets(coordinates, labels, row_refs, centers, center_refs, radii,
                         classes, *, min_known_fraction=MIN_KNOWN_FRACTION):
    """Count full-section neighbors, excluding the center by identity, not position."""
    tree = cKDTree(coordinates)
    shape = (len(centers), len(radii))
    composition = np.zeros((*shape, classes + 1), dtype=np.float32)
    density = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    for start in range(0, len(centers), 256):
        nearby = tree.query_ball_point(centers[start:start + 256], max(radii))
        for offset, neighbors in enumerate(nearby):
            i = start + offset
            take = np.asarray(neighbors, dtype=np.int64)
            take = take[row_refs[take] != center_refs[i]]
            distances = np.linalg.norm(coordinates[take] - centers[i], axis=1)
            for s, radius in enumerate(radii):
                types = labels[take[distances <= radius]]
                n = len(types)
                density[i, s] = n
                if not n:
                    continue
                known = (types >= 0) & (types < classes)
                counts = np.bincount(np.where(known, types, classes), minlength=classes + 1)
                composition[i, s] = counts / n
                valid[i, s] = float(known.mean()) >= min_known_fraction
    return composition, density, valid


def collect_full_sections(parent, source, cache, geometry_proof=None):
    """Cache only verified units represented by selected centers, split by split."""
    import pandas as pd
    parent, source, cache = map(Path, (parent, source, cache))
    fingerprint = {"parent": sha(parent / "manifest.json"),
                   "source": {s: sha(source / f"{s}-manifest.json") for s in ("train", "validation", "test")}}
    extra = set()
    if geometry_proof is not None:
        fingerprint["geometry_extension"] = sha(geometry_proof)
        extra = set(json.loads(Path(geometry_proof).read_text())["verified_sample_keys"])
    done = cache / "metadata.json"
    if done.exists():
        metadata = json.loads(done.read_text())
        if metadata["fingerprint"] != fingerprint:
            raise ValueError("full-section cache fingerprint mismatch")
        return metadata
    cache.mkdir(parents=True, exist_ok=True)
    sections = []
    for split in ("train", "validation", "test"):
        identity = pd.read_parquet(parent / split / "row_identity.parquet")
        samples = np.load(parent / split / "sample.npy")
        allowed = {}
        for i, record in enumerate(identity.itertuples()):
            if record.section_verified or record.sample_key in extra:
                allowed[int(samples[i])] = dict(source_id=int(record.source_id), sample_key=record.sample_key)
        parts = defaultdict(lambda: [[], [], []])
        offset = 0
        files = sorted(source.glob(f"{split}-shard-*-row_ptr.npy"))
        if not files:
            raise FileNotFoundError(f"no full metadata shards for {split}")
        for number, file in enumerate(files):
            stem = str(file).removesuffix("-row_ptr.npy")
            sample = np.load(stem + "-evaluation_sample.npy", mmap_mode="r")
            n = len(sample)
            selected = np.isin(sample, list(allowed))
            if selected.any():
                coordinates = np.load(stem + "-spatial.npy", mmap_mode="r")
                labels = np.load(stem + "-evaluation_cell_type.npy", mmap_mode="r")
                sources = np.load(stem + "-row_source_id.npy", mmap_mode="r")
                selected &= np.isfinite(coordinates[:, :2]).all(axis=1)
                for key in np.unique(sample[selected]):
                    take = np.flatnonzero(selected & (sample == key))
                    if not np.all(sources[take] == allowed[int(key)]["source_id"]):
                        raise ValueError("spatial sample/source identity mismatch")
                    block = parts[int(key)]
                    block[0].append(np.asarray(coordinates[take, :2], dtype=np.float32))
                    block[1].append(np.asarray(labels[take], dtype=np.int64))
                    block[2].append(take + offset)
            offset += n
            if number % 100 == 0:
                print(json.dumps(dict(stage="full_metadata", split=split, shard=number, total=len(files))), flush=True)
        expected = json.loads((source / f"{split}-manifest.json").read_text())["rows"]
        if offset != expected:
            raise ValueError("full corpus metadata row count mismatch")
        for key, blocks in parts.items():
            arrays = dict(zip(("coordinates", "labels", "row_refs"), (np.concatenate(v) for v in blocks)))
            file = cache / f"{split}-{key}.npz"
            np.savez(file, **arrays)
            sections.append(dict(split=split, sample_id=key, path=file.name, rows=len(arrays["labels"]),
                                 known_fraction=float((arrays["labels"] >= 0).mean()), **allowed[key]))
        print(json.dumps(dict(stage="metadata_done", split=split, sections=len(parts), full_rows=offset)), flush=True)
    metadata = dict(fingerprint=fingerprint, sections=sections)
    done.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_niche_dataset(parent, source, output, cache, geometry_proof=None, evaluation_label_contract=None):
    import pandas as pd
    parent, source, output, cache = map(Path, (parent, source, output, cache))
    if output.exists():
        raise FileExistsError("new niche dataset output must not exist")
    metadata = collect_full_sections(parent, source, cache, geometry_proof)
    if evaluation_label_contract:
        metadata = materialize_evaluation_neighbor_labels(metadata, cache, evaluation_label_contract)
    calibration = defaultdict(list)
    vocabulary = set()
    for record in metadata["sections"]:
        if record["split"] != "train" or record["known_fraction"] < MIN_KNOWN_FRACTION:
            continue
        with np.load(cache / record["path"]) as data:
            x, labels = data["coordinates"], data["labels"]
            vocabulary.update(int(v) for v in np.unique(labels) if v >= 0)
            if len(x) <= max(SCALES):
                continue
            rng = np.random.default_rng(20260915)
            selected = rng.choice(len(x), min(len(x), 256), replace=False)
            distances, _ = cKDTree(x).query(x[selected], k=[k + 1 for k in SCALES], workers=1)
            calibration[record["source_id"]].append(distances)
    if len(vocabulary) < 2 or not calibration:
        raise ValueError("no complete training sections with usable cell-type annotations")
    radii = {key: np.median(np.concatenate(blocks), axis=0).tolist() for key, blocks in calibration.items()}
    if any(not np.isfinite(v).all() or min(v) <= 0 for v in radii.values()):
        raise ValueError("invalid train-fitted spatial radii")
    vocabulary = np.asarray(sorted(vocabulary), dtype=np.int64)
    classes = len(vocabulary)
    shutil.copytree(parent, output)
    manifest = json.loads((parent / "manifest.json").read_text())
    manifest["niche_policy"] = "full-section cell-type proportions; no anatomy targets; unknown neighbor bin retained"
    report = dict(parent_manifest_sha256=sha(parent / "manifest.json"), same_rows_and_splits=True,
                  radii_fit_split="train", source_metadata=metadata["fingerprint"], full_sections=metadata["sections"],
                  class_count=classes + 1, unknown_class_index=classes, scales=list(SCALES),
                  vocabulary_codes=vocabulary.tolist(),
                  radii_by_source=radii, min_known_fraction=MIN_KNOWN_FRACTION, splits={})
    report["heldout_target_materialization"] = metadata.get("heldout_target_materialization", [])
    if geometry_proof:
        shutil.copy2(geometry_proof, output / "niche_geometry_extension.json")
    vocab = json.loads((source / "joint_metadata_vocab.json").read_text())["cell_type"]
    names = {label_code(name): name for name in vocab}
    (output / "niche_vocabulary.json").write_text(json.dumps([
        dict(index=i, source_code=int(code), label=names.get(int(code), str(code))) for i, code in enumerate(vocabulary)
    ] + [dict(index=classes, source_code=-1, label="unknown_or_unseen_in_training")], indent=2) + "\n")
    for split in ("train", "validation", "test"):
        directory = output / split
        identity = pd.read_parquet(directory / "row_identity.parquet")
        samples = np.load(directory / "sample.npy")
        centers = np.load(directory / "coordinates.npy")[:, :2]
        refs = np.load(directory / "source_row_ref.npy")
        n = len(refs)
        shutil.copy2(directory / "label_niche.npy", directory / "label_niche_anatomy_proxy.npy")
        target = np.lib.format.open_memmap(directory / "label_niche.npy", mode="w+", dtype=np.float32,
                                          shape=(n, len(SCALES), classes + 1))
        target[:] = 0
        density = np.zeros((n, len(SCALES)), dtype=np.float32)
        valid = np.zeros((n, len(SCALES)), dtype=bool)
        density_valid = np.zeros_like(valid)
        for record in metadata["sections"]:
            if record["split"] != split or record["source_id"] not in radii:
                continue
            selected = np.flatnonzero((samples == record["sample_id"]) & np.isfinite(centers).all(axis=1))
            if not len(selected):
                continue
            with np.load(cache / record["path"]) as data:
                # Verify that every sampled center belongs to this complete unit.
                indices = np.searchsorted(data["row_refs"], refs[selected])
                if np.any(indices >= len(data["row_refs"])) or not np.array_equal(data["row_refs"][indices], refs[selected]):
                    raise ValueError("center/full-section identity mismatch")
                if not np.allclose(data["coordinates"][indices], centers[selected], atol=1e-5):
                    raise ValueError("center/full-section coordinates differ")
                raw = data["labels"]
                encoded = np.searchsorted(vocabulary, raw)
                known = (encoded < classes) & (vocabulary[encoded.clip(0, classes - 1)] == raw)
                encoded = np.where(known, encoded, -1)
                p, d, m = neighborhood_targets(data["coordinates"], encoded, data["row_refs"],
                    centers[selected], refs[selected], radii[record["source_id"]], classes)
            target[selected], density[selected], valid[selected] = p, d, m
            density_valid[selected] = True
        target.flush()
        del target
        for name, array in {"label_niche_density": density, "mask_niche": valid,
                            "mask_niche_density": density_valid}.items():
            np.save(directory / f"{name}.npy", array)
        for name in ("label_niche", "label_niche_density", "mask_niche", "mask_niche_density", "label_niche_anatomy_proxy"):
            file = directory / f"{name}.npy"
            array = np.load(file, mmap_mode="r")
            manifest["splits"][split]["files"][name] = dict(path=str(file.relative_to(output)), sha256=sha(file),
                                                          dtype=str(array.dtype), shape=list(array.shape))
        report["splits"][split] = dict(rows=n, valid_centers=int(valid.any(1).sum()),
                                      valid_per_scale=valid.sum(0).tolist(), density_centers=int(density_valid.any(1).sum()))
        manifest["splits"][split].setdefault("task_rows", {})["spatial_niche"] = int(valid.any(1).sum())
        # No fallback to proxy labels is permitted for an unavailable split.
        print(json.dumps(dict(stage="targets", split=split, **report["splits"][split])), flush=True)
        if not np.all(valid.any(0)):
            raise ValueError(f"{split} lacks valid composition targets at one or more fixed radii")
    manifest["niche_contract"] = {k: report[k] for k in ("class_count", "unknown_class_index", "scales", "radii_by_source", "min_known_fraction")}
    manifest["niche_contract"].update(mode="composition", input="expression_only", receipt="niche_target_receipt.json")
    (output / "niche_target_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest["niche_contract"]["artifacts"] = [dict(path=name, sha256=sha(output / name))
        for name in ("niche_target_receipt.json", "niche_vocabulary.json", "niche_geometry_extension.json")
        if (output / name).exists()]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report
