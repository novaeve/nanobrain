"""Fetch and validate the frozen 100k nanoBrain dataset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from .data import verify_subset_manifest
from .environment import read_env_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "NANOBRAIN_DATA_DIR",
        ROOT / "assets" / "nanobrain_v3_100k_metadata_v2",
    )
)
EXPECTED_ROWS = {"train": 60_000, "validation": 20_000, "test": 20_000}
EXPECTED_SPLIT_ID = "BRAINWM-HUMAN-GLOBAL-6-2-2-V3"


def _validate_release(root: Path) -> dict:
    manifest = verify_subset_manifest(root, verify_hashes=True)
    rows = {name: int(manifest["splits"][name]["rows"]) for name in EXPECTED_ROWS}
    if rows != EXPECTED_ROWS:
        raise ValueError(f"expected the frozen 100k split {EXPECTED_ROWS}, found {rows}")
    if manifest.get("split_id") != EXPECTED_SPLIT_ID:
        raise ValueError("dataset split contract does not match the frozen nanoBrain V3 split")
    if manifest.get("identity_repair", {}).get("version") != 2:
        raise ValueError("expected the metadata-v2 dataset with repaired row identity")
    if manifest.get("normalization", {}).get("fit_split") != "train":
        raise ValueError("dataset normalization must be fit on train only")
    return manifest


def _safe_relative(name: str) -> Path:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if path.parts[0].endswith(":"):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return Path(*path.parts)


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            relative = _safe_relative(item.filename)
            target = destination / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        for item in bundle:
            relative = _safe_relative(item.name)
            if item.isdir():
                (destination / relative).mkdir(parents=True, exist_ok=True)
                continue
            if not item.isfile():
                raise ValueError(f"archive contains a non-regular file: {item.name!r}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(item)
            if source is None:
                raise ValueError(f"could not read archive member: {item.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _find_dataset_root(extracted: Path) -> Path:
    if (extracted / "manifest.json").is_file():
        return extracted
    matches = [path.parent for path in extracted.rglob("manifest.json")]
    if len(matches) != 1:
        raise ValueError("archive must contain exactly one dataset root with manifest.json")
    return matches[0]


def _download(url: str, target: Path) -> None:
    with urlopen(url, timeout=60) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(
    *,
    destination: Path = DEFAULT_DATA_DIR,
    url: str | None = None,
    archive: Path | None = None,
    expected_sha256: str | None = None,
) -> dict:
    destination = destination.expanduser().resolve()
    placeholder = False
    if destination.is_dir() and (destination / "manifest.json").is_file():
        return _validate_release(destination)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"dataset destination is not a directory: {destination}")
        unexpected = {path.name for path in destination.iterdir()} - {"README.md", ".gitkeep"}
        if unexpected:
            raise FileExistsError(
                f"dataset destination contains unrecognized files: {destination}; "
                "use an empty destination or the frozen complete dataset"
            )
        placeholder = True
    if archive is None and not url:
        raise FileNotFoundError(
            f"dataset is not present at {destination}; set NANOBRAIN_DATA_URL or pass --url"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nanobrain-data-", dir=destination.parent) as temp_name:
        temp = Path(temp_name)
        bundle_path = archive.expanduser().resolve() if archive else temp / "dataset.bundle"
        if archive is None:
            _download(str(url), bundle_path)
        if expected_sha256 and _sha256(bundle_path).lower() != expected_sha256.lower():
            raise ValueError(
                "downloaded dataset bundle SHA-256 does not match NANOBRAIN_DATA_SHA256"
            )

        extracted = temp / "extracted"
        extracted.mkdir()
        if zipfile.is_zipfile(bundle_path):
            _extract_zip(bundle_path, extracted)
        elif tarfile.is_tarfile(bundle_path):
            _extract_tar(bundle_path, extracted)
        else:
            raise ValueError("dataset bundle must be a ZIP or TAR archive")
        root = _find_dataset_root(extracted)
        manifest = _validate_release(root)
        if placeholder:
            backup = temp / "placeholder"
            destination.rename(backup)
            try:
                shutil.move(str(root), str(destination))
            except BaseException:
                if not destination.exists():
                    backup.rename(destination)
                raise
        else:
            if destination.exists():
                raise FileExistsError(
                    f"dataset destination appeared during download: {destination}"
                )
            shutil.move(str(root), str(destination))
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="download or verify the frozen 100k split")
    local_env = read_env_file(ROOT / ".env")
    prepare.add_argument(
        "--url", default=os.environ.get("NANOBRAIN_DATA_URL") or local_env.get("NANOBRAIN_DATA_URL")
    )
    prepare.add_argument("--archive", type=Path, help="local ZIP or TAR bundle")
    prepare.add_argument(
        "--destination",
        type=Path,
        default=Path(os.environ.get("NANOBRAIN_DATA_DIR", DEFAULT_DATA_DIR)),
    )
    prepare.add_argument(
        "--sha256",
        default=os.environ.get("NANOBRAIN_DATA_SHA256") or local_env.get("NANOBRAIN_DATA_SHA256"),
    )
    args = parser.parse_args()
    try:
        manifest = prepare_dataset(
            destination=args.destination,
            url=args.url,
            archive=args.archive,
            expected_sha256=args.sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "data_dir": str(args.destination.expanduser().resolve()),
                "rows": {name: int(manifest["splits"][name]["rows"]) for name in EXPECTED_ROWS},
                "total_rows": sum(EXPECTED_ROWS.values()),
                "split_id": manifest["split_id"],
                "metadata_version": manifest["identity_repair"]["version"],
                "normalization_fit_split": manifest["normalization"]["fit_split"],
                "checksums_verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
