"""Download, checksum, and extract the published Cajal dataset using Python's standard library."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(destination, manifest):
    for name, record in manifest["datasets"].items():
        folder = destination / name
        if sha256(folder / "manifest.json") != record["manifest_sha256"]:
            raise ValueError(f"Dataset manifest checksum mismatch: {name}")
        actual = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if actual["rows"] != record["rows"]:
            raise ValueError(f"Dataset row count mismatch: {name}")
    for relative, expected in manifest["files"].items():
        path = destination / relative
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Dataset file checksum mismatch: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=REPOSITORY / "assets/cajal")
    parser.add_argument("--archive-dir", type=Path, default=REPOSITORY / "assets/cajal-release")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reference/data-release.json").read_text(encoding="utf-8"))
    if args.verify_only:
        verify_dataset(args.destination, manifest)
        print("All dataset manifests, row counts, and file checksums verified.")
        return

    args.archive_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/novaeve/nanobrain/releases/download/{manifest['release_tag']}"
    for part in manifest["parts"]:
        path = args.archive_dir / part["name"]
        if not path.exists() or path.stat().st_size != part["bytes"] or sha256(path) != part["sha256"]:
            temporary = path.with_name(path.name + ".download")
            print(f"Downloading {part['name']}", flush=True)
            request = urllib.request.Request(f"{url}/{part['name']}", headers={"User-Agent": "nanoBrain-data"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if temporary.stat().st_size != part["bytes"] or sha256(temporary) != part["sha256"]:
                raise ValueError(f"Archive part checksum mismatch: {part['name']}")
            temporary.replace(path)

    # A temporary combined archive keeps extraction compatible with standard tar readers.
    with tempfile.TemporaryFile(dir=args.archive_dir) as combined:
        for part in manifest["parts"]:
            with (args.archive_dir / part["name"]).open("rb") as source:
                shutil.copyfileobj(source, combined, length=8 * 1024 * 1024)
        if combined.tell() != manifest["archive_bytes"]:
            raise ValueError("Combined archive size mismatch")
        combined.seek(0)
        digest = hashlib.file_digest(combined, "sha256").hexdigest()
        if digest != manifest["archive_sha256"]:
            raise ValueError("Combined archive checksum mismatch")
        combined.seek(0)
        args.destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=combined, mode="r:gz") as archive:
            allowed = set(manifest["files"])
            for entry in archive:
                if entry.isdir() and entry.name.strip("/") in manifest["datasets"]:
                    continue
                if not entry.isfile() or entry.name not in allowed:
                    raise ValueError(f"Unexpected archive entry: {entry.name}")
                archive.extract(entry, args.destination, filter="data")
    verify_dataset(args.destination, manifest)
    print(f"Dataset ready at {args.destination.resolve()}")


if __name__ == "__main__":
    main()
