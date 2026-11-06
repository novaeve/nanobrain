"""Exercise the portable multipart dataset entrypoint with tiny local archives."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


def setup_release(tmp_path, unsafe=False):
    script = Path(__file__).resolve().parents[1] / "scripts/download_data.py"
    spec = importlib.util.spec_from_file_location("cajal_data_download", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path / "variant"
    (module.ROOT / "reference").mkdir(parents=True)
    cache = tmp_path / "archives"
    cache.mkdir()
    files = {"unified_1m/manifest.json": b'{"rows": 3}', "unified_1m/values.npy": b"synthetic-counts"}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, value in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            archive.addfile(entry, io.BytesIO(value))
        if unsafe:
            entry = tarfile.TarInfo("../escape.txt")
            entry.size = 4
            archive.addfile(entry, io.BytesIO(b"fail"))
    compressed = buffer.getvalue()
    manifest = {
        "release_tag": "test-only",
        "archive_bytes": len(compressed),
        "archive_sha256": hashlib.sha256(compressed).hexdigest(),
        "datasets": {
            "unified_1m": {"rows": 3, "manifest_sha256": hashlib.sha256(files["unified_1m/manifest.json"]).hexdigest()}
        },
        "files": {name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in files.items()},
        "parts": [],
    }
    for index, start in enumerate(range(0, len(compressed), 64)):
        value = compressed[start : start + 64]
        name = f"data.tar.gz.part{index:03d}"
        (cache / name).write_bytes(value)
        manifest["parts"].append({"name": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()})
    (module.ROOT / "reference/data-release.json").write_text(json.dumps(manifest), encoding="utf-8")
    return module, cache, tmp_path / "data"


def run(module, cache, destination, monkeypatch, *options):
    monkeypatch.setattr(sys, "argv", ["download_data", "--archive-dir", str(cache), "--destination", str(destination), *options])
    module.main()


def test_cached_multipart_archive_extracts_and_verifies(tmp_path, monkeypatch):
    module, cache, destination = setup_release(tmp_path)
    run(module, cache, destination, monkeypatch)
    assert (destination / "unified_1m/values.npy").read_bytes() == b"synthetic-counts"
    run(module, cache, destination, monkeypatch, "--verify-only")


def test_changed_extracted_file_fails_verification(tmp_path, monkeypatch):
    module, cache, destination = setup_release(tmp_path)
    run(module, cache, destination, monkeypatch)
    (destination / "unified_1m/values.npy").write_bytes(b"changed-counts")
    with pytest.raises(ValueError, match="file checksum mismatch"):
        run(module, cache, destination, monkeypatch, "--verify-only")


def test_undeclared_archive_path_is_rejected(tmp_path, monkeypatch):
    module, cache, destination = setup_release(tmp_path, unsafe=True)
    with pytest.raises(ValueError, match="Unexpected archive entry"):
        run(module, cache, destination, monkeypatch)
    assert not (tmp_path / "escape.txt").exists()


def test_corrupt_download_does_not_replace_cached_part(tmp_path, monkeypatch):
    module, cache, destination = setup_release(tmp_path)
    first = sorted(cache.iterdir())[0]
    first.write_bytes(b"corrupted")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"also-corrupted"))
    with pytest.raises(ValueError, match="part checksum mismatch"):
        run(module, cache, destination, monkeypatch)
    assert first.read_bytes() == b"corrupted"
