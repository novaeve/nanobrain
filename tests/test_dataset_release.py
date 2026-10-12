from __future__ import annotations

import zipfile

import pytest

from nanobrain.dataset_release import EXPECTED_ROWS, _extract_zip, prepare_dataset


def test_frozen_release_rows_remain_100k() -> None:
    assert EXPECTED_ROWS == {"train": 60_000, "validation": 20_000, "test": 20_000}
    assert sum(EXPECTED_ROWS.values()) == 100_000


def test_zip_extraction_rejects_parent_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../../outside.txt", "not allowed")

    with pytest.raises(ValueError, match="unsafe archive member"):
        _extract_zip(archive, tmp_path / "out")

    assert not (tmp_path / "outside.txt").exists()


def test_prepare_accepts_tracked_empty_destination(tmp_path, monkeypatch) -> None:
    import nanobrain.dataset_release as release

    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / ".gitkeep").write_text("")
    (destination / "README.md").write_text("download destination")
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dataset/manifest.json", "{}")
        bundle.writestr("dataset/train/features.npy", "fixture payload")

    monkeypatch.setattr(
        release, "_validate_release", lambda root: {"validated": (root / "manifest.json").is_file()}
    )

    manifest = prepare_dataset(destination=destination, archive=archive)

    assert manifest == {"validated": True}
    assert (destination / "train/features.npy").read_text() == "fixture payload"
    assert not (destination / ".gitkeep").exists()
