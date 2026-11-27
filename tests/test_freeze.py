"""A frozen campaign cannot change: definition, judges, seed model and data are locked."""

import json
import shutil
from pathlib import Path

import pytest

from nanobrain.infra import campaign, freeze
from nanobrain.infra.contracts import ROOT


@pytest.fixture
def setup(tmp_path):
    folder = tmp_path / "campaign"
    shutil.copytree(campaign.directory(), folder, ignore=shutil.ignore_patterns(freeze.LOCK))
    data = {}
    for name in ("pretrain", "annotation"):
        root = tmp_path / "data" / name
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({"name": name}))
        data[name] = str(root)
    return folder, {"data": data}


def test_unfrozen_campaign_is_refused(setup):
    folder, cfg = setup
    with pytest.raises(ValueError, match="not frozen"):
        freeze.verify(cfg, campaign.load(folder))


def test_freeze_then_verify(setup):
    folder, cfg = setup
    definition = campaign.load(folder)
    lock = freeze.freeze(cfg, definition)
    assert any(k.startswith("brainwm_unified/") for k in lock["files"])
    assert "scripts/predict_eval.py" in lock["files"] and set(lock["data"]) == {"annotation", "pretrain"}
    assert freeze.verify(cfg, definition) == freeze.verify(cfg, definition)
    assert freeze.freeze(cfg, definition)["frozen_at"] == lock["frozen_at"]


def test_changed_definition_or_data_is_refused(setup):
    folder, cfg = setup
    definition = campaign.load(folder)
    freeze.freeze(cfg, definition)
    manifest = Path(cfg["data"]["annotation"]) / "manifest.json"
    original = manifest.read_text()
    manifest.write_text(original + " ")
    with pytest.raises(ValueError, match="data:annotation"):
        freeze.verify(cfg, definition)
    manifest.write_text(original)
    freeze.verify(cfg, definition)
    toml = folder / "campaign.toml"
    toml.write_text(toml.read_text().replace("regression_floor = 0.005", "regression_floor = 0.01"))
    with pytest.raises(ValueError, match="campaign.toml"):
        freeze.verify(cfg, definition)
    with pytest.raises(ValueError, match="new campaign id"):
        freeze.freeze(cfg, definition)


def test_moved_data_root_is_refused(setup, tmp_path):
    folder, cfg = setup
    definition = campaign.load(folder)
    freeze.freeze(cfg, definition)
    moved = tmp_path / "elsewhere"
    shutil.copytree(cfg["data"]["pretrain"], moved)
    with pytest.raises(ValueError, match="data:pretrain"):
        freeze.verify({"data": {**cfg["data"], "pretrain": str(moved)}}, definition)


def test_cajal_campaign_lock_matches_its_tag():
    """A frozen campaign runs from campaign/<id>: every locked file there has the hash in the lock."""
    import hashlib
    import subprocess

    from nanobrain.infra.contracts import load_config

    definition = campaign.load()
    tag = f"campaign/{campaign.directory().name}"
    if subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", tag], check=False, capture_output=True
    ).returncode:
        pytest.skip(f"{tag} is not in this clone")
    lock = json.loads(
        subprocess.check_output(["git", "-C", str(ROOT), "show", f"{tag}:{freeze.lock_path(definition).relative_to(ROOT).as_posix()}"])
    )
    assert lock["campaign_id"] == definition["campaign"]["id"]
    for path, expected in lock["files"].items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{tag}:{path}"])
        assert hashlib.sha256(blob).hexdigest() == expected, path
    cfg = load_config()
    if all((Path(p) / "manifest.json").is_file() for p in cfg["data"].values()):
        assert freeze.snapshot(definition, cfg)["data"] == lock["data"]


def test_declared_label_files_are_locked_with_their_data(tmp_path):
    definition = {**campaign.load(), "freeze": {**campaign.load()["freeze"], "data_files": {"a": ["age.npy"]}}}
    root = tmp_path / "a"
    root.mkdir()
    (root / "manifest.json").write_text("{}")
    cfg = {"data": {"a": str(root)}}
    with pytest.raises(FileNotFoundError, match="age.npy"):
        freeze.snapshot(definition, cfg)
    (root / "age.npy").write_bytes(b"1")
    before = freeze.snapshot(definition, cfg)["data"]["a"]
    assert set(before["files"]) == {"age.npy"}
    (root / "age.npy").write_bytes(b"2")
    assert freeze.snapshot(definition, cfg)["data"]["a"] != before
    with pytest.raises(ValueError, match="unknown data"):
        freeze.snapshot(definition, {"data": {"b": str(root)}})


def test_age_draft_is_defined_but_not_frozen():
    draft = campaign.load(str(ROOT / "campaigns/cajal-v1-age"))
    assert not freeze.lock_path(draft).exists()
    assert "age" in draft["objectives"]["names"] and draft["campaign"]["id"] != campaign.load()["campaign"]["id"]
    age = [m for m in draft["metrics"] if m["group"] == "age"]
    assert age and all(m["direction"] == "min" for m in age)
    assert set(draft["freeze"]["data_files"]) <= {"pretrain", "annotation"}
