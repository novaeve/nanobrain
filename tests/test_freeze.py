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
    shutil.copytree(ROOT / "campaigns/cajal-v1", folder, ignore=shutil.ignore_patterns(freeze.LOCK))
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


def test_cajal_campaign_is_frozen_and_intact():
    from nanobrain.infra.contracts import load_config

    cfg = load_config()
    missing = [p for p in cfg["data"].values() if not (Path(p) / "manifest.json").is_file()]
    if missing:
        pytest.skip(f"campaign data not on this machine: {missing}")
    assert (campaign.directory() / freeze.LOCK).is_file()
    freeze.verify(cfg, campaign.load())
