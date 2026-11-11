"""Historical checkpoints cannot become hidden candidate training dependencies."""

import copy
import json

import pytest

from cajal_research import contracts as C
from cajal_research.agent import SYSTEM, ToolSession
from cajal_research.code_policy import inspect_source
from cajal_research.components import catalog
from cajal_research.training_policy import identity_for, prepare_training, validate_identity


def test_config_rejects_reintroducing_a_pretrained_teacher(tmp_path):
    cfg = C.load_config()
    assert cfg["weight_policy"] == "random_init_self_contained" and not cfg.get("trusted_teachers")
    for mutation in [{"trusted_teachers": {"old": {"path": "/tmp/old.pt"}}}, {"weight_policy": "pretrained"}]:
        path = tmp_path / "config.json"
        C.save(path, dict(cfg, **mutation))
        with pytest.raises(ValueError, match="independently"):
            C.load_config(path)


def test_teacher_is_not_exposed_by_tools_or_catalogue(tmp_path):
    session = ToolSession(None, {}, None, [], set(), {}, tmp_path / "unused")
    contract = session.call("inspect_contract", {})
    schema = contract["schema"]["properties"]["train"]["properties"]["objectives"]["items"]["properties"]
    assert "teacher_gram" not in schema["name"]["enum"] and "teacher" not in schema
    assert "random initialization" in contract["weight_policy"]
    assert "NEVER trained weights" in SYSTEM
    teacher = next(x for x in catalog() if x["id"] == "cajal.teacher_gram")
    assert teacher["available"] is False


@pytest.mark.parametrize("attribute", ["load", "load_state_dict", "from_pretrained"])
def test_custom_sources_cannot_load_checkpoint_weights(attribute):
    with pytest.raises(ValueError):
        inspect_source(f"import torch\ndef build(hidden_dim):\n    return torch.{attribute}('old.pt')\n", "build")


def test_fresh_run_and_own_interrupted_run_have_distinct_initialization(tmp_path):
    cfg, bundle = C.load_config(), C.reference_bundle()
    assert prepare_training(tmp_path, bundle, cfg, 0) == []
    identity_path = tmp_path / "training-identity.json"
    identity = json.loads(identity_path.read_text())
    assert identity["candidate_id"] == C.digest(bundle) and identity["external_weight_dependencies"] == []
    (tmp_path / "train").mkdir()
    (tmp_path / "train/latest.pt").write_bytes(b"own checkpoint")
    assert prepare_training(tmp_path, bundle, cfg, 0) == ["--resume"]
    assert json.loads(identity_path.read_text()) == identity
    other = copy.deepcopy(bundle)
    other["train"]["lr"] = 0.0005
    with pytest.raises(ValueError, match="different candidate"):
        prepare_training(tmp_path, other, cfg, 0)
    with pytest.raises(ValueError, match="seed"):
        prepare_training(tmp_path, bundle, cfg, 1)


def test_a_foreign_or_unidentified_checkpoint_cannot_be_used_to_warm_start(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train/latest.pt").write_bytes(b"foreign checkpoint")
    with pytest.raises(ValueError, match="unidentified checkpoint"):
        prepare_training(tmp_path, C.reference_bundle(), C.load_config(), 0)
    assert not (tmp_path / "training-identity.json").exists()


def test_recipe_changes_cannot_relabel_an_existing_own_checkpoint(tmp_path):
    cfg, bundle = C.load_config(), C.reference_bundle()
    prepare_training(tmp_path, bundle, cfg, 0)
    changed = dict(cfg, protocol_id=cfg["protocol_id"] + "-other")
    with pytest.raises(ValueError, match="training recipe"):
        prepare_training(tmp_path, bundle, changed, 0)


def test_runtime_checks_reject_legacy_teacher_bundles_before_any_launch(tmp_path):
    cfg, bundle = C.load_config(), C.reference_bundle()
    bundle["train"]["objectives"] = [{"name": "teacher_gram", "teacher": "cell_type_30k", "weight": 1.0}]
    with pytest.raises(ValueError):
        prepare_training(tmp_path, bundle, cfg, 0)
    assert not (tmp_path / "training-identity.json").exists()


def test_historical_results_cannot_be_relabelled_as_independent_training():
    cfg, bundle = C.load_config(), C.reference_bundle()
    receipt = {
        "weight_policy": cfg["weight_policy"],
        "external_weight_dependencies": [],
        "training_identity": identity_for(bundle, cfg, 0),
    }
    validate_identity(receipt, bundle, cfg, 0)
    for key, value in [
        ("weight_policy", "teacher_assisted"),
        ("external_weight_dependencies", ["teacher.pt"]),
        ("training_identity", {}),
    ]:
        changed = dict(receipt, **{key: value})
        with pytest.raises(ValueError, match="does not attest"):
            validate_identity(changed, bundle, cfg, 0)
