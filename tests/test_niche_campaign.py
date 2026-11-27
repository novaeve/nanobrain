import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def describe_campaign(directory):
    code = """
import json
from nanobrain.infra import contracts as c
cfg, root = c.load_config(), c.reference_bundle()
original = c.resolved_document(root, cfg, 0, include_optional_tasks=False)
candidate = c.resolved_document(root, cfg, 0)
root["train"]["objectives"] = []
custom = c.resolved_document(root, cfg, 0)
print(json.dumps({
    "base_hash": c.digest(c.canonical_document(original)),
    "tasks": c.TASKS,
    "metrics": c.METRICS,
    "data_key": c.DATA_KEY,
    "components": candidate["model"]["research_components"],
    "objectives": [item["name"] for item in candidate["train"]["objectives"]],
    "custom_objectives": custom["train"]["objectives"],
    "weighted_tasks": c.candidate_schema()["properties"]["train"]["properties"]["task_weights"]["required"],
}))
"""
    env = dict(os.environ, NANOBRAIN_CAMPAIGN=str(ROOT / "campaigns" / directory))
    return json.loads(subprocess.check_output([sys.executable, "-B", "-c", code], cwd=ROOT, env=env, text=True))


def test_niche_is_an_opt_in_sixth_objective():
    default = describe_campaign("cajal-v3")
    niche = describe_campaign("cajal-v3-niche")
    assert default["base_hash"] == niche["base_hash"]
    assert len(default["tasks"]) == 5 and not default["components"]
    assert niche["tasks"] == [*default["tasks"], "niche_composition"]
    assert niche["metrics"] == [*default["metrics"], "Niche composition MAE"]
    assert "niche" not in default["data_key"]
    assert niche["data_key"] == {**default["data_key"], "niche": "spatial"}
    assert niche["weighted_tasks"] == default["weighted_tasks"]
    assert niche["components"] == [{"slot": "niche", "name": "cajal.niche_composition", "params": {}}]
    assert niche["objectives"] == [*default["objectives"], "niche_composition"]
    assert niche["custom_objectives"] == [{"name": "niche_composition", "weight": 1.0}]
