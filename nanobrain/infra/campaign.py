"""Campaign definitions: the open choices a user writes before evolution, locked once a campaign starts.

A campaign directory holds campaign.toml (data, objectives, metrics, budget, resources, gate and search
space) and the agent brief. One process runs one campaign, chosen by NANOBRAIN_CAMPAIGN (a directory)
and defaulting to campaigns/cajal-v1.
"""

import os
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "campaigns/cajal-v1"
SECTIONS = ("campaign", "budget", "resources", "objectives", "gate", "metrics", "search", "freeze")


def directory():
    return Path(os.environ.get("NANOBRAIN_CAMPAIGN", DEFAULT)).resolve()


@lru_cache(None)
def load(path=None):
    folder = Path(path).resolve() if path else directory()
    doc = tomllib.loads((folder / "campaign.toml").read_text())
    missing = [s for s in SECTIONS if s not in doc]
    if missing:
        raise ValueError(f"campaign definition lacks sections: {missing}")
    names = [m["name"] for m in doc["metrics"]]
    if len(set(names)) != len(names) or not names:
        raise ValueError("metric names must be unique and nonempty")
    for m in doc["metrics"]:
        low, high = m["range"]
        if not low < high:
            raise ValueError(f"invalid range for metric {m['name']}")
    doc["directory"] = str(folder)
    return doc


def brief(path=None):
    doc = load(path)
    text = (Path(doc["directory"]) / doc["campaign"]["agent_brief"]).read_text()
    return text.removesuffix("\n")
