"""Machine-readable contract supplied to the LLM and saved with every campaign."""
from __future__ import annotations

from . import modules as _modules
from .genome import SEARCH_SPACE
from .registry import backend_catalog

MODULE_FIELDS = ("encoder", "attention", "spatial", "residual", "pairing", "flow",
                 "classification_head", "regression_head")


def component_catalog():
    catalog = backend_catalog()
    for kind, entries in catalog.items():
        for entry in entries:
            if kind == "operation":
                entry["searchable"] = False
                entry["graph_searchable"] = True
                entry["input_contract"] = "Explicit graph tensor contract; see graph component catalog"
                continue
            field = kind if kind != "head" else "classification_head"
            entry["searchable"] = entry["name"] in SEARCH_SPACE.get(field, ()) or (
                kind == "head" and entry["name"] in SEARCH_SPACE["regression_head"])
            entry["input_contract"] = {
                "encoder": "features[B,G] -> state[B,H] or tokens[B,T,H]",
                "attention": "state[B,H] or tokens[B,T,H] -> same shape; tokens then mean-pool",
                "spatial": "state[B,H], neighbor indices[B,K], bool mask[B,K] -> state[B,H]; absent neighbors = identity",
                "residual": "state[B,H] -> state[B,H]",
                "pairing": "source[P,H], target[P,H] -> barycenter[P,H], plan[P,P] (detached OT)",
                "flow": "state[P,H], context[P,H], time[P,3]=(tau,source_years,delta_years) -> velocity[P,H]",
                "head": "state[B,H] -> logits[B,C] or expression[B,G]",
            }[kind]
            keys = {
                "encoder": ("hidden_dim", "encoder_depth", "dropout"),
                "attention": ("hidden_dim", "attention_heads", "dropout"),
                "spatial": ("hidden_dim",), "residual": ("hidden_dim",),
                "pairing": ("sinkhorn_epsilon", "sinkhorn_iterations"),
                "flow": ("hidden_dim", "flow_depth", "attention_heads", "dropout"),
                "head": ("hidden_dim", "dropout"),
            }[kind]
            entry["parameter_ranges"] = {key: list(SEARCH_SPACE[key]) for key in keys}
    return catalog
