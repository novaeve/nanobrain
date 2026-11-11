"""Backbone/training-only search with tutorial task adapters held fixed."""
from __future__ import annotations

import copy
import json

from .architecture import Shape, canonical_architecture, canonical_graph, graph_catalog
from .objectives import default_objectives, objective_catalog
from .genome import SEARCH_SPACE


SEARCH_SCOPE = "backbone_training_fixed_heads_v1"
BACKBONE_KINDS = ("encoder", "attention", "spatial", "residual", "operation")


def fixed_heads():
    return {task: {"name": "classification" if task in {"annotation", "integration", "niche"}
                   else "regression", "params": {}}
            for task in ("annotation", "integration", "integration_reconstruction",
                         "niche", "development", "completion")}


def fixed_flow():
    # Width is inferred from the backbone interface, never a separate head edit.
    return {"nodes": [{"id": "velocity", "kind": "flow", "name": "fmt",
                       "inputs": ["state", "context", "time"],
                       "params": {"heads": 4, "depth": 2}, "repeat": 1}],
            "output": "velocity"}


def validate_search_scope(genome, scope):
    if scope is None:
        return
    if scope != SEARCH_SCOPE:
        raise ValueError(f"unsupported search_scope: {scope}")
    if genome.get("architecture") is None:
        raise ValueError("fixed-head search requires a graph genome")
    architecture = canonical_architecture(genome["architecture"])
    if architecture["version"] != 3 or "endpoint" in architecture:
        raise ValueError("fixed-head search requires version 3 and forbids endpoint adapters")
    if architecture["heads"] != fixed_heads():
        raise ValueError("task heads are fixed tutorial classification/regression heads; params/pre are not searchable")
    width = architecture["width"]
    expected = canonical_graph(fixed_flow(), {"state": Shape(width), "context": Shape(width), "time": Shape(3)})
    if architecture["flow"] != expected:
        raise ValueError("task adaptation flow is fixed to tutorial FMT (depth=2, heads=4)")
    if architecture["pairing"]["name"] != "sinkhorn":
        raise ValueError("tutorial Sinkhorn pairing is fixed; its training parameters remain searchable")
    for task, objective in default_objectives().items():
        if architecture["objectives"][task]["name"] != objective["name"]:
            raise ValueError(f"task objective family is fixed: {task}; loss parameters remain searchable")


def allowed_patch_path(path):
    roots = ("/learning_rate", "/architecture/width", "/architecture/backbone",
             "/architecture/pairing/params", "/architecture/solver")
    if any(path == root or path.startswith(root + "/") for root in roots):
        return True
    return any(path == f"/architecture/objectives/{task}/params" or
               path.startswith(f"/architecture/objectives/{task}/params/")
               for task in default_objectives())


def adapt_interface_width(genome, scope, parent_widths=()):
    """Propagate a backbone width edit through the fixed FMT input interface."""
    if scope is None:
        return genome
    result = copy.deepcopy(genome)
    architecture = result.get("architecture", {})
    width = architecture.get("width")
    nodes = architecture.get("flow", {}).get("nodes", [])
    if len(nodes) == 1 and nodes[0].get("kind") == "flow" and nodes[0].get("name") == "fmt":
        params = nodes[0].get("params", {})
        if "hidden_dim" in params:
            if params["hidden_dim"] not in (width, *parent_widths):
                raise ValueError("fixed FMT hidden_dim is derived from the backbone width")
            params["hidden_dim"] = width
    return result


def search_catalog():
    catalog = graph_catalog()
    return {kind: catalog[kind] for kind in BACKBONE_KINDS}


def build_prompt(protocol, parents, initial_reference, previous_generations, count):
    context = lambda x: {k: x[k] for k in ("architecture_id", "genome", "metrics", "score",
                                          "runtime_seconds", "parameter_count", "diagnostics") if k in x}
    losses = objective_catalog()
    fixed_loss_names = {item["name"] for item in default_objectives().values()}
    losses["registry"] = {k: v for k, v in losses["registry"].items() if k in fixed_loss_names}
    return (
        'Propose nanoBrain backbone computation graphs and permitted training parameters. '
        'Respond with JSON {"candidates": [...]} only. '
        f'Exactly {count} distinct candidates. Each needs parent_ids, target_tasks, hypothesis, expected_tradeoff, '
        'and EITHER a full genome with exactly architecture and learning_rate OR an RFC6902 patch against one parent. '
        'Use only current parent IDs. Never return both genome and patch. '
        'Only architecture.backbone, architecture.width, learning_rate, objectives.<task>.params, '
        'pairing.params and solver are mutable. Full genomes must preserve every other field. '
        'TASK ADAPTERS ARE FIXED: annotation/niche/integration use classification heads; '
        'integration_reconstruction/development/completion use regression heads. All head params are empty; '
        'head pre-graphs are forbidden. Development uses the fixed FMT flow (depth=2, heads=4), '
        'Sinkhorn pairing and conditional_flow_matching objective. No flow edits, endpoint adapters, '
        'alternative heads, or objective-family changes. Fixed heads and FMT input width follow architecture.width automatically. '
        'Classification is Linear(H,C); regression is LayerNorm(H) then Linear(H,G). '
        'All task heads and FMT weights train normally from scratch; fixed means fixed architecture, not frozen weights. '
        'Only the supplied registered backbone components may be used; no Python/code or new components. '
        'Insert/remove/reorder/replace/repeat compatible backbone nodes; parallel paths with add/subtract/concat/stack are allowed. '
        'Nodes are topologically ordered with unique IDs; inputs reference prior nodes or features[B,G], '
        'batch_id[B,1], nonnegative_features[B,G]. All nodes must contribute to the output [B,width]. '
        'Max 32 nodes / 64 expanded layers; repeat=1..8 only for unary shape-preserving nodes. '
        'Width=16..512 must be divisible by 4 for fixed FMT. Only explicit catalog factory params are accepted; '
        'input_dim/output_dim are inferred and the data input_dim/version stay fixed. '
        'Token encoders need explicit pooling before rank-2 outputs/spatial layers. '
        'At most one nonidentity spatial hop per path. scpoli_conditional takes [features_or_state,batch_id]; '
        'scprint_value must take nonnegative_features; scfoundation_autobin and scprint_value create gene tokens. '
        'harmony/scanorama/bbknn are input-only cached transforms: inputs=["features"], repeat=1, strength in [0,1], '
        'no hidden_dim; use only before an encoder. Raw/corrected branches may be merged. '
        'Include each of harmony, scanorama and bbknn at least once across the generation. '
        'celltypist_linear and other task heads are not in the search space. '
        'Training-related choices include the listed learning rates, loss regression type/weights/label smoothing/ '
        'adversarial strength, Sinkhorn epsilon/iterations and Euler/Heun solver/rollout steps. '
        'Required loss terms remain positive. Training steps, batch sizes, seeds, splits, checkpoint policy, '
        'evaluators, promotion gates and task definitions remain fixed by protocol. '
        'Each candidate trains independently from scratch for each task with separate weights/optimizers/checkpoints. '
        'Selection uses test metrics in this existing exploratory protocol; fixed score and per-task guards decide promotion. '
        'Explain the metric hypothesis and tradeoff. ID-only changes are not novel candidates.\n'
        + json.dumps(dict(search_scope=SEARCH_SCOPE, component_catalog=search_catalog(),
                          objective_catalog=losses, learning_rates=SEARCH_SPACE["learning_rate"],
                          protocol=protocol, parents=[context(x) for x in parents],
                          initial_reference=context(initial_reference), previous_generations=previous_generations),
                     ensure_ascii=False))
