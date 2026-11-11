"""Cajal adaptation boundaries and real CPU numerical probes; no LLM/real training."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from nanobrain.agent.react import ToolSession, propose
from nanobrain.infra import contracts as C
from nanobrain.infra import gate as selection
from nanobrain.infra.store import Store


def candidate(**train):
    b = C.reference_bundle()
    b["parent_ids"] = ["a" * 64]
    b["hypothesis"] = "Test a declared change from the Cajal recipe."
    b["train"] = train
    return b


def plugin_candidate():
    b = candidate(lr=0.0005)
    b["files"] = {
        "modules/residual.py": """import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        return x + 0.1 * self.linear(self.norm(x))

def build(hidden_dim):
    return Residual(hidden_dim)
"""
    }
    b["plugins"] = [
        {
            "name": "custom_residual",
            "kind": "token_transform",
            "file": "modules/residual.py",
            "parameters": {},
            "settings": {},
        }
    ]
    return b


def test_default_is_read_only_plan(tmp_path, monkeypatch, capsys):
    from nanobrain import cli
    from nanobrain.agent.api_config import APIConfig

    monkeypatch.setattr(APIConfig, "from_env", lambda *a, **k: pytest.fail("plan read API credentials"))
    monkeypatch.setattr(cli.Workflow, "execute", lambda *a: pytest.fail("plan started evolution"))
    output = tmp_path / "must_not_exist"
    cli.main(["--output", str(output)])
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "prepared_not_started"
    assert not doc["llm_called"] and not doc["real_training_started"]
    assert doc["joint_trainings_per_generation"] == 10
    assert doc["maximum_joint_trainings"] == 300
    assert doc["reference_trainings"] == 0
    assert doc["seeds"] == [0] and doc["seed_count"] == 1
    assert not doc["uncertainty_estimated"]
    assert not output.exists()


def test_reference_is_actual_cajal_v1():
    ref = C.reference_bundle()
    C.validate_bundle(ref)
    assert ref["model"] == ref["train"] == ref["files"] == {}
    assert ref["plugins"] == []
    doc = C.reference_document()
    assert doc["ModelConfig"]["n_experts"] == 32
    assert doc["ModelConfig"]["d_model"] == 256
    assert doc["ModelConfig"]["n_core"] == 2
    assert doc["ModelConfig"]["attention_backend"] == "fa3"
    assert doc["TrainConfig"]["balancer"] == "grad_equal"
    assert doc["TrainConfig"]["balancer_params"] == {"every": 50}
    assert set(doc["TrainConfig"]["task_weights"]) == set(C.TASKS)


def test_one_model_document_and_fixed_budget():
    cfg = C.load_config()
    doc = C.resolved_document(C.reference_bundle(), cfg, cfg["seeds"][0])
    assert set(doc) == {"model", "data", "train"}
    assert doc["train"]["steps"] == 30000 and doc["train"]["stop_at"] == 10000
    assert doc["data"]["root"] == "/data/brainwm/unified_1m"
    assert all(w > 0 for w in doc["train"]["task_weights"].values())
    assert doc["model"]["research_plugins"] == {}


@pytest.mark.parametrize("balancer", ["initial", "famo"])
def test_balancer_switch_uses_its_own_parameters(balancer):
    from brainwm_unified.balance import build_balancer

    doc = C.resolved_document(candidate(balancer=balancer), C.load_config(), 0)
    assert doc["train"]["balancer_params"] == {}
    build_balancer(balancer, 200, doc["train"]["balancer_params"])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "n_genes", 256),
        ("model", "n_batches", 2),
        ("model", "likelihood", "mse_log"),
        ("model", "cell_type_levels", [2]),
        ("model", "research_plugins", {}),
        ("train", "steps", 20),
        ("train", "stop_at", 2),
        ("train", "out_dir", "/tmp"),
        ("train", "task_schedule", []),
        ("train", "research_plugins", {}),
        ("train", "task_weights", {"gene": 1}),
        ("train", "lr", float("nan")),
        ("model", "d_model", True),
        ("model", "n_heads", 8),
        ("model", "top_k", 64),
    ],
)
def test_immutable_and_invalid_changes_rejected(section, key, value):
    b = candidate()
    b[section][key] = value
    with pytest.raises(ValueError):
        C.validate_bundle(b)


def test_all_five_weights_required_positive():
    b = candidate(task_weights=dict.fromkeys(C.TASKS, 1.0))
    C.validate_bundle(b)
    b["train"]["task_weights"]["spatial"] = 0.0
    with pytest.raises(ValueError):
        C.validate_bundle(b)


def test_source_io_and_unconnected_files_rejected():
    b = plugin_candidate()
    b["files"]["modules/residual.py"] = "import os\ndef build(hidden_dim):\n    return os.system('id')\n"
    with pytest.raises(ValueError):
        C.validate_bundle(b)
    b = plugin_candidate()
    b["files"]["modules/unused.py"] = "def build(hidden_dim):\n    return hidden_dim\n"
    with pytest.raises(ValueError):
        C.validate_bundle(b)


def test_semantic_dedup_defaults_and_ancestry():
    a = candidate(lr=0.0005)
    b = copy.deepcopy(a)
    b["parent_ids"] = ["b" * 64]
    b["hypothesis"] = "Different prose, same numerical experiment."
    b["model"]["d_model"] = 256
    assert C.digest(a) != C.digest(b)
    assert C.semantic_id(a) == C.semantic_id(b)


def test_single_seed_has_no_invented_standard_deviation():
    seeds = C.load_config()["seeds"]
    rows = [
        {"status": "complete", "seed": s, "metrics": dict.fromkeys(C.METRICS, 0.6 + i * 0.01)}
        for i, s in enumerate(seeds)
    ]
    result = selection.aggregate(rows, seeds)
    assert result["metrics"][C.METRICS[0]] == pytest.approx(0.6)
    assert result["seed_count"] == 1 and not result["uncertainty_estimated"]
    assert all(value is None for value in result["std"].values())
    for broken in [[], [rows[0], rows[0]]]:
        with pytest.raises(ValueError):
            selection.aggregate(broken, seeds)
    rows[0]["metrics"].pop(C.METRICS[-1])
    with pytest.raises(ValueError):
        selection.aggregate(rows, seeds)


def test_nan_metrics_fail_closed():
    seeds = C.load_config()["seeds"]
    rows = [{"status": "complete", "seed": s, "metrics": dict.fromkeys(C.METRICS, 0.6)} for s in seeds]
    rows[0]["metrics"][C.METRICS[2]] = float("nan")
    with pytest.raises(ValueError):
        selection.aggregate(rows, seeds)


def test_champion_needs_improvement_above_spread_and_no_regression():
    parent = {"metrics": dict.fromkeys(C.METRICS, 0.5), "std": dict.fromkeys(C.METRICS, 0.01)}
    child = copy.deepcopy(parent)
    baseline = dict.fromkeys(C.METRICS, 0.6)
    child["metrics"][C.METRICS[0]] = 0.505
    assert not selection.promote(child, parent, parent, baseline, [C.METRICS[0]])["passed"]
    child["metrics"] = dict.fromkeys(C.METRICS, 0.53)
    assert selection.promote(child, parent, parent, baseline, list(C.METRICS))["passed"]
    child["metrics"][C.METRICS[-1]] = 0.47
    assert not selection.promote(child, parent, parent, baseline, list(C.METRICS))["passed"]


def test_store_is_immutable_and_tree_keeps_failures(tmp_path):
    store = Store(tmp_path)
    a = store.stage(C.reference_bundle(), origin={"generation": 0})
    b = candidate(lr=0.0005)
    b["parent_ids"] = [a]
    i = store.stage(b, [a], {"generation": 1})
    store.put(i, "result", {"candidate_id": i, "status": "failed", "reason": "synthetic failure", "generation": 1})
    tree = store.tree_summary()
    assert tree["complete"] and not tree["truncated"] and len(tree["nodes"]) == 2
    assert [a, i] in tree["edges"]
    p = store.candidates / i / "bundle.json"
    changed = json.loads(p.read_text())
    changed["train"]["lr"] = 0.0002
    C.save(p, changed)
    with pytest.raises(ValueError):
        store.bundle(i)


def test_token_plugins_cannot_mix_cells():
    from nanobrain.infra.plugins import apply_transforms

    class Mean(torch.nn.Module):
        def forward(self, x):
            return x + x.mean(1, keepdim=True)

    modules = torch.nn.ModuleList([Mean()])
    x = torch.arange(12.0).reshape(6, 2)
    doc = torch.tensor([0, 0, 0, 1, 1, -1])
    a = apply_transforms(modules, x, doc)
    y = x.clone()
    y[3:5] += 1000
    b = apply_transforms(modules, y, doc)
    assert torch.equal(a[:3], b[:3])
    assert torch.equal(a[-1], x[-1])


def test_empty_extension_preserves_stock_parameters_and_forward():
    from brainwm_unified import model as M
    from brainwm_unified.config import ModelConfig

    path = C.ROOT / "reference/stock_model.py"
    spec = importlib.util.spec_from_file_location("brainwm_unified.stock_model", path)
    stock = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stock
    spec.loader.exec_module(stock)
    cfg = ModelConfig(
        n_genes=24,
        n_batches=2,
        cell_type_levels=[3],
        anatomy_levels=[2],
        d_model=32,
        n_heads=4,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        n_experts=4,
        top_k=2,
        expert_hidden=32,
        z_dim=8,
        decoder_hidden=32,
        spatial_d=16,
        spatial_layers=1,
        max_loops=2,
        compile_blocks=False,
        activation_checkpointing=False,
        moe_backend="loop",
    )
    torch.manual_seed(9)
    a = stock.UnifiedModel(cfg).eval()
    torch.manual_seed(9)
    b = M.UnifiedModel(cfg).eval()
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    batch = {
        "tok_gene": torch.tensor([24, 0, 1, 24, 0, 1]),
        "tok_doc": torch.tensor([0, 0, 0, 1, 1, 1]),
        "tok_cls": torch.tensor([True, False, False, True, False, False]),
        "tok_in": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.5, 0.0]),
        "tok_masked": torch.tensor([False, False, True, False, False, True]),
    }
    with torch.no_grad():
        x = a(batch, stock.dense_mask(batch["tok_doc"]), 1)
        y = b(batch, M.dense_mask(batch["tok_doc"]), 1)
    assert all(torch.equal(x[k], y[k]) for k in x)


def test_mock_react_uses_tools_no_real_llm(tmp_path):
    store = Store(tmp_path / "lineage")
    root = store.stage(C.reference_bundle())
    b = candidate(lr=0.0005)
    b["parent_ids"] = [root]
    identifier = C.digest(b)
    session = ToolSession(
        store,
        {},
        lambda *a: {"status": "accepted"},
        [root],
        [C.semantic_id(C.reference_bundle())],
        {"generation": 1},
        tmp_path / "submission.json",
    )
    calls = [
        ("write_candidate", {"bundle": b}),
        ("validate_candidate", {"candidate_id": identifier}),
        ("submit_candidate", {"candidate_id": identifier}),
    ]

    class Client:
        def chat(self, messages, **kwargs):
            name, args = calls.pop(0)
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        }
                    }
                ]
            }

    result = propose(Client(), session, {"test": True})
    assert result["candidate_id"] == identifier and (tmp_path / "submission.json").is_file()


@pytest.mark.parametrize(
    "variant",
    ["reference", "module", "module_loss_optimizer", "component_hybrid", "legacy_prototype", "regularizer_budget"],
)
def test_linux_sandbox_five_joint_gradients_update_and_reload(tmp_path, variant):
    from nanobrain.infra.sandbox import clean_environment, command, launch

    b = plugin_candidate() if variant in ("module", "module_loss_optimizer") else C.reference_bundle()
    if variant == "component_hybrid":
        b["components"] = [
            {"slot": slot, "name": name, "params": {}}
            for slot, name in [
                ("value_embedding", "legacy.encoder.scfoundation_autobin"),
                ("block_attention", "legacy.attention.bert"),
                ("block_mixer", "legacy.operation.scbrain_ffn"),
                ("latent", "scvi.vae_encoder"),
                ("classifier", "scanvi.classifier"),
                ("decoder", "scvi.nb_decoder"),
                ("neighborhood", "legacy.spatial.graphst_sparse"),
            ]
        ]
    if variant == "legacy_prototype":
        b["components"] = [
            {"slot": slot, "name": name, "params": {}}
            for slot, name in [
                ("value_embedding", "legacy.encoder.scprint_value"),
                ("token", "legacy.operation.sctab_glu"),
                ("classifier", "legacy.head.scpoli_prototype"),
                ("neighborhood", "legacy.spatial.stagate_gat"),
            ]
        ]
    if variant == "regularizer_budget":
        b["train"]["objectives"] = [
            {
                "name": "sigreg",
                "weight": 1.0,
                "start": 100,
                "budget": {"cell_type": 1.0},
                "dual_lr": 0.01,
            }
        ]
    if variant == "module_loss_optimizer":
        b["train"]["optimizer"] = "custom"
        b["files"]["losses/gene.py"] = """import torch.nn.functional as F
def compute(prediction, target):
    return F.smooth_l1_loss(prediction, target)
"""
        b["files"]["optimizers/sgd.py"] = """import torch
from torch.optim import Optimizer
class Descent(Optimizer):
    def __init__(self, parameters, lr, weight_decay):
        super().__init__(parameters, dict(lr=lr, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                    p.add_(p.grad, alpha=-group['lr'])
def build(parameters, lr, weight_decay):
    return Descent(parameters, lr, weight_decay)
"""
        b["plugins"] += [
            {"name": "custom_gene", "kind": "gene_loss", "file": "losses/gene.py", "parameters": {}, "settings": {}},
            {
                "name": "custom_descent",
                "kind": "optimizer",
                "file": "optimizers/sgd.py",
                "parameters": {},
                "settings": {},
            },
        ]
    path = tmp_path / "bundle.json"
    C.save(path, b)
    args = command(C.ROOT, tmp_path / "probe", ["probe", "--bundle", str(path)], inputs=[path])
    assert "--unshare-net" in args and set(clean_environment()) == {"PATH", "LANG"}
    assert "/data/brainwm/unified_1m" not in args and "/data/brainwm/eval_10k" not in args
    result = launch(C.ROOT, tmp_path / "probe", ["probe", "--bundle", str(path)], inputs=[path], timeout=600)
    assert result["status"] == "accepted" and result["synthetic_only"]
    assert result["shared_encoder_gradients"] and result["optimizer_update"] and result["checkpoint_reload"]
    assert set(result["tasks"]) == set(C.TASKS)
    assert not result["benchmark_data_mounted"] and not result["gpu_used"]
    assert {"cell_type:0", "anatomy:0", "reconstruction", "spatial"} <= set(result["checkpoint_prediction_fields"])
    assert result["external_weight_dependencies"] == []


def test_joint_execution_adapter_and_resume_without_real_training(tmp_path, monkeypatch):
    """Exercise train/predict/score orchestration with file-producing fakes, no GPU/API."""
    import csv
    import threading
    import time

    from nanobrain.infra.workflow import Workflow

    cfg = C.load_config()
    workflow = Workflow(cfg, tmp_path)
    workflow.store = Store(tmp_path / "lineage")
    identifier = workflow.store.stage(candidate(lr=0.0005))
    calls = []
    lock = threading.Lock()
    predictions = {"active": 0, "peak": 0}

    def fake_worker(directory, args, inputs, cpu=False):
        directory.mkdir(parents=True, exist_ok=True)
        calls.append(args)
        if args[0] == "train":
            from nanobrain.infra.training_policy import identity_for

            path = directory / "train/latest.pt"
            path.parent.mkdir()
            path.write_bytes(b"one jointly trained model (mock)")
            result = {
                "status": "complete",
                "seed": int(args[args.index("--seed") + 1]),
                "weight_policy": cfg["weight_policy"],
                "external_weight_dependencies": [],
                "training_identity": identity_for(workflow.store.bundle(identifier), cfg, 0),
                "artifact_hashes": {"train/latest.pt": C.sha(path)},
            }
        elif args[0] == "predict":
            with lock:
                predictions["active"] += 1
                predictions["peak"] = max(predictions["peak"], predictions["active"])
            time.sleep(0.05)
            path = directory / "prediction.npz"
            path.write_bytes(b"mock predictions")
            result = {"status": "complete", "artifact_hashes": {"prediction.npz": C.sha(path)}}
            with lock:
                predictions["active"] -= 1
        else:
            path = directory / "scores/summary.csv"
            path.parent.mkdir()
            with path.open("w") as fh:
                w = csv.writer(fh)
                w.writerow(["method", *C.METRICS])
                w.writerow(["candidate", *[0.6] * len(C.METRICS)])
            result = {"status": "complete", "artifact_hashes": {"scores/summary.csv": C.sha(path)}}
        C.save(directory / "receipt.json", result)
        return result

    monkeypatch.setattr(workflow, "worker", fake_worker)
    result = workflow.evaluate_seed(identifier, cfg["seeds"][0])
    assert result["joint_model_count"] == 1 and set(result["metrics"]) == set(C.METRICS)
    assert sum(c[0] == "train" for c in calls) == 1
    assert len({c[c.index("--checkpoint") + 1] for c in calls if c[0] == "predict"}) == 1
    assert predictions["peak"] == cfg["task_evaluation_workers"] == 2
    before = len(calls)
    workflow.evaluate_seed(identifier, cfg["seeds"][0])
    assert len(calls) == before
    checkpoint = Path(next(k for k in result["artifact_hashes"] if k.endswith("latest.pt")))
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError):
        workflow.evaluate_seed(identifier, cfg["seeds"][0])


def test_complete_mock_campaign_resume_and_contract_freeze(tmp_path, monkeypatch):
    """Control-flow integration only: all training and LLM responses are simulated."""
    from nanobrain.infra import workflow as W

    cfg = C.load_config()
    cfg.update(generations=1, candidates_per_generation=1)
    monkeypatch.setattr(W, "code_fingerprint", lambda: {"git_sha": "frozen-test-code"})
    monkeypatch.setattr(W, "input_fingerprint", lambda cfg: {"mock_data": "fixed"})
    monkeypatch.setattr(W, "campaign_fingerprint", lambda cfg: "frozen-test-campaign")
    workflow = W.Workflow(cfg, tmp_path)
    monkeypatch.setattr(workflow, "validate", lambda *a: {"status": "accepted"})
    evaluated = []

    def evaluate(identifier, generation):
        p = workflow.store.candidates / identifier / "result.json"
        if p.exists():
            return json.loads(p.read_text())
        evaluated.append(identifier)
        metrics = dict.fromkeys(C.METRICS, 0.6 + generation * 0.03)
        node = {
            "candidate_id": identifier,
            "generation": generation,
            "status": "complete",
            "all_seeds_passed": True,
            "metrics": metrics,
            "std": dict.fromkeys(C.METRICS, 0.001),
            "seed_results": [{"seed": s, "metrics": metrics} for s in cfg["seeds"]],
        }
        workflow.store.put(identifier, "result", node)
        return node

    monkeypatch.setattr(workflow, "evaluate", evaluate)
    b = candidate(lr=0.0005)
    b["parent_ids"] = [C.digest(C.reference_bundle())]
    identifier = C.digest(b)
    calls = [
        ("write_candidate", {"bundle": b}),
        ("validate_candidate", {"candidate_id": identifier}),
        ("submit_candidate", {"candidate_id": identifier}),
    ]

    class Client:
        def chat(self, messages, **kwargs):
            name, args = calls.pop(0)
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "mock",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        }
                    }
                ]
            }

    state = workflow.execute(Client())
    assert len(state["generations"]) == 1 and len(evaluated) == 2
    assert (tmp_path / "standard_metrics.csv").is_file()
    assert not json.loads((tmp_path / "frozen-champion.json").read_text())["scaling_started"]
    before = (tmp_path / "decisions/g01-p00.json").read_bytes()
    workflow.execute(Client())
    assert len(evaluated) == 2 and (tmp_path / "decisions/g01-p00.json").read_bytes() == before
    cfg["token_budget"] //= 2
    with pytest.raises(ValueError, match="resume requires identical"):
        workflow.execute(Client())


def test_resource_interruption_does_not_freeze_a_failed_candidate(tmp_path, monkeypatch):
    from nanobrain.infra.workflow import InfrastructureError, Workflow

    workflow = Workflow(C.load_config(), tmp_path)
    workflow.store = Store(tmp_path / "lineage")
    identifier = workflow.store.stage(C.reference_bundle())

    def unavailable(*args):
        raise InfrastructureError("temporary GPU resource exhaustion (mock)")

    monkeypatch.setattr(workflow, "evaluate_seed", unavailable)
    with pytest.raises(InfrastructureError):
        workflow.evaluate(identifier, 0)
    assert not (workflow.store.candidates / identifier / "result.json").exists()


def test_current_cajal_data_and_sampling_controls_are_distinct():
    from brainwm_unified.config import DataConfig

    b = candidate(warmup_steps=2000, min_lr_ratio=0.2, lr_schedule="linear", kl_warmup_steps=8000)
    b["data"] = {"dissociated_cells": 256, "spatial_groups": 32, "max_nonzero_tokens": 2048, "zero_tokens": 128}
    C.validate_bundle(b)
    doc = C.resolved_document(b, C.load_config(), 0)
    assert doc["data"]["root"] == "/data/brainwm/unified_1m"
    assert doc["data"]["cell_fraction"] == 1.0
    assert doc["data"]["dissociated_cells"] == 256 and doc["data"]["spatial_groups"] == 32
    DataConfig(**doc["data"])
    for field, value in [("root", "/tmp/100k"), ("seed", 7), ("cell_fraction", 0.1), ("neighbors", 8)]:
        invalid = copy.deepcopy(b)
        invalid["data"][field] = value
        with pytest.raises(ValueError):
            C.validate_bundle(invalid)


def test_optimizer_and_balancer_parameters_reach_actual_implementations():
    from brainwm_unified.balance import build_balancer
    from brainwm_unified.config import TrainConfig
    from brainwm_unified.train import lr_lambda
    from nanobrain.infra.plugins import make_optimizers

    b = candidate(
        optimizer="adamw",
        optimizer_params={"adam_beta1": 0.85, "adam_beta2": 0.99, "adam_eps": 1e-7},
        balancer_params={"every": 20, "beta": 0.8},
        warmup_steps=0,
        lr_schedule="linear",
        min_lr_ratio=0.1,
    )
    doc = C.resolved_document(b, C.load_config(), 0)
    cfg = TrainConfig(**doc["train"])
    model = torch.nn.Linear(3, 2)
    (optimizer,) = make_optimizers(model, cfg, {}, ([], []))
    assert optimizer.param_groups[0]["betas"] == (0.85, 0.99)
    assert optimizer.param_groups[0]["eps"] == 1e-7
    balancer = build_balancer(cfg.balancer, cfg.norm_steps, cfg.balancer_params)
    assert balancer.every == 20 and balancer.beta == 0.8
    assert lr_lambda(cfg)(cfg.steps) == pytest.approx(0.1)
    b["train"]["optimizer"] = "sgd"
    with pytest.raises(ValueError):
        C.validate_bundle(b)


def test_historical_teacher_is_outside_the_candidate_contract():
    b = candidate(objectives=[{"name": "teacher_gram", "weight": 1.0, "teacher": "cell_type_30k"}])
    before = C.digest(b)
    with pytest.raises(ValueError):
        C.resolved_document(b, C.load_config(), 0)
    assert C.digest(b) == before
    b["train"]["objectives"][0]["checkpoint"] = "/tmp/foreign.pt"
    with pytest.raises(ValueError):
        C.validate_bundle(b)


def test_component_catalog_is_merged_and_does_not_import_legacy_data():
    from nanobrain.infra.components import catalog, legacy

    rows = catalog()
    ids = {r["id"] for r in rows}
    assert {
        "cajal.moe",
        "cajal.teacher_gram",
        "cajal.grouped_muon",
        "scvi.vae_encoder",
        "scvi.nb_decoder",
        "scanvi.classifier",
    } <= ids
    assert "legacy.attention.differential_linear" in ids
    assert any(r["id"] == "baseline.XGBoost" and r["role"] == "whole_method_baseline" for r in rows)
    legacy()
    assert "nanobrain_v1.data" not in sys.modules and "nanobrain_v1.training" not in sys.modules
    for row in rows:
        if row.get("verification", {}).get("status") == "unavailable":
            b = candidate()
            b["components"] = [{"slot": row["slots"][0], "name": row["id"], "params": {}}]
            with pytest.raises(ValueError):
                C.validate_bundle(b)


def test_baseline_snapshot_matches_current_cajal_scores():
    opponents = json.loads((C.ROOT / "reference/opponents.json").read_text())
    assert opponents["source_csv_sha256"] == C.sha(C.ROOT / "reference/cajal_existing_five_tasks.csv")
    assert len(opponents["all_existing_baselines"]) == 13
    assert len(opponents["available_method_names"]) == 63
    assert opponents["best_method"]["Integration batch"] == "scANVI"
    assert opponents["best_opponents"]["Integration batch"] == pytest.approx(0.4535480119005794)
    assert any(row["method"] == "cajal_v1@10000" for row in opponents["cajal_reference_performance"])


def test_available_components_preserve_masked_spatial_causality():
    from brainwm_unified import model as M
    from brainwm_unified.config import ModelConfig
    from nanobrain.infra.components import catalog

    rows = [r for r in catalog() if r["available"] and "neighborhood" in r["slots"]]
    cfg = ModelConfig(
        n_genes=24,
        n_batches=2,
        cell_type_levels=[3],
        anatomy_levels=[2],
        d_model=32,
        n_heads=4,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        n_experts=4,
        top_k=2,
        expert_hidden=32,
        z_dim=8,
        decoder_hidden=32,
        spatial_d=16,
        spatial_layers=1,
        compile_blocks=False,
        activation_checkpointing=False,
        moe_backend="loop",
        max_loops=2,
    )
    z = torch.randn(2, 5, 8)
    rel = torch.randn(2, 5, 3)
    masked = torch.zeros(2, 5, dtype=torch.bool)
    masked[:, 0] = True
    pad = torch.zeros_like(masked)
    for row in rows:
        cfg.research_components = [{"slot": "neighborhood", "name": row["id"], "params": {}}]
        model = M.UnifiedModel(cfg).eval()
        changed = z.clone()
        changed[:, 0] += 10000
        with torch.no_grad():
            a = model.neighborhood(z, rel, masked, pad)
            b = model.neighborhood(changed, rel, masked, pad)
        assert torch.allclose(a[:, 0], b[:, 0]), row["id"]


def test_sandbox_username_allows_torch_compile_initialization(tmp_path):
    import subprocess

    from nanobrain.infra.sandbox import clean_environment, command

    args = command(C.ROOT, tmp_path / "worker", [])
    separator = args.index("--")
    python = args[separator + 1]
    code = (
        "import getpass, torch; "
        "assert getpass.getuser() == 'cajal-worker'; "
        "torch.compile(torch.nn.Linear(2, 2)); "
        "print('sandbox compile initialization passed')"
    )
    args[separator:] = ["--", python, "-B", "-c", code]
    result = subprocess.run(args, env=clean_environment(), capture_output=True, text=True, timeout=120, check=True)
    assert "sandbox compile initialization passed" in result.stdout
