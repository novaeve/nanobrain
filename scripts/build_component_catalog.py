#!/usr/bin/env python3
"""Inventory and CPU-test frozen components. Never run an evolution or read cell data."""

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanobrain.infra import components as C
from nanobrain.infra.contracts import save, sha


def native_entries():
    items = [
        ("cajal.attention", ["block_attention"], "model", "Attention", "model.n_heads"),
        ("cajal.moe", ["block_mixer"], "model", "MoE", "model.n_experts/top_k/router_score"),
        ("cajal.swiglu", ["token", "block_mixer"], "model", "SwiGLU", None),
        ("cajal.recurrent_encoder", [], "model", "Encoder", "model.n_prelude/n_core/n_coda/loops"),
        ("cajal.value_mlp", ["value_embedding"], "model", "Encoder", None),
        ("cajal.nbeats_block", [], "model", "Block", "model.nbeats"),
        ("cajal.linear_latent", ["latent"], "model", "UnifiedModel", "model.z_dim/latent_reg"),
        ("cajal.hierarchical_head", ["classifier"], "model", "HierarchicalHead", None),
        ("cajal.nb_decoder", ["decoder"], "model", "ExpressionDecoder", "model.decoder"),
        ("cajal.unified_decoder", ["decoder"], "model", "UnifiedExpressionDecoder", "model.decoder"),
        (
            "cajal.neighborhood_transformer",
            ["neighborhood"],
            "model",
            "NeighborhoodModel",
            "model.spatial_d/spatial_layers",
        ),
        ("cajal.batch_discriminator", [], "model", "BatchDiscriminator", "model.batch_adversary"),
        ("cajal.mog_prior", [], "model", "MoGPrior", "model.latent_reg/mog_components"),
        ("cajal.nb_likelihood", [], "objectives", "expression_nll", None),
        ("cajal.initial_loss", [], "balance", "InitialLoss", "train.balancer"),
        ("cajal.grad_equal", [], "balance", "GradEqual", "train.balancer/balancer_params"),
        ("cajal.famo", [], "balance", "FAMO", "train.balancer/balancer_params"),
        ("cajal.cond_hsic", [], "regularizers", "CondHSIC", "train.objectives"),
        ("cajal.sigreg", [], "regularizers", "SIGReg", "train.objectives"),
        ("cajal.batch_swap", [], "regularizers", "BatchSwap", "train.objectives"),
        ("cajal.teacher_gram", [], "regularizers", "TeacherGram", "train.objectives + named frozen teacher"),
        ("cajal.batch_adversary", [], "regularizers", "BatchAdversary", "train.objectives"),
        ("cajal.grouped_muon", [], "optim", "GroupedMuon", "train.optimizer/optimizer_params"),
        ("cajal.kl_capacity", [], "balance", "CapacityWeight", "train.kl_target/kl_dual_lr"),
        ("cajal.budgeted_weight", [], "balance", "DualWeight", "train.objectives.budget/dual_lr"),
    ]
    result = []
    for name, slots, module, symbol, field in items:
        result.append(
            {
                "id": name,
                "source_family": "Cajal",
                "source": f"brainwm_unified.{module}.{symbol}",
                "source_file": f"brainwm_unified/{module}.py",
                "source_sha256": sha(ROOT / f"brainwm_unified/{module}.py"),
                "slots": slots,
                "available": True,
                "role": "component",
                "search_field": field,
                "parameter_ranges": {"expansion": C.PARAM_RANGES["expansion"]} if name == "cajal.swiglu" else {},
                "note": "Existing Cajal implementation; empty candidate selections preserve the seed architecture.",
            }
        )
    for name, slots, source in [
        ("scvi.vae_encoder", ["latent"], "scvi.nn.Encoder"),
        ("scvi.nb_decoder", ["decoder"], "scvi.nn.DecoderSCVI"),
        ("scanvi.classifier", ["classifier"], "scvi.module._classifier.Classifier"),
    ]:
        module_name, symbol = source.rsplit(".", 1)
        source_file = inspect.getsourcefile(getattr(importlib.import_module(module_name), symbol))
        result.append(
            {
                "id": name,
                "source_family": "scVI/scANVI",
                "source": source,
                "source_file": source_file,
                "source_sha256": sha(source_file),
                "slots": slots,
                "available": True,
                "role": "component",
                "parameter_ranges": {
                    k: C.PARAM_RANGES[k] for k in ("depth", "dropout") if k != "dropout" or name != "scvi.nb_decoder"
                },
                "note": "Official installed module, adapted to Cajal tensors/objectives; not a complete baseline model.",
            }
        )
    return result


def legacy_entries():
    registry = C.legacy()
    result = []
    token_enc = {
        "mlp",
        "residual_mlp",
        "scbrain_swiglu",
        "scnode_mlp",
        "scvi_fc",
        "sctab_tabnet",
        "scnym_mlp",
        "sedr_mlp",
    }
    token_ops = {
        "linear",
        "layer_norm",
        "rms_norm",
        "gelu",
        "silu",
        "dropout",
        "scbrain_ffn",
        "scfit_ffn",
        "scfit_attention",
        "sctab_glu",
    }
    mixers = {"linear", "layer_norm", "rms_norm", "gelu", "silu", "dropout", "scbrain_ffn", "scfit_ffn"}
    for kind, values in registry.backend_catalog().items():
        for item in values:
            name = item["name"]
            slots = []
            flat = False
            if kind == "encoder" and name in token_enc:
                slots, flat = ["token"], True
            elif kind == "encoder" and name in ("scfoundation_autobin", "scprint_value"):
                slots = ["value_embedding"]
            elif kind == "attention" and name != "identity":
                slots = ["token", "block_attention"]
            elif kind == "operation" and name in token_ops:
                slots = ["token"] + (["block_mixer"] if name in mixers else [])
                flat = name == "sctab_glu"
            elif kind == "residual" and name != "none":
                slots, flat = ["token"], True
            elif kind == "head" and name != "regression":
                slots = ["classifier"]
            elif kind == "spatial" and name != "identity":
                slots = ["neighborhood"]
            factory = registry.registry(kind).resolve(name).factory
            signature = inspect.signature(factory)
            parameters = {k: C.PARAM_RANGES[k] for k in signature.parameters if k in C.PARAM_RANGES}
            source_file = inspect.getsourcefile(factory)
            record = {
                "id": f"legacy.{kind}.{name}",
                "legacy_kind": kind,
                "legacy_name": name,
                "source_family": "nanoBrain existing component library",
                "source": item["source"],
                "source_file": str(Path(source_file).relative_to(ROOT))
                if source_file and Path(source_file).is_relative_to(ROOT)
                else source_file,
                "source_sha256": sha(source_file) if source_file else None,
                "slots": slots,
                "available": bool(slots),
                "flat_input": flat,
                "role": "component",
                "parameter_ranges": parameters,
                "original_contract": item,
                "note": "Frozen original adapter; Cajal token adapters act on one cell's learned gene-token features, not a 256-gene expression dataset.",
            }
            if not slots:
                record["reason"] = (
                    "Preserved in merged source catalogue; requires a different task/input contract (e.g. development, nonnegative expression or a dataset-wide transform)."
                )
            result.append(record)
    return result


def check_entry(name, slot=None):
    import torch

    from brainwm_unified import model as M
    from brainwm_unified.config import ModelConfig

    torch.set_num_threads(2)
    torch.manual_seed(0)
    entry = C.get(name)
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
    spec = {"slot": slot or entry["slots"][0], "name": name, "params": {}}
    cfg.research_components = [spec]
    if name == "cajal.unified_decoder":
        cfg.decoder = "unified"
    model = M.UnifiedModel(cfg)
    # Use the real shared five-objective interfaces on synthetic data only.
    import tempfile

    from brainwm_unified.objectives import compute_losses
    from nanobrain.infra.worker import synthetic

    with tempfile.TemporaryDirectory() as d:
        b, maps, panels = synthetic(d)
        output = model(b, M.dense_mask(b["tok_doc"]), 1)
        losses, _ = compute_losses(model, b, output, panels, maps, 1.0)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite five-task loss")
        loss.backward()
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()):
            raise ValueError("component disconnected shared encoder")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite gradient")
    return {"status": "passed", "synthetic_only": True, "tasks": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-entry")
    parser.add_argument("--probe-slot")
    parser.add_argument("--retry-unavailable", action="store_true")
    args = parser.parse_args()
    if args.probe_entry:
        try:
            result = check_entry(args.probe_entry, args.probe_slot)
        except Exception as exc:  # noqa: BLE001 - each optional source gets an explicit availability receipt
            result = {"status": "unavailable", "reason": type(exc).__name__ + ": " + str(exc)[:1000]}
        print(json.dumps(result))
        return
    entries = native_entries() + legacy_entries()
    if args.retry_unavailable:
        previous = {e["id"]: e for e in json.loads((C.LIBRARY / "catalog.json").read_text())}
        for entry in entries:
            old = previous.get(entry["id"], {})
            verified = old.get("verification", {})
            if old.get("slots") == entry["slots"]:
                by_slot = verified.get("by_slot", {})
                if not by_slot and verified.get("status") == "passed" and entry["slots"]:
                    by_slot = {entry["slots"][0]: verified}
                entry["verification"] = {"by_slot": {s: v for s, v in by_slot.items() if v["status"] == "passed"}}
    for entry in entries:
        if entry["id"] == "legacy.operation.sctab_glu":
            entry["flat_input"] = True
    save(C.LIBRARY / "catalog.json", entries)
    audit = []
    for entry in entries:
        if not entry["slots"]:
            continue
        by_slot = entry.get("verification", {}).get("by_slot", {})
        for slot in entry["slots"]:
            if args.retry_unavailable and by_slot.get(slot, {}).get("status") == "passed":
                continue
            try:
                proc = subprocess.run(
                    [sys.executable, "-B", __file__, "--probe-entry", entry["id"], "--probe-slot", slot],
                    check=False,
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "2",
                        "MPLCONFIGDIR": str(ROOT / "preparation/matplotlib-cache"),
                        "HF_HOME": str(ROOT / "preparation/hf-cache"),
                        "XDG_CACHE_HOME": str(ROOT / "preparation/xdg-cache"),
                        "HF_HUB_OFFLINE": "1",
                        "NUMBA_CACHE_DIR": str(ROOT / "preparation/numba-cache"),
                    },
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                try:
                    result = json.loads(proc.stdout.splitlines()[-1])
                except (ValueError, IndexError):
                    result = {"status": "unavailable", "reason": (proc.stderr + proc.stdout)[-1500:]}
            except subprocess.TimeoutExpired:
                result = {"status": "unavailable", "reason": "CPU interface probe exceeded 180 seconds"}
            by_slot[slot] = result
            print(entry["id"], slot, result["status"], result.get("reason", ""), flush=True)
        entry["available"] = all(v["status"] == "passed" for v in by_slot.values())
        entry["verification"] = {
            "status": "passed" if entry["available"] else "unavailable",
            "by_slot": by_slot,
            "synthetic_only": True,
            "tasks": 5,
        }
        if not entry["available"]:
            entry["reason"] = "; ".join(v.get("reason", "") for v in by_slot.values() if v["status"] != "passed")
        audit.append(dict(id=entry["id"], **entry["verification"]))
    opponents = json.loads((ROOT / "reference/opponents.json").read_text())
    for row in opponents["all_existing_baselines"]:
        entries.append(
            {
                "id": "baseline." + row["method"],
                "source_family": "Cajal existing whole-method benchmark",
                "source": opponents["source_csv"],
                "role": "whole_method_baseline",
                "slots": [],
                "available": True,
                "metrics": {k: v for k, v in row.items() if k != "method"},
                "regime": "transductive" if "transductive" in row["method"].lower() else "existing_Cajal_recipe",
                "promotion_frontier": any(r["method"] == row["method"] for r in opponents["rows"]),
                "note": "Existing complete-method performance; a fitted tree/kNN is not a differentiable Cajal layer.",
            }
        )
    save(C.LIBRARY / "catalog.json", entries)
    save(
        C.LIBRARY / "verification.json",
        {
            "synthetic_only": True,
            "real_training_started": False,
            "entries": [dict(id=e["id"], **e["verification"]) for e in entries if "verification" in e],
            "total_entries": len(entries),
            "runnable_structural_components": sum(bool(e["slots"]) and e["available"] for e in entries),
        },
    )


if __name__ == "__main__":
    main()
