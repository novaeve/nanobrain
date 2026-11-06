"""Joint pretraining loop: all five objectives on every step.

Writes `train.jsonl` (per-task losses, weights, metrics, MoE load, speed),
`grad_probe.jsonl` (per-task gradient norms and cosines on the encoder) and
`latest.pt` (resumable state) under the output directory.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from . import experiment, record
from . import model as M
from .balance import CapacityWeight, DualWeight, build_balancer
from .config import DataConfig, TrainConfig, to_dict
from .data import BatchStream, Subset, load_level_maps, restrict
from .objectives import TASKS, compute_losses
from .optim import GroupedMuon
from .regularizers import OBJECTIVES

MUON = {"muon": torch.optim.Muon, "grouped_muon": GroupedMuon}


def lr_lambda(cfg: TrainConfig):
    def f(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        p = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    return f


def task_weights_at(cfg: TrainConfig, step: int) -> dict:
    """The task weights in force at `step`: the first schedule phase not yet over, else task_weights."""
    for phase in cfg.task_schedule:
        if step < phase["until"]:
            return phase["task_weights"]
    return cfg.task_weights


def to_device(b: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def probe_group(name: str) -> str:
    """Encoder parameter -> layer group: embed, prelude.i, core.i, coda.i, or other."""
    parts = name.split(".")
    if parts[0] in ("prelude", "core", "coda"):
        return f"{parts[0]}.{parts[1]}"
    if parts[0] in ("gene_emb", "value_emb", "mask_emb"):
        return "embed"
    return "other"


def grad_probe(model, losses: dict) -> dict:
    """Per-loss gradient norms and pairwise cosines on the whole encoder and per layer group.

    Covers every entry of `losses`: the five tasks plus any extra objectives. Whole-encoder keys are `norm/<task>` and `cos/<a>~<b>`; per-group keys add the
    group, e.g. `cos/core.0/gene~spatial`, so a conflict confined to some layers is
    not averaged away by the others.
    """
    named = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = [probe_group(n) for n, _ in named]
    names = list(losses)
    flat = {}
    for t in names:
        grads = torch.autograd.grad(losses[t], params, retain_graph=True, allow_unused=True)
        flat[t] = [(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(grads, params)]

    def summarize(select, prefix):
        vec = {t: torch.cat([v for v, keep in zip(flat[t], select) if keep]) for t in names}
        out = {f"norm/{prefix}{t}": vec[t].norm().item() for t in names}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                out[f"cos/{prefix}{a}~{b}"] = torch.nn.functional.cosine_similarity(vec[a], vec[b], dim=0).item()
        return out

    out = summarize([True] * len(params), "")
    for g in dict.fromkeys(groups):
        out.update(summarize([x == g for x in groups], f"{g}/"))
    return out


def moe_load_stats(model) -> dict:
    """Max/mean expert load per MoE layer and slot since the last bias update."""
    stats = {}
    for li, moe in enumerate(model.encoder.moe_layers()):
        for s in range(moe.load.shape[0]):
            load = moe.load[s]
            if load.sum() > 0:
                stats[f"moe/L{li}s{s}"] = (load.max() / load.mean()).item()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", help="experiment TOML (configs/experiments/*.toml) without a sweep; flags below override it"
    )
    ap.add_argument("--data", default=None, help=f"data root (default {DataConfig.root})")
    ap.add_argument("--out", default=None, help="output directory (default: the config's run.out)")
    ap.add_argument("--steps", type=int, default=None, help=f"training steps (default {TrainConfig.steps})")
    ap.add_argument("--value-scale", default="counts", choices=["counts", "log1p_cp10k"])
    ap.add_argument(
        "--set", nargs="*", action="extend", default=[], help="config overrides (repeatable), e.g. model.d_model=512"
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args(argv)

    doc = experiment.load(args.config) if args.config else {}
    try:  # a sweep config runs one member, chosen by --set (as `cajal run` launches it)
        doc = experiment.pin_sweep(doc, args.set)
    except ValueError as e:
        raise SystemExit(f"{args.config}: {e}") from None
    flags = [f"data.root={json.dumps(args.data)}"] * bool(args.data) + [f"train.steps={args.steps}"] * bool(args.steps)
    doc = experiment.override(doc, [*flags, *args.set])
    mcfg, dcfg, tcfg = experiment.build(doc)
    tcfg.out_dir = args.out or doc.get("run", {}).get("out")
    if not tcfg.out_dir:
        raise SystemExit("give --out or a config with run.out")
    if args.no_compile:
        tcfg.compile = False

    ds = restrict(Subset(dcfg.root, args.value_scale), dcfg.cell_fraction)
    level_maps = {
        "cell_type": load_level_maps(os.path.join(dcfg.root, "cell_type_vocab.json")),
        "anatomy": load_level_maps(os.path.join(dcfg.root, "anatomy_vocab.json")),
    }
    mcfg.cell_type_levels = [int(m.max()) + 1 for m in level_maps["cell_type"]]
    mcfg.anatomy_levels = [int(m.max()) + 1 for m in level_maps["anatomy"]]
    mcfg.n_batches = int(ds.integration_batch.max()) + 1
    if mcfg.likelihood == "nb" and args.value_scale != "counts":
        mcfg.likelihood = "mse_log"

    device = torch.device("cuda")
    os.makedirs(tcfg.out_dir, exist_ok=True)
    with open(os.path.join(tcfg.out_dir, "config.json"), "w") as fh:
        json.dump(to_dict(mcfg, dcfg, tcfg), fh, indent=2)
    record.write(
        tcfg.out_dir,
        "train",
        experiment=args.config,
        run=doc.get("run", {}),
        argv=argv or sys.argv[1:],
        config=to_dict(mcfg, dcfg, tcfg),
        data=record.data_fingerprint(dcfg.root),
        status="running",
    )

    torch.manual_seed(dcfg.seed)  # initial weights follow the run's seed, like its data order
    model = M.UnifiedModel(mcfg).to(device)
    if mcfg.input_norm == "pearson":  # residual parameters from the pretraining set (a resume reloads them)
        model.encoder.gene_rate[: mcfg.n_genes] = torch.from_numpy(ds.gene_rates(mcfg.n_genes)).to(device)
        model.encoder.residual_clip.fill_(math.sqrt(ds.n_cells))
    if tcfg.compile:
        M.enable_compiled_flex()
    panels = ds.panel_matrix(mcfg.n_genes).to(device)
    maps = {k: [torch.as_tensor(m, device=device) for m in v] for k, v in level_maps.items()}

    muon_params, adamw_params = M.param_groups(model)
    if tcfg.optimizer == "muon" and any(p.ndim == 3 for p in muon_params):
        raise ValueError("torch Muon only takes 2-D weights; stacked expert weights need train.optimizer=grouped_muon")
    from cajal_research.plugins import make_optimizers

    opts = make_optimizers(model, tcfg, MUON, (muon_params, adamw_params))
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda(tcfg)) for o in opts]
    balancer = build_balancer(tcfg.balancer, tcfg.norm_steps, tcfg.balancer_params)

    extras, extra_weights, duals, starts = {}, {}, {}, {}
    for spec in tcfg.objectives:
        spec = dict(spec)
        name, weight = spec.pop("name"), float(spec.pop("weight", 1.0))
        starts[name] = int(spec.pop("start", 0))
        budget, dual_lr = spec.pop("budget", None), spec.pop("dual_lr", None)
        extras[name], extra_weights[name] = OBJECTIVES.build(name, **spec), weight
        if budget:
            duals[name] = DualWeight(budget, init=weight, **({"lr": dual_lr} if dual_lr else {}))
    kl_dual = CapacityWeight(tcfg.kl_target, init=tcfg.kl_weight, lr=tcfg.kl_dual_lr) if tcfg.kl_target else None

    step = 0
    ckpt_path = os.path.join(tcfg.out_dir, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["opts"]):
            o.load_state_dict(s)
        for sc, s in zip(scheds, state["scheds"]):
            sc.load_state_dict(s)
        balancer.load_state_dict(state["balancer"] if "balancer" in state else state["normalizer"])
        for name, s in state.get("duals", {}).items():
            duals[name].load_state_dict(s)
        if kl_dual and "kl_dual" in state:
            kl_dual.load_state_dict(state["kl_dual"])
        step = state["step"]
        print(f"resumed at step {step}", flush=True)

    loader = torch.utils.data.DataLoader(
        BatchStream(ds, dcfg, mcfg.n_genes, seed_offset=step, total_steps=tcfg.steps),
        batch_size=None,
        num_workers=tcfg.num_workers,
        pin_memory=True,
        persistent_workers=tcfg.num_workers > 0,
        prefetch_factor=4 if tcfg.num_workers else None,
    )
    rng = np.random.default_rng([dcfg.seed, step, 1])
    train_log = os.path.join(tcfg.out_dir, "train.jsonl")
    probe_path = os.path.join(tcfg.out_dir, "grad_probe.jsonl")
    with open(train_log, "a") as log, open(probe_path, "a") as probe_log:
        print(f"params {M.count_params(model) / 1e6:.2f}M, cells {ds.n_cells}", flush=True)

        model.train()
        t_last = time.time()
        for batch in loader:
            if step >= (tcfg.stop_at or tcfg.steps):
                break
            b = M.pad_tokens(to_device(batch, device))
            attn_mask = M.attention_layout(b["tok_doc"], mcfg.attention_backend)
            loops = int(rng.integers(mcfg.min_loops, mcfg.max_loops + 1))
            kl_weight = (kl_dual.weight if kl_dual else tcfg.kl_weight) * min(1.0, step / max(1, tcfg.kl_warmup_steps))

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b, attn_mask, loops)
                active = {n: f for n, f in extras.items() if step >= starts[n]}
                losses, metrics = compute_losses(model, b, out, panels, maps, kl_weight, active)
            if kl_dual:
                kl_dual.update(metrics["integration/kl"])
            scales = balancer.scales(step, losses, model)
            weights = {n: extra_weights[n] for n in active}
            weights.update({name: d.update(losses) for name, d in duals.items() if name in active})
            task_weights = task_weights_at(tcfg, step)
            total = sum(task_weights[t] * scales[t] * losses[t] for t in TASKS)
            total = total + sum(w * scales.get(name, 1.0) * losses[name] for name, w in weights.items())

            if tcfg.grad_probe_every and step % tcfg.grad_probe_every == 0:
                probe = grad_probe(model, losses)
                probe_log.write(json.dumps({"step": step, **probe}) + "\n")
                probe_log.flush()

            for o in opts:
                o.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            for o in opts:
                o.step()
            for sc in scheds:
                sc.step()
            if balancer.wants_after_step:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    after, _ = compute_losses(model, b, model(b, attn_mask, loops), panels, maps, kl_weight, active)
                balancer.after_step(after)

            if step % tcfg.log_every == 0:
                now = time.time()
                rec = {
                    "step": step,
                    "loops": loops,
                    "lr": scheds[0].get_last_lr()[0],
                    "kl_weight": kl_weight,
                    "total": total.item(),
                    "grad_norm": grad_norm.item(),
                    **{f"loss/{t}": losses[t].item() for t in (*TASKS, *active)},
                    **{f"scale/{t}": v for t, v in scales.items()},
                    **{f"weight/{name}": w for name, w in weights.items()},
                    **{f"famo/{t}": v for t, v in getattr(balancer, "weights", dict)().items()},
                    **{k: float(v) for k, v in metrics.items()},
                    **moe_load_stats(model),
                    "cells": int(b["tok_cls"].sum()),
                    "tokens": int((b["tok_doc"] >= 0).sum()),
                    "sec_per_step": (now - t_last) / (tcfg.log_every if step else 1),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                t_last = now
            for moe in model.encoder.moe_layers():
                moe.update_bias(mcfg.router_bias_rate)

            step += 1
            if step % tcfg.ckpt_every == 0 or step in (tcfg.steps, tcfg.stop_at):
                state = {
                    "model": model.state_dict(),
                    "opts": [o.state_dict() for o in opts],
                    "scheds": [s.state_dict() for s in scheds],
                    "balancer": balancer.state_dict(),
                    "duals": {name: d.state_dict() for name, d in duals.items()},
                    "kl_dual": kl_dual.state_dict() if kl_dual else None,
                    "step": step,
                    "config": to_dict(mcfg, dcfg, tcfg),
                }
                torch.save(state, ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
                if step % (tcfg.keep_every or 10 * tcfg.ckpt_every) == 0:
                    keep = {k: state[k] for k in ("model", "step", "config")} if tcfg.keep_weights_only else state
                    torch.save(keep, os.path.join(tcfg.out_dir, f"step{step}.pt"))
                record.write(
                    tcfg.out_dir,
                    "train",
                    step=step,
                    status="done" if step == (tcfg.stop_at or tcfg.steps) else "running",
                )


if __name__ == "__main__":
    main()
