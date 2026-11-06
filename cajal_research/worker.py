"""Trusted worker entrypoints; each process loads torch OR the scoring stack."""

import argparse
import json
import resource
import runpy
import sys
from pathlib import Path

from .contracts import ROOT, TASKS, load_config, resolved_document, save, sha, validate_bundle


def synthetic(root):
    import numpy as np
    import torch

    from brainwm_unified.config import DataConfig
    from brainwm_unified.data import Subset, build_batch, load_level_maps

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n, g, k = 12, 24, 4
    rng = np.random.default_rng(27)
    x = rng.poisson(2.0, (n, g)).astype(np.float32)
    spatial = rng.normal(size=(n, 3)).astype(np.float32)
    neighbors = np.array([[(i + j + 1) % n for j in range(k)] for i in range(n)], dtype=np.int64)
    arrays = {
        "values": x.ravel(),
        "gene_index": np.tile(np.arange(g, dtype=np.int32), n),
        "row_ptr": np.arange(n + 1, dtype=np.int64) * g,
        "cell_type": np.arange(n) % 3,
        "anatomy": np.arange(n) % 2,
        "integration_batch": np.arange(n) % 2,
        "panel_id": np.zeros(n, np.int64),
        "spatial": spatial,
        "neighbors": neighbors,
        "neighbor_dist": np.ones((n, k), np.float32),
        "panel_gene_ptr": np.array([0, g]),
        "panel_gene_index": np.arange(g),
    }
    for key, value in arrays.items():
        np.save(root / (key + ".npy"), value)
    save(root / "cell_type_vocab.json", {"levels": ["cell_type"], "cell_type": [{"index": i} for i in range(3)]})
    save(root / "anatomy_vocab.json", {"levels": ["anatomy"], "anatomy": [{"index": i} for i in range(2)]})
    ds = Subset(str(root))
    cfg = DataConfig(root=str(root), dissociated_cells=3, spatial_groups=2, neighbors=k, zero_tokens=2)
    b = build_batch(ds, cfg, g, np.random.default_rng(0))
    maps = {
        name: [torch.as_tensor(v) for v in load_level_maps(str(root / (name + "_vocab.json")))]
        for name in ("cell_type", "anatomy")
    }
    return b, maps, ds.panel_matrix(g)


def probe(bundle):
    import copy

    import torch

    from brainwm_unified import model as M
    from brainwm_unified.config import ModelConfig, TrainConfig
    from brainwm_unified.objectives import compute_losses

    from .contracts import reference_document
    from .plugins import make_optimizers

    validate_bundle(bundle)
    torch.set_num_threads(2)
    torch.manual_seed(11)
    payload = {"plugins": bundle["plugins"], "files": bundle["files"]} if bundle["plugins"] else {}
    base = reference_document()["ModelConfig"]
    full = ModelConfig(**{**base, **bundle["model"], "compile_blocks": False, "research_plugins": payload})
    # Count the actual full Cajal candidate before the small synthetic numerical probe.
    big = M.UnifiedModel(full)
    count = sum(p.numel() for p in big.parameters())
    del big
    if count > 200_000_000:
        raise ValueError("candidate exceeds 200M trainable parameters")
    cfg = copy.deepcopy(full)
    cfg.n_genes, cfg.n_batches = 24, 2
    cfg.cell_type_levels, cfg.anatomy_levels = [3], [2]
    cfg.d_model, cfg.n_heads, cfg.expert_hidden = 32, 4, 32
    cfg.n_prelude = cfg.n_core = cfg.n_coda = 1
    cfg.n_hash_layers = min(cfg.n_hash_layers, 1)
    cfg.n_experts, cfg.top_k = min(cfg.n_experts, 4), min(cfg.top_k, 4)
    cfg.z_dim, cfg.decoder_hidden, cfg.spatial_d, cfg.spatial_layers = 8, 32, 16, 1
    cfg.loops = cfg.min_loops = 1
    cfg.max_loops = 2
    cfg.activation_checkpointing, cfg.attention_backend, cfg.moe_backend = False, "flex", "loop"
    b, maps, panels = synthetic("/work/synthetic")
    model = M.UnifiedModel(cfg)
    if cfg.input_norm == "pearson":
        model.encoder.gene_rate.fill_(1.0 / 24)
        model.encoder.residual_clip.fill_(4.0)
    mask = M.dense_mask(b["tok_doc"])
    train_cfg = TrainConfig(**bundle["train"], research_plugins=payload)
    if train_cfg.optimizer in ("grouped_muon", "muon"):
        # CPU tensor contract probe; production retains Cajal's actual grouped Muon path.
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    else:
        (optimizer,) = make_optimizers(model, train_cfg, {}, ([], []))
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    out = model(b, mask, loops=1)
    losses, _ = compute_losses(model, b, out, panels, maps, 1.0)
    if set(losses) != set(TASKS) or not all(torch.isfinite(v) for v in losses.values()):
        raise ValueError("all five Cajal objectives must be finite")
    for task in TASKS:
        grads = torch.autograd.grad(
            losses[task], list(model.encoder.parameters()), retain_graph=True, allow_unused=True
        )
        if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
            raise ValueError(f"{task} must train the shared encoder")
    sum(losses.values()).backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite gradient")
    optimizer.step()
    if not any(not torch.equal(before[k], p) for k, p in model.named_parameters()):
        raise ValueError("optimizer made no update")
    for name, p in model.encoder.research_transforms.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError(f"disconnected plugin parameter {name}")
    model.eval()
    with torch.no_grad():
        expected = model(b, mask, 1)
        saved = Path("/work/probe.pt")
        import dataclasses

        from brainwm_unified import infer

        torch.save({"model": model.state_dict(), "step": 1, "config": {"ModelConfig": dataclasses.asdict(cfg)}}, saved)
        restored, _ = infer.load_model(str(saved), device="cpu", attention_backend="flex")
        actual = restored(b, mask, 1)
        if not all(torch.allclose(expected[k], actual[k], atol=1e-6) for k in ("mu", "gene_pred")):
            raise ValueError("checkpoint reload changed predictions")
    from brainwm_unified import record

    return {
        "status": "accepted",
        "full_parameter_count": count,
        "tasks": list(TASKS),
        "shared_encoder_gradients": True,
        "optimizer_update": True,
        "checkpoint_reload": True,
        "synthetic_only": True,
        "benchmark_data_mounted": False,
        "gpu_used": False,
        "code_fingerprint": record.fingerprint(),
    }


def script(name, argv):
    sys.argv = [name, *argv]
    sys.path.insert(0, str(ROOT / "scripts"))
    runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "train", "predict", "score"])
    ap.add_argument("--bundle")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task", choices=["annotation", "integration", "genes", "spatial"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--prediction")
    ap.add_argument("--config", default=str(ROOT / "configs/cajal_react.json"))
    args = ap.parse_args()
    if args.mode == "probe":
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
        result = probe(json.loads(Path(args.bundle).read_text()))
    elif args.mode == "train":
        from brainwm_unified import train

        cfg = load_config(args.config)
        bundle = json.loads(Path(args.bundle).read_text())
        doc = resolved_document(bundle, cfg, args.seed)
        overrides = [
            f"{section}.{k}={json.dumps(v)}"
            for section, values in doc.items()
            for k, v in values.items()
            if not (section == "train" and k == "out_dir")
        ]
        resume = ["--resume"] if Path("/work/train/latest.pt").is_file() else []
        train.main(["--out", "/work/train", *resume, "--set", *overrides])
        receipt = json.loads(Path("/work/train/record.json").read_text())
        if (
            receipt.get("status") != "done"
            or receipt.get("step") != cfg["stop_at"]
            or receipt["fingerprint"]["git_dirty"]
        ):
            raise ValueError("training receipt is incomplete or code is dirty")
        result = {
            "status": "complete",
            "seed": args.seed,
            "step": cfg["stop_at"],
            "record": receipt,
            "checkpoint": "train/latest.pt",
            "shared_model_count": 1,
            "artifact_hashes": {"train/latest.pt": sha("/work/train/latest.pt")},
        }
    elif args.mode == "predict":
        cfg = load_config(args.config)
        key = "annotation" if args.task == "genes" else args.task
        script(
            "predict_eval.py",
            [
                "--checkpoint",
                args.checkpoint,
                "--data",
                cfg["data"][key],
                "--out",
                "/work/prediction.npz",
                "--seed",
                str(cfg["evaluation_seed"]),
                "--token-budget",
                str(cfg["token_budget"]),
                "--task",
                "embed" if args.task in ("annotation", "integration") else args.task,
            ],
        )
        artifacts = {p.name: sha(p) for p in Path("/work").glob("*.npz")}
        result = {"status": "complete", "task": args.task, "prediction": "prediction.npz", "artifact_hashes": artifacts}
    else:
        cfg = load_config(args.config)
        key = "annotation" if args.task == "genes" else args.task
        common = [
            "--data",
            cfg["data"][key],
            "--pretrain",
            cfg["data"]["pretrain"],
            "--pred",
            "candidate=" + args.prediction,
            "--out",
            "/work/scores",
        ]
        if args.task in ("annotation", "integration"):
            script("eval_zero_shot.py", [*common, "--cache", str(ROOT / "reference/baseline_cache")])
        else:
            import shutil

            cache = Path("/work/imputation-cache")
            shutil.copytree(ROOT / "reference/imputation_cache", cache, dirs_exist_ok=True)
            if args.task == "genes":
                mask = Path(args.prediction).parent / (
                    "genes_mask_" + Path(cfg["data"][key]).name + "_seed" + str(cfg["evaluation_seed"]) + ".npz"
                )
                common += ["--mask", str(mask)]
            script(
                "eval_imputation.py",
                [*common, "--task", args.task, "--cache", str(cache), "--seed", str(cfg["evaluation_seed"])],
            )
        result = {
            "status": "complete",
            "task": args.task,
            "scores": "scores/summary.csv",
            "artifact_hashes": {"scores/summary.csv": sha("/work/scores/summary.csv")},
        }
    save("/work/receipt.json", result)


if __name__ == "__main__":
    main()
