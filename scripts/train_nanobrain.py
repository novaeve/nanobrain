#!/usr/bin/env python3
"""Train an independent nanoBrain task on the frozen 100k metadata-v2 dataset.

The run uses a finite step budget and writes model, validation, and data receipts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.data import (  # noqa: E402
    infer_label_cardinalities,
    load_v3_subset,
    verify_subset_manifest,
)
from nanobrain.genome import CandidateGenome, default_genome  # noqa: E402
from nanobrain.model import NanoBrainModelConfig  # noqa: E402
from nanobrain.training import TrainConfig, train_candidate  # noqa: E402


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get(
            "NANOBRAIN_DATA_DIR",
            ROOT / "assets" / "nanobrain_v3_100k_metadata_v2",
        )),
        help="materialized nanoBrain subset directory",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--task", choices=("annotation", "integration", "development", "spatial_niche"), default="annotation",
                        help="one task per independent model/optimizer/checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--development-cells", type=int, default=64)
    parser.add_argument("--validation-development-groups", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--checkpoint-policy", choices=("all", "final_weights"), default="all")
    parser.add_argument(
        "--loss-balance",
        choices=("ema_equal", "detached_max_equal", "fixed"),
        default="ema_equal",
    )
    parser.add_argument("--loss-balance-decay", type=float, default=0.99)
    parser.add_argument("--integration-adversarial-warmup-steps", type=int, default=500)
    parser.add_argument("--integration-logit-clip", type=float, default=20.0)
    parser.add_argument("--stability-max-loss", type=float, default=1.0e8)
    parser.add_argument("--stability-max-gradient-norm", type=float, default=1.0e6)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--genome-json",
        type=Path,
        default=None,
        help="optional JSON file containing a CandidateGenome dictionary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "nanobrain" / "smoke",
        help="directory for checkpoint and receipt",
    )
    parser.add_argument("--verify", action="store_true", help="verify every subset checksum before loading")
    parser.add_argument("--plugin", action="append", default=[],
                        help="import a module that registers custom model components; repeatable")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.validation_batch_size < 1:
        parser.error("steps and batch sizes must be positive")
    if not 0.0 <= args.loss_balance_decay < 1.0:
        parser.error("--loss-balance-decay must be in [0, 1)")

    plugins = []
    for name in args.plugin:
        try:
            module = importlib.import_module(name)
        except (ImportError, ValueError) as exc:
            parser.error(f"cannot import plugin {name!r}: {exc}")
        path = Path(module.__file__) if module.__file__ else None
        plugins.append({
            "module": name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            if path and path.is_file() else None,
        })

    started = time.perf_counter()
    manifest = verify_subset_manifest(
        args.data,
        verify_hashes=args.verify,
        data_splits=("train", "validation"),
    )
    splits = load_v3_subset(args.data, verify=False, include_test=False)
    cardinalities = infer_label_cardinalities(splits)
    development_dim = (
        splits.train.input_dim
        if splits.development_train is not None
        else int(splits.train.labels["development"].shape[-1])
    )
    condition_dim = int(
        splits.train.development_condition.shape[-1]
        if splits.train.development_condition is not None
        and splits.train.development_condition.ndim > 1
        else 1
    )
    model_config = NanoBrainModelConfig(
        input_dim=splits.train.input_dim,
        annotation_classes=cardinalities["annotation"],
        niche_classes=cardinalities["niche"],
        integration_classes=cardinalities["integration"],
        development_dim=development_dim,
        development_condition_dim=condition_dim,
    )
    if args.genome_json is None:
        genome = default_genome(seed=args.seed)
    else:
        try:
            genome_payload = json.loads(args.genome_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --genome-json {args.genome_json}: {exc}")
        if not isinstance(genome_payload, dict):
            parser.error("--genome-json must contain a JSON object")
        try:
            genome = CandidateGenome.from_dict(genome_payload)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid candidate genome: {exc}")
    train_config = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        seed=args.seed,
        device=args.device,
        development_cells=args.development_cells,
        validation_development_groups=args.validation_development_groups,
        eval_interval=args.eval_interval,
        precision=args.precision,
        loss_balance_mode=args.loss_balance,
        loss_balance_decay=args.loss_balance_decay,
        integration_adversarial_warmup_steps=args.integration_adversarial_warmup_steps,
        integration_logit_clip=args.integration_logit_clip,
        stability_max_loss=args.stability_max_loss,
        stability_max_gradient_norm=args.stability_max_gradient_norm,
        task=args.task,
        checkpoint_policy=args.checkpoint_policy,
    )
    receipt = train_candidate(
        splits,
        model_config,
        genome,
        train_config,
        output_dir=args.output,
    )
    receipt["data_manifest"] = {
        "path": str(Path(args.data).resolve() / "manifest.json"),
        "format": manifest["format"],
        "split_id": manifest["split_id"],
        "rows": {
            split: int(manifest["splits"][split]["rows"])
            for split in ("train", "validation", "test")
        },
        "source_shards": manifest["source_shards"],
        "h5ad_in_training_loop": False,
        "loaded_splits": ["train", "validation"],
        "test_opened": False,
    }
    receipt["label_cardinalities"] = cardinalities
    receipt["plugins"] = plugins
    receipt["elapsed_wall_seconds"] = time.perf_counter() - started
    # Preserve a failed-stability receipt so the campaign can record and skip
    # the candidate instead of mistaking an early-stop run for a valid model.
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / f"receipt-{receipt['candidate_id']}-steps{args.steps}.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"receipt": str(receipt_path), **receipt}, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
