import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from nanobrain.data import DatasetSplits, make_synthetic_splits
from nanobrain.development import DevelopmentPairDataset
from nanobrain.genome import CandidateGenome
from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
from nanobrain.task_training import TASKS, configure_task, row_task_loss
from nanobrain.training import TrainConfig, train_candidate, _seed_everything
from scripts.run_nanobrain_baselines import _run_nanobrain


def fixture_data():
    splits = make_synthetic_splits(train_rows=24, validation_rows=12, test_rows=8,
        input_dim=8, annotation_classes=3, niche_classes=2, integration_classes=2,
        development_dim=8, seed=17)
    for dataset in (splits.train, splits.validation):
        dataset.completion_target = dataset.features.clone()
        dataset.completion_mask = torch.zeros_like(dataset.features, dtype=torch.bool)
        dataset.completion_mask[:, 0] = True
        dataset.features = dataset.features.clone().masked_fill(dataset.completion_mask, 0)
        dataset.feature_observed = ~dataset.completion_mask
    def pairs(dataset, split):
        n = dataset.size // 2
        return DevelopmentPairDataset(dataset, np.arange(n), np.arange(n, 2 * n),
            np.array([0, n]), [{"source_time_days": 10.0, "target_time_days": 20.0,
                               "interval_id": "DEV-FIXTURE", "uberon_id": "UBERON:fixture", "cells": n}], split=split)
    return DatasetSplits(splits.train, splits.validation, None,
                         pairs(splits.train, "train"), pairs(splits.validation, "validation"))


@pytest.mark.parametrize("task", TASKS)
def test_independent_training_updates_only_task_path_and_evaluates_only_that_task(tmp_path, task):
    splits = fixture_data()
    genome = CandidateGenome(hidden_dim=32, encoder_depth=1, attention="identity", attention_heads=1,
                             spatial="identity", residual="none", pairing="identity", flow="mlp", seed=17)
    config = NanoBrainModelConfig(input_dim=8, annotation_classes=3, niche_classes=2,
                                 integration_classes=2, development_dim=8)
    _seed_everything(genome.seed)
    initial = NanoBrainModel(config, genome)
    receipt = train_candidate(splits, config, genome,
        TrainConfig(steps=2, batch_size=8, validation_batch_size=12,
                    validation_development_groups=1, development_cells=4,
                    precision="fp32", device="cpu", task=task), output_dir=tmp_path)
    payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=False)
    assert receipt["optimizer_changed_parameter"] and receipt["training_task"] == task
    assert receipt["loss_balance"]["mode"] == "fixed"
    trainable = set(receipt["trainable_parameters"])
    changed = {name for name, value in initial.state_dict().items() if not torch.equal(value, payload["model"][name])}
    assert changed and changed <= trainable
    unrelated = {"annotation": "niche_head", "spatial_niche": "annotation_head",
                 "integration": "annotation_head", "development": "annotation_head",
                 "spatial_completion": "annotation_head"}[task]
    assert all(torch.equal(v, payload["model"][n]) for n, v in initial.state_dict().items() if n.startswith(unrelated))
    assert task in receipt["train_steps"][-1]
    assert not ((set(TASKS) - {task}) & set(receipt["train_steps"][-1]))
    if task == "integration":
        assert "integration_reconstruction" in receipt["train_steps"][-1]
    rows = _run_nanobrain(splits, Path(receipt["checkpoint"]), tmp_path / "evaluation",
                         "cpu", 12, evaluation_seed=5)
    assert rows and {row["task"] for row in rows} == {task}
    assert {row["training_regime"] for row in rows} == {"independent_task_from_scratch"}
    assert not any(row["status"] in {"error", "failed"} for row in rows), rows
    if task == "annotation":
        bundle = tmp_path / "task_checkpoints.json"
        bundle.write_text(json.dumps(dict(format="nanobrain_task_checkpoints_v1", genome=genome.to_dict(),
            tasks={task: dict(path=receipt["checkpoint"], sha256=hashlib.sha256(Path(receipt["checkpoint"]).read_bytes()).hexdigest())})))
        bundled = _run_nanobrain(splits, bundle, tmp_path / "bundle_evaluation", "cpu", 12, evaluation_seed=5)
        assert {(r["task"], r["metric"]) for r in bundled} == {(r["task"], r["metric"]) for r in rows}
        Path(receipt["checkpoint"]).write_bytes(b"corrupt checkpoint")
        with pytest.raises(ValueError, match="checksum mismatch"):
            _run_nanobrain(splits, bundle, tmp_path / "tampered", "cpu", 12)


def test_integration_objective_does_not_read_other_tasks_labels():
    splits = fixture_data()
    genome = CandidateGenome(hidden_dim=32, attention="identity", spatial="identity", dropout=0, seed=5)
    model = NanoBrainModel(NanoBrainModelConfig(8, 3, 2, 2, 8), genome)
    configure_task(model, "integration")
    batch = splits.train.batch(np.arange(8))
    original, _ = row_task_loss(model, batch, "integration")
    batch.labels = {"integration": batch.labels["integration"]}
    batch.label_masks = {"integration": batch.label_masks["integration"]}
    without_other_labels, _ = row_task_loss(model, batch, "integration")
    torch.testing.assert_close(original.total, without_other_labels.total)


def test_public_protocol_keeps_four_independent_tasks_and_100k_split():
    from pathlib import Path
    p = json.loads(Path("configs/nanobrain_public_100k.json").read_text())
    assert p["training_mode"] == "independent"
    assert len(p["tasks"]) == 4 and "spatial_completion" not in p["tasks"]
    assert p["rows"] == {"train": 60000, "validation": 20000, "test": 20000}
