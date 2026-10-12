import json

import torch

from nanobrain.data import DatasetSplits, make_synthetic_splits
from nanobrain.genome import CandidateGenome
from nanobrain.model import NanoBrainModelConfig
from nanobrain.training import TrainConfig, train_candidate


def test_final_weights_policy_preserves_metrics_and_final_step_checkpoint(tmp_path):
    splits = make_synthetic_splits(train_rows=12, validation_rows=8, test_rows=4,
        input_dim=8, annotation_classes=3, niche_classes=2, integration_classes=2, development_dim=3)
    genome = CandidateGenome(hidden_dim=32, encoder="identity", attention="identity",
        spatial="identity", residual="none", flow="none")
    receipt = train_candidate(DatasetSplits(splits.train, splits.validation, None),
        NanoBrainModelConfig(8, 3, 2, 2, development_dim=3), genome,
        TrainConfig(steps=2, batch_size=6, eval_interval=1, device="cpu",
                    task="annotation", checkpoint_policy="final_weights"), output_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    assert len(receipt["validation_history"]) == 2 and receipt["best_checkpoint"] is None
    checkpoint = torch.load(receipt["checkpoint"], weights_only=False)
    assert checkpoint["step"] == 2 and "optimizer" not in checkpoint
    assert receipt["optimizer_changed_parameter"]


def test_training_streams_validation_and_versioned_checkpoints(tmp_path):
    splits = make_synthetic_splits(
        train_rows=24,
        validation_rows=12,
        test_rows=8,
        input_dim=8,
        annotation_classes=3,
        niche_classes=2,
        integration_classes=2,
        development_dim=3,
        seed=17,
    )
    splits = DatasetSplits(splits.train, splits.validation, None)
    genome = CandidateGenome(
        attention="identity",
        attention_heads=1,
        encoder="identity",
        encoder_depth=1,
        flow="none",
        flow_depth=1,
        hidden_dim=32,
        pairing="identity",
        residual="none",
        spatial="identity",
        seed=17,
    )
    config = NanoBrainModelConfig(
        input_dim=8,
        annotation_classes=3,
        niche_classes=2,
        integration_classes=2,
        development_dim=3,
        development_condition_dim=1,
    )

    receipt = train_candidate(
        splits,
        config,
        genome,
        TrainConfig(
            steps=2,
            batch_size=8,
            validation_batch_size=6,
            device="cpu",
            eval_interval=1,
            precision="fp32",
        ),
        output_dir=tmp_path,
    )

    history = [
        json.loads(line)
        for line in (tmp_path / "training_history.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in history].count("validation") == 2
    assert history[0]["event"] == "run_start"
    assert history[-1]["event"] == "run_complete"
    assert all(record.get("test_opened") is False for record in history if "test_opened" in record)
    versions = sorted(tmp_path.glob(f"candidate-{genome.genome_id}-step-*.pt"))
    assert len(versions) == 2
    latest = torch.load(
        tmp_path / f"candidate-{genome.genome_id}-latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert latest["step"] == 2
    assert "optimizer" in latest and "loss_balancer" in latest
    assert receipt["best_checkpoint"] is not None
    assert receipt["test_opened"] is False
    best_record = max(receipt["validation_history"], key=lambda record: record["quality"])
    best = torch.load(receipt["best_checkpoint"], map_location="cpu", weights_only=False)
    selected_version = torch.load(
        tmp_path / f"candidate-{genome.genome_id}-step-{int(best_record['step']):08d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert best["step"] == int(best_record["step"])
    assert best["validation_quality"] == best_record["quality"]
    assert all(
        torch.equal(best["model"][name], selected_version["model"][name])
        for name in best["model"]
    )
