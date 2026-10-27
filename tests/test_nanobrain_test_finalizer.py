import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import finalize_nanobrain_test_campaign as finalizer


FIELDS = [
    "split_id",
    "subset",
    "task",
    "method",
    "method_family",
    "training_regime",
    "fit_rows",
    "eval_rows",
    "status",
    "runtime_seconds",
    "prediction_artifact",
    "metric",
    "value",
    "direction",
    "stratum_type",
    "stratum_value",
    "notes",
    "error",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _row(method: str, task: str, metric: str, value: float) -> dict[str, str]:
    return {
        "split_id": finalizer.SPLIT_ID,
        "subset": finalizer.SUBSET,
        "task": task,
        "method": method,
        "method_family": "test",
        "training_regime": "fit",
        "fit_rows": "60",
        "eval_rows": "20",
        "status": "complete",
        "runtime_seconds": "1",
        "prediction_artifact": "prediction.npz",
        "metric": metric,
        "value": str(value),
        "direction": "min" if metric in finalizer.MINIMIZE_METRICS else "max",
        "stratum_type": "overall",
        "stratum_value": "overall",
        "notes": "fixture",
        "error": "",
    }


def _write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _manifest(data_hash: str, **extra) -> dict:
    return {
        "split_id": finalizer.SPLIT_ID,
        "fit_split": "train",
        "evaluation_split": "test",
        "test_opened": True,
        "data_manifest_sha256": data_hash,
        **extra,
    }


def _campaign_fixture(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "final"
    campaign = parent / "test-v3"
    data_manifest = tmp_path / "manifest.json"
    data_manifest.write_text('{"format":"fixture"}\n', encoding="utf-8")
    data_hash = hashlib.sha256(data_manifest.read_bytes()).hexdigest()
    protocol_id = "TEST-PROTOCOL"
    seeds = [11, 22, 33]
    _write_json(parent / "protocol_freeze.json", {"protocol_id": protocol_id, "test_opened": False})
    _write_json(parent / "test_label_open_receipt.json", {"protocol_id": protocol_id, "status": "complete"})
    _write_json(
        parent / "compact_train_label_filter_receipt.json",
        {
            "protocol_id": protocol_id,
            "status": "complete",
            "after": {"manifest": {"sha256": data_hash}},
        },
    )
    _write_json(
        campaign / "campaign_intent.json",
        {
            "protocol_id": protocol_id,
            "fit_split": "train",
            "evaluation_split": "test",
            "status": "running",
            "runs": {"nanobrain": seeds},
            "prior_attempts": [{"path": "../test", "status": "aborted"}],
        },
    )

    for seed in seeds:
        candidate = f"candidate{seed}"
        directory = campaign / f"nanobrain-seed-{seed}"
        checkpoint = tmp_path / f"train-{seed}" / f"candidate-{candidate}-best.pt"
        checkpoint.parent.mkdir(parents=True)
        final_checkpoint = checkpoint.parent / f"candidate-{candidate}.pt"
        torch.save(
            {
                "model": {"weight": torch.tensor([float(seed)])},
                "step": 1,
                "validation_quality": 0.9,
            },
            checkpoint,
        )
        torch.save(
            {"model": {"weight": torch.tensor([float(seed + 1)])}, "step": 2000},
            final_checkpoint,
        )
        rows = [
            _row(f"nanobrain_candidate-{candidate}", task, metric, index + seed / 100.0)
            for index, (task, metric) in enumerate(finalizer.KEY_METRICS)
        ]
        _write_table(directory / "benchmark_table.csv", rows)
        _write_json(directory / "run_manifest.json", _manifest(data_hash, checkpoint=str(checkpoint)))
        _write_json(
            checkpoint.parent / f"receipt-{candidate}-steps2000.json",
            {
                "status": "complete",
                "fit_split": "train",
                "evaluation_split": "validation",
                "test_opened": False,
                "steps": 2000,
                "checkpoint": str(final_checkpoint),
                "best_validation_quality": 0.9,
                "validation_history": [
                    {"step": 1, "quality": 0.9},
                    {"step": 2000, "quality": 0.8},
                ],
                "genome": {"seed": seed, "hidden_dim": 64},
                "train": {"seed": seed, "batch_size": 256},
            },
        )

    internal = campaign / "internal-baselines"
    _write_table(internal / "benchmark_table.csv", [_row("majority", "annotation", "accuracy", 0.1)])
    _write_json(internal / "run_manifest.json", _manifest(data_hash))

    external = campaign / "external-baselines"
    external_rows = [
        _row(method, "annotation", "accuracy", 0.2)
        for method in sorted(finalizer.EXPECTED_EXTERNAL_METHODS)
    ]
    _write_table(external / "benchmark_table.csv", external_rows)
    _write_json(
        external / "run_manifest.json",
        _manifest(
            data_hash,
            methods_requested=sorted(finalizer.EXPECTED_EXTERNAL_METHODS),
            method_status={method: "complete" for method in finalizer.EXPECTED_EXTERNAL_METHODS},
        ),
    )
    return campaign, data_manifest


def test_finalizer_audits_and_writes_complete_campaign(tmp_path):
    campaign, data_manifest = _campaign_fixture(tmp_path)

    result = finalizer.finalize(campaign, data_manifest=data_manifest)
    first_receipt = (campaign / "campaign_receipt.json").read_bytes()
    repeated = finalizer.finalize(campaign, data_manifest=data_manifest)

    assert result["status"] == "complete"
    assert result["row_counts"] == {
        "per_seed_metrics": 24,
        "aggregate_metrics": 8,
        "final_benchmark_long": 22,
    }
    intent = json.loads((campaign / "campaign_intent.json").read_text(encoding="utf-8"))
    receipt = json.loads((campaign / "campaign_receipt.json").read_text(encoding="utf-8"))
    assert intent["status"] == "complete"
    assert receipt["independent_seeds"] == [11, 22, 33]
    assert receipt["audit"]["external_methods_complete"] == 13
    assert receipt["checkpoint_metadata_audit"]["legacy_step_metadata_mismatch"] is False
    assert (campaign / "campaign_receipt.json").read_bytes() == first_receipt
    assert repeated["receipt_sha256"] == result["receipt_sha256"]
    assert (campaign / "campaign_intent.pre_finalize.json").is_file()
    assert (campaign / "REPORT.md").is_file()


def test_finalizer_rejects_changed_data_manifest(tmp_path):
    campaign, data_manifest = _campaign_fixture(tmp_path)
    data_manifest.write_text('{"format":"tampered"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="data manifest hash mismatch"):
        finalizer.finalize(campaign, data_manifest=data_manifest)
