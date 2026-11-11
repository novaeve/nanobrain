import csv
import json

import pytest

from scripts.run_nanobrain_campaign import _history_summary, _prepare_output


def _write_receipt(path, candidate_id, quality):
    path.write_text(
        json.dumps({
            "candidate_id": candidate_id,
            "genome": {"encoder": "mlp"},
            "quality": quality,
            "runtime_seconds": 1.0,
            "validation": {"annotation_accuracy": quality},
        }) + "\n",
        encoding="utf-8",
    )


def test_explicit_nonempty_campaign_output_is_rejected(tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    (output / "old-artifact.pt").write_bytes(b"stale")

    with pytest.raises(ValueError, match="must be empty"):
        _prepare_output(output)


def test_history_uses_manifest_candidate_receipt_and_evaluator(tmp_path):
    campaign_dir = tmp_path / "campaign"
    candidate_dir = campaign_dir / "candidate_00"
    candidate_dir.mkdir(parents=True)
    campaign = {
        "training": {"steps": 300},
        "candidates": [{"index": 0, "candidate_id": "current", "genome": {"encoder": "mlp"}}],
    }
    (campaign_dir / "campaign.json").write_text(json.dumps(campaign) + "\n", encoding="utf-8")
    _write_receipt(candidate_dir / "receipt-old-steps300.json", "old", 0.99)
    _write_receipt(candidate_dir / "receipt-current-steps300.json", "current", 0.5)

    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    fields = ["task", "method", "status", "metric", "value", "stratum_type", "stratum_value"]
    with (evaluation / "benchmark_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "task": "annotation",
            "method": "nanobrain_candidate-current",
            "status": "complete",
            "metric": "accuracy",
            "value": "0.75",
            "stratum_type": "overall",
            "stratum_value": "overall",
        })

    history = _history_summary(campaign_dir, [evaluation], "current")

    assert len(history) == 1
    assert history[0]["candidate_id"] == "current"
    assert history[0]["quality"] == 0.5
    assert history[0]["benchmark_validation"]["tasks"]["annotation"]["accuracy"] == 0.75
