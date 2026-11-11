import csv
import json

from scripts.summarize_nanobrain_benchmark import audit


def _write_table(path, artifact):
    fields = [
        "split_id", "task", "method", "fit_rows", "eval_rows", "status",
        "prediction_artifact", "metric", "stratum_type", "stratum_value", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
            "task": "annotation", "method": "toy", "fit_rows": "2", "eval_rows": "1",
            "status": "complete", "prediction_artifact": str(artifact.relative_to(path.parent)),
            "metric": "accuracy", "stratum_type": "overall", "stratum_value": "overall", "error": "",
        })


def test_audit_marks_artifact_and_receipt_complete(tmp_path):
    table = tmp_path / "benchmark_table.csv"
    artifact = tmp_path / "predictions" / "toy__annotation.npz"
    artifact.parent.mkdir()
    artifact.write_bytes(b"npz")
    artifact.with_suffix(".metrics.json").write_text(json.dumps({"format": "test"}) + "\n", encoding="utf-8")
    _write_table(table, artifact)

    report = audit(table)

    assert report["error_rows"] == 0
    assert report["task_summary"]["annotation"]["complete_methods"] == ["toy"]
    assert report["entries"][0]["status"] == "complete"


def test_audit_marks_missing_artifact_incomplete(tmp_path):
    table = tmp_path / "benchmark_table.csv"
    _write_table(table, tmp_path / "predictions" / "missing.npz")

    report = audit(table)

    assert report["entries"][0]["status"] == "incomplete"
    assert report["task_summary"]["annotation"]["complete_methods"] == []


def test_audit_includes_external_status_ledger(tmp_path):
    table = tmp_path / "benchmark_table.csv"
    artifact = tmp_path / "predictions" / "toy__annotation.npz"
    artifact.parent.mkdir()
    artifact.write_bytes(b"npz")
    artifact.with_suffix(".metrics.json").write_text("{}\n", encoding="utf-8")
    _write_table(table, artifact)
    (tmp_path / "method_status.json").write_text(
        json.dumps([
            {"method": "toy", "status": "complete"},
            {"method": "needs_asset", "status": "external_asset_required"},
        ]) + "\n",
        encoding="utf-8",
    )

    report = audit(table)

    assert report["external_status_counts"] == {
        "complete": 1,
        "external_asset_required": 1,
    }
    assert report["unmeasured_external_methods"] == ["needs_asset"]
    assert report["status_errors"] == []
