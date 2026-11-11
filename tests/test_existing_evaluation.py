"""Reuse Cajal data/metrics/results; tests use small synthetic predictions only."""

import csv
import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cajal_research import contracts as C


def scorer(name):
    spec = importlib.util.spec_from_file_location("test_" + name, C.ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def embedding(tmp_path):
    path = tmp_path / "prediction.npz"
    labels = np.arange(12) % 3
    np.savez(
        path,
        step=10000,
        cell_type=labels,
        anatomy=labels,
        batch=np.arange(12) % 2,
        mu=np.arange(48).reshape(12, 4),
        cell_type_logits_0=np.eye(3)[labels],
        anatomy_logits_0=np.eye(3)[labels],
    )
    return path


def fail_baseline(*args, **kwargs):
    pytest.fail("existing baseline was loaded, fitted, or scored")


def read_summary(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_annotation_predictions_only_uses_original_metrics_without_baselines(tmp_path, monkeypatch):
    module = scorer("eval_zero_shot")
    prediction = embedding(tmp_path)
    monkeypatch.setattr(module, "load_baselines", fail_baseline)
    monkeypatch.setattr(module, "prepare_features", fail_baseline)
    monkeypatch.setattr(module, "load_counts", fail_baseline)
    monkeypatch.setattr(module, "level_parents", lambda _: {kind: np.array([0, 0, 1]) for kind in module.KINDS})
    out = tmp_path / "scores"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--pred",
            f"candidate={prediction}",
            "--data",
            str(tmp_path),
            "--out",
            str(out),
            "--predictions-only",
            "--task",
            "annotation",
        ],
    )
    module.main()
    rows = read_summary(out / "summary.csv")
    assert [r[""] for r in rows] == ["candidate"]
    assert float(rows[0]["Cell type fine macro-F1"]) == 1.0
    assert float(rows[0]["Region fine macro-F1"]) == 1.0
    assert all(not key.startswith("Integration") for key in rows[0])


def test_integration_keeps_cached_PCR_reference_and_scores_only_candidate(tmp_path, monkeypatch):
    module = scorer("eval_zero_shot")
    prediction = embedding(tmp_path)
    monkeypatch.setattr(module, "load_baselines", fail_baseline)
    monkeypatch.setattr(module, "prepare_features", fail_baseline)
    monkeypatch.setattr(module, "level_parents", lambda _: {})
    monkeypatch.setattr(module, "load_counts", lambda _: sparse.csr_matrix(np.ones((12, 3))))

    class Transform:
        def transform(self, values):
            return values

        fit = fail_baseline

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "features.pkl").write_bytes(b"existing frozen prep")
    monkeypatch.setattr(
        module.pickle, "load", lambda _: {"genes": np.arange(3), "scaler": Transform(), "pca": Transform()}
    )

    class Benchmark:
        def __init__(self, adata, **kwargs):
            assert kwargs["embedding_obsm_keys"] == ["candidate"]
            assert kwargs["pre_integrated_embedding_obsm_key"] == "PCA (unintegrated)"
            assert "PCA (unintegrated)" in adata.obsm

        def prepare(self, **kwargs):
            assert kwargs["neighbor_computer"] is module.gpu_neighbors

        def benchmark(self):
            pass

        def get_results(self, min_max_scale):
            assert min_max_scale is False
            return pd.DataFrame(
                {"Bio conservation": [0.6], "Batch correction": [0.4], "Total": [0.52]}, index=["candidate"]
            )

    benchmark_module = types.ModuleType("scib_metrics.benchmark")
    benchmark_module.Benchmarker = Benchmark
    monkeypatch.setitem(sys.modules, "scib_metrics.benchmark", benchmark_module)
    out = tmp_path / "scores"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--pred",
            f"candidate={prediction}",
            "--data",
            str(tmp_path),
            "--out",
            str(out),
            "--cache",
            str(cache),
            "--predictions-only",
            "--task",
            "integration",
        ],
    )
    module.main()
    rows = read_summary(out / "summary.csv")
    assert [r[""] for r in rows] == ["candidate"]
    assert float(rows[0]["Integration bio"]) == 0.6 and float(rows[0]["Integration batch"]) == 0.4
    assert all(not key.startswith("Cell type") for key in rows[0])


@pytest.mark.parametrize("task", ["genes", "spatial"])
def test_imputation_predictions_only_preserves_original_summaries(task, tmp_path, monkeypatch):
    module = scorer("eval_imputation")
    from brainwm_unified.impute_metrics import row_stats

    rng = np.random.default_rng(0)
    stats = row_stats(rng.random(60), rng.random(60), np.arange(0, 61, 15))
    prediction = tmp_path / "prediction.npz"
    labels = np.array([0, 1, 0, 1])
    np.save(tmp_path / "cell_type.npy", labels)
    np.savez(prediction, **stats, task=task, cell_type=labels, cell_type_logits_0=np.eye(2)[labels])
    monkeypatch.setattr(module, "Cells", fail_baseline)
    monkeypatch.setattr(module, "add_baselines", fail_baseline)
    out = tmp_path / "scores"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--pred",
            f"candidate={prediction}",
            "--data",
            "/not-read",
            "--pretrain",
            str(tmp_path),
            "--task",
            task,
            "--out",
            str(out),
            "--predictions-only",
        ],
    )
    module.main()
    rows = read_summary(out / "summary.csv")
    assert [r["method"] for r in rows] == ["candidate"]
    assert (
        float(rows[0][module.TASK_NAME[task] + " Pearson (cell median)"])
        == module.summarize(stats)["pearson_cell_median"]
    )


@pytest.mark.parametrize("task", ["annotation", "integration", "genes", "spatial"])
def test_campaign_worker_never_requests_baseline_work(task, monkeypatch):
    from cajal_research import worker

    calls = []
    monkeypatch.setattr(worker, "script", lambda name, args: calls.append((name, args)))
    monkeypatch.setattr(worker, "save", lambda *a: None)
    monkeypatch.setattr(worker, "sha", lambda *a: "synthetic-receipt")
    monkeypatch.setattr(sys, "argv", ["worker", "score", "--task", task, "--prediction", "/not-read/prediction.npz"])
    worker.main()
    _name, args = calls[0]
    assert "--predictions-only" in args and "--fit" not in args and "--prepare" not in args
    assert args[args.index("--task") + 1] == task


def test_initial_reference_reuses_existing_checkpoint_and_results(tmp_path, monkeypatch):
    from cajal_research.store import Store
    from cajal_research.workflow import Workflow

    workflow = Workflow(C.load_config(), tmp_path)
    workflow.store = Store(tmp_path / "lineage")
    identifier = workflow.store.stage(C.reference_bundle())
    monkeypatch.setattr(workflow, "worker", fail_baseline)
    result = workflow.evaluate(identifier, 0)
    assert result["status"] == "complete"
    (row,) = result["seed_results"]
    assert row["method"] == "cajal_v1@10000" and row["source"] == "existing_Cajal_project_result"
    assert not row["training_executed_by_campaign"] and not row["evaluation_executed_by_campaign"]
    assert not (tmp_path / "jobs").exists()
    snapshot = json.loads((C.ROOT / "reference/existing_cajal_v1_10k.json").read_text())
    assert row["metrics"] == snapshot["metrics"]


def test_changed_existing_reference_fails_without_refitting(monkeypatch):
    from cajal_research import existing_reference

    snapshot = json.loads((C.ROOT / "reference/existing_cajal_v1_10k.json").read_text())

    def changed(path):
        return "changed" if str(path) == snapshot["checkpoint"] else C.sha(path)

    monkeypatch.setattr(existing_reference, "sha", changed)
    with pytest.raises(ValueError, match="artifact changed"):
        existing_reference.load(C.load_config())


def test_llm_uses_cajal_scores_and_reuses_legacy_components_only(tmp_path):
    from cajal_research.agent import SYSTEM, ToolSession
    from cajal_research.workflow import baseline

    opponents = baseline()
    session = ToolSession(None, opponents, None, [], [], {}, tmp_path / "unused.json")
    detail = session.call("baseline_detail", {})
    assert detail["source_csv"] == "/data/brainwm/BrainWM-Unified/runs/eval/five_tasks.csv"
    policy = detail["performance_input_policy"]
    assert policy["legacy_component_implementations_reused"]
    assert not policy["existing_baselines_refitted_or_rescored"] and not policy["sealed_V4_test_used"]
    assert "three paired seeds" not in detail["regime"]
    assert "Do not use legacy nanoBrain experiment scores" in SYSTEM
    entries = []
    offset = 0
    while True:
        page = session.call("list_components", {"offset": offset, "available_only": False})
        entries.extend(page["entries"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    legacy = [entry for entry in entries if entry["id"].startswith("legacy.")]
    assert legacy and any(entry["available"] for entry in legacy)
    assert all("metrics" not in entry for entry in legacy)
    scored = [entry for entry in entries if "metrics" in entry]
    assert len(scored) == 13
    assert all(entry["source_family"] == "Cajal existing whole-method benchmark" for entry in scored)
