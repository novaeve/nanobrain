import json
from dataclasses import fields

import numpy as np
import pytest

from nanobrain.baselines import AdapterContext, Availability, build, names
from nanobrain.baselines.integration import _bbknn_exact_backend
from nanobrain.data import load_v3_subset
from scripts.merge_nanobrain_pilots import _collect_method_status


EXPECTED_ADAPTERS = {
    "bbknn",
    "celltypist",
    "geneformer_v1_10m",
    "graphst",
    "harmony",
    "scanorama",
    "scfit",
    "scgpt_brain",
    "scgpt_human",
    "scnode",
    "scuniflow",
    "scvi_scanvi",
    "spagcn",
    "stagate",
    "trajectorynet",
    "uce",
    "xgboost",
}


def _write_split(root, split, rows=2):
    directory = root / split
    directory.mkdir(parents=True)
    np.save(directory / "features.npy", np.zeros((rows, 4), dtype=np.float32))
    for name in ("annotation", "niche", "integration"):
        np.save(directory / f"label_{name}.npy", np.zeros(rows, dtype=np.int64))
    np.save(directory / "label_development.npy", np.zeros((rows, 1), dtype=np.float32))
    for name in ("annotation", "niche", "integration", "development"):
        np.save(directory / f"mask_{name}.npy", np.ones(rows, dtype=bool))


def _write_status_run(root, status, *, test_opened=False):
    root.mkdir(parents=True)
    (root / "method_status.json").write_text(
        json.dumps([{"method": "toy", "status": status, "tasks": "annotation"}]) + "\n",
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps({"test_opened": test_opened}) + "\n",
        encoding="utf-8",
    )


def test_adapter_registry_is_complete_and_buildable():
    assert set(names()) == EXPECTED_ADAPTERS
    for method_id in names():
        adapter = build(method_id)
        assert adapter.method_id == method_id
        assert adapter.tasks


def test_availability_only_runs_ready_methods():
    assert Availability("ready").runnable is True
    for status in (
        "dependency_missing",
        "external_asset_required",
        "resource_contract_incompatible",
        "task_contract_incompatible",
        "failed",
    ):
        assert Availability(status).runnable is False


def test_bbknn_singleton_batch_keeps_two_dimensional_neighbor_result():
    matrix = pytest.importorskip("bbknn.matrix")
    pca = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 0.1], [0.9, -0.1]],
        dtype=np.float32,
    )
    distances, connectivities, _ = matrix.bbknn(
        pca=pca,
        batch_list=np.asarray(["singleton", "common", "common", "common"]),
        neighbors_within_batch=1,
        n_pcs=2,
        computation=_bbknn_exact_backend(1),
        metric="euclidean",
    )
    assert distances.shape == (4, 4)
    assert connectivities.shape == (4, 4)


def test_adapter_context_cannot_expose_test_split():
    assert "test" not in {field.name for field in fields(AdapterContext)}


def test_train_validation_loader_does_not_open_sealed_test(tmp_path):
    _write_split(tmp_path, "train")
    _write_split(tmp_path, "validation")
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "format": "nanobrain_v3_subset_v1",
            "split_id": "BRAINWM-HUMAN-GLOBAL-6-2-2-V3",
            "splits": {
                "train": {"rows": 2, "files": {}},
                "validation": {"rows": 2, "files": {}},
                "test": {"rows": 2, "files": {}},
            },
        }) + "\n",
        encoding="utf-8",
    )

    splits = load_v3_subset(tmp_path, include_test=False)

    assert splits.train.size == 2
    assert splits.validation.size == 2
    assert splits.test is None
    assert not (tmp_path / "test").exists()


def test_status_collection_prefers_measured_complete(tmp_path):
    unavailable = tmp_path / "unavailable"
    complete = tmp_path / "complete"
    _write_status_run(unavailable, "external_asset_required")
    _write_status_run(complete, "complete")

    records, sources = _collect_method_status([
        ("unavailable", unavailable),
        ("complete", complete),
    ])

    assert records == [{
        "method": "toy",
        "status": "complete",
        "tasks": "annotation",
        "source_run": "complete",
    }]
    assert len(sources) == 2


def test_status_collection_rejects_opened_test(tmp_path):
    run = tmp_path / "run"
    _write_status_run(run, "complete", test_opened=True)

    with pytest.raises(ValueError, match="sealed test"):
        _collect_method_status([("run", run)])
