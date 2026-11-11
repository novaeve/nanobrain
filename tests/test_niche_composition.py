from pathlib import Path

import numpy as np
import pytest
import torch

from brainwm_benchmark.metrics import niche_composition, wasserstein2
from nanobrain.data import DatasetSplits, make_synthetic_splits, infer_label_cardinalities, niche_model_options
from nanobrain.niche_data import neighborhood_targets


def composition_fixture():
    splits = make_synthetic_splits(train_rows=32, validation_rows=16, test_rows=8,
        input_dim=8, annotation_classes=3, niche_classes=2, integration_classes=2,
        development_dim=8, seed=19)
    for ds in (splits.train, splits.validation):
        x = ds.coordinates[:, :2].numpy()
        p, d, m = neighborhood_targets(x, ds.labels["annotation"].numpy(), np.arange(ds.size),
            x, np.arange(ds.size), [2., 100.], 3, min_known_fraction=0)
        ds.labels["niche"] = torch.from_numpy(p)
        ds.labels["niche_density"] = torch.from_numpy(d)
        ds.masks["niche"] = torch.from_numpy(m)
        ds.masks["niche_density"] = torch.ones_like(ds.masks["niche"])
    return DatasetSplits(splits.train, splits.validation, None)


def test_full_neighborhood_counts_exclude_only_center_identity_and_retain_unknown_mass():
    x = np.array([[0., 0.], [0., 0.], [1., 0.], [2., 0.]])
    p, density, valid = neighborhood_targets(x, np.array([0, 1, 2, -1]), np.arange(10, 14),
        x[:1], np.array([10]), [.1, 1.1, 3.], 3)
    np.testing.assert_allclose(density, [[1, 2, 3]])
    np.testing.assert_allclose(p[0, 0], [0, 1, 0, 0])
    np.testing.assert_allclose(p[0, 1], [0, .5, .5, 0])
    np.testing.assert_allclose(p[0, 2], [0, 1/3, 1/3, 1/3])
    assert valid.tolist() == [[True, True, False]]


def test_true_proportions_and_equal_radius_weighting():
    truth = np.array([[[1., 0.], [.5, .5]], [[0., 1.], [0., 0.]]])
    prediction = np.array([[[.8, .2], [1., 0.]], [[0., 1.], [.5, .5]]])
    result = niche_composition(dict(target_composition=truth, prediction_composition=prediction,
                                   composition_mask=np.array([[1, 1], [1, 0]], bool)))
    assert result["composition_mae"] == pytest.approx((.1 + .5) / 2)
    assert result["composition_mae_scale_0_type_0"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="probability vectors"):
        niche_composition(dict(target_composition=truth, prediction_composition=prediction * 2))


def test_exact_w2_is_unpaired_and_uses_euclidean_units():
    x = np.array([[0., 0.], [2., 2.]])
    assert wasserstein2(x, x[::-1]) == pytest.approx(0)
    assert wasserstein2(x, x + 1) == pytest.approx(np.sqrt(2))
    assert wasserstein2(np.zeros((2, 2)), np.ones((3, 2))) == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("graph", [False, True])
def test_composition_heads_train_reload_and_match_validation_mae(tmp_path, graph):
    from nanobrain.genome import CandidateGenome
    from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
    from nanobrain.architecture_migration import legacy_to_graph
    from nanobrain.training import TrainConfig, train_candidate
    from scripts.run_nanobrain_baselines import _run_nanobrain

    splits = composition_fixture()
    genome = CandidateGenome(hidden_dim=32, attention="identity", spatial="neighbor_mean",
                             flow="mlp", pairing="identity", dropout=0, seed=19)
    if graph:
        genome = CandidateGenome.from_dict(dict(architecture=legacy_to_graph(genome, 8), learning_rate=.001, seed=19))
    cardinalities = infer_label_cardinalities(splits)
    config = NanoBrainModelConfig(8, 3, cardinalities["niche"], 2, 8, **niche_model_options(splits))
    receipt = train_candidate(splits, config, genome,
        TrainConfig(steps=2, batch_size=8, eval_interval=1, validation_batch_size=8,
                    task="spatial_niche", checkpoint_policy="best_weights", precision="fp32", device="cpu"),
        output_dir=tmp_path)
    best = torch.load(receipt["best_checkpoint"], weights_only=False, map_location="cpu")
    model = NanoBrainModel(config, genome)
    model.load_state_dict(best["model"])
    assert receipt["optimizer_changed_parameter"]
    assert any(name.startswith("niche_density_head") for name in receipt["trainable_parameters"])
    rows = _run_nanobrain(splits, Path(receipt["best_checkpoint"]), tmp_path / "eval", "cpu", 8)
    assert all(row["status"] == "complete" for row in rows), rows
    metric = next(row for row in rows if row.get("metric") == "composition_mae")
    assert metric["direction"] == "min"
    assert metric["value"] == pytest.approx(best["validation"]["spatial_niche.composition_mae"], abs=1e-7)
    assert not any(row.get("metric") == "f1_macro" for row in rows)


def test_niche_input_hides_neighbors_and_targets():
    from dataclasses import replace
    from nanobrain.genome import CandidateGenome
    from nanobrain.model import NanoBrainModel, NanoBrainModelConfig
    from nanobrain.task_training import task_input
    splits = composition_fixture()
    net = NanoBrainModel(NanoBrainModelConfig(8, 3, 4, 2, 8, niche_mode="composition", niche_scales=2),
                         CandidateGenome(hidden_dim=32, attention="identity", spatial="neighbor_mean", dropout=0)).eval()
    batch = splits.train.batch(np.arange(8))
    altered = replace(batch, coordinates=torch.randn_like(batch.coordinates) * 100,
                      neighbor_features=torch.randn(10, 8) * 100,
                      labels={**batch.labels, "niche": batch.labels["niche"].roll(1, -1)})
    with torch.no_grad():
        a = net.predict_niche(net.encode(task_input(batch, "spatial_niche")))[0]
        b = net.predict_niche(net.encode(task_input(altered, "spatial_niche")))[0]
    torch.testing.assert_close(a, b)


def test_classical_composition_baselines_use_training_targets_only(tmp_path):
    from scripts.run_nanobrain_baselines import _run_spatial
    splits = composition_fixture()
    rows = _run_spatial(splits.train, splits.validation, tmp_path, 19, {"niche_global_mean", "spatial_no_structure"})
    assert {r["method"] for r in rows} == {"niche_global_mean", "spatial_no_structure"}
    assert all(r["status"] == "complete" for r in rows)
    assert len([r for r in rows if r.get("metric") == "composition_mae"]) == 2


def test_derived_manifest_counts_composition_eligible_centers(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from nanobrain import niche_data

    parent, source, cache, output = [tmp_path / name for name in ("parent", "source", "cache", "output")]
    for path in (parent, source, cache):
        path.mkdir()
    (source / "joint_metadata_vocab.json").write_text(json.dumps({"cell_type": []}))
    manifest = {"splits": {}}
    sections = []
    xy = np.column_stack([np.arange(121), np.zeros(121)]).astype(np.float32)
    for split in ("train", "validation", "test"):
        directory = parent / split
        directory.mkdir()
        pd.DataFrame({"row_id": [0, 1, 2]}).to_parquet(directory / "row_identity.parquet")
        np.save(directory / "sample.npy", [1, 1, 999])
        np.save(directory / "coordinates.npy", xy[[0, 1, 120]])
        np.save(directory / "source_row_ref.npy", [0, 1, 120])
        np.save(directory / "label_niche.npy", [0, 1, 1])
        np.save(directory / "mask_niche.npy", [True, True, True])
        file = f"{split}.npz"
        np.savez(cache / file, coordinates=xy, labels=np.arange(121) % 2, row_refs=np.arange(121))
        sections.append(dict(split=split, sample_id=1, source_id=7, path=file, known_fraction=1.0))
        manifest["splits"][split] = dict(rows=3, files={}, task_rows={"spatial_niche": 3})
    (parent / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(niche_data, "collect_full_sections",
                        lambda *args: dict(fingerprint={"fixture": True}, sections=sections))

    receipt = niche_data.build_niche_dataset(parent, source, output, cache)
    derived = json.loads((output / "manifest.json").read_text())
    for split in manifest["splits"]:
        eligible = int(np.load(output / split / "mask_niche.npy").any(1).sum())
        assert eligible == 2
        assert derived["splits"][split]["task_rows"]["spatial_niche"] == eligible
        assert receipt["splits"][split]["valid_centers"] == eligible
        assert np.load(parent / split / "mask_niche.npy").sum() == 3
