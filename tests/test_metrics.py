import numpy as np
from brainwm_benchmark import evaluate
from brainwm_benchmark.metrics import _knn_graph


def test_knn_graph_keeps_closest_neighbor_excludes_self_and_handles_small_sets():
    x = np.array([[0.0], [1.0], [4.0]])
    nearest, graph = _knn_graph(x, k=1)
    np.testing.assert_array_equal(nearest[:, 0], [1, 0, 1])
    assert not graph.diagonal().any()
    neighbors, graph = _knn_graph(x, k=15)
    assert neighbors.shape == (3, 2)
    assert all(i not in neighbors[i] for i in range(3))
    two, _ = _knn_graph(x[:2], k=15)
    np.testing.assert_array_equal(two[:, 0], [1, 0])


def test_annotation_perfect():
    y = np.array([0, 1, 2, 1])
    prob = np.eye(3)[y]
    out = evaluate("annotation", {"y_true": y, "y_pred": y, "y_prob": prob})
    assert out["accuracy"] == 1.0
    assert out["f1_macro"] == 1.0
    assert out["ece"] == 0.0


def test_completion_identity():
    x = np.arange(20, dtype=float).reshape(5, 4)
    out = evaluate("spatial_completion", {"target": x, "prediction": x})
    assert out["rmse"] == 0.0
    assert out["mae"] == 0.0
    assert np.isclose(out["pearson_global"], 1.0)


def test_development_identity():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 5))
    out = evaluate("development", {"target": x, "prediction": x})
    assert out["population_mse"] == 0.0
    assert np.isclose(out["gene_pearson"], 1.0)


def test_development_population_groups_prevent_cross_interval_cancellation():
    target = np.concatenate((np.zeros((4, 2)), np.full((4, 2), 10.0)))
    prediction = np.concatenate((np.full((4, 2), 10.0), np.zeros((4, 2))))

    pooled = evaluate("development", {"target": target, "prediction": prediction})
    grouped = evaluate(
        "development",
        {
            "target": target,
            "prediction": prediction,
            "group_ptr": np.asarray([0, 4, 8], dtype=np.int64),
            "population_unpaired": np.asarray([True]),
        },
    )

    assert pooled["population_mse"] == 0.0
    assert grouped["population_mse"] == 100.0


def test_spatial_and_integration_are_finite():
    rng = np.random.default_rng(11)
    embedding = rng.normal(size=(30, 4))
    labels = np.repeat(np.arange(3), 10)
    batches = np.tile(np.arange(2), 15)
    integ = evaluate("integration", {"embedding": embedding, "biology_label": labels, "batch_label": batches})
    assert all(np.isfinite(list(integ.values())))
    spatial = evaluate("spatial_niche", {"embedding": embedding, "coordinates": rng.normal(size=(30, 2)), "labels": labels, "clusters": labels})
    assert spatial["ari"] == 1.0
    assert all(np.isfinite(list(spatial.values())))
