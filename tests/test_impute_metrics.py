"""Imputation metrics: sufficient statistics reproduce the direct formulas (CPU)."""

import itertools

import numpy as np

from brainwm_unified.impute_metrics import panel_profile, row_stats, summarize


def test_summary_from_row_stats_matches_direct_formulas():
    rng = np.random.default_rng(0)
    ptr = np.r_[0, np.cumsum([20, 30, 5, 40])]
    true = np.abs(rng.normal(size=ptr[-1]))
    true[::3] = 0
    pred = true + rng.normal(scale=0.3, size=ptr[-1])
    got = summarize(row_stats(pred, true, ptr))
    rows = [np.corrcoef(pred[a:b], true[a:b])[0, 1] for a, b in itertools.pairwise(ptr) if b - a >= 10]
    nz = true > 0
    assert np.isclose(got["pearson_cell_median"], np.median(rows))
    assert np.isclose(got["mse"], np.mean((pred - true) ** 2))
    assert np.isclose(got["mse_nonzero"], np.mean((pred[nz] - true[nz]) ** 2))
    assert np.isclose(got["pearson_global"], np.corrcoef(pred, true)[0, 1])


def test_stats_add_up_across_files():
    rng = np.random.default_rng(1)
    ptr = np.r_[0, np.cumsum([15, 25, 12])]
    true, pred = rng.random(ptr[-1]), rng.random(ptr[-1])
    whole = row_stats(pred, true, ptr)
    first, second = row_stats(pred[:15], true[:15], ptr[:2]), row_stats(pred[15:], true[15:], ptr[1:] - 15)
    joined = {k: np.r_[first[k], second[k]] for k in whole}
    assert summarize(joined) == summarize(whole)


def test_panel_profile_normalises_by_the_whole_library():
    row_ptr = np.array([0, 3])
    gene_index, values = np.array([7, 2, 30]), np.array([1.0, 3.0, 6.0])
    prof = panel_profile(row_ptr, gene_index, values, 0, np.array([2, 5, 7]))  # gene 30 is off this panel
    np.testing.assert_allclose(prof, np.log1p(np.array([3.0, 0.0, 1.0]) / 10 * 1e4), rtol=1e-6)
