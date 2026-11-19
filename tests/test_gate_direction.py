"""Metric direction and per-metric tolerance in the promotion gate."""

import pytest

from nanobrain.infra import contracts as C
from nanobrain.infra import gate


def test_cajal_metrics_are_larger_is_better_with_the_floor():
    assert set(C.METRIC_SIGN.values()) == {1}
    assert set(C.METRIC_TOLERANCE.values()) == {C.GATE_FLOOR}


def test_smaller_is_better_metric_with_its_own_tolerance(monkeypatch):
    error = C.METRICS[0]
    monkeypatch.setitem(gate.METRIC_SIGN, error, -1)
    monkeypatch.setitem(gate.METRIC_TOLERANCE, error, 0.5)
    baseline = {k: 0.5 for k in C.METRICS} | {error: 8.0}
    parent = {"metrics": dict(baseline)}
    lower = {"metrics": baseline | {error: 7.0}}
    assert gate.promote(lower, parent, parent, baseline, [error])["passed"]
    small_rise = {"metrics": baseline | {error: 8.4, C.METRICS[1]: 0.6}}
    assert not gate.promote(small_rise, parent, parent, baseline, [C.METRICS[1]])["regressions"]
    large_rise = {"metrics": baseline | {error: 8.6, C.METRICS[1]: 0.6}}
    assert gate.promote(large_rise, parent, parent, baseline, [C.METRICS[1]])["regressions"] == [error]
    assert gate.margins(lower, baseline)[0] == pytest.approx(1 / 8)


def test_definition_rejects_a_bad_direction(tmp_path):
    from nanobrain.infra import campaign

    src = campaign.directory()
    text = (src / "campaign.toml").read_text().replace(
        'group = "annotation"\nrange = [0.0, 1.0]', 'group = "annotation"\nrange = [0.0, 1.0]\ndirection = "up"', 1
    )
    (tmp_path / "campaign.toml").write_text(text)
    with pytest.raises(ValueError, match="direction"):
        campaign.load(str(tmp_path))


def test_cajal_evaluates_its_four_groups_in_order():
    assert tuple(C.METRIC_GROUPS) == ("annotation", "integration", "genes", "spatial")
    assert C.DATA_KEY == {"genes": "annotation", "age": "annotation"}
