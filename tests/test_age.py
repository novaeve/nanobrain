"""Age task: labels in days post fertilization, the interval-censored AFT likelihood (logistic and normal laws),
held-out groups, donor weights, the opt-in head, the balancer with a detached loss, end to end, evaluation."""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_gpu import TINY_RUN, _has_fa3
from test_unified import make, subset_dir, tiny_config  # noqa: F401  (subset_dir is a fixture)

from brainwm_unified import age
from brainwm_unified import model as M
from brainwm_unified.objectives import compute_losses

DEVICE = torch.device("cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
YEAR = 365.25


def test_labels_are_days_post_fertilization_at_their_reporting_resolution():
    nan = np.nan
    lower = np.array([73 * YEAR, -168.0, 100.0, 90 * YEAR, nan, nan, 50.0, nan])
    upper = np.array([73 * YEAR, -168.0, 400.0, nan, nan, nan, 50.0, nan])
    point = np.array([73 * YEAR, -168.0, 250.0, nan, -212.0, nan, 50.0, nan])
    status = ["exact", "exact", "range_midpoint", "right_censored", "author_reported_point", "exact"]
    status = np.array([*status, "categorical_only", "categorical_only"])
    basis = ["postnatal_y", "postconception_week", "postnatal_y_range", "postnatal_y_censored", "", "", "", ""]
    stage = np.array(["adult", "prenatal", "adult", "adult", "prenatal", "unknown", "unknown", "postnatal"])
    lo, hi = age.label_interval(lower, upper, point, status, np.array(basis), stage)
    np.testing.assert_allclose(lo[0], 266 + 73 * YEAR, rtol=1e-6)  # "73y": a whole year, ~26,929-27,294 dpf
    np.testing.assert_allclose(hi[0] - lo[0], YEAR, rtol=1e-6)
    assert (lo[1], hi[1]) == (98.0, 105.0)  # 14 pcw: the 15th week after fertilization
    assert (lo[2], hi[2]) == (366.0, 666.0)  # a range as given
    assert lo[3] == np.float32(266 + 90 * YEAR) and np.isnan(hi[3])  # "> 90 y"
    assert (lo[4], hi[4]) == (54.0, 55.0)  # Carnegie 22: a day
    assert np.isnan([lo[5], hi[5], lo[6], hi[6]]).all()  # no number and no known life stage
    assert lo[7] == 266.0 and hi[7] == np.float32(266 + 18 * YEAR)  # categorical "infant": its life stage
    assert (lo[np.isfinite(lo)] >= 0).all()
    with pytest.raises(ValueError, match="before fertilization"):
        age.label_interval([-300.0], [-300.0], [nan], ["exact"], [""])


def test_log_mass_is_stable_in_both_tails_for_both_laws():
    from scipy.special import log_ndtr

    a = torch.tensor([-1.0, 8.0, -40.0, -torch.inf, 2.0, 0.3], dtype=torch.float64)
    b = torch.tensor([0.5, 9.0, -39.0, 1.0, torch.inf, 0.3000001], dtype=torch.float64)
    got = age.log_mass(a, b, "gauss")
    lb, la = log_ndtr(b.numpy()), log_ndtr(a.numpy())  # scipy's accurate log CDF (torch's ndtr loses the far tail)
    with np.errstate(divide="ignore"):
        ref = np.where(np.isinf(la), lb, lb + np.log(-np.expm1(la - lb)))
    up = a.numpy() > 0  # upper tail by symmetry
    lu, ll = log_ndtr(-a.numpy()), log_ndtr(-b.numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = np.where(up, np.where(np.isinf(ll), lu, lu + np.log(-np.expm1(ll - lu))), ref)
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-9)
    assert got[1] > -35.02 and got[2] < -700  # 8 < x < 9: -35.0136 (naive torch ndtr gives -35.032)
    got = age.log_mass(a, b, "logistic")
    keep = [0, 1, 3, 4, 5]
    sig = torch.sigmoid
    naive = torch.log(torch.where(a > 0, sig(-a) - sig(-b), sig(b) - sig(a)))
    torch.testing.assert_close(got[keep], naive[keep], rtol=1e-6, atol=0)
    assert torch.isfinite(got).all()
    assert abs(got[2].item() - (-39.0 + math.log(1 - math.exp(-1.0)))) < 1e-6  # e^b - e^a in the far tail


@pytest.mark.parametrize("kind", age.KINDS)
def test_censored_likelihood_learns_a_calibrated_interval(kind):
    torch.manual_seed(0)
    n, m, s = 6000, math.log(20000.0), 0.15
    u = torch.rand(n).clamp(1e-6, 1 - 1e-6)
    noise = torch.log(u / (1 - u)) if kind == "logistic" else math.sqrt(2) * torch.erfinv(2 * u - 1)
    true = m + s * noise
    lo = torch.exp(true).floor()  # day-resolution labels
    hi = lo + 1.0
    raw = torch.zeros(1, 2, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        age.loss(raw.expand(n, 2), lo, hi, kind).backward()
        opt.step()
    mu, scale = age.params(raw.detach())
    assert abs(mu.item() - m) < 0.02 and abs(scale.item() - s) < 0.015
    pl, ph = age.bounds(raw.detach().expand(n, 2), kind, 0.1)
    assert abs(((pl <= true) & (true <= ph)).float().mean().item() - 0.9) < 0.02  # 90% interval covers 90%


def test_unlabelled_censored_and_weighted_cells():
    raw = torch.randn(4, 2, requires_grad=True)
    nan = float("nan")
    lo = torch.tensor([nan, 1000.0, 30000.0, nan])
    hi = torch.tensor([nan, 1007.0, nan, 200.0])  # none, a week, right-censored, left-censored
    per = age.censored_nll(raw, lo, hi)
    assert per[0] == 0 and torch.isfinite(per).all() and (per[1:] > 0).all()
    age.loss(raw, lo, hi).backward()
    assert raw.grad[0].abs().sum() == 0 and raw.grad[1:].abs().sum() > 0
    empty = age.loss(raw, torch.full((4,), nan), torch.full((4,), nan))
    assert empty.item() == 0 and empty.requires_grad
    w = torch.tensor([5.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(age.loss(raw, lo, hi, weight=w), per[1])  # weights over labelled cells only


def test_bounds_scale_floor_and_initialisation():
    zero = torch.zeros(1, 2)
    mu, s = age.params(zero)
    assert abs(mu.item() - age.LOG_CENTER) < 1e-6 and abs(s.item() - (1.0 + age.SCALE_MIN)) < 1e-3
    assert age.params(torch.tensor([[0.0, -50.0]]))[1].item() >= age.SCALE_MIN - 1e-7  # float32
    for kind, k in (("logistic", math.log(19.0)), ("gauss", 1.6448536)):
        pl, ph = age.bounds(zero, kind, 0.1)
        assert abs(((ph - pl) / 2 / s).item() - k) < 1e-5


def test_holdout_groups_and_donor_weights():
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(100), 50)
    centre = np.repeat(np.exp(rng.uniform(np.log(30), np.log(30000), 100)), 50)
    lo, hi = centre.astype(np.float32), (centre + 1).astype(np.float32)
    lo[group >= 90], hi[group >= 90] = np.nan, np.nan  # ten groups without an age
    held = age.holdout_groups(group, lo, hi, 0.2, seed=1)
    assert np.array_equal(held, age.holdout_groups(group, lo, hi, 0.2, seed=1)) and 16 <= len(held) <= 20
    assert (held < 90).all()
    order = np.argsort(np.log(centre[::50][:90]))
    assert len(np.intersect1d(held, order[:18])) >= 2 and len(np.intersect1d(held, order[-18:])) >= 2  # stratified
    assert len(age.holdout_groups(group, lo, hi, 0.0)) == 0
    donor = np.repeat(np.arange(4), [10, 20, 30, 40])
    labelled = np.ones(100, bool)
    labelled[:5] = False
    w = age.group_weights(donor, labelled)
    totals = [w[donor == d].sum() for d in range(4)]
    assert np.allclose(totals, totals[0]) and abs(w[labelled].mean() - 1) < 1e-6 and (w[~labelled] == 0).all()


def age_batch(b, n):
    rng = np.random.default_rng(0)
    c = np.exp(rng.uniform(np.log(30), np.log(30000), n))
    b["age_lo"] = torch.tensor(c, dtype=torch.float32, device=DEVICE)
    b["age_hi"] = b["age_lo"] + 7.0
    return b


def test_default_is_unchanged_and_detach_leaves_the_encoder_alone(subset_dir):  # noqa: F811
    assert M.ModelConfig().age_head == "" and not hasattr(M.UnifiedModel(tiny_config()), "age_head")
    with pytest.raises(ValueError, match="age_head"):
        M.UnifiedModel(tiny_config(age_head="beta"))
    _, b, maps, panels = make(subset_dir)
    for detach in (False, True):
        torch.manual_seed(0)
        model = M.UnifiedModel(tiny_config(age_head="logistic", age_detach=detach)).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b, M.dense_mask(b["tok_doc"]), loops=2)
            raw = model.age_head(out["z"])
            losses, metrics = compute_losses(model, age_batch(dict(b), out["z"].shape[0]), out, panels, maps, 1.0)
        assert raw.dtype == torch.float32  # the head runs outside autocast
        assert list(losses)[-1] == "age" and torch.isfinite(losses["age"]) and "age/coverage" in metrics
        enc = [p for p in model.encoder.parameters() if p.requires_grad]
        grads = torch.autograd.grad(losses["age"], enc, retain_graph=True, allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in grads) != detach


def data_with_age(src: str, tmp_path: Path) -> str:
    root = tmp_path / "age_data"
    shutil.copytree(src, root)
    n = len(np.load(root / "cell_type.npy"))
    rng = np.random.default_rng(0)
    lo = np.exp(rng.uniform(np.log(30), np.log(30000), n)).astype(np.float32)
    lo[::7] = np.nan  # some cells without an age
    np.save(root / "age_lower_dpf.npy", lo)
    np.save(root / "age_upper_dpf.npy", (lo + 7).astype(np.float32))
    np.save(root / "donor.npy", np.arange(n) % 10)
    np.save(root / "dataset.npy", (np.arange(n) % 10) // 2)  # five datasets of two donors
    return str(root)


@pytest.mark.skipif(not _has_fa3(), reason="FA3 not built")
def test_age_trains_end_to_end_and_predicts(subset_dir, tmp_path, capsys):  # noqa: F811
    from brainwm_unified import infer, train
    from brainwm_unified.config import DataConfig
    from brainwm_unified.data import Subset

    data = data_with_age(subset_dir, tmp_path)
    run = str(tmp_path / "age_run")
    sets = [*TINY_RUN, "model.age_head=logistic", "data.age_holdout=0.4", "data.age_donor_weight=true"]
    sets += ['train.balancer="grad_equal"', 'train.balancer_params={"every": 1}']
    train.main(["--data", data, "--steps", "4", "--out", run, "--set", *sets])
    h = np.load(Path(run, "age_holdout.npz"))
    assert str(h["unit"]) == "dataset" and 1 <= len(h["ids"]) <= 3 and "datasets held out" in capsys.readouterr().out
    rows = [json.loads(line) for line in Path(run, "train.jsonl").read_text().splitlines()]
    assert all(math.isfinite(r["loss/age"]) and "age/coverage" in r for r in rows)
    model, _ = infer.load_model(str(Path(run, "latest.pt")), compile_blocks=False)
    torch.manual_seed(0)  # train initialises with the data seed
    fresh = M.UnifiedModel(model.cfg)
    assert (model.age_head.net[2].weight.cpu() - fresh.age_head.net[2].weight).abs().max() > 0  # in the total loss
    ds = Subset(data)
    res = infer.predict(model, ds, np.arange(ds.n_cells), DataConfig(root=data))
    assert res["age"].shape == (ds.n_cells, 2) and np.isfinite(res["age"]).all()

    detached = str(tmp_path / "age_detached")  # a loss with no encoder path under grad_equal keeps its scale
    train.main(["--data", data, "--steps", "4", "--out", detached, "--set", *sets, "model.age_detach=true"])
    rows = [json.loads(line) for line in Path(detached, "train.jsonl").read_text().splitlines()]
    scales = [r["scale/age"] for r in rows if "scale/age" in r]  # after the balancer's initial phase
    assert scales and all(math.isfinite(v) and v < 1e6 for v in scales)


def test_eval_age_scores_and_dataset_baseline(tmp_path):
    import eval_age

    lo = np.array([100.0, 1000.0, 20000.0, 30000.0], np.float32)
    hi = lo + 7
    c = np.log(age.label_center(lo, hi))
    s = eval_age.scores(c - 0.1, c + 0.1, lo, hi, 0.1)
    assert s["coverage"] == 1.0 and s["abs_err_log_median"] < 1e-6
    assert abs(s["width_relative_median"] - (math.exp(0.2) - 1)) < 1e-6
    rep = eval_age.report(c - 0.1, c + 0.1, lo, hi, np.array([True, False, False, True]), 0.1)
    assert rep["held_out/all"]["n_closed"] == 2 and rep["seen/all"]["n_closed"] == 2 and "held_out/prenatal" in rep
    root = tmp_path / "train"
    root.mkdir()
    np.save(root / "age_lower_dpf.npy", lo)
    np.save(root / "age_upper_dpf.npy", hi)
    np.save(root / "dataset.npy", np.array([0, 0, 1, 1]))
    np.save(root / "donor.npy", np.array([0, 1, 2, 3]))
    pl, ph = eval_age.dataset_mean(str(root), "donor", np.array([3]), 0.1, np.array([0, 1, 5]))
    assert pl[1] == ph[1] == pytest.approx(c[2])  # dataset 1 has one donor left
    assert pl[2] < ph[2]  # an unseen dataset falls back to the global interval
    pl, ph = eval_age.dataset_mean(str(root), "dataset", np.array([1]), 0.1, np.array([1]))
    assert pl[0] < ph[0]  # a held-out dataset: global, never its own ages


def test_age_experiments_resolve():
    from brainwm_unified import experiment

    paths = sorted(Path("configs/experiments/age").glob("[!_]*.toml"))
    assert len(paths) == 4
    for path in paths:
        for run in experiment.resolve(path):
            mcfg, dcfg, _ = experiment.build(run)
            assert (
                mcfg.age_head in age.KINDS and dcfg.age_holdout_unit == "dataset" and run["status"] in experiment.STATUS
            )
            M.UnifiedModel(mcfg)


def test_eval_age_summary_names_the_candidate_row(tmp_path):
    import csv
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_age", Path(__file__).parents[1] / "scripts/eval_age.py")
    eval_age = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_age)
    held = {"held_out/all": {"interval_score_log": 1.5, "coverage": 0.9}}
    eval_age.write_summary({"npz": "x", "model": held, "dataset_mean": held}, str(tmp_path), "candidate")
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert [r["method"] for r in rows] == ["candidate", "dataset_mean"]
    assert float(rows[0]["Age interval score, held out (log days)"]) == 1.5
    assert rows[0]["Age median error, seen (days)"] == "nan"
