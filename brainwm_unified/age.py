"""Age as a pretraining task (opt-in, model.age_head): an interval in days since fertilization from the latent.

Unit: days post fertilization (dpf), positive from the zygote on. V4 stores days relative to birth with
birth = 266 dpf (checked: 14 / 18 / 22 pcw -> -168 / -140 / -112; Carnegie 18 / 22 -> 44 / 54 dpf).

Labels are intervals [lo, hi) in dpf (NaN = open end): a reported age is truncated, so "73y" covers a year and
"14pcw" a week (RESOLUTION_DAYS); ranges and censored ages ("> 90 y") are intervals as given; a categorical-only
age falls back to its life stage's interval. The labels are interval-censored: the likelihood of a
label is the probability of its interval.

Model: an accelerated-failure-time law on u = log dpf with interval censoring. The head maps the latent to two
numbers (d -> 2): a location and a scale on u. A constant scale in u is a constant relative tolerance, so it is
tight for the young (fast change) and loose for the old (+-5% is ~3 days at 8 weeks, 2.5 years at 50 years).

  logistic  log-logistic AFT (default): closed-form CDF, bounded influence of a wrong label (|dl/dmu| <= 1/s)
  gauss     log-normal AFT (the comparison)

Loss: -log P(lo <= age < hi), the log score of the interval (strictly proper; a too wide interval loses
likelihood, a too narrow one that misses loses more; no separate width penalty, which would make it improper).
The reported interval is the central 1 - alpha one. Design review (2026-11-13): logistic over Gaussian, a scale
floor above the label widths, fp32 head, dataset-level hold-out as the primary test, donor weighting as an option.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.special import log_ndtr

GESTATION_DAYS = 266.0
YEAR = 365.25
RESOLUTION_DAYS = {
    "postnatal_y": YEAR,
    "postnatal_mo": YEAR / 12,
    "postconception_week": 7.0,
    "weeks_post_conception": 7.0,
    "lmp_month": YEAR / 12,
}
MIN_WIDTH_DAYS = 1.0  # an exact day is the day itself
NO_TIME = ("categorical_only", "unknown")
LIFE_STAGE_DPF = {  # V4 life stages (checked on samples.parquet): prenatal < birth, postnatal 0-18 y, adult >= 18 y
    "prenatal": (0.0, GESTATION_DAYS),
    "postnatal": (GESTATION_DAYS, GESTATION_DAYS + 18 * YEAR),
    "adult": (GESTATION_DAYS + 18 * YEAR, np.nan),
}
KINDS = ("logistic", "gauss")
LOG_CENTER = 8.75  # the head starts at the donor-weighted mean log dpf (~17 years)
SCALE_MIN = 0.02  # 2% relative: above a year's width at 73 (0.0137) so a training donor cannot be memorised to 0
SCALE_OFFSET = 0.5413  # softplus^-1(1): the scale starts at ~1 in log days


def label_interval(lower, upper, point, status, basis, life_stage=None):
    """Per-cell label [lo, hi) in dpf from V4's birth-relative arrays and time_status / time_basis / life_stage."""
    lower, upper, point = (np.asarray(x, np.float64) + GESTATION_DAYS for x in (lower, upper, point))
    res = np.array([max(RESOLUTION_DAYS.get(b, MIN_WIDTH_DAYS), MIN_WIDTH_DAYS) for b in basis])
    status = np.asarray(status)
    lo, hi = np.full(len(lower), np.nan), np.full(len(lower), np.nan)
    both = np.isfinite(lower) & np.isfinite(upper)
    span = both & (upper > lower)
    lo[span], hi[span] = lower[span], upper[span]
    pt = both & ~span  # a point given as lower = upper
    lo[pt], hi[pt] = lower[pt], lower[pt] + res[pt]
    right = np.isfinite(lower) & ~np.isfinite(upper)  # "> 90 y"
    lo[right] = lower[right]
    left = ~np.isfinite(lower) & np.isfinite(upper)
    hi[left] = upper[left]
    only_point = ~np.isfinite(lower) & ~np.isfinite(upper) & np.isfinite(point)
    lo[only_point], hi[only_point] = point[only_point], point[only_point] + res[only_point]
    none = np.isin(status, NO_TIME) | ~(np.isfinite(lo) | np.isfinite(hi))
    lo[none], hi[none] = np.nan, np.nan
    if life_stage is not None:  # a categorical-only age is still its life stage's interval
        for stage, (a, b) in LIFE_STAGE_DPF.items():
            m = none & (np.asarray(life_stage) == stage)
            lo[m], hi[m] = a, b
    if (lo[np.isfinite(lo)] < 0).any() or (hi[np.isfinite(hi)] <= 0).any():
        raise ValueError("an age before fertilization: check the gestation constant")
    return lo.astype(np.float32), hi.astype(np.float32)


class AgeHead(nn.Module):
    """Latent -> (location, scale) on log dpf (d -> 2), in fp32 outside autocast (bf16 spacing on u in [8, 16)
    is 0.0625, ~6% of the age)."""

    def __init__(self, z_dim: int, hidden: int, kind: str = "logistic"):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"age_head {kind!r}; valid: {', '.join(KINDS)}")
        self.kind = kind
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        with torch.autocast(z.device.type, enabled=False):
            return self.net(z.float())


def params(raw: torch.Tensor):
    """(location, scale) on log dpf."""
    return LOG_CENTER + raw[..., 0], F.softplus(raw[..., 1] + SCALE_OFFSET) + SCALE_MIN


def half_width(kind: str, alpha: float) -> float:
    """Central 1 - alpha interval of the standard law: +- this many scales."""
    if kind == "logistic":
        return math.log((2 - alpha) / alpha)
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(1 - alpha, dtype=torch.float64)).item()


def bounds(raw: torch.Tensor, kind: str, alpha: float = 0.1):
    """The predicted interval in log dpf (exp gives days)."""
    mu, s = params(raw)
    k = half_width(kind, alpha)
    return mu - k * s, mu + k * s


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x <= 0 (Maechler 2012: expm1 near 0, log1p in the tail)."""
    x = x.clamp(max=-1e-12)
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def log_mass(a: torch.Tensor, b: torch.Tensor, kind: str = "gauss") -> torch.Tensor:
    """log(F(b) - F(a)) for standardised a < b (either may be infinite), stable in both tails (float64 inside).

    logistic: F(b) - F(a) = sigmoid(b) sigmoid(-a) (1 - e^(a - b)), every factor in log space.
    gauss:    upper tail by symmetry, Phi(b) - Phi(a) = Phi(-a) - Phi(-b) when a > 0, then log_ndtr."""
    a, b = a.double(), b.double()
    if kind == "logistic":
        return F.logsigmoid(b) + F.logsigmoid(-a) + _log1mexp(a - b)
    flip = a > 0
    lo, hi = torch.where(flip, -b, a), torch.where(flip, -a, b)
    lb, la = log_ndtr(hi), log_ndtr(lo)
    return lb + _log1mexp(la - lb)


def _log_label(lo: torch.Tensor, hi: torch.Tensor):
    """Label bounds in log dpf, open ends as -inf / +inf, and the mask of labelled cells."""
    lab = torch.isfinite(lo) | torch.isfinite(hi)
    a = torch.where(torch.isfinite(lo), torch.log(lo.double().clamp(min=1e-6)), -torch.inf)
    b = torch.where(torch.isfinite(hi), torch.log(hi.double().clamp(min=1e-6)), torch.inf)
    return a, b, lab


def _standardise(x: torch.Tensor, mu: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """(x - mu) / s, with infinite bounds kept infinite but off the graph (else 0 x inf = NaN in backward)."""
    finite = torch.isfinite(x)
    z = (torch.where(finite, x, mu.detach()) - mu) / s
    return torch.where(finite, z, x)


def censored_nll(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, kind: str = "logistic") -> torch.Tensor:
    """Per-cell -log P(lo <= age < hi); 0 for unlabelled cells."""
    a, b, lab = _log_label(lo, hi)
    mu, s = (x.double() for x in params(raw))
    nll = -log_mass(_standardise(a, mu, s), _standardise(b, mu, s), kind)
    return torch.where(lab, nll, torch.zeros_like(nll)).float()


def loss(raw, lo, hi, kind: str = "logistic", weight: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted mean over labelled cells (0, keeping the graph, when the batch has none)."""
    lab = (torch.isfinite(lo) | torch.isfinite(hi)).float()
    w = lab if weight is None else lab * weight.float()
    return (censored_nll(raw, lo, hi, kind) * w).sum() / w.sum().clamp(min=1e-12) + 0.0 * raw.sum()


def label_center(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """A representative age in dpf: the closed interval's log-midpoint, else its finite end."""
    both = np.isfinite(lo) & np.isfinite(hi)
    c = np.where(np.isfinite(lo), lo, hi).astype(np.float64)
    c[both] = np.sqrt(np.maximum(lo[both], 1e-6) * np.maximum(hi[both], 1e-6))
    return c


def holdout_groups(groups: np.ndarray, lo: np.ndarray, hi: np.ndarray, fraction: float, seed: int = 0, bins: int = 5):
    """Groups (datasets or donors) whose age labels are masked in training: a seeded `fraction` of the labelled
    groups, stratified by quantile bins of the group's median log age."""
    if fraction <= 0:
        return np.zeros(0, groups.dtype)
    c = label_center(lo, hi)
    ok = np.isfinite(c)
    ids = np.unique(groups[ok])
    order = np.argsort(groups[ok], kind="stable")
    gs, lc = groups[ok][order], np.log(c[ok][order])
    starts = np.searchsorted(gs, ids)
    med = np.array([np.median(x) for x in np.split(lc, starts[1:])])
    bins = max(1, min(bins, len(ids) // 5))  # at least five groups per band
    edges = np.quantile(med, np.linspace(0, 1, bins + 1)[1:-1])
    band = np.searchsorted(edges, med, side="right")
    rng = np.random.default_rng(seed)
    held = [g for k in range(bins) for g in rng.permutation(ids[band == k])[: round(fraction * (band == k).sum())]]
    return np.sort(np.asarray(held, groups.dtype))


def group_weights(groups: np.ndarray, labelled: np.ndarray) -> np.ndarray:
    """Per-cell weight 1 / (labelled cells of its group), scaled to mean 1 over labelled cells: every donor counts
    once (the effective sample size of a donor-level label is the number of donors, not of cells)."""
    w = np.zeros(len(groups), np.float32)
    _, inv, counts = np.unique(groups[labelled], return_inverse=True, return_counts=True)
    w[labelled] = 1.0 / counts[inv]
    return (w * labelled.sum() / max(w.sum(), 1e-12)).astype(np.float32)
