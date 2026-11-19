"""Age-interval scores in days post fertilization: groups whose ages training never saw vs the rest.

    .venv/bin/python scripts/eval_age.py runs/eval/annotation/<run>_step10000.npz \
        --holdout runs/age/<run>/age_holdout.npz --train-root /data/brainwm/unified_1m \
        [--pred ridge=runs/eval/age/ridge_eval_10k.npz]

Every prediction is an interval in log dpf (exp gives days). Per group (held_out: datasets or donors whose ages
the run never saw, from its age_holdout.npz; seen: the rest) and age band
(prenatal < 266 dpf, 0-18 y, 18-65 y, > 65 y, all), on cells with a closed label interval: coverage (the
label's log-midpoint inside the predicted interval; target 1 - alpha), median width in days and relative to the
age, interval score in log days, median |error| of the interval centre in log days and in days; right-censored
labels ("> 90 y") report whether the prediction reaches above the bound. Baseline 'dataset_mean': each cell's
dataset's mean log age and its alpha/2, 1 - alpha/2 quantiles over the training cells not held out, else the
global ones (it can only read the dataset: a model at its level reads batch, not biology). Prints JSON.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

from brainwm_unified import age

BANDS = {
    "prenatal": (0.0, age.GESTATION_DAYS),
    "0-18y": (age.GESTATION_DAYS, age.GESTATION_DAYS + 18 * 365.25),
    "18-65y": (age.GESTATION_DAYS + 18 * 365.25, age.GESTATION_DAYS + 65 * 365.25),
    ">65y": (age.GESTATION_DAYS + 65 * 365.25, np.inf),
    "all": (0.0, np.inf),
}


def scores(log_lo, log_hi, lo, hi, alpha):
    """Metrics of predicted log-dpf intervals against label intervals [lo, hi) in dpf (NaN = open)."""
    closed = np.isfinite(lo) & np.isfinite(hi)
    c = np.log(age.label_center(lo, hi))
    centre = 0.5 * (log_lo + log_hi)
    out = {"n_closed": int(closed.sum())}
    if closed.any():
        cc, pl, ph = c[closed], log_lo[closed], log_hi[closed]
        a, b = np.log(lo[closed]), np.log(hi[closed])
        gap = np.clip(pl - b, 0, None) + np.clip(a - ph, 0, None)
        out |= {
            "coverage": float(((pl <= cc) & (cc <= ph)).mean()),
            "width_days_median": float(np.median(np.exp(ph) - np.exp(pl))),
            "width_relative_median": float(np.median(np.exp(ph - pl) - 1)),
            "interval_score_log": float(np.mean((ph - pl) + (2 / alpha) * gap)),
            "abs_err_log_median": float(np.median(np.abs(centre[closed] - cc))),
            "abs_err_days_median": float(np.median(np.abs(np.exp(centre[closed]) - np.exp(cc)))),
        }
    right = np.isfinite(lo) & ~np.isfinite(hi)
    if right.any():
        out["right_censored_reached"] = float((log_hi[right] >= np.log(lo[right])).mean())
    return out


# Columns of summary.csv (--out): metric name -> (group/band, score). A campaign picks its metrics from these.
SUMMARY = {
    "Age interval score, held out (log days)": ("held_out/all", "interval_score_log"),
    "Age coverage, held out": ("held_out/all", "coverage"),
    "Age median error, held out (days)": ("held_out/all", "abs_err_days_median"),
    "Age median width, held out (days)": ("held_out/all", "width_days_median"),
    "Age median error, seen (days)": ("seen/all", "abs_err_days_median"),
}


def write_summary(out, folder, name):
    """summary.csv with one row per method ("model" is written as `name`) and the SUMMARY columns."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", *SUMMARY])
        for method, res in out.items():
            if isinstance(res, dict):
                row = [res.get(group, {}).get(score, float("nan")) for group, score in SUMMARY.values()]
                w.writerow([name if method == "model" else method, *row])


def report(log_lo, log_hi, lo, hi, hidden, alpha):
    c = age.label_center(lo, hi)
    groups = {"held_out": hidden, "seen": ~hidden}
    res = {}
    for g, gm in groups.items():
        for band, (b0, b1) in BANDS.items():
            m = gm & np.isfinite(c) & (c >= b0) & (c < b1)
            if m.any():
                res[f"{g}/{band}"] = scores(log_lo[m], log_hi[m], lo[m], hi[m], alpha)
    return res


def dataset_mean(train_root, unit, held, alpha, datasets):
    """Per-dataset interval from the training cells not held out (log-age mean and quantiles), else global."""
    lo = np.load(f"{train_root}/age_lower_dpf.npy")
    hi = np.load(f"{train_root}/age_upper_dpf.npy")
    ds = np.load(f"{train_root}/dataset.npy")
    group = np.load(f"{train_root}/{unit}.npy")
    c = age.label_center(lo, hi)
    ok = np.isfinite(lo) & np.isfinite(hi) & ~np.isin(group, held)
    lc = np.log(c[ok])
    glob = (np.mean(lc), *np.quantile(lc, [alpha / 2, 1 - alpha / 2]))
    table = {}
    for d in np.unique(ds[ok]):
        x = lc[ds[ok] == d]
        table[d] = (np.mean(x), *np.quantile(x, [alpha / 2, 1 - alpha / 2]))
    got = np.array([table.get(d, glob) for d in datasets])
    return got[:, 1], got[:, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--holdout", required=True, help="the run's age_holdout.npz (unit, ids)")
    ap.add_argument("--train-root", default="/data/brainwm/unified_1m")
    ap.add_argument("--pred", action="append", default=[], help="NAME=npz with age_log_lo / age_log_hi")
    ap.add_argument("--out", help="also write <out>/summary.csv (one row per method)")
    ap.add_argument("--name", default="model", help="row name of the model's own prediction in summary.csv")
    args = ap.parse_args()
    z = np.load(args.npz)
    h = np.load(args.holdout)
    unit, held = str(h["unit"]), h["ids"]
    lo, hi = z["age_lower_dpf"], z["age_upper_dpf"]
    hidden = np.isin(z[unit], held)
    alpha = float(z["age_alpha"]) if "age_alpha" in z.files else 0.1
    out = {"npz": args.npz, "alpha": alpha, "holdout_unit": unit, "held_out": len(held)}
    if "age_raw" in z.files:
        pl, ph = age.bounds(torch.as_tensor(z["age_raw"]), str(z["age_kind"]), alpha)
        out["model"] = report(pl.numpy(), ph.numpy(), lo, hi, hidden, alpha)
    base = dataset_mean(args.train_root, unit, held, alpha, z["dataset"])
    out["dataset_mean"] = report(*base, lo, hi, hidden, alpha)
    for spec in args.pred:
        name, path = spec.split("=", 1)
        p = np.load(path)
        out[name] = report(p["age_log_lo"], p["age_log_hi"], lo, hi, hidden, alpha)
    if args.out:
        if "model" not in out:
            raise ValueError(f"{args.npz} has no age prediction: the model has no age head")
        write_summary(out, args.out, args.name)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
