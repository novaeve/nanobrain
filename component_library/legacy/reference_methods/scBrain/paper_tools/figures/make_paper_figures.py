#!/usr/bin/env python3
"""Build paper figures from completed benchmark artifacts.

This script reads existing logs, CSV summaries, and saved latent features only.
No training or new experiments are performed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from src.models.mjm import mjm, mjm_1
from src.models.mjm_nlevel import mjm_1_nlevel
from src.utils.dataloader import build_dataloaders
from src.utils.dataloader_kukanja import build_dataloaders_kukanja
EXPERIMENTS = PROJECT / "experiments"
DATA = PROJECT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
SEAAD_H5AD = Path(
    "./data/case_study/"
    "input_bundle_20260408_061939_336043/"
    "prepared_SEAAD_MTG_MERFISH.2024-12-11.h5ad"
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.7,
    }
)

PALETTE = {
    "mjm": "#C84C09",
    "mjm_light": "#F2B27E",
    "foundation": "#2A6F97",
    "foundation_light": "#9BC1D9",
    "conventional": "#5A5A5A",
    "conventional_light": "#B6B6B6",
    "batch": "#4C78A8",
    "bio": "#76B7B2",
    "overall": "#C84C09",
    "correct": "#1B8A5A",
    "wrong": "#B97A9F",
    "grid": "#D8D8D8",
}

LEVEL_LABELS = {
    "SEA_AD_MTP_ST": ["Class", "Subclass", "Supertype"],
    "Kukanja_MS": ["Level0", "Level1.5", "Level2", "Level3"],
    "Kukanja_EAE": ["Broad", "Medium", "Medium+DA", "High"],
}

LEVEL_KEYS = {
    "SEA-AD": ["CLASS", "SUBCLASS", "SUPERTYPE"],
    "Kukanja MS": ["LEVEL0", "LEVEL1_5", "LEVEL2", "LEVEL3"],
    "Kukanja EAE": ["ANNOTATION_BROAD", "ANNOTATION_MEDIUM", "ANNOTATION_MEDIUM_DA", "ANNOTATION_HIGH"],
}


def friendly_batch_label(name: str) -> str:
    mapping = {
        "mjm (origin)": "scBrain",
        "mjm (dec256)": "scBrain (decoder 256)",
        "mjm_1 (feat,dec128)": "scBrain\n(feature residual, 128)",
        "mjm_1 (logit,dec128)": "scBrain\n(logit residual, 128)",
        "mjm_1 (none,dec128)": "scBrain\n(independent, 128)",
        "mjm_1 (feat,dec256)": "scBrain\n(feature residual, 256)",
        "mjm_1 (feat,dec64)": "scBrain\n(feature residual, 64)",
        "mjm_2a (depth)": "scBrain\n(depth)",
        "mjm_2b (vol)": "scBrain\n(volume)",
        "mjm_2c (FiLM)": "scBrain\n(FiLM)",
        "mjm_2a_lr (depth)": "scBrain linear probe\n(depth)",
        "mjm_2c_lr (FiLM)": "scBrain linear probe\n(FiLM)",
        "mjm_lr": "scBrain linear probe",
        "mjm_1_lr": "scBrain linear probe\n(feature residual)",
        "mjm_attn": "scBrain\n(attention)",
        "scGPT (scratch)": "scGPT (scratch)",
        "scGPT_brain (freeze)": "scGPT-Brain\n(frozen)",
        "scGPT_brain (full)": "scGPT-Brain\n(fine-tuned)",
        "scGPT_human (freeze)": "scGPT-Human\n(frozen)",
        "scGPT_human (full)": "scGPT-Human\n(fine-tuned)",
    }
    return mapping.get(name, name.replace("_", " "))


def save(fig, stem):
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)


def style_axis(ax, xgrid=False, ygrid=False):
    ax.set_facecolor("white")
    if xgrid:
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)
    if ygrid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def add_panel_label(ax, label: str):
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def parse_metric_log(path: Path):
    """Parse macro/micro F1 metrics from a log file."""
    macro_pat = re.compile(
        r"\[(?P<level>[A-Z0-9_]+)\]\s+Macro\s+-\s+(?:P|Precision):\s*(?P<p>[0-9.]+)\s*(?:\||)\s*(?:R|Recall):\s*(?P<r>[0-9.]+)\s*(?:\||)\s*F1:\s*(?P<f1>[0-9.]+)"
    )
    micro_pat = re.compile(
        r"\[(?P<level>[A-Z0-9_]+)\]\s+Micro\s+-\s+(?:(?:P|Precision):\s*[0-9.]+\s*(?:\||)\s*(?:R|Recall):\s*[0-9.]+\s*(?:\||)\s*)?F1:\s*(?P<f1>[0-9.]+)"
    )
    auc_pat = re.compile(
        r"\[(?P<level>[A-Z0-9_]+)\]\s+AUC-ROC\s+-\s+Macro:?\s*(?P<macro>[0-9.]+)\s*(?:\|\s*|\s+)"
        r"Micro:?\s*(?P<micro>[0-9.]+)"
    )
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = macro_pat.search(line)
        if m:
            level = m.group("level")
            out.setdefault(level, {})
            out[level]["macro_p"] = float(m.group("p"))
            out[level]["macro_r"] = float(m.group("r"))
            out[level]["macro_f1"] = float(m.group("f1"))
            continue
        m = micro_pat.search(line)
        if m:
            level = m.group("level")
            out.setdefault(level, {})
            out[level]["micro_f1"] = float(m.group("f1"))
            continue
        m = auc_pat.search(line)
        if m:
            level = m.group("level")
            out.setdefault(level, {})
            out[level]["auc_macro"] = float(m.group("macro"))
            out[level]["auc_micro"] = float(m.group("micro"))
    return out


def harmonic_f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def parse_per_class_log(path: Path):
    level_pat = re.compile(r"\[(?P<level>[A-Z0-9_]+)\].*Per-class(?: P/R)?:\s*(?P<body>.*)$")
    class_pat = re.compile(r"C(?P<idx>\d+)\(S=(?P<sup>\d+)\):\s*P=(?P<p>[0-9.]+)/R=(?P<r>[0-9.]+)")
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = level_pat.search(line)
        if not m:
            continue
        rows = []
        for cm in class_pat.finditer(m.group("body")):
            p = float(cm.group("p"))
            r = float(cm.group("r"))
            rows.append(
                {
                    "idx": int(cm.group("idx")),
                    "support": int(cm.group("sup")),
                    "precision": p,
                    "recall": r,
                    "f1": harmonic_f1(p, r),
                }
            )
        out[m.group("level")] = rows
    return out


def load_split_labels(npz_path: Path, dataset: str):
    data = np.load(npz_path, allow_pickle=True)
    if dataset == "SEA_AD_MTP_ST":
        batch = data["batch_donor"]
        unique = np.unique(batch)
        rng = np.random.RandomState(3028)
        rng.shuffle(unique)
        test_mask = np.isin(batch, unique[:6])
        return {
            "test_mask": test_mask,
            "batch": batch[test_mask],
            "levels": {
                "Class": data["y_class"][test_mask],
                "Subclass": data["y_subclass"][test_mask],
                "Supertype": data["y_supertype"][test_mask],
            },
            "meta": data["meta"].item(),
        }
    if dataset == "Kukanja_MS":
        sample_ids = data["sample_ids"]
        unique = np.unique(sample_ids)
        rng = np.random.RandomState(3028)
        rng.shuffle(unique)
        n_test, n_val = 2, 1
        test_mask = np.isin(sample_ids, unique[:n_test])
        meta = data["meta"].item()
        names = meta["label_names"]
        levels = {name: data[f"y_{name}"][test_mask] for name in names}
        return {
            "test_mask": test_mask,
            "batch": sample_ids[test_mask],
            "levels": levels,
            "meta": meta,
        }
    if dataset == "Kukanja_EAE":
        sample_ids = data["sample_ids"]
        unique = np.unique(sample_ids)
        rng = np.random.RandomState(3028)
        rng.shuffle(unique)
        n_samples = len(unique)
        n_test = max(2, int(n_samples * 0.2))
        n_val = max(1, int(n_samples * 0.1))
        test_mask = np.isin(sample_ids, unique[:n_test])
        meta = data["meta"].item()
        levels = {
            "Broad": data["y_annotation_broad"][test_mask],
            "Medium": data["y_annotation_medium"][test_mask],
            "Medium_DA": data["y_annotation_medium_da"][test_mask],
            "High": data["y_annotation_high"][test_mask],
        }
        return {
            "test_mask": test_mask,
            "batch": sample_ids[test_mask],
            "levels": levels,
            "meta": meta,
        }
    raise ValueError(dataset)


def select_top_classes(labels, inv_map, n=8):
    vals, cnt = np.unique(labels, return_counts=True)
    order = np.lexsort((vals, -cnt))[:n]
    selected = [int(vals[i]) for i in order]
    names = [inv_map.get(v, f"C{v}") for v in selected]
    return selected, names, cnt[order]


def compact_case_label(label: str) -> str:
    mapping = {
        "Neu_Ex_Glu_B": "Exc-B",
        "Neu_In_Gly": "In-Gly",
        "Neu_In_GABA_B": "In-GABA-B",
        "VEC_A": "VEC-A",
        "MOL5/6": "MOL5/6",
        "MOL2": "MOL2",
        "Astro": "Astro",
        "Schw": "Schw",
        "Other": "Other",
    }
    return mapping.get(label, label.replace("_", "-"))


def load_seaad_category_names():
    adata = ad.read_h5ad(SEAAD_H5AD, backed="r")
    try:
        return {
            "Subclass": list(adata.obs["Subclass"].dtype.categories),
            "Supertype": list(adata.obs["Supertype"].dtype.categories),
        }
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


@torch.no_grad()
def extract_mjm_c3_features(x_np, ckpt_rel, residual_mode, dec_hidden_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mjm_1(
        input_dim=140,
        latent_dim=20,
        e_layers=3,
        d_layers=1,
        enc_hidden_dim=256,
        dec_hidden_dim=dec_hidden_dim,
        expansion_factor=2.67,
        dropout=0.3,
        output_num=[3, 24, 137],
        residual_mode=residual_mode,
    )
    model.load_state_dict(torch.load(PROJECT / ckpt_rel, map_location=device))
    model.to(device).eval()
    batch_size = 32768 if device.type == "cuda" else 4096
    outputs = []
    for start in range(0, len(x_np), batch_size):
        xb = torch.from_numpy(x_np[start : start + batch_size]).to(device)
        h1 = model.E1(xb)
        h2 = model.E2(h1)
        h = model.E3(h2)
        c1 = model.D1(h1)
        logits1 = model.head1(c1)
        if residual_mode == "feature":
            c2 = model.D2(h2) + c1
        elif residual_mode == "logit":
            c2 = model.D2(h2) + model.proj_c1_to_c2(logits1)
        else:
            c2 = model.D2(h2)
        logits2 = model.head2(c2)
        if residual_mode == "feature":
            c3 = model.D3(h) + c2
        elif residual_mode == "logit":
            c3 = model.D3(h) + model.proj_c2_to_c3(logits2)
        else:
            c3 = model.D3(h)
        outputs.append(c3.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def linear_probe_macro_f1(features, labels):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pred = cross_val_predict(clf, features, labels, cv=cv, method="predict", n_jobs=1)
    return f1_score(labels, pred, average="macro")


def project_representation(features):
    scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=2).fit_transform(scaled)


def draw_boundary_panel(ax, coords, labels, class_names, class_colors, title, score):
    unique = np.unique(labels)
    code_to_idx = {int(code): idx for idx, code in enumerate(unique)}
    y_idx = np.array([code_to_idx[int(x)] for x in labels], dtype=int)

    plane_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3.0, gamma="scale"))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pred_cv = cross_val_predict(plane_clf, coords, y_idx, cv=cv, method="predict", n_jobs=1)
    mis = pred_cv != y_idx

    plane_fit = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3.0, gamma="scale"))
    plane_fit.fit(coords, y_idx)

    x_pad = 0.08 * max(coords[:, 0].ptp(), 1e-6)
    y_pad = 0.08 * max(coords[:, 1].ptp(), 1e-6)
    x_min, x_max = coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad
    y_min, y_max = coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 320),
        np.linspace(y_min, y_max, 320),
    )
    zz = plane_fit.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    bg_colors = [sns.desaturate(c, 0.55) for c in class_colors]
    cmap = ListedColormap(bg_colors)
    levels = np.arange(len(unique) + 1) - 0.5
    ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.24, antialiased=True)
    ax.contour(xx, yy, zz, levels=levels, colors="white", linewidths=0.5, alpha=0.9)

    for idx, (name, color) in enumerate(zip(class_names, class_colors)):
        pts = coords[y_idx == idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            color=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            rasterized=True,
            label=name,
        )
    if mis.any():
        ax.scatter(
            coords[mis, 0],
            coords[mis, 1],
            s=28,
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.8,
            rasterized=True,
        )

    ax.set_title(f"{title}\nMacro-F1 = {score:.3f}", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#FCFCFC")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#D6D6D6")


def plot_primary_overview():
    """Grouped bars for the primary endpoint comparison across datasets."""
    data = {
        "SEA-AD": [
            ("scBrain", 0.7611, 0.8851, PALETTE["mjm"], PALETTE["mjm_light"]),
            ("Geneformer\n(fine-tuned)", 0.7489, 0.8786, PALETTE["foundation"], PALETTE["foundation_light"]),
            ("Logistic regression", 0.8080, 0.8995, PALETTE["conventional"], PALETTE["conventional_light"]),
        ],
        "Kukanja MS": [
            ("scBrain", 0.8551, 0.8977, PALETTE["mjm"], PALETTE["mjm_light"]),
            ("Geneformer\n(fine-tuned)", 0.8448, 0.9014, PALETTE["foundation"], PALETTE["foundation_light"]),
            ("Stack", 0.8020, 0.8721, PALETTE["conventional"], PALETTE["conventional_light"]),
        ],
        "Kukanja EAE": [
            ("scBrain", 0.6971, 0.8006, PALETTE["mjm"], PALETTE["mjm_light"]),
            ("UCE\n(fine-tuned)", 0.5770, 0.7160, PALETTE["foundation"], PALETTE["foundation_light"]),
            ("scSimilarity", 0.6275, 0.7612, PALETTE["conventional"], PALETTE["conventional_light"]),
        ],
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.25), sharey=False)
    for ax, (title, rows) in zip(axes, data.items()):
        y = np.arange(len(rows))
        h = 0.30
        ax.barh(y - h / 2, [r[1] for r in rows], height=h, color=[r[3] for r in rows], edgecolor="none")
        ax.barh(y + h / 2, [r[2] for r in rows], height=h, color=[r[4] for r in rows], edgecolor="none")
        for yy, (label, macro, micro, dark, light) in zip(y, rows):
            ax.text(macro + 0.012, yy - h / 2, f"{macro:.3f}", va="center", ha="left", fontsize=7.2, color=dark)
            ax.text(micro + 0.012, yy + h / 2, f"{micro:.3f}", va="center", ha="left", fontsize=7.2, color=dark)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        style_axis(ax, xgrid=True)
        ax.set_title(title, pad=8)
        ax.set_xlabel("F1")
    axes[0].set_ylabel("Model")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#666666"),
        plt.Rectangle((0, 0), 1, 1, color="#BBBBBB"),
    ]
    axes[-1].legend(handles, ["Macro-F1", "Micro-F1"], frameon=False, loc="lower right")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_primary_overview")
    plt.close(fig)


def plot_hierarchy_lines():
    """Macro F1 across hierarchy levels for scBrain vs the strongest baseline."""
    seaad = {
        "scBrain": parse_metric_log(EXPERIMENTS / "mjm_1/SEA_AD_MTP_ST/record_e3_d1_enc256_dec256_z20_s3028_resfeature.log"),
        "Logistic regression": parse_metric_log(EXPERIMENTS / "sklearn_baselines/lr/SEA_AD_MTP_ST/record_s3028.log"),
    }
    ms = {
        "scBrain": parse_metric_log(EXPERIMENTS / "kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log"),
        "Stack": parse_metric_log(EXPERIMENTS / "kukanja/ms_stack_full_g5.launch.log"),
    }
    eae = {
        "scBrain": parse_metric_log(EXPERIMENTS / "kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log"),
        "UCE": parse_metric_log(EXPERIMENTS / "kukanja/Kukanja_EAE/record_uce_full_mouse_s3028.log"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    plots = [
        ("SEA-AD vs Logistic regression", LEVEL_KEYS["SEA-AD"], ["Class", "Subclass", "Supertype"], seaad),
        ("Kukanja MS vs Stack", LEVEL_KEYS["Kukanja MS"], ["Level0", "Level1.5", "Level2", "Level3"], ms),
        (
            "Kukanja EAE vs UCE",
            LEVEL_KEYS["Kukanja EAE"],
            ["Broad", "Medium", "Medium+DA", "High"],
            eae,
        ),
    ]
    for ax, (title, keys, labels, methods) in zip(axes, plots):
        x = np.arange(len(keys))
        for name, metrics, color, marker in [
            ("scBrain", methods["scBrain"], PALETTE["mjm"], "o"),
            (list(methods.keys())[1], list(methods.values())[1], PALETTE["conventional"], "s"),
        ]:
            vals = [metrics[level]["macro_f1"] for level in keys]
            ax.plot(x, vals, marker=marker, ms=5, lw=2, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.3, 1.0)
        style_axis(ax, ygrid=True)
        ax.set_title(title)
    axes[0].set_ylabel("Macro-F1")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_hierarchy_levels")
    plt.close(fig)


def plot_level_metrics():
    """Detailed level-wise precision/recall/F1 and AUC for the best scBrain checkpoints."""
    configs = [
        (
            "SEA-AD",
            LEVEL_KEYS["SEA-AD"],
            ["Class", "Subclass", "Supertype"],
            parse_metric_log(EXPERIMENTS / "mjm_1/SEA_AD_MTP_ST/record_e3_d1_enc256_dec256_z20_s3028_resfeature.log"),
        ),
        (
            "Kukanja MS",
            LEVEL_KEYS["Kukanja MS"],
            ["Level0", "Level1.5", "Level2", "Level3"],
            parse_metric_log(EXPERIMENTS / "kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log"),
        ),
        (
            "Kukanja EAE",
            LEVEL_KEYS["Kukanja EAE"],
            ["Broad", "Medium", "Medium+DA", "High"],
            parse_metric_log(EXPERIMENTS / "kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log"),
        ),
    ]
    metric_colors = {
        "macro_p": "#0072B2",
        "macro_r": "#E69F00",
        "macro_f1": PALETTE["mjm"],
        "auc_macro": "#009E73",
        "auc_micro": "#CC79A7",
    }
    metric_labels = {
        "macro_p": "Macro P",
        "macro_r": "Macro R",
        "macro_f1": "Macro F1",
        "auc_macro": "Macro AUC",
        "auc_micro": "Micro AUC",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.3), sharex="col")
    top_handles = None
    bottom_handles = None
    for col, (title, keys, labels, metrics) in enumerate(configs):
        x = np.arange(len(keys))

        ax = axes[0, col]
        for key in ["macro_p", "macro_r", "macro_f1"]:
            line = ax.plot(
                x,
                [metrics[k][key] for k in keys],
                marker="o",
                ms=4.5,
                lw=2,
                color=metric_colors[key],
                label=metric_labels[key],
            )[0]
            if col == 0:
                top_handles = top_handles or []
                top_handles.append(line)
        ax.set_title(title)
        ax.set_ylim(0.55 if title == "SEA-AD" else 0.70 if title == "Kukanja EAE" else 0.78, 1.0)
        style_axis(ax, ygrid=True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")

        ax2 = axes[1, col]
        for key in ["auc_macro", "auc_micro"]:
            line = ax2.plot(
                x,
                [metrics[k][key] for k in keys],
                marker="o",
                ms=4.5,
                lw=2,
                color=metric_colors[key],
                label=metric_labels[key],
            )[0]
            if col == 0:
                bottom_handles = bottom_handles or []
                bottom_handles.append(line)
        ax2.set_ylim(0.95 if title == "SEA-AD" else 0.97, 1.0)
        style_axis(ax2, ygrid=True)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=18, ha="right")

    axes[0, 0].set_ylabel("Precision / Recall / F1")
    axes[1, 0].set_ylabel("AUC")
    fig.legend(top_handles, [h.get_label() for h in top_handles], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.legend(bottom_handles, [h.get_label() for h in bottom_handles], frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save(fig, "fig_level_metrics")
    plt.close(fig)


def plot_classwise_distributions():
    configs = [
        (
            "SEA-AD",
            EXPERIMENTS / "mjm_1/SEA_AD_MTP_ST/record_e3_d1_enc256_dec256_z20_s3028_resfeature.log",
            LEVEL_KEYS["SEA-AD"],
            ["Class", "Subclass", "Supertype"],
        ),
        (
            "Kukanja MS",
            EXPERIMENTS / "kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log",
            LEVEL_KEYS["Kukanja MS"],
            ["Level0", "Level1.5", "Level2", "Level3"],
        ),
        (
            "Kukanja EAE",
            EXPERIMENTS / "kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log",
            LEVEL_KEYS["Kukanja EAE"],
            ["Broad", "Medium", "Medium+DA", "High"],
        ),
    ]
    level_colors = ["#D5E7F4", "#A7CDE2", "#6EA6C8", "#2A6F97"]
    rng = np.random.RandomState(3028)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.1),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )
    legend_sizes = None
    legend_supports = None
    for ax, (title, path, level_keys, display_labels) in zip(axes, configs):
        metrics = parse_metric_log(path)
        class_rows = parse_per_class_log(path)
        plot_rows = []
        tick_labels = []
        support_vals = []
        for level_key, display_label in zip(level_keys, display_labels):
            rows = class_rows[level_key]
            tick_labels.append(f"{display_label}\n(n={len(rows)})")
            for row in rows:
                support_vals.append(row["support"])
                plot_rows.append(
                    {
                        "level": display_label,
                        "f1": row["f1"],
                        "support": row["support"],
                    }
                )
        df = pd.DataFrame(plot_rows)

        sns.boxplot(
            data=df,
            x="level",
            y="f1",
            ax=ax,
            color="#EEF2F5",
            width=0.56,
            fliersize=0,
            linewidth=1.0,
        )

        support_vals = np.asarray(support_vals, dtype=float)
        sup_min = np.log10(max(1.0, support_vals.min()))
        sup_max = np.log10(max(1.0, support_vals.max()))
        denom = max(sup_max - sup_min, 1e-6)
        support_quantiles = np.quantile(support_vals, [0.2, 0.6, 0.95]).astype(int)
        if legend_supports is None:
            legend_supports = support_quantiles.tolist()
            legend_sizes = []
            for support in legend_supports:
                scaled = (np.log10(max(1, support)) - sup_min) / denom
                legend_sizes.append(16 + 52 * scaled)

        for pos, (level_key, display_label) in enumerate(zip(level_keys, display_labels)):
            rows = class_rows[level_key]
            xs = pos + rng.uniform(-0.17, 0.17, size=len(rows))
            ys = np.array([row["f1"] for row in rows], dtype=float)
            sizes = []
            for row in rows:
                scaled = (np.log10(max(1, row["support"])) - sup_min) / denom
                sizes.append(16 + 52 * scaled)
            ax.scatter(
                xs,
                ys,
                s=sizes,
                color=level_colors[min(pos, len(level_colors) - 1)],
                alpha=0.52,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
            ax.scatter(
                [pos],
                [metrics[level_key]["macro_f1"]],
                marker="_",
                s=300,
                color=PALETTE["mjm"],
                linewidths=2.0,
                zorder=4,
            )
            ax.text(
                pos,
                metrics[level_key]["macro_f1"] + 0.035,
                f"{metrics[level_key]['macro_f1']:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color=PALETTE["mjm"],
            )

        ax.set_title(title, pad=8)
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("")
        ax.set_ylim(0.0, 1.02)
        style_axis(ax, ygrid=True)
    axes[0].set_ylabel("Per-class F1")
    for ax in axes[1:]:
        ax.set_ylabel("")

    macro_handle = plt.Line2D(
        [0],
        [0],
        marker="_",
        color=PALETTE["mjm"],
        markersize=12,
        linewidth=0,
        label="Macro-F1",
    )
    support_handles = []
    support_labels = []
    if legend_supports is not None:
        for size, support in zip(legend_sizes, legend_supports):
            support_handles.append(
                plt.scatter(
                    [],
                    [],
                    s=size,
                    color=level_colors[2],
                    alpha=0.52,
                    edgecolors="white",
                    linewidths=0.35,
                )
            )
            support_labels.append(f"support {support}")
    fig.legend(
        [macro_handle] + support_handles,
        ["Macro-F1 marker"] + support_labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.text(
        0.5,
        -0.01,
        "Boxplots summarize classwise spread; bubble size scales with class support.",
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save(fig, "fig_classwise_distributions")
    plt.close(fig)


@torch.no_grad()
def plot_roc_curves():
    """Micro-average ROC curves for the best residual scBrain checkpoint at each dataset/level."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = [
        (
            "SEA-AD",
            ["Class", "Subclass", "Supertype"],
            ["CLASS", "SUBCLASS", "SUPERTYPE"],
            build_dataloaders(
                npz_path=str(DATA / "global_data.npz"),
                batch_size=2048,
                seed=3028,
                num_workers=0,
            ),
            mjm_1(
                input_dim=140,
                latent_dim=20,
                e_layers=3,
                d_layers=1,
                enc_hidden_dim=256,
                dec_hidden_dim=256,
                expansion_factor=2.67,
                dropout=0.3,
                output_num=[3, 24, 137],
                residual_mode="feature",
            ),
            EXPERIMENTS / "mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec256_z20_s3028_resfeature/best_model.pth",
        ),
        (
            "Kukanja MS",
            ["Level0", "Level1.5", "Level2", "Level3"],
            ["LEVEL0", "LEVEL1_5", "LEVEL2", "LEVEL3"],
            build_dataloaders_kukanja(
                str(DATA / "kukanja/kukanja_ms.npz"),
                batch_size=2048,
                seed=3028,
                num_workers=0,
            ),
            mjm_1_nlevel(
                input_dim=266,
                latent_dim=20,
                e_layers=3,
                d_layers=1,
                enc_hidden_dim=256,
                dec_hidden_dim=128,
                expansion_factor=2.67,
                dropout=0.3,
                output_num=[13, 14, 25, 27],
                residual_mode="feature",
            ),
            EXPERIMENTS / "kukanja/Kukanja_MS/checkpoints/mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature/best_model.pth",
        ),
        (
            "Kukanja EAE",
            ["Broad", "Medium", "Medium+DA", "High"],
            ["ANNOTATION_BROAD", "ANNOTATION_MEDIUM", "ANNOTATION_MEDIUM_DA", "ANNOTATION_HIGH"],
            build_dataloaders_kukanja(
                str(DATA / "kukanja/kukanja_eae.npz"),
                batch_size=2048,
                seed=3028,
                num_workers=0,
            ),
            mjm_1_nlevel(
                input_dim=239,
                latent_dim=20,
                e_layers=3,
                d_layers=1,
                enc_hidden_dim=256,
                dec_hidden_dim=256,
                expansion_factor=2.67,
                dropout=0.3,
                output_num=[8, 16, 23, 41],
                residual_mode="feature",
            ),
            EXPERIMENTS / "kukanja/Kukanja_EAE/checkpoints/mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature/best_model.pth",
        ),
    ]
    colors = ["#0072B2", "#E69F00", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.65), sharex=True, sharey=True)
    for ax, (title, labels, levels, loaders, model, ckpt) in zip(axes, items):
        _, _, test_loader = loaders[:3]
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()

        probs = {lvl: [] for lvl in levels}
        trues = {lvl: [] for lvl in levels}
        for batch in test_loader:
            if title == "SEA-AD":
                x = torch.log1p(batch["X"].to(device))
                recon, logits_list, _ = model(x)
                y_list = [batch["y_class"].cpu().numpy(), batch["y_subclass"].cpu().numpy(), batch["y_supertype"].cpu().numpy()]
            else:
                x = torch.log1p(batch["X"].to(device))
                recon, logits_list, _ = model(x)
                if title == "Kukanja MS":
                    y_list = [batch[f"y_{name}"].cpu().numpy() for name in ["level0", "level1_5", "level2", "level3"]]
                else:
                    y_list = [batch[f"y_{name}"].cpu().numpy() for name in ["annotation_broad", "annotation_medium", "annotation_medium_da", "annotation_high"]]
            for lvl, y_true, logit in zip(levels, y_list, logits_list):
                probs[lvl].append(F.softmax(logit, dim=-1).cpu().numpy())
                trues[lvl].append(y_true)

        for color, lvl, short_label in zip(colors, levels, labels):
            y_true = np.concatenate(trues[lvl], axis=0)
            y_prob = np.concatenate(probs[lvl], axis=0)
            y_onehot = np.zeros_like(y_prob)
            y_onehot[np.arange(len(y_true)), y_true] = 1
            fpr, tpr, _ = roc_curve(y_onehot.ravel(), y_prob.ravel())
            auc_val = roc_auc_score(y_onehot.ravel(), y_prob.ravel())
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{short_label} ({auc_val:.4f})")

        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
        ax.set_title(title)
        ax.set_xlim(0, 0.10)
        ax.set_ylim(0, 1)
        style_axis(ax, xgrid=True, ygrid=True)
        ax.set_xlabel("False Positive Rate (0-0.10)")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("True Positive Rate")
    fig.tight_layout()
    save(fig, "fig_roc_curves")
    plt.close(fig)


def plot_batch_integration():
    """Grouped horizontal bars for scIB batch integration."""
    csv = EXPERIMENTS / "batch_integration/SEA_AD_MTP_ST/scib_fast_results.csv"
    df = pd.read_csv(csv, index_col=0)
    order = df.sort_values("overall", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    y = np.arange(len(order))
    h = 0.20
    ax.barh(y - h, order["batch_score"], height=h, color=PALETTE["batch"], label="Batch")
    ax.barh(y, order["bio_score"], height=h, color=PALETTE["bio"], label="Bio")
    ax.barh(y + h, order["overall"], height=h, color=PALETTE["overall"], label="Overall")
    for yy, row in zip(y, order.itertuples()):
        ax.text(row.overall + 0.008, yy + h, f"{row.overall:.3f}", va="center", fontsize=7.2, color=PALETTE["overall"])
    ax.set_yticks(y)
    ax.set_yticklabels([friendly_batch_label(idx) for idx in order.index])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.8)
    style_axis(ax, xgrid=True)
    ax.set_xlabel("scIB score")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_batch_integration")
    plt.close(fig)


def plot_ablation():
    """SEA-AD residual-mode and decoder-width ablations."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    left = [
        ("independent\n(128)", 0.6887, 0.8661),
        ("logit\n(128)", 0.6997, 0.8681),
        ("feature\n(128)", 0.7525, 0.8843),
    ]
    right = [
        ("64", 0.7400, 0.8824),
        ("128", 0.7525, 0.8843),
        ("256", 0.7611, 0.8851),
    ]
    for ax, rows, title in zip(axes, [left, right], ["Residual coupling", "Decoder width"]):
        x = np.arange(len(rows))
        w = 0.34
        ax.bar(x - w / 2, [r[1] for r in rows], width=w, color=PALETTE["mjm"], edgecolor="none")
        ax.bar(x + w / 2, [r[2] for r in rows], width=w, color=PALETTE["mjm_light"], edgecolor="none")
        for xx, (_, macro, micro) in zip(x, rows):
            ax.text(xx - w / 2, macro + 0.008, f"{macro:.3f}", ha="center", va="bottom", fontsize=7.2, color=PALETTE["mjm"])
            ax.text(xx + w / 2, micro + 0.008, f"{micro:.3f}", ha="center", va="bottom", fontsize=7.2, color="#7A4B1D")
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_title(title)
        style_axis(ax, ygrid=True)
        ax.set_ylim(0.62, 0.9)
    axes[0].set_ylabel("F1")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["mjm"]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["mjm_light"]),
    ]
    axes[1].legend(handles, ["Macro-F1", "Micro-F1"], frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_ablation")
    plt.close(fig)


def _posthoc_margin_and_confusion(latent, y_true, class_names, top_k=8):
    """Cross-validated linear probe on latent features for visualization only."""
    if len(y_true) > 20000:
        sampler = StratifiedShuffleSplit(n_splits=1, train_size=20000, random_state=42)
        subset_idx, _ = next(sampler.split(latent, y_true))
        latent = latent[subset_idx]
        y_true = y_true[subset_idx]

    selected, selected_names, _ = select_top_classes(y_true, class_names, n=top_k)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    prob = cross_val_predict(clf, latent, y_true, cv=cv, method="predict_proba", n_jobs=-1)
    pred = prob.argmax(axis=1)

    # Normalize selected classes and bucket the rest as Other.
    sel = set(selected)
    y_true_sel = np.array([y if int(y) in sel else -1 for y in y_true])
    y_pred_sel = np.array([y if int(y) in sel else -1 for y in pred])
    lab_order = selected + [-1]
    lab_names = selected_names + ["Other"]
    cm = confusion_matrix(y_true_sel, y_pred_sel, labels=lab_order)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # Confidence margin = p(true) - max_{c != true} p(c)
    true_prob = prob[np.arange(len(y_true)), y_true]
    prob_copy = prob.copy()
    prob_copy[np.arange(len(y_true)), y_true] = -1
    best_other = prob_copy.max(axis=1)
    margin = true_prob - best_other
    correct = pred == y_true
    return cm, lab_names, margin, correct, selected, selected_names


def plot_case_study():
    split = load_split_labels(DATA / "kukanja/kukanja_eae.npz", "Kukanja_EAE")
    y_true = split["levels"]["High"]
    meta = split["meta"]
    inv = {v: k for k, v in meta["label_maps"]["annotation_high"].items()}
    # Use the same completed test split order as the saved features.
    mjm = np.load(EXPERIMENTS / "kukanja/Kukanja_EAE/checkpoints/mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature/test_features.npz", allow_pickle=True)
    uce = np.load(EXPERIMENTS / "kukanja/Kukanja_EAE/checkpoints/uce_full_mouse_s3028/test_features.npz", allow_pickle=True)
    mjm_cm, labels, mjm_margin, mjm_correct, _, _ = _posthoc_margin_and_confusion(
        mjm["latent"], y_true, inv, top_k=8
    )
    uce_cm, _, uce_margin, uce_correct, _, _ = _posthoc_margin_and_confusion(
        uce["cell_emb"], y_true, inv, top_k=8
    )

    fig = plt.figure(figsize=(10.8, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78], width_ratios=[1.0, 1.0, 0.08], hspace=0.34, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, :])

    short_labels = [compact_case_label(x) for x in labels]

    # Confusion matrices
    for ax, cm, title, show_cbar in [(ax0, mjm_cm, "scBrain", False), (ax1, uce_cm, "UCE", True)]:
        sns.heatmap(
            cm,
            ax=ax,
            cmap="mako",
            vmin=0,
            vmax=1,
            square=True,
            cbar=show_cbar,
            cbar_ax=cax if show_cbar else None,
            cbar_kws={"label": "Row-normalized recall"},
            xticklabels=short_labels,
            yticklabels=short_labels,
        )
        ax.set_title(title, pad=8)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if ax is ax0 else "")
    add_panel_label(ax0, "A")
    add_panel_label(ax1, "B")

    # Confidence margins
    margin_df = pd.DataFrame(
        {
            "margin": np.concatenate([mjm_margin, uce_margin]),
            "prediction_quality": np.where(
                np.concatenate([mjm_correct, uce_correct]),
                "correct predictions",
                "incorrect predictions",
            ),
            "model": ["scBrain"] * len(mjm_margin) + ["UCE"] * len(uce_margin),
        }
    )
    sns.boxplot(
        data=margin_df,
        x="model",
        y="margin",
        hue="prediction_quality",
        hue_order=["correct predictions", "incorrect predictions"],
        palette={
            "correct predictions": PALETTE["correct"],
            "incorrect predictions": PALETTE["wrong"],
        },
        ax=ax2,
        width=0.6,
        fliersize=1.5,
    )
    ax2.axhline(0.0, color="#999999", lw=1, ls="--")
    style_axis(ax2, ygrid=True)
    ax2.set_title("Prediction confidence margin", pad=8)
    ax2.set_xlabel("")
    ax2.set_ylabel(r"$p(\mathrm{true}) - p(\mathrm{runner\text{-}up})$")
    ax2.legend(frameon=False, title="", ncol=2, loc="lower left")
    add_panel_label(ax2, "C")
    fig.tight_layout()
    save(fig, "fig_case_study")
    plt.close(fig)


def plot_representation_boundary_case_study():
    category_names = load_seaad_category_names()
    subclass_names = category_names["Subclass"]
    supertype_names = category_names["Supertype"]

    data = np.load(DATA / "global_data.npz", allow_pickle=True)
    batch = data["batch_donor"]
    unique = np.unique(batch)
    rng = np.random.RandomState(3028)
    rng.shuffle(unique)
    test_mask = np.isin(batch, unique[:6])

    x_all = np.log1p(data["X"][test_mask].astype("float32"))
    y_super = data["y_supertype"][test_mask]
    y_sub = data["y_subclass"][test_mask]

    pax6_subclass = subclass_names.index("Pax6")
    family_mask = y_sub == pax6_subclass
    x_family = x_all[family_mask]
    y_family = y_super[family_mask]

    selected_codes = sorted(np.unique(y_family), key=lambda idx: supertype_names[int(idx)])
    class_names = [supertype_names[int(idx)].replace("_", "-") for idx in selected_codes]
    support = [int((y_family == idx).sum()) for idx in selected_codes]

    geneformer = np.load(
        EXPERIMENTS / "Geneformer/SEA_AD_MTP_ST/checkpoints/full_s3028/test_features.npz",
        allow_pickle=True,
    )["cell_emb"][family_mask]

    features = [
        ("Classic linear probe", x_family),
        ("Geneformer", geneformer),
        (
            "scBrain (feature residual, 256)",
            extract_mjm_c3_features(
                x_family,
                "experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec256_z20_s3028_resfeature/best_model.pth",
                "feature",
                256,
            ),
        ),
        (
            "scBrain (independent, 128)",
            extract_mjm_c3_features(
                x_family,
                "experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_resnone/best_model.pth",
                "none",
                128,
            ),
        ),
        (
            "scBrain (logit residual, 128)",
            extract_mjm_c3_features(
                x_family,
                "experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_reslogit/best_model.pth",
                "logit",
                128,
            ),
        ),
        (
            "scBrain (feature residual, 128)",
            extract_mjm_c3_features(
                x_family,
                "experiments/mjm_1/SEA_AD_MTP_ST/checkpoints/e3_d1_enc256_dec128_z20_s3028_resfeature/best_model.pth",
                "feature",
                128,
            ),
        ),
    ]

    class_colors = ["#2563EB", "#F59E0B", "#059669", "#DC2626"]
    fig, axes = plt.subplots(2, 3, figsize=(11.3, 7.2))
    fig.suptitle("SEA-AD representation-space case study on the Pax6 supertype family", y=0.985, fontsize=11.4)
    fig.text(
        0.5,
        0.955,
        ", ".join(f"{name} (n={cnt})" for name, cnt in zip(class_names, support)),
        ha="center",
        va="center",
        fontsize=8.7,
        color="#555555",
    )
    fig.text(0.012, 0.72, "Baselines vs final model", rotation=90, va="center", ha="center", fontsize=9.5)
    fig.text(0.012, 0.29, "Decoder coupling variants", rotation=90, va="center", ha="center", fontsize=9.5)

    for label, ax, (title, feat) in zip(list("ABCDEF"), axes.flat, features):
        coords = project_representation(feat)
        score = linear_probe_macro_f1(feat, y_family)
        draw_boundary_panel(ax, coords, y_family, class_names, class_colors, title, score)
        add_panel_label(ax, label)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6.5,
            label=f"{name} (n={cnt})",
        )
        for name, cnt, color in zip(class_names, support, class_colors)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            markersize=6.7,
            label="2D boundary error",
        )
    )
    fig.legend(
        handles=handles,
        ncol=5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0.03, 0.07, 1.0, 0.93))
    save(fig, "fig_representation_boundary_case_study")
    plt.close(fig)


def plot_appendix_baseline_landscape():
    """Heatmap of fine-level Macro F1 across completed baselines."""
    rows = [
        ("scBrain-res256", "scBrain"),
        ("scBrain-res128", "scBrain"),
        ("scBrain-logit128", "scBrain"),
        ("scBrain-ind128", "scBrain"),
        ("Geneformer-ft", "Foundation"),
        ("Geneformer-fr", "Foundation"),
        ("scGPT-Brain-ft", "Foundation"),
        ("scGPT-Human-ft", "Foundation"),
        ("scGPT-Human-fr", "Foundation"),
        ("UCE-ft", "Foundation"),
        ("UCE-fr", "Foundation"),
        ("LogReg", "Conventional"),
        ("CellTypist", "Conventional"),
        ("XGBoost", "Conventional"),
        ("SVM", "Conventional"),
        ("scSimilarity", "Conventional"),
        ("Stack", "Conventional"),
        ("TACCO", "Conventional"),
        ("Spatial GNN", "Conventional"),
    ]
    values = {
        "SEA-AD": {
            "scBrain-res256": 0.7611,
            "scBrain-res128": 0.7525,
            "scBrain-logit128": 0.6997,
            "scBrain-ind128": 0.6887,
            "Geneformer-ft": 0.7489,
            "Geneformer-fr": 0.5420,
            "scGPT-Brain-ft": 0.6819,
            "scGPT-Human-ft": 0.6701,
            "scGPT-Human-fr": 0.2386,
            "UCE-ft": 0.5118,
            "UCE-fr": 0.3403,
            "LogReg": 0.8080,
            "CellTypist": 0.7324,
            "XGBoost": 0.6274,
            "SVM": 0.5890,
            "scSimilarity": 0.4292,
            "Stack": 0.5805,
            "TACCO": 0.4576,
            "Spatial GNN": 0.3857,
        },
        "Kukanja MS": {
            "scBrain-res256": 0.8496,
            "scBrain-res128": 0.8551,
            "scBrain-logit128": 0.8426,
            "scBrain-ind128": 0.8445,
            "Geneformer-ft": 0.8448,
            "Geneformer-fr": 0.7829,
            "scGPT-Human-ft": 0.8297,
            "scGPT-Human-fr": 0.6022,
            "UCE-ft": 0.7212,
            "UCE-fr": 0.6130,
            "scSimilarity": 0.7301,
            "Stack": 0.8020,
        },
        "Kukanja EAE": {
            "scBrain-res256": 0.6971,
            "scBrain-res128": 0.6866,
            "scBrain-logit128": 0.6940,
            "scBrain-ind128": 0.6887,
            "UCE-ft": 0.5770,
            "UCE-fr": 0.4439,
            "scSimilarity": 0.6275,
            "Stack": 0.0014,
        },
    }
    row_names = [name for name, _ in rows]
    df = pd.DataFrame(index=row_names, columns=["SEA-AD", "Kukanja MS", "Kukanja EAE"], dtype=float)
    for col, col_vals in values.items():
        for row, val in col_vals.items():
            df.loc[row, col] = val

    annot = df.applymap(lambda v: "" if pd.isna(v) else f"{v:.3f}")

    row_families = [family for _, family in rows]
    family_blocks = []
    start = 0
    current = row_families[0]
    for idx, family in enumerate(row_families + ["__end__"]):
        if idx == len(row_families) or family != current:
            family_blocks.append((current, start, idx))
            if idx < len(row_families):
                start = idx
                current = family

    fig = plt.figure(figsize=(8.0, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.18, 1.0], wspace=0.03)
    ax_band = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        df,
        ax=ax,
        cmap="crest",
        vmin=0.0,
        vmax=0.9,
        mask=df.isna(),
        linewidths=0.5,
        linecolor="#ffffff",
        annot=annot.values,
        fmt="",
        cbar_kws={"label": "Fine-level Macro F1", "shrink": 0.88},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0, labelsize=7.8)
    for _, start, end in family_blocks[:-1]:
        ax.hlines(end, *ax.get_xlim(), colors="#444444", linewidth=1.3)

    band_colors = {
        "scBrain": "#F2E1D5",
        "Foundation": "#DCEAF1",
        "Conventional": "#E6E6E6",
    }
    ax_band.set_xlim(0, 1)
    ax_band.set_ylim(len(df), 0)
    for family, start, end in family_blocks:
        ax_band.axhspan(start, end, color=band_colors.get(family, "#EFEFEF"), ec="white")
        ax_band.text(0.5, (start + end) / 2, family, rotation=90, va="center", ha="center", fontsize=8.8)
    ax_band.axis("off")
    fig.tight_layout()
    save(fig, "fig_appendix_baseline_landscape")
    plt.close(fig)


PLOT_BUILDERS = {
    "primary_overview": plot_primary_overview,
    "hierarchy_levels": plot_hierarchy_lines,
    "level_metrics": plot_level_metrics,
    "classwise_distributions": plot_classwise_distributions,
    "roc_curves": plot_roc_curves,
    "batch_integration": plot_batch_integration,
    "ablation": plot_ablation,
    "case_study": plot_case_study,
    "representation_boundary_case_study": plot_representation_boundary_case_study,
    "baseline_landscape": plot_appendix_baseline_landscape,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    targets = argv or list(PLOT_BUILDERS.keys())
    failed = []
    for target in targets:
        if target not in PLOT_BUILDERS:
            valid = ", ".join(sorted(PLOT_BUILDERS))
            raise SystemExit(f"Unknown figure target '{target}'. Valid targets: {valid}")
        try:
            PLOT_BUILDERS[target]()
        except FileNotFoundError as exc:
            print(f"SKIP {target}: missing artifact {exc.filename}")
        except RuntimeError as exc:
            failed.append((target, str(exc)))
    if failed:
        details = "\n".join(f"  - {target}: {message}" for target, message in failed)
        raise SystemExit(f"Figure generation failed:\n{details}")


if __name__ == "__main__":
    main()
