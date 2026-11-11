#!/usr/bin/env python3
"""Generate level-wise appendix tables from completed benchmark logs."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
EXPERIMENTS = PROJECT / "experiments"
DATA = PROJECT / "data"
OUT = ROOT / "appendix_level_metrics.tex"


DATASETS = {
    "SEA-AD": {
        "log": EXPERIMENTS / "mjm_1/SEA_AD_MTP_ST/record_e3_d1_enc256_dec256_z20_s3028_resfeature.log",
        "summary_levels": [
            ("CLASS", "Class"),
            ("SUBCLASS", "Subclass"),
            ("SUPERTYPE", "Supertype"),
        ],
        "class_levels": [
            ("CLASS", "Class"),
            ("SUBCLASS", "Subclass"),
        ],
        "label_map": None,
    },
    "Kukanja MS": {
        "log": EXPERIMENTS / "kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log",
        "summary_levels": [
            ("LEVEL0", "Level0"),
            ("LEVEL1_5", "Level1.5"),
            ("LEVEL2", "Level2"),
            ("LEVEL3", "Level3"),
        ],
        "class_levels": [
            ("LEVEL0", "Level0"),
            ("LEVEL1_5", "Level1.5"),
            ("LEVEL2", "Level2"),
        ],
        "label_map": DATA / "kukanja/kukanja_ms.npz",
    },
    "Kukanja EAE": {
        "log": EXPERIMENTS / "kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log",
        "summary_levels": [
            ("ANNOTATION_BROAD", "Broad"),
            ("ANNOTATION_MEDIUM", "Medium"),
            ("ANNOTATION_MEDIUM_DA", "Medium_DA"),
            ("ANNOTATION_HIGH", "High"),
        ],
        "class_levels": [
            ("ANNOTATION_BROAD", "Broad"),
            ("ANNOTATION_MEDIUM", "Medium"),
            ("ANNOTATION_MEDIUM_DA", "Medium_DA"),
        ],
        "label_map": DATA / "kukanja/kukanja_eae.npz",
    },
}


MACRO_RE = re.compile(
    r"\[(?P<lvl>[A-Z0-9_]+)\]\s+Macro\s+-\s+(?:Precision|P):\s*(?P<p>[0-9.]+)\s*(?:\|\s*|\s+)"
    r"(?:Recall|R):\s*(?P<r>[0-9.]+)\s*(?:\|\s*|\s+)"
    r"F1:?\s*(?P<f1>[0-9.]+)"
)
MICRO_RE = re.compile(
    r"\[(?P<lvl>[A-Z0-9_]+)\]\s+Micro\s+-\s+(?:Precision|P):\s*(?P<p>[0-9.]+)\s*(?:\|\s*|\s+)"
    r"(?:Recall|R):\s*(?P<r>[0-9.]+)\s*(?:\|\s*|\s+)"
    r"F1:?\s*(?P<f1>[0-9.]+)"
)
AUC_RE = re.compile(
    r"\[(?P<lvl>[A-Z0-9_]+)\]\s+AUC-ROC\s+-\s+Macro:?\s*(?P<macro>[0-9.]+)\s*(?:\|\s*|\s+)"
    r"Micro:?\s*(?P<micro>[0-9.]+)"
)
PER_CLASS_RE = re.compile(
    r"C(?P<idx>\d+)\(S=(?P<sup>\d+)\):\s*P=(?P<p>[0-9.]+)/R=(?P<r>[0-9.]+)"
)


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def read_label_names(npz_path: Path) -> dict[str, list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    meta = data["meta"].item()
    out: dict[str, list[str]] = {}
    for key, mapping in meta["label_maps"].items():
        inv = {v: k for k, v in mapping.items()}
        out[key.upper()] = [inv[i] for i in range(len(inv))]
    return out


def parse_log(path: Path):
    text = path.read_text(errors="replace").splitlines()
    metrics: dict[str, dict] = {}
    for line in text:
        m = MACRO_RE.search(line)
        if m:
            lvl = m.group("lvl")
            metrics.setdefault(lvl, {})
            metrics[lvl]["macro_p"] = float(m.group("p"))
            metrics[lvl]["macro_r"] = float(m.group("r"))
            metrics[lvl]["macro_f1"] = float(m.group("f1"))
            continue
        m = MICRO_RE.search(line)
        if m:
            lvl = m.group("lvl")
            metrics.setdefault(lvl, {})
            metrics[lvl]["micro_p"] = float(m.group("p"))
            metrics[lvl]["micro_r"] = float(m.group("r"))
            metrics[lvl]["micro_f1"] = float(m.group("f1"))
            continue
        m = AUC_RE.search(line)
        if m:
            lvl = m.group("lvl")
            metrics.setdefault(lvl, {})
            metrics[lvl]["auc_macro"] = float(m.group("macro"))
            metrics[lvl]["auc_micro"] = float(m.group("micro"))
            continue
        m = re.search(r"\[(?P<lvl>[A-Z0-9_]+)\].*Per-class(?: P/R)?:\s*(?P<body>.*)$", line)
        if m:
            lvl = m.group("lvl")
            metrics.setdefault(lvl, {})
            cls = []
            for cm in PER_CLASS_RE.finditer(m.group("body")):
                idx = int(cm.group("idx"))
                sup = int(cm.group("sup"))
                p = float(cm.group("p"))
                r = float(cm.group("r"))
                cls.append({
                    "idx": idx,
                    "support": sup,
                    "precision": p,
                    "recall": r,
                    "f1": f1(p, r),
                })
            metrics[lvl]["classes"] = cls
    return metrics


def tex_escape(s: str) -> str:
    return (
        s.replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def fmt(v: float) -> str:
    return f"{v:.4f}"


def shaded(text: str, bg_color: str | None) -> str:
    if bg_color is None:
        return text
    return rf"\cellcolor[rgb]{{{bg_color}}} {text}"


def combined_summary_table(dataset_rows: list[tuple[str, list[tuple[str, str, dict]]]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Unified level-wise precision, recall, F1, and AUC summary for the best residual scBrain checkpoint on each dataset.}",
        r"\label{tab:appendix_level_summary}",
        r"\centering",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Level}} & \multicolumn{3}{c}{\textbf{Macro}} & \multicolumn{3}{c}{\textbf{Micro}} & \multicolumn{2}{c}{\textbf{AUC-ROC}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
        r"& & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Macro} & \textbf{Micro} \\",
        r"\midrule",
    ]
    stripe_bg = "0.949,0.949,0.949"
    body_row_idx = 0
    for dataset_idx, (dataset, rows) in enumerate(dataset_rows):
        for level_idx, (_, lvl_name, m) in enumerate(rows):
            bg = stripe_bg if body_row_idx % 2 == 1 else None
            dataset_cell = ""
            if level_idx == 0:
                dataset_cell = rf"\multirow{{{len(rows)}}}{{*}}{{\textbf{{{tex_escape(dataset)}}}}}"
            lines.append(
                " & ".join(
                    [
                        dataset_cell,
                        shaded(tex_escape(lvl_name), bg),
                        shaded(fmt(m["macro_p"]), bg),
                        shaded(fmt(m["macro_r"]), bg),
                        shaded(fmt(m["macro_f1"]), bg),
                        shaded(fmt(m["micro_p"]), bg),
                        shaded(fmt(m["micro_r"]), bg),
                        shaded(fmt(m["micro_f1"]), bg),
                        shaded(fmt(m["auc_macro"]), bg),
                        shaded(fmt(m["auc_micro"]), bg),
                    ]
                )
                + r" \\"
            )
            body_row_idx += 1
        if dataset_idx != len(dataset_rows) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\vspace{0.3em}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize\textbf{Notes.} AUC values are one-vs-rest ROC areas computed from the same held-out test predictions as the precision, recall, and F1 entries.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def class_table(dataset: str, level_name: str, class_names: list[str], cls_rows: list[dict]) -> str:
    title = f"{dataset} {tex_escape(level_name)} per-class precision, recall, and F1 for scBrain."
    label = f"tab:{dataset.lower().replace(' ', '_')}_{level_name.lower().replace('.', '').replace(' ', '_')}_perclass"
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{longtable}{@{}lrrrr@{}}",
        rf"\caption{{{title}}}\label{{{label}}}\\",
        r"\toprule",
        r"Class & Support & Precision & Recall & F1 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Class & Support & Precision & Recall & F1 \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    stripe_bg = "0.949,0.949,0.949"
    for idx, row in enumerate(cls_rows):
        bg = stripe_bg if idx % 2 == 1 else None
        idx = row["idx"]
        cname = class_names[idx] if idx < len(class_names) else f"C{idx}"
        lines.append(
            " & ".join(
                [
                    shaded(tex_escape(cname), bg),
                    shaded(str(row["support"]), bg),
                    shaded(fmt(row["precision"]), bg),
                    shaded(fmt(row["recall"]), bg),
                    shaded(fmt(row["f1"]), bg),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\normalsize", ""])
    return "\n".join(lines)


def build() -> str:
    parts = [
        r"\section{Level-Wise Detailed Results}",
        r"\label{app:levelwise_results}",
        r"All values below come from the same best residual scBrain checkpoint family used in the main text and appendix figures. The exact level-wise aggregates are retained here as one compact reference table, while Figure~\ref{fig:classwise_distributions} summarizes the intermediate classwise F1 distributions visually.",
        "",
    ]
    all_summary_rows = []
    for dataset, cfg in DATASETS.items():
        parsed = parse_log(cfg["log"])
        summary_rows = []
        for lvl_key, lvl_name in cfg["summary_levels"]:
            m = parsed[lvl_key]
            summary_rows.append((lvl_key, lvl_name, m))
        all_summary_rows.append((dataset, summary_rows))
    parts.append(combined_summary_table(all_summary_rows))
    parts.extend(
        [
            r"The finest-level exact per-class precision/recall/F1 tables remain in Appendix~\ref{app:per_class}. To keep the appendix readable, the lower and intermediate hierarchy levels are summarized visually rather than repeated as long numeric tables.",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    OUT.write_text(build())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
