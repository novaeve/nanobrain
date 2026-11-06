#!/usr/bin/env python3
"""Generate reproducibility-oriented appendix tables from local artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "appendix_reproducibility.tex"
DATA = PROJECT / "data"
EXPERIMENTS = PROJECT / "experiments"

sys.path.insert(0, str(PROJECT))
from src.models.scbrain import scBrain, scBrainNLevel
from src.utils.dataloader import build_dataloaders
from src.utils.dataloader_kukanja import build_dataloaders_kukanja


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_float(value: float) -> str:
    return f"{value:.4f}"


def model_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def dataset_rows() -> list[dict[str, str]]:
    required = [
        DATA / "global_data.npz",
        DATA / "kukanja/kukanja_ms.npz",
        DATA / "kukanja/kukanja_eae.npz",
    ]
    if any(not path.exists() for path in required):
        return [
            {
                "dataset": "SEA-AD",
                "cells": "not packaged",
                "genes": "140",
                "split_cells": "requires data/global_data.npz",
                "split_units": "18 / 3 / 6 donors",
                "levels": "Class=3, Subclass=24, Supertype=137",
                "spatial": "Packed data are excluded from the code release; see docs/ASSET_MANIFEST.md.",
            },
            {
                "dataset": "Kukanja MS",
                "cells": "not packaged",
                "genes": "266",
                "split_cells": "requires data/kukanja/kukanja_ms.npz",
                "split_units": "3 / 1 / 2 samples",
                "levels": "Level0=13, Level1.5=14, Level2=25, Level3=27",
                "spatial": "Packed data are excluded from the code release; see docs/ASSET_MANIFEST.md.",
            },
            {
                "dataset": "Kukanja EAE",
                "cells": "not packaged",
                "genes": "239",
                "split_cells": "requires data/kukanja/kukanja_eae.npz",
                "split_units": "37 / 5 / 10 samples",
                "levels": "Broad=8, Medium=16, Medium+DA=23, High=41",
                "spatial": "Packed data are excluded from the code release; see docs/ASSET_MANIFEST.md.",
            },
        ]

    sea = np.load(DATA / "global_data.npz", allow_pickle=True)
    sea_meta = sea["meta"].item()
    sea_train, sea_val, sea_test = build_dataloaders(
        str(DATA / "global_data.npz"),
        batch_size=256,
        seed=3028,
        num_workers=0,
    )

    ms = np.load(DATA / "kukanja/kukanja_ms.npz", allow_pickle=True)
    _, _, _, ms_info = build_dataloaders_kukanja(
        str(DATA / "kukanja/kukanja_ms.npz"),
        batch_size=512,
        seed=3028,
        num_workers=0,
    )
    ms_meta = ms["meta"].item()

    eae = np.load(DATA / "kukanja/kukanja_eae.npz", allow_pickle=True)
    _, _, _, eae_info = build_dataloaders_kukanja(
        str(DATA / "kukanja/kukanja_eae.npz"),
        batch_size=512,
        seed=3028,
        num_workers=0,
    )
    eae_meta = eae["meta"].item()

    return [
        {
            "dataset": "SEA-AD",
            "cells": fmt_int(int(sea["X"].shape[0])),
            "genes": str(int(sea["X"].shape[1])),
            "split_cells": (
                f"{fmt_int(len(sea_train.dataset))} / "
                f"{fmt_int(len(sea_val.dataset))} / "
                f"{fmt_int(len(sea_test.dataset))}"
            ),
            "split_units": "18 / 3 / 6 donors",
            "levels": (
                f"Class={sea_meta['num_class']}, "
                f"Subclass={sea_meta['num_subclass']}, "
                f"Supertype={sea_meta['num_supertype']}"
            ),
            "spatial": "Headline tables use gene-only input. When enabled, x-y is z-scored per donor and scaled by 2.",
        },
        {
            "dataset": "Kukanja MS",
            "cells": fmt_int(int(ms["X"].shape[0])),
            "genes": str(int(ms["X"].shape[1])),
            "split_cells": (
                f"{fmt_int(ms_info['n_train'])} / "
                f"{fmt_int(ms_info['n_val'])} / "
                f"{fmt_int(ms_info['n_test'])}"
            ),
            "split_units": "3 / 1 / 2 samples",
            "levels": (
                f"Level0={ms_meta['num_level0']}, "
                f"Level1.5={ms_meta['num_level1_5']}, "
                f"Level2={ms_meta['num_level2']}, "
                f"Level3={ms_meta['num_level3']}"
            ),
            "spatial": "Uses the packed spatial_norm field supplied with the dataset.",
        },
        {
            "dataset": "Kukanja EAE",
            "cells": fmt_int(int(eae["X"].shape[0])),
            "genes": str(int(eae["X"].shape[1])),
            "split_cells": (
                f"{fmt_int(eae_info['n_train'])} / "
                f"{fmt_int(eae_info['n_val'])} / "
                f"{fmt_int(eae_info['n_test'])}"
            ),
            "split_units": "37 / 5 / 10 samples",
            "levels": (
                f"Broad={eae_meta['num_annotation_broad']}, "
                f"Medium={eae_meta['num_annotation_medium']}, "
                f"Medium+DA={eae_meta['num_annotation_medium_da']}, "
                f"High={eae_meta['num_annotation_high']}"
            ),
            "spatial": "Uses the packed spatial_norm field supplied with the dataset.",
        },
    ]


def dataset_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Exact dataset, split, and hierarchy details used by the reported benchmarks. Split-cell counts are shown as train / validation / test.}",
        r"\label{tab:appendix_dataset_details}",
        r"\centering",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{lllcclp{5.2cm}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Cells} & \textbf{Genes} & \textbf{Split Cells} & \textbf{Split Units} & \textbf{Levels} & \textbf{Spatial Handling} \\",
        r"\midrule",
    ]
    stripe = "0.949,0.949,0.949"
    for idx, row in enumerate(rows):
        bg = stripe if idx % 2 == 1 else None
        def shade(text: str) -> str:
            if bg is None:
                return text
            return rf"\cellcolor[rgb]{{{bg}}} {text}"
        lines.append(
            " & ".join(
                [
                    shade(tex_escape(row["dataset"])),
                    shade(row["cells"]),
                    shade(row["genes"]),
                    shade(tex_escape(row["split_cells"])),
                    shade(tex_escape(row["split_units"])),
                    shade(tex_escape(row["levels"])),
                    shade(tex_escape(row["spatial"])),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def config_rows() -> list[dict[str, str]]:
    sea_model = scBrain(
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
    )
    ms_model = scBrainNLevel(
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
    )
    eae_model = scBrainNLevel(
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
    )
    return [
        {
            "dataset": "SEA-AD",
            "run_id": "e3_d1_enc256_dec256_z20_s3028_resfeature",
            "model": "scBrain",
            "input_dim": "140",
            "levels": "3 / 24 / 137",
            "enc_dec": "256 / 256",
            "latent": "20",
            "batch": "512",
            "params": fmt_int(model_params(sea_model)),
        },
        {
            "dataset": "Kukanja MS",
            "run_id": "scbrain_e3_d1_enc256_dec128_z20_s3028_resfeature",
            "model": "scBrainNLevel",
            "input_dim": "266",
            "levels": "13 / 14 / 25 / 27",
            "enc_dec": "256 / 128",
            "latent": "20",
            "batch": "512",
            "params": fmt_int(model_params(ms_model)),
        },
        {
            "dataset": "Kukanja EAE",
            "run_id": "scbrain_e3_d1_enc256_dec256_z20_s3028_resfeature",
            "model": "scBrainNLevel",
            "input_dim": "239",
            "levels": "8 / 16 / 23 / 41",
            "enc_dec": "256 / 256",
            "latent": "20",
            "batch": "512",
            "params": fmt_int(model_params(eae_model)),
        },
    ]


def config_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Best residual scBrain checkpoints used throughout the paper. `Levels' lists the class counts from coarse to fine. `Enc / Dec' gives encoder and decoder hidden widths.}",
        r"\label{tab:appendix_scbrain_configs}",
        r"\centering",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{lllcccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Run ID} & \textbf{Model} & \textbf{Input} & \textbf{Levels} & \textbf{Enc / Dec} & \textbf{Latent} & \textbf{Batch} & \textbf{Params} \\",
        r"\midrule",
    ]
    stripe = "0.949,0.949,0.949"
    for idx, row in enumerate(rows):
        bg = stripe if idx % 2 == 1 else None
        def shade(text: str) -> str:
            if bg is None:
                return text
            return rf"\cellcolor[rgb]{{{bg}}} {text}"
        lines.append(
            " & ".join(
                [
                    shade(tex_escape(row["dataset"])),
                    shade(r"\texttt{" + tex_escape(row["run_id"]) + "}"),
                    shade(r"\texttt{" + tex_escape(row["model"]) + "}"),
                    shade(row["input_dim"]),
                    shade(tex_escape(row["levels"])),
                    shade(tex_escape(row["enc_dec"])),
                    shade(row["latent"]),
                    shade(row["batch"]),
                    shade(row["params"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def pretty_batch_label(name: str) -> str:
    mapping = {
        "scGPT_human (full)": "scGPT (Human)",
        "scGPT_brain (full)": "scGPT (Brain)",
        "scGPT (scratch)": "scGPT (scratch)",
        "mjm_2c_lr (FiLM)": "scBrain linear probe (FiLM)",
        "mjm_2a_lr (depth)": "scBrain linear probe (depth)",
        "mjm_1_lr": "scBrain linear probe (feature residual)",
        "mjm_lr": "scBrain linear probe",
        "mjm_1 (none,dec128)": "scBrain (independent decoders)",
        "mjm (dec256)": "scBrain",
        "Geneformer (full)": "Geneformer",
    }
    return mapping.get(name, name)


def batch_rows() -> list[dict[str, str]]:
    csv_path = EXPERIMENTS / "batch_integration/SEA_AD_MTP_ST/scib_fast_results.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, index_col=0)
    top = df.sort_values("overall", ascending=False).head(10)
    rows = []
    for name, row in top.iterrows():
        rows.append(
            {
                "method": pretty_batch_label(name),
                "batch": fmt_float(float(row["batch_score"])),
                "bio": fmt_float(float(row["bio_score"])),
                "overall": fmt_float(float(row["overall"])),
                "asw_b": fmt_float(float(row["ASW_batch"])),
                "graph": fmt_float(float(row["graph_conn"])),
                "pcr_b": fmt_float(float(row["PCR_batch"])),
                "asw_l": fmt_float(float(row["ASW_label"])),
                "nmi": fmt_float(float(row["NMI"])),
                "ari": fmt_float(float(row["ARI"])),
            }
        )
    return rows


def batch_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Exact SEA-AD batch-integration scores for the ten strongest reportable embedding models ranked by overall scIB score. This table complements Figure~\ref{fig:batch_integration}.}",
        r"\label{tab:appendix_batch_integration}",
        r"\centering",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Batch} & \textbf{Bio} & \textbf{Overall} & \textbf{ASW$_b$} & \textbf{Graph} & \textbf{PCR$_b$} & \textbf{ASW$_\ell$} & \textbf{NMI} & \textbf{ARI} \\",
        r"\midrule",
    ]
    stripe = "0.949,0.949,0.949"
    for idx, row in enumerate(rows):
        bg = stripe if idx % 2 == 1 else None
        def shade(text: str) -> str:
            if bg is None:
                return text
            return rf"\cellcolor[rgb]{{{bg}}} {text}"
        lines.append(
            " & ".join(
                [
                    shade(tex_escape(row["method"])),
                    shade(row["batch"]),
                    shade(row["bio"]),
                    shade(row["overall"]),
                    shade(row["asw_b"]),
                    shade(row["graph"]),
                    shade(row["pcr_b"]),
                    shade(row["asw_l"]),
                    shade(row["nmi"]),
                    shade(row["ari"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\vspace{0.3em}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize\textbf{Notes.} `Batch' is the mean of ASW$_b$, graph connectivity, and PCR$_b$. `Bio' is the mean of ASW$_\ell$, NMI, and ARI. `Overall' uses the scIB-style weighting $0.4 \times \mathrm{Batch} + 0.6 \times \mathrm{Bio}$.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> str:
    parts = [
        r"\section{Reproducibility Details}",
        r"\label{app:reproducibility}",
        r"All numbers in this section are generated from the packed NPZ files in \texttt{data/} and the completed logs or CSV summaries in \texttt{experiments/}. The goal is to make the reported hierarchy-level results and split protocol easy to reconstruct without rereading the training code.",
        r"Specifically, the packed data artifacts are \texttt{global\_data.npz}, \texttt{kukanja\_ms.npz}, and \texttt{kukanja\_eae.npz} under \texttt{data/}; the batch-integration summary is \texttt{scib\_fast\_results.csv} under \texttt{experiments/batch\_integration/SEA\_AD\_MTP\_ST/}; and the manuscript-side summary tables in this directory are regenerated by \texttt{build\_main\_results\_table.py}, \texttt{build\_levelwise\_appendix.py}, and \texttt{build\_reproducibility\_appendix.py}.",
        "",
        r"\subsection{Dataset and Split Details}",
        r"The primary SEA-AD benchmark uses deterministic donor splits with seed 3028, while both Kukanja datasets use deterministic sample-based splits with the same seed. The exact counts and hierarchy cardinalities are given in Table~\ref{tab:appendix_dataset_details}.",
        "",
        dataset_table(dataset_rows()),
        r"\subsection{Optimization and Architecture Details}",
        r"The reported scBrain checkpoints share the same optimization recipe: \texttt{log1p} input normalization, Adam optimizer with learning rate $10^{-4}$ and weight decay $10^{-4}$, cosine annealing to $10^{-6}$ over at most 500 epochs, early stopping with patience 30 on the validation total loss, and gradient clipping at norm 5.0. SEA-AD minimizes reconstruction loss plus three classification losses, with confidence weighting on the supertype term; the Kukanja runs use reconstruction loss plus uniform cross-entropy at each hierarchy level.",
        "",
        config_table(config_rows()),
        r"\subsection{Metric Definitions}",
        r"Macro-F1 is the unweighted mean of per-class F1 scores, so it is sensitive to rare classes. Micro-F1 aggregates all predictions before computing precision and recall; in this single-label setting it is equivalent to overall accuracy. Appendix AUC values are one-vs-rest ROC areas computed from the same held-out predictions as the F1 summaries. For classes absent from a given test split, the macro AUC only averages over valid positive-negative one-vs-rest problems, matching the training logs.",
        "",
        r"For SEA-AD batch integration, ASW$_b$, graph connectivity, and PCR$_b$ measure batch removal, while ASW$_\ell$, NMI, and ARI measure biological conservation. The manuscript reports the same aggregate used in the evaluation script: $\mathrm{Batch}=\frac{1}{3}(\mathrm{ASW}_b+\mathrm{Graph}+\mathrm{PCR}_b)$, $\mathrm{Bio}=\frac{1}{3}(\mathrm{ASW}_\ell+\mathrm{NMI}+\mathrm{ARI})$, and $\mathrm{Overall}=0.4 \times \mathrm{Batch}+0.6 \times \mathrm{Bio}$.",
        "",
        r"\subsection{Exact SEA-AD Batch-Integration Scores}",
        r"Figure~\ref{fig:batch_integration} shows only the top-level bar summary. Table~\ref{tab:appendix_batch_integration} adds the exact component scores for the ten strongest reportable embedding models.",
        "",
        batch_table(batch_rows()),
    ]
    return "\n".join(parts)


def main() -> None:
    OUT.write_text(build())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
