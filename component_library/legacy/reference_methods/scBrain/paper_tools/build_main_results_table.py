#!/usr/bin/env python3
"""Generate SEA-AD main table, appendix transfer tables, and audit files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

MAIN_TEX = ROOT / "main_results_table.tex"
APPENDIX_TEX = ROOT / "appendix_transfer_tables.tex"
AUDIT_MD = ROOT / "main_results_table_audit.md"


LEVELS = [
    ("CLASS", "Class"),
    ("SUBCLASS", "Subclass"),
    ("SUPERTYPE", "Supertype"),
]

KUKANJA_DATASETS = {
    "Kukanja MS": [
        ("LEVEL0", "Level0"),
        ("LEVEL1_5", "Level1.5"),
        ("LEVEL2", "Level2"),
        ("LEVEL3", "Level3"),
    ],
    "Kukanja EAE": [
        ("ANNOTATION_BROAD", "Broad"),
        ("ANNOTATION_MEDIUM", "Medium"),
        ("ANNOTATION_MEDIUM_DA", "Medium+DA"),
        ("ANNOTATION_HIGH", "High"),
    ],
}

HIGHLIGHT_LEVELS = {"CLASS", "SUBCLASS", "SUPERTYPE"}
DATASET_REASON_FIELD = {
    "SEA-AD": "seaad_missing_reason",
    "Kukanja MS": "ms_missing_reason",
    "Kukanja EAE": "eae_missing_reason",
}

FAMILY_DISPLAY = {
    "Conventional": "Classic",
    "Spatial / specialized": "Spatial",
    "Foundation (freeze)": "Freeze",
    "Foundation (fine-tuned)": "FT",
    "Semi-supervised": "Semi",
    "Ours": "Ours",
}

SEAAD_MAIN_LAYOUT = {
    "XGBoost": {"group": "Classic", "mode": "", "order": 1},
    "SVM": {"group": "Classic", "mode": "", "order": 2},
    "CellTypist": {"group": "Classic", "mode": "", "order": 3},
    "TACCO (OT)": {"group": "Classic", "mode": "", "order": 4},
    "Geneformer (freeze)": {"group": "Foundation Models", "mode": "Freeze", "order": 1},
    "scGPT-Brain (freeze)": {"group": "Foundation Models", "mode": "Freeze", "order": 2},
    "scGPT-Human (freeze)": {"group": "Foundation Models", "mode": "Freeze", "order": 3},
    "UCE (freeze)": {"group": "Foundation Models", "mode": "Freeze", "order": 4},
    "scFoundation (freeze)": {"group": "Foundation Models", "mode": "Freeze", "order": 5},
    "HEIST (freeze)": {"group": "Foundation Models", "mode": "Freeze", "order": 6},
    "Geneformer (fine-tuned)": {"group": "Foundation Models", "mode": "FT", "order": 7},
    "scGPT-Brain (fine-tuned)": {"group": "Foundation Models", "mode": "FT", "order": 8},
    "scGPT-Human (fine-tuned)": {"group": "Foundation Models", "mode": "FT", "order": 9},
    "UCE (fine-tuned)": {"group": "Foundation Models", "mode": "FT", "order": 10},
    "HEIST (full)": {"group": "Foundation Models", "mode": "FT", "order": 11},
    "scSimilarity": {"group": "Non-Foundation", "mode": "", "order": 1},
    "Stack (freeze)": {"group": "Non-Foundation", "mode": "", "order": 2},
    "Stack (full)": {"group": "Non-Foundation", "mode": "", "order": 3},
    "Spatial GNN": {"group": "Non-Foundation", "mode": "", "order": 4},
    "Novae (freeze)": {"group": "Non-Foundation", "mode": "", "order": 5},
    "GraphST": {"group": "Non-Foundation", "mode": "", "order": 6},
    "scBrain (feature residual)": {"group": "Non-Foundation", "mode": "", "order": 99},
}

SEAAD_MAIN_GROUP_ORDER = [
    "Classic",
    "Foundation Models",
    "Non-Foundation",
]

SEAAD_MAIN_MODE_ORDER = {
    "Classic": [""],
    "Foundation Models": ["Freeze", "FT"],
    "Non-Foundation": [""],
}

SEAAD_MAIN_EXCLUDED_LABELS = {
    "Logistic regression",
}


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    level: str
    start_marker: Optional[str] = None
    end_marker: Optional[str] = None
    display_level: Optional[str] = None


@dataclass(frozen=True)
class MethodSpec:
    family: str
    label: str
    seaad: dict[str, SourceSpec]
    note: str = ""
    ours: bool = False
    appendix_family: Optional[str] = None
    ms: Optional[dict[str, SourceSpec]] = None
    eae: Optional[dict[str, SourceSpec]] = None
    missing_reason: str = ""
    seaad_missing_reason: str = ""
    ms_missing_reason: str = ""
    eae_missing_reason: str = ""


MACRO_RE = re.compile(
    r"\[(?P<level>[A-Z0-9_]+)\]\s+Macro\s+-\s+(?:Precision|P):\s*(?P<p>[0-9.]+).*?F1:?\s*(?P<f1>[0-9.]+)"
)
MICRO_RE = re.compile(
    r"\[(?P<level>[A-Z0-9_]+)\]\s+Micro\s+-\s+(?:Precision|P):\s*(?P<p>[0-9.]+).*?F1:?\s*(?P<f1>[0-9.]+)"
)
MICRO_F1_ONLY_RE = re.compile(r"\[(?P<level>[A-Z0-9_]+)\]\s+Micro\s+-\s+F1:?\s*(?P<f1>[0-9.]+)")
MACRO_AUC_RE = re.compile(r"\[(?P<level>[A-Z0-9_]+)\]\s+AUC-ROC\s+-\s+Macro:?\s*(?P<macro>[0-9.]+).*?Micro:?\s*(?P<micro>[0-9.]+)")


def seaad_spec(path: str, level: str, *, start_marker: Optional[str] = None, end_marker: Optional[str] = None,
               display_level: Optional[str] = None) -> SourceSpec:
    return SourceSpec(PROJECT / path, level, start_marker, end_marker, display_level)


MAIN_METHODS = [
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="Logistic regression",
        seaad={lvl: seaad_spec("experiments/sklearn_baselines/lr/SEA_AD_MTP_ST/record_s3028.log", lvl) for lvl, _ in LEVELS},
        note="Best conventional SEA-AD baseline.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="XGBoost",
        seaad={lvl: seaad_spec("experiments/sklearn_baselines/xgboost/SEA_AD_MTP_ST/record_s3028.log", lvl) for lvl, _ in LEVELS},
        note="Tree baseline on raw Cell×Gene features.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="SVM",
        seaad={lvl: seaad_spec("experiments/sklearn_baselines/svm/SEA_AD_MTP_ST/record_s3028.log", lvl) for lvl, _ in LEVELS},
        note="RBF-kernel baseline on raw Cell×Gene features.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="CellTypist",
        seaad={
            "CLASS": seaad_spec("experiments/CellTypist/logs/record_s3028.log", "RETRAINED_CLASS", display_level="RETRAINED_CLASS"),
            "SUBCLASS": seaad_spec("experiments/CellTypist/logs/record_s3028.log", "RETRAINED_SUBCLASS", display_level="RETRAINED_SUBCLASS"),
            "SUPERTYPE": seaad_spec("experiments/CellTypist/logs/record_s3028.log", "RETRAINED_SUPERTYPE", display_level="RETRAINED_SUPERTYPE"),
        },
        note="Retrained in-domain CellTypist.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="TACCO (OT)",
        seaad={
            "CLASS": seaad_spec("experiments/TACCO/logs/record_s3028.log", "OT_CLASS", display_level="OT_CLASS"),
            "SUBCLASS": seaad_spec("experiments/TACCO/logs/record_s3028.log", "OT_SUBCLASS", display_level="OT_SUBCLASS"),
            "SUPERTYPE": seaad_spec("experiments/TACCO/logs/record_s3028.log", "OT_SUPERTYPE", display_level="OT_SUPERTYPE"),
        },
        note="Optimal-transport transfer baseline.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="scSimilarity",
        seaad={lvl: seaad_spec("experiments/scSimilarity/SEA_AD_MTP_ST/record_full_s3028.log", lvl) for lvl, _ in LEVELS},
        note="Full fine-tuning reported by benchmark.",
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/ms_scsimilarity_g7.launch.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/ms_scsimilarity_g7.launch.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/ms_scsimilarity_g7.launch.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/ms_scsimilarity_g7.launch.log", "LEVEL3"),
        },
        eae={
            "ANNOTATION_BROAD": seaad_spec(
                "experiments/kukanja/eae_new_run.log",
                "ANNOTATION_BROAD",
                start_marker="[8] scSimilarity full",
                end_marker="model_name='UCE'",
            ),
            "ANNOTATION_MEDIUM": seaad_spec(
                "experiments/kukanja/eae_new_run.log",
                "ANNOTATION_MEDIUM",
                start_marker="[8] scSimilarity full",
                end_marker="model_name='UCE'",
            ),
            "ANNOTATION_MEDIUM_DA": seaad_spec(
                "experiments/kukanja/eae_new_run.log",
                "ANNOTATION_MEDIUM_DA",
                start_marker="[8] scSimilarity full",
                end_marker="model_name='UCE'",
            ),
            "ANNOTATION_HIGH": seaad_spec(
                "experiments/kukanja/eae_new_run.log",
                "ANNOTATION_HIGH",
                start_marker="[8] scSimilarity full",
                end_marker="model_name='UCE'",
            ),
        },
    ),
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="Stack (full)",
        seaad={lvl: seaad_spec("experiments/Stack/SEA_AD_MTP_ST/record_full_s3028.log", lvl) for lvl, _ in LEVELS},
        note="Full Stack adaptation.",
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/ms_stack_full_g5.launch.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/ms_stack_full_g5.launch.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/ms_stack_full_g5.launch.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/ms_stack_full_g5.launch.log", "LEVEL3"),
        },
        eae={
            "ANNOTATION_BROAD": seaad_spec("experiments/kukanja/stack_full_eae_gpu1.launch.log", "ANNOTATION_BROAD"),
            "ANNOTATION_MEDIUM": seaad_spec("experiments/kukanja/stack_full_eae_gpu1.launch.log", "ANNOTATION_MEDIUM"),
            "ANNOTATION_MEDIUM_DA": seaad_spec("experiments/kukanja/stack_full_eae_gpu1.launch.log", "ANNOTATION_MEDIUM_DA"),
            "ANNOTATION_HIGH": seaad_spec("experiments/kukanja/stack_full_eae_gpu1.launch.log", "ANNOTATION_HIGH"),
        },
    ),
    MethodSpec(
        family="Conventional",
        appendix_family="Conventional",
        label="Stack (freeze)",
        seaad={},
        note="Frozen Stack transfer on external-validation datasets only.",
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/ms_stack_freeze_g5.launch.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/ms_stack_freeze_g5.launch.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/ms_stack_freeze_g5.launch.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/ms_stack_freeze_g5.launch.log", "LEVEL3"),
        },
        eae={
            "ANNOTATION_BROAD": seaad_spec("experiments/kukanja/eae_stack_freeze_g6.launch.log", "ANNOTATION_BROAD"),
            "ANNOTATION_MEDIUM": seaad_spec("experiments/kukanja/eae_stack_freeze_g6.launch.log", "ANNOTATION_MEDIUM"),
            "ANNOTATION_MEDIUM_DA": seaad_spec("experiments/kukanja/eae_stack_freeze_g6.launch.log", "ANNOTATION_MEDIUM_DA"),
            "ANNOTATION_HIGH": seaad_spec("experiments/kukanja/eae_stack_freeze_g6.launch.log", "ANNOTATION_HIGH"),
        },
        missing_reason="not evaluated on SEA-AD",
        seaad_missing_reason="not evaluated on SEA-AD",
    ),
    MethodSpec(
        family="Spatial / specialized",
        appendix_family="Spatial / specialized",
        label="Spatial GNN",
        seaad={lvl: seaad_spec("experiments/spatial_gnn/SEA_AD_MTP_ST/record_k15_gcn2_h256_z128_s3028.log", lvl) for lvl, _ in LEVELS},
        note="Graph baseline using spatial neighborhoods.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Spatial / specialized",
        appendix_family="Spatial / specialized",
        label="HEIST (freeze)",
        seaad={lvl: seaad_spec("experiments/HEIST/SEA_AD_MTP_ST/record_freeze_s3028.log", lvl) for lvl, _ in LEVELS},
        note="HEIST encoder frozen; included for completeness.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Spatial / specialized",
        appendix_family="Spatial / specialized",
        label="HEIST (full)",
        seaad={lvl: seaad_spec("experiments/HEIST/SEA_AD_MTP_ST/record_full_s3028.log", lvl) for lvl, _ in LEVELS},
        note="HEIST end-to-end fine-tuning.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Spatial / specialized",
        appendix_family="Spatial / specialized",
        label="Novae (freeze)",
        seaad={lvl: seaad_spec("experiments/Novae/SEA_AD_MTP_ST/record_freeze_s3028.log", lvl) for lvl, _ in LEVELS},
        note="Frozen Novae encoder baseline.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Spatial / specialized",
        appendix_family="Spatial / specialized",
        label="GraphST",
        seaad={lvl: seaad_spec("experiments/GraphST/SEA_AD_MTP_ST/record_d64_e600_s3028.log", lvl) for lvl, _ in LEVELS},
        note="GraphST spot-aware encoder baseline.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Foundation (freeze)",
        appendix_family="Foundation (freeze)",
        label="Geneformer (freeze)",
        seaad={lvl: seaad_spec("experiments/Geneformer/SEA_AD_MTP_ST/record_freeze_s3028.log", lvl) for lvl, _ in LEVELS},
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_freeze_s3028.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_freeze_s3028.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_freeze_s3028.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_freeze_s3028.log", "LEVEL3"),
        },
        eae=None,
        missing_reason="not evaluated on Kukanja EAE",
    ),
    MethodSpec(
        family="Foundation (freeze)",
        appendix_family="Foundation (freeze)",
        label="scGPT-Brain (freeze)",
        seaad={lvl: seaad_spec("experiments/scGPT_brain/SEA_AD_MTP_ST/record_freeze_s3028.log", lvl) for lvl, _ in LEVELS},
        ms=None,
        eae={
            "ANNOTATION_BROAD": seaad_spec("experiments/kukanja/Kukanja_EAE/record_scgpt_freeze_brain_s3028.log", "ANNOTATION_BROAD"),
            "ANNOTATION_MEDIUM": seaad_spec("experiments/kukanja/Kukanja_EAE/record_scgpt_freeze_brain_s3028.log", "ANNOTATION_MEDIUM"),
            "ANNOTATION_MEDIUM_DA": seaad_spec("experiments/kukanja/Kukanja_EAE/record_scgpt_freeze_brain_s3028.log", "ANNOTATION_MEDIUM_DA"),
            "ANNOTATION_HIGH": seaad_spec("experiments/kukanja/Kukanja_EAE/record_scgpt_freeze_brain_s3028.log", "ANNOTATION_HIGH"),
        },
        missing_reason="not evaluated on Kukanja MS",
        ms_missing_reason="not evaluated on Kukanja MS",
        eae_missing_reason="Kukanja EAE log exists but no completed final annotation metrics were found",
    ),
    MethodSpec(
        family="Foundation (freeze)",
        appendix_family="Foundation (freeze)",
        label="scGPT-Human (freeze)",
        seaad={lvl: seaad_spec("experiments/scGPT_human/SEA_AD_MTP_ST/record_freeze_s3028.log", lvl) for lvl, _ in LEVELS},
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_freeze_human_s3028.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_freeze_human_s3028.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_freeze_human_s3028.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_freeze_human_s3028.log", "LEVEL3"),
        },
        eae=None,
        missing_reason="not evaluated on Kukanja EAE",
    ),
    MethodSpec(
        family="Foundation (freeze)",
        appendix_family="Foundation (freeze)",
        label="UCE (freeze)",
        seaad={lvl: seaad_spec("experiments/UCE/SEA_AD_MTP_ST/record_freeze_s3028.log", lvl) for lvl, _ in LEVELS},
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_freeze_human_s3028.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_freeze_human_s3028.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_freeze_human_s3028.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_freeze_human_s3028.log", "LEVEL3"),
        },
        eae={
            "ANNOTATION_BROAD": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_freeze_mouse_s3028.log", "ANNOTATION_BROAD"),
            "ANNOTATION_MEDIUM": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_freeze_mouse_s3028.log", "ANNOTATION_MEDIUM"),
            "ANNOTATION_MEDIUM_DA": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_freeze_mouse_s3028.log", "ANNOTATION_MEDIUM_DA"),
            "ANNOTATION_HIGH": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_freeze_mouse_s3028.log", "ANNOTATION_HIGH"),
        },
    ),
    MethodSpec(
        family="Foundation (freeze)",
        appendix_family="Foundation (freeze)",
        label="scFoundation (freeze)",
        seaad={lvl: seaad_spec("experiments/scFoundation/SEA_AD_MTP_ST/record_freeze_s3028.log", lvl) for lvl, _ in LEVELS},
        note="Test-only backfilled on 2026-05-03 from existing checkpoint.",
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Foundation (freeze)",
        appendix_family="Foundation (freeze)",
        label="Nicheformer (freeze)",
        seaad={},
        note="Checkpoint exists, but finished SEA-AD test metrics are not available yet.",
        ms=None,
        eae=None,
        missing_reason="SEA-AD checkpoint exists but test did not finish; Kukanja not evaluated",
        seaad_missing_reason="SEA-AD checkpoint exists; repeated test-only evaluation stopped after loading the best model and produced no final metrics",
        ms_missing_reason="not evaluated on Kukanja MS",
        eae_missing_reason="not evaluated on Kukanja EAE",
    ),
    MethodSpec(
        family="Foundation (fine-tuned)",
        appendix_family="Foundation (fine-tuned)",
        label="Geneformer (fine-tuned)",
        seaad={lvl: seaad_spec("experiments/Geneformer/SEA_AD_MTP_ST/record_full_s3028.log", lvl) for lvl, _ in LEVELS},
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_full_s3028.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_full_s3028.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_full_s3028.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/Kukanja_MS/record_geneformer_full_s3028.log", "LEVEL3"),
        },
        eae=None,
        missing_reason="not evaluated on Kukanja EAE",
    ),
    MethodSpec(
        family="Foundation (fine-tuned)",
        appendix_family="Foundation (fine-tuned)",
        label="scGPT-Brain (fine-tuned)",
        seaad={lvl: seaad_spec("experiments/scGPT_brain/SEA_AD_MTP_ST/record_full_s3028.log", lvl) for lvl, _ in LEVELS},
        ms=None,
        eae=None,
        missing_reason="not evaluated on Kukanja",
    ),
    MethodSpec(
        family="Foundation (fine-tuned)",
        appendix_family="Foundation (fine-tuned)",
        label="scGPT-Human (fine-tuned)",
        seaad={lvl: seaad_spec("experiments/scGPT_human/SEA_AD_MTP_ST/record_full_s3028.log", lvl) for lvl, _ in LEVELS},
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_full_human_s3028.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_full_human_s3028.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_full_human_s3028.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/Kukanja_MS/record_scgpt_full_human_s3028.log", "LEVEL3"),
        },
        eae=None,
        missing_reason="not evaluated on Kukanja EAE",
    ),
    MethodSpec(
        family="Foundation (fine-tuned)",
        appendix_family="Foundation (fine-tuned)",
        label="UCE (fine-tuned)",
        seaad={lvl: seaad_spec("experiments/UCE/SEA_AD_MTP_ST/record_full_s3028.log", lvl) for lvl, _ in LEVELS},
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_full_human_s3028.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_full_human_s3028.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_full_human_s3028.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/Kukanja_MS/record_uce_full_human_s3028.log", "LEVEL3"),
        },
        eae={
            "ANNOTATION_BROAD": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_full_mouse_s3028.log", "ANNOTATION_BROAD"),
            "ANNOTATION_MEDIUM": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_full_mouse_s3028.log", "ANNOTATION_MEDIUM"),
            "ANNOTATION_MEDIUM_DA": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_full_mouse_s3028.log", "ANNOTATION_MEDIUM_DA"),
            "ANNOTATION_HIGH": seaad_spec("experiments/kukanja/Kukanja_EAE/record_uce_full_mouse_s3028.log", "ANNOTATION_HIGH"),
        },
    ),
    MethodSpec(
        family="Foundation (fine-tuned)",
        appendix_family="Foundation (fine-tuned)",
        label="scFoundation (fine-tuned)",
        seaad={},
        note="Existing checkpoint found; SEA-AD test-only backfill was retried but did not finish.",
        ms=None,
        eae=None,
        missing_reason="SEA-AD full checkpoint exists but final test metrics were missing at manuscript freeze",
        seaad_missing_reason="SEA-AD best checkpoint exists; test-only backfill was retried on 2026-05-03 but stalled during inference and produced no final metrics",
        ms_missing_reason="not evaluated on Kukanja MS",
        eae_missing_reason="not evaluated on Kukanja EAE",
    ),
    MethodSpec(
        family="Foundation (fine-tuned)",
        appendix_family="Foundation (fine-tuned)",
        label="Nicheformer (fine-tuned)",
        seaad={},
        note="Training log exists but no saved best checkpoint for final SEA-AD test.",
        ms=None,
        eae=None,
        missing_reason="SEA-AD training incomplete or checkpoint missing; Kukanja not evaluated",
        seaad_missing_reason="SEA-AD training log exists, but no saved best checkpoint was found for a test-only backfill",
        ms_missing_reason="not evaluated on Kukanja MS",
        eae_missing_reason="not evaluated on Kukanja EAE",
    ),
    MethodSpec(
        family="Ours",
        appendix_family="Ours",
        label="scBrain (feature residual)",
        seaad={lvl: seaad_spec("experiments/mjm_1/SEA_AD_MTP_ST/record_e3_d1_enc256_dec256_z20_s3028_resfeature.log", lvl) for lvl, _ in LEVELS},
        note="Residual hierarchical decoder with feature-level coupling.",
        ours=True,
        ms={
            "LEVEL0": seaad_spec("experiments/kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log", "LEVEL0"),
            "LEVEL1_5": seaad_spec("experiments/kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log", "LEVEL1_5"),
            "LEVEL2": seaad_spec("experiments/kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log", "LEVEL2"),
            "LEVEL3": seaad_spec("experiments/kukanja/Kukanja_MS/record_mjm1_e3_d1_enc256_dec128_z20_s3028_resfeature.log", "LEVEL3"),
        },
        eae={
            "ANNOTATION_BROAD": seaad_spec("experiments/kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log", "ANNOTATION_BROAD"),
            "ANNOTATION_MEDIUM": seaad_spec("experiments/kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log", "ANNOTATION_MEDIUM"),
            "ANNOTATION_MEDIUM_DA": seaad_spec("experiments/kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log", "ANNOTATION_MEDIUM_DA"),
            "ANNOTATION_HIGH": seaad_spec("experiments/kukanja/Kukanja_EAE/record_mjm1_e3_d1_enc256_dec256_z20_s3028_resfeature.log", "ANNOTATION_HIGH"),
        },
    ),
]


FAMILY_ORDER = [
    "Conventional",
    "Spatial / specialized",
    "Foundation (freeze)",
    "Foundation (fine-tuned)",
    "Semi-supervised",
    "Ours",
]


APPENDIX_METHODS = [method for method in MAIN_METHODS if method.appendix_family is not None]


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read()


def slice_text(text: str, spec: SourceSpec) -> tuple[str, int]:
    start = 0
    end = len(text)
    if spec.start_marker:
        found = text.find(spec.start_marker)
        if found != -1:
            start = found
    if spec.end_marker:
        found = text.find(spec.end_marker, start + 1)
        if found != -1:
            end = found
    return text[start:end], start


def line_no(text: str, char_offset: int) -> int:
    return text.count("\n", 0, char_offset) + 1


def rel_source(path: Path) -> str:
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return path.name


def last_match(pattern: re.Pattern[str], scoped: str, level: str, full_text: str, offset: int) -> tuple[Optional[re.Match[str]], Optional[int]]:
    found = None
    lineno = None
    for match in pattern.finditer(scoped):
        if match.group("level") == level:
            found = match
            lineno = line_no(full_text, offset + match.start())
    return found, lineno


def extract_entry(spec: Optional[SourceSpec]) -> Optional[dict[str, object]]:
    if spec is None or not spec.path.exists():
        return None

    full_text = read_text(spec.path)
    scoped, offset = slice_text(full_text, spec)

    macro_match, macro_line = last_match(MACRO_RE, scoped, spec.level, full_text, offset)
    micro_match, micro_line = last_match(MICRO_RE, scoped, spec.level, full_text, offset)
    micro_only_match, micro_only_line = last_match(MICRO_F1_ONLY_RE, scoped, spec.level, full_text, offset)
    auc_match, auc_line = last_match(MACRO_AUC_RE, scoped, spec.level, full_text, offset)

    if macro_match is None:
        return None

    micro_f1 = None
    if micro_match is not None:
        micro_f1 = float(micro_match.group("f1"))
    elif micro_only_match is not None:
        micro_f1 = float(micro_only_match.group("f1"))
        micro_line = micro_only_line

    if micro_f1 is None:
        return None

    entry = {
        "macro_f1": float(macro_match.group("f1")),
        "micro_f1": micro_f1,
        "path": spec.path,
        "level": spec.display_level or spec.level,
        "macro_line": macro_line,
        "micro_line": micro_line,
    }

    if auc_match is not None:
        entry["auc_macro"] = float(auc_match.group("macro"))
        entry["auc_micro"] = float(auc_match.group("micro"))
        entry["auc_line"] = auc_line

    return entry


def collect_method_data(method: MethodSpec) -> dict[str, object]:
    seaad = {level: extract_entry(method.seaad.get(level)) for level, _ in LEVELS}

    ms = None
    if method.ms:
        ms = {level: extract_entry(method.ms.get(level)) for level, _ in KUKANJA_DATASETS["Kukanja MS"]}

    eae = None
    if method.eae:
        eae = {level: extract_entry(method.eae.get(level)) for level, _ in KUKANJA_DATASETS["Kukanja EAE"]}

    return {
        "family": method.family,
        "appendix_family": method.appendix_family or method.family,
        "label": method.label,
        "note": method.note,
        "ours": method.ours,
        "missing_reason": method.missing_reason,
        "seaad_missing_reason": method.seaad_missing_reason,
        "ms_missing_reason": method.ms_missing_reason,
        "eae_missing_reason": method.eae_missing_reason,
        "SEA-AD": seaad,
        "Kukanja MS": ms,
        "Kukanja EAE": eae,
    }


def collect_all() -> list[dict[str, object]]:
    return [collect_method_data(method) for method in MAIN_METHODS]


def tex_escape(text: str) -> str:
    return (
        text.replace("&", r"\&")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def display_family_name(family: str) -> str:
    return FAMILY_DISPLAY.get(family, family)


def display_label_name(row: dict[str, object]) -> str:
    label = str(row["label"])
    family = str(row["family"])
    if family.startswith("Foundation") and str(row.get("table_context", "")) != "seaad_main":
        label = re.sub(r" \((freeze|fine-tuned)\)$", "", label)
    if family == "Ours" and label == "scBrain (feature residual)":
        return "scBrain"
    label = label.replace("scGPT-Brain", "scGPT (Brain)")
    label = label.replace("scGPT-Human", "scGPT (Human)")
    if str(row.get("table_context", "")) == "seaad_main":
        label = re.sub(r" \((freeze|fine-tuned|full|FT)\)$", "", label)
    return label


def fmt(
    value: Optional[float],
    mark: Optional[str] = None,
    *,
    shade: bool = False,
    bg_color: Optional[str] = None,
) -> str:
    if value is None:
        text = r"N/A"
    else:
        text = f"{value:.4f}"
        if mark == "best":
            text = rf"\best{{{text}}}"
        elif mark == "second":
            text = rf"\second{{{text}}}"
    if bg_color is not None:
        return rf"\cellcolor[rgb]{{{bg_color}}} {text}"
    if shade:
        return rf"\cellcolor[rgb]{{0.949,0.949,0.949}} {text}"
    return text


def shaded_text(text: str, *, shade: bool = False, bg_color: Optional[str] = None) -> str:
    escaped = tex_escape(text)
    if bg_color is not None:
        return rf"\cellcolor[rgb]{{{bg_color}}} {escaped}"
    if shade:
        return rf"\cellcolor[rgb]{{0.949,0.949,0.949}} {escaped}"
    return escaped


def rotated_group_label(text: str) -> str:
    multiline = {
        "Foundation Models": r"Foundation\\Models",
        "Non-Foundation": r"Non-\\Foundation",
    }
    if text in multiline:
        return rf"\rotatebox[origin=c]{{90}}{{\parbox{{1.45cm}}{{\centering\textbf{{{multiline[text]}}}}}}}"
    return rf"\rotatebox[origin=c]{{90}}{{\textbf{{{tex_escape(text)}}}}}"


def has_complete_seaad_result(row: dict[str, object]) -> bool:
    seaad = row["SEA-AD"]
    return all(seaad.get(level) is not None for level, _ in LEVELS)


def seaad_main_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        label = str(row["label"])
        if label in SEAAD_MAIN_EXCLUDED_LABELS:
            continue
        if label not in SEAAD_MAIN_LAYOUT:
            continue
        if not has_complete_seaad_result(row):
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: (SEAAD_MAIN_GROUP_ORDER.index(SEAAD_MAIN_LAYOUT[str(row["label"])]["group"]), SEAAD_MAIN_LAYOUT[str(row["label"])]["order"]))
    return filtered


def is_not_evaluated_reason(reason: str) -> bool:
    return reason.startswith("not evaluated")


def missing_cell(reason: str, *, bg_color: Optional[str] = None) -> str:
    text = r"---" if is_not_evaluated_reason(reason) else r"N/A"
    if bg_color is not None:
        return rf"\multicolumn{{1}}{{c}}{{\cellcolor[rgb]{{{bg_color}}} {text}}}"
    return rf"\multicolumn{{1}}{{c}}{{{text}}}"


def has_reported_result(entry_map: Optional[dict[str, Optional[dict[str, object]]]]) -> bool:
    if not entry_map:
        return False
    return any(cell is not None for cell in entry_map.values())


def appendix_row_should_be_shown(row: dict[str, object], dataset_name: str) -> bool:
    entry_map = row[dataset_name]
    if has_reported_result(entry_map):
        return True
    return not is_not_evaluated_reason(dataset_missing_reason(row, dataset_name))


def appendix_rows_for_dataset(rows: list[dict[str, object]], dataset_name: str) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        label = str(row["label"])
        if label not in SEAAD_MAIN_LAYOUT:
            continue
        if not appendix_row_should_be_shown(row, dataset_name):
            continue
        filtered.append({**row, "table_context": "appendix"})
    filtered.sort(
        key=lambda row: (
            SEAAD_MAIN_GROUP_ORDER.index(SEAAD_MAIN_LAYOUT[str(row["label"])]["group"]),
            SEAAD_MAIN_LAYOUT[str(row["label"])]["order"],
        )
    )
    return filtered


def rank_marks(rows: list[dict[str, object]], dataset: str, level: str, metric: str) -> dict[str, str]:
    values = []
    for row in rows:
        entry = row[dataset]
        if entry is None:
            continue
        cell = entry.get(level)
        if cell is not None and metric in cell and cell[metric] is not None:
            values.append((row["label"], float(cell[metric])))
    values.sort(key=lambda item: item[1], reverse=True)
    marks: dict[str, str] = {}
    if not values:
        return marks

    eps = 1e-12
    best_value = values[0][1]
    for label, value in values:
        if abs(value - best_value) <= eps:
            marks[label] = "best"

    remaining = [(label, value) for label, value in values if value < best_value - eps]
    if not remaining:
        return marks

    second_value = remaining[0][1]
    for label, value in remaining:
        if abs(value - second_value) <= eps:
            marks[label] = "second"
    return marks


def family_counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for family in FAMILY_ORDER:
        counts[family] = sum(1 for row in rows if row[key] == family)
    return counts


def dataset_missing_reason(row: dict[str, object], dataset_name: str) -> str:
    field = DATASET_REASON_FIELD[dataset_name]
    explicit = str(row.get(field, "") or "").strip()
    if explicit:
        return explicit

    generic = str(row.get("missing_reason", "") or "").strip()
    if generic == "not evaluated on Kukanja":
        if dataset_name == "Kukanja MS":
            return "not evaluated on Kukanja MS"
        if dataset_name == "Kukanja EAE":
            return "not evaluated on Kukanja EAE"
    if generic:
        return generic

    if dataset_name == "SEA-AD":
        return "no usable SEA-AD test metrics"
    if dataset_name == "Kukanja MS":
        return "not evaluated on Kukanja MS"
    return "not evaluated on Kukanja EAE"


def build_main_tex(rows: list[dict[str, object]]) -> str:
    rows = seaad_main_rows(rows)
    rows = [{**row, "table_context": "seaad_main"} for row in rows]
    highlight_bg = "1.0,0.94,0.96"
    stripe_bg = "0.949,0.949,0.949"
    body_row_idx = 0
    marks = {}
    for level, _ in LEVELS:
        if level in HIGHLIGHT_LEVELS:
            marks[(level, "macro_f1")] = rank_marks(rows, "SEA-AD", level, "macro_f1")
            marks[(level, "micro_f1")] = rank_marks(rows, "SEA-AD", level, "micro_f1")
            marks[(level, "auc_micro")] = rank_marks(rows, "SEA-AD", level, "auc_micro")

    lines = [
        r"% Auto-generated by build_main_results_table.py",
        r"\newcommand{\best}[1]{\textcolor{red}{\textbf{#1}}}",
        r"\newcommand{\second}[1]{\textcolor{blue}{\underline{#1}}}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{SEA-AD donor-split annotation benchmark across the class, subclass, and supertype hierarchy levels. Rows are organized into Classic, Foundation Models, and Non-Foundation baselines; foundation models are further split into frozen transfer and fine-tuned transfer.}",
        r"\label{tab:main_results}",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{lllccccccccc}",
        r"\toprule",
        r"& & \multirow{2}{*}{\textbf{Method}} & \multicolumn{3}{c}{\textbf{Class}} & \multicolumn{3}{c}{\textbf{Subclass}} & \multicolumn{3}{c}{\textbf{Supertype}} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        r"& & & \textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{AUC-ROC} & \textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{AUC-ROC} & \textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{AUC-ROC} \\",
        r"\midrule",
    ]

    for group in SEAAD_MAIN_GROUP_ORDER:
        group_rows = [row for row in rows if SEAAD_MAIN_LAYOUT.get(str(row["label"]))["group"] == group]
        if not group_rows:
            continue
        group_span = len(group_rows)
        group_started = False
        mode_order = SEAAD_MAIN_MODE_ORDER[group]
        for mode_idx, mode in enumerate(mode_order):
            mode_rows = [row for row in group_rows if SEAAD_MAIN_LAYOUT[str(row["label"])]["mode"] == mode]
            if not mode_rows:
                continue
            mode_span = len(mode_rows)
            for idx, row in enumerate(mode_rows):
                seaad = row["SEA-AD"]
                row_bg = highlight_bg if row["ours"] else (stripe_bg if body_row_idx % 2 == 1 else None)
                cells = []
                for level_idx, (level, _) in enumerate(LEVELS):
                    cell = seaad.get(level)
                    cells.append(
                        fmt(
                            float(cell["macro_f1"]),
                            marks.get((level, "macro_f1"), {}).get(row["label"]),
                            bg_color=row_bg,
                        )
                    )
                    cells.append(
                        fmt(
                            float(cell["micro_f1"]),
                            marks.get((level, "micro_f1"), {}).get(row["label"]),
                            bg_color=row_bg,
                        )
                    )
                    auc_value = None if cell.get("auc_micro") is None else float(cell["auc_micro"])
                    cells.append(
                        fmt(
                            auc_value,
                            marks.get((level, "auc_micro"), {}).get(row["label"]) if auc_value is not None else None,
                            bg_color=row_bg,
                        )
                    )

                category_cell = ""
                if not group_started:
                    category_cell = (
                        rf"\multirow{{{group_span}}}{{*}}{{{rotated_group_label(group)}}}"
                    )
                    group_started = True

                mode_cell = ""
                if mode and idx == 0:
                    mode_cell = (
                        rf"\multirow{{{mode_span}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{tex_escape(mode)}}}}}}}"
                    )

                if row["ours"]:
                    lines.append(r"\cmidrule(lr){2-12}")

                lines.append(
                    f"{category_cell} & {mode_cell} & {shaded_text(display_label_name(row), bg_color=row_bg)} & "
                    + " & ".join(cells)
                    + r" \\"
                )
                body_row_idx += 1
            if mode and mode_idx != len(mode_order) - 1 and any(
                SEAAD_MAIN_LAYOUT[str(row["label"])]["mode"] == next_mode
                for next_mode in mode_order[mode_idx + 1:]
                for row in group_rows
            ):
                lines.append(r"\cmidrule(lr){2-12}")
        if group != SEAAD_MAIN_GROUP_ORDER[-1]:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\vspace{0.3em}",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize\textbf{Notes.} Best and second-best entries are highlighted in red boldface and blue underline. AUC-ROC reports the micro-average one-vs-rest area under the ROC curve when available in the original evaluation log; methods without recorded ROC summaries are marked as `N/A`. The narrow left columns encode baseline family and, for foundation models only, transfer mode (Freeze or FT). Table~\ref{tab:main_results} includes only the selected methods with complete SEA-AD test metrics at all three hierarchy levels. Omitted attempted runs are documented in Appendix~\ref{app:coverage}.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_appendix_tex(rows: list[dict[str, object]]) -> str:
    sections = []
    highlight_bg = "1.0,0.94,0.96"
    stripe_bg = "0.949,0.949,0.949"
    for dataset_name, level_defs in KUKANJA_DATASETS.items():
        dataset_rows = appendix_rows_for_dataset(rows, dataset_name)
        marks = {}
        for level, _ in level_defs:
            marks[(level, "macro_f1")] = rank_marks(dataset_rows, dataset_name, level, "macro_f1")
            marks[(level, "micro_f1")] = rank_marks(dataset_rows, dataset_name, level, "micro_f1")
            marks[(level, "auc_micro")] = rank_marks(dataset_rows, dataset_name, level, "auc_micro")

        label = "appendix_kukanja_transfer" if dataset_name == "Kukanja MS" else "appendix_kukanja_eae"
        title = (
            "Kukanja MS external validation benchmark."
            if dataset_name == "Kukanja MS"
            else "Kukanja EAE external validation benchmark."
        )
        last_col = 3 + 3 * len(level_defs)
        level_headers = " & ".join(
            rf"\multicolumn{{3}}{{c}}{{\textbf{{{tex_escape(display)}}}}}" for _, display in level_defs
        )
        cmidrules = "".join(
            rf"\cmidrule(lr){{{4 + 3 * idx}-{6 + 3 * idx}}}" for idx in range(len(level_defs))
        )
        sub_headers = " & ".join(
            [r"\textbf{Macro-F1} & \textbf{Micro-F1} & \textbf{AUC-ROC}" for _ in level_defs]
        )
        visible_groups = [
            group
            for group in SEAAD_MAIN_GROUP_ORDER
            if any(SEAAD_MAIN_LAYOUT[str(row["label"])]["group"] == group for row in dataset_rows)
        ]

        lines = [
            r"\begin{table*}[t]",
            rf"\caption{{{title} The layout mirrors Table~\ref{{tab:main_results}} and reports Macro-F1, Micro-F1, and AUC-ROC at each hierarchy depth.}}",
            rf"\label{{tab:{label}}}",
            r"\centering",
            r"\footnotesize",
            r"\begin{adjustbox}{max width=\textwidth}",
            r"\setlength{\tabcolsep}{2.5pt}",
            r"\renewcommand{\arraystretch}{0.95}",
            rf"\begin{{tabular}}{{lll{'c' * (3 * len(level_defs))}}}",
            r"\toprule",
            r"& & \multirow{2}{*}{\textbf{Method}} & " + level_headers + r" \\",
            cmidrules,
            rf"& & & {sub_headers} \\",
            r"\midrule",
        ]

        body_row_idx = 0
        for group_idx, group in enumerate(visible_groups):
            group_rows = [
                row
                for row in dataset_rows
                if SEAAD_MAIN_LAYOUT[str(row["label"])]["group"] == group
            ]
            if not group_rows:
                continue
            group_span = len(group_rows)
            group_started = False
            mode_order = SEAAD_MAIN_MODE_ORDER[group]
            for mode_idx, mode in enumerate(mode_order):
                mode_rows = [row for row in group_rows if SEAAD_MAIN_LAYOUT[str(row["label"])]["mode"] == mode]
                if not mode_rows:
                    continue
                mode_span = len(mode_rows)
                for idx, row in enumerate(mode_rows):
                    row_bg = highlight_bg if row["ours"] else (stripe_bg if body_row_idx % 2 == 1 else None)
                    category_cell = ""
                    if not group_started:
                        category_cell = (
                            rf"\multirow{{{group_span}}}{{*}}{{{rotated_group_label(group)}}}"
                        )
                        group_started = True

                    mode_cell = ""
                    if mode and idx == 0:
                        mode_cell = (
                            rf"\multirow{{{mode_span}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{{tex_escape(mode)}}}}}}}"
                        )

                    if row["ours"]:
                        lines.append(rf"\cmidrule(lr){{2-{last_col}}}")

                    entry_map = row[dataset_name]
                    reason = dataset_missing_reason(row, dataset_name)
                    cells = []
                    for level, _ in level_defs:
                        cell = None if entry_map is None else entry_map.get(level)
                        if cell is None:
                            cells.extend(
                                [
                                    missing_cell(reason, bg_color=row_bg),
                                    missing_cell(reason, bg_color=row_bg),
                                    missing_cell(reason, bg_color=row_bg),
                                ]
                            )
                            continue
                        cells.append(
                            fmt(
                                float(cell["macro_f1"]),
                                marks[(level, "macro_f1")].get(row["label"]),
                                bg_color=row_bg,
                            )
                        )
                        cells.append(
                            fmt(
                                float(cell["micro_f1"]),
                                marks[(level, "micro_f1")].get(row["label"]),
                                bg_color=row_bg,
                            )
                        )
                        auc_value = None if cell.get("auc_micro") is None else float(cell["auc_micro"])
                        cells.append(
                            fmt(
                                auc_value,
                                marks[(level, "auc_micro")].get(row["label"]) if auc_value is not None else None,
                                bg_color=row_bg,
                            )
                        )
                    lines.append(
                        f"{category_cell} & {mode_cell} & {shaded_text(display_label_name(row), bg_color=row_bg)} & "
                        + " & ".join(cells)
                        + r" \\"
                    )
                    body_row_idx += 1
                if mode and mode_idx != len(mode_order) - 1 and any(
                    SEAAD_MAIN_LAYOUT[str(row["label"])]["mode"] == next_mode
                    for next_mode in mode_order[mode_idx + 1:]
                    for row in group_rows
                ):
                    lines.append(rf"\cmidrule(lr){{2-{last_col}}}")
            if group_idx != len(visible_groups) - 1:
                lines.append(r"\midrule")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{adjustbox}",
                r"\vspace{0.3em}",
                r"\begin{minipage}{0.98\textwidth}",
                r"\footnotesize\textbf{Notes.} Best and second-best entries are highlighted in red boldface and blue underline. AUC-ROC is the micro-average one-vs-rest score. Unrun method--dataset pairs are omitted, while `N/A` marks attempted runs without recoverable final metrics; Appendix~\ref{app:coverage} lists the remaining coverage gaps.",
                r"\end{minipage}",
                r"\end{table*}",
                "",
            ]
        )
        sections.append("\n".join(lines))

    return "\n".join(sections)


def build_audit(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Main Results Table Audit",
        "",
        "This file is generated by `build_main_results_table.py`.",
        "",
        "## SEA-AD Main Table",
        "",
        "| Method | Family | Level | Macro-F1 | Micro-F1 | AUC-Macro | AUC-Micro | Source / reason |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        for level_key, level_name in LEVELS:
            entry = row["SEA-AD"].get(level_key)
            if entry is None:
                reason = dataset_missing_reason(row, "SEA-AD")
                lines.append(
                    f"| {row['label']} | {row['family']} | {level_name} | -- | -- | -- | -- | {reason} |"
                )
                continue
            source_lines = [f"L{entry['macro_line']}", f"L{entry['micro_line']}"]
            if entry.get("auc_line") is not None:
                source_lines.append(f"L{entry['auc_line']}")
            source = f"`{rel_source(entry['path'])}#{','.join(source_lines)} ({entry['level']})`"
            auc_macro = f"{entry.get('auc_macro', float('nan')):.4f}" if "auc_macro" in entry else "--"
            auc_micro = f"{entry.get('auc_micro', float('nan')):.4f}" if "auc_micro" in entry else "--"
            lines.append(
                f"| {row['label']} | {row['family']} | {level_name} | {entry['macro_f1']:.4f} | {entry['micro_f1']:.4f} | {auc_macro} | {auc_micro} | {source} |"
            )

    lines.extend(
        [
            "",
            "## Kukanja Appendix Tables",
            "",
            "| Method | Dataset | Level | Macro-F1 | Micro-F1 | AUC-Macro | AUC-Micro | Source / reason |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )

    for row in rows:
        for dataset_name, level_defs in KUKANJA_DATASETS.items():
            entry_map = row[dataset_name]
            for level_key, level_name in level_defs:
                cell = None if entry_map is None else entry_map.get(level_key)
                if cell is None:
                    lines.append(
                        f"| {row['label']} | {dataset_name} | {level_name} | -- | -- | -- | -- | {dataset_missing_reason(row, dataset_name)} |"
                    )
                else:
                    source_lines = [f"L{cell['macro_line']}", f"L{cell['micro_line']}"]
                    if cell.get("auc_line") is not None:
                        source_lines.append(f"L{cell['auc_line']}")
                    source = f"`{rel_source(cell['path'])}#{','.join(source_lines)} ({cell['level']})`"
                    auc_macro = f"{cell.get('auc_macro', float('nan')):.4f}" if "auc_macro" in cell else "--"
                    auc_micro = f"{cell.get('auc_micro', float('nan')):.4f}" if "auc_micro" in cell else "--"
                    lines.append(
                        f"| {row['label']} | {dataset_name} | {level_name} | {cell['macro_f1']:.4f} | {cell['micro_f1']:.4f} | {auc_macro} | {auc_micro} | {source} |"
                    )

    lines.extend(
        [
            "",
            "## Missing-result summary",
            "",
            "- `Nicheformer (freeze)`: `experiments/Nicheformer/SEA_AD_MTP_ST/checkpoints/freeze_s3028/best_model.pth` exists, but the SEA-AD test log stops after `Loading best model for test-set evaluation...`; no finished metrics are available yet.",
            "- `scFoundation (fine-tuned)`: `experiments/scFoundation/SEA_AD_MTP_ST/checkpoints/full_s3028/best_model.pth` exists, and test-only reevaluation was retried on 2026-05-03, but the run stalled during inference before printing final metrics.",
            "- `Nicheformer (fine-tuned)`: the training log exists, but no saved `checkpoints/full_s3028/best_model.pth` was found, so final test-only backfill is not currently possible.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = collect_all()
    MAIN_TEX.write_text(build_main_tex(rows))
    APPENDIX_TEX.write_text(build_appendix_tex(rows))
    AUDIT_MD.write_text(build_audit(rows))
    print(f"Wrote {MAIN_TEX}")
    print(f"Wrote {APPENDIX_TEX}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
