"""
Global configuration for STRING spectral embedding validation.
"""
import os
from pathlib import Path

# ── Species ──────────────────────────────────────────────────────────────
SPECIES_ID = 9606  # human (change to 10090 for mouse)

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = os.environ.get("SCUNIFLOW_STRING_DATA", str(REPO_ROOT / "data" / "string_spectral"))
RESULTS_DIR = os.environ.get("SCUNIFLOW_STRING_RESULTS", str(REPO_ROOT / "results" / "string_spectral"))
FIGURES_DIR = os.environ.get("SCUNIFLOW_STRING_FIGURES", str(REPO_ROOT / "results" / "string_spectral_figures"))

# ── STRING download URLs (v12.0) ────────────────────────────────────────
STRING_BASE = "https://stringdb-downloads.org/download"
STRING_LINKS_URL = (
    f"{STRING_BASE}/protein.links.v12.0/{SPECIES_ID}.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    f"{STRING_BASE}/protein.info.v12.0/{SPECIES_ID}.protein.info.v12.0.txt.gz"
)
STRING_ALIASES_URL = (
    f"{STRING_BASE}/protein.aliases.v12.0/{SPECIES_ID}.protein.aliases.v12.0.txt.gz"
)

# ── Score thresholds for edge filtering ──────────────────────────────────
SCORE_THRESHOLDS = [400, 700, 900]  # STRING scores are 0-1000 integers

# ── Spectral embedding ──────────────────────────────────────────────────
N_COMPONENTS = 2
RANDOM_SEED = 42

# ── Hub filtering ────────────────────────────────────────────────────────
HUB_PERCENTILE = 99.5  # remove top 0.5% degree nodes

# ── KEGG ─────────────────────────────────────────────────────────────────
KEGG_SPECIES_CODE = "hsa"  # human; use "mmu" for mouse
KEGG_MIN_PATHWAY_SIZE = 15  # only evaluate pathways with >= this many genes in graph
KEGG_MAX_PATHWAY_SIZE = 500

# ── Evaluation ───────────────────────────────────────────────────────────
N_PERMUTATIONS = 200  # for permutation-based significance test
TOP_K_NEIGHBORS = 15  # for kNN-based pathway enrichment test
