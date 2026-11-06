"""
Step 8: Augmented gene interaction graph -> spectral coordinates for ALL model genes.

Coverage strategy
-----------------
Case A  STRING thr=200             ~53% of model genes covered
                                   (thr 400->200 gains only ~42 genes, so 200 is used
                                    purely as the base without further threshold tuning)
Case B1 GTEx brain co-expression   |r| >= 0.3, edge weight 0.8
Case B2 Genomic proximity          same chromosome, <= 1 Mb, edge weight 0.2
Case B3 GO BP Jaccard              Jaccard(BP_terms) >= 0.1, edge weight 0.15
Case C  Truly isolated genes       placed uniformly on 190x190 grid border;
                                   P2 joint training will relocate them via CFM gradients

Outputs
-------
results/<dataset_name>_augmented_gene_coords.npy
                                    shape (G, 2), values in [0, grid_size)
                                    indexed by sorted Ensembl rank
                                    (same order as model gene_ids = argsort(argsort(ensembl_ids)))
results/<dataset_name>_augmented_gene_meta.json
                                    coverage stats + sorted gene list

Example
-------
python ./string_spectral/step8_augmented_graph.py \
    --h5ad data/pbmc68k_raw_counts.h5ad \
    --grid_size 181

Default outputs for the example above:
results/pbmc68k_raw_counts_augmented_gene_coords.npy
results/pbmc68k_raw_counts_augmented_gene_meta.json

For the default SEA-AD h5ad path, outputs are:
results/seaad_augmented_gene_coords.npy
results/seaad_augmented_gene_meta.json
"""

import argparse
import gzip
import io
import json
import math
import os
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import requests

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import scanpy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

GRID_SIZE  = 190       # default; can be overridden by CLI
BORDER_W   = 2         # grid border width reserved for Case C isolated nodes

STRING_THR = 200
GTEX_R_THR = 0.3       # Pearson |r| threshold for GTEx edges
GTEX_W     = 0.8       # edge weight for GTEx co-expression
GENOMIC_BP = 1_000_000 # 1 Mb window
GENOMIC_W  = 0.2       # edge weight for genomic proximity
GO_JAC_THR = 0.1       # Jaccard threshold for GO BP similarity
GO_W       = 0.15      # edge weight for GO BP edges

H5AD_PATH       = "data/external/MTG/92b37feb-aa2c-40d7-bd90-0a9b5ddb3b27.h5ad"
STRING_PARQUET  = DATA_DIR / "string_links.parquet"
PROTEIN_TO_GENE = DATA_DIR / "protein_to_gene.json"

GTEX_URL   = (
    "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)
GTEX_LOCAL = DATA_DIR / "GTEx_median_tpm.gct.gz"

GOA_URL    = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
GOA_LOCAL  = DATA_DIR / "goa_human.gaf.gz"

ENSEMBL_REST = "https://rest.ensembl.org"

BRAIN_TISSUES = [
    "Brain - Amygdala",
    "Brain - Anterior cingulate cortex (BA24)",
    "Brain - Caudate (basal ganglia)",
    "Brain - Cerebellar Hemisphere",
    "Brain - Cerebellum",
    "Brain - Cortex",
    "Brain - Frontal Cortex (BA9)",
    "Brain - Hippocampus",
    "Brain - Hypothalamus",
    "Brain - Nucleus accumbens (basal ganglia)",
    "Brain - Putamen (basal ganglia)",
    "Brain - Spinal cord (cervical c-1)",
    "Brain - Substantia nigra",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def download_file(url, dest, desc=""):
    """Returns dest path on success, None on HTTP error (caller should skip gracefully)."""
    dest = Path(dest)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest
    print(f"  [download] {desc or dest.name} ...")
    r = requests.get(url, stream=True, timeout=180)
    if not r.ok:
        print(f"  [warn] download failed: HTTP {r.status_code} for {url}")
        return None
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done  = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()
    return dest


def ensembl_batch_lookup(ensembl_ids, fields=None, batch=200):
    """POST /lookup/id for up to `batch` IDs at a time. Returns {eid: record}."""
    url     = f"{ENSEMBL_REST}/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    params  = {}
    if fields:
        params["fields"] = ",".join(fields)
    out = {}
    n   = len(ensembl_ids)
    for start in range(0, n, batch):
        chunk = ensembl_ids[start : start + batch]
        body  = json.dumps({"ids": chunk})
        for attempt in range(4):
            try:
                r = requests.post(url, headers=headers, data=body,
                                  params=params, timeout=90)
                if r.status_code == 200:
                    out.update(r.json())
                    break
                elif r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"  [rate-limit] sleeping {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"  [warn] Ensembl HTTP {r.status_code} at chunk {start}")
                    break
            except Exception as exc:
                print(f"  [warn] Ensembl lookup error: {exc}")
                time.sleep(10)
        if (start // batch) % 10 == 0:
            print(f"    Ensembl lookup: {min(start+batch, n)}/{n} ...", flush=True)
    return out


# ── Step 1: Load model gene list ──────────────────────────────────────────────

def load_model_genes(h5ad_path):
    """
    Returns
    -------
    sorted_ids  : list[str]  length G, Ensembl IDs sorted alphabetically
                             (this is the gene_ids rank order used by the model)
    symbol_of   : dict       ensembl_id -> gene symbol
    """
    adata = sc.read_h5ad(h5ad_path, backed="r")
    eids = list(adata.var.index.astype(str))
    if "feature_name" in adata.var.columns:
        symbols = [str(x) for x in adata.var["feature_name"]]
    elif "SYMBOL" in adata.var.columns:
        symbols = [str(x) for x in adata.var["SYMBOL"]]
    elif "gene_name" in adata.var.columns:
        symbols = [str(x) for x in adata.var["gene_name"]]
    else:
        symbols = eids
    symbol_of = dict(zip(eids, symbols))
    sorted_ids = sorted(eids)
    print(f"[genes] {len(sorted_ids):,} model genes")
    return sorted_ids, symbol_of


# ── Step 2: Case A — STRING base graph ───────────────────────────────────────

def build_string_graph(sorted_ids, symbol_of):
    """
    Build STRING graph at STRING_THR, using ALL model genes as nodes.
    Returns (nx.Graph with Ensembl IDs, list of isolated Ensembl IDs).
    """
    p2g        = json.load(open(PROTEIN_TO_GENE))
    links      = pd.read_parquet(STRING_PARQUET)
    edges      = links[links["combined_score"] >= STRING_THR]
    sym_to_eid = {symbol_of[e]: e for e in sorted_ids}

    # Vectorized: map protein IDs -> Ensembl IDs in one pass
    e1_series = edges["protein1"].map(p2g).map(sym_to_eid)
    e2_series = edges["protein2"].map(p2g).map(sym_to_eid)
    w_series  = edges["combined_score"] / 1000.0

    valid = e1_series.notna() & e2_series.notna() & (e1_series != e2_series)
    df = pd.DataFrame({
        "e1": e1_series[valid].values,
        "e2": e2_series[valid].values,
        "w":  w_series[valid].values,
    })
    # Canonicalize edge direction, then keep max weight per pair
    swap = df["e1"] > df["e2"]
    df.loc[swap, ["e1", "e2"]] = df.loc[swap, ["e2", "e1"]].values
    max_w_df = df.groupby(["e1", "e2"], as_index=False)["w"].max()

    G = nx.Graph()
    G.add_nodes_from(sorted_ids)
    covered = set()
    for row in max_w_df.itertuples(index=False):
        G.add_edge(row.e1, row.e2, weight=row.w)
        covered.update([row.e1, row.e2])

    isolated = [e for e in sorted_ids if e not in covered]
    print(f"[string thr={STRING_THR}]  covered: {len(covered):,}  "
          f"isolated: {len(isolated):,}")
    return G, isolated


# ── Step 3: Case B1 — GTEx brain co-expression ───────────────────────────────

def load_gtex_brain(target_eids):
    """
    Download GTEx median TPM, filter to brain tissues and target genes.
    Returns (X: np.ndarray (n, n_tissues), kept_eids: list[str]).
    """
    if download_file(GTEX_URL, GTEX_LOCAL, "GTEx v8 median TPM (~50 MB)") is None:
        return np.zeros((0, len(BRAIN_TISSUES)), dtype=np.float32), []

    print("  [gtex] parsing GCT ...")
    target_set = set(target_eids)
    records = {}

    with gzip.open(GTEX_LOCAL, "rt") as f:
        next(f)  # #1.2
        next(f)  # dims
        header     = next(f).rstrip("\n").split("\t")
        brain_cols = [header.index(bt) for bt in BRAIN_TISSUES if bt in header]
        n_found    = len(brain_cols)
        print(f"  [gtex] {n_found}/{len(BRAIN_TISSUES)} brain tissue columns found")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid   = parts[0].split(".")[0]   # strip Ensembl version
            if eid not in target_set:
                continue
            records[eid] = np.array([float(parts[c]) for c in brain_cols], dtype=np.float32)

    kept_eids = [e for e in target_eids if e in records]
    X = np.stack([records[e] for e in kept_eids]) if kept_eids else np.zeros((0, n_found))
    print(f"  [gtex] {len(kept_eids):,} / {len(target_eids):,} isolated genes have GTEx data")
    return X, kept_eids


def add_gtex_edges(G, isolated):
    """Add GTEx brain co-expression edges for isolated genes (|r| >= GTEX_R_THR)."""
    if not isolated:
        return G, []

    X, kept_eids = load_gtex_brain(isolated)
    if len(kept_eids) < 2:
        print("[gtex] too few genes with GTEx data, skipping")
        return G, isolated

    # Z-score each gene across brain tissues
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    Xz  = (X - mu) / std    # (n, n_tissues)
    n_t = Xz.shape[1]

    # Chunked pairwise Pearson r to avoid allocating (n x n) matrix at once
    n       = len(kept_eids)
    CHUNK   = 2000
    n_edges = 0
    for i in range(0, n, CHUNK):
        Xi = Xz[i : i + CHUNK]
        for j in range(i, n, CHUNK):
            Xj  = Xz[j : j + CHUNK]
            cor = (Xi @ Xj.T) / n_t     # approximate Pearson r
            rows, cols = np.where(np.abs(cor) >= GTEX_R_THR)

            # Filter to upper triangle in global index space before the Python loop
            gi_arr = i + rows
            gj_arr = j + cols
            upper  = gi_arr < gj_arr
            gi_arr, gj_arr = gi_arr[upper], gj_arr[upper]
            w_arr  = np.abs(cor[rows[upper], cols[upper]]) * GTEX_W

            for gi, gj, w_edge in zip(gi_arr, gj_arr, w_arr):
                e1, e2 = kept_eids[gi], kept_eids[gj]
                w_edge = float(w_edge)
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] = max(G[e1][e2]["weight"], w_edge)
                else:
                    G.add_edge(e1, e2, weight=w_edge)
                    n_edges += 1

    still_isolated = [e for e in isolated if G.degree(e) == 0]
    print(f"[gtex] +{n_edges:,} edges  |  still isolated: {len(still_isolated):,}")
    return G, still_isolated


# ── Step 4: Case B2 — Genomic proximity ──────────────────────────────────────

def add_genomic_edges(G, isolated):
    """Add edges for isolated genes within GENOMIC_BP of each other via Ensembl REST."""
    if not isolated:
        return G, []

    print(f"[genomic] fetching coordinates for {len(isolated):,} genes via Ensembl REST ...")
    records = ensembl_batch_lookup(
        isolated,
        fields=["id", "seq_region_name", "start", "end"],
    )

    # chrom -> sorted list of (start, end, ensembl_id)
    chrom_map = defaultdict(list)
    for eid, rec in records.items():
        if isinstance(rec, dict) and "seq_region_name" in rec:
            chrom = str(rec["seq_region_name"])
            s     = int(rec.get("start", 0))
            e     = int(rec.get("end",   0))
            chrom_map[chrom].append((s, e, eid))

    n_edges = 0
    for chrom, entries in chrom_map.items():
        entries.sort()
        for i in range(len(entries)):
            si, ei, gid_i = entries[i]
            for j in range(i + 1, len(entries)):
                sj, ej, gid_j = entries[j]
                if sj - ei > GENOMIC_BP:
                    break
                dist   = max(0, max(si, sj) - min(ei, ej))
                w_edge = GENOMIC_W * (1.0 - dist / GENOMIC_BP)
                if not G.has_edge(gid_i, gid_j):
                    G.add_edge(gid_i, gid_j, weight=w_edge)
                    n_edges += 1

    still_isolated = [e for e in isolated if G.degree(e) == 0]
    print(f"[genomic] +{n_edges:,} edges  |  still isolated: {len(still_isolated):,}")
    return G, still_isolated


# ── Step 5: Case B3 — GO BP Jaccard ──────────────────────────────────────────

def load_go_annotations():
    """
    Download goa_human.gaf.gz and parse Gene Ontology BP annotations.
    Returns dict: gene_symbol -> set of GO:XXXXXXX term IDs (aspect P only).
    """
    if download_file(GOA_URL, GOA_LOCAL, "GO Annotation File human (~200 MB)") is None:
        return {}
    print("  [go] parsing GAF ...")
    gene_go = defaultdict(set)
    with gzip.open(GOA_LOCAL, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            symbol = cols[2]
            go_id  = cols[4]
            aspect = cols[8]   # P = biological_process
            if aspect == "P":
                gene_go[symbol].add(go_id)
    print(f"  [go] {len(gene_go):,} genes with BP annotations")
    return dict(gene_go)


def add_go_edges(G, isolated, symbol_of):
    """Add GO BP Jaccard edges for isolated genes (Jaccard >= GO_JAC_THR)."""
    if not isolated:
        return G, []

    gene_go = load_go_annotations()
    if not gene_go:
        print("[go] skipping (no annotation data)")
        return G, isolated

    # Collect isolated genes that have GO BP annotations
    iso_go = [(e, gene_go[symbol_of[e]])
              for e in isolated
              if symbol_of[e] in gene_go and len(gene_go[symbol_of[e]]) > 0]

    n_edges = 0
    for i in range(len(iso_go)):
        ei, terms_i = iso_go[i]
        for j in range(i + 1, len(iso_go)):
            ej, terms_j = iso_go[j]
            inter = len(terms_i & terms_j)
            if inter == 0:
                continue
            jac = inter / len(terms_i | terms_j)
            if jac >= GO_JAC_THR:
                w_edge = jac * GO_W
                if not G.has_edge(ei, ej):
                    G.add_edge(ei, ej, weight=w_edge)
                    n_edges += 1

    still_isolated = [e for e in isolated if G.degree(e) == 0]
    print(f"[go] +{n_edges:,} edges  |  still isolated: {len(still_isolated):,}")
    return G, still_isolated


# ── Step 6: Spectral embedding ────────────────────────────────────────────────

def spectral_embed(G, grid_size: int, n_components=2):
    """
    Laplacian eigenmap on the largest connected component.
    Small secondary components get random positions inside the main zone.
    Returns dict: {ensembl_id: np.array([row, col])} for all non-isolated genes.
    """
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    main_nodes = components[0]
    lo = float(BORDER_W)
    hi = float(grid_size - BORDER_W)

    print(f"[spectral] largest component: {len(main_nodes):,} nodes  "
          f"| {len(components)} total components")

    # Build normalized Laplacian for main component
    G_main = G.subgraph(main_nodes).copy()
    nodes  = sorted(G_main.nodes())
    n      = len(nodes)
    idx    = {v: i for i, v in enumerate(nodes)}

    rows_sp, cols_sp, data_sp = [], [], []
    for u, v, d in G_main.edges(data=True):
        w = d.get("weight", 1.0)
        i, j = idx[u], idx[v]
        rows_sp += [i, j]
        cols_sp += [j, i]
        data_sp += [w, w]

    A      = sp.csr_matrix((data_sp, (rows_sp, cols_sp)), shape=(n, n))
    deg    = np.array(A.sum(axis=1)).ravel()
    d_invsq = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L_sym  = sp.eye(n) - d_invsq @ A @ d_invsq

    print(f"[spectral] computing eigenvectors (n={n:,}) ...")
    eigenvalues, eigenvectors = spla.eigsh(L_sym, k=n_components + 1, which="SM")
    order = np.argsort(eigenvalues)
    vecs  = eigenvectors[:, order[1 : n_components + 1]]   # skip trivial eigenvector

    # Normalize to [BORDER_W, GRID_SIZE - BORDER_W)
    for d in range(n_components):
        v = vecs[:, d]
        v_min, v_max = v.min(), v.max()
        if v_max > v_min:
            vecs[:, d] = lo + (v - v_min) / (v_max - v_min) * (hi - lo)
        else:
            vecs[:, d] = (lo + hi) / 2.0

    coords_dict = {nodes[i]: vecs[i].astype(np.float32) for i in range(n)}

    # Small secondary components: scatter randomly inside the main zone
    rng = np.random.RandomState(42)
    for comp in components[1:]:
        anchor = rng.uniform(lo, hi, size=2)
        for e in comp:
            coords_dict[e] = (anchor + rng.randn(2) * 0.5).astype(np.float32)

    return coords_dict


# ── Step 7: Case C — place isolated genes on grid border ─────────────────────

def assign_border_positions(isolated_eids, grid_size: int):
    """
    Distribute truly isolated genes uniformly around the grid border.
    Returns dict: {ensembl_id: np.array([row, col])}.
    """
    border = []
    g = grid_size
    for c in range(g):          border.append((0, c))           # top row
    for r in range(1, g):       border.append((r, g - 1))       # right col
    for c in range(g - 2, -1, -1): border.append((g - 1, c))   # bottom row
    for r in range(g - 2, 0, -1):  border.append((r, 0))        # left col

    rng = np.random.RandomState(0)
    positions = {}
    n_border = len(border)
    for i, eid in enumerate(isolated_eids):
        r, c = border[i % n_border]
        jitter = rng.uniform(-0.3, 0.3, size=2)
        positions[eid] = np.array([r + jitter[0], c + jitter[1]], dtype=np.float32)
    return positions


# ── Step 8: Assemble final (G, 2) array ──────────────────────────────────────

def assemble_coords(sorted_ids, coords_dict, border_dict, grid_size: int):
    """
    Build final (G, 2) float32 array in sorted Ensembl rank order.
    Fallback for any gene not in either dict: grid center.
    """
    coords = np.full((len(sorted_ids), 2), grid_size / 2.0, dtype=np.float32)
    for rank, eid in enumerate(sorted_ids):
        if eid in coords_dict:
            coords[rank] = coords_dict[eid]
        elif eid in border_dict:
            coords[rank] = border_dict[eid]
    return coords


# ── Main ──────────────────────────────────────────────────────────────────────

def infer_dataset_name(h5ad_path: Path):
    stem = h5ad_path.stem.lower()
    full = str(h5ad_path).lower()
    if "sea-ad" in full or "sea_ad" in full or stem.startswith("sea_ad_"):
        return "seaad"
    return h5ad_path.stem


def default_output_paths(h5ad_path: Path):
    dataset_name = infer_dataset_name(h5ad_path)
    out_npy = RESULTS_DIR / f"{dataset_name}_augmented_gene_coords.npy"
    out_meta = RESULTS_DIR / f"{dataset_name}_augmented_gene_meta.json"
    return out_npy, out_meta

def main():
    parser = argparse.ArgumentParser(
        description="Build augmented graph spectral coordinates for all genes in an h5ad"
    )
    parser.add_argument(
        "--h5ad",
        type=str,
        default=H5AD_PATH,
        help="Input h5ad path; var_names should be Ensembl IDs",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=GRID_SIZE,
        help="Grid size, must match scUniFlow model.grid_size",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default=None,
        help="Optional output .npy path; defaults to results/<h5ad_stem>_augmented_gene_coords.npy",
    )
    parser.add_argument(
        "--output_meta",
        type=str,
        default=None,
        help="Optional output .json path; defaults to results/<h5ad_stem>_augmented_gene_meta.json",
    )
    args = parser.parse_args()

    h5ad_path = Path(args.h5ad).resolve()
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad does not exist: {h5ad_path}")

    grid_size = int(args.grid_size)
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    default_npy, default_meta = default_output_paths(h5ad_path)
    out_npy = Path(args.output_npy).resolve() if args.output_npy else default_npy
    out_meta = Path(args.output_meta).resolve() if args.output_meta else default_meta
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    print("=" * 65)
    print("Step 8: Augmented gene graph -> spectral coords for all genes")
    print("=" * 65)
    print(f"[input] h5ad={h5ad_path}")
    print(f"[input] grid_size={grid_size}")

    # 1. Model gene list
    sorted_ids, symbol_of = load_model_genes(h5ad_path)

    # 2. Case A: STRING base
    print("\n[Case A] STRING thr=200")
    G_graph, isolated = build_string_graph(sorted_ids, symbol_of)

    # 3. Case B1: GTEx
    print("\n[Case B1] GTEx brain co-expression")
    G_graph, isolated = add_gtex_edges(G_graph, isolated)

    # 4. Case B2: Genomic proximity
    print("\n[Case B2] Genomic proximity (Ensembl REST)")
    G_graph, isolated = add_genomic_edges(G_graph, isolated)

    # 5. Case B3: GO BP Jaccard
    print("\n[Case B3] GO BP Jaccard similarity")
    G_graph, isolated = add_go_edges(G_graph, isolated, symbol_of)

    print(f"\n[summary] After all augmentation:")
    print(f"  connected genes : {len(sorted_ids) - len(isolated):,}")
    print(f"  isolated (Case C): {len(isolated):,}")

    # 6. Spectral embedding
    print("\n[Spectral embedding]")
    coords_dict = spectral_embed(G_graph, grid_size=grid_size)

    # 7. Case C: border placement
    print(f"\n[Case C] Placing {len(isolated):,} isolated genes on grid border")
    border_dict = assign_border_positions(isolated, grid_size=grid_size)

    # 8. Assemble
    coords = assemble_coords(sorted_ids, coords_dict, border_dict, grid_size=grid_size)

    # 9. Save
    np.save(out_npy, coords)

    n_main = sum(1 for e in sorted_ids if e in coords_dict)
    meta = {
        "source_h5ad"       : str(h5ad_path),
        "grid_size"         : grid_size,
        "n_genes"           : len(sorted_ids),
        "string_thr"        : STRING_THR,
        "n_connected"       : n_main,
        "n_isolated_border" : len(isolated),
        "coord_min"         : float(coords.min()),
        "coord_max"         : float(coords.max()),
        "sorted_gene_list"  : sorted_ids,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[saved] {out_npy}  shape={coords.shape}")
    print(f"[saved] {out_meta}")
    print(f"coord range: [{coords.min():.2f}, {coords.max():.2f}]")


if __name__ == "__main__":
    main()
