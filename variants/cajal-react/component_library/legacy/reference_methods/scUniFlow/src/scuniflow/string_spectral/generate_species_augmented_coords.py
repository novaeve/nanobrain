#!/usr/bin/env python3
"""
Generate spectral coordinates for an h5ad using species-aware protein-network data.

This is intended for scUniFlow training on non-SEA-AD datasets such as mouse brain
snRNA-seq. The pipeline is:

1. Load all model genes from h5ad (Ensembl IDs + gene symbols)
2. Download / parse species-specific STRING network
3. Build the base protein interaction graph in Ensembl space
4. Optionally augment isolated genes with GO BP Jaccard edges
5. Run spectral embedding on the connected graph
6. Place truly isolated genes on the grid border
"""

import argparse
import gzip
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

from step8_augmented_graph import (
    assemble_coords,
    assign_border_positions,
    load_model_genes,
    spectral_embed,
)


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


SPECIES_PRESETS = {
    "human": {
        "species_id": 9606,
        "go_url": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
        "go_cache": "goa_human.gaf.gz",
        "go_source": "goa_human",
    },
    "mouse": {
        "species_id": 10090,
        "go_url": "https://current.geneontology.org/annotations/mgi.gaf.gz",
        "go_cache": "mgi.gaf.gz",
        "go_source": "mgi",
    },
}

STRING_BASE = "https://stringdb-downloads.org/download"


def download_file(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"[skip] {desc}: {dest.name} already exists")
        return dest
    print(f"[download] {desc} -> {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def species_paths(species_name: str):
    preset = SPECIES_PRESETS[species_name]
    sid = int(preset["species_id"])
    return {
        "links_gz": DATA_DIR / f"{sid}.protein.links.v12.0.txt.gz",
        "info_gz": DATA_DIR / f"{sid}.protein.info.v12.0.txt.gz",
        "links_parquet": DATA_DIR / f"{sid}.string_links.parquet",
        "protein_to_gene": DATA_DIR / f"{sid}.protein_to_gene.json",
        "go_gaf": DATA_DIR / preset["go_cache"],
    }


def ensure_string_files(species_name: str):
    preset = SPECIES_PRESETS[species_name]
    sid = int(preset["species_id"])
    paths = species_paths(species_name)
    download_file(
        f"{STRING_BASE}/protein.links.v12.0/{sid}.protein.links.v12.0.txt.gz",
        paths["links_gz"],
        f"STRING {sid} protein links",
    )
    download_file(
        f"{STRING_BASE}/protein.info.v12.0/{sid}.protein.info.v12.0.txt.gz",
        paths["info_gz"],
        f"STRING {sid} protein info",
    )
    return paths


def load_string_links(species_name: str) -> pd.DataFrame:
    paths = ensure_string_files(species_name)
    cache = paths["links_parquet"]
    if cache.exists():
        print(f"[cache] loading {cache.name}")
        return pd.read_parquet(cache)

    print("[parse] reading STRING links ...")
    df = pd.read_csv(paths["links_gz"], sep=" ", compression="gzip")
    df.columns = ["protein1", "protein2", "combined_score"]
    df.to_parquet(cache, index=False)
    return df


def load_protein_to_gene(species_name: str) -> dict:
    paths = ensure_string_files(species_name)
    cache = paths["protein_to_gene"]
    if cache.exists():
        print(f"[cache] loading {cache.name}")
        with cache.open() as f:
            return json.load(f)

    print("[parse] reading STRING protein info ...")
    df = pd.read_csv(paths["info_gz"], sep="\t", compression="gzip")
    mapping = dict(zip(df["#string_protein_id"], df["preferred_name"]))
    with cache.open("w") as f:
        json.dump(mapping, f)
    return mapping


def build_string_graph(sorted_ids, symbol_of, species_name: str, string_threshold: int):
    links = load_string_links(species_name)
    protein_to_gene = load_protein_to_gene(species_name)
    edges = links[links["combined_score"] >= int(string_threshold)]

    symbol_to_eids = defaultdict(list)
    for eid in sorted_ids:
        symbol_to_eids[str(symbol_of[eid])].append(eid)

    e1_symbols = edges["protein1"].map(protein_to_gene)
    e2_symbols = edges["protein2"].map(protein_to_gene)
    w_series = edges["combined_score"] / 1000.0

    G = nx.Graph()
    G.add_nodes_from(sorted_ids)

    n_edge_events = 0
    for sym1, sym2, w in zip(e1_symbols, e2_symbols, w_series):
        if pd.isna(sym1) or pd.isna(sym2) or sym1 == sym2:
            continue
        eids1 = symbol_to_eids.get(str(sym1), [])
        eids2 = symbol_to_eids.get(str(sym2), [])
        if not eids1 or not eids2:
            continue
        w = float(w)
        for e1 in eids1:
            for e2 in eids2:
                if e1 == e2:
                    continue
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] = max(float(G[e1][e2]["weight"]), w)
                else:
                    G.add_edge(e1, e2, weight=w)
                    n_edge_events += 1

    isolated = [eid for eid in sorted_ids if G.degree(eid) == 0]
    print(
        f"[string] threshold={string_threshold} nodes={G.number_of_nodes():,} "
        f"edges={G.number_of_edges():,} added={n_edge_events:,} isolated={len(isolated):,}"
    )
    return G, isolated


def load_go_annotations(species_name: str) -> dict:
    preset = SPECIES_PRESETS[species_name]
    paths = species_paths(species_name)
    if not paths["go_gaf"].exists():
        try:
            download_file(
                preset["go_url"],
                paths["go_gaf"],
                f"{preset['go_source']} GO annotations",
            )
        except Exception as exc:
            print(f"[warn] failed to download GO annotations: {exc}")
            return {}

    print(f"[parse] reading {paths['go_gaf'].name} ...")
    gene_go = defaultdict(set)
    with gzip.open(paths["go_gaf"], "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            symbol = cols[2]
            go_id = cols[4]
            aspect = cols[8]
            if aspect == "P":
                gene_go[symbol].add(go_id)
    print(f"[go] {len(gene_go):,} genes with BP annotations")
    return dict(gene_go)


def add_go_edges(G, isolated, symbol_of, species_name: str, go_jaccard_threshold: float, go_weight: float):
    if not isolated:
        return G, []

    gene_go = load_go_annotations(species_name)
    if not gene_go:
        print("[go] skipping (no annotation data)")
        return G, isolated

    iso_go = []
    for eid in isolated:
        terms = gene_go.get(str(symbol_of[eid]))
        if terms:
            iso_go.append((eid, terms))

    n_edges = 0
    for i in range(len(iso_go)):
        ei, terms_i = iso_go[i]
        for j in range(i + 1, len(iso_go)):
            ej, terms_j = iso_go[j]
            inter = len(terms_i & terms_j)
            if inter == 0:
                continue
            jac = inter / len(terms_i | terms_j)
            if jac >= float(go_jaccard_threshold):
                w = float(jac * go_weight)
                if not G.has_edge(ei, ej):
                    G.add_edge(ei, ej, weight=w)
                    n_edges += 1

    still_isolated = [eid for eid in isolated if G.degree(eid) == 0]
    print(f"[go] +{n_edges:,} edges | still isolated: {len(still_isolated):,}")
    return G, still_isolated


def infer_output_paths(h5ad_path: Path, species_name: str):
    stem = h5ad_path.stem
    return (
        RESULTS_DIR / f"{stem}_{species_name}_augmented_gene_coords.npy",
        RESULTS_DIR / f"{stem}_{species_name}_augmented_gene_meta.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate species-aware augmented spectral coords for scUniFlow"
    )
    parser.add_argument("--h5ad", required=True, help="Input h5ad path with Ensembl var_names")
    parser.add_argument("--species", choices=sorted(SPECIES_PRESETS.keys()), required=True)
    parser.add_argument("--grid_size", type=int, required=True)
    parser.add_argument("--string_threshold", type=int, default=200)
    parser.add_argument("--go_jaccard_threshold", type=float, default=0.1)
    parser.add_argument("--go_weight", type=float, default=0.15)
    parser.add_argument("--skip_go", action="store_true")
    parser.add_argument("--output_npy", default="")
    parser.add_argument("--output_meta", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    h5ad_path = Path(args.h5ad).resolve()
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad does not exist: {h5ad_path}")

    default_npy, default_meta = infer_output_paths(h5ad_path, args.species)
    output_npy = Path(args.output_npy).resolve() if args.output_npy else default_npy
    output_meta = Path(args.output_meta).resolve() if args.output_meta else default_meta
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    print("=" * 72)
    print("Species-aware protein-network spectral coordinates")
    print("=" * 72)
    print(f"[input] h5ad={h5ad_path}")
    print(f"[input] species={args.species}")
    print(f"[input] grid_size={args.grid_size}")

    sorted_ids, symbol_of = load_model_genes(h5ad_path)

    print("\n[Case A] STRING base graph")
    G_graph, isolated_after_string = build_string_graph(
        sorted_ids=sorted_ids,
        symbol_of=symbol_of,
        species_name=args.species,
        string_threshold=int(args.string_threshold),
    )
    isolated = list(isolated_after_string)

    isolated_after_go = list(isolated)
    if not args.skip_go:
        print("\n[Case B] GO BP augmentation")
        G_graph, isolated_after_go = add_go_edges(
            G_graph,
            isolated,
            symbol_of=symbol_of,
            species_name=args.species,
            go_jaccard_threshold=float(args.go_jaccard_threshold),
            go_weight=float(args.go_weight),
        )
        isolated = list(isolated_after_go)

    print("\n[Spectral embedding]")
    coords_dict = spectral_embed(G_graph, grid_size=int(args.grid_size))

    print(f"\n[Case C] Placing {len(isolated):,} isolated genes on grid border")
    border_dict = assign_border_positions(isolated, grid_size=int(args.grid_size))
    coords = assemble_coords(sorted_ids, coords_dict, border_dict, grid_size=int(args.grid_size))
    np.save(output_npy, coords)

    meta = {
        "source_h5ad": str(h5ad_path),
        "species": str(args.species),
        "grid_size": int(args.grid_size),
        "n_genes": int(len(sorted_ids)),
        "string_threshold": int(args.string_threshold),
        "used_sources": ["STRING"] + ([] if args.skip_go else ["GO_BP"]),
        "n_connected_after_string": int(len(sorted_ids) - len(isolated_after_string)),
        "n_connected_after_go": int(len(sorted_ids) - len(isolated_after_go)),
        "n_isolated_border": int(len(isolated)),
        "coord_shape": list(coords.shape),
        "coord_min": float(coords.min()) if coords.size else 0.0,
        "coord_max": float(coords.max()) if coords.size else 0.0,
        "sorted_gene_list": sorted_ids,
    }
    with output_meta.open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[saved] {output_npy} shape={coords.shape}")
    print(f"[saved] {output_meta}")


if __name__ == "__main__":
    main()
