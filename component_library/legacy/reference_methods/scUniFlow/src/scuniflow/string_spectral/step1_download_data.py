"""
Step 1-2: Download STRING network and KEGG pathway annotations.

Downloads:
  - STRING protein links (edges with scores)
  - STRING protein info (protein -> gene symbol mapping)
  - KEGG pathway gene sets via REST API
"""
import os
import gzip
import shutil
import urllib.request
import json
import time

import pandas as pd

from config import (
    SPECIES_ID, DATA_DIR, KEGG_SPECIES_CODE,
    STRING_LINKS_URL, STRING_INFO_URL,
)


def download_file(url: str, dest: str, desc: str = "") -> str:
    """Download a file if it doesn't already exist."""
    if os.path.exists(dest):
        print(f"[skip] {desc or dest} already exists")
        return dest
    print(f"[download] {desc or url} ...")
    tmp = dest + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, dest)
    print(f"  -> saved to {dest}")
    return dest


def download_string_data():
    """Download STRING links and info files."""
    os.makedirs(DATA_DIR, exist_ok=True)

    links_gz = os.path.join(DATA_DIR, f"{SPECIES_ID}.protein.links.v12.0.txt.gz")
    info_gz = os.path.join(DATA_DIR, f"{SPECIES_ID}.protein.info.v12.0.txt.gz")

    download_file(STRING_LINKS_URL, links_gz, "STRING protein links")
    download_file(STRING_INFO_URL, info_gz, "STRING protein info")

    return links_gz, info_gz


def parse_string_links(links_gz: str) -> pd.DataFrame:
    """Parse STRING links file into a DataFrame with columns [protein1, protein2, score]."""
    cache = os.path.join(DATA_DIR, "string_links.parquet")
    if os.path.exists(cache):
        print("[cache] loading string_links.parquet")
        return pd.read_parquet(cache)

    print("[parse] reading STRING links ...")
    df = pd.read_csv(links_gz, sep=" ", compression="gzip")
    df.columns = ["protein1", "protein2", "combined_score"]
    df.to_parquet(cache, index=False)
    print(f"  -> {len(df):,} edges parsed")
    return df


def parse_string_info(info_gz: str) -> dict:
    """Build mapping: STRING protein ID -> preferred gene symbol."""
    cache = os.path.join(DATA_DIR, "protein_to_gene.json")
    if os.path.exists(cache):
        print("[cache] loading protein_to_gene.json")
        with open(cache) as f:
            return json.load(f)

    print("[parse] reading STRING protein info ...")
    df = pd.read_csv(info_gz, sep="\t", compression="gzip")
    # columns: #string_protein_id  preferred_name  protein_size  annotation
    mapping = dict(zip(
        df["#string_protein_id"],
        df["preferred_name"],
    ))
    with open(cache, "w") as f:
        json.dump(mapping, f)
    print(f"  -> {len(mapping):,} protein-to-gene mappings")
    return mapping


def _parse_kegg_gene_symbols_local(list_file: str) -> dict:
    """
    Parse local kegg_list_hsa.txt -> {Entrez_ID: gene_symbol}.
    Format: "hsa:ID\\tCDS\\tchromosome\\tSYMBOL, ALIAS1; full name"
    """
    mapping = {}
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            gene_id = parts[0].replace(f"{KEGG_SPECIES_CODE}:", "")
            if len(parts) >= 4:
                desc = parts[3]
            else:
                desc = parts[-1]
            symbol = desc.split(",")[0].split(";")[0].strip()
            mapping[gene_id] = symbol
    return mapping


def download_kegg_pathways() -> dict:
    """
    Load KEGG pathway gene sets from local files.

    Expects two pre-downloaded files in DATA_DIR:
      - kegg_link_hsa_pathway.txt  (from https://rest.kegg.jp/link/hsa/pathway)
      - kegg_list_hsa.txt          (from https://rest.kegg.jp/list/hsa)

    Returns: {pathway_id: {"name": str, "genes": [gene_symbol, ...]}}
    """
    cache = os.path.join(DATA_DIR, "kegg_pathways.json")
    if os.path.exists(cache):
        print("[cache] loading kegg_pathways.json")
        with open(cache) as f:
            return json.load(f)

    link_file = os.path.join(DATA_DIR, "kegg_link_hsa_pathway.txt")
    list_file = os.path.join(DATA_DIR, "kegg_list_hsa.txt")

    if not os.path.exists(link_file) or not os.path.exists(list_file):
        raise FileNotFoundError(
            f"Please download KEGG files first:\n"
            f"  1. https://rest.kegg.jp/link/hsa/pathway -> {link_file}\n"
            f"  2. https://rest.kegg.jp/list/hsa          -> {list_file}"
        )

    # 1. Build Entrez ID -> gene symbol mapping
    print("[parse] reading kegg_list_hsa.txt ...")
    gene_id_to_symbol = _parse_kegg_gene_symbols_local(list_file)
    print(f"  -> {len(gene_id_to_symbol):,} gene ID-to-symbol mappings")

    # 2. Parse pathway -> gene links
    # Format: "path:hsa00010\thsa:10327"
    print("[parse] reading kegg_link_hsa_pathway.txt ...")
    pathway_genes = {}  # pid -> list of Entrez IDs
    pathway_names = {}
    with open(link_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pid = parts[0].replace("path:", "")
            gene_id = parts[1].replace(f"{KEGG_SPECIES_CODE}:", "")
            if pid not in pathway_genes:
                pathway_genes[pid] = []
            pathway_genes[pid].append(gene_id)

    # 3. Convert to gene symbols and build final dict
    pathways = {}
    for pid, gene_ids in pathway_genes.items():
        symbols = []
        for gid in gene_ids:
            sym = gene_id_to_symbol.get(gid)
            if sym:
                symbols.append(sym)
        pathways[pid] = {
            "name": pid,  # will be just the ID; we don't have full names locally
            "genes": symbols,
        }

    # 4. Try to get pathway names from KEGG list/pathway API cache or link file
    # Pathway names are not in the downloaded files, so use pathway IDs as names
    # Optionally fetch names if network available
    try:
        url_list = f"https://rest.kegg.jp/list/pathway/{KEGG_SPECIES_CODE}"
        resp = urllib.request.urlopen(url_list, timeout=10).read().decode()
        for line in resp.strip().split("\n"):
            parts = line.split("\t")
            pid = parts[0].replace("path:", "")
            name = parts[1].split(" - ")[0] if " - " in parts[1] else parts[1]
            if pid in pathways:
                pathways[pid]["name"] = name
        print("  -> fetched pathway names from KEGG")
    except Exception:
        print("  -> [info] could not fetch pathway names, using pathway IDs")

    with open(cache, "w") as f:
        json.dump(pathways, f, indent=2)

    n_with_genes = sum(1 for p in pathways.values() if p["genes"])
    total_genes = len({g for p in pathways.values() for g in p["genes"]})
    print(f"  -> {len(pathways)} pathways ({n_with_genes} with genes), "
          f"{total_genes:,} unique gene symbols")
    return pathways


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1-2: Download STRING data and KEGG pathways")
    print("=" * 60)

    links_gz, info_gz = download_string_data()
    links_df = parse_string_links(links_gz)
    protein_to_gene = parse_string_info(info_gz)
    kegg_pathways = download_kegg_pathways()

    print("\n[summary]")
    print(f"  STRING edges:       {len(links_df):>12,}")
    print(f"  Protein-gene maps:  {len(protein_to_gene):>12,}")
    print(f"  KEGG pathways:      {len(kegg_pathways):>12,}")
    total_genes = len({g for p in kegg_pathways.values() for g in p["genes"]})
    print(f"  KEGG unique genes:  {total_genes:>12,}")
