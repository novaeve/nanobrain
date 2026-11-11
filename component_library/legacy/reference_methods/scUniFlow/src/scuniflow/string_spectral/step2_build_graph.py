"""
Step 3: Build and preprocess the STRING graph.

For each score threshold and weighting scheme:
  - Build graph from STRING edges
  - Extract largest connected component (GCC)
  - Optionally remove hub nodes
  - Record graph statistics
"""
import os
import json
import pickle

import numpy as np
import pandas as pd
import networkx as nx

from config import (
    DATA_DIR, RESULTS_DIR, SCORE_THRESHOLDS,
    HUB_PERCENTILE, KEGG_MIN_PATHWAY_SIZE, KEGG_MAX_PATHWAY_SIZE,
)


def build_graphs(links_df: pd.DataFrame, protein_to_gene: dict) -> dict:
    """
    Build multiple graph variants for different thresholds and weighting.

    Returns dict keyed by (threshold, weighted_bool, hub_removed_bool) -> {
        "graph": nx.Graph,
        "gene_list": list of gene symbols (ordered),
        "stats": dict,
    }
    """
    results = {}

    for threshold in SCORE_THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"Score threshold: {threshold}")
        print(f"{'='*60}")

        # Filter edges by threshold
        mask = links_df["combined_score"] >= threshold
        edges = links_df[mask].copy()
        print(f"  Edges after filtering: {len(edges):,}")

        for weighted in [False, True]:
            w_label = "weighted" if weighted else "unweighted"

            G = nx.Graph()
            for _, row in edges.iterrows():
                p1 = row["protein1"]
                p2 = row["protein2"]
                g1 = protein_to_gene.get(p1)
                g2 = protein_to_gene.get(p2)
                if g1 is None or g2 is None:
                    continue
                if g1 == g2:
                    continue  # skip self-loops
                if weighted:
                    w = row["combined_score"] / 1000.0
                    if G.has_edge(g1, g2):
                        # keep max score if duplicates
                        G[g1][g2]["weight"] = max(G[g1][g2]["weight"], w)
                    else:
                        G.add_edge(g1, g2, weight=w)
                else:
                    G.add_edge(g1, g2)

            print(f"  [{w_label}] Nodes: {G.number_of_nodes():,}, "
                  f"Edges: {G.number_of_edges():,}")

            for remove_hubs in [False, True]:
                h_label = "no_hub_removal" if not remove_hubs else "hub_removed"
                G_proc = G.copy()

                # Remove hub nodes
                if remove_hubs and G_proc.number_of_nodes() > 0:
                    degrees = np.array([d for _, d in G_proc.degree()])
                    cutoff = np.percentile(degrees, HUB_PERCENTILE)
                    hubs = [n for n, d in G_proc.degree() if d > cutoff]
                    G_proc.remove_nodes_from(hubs)
                    print(f"    [{h_label}] Removed {len(hubs)} hubs "
                          f"(degree > {cutoff:.0f})")

                # Extract largest connected component
                if G_proc.number_of_nodes() == 0:
                    print(f"    [{h_label}] Graph is empty, skipping")
                    continue

                gcc_nodes = max(nx.connected_components(G_proc), key=len)
                G_gcc = G_proc.subgraph(gcc_nodes).copy()
                gene_list = sorted(G_gcc.nodes())

                stats = {
                    "threshold": threshold,
                    "weighted": weighted,
                    "hub_removed": remove_hubs,
                    "n_nodes": G_gcc.number_of_nodes(),
                    "n_edges": G_gcc.number_of_edges(),
                    "avg_degree": np.mean([d for _, d in G_gcc.degree()]),
                    "n_connected_components_before_gcc": nx.number_connected_components(G_proc),
                }

                print(f"    [{h_label}] GCC: {stats['n_nodes']:,} nodes, "
                      f"{stats['n_edges']:,} edges, "
                      f"avg_degree={stats['avg_degree']:.1f}")

                key = (threshold, weighted, remove_hubs)
                results[key] = {
                    "graph": G_gcc,
                    "gene_list": gene_list,
                    "stats": stats,
                }

    return results


def annotate_with_kegg(graph_variants: dict, kegg_pathways: dict) -> dict:
    """
    For each graph variant, compute overlap with KEGG pathways.
    Adds 'pathway_gene_sets' to each variant: {pathway_id: set of genes in graph}.
    """
    for key, variant in graph_variants.items():
        gene_set = set(variant["gene_list"])
        pathway_gene_sets = {}

        for pid, pinfo in kegg_pathways.items():
            overlap = gene_set & set(pinfo["genes"])
            if KEGG_MIN_PATHWAY_SIZE <= len(overlap) <= KEGG_MAX_PATHWAY_SIZE:
                pathway_gene_sets[pid] = {
                    "name": pinfo["name"],
                    "genes_in_graph": sorted(overlap),
                    "size": len(overlap),
                }

        variant["pathway_gene_sets"] = pathway_gene_sets
        print(f"  {key}: {len(pathway_gene_sets)} usable KEGG pathways")

    return graph_variants

# ==============================================================================
# 先验图谱扩充补全模块 (Augmented Gene Graph Builder)
# 处理策略：当全量基因不在 STRING 中，启动跨数据库与染色体物理先验对孤立基座补齐
# ==============================================================================

def query_gtex_coexpression(missing_genes, gene_list):
    """
    [开发接口] 查 GTEx 共表达：对于 STRING 中缺失的，利用真实表达相关性补边。
    需后续传入真实的 GTEx 相关性矩阵计算。
    """
    # 示例返回格式: return [ (g1, g2, weight), ... ]
    return []

def find_genomic_neighbors(gene, gene_positions_dict, window=1000000):
    """
    [开发接口] 根据染色体坐标 (chrom, start, end) 检索 1Mb 临近位置。
    """
    if gene not in gene_positions_dict:
        return []
    chrom, start, end = gene_positions_dict[gene]
    neighbors = []
    for g, (c, s, e) in gene_positions_dict.items():
        if g != gene and c == chrom:
            dist = max(0, max(start, s) - min(end, e))
            if dist < window:
                neighbors.append(g)
    return neighbors

def build_augmented_gene_graph(gene_list, links_df, protein_to_gene, gene_positions_dict=None, threshold=400):
    """
    包含从 STRING 建基干、由 GTEx 补遗漏、依染色体补弱边的全方位融合构图方法。
    """
    G = nx.Graph()
    # 强制将要求的全量基因注册入图以防止被当作游离点抛弃丢失
    G.add_nodes_from(gene_list)
    
    # 1. 挂载 STRING 主干网络已知硬边
    mask = links_df["combined_score"] >= threshold
    edges = links_df[mask]
    
    string_genes_found = set()
    for _, row in edges.iterrows():
        g1, g2 = protein_to_gene.get(row["protein1"]), protein_to_gene.get(row["protein2"])
        if g1 and g2 and g1 != g2 and g1 in gene_list and g2 in gene_list:
            w = row["combined_score"] / 1000.0
            G.add_edge(g1, g2, weight=w)
            string_genes_found.update([g1, g2])
            
    # 2. 对 STRING 缺失的节点，通过 GTEx RNA-seq 表达相似度补修桥梁
    missing_genes = [g for g in gene_list if g not in string_genes_found]
    for g1, g2, w in query_gtex_coexpression(missing_genes, gene_list):
        if g1 in gene_list and g2 in gene_list:
            G.add_edge(g1, g2, weight=w)
            
    # 3. 对于极其偏门，两个大库均不存在网络连接的绝对孤岛节点
    # 利用基因组物理位点临近共表达猜想补上极弱引线 (权重 0.1 等代 100分)
    if gene_positions_dict is not None:
        isolated = [g for g in gene_list if G.degree(g) == 0]
        for g in isolated:
            for n in find_genomic_neighbors(g, gene_positions_dict, window=1000000):
                if n in gene_list:
                    G.add_edge(g, n, weight=0.1) 
                    
    return G



def save_graph_variants(graph_variants: dict):
    """Persist graph variants to disk."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "graph_variants.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(graph_variants, f)
    print(f"\n[saved] graph variants -> {save_path}")

    # Also save summary stats as CSV
    rows = []
    for key, variant in graph_variants.items():
        row = variant["stats"].copy()
        row["n_kegg_pathways"] = len(variant.get("pathway_gene_sets", {}))
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(RESULTS_DIR, "graph_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[saved] graph stats -> {stats_path}")


if __name__ == "__main__":
    from step1_download_data import (
        download_string_data, parse_string_links,
        parse_string_info, download_kegg_pathways,
    )

    print("=" * 60)
    print("Step 3: Build and preprocess STRING graphs")
    print("=" * 60)

    links_gz, info_gz = download_string_data()
    links_df = parse_string_links(links_gz)
    protein_to_gene = parse_string_info(info_gz)
    kegg_pathways = download_kegg_pathways()

    graph_variants = build_graphs(links_df, protein_to_gene)
    graph_variants = annotate_with_kegg(graph_variants, kegg_pathways)
    save_graph_variants(graph_variants)
