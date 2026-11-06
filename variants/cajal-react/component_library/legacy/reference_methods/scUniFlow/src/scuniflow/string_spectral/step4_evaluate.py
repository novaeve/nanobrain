"""
Step 5-6: Quantitative evaluation of spectral embedding quality.

Metrics:
  1. Silhouette Score: measures how well KEGG pathway genes cluster together
  2. Intra-pathway distance ratio: avg intra-pathway dist / avg inter-gene dist
  3. kNN pathway enrichment: for each gene, check if its k-nearest neighbors
     are enriched in same-pathway genes (Fisher exact test)
  4. Permutation test: compare observed silhouette to random label shuffles
"""
import os
import pickle
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import fisher_exact
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from config import (
    RESULTS_DIR, N_PERMUTATIONS, TOP_K_NEIGHBORS,
    KEGG_MIN_PATHWAY_SIZE,
)


def compute_silhouette_per_pathway(coords: np.ndarray,
                                   gene_list: list,
                                   pathway_gene_sets: dict) -> dict:
    """
    For each pathway, compute silhouette score treating pathway genes as one
    cluster and all other genes as another.
    """
    results = {}
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    for pid, pinfo in pathway_gene_sets.items():
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        if len(pw_genes) < KEGG_MIN_PATHWAY_SIZE:
            continue

        pw_idx = [gene_to_idx[g] for g in pw_genes]
        labels = np.zeros(len(gene_list), dtype=int)
        labels[pw_idx] = 1

        # Need at least 2 labels and some points in each
        if labels.sum() < 2 or (labels == 0).sum() < 2:
            continue

        try:
            sil = silhouette_score(coords, labels)
            results[pid] = {
                "name": pinfo["name"],
                "size": len(pw_genes),
                "silhouette": float(sil),
            }
        except Exception:
            pass

    return results


def compute_distance_ratio(coords: np.ndarray,
                           gene_list: list,
                           pathway_gene_sets: dict) -> dict:
    """
    Intra-pathway distance ratio: mean(intra-pathway pairwise dist) /
    mean(all pairwise dist) for a random subsample.

    Ratio < 1 means pathway genes are closer than average.
    """
    results = {}
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Compute global mean distance on a subsample to avoid O(n^2) cost
    np.random.seed(42)
    n = len(gene_list)
    sample_size = min(5000, n)
    sample_idx = np.random.choice(n, sample_size, replace=False)
    global_dists = pdist(coords[sample_idx])
    global_mean = np.mean(global_dists)

    for pid, pinfo in pathway_gene_sets.items():
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        if len(pw_genes) < KEGG_MIN_PATHWAY_SIZE:
            continue

        pw_idx = np.array([gene_to_idx[g] for g in pw_genes])
        pw_coords = coords[pw_idx]
        intra_dists = pdist(pw_coords)
        if len(intra_dists) == 0:
            continue

        intra_mean = np.mean(intra_dists)
        ratio = intra_mean / global_mean if global_mean > 0 else np.nan

        results[pid] = {
            "name": pinfo["name"],
            "size": len(pw_genes),
            "intra_mean_dist": float(intra_mean),
            "global_mean_dist": float(global_mean),
            "ratio": float(ratio),
        }

    return results


def compute_knn_enrichment(coords: np.ndarray,
                           gene_list: list,
                           pathway_gene_sets: dict,
                           k: int = TOP_K_NEIGHBORS) -> dict:
    """
    For each pathway, test whether pathway genes' k-nearest neighbors are
    enriched for same-pathway genes using Fisher exact test.

    For each pathway gene, count how many of its kNN are also in the pathway.
    Aggregate with a one-sided Fisher exact test.
    """
    results = {}
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n = len(gene_list)

    # Fit kNN model
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    # indices[:,0] is the point itself, skip it
    knn_idx = indices[:, 1:]

    for pid, pinfo in pathway_gene_sets.items():
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        if len(pw_genes) < KEGG_MIN_PATHWAY_SIZE:
            continue

        pw_idx_set = set(gene_to_idx[g] for g in pw_genes)
        pw_size = len(pw_idx_set)

        # Count: for pathway genes, how many kNN are in pathway
        hits_in = 0   # pathway gene's neighbor is in pathway
        hits_out = 0  # pathway gene's neighbor is NOT in pathway

        for idx in pw_idx_set:
            neighbors = knn_idx[idx]
            hits_in += sum(1 for nb in neighbors if nb in pw_idx_set)
            hits_out += sum(1 for nb in neighbors if nb not in pw_idx_set)

        # Expected by chance
        # Contingency table for Fisher test:
        # |                  | Neighbor in pathway | Neighbor not in pathway |
        # | Pathway genes    | hits_in             | hits_out                |
        # | Non-pathway genes| expected_in         | expected_out            |
        non_pw_idx = [i for i in range(n) if i not in pw_idx_set]
        hits_in_nonpw = 0
        hits_out_nonpw = 0
        for idx in non_pw_idx:
            neighbors = knn_idx[idx]
            hits_in_nonpw += sum(1 for nb in neighbors if nb in pw_idx_set)
            hits_out_nonpw += sum(1 for nb in neighbors if nb not in pw_idx_set)

        table = [[hits_in, hits_out],
                 [hits_in_nonpw, hits_out_nonpw]]

        try:
            odds_ratio, pvalue = fisher_exact(table, alternative="greater")
        except Exception:
            odds_ratio, pvalue = np.nan, np.nan

        results[pid] = {
            "name": pinfo["name"],
            "size": pw_size,
            "hits_in_pathway": hits_in,
            "total_pw_neighbors": hits_in + hits_out,
            "odds_ratio": float(odds_ratio) if not np.isinf(odds_ratio) else 999.0,
            "pvalue": float(pvalue),
        }

    return results


def permutation_test_silhouette(coords: np.ndarray,
                                gene_list: list,
                                pathway_gene_sets: dict,
                                n_perms: int = N_PERMUTATIONS) -> dict:
    """
    For top pathways, compare observed silhouette to distribution under
    random label permutation.
    """
    results = {}
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Pick top 20 largest pathways for efficiency
    sorted_pws = sorted(pathway_gene_sets.items(),
                        key=lambda x: x[1]["size"], reverse=True)[:20]

    for pid, pinfo in sorted_pws:
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        if len(pw_genes) < KEGG_MIN_PATHWAY_SIZE:
            continue

        pw_idx = [gene_to_idx[g] for g in pw_genes]
        pw_size = len(pw_idx)

        # Observed
        labels = np.zeros(len(gene_list), dtype=int)
        labels[pw_idx] = 1
        try:
            obs_sil = silhouette_score(coords, labels)
        except Exception:
            continue

        # Permutations: randomly assign same number of genes as "pathway"
        perm_sils = []
        rng = np.random.RandomState(42)
        n = len(gene_list)
        for _ in range(n_perms):
            perm_labels = np.zeros(n, dtype=int)
            perm_idx = rng.choice(n, pw_size, replace=False)
            perm_labels[perm_idx] = 1
            try:
                perm_sils.append(silhouette_score(coords, perm_labels))
            except Exception:
                pass

        perm_sils = np.array(perm_sils)
        # p-value: fraction of permutations with silhouette >= observed
        pvalue = np.mean(perm_sils >= obs_sil) if len(perm_sils) > 0 else np.nan

        results[pid] = {
            "name": pinfo["name"],
            "size": pw_size,
            "observed_silhouette": float(obs_sil),
            "perm_mean": float(np.mean(perm_sils)),
            "perm_std": float(np.std(perm_sils)),
            "pvalue": float(pvalue),
            "significant": bool(pvalue < 0.05),
        }

        print(f"  {pid} ({pinfo['name'][:30]}): sil={obs_sil:.4f}, "
              f"perm_mean={np.mean(perm_sils):.4f}, p={pvalue:.4f}")

    return results


def run_evaluation(graph_variants: dict) -> dict:
    """Run all evaluation metrics on each graph variant and embedding method."""
    all_results = {}

    for key, variant in graph_variants.items():
        if "embeddings" not in variant:
            continue

        pathway_gene_sets = variant.get("pathway_gene_sets", {})
        if not pathway_gene_sets:
            continue

        gene_list = variant["gene_list"]

        for method in ["sklearn", "manual"]:
            coords = variant["embeddings"].get(method)
            if coords is None:
                continue

            label = f"thr{key[0]}_w{key[1]}_hub{key[2]}_{method}"
            print(f"\n{'='*60}")
            print(f"Evaluating: {label}")
            print(f"  nodes={len(gene_list)}, pathways={len(pathway_gene_sets)}")
            print(f"{'='*60}")

            eval_result = {}

            # 1. Silhouette per pathway
            print("\n[1] Silhouette per pathway ...")
            eval_result["silhouette"] = compute_silhouette_per_pathway(
                coords, gene_list, pathway_gene_sets
            )
            sil_values = [v["silhouette"] for v in eval_result["silhouette"].values()]
            if sil_values:
                print(f"    mean={np.mean(sil_values):.4f}, "
                      f"median={np.median(sil_values):.4f}")

            # 2. Distance ratio
            print("\n[2] Intra-pathway distance ratio ...")
            eval_result["distance_ratio"] = compute_distance_ratio(
                coords, gene_list, pathway_gene_sets
            )
            ratios = [v["ratio"] for v in eval_result["distance_ratio"].values()]
            if ratios:
                frac_below1 = np.mean(np.array(ratios) < 1.0)
                print(f"    mean_ratio={np.mean(ratios):.4f}, "
                      f"fraction<1={frac_below1:.2%}")

            # 3. kNN enrichment
            print("\n[3] kNN pathway enrichment ...")
            eval_result["knn_enrichment"] = compute_knn_enrichment(
                coords, gene_list, pathway_gene_sets
            )
            pvals = [v["pvalue"] for v in eval_result["knn_enrichment"].values()]
            if pvals:
                frac_sig = np.mean(np.array(pvals) < 0.05)
                print(f"    fraction significant (p<0.05): {frac_sig:.2%}")

            # 4. Permutation test (only on one representative config to save time)
            if method == "sklearn":
                print("\n[4] Permutation test (top 20 pathways) ...")
                eval_result["permutation_test"] = permutation_test_silhouette(
                    coords, gene_list, pathway_gene_sets
                )

            all_results[label] = eval_result

    return all_results


def save_evaluation(all_results: dict):
    """Save evaluation results."""
    save_path = os.path.join(RESULTS_DIR, "evaluation_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n[saved] evaluation -> {save_path}")

    # Also save a summary CSV
    rows = []
    for label, evals in all_results.items():
        row = {"config": label}

        # Silhouette summary
        if "silhouette" in evals:
            sils = [v["silhouette"] for v in evals["silhouette"].values()]
            row["sil_mean"] = np.mean(sils) if sils else np.nan
            row["sil_median"] = np.median(sils) if sils else np.nan
            row["n_pathways_sil"] = len(sils)

        # Distance ratio summary
        if "distance_ratio" in evals:
            ratios = [v["ratio"] for v in evals["distance_ratio"].values()]
            row["ratio_mean"] = np.mean(ratios) if ratios else np.nan
            row["frac_ratio_lt1"] = (
                np.mean(np.array(ratios) < 1.0) if ratios else np.nan
            )

        # kNN enrichment summary
        if "knn_enrichment" in evals:
            pvals = [v["pvalue"] for v in evals["knn_enrichment"].values()]
            row["knn_frac_sig"] = (
                np.mean(np.array(pvals) < 0.05) if pvals else np.nan
            )
            row["n_pathways_knn"] = len(pvals)

        # Permutation test summary
        if "permutation_test" in evals:
            perms = evals["permutation_test"]
            n_sig = sum(1 for v in perms.values() if v["significant"])
            row["perm_n_tested"] = len(perms)
            row["perm_n_significant"] = n_sig

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] summary -> {summary_path}")


if __name__ == "__main__":
    load_path = os.path.join(RESULTS_DIR, "graph_variants_with_embeddings.pkl")
    print(f"[load] {load_path}")
    with open(load_path, "rb") as f:
        graph_variants = pickle.load(f)

    all_results = run_evaluation(graph_variants)
    save_evaluation(all_results)
