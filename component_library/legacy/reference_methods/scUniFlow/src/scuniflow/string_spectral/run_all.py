#!/usr/bin/env python3
"""
Main pipeline: STRING spectral embedding validation.

Runs all steps sequentially:
  Step 1-2: Download STRING data + KEGG pathways
  Step 3:   Build and preprocess graphs
  Step 4:   Spectral embedding (sklearn + manual)
  Step 5-6: Quantitative evaluation
  Step 7:   Visualization

Usage:
    python run_all.py              # run full pipeline
    python run_all.py --from 3     # resume from step 3
"""
import argparse
import os
import pickle
import time

from config import RESULTS_DIR, FIGURES_DIR


def step1_2():
    """Download data and prepare gene/pathway mappings."""
    from step1_download_data import (
        download_string_data, parse_string_links,
        parse_string_info, download_kegg_pathways,
    )

    links_gz, info_gz = download_string_data()
    links_df = parse_string_links(links_gz)
    protein_to_gene = parse_string_info(info_gz)
    kegg_pathways = download_kegg_pathways()

    print(f"\n[summary] edges={len(links_df):,}, "
          f"genes={len(protein_to_gene):,}, "
          f"pathways={len(kegg_pathways)}")

    return links_df, protein_to_gene, kegg_pathways


def step3(links_df, protein_to_gene, kegg_pathways):
    """Build and preprocess graphs."""
    from step2_build_graph import build_graphs, annotate_with_kegg, save_graph_variants

    graph_variants = build_graphs(links_df, protein_to_gene)
    graph_variants = annotate_with_kegg(graph_variants, kegg_pathways)
    save_graph_variants(graph_variants)
    return graph_variants


def step4(graph_variants):
    """Run spectral embeddings."""
    from step3_spectral_embedding import run_all_embeddings, save_embeddings

    graph_variants = run_all_embeddings(graph_variants)
    save_embeddings(graph_variants)
    return graph_variants


def step5_6(graph_variants):
    """Evaluate embeddings."""
    from step4_evaluate import run_evaluation, save_evaluation

    all_results = run_evaluation(graph_variants)
    save_evaluation(all_results)
    return all_results


def step7(graph_variants, all_results):
    """Generate figures."""
    from step5_visualize import generate_all_figures

    generate_all_figures(graph_variants, all_results)


def main():
    parser = argparse.ArgumentParser(
        description="STRING spectral embedding validation pipeline"
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Resume from this step (1-7). Steps 1-2 are combined, 5-6 are combined."
    )
    args = parser.parse_args()
    start = args.from_step

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t_total = time.time()

    if start <= 2:
        print("\n" + "=" * 70)
        print("STEP 1-2: Download STRING data + KEGG pathways")
        print("=" * 70)
        t0 = time.time()
        links_df, protein_to_gene, kegg_pathways = step1_2()
        print(f"  [time] {time.time() - t0:.1f}s")
    else:
        links_df = protein_to_gene = kegg_pathways = None

    if start <= 3:
        if links_df is None:
            # Need data for step 3
            links_df, protein_to_gene, kegg_pathways = step1_2()

        print("\n" + "=" * 70)
        print("STEP 3: Build and preprocess graphs")
        print("=" * 70)
        t0 = time.time()
        graph_variants = step3(links_df, protein_to_gene, kegg_pathways)
        print(f"  [time] {time.time() - t0:.1f}s")
    else:
        graph_variants = None

    if start <= 4:
        if graph_variants is None:
            path = os.path.join(RESULTS_DIR, "graph_variants.pkl")
            print(f"[load] {path}")
            with open(path, "rb") as f:
                graph_variants = pickle.load(f)

        print("\n" + "=" * 70)
        print("STEP 4: Spectral embedding")
        print("=" * 70)
        t0 = time.time()
        graph_variants = step4(graph_variants)
        print(f"  [time] {time.time() - t0:.1f}s")

    if start <= 6:
        if graph_variants is None:
            path = os.path.join(RESULTS_DIR, "graph_variants_with_embeddings.pkl")
            print(f"[load] {path}")
            with open(path, "rb") as f:
                graph_variants = pickle.load(f)

        print("\n" + "=" * 70)
        print("STEP 5-6: Evaluation")
        print("=" * 70)
        t0 = time.time()
        all_results = step5_6(graph_variants)
        print(f"  [time] {time.time() - t0:.1f}s")
    else:
        all_results = None

    if start <= 7:
        if graph_variants is None:
            path = os.path.join(RESULTS_DIR, "graph_variants_with_embeddings.pkl")
            with open(path, "rb") as f:
                graph_variants = pickle.load(f)
        if all_results is None:
            path = os.path.join(RESULTS_DIR, "evaluation_results.pkl")
            with open(path, "rb") as f:
                all_results = pickle.load(f)

        print("\n" + "=" * 70)
        print("STEP 7: Visualization")
        print("=" * 70)
        t0 = time.time()
        step7(graph_variants, all_results)
        print(f"  [time] {time.time() - t0:.1f}s")

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE  (total: {time.time() - t_total:.1f}s)")
    print(f"  Results:  {RESULTS_DIR}")
    print(f"  Figures:  {FIGURES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
