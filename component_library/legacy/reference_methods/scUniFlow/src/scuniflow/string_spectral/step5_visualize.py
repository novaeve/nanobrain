"""
Step 7: Visualization of spectral embeddings and evaluation results.

Generates:
  1. Scatter plots of 2D embeddings colored by top KEGG pathways
  2. Bar charts of silhouette scores across pathways
  3. Distance ratio distribution plots
  4. kNN enrichment volcano-like plots
  5. Comparison heatmap across configurations
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns

from config import FIGURES_DIR, RESULTS_DIR


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.figsize": (10, 8),
})


def plot_embedding_with_pathways(coords, gene_list, pathway_gene_sets,
                                 title, save_path, n_pathways=8):
    """
    Scatter plot of 2D embedding, highlighting top-N pathway gene sets.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Sort pathways by size, pick top N
    sorted_pws = sorted(pathway_gene_sets.items(),
                        key=lambda x: x[1]["size"], reverse=True)[:n_pathways]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: all genes in grey
    ax.scatter(coords[:, 0], coords[:, 1], s=1, c="lightgrey", alpha=0.3,
               rasterized=True, label="_nolegend_")

    colors = cm.get_cmap("tab10", n_pathways)
    legend_handles = []

    for i, (pid, pinfo) in enumerate(sorted_pws):
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        idx = [gene_to_idx[g] for g in pw_genes]
        c = colors(i)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, c=[c], alpha=0.7,
                   rasterized=True)
        short_name = pinfo["name"][:35]
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=6, label=f"{short_name} (n={len(pw_genes)})")
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Spectral dim 1")
    ax.set_ylabel("Spectral dim 2")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [fig] {save_path}")


def plot_silhouette_bar(silhouette_results, title, save_path, top_n=30):
    """Bar chart of silhouette scores for top pathways."""
    if not silhouette_results:
        return

    df = pd.DataFrame([
        {"pathway": f"{v['name'][:30]} (n={v['size']})", "silhouette": v["silhouette"]}
        for v in silhouette_results.values()
    ]).sort_values("silhouette", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in df["silhouette"]]
    ax.barh(range(len(df)), df["silhouette"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["pathway"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Silhouette Score")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [fig] {save_path}")


def plot_distance_ratio_hist(ratio_results, title, save_path):
    """Histogram of distance ratios across pathways."""
    if not ratio_results:
        return

    ratios = [v["ratio"] for v in ratio_results.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="ratio=1 (random)")
    median_r = np.median(ratios)
    ax.axvline(median_r, color="green", linestyle="-", linewidth=1.5,
               label=f"median={median_r:.3f}")
    ax.set_xlabel("Intra-pathway / Global distance ratio")
    ax.set_ylabel("Number of pathways")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [fig] {save_path}")


def plot_knn_enrichment(knn_results, title, save_path):
    """Volcano-style plot: -log10(p-value) vs log2(odds ratio)."""
    if not knn_results:
        return

    data = []
    for pid, v in knn_results.items():
        or_val = v["odds_ratio"]
        pval = v["pvalue"]
        if np.isnan(or_val) or np.isnan(pval) or or_val <= 0:
            continue
        data.append({
            "name": v["name"][:25],
            "log2_OR": np.log2(or_val) if or_val < 999 else 10,
            "neg_log10_p": -np.log10(max(pval, 1e-300)),
            "significant": pval < 0.05,
        })

    if not data:
        return

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 7))
    sig = df[df["significant"]]
    nsig = df[~df["significant"]]
    ax.scatter(nsig["log2_OR"], nsig["neg_log10_p"], s=15, c="grey", alpha=0.5)
    ax.scatter(sig["log2_OR"], sig["neg_log10_p"], s=20, c="#e74c3c", alpha=0.7)
    ax.axhline(-np.log10(0.05), color="blue", linestyle="--", linewidth=0.8,
               label="p=0.05")
    ax.set_xlabel("log2(Odds Ratio)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [fig] {save_path}")


def plot_comparison_heatmap(summary_csv, save_path):
    """Heatmap comparing key metrics across configurations."""
    try:
        df = pd.read_csv(summary_csv)
    except pd.errors.EmptyDataError:
        print("  [skip] evaluation_summary.csv is empty")
        return
    if len(df) == 0:
        return

    metrics = [c for c in ["sil_mean", "frac_ratio_lt1", "knn_frac_sig"]
               if c in df.columns]
    if not metrics:
        return

    plot_df = df.set_index("config")[metrics].copy()
    # Rename for readability
    rename = {
        "sil_mean": "Mean Silhouette",
        "frac_ratio_lt1": "Frac(ratio<1)",
        "knn_frac_sig": "Frac(kNN sig)",
    }
    plot_df = plot_df.rename(columns=rename)

    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.5)))
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                linewidths=0.5)
    ax.set_title("Metric comparison across configurations")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [fig] {save_path}")


def generate_all_figures(graph_variants: dict, all_eval_results: dict):
    """Generate all figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for key, variant in graph_variants.items():
        if "embeddings" not in variant:
            continue

        pw_sets = variant.get("pathway_gene_sets", {})
        gene_list = variant["gene_list"]
        thr, w, hub = key
        prefix = f"thr{thr}_w{w}_hub{hub}"

        for method in ["sklearn", "manual"]:
            coords = variant["embeddings"].get(method)
            if coords is None:
                continue

            tag = f"{prefix}_{method}"
            eval_key = f"thr{thr}_w{w}_hub{hub}_{method}"

            # 1. Embedding scatter with pathways
            plot_embedding_with_pathways(
                coords, gene_list, pw_sets,
                title=f"Spectral Embedding ({method})\n"
                      f"thr={thr}, weighted={w}, hub_removed={hub}",
                save_path=os.path.join(FIGURES_DIR, f"embedding_{tag}.png"),
            )

            # 2-4. Evaluation plots
            if eval_key in all_eval_results:
                evals = all_eval_results[eval_key]

                if "silhouette" in evals:
                    plot_silhouette_bar(
                        evals["silhouette"],
                        title=f"Silhouette Scores ({tag})",
                        save_path=os.path.join(FIGURES_DIR,
                                                f"silhouette_{tag}.png"),
                    )

                if "distance_ratio" in evals:
                    plot_distance_ratio_hist(
                        evals["distance_ratio"],
                        title=f"Distance Ratio ({tag})",
                        save_path=os.path.join(FIGURES_DIR,
                                                f"dist_ratio_{tag}.png"),
                    )

                if "knn_enrichment" in evals:
                    plot_knn_enrichment(
                        evals["knn_enrichment"],
                        title=f"kNN Enrichment ({tag})",
                        save_path=os.path.join(FIGURES_DIR,
                                                f"knn_enrichment_{tag}.png"),
                    )

    # 5. Comparison heatmap
    summary_csv = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    if os.path.exists(summary_csv):
        plot_comparison_heatmap(
            summary_csv,
            save_path=os.path.join(FIGURES_DIR, "comparison_heatmap.png"),
        )


if __name__ == "__main__":
    emb_path = os.path.join(RESULTS_DIR, "graph_variants_with_embeddings.pkl")
    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.pkl")

    print(f"[load] {emb_path}")
    with open(emb_path, "rb") as f:
        graph_variants = pickle.load(f)

    print(f"[load] {eval_path}")
    with open(eval_path, "rb") as f:
        all_eval_results = pickle.load(f)

    generate_all_figures(graph_variants, all_eval_results)
