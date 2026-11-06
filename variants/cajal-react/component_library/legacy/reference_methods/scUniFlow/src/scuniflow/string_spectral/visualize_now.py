"""
Quick visualization of existing embeddings (independent of evaluation).
Can run while evaluation is still in progress.
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def plot_embedding(coords, gene_list, pathway_gene_sets, title, save_path, n_pathways=8):
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    sorted_pws = sorted(pathway_gene_sets.items(),
                        key=lambda x: x[1]["size"], reverse=True)

    # Filter out mega-pathways (>400 genes) for cleaner visualization
    filtered = [p for p in sorted_pws if p[1]["size"] <= 400]
    if len(filtered) < n_pathways:
        filtered = sorted_pws
    selected = filtered[:n_pathways]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, c="lightgrey", alpha=0.2,
               rasterized=True)

    colors = cm.get_cmap("tab10", n_pathways)
    handles = []
    for i, (pid, pinfo) in enumerate(selected):
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        idx = [gene_to_idx[g] for g in pw_genes]
        c = colors(i)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=10, c=[c], alpha=0.7,
                   rasterized=True)
        short_name = pinfo["name"][:40]
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=7, label=f"{short_name} (n={len(pw_genes)})")
        )

    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Spectral dim 1")
    ax.set_ylabel("Spectral dim 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {save_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    emb_path = os.path.join(RESULTS_DIR, "graph_variants_with_embeddings.pkl")
    print(f"[load] {emb_path}")
    with open(emb_path, "rb") as f:
        gv = pickle.load(f)

    count = 0
    for key, variant in gv.items():
        if "embeddings" not in variant:
            continue
        pw_sets = variant.get("pathway_gene_sets", {})
        if not pw_sets:
            continue

        gene_list = variant["gene_list"]
        thr, w, hub = key

        for method in ["sklearn", "manual"]:
            coords = variant["embeddings"].get(method)
            if coords is None:
                continue
            tag = f"thr{thr}_w{w}_hub{hub}_{method}"
            plot_embedding(
                coords, gene_list, pw_sets,
                title=f"Spectral Embedding ({method})\n"
                      f"threshold={thr}, weighted={w}, hub_removed={hub}, "
                      f"n={len(gene_list)}",
                save_path=os.path.join(FIGURES_DIR, f"embedding_{tag}.png"),
            )
            count += 1

    print(f"\nDone! Generated {count} figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
