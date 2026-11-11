"""
Step 6 (Hilbert): Visualize genes mapped onto a 2D matrix via Hilbert curve.

Pipeline step ⑥→⑦ in the full workflow:
  - Take 2D spectral coordinates for each gene
  - Sort genes along a Hilbert curve to get a 1D ordering that preserves 2D locality
  - Map the 1D ordering into a grid (matrix), so nearby genes in 2D stay nearby in the matrix
  - Visualize which matrix cells belong to specific KEGG pathways
"""
import os
import math
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from hilbertcurve.hilbertcurve import HilbertCurve

from config import RESULTS_DIR, FIGURES_DIR


def coords_to_hilbert_order(coords: np.ndarray, hilbert_p: int = 8) -> np.ndarray:
    """
    Map 2D spectral coordinates to a Hilbert curve index, return sorted order.

    1. Normalize coords to [0, 2^p - 1] integer grid
    2. Compute Hilbert distance for each point
    3. Return argsort by Hilbert distance (= the 1D ordering)
    """
    n = coords.shape[0]
    side = 2 ** hilbert_p  # e.g. 256 for p=8

    # Normalize to [0, side-1]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (coords - mins) / ranges * (side - 1)
    int_coords = np.clip(normed.astype(int), 0, side - 1)

    # Compute Hilbert distance for each point
    hc = HilbertCurve(hilbert_p, 2)
    h_distances = np.array([
        hc.distance_from_point([int(int_coords[i, 0]), int(int_coords[i, 1])])
        for i in range(n)
    ])

    # Sort by Hilbert distance
    order = np.argsort(h_distances)
    return order


def order_to_matrix_position(n_genes: int) -> tuple:
    """
    Compute matrix dimensions and mapping from 1D position to (row, col).
    Returns (grid_size, pos_to_rc) where pos_to_rc[i] = (row, col).
    """
    grid_size = math.ceil(math.sqrt(n_genes))
    pos_to_rc = {}
    for pos in range(n_genes):
        row = pos // grid_size
        col = pos % grid_size
        pos_to_rc[pos] = (row, col)
    return grid_size, pos_to_rc


def plot_hilbert_matrix(gene_list, hilbert_order, pathway_gene_sets,
                        title, save_path, n_pathways=8):
    """
    Visualize the Hilbert-mapped matrix, coloring cells by KEGG pathway membership.
    """
    n = len(gene_list)
    grid_size, pos_to_rc = order_to_matrix_position(n)

    # Build gene -> matrix position mapping
    # hilbert_order[i] = index in gene_list that gets position i
    gene_idx_to_pos = np.zeros(n, dtype=int)
    for pos, gene_idx in enumerate(hilbert_order):
        gene_idx_to_pos[gene_idx] = pos

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Select top pathways (filter mega-pathways, pick medium-sized ones)
    sorted_pws = sorted(pathway_gene_sets.items(),
                        key=lambda x: x[1]["size"], reverse=True)
    filtered = [p for p in sorted_pws if 30 <= p[1]["size"] <= 350]
    if len(filtered) < n_pathways:
        filtered = sorted_pws
    selected = filtered[:n_pathways]

    # --- Plot 1: Overview matrix with all selected pathways ---
    fig, ax = plt.subplots(figsize=(12, 12))
    matrix = np.full((grid_size, grid_size), np.nan)

    # Fill background: occupied cells = 0 (grey), empty = nan (white)
    for pos in range(n):
        r, c = pos_to_rc[pos]
        matrix[r, c] = 0

    # Assign pathway colors (1-based index)
    pw_colors = {}
    color_list = plt.colormaps["tab10"](np.linspace(0, 1, n_pathways))

    for pw_i, (pid, pinfo) in enumerate(selected, 1):
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        for g in pw_genes:
            idx = gene_to_idx[g]
            pos = gene_idx_to_pos[idx]
            r, c = pos_to_rc[pos]
            matrix[r, c] = pw_i
        pw_colors[pw_i] = color_list[pw_i - 1]

    # Build custom colormap: 0=lightgrey, 1..n=pathway colors
    all_colors = [(0.92, 0.92, 0.92, 1.0)]  # 0 = grey background
    for i in range(1, n_pathways + 1):
        all_colors.append(tuple(pw_colors[i]))
    cmap = mcolors.ListedColormap(all_colors)
    bounds = np.arange(-0.5, n_pathways + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest",
              aspect="equal")
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel(f"Column (grid {grid_size}×{grid_size})")
    ax.set_ylabel("Row")

    # Legend
    from matplotlib.patches import Patch
    handles = []
    for pw_i, (pid, pinfo) in enumerate(selected, 1):
        name = pinfo["name"][:35]
        size = len([g for g in pinfo["genes_in_graph"] if g in gene_to_idx])
        handles.append(Patch(facecolor=pw_colors[pw_i],
                             label=f"{name} (n={size})"))
    handles.insert(0, Patch(facecolor=(0.92, 0.92, 0.92), label="Other genes"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")

    # --- Plot 2: Per-pathway individual views ---
    n_cols = 4
    n_rows = math.ceil(n_pathways / n_cols)
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for pw_i, (pid, pinfo) in enumerate(selected):
        ax = axes[pw_i]
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]

        # Background
        bg = np.full((grid_size, grid_size), np.nan)
        for pos in range(n):
            r, c = pos_to_rc[pos]
            bg[r, c] = 0.85
        ax.imshow(bg, cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                  aspect="equal")

        # Highlight pathway genes
        pw_rows, pw_cols = [], []
        for g in pw_genes:
            idx = gene_to_idx[g]
            pos = gene_idx_to_pos[idx]
            r, c = pos_to_rc[pos]
            pw_rows.append(r)
            pw_cols.append(c)

        ax.scatter(pw_cols, pw_rows, s=3, c=[color_list[pw_i]],
                   alpha=0.8, edgecolors="none")
        name = pinfo["name"][:30]
        ax.set_title(f"{name} (n={len(pw_genes)})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(selected), len(axes)):
        axes[i].set_visible(False)

    fig2.suptitle(title.replace("Hilbert Matrix", "Per-pathway Hilbert"),
                  fontsize=13)
    fig2.tight_layout()
    save_path2 = save_path.replace(".png", "_per_pathway.png")
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [saved] {save_path2}")


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
            print(f"\n[Hilbert] {tag}, n={len(gene_list)}")

            # Compute Hilbert ordering from 2D coords
            hilbert_order = coords_to_hilbert_order(coords)

            # Visualize
            grid_size = math.ceil(math.sqrt(len(gene_list)))
            plot_hilbert_matrix(
                gene_list, hilbert_order, pw_sets,
                title=f"Hilbert Matrix ({method})\n"
                      f"thr={thr}, weighted={w}, hub={hub}, "
                      f"grid={grid_size}×{grid_size}",
                save_path=os.path.join(FIGURES_DIR, f"hilbert_{tag}.png"),
            )
            count += 1

    print(f"\nDone! Generated {count * 2} Hilbert figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
