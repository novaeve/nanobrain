"""
Improved embedding: Spectral (k eigenvectors) -> UMAP -> Hilbert matrix.

Instead of using only eigenvectors v1, v2 as 2D coordinates,
use the first k (e.g. 50) non-trivial eigenvectors of the normalized Laplacian
as a high-dimensional embedding, then apply UMAP to get 2D coordinates
that better preserve cluster structure.
"""
import os
import math
import pickle
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from hilbertcurve.hilbertcurve import HilbertCurve

from config import RESULTS_DIR, FIGURES_DIR, RANDOM_SEED

# ── Parameters ───────────────────────────────────────────────────────────
K_EIGENVECTORS = 50      # number of non-trivial eigenvectors to use
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
HILBERT_P = 8            # Hilbert curve order (2^8 = 256 grid)

# Only run these representative configs to save time
SELECTED_CONFIGS = [
    (400, False, False),  # baseline: loose threshold, unweighted, keep hubs
    (400, False, True),   # loose + hub removed
    (700, False, False),  # mid threshold
    (700, True, False),   # mid threshold + weighted
    (900, False, False),  # strict threshold
]


def compute_spectral_umap(G: nx.Graph, gene_list: list, weighted: bool,
                          k: int = K_EIGENVECTORS) -> np.ndarray:
    """
    1. Build adjacency matrix
    2. Compute normalized Laplacian
    3. Extract first k non-trivial eigenvectors
    4. UMAP to 2D
    """
    n = len(gene_list)
    print(f"    [1/3] Adjacency matrix ({n} nodes) ...")
    if weighted:
        A = nx.adjacency_matrix(G, nodelist=gene_list, weight="weight").astype(np.float64)
    else:
        A = nx.adjacency_matrix(G, nodelist=gene_list, weight=None).astype(np.float64)

    print(f"    [2/3] Normalized Laplacian eigenvectors (k={k}) ...")
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    L_norm = sp.eye(n) - d_inv_sqrt @ A @ d_inv_sqrt

    # k+1 smallest eigenvalues (skip the trivial zero eigenvalue)
    actual_k = min(k + 1, n - 2)
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(L_norm, k=actual_k, which="SM")
    print(f"      eigsh done in {time.time() - t0:.1f}s")

    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    # Skip first (constant) eigenvector
    coords_high = eigenvectors[:, 1:actual_k]
    print(f"      high-dim embedding shape: {coords_high.shape}")

    print(f"    [3/3] UMAP {coords_high.shape[1]}D -> 2D ...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
    )
    coords_2d = reducer.fit_transform(coords_high)
    print(f"      UMAP done in {time.time() - t0:.1f}s")

    return coords_2d


def coords_to_hilbert_order(coords: np.ndarray) -> np.ndarray:
    """Map 2D coordinates to Hilbert curve ordering."""
    n = coords.shape[0]
    side = 2 ** HILBERT_P

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (coords - mins) / ranges * (side - 1)
    int_coords = np.clip(normed.astype(int), 0, side - 1)

    hc = HilbertCurve(HILBERT_P, 2)
    h_distances = np.array([
        hc.distance_from_point([int(int_coords[i, 0]), int(int_coords[i, 1])])
        for i in range(n)
    ])
    return np.argsort(h_distances)


def plot_embedding_2d(coords, gene_list, pathway_gene_sets, title, save_path,
                      n_pathways=8):
    """Scatter plot of UMAP 2D embedding colored by pathways."""
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    sorted_pws = sorted(pathway_gene_sets.items(),
                        key=lambda x: x[1]["size"], reverse=True)
    filtered = [p for p in sorted_pws if 30 <= p[1]["size"] <= 350]
    if len(filtered) < n_pathways:
        filtered = sorted_pws
    selected = filtered[:n_pathways]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, c="lightgrey", alpha=0.2,
               rasterized=True)

    colors = plt.colormaps["tab10"](np.linspace(0, 1, n_pathways))
    handles = []
    for i, (pid, pinfo) in enumerate(selected):
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        idx = [gene_to_idx[g] for g in pw_genes]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=10, c=[colors[i]],
                   alpha=0.7, rasterized=True)
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i],
                   markersize=7,
                   label=f"{pinfo['name'][:35]} (n={len(pw_genes)})")
        )

    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_hilbert_matrix(gene_list, hilbert_order, pathway_gene_sets,
                        title, save_path, n_pathways=8):
    """Hilbert matrix overview + per-pathway subplots."""
    n = len(gene_list)
    grid_size = math.ceil(math.sqrt(n))
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    gene_idx_to_pos = np.zeros(n, dtype=int)
    for pos, gene_idx in enumerate(hilbert_order):
        gene_idx_to_pos[gene_idx] = pos

    sorted_pws = sorted(pathway_gene_sets.items(),
                        key=lambda x: x[1]["size"], reverse=True)
    filtered = [p for p in sorted_pws if 30 <= p[1]["size"] <= 350]
    if len(filtered) < n_pathways:
        filtered = sorted_pws
    selected = filtered[:n_pathways]

    color_list = plt.colormaps["tab10"](np.linspace(0, 1, n_pathways))

    # ── Overview matrix ──
    fig, ax = plt.subplots(figsize=(12, 12))
    matrix = np.full((grid_size, grid_size), np.nan)
    for pos in range(n):
        r, c = pos // grid_size, pos % grid_size
        matrix[r, c] = 0

    pw_colors = {}
    for pw_i, (pid, pinfo) in enumerate(selected, 1):
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]
        for g in pw_genes:
            idx = gene_to_idx[g]
            pos = gene_idx_to_pos[idx]
            r, c = pos // grid_size, pos % grid_size
            matrix[r, c] = pw_i
        pw_colors[pw_i] = color_list[pw_i - 1]

    all_colors = [(0.92, 0.92, 0.92, 1.0)]
    for i in range(1, n_pathways + 1):
        all_colors.append(tuple(pw_colors[i]))
    cmap = mcolors.ListedColormap(all_colors)
    bounds = np.arange(-0.5, n_pathways + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel(f"Column (grid {grid_size}×{grid_size})")
    ax.set_ylabel("Row")

    handles = [Patch(facecolor=(0.92, 0.92, 0.92), label="Other genes")]
    for pw_i, (pid, pinfo) in enumerate(selected, 1):
        size = len([g for g in pinfo["genes_in_graph"] if g in gene_to_idx])
        handles.append(Patch(facecolor=pw_colors[pw_i],
                             label=f"{pinfo['name'][:35]} (n={size})"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")

    # ── Per-pathway subplots ──
    n_cols = 4
    n_rows = math.ceil(n_pathways / n_cols)
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for pw_i, (pid, pinfo) in enumerate(selected):
        ax = axes[pw_i]
        pw_genes = [g for g in pinfo["genes_in_graph"] if g in gene_to_idx]

        bg = np.full((grid_size, grid_size), np.nan)
        for pos in range(n):
            r, c = pos // grid_size, pos % grid_size
            bg[r, c] = 0.85
        ax.imshow(bg, cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                  aspect="equal")

        pw_rows, pw_cols = [], []
        for g in pw_genes:
            idx = gene_to_idx[g]
            pos = gene_idx_to_pos[idx]
            pw_rows.append(pos // grid_size)
            pw_cols.append(pos % grid_size)

        ax.scatter(pw_cols, pw_rows, s=3, c=[color_list[pw_i]],
                   alpha=0.8, edgecolors="none")
        ax.set_title(f"{pinfo['name'][:30]} (n={len(pw_genes)})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(selected), len(axes)):
        axes[i].set_visible(False)

    fig2.suptitle(title.replace("Hilbert Matrix", "Per-pathway Hilbert"), fontsize=13)
    fig2.tight_layout()
    fig2.savefig(save_path.replace(".png", "_per_pathway.png"), dpi=150,
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"  [saved] {save_path.replace('.png', '_per_pathway.png')}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    emb_path = os.path.join(RESULTS_DIR, "graph_variants_with_embeddings.pkl")
    print(f"[load] {emb_path}")
    with open(emb_path, "rb") as f:
        gv = pickle.load(f)

    # Store new UMAP embeddings for later use
    umap_results = {}

    for key in SELECTED_CONFIGS:
        if key not in gv:
            print(f"[skip] config {key} not found")
            continue

        variant = gv[key]
        pw_sets = variant.get("pathway_gene_sets", {})
        if not pw_sets:
            print(f"[skip] {key}: no pathway annotations")
            continue

        G = variant["graph"]
        gene_list = variant["gene_list"]
        thr, w, hub = key
        tag = f"thr{thr}_w{w}_hub{hub}"

        print(f"\n{'='*60}")
        print(f"Config: {tag}, n={len(gene_list)}")
        print(f"{'='*60}")

        # Compute spectral + UMAP embedding
        coords_2d = compute_spectral_umap(G, gene_list, w)
        umap_results[key] = coords_2d

        # 1. UMAP 2D scatter with pathways
        plot_embedding_2d(
            coords_2d, gene_list, pw_sets,
            title=f"Spectral({K_EIGENVECTORS}D) + UMAP\n"
                  f"thr={thr}, weighted={w}, hub={hub}, n={len(gene_list)}",
            save_path=os.path.join(FIGURES_DIR, f"umap_{tag}.png"),
        )

        # 2. Hilbert mapping
        hilbert_order = coords_to_hilbert_order(coords_2d)
        grid_size = math.ceil(math.sqrt(len(gene_list)))
        plot_hilbert_matrix(
            gene_list, hilbert_order, pw_sets,
            title=f"Spectral+UMAP Hilbert Matrix\n"
                  f"thr={thr}, weighted={w}, hub={hub}, "
                  f"grid={grid_size}×{grid_size}",
            save_path=os.path.join(FIGURES_DIR, f"umap_hilbert_{tag}.png"),
        )

    # Save UMAP embeddings
    save_path = os.path.join(RESULTS_DIR, "umap_embeddings.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(umap_results, f)
    print(f"\n[saved] UMAP embeddings -> {save_path}")
    print("Done!")


if __name__ == "__main__":
    main()
