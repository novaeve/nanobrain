"""
Step 4: Spectral embedding on STRING graph variants.

Two implementation routes:
  Route 1: sklearn SpectralEmbedding (Laplacian Eigenmaps)
  Route 2: Manual normalized Laplacian eigendecomposition

For each graph variant, compute 2D embeddings and save coordinates.
"""
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.manifold import SpectralEmbedding
import networkx as nx

from config import N_COMPONENTS, RANDOM_SEED, RESULTS_DIR


def _adjacency_matrix(G: nx.Graph, gene_list: list, weighted: bool) -> sp.csr_matrix:
    """Get sparse adjacency matrix in consistent node order."""
    if weighted:
        A = nx.adjacency_matrix(G, nodelist=gene_list, weight="weight")
    else:
        A = nx.adjacency_matrix(G, nodelist=gene_list, weight=None)
    return A.astype(np.float64)


def embed_sklearn(A: sp.csr_matrix, n_components: int = N_COMPONENTS) -> np.ndarray:
    """
    Route 1: sklearn SpectralEmbedding with precomputed affinity.
    Input is the adjacency matrix (affinity='precomputed').
    """
    se = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        random_state=RANDOM_SEED,
    )
    coords = se.fit_transform(A.toarray())
    return coords


def embed_manual_laplacian(A: sp.csr_matrix,
                           n_components: int = N_COMPONENTS) -> np.ndarray:
    """
    Route 2: Manual normalized Laplacian spectral embedding.

    Computes L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
    Then finds the smallest non-trivial eigenvectors.
    """
    n = A.shape[0]
    # Degree vector
    degrees = np.array(A.sum(axis=1)).flatten()
    # Avoid division by zero for isolated nodes (shouldn't happen in GCC)
    degrees[degrees == 0] = 1.0

    # D^{-1/2}
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))

    # Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    L_norm = sp.eye(n) - d_inv_sqrt @ A @ d_inv_sqrt

    # Find smallest eigenvalues (k+1 to skip the trivial zero eigenvalue)
    k = n_components + 1
    # Use 'SM' (smallest magnitude) to get the smallest eigenvalues
    eigenvalues, eigenvectors = eigsh(L_norm, k=k, which="SM")

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the first eigenvector (constant, eigenvalue ~0)
    coords = eigenvectors[:, 1:n_components + 1]

    return coords


def run_all_embeddings(graph_variants: dict) -> dict:
    """
    For each graph variant, compute embeddings using both methods.
    Returns dict with same keys, adding 'embeddings' sub-dict.
    """
    for key, variant in graph_variants.items():
        G = variant["graph"]
        gene_list = variant["gene_list"]
        weighted = variant["stats"]["weighted"]
        n = len(gene_list)

        print(f"\n{'='*60}")
        print(f"Embedding: threshold={key[0]}, weighted={key[1]}, "
              f"hub_removed={key[2]}, n={n}")
        print(f"{'='*60}")

        A = _adjacency_matrix(G, gene_list, weighted)

        # Route 1: sklearn
        t0 = time.time()
        try:
            coords_sklearn = embed_sklearn(A)
            t_sklearn = time.time() - t0
            print(f"  [sklearn] done in {t_sklearn:.1f}s")
        except Exception as e:
            print(f"  [sklearn] FAILED: {e}")
            coords_sklearn = None
            t_sklearn = None

        # Route 2: manual normalized Laplacian
        t0 = time.time()
        try:
            coords_manual = embed_manual_laplacian(A)
            t_manual = time.time() - t0
            print(f"  [manual]  done in {t_manual:.1f}s")
        except Exception as e:
            print(f"  [manual]  FAILED: {e}")
            coords_manual = None
            t_manual = None

        variant["embeddings"] = {
            "sklearn": coords_sklearn,
            "manual": coords_manual,
            "time_sklearn": t_sklearn,
            "time_manual": t_manual,
        }

    return graph_variants


def save_embeddings(graph_variants: dict):
    """Save all embeddings to disk."""
    save_path = os.path.join(RESULTS_DIR, "graph_variants_with_embeddings.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(graph_variants, f)
    print(f"\n[saved] embeddings -> {save_path}")


if __name__ == "__main__":
    # Load pre-built graph variants
    load_path = os.path.join(RESULTS_DIR, "graph_variants.pkl")
    print(f"[load] {load_path}")
    with open(load_path, "rb") as f:
        graph_variants = pickle.load(f)

    graph_variants = run_all_embeddings(graph_variants)
    save_embeddings(graph_variants)
