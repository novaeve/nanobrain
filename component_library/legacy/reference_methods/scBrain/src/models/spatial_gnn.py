"""
STELLAR-like Spatial GNN for hierarchical cell annotation.

Architecture:
  1. Build KNN spatial graph (precomputed)
  2. Gene expression encoder (Linear → hidden)
  3. GCN message passing layers (aggregate neighbor features)
  4. Hierarchical classification heads (same residual design as mjm_1)
  5. Reconstruction decoder

The spatial graph encodes cell neighborhood structure — cells that are
physically close share information through message passing, compensating
for the limited gene panel in MERFISH data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree


# ── Graph construction ──────────────────────────────────────────────────────

def build_knn_graph(spatial_coords, k=15):
    """
    Build a KNN spatial graph from 2D coordinates.
    Returns edge_index [2, E] as a LongTensor (COO format).
    """
    tree = cKDTree(spatial_coords)
    _, indices = tree.query(spatial_coords, k=k + 1)  # +1 because self is included
    n = len(spatial_coords)
    src = np.repeat(np.arange(n), k)
    dst = indices[:, 1:].flatten()  # exclude self-loop
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()


# ── GCN Layer (pure PyTorch) ────────────────────────────────────────────────

class GCNConv(nn.Module):
    """Simple GCN convolution: h' = D^{-1} A X W + b (mean aggregation)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: [N, in_dim]
        edge_index: [2, E] (src → dst)
        """
        src, dst = edge_index
        N = x.size(0)

        # Transform
        h = self.linear(x)  # [N, out_dim]

        # Aggregate (mean of neighbors)
        # Use scatter_mean via index_add
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])

        # Degree normalization
        deg = torch.zeros(N, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        agg = agg / deg

        return agg


# ── Spatial GNN Model ───────────────────────────────────────────────────────

class SpatialGNN(nn.Module):
    """
    STELLAR-inspired spatial GNN for hierarchical cell annotation.

    Pipeline:
        X [B, 140] → gene_encoder → h0 [B, hidden]
        h0 → GCN_1 → h1 → GCN_2 → h2 (spatial context aggregation)
        h2 → head_class    → logits_class [B, 3]
        h2 → head_subclass → logits_subclass [B, 24]  (+ residual from class)
        h2 → head_supertype→ logits_supertype [B, 137] (+ residual from subclass)
        h2 → recon_decoder → x_hat [B, 140]
    """
    def __init__(self,
                 input_dim=140,
                 hidden_dim=256,
                 latent_dim=128,
                 n_gcn_layers=2,
                 dropout=0.3,
                 output_num=[3, 24, 137]):
        super().__init__()

        # Gene expression encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for _ in range(n_gcn_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn_norms.append(nn.LayerNorm(hidden_dim))
        self.gcn_dropout = nn.Dropout(dropout)

        # Projection to latent
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Hierarchical classification heads (with cross-level feature residual)
        dec_dim = latent_dim
        self.dec1 = nn.Sequential(nn.Linear(dec_dim, dec_dim), nn.SiLU(), nn.Dropout(dropout))
        self.dec2 = nn.Sequential(nn.Linear(dec_dim, dec_dim), nn.SiLU(), nn.Dropout(dropout))
        self.dec3 = nn.Sequential(nn.Linear(dec_dim, dec_dim), nn.SiLU(), nn.Dropout(dropout))

        self.head1 = nn.Linear(dec_dim, output_num[0])
        self.head2 = nn.Linear(dec_dim, output_num[1])
        self.head3 = nn.Linear(dec_dim, output_num[2])

        # Reconstruction
        self.recon_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: [N, input_dim] gene expression (log1p normalized)
            edge_index: [2, E] spatial KNN graph
        Returns:
            recon: [N, input_dim]
            logits: [logits_class, logits_subclass, logits_supertype]
            z: [N, latent_dim]
        """
        # Encode gene expression
        h = self.gene_encoder(x)

        # GCN message passing with residual connections
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_new = gcn(h, edge_index)
            h = norm(h + h_new)  # residual + norm
            h = F.silu(h)
            h = self.gcn_dropout(h)

        # Project to latent
        z = self.to_latent(h)

        # Hierarchical decoding with feature residuals
        c1 = self.dec1(z)
        logits1 = self.head1(c1)

        c2 = self.dec2(z) + c1
        logits2 = self.head2(c2)

        c3 = self.dec3(z) + c2
        logits3 = self.head3(c3)

        # Reconstruction
        recon = self.recon_decoder(z)

        return recon, [logits1, logits2, logits3], z
