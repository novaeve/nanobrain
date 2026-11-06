"""
HEIST-inspired hierarchical graph encoder for cell-type annotation.

Reimplements the core HEIST architecture (multi-level GINConv +
TransformerConv + cross-message-passing) in pure PyTorch -- no
torch_geometric dependency.

Pipeline per forward call:
  X [B, 140], spatial [B, 2], batch_id [B]
  -> build k-NN spatial graph (high-level: cell-cell)
  -> build per-cell gene co-expression graph (low-level: gene-gene)
  -> sinusoidal positional encoding on both levels
  -> N x MultiLevelGraphLayer (GIN + MHA on high, TransformerConv on low,
     cross-message-passing between levels)
  -> pool low-level gene embeddings per cell -> concat with high-level
  -> 3 x classification heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Pure-PyTorch graph convolution layers (no torch_geometric)
# ---------------------------------------------------------------------------

class GINConvPure(nn.Module):
    """GIN convolution: h_v' = MLP((1+eps)*h_v + sum_{u in N(v)} h_u)."""

    def __init__(self, in_dim, out_dim, train_eps=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        eps_init = 0.0
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps_init))
        else:
            self.register_buffer("eps", torch.tensor(eps_init))

    def forward(self, x, edge_index):
        src, dst = edge_index  # src -> dst message
        N = x.size(0)
        agg = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, x[src])
        out = (1 + self.eps) * x + agg
        return self.mlp(out)


class TransformerConvPure(nn.Module):
    """Simplified graph transformer convolution with multi-head attention."""

    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        assert self.head_dim * heads == out_dim
        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index
        Q = self.q_proj(x)[dst]  # queries at destination nodes
        K = self.k_proj(x)[src]  # keys at source nodes
        V = self.v_proj(x)[src]  # values at source nodes

        N = x.size(0)
        # Reshape for multi-head: [E, heads, head_dim]
        Q = Q.view(-1, self.heads, self.head_dim)
        K = K.view(-1, self.heads, self.head_dim)
        V = V.view(-1, self.heads, self.head_dim)

        # Attention scores per edge
        attn = (Q * K).sum(dim=-1) / self.scale  # [E, heads]

        # Softmax per destination node (scatter softmax)
        attn_max = torch.zeros(N, self.heads, device=x.device, dtype=x.dtype)
        attn_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(attn), attn, reduce="amax", include_self=True)
        attn = attn - attn_max[dst]
        attn = attn.exp()

        attn_sum = torch.zeros(N, self.heads, device=x.device, dtype=x.dtype)
        attn_sum.index_add_(0, dst, attn)
        attn = attn / (attn_sum[dst] + 1e-8)

        # Weighted aggregation
        msg = attn.unsqueeze(-1) * V  # [E, heads, head_dim]
        msg = msg.view(-1, self.heads * self.head_dim)  # [E, out_dim]

        out = torch.zeros(N, self.heads * self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Cross-message-passing (HEIST-style)
# ---------------------------------------------------------------------------

class CrossMessagePassing(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Q = nn.Linear(d, d, bias=False)
        self.K = nn.Linear(d, d, bias=False)
        self.V = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d)

    def forward(self, to_emb, from_emb):
        q = self.Q(to_emb)
        k = self.K(from_emb)
        v = self.V(to_emb)
        weights = (q * k).sum(dim=-1, keepdim=True) / self.scale
        return weights * v


# ---------------------------------------------------------------------------
# Multi-level graph layer (mirrors HEIST MultiLevelGraphLayer)
# ---------------------------------------------------------------------------

class MultiLevelGraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, cross_mp=True):
        super().__init__()
        self.conv_high = GINConvPure(in_dim, out_dim)
        self.multi_head = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.mha_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.conv_low = TransformerConvPure(in_dim, out_dim, heads=num_heads)
        self.norm_in = nn.LayerNorm(in_dim)
        self.cross_mp = cross_mp
        if cross_mp:
            self.cross_lh = CrossMessagePassing(out_dim)
            self.cross_hl = CrossMessagePassing(out_dim)

    def forward(self, high_emb, high_edge_index, low_emb, low_edge_index,
                low_batch):
        """
        high_emb:        [N_cells, in_dim]
        high_edge_index: [2, E_high]
        low_emb:         [N_genes_total, in_dim]
        low_edge_index:  [2, E_low]
        low_batch:       [N_genes_total] mapping gene node -> cell index
        """
        h = self.norm_in(high_emb)
        l = self.norm_in(low_emb)

        # High-level: GIN + self-attention
        h_gin = self.conv_high(h, high_edge_index)
        h_unsq = h.unsqueeze(1)  # [N, 1, D] for MHA
        h_mha, _ = self.multi_head(h_unsq, h_unsq, h_unsq)
        h_mha = self.mha_proj(h_mha.squeeze(1))
        h_out = h_gin + h_mha

        # Low-level: TransformerConv
        l_out = self.conv_low(l, low_edge_index)

        if self.cross_mp:
            # low -> high: pool low per cell, cross-attend
            N_cells = high_emb.size(0)
            D = l_out.size(1)
            low_pool = torch.zeros(N_cells, D, device=l_out.device, dtype=l_out.dtype)
            low_pool.index_add_(0, low_batch, l_out)
            cnt = torch.zeros(N_cells, 1, device=l_out.device, dtype=l_out.dtype)
            cnt.index_add_(0, low_batch, torch.ones(low_batch.size(0), 1, device=l_out.device))
            low_pool = low_pool / cnt.clamp(min=1)

            h_out = self.cross_hl(h_out, low_pool)

            # high -> low: broadcast cell embedding to each gene node
            h_per_gene = h_out[low_batch]
            l_out = self.cross_lh(l_out, h_per_gene)

            return F.gelu(h_out), F.gelu(l_out)
        else:
            return F.gelu(h_out), F.gelu(l_out)


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (adapted from HEIST pe.py)
# ---------------------------------------------------------------------------

def spatial_sinusoidal_pe(coords, pe_dim):
    """
    2D sinusoidal PE for cell spatial coordinates.
    coords: [N, 2]  ->  pe: [N, pe_dim]
    """
    x, y = coords[:, 0], coords[:, 1]
    half = pe_dim // 2
    quarter = half // 2
    device = coords.device

    i = torch.arange(quarter, device=device, dtype=torch.float32)
    div = 10000.0 ** (4 * i / half)

    pe_x = torch.zeros(coords.size(0), half, device=device)
    pe_x[:, 0::2] = torch.sin(x.unsqueeze(-1) / div)
    pe_x[:, 1::2] = torch.cos(x.unsqueeze(-1) / div)

    pe_y = torch.zeros(coords.size(0), half, device=device)
    pe_y[:, 0::2] = torch.sin(y.unsqueeze(-1) / div)
    pe_y[:, 1::2] = torch.cos(y.unsqueeze(-1) / div)

    return torch.cat([pe_x, pe_y], dim=-1)


def gene_rank_pe(gene_expr, gene_batch, pe_dim, n_genes=None):
    """
    Rank-based sinusoidal PE for gene nodes within each cell (vectorized).
    gene_expr: [N_genes_total]
    gene_batch: [N_genes_total] cell index per gene node
    n_genes: number of genes per cell (if known, enables fast path)
    Returns: [N_genes_total, pe_dim]
    """
    device = gene_expr.device
    N = gene_expr.size(0)

    if n_genes is not None and N == (N // n_genes) * n_genes:
        # Fast path: all cells have same number of genes
        B = N // n_genes
        expr_2d = gene_expr.view(B, n_genes)
        ranks = torch.argsort(torch.argsort(-expr_2d, dim=1), dim=1)
        rank_norm = ranks.float() / max(n_genes - 1, 1)
        rank_norm = rank_norm.reshape(-1)
    else:
        rank_norm = torch.zeros(N, dtype=torch.float32, device=device)
        unique_cells = gene_batch.unique()
        for c in unique_cells:
            mask = gene_batch == c
            vals = gene_expr[mask]
            ranks = torch.argsort(torch.argsort(-vals))
            n = ranks.size(0)
            rn = ranks.float() / max(n - 1, 1)
            rank_norm[mask] = rn

    half = pe_dim // 2
    div = 10000.0 ** (2 * torch.arange(0, pe_dim, 2, device=device).float() / half)
    pe = torch.zeros(N, pe_dim, device=device)
    pe[:, 0::2] = torch.sin(rank_norm.unsqueeze(-1) / div)
    pe[:, 1::2] = torch.cos(rank_norm.unsqueeze(-1) / div)
    return pe


# ---------------------------------------------------------------------------
# Graph construction utilities
# ---------------------------------------------------------------------------

def build_knn_graph_batched(spatial, batch_id, k=10):
    """
    Build k-NN spatial graph per donor within a batch.
    Returns edge_index [2, E] on the same device as spatial.
    """
    device = spatial.device
    spatial_np = spatial.detach().cpu().numpy()
    batch_np = batch_id.detach().cpu().numpy()

    all_src, all_dst = [], []
    for donor in np.unique(batch_np):
        idx = np.where(batch_np == donor)[0]
        if len(idx) < 2:
            continue
        coords = spatial_np[idx]
        actual_k = min(k, len(idx) - 1)
        tree = cKDTree(coords)
        _, neighbors = tree.query(coords, k=actual_k + 1)
        neighbors = neighbors[:, 1:]  # drop self
        n = len(idx)
        src = np.repeat(idx, actual_k)
        dst = idx[neighbors.flatten()]
        all_src.append(src)
        all_dst.append(dst)

    if all_src:
        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
    else:
        src = np.array([], dtype=np.int64)
        dst = np.array([], dtype=np.int64)

    edge_index = torch.from_numpy(np.stack([src, dst])).long().to(device)
    return edge_index


def build_gene_graph_batched(X, batch_size, n_genes, top_k=5):
    """
    Build per-cell gene co-expression graphs (vectorized).

    For each cell, genes are nodes. Edges connect each gene to its top-k
    nearest neighbors in expression-value space (sorted-rank proximity).

    Returns:
        low_edge_index: [2, E_low] global gene-node indices
        low_batch:      [B * n_genes] mapping each gene node to its cell
        gene_features:  [B * n_genes, 1] gene expression values
    """
    device = X.device
    B, G = X.shape

    gene_features = X.reshape(B * G, 1)
    low_batch = torch.arange(B, device=device).unsqueeze(1).expand(B, G).reshape(-1)

    actual_k = min(top_k, G - 1)

    # Vectorized: sort genes per cell, connect each to its k nearest in rank
    X_cpu = X.detach().cpu().numpy()
    order = np.argsort(X_cpu, axis=1)  # [B, G] sorted gene indices per cell

    # For each gene at rank r, connect to genes at ranks [r-k, r+k] (excluding self)
    offsets = np.arange(B)[:, None] * G  # [B, 1]
    all_src = []
    all_dst = []
    for delta in range(-actual_k, actual_k + 1):
        if delta == 0:
            continue
        # Shift the rank array
        if delta > 0:
            src_ranks = np.arange(0, G - delta)
            dst_ranks = src_ranks + delta
        else:
            src_ranks = np.arange(-delta, G)
            dst_ranks = src_ranks + delta
        # Global gene indices
        src_genes = order[:, src_ranks] + offsets  # [B, len(src_ranks)]
        dst_genes = order[:, dst_ranks] + offsets
        all_src.append(src_genes.ravel())
        all_dst.append(dst_genes.ravel())

    if all_src:
        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        low_edge_index = torch.from_numpy(np.stack([src, dst])).long().to(device)
    else:
        low_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return low_edge_index, low_batch, gene_features


# ---------------------------------------------------------------------------
# Classification head (same pattern as Nicheformer)
# ---------------------------------------------------------------------------

class ClsDecoder(nn.Module):
    def __init__(self, d_model, n_cls, nlayers=3):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers += [nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU()]
        layers.append(nn.Linear(d_model, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HEISTEncoder(nn.Module):
    """
    Pure-PyTorch reimplementation of HEIST's GraphEncoder.

    Processes two graph levels:
      - High-level: cell-cell spatial graph (nodes=cells, features=spatial coords)
      - Low-level:  gene co-expression graph (nodes=genes, features=expression)

    Uses sinusoidal PE, multi-level GIN+Transformer layers, and
    cross-message-passing between levels.
    """

    def __init__(self, pe_dim=32, init_dim=64, hidden_dim=128,
                 output_dim=128, num_layers=3, num_heads=4,
                 cross_message_passing=True, positional_encoding=True):
        super().__init__()
        self.pe_dim = pe_dim
        self.positional_encoding = positional_encoding
        self.cross_mp = cross_message_passing

        high_in = 2 + pe_dim if positional_encoding else 2
        low_in = 1 + pe_dim if positional_encoding else 1

        self.mlp_high = nn.Sequential(nn.Linear(high_in, init_dim), nn.GELU())
        self.mlp_low = nn.Sequential(nn.Linear(low_in, init_dim), nn.GELU())

        self.convs = nn.ModuleList()
        self.convs.append(MultiLevelGraphLayer(init_dim, hidden_dim, num_heads, cross_message_passing))
        for _ in range(num_layers - 2):
            self.convs.append(MultiLevelGraphLayer(hidden_dim, hidden_dim, num_heads, cross_message_passing))
        self.convs.append(MultiLevelGraphLayer(hidden_dim, output_dim, num_heads, cross_message_passing))

        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, spatial, X, high_edge_index, low_edge_index, low_batch):
        """
        spatial:         [B, 2]
        X:               [B, G]
        high_edge_index: [2, E_high]
        low_edge_index:  [2, E_low]
        low_batch:       [B*G] gene-node -> cell mapping
        Returns: high_emb [B, output_dim]
        """
        B, G = X.shape
        device = X.device

        # Gene node features
        gene_features = X.reshape(B * G, 1)

        # Positional encoding
        if self.positional_encoding:
            high_pe = spatial_sinusoidal_pe(spatial, self.pe_dim)
            high_in = torch.cat([spatial, high_pe], dim=-1)
            low_pe = gene_rank_pe(gene_features.squeeze(-1), low_batch, self.pe_dim, n_genes=G)
            low_in = torch.cat([gene_features, low_pe], dim=-1)
        else:
            high_in = spatial
            low_in = gene_features

        high_emb = self.mlp_high(high_in)
        low_emb = self.mlp_low(low_in)

        # Multi-level graph layers with residual
        high_emb, low_emb = self.convs[0](
            high_emb, high_edge_index, low_emb, low_edge_index, low_batch
        )
        for layer in self.convs[1:]:
            h_new, l_new = layer(
                high_emb, high_edge_index, low_emb, low_edge_index, low_batch
            )
            high_emb = h_new + high_emb
            low_emb = l_new + low_emb

        return self.norm(high_emb)


class HEISTForAnnotation(nn.Module):
    """
    HEIST encoder + hierarchical classification heads for cell-type
    annotation on MERFISH spatial-transcriptomics data.

    Builds spatial k-NN and gene co-expression graphs on-the-fly from
    the flat tensors provided by our dataloader.

    Args:
        checkpoint_path: optional path to a pretrained HEIST checkpoint.
        output_num:      [n_class, n_subclass, n_supertype].
        dropout:         dropout rate (applied before cls heads).
        freeze_backbone: freeze encoder weights; train heads only.
        k_spatial:       k for spatial k-NN graph.
        k_gene:          k for gene co-expression graph.
        pe_dim:          positional encoding dimension.
        hidden_dim:      hidden dimension in graph layers.
        output_dim:      encoder output embedding dimension.
        num_layers:      number of multi-level graph layers.
        num_heads:       attention heads.
    """

    def __init__(
        self,
        checkpoint_path=None,
        output_num=None,
        dropout=0.2,
        freeze_backbone=False,
        k_spatial=10,
        k_gene=5,
        pe_dim=32,
        hidden_dim=128,
        output_dim=128,
        num_layers=3,
        num_heads=4,
        cross_message_passing=True,
        positional_encoding=True,
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self.k_spatial = k_spatial
        self.k_gene = k_gene
        self.output_dim = output_dim

        self.encoder = HEISTEncoder(
            pe_dim=pe_dim,
            init_dim=hidden_dim // 2,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            cross_message_passing=cross_message_passing,
            positional_encoding=positional_encoding,
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Support both raw state_dict and wrapped checkpoint formats
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            print(
                f"[HEIST] Loaded checkpoint from {checkpoint_path}\n"
                f"        Missing: {len(missing)}  Unexpected: {len(unexpected)}"
            )
        else:
            print("[HEIST] No checkpoint provided -- training from scratch.")

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[HEIST] Encoder weights frozen.")

        self.freeze_backbone = freeze_backbone
        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.cls_head_class = ClsDecoder(output_dim, output_num[0])
        self.cls_head_subclass = ClsDecoder(output_dim, output_num[1])
        self.cls_head_supertype = ClsDecoder(output_dim, output_num[2])

    def forward(self, X, spatial=None, batch_id=None):
        """
        Args:
            X:        [B, 140] raw gene expression
            spatial:  [B, 2]   spatial coordinates
            batch_id: [B]      donor/batch index per cell

        Returns:
            logits:   list of [(B, 3), (B, 24), (B, 137)]
            cell_emb: (B, output_dim)
        """
        B, G = X.shape
        device = X.device

        # Default spatial / batch_id if not provided
        if spatial is None:
            spatial = torch.zeros(B, 2, device=device)
        if batch_id is None:
            batch_id = torch.zeros(B, dtype=torch.long, device=device)

        # Build graphs on-the-fly
        high_edge_index = build_knn_graph_batched(spatial, batch_id, k=self.k_spatial)
        low_edge_index, low_batch, _ = build_gene_graph_batched(
            X, B, G, top_k=self.k_gene
        )

        # Encode
        if self.freeze_backbone:
            with torch.no_grad():
                cell_emb = self.encoder(spatial, X, high_edge_index, low_edge_index, low_batch)
        else:
            cell_emb = self.encoder(spatial, X, high_edge_index, low_edge_index, low_batch)

        cell_emb_drop = self.dropout(cell_emb)

        # Classification
        logit_c = self.cls_head_class(cell_emb_drop)
        logit_sc = self.cls_head_subclass(cell_emb_drop)
        logit_st = self.cls_head_supertype(cell_emb_drop)

        return [logit_c, logit_sc, logit_st], cell_emb
