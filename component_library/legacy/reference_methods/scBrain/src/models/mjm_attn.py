"""
mjm_attn: Transformer-encoder MJM with linear attention + 2D spatial RoPE.

Architecture (based on mjm_1_lr cascaded structure):
  X [B,140] → input_proj → h [B,256]
  E1: SpatialTransformerBlock × e_layers  → h1
  E2: SpatialTransformerBlock × e_layers  → h2
  E3: SpatialTransformerBlock × e_layers + Linear → H [B,20]
  LR heads: h1→class, h2→subclass, H→supertype
  Recon: SwiGLU(H) → x_hat

Attention operates on the cell dimension (cells = tokens).
2D RoPE encodes spatial (x, y) coordinates.
Donor mask ensures cells only attend within the same donor.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mjm import RMSNorm, SwiGLU


# ── 2D Rotary Position Encoding ──────────────────────────────────────────────

class SpatialRoPE2D(nn.Module):
    """2D RoPE: split dim in half, apply 1D RoPE(x) to first half, RoPE(y) to second."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"
        self.half_dim = dim // 2
        quarter = dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, spatial):
        """
        Args:
            spatial: [B, 2] — (x, y) coordinates
        Returns:
            cos, sin: [B, half_dim] each, to be applied per-head
        """
        x_pos = spatial[:, 0:1]  # [B, 1]
        y_pos = spatial[:, 1:2]  # [B, 1]

        # [B, quarter]
        x_angles = x_pos * self.inv_freq.unsqueeze(0)
        y_angles = y_pos * self.inv_freq.unsqueeze(0)

        # Concat x and y angles → [B, half_dim]
        angles = torch.cat([x_angles, y_angles], dim=-1)
        return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE rotation to tensor x [B, n_heads, head_dim].
    cos, sin: [B, half_dim] where half_dim = head_dim.
    """
    # x shape: [B, n_heads, head_dim]
    # cos/sin shape: [B, half_dim] → [B, 1, half_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # Split into pairs for rotation
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    cos_half = cos[..., :x1.shape[-1]]
    sin_half = sin[..., :x1.shape[-1]]
    out1 = x1 * cos_half - x2 * sin_half
    out2 = x1 * sin_half + x2 * cos_half
    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ── Linear Attention with Donor Mask ─────────────────────────────────────────

class LinearAttention(nn.Module):
    """Multi-head linear attention with 2D RoPE and donor masking.

    Linear attention: φ(Q) @ (φ(K)^T @ V) instead of softmax(QK^T) @ V.
    Feature map φ(x) = elu(x) + 1.
    Complexity: O(B · d²) instead of O(B² · d).

    With donor mask, we fall back to masked quadratic attention per donor group,
    but since each donor has ~B/n_donors tokens, this is still efficient.
    """

    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _feature_map(self, x):
        """φ(x) = elu(x) + 1, ensures non-negative."""
        return F.elu(x) + 1.0

    def forward(self, x, cos, sin, donor_mask=None):
        """
        Args:
            x: [B, dim]
            cos, sin: [B, half_dim] from SpatialRoPE2D
            donor_mask: [B, B] bool, True where cells share donor
        Returns:
            [B, dim]
        """
        B, D = x.shape

        # Project and reshape to [B, n_heads, head_dim]
        Q = self.q_proj(x).view(B, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(B, self.n_heads, self.head_dim)

        # Apply 2D RoPE to Q, K
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        if donor_mask is not None:
            # Masked attention (quadratic but sparse via donor groups)
            # [B, n_heads, B] = [B, n_heads, head_dim] @ [B, n_heads, head_dim]^T
            phi_Q = self._feature_map(Q)
            phi_K = self._feature_map(K)

            # Attention scores: [B, n_heads, B]
            attn = torch.einsum('bnh,cnh->bnc', phi_Q, phi_K)

            # Apply donor mask: [B, B] → [B, 1, B]
            mask = donor_mask.unsqueeze(1)  # [B, 1, B]
            attn = attn.masked_fill(~mask, 0.0)

            # Normalize
            attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            attn = attn / attn_sum
            attn = self.dropout(attn)

            # Weighted sum: [B, n_heads, head_dim]
            out = torch.einsum('bnc,cnh->bnh', attn, V)
        else:
            # Pure linear attention (no mask): O(B·d²)
            phi_Q = self._feature_map(Q)
            phi_K = self._feature_map(K)
            # KV: [n_heads, head_dim, head_dim]
            KV = torch.einsum('bnh,bnk->nhk', phi_K, V)
            # Numerator: [B, n_heads, head_dim]
            out = torch.einsum('bnh,nhk->bnk', phi_Q, KV)
            # Denominator: [B, n_heads, 1]
            Z = torch.einsum('bnh,nh->bn', phi_Q,
                             phi_K.sum(dim=0)).unsqueeze(-1).clamp(min=1e-6)
            out = out / Z

        out = out.reshape(B, D)
        return self.out_proj(self.dropout(out))


# ── Spatial Transformer Block ────────────────────────────────────────────────

class SpatialTransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → LinearAttn → residual → RMSNorm → SwiGLU → residual."""

    def __init__(self, dim, n_heads=8, expansion_factor=2.67, dropout=0.3):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = LinearAttention(dim, n_heads=n_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim, expansion_factor=expansion_factor, dropout=dropout)

    def forward(self, x, cos, sin, donor_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, donor_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ── mjm_attn: Transformer Encoder + LR Heads ────────────────────────────────

class mjm_attn(nn.Module):
    """
    MJM with Transformer encoder (linear attention + 2D spatial RoPE).

    Three-stage cascaded encoder (like mjm_1), each stage is a stack of
    SpatialTransformerBlocks. LR classification heads (like mjm_1_lr).

    Forward: (X, spatial, batch_id) → (recon, [logits1, logits2, logits3], H)
    """

    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 n_heads=8,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, enc_hidden_dim)

        # 2D RoPE
        self.rope = SpatialRoPE2D(enc_hidden_dim // n_heads)

        # E1: TransformerBlock × e_layers
        self.E1 = nn.ModuleList([
            SpatialTransformerBlock(enc_hidden_dim, n_heads, expansion_factor, dropout)
            for _ in range(e_layers)
        ])

        # E2: TransformerBlock × e_layers
        self.E2 = nn.ModuleList([
            SpatialTransformerBlock(enc_hidden_dim, n_heads, expansion_factor, dropout)
            for _ in range(e_layers)
        ])

        # E3: TransformerBlock × e_layers + latent projection
        self.E3 = nn.ModuleList([
            SpatialTransformerBlock(enc_hidden_dim, n_heads, expansion_factor, dropout)
            for _ in range(e_layers)
        ])
        self.latent_proj = nn.Linear(enc_hidden_dim, latent_dim)

        # Reconstruction decoder
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR classification heads
        self.head1 = nn.Linear(enc_hidden_dim, output_num[0])   # h1 → class
        self.head2 = nn.Linear(enc_hidden_dim, output_num[1])   # h2 → subclass
        self.head3 = nn.Linear(latent_dim, output_num[2])       # H  → supertype

    def forward(self, x, spatial, batch_id):
        """
        Args:
            x:        [B, input_dim] gene expression (log1p normalized)
            spatial:  [B, 2] spatial coordinates (x, y)
            batch_id: [B] donor IDs (for donor mask)
        """
        B = x.shape[0]

        # Input projection
        h = self.input_proj(x)  # [B, enc_hidden_dim]

        # 2D RoPE from spatial coords
        cos, sin = self.rope(spatial)

        # Donor mask: [B, B] bool
        donor_mask = (batch_id.unsqueeze(0) == batch_id.unsqueeze(1))

        # E1
        for block in self.E1:
            h = block(h, cos, sin, donor_mask)
        h1 = h

        # E2
        for block in self.E2:
            h = block(h, cos, sin, donor_mask)
        h2 = h

        # E3
        for block in self.E3:
            h = block(h, cos, sin, donor_mask)
        H = self.latent_proj(h)  # [B, latent_dim]

        # Classification
        logits1 = self.head1(h1)
        logits2 = self.head2(h2)
        logits3 = self.head3(H)

        # Reconstruction
        recon = self.recon_decoder(H)

        return recon, [logits1, logits2, logits3], H


# ── mjm_attn_dec: Transformer Encoder + Nonlinear Decoder Heads ──────────────

class mjm_attn_dec(nn.Module):
    """
    Same Transformer encoder as mjm_attn, but with mjm_1-style nonlinear
    decoder blocks (Linear+FFN) + feature residual + Linear classification heads.

    Comparison: mjm_attn (LR heads) vs mjm_attn_dec (decoder heads).
    """

    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 n_heads=8,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 ):
        super().__init__()
        from src.models.mjm import _make_dec_block

        # ── Encoder (identical to mjm_attn) ──────────────────────────────────
        self.input_proj = nn.Linear(input_dim, enc_hidden_dim)
        self.rope = SpatialRoPE2D(enc_hidden_dim // n_heads)

        self.E1 = nn.ModuleList([
            SpatialTransformerBlock(enc_hidden_dim, n_heads, expansion_factor, dropout)
            for _ in range(e_layers)
        ])
        self.E2 = nn.ModuleList([
            SpatialTransformerBlock(enc_hidden_dim, n_heads, expansion_factor, dropout)
            for _ in range(e_layers)
        ])
        self.E3 = nn.ModuleList([
            SpatialTransformerBlock(enc_hidden_dim, n_heads, expansion_factor, dropout)
            for _ in range(e_layers)
        ])
        self.latent_proj = nn.Linear(enc_hidden_dim, latent_dim)

        # ── Decoder blocks (from mjm_1, feature residual) ───────────────────
        self.D1 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D2 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D3 = _make_dec_block(latent_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)

        # ── Classification heads ─────────────────────────────────────────────
        self.head1 = nn.Linear(dec_hidden_dim, output_num[0])
        self.head2 = nn.Linear(dec_hidden_dim, output_num[1])
        self.head3 = nn.Linear(dec_hidden_dim, output_num[2])

        # ── Reconstruction decoder ───────────────────────────────────────────
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

    def forward(self, x, spatial, batch_id):
        B = x.shape[0]
        h = self.input_proj(x)
        cos, sin = self.rope(spatial)
        donor_mask = (batch_id.unsqueeze(0) == batch_id.unsqueeze(1))

        for block in self.E1:
            h = block(h, cos, sin, donor_mask)
        h1 = h

        for block in self.E2:
            h = block(h, cos, sin, donor_mask)
        h2 = h

        for block in self.E3:
            h = block(h, cos, sin, donor_mask)
        H = self.latent_proj(h)

        # Hierarchical decoding with feature residual
        C1      = self.D1(h1)
        logits1 = self.head1(C1)

        C2      = self.D2(h2) + C1
        logits2 = self.head2(C2)

        C3      = self.D3(H) + C2
        logits3 = self.head3(C3)

        recon = self.recon_decoder(H)
        return recon, [logits1, logits2, logits3], H


# ── mjm_attn_nlevel: N-level Transformer Encoder + LR Heads ───────────────

class mjm_attn_nlevel(nn.Module):
    """
    N-level generalization of mjm_attn.

    N-stage cascaded transformer encoder with spatial RoPE + donor mask.
    LR classification heads from each stage output.
    Forward: (X, spatial, batch_id) -> (recon, logits_list, H)
    """

    def __init__(self,
                 input_dim=266,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 n_heads=8,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=None,
                 ):
        super().__init__()
        if output_num is None:
            output_num = [13, 14, 25, 27]
        assert len(output_num) >= 2
        self.n_levels = len(output_num)
        self.output_num = output_num

        self.input_proj = nn.Linear(input_dim, enc_hidden_dim)
        self.rope = SpatialRoPE2D(enc_hidden_dim // n_heads)

        # N encoder stages
        self.stages = nn.ModuleList()
        for i in range(self.n_levels):
            self.stages.append(nn.ModuleList([
                SpatialTransformerBlock(enc_hidden_dim, n_heads, expansion_factor, dropout)
                for _ in range(e_layers)
            ]))

        # Last stage projects to latent
        self.latent_proj = nn.Linear(enc_hidden_dim, latent_dim)

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR heads
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim if i == self.n_levels - 1 else enc_hidden_dim,
                      n_cls)
            for i, n_cls in enumerate(output_num)
        ])

    def forward(self, x, spatial, batch_id):
        h = self.input_proj(x)
        cos, sin = self.rope(spatial)
        donor_mask = (batch_id.unsqueeze(0) == batch_id.unsqueeze(1))

        h_list = []
        for i, stage in enumerate(self.stages):
            for block in stage:
                h = block(h, cos, sin, donor_mask)
            if i == self.n_levels - 1:
                h_list.append(self.latent_proj(h))
            else:
                h_list.append(h)

        H = h_list[-1]
        recon = self.recon_decoder(H)
        logits_list = [self.heads[i](h_list[i]) for i in range(self.n_levels)]
        return recon, logits_list, H
