"""
mjm_2a/b/c: three variants that inject brain-specific spatial/morphological context
into the mjm_1 hierarchical encoder-decoder.

All variants preserve the EXACT mjm_1 hierarchical logic:
  X -> E1->h1 -> E2->h2 -> E3->H
       D1->C1   D2->C2+C1  D3->C3+C2  + ReconDecoder

mjm_2a: Laminar Depth Injection
  - DepthMLP(depth_imputed[B,1]) -> depth_feat[B,dec_hidden_dim]
  - Injected as residual into C2 (Subclass decoder), after C1 residual
  - Auxiliary depth regression head on h2

mjm_2b: Morpho-Spatial Fusion
  - Input = concat(X[B,140], cell_volume_norm[B,1], x_tiled[B,1], y_tiled[B,1]) -> [B,143]
  - Linear(143->140) projects back to gene-expression space before E1
  - Auxiliary volume regression head on H (latent)

mjm_2c: CPS-Conditioned FiLM Modulation
  - ContextMLP([cps, x_tiled, y_tiled] -> [B,3]) -> gamma, beta [B, enc_hidden_dim]
  - FiLM modulates h1 (E1 output): h1_mod = gamma * h1 + beta
  - Everything downstream (E2, E3, D1/D2/D3) sees disease-space-conditioned features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Re-use building blocks from mjm.py ─────────────────────────────────────
from src.models.mjm import RMSNorm, SwiGLU, FFN, _make_dec_block


# ────────────────────────────────────────────────────────────────────────────
# Plan A: Laminar Depth Injection
# ────────────────────────────────────────────────────────────────────────────
class mjm_2a(nn.Module):
    """
    mjm_1 architecture + depth context injected at the Subclass decoder level.

    depth_imputed [B,1] -> DepthMLP -> depth_feat [B, dec_hidden_dim]
    C2 = D2(h2) + C1 + depth_feat          (depth residual at Subclass level)
    C3 = D3(H)  + C2                       (Supertype still gets depth via C2)

    Auxiliary head: depth_head(h2) -> scalar  (only supervised on has_depth cells)
    """
    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 depth_mlp_dim=64,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 residual_mode='feature',
                 ):
        super().__init__()
        assert residual_mode == 'feature', "mjm_2a only supports feature residual mode"
        self.residual_mode = residual_mode

        # ── Encoders (identical to mjm_1) ───────────────────────────────────
        self.E1 = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
        )
        self.E2 = FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                      expansion_factor=expansion_factor, dropout=dropout)
        self.E3 = nn.Sequential(
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        # ── Decoders (identical to mjm_1) ───────────────────────────────────
        self.D1 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D2 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D3 = _make_dec_block(latent_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)

        # ── Classification heads (identical to mjm_1) ───────────────────────
        self.head1 = nn.Linear(dec_hidden_dim, output_num[0])
        self.head2 = nn.Linear(dec_hidden_dim, output_num[1])
        self.head3 = nn.Linear(dec_hidden_dim, output_num[2])

        # ── Reconstruction decoder (identical to mjm_1) ─────────────────────
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # ── NEW: Depth context branch ────────────────────────────────────────
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, depth_mlp_dim),
            nn.SiLU(),
            nn.Linear(depth_mlp_dim, dec_hidden_dim),
        )
        # Auxiliary depth regression head (applied on h2)
        self.depth_head = nn.Linear(enc_hidden_dim, 1)

    def forward(self, x, depth_imputed):
        """
        Args:
            x:             [B, input_dim] gene expression (log1p normalized)
            depth_imputed: [B, 1] imputed normalized depth from pia
        Returns:
            recon, [logits1, logits2, logits3], H, depth_pred
        """
        # ── Encoding ────────────────────────────────────────────────────────
        h1 = self.E1(x)    # [B, enc_hidden_dim]
        h2 = self.E2(h1)   # [B, enc_hidden_dim]
        H  = self.E3(h2)   # [B, latent_dim]

        # ── Depth context ────────────────────────────────────────────────────
        depth_feat = self.depth_mlp(depth_imputed)  # [B, dec_hidden_dim]

        # ── Hierarchical decoding + classification ───────────────────────────
        C1      = self.D1(h1)           # [B, dec_hidden_dim]
        logits1 = self.head1(C1)        # [B, n1]

        # C2: mjm_1 feature residual + depth injection at Subclass level
        C2      = self.D2(h2) + C1 + depth_feat   # [B, dec_hidden_dim]
        logits2 = self.head2(C2)                   # [B, n2]

        # C3: standard feature residual (depth flows through C2)
        C3      = self.D3(H) + C2      # [B, dec_hidden_dim]
        logits3 = self.head3(C3)       # [B, n3]

        # ── Reconstruction ───────────────────────────────────────────────────
        recon = self.recon_decoder(H)  # [B, input_dim]

        # ── Auxiliary depth prediction ───────────────────────────────────────
        depth_pred = self.depth_head(h2)  # [B, 1]

        return recon, [logits1, logits2, logits3], H, depth_pred


# ────────────────────────────────────────────────────────────────────────────
# Plan B: Morpho-Spatial Fusion
# ────────────────────────────────────────────────────────────────────────────
class mjm_2b(nn.Module):
    """
    mjm_1 + cell morphology (volume) and spatial tiled coords fused at input.

    Input fusion:
        [X(140), cell_volume_norm(1), x_tiled_norm(1), y_tiled_norm(1)] -> [B,143]
        Linear(143->140) -> X_proj [B,140]
        -> E1->E2->E3 (identical to mjm_1)

    Auxiliary volume regression head on H (latent):
        volume_head(H) -> [B,1], only supervised on has_volume cells

    has_volume mask ensures 80.5% cells without volume still train normally
    through the spatial tiled coordinates.
    """
    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 residual_mode='feature',
                 ):
        super().__init__()
        assert residual_mode == 'feature'
        self.residual_mode = residual_mode

        # ── NEW: Input fusion projection ─────────────────────────────────────
        # X(140) + volume(1) + x_tiled(1) + y_tiled(1) = 143
        self.input_proj = nn.Linear(input_dim + 3, input_dim)

        # ── Encoders (identical to mjm_1) ───────────────────────────────────
        self.E1 = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
        )
        self.E2 = FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                      expansion_factor=expansion_factor, dropout=dropout)
        self.E3 = nn.Sequential(
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        # ── Decoders (identical to mjm_1) ───────────────────────────────────
        self.D1 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D2 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D3 = _make_dec_block(latent_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)

        # ── Classification heads (identical to mjm_1) ───────────────────────
        self.head1 = nn.Linear(dec_hidden_dim, output_num[0])
        self.head2 = nn.Linear(dec_hidden_dim, output_num[1])
        self.head3 = nn.Linear(dec_hidden_dim, output_num[2])

        # ── Reconstruction decoder (identical to mjm_1) ─────────────────────
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # ── NEW: Auxiliary volume regression head ────────────────────────────
        self.volume_head = nn.Linear(latent_dim, 1)

    def forward(self, x, cell_volume_norm, spatial_tiled_norm):
        """
        Args:
            x:                  [B, 140] gene expression (log1p)
            cell_volume_norm:   [B, 1]   z-score normalized volume (0 if missing)
            spatial_tiled_norm: [B, 2]   normalized tiled spatial coords
        Returns:
            recon, [logits1, logits2, logits3], H, vol_pred
        """
        # ── Input fusion ────────────────────────────────────────────────────
        x_aug  = torch.cat([x, cell_volume_norm, spatial_tiled_norm], dim=-1)  # [B,143]
        x_proj = self.input_proj(x_aug)  # [B,140]

        # ── Encoding ────────────────────────────────────────────────────────
        h1 = self.E1(x_proj)  # [B, enc_hidden_dim]
        h2 = self.E2(h1)      # [B, enc_hidden_dim]
        H  = self.E3(h2)      # [B, latent_dim]

        # ── Hierarchical decoding (identical to mjm_1 feature mode) ─────────
        C1      = self.D1(h1)
        logits1 = self.head1(C1)

        C2      = self.D2(h2) + C1
        logits2 = self.head2(C2)

        C3      = self.D3(H) + C2
        logits3 = self.head3(C3)

        recon = self.recon_decoder(H)

        # ── Auxiliary volume prediction ──────────────────────────────────────
        vol_pred = self.volume_head(H)  # [B, 1]

        return recon, [logits1, logits2, logits3], H, vol_pred


# ────────────────────────────────────────────────────────────────────────────
# Plan C: CPS-Conditioned FiLM Modulation
# ────────────────────────────────────────────────────────────────────────────
class mjm_2c(nn.Module):
    """
    mjm_1 + FiLM modulation of h1 conditioned on CPS and spatial_tiled.

    Context = [cps(1), x_tiled_norm(1), y_tiled_norm(1)] -> [B,3]
    ContextMLP(3->film_hidden->enc_hidden_dim*2) -> gamma[B,d], beta[B,d]
    h1_mod = gamma * h1 + beta

    Everything downstream (E2->E3->D1/D2/D3) sees disease-and-space-conditioned
    features. CPS is available for ALL 1.9M cells, so no cells are wasted.

    No auxiliary loss needed — FiLM supervision comes from the classification
    and reconstruction losses backpropagating through h1_mod.
    """
    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 film_hidden=64,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 residual_mode='feature',
                 ):
        super().__init__()
        assert residual_mode == 'feature'
        self.residual_mode = residual_mode
        self.enc_hidden_dim = enc_hidden_dim

        # ── Encoders (identical to mjm_1) ───────────────────────────────────
        self.E1 = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
        )
        self.E2 = FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                      expansion_factor=expansion_factor, dropout=dropout)
        self.E3 = nn.Sequential(
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        # ── Decoders (identical to mjm_1) ───────────────────────────────────
        self.D1 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D2 = _make_dec_block(enc_hidden_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)
        self.D3 = _make_dec_block(latent_dim, dec_hidden_dim,
                                  d_layers, expansion_factor, dropout)

        # ── Classification heads (identical to mjm_1) ───────────────────────
        self.head1 = nn.Linear(dec_hidden_dim, output_num[0])
        self.head2 = nn.Linear(dec_hidden_dim, output_num[1])
        self.head3 = nn.Linear(dec_hidden_dim, output_num[2])

        # ── Reconstruction decoder (identical to mjm_1) ─────────────────────
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # ── NEW: FiLM context network ────────────────────────────────────────
        # Input: [cps, x_tiled_norm, y_tiled_norm] -> 3 dims
        self.film_net = nn.Sequential(
            nn.Linear(3, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, enc_hidden_dim * 2),  # -> gamma + beta
        )

    def forward(self, x, cps, spatial_tiled_norm):
        """
        Args:
            x:                  [B, 140] gene expression (log1p)
            cps:                [B, 1]   continuous pseudo-progression score
            spatial_tiled_norm: [B, 2]   normalized tiled spatial coords
        Returns:
            recon, [logits1, logits2, logits3], H
        """
        # ── E1: initial encoding ─────────────────────────────────────────────
        h1 = self.E1(x)  # [B, enc_hidden_dim]

        # ── FiLM modulation on h1 ────────────────────────────────────────────
        context = torch.cat([cps, spatial_tiled_norm], dim=-1)  # [B, 3]
        film_params = self.film_net(context)                     # [B, enc_hidden_dim*2]
        gamma, beta = film_params.chunk(2, dim=-1)               # each [B, enc_hidden_dim]
        # scale gamma around 1 (like layer norm scale), beta is bias
        h1_mod = (1.0 + gamma) * h1 + beta                      # [B, enc_hidden_dim]

        # ── E2, E3 ───────────────────────────────────────────────────────────
        h2 = self.E2(h1_mod)  # [B, enc_hidden_dim]
        H  = self.E3(h2)      # [B, latent_dim]

        # ── Hierarchical decoding (identical to mjm_1 feature mode) ─────────
        C1      = self.D1(h1_mod)
        logits1 = self.head1(C1)

        C2      = self.D2(h2) + C1
        logits2 = self.head2(C2)

        C3      = self.D3(H) + C2
        logits3 = self.head3(C3)

        recon = self.recon_decoder(H)

        return recon, [logits1, logits2, logits3], H
