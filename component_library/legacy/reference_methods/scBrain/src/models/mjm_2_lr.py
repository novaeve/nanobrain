"""
mjm_2a_lr / mjm_2b_lr / mjm_2c_lr: LR-head variants of mjm_2 models.

Same encoder + context injection as originals, but classification heads are
single Linear layers instead of SwiGLU/FFN decoder blocks + Linear heads.
"""
import torch
import torch.nn as nn

from src.models.mjm import FFN, SwiGLU


class mjm_2a_lr(nn.Module):
    """mjm_2a with LR heads. Depth auxiliary head kept, decoder injection removed."""

    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=128, depth_mlp_dim=64,
                 residual_mode='feature',
                 ):
        super().__init__()

        # Encoders (identical to mjm_2a)
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

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR heads
        self.head1 = nn.Linear(enc_hidden_dim, output_num[0])
        self.head2 = nn.Linear(enc_hidden_dim, output_num[1])
        self.head3 = nn.Linear(latent_dim, output_num[2])

        # Auxiliary depth head (kept from mjm_2a)
        self.depth_head = nn.Linear(enc_hidden_dim, 1)

    def forward(self, x, depth_imputed):
        h1 = self.E1(x)
        h2 = self.E2(h1)
        H  = self.E3(h2)

        logits1 = self.head1(h1)
        logits2 = self.head2(h2)
        logits3 = self.head3(H)

        recon = self.recon_decoder(H)
        depth_pred = self.depth_head(h2)
        return recon, [logits1, logits2, logits3], H, depth_pred


class mjm_2b_lr(nn.Module):
    """mjm_2b with LR heads. Input fusion + volume auxiliary kept."""

    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=128, residual_mode='feature',
                 ):
        super().__init__()

        # Input fusion (identical to mjm_2b)
        self.input_proj = nn.Linear(input_dim + 3, input_dim)

        # Encoders
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

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR heads
        self.head1 = nn.Linear(enc_hidden_dim, output_num[0])
        self.head2 = nn.Linear(enc_hidden_dim, output_num[1])
        self.head3 = nn.Linear(latent_dim, output_num[2])

        # Auxiliary volume head (kept from mjm_2b)
        self.volume_head = nn.Linear(latent_dim, 1)

    def forward(self, x, cell_volume_norm, spatial_tiled_norm):
        x_aug  = torch.cat([x, cell_volume_norm, spatial_tiled_norm], dim=-1)
        x_proj = self.input_proj(x_aug)

        h1 = self.E1(x_proj)
        h2 = self.E2(h1)
        H  = self.E3(h2)

        logits1 = self.head1(h1)
        logits2 = self.head2(h2)
        logits3 = self.head3(H)

        recon = self.recon_decoder(H)
        vol_pred = self.volume_head(H)
        return recon, [logits1, logits2, logits3], H, vol_pred


class mjm_2c_lr(nn.Module):
    """mjm_2c with LR heads. FiLM modulation on h1 kept."""

    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 film_hidden=64,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=128, residual_mode='feature',
                 ):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim

        # Encoders
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

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR heads
        self.head1 = nn.Linear(enc_hidden_dim, output_num[0])
        self.head2 = nn.Linear(enc_hidden_dim, output_num[1])
        self.head3 = nn.Linear(latent_dim, output_num[2])

        # FiLM context network (kept from mjm_2c)
        self.film_net = nn.Sequential(
            nn.Linear(3, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, enc_hidden_dim * 2),
        )

    def forward(self, x, cps, spatial_tiled_norm):
        h1 = self.E1(x)

        # FiLM modulation
        context = torch.cat([cps, spatial_tiled_norm], dim=-1)
        film_params = self.film_net(context)
        gamma, beta = film_params.chunk(2, dim=-1)
        h1_mod = (1.0 + gamma) * h1 + beta

        h2 = self.E2(h1_mod)
        H  = self.E3(h2)

        logits1 = self.head1(h1_mod)
        logits2 = self.head2(h2)
        logits3 = self.head3(H)

        recon = self.recon_decoder(H)
        return recon, [logits1, logits2, logits3], H
