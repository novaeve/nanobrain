"""
mjm_lr / mjm_1_lr: LR-head variants of mjm / mjm_1.

Identical encoder and reconstruction decoder, but classification heads are
single Linear layers (logistic regression under softmax + cross-entropy)
instead of SwiGLU/FFN decoder blocks + Linear heads.
"""
import torch
import torch.nn as nn

from src.models.mjm import FFN, SwiGLU


class mjm_lr(nn.Module):
    """mjm with LR classification heads: z → Linear(latent_dim, n_classes)."""

    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=256, is_hierarchical=True,
                 ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        self.recon_decoder = SwiGLU(latent_dim, input_dim)

        # LR heads: direct linear from latent
        self.head1 = nn.Linear(latent_dim, output_num[0])
        self.head2 = nn.Linear(latent_dim, output_num[1])
        self.head3 = nn.Linear(latent_dim, output_num[2])

    def forward(self, x):
        z = self.encoder(x)
        recon = self.recon_decoder(z)
        logits1 = self.head1(z)
        logits2 = self.head2(z)
        logits3 = self.head3(z)
        return recon, [logits1, logits2, logits3], z


class mjm_1_lr(nn.Module):
    """mjm_1 with LR classification heads: h1/h2/H → Linear directly."""

    def __init__(self,
                 input_dim=140,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=[3, 24, 137],
                 spatial_dim=0,
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=128, residual_mode='feature',
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim

        # Encoders (identical to mjm_1)
        self.E1 = nn.Sequential(
            nn.Linear(input_dim + spatial_dim, enc_hidden_dim),
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

        # Reconstruction decoder (identical to mjm_1)
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR heads: direct linear from encoder outputs
        self.head1 = nn.Linear(enc_hidden_dim, output_num[0])   # h1 → class
        self.head2 = nn.Linear(enc_hidden_dim, output_num[1])   # h2 → subclass
        self.head3 = nn.Linear(latent_dim, output_num[2])       # H  → supertype

    def forward(self, x):
        h1 = self.E1(x)
        h2 = self.E2(h1)
        H  = self.E3(h2)

        logits1 = self.head1(h1)
        logits2 = self.head2(h2)
        logits3 = self.head3(H)

        recon = self.recon_decoder(H)
        return recon, [logits1, logits2, logits3], H
