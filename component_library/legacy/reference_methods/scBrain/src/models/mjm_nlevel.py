"""
mjm_1_nlevel / mjm_2c_nlevel: Generalized N-level cascaded encoder-decoder.

Extends the original 3-level mjm_1 to support arbitrary number of hierarchy levels.
Used for Kukanja (4-level) and other multi-level datasets.

Architecture (N levels):
  X -> E1->h1 -> E2->h2 -> ... -> EN->H (latent)
       D1->C1    D2->C2+C1        DN->CN+C(N-1)  + ReconDecoder
       head1     head2             headN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mjm import RMSNorm, SwiGLU, FFN, _make_dec_block


class mjm_1_nlevel(nn.Module):
    """
    N-level cascaded encoder-decoder with cross-layer residual classification.

    Encoder chain:  E1 (Linear+FFN) -> E2 (FFN) -> ... -> EN (FFN+Linear->latent)
    Decoder chain:  D1(h1)->C1, D2(h2)+C1->C2, ..., DN(H)+C(N-1)->CN
    Heads:          head_i(Ci) -> logits_i

    Residual modes:
        'feature': C_{i+1} = D_{i+1}(...) + C_i
        'logit':   C_{i+1} = D_{i+1}(...) + proj(logits_i)
        'none':    no residual
    """
    def __init__(self,
                 input_dim=266,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=None,
                 residual_mode='feature',
                 ):
        super().__init__()
        if output_num is None:
            output_num = [13, 14, 25, 27]
        assert len(output_num) >= 2, f"Need at least 2 levels, got {len(output_num)}"
        assert residual_mode in ('feature', 'logit', 'none')

        self.n_levels = len(output_num)
        self.output_num = output_num
        self.residual_mode = residual_mode
        self.latent_dim = latent_dim
        self.enc_hidden_dim = enc_hidden_dim

        # -- Encoders --
        # E1: Linear(input_dim -> enc_hidden_dim) + FFN
        # E2..E(N-1): FFN only
        # EN: FFN + Linear(enc_hidden_dim -> latent_dim)
        encoders = []
        for i in range(self.n_levels):
            if i == 0:
                encoders.append(nn.Sequential(
                    nn.Linear(input_dim, enc_hidden_dim),
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                ))
            elif i == self.n_levels - 1:
                encoders.append(nn.Sequential(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                    nn.Linear(enc_hidden_dim, latent_dim),
                ))
            else:
                encoders.append(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout)
                )
        self.encoders = nn.ModuleList(encoders)

        # -- Decoders --
        # D1..D(N-1): input from enc_hidden_dim
        # DN: input from latent_dim
        decoders = []
        for i in range(self.n_levels):
            in_dim = latent_dim if i == self.n_levels - 1 else enc_hidden_dim
            decoders.append(_make_dec_block(in_dim, dec_hidden_dim,
                                            d_layers, expansion_factor, dropout))
        self.decoders = nn.ModuleList(decoders)

        # -- Classification heads --
        self.heads = nn.ModuleList([
            nn.Linear(dec_hidden_dim, n_cls) for n_cls in output_num
        ])

        # -- Residual projections (logit mode only) --
        if residual_mode == 'logit':
            self.residual_projs = nn.ModuleList([
                nn.Linear(output_num[i], dec_hidden_dim)
                for i in range(self.n_levels - 1)
            ])

        # -- Reconstruction decoder --
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

    def forward(self, x):
        # -- Encode --
        h_list = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            h_list.append(h)

        H = h_list[-1]  # latent representation

        # -- Decode + classify with residual chain --
        logits_list = []
        C_prev = None
        for i in range(self.n_levels):
            Ci = self.decoders[i](h_list[i])

            if i > 0 and self.residual_mode == 'feature':
                Ci = Ci + C_prev
            elif i > 0 and self.residual_mode == 'logit':
                Ci = Ci + self.residual_projs[i - 1](logits_list[-1])

            logits_i = self.heads[i](Ci)
            logits_list.append(logits_i)
            C_prev = Ci

        # -- Reconstruct --
        recon = self.recon_decoder(H)

        return recon, logits_list, H


class mjm_nlevel(nn.Module):
    """
    N-level flat encoder + hierarchical decoder (generalizes mjm base).

    Encoder: Linear(input_dim -> enc_hidden_dim) -> FFN -> Linear(-> latent_dim)
    Decoder: N independent decoder blocks from latent, with optional residual chain.
    """
    def __init__(self,
                 input_dim=266,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=None,
                 is_hierarchical=True,
                 ):
        super().__init__()
        if output_num is None:
            output_num = [13, 14, 25, 27]
        self.n_levels = len(output_num)
        self.output_num = output_num
        self.is_hierarchical = is_hierarchical

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        self.decoders = nn.ModuleList([
            SwiGLU(dec_hidden_dim, dec_hidden_dim, dropout=dropout)
            for _ in range(self.n_levels)
        ])
        self.input_proj = nn.Linear(latent_dim, dec_hidden_dim) if latent_dim != dec_hidden_dim else nn.Identity()
        self.heads = nn.ModuleList([
            nn.Linear(dec_hidden_dim, n_cls) for n_cls in output_num
        ])

    def forward(self, x):
        z = self.encoder(x)
        recon = self.recon_decoder(z)

        h = self.input_proj(z)
        logits_list = []
        curr = h
        for i in range(self.n_levels):
            if self.is_hierarchical:
                curr = curr + self.decoders[i](curr)
                logits_list.append(self.heads[i](curr))
            else:
                logits_list.append(self.heads[i](self.decoders[i](h)))

        return recon, logits_list, z


class mjm_lr_nlevel(nn.Module):
    """
    N-level flat encoder + LR heads (generalizes mjm_lr).

    Encoder: Linear -> FFN -> Linear -> latent
    Heads: N x Linear(latent_dim -> n_cls)
    """
    def __init__(self,
                 input_dim=266,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=None,
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=256, is_hierarchical=True,
                 ):
        super().__init__()
        if output_num is None:
            output_num = [13, 14, 25, 27]
        self.n_levels = len(output_num)
        self.output_num = output_num

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_hidden_dim),
            FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                expansion_factor=expansion_factor, dropout=dropout),
            nn.Linear(enc_hidden_dim, latent_dim),
        )

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        self.heads = nn.ModuleList([
            nn.Linear(latent_dim, n_cls) for n_cls in output_num
        ])

    def forward(self, x):
        z = self.encoder(x)
        recon = self.recon_decoder(z)
        logits_list = [h(z) for h in self.heads]
        return recon, logits_list, z


class mjm_1_lr_nlevel(nn.Module):
    """
    N-level cascaded encoder + LR heads (generalizes mjm_1_lr).

    Encoder chain: E1(Linear+FFN) -> E2(FFN) -> ... -> EN(FFN+Linear->latent)
    Heads: head_i(h_i) -> logits_i  (direct Linear, no decoder blocks)
    """
    def __init__(self,
                 input_dim=266,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=None,
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=128, residual_mode='feature',
                 ):
        super().__init__()
        if output_num is None:
            output_num = [13, 14, 25, 27]
        assert len(output_num) >= 2
        self.n_levels = len(output_num)
        self.output_num = output_num
        self.enc_hidden_dim = enc_hidden_dim

        # Encoders (same structure as mjm_1_nlevel)
        encoders = []
        for i in range(self.n_levels):
            if i == 0:
                encoders.append(nn.Sequential(
                    nn.Linear(input_dim, enc_hidden_dim),
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                ))
            elif i == self.n_levels - 1:
                encoders.append(nn.Sequential(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                    nn.Linear(enc_hidden_dim, latent_dim),
                ))
            else:
                encoders.append(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout)
                )
        self.encoders = nn.ModuleList(encoders)

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR heads: direct linear from encoder outputs
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim if i == self.n_levels - 1 else enc_hidden_dim,
                      n_cls)
            for i, n_cls in enumerate(output_num)
        ])

    def forward(self, x):
        h_list = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            h_list.append(h)

        H = h_list[-1]
        recon = self.recon_decoder(H)
        logits_list = [self.heads[i](h_list[i]) for i in range(self.n_levels)]
        return recon, logits_list, H


class mjm_2c_nlevel(nn.Module):
    """
    N-level mjm_1 + FiLM modulation of h1 conditioned on disease score + spatial coords.

    Context = [disease_score(1), x_coord(1), y_coord(1)] -> [B,3]
    FiLM: gamma, beta -> h1_mod = (1+gamma)*h1 + beta
    """
    def __init__(self,
                 input_dim=266,
                 latent_dim=20,
                 e_layers=3,
                 d_layers=1,
                 enc_hidden_dim=256,
                 dec_hidden_dim=128,
                 film_hidden=64,
                 film_input_dim=3,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=None,
                 residual_mode='feature',
                 ):
        super().__init__()
        if output_num is None:
            output_num = [13, 14, 25, 27]
        assert len(output_num) >= 2
        assert residual_mode == 'feature', "mjm_2c only supports feature residual"

        self.n_levels = len(output_num)
        self.output_num = output_num
        self.residual_mode = residual_mode
        self.enc_hidden_dim = enc_hidden_dim

        # -- Encoders (same structure as mjm_1_nlevel) --
        encoders = []
        for i in range(self.n_levels):
            if i == 0:
                encoders.append(nn.Sequential(
                    nn.Linear(input_dim, enc_hidden_dim),
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                ))
            elif i == self.n_levels - 1:
                encoders.append(nn.Sequential(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                    nn.Linear(enc_hidden_dim, latent_dim),
                ))
            else:
                encoders.append(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout)
                )
        self.encoders = nn.ModuleList(encoders)

        # -- Decoders --
        decoders = []
        for i in range(self.n_levels):
            in_dim = latent_dim if i == self.n_levels - 1 else enc_hidden_dim
            decoders.append(_make_dec_block(in_dim, dec_hidden_dim,
                                            d_layers, expansion_factor, dropout))
        self.decoders = nn.ModuleList(decoders)

        # -- Classification heads --
        self.heads = nn.ModuleList([
            nn.Linear(dec_hidden_dim, n_cls) for n_cls in output_num
        ])

        # -- Reconstruction decoder --
        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # -- FiLM context network --
        self.film_net = nn.Sequential(
            nn.Linear(film_input_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, enc_hidden_dim * 2),
        )

    def forward(self, x, disease_score, spatial_norm):
        """
        Args:
            x:              [B, input_dim] gene expression (log1p)
            disease_score:  [B, 1] disease conditioning score
            spatial_norm:   [B, 2] normalized spatial coordinates
        """
        # -- E1 --
        h1 = self.encoders[0](x)

        # -- FiLM modulation --
        context = torch.cat([disease_score, spatial_norm], dim=-1)
        film_params = self.film_net(context)
        gamma, beta = film_params.chunk(2, dim=-1)
        h1_mod = (1.0 + gamma) * h1 + beta

        # -- E2..EN --
        h_list = [h1_mod]
        h = h1_mod
        for enc in self.encoders[1:]:
            h = enc(h)
            h_list.append(h)

        H = h_list[-1]

        # -- Decode + classify --
        logits_list = []
        C_prev = None
        for i in range(self.n_levels):
            Ci = self.decoders[i](h_list[i])
            if i > 0:
                Ci = Ci + C_prev
            logits_i = self.heads[i](Ci)
            logits_list.append(logits_i)
            C_prev = Ci

        recon = self.recon_decoder(H)
        return recon, logits_list, H


class mjm_2c_lr_nlevel(nn.Module):
    """
    N-level mjm_1_lr + FiLM modulation (generalizes mjm_2c_lr).

    Cascaded encoder with FiLM on h1, LR classification heads.
    Context = [disease_score(1), x_coord(1), y_coord(1)] -> [B,3]
    """
    def __init__(self,
                 input_dim=266,
                 latent_dim=20,
                 e_layers=3,
                 enc_hidden_dim=256,
                 film_hidden=64,
                 film_input_dim=3,
                 expansion_factor=2.67,
                 dropout=0.3,
                 output_num=None,
                 # unused, kept for CLI compatibility
                 d_layers=1, dec_hidden_dim=128, residual_mode='feature',
                 ):
        super().__init__()
        if output_num is None:
            output_num = [13, 14, 25, 27]
        assert len(output_num) >= 2
        self.n_levels = len(output_num)
        self.output_num = output_num
        self.enc_hidden_dim = enc_hidden_dim

        # Encoders (same as mjm_1_lr_nlevel)
        encoders = []
        for i in range(self.n_levels):
            if i == 0:
                encoders.append(nn.Sequential(
                    nn.Linear(input_dim, enc_hidden_dim),
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                ))
            elif i == self.n_levels - 1:
                encoders.append(nn.Sequential(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout),
                    nn.Linear(enc_hidden_dim, latent_dim),
                ))
            else:
                encoders.append(
                    FFN(n_layers=e_layers, model_dim=enc_hidden_dim,
                        expansion_factor=expansion_factor, dropout=dropout)
                )
        self.encoders = nn.ModuleList(encoders)

        self.recon_decoder = SwiGLU(latent_dim, input_dim, dropout=dropout)

        # LR heads
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim if i == self.n_levels - 1 else enc_hidden_dim,
                      n_cls)
            for i, n_cls in enumerate(output_num)
        ])

        # FiLM context network
        self.film_net = nn.Sequential(
            nn.Linear(film_input_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, enc_hidden_dim * 2),
        )

    def forward(self, x, disease_score, spatial_norm):
        # E1
        h1 = self.encoders[0](x)

        # FiLM modulation
        context = torch.cat([disease_score, spatial_norm], dim=-1)
        film_params = self.film_net(context)
        gamma, beta = film_params.chunk(2, dim=-1)
        h1_mod = (1.0 + gamma) * h1 + beta

        # E2..EN
        h_list = [h1_mod]
        h = h1_mod
        for enc in self.encoders[1:]:
            h = enc(h)
            h_list.append(h)

        H = h_list[-1]
        recon = self.recon_decoder(H)
        logits_list = [self.heads[i](h_list[i]) for i in range(self.n_levels)]
        return recon, logits_list, H
