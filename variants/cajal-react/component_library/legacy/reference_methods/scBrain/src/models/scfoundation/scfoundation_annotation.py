"""
scFoundation fine-tuning wrapper for hierarchical cell-type annotation.

Pipeline per forward call:
  raw X (B, 140)  ->  map to scFoundation's 19264-gene space (zero-fill unmapped)
  ->  normalize: log1p(x / sum * 1e4)
  ->  append T and S resolution tokens  ->  (B, 19266)
  ->  gather non-zero genes (sparse input)
  ->  token_emb + pos_emb  ->  encoder (12-layer Transformer)
  ->  pool: [last, second-last, max, mean]  ->  cell embedding (B, 3072)
  ->  3 classification heads
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "./external/scFoundation/model")
from pretrainmodels import select_model
from load import gatherData


class ClsDecoder(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers += [nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU()]
        layers.append(nn.Linear(d_model, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ScFoundationForAnnotation(nn.Module):
    """
    Pre-trained scFoundation model fine-tuned for hierarchical cell-type
    annotation on MERFISH spatial-transcriptomics data.

    Args:
        model_path:      path to scFoundation .pt weights (perturblab/scfoundation-cell).
        config_path:     path to config.json for the model.
        gene_names:      list of MERFISH gene symbols (same order as data columns).
        output_num:      [n_class, n_subclass, n_supertype].
        dropout:         dropout for the classification heads.
        freeze_backbone: freeze scFoundation encoder weights; train heads only.
        pool_type:       'all' (4x concat) or 'max'.
    """

    SCF_GENE_TSV = "./weights/scfoundation/OS_scRNA_gene_index.19264.tsv"
    N_SCF_GENES = 19264

    def __init__(
        self,
        model_path: str,
        config_path: str,
        gene_names: list,
        output_num: list = None,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        pool_type: str = "all",
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self.gene_names = gene_names
        self.n_merfish_genes = len(gene_names)
        self.pool_type = pool_type
        self.freeze_backbone = freeze_backbone

        # -- Load config --
        with open(config_path) as f:
            cfg = json.load(f)

        self.model_config = self._build_config(cfg)
        self.hidden_dim = cfg["encoder_hidden_dim"]  # 768

        # -- Build and load backbone --
        backbone = select_model(self.model_config)
        state_dict = torch.load(model_path, map_location="cpu")
        backbone.load_state_dict(state_dict)

        # Extract encoder components
        self.token_emb = backbone.token_emb
        self.pos_emb = backbone.pos_emb
        self.encoder = backbone.encoder

        # Embedding dim depends on pool_type
        if pool_type == "all":
            self.embed_dim = self.hidden_dim * 4  # 3072
        else:
            self.embed_dim = self.hidden_dim  # 768

        print(
            f"[scFoundation] Loaded from {model_path}\n"
            f"        hidden_dim={self.hidden_dim}, pool={pool_type}, "
            f"embed_dim={self.embed_dim}"
        )

        # -- Build gene mapping: MERFISH index -> scFoundation index --
        gene_df = pd.read_csv(self.SCF_GENE_TSV, header=0, delimiter="\t")
        scf_gene_to_idx = {g: i for i, g in enumerate(gene_df["gene_name"])}

        merfish_indices = []
        scf_indices = []
        for i, g in enumerate(gene_names):
            if g in scf_gene_to_idx:
                merfish_indices.append(i)
                scf_indices.append(scf_gene_to_idx[g])

        self.register_buffer("_merfish_idx", torch.tensor(merfish_indices, dtype=torch.long))
        self.register_buffer("_scf_idx", torch.tensor(scf_indices, dtype=torch.long))
        print(f"[scFoundation] Gene coverage: {len(merfish_indices)}/{self.n_merfish_genes}")

        # -- Freeze backbone if requested --
        if freeze_backbone:
            for p in self.token_emb.parameters():
                p.requires_grad = False
            for p in self.pos_emb.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[scFoundation] Backbone weights frozen.")

        # -- Batch norm on embeddings --
        self.emb_norm = nn.BatchNorm1d(self.embed_dim, affine=False, eps=1e-6)

        # -- Classification heads --
        self.cls_head_class = ClsDecoder(self.embed_dim, output_num[0])
        self.cls_head_subclass = ClsDecoder(self.embed_dim, output_num[1])
        self.cls_head_supertype = ClsDecoder(self.embed_dim, output_num[2])

    # -----------------------------------------------------------------

    @staticmethod
    def _build_config(cfg):
        return {
            "model": "mae_autobin",
            "n_class": cfg["num_tokens"],
            "seq_len": cfg["num_tokens"] + 2,
            "bin_alpha": cfg["bin_alpha"],
            "bin_num": cfg["bin_num"],
            "pad_token_id": cfg["pad_token_id"],
            "mask_token_id": cfg["mask_token_id"],
            "encoder": {
                "hidden_dim": cfg["encoder_hidden_dim"],
                "depth": cfg["encoder_depth"],
                "heads": cfg["encoder_heads"],
                "dim_head": cfg.get("encoder_dim_head", 64),
                "module_type": cfg["encoder_module_type"],
                "ff_dropout": cfg.get("ff_dropout", 0.0),
                "attn_dropout": cfg.get("attn_dropout", 0.0),
            },
            "decoder": {
                "hidden_dim": cfg["decoder_hidden_dim"],
                "depth": cfg["decoder_depth"],
                "heads": cfg["decoder_heads"],
                "dim_head": cfg.get("decoder_dim_head", 64),
                "module_type": cfg["decoder_module_type"],
                "ff_dropout": cfg.get("ff_dropout", 0.0),
                "attn_dropout": cfg.get("attn_dropout", 0.0),
            },
        }

    # -----------------------------------------------------------------

    def _map_genes(self, X: torch.Tensor) -> torch.Tensor:
        """Map (B, 140) MERFISH expression to (B, 19264) scFoundation gene space."""
        B = X.shape[0]
        full = torch.zeros(B, self.N_SCF_GENES, device=X.device, dtype=X.dtype)
        full[:, self._scf_idx] = X[:, self._merfish_idx]
        return full

    def _prepare_input(self, X_raw: torch.Tensor):
        """
        Prepare scFoundation input from raw MERFISH counts.

        Returns:
            gathered_x:       (B, L) gathered non-zero expression values
            x_padding:        (B, L) padding mask
            position_gene_ids: (B, L) gene position IDs
        """
        # Map to 19264-gene space
        full_expr = self._map_genes(X_raw)  # (B, 19264)

        # Normalize: log1p(x / sum * 1e4)
        row_sums = full_expr.sum(dim=1, keepdim=True)
        x_norm = torch.log1p(full_expr / (row_sums + 1e-8) * 1e4)

        # Append T and S resolution tokens (tgthighres='a5')
        totalcount = torch.log10(row_sums.squeeze(1) + 1)
        t_token = totalcount + 5.0
        s_token = totalcount
        pretrain_x = torch.cat([x_norm, t_token.unsqueeze(1), s_token.unsqueeze(1)], dim=1)  # (B, 19266)

        # Gather non-zero entries (scFoundation's sparse input)
        value_labels = pretrain_x > 0
        gathered_x, x_padding = gatherData(pretrain_x, value_labels, self.model_config["pad_token_id"])

        data_gene_ids = torch.arange(self.N_SCF_GENES + 2, device=X_raw.device).repeat(X_raw.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.model_config["pad_token_id"])

        return gathered_x, x_padding, position_gene_ids

    def _encode(self, X_raw: torch.Tensor):
        """Run encoder and return cell embeddings."""
        gathered_x, x_padding, position_gene_ids = self._prepare_input(X_raw)

        x = self.token_emb(torch.unsqueeze(gathered_x, 2).float(), output_weight=0)
        pos = self.pos_emb(position_gene_ids)
        x = x + pos

        if self.freeze_backbone:
            with torch.no_grad():
                enc_out = self.encoder(x, x_padding)
        else:
            enc_out = self.encoder(x, x_padding)

        # Pool
        if self.pool_type == "all":
            emb1 = enc_out[:, -1, :]
            emb2 = enc_out[:, -2, :]
            emb3, _ = torch.max(enc_out[:, :-2, :], dim=1)
            emb4 = torch.mean(enc_out[:, :-2, :], dim=1)
            cell_emb = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        else:
            cell_emb, _ = torch.max(enc_out, dim=1)

        return cell_emb

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, 140) raw gene expression.
        Returns:
            logits:   list of [(B, 3), (B, 24), (B, 137)]
            cell_emb: (B, embed_dim)
        """
        cell_emb = self._encode(X)
        cell_emb = self.emb_norm(cell_emb)

        logit_class = self.cls_head_class(cell_emb)
        logit_subclass = self.cls_head_subclass(cell_emb)
        logit_supertype = self.cls_head_supertype(cell_emb)

        return [logit_class, logit_subclass, logit_supertype], cell_emb
