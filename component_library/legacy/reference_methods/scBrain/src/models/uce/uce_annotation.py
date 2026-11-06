"""
UCE fine-tuning wrapper for hierarchical cell-type annotation.

Pipeline per forward call:
  raw X (N×140)  →  build chromosome-ordered gene sentences
  →  pe_embedding lookup + L2-norm  →  TransformerModel backbone (4-layer)
  →  CLS embedding (position 0)  →  3 × ClsDecoder heads

Gene sentence structure (per cell, using non-zero genes):
  [CLS] [CHROM_OPEN_A] gene_a1 gene_a2 ... [CHROM_CLOSE] [CHROM_OPEN_B] ...
Chromosomes are visited in shuffled order; genes within each chromosome
are sorted by genomic start position.
"""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "./external/UCE")
from model import TransformerModel

# UCE special token indices (as defined in eval_data.py)
CLS_TOKEN_IDX      = 3
PAD_TOKEN_IDX      = 0
CHROM_TOKEN_RIGHT  = 2        # closing bracket for a chromosome block
CHROM_TOKEN_OFFSET = 143574   # chrom_code + CHROM_TOKEN_OFFSET = chrom open token


class ClsDecoder(nn.Module):
    """Two-layer classification head (matches scGPT's ClsDecoder structure)."""
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers += [nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU()]
        layers.append(nn.Linear(d_model, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UCEForAnnotation(nn.Module):
    """
    Pre-trained UCE (4-layer) fine-tuned for hierarchical cell-type
    annotation on MERFISH spatial-transcriptomics data.

    Args:
        model_dir:       path to directory containing 4layer_model.torch,
                         all_tokens.torch, species_chrom.csv,
                         species_offsets.pkl.
        gene_names:      list of MERFISH gene symbols (same order as the
                         data matrix columns).
        output_num:      [n_class, n_subclass, n_supertype].
        dropout:         dropout for the classification heads (unused here,
                         kept for interface parity).
        freeze_backbone: freeze transformer weights; train heads only.
    """

    def __init__(
        self,
        model_dir: str,
        gene_names: list,
        output_num: list = None,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.d_model = 1280

        # ── backbone ──────────────────────────────────────────────────────────
        self.backbone = TransformerModel(
            token_dim  = 5120,
            d_model    = 1280,
            nhead      = 20,
            d_hid      = 5120,
            nlayers    = 4,
            output_dim = 1280,
            dropout    = 0.05,
        )

        full_state = torch.load(f"{model_dir}/4layer_model.torch", map_location="cpu")
        # 4layer_model.torch includes pe_embedding.weight; exclude it for the backbone
        backbone_state = {k: v for k, v in full_state.items()
                          if not k.startswith("pe_embedding")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"[UCE] Loaded backbone from {model_dir}/4layer_model.torch"
              f" | missing: {len(missing)} | unexpected: {len(unexpected)}")

        # ── pe_embedding: lookup table for all UCE tokens ─────────────────────
        all_tokens = torch.load(f"{model_dir}/all_tokens.torch", map_location="cpu")
        self.pe_embedding = nn.Embedding.from_pretrained(all_tokens, freeze=True)
        print(f"[UCE] pe_embedding shape: {all_tokens.shape}")

        # ── precompute per-gene metadata ──────────────────────────────────────
        chrom_df = pd.read_csv(f"{model_dir}/species_chrom.csv")
        # categorical codes must be computed over the full DF to match pretraining
        chrom_df["spec_chrom"] = pd.Categorical(
            chrom_df["species"] + "_" + chrom_df["chromosome"].astype(str)
        )
        human_df = chrom_df[chrom_df["species"] == "human"].reset_index(drop=True)

        with open(f"{model_dir}/species_offsets.pkl", "rb") as f:
            offsets = pickle.load(f)
        human_offset = offsets["human"]  # 13466

        gene_to_info = {
            row["gene_symbol"]: {
                "token_idx":  human_offset + i,
                "chrom_code": int(human_df["spec_chrom"].cat.codes[i]),
                "start":      int(row["start"]),
            }
            for i, row in human_df.iterrows()
        }

        token_idxs  = []
        chrom_codes = []
        starts      = []
        valid_mask  = []

        for g in gene_names:
            info = gene_to_info.get(g)
            if info is None:
                token_idxs.append(PAD_TOKEN_IDX)
                chrom_codes.append(-1)
                starts.append(0)
                valid_mask.append(False)
            else:
                token_idxs.append(info["token_idx"])
                chrom_codes.append(info["chrom_code"])
                starts.append(info["start"])
                valid_mask.append(True)

        self.register_buffer("_token_idxs",  torch.tensor(token_idxs,  dtype=torch.long))
        self.register_buffer("_chrom_codes", torch.tensor(chrom_codes, dtype=torch.long))
        self.register_buffer("_starts",      torch.tensor(starts,      dtype=torch.long))
        self.register_buffer("_valid_mask",  torch.tensor(valid_mask,  dtype=torch.bool))

        n_mapped = self._valid_mask.sum().item()
        print(f"[UCE] Gene coverage: {n_mapped}/{self.n_genes}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[UCE] Backbone weights frozen.")

        # ── hierarchical classification heads ─────────────────────────────────
        self.cls_head_class     = ClsDecoder(self.d_model, output_num[0])
        self.cls_head_subclass  = ClsDecoder(self.d_model, output_num[1])
        self.cls_head_supertype = ClsDecoder(self.d_model, output_num[2])

    # ─────────────────────────────────────────────────────────────────────────

    def _build_sentences(self, X: torch.Tensor):
        """
        Build chromosome-ordered gene sentences for a batch of cells.

        Args:
            X: (B, G) raw expression (non-negative).

        Returns:
            sentences: LongTensor (seq_len, B)
            mask:      FloatTensor (B, seq_len) – 1 valid, 0 pad
        """
        B      = X.shape[0]
        device = X.device

        X_np         = X.detach().cpu().numpy()
        token_idxs   = self._token_idxs.cpu().numpy()
        chrom_codes  = self._chrom_codes.cpu().numpy()
        starts_np    = self._starts.cpu().numpy()
        valid_mask   = self._valid_mask.cpu().numpy()

        all_sentences = []
        max_len = 0

        for b in range(B):
            expr = X_np[b]
            sel  = valid_mask & (expr > 0)
            sel_idx = np.where(sel)[0]

            sent = [CLS_TOKEN_IDX]

            if len(sel_idx) > 0:
                uq_chroms = np.unique(chrom_codes[sel_idx])
                np.random.shuffle(uq_chroms)  # shuffled chrom order (matches pretraining)

                for chrom in uq_chroms:
                    g_in_chrom = sel_idx[chrom_codes[sel_idx] == chrom]
                    order      = np.argsort(starts_np[g_in_chrom])
                    g_in_chrom = g_in_chrom[order]

                    sent.append(int(chrom) + CHROM_TOKEN_OFFSET)  # open bracket
                    for gi in g_in_chrom:
                        sent.append(int(token_idxs[gi]))
                    sent.append(CHROM_TOKEN_RIGHT)                 # close bracket

            all_sentences.append(sent)
            max_len = max(max_len, len(sent))

        # pad and stack
        sentences_padded = torch.full((B, max_len), PAD_TOKEN_IDX, dtype=torch.long)
        mask_padded      = torch.zeros(B, max_len, dtype=torch.float)

        for b, sent in enumerate(all_sentences):
            n = len(sent)
            sentences_padded[b, :n] = torch.tensor(sent, dtype=torch.long)
            mask_padded[b, :n]      = 1.0

        # backbone expects (seq_len, B) layout
        sentences_padded = sentences_padded.t().contiguous().to(device)
        mask_padded      = mask_padded.to(device)
        return sentences_padded, mask_padded

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, G) raw gene expression, non-negative float32.

        Returns:
            logits:   list of [(B, 3), (B, 24), (B, 137)]
            cell_emb: (B, 1280)
        """
        # 1. Build token sentences
        sentences, mask = self._build_sentences(X)   # (seq_len, B), (B, seq_len)

        # 2. Embedding lookup + L2-normalize
        emb = self.pe_embedding(sentences)            # (seq_len, B, 5120)
        emb = nn.functional.normalize(emb, dim=2)

        # 3. Backbone → CLS embedding at position 0
        _, cell_emb = self.backbone(emb, mask=mask)   # (B, 1280)

        # 4. Classification heads
        logit_class     = self.cls_head_class(cell_emb)
        logit_subclass  = self.cls_head_subclass(cell_emb)
        logit_supertype = self.cls_head_supertype(cell_emb)

        return [logit_class, logit_subclass, logit_supertype], cell_emb
