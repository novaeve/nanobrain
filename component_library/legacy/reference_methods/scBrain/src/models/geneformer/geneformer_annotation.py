"""
Geneformer (316M) fine-tuning wrapper for hierarchical cell-type annotation.

Pipeline per forward call:
  raw X (N×140)  →  total-count normalise (→ 10000)
  →  divide by gene median expression  →  rank-order (descending)
  →  BertModel (18 layers, 1152-dim)  →  mean-pool gene positions
  →  3 × classification heads
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from transformers import BertConfig, BertModel
from safetensors.torch import load_file


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


class GeneformerForAnnotation(nn.Module):
    """
    Pre-trained Geneformer (316M, gc104M) fine-tuned for hierarchical
    cell-type annotation on MERFISH spatial-transcriptomics data.

    Args:
        code_dir:        path to Geneformer_code_only directory (contains
                         config.json and the geneformer/ package with
                         token/gene-name/gene-median dictionaries).
        weights_path:    path to model.safetensors (316M weights).
        gene_names:      list of MERFISH gene symbols (same order as data
                         matrix columns).
        output_num:      [n_class, n_subclass, n_supertype].
        dropout:         dropout for the classification heads.
        freeze_backbone: freeze BERT weights; train heads only.
        max_seq_len:     maximum input sequence length (default 1024).
    """

    CLS_TOKEN_ID = 2   # <cls> in gc104M token dictionary

    def __init__(
        self,
        code_dir: str,
        weights_path: str,
        gene_names: list,
        output_num: list = None,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self.gene_names   = gene_names
        self.n_genes      = len(gene_names)
        self.max_seq_len  = max_seq_len

        # ── load dictionaries ─────────────────────────────────────────────────
        gf_dir = os.path.join(code_dir, "geneformer")

        with open(os.path.join(gf_dir, "token_dictionary_gc104M.pkl"), "rb") as f:
            token_dict: dict = pickle.load(f)         # Ensembl_ID → token_id

        with open(os.path.join(gf_dir, "gene_name_id_dict_gc104M.pkl"), "rb") as f:
            gene_name_to_ens: dict = pickle.load(f)   # gene_symbol → Ensembl_ID

        with open(os.path.join(gf_dir, "gene_median_dictionary_gc104M.pkl"), "rb") as f:
            gene_median_dict: dict = pickle.load(f)   # Ensembl_ID → median

        # ── per-gene token IDs and medians ────────────────────────────────────
        pad_token_id = token_dict.get("<pad>", 0)
        token_ids  = []
        medians    = []
        valid_mask = []

        for g in gene_names:
            ens    = gene_name_to_ens.get(g)
            tok    = token_dict.get(ens) if ens else None
            median = gene_median_dict.get(ens) if ens else None

            if tok is None or median is None or median <= 0:
                token_ids.append(pad_token_id)
                medians.append(1.0)
                valid_mask.append(False)
            else:
                token_ids.append(tok)
                medians.append(float(median))
                valid_mask.append(True)

        self.register_buffer("_token_ids",  torch.tensor(token_ids,  dtype=torch.long))
        self.register_buffer("_medians",    torch.tensor(medians,    dtype=torch.float32))
        self.register_buffer("_valid_mask", torch.tensor(valid_mask, dtype=torch.bool))
        self.pad_token_id = pad_token_id

        n_mapped = self._valid_mask.sum().item()
        print(f"[Geneformer] Gene coverage: {n_mapped}/{self.n_genes}")

        # ── BERT backbone ─────────────────────────────────────────────────────
        config = BertConfig.from_pretrained(code_dir)
        self.backbone = BertModel(config, add_pooling_layer=False)
        self.d_model  = config.hidden_size   # 1152

        # load from safetensors; the file stores bert.* + cls.* keys
        state = load_file(weights_path)
        bert_state = {
            k[len("bert."):]: v
            for k, v in state.items()
            if k.startswith("bert.")
        }
        missing, unexpected = self.backbone.load_state_dict(bert_state, strict=False)
        print(
            f"[Geneformer] Loaded weights from {weights_path}\n"
            f"             Missing keys   : {len(missing)}  {missing[:3]}\n"
            f"             Unexpected keys: {len(unexpected)}  {unexpected[:3]}"
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[Geneformer] Backbone weights frozen.")

        # ── hierarchical classification heads ─────────────────────────────────
        self.cls_head_class     = ClsDecoder(self.d_model, output_num[0])
        self.cls_head_subclass  = ClsDecoder(self.d_model, output_num[1])
        self.cls_head_supertype = ClsDecoder(self.d_model, output_num[2])

    # ─────────────────────────────────────────────────────────────────────────

    def _tokenize(self, X: torch.Tensor):
        """
        Geneformer tokenisation: rank-order genes by normalised expression.

        Args:
            X: (B, G) raw counts, float32.

        Returns:
            input_ids:      (B, seq_len) LongTensor
            attention_mask: (B, seq_len) FloatTensor  (1=real, 0=pad)
        """
        B      = X.shape[0]
        device = X.device

        # 1. total-count normalise to 10 000
        totals   = X.sum(dim=1, keepdim=True).clamp(min=1.0)
        X_norm   = X / totals * 1e4

        # 2. divide by per-gene median; mask invalid genes
        X_norm   = X_norm / self._medians.unsqueeze(0)   # (B, G)
        X_norm   = X_norm * self._valid_mask.unsqueeze(0).float()  # zero out invalid

        # 3. rank-order by normalised expression (descending)
        #    Only non-zero genes are included in the sentence.
        rank_order = torch.argsort(X_norm, dim=1, descending=True)  # (B, G)

        # determine how many non-zero tokens per cell
        # (cap at max_seq_len - 1 to leave room for CLS)
        nonzero_counts = (X_norm > 0).sum(dim=1).clamp(max=self.max_seq_len - 1)  # (B,)
        max_n_genes = int(nonzero_counts.max().item())

        seq_len = max_n_genes + 1  # +1 for CLS

        # build input_ids: CLS + gene tokens in rank order
        gene_token_ids = self._token_ids[rank_order[:, :max_n_genes]]  # (B, max_n_genes)

        # CLS token prepended
        cls_col    = torch.full((B, 1), self.CLS_TOKEN_ID, dtype=torch.long, device=device)
        input_ids  = torch.cat([cls_col, gene_token_ids], dim=1)          # (B, seq_len)

        # attention mask: 1 for CLS + actual genes, 0 for padded positions
        # nonzero_counts is the number of non-zero genes per cell
        gene_mask = (torch.arange(max_n_genes, device=device).unsqueeze(0)
                     < nonzero_counts.unsqueeze(1)).float()               # (B, max_n_genes)
        cls_mask  = torch.ones(B, 1, device=device)
        attention_mask = torch.cat([cls_mask, gene_mask], dim=1)          # (B, seq_len)

        return input_ids, attention_mask

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, G) raw gene expression, non-negative float32.

        Returns:
            logits:   list of [(B, 3), (B, 24), (B, 137)]
            cell_emb: (B, 1152)
        """
        input_ids, attention_mask = self._tokenize(X)

        # BERT forward
        outputs = self.backbone(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # (B, seq_len, 1152)

        # mean-pool over gene positions (exclude CLS at position 0)
        gene_attn_mask = attention_mask.clone()
        gene_attn_mask[:, 0] = 0.0                               # mask out CLS
        count    = gene_attn_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        cell_emb = (last_hidden * gene_attn_mask.unsqueeze(-1)).sum(dim=1) / count  # (B, 1152)

        # classification heads
        logit_class     = self.cls_head_class(cell_emb)
        logit_subclass  = self.cls_head_subclass(cell_emb)
        logit_supertype = self.cls_head_supertype(cell_emb)

        return [logit_class, logit_subclass, logit_supertype], cell_emb
