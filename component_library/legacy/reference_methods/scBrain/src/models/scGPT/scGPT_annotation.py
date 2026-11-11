"""
scGPT fine-tuning wrapper for hierarchical cell-type annotation.

Pipeline per forward call:
  raw X (N×140)  →  log1p  →  bin (scGPT-style, n_bins=51)
  →  tokenise (prepend <cls>)  →  TransformerModel backbone
  →  cell_emb (CLS position)  →  3 × ClsDecoder heads
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .scGPT import TransformerModel, ClsDecoder
from .binning import scgpt_binning_torch


class scGPTForAnnotation(nn.Module):
    """
    Pre-trained scGPT brain model fine-tuned for hierarchical cell-type
    annotation on MERFISH spatial-transcriptomics data.

    Args:
        checkpoint_dir: path to folder containing args.json, vocab.json,
                        best_model.pt  (brain checkpoint).
        gene_ids:       LongTensor of shape (140,) – vocab indices for the
                        140 valid MERFISH genes (from merfish_indices.npy).
        output_num:     list of [n_class, n_subclass, n_supertype].
        n_bins:         number of expression bins (must match pre-training, 51).
        dropout:        dropout rate for the classification heads.
        freeze_backbone: if True, the transformer weights are frozen and only
                        the classification heads are trained.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        gene_ids: torch.Tensor,
        output_num: list = None,
        n_bins: int = 51,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self.n_bins = n_bins

        # ── load checkpoint config ───────────────────────────────────────────
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "args.json") as f:
            cfg = json.load(f)

        d_model     = cfg["embsize"]          # 512
        nhead       = cfg["nheads"]           # 8
        d_hid       = cfg["d_hid"]            # 512
        nlayers     = cfg["nlayers"]          # 12
        nlayers_cls = cfg.get("n_layers_cls", 3)
        pad_token   = cfg.get("pad_token", "<pad>")
        pad_value   = cfg.get("pad_value", -2)
        self.d_model = d_model

        # ── load vocab (plain dict) ──────────────────────────────────────────
        with open(checkpoint_dir / "vocab.json") as f:
            vocab_dict = json.load(f)

        self.cls_token_id = vocab_dict["<cls>"]
        self.pad_token_id = vocab_dict[pad_token]
        self.pad_value    = pad_value

        # ── register gene IDs as a non-trainable buffer ──────────────────────
        self.register_buffer("gene_ids", gene_ids.long())  # (140,)

        # ── backbone ─────────────────────────────────────────────────────────
        self.backbone = TransformerModel(
            ntoken             = len(vocab_dict),
            d_model            = d_model,
            nhead              = nhead,
            d_hid              = d_hid,
            nlayers            = nlayers,
            nlayers_cls        = nlayers_cls,
            n_cls              = 1,           # placeholder; we use our own heads
            vocab              = vocab_dict,
            dropout            = dropout,
            pad_token          = pad_token,
            pad_value          = pad_value,
            do_mvc             = False,
            do_dab             = False,
            use_batch_labels   = False,
            domain_spec_batchnorm = False,
            input_emb_style    = "continuous",
            n_input_bins       = n_bins,
            cell_emb_style     = "cls",
            explicit_zero_prob = False,
            use_fast_transformer = False,
            pre_norm           = False,
        )

        # load pre-trained weights
        model_file = checkpoint_dir / "best_model.pt"
        pretrained_state = torch.load(model_file, map_location="cpu")

        # The brain checkpoint was saved with FlashAttention which stores a
        # single fused "Wqkv" projection.  Standard PyTorch MultiheadAttention
        # expects "in_proj_weight" / "in_proj_bias" instead.
        # The tensors are identical in layout so we just rename the keys.
        converted = {}
        for k, v in pretrained_state.items():
            new_k = k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight") \
                     .replace("self_attn.Wqkv.bias",   "self_attn.in_proj_bias")
            converted[new_k] = v

        missing, unexpected = self.backbone.load_state_dict(converted, strict=False)
        # Expected missing: cls_decoder (not trained in pre-training)
        # Expected unexpected: flag_encoder, mvc_decoder (not in our model)
        print(
            f"[scGPT] Loaded pre-trained weights from {model_file}\n"
            f"        Missing keys  : {len(missing)}  {missing[:3]}\n"
            f"        Unexpected keys: {len(unexpected)}  {unexpected[:3]}"
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[scGPT] Backbone weights frozen.")

        # ── hierarchical classification heads ────────────────────────────────
        self.cls_head_class     = ClsDecoder(d_model, output_num[0], nlayers=nlayers_cls)
        self.cls_head_subclass  = ClsDecoder(d_model, output_num[1], nlayers=nlayers_cls)
        self.cls_head_supertype = ClsDecoder(d_model, output_num[2], nlayers=nlayers_cls)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (batch, 140)  raw gene expression, non-negative float32.

        Returns:
            logits:   list of [(batch, 3), (batch, 24), (batch, 137)]
            cell_emb: (batch, d_model)
        """
        batch_size = X.shape[0]
        device     = X.device

        # 1. log1p normalise (consistent with MJM trainer)
        X_norm = torch.log1p(X)

        # 2. scGPT-style per-cell quantile binning
        X_binned = scgpt_binning_torch(X_norm, n_bins=self.n_bins).float()  # (B, 140)

        # 3. Prepend <cls> token
        cls_ids  = X.new_full((batch_size, 1), self.cls_token_id, dtype=torch.long)
        cls_vals = X.new_zeros(batch_size, 1)

        gene_ids_exp = self.gene_ids.unsqueeze(0).expand(batch_size, -1)  # (B, 140)
        src    = torch.cat([cls_ids, gene_ids_exp], dim=1)    # (B, 141)
        values = torch.cat([cls_vals, X_binned],    dim=1)    # (B, 141)

        # 4. No padding (all 140 genes always present)
        src_key_padding_mask = torch.zeros(
            batch_size, src.shape[1], dtype=torch.bool, device=device
        )

        # 5. Backbone → cell embedding from CLS position
        output   = self.backbone(src, values, src_key_padding_mask)
        cell_emb = output["cell_emb"]  # (B, d_model)

        # 6. Classification heads
        logit_class     = self.cls_head_class(cell_emb)
        logit_subclass  = self.cls_head_subclass(cell_emb)
        logit_supertype = self.cls_head_supertype(cell_emb)

        return [logit_class, logit_subclass, logit_supertype], cell_emb
