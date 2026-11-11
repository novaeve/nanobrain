"""
Nicheformer fine-tuning wrapper for hierarchical cell-type annotation.

Pipeline per forward call:
  raw X (N×140)  →  size-factor normalise (→ 10000)
  →  divide by MERFISH technology mean  →  rank-order (descending)
  →  prepend auxiliary tokens [specie=5, assay=7, modality=4]
  →  Nicheformer backbone (12 layers, 512-dim)
  →  mean-pool positions 3: (skip 3 auxiliary tokens)
  →  3 × classification heads
"""

import importlib.metadata
import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as grad_ckpt
import anndata

# Nicheformer's __init__.py calls importlib.metadata.version("nicheformer")
# which fails when the package is not pip-installed.  Patch before import.
_real_version = importlib.metadata.version
def _patched_version(pkg):
    if pkg == "nicheformer":
        return "0.0.1"
    return _real_version(pkg)
importlib.metadata.version = _patched_version

sys.path.insert(0, "./external/nicheformer/src")
from nicheformer.models._nicheformer import Nicheformer


# Nicheformer auxiliary token IDs (from pre-training configuration)
SPECIE_TOKEN  = 5   # human
ASSAY_TOKEN   = 7   # MERFISH
MODALITY_TOKEN = 4  # spatial
AUX_TOKENS     = 3  # number of auxiliary tokens prepended

PADDING_TOKEN = 1   # token_id = 0 maps to 1 inside complete_masking
AUX_TOKEN_OFFSET = 30   # gene token_id = vocab_position + 30


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


class NicheformerForAnnotation(nn.Module):
    """
    Pre-trained Nicheformer fine-tuned for hierarchical cell-type
    annotation on MERFISH spatial-transcriptomics data.

    Args:
        checkpoint_path: path to nicheformer.ckpt (PyTorch Lightning checkpoint).
        vocab_path:      path to model.h5ad whose var_names are Ensembl IDs
                         (Nicheformer gene vocabulary, 20310 genes).
        merfish_mean_path: path to merfish_mean_script.npy (20310,) – technology
                           mean per gene; NaN for genes not in MERFISH.
        gene_name_to_ens_path: path to gene_name_id_dict_gc104M.pkl used to
                           convert MERFISH gene symbols to Ensembl IDs.
        gene_names:      list of MERFISH gene symbols (same order as data columns).
        output_num:      [n_class, n_subclass, n_supertype].
        dropout:         dropout for the classification heads.
        freeze_backbone: freeze Nicheformer weights; train heads only.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        merfish_mean_path: str,
        gene_name_to_ens_path: str,
        gene_names: list,
        output_num: list = None,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        specie_token: int = SPECIE_TOKEN,
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self._specie_token = specie_token

        self.gene_names = gene_names
        self.n_genes    = len(gene_names)

        # ── load Nicheformer from Lightning checkpoint ─────────────────────────
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hp   = ckpt["hyper_parameters"]
        # remove keys not accepted by __init__
        hp.pop("autoregressive", None)
        hp.pop("pool", None)
        hp.pop("karpathy", None)

        self.backbone = Nicheformer(**hp)
        missing, unexpected = self.backbone.load_state_dict(
            ckpt["state_dict"], strict=False
        )
        print(
            f"[Nicheformer] Loaded checkpoint from {checkpoint_path}\n"
            f"              Missing keys   : {len(missing)}  {missing[:3]}\n"
            f"              Unexpected keys: {len(unexpected)}  {unexpected[:3]}"
        )
        self.d_model = hp["dim_model"]  # 512

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[Nicheformer] Backbone weights frozen.")
        else:
            # Enable gradient checkpointing for full fine-tuning to save memory
            for layer in self.backbone.encoder.layers:
                layer.gradient_checkpointing = False  # handled manually below
            self.backbone.encoder.enable_nested_tensor = False
            print("[Nicheformer] Gradient checkpointing enabled for full fine-tuning.")

        # ── load Nicheformer gene vocabulary ───────────────────────────────────
        model_adata = anndata.read_h5ad(vocab_path, backed="r")
        var_names   = list(model_adata.var_names)         # Ensembl IDs, 20310
        ens_to_pos  = {ens: i for i, ens in enumerate(var_names)}

        # ── load MERFISH technology mean ───────────────────────────────────────
        merfish_mean = np.load(merfish_mean_path)         # (20310,)

        # ── gene name → Ensembl ID mapping ────────────────────────────────────
        with open(gene_name_to_ens_path, "rb") as f:
            gene_name_to_ens: dict = pickle.load(f)

        # ── per-gene vocab positions, means, and validity flags ───────────────
        vocab_positions = []
        means           = []
        valid_mask      = []

        for g in gene_names:
            ens = gene_name_to_ens.get(g)
            pos = ens_to_pos.get(ens) if ens else None

            if pos is None:
                vocab_positions.append(0)
                means.append(1.0)
                valid_mask.append(False)
                continue

            m = merfish_mean[pos]
            if np.isnan(m) or m == 0:
                # gene in vocab but no valid MERFISH mean: use mean=1 (no scaling)
                means.append(1.0)
            else:
                means.append(float(m))

            vocab_positions.append(pos)
            valid_mask.append(True)

        self.register_buffer("_vocab_pos",  torch.tensor(vocab_positions, dtype=torch.long))
        self.register_buffer("_means",      torch.tensor(means,           dtype=torch.float32))
        self.register_buffer("_valid_mask", torch.tensor(valid_mask,      dtype=torch.bool))

        n_mapped = self._valid_mask.sum().item()
        print(f"[Nicheformer] Gene coverage: {n_mapped}/{self.n_genes}")

        # ── hierarchical classification heads ─────────────────────────────────
        self.cls_head_class     = ClsDecoder(self.d_model, output_num[0])
        self.cls_head_subclass  = ClsDecoder(self.d_model, output_num[1])
        self.cls_head_supertype = ClsDecoder(self.d_model, output_num[2])

    # ─────────────────────────────────────────────────────────────────────────

    def _tokenize(self, X: torch.Tensor):
        """
        Nicheformer tokenisation for a batch of MERFISH cells.

        Returns:
            token_ids:      (B, context_length) LongTensor
            attention_mask: (B, context_length) BoolTensor  (True = padding/ignore)
        """
        B      = X.shape[0]
        device = X.device

        context_length = self.backbone.hparams.context_length  # 1500

        # 1. Size-factor normalise to 10000 (on valid genes only)
        #    Use only valid genes for the total count to match Nicheformer training.
        X_sf = X.clone()
        totals = X_sf.sum(dim=1, keepdim=True).clamp(min=1.0)
        X_sf   = X_sf / totals * 1e4                              # (B, G)

        # 2. Divide by per-gene technology mean
        X_norm = X_sf / self._means.unsqueeze(0)                  # (B, G)
        X_norm = X_norm * self._valid_mask.unsqueeze(0).float()   # zero invalid genes

        # 3. For each cell: sort non-zero genes by X_norm descending
        #    token_id = vocab_position + AUX_TOKEN_OFFSET
        X_np      = X_norm.detach().cpu().numpy()
        vocab_pos = self._vocab_pos.cpu().numpy()
        valid     = self._valid_mask.cpu().numpy()

        # Build sequences (without auxiliary tokens yet)
        max_gene_len = context_length - AUX_TOKENS
        gene_seqs = np.ones((B, max_gene_len), dtype=np.int32)  # 1 = padding token

        for b in range(B):
            expr     = X_np[b]
            nonzero  = np.nonzero(expr)[0]
            if len(nonzero) == 0:
                continue
            sorted_i = nonzero[np.argsort(-expr[nonzero])][:max_gene_len]
            gene_seqs[b, :len(sorted_i)] = vocab_pos[sorted_i] + AUX_TOKEN_OFFSET

        gene_seqs_t = torch.from_numpy(gene_seqs).long().to(device)  # (B, max_gene_len)

        # 4. Prepend auxiliary tokens: [specie, assay, modality]
        aux = torch.tensor(
            [[self._specie_token, ASSAY_TOKEN, MODALITY_TOKEN]],
            dtype=torch.long, device=device
        ).expand(B, -1)                                            # (B, 3)

        token_ids = torch.cat([aux, gene_seqs_t], dim=1)          # (B, context_length)

        # 5. Attention mask: True where token == PADDING_TOKEN
        attention_mask = (token_ids == PADDING_TOKEN)              # (B, context_length)

        return token_ids, attention_mask

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, G) raw gene expression, non-negative float32.

        Returns:
            logits:   list of [(B, 3), (B, 24), (B, 137)]
            cell_emb: (B, 512)
        """
        token_ids, attention_mask = self._tokenize(X)

        # Backbone forward:
        # - freeze: torch.no_grad() drops all intermediate activations (~50GB saved)
        # - full:   gradient checkpointing recomputes activations layer by layer
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(token_ids, attention_mask)
        else:
            # Gradient checkpointing via per-layer recompute
            # Replicate Nicheformer.forward() embedding step
            src = self.backbone.embeddings(token_ids)             # (B, ctx, d_model)
            if self.backbone.hparams.learnable_pe:
                pos = self.backbone.positional_embedding(
                    self.backbone.pos.to(src.device)
                )
                src = self.backbone.dropout(src + pos)
            else:
                src = self.backbone.positional_embedding(src)

            key_padding_mask = attention_mask                      # True = ignore

            def make_layer_fn(layer):
                def fn(x, mask):
                    return layer(x, src_key_padding_mask=mask)
                return fn

            for layer in self.backbone.encoder.layers:
                src = grad_ckpt.checkpoint(
                    make_layer_fn(layer), src, key_padding_mask,
                    use_reentrant=False
                )
            if self.backbone.encoder.norm is not None:
                src = self.backbone.encoder.norm(src)
            outputs = {"transformer_output": src}

        hidden_state = outputs["transformer_output"]               # (B, ctx, 512)

        # Mean-pool positions AUX_TOKENS: (skip the 3 auxiliary tokens)
        gene_hidden  = hidden_state[:, AUX_TOKENS:, :]             # (B, ctx-3, 512)
        gene_mask    = ~attention_mask[:, AUX_TOKENS:]             # True = valid gene
        count        = gene_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        cell_emb     = (gene_hidden * gene_mask.unsqueeze(-1).float()).sum(dim=1) / count

        # Classification heads
        logit_class     = self.cls_head_class(cell_emb)
        logit_subclass  = self.cls_head_subclass(cell_emb)
        logit_supertype = self.cls_head_supertype(cell_emb)

        return [logit_class, logit_subclass, logit_supertype], cell_emb
