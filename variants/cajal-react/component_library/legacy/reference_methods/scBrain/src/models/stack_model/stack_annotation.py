"""
Stack fine-tuning wrapper for hierarchical cell-type annotation.

Pipeline per forward call:
  raw X (B, 140)  ->  map to Stack's 15012-gene space (zero-fill unmapped)
  ->  reshape to (B, 1, n_genes)  ->  log1p
  ->  Stack encoder (gene_reduction + tabular attention layers)
  ->  cell embeddings (B, n_hidden * token_dim)
  ->  3 classification heads
"""

import sys
import pickle
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "./external/stack/src")
from stack.model_loading import load_model_from_checkpoint


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


class StackForAnnotation(nn.Module):
    """
    Pre-trained Stack model fine-tuned for hierarchical cell-type
    annotation on MERFISH spatial-transcriptomics data.

    Args:
        checkpoint_path: path to Stack .ckpt checkpoint.
        gene_list_path:  path to basecount_1000per_15000max.pkl (Stack gene vocab).
        gene_names:      list of MERFISH gene symbols (same order as data columns).
        output_num:      [n_class, n_subclass, n_supertype].
        dropout:         dropout for the classification heads.
        freeze_backbone: freeze Stack encoder weights; train heads only.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gene_list_path: str,
        gene_names: list,
        output_num: list = None,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if output_num is None:
            output_num = [3, 24, 137]

        self.gene_names = gene_names
        self.n_merfish_genes = len(gene_names)

        # -- Load Stack backbone from checkpoint --
        self.backbone = load_model_from_checkpoint(
            checkpoint_path, model_class="StateICLModel", device=torch.device("cpu"),
        )
        self.backbone.train()  # load_model sets eval; we want train mode

        self.n_stack_genes = self.backbone.n_genes      # 15012
        self.n_hidden = self.backbone.n_hidden           # 100
        self.token_dim = self.backbone.token_dim         # 16
        self.embed_dim = self.n_hidden * self.token_dim  # 1600

        print(
            f"[Stack] Loaded checkpoint from {checkpoint_path}\n"
            f"        n_genes={self.n_stack_genes}, n_hidden={self.n_hidden}, "
            f"token_dim={self.token_dim}, embed_dim={self.embed_dim}"
        )

        # -- Build gene mapping: MERFISH index -> Stack index --
        with open(gene_list_path, "rb") as f:
            stack_genes = pickle.load(f)
        stack_gene_to_idx = {g: i for i, g in enumerate(stack_genes)}

        merfish_indices = []  # indices into MERFISH 140-dim
        stack_indices = []    # corresponding indices into Stack 15012-dim
        for i, g in enumerate(gene_names):
            if g in stack_gene_to_idx:
                merfish_indices.append(i)
                stack_indices.append(stack_gene_to_idx[g])

        self.register_buffer("_merfish_idx", torch.tensor(merfish_indices, dtype=torch.long))
        self.register_buffer("_stack_idx", torch.tensor(stack_indices, dtype=torch.long))
        print(f"[Stack] Gene coverage: {len(merfish_indices)}/{self.n_merfish_genes}")

        # -- Freeze backbone if requested --
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[Stack] Backbone weights frozen.")

        # -- Classification heads --
        self.cls_head_class = ClsDecoder(self.embed_dim, output_num[0])
        self.cls_head_subclass = ClsDecoder(self.embed_dim, output_num[1])
        self.cls_head_supertype = ClsDecoder(self.embed_dim, output_num[2])

    # -----------------------------------------------------------------

    def _map_genes(self, X: torch.Tensor) -> torch.Tensor:
        """Map (B, 140) MERFISH expression to (B, 1, n_stack_genes)."""
        B = X.shape[0]
        full = torch.zeros(B, self.n_stack_genes, device=X.device, dtype=X.dtype)
        full[:, self._stack_idx] = X[:, self._merfish_idx]
        return full.unsqueeze(1)  # (B, 1, n_stack_genes)

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: (B, 140) raw gene expression.

        Returns:
            logits:   list of [(B, 3), (B, 24), (B, 137)]
            cell_emb: (B, embed_dim)
        """
        # Map to Stack gene space: (B, 1, 15012)
        features = self._map_genes(X)

        # Run Stack encoder (log1p + gene_reduction + attention)
        features_log = torch.log1p(features)

        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self.backbone._reduce_and_tokenize(features_log)
                x = self.backbone._run_attention_layers(tokens)
        else:
            tokens = self.backbone._reduce_and_tokenize(features_log)
            x = self.backbone._run_attention_layers(tokens)

        # x shape: (B, 1, n_hidden, token_dim) -> flatten to (B, embed_dim)
        cell_emb = x.reshape(X.shape[0], -1)

        # Classification heads
        logit_class = self.cls_head_class(cell_emb)
        logit_subclass = self.cls_head_subclass(cell_emb)
        logit_supertype = self.cls_head_supertype(cell_emb)

        return [logit_class, logit_subclass, logit_supertype], cell_emb
