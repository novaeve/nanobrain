"""
N-level foundation model wrappers for Kukanja.
These extend the existing FM model classes to support N classification heads.
"""
import torch
import torch.nn as nn


class ClsDecoder(nn.Module):
    """Shared classification decoder head."""
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers += [nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU()]
        layers.append(nn.Linear(d_model, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GeneformerNLevel(nn.Module):
    """
    N-level Geneformer wrapper. Loads pretrained backbone from
    GeneformerForAnnotation, replaces heads with N-level ModuleList.
    """
    def __init__(self, code_dir, weights_path, gene_names,
                 output_num, dropout=0.2, freeze_backbone=False, max_seq_len=1024):
        super().__init__()
        from src.models.geneformer.geneformer_annotation import GeneformerForAnnotation

        # Create base model with dummy 3-level output (just to load backbone)
        dummy_out = output_num[:3] if len(output_num) >= 3 else output_num + [2] * (3 - len(output_num))
        self._base = GeneformerForAnnotation(
            code_dir=code_dir, weights_path=weights_path,
            gene_names=gene_names, output_num=dummy_out,
            dropout=dropout, freeze_backbone=freeze_backbone,
            max_seq_len=max_seq_len,
        )

        d_model = self._base.d_model

        # Replace with N-level heads
        self.heads = nn.ModuleList([
            ClsDecoder(d_model, n) for n in output_num
        ])

        # Remove old heads to avoid confusion
        del self._base.cls_head_class
        del self._base.cls_head_subclass
        del self._base.cls_head_supertype

    def forward(self, X):
        input_ids, attention_mask = self._base._tokenize(X)
        outputs = self._base.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        gene_attn_mask = attention_mask.clone()
        gene_attn_mask[:, 0] = 0.0
        count = gene_attn_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        cell_emb = (last_hidden * gene_attn_mask.unsqueeze(-1)).sum(dim=1) / count

        logits = [h(cell_emb) for h in self.heads]
        return logits, cell_emb


class UCENLevel(nn.Module):
    """
    N-level UCE wrapper. Loads pretrained backbone from UCEForAnnotation,
    replaces heads with N-level ModuleList.
    """
    def __init__(self, model_dir, gene_names, output_num,
                 dropout=0.2, freeze_backbone=False, species='human'):
        super().__init__()
        from src.models.uce.uce_annotation import UCEForAnnotation

        dummy_out = output_num[:3] if len(output_num) >= 3 else output_num + [2] * (3 - len(output_num))
        self._base = UCEForAnnotation(
            model_dir=model_dir, gene_names=gene_names,
            output_num=dummy_out, dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        # For mouse species, we need to override the species filter
        if species == 'mouse':
            self._override_species_mouse(model_dir, gene_names)

        d_model = self._base.d_model

        self.heads = nn.ModuleList([
            ClsDecoder(d_model, n) for n in output_num
        ])

        del self._base.cls_head_class
        del self._base.cls_head_subclass
        del self._base.cls_head_supertype

    def _override_species_mouse(self, model_dir, gene_names):
        """Override UCE's gene mapping to use mouse genes instead of human."""
        import os, pickle
        import pandas as pd
        import torch
        import numpy as np

        chrom_csv = os.path.join(model_dir, 'species_chrom.csv')
        offsets_pkl = os.path.join(model_dir, 'species_offsets.pkl')

        chrom_df = pd.read_csv(chrom_csv)
        # Reconstruct spec_chrom codes the same way as base class
        chrom_df["spec_chrom"] = pd.Categorical(
            chrom_df["species"] + "_" + chrom_df["chromosome"].astype(str)
        )

        mouse_df = chrom_df[chrom_df['species'] == 'mouse'].reset_index(drop=True)
        if mouse_df.empty:
            print("[UCE] WARNING: No mouse genes found in species_chrom.csv")
            return

        with open(offsets_pkl, 'rb') as f:
            offsets = pickle.load(f)
        mouse_offset = offsets.get('mouse', 0)

        # Build gene lookup (case-insensitive: UCE uses UPPERCASE, EAE uses Title Case)
        gene_to_info = {}
        for i, row in mouse_df.iterrows():
            gene = row['gene_symbol']
            gene_to_info[gene.upper()] = {
                "token_idx":  mouse_offset + i,
                "chrom_code": int(mouse_df["spec_chrom"].cat.codes[i]),
                "start":      int(row["start"]),
            }

        # Rebuild all four buffers
        from src.models.uce.uce_annotation import PAD_TOKEN_IDX
        token_idxs = []
        chrom_codes = []
        starts = []
        valid_mask = []

        for g in gene_names:
            info = gene_to_info.get(g.upper())
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

        valid_count = sum(valid_mask)
        print(f"[UCE-mouse] Gene coverage: {valid_count}/{len(gene_names)}")

        # Replace the base class buffers
        self._base._token_idxs  = torch.tensor(token_idxs,  dtype=torch.long)
        self._base._chrom_codes = torch.tensor(chrom_codes, dtype=torch.long)
        self._base._starts      = torch.tensor(starts,      dtype=torch.long)
        self._base._valid_mask  = torch.tensor(valid_mask,  dtype=torch.bool)
        self._base.register_buffer("_token_idxs",  self._base._token_idxs)
        self._base.register_buffer("_chrom_codes", self._base._chrom_codes)
        self._base.register_buffer("_starts",      self._base._starts)
        self._base.register_buffer("_valid_mask",  self._base._valid_mask)

    def forward(self, X):
        # Bypass base forward (heads deleted), call backbone directly
        import torch.nn.functional as F
        sentences, mask = self._base._build_sentences(X)
        emb = self._base.pe_embedding(sentences)
        emb = F.normalize(emb, dim=2)
        _, cell_emb = self._base.backbone(emb, mask=mask)
        logits = [h(cell_emb) for h in self.heads]
        return logits, cell_emb


class ScGPTNLevel(nn.Module):
    """
    N-level scGPT wrapper. Converts gene_names to vocab IDs, loads pretrained
    backbone from scGPTForAnnotation, replaces heads with N-level ModuleList.
    """
    def __init__(self, ckpt_dir, gene_names, output_num,
                 dropout=0.2, freeze_backbone=False):
        super().__init__()
        import json
        from pathlib import Path
        from src.models.scGPT.scGPT_annotation import scGPTForAnnotation

        # Convert gene names to vocab IDs
        ckpt_path = Path(ckpt_dir)
        with open(ckpt_path / "vocab.json") as f:
            vocab = json.load(f)

        gene_ids = []
        valid = 0
        for g in gene_names:
            if g in vocab:
                gene_ids.append(vocab[g])
                valid += 1
            else:
                gene_ids.append(vocab.get("<pad>", 0))
        print(f"[scGPT] Gene coverage: {valid}/{len(gene_names)}")
        gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long)

        dummy_out = output_num[:3] if len(output_num) >= 3 else output_num + [2] * (3 - len(output_num))
        self._base = scGPTForAnnotation(
            checkpoint_dir=ckpt_dir, gene_ids=gene_ids_tensor,
            output_num=dummy_out, dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        d_model = self._base.d_model

        self.heads = nn.ModuleList([
            ClsDecoder(d_model, n) for n in output_num
        ])

        del self._base.cls_head_class
        del self._base.cls_head_subclass
        del self._base.cls_head_supertype

    def forward(self, X):
        # Bypass base forward (heads deleted), call backbone directly
        from src.models.scGPT.binning import scgpt_binning_torch
        batch_size = X.shape[0]
        device = X.device

        X_norm = torch.log1p(X)
        X_binned = scgpt_binning_torch(X_norm, n_bins=self._base.n_bins).float()

        cls_ids  = X.new_full((batch_size, 1), self._base.cls_token_id, dtype=torch.long)
        cls_vals = X.new_zeros(batch_size, 1)

        gene_ids_exp = self._base.gene_ids.unsqueeze(0).expand(batch_size, -1)
        src    = torch.cat([cls_ids, gene_ids_exp], dim=1)
        values = torch.cat([cls_vals, X_binned],    dim=1)

        src_key_padding_mask = torch.zeros(
            batch_size, src.shape[1], dtype=torch.bool, device=device
        )

        output   = self._base.backbone(src, values, src_key_padding_mask)
        cell_emb = output["cell_emb"]

        logits = [h(cell_emb) for h in self.heads]
        return logits, cell_emb


class StackNLevel(nn.Module):
    """
    N-level Stack wrapper. Loads pretrained backbone from StackForAnnotation,
    replaces heads with N-level ModuleList.
    """
    def __init__(self, checkpoint_path, gene_list_path, gene_names,
                 output_num, dropout=0.2, freeze_backbone=False):
        super().__init__()
        from src.models.stack_model.stack_annotation import StackForAnnotation

        dummy_out = output_num[:3] if len(output_num) >= 3 else output_num + [2] * (3 - len(output_num))
        self._base = StackForAnnotation(
            checkpoint_path=checkpoint_path,
            gene_list_path=gene_list_path,
            gene_names=gene_names,
            output_num=dummy_out,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        d_model = self._base.embed_dim
        self.heads = nn.ModuleList([
            ClsDecoder(d_model, n) for n in output_num
        ])

        del self._base.cls_head_class
        del self._base.cls_head_subclass
        del self._base.cls_head_supertype

    def forward(self, X):
        features = self._base._map_genes(X)
        features_log = torch.log1p(features)

        if self._base.freeze_backbone:
            with torch.no_grad():
                tokens = self._base.backbone._reduce_and_tokenize(features_log)
                x = self._base.backbone._run_attention_layers(tokens)
        else:
            tokens = self._base.backbone._reduce_and_tokenize(features_log)
            x = self._base.backbone._run_attention_layers(tokens)

        cell_emb = x.reshape(X.shape[0], -1)
        logits = [h(cell_emb) for h in self.heads]
        return logits, cell_emb


class ScSimilarityNLevel(nn.Module):
    """
    N-level scSimilarity wrapper trained from scratch on the current gene panel.
    """
    def __init__(self, n_genes, output_num, latent_dim=128, hidden_dim=None,
                 dropout=0.5, input_dropout=0.4, freeze_backbone=False):
        super().__init__()
        from scimilarity.nn_models import Encoder

        if hidden_dim is None:
            hidden_dim = [512, 512]

        self.encoder = Encoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            input_dropout=input_dropout,
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.heads = nn.ModuleList([
            ClsDecoder(latent_dim, n) for n in output_num
        ])

    def forward(self, X):
        X_norm = torch.log1p(X)
        if self.freeze_backbone:
            with torch.no_grad():
                cell_emb = self.encoder(X_norm)
        else:
            cell_emb = self.encoder(X_norm)
        logits = [h(cell_emb) for h in self.heads]
        return logits, cell_emb


class NicheformerNLevel(nn.Module):
    """
    N-level Nicheformer wrapper for human Kukanja datasets.
    """
    def __init__(self, checkpoint_path, vocab_path, merfish_mean_path,
                 gene_name_to_ens_path, gene_names, output_num,
                 dropout=0.2, freeze_backbone=False, specie_token=5):
        super().__init__()
        from src.models.nicheformer.nicheformer_annotation import NicheformerForAnnotation

        dummy_out = output_num[:3] if len(output_num) >= 3 else output_num + [2] * (3 - len(output_num))
        self._base = NicheformerForAnnotation(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            merfish_mean_path=merfish_mean_path,
            gene_name_to_ens_path=gene_name_to_ens_path,
            gene_names=gene_names,
            output_num=dummy_out,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            specie_token=specie_token,
        )

        for head in [self._base.cls_head_class, self._base.cls_head_subclass, self._base.cls_head_supertype]:
            for p in head.parameters():
                p.requires_grad = False

        d_model = self._base.d_model
        self.heads = nn.ModuleList([
            ClsDecoder(d_model, n) for n in output_num
        ])

    def forward(self, X):
        _, cell_emb = self._base(X)
        logits = [h(cell_emb) for h in self.heads]
        return logits, cell_emb


class ScFoundationNLevel(nn.Module):
    """
    N-level scFoundation wrapper.
    """
    def __init__(self, model_path, config_path, gene_names, output_num,
                 dropout=0.2, freeze_backbone=False, pool_type='all'):
        super().__init__()
        from src.models.scfoundation.scfoundation_annotation import ScFoundationForAnnotation

        dummy_out = output_num[:3] if len(output_num) >= 3 else output_num + [2] * (3 - len(output_num))
        self._base = ScFoundationForAnnotation(
            model_path=model_path,
            config_path=config_path,
            gene_names=gene_names,
            output_num=dummy_out,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pool_type=pool_type,
        )

        for head in [self._base.cls_head_class, self._base.cls_head_subclass, self._base.cls_head_supertype]:
            for p in head.parameters():
                p.requires_grad = False

        d_model = self._base.embed_dim
        self.heads = nn.ModuleList([
            ClsDecoder(d_model, n) for n in output_num
        ])

    def forward(self, X):
        cell_emb = self._base._encode(X)
        cell_emb = self._base.emb_norm(cell_emb)
        logits = [h(cell_emb) for h in self.heads]
        return logits, cell_emb
