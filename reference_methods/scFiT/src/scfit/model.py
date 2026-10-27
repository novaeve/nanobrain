import math

import torch
import torch.nn as nn

from scfit.utils import build_mask_states


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / max(half_dim - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: float = 32.0):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.max_value = float(max_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, min=0.0, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class SequenceConditionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cell_types: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_cell_types = int(max(n_cell_types, 0))

        self.flow_time_embed = SinusoidalTimeEmbedding(d_model)
        self.flow_time_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.bio_time_embed = SinusoidalTimeEmbedding(d_model)
        self.bio_time_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.cell_type_embed = nn.Embedding(self.n_cell_types, d_model) if self.n_cell_types > 0 else None
        self.cell_type_proj = nn.Linear(d_model, d_model) if self.n_cell_types > 0 else None

        if self.cell_type_embed is not None:
            nn.init.normal_(self.cell_type_embed.weight, mean=0.0, std=0.02)
        if self.cell_type_proj is not None:
            nn.init.zeros_(self.cell_type_proj.bias)

    @staticmethod
    def _ensure_2d_scalar(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        if x.ndim == 0:
            x = x.view(1, 1).expand(ref.shape[0], 1)
        elif x.ndim == 1:
            x = x.unsqueeze(1)
        return x.to(device=ref.device, dtype=torch.float32)

    def forward(
        self,
        t_flow: torch.Tensor,
        t_src: torch.Tensor = None,
        t_tgt: torch.Tensor = None,
        cell_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        t_flow = self._ensure_2d_scalar(t_flow, t_flow)
        c_flow = self.flow_time_proj(self.flow_time_embed(t_flow))
        c_all = c_flow

        t_src = self._ensure_2d_scalar(t_src, t_flow)
        t_tgt = self._ensure_2d_scalar(t_tgt, t_flow)
        if t_src is not None and t_tgt is not None:
            bio_time = torch.cat(
                [self.bio_time_embed(t_src), self.bio_time_embed(t_tgt)],
                dim=-1,
            )
            c_bio = self.bio_time_proj(bio_time)
            c_all = c_all + c_bio

        if self.cell_type_embed is not None and self.cell_type_proj is not None and cell_type_ids is not None:
            if cell_type_ids.ndim > 1:
                cell_type_ids = cell_type_ids.view(-1)
            cell_type_emb = self.cell_type_embed(cell_type_ids.to(device=t_flow.device, dtype=torch.long))
            c_type = self.cell_type_proj(cell_type_emb)
            c_all = c_all + c_type

        return c_all


class AdaLNTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_mod: nn.Module = None,
        ffn_mod: nn.Module = None,
        use_adaln: bool = True,
    ):
        super().__init__()
        self.use_adaln = bool(use_adaln)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.attn_mod = attn_mod if attn_mod is not None else nn.Linear(d_model, d_model * 2)
        self.ffn_mod = ffn_mod if ffn_mod is not None else nn.Linear(d_model, d_model * 2)

        if attn_mod is None:
            nn.init.zeros_(self.attn_mod.weight)
            nn.init.zeros_(self.attn_mod.bias)
        if ffn_mod is None:
            nn.init.zeros_(self.ffn_mod.weight)
            nn.init.zeros_(self.ffn_mod.bias)

    @staticmethod
    def _modulate(x: torch.Tensor, mod: torch.Tensor, cls_mask: torch.Tensor = None) -> torch.Tensor:
        scale, shift = mod.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        x_mod = x * (1.0 + scale) + shift
        if cls_mask is not None:
            x_mod = torch.where(cls_mask.unsqueeze(-1), x, x_mod)
        return x_mod

    @staticmethod
    def _cls_received_attention(
        attn_weights: torch.Tensor,
        padding_mask: torch.Tensor,
        cls_mask: torch.Tensor,
    ) -> torch.Tensor:
        cls_as_key = cls_mask[:, None, None, :].to(attn_weights.dtype)
        cls_received = (attn_weights * cls_as_key).sum(dim=-1)
        valid_queries = ((~padding_mask) & (~cls_mask))[:, None, :].expand(-1, attn_weights.shape[1], -1)
        valid_queries = valid_queries.to(attn_weights.dtype)
        denom = valid_queries.sum().clamp_min(1.0)
        return (cls_received * valid_queries).sum() / denom

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        padding_mask: torch.Tensor,
        cls_mask: torch.Tensor = None,
        collect_attn_stats: bool = False,
    ):
        if self.use_adaln:
            x_norm = self._modulate(self.norm1(x), self.attn_mod(t_emb), cls_mask=cls_mask)
        else:
            x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=padding_mask,
            need_weights=collect_attn_stats,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)

        if self.use_adaln:
            x_norm = self._modulate(self.norm2(x), self.ffn_mod(t_emb), cls_mask=cls_mask)
        else:
            x_norm = self.norm2(x)
        x = x + self.dropout2(self.ffn(x_norm))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        cls_attn = None
        if collect_attn_stats and attn_weights is not None and cls_mask is not None:
            cls_attn = self._cls_received_attention(attn_weights, padding_mask, cls_mask)
        return x, cls_attn


class ExprHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TypeHead(nn.Module):
    def __init__(self, d_model: int, n_cell_types: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, int(n_cell_types)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PseudotimeHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BinHead(nn.Module):
    def __init__(self, d_model: int, n_bins: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class scFiTModel(nn.Module):
    def __init__(
        self,
        n_genes: int,
        top_k_genes: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        d_ff: int = 2048,
        nlayers: int = 8,
        dropout: float = 0.1,
        value_max: float = 32.0,
        share_adaln_projection: bool = False,
        time_conditioning_mode: str = "adaln",
        aux_bin_ce: bool = False,
        n_bins: int = 0,
        n_cell_types: int = 0,
        pseudotime_head: bool = False,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.top_k_genes = int(top_k_genes)
        self.n_cell_types = int(max(n_cell_types, 0))
        self.pad_token_id = self.n_genes
        self.cls_token_id = self.n_genes + 1
        self.time_conditioning_mode = str(time_conditioning_mode).strip().lower()
        if self.time_conditioning_mode not in {"adaln", "concat"}:
            raise ValueError(f"unsupported time_conditioning_mode={time_conditioning_mode}")

        self.gene_encoder = nn.Embedding(self.n_genes + 2, d_model, padding_idx=self.pad_token_id)
        self.gene_norm = nn.LayerNorm(d_model)
        self.value_encoder = ContinuousValueEncoder(d_model, dropout=dropout, max_value=value_max)
        self.mask_encoder = nn.Embedding(4, d_model, padding_idx=0)
        self.sequence_condition = SequenceConditionEncoder(
            d_model=d_model,
            n_cell_types=n_cell_types,
            dropout=dropout,
        )
        self.concat_condition_proj = (
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            if self.time_conditioning_mode == "concat"
            else None
        )

        self.shared_attn_mod = None
        self.shared_ffn_mod = None
        if share_adaln_projection and self.time_conditioning_mode == "adaln":
            self.shared_attn_mod = nn.Linear(d_model, d_model * 2)
            self.shared_ffn_mod = nn.Linear(d_model, d_model * 2)
            nn.init.zeros_(self.shared_attn_mod.weight)
            nn.init.zeros_(self.shared_attn_mod.bias)
            nn.init.zeros_(self.shared_ffn_mod.weight)
            nn.init.zeros_(self.shared_ffn_mod.bias)

        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    d_model,
                    nhead,
                    d_ff,
                    dropout=dropout,
                    attn_mod=self.shared_attn_mod,
                    ffn_mod=self.shared_ffn_mod,
                    use_adaln=(self.time_conditioning_mode == "adaln"),
                )
                for _ in range(nlayers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.x1_head = ExprHead(d_model, dropout=dropout)
        self.bin_head = BinHead(d_model, int(n_bins), dropout=dropout) if aux_bin_ce and int(n_bins) > 1 else None
        self.type_head = TypeHead(d_model, self.n_cell_types, dropout=dropout) if self.n_cell_types > 1 else None
        self.pseudotime_head = PseudotimeHead(d_model, dropout=dropout) if bool(pseudotime_head) else None

        nn.init.normal_(self.gene_encoder.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_encoder.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.gene_encoder.weight[self.pad_token_id].zero_()
            self.mask_encoder.weight[0].zero_()

    def encode(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        mask_states: torch.Tensor,
        padding_mask: torch.Tensor,
        t: torch.Tensor,
        t_src: torch.Tensor = None,
        t_tgt: torch.Tensor = None,
        cell_type_ids: torch.Tensor = None,
        return_cls_attention: bool = False,
    ):
        cls_mask = gene_ids.eq(self.cls_token_id)
        gene_emb = self.gene_norm(self.gene_encoder(gene_ids))
        value_emb = self.value_encoder(expr_values)
        mask_emb = self.mask_encoder(mask_states)
        x = gene_emb + value_emb + mask_emb
        c_all = self.sequence_condition(
            t_flow=t,
            t_src=t_src,
            t_tgt=t_tgt,
            cell_type_ids=cell_type_ids,
        )
        if self.concat_condition_proj is not None:
            cond_tokens = c_all.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = self.concat_condition_proj(torch.cat([x, cond_tokens], dim=-1))
        cls_attn_values = []
        for block in self.blocks:
            x, cls_attn = block(
                x,
                c_all,
                padding_mask,
                cls_mask=cls_mask,
                collect_attn_stats=return_cls_attention,
            )
            if cls_attn is not None:
                cls_attn_values.append(cls_attn)
        hidden = self.final_norm(x)
        if not return_cls_attention:
            return hidden

        stats = {}
        if cls_attn_values:
            layer_attn = torch.stack(cls_attn_values)
            stats = {
                "cls_attn_received_mean": layer_attn.mean(),
                "cls_attn_received_last": layer_attn[-1],
            }
        return hidden, stats

    def get_cell_embedding(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        mask_states: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
        t: torch.Tensor = None,
        t_src: torch.Tensor = None,
        t_tgt: torch.Tensor = None,
        cell_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = gene_ids.eq(self.pad_token_id)
        cls_mask = gene_ids.eq(self.cls_token_id)
        if mask_states is None:
            empty_mask = torch.zeros_like(padding_mask)
            mask_states = build_mask_states(padding_mask, empty_mask, cls_mask=cls_mask)
        if t is None:
            t = torch.ones((gene_ids.shape[0], 1), device=gene_ids.device, dtype=torch.float32)
        hidden = self.encode(
            gene_ids,
            expr_values,
            mask_states,
            padding_mask,
            t,
            t_src=t_src,
            t_tgt=t_tgt,
            cell_type_ids=cell_type_ids,
        )
        return hidden[:, 0, :]

    def forward(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        mask_states: torch.Tensor,
        padding_mask: torch.Tensor,
        t: torch.Tensor,
        t_src: torch.Tensor = None,
        t_tgt: torch.Tensor = None,
        cell_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden = self.encode(
            gene_ids,
            expr_values,
            mask_states,
            padding_mask,
            t,
            t_src=t_src,
            t_tgt=t_tgt,
            cell_type_ids=cell_type_ids,
        )
        pred = self.x1_head(hidden)
        return pred.masked_fill(padding_mask, 0.0)

    def forward_with_aux(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        mask_states: torch.Tensor,
        padding_mask: torch.Tensor,
        t: torch.Tensor,
        t_src: torch.Tensor = None,
        t_tgt: torch.Tensor = None,
        cell_type_ids: torch.Tensor = None,
        return_cls_attention: bool = False,
    ):
        cls_attention_stats = None
        if return_cls_attention:
            hidden, cls_attention_stats = self.encode(
                gene_ids,
                expr_values,
                mask_states,
                padding_mask,
                t,
                t_src=t_src,
                t_tgt=t_tgt,
                cell_type_ids=cell_type_ids,
                return_cls_attention=True,
            )
        else:
            hidden = self.encode(
                gene_ids,
                expr_values,
                mask_states,
                padding_mask,
                t,
                t_src=t_src,
                t_tgt=t_tgt,
                cell_type_ids=cell_type_ids,
            )
        pred = self.x1_head(hidden).masked_fill(padding_mask, 0.0)
        pred = pred.masked_fill(gene_ids.eq(self.cls_token_id), 0.0)
        bin_logits = None
        if self.bin_head is not None:
            bin_logits = self.bin_head(hidden)
            bin_logits = bin_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            bin_logits = bin_logits.masked_fill(gene_ids.eq(self.cls_token_id).unsqueeze(-1), 0.0)
        cell_emb = hidden[:, 0, :]
        type_logits = self.type_head(cell_emb) if self.type_head is not None else None
        pseudo_pred = self.pseudotime_head(cell_emb) if self.pseudotime_head is not None else None
        return pred, bin_logits, type_logits, pseudo_pred, cell_emb, cls_attention_stats

    def forward_train(
        self,
        gene_ids: torch.Tensor,
        x_1: torch.Tensor,
        padding_mask: torch.Tensor,
        mask_indicator: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor,
        t_src: torch.Tensor = None,
        t_tgt: torch.Tensor = None,
        cell_type_ids: torch.Tensor = None,
        return_cls_attention: bool = False,
    ):
        x_t = (1.0 - t) * x_0 + t * x_1
        input_expr = torch.where(mask_indicator, x_t, x_1)
        cls_mask = gene_ids.eq(self.cls_token_id)
        mask_states = build_mask_states(padding_mask, mask_indicator, cls_mask=cls_mask)
        x1_pred, bin_logits, type_logits, pseudo_pred, cell_emb, cls_attention_stats = self.forward_with_aux(
            gene_ids,
            input_expr,
            mask_states,
            padding_mask,
            t,
            t_src=t_src,
            t_tgt=t_tgt,
            cell_type_ids=cell_type_ids,
            return_cls_attention=return_cls_attention,
        )
        return x1_pred, x_t, bin_logits, type_logits, pseudo_pred, cell_emb, cls_attention_stats

    @torch.no_grad()
    def inference(
        self,
        gene_ids: torch.Tensor,
        observed_x1: torch.Tensor,
        padding_mask: torch.Tensor,
        mask_indicator: torch.Tensor,
        n_steps: int = 10,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        t_src: torch.Tensor = None,
        t_tgt: torch.Tensor = None,
        cell_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        x_t = observed_x1.clone()
        valid_mask = ~padding_mask
        cls_mask = gene_ids.eq(self.cls_token_id)
        masked_valid = mask_indicator & valid_mask & (~cls_mask)
        prior_std = max(float(prior_std), 1e-6)
        x_t[masked_valid] = torch.randn_like(x_t[masked_valid]) * prior_std + float(prior_mean)

        dt = 1.0 / max(int(n_steps), 1)
        eps = 1e-3

        for step in range(int(n_steps)):
            t = torch.full((gene_ids.shape[0], 1), step * dt, device=gene_ids.device, dtype=torch.float32)
            mask_states = build_mask_states(padding_mask, mask_indicator, cls_mask=cls_mask)
            x1_pred = self.forward(
                gene_ids,
                x_t,
                mask_states,
                padding_mask,
                t,
                t_src=t_src,
                t_tgt=t_tgt,
                cell_type_ids=cell_type_ids,
            )
            denom = (1.0 - t).clamp_min(eps)
            velocity = (x1_pred - x_t) / denom
            x_t[masked_valid] = x_t[masked_valid] + velocity[masked_valid] * dt
            x_t[masked_valid] = torch.clamp(x_t[masked_valid], min=0.0)
            x_t[valid_mask & (~mask_indicator)] = observed_x1[valid_mask & (~mask_indicator)]
            x_t[cls_mask] = 0.0
            x_t[padding_mask] = 0.0

        return x_t

    @torch.no_grad()
    def temporal_inference(
        self,
        gene_ids: torch.Tensor,
        src_expr: torch.Tensor,
        padding_mask: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        cell_type_ids: torch.Tensor = None,
        n_steps: int = 10,
    ) -> torch.Tensor:
        x_t = src_expr.clone()
        valid_mask = ~padding_mask
        cls_mask = gene_ids.eq(self.cls_token_id)
        evolve_mask = valid_mask & (~cls_mask)

        dt = 1.0 / max(int(n_steps), 1)
        eps = 1e-3
        full_mask_indicator = evolve_mask

        for step in range(int(n_steps)):
            t_flow = torch.full((gene_ids.shape[0], 1), step * dt, device=gene_ids.device, dtype=torch.float32)
            mask_states = build_mask_states(padding_mask, full_mask_indicator, cls_mask=cls_mask)
            tgt_pred = self.forward(
                gene_ids,
                x_t,
                mask_states,
                padding_mask,
                t_flow,
                t_src=t_src,
                t_tgt=t_tgt,
                cell_type_ids=cell_type_ids,
            )
            denom = (1.0 - t_flow).clamp_min(eps)
            velocity = (tgt_pred - x_t) / denom
            x_t[evolve_mask] = x_t[evolve_mask] + velocity[evolve_mask] * dt
            x_t[evolve_mask] = torch.clamp(x_t[evolve_mask], min=0.0)
            x_t[cls_mask] = 0.0
            x_t[padding_mask] = 0.0

        return x_t
