"""Unified encoder and pretraining heads.

Encoder: gene tokens (gene id + value) of many cells packed into one sequence,
attention isolated per cell, a prelude / looped core / coda stack of
attention + fine-grained MoE blocks with an N-BEATS backcast/forecast residual,
and a per-cell variational latent z read from the CLS token.

Heads on z: scVI-style batch-conditioned expression decoder (integration),
residual hierarchical classifiers (cell type, anatomy), and a neighbourhood
transformer that reconstructs masked spatial neighbours. Masked gene values
are read from the token states (gene MAE). With decoder = "unified", masked genes,
reconstruction and spatial neighbours all go through one batch- and task-conditioned
decoder (UnifiedExpressionDecoder).
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.profiler import record_function
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from . import deepinfra
from .config import ModelConfig

_flex = flex_attention


def enable_compiled_flex():
    global _flex
    # The gradient probe backpropagates each task separately with retain_graph, which
    # compiled backward graphs only allow when their buffers are not donated.
    import torch._functorch.config as functorch_config

    functorch_config.donated_buffer = False
    _flex = torch.compile(flex_attention, dynamic=True)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int, d_out: int | None = None):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_out or d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Bidirectional attention within each cell, with QK-norm."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.q_norm = nn.RMSNorm(self.dh)
        self.k_norm = nn.RMSNorm(self.dh)

    def forward(self, x, attn_mask):
        n = x.shape[0]
        with record_function("attn.qkv"):
            q, k, v = self.qkv(x).view(n, 3, self.n_heads, self.dh).unbind(1)
            q, k = self.q_norm(q), self.k_norm(k)
            if not isinstance(attn_mask, JaggedLayout):
                q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # (1, H, N, dh)
        with record_function("attn.core"):
            if isinstance(attn_mask, JaggedLayout):
                return self.out(jagged_attention(q, k, v, attn_mask).reshape(n, -1))
            if isinstance(attn_mask, torch.Tensor):  # dense boolean mask (CPU / tests)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                o = _flex(q, k, v, block_mask=attn_mask)
        with record_function("attn.out"):
            return self.out(o.squeeze(0).transpose(0, 1).reshape(n, -1))


def top_k_indices(v: torch.Tensor, k: int) -> torch.Tensor:
    """torch.topk(v, k).indices for a small last dim, as k fusible argmax passes (topk is not lowered by inductor)."""
    picks = []
    for _ in range(k):
        i = v.argmax(-1, keepdim=True)
        picks.append(i)
        v = v.scatter(-1, i, float("-inf"))
    return torch.cat(picks, -1)


class _PermuteRows(torch.autograd.Function):
    """y[idx] for a permutation idx; the gradient is gathered back with its inverse."""

    @staticmethod
    def forward(ctx, y, idx, inverse):
        ctx.save_for_backward(inverse)
        return y[idx]

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return grad[inverse], None, None


def permute_rows(y, idx, inverse):
    return _PermuteRows.apply(y, idx, inverse)


class MoE(nn.Module):
    """Shared experts plus top-k routed fine-grained experts (DeepSeek-V3/V4 style).

    Gates come from sqrt(softplus) (V4) or sigmoid (V3) scores. Load balancing
    is aux-loss-free: a per-expert bias, updated outside the gradient, shifts
    which experts are selected. With `hash_route` the experts are instead fixed
    per gene id (V4 hash routing, keyed by gene rather than by text token).
    `n_slots` gives a looped block its own router and bias per loop.
    """

    def __init__(self, d: int, cfg: ModelConfig, n_slots: int = 1, hash_route: bool = False):
        super().__init__()
        self.top_k, self.n_experts = cfg.top_k, cfg.n_experts
        self.backend, self.score_fn = cfg.moe_backend, cfg.router_score
        e, h = cfg.n_experts, cfg.expert_hidden
        # Routed experts as stacked tensors: w13 holds each expert's gate (w1) and up (w3)
        # projections for one fused grouped GEMM; `muon_split` tells Muon to orthogonalise
        # w1 and w3 as separate matrices, exactly as for separate nn.Linear weights.
        self.w13 = nn.Parameter(torch.empty(e, 2 * h, d).uniform_(-(d**-0.5), d**-0.5))
        self.w13.muon_split = 2
        self.w2 = nn.Parameter(torch.empty(e, d, h).uniform_(-(h**-0.5), h**-0.5))
        self.shared = SwiGLU(d, cfg.expert_hidden * cfg.n_shared_experts)
        self.routers = nn.ModuleList(nn.Linear(d, cfg.n_experts, bias=False) for _ in range(n_slots))
        self.register_buffer("route_bias", torch.zeros(n_slots, cfg.n_experts))
        self.register_buffer("load", torch.zeros(n_slots, cfg.n_experts))
        self.hash_route = hash_route
        if hash_route:  # k distinct experts per gene id (the last id is CLS), fixed by seed
            g = torch.Generator().manual_seed(0)
            table = torch.rand(cfg.n_genes + 1, cfg.n_experts, generator=g).argsort(-1)[:, : cfg.top_k]
            self.register_buffer("hash_table", table)

    def forward(self, x, slot: int = 0, gene=None):
        with record_function("moe.route"):
            logits = self.routers[slot](x).float()
            scores = torch.sqrt(F.softplus(logits)) if self.score_fn == "sqrtsoftplus" else torch.sigmoid(logits)
            if self.hash_route:
                idx = self.hash_table[gene]
            else:
                idx = top_k_indices(scores + self.route_bias[slot], self.top_k)
            gate = scores.gather(-1, idx)
            gate = (gate / gate.sum(-1, keepdim=True)).to(x.dtype)
            flat = idx.flatten()
            # Fixed-size count (no host sync, unlike bincount which must find the max first).
            counts = torch.zeros(self.n_experts, dtype=torch.long, device=x.device).scatter_add_(
                0, flat, torch.ones_like(flat)
            )
            if self.training:
                self.load[slot] += counts.float()
            order = torch.argsort(flat, stable=True)  # (token, slot) pairs grouped by expert
            inverse = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=x.device))
        with record_function("moe.dispatch"):
            # Gathers in both directions (the ScatterMoE idea, without its kernels): no
            # scatter or atomic add, so inductor fuses each with its neighbouring ops.
            n, k = x.shape[0], self.top_k
            x_sorted = permute_rows(x.unsqueeze(1).expand(n, k, x.shape[1]).reshape(n * k, -1), order, inverse)
        with record_function("moe.experts"):
            backend = self.backend or deepinfra.default_backend(x.device)
            y = deepinfra.grouped_swiglu_fused(x_sorted, self.w13, self.w2, counts, backend)
        with record_function("moe.shared"):
            shared = self.shared(x)
        with record_function("moe.combine"):
            routed = (permute_rows(y, inverse, order).view(n, k, -1) * gate.unsqueeze(-1).to(y.dtype)).sum(1)
            return shared + routed.to(shared.dtype)

    @torch.no_grad()
    def update_bias(self, rate: float):
        """Raise the bias of under-used experts, lower over-used ones; reset counts."""
        used = self.load.sum(-1, keepdim=True) > 0
        delta = torch.sign(self.load.mean(-1, keepdim=True) - self.load)
        self.route_bias += torch.where(used, rate * delta, torch.zeros_like(delta))
        self.load.zero_()


class Block(nn.Module):
    """Attention + MoE. With N-BEATS, the block's output f yields a backcast
    subtracted from its input and a forecast added to the running sum."""

    def __init__(self, cfg: ModelConfig, n_slots: int = 1, hash_route: bool = False):
        super().__init__()
        d = cfg.d_model
        self.nbeats = cfg.nbeats
        self.norm1, self.norm2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = Attention(d, cfg.n_heads)
        self.moe = MoE(d, cfg, n_slots, hash_route)
        self.recompute_moe = False  # set by the encoder for recompute_granularity="moe"
        if self.nbeats:
            self.backcast = nn.Linear(d, d, bias=False)
            self.forecast = nn.Linear(d, d, bias=False)

    def forward(self, x, total, attn_mask, gene, slot: int = 0):
        a = self.attn(self.norm1(x), attn_mask)
        if self.training and self.recompute_moe:
            # Keep attention activations; recompute only the MoE branch, which holds most of them.
            f = a + checkpoint(self._moe_branch, x, a, slot, gene, use_reentrant=False)
        else:
            f = a + self._moe_branch(x, a, slot, gene)
        if not self.nbeats:
            return x + f, total
        with record_function("nbeats"):
            return x - self.backcast(f), total + self.forecast(f)

    def _moe_branch(self, x, a, slot, gene):
        return self.moe(self.norm2(x + a), slot, gene)


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.gene_emb = nn.Embedding(cfg.n_genes + 1, d)  # last row is the CLS token
        self.value_emb = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.mask_emb = nn.Parameter(torch.zeros(d))
        if cfg.input_norm == "pearson":  # set from the pretraining set (train.py), saved with the weights
            self.register_buffer("gene_rate", torch.zeros(cfg.n_genes + 1, dtype=torch.float64))
            self.register_buffer("residual_clip", torch.tensor(1.0, dtype=torch.float64))
        elif cfg.input_norm != "log1p":
            raise ValueError(f"input_norm {cfg.input_norm!r}; valid: log1p, pearson")
        self.prelude = nn.ModuleList(Block(cfg, hash_route=i < cfg.n_hash_layers) for i in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg, n_slots=cfg.max_loops) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.inject = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.inject.weight)
        self.final_norm = nn.RMSNorm(d)
        nn.init.normal_(self.mask_emb, std=0.02)

    def pearson_residuals(self, b) -> torch.Tensor:
        """tok_in (log1p CP10K of the cell's library) as analytic Pearson residuals under gene_rate."""
        n_cells = len(b["cnt_ptr"]) - 1
        cell = torch.repeat_interleave(torch.arange(n_cells, device=b["cnt_val"].device), b["cnt_ptr"].diff())
        lib = torch.zeros(n_cells, dtype=torch.float64, device=cell.device).index_add_(0, cell, b["cnt_val"].double())
        n = lib[b["tok_doc"].clamp(min=0)]
        x = torch.expm1(b["tok_in"].double()) * n / 1e4
        mu = n * self.gene_rate[b["tok_gene"]]
        sd = torch.sqrt(mu + mu * mu / 100.0)
        r = torch.where(sd > 0, (x - mu) / sd.clamp(min=1e-12), torch.zeros_like(mu))
        return r.clamp(-self.residual_clip, self.residual_clip).to(b["tok_in"].dtype)

    def embed(self, b):
        tok_in = self.pearson_residuals(b) if self.cfg.input_norm == "pearson" else b["tok_in"]
        value = self.value_emb(tok_in.unsqueeze(-1))
        value = torch.where(b["tok_masked"].unsqueeze(-1), self.mask_emb.to(value.dtype), value)
        value = value * (~b["tok_cls"]).unsqueeze(-1)
        return self.gene_emb(b["tok_gene"]) + value

    def run_block(self, blk, x, total, attn_mask, gene, slot: int = 0, core: bool = False):
        scope = self.cfg.recompute_scope
        recompute = self.cfg.activation_checkpointing and (scope == "all" or (scope == "core" and core))
        blk.recompute_moe = recompute and self.cfg.recompute_granularity == "moe"
        if not (self.training and recompute) or blk.recompute_moe:
            return blk(x, total, attn_mask, gene, slot)
        # Recomputing a block in backward repeats its routing, so MoE load counts are
        # recorded twice; the doubling is uniform and leaves bias updates unchanged.
        # Selective checkpointing keeps the attention kernel's outputs, so backward
        # recomputes only the cheaper projections and MoE.
        context = SAVE_ATTENTION if self.cfg.selective_checkpointing else None
        kwargs = {"context_fn": context} if context else {}
        return checkpoint(blk, x, total, attn_mask, gene, slot, use_reentrant=False, **kwargs)

    def forward(self, b, attn_mask, loops: int):
        e, gene = self.embed(b), b["tok_gene"]
        x, total = e, torch.zeros_like(e)
        for blk in self.prelude:
            x, total = self.run_block(blk, x, total, attn_mask, gene)
        for t in range(loops):
            if self.cfg.loop_injection:
                x = x + self.inject(e)
            for blk in self.core:
                x, total = self.run_block(blk, x, total, attn_mask, gene, slot=t, core=True)
        for blk in self.coda:
            x, total = self.run_block(blk, x, total, attn_mask, gene)
        return self.final_norm(total if self.cfg.nbeats else x)

    def moe_layers(self):
        return [blk.moe for blk in (*self.prelude, *self.core, *self.coda)]


class HierarchicalHead(nn.Module):
    """scBrain-style residual chain: one SwiGLU step and classifier per level, coarse to fine."""

    def __init__(self, d_in: int, hidden: int, level_sizes: list[int]):
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.steps = nn.ModuleList(SwiGLU(hidden, hidden) for _ in level_sizes)
        self.heads = nn.ModuleList(nn.Linear(hidden, n) for n in level_sizes)

    def forward(self, z):
        h, logits = self.proj(z), []
        for step, head in zip(self.steps, self.heads):
            h = h + step(h)
            logits.append(head(h))
        return logits


class ExpressionDecoder(nn.Module):
    """p(x | z, batch): gene proportions over the vocabulary, as in scVI."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.batch_emb = nn.Embedding(cfg.n_batches, cfg.z_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * cfg.z_dim, cfg.decoder_hidden), nn.LayerNorm(cfg.decoder_hidden), nn.SiLU()
        )
        self.out = nn.Linear(cfg.decoder_hidden, cfg.n_genes)
        self.log_theta = nn.Parameter(torch.zeros(cfg.n_genes))  # NB inverse dispersion per gene

    def forward(self, z, batch, task: str = "reconstruct"):
        return self.out(self.mlp(torch.cat([z, self.batch_emb(batch)], -1)))


class UnifiedExpressionDecoder(nn.Module):
    """One decoder for every expression output, conditioned on batch and task.

    Inputs come from the latent (reconstruction, spatial) or from a masked gene's token state
    (imputation), are projected to a shared width, modulated by FiLM of batch + task embeddings
    (zero-initialised, so it starts unconditioned) and passed through one shared residual MLP.
    Every output is read against the same gene output embeddings: logits over all genes for the
    NB likelihood, or one value per masked token for that token's gene. Because each output sees the
    batch, the encoder need not keep batch to predict expression (the gene head of the separate
    design did not see it).
    """

    TASKS = ("reconstruct", "spatial", "impute")

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.decoder_hidden
        self.from_latent = nn.Linear(cfg.z_dim, hidden)
        self.from_token = nn.Linear(cfg.d_model, hidden)
        self.batch_emb = nn.Embedding(cfg.n_batches, hidden)
        self.task_emb = nn.Embedding(len(self.TASKS), hidden)
        self.norm = nn.LayerNorm(hidden)
        self.film = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, cfg.n_genes)  # gene output embeddings, shared by every task
        self.value_scale = nn.Parameter(torch.ones(()))
        self.log_theta = nn.Parameter(torch.zeros(cfg.n_genes))  # NB inverse dispersion per gene

    def hidden(self, x, batch, task: str):
        cond = self.batch_emb(batch) + self.task_emb.weight[self.TASKS.index(task)]
        gamma, beta = self.film(cond).chunk(2, -1)
        h = self.norm(x) * (1 + gamma) + beta
        return h + self.mlp(h)

    def forward(self, z, batch, task: str = "reconstruct"):
        """Logits over all genes for latents `z` (reconstruction or spatial)."""
        return self.out(self.hidden(self.from_latent(z), batch, task))

    def impute(self, h_tok, batch, genes):
        """One value (log1p CP10K) per masked token, read against its gene's output embedding."""
        h = self.hidden(self.from_token(h_tok), batch, "impute")
        return self.value_scale * (h * self.out.weight[genes]).sum(-1) + self.out.bias[genes]


class NeighborhoodModel(nn.Module):
    """Predicts the latents of masked cells from the visible cells of a spatial group."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.spatial_d
        self.inp = nn.Linear(cfg.z_dim, d)
        self.pos = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.mask_emb = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, cfg.spatial_layers)
        self.out = nn.Linear(d, cfg.z_dim)
        self.residual_mean = cfg.spatial_residual_mean
        if self.residual_mean:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, z_members, rel, masked, pad):
        h = torch.where(masked.unsqueeze(-1), self.mask_emb, self.inp(z_members)) + self.pos(rel)
        out = self.out(self.tf(h, src_key_padding_mask=pad))
        if self.residual_mean:  # every member's prediction is a correction to the visible members' mean
            seen = (~masked & ~pad).unsqueeze(-1).to(z_members.dtype)
            out = out + ((z_members * seen).sum(1, keepdim=True) / seen.sum(1, keepdim=True).clamp(min=1)).to(out.dtype)
        return out


class GradReverse(torch.autograd.Function):
    """Identity forward; backward multiplies the gradient by -coeff (Ganin & Lempitsky 2015)."""

    @staticmethod
    def forward(ctx, x, coeff: float):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


class BatchDiscriminator(nn.Module):
    """Predicts a cell's integration batch from its latent; trained against the encoder by gradient reversal."""

    def __init__(self, z_dim: int, hidden: int, n_batches: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_batches))

    def forward(self, z: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
        return self.net(GradReverse.apply(z, coeff))


class MoGPrior(nn.Module):
    """Learned mixture-of-Gaussians prior over z (a VampPrior-like replacement for N(0, I))."""

    def __init__(self, n_components: int, z_dim: int):
        super().__init__()
        self.means = nn.Parameter(torch.randn(n_components, z_dim))
        self.logvars = nn.Parameter(torch.zeros(n_components, z_dim))
        self.logits = nn.Parameter(torch.zeros(n_components))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float().unsqueeze(1)  # (n, 1, z)
        var = self.logvars.exp()
        comp = -0.5 * (((z - self.means) ** 2) / var + self.logvars + math.log(2 * math.pi)).sum(-1)
        return torch.logsumexp(comp + torch.log_softmax(self.logits, 0), dim=1)


class UnifiedModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoder = Encoder(cfg)
        if cfg.decoder == "separate":
            self.gene_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        elif cfg.decoder != "unified":
            raise ValueError(f"decoder {cfg.decoder!r}; valid: separate, unified")
        self.to_latent = nn.Linear(d, 2 * cfg.z_dim)
        if cfg.latent_reg == "mog":
            self.prior = MoGPrior(cfg.mog_components, cfg.z_dim)
        if cfg.residual_dim:
            self.to_residual = nn.Linear(d, cfg.residual_dim)
            self.res_proj = nn.Linear(cfg.residual_dim, cfg.z_dim, bias=False)
            nn.init.zeros_(self.res_proj.weight)
        self.decoder = ExpressionDecoder(cfg) if cfg.decoder == "separate" else UnifiedExpressionDecoder(cfg)
        self.cell_type_head = HierarchicalHead(cfg.z_dim, d, cfg.cell_type_levels)
        self.anatomy_head = HierarchicalHead(cfg.z_dim, d, cfg.anatomy_levels)
        self.neighborhood = NeighborhoodModel(cfg)
        if cfg.batch_adversary:
            self.batch_adversary = BatchDiscriminator(cfg.z_dim, cfg.batch_adversary, cfg.n_batches)
        if cfg.compile_blocks:
            # Regional compilation (as in torchtitan): each encoder block is compiled on its own,
            # with dynamic shapes since token counts and cell lengths change every step.
            # All blocks share Block.forward, and each (hash routing, loop slot, recompute, MoE load
            # buffer shape) variant is its own graph: ~20 with core-only MoE recompute, more than 64 with
            # whole-block recompute of every block. The limit must never be hit: past it a variant runs
            # eager, and a block recomputed in backward then saves different tensors than its compiled
            # forward did (CheckpointError), so the limits are set well above any configuration's count.
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 1024)
            torch._dynamo.config.accumulated_recompile_limit = max(
                torch._dynamo.config.accumulated_recompile_limit, 16384
            )
            for blk in (*self.encoder.prelude, *self.encoder.core, *self.encoder.coda):
                blk.compile(dynamic=True)

    def forward(self, b, attn_mask, loops: int | None = None):
        loops = loops or self.cfg.loops
        h = self.encoder(b, attn_mask, loops)
        cls = h[b["tok_cls"]]
        mu, logvar = self.to_latent(cls).float().chunk(2, -1)
        logvar = logvar.clamp(-10, 10)
        stochastic = self.training and self.cfg.latent_reg != "none"
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if stochastic else mu
        out = {
            "gene_pred": self.impute(h, b),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }
        if self.cfg.residual_dim:
            out["residual"] = self.to_residual(cls).float()
        return out

    def impute(self, h, b) -> torch.Tensor:
        """Predicted values of the masked tokens, from the encoder's token states `h`."""
        masked = b["tok_masked"]
        if self.cfg.decoder == "separate":
            return self.gene_head(h[masked]).squeeze(-1).float()
        cell_batch = b["batch"][b["tok_doc"][masked]]
        return self.decoder.impute(h[masked], cell_batch, b["tok_gene"][masked]).float()

    def recon_input(self, out: dict) -> torch.Tensor:
        """What the expression decoder reads: z, or detached mu, plus the residual latent if any."""
        base = out["mu"].detach() if self.cfg.recon_detach_mu else out["z"]
        if self.cfg.residual_dim:
            base = base + self.res_proj(out["residual"])
        return base


@dataclass
class JaggedLayout:
    """Cell boundaries for variable-length attention: `offsets` (cells + 1) over the first `n_real` tokens."""

    offsets: torch.Tensor
    n_real: int
    max_len: int = 0
    kernel: str = "sdpa"  # "sdpa" (nested jagged tensors) or "fa3" (FlashAttention-3 varlen)


def jagged_layout(doc: torch.Tensor, kernel: str = "sdpa") -> JaggedLayout:
    """Cells are contiguous runs of `doc`; padding (-1) sits at the end and is left out."""
    lengths = torch.bincount(doc[doc >= 0])
    offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    return JaggedLayout(offsets, int(offsets[-1]), int(lengths.max()), kernel)


def jagged_attention(q, k, v, layout: JaggedLayout):
    """Per-cell attention through nested jagged tensors and SDPA (FlashAttention varlen).

    q, k, v are (N, heads, dh) packed tokens; padding rows past `n_real` get zeros.
    """
    nr = layout.n_real
    if layout.kernel == "fa3":
        from flash_attn_interface import flash_attn_varlen_func

        cu = layout.offsets.to(torch.int32)
        # FA3 runs in bf16; under autocast q and k arrive in fp32 from their RMSNorm.
        qb, kb, vb = (t[:nr].bfloat16() for t in (q, k, v))
        out = flash_attn_varlen_func(qb, kb, vb, cu, cu, layout.max_len, layout.max_len)
        out = (out[0] if isinstance(out, tuple) else out).to(q.dtype)
    else:
        nested = [torch.nested.nested_tensor_from_jagged(t[:nr], layout.offsets).transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(*nested).transpose(1, 2).values()
    return torch.cat([out, out.new_zeros(q.shape[0] - nr, *out.shape[1:])]) if q.shape[0] > nr else out


_ATTENTION_OPS = {
    torch.ops.aten._flash_attention_forward.default,
    torch.ops.aten._efficient_attention_forward.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
}


try:  # FlashAttention-3 registers its forward as a custom op when imported
    import flash_attn_interface  # noqa: F401

    _ATTENTION_OPS.add(torch.ops.flash_attn_3._flash_attn_forward.default)
except ImportError:
    pass


def _save_attention(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if op in _ATTENTION_OPS else CheckpointPolicy.PREFER_RECOMPUTE


SAVE_ATTENTION = partial(create_selective_checkpoint_contexts, _save_attention)


def attention_layout(doc: torch.Tensor, backend: str):
    """What Attention needs for this packed batch: a FlexAttention BlockMask or a JaggedLayout."""
    if backend in ("jagged", "fa3"):
        return jagged_layout(doc, "fa3" if backend == "fa3" else "sdpa")
    return block_mask(doc)


def dense_mask(doc: torch.Tensor) -> torch.Tensor:
    """(1, 1, N, N) boolean mask allowing attention only within the same cell."""
    return (doc[:, None] == doc[None, :]).unsqueeze(0).unsqueeze(0)


def block_mask(doc: torch.Tensor, block: int = 128):
    """FlexAttention block mask for the packed sequence, built from cell boundaries.

    Cells are contiguous runs of equal `doc` (padding, -1, sits at the end), so the
    key blocks a query block needs form one contiguous range: from the block holding
    the start of its first cell to the block holding the end of its last cell. This
    is O(number of blocks) per query block, unlike evaluating the mask over all N x N
    positions. `doc` must be padded to a multiple of `block`.
    """
    n = doc.shape[0]
    n_blocks = n // block
    d = torch.where(doc < 0, doc.max() + 1, doc)
    start = torch.searchsorted(d, d, right=False)
    end = torch.searchsorted(d, d, right=True)
    first = start[::block] // block
    last = (end[block - 1 :: block] - 1) // block
    num = last - first + 1
    # FlexAttention expects the index table to span every key block; entries past `num` are ignored.
    idx = torch.minimum(first[:, None] + torch.arange(n_blocks, device=doc.device), last[:, None])

    def same_cell(b, h, q, kv):
        return doc[q] == doc[kv]

    return BlockMask.from_kv_blocks(
        num.view(1, 1, -1).int(),
        idx.view(1, 1, n_blocks, n_blocks).int(),
        BLOCK_SIZE=block,
        mask_mod=same_cell,
        seq_lengths=(n, n),
    )


def pad_tokens(b: dict, multiple: int = 128) -> dict:
    """Pad the token arrays so the packed length is a multiple of `multiple`.

    Padding tokens form their own document (-1), are never masked or CLS, and
    use the CLS gene id with zero value, so they influence nothing.
    """
    n = b["tok_gene"].shape[0]
    pad = (-n) % multiple
    if not pad:
        return b
    out = dict(b)
    fill = {
        "tok_gene": b["tok_gene"].max().item(),
        "tok_in": 0.0,
        "tok_target": 0.0,
        "tok_masked": False,
        "tok_cls": False,
        "tok_doc": -1,
    }
    for k, v in fill.items():
        out[k] = torch.cat([b[k], torch.full((pad,), v, dtype=b[k].dtype, device=b[k].device)])
    return out


def param_groups(model: nn.Module):
    """Hidden matrices of the encoder blocks (2-D, or stacked 3-D experts) go to Muon; the rest to AdamW."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        in_blocks = name.startswith(("encoder.prelude", "encoder.core", "encoder.coda"))
        if in_blocks and p.ndim in (2, 3) and "routers" not in name:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def upgrade_state_dict(state: dict) -> dict:
    """Checkpoints from before stacked experts: moe.experts.<e>.w{1,2,3}.weight -> moe.w13, moe.w2."""
    out, experts = {}, {}
    for key, value in state.items():
        prefix, sep, rest = key.partition(".moe.experts.")
        if not sep:
            out[key] = value
            continue
        e, name = rest.split(".")[:2]
        experts.setdefault(prefix, {})[(int(e), name)] = value
    for prefix, w in experts.items():
        n = 1 + max(e for e, _ in w)
        out[f"{prefix}.moe.w13"] = torch.stack([torch.cat([w[(e, "w1")], w[(e, "w3")]]) for e in range(n)])
        out[f"{prefix}.moe.w2"] = torch.stack([w[(e, "w2")] for e in range(n)])
    return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
