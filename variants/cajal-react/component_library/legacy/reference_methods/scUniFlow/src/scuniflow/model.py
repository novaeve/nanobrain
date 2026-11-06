import math

import numpy as np
import torch
import torch.nn as nn



def _hilbert_distances(rows: np.ndarray, cols: np.ndarray, grid_size: int) -> np.ndarray:
    """Compute Hilbert distances for 2-D integer grid coordinates without extra deps."""
    x = rows.astype(np.int64, copy=True)
    y = cols.astype(np.int64, copy=True)

    order = int(math.ceil(math.log2(max(grid_size, 2))))
    side = 1 << order
    d = np.zeros_like(x, dtype=np.int64)
    s = side // 2

    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += (s * s) * ((3 * rx) ^ ry)

        swap_mask = ry == 0
        flip_mask = swap_mask & (rx == 1)

        if np.any(flip_mask):
            x[flip_mask] = (side - 1) - x[flip_mask]
            y[flip_mask] = (side - 1) - y[flip_mask]

        if np.any(swap_mask):
            tmp = x[swap_mask].copy()
            x[swap_mask] = y[swap_mask]
            y[swap_mask] = tmp

        s //= 2

    return d


def _ordered_slot_indices(grid_size: int, order_mode: str = "hilbert"):
    n_slots = grid_size * grid_size
    slot_lin = np.arange(n_slots, dtype=np.int64)
    slot_rows = slot_lin // grid_size
    slot_cols = slot_lin % grid_size

    if order_mode == "hilbert":
        slot_key = _hilbert_distances(slot_rows, slot_cols, grid_size)
        slot_order = np.lexsort((slot_lin, slot_cols, slot_rows, slot_key))
    elif order_mode == "row_major":
        slot_order = np.lexsort((slot_lin, slot_cols, slot_rows))
    else:
        raise ValueError(f"Unsupported slot order_mode: {order_mode}")

    return slot_order, slot_rows, slot_cols


def build_ordered_gene_coords(n_genes: int, grid_size: int, order_mode: str = "hilbert"):
    """
    Generate deterministic 2-D coordinates that preserve the current gene order.

    Gene i is assigned to the i-th slot under the chosen grid traversal order.
    This is useful when we want a coordinate-based hard router without relying
    on an external protein-network spectral embedding.
    """
    n_slots = grid_size * grid_size
    if n_genes > n_slots:
        raise ValueError(
            f"Grid {grid_size}x{grid_size} has only {n_slots} slots for {n_genes} genes"
        )

    slot_order, slot_rows, slot_cols = _ordered_slot_indices(
        grid_size=grid_size,
        order_mode=order_mode,
    )
    chosen = slot_order[:n_genes]
    return np.stack([slot_rows[chosen], slot_cols[chosen]], axis=1).astype(np.float32)


def build_hard_permutation_from_coords(
    gene_coords_grid,
    grid_size: int,
    slot_order_mode: str = "hilbert",
    gene_order_mode: str = "hilbert",
):
    """
    Turn continuous spectral coordinates into a fixed slot assignment.

    We sort genes and grid slots by Hilbert distance so nearby spectral points
    stay nearby on the 2-D grid while producing a strict one-gene-per-slot map.
    """
    coords = np.asarray(gene_coords_grid, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"gene_coords_grid must have shape (G, 2), got {coords.shape}")

    n_genes = coords.shape[0]
    n_slots = grid_size * grid_size
    if n_genes > n_slots:
        raise ValueError(
            f"Grid {grid_size}x{grid_size} has only {n_slots} slots for {n_genes} genes"
        )

    rows = np.clip(np.rint(coords[:, 0]).astype(np.int64), 0, grid_size - 1)
    cols = np.clip(np.rint(coords[:, 1]).astype(np.int64), 0, grid_size - 1)
    if gene_order_mode == "hilbert":
        gene_key = _hilbert_distances(rows, cols, grid_size)
        gene_order = np.lexsort((np.arange(n_genes), coords[:, 1], coords[:, 0], gene_key))
    elif gene_order_mode == "row_major":
        gene_order = np.lexsort((np.arange(n_genes), coords[:, 1], coords[:, 0]))
    else:
        raise ValueError(f"Unsupported gene order_mode: {gene_order_mode}")

    slot_order, _, _ = _ordered_slot_indices(
        grid_size=grid_size,
        order_mode=slot_order_mode,
    )

    slot_to_gene = np.full((n_slots,), -1, dtype=np.int64)
    slot_to_gene[slot_order[:n_genes]] = gene_order

    gene_to_slot = np.empty((n_genes,), dtype=np.int64)
    gene_to_slot[gene_order] = slot_order[:n_genes]
    return slot_to_gene, gene_to_slot


class SpectralRouter(nn.Module):
    """
    Fixed hard router derived from spectral coordinates.

    The routing itself is non-trainable: every global gene id maps to exactly one
    slot on the grid. The trainable part is the gene identity embedding looked up
    after routing.
    """

    def __init__(self, n_genes=30000, d_model=512, grid_size=174):
        super().__init__()
        self.D = d_model
        self.L = grid_size * grid_size
        self.G = n_genes
        self.grid_size = grid_size

        self.gene_embedding = nn.Embedding(self.G, self.D)
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)

        slot_to_gene = torch.full((self.L,), -1, dtype=torch.long)
        slot_to_gene[: self.G] = torch.arange(self.G, dtype=torch.long)
        gene_to_slot = torch.arange(self.G, dtype=torch.long)
        slot_is_real = (slot_to_gene >= 0).to(torch.float32)

        self.register_buffer("slot_to_gene", slot_to_gene)
        self.register_buffer("gene_to_slot", gene_to_slot)
        self.register_buffer("slot_is_real", slot_is_real)

    def init_from_spectral(
        self,
        gene_coords_grid,
        slot_order_mode: str = "hilbert",
        gene_order_mode: str = "hilbert",
    ):
        """
        Freeze routing from spectral coordinates and initialize gene embeddings
        with Fourier features of the same coordinates.
        """
        slot_to_gene_np, gene_to_slot_np = build_hard_permutation_from_coords(
            gene_coords_grid,
            grid_size=self.grid_size,
            slot_order_mode=slot_order_mode,
            gene_order_mode=gene_order_mode,
        )

        device = self.gene_embedding.weight.device
        coords = torch.as_tensor(gene_coords_grid, dtype=torch.float32, device=device)
        freqs = 1.0 / (
            10000
            ** (
                torch.arange(self.D // 4, dtype=torch.float32, device=device)
                / max(self.D // 4, 1)
            )
        )

        c = coords / float(self.grid_size) * math.pi
        emb = torch.zeros(coords.shape[0], self.D, device=device)
        ax = c[:, 0:1] * freqs.unsqueeze(0)
        ay = c[:, 1:2] * freqs.unsqueeze(0)
        emb[:, 0::4] = ax.sin()
        emb[:, 1::4] = ax.cos()
        emb[:, 2::4] = ay.sin()
        emb[:, 3::4] = ay.cos()

        with torch.no_grad():
            self.slot_to_gene.copy_(torch.from_numpy(slot_to_gene_np).to(self.slot_to_gene.device))
            self.gene_to_slot.copy_(torch.from_numpy(gene_to_slot_np).to(self.gene_to_slot.device))
            self.slot_is_real.copy_((self.slot_to_gene >= 0).to(torch.float32))
            self.gene_embedding.weight.copy_(emb)

    def route_scalar(self, gene_ids, scalar_val):
        """Hard-route per-gene scalars from input column order to fixed grid slots."""
        B, G = scalar_val.shape
        out = scalar_val.new_zeros(B, self.L)
        slot_idx = self.gene_to_slot[gene_ids]
        out.scatter_(1, slot_idx, scalar_val)
        return out

    def inverse_route(self, routed_context, gene_ids):
        """Map grid features back to the original input column order."""
        idx = self.gene_to_slot[gene_ids]
        if routed_context.ndim == 2:
            return routed_context.gather(1, idx)
        idx = idx.unsqueeze(-1).expand(-1, -1, routed_context.shape[-1])
        return routed_context.gather(1, idx)

    def routed_gene_embeddings(self, batch_size: int):
        safe_gene_ids = self.slot_to_gene.clamp_min(0)
        gene_emb = self.gene_embedding(safe_gene_ids)
        gene_emb = gene_emb * self.slot_is_real.unsqueeze(-1)
        return gene_emb.unsqueeze(0).expand(batch_size, -1, -1)


class VectorFieldNet(nn.Module):
    def __init__(self, d_model=512, d_hidden=512):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )
        self.binary_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_hidden // 2),
            nn.SiLU(),
            nn.LayerNorm(d_hidden // 2),
            nn.Linear(d_hidden // 2, 1),
        )

    def _build_input(self, x_t, context, t):
        B, n_mask = x_t.shape

        if len(t.shape) == 1:
            if t.shape[0] == 1:
                t = t.expand(B, 1)
            else:
                t = t.view(B, 1)

        t_emb = self.time_embed(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, n_mask, -1)

        return torch.cat([context, t_emb, x_t.unsqueeze(-1)], dim=-1)

    def forward(self, x_t, context, t):
        x_in = self._build_input(x_t, context, t)
        x1_pred = self.mlp(x_in)
        return x1_pred.squeeze(-1)

    def binary_forward(self, x_t, context, t):
        x_in = self._build_input(x_t, context, t)
        binary_logit = self.binary_mlp(x_in)
        return binary_logit.squeeze(-1)


class scRNAModel(nn.Module):
    def __init__(
        self,
        n_genes=30000,
        d_model=512,
        grid_size=174,
        n_layers=8,
        d_state_row=64,
        d_state_col=16,
        activation_checkpointing=True,
    ):
        super().__init__()
        # Keep the independent router/velocity components importable without
        # requiring CUDA Mamba kernels. The full encoder still requires Mamba.
        from gene_mamba import GeneMamba
        self.router = SpectralRouter(
            n_genes=n_genes,
            d_model=d_model,
            grid_size=grid_size,
        )
        self.mamba_encoder = GeneMamba(
            n_genes=n_genes,
            grid_size=grid_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state_row=d_state_row,
            d_state_col=d_state_col,
            activation_checkpointing=activation_checkpointing,
        )
        self.vector_field = VectorFieldNet(d_model=d_model)

    def init_from_spectral(
        self,
        gene_coords_grid,
        slot_order_mode: str = "hilbert",
        gene_order_mode: str = "hilbert",
    ):
        self.router.init_from_spectral(
            gene_coords_grid,
            slot_order_mode=slot_order_mode,
            gene_order_mode=gene_order_mode,
        )

    def compute_context(self, gene_ids, expr_values, mask_indicator, temperature=1.0):
        """
        Build Mamba context under a fixed hard spectral routing.

        `temperature` is kept for backward-compatible call sites, but routing is
        now deterministic and ignores it.
        """
        del temperature
        B = expr_values.shape[0]
        gene_emb = self.router.routed_gene_embeddings(B)
        expr_permuted = self.router.route_scalar(gene_ids, expr_values)
        mask_permuted = self.router.route_scalar(gene_ids, mask_indicator)

        mamba_out = self.mamba_encoder(gene_emb, expr_permuted, mask_permuted)
        context = self.router.inverse_route(mamba_out, gene_ids)
        return context, expr_permuted

    def forward_train(self, gene_ids, x_1, mask_indicator, t, x_0, temperature=1.0):
        x_t = (1 - t) * x_0 + t * x_1
        mask_features = torch.where(mask_indicator == 1, x_t, x_1)

        context, expr_permuted = self.compute_context(
            gene_ids,
            mask_features,
            mask_indicator,
            temperature=temperature,
        )

        B, _ = mask_indicator.shape
        predictions = []
        targets = []
        x1_hats = []
        x1_targets = []
        for b in range(B):
            m_idx = mask_indicator[b].nonzero(as_tuple=True)[0]
            ctx_m = context[b, m_idx, :].unsqueeze(0)
            xt_m = x_t[b, m_idx].unsqueeze(0)

            t_b = t[b].unsqueeze(0)
            x1_pred = self.vector_field(xt_m, ctx_m, t_b)
            x1_true = x_1[b, m_idx]

            predictions.append(x1_pred.squeeze(0))
            targets.append(x1_true)
            x1_hats.append(x1_pred.squeeze(0))
            x1_targets.append(x1_true)

        return predictions, targets, x1_hats, x1_targets, expr_permuted

    def forward_train_velocity(self, gene_ids, x_1, mask_indicator, t, x_0, temperature=1.0):
        """
        Ablation path: train the vector field to predict velocity directly
        instead of predicting x_1 and deriving velocity during rollout.
        """
        x_t = (1 - t) * x_0 + t * x_1
        mask_features = torch.where(mask_indicator == 1, x_t, x_1)

        context, expr_permuted = self.compute_context(
            gene_ids,
            mask_features,
            mask_indicator,
            temperature=temperature,
        )

        B, _ = mask_indicator.shape
        predictions = []
        targets = []
        v_hats = []
        v_targets = []
        for b in range(B):
            m_idx = mask_indicator[b].nonzero(as_tuple=True)[0]
            ctx_m = context[b, m_idx, :].unsqueeze(0)
            xt_m = x_t[b, m_idx].unsqueeze(0)

            t_b = t[b].unsqueeze(0)
            v_pred = self.vector_field(xt_m, ctx_m, t_b)
            v_true = x_1[b, m_idx] - x_0[b, m_idx]

            predictions.append(v_pred.squeeze(0))
            targets.append(v_true)
            v_hats.append(v_pred.squeeze(0))
            v_targets.append(v_true)

        return predictions, targets, v_hats, v_targets, expr_permuted

    def forward_train_dual(self, gene_ids, x_1, mask_indicator, t, x_0, temperature=1.0):
        x_t = (1 - t) * x_0 + t * x_1
        mask_features = torch.where(mask_indicator == 1, x_t, x_1)
        observed_features = torch.where(mask_indicator == 1, torch.zeros_like(x_1), x_1)

        context, expr_permuted = self.compute_context(
            gene_ids,
            mask_features,
            mask_indicator,
            temperature=temperature,
        )
        binary_context, _ = self.compute_context(
            gene_ids,
            observed_features,
            mask_indicator,
            temperature=temperature,
        )

        B, _ = mask_indicator.shape
        predictions = []
        binary_logits = []
        targets = []
        x1_hats = []
        x1_targets = []
        for b in range(B):
            m_idx = mask_indicator[b].nonzero(as_tuple=True)[0]
            ctx_m = context[b, m_idx, :].unsqueeze(0)
            binary_ctx_m = binary_context[b, m_idx, :].unsqueeze(0)
            xt_m = x_t[b, m_idx].unsqueeze(0)
            observed_m = observed_features[b, m_idx].unsqueeze(0)

            t_b = t[b].unsqueeze(0)
            t_binary = torch.zeros_like(t_b)
            x1_pred = self.vector_field(xt_m, ctx_m, t_b)
            binary_logit = self.vector_field.binary_forward(observed_m, binary_ctx_m, t_binary)
            x1_true = x_1[b, m_idx]

            predictions.append(x1_pred.squeeze(0))
            binary_logits.append(binary_logit.squeeze(0))
            targets.append(x1_true)
            x1_hats.append(x1_pred.squeeze(0))
            x1_targets.append(x1_true)

        return predictions, binary_logits, targets, x1_hats, x1_targets, expr_permuted

    def predict_binary_logits(self, gene_ids, expr_values, mask_indicator, t, temperature=1.0):
        context, _ = self.compute_context(
            gene_ids,
            expr_values,
            mask_indicator,
            temperature=temperature,
        )

        B, G = mask_indicator.shape
        out = expr_values.new_zeros(B, G)
        for b in range(B):
            m_idx = mask_indicator[b].nonzero(as_tuple=True)[0]
            if m_idx.numel() == 0:
                continue
            ctx_m = context[b, m_idx, :].unsqueeze(0)
            xt_m = expr_values[b, m_idx].unsqueeze(0)
            t_b = t[b].unsqueeze(0)
            out[b, m_idx] = self.vector_field.binary_forward(xt_m, ctx_m, t_b).squeeze(0)
        return out

    def inference(
        self,
        gene_ids,
        known_genes,
        mask_indicator,
        n_steps=10,
        temperature=0.01,
        x1_true=None,
        debug_step0_stats=False,
        prior_mean=0.0,
        prior_std=1.0,
        velocity_bias=0.0,
        init_from_x1_true=False,
        initial_masked_values=None,
    ):
        del temperature
        B, G = known_genes.shape
        dt = 1.0 / n_steps
        eps = 1e-3

        x_t = known_genes.clone()
        mask_bool = mask_indicator == 1
        if init_from_x1_true and x1_true is not None:
            x_t[mask_bool] = x1_true[mask_bool]
        elif initial_masked_values is not None:
            x_t[mask_bool] = initial_masked_values[mask_bool]
        else:
            prior_std = max(float(prior_std), 1e-6)
            x_t[mask_bool] = torch.randn_like(x_t[mask_bool]) * prior_std + float(prior_mean)

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=known_genes.device)

            context, _ = self.compute_context(gene_ids, x_t, mask_indicator)

            for b in range(B):
                m_idx = mask_indicator[b].nonzero(as_tuple=True)[0]
                if len(m_idx) == 0:
                    continue
                ctx_m = context[b, m_idx, :].unsqueeze(0)
                xt_m = x_t[b, m_idx].unsqueeze(0)

                x1_pred = self.vector_field(xt_m, ctx_m, t[b].unsqueeze(0))
                denom = (1.0 - t[b].unsqueeze(0)).clamp_min(eps)
                v_pred = (x1_pred - xt_m) / denom
                v_pred = v_pred + float(velocity_bias)
                if debug_step0_stats and i == 0 and b == 0:
                    x1_ref = x1_true if x1_true is not None else known_genes
                    print(
                        f"[inference debug] step 0: "
                        f"x1_pred mean={x1_pred.mean().item():.6f}, std={x1_pred.std().item():.6f}"
                    )
                    print(
                        f"[inference debug] step 0: "
                        f"x_t(masked) mean={xt_m.mean().item():.6f}, std={xt_m.std().item():.6f}"
                    )
                    x1_masked = x1_ref[b, m_idx]
                    print(
                        f"[inference debug] step 0: "
                        f"x1_true(masked) mean={x1_masked.mean().item():.6f}, std={x1_masked.std().item():.6f}"
                    )
                    v_true = x1_masked.unsqueeze(0) - xt_m
                    xp_flat = x1_pred.flatten().float()
                    xt_flat = x1_masked.unsqueeze(0).flatten().float()
                    vp_flat = v_pred.flatten().float()
                    vt_flat = v_true.flatten().float()
                    if xp_flat.numel() >= 2 and xp_flat.std().item() > 0 and xt_flat.std().item() > 0:
                        corr_x1 = torch.corrcoef(torch.stack([xp_flat, xt_flat]))[0, 1].item()
                    else:
                        corr_x1 = float("nan")
                    if vp_flat.numel() >= 2 and vp_flat.std().item() > 0 and vt_flat.std().item() > 0:
                        corr = torch.corrcoef(torch.stack([vp_flat, vt_flat]))[0, 1].item()
                    else:
                        corr = float("nan")
                    print(f"[inference debug] step 0: x1_pred vs x1_true pearson={corr_x1:.6f}")
                    print(f"[inference debug] step 0: v_pred vs v_true pearson={corr:.6f}")
                    print(
                        f"[inference debug] step 0: "
                        f"v_true mean={v_true.mean().item():.6f}, std={v_true.std().item():.6f}"
                    )

                x_t[b, m_idx] = x_t[b, m_idx] + v_pred.squeeze(0) * dt
                x_t[b, m_idx] = torch.clamp(x_t[b, m_idx], min=0.0)

            x_t[~mask_bool] = known_genes[~mask_bool]
            if debug_step0_stats and x1_true is not None and B > 0:
                m0 = mask_bool[0]
                if m0.any():
                    dist = torch.abs(x_t[0, m0] - x1_true[0, m0]).mean().item()
                    x_t_mean = x_t[0, m0].mean().item()
                    print(
                        f"[inference trace] step {i}: t={t_val:.3f}, "
                        f"dist_to_x1={dist:.6f}, x_t_mean={x_t_mean:.6f}"
                    )

        return x_t

    def inference_velocity(
        self,
        gene_ids,
        known_genes,
        mask_indicator,
        n_steps=10,
        temperature=0.01,
        prior_mean=0.0,
        prior_std=1.0,
        initial_masked_values=None,
    ):
        """Rollout for the velocity-only ablation."""
        del temperature
        B, G = known_genes.shape
        dt = 1.0 / n_steps

        x_t = known_genes.clone()
        mask_bool = mask_indicator == 1
        if initial_masked_values is not None:
            x_t[mask_bool] = initial_masked_values[mask_bool]
        else:
            prior_std = max(float(prior_std), 1e-6)
            x_t[mask_bool] = torch.randn_like(x_t[mask_bool]) * prior_std + float(prior_mean)

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=known_genes.device)
            context, _ = self.compute_context(gene_ids, x_t, mask_indicator)

            for b in range(B):
                m_idx = mask_indicator[b].nonzero(as_tuple=True)[0]
                if len(m_idx) == 0:
                    continue
                ctx_m = context[b, m_idx, :].unsqueeze(0)
                xt_m = x_t[b, m_idx].unsqueeze(0)
                v_pred = self.vector_field(xt_m, ctx_m, t[b].unsqueeze(0))
                x_t[b, m_idx] = x_t[b, m_idx] + v_pred.squeeze(0) * dt
                x_t[b, m_idx] = torch.clamp(x_t[b, m_idx], min=0.0)

            x_t[~mask_bool] = known_genes[~mask_bool]

        return x_t
