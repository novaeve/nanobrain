"""The five pretraining objectives, all computed on every step.

gene        masked gene values reconstructed from token states (cell x gene MAE)
integration scVI ELBO: expression from (z, batch) plus KL(q(z|x) || N(0, I))
cell_type   residual hierarchical cross-entropy from z, every level
anatomy     same, for brain region
spatial     masked neighbours' expression and cell type predicted from the
            visible cells of their spatial group (spatial MAE)
"""

import math

import torch
import torch.nn.functional as F

TASKS = ("gene", "integration", "cell_type", "anatomy", "spatial")


def dense_counts(b: dict, n_cells: int, n_genes: int) -> torch.Tensor:
    lengths = b["cnt_ptr"].diff()
    rows = torch.repeat_interleave(torch.arange(n_cells, device=lengths.device), lengths)
    x = torch.zeros(n_cells, n_genes, device=lengths.device)
    x[rows, b["cnt_gene"]] = b["cnt_val"].float()
    return x


def expression_nll(logits, counts, measured, log_theta, likelihood: str) -> torch.Tensor:
    """Per-cell negative log-likelihood summed over measured genes (unmeasured genes are ignored)."""
    # A large finite negative (not -inf) keeps unmeasured genes out of the softmax without NaN gradients.
    logits = logits.float().masked_fill(~measured, -1e4)
    log_rho = torch.log_softmax(logits, -1)
    if likelihood == "nb":
        lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
        log_mu = log_rho + torch.log(lib)
        theta = torch.exp(log_theta.float()).clamp(max=1e6)
        log_theta_mu = torch.logaddexp(torch.log(theta), log_mu)
        ll = (
            torch.lgamma(counts + theta)
            - torch.lgamma(theta)
            - torch.lgamma(counts + 1)
            + theta * (torch.log(theta) - log_theta_mu)
            + counts * (log_mu - log_theta_mu)
        )
    else:  # mse_log
        lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
        target = torch.log1p(counts / lib * 1e4)
        ll = -((torch.log1p(torch.exp(log_rho) * 1e4) - target) ** 2)
    return -(ll.masked_fill(~measured, 0.0)).sum(-1)


def hierarchical_ce(logits: list[torch.Tensor], labels: torch.Tensor, level_maps: list[torch.Tensor]):
    """Mean cross-entropy over levels, plus fine-level accuracy."""
    losses = [F.cross_entropy(lg.float(), m[labels]) for lg, m in zip(logits, level_maps)]
    acc = (logits[-1].argmax(-1) == labels).float().mean()
    return torch.stack(losses).mean(), acc


def latent_penalty(model, out: dict) -> torch.Tensor:
    """KL of the integration objective, per the model's latent_reg."""
    mu, logvar, z = out["mu"], out["logvar"], out["z"]
    if model.cfg.latent_reg == "kl":
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(-1).mean()
    if model.cfg.latent_reg == "mog":  # single-sample Monte Carlo KL(q(z|x) || p_mog(z))
        log_q = -0.5 * (((z - mu) ** 2) / logvar.exp() + logvar + math.log(2 * math.pi)).sum(-1)
        return (log_q - model.prior.log_prob(z)).mean()
    return mu.new_zeros(())


def compute_losses(
    model, b: dict, out: dict, panels: torch.Tensor, level_maps: dict, kl_weight: float, extras: dict | None = None
):
    """The five task losses (plus any extra regularisers in `extras`: name -> callable(model, b, out))."""
    cfg = model.cfg
    z = out["z"]
    n_cells = z.shape[0]
    counts = dense_counts(b, n_cells, cfg.n_genes)
    measured = panels[b["panel"]]
    dec = model.decoder

    losses, metrics = {}, {}
    losses["gene"] = F.mse_loss(out["gene_pred"], b["tok_target"][b["tok_masked"]].float())

    recon = expression_nll(
        dec(model.recon_input(out), b["batch"]), counts, measured, dec.log_theta, cfg.likelihood
    ).mean()
    kl = latent_penalty(model, out)
    losses["integration"] = recon + kl_weight * kl
    metrics["integration/recon"], metrics["integration/kl"] = recon, kl

    losses["cell_type"], metrics["cell_type/acc"] = hierarchical_ce(
        model.cell_type_head(z), b["cell_type"], level_maps["cell_type"]
    )
    losses["anatomy"], metrics["anatomy/acc"] = hierarchical_ce(
        model.anatomy_head(z), b["anatomy"], level_maps["anatomy"]
    )

    members, masked = b["grp_members"], b["grp_masked"]
    pad = members < 0
    safe = members.clamp(min=0)
    pred = model.neighborhood(z[safe], b["grp_rel"], masked, pad)[masked]
    target_cells = safe[masked]
    n_measured = measured[target_cells].sum(-1).clamp(min=1)
    nll = expression_nll(
        dec(pred, b["batch"][target_cells], task="spatial"),
        counts[target_cells],
        measured[target_cells],
        dec.log_theta,
        cfg.likelihood,
    )
    sp_recon = (nll / n_measured).mean()
    sp_ce = F.cross_entropy(model.cell_type_head(pred)[-1].float(), b["cell_type"][target_cells])
    losses["spatial"] = sp_recon + sp_ce
    metrics["spatial/recon_per_gene"], metrics["spatial/cell_type_ce"] = sp_recon, sp_ce
    if cfg.spatial_latent:  # the hidden cell's own latent mean, seen from its own tokens, as a denoised target
        sp_latent = F.mse_loss(pred.float(), out["mu"][target_cells].detach().float())
        losses["spatial"] = losses["spatial"] + cfg.spatial_latent * sp_latent
        metrics["spatial/latent_mse"] = sp_latent
    for name, fn in (extras or {}).items():
        losses[name] = fn(model, b, out)
    if model.cfg.research_plugins:
        from cajal_research.plugins import replace_losses

        losses = replace_losses(model.cfg.research_plugins, model, b, out, level_maps, losses)
    return losses, metrics
