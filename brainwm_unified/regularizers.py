"""Extra objectives for integration, registered by name and configured in TOML:

    [[train.objectives]]
    name = "cond_hsic"      # registered name
    weight = 10.0           # added to the total as weight * loss (not task-normalised)
    label = "cell_type"     # the objective's own parameters

Each is a dataclass called as fn(model, batch, out) -> scalar loss on the latent mean.

  cond_hsic     z independent of batch given the label (class-conditional HSIC)      TCRI / CIRCE / ConDo
  sigreg        isotropic-Gaussian latent via random-projection Epps-Pulley tests    LeJEPA (SIGReg)
  batch_swap    counterfactual: decode under another batch, re-encode, match mu     sysVI cycle, CFGen, JEPA
  teacher_gram  match a frozen teacher's pairwise latent geometry                  REPA / VA-VAE / DINOv3 Gram
  batch_adversary  a batch discriminator on mu behind gradient reversal            DANN, DIRAC, scDREAMER
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import model as M
from .registry import Registry

OBJECTIVES = Registry("objective")


@OBJECTIVES.register("cond_hsic")
@dataclass
class CondHSIC:
    """z independent of batch given the label: size-weighted HSIC of mu and batch within each label class.

    Gaussian kernel on mu (median bandwidth), delta kernel on batch, batch kernel centred within each
    class; classes seen in one batch contribute nothing, so biology that differs between batches is
    not pulled together, only the same label across batches.
    """

    label: str = "cell_type"

    def __call__(self, model, b: dict, out: dict) -> torch.Tensor:
        mu = out["mu"].float()
        y, g = b[self.label], b["batch"]
        n = len(mu)
        same = (y[:, None] == y[None, :]).float()
        n_c = same.sum(1)
        d2 = torch.cdist(mu, mu).pow(2)
        bandwidth = d2.detach()[d2.detach() > 0].median().clamp(min=1e-6)
        k = torch.exp(-d2 / bandwidth)
        L = (g[:, None] == g[None, :]).float()
        row = (L * same).sum(1) / n_c  # mean of L over the cell's class
        grand = (same @ row) / n_c  # mean of L over the class block
        lc = (L - row[:, None] - row[None, :] + grand[:, None]) * same
        return ((k * lc).sum(1) / (n_c * n)).sum()


@OBJECTIVES.register("sigreg")
@dataclass
class SIGReg:
    """Sketched isotropic Gaussian regulariser: for random unit directions, the projected latent's
    empirical characteristic function is matched to N(0, 1)'s (Epps-Pulley statistic), averaged."""

    n_proj: int = 256
    t_max: float = 5.0
    n_t: int = 33

    def __call__(self, model, b: dict, out: dict) -> torch.Tensor:
        x = out["mu"].float()
        a = torch.randn(x.shape[1], self.n_proj, device=x.device)
        p = x @ (a / a.norm(dim=0, keepdim=True))  # (n, P)
        t = torch.linspace(-self.t_max, self.t_max, self.n_t, device=x.device)
        tp = p.unsqueeze(-1) * t  # (n, P, T)
        target = torch.exp(-0.5 * t**2)
        err = (torch.cos(tp).mean(0) - target) ** 2 + torch.sin(tp).mean(0) ** 2
        return torch.trapezoid(err * target, t).mean()


@OBJECTIVES.register("batch_swap")
@dataclass
class BatchSwap:
    """Batch as augmentation: decode n_cells latents under a random other batch, re-encode the
    counterfactual (same genes, decoded values, nothing masked) and match its mu to the original
    (cosine, original detached). The decoder output is detached: only the encoder learns invariance."""

    n_cells: int = 32

    def __call__(self, model, b: dict, out: dict) -> torch.Tensor:
        mu = out["mu"]
        n_cells = mu.shape[0]
        k = min(self.n_cells, n_cells)
        keep = torch.zeros(n_cells, dtype=torch.bool, device=mu.device)
        keep[torch.randperm(n_cells, device=mu.device)[:k]] = True
        new_id = torch.cumsum(keep.long(), 0) - 1
        doc = b["tok_doc"]
        tok = (doc >= 0) & keep[doc.clamp(min=0)]
        with torch.no_grad():
            n_b = model.cfg.n_batches
            other = (b["batch"][keep] + torch.randint(1, n_b, (k,), device=mu.device)) % n_b
            rho = torch.softmax(model.decoder(mu[keep].detach().float(), other).float(), -1)
            genes, cls = b["tok_gene"][tok], b["tok_cls"][tok]
            d = new_id[doc[tok]]
            vals = torch.log1p(rho[d, genes.clamp(max=rho.shape[1] - 1)] * 1e4).masked_fill(cls, 0.0)
        sub = M.pad_tokens(
            {
                "tok_gene": genes,
                "tok_in": vals.to(b["tok_in"].dtype),
                "tok_target": vals.to(b["tok_in"].dtype),
                "tok_masked": torch.zeros_like(cls),
                "tok_cls": cls,
                "tok_doc": d,
            }
        )
        layout = (
            M.dense_mask(sub["tok_doc"])
            if mu.device.type == "cpu"
            else M.attention_layout(sub["tok_doc"], model.cfg.attention_backend)
        )
        h = model.encoder(sub, layout, model.cfg.loops)
        mu2 = model.to_latent(h[sub["tok_cls"]]).float().chunk(2, -1)[0]
        return (1 - F.cosine_similarity(mu2, mu[keep].detach().float(), dim=-1)).mean()


@OBJECTIVES.register("teacher_gram")
@dataclass
class TeacherGram:
    """Inherit a frozen teacher's latent geometry: match the cosine-similarity (Gram) matrix of mu to the
    teacher's on the same batch. The teacher is loaded once from `checkpoint` and never trained."""

    checkpoint: str = ""
    _teacher: object = field(default=None, init=False, repr=False, compare=False)

    def __call__(self, model, b: dict, out: dict) -> torch.Tensor:
        if self._teacher is None:
            from .infer import load_model

            self._teacher, _ = load_model(self.checkpoint, device=out["mu"].device)
            self._teacher.requires_grad_(False)
        with torch.no_grad():
            layout = (
                M.dense_mask(b["tok_doc"])
                if out["mu"].device.type == "cpu"
                else M.attention_layout(b["tok_doc"], self._teacher.cfg.attention_backend)
            )
            t_mu = self._teacher(b, layout, self._teacher.cfg.loops)["mu"].float()
        s, t = F.normalize(out["mu"].float(), dim=-1), F.normalize(t_mu, dim=-1)
        return ((s @ s.T - t @ t.T) ** 2).mean()


@OBJECTIVES.register("batch_adversary")
@dataclass
class BatchAdversary:
    """The discriminator learns to tell each cell's integration batch from mu; reversed, the same loss pushes
    the encoder to make batches indistinguishable. Needs model.batch_adversary (its hidden width) > 0; the
    discriminator is part of the model, so it is optimised and checkpointed with it. coeff scales the
    reversed gradient (the balancer already sets the loss's overall share)."""

    coeff: float = 1.0

    def __call__(self, model, b: dict, out: dict) -> torch.Tensor:
        if not hasattr(model, "batch_adversary"):
            raise ValueError("batch_adversary needs model.batch_adversary > 0 (the discriminator's hidden width)")
        logits = model.batch_adversary(out["mu"].float(), self.coeff)
        return F.cross_entropy(logits.float(), b["batch"])
