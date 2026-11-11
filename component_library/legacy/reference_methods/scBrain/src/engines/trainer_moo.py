"""
Multi-Objective Optimization trainer for MJM_1 / MJM_1_LR.

Supports 4 strategies:
  - equal:       standard equal-weight sum (baseline)
  - uncertainty:  Kendall et al. 2018 — learnable homoscedastic uncertainty
  - pcgrad:      Yu et al. 2020 — gradient surgery for conflicting tasks
  - gradnorm:    Chen et al. 2018 — adaptive gradient norm balancing

Tasks:
  - recon:     MSE reconstruction loss
  - class:     CE for 3-class
  - subclass:  CE for 24-class
  - supertype: confidence-weighted CE for 137-class
"""
import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.engines.trainer import TrainerEngine


class TrainerEngine_MOO(TrainerEngine):

    def __init__(self, model, optimizer, scheduler, device, logger, args):
        super().__init__(model, optimizer, scheduler, device, logger, args)
        self.moo_method = getattr(args, 'moo_method', 'equal')
        self.spatial_dim = getattr(args, 'spatial_dim', 0)
        self._init_moo()

    # ── MOO initialization ──────────────────────────────────────────────
    def _init_moo(self):
        if self.moo_method == 'uncertainty':
            self._init_uncertainty()
        elif self.moo_method == 'gradnorm':
            self._init_gradnorm()

    def _init_uncertainty(self):
        """Kendall 2018: learnable log-variance per task."""
        self.log_vars = nn.ParameterDict({
            'recon': nn.Parameter(torch.zeros(1, device=self.device)),
            'class': nn.Parameter(torch.zeros(1, device=self.device)),
            'subclass': nn.Parameter(torch.zeros(1, device=self.device)),
            'supertype': nn.Parameter(torch.zeros(1, device=self.device)),
        })
        # Add log_var params to optimizer
        self.optimizer.add_param_group({
            'params': list(self.log_vars.parameters()),
            'lr': self.optimizer.defaults['lr'],
        })

    def _init_gradnorm(self):
        """Chen 2018: learnable task weights with gradient norm balancing."""
        self.task_weights = nn.Parameter(
            torch.ones(4, device=self.device), requires_grad=True)
        self.alpha_gn = getattr(self.args, 'alpha_gradnorm', 1.5)
        self.initial_losses = None  # set on first step
        # Separate optimizer for weights with higher LR
        self.weight_optimizer = torch.optim.Adam(
            [self.task_weights], lr=self.optimizer.defaults['lr'] * 10)

    # ── Compute individual task losses ──────────────────────────────────
    def _prepare_input(self, batch_data):
        """Prepare model input, optionally concatenating spatial coords."""
        X = batch_data['X'].to(self.device)
        X = self._normalize(X)
        if self.spatial_dim > 0 and 'spatial_zscore' in batch_data:
            spatial = batch_data['spatial_zscore'].to(self.device)
            X_input = torch.cat([X, spatial], dim=1)
        else:
            X_input = X
        return X_input, X  # X_input for model, X (gene only) for recon target

    def compute_task_losses(self, batch_data):
        """Return 4 individual loss tensors + logits + latent."""
        X_input, X_target = self._prepare_input(batch_data)
        y_c = batch_data['y_class'].to(self.device)
        y_sc = batch_data['y_subclass'].to(self.device)
        y_st = batch_data['y_supertype'].to(self.device)
        confidence = batch_data['confidence'].to(self.device)

        recon_X, logits, z = self.model(X_input)
        logit_c, logit_sc, logit_st = logits

        loss_recon = F.mse_loss(recon_X, X_target)
        loss_c = F.cross_entropy(logit_c, y_c)
        loss_sc = F.cross_entropy(logit_sc, y_sc)
        loss_st = (F.cross_entropy(logit_st, y_st, reduction='none') * confidence).mean()

        losses = {'recon': loss_recon, 'class': loss_c,
                  'subclass': loss_sc, 'supertype': loss_st}
        logits_dict = {'class': logit_c, 'subclass': logit_sc, 'supertype': logit_st}
        return losses, logits_dict, z

    # ── Override compute_loss for compatibility with test() ─────────────
    def compute_loss(self, batch_data):
        losses, logits_dict, z = self.compute_task_losses(batch_data)
        total = sum(losses.values())
        metrics = {'total': total.item(),
                   'recon': losses['recon'].item(),
                   'class': losses['class'].item(),
                   'subclass': losses['subclass'].item(),
                   'supertype': losses['supertype'].item()}
        return total, losses['recon'], metrics, logits_dict, z

    # ── MOO-specific training steps ─────────────────────────────────────
    def _step_equal(self, losses):
        total = sum(losses.values())
        total.backward()
        return total

    def _step_uncertainty(self, losses):
        total = torch.tensor(0.0, device=self.device)
        for key, loss in losses.items():
            log_var = self.log_vars[key]
            precision = torch.exp(-log_var)
            total = total + precision * loss + log_var
        total.backward()
        return total

    def _step_pcgrad(self, losses):
        """2-task PCGrad: recon vs classification (sum of 3 CE)."""
        loss_recon = losses['recon']
        loss_cls = losses['class'] + losses['subclass'] + losses['supertype']

        # Collect shared encoder parameters
        shared_params = []
        if hasattr(self.model, 'E1'):
            shared_params += list(self.model.E1.parameters())
            shared_params += list(self.model.E2.parameters())
            shared_params += list(self.model.E3.parameters())
        elif hasattr(self.model, 'encoder'):
            shared_params += list(self.model.encoder.parameters())

        # Compute per-task gradients on shared params
        g_recon = torch.autograd.grad(loss_recon, shared_params,
                                       retain_graph=True, allow_unused=True)
        g_cls = torch.autograd.grad(loss_cls, shared_params,
                                     retain_graph=True, allow_unused=True)

        # Flatten
        g_recon_flat = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
                                   for g, p in zip(g_recon, shared_params)])
        g_cls_flat = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
                                 for g, p in zip(g_cls, shared_params)])

        # Project if conflicting
        dot = torch.dot(g_recon_flat, g_cls_flat)
        if dot < 0:
            # Project g_recon onto normal plane of g_cls
            g_recon_flat = g_recon_flat - (dot / (g_cls_flat.norm() ** 2 + 1e-8)) * g_cls_flat
            # Project g_cls onto normal plane of g_recon (original)
            g_cls_orig = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
                                     for g, p in zip(g_cls, shared_params)])
            dot2 = torch.dot(g_cls_orig, torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
                                                      for g, p in zip(g_recon, shared_params)]))
            g_recon_orig_flat = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=self.device)
                                            for g, p in zip(g_recon, shared_params)])
            g_cls_flat = g_cls_orig - (dot2 / (g_recon_orig_flat.norm() ** 2 + 1e-8)) * g_recon_orig_flat

        # Apply combined gradient to shared params
        combined = g_recon_flat + g_cls_flat
        offset = 0
        for p in shared_params:
            numel = p.numel()
            if p.grad is None:
                p.grad = combined[offset:offset + numel].reshape(p.shape)
            else:
                p.grad += combined[offset:offset + numel].reshape(p.shape)
            offset += numel

        # Normal backward for private params (zero out shared grads first won't work,
        # so we do a full backward and then overwrite shared grads)
        # Actually: we already set shared grads. Now backward total for private params.
        total = loss_recon + loss_cls
        # Compute grads for private params only
        private_params = [p for p in self.model.parameters()
                          if not any(p is sp for sp in shared_params)]
        if private_params:
            private_grads = torch.autograd.grad(total, private_params, allow_unused=True)
            for p, g in zip(private_params, private_grads):
                if g is not None:
                    p.grad = g

        return total.detach()

    def _step_gradnorm(self, losses):
        """GradNorm (Chen 2018): adaptive weight balancing via gradient norms.

        Key fixes vs naive implementation:
        - Direct weights with renormalization (no softmax)
        - Clamp weights to [0.25, 4.0] to prevent explosion
        - Separate weight optimizer
        - Two-step: (1) backward weighted loss for model, (2) backward GradNorm loss for weights
        """
        loss_list = [losses['recon'], losses['class'],
                     losses['subclass'], losses['supertype']]

        # Record initial losses on first call
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(
                [l.item() for l in loss_list], device=self.device)

        # Direct weights (renormalized to sum=4)
        with torch.no_grad():
            w_sum = self.task_weights.data.sum()
            self.task_weights.data = self.task_weights.data / w_sum * 4.0
            self.task_weights.data.clamp_(0.25, 4.0)
        weights = self.task_weights

        # Step 1: backward weighted task loss for model parameters
        weighted_total = sum(w.detach() * l for w, l in zip(weights, loss_list))
        weighted_total.backward(retain_graph=True)

        # Step 2: compute GradNorm loss for weight update
        # Find last shared layer weight for reference
        if hasattr(self.model, 'E3'):
            ref_param = None
            for p in self.model.E3.parameters():
                ref_param = p
        else:
            ref_param = list(self.model.encoder.parameters())[-1]

        gnorms = []
        for w, l in zip(weights, loss_list):
            g = torch.autograd.grad(w * l, ref_param, retain_graph=True,
                                     create_graph=True, allow_unused=True)[0]
            gnorms.append(g.norm() if g is not None
                          else torch.tensor(0.0, device=self.device, requires_grad=False))

        gnorms = torch.stack(gnorms)
        mean_gnorm = gnorms.mean().detach()

        # Inverse training rate
        loss_ratios = torch.stack(
            [l.detach() / (il + 1e-8)
             for l, il in zip(loss_list, self.initial_losses)])
        inv_rate = loss_ratios / (loss_ratios.mean() + 1e-8)
        target_gnorms = mean_gnorm * (inv_rate ** self.alpha_gn)

        gn_loss = (gnorms - target_gnorms.detach()).abs().sum()

        # Update weights only
        self.weight_optimizer.zero_grad()
        gn_loss.backward()
        self.weight_optimizer.step()

        return weighted_total.detach()

    # ── Training epoch ──────────────────────────────────────────────────
    def train_epoch(self, dataloader):
        self.model.train()
        keys = ['total', 'recon', 'class', 'subclass', 'supertype']
        epoch_metrics = {k: 0.0 for k in keys}
        conflict_count = 0
        total_steps = 0

        for batch_data in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            losses, _, _ = self.compute_task_losses(batch_data)

            if self.moo_method == 'equal':
                total = self._step_equal(losses)
            elif self.moo_method == 'uncertainty':
                total = self._step_uncertainty(losses)
            elif self.moo_method == 'pcgrad':
                total = self._step_pcgrad(losses)
            elif self.moo_method == 'gradnorm':
                total = self._step_gradnorm(losses)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            epoch_metrics['total'] += sum(l.item() for l in losses.values())
            for k in ['recon', 'class', 'subclass', 'supertype']:
                epoch_metrics[k] += losses[k].item()
            total_steps += 1

        result = {k: v / len(dataloader) for k, v in epoch_metrics.items()}

        # Log MOO-specific info
        if self.moo_method == 'uncertainty':
            var_str = ' | '.join(
                f"{k}: {v.item():.3f}" for k, v in self.log_vars.items())
            self.logger.info(f"  [Uncertainty] log_var: {var_str}")
        elif self.moo_method == 'gradnorm':
            w = torch.softmax(self.task_weights, dim=0) * 4.0
            w_str = f"recon={w[0]:.3f} cls={w[1]:.3f} sub={w[2]:.3f} st={w[3]:.3f}"
            self.logger.info(f"  [GradNorm] weights: {w_str}")

        return result

    # ── Validation epoch (no gradient updates) ──────────────────────────
    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        keys = ['total', 'recon', 'class', 'subclass', 'supertype']
        epoch_metrics = {k: 0.0 for k in keys}

        for batch_data in tqdm(dataloader, desc="Evaluating"):
            losses, _, _ = self.compute_task_losses(batch_data)
            epoch_metrics['total'] += sum(l.item() for l in losses.values())
            for k in ['recon', 'class', 'subclass', 'supertype']:
                epoch_metrics[k] += losses[k].item()

        return {k: v / len(dataloader) for k, v in epoch_metrics.items()}

