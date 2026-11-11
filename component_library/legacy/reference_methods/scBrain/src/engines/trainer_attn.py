"""
Trainer for mjm_attn: passes spatial coords and batch_id to model.
Extends TrainerEngine — only overrides compute_loss, eval_epoch, and test.
"""
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.engines.trainer import TrainerEngine


class TrainerEngine_attn(TrainerEngine):

    def compute_loss(self, batch_data):
        X          = batch_data['X'].to(self.device)
        spatial    = batch_data['spatial'].to(self.device)
        batch_id   = batch_data['batch_id'].to(self.device)
        y_c        = batch_data['y_class'].to(self.device)
        y_sc       = batch_data['y_subclass'].to(self.device)
        y_st       = batch_data['y_supertype'].to(self.device)
        confidence = batch_data['confidence'].to(self.device)

        X = self._normalize(X)
        recon_X, logits, z = self.model(X, spatial, batch_id)
        logit_c, logit_sc, logit_st = logits

        loss_recon = F.mse_loss(recon_X, X)
        loss_c     = F.cross_entropy(logit_c, y_c)
        loss_sc    = F.cross_entropy(logit_sc, y_sc)
        loss_st    = (F.cross_entropy(logit_st, y_st, reduction='none') * confidence).mean()

        total_loss = loss_recon + loss_c + loss_sc + loss_st

        metrics = {
            'total': total_loss, 'recon': loss_recon.item(),
            'class': loss_c.item(), 'subclass': loss_sc.item(),
            'supertype': loss_st.item(),
        }
        logits_dict = {'class': logit_c, 'subclass': logit_sc, 'supertype': logit_st}
        return total_loss, loss_recon, metrics, logits_dict, z

    def eval_epoch(self, dataloader):
        """Validation: pure eval, no gradient updates."""
        self.model.eval()
        epoch_metrics = {k: 0.0 for k in ['total', 'recon', 'class', 'subclass', 'supertype']}
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="Evaluating"):
                _, _, metrics, _, _ = self.compute_loss(batch_data)
                for k in epoch_metrics:
                    epoch_metrics[k] += metrics[k]
        return {k: v / len(dataloader) for k, v in epoch_metrics.items()}
