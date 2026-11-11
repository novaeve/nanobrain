"""
Trainer for mjm_2a: Laminar Depth Injection.
Extends TrainerEngine logic; does NOT modify trainer.py.
"""
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.engines.trainer import TrainerEngine


class TrainerEngine_2a(TrainerEngine):
    """
    Adds depth imputation input and auxiliary depth regression loss.
    depth_imputer: a callable model(X) -> depth_pred [B,1]
    lambda_depth: weight for auxiliary depth MSE loss
    """
    def __init__(self, model, optimizer, scheduler, device, logger, args,
                 depth_imputer=None, lambda_depth=0.1):
        super().__init__(model, optimizer, scheduler, device, logger, args)
        self.depth_imputer = depth_imputer
        self.lambda_depth  = lambda_depth

    def compute_loss(self, batch_data):
        X          = batch_data['X'].to(self.device)
        y_c        = batch_data['y_class'].to(self.device)
        y_sc       = batch_data['y_subclass'].to(self.device)
        y_st       = batch_data['y_supertype'].to(self.device)
        confidence = batch_data['confidence'].to(self.device)
        depth_true = batch_data['depth_norm'].to(self.device).unsqueeze(-1)   # [B,1]
        has_depth  = batch_data['has_depth'].to(self.device)                   # [B]

        X_norm = self._normalize(X)  # log1p

        # Impute depth for all cells
        with torch.no_grad():
            depth_imputed = self.depth_imputer(X_norm)  # [B,1]

        recon_X, logits, z, depth_pred = self.model(X_norm, depth_imputed)
        logit_c, logit_sc, logit_st = logits

        loss_recon = F.mse_loss(recon_X, X_norm)
        loss_c     = F.cross_entropy(logit_c, y_c)
        loss_sc    = F.cross_entropy(logit_sc, y_sc)
        loss_st_uw = F.cross_entropy(logit_st, y_st, reduction='none')
        loss_st    = (loss_st_uw * confidence).mean()

        # Auxiliary depth loss (only on cells that have real depth labels)
        if has_depth.sum() > 0:
            loss_depth = F.mse_loss(
                depth_pred[has_depth.bool()],
                depth_true[has_depth.bool()]
            )
        else:
            loss_depth = torch.tensor(0.0, device=self.device)

        total_loss = (1.0 * loss_recon + 1.0 * loss_c +
                      1.0 * loss_sc   + 1.0 * loss_st +
                      self.lambda_depth * loss_depth)

        metrics = {
            'total':    total_loss,
            'recon':    loss_recon.item(),
            'class':    loss_c.item(),
            'subclass': loss_sc.item(),
            'supertype':loss_st.item(),
            'depth':    loss_depth.item(),
        }
        logits_dict = {'class': logit_c, 'subclass': logit_sc, 'supertype': logit_st}
        return total_loss, loss_recon, metrics, logits_dict, z

    def train_epoch(self, dataloader):
        self.model.train()
        keys = ['total', 'recon', 'class', 'subclass', 'supertype', 'depth']
        epoch_metrics = {k: 0.0 for k in keys}
        for batch_data in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            loss, _, metrics, _, _ = self.compute_loss(batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
        return {k: v / len(dataloader) for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        keys = ['total', 'recon', 'class', 'subclass', 'supertype', 'depth']
        epoch_metrics = {k: 0.0 for k in keys}
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            _, _, metrics, _, _ = self.compute_loss(batch_data)
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
        return {k: v / len(dataloader) for k, v in epoch_metrics.items()}

    def train(self, train_loader, val_loader):
        os.makedirs(self.args.save_dir, exist_ok=True)
        best_model_path = os.path.join(self.args.save_dir, 'best_model.pth')
        patience_counter = 0
        best_val_metrics = None

        for epoch in range(1, self.args.max_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.eval_epoch(val_loader)
            self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.args.max_epochs}] "
                f"Time:{time.time()-t0:.1f}s LR:{self.scheduler.get_last_lr()[0]:.2e}"
            )
            self.logger.info(
                f"  [Train] total:{train_metrics['total']:.4f} "
                f"recon:{train_metrics['recon']:.4f} cls:{train_metrics['class']:.4f} "
                f"sub:{train_metrics['subclass']:.4f} sup:{train_metrics['supertype']:.4f} "
                f"depth:{train_metrics['depth']:.4f}"
            )
            self.logger.info(
                f"  [Val]   total:{val_metrics['total']:.4f} "
                f"recon:{val_metrics['recon']:.4f} cls:{val_metrics['class']:.4f} "
                f"sub:{val_metrics['subclass']:.4f} sup:{val_metrics['supertype']:.4f} "
                f"depth:{val_metrics['depth']:.4f}"
            )

            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                best_val_metrics   = val_metrics
                patience_counter   = 0
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info('Best model saved!')
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info('Early stopping triggered!')
                    break

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info('Loading best model for test...')
        best_model_path = os.path.join(self.args.save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        all_y_true = {'class': [], 'subclass': [], 'supertype': []}
        all_y_pred = {'class': [], 'subclass': [], 'supertype': []}
        all_y_prob = {'class': [], 'subclass': [], 'supertype': []}
        res = {'latent': [], 'spatial': [], 'batch': [], 'supertype': [], 'cps': []}
        epoch_metrics = {'total': 0.0, 'recon': 0.0, 'class': 0.0,
                         'subclass': 0.0, 'supertype': 0.0, 'depth': 0.0}

        for batch_data in test_loader:
            _, _, metrics, logits_dict, z = self.compute_loss(batch_data)
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            res['latent'].append(z.cpu())
            res['spatial'].append(batch_data['spatial'].cpu())
            res['batch'].append(batch_data['batch_id'].cpu())
            res['supertype'].append(batch_data['y_supertype'].cpu())
            res['cps'].append(batch_data['cps'].cpu())
            for t in ['class', 'subclass', 'supertype']:
                all_y_true[t].append(batch_data[f'y_{t}'].cpu().numpy())
                all_y_pred[t].append(logits_dict[t].argmax(-1).cpu().numpy())
                all_y_prob[t].append(F.softmax(logits_dict[t], -1).cpu().numpy())

        test_metrics = {k: v / len(test_loader) for k, v in epoch_metrics.items()}
        self.logger.info(f"  [Test] {test_metrics}")
        final_y_true = {k: np.concatenate(v) for k, v in all_y_true.items()}
        final_y_pred = {k: np.concatenate(v) for k, v in all_y_pred.items()}
        final_y_prob = {k: np.concatenate(v, axis=0) for k, v in all_y_prob.items()}
        self.log_classification_metrics(final_y_true, final_y_pred, prefix='Test')
        num_classes_dict = {
            'class':     self.args.output_num[0],
            'subclass':  self.args.output_num[1],
            'supertype': self.args.output_num[2],
        }
        self.log_auc_roc_metrics(final_y_true, final_y_prob, num_classes_dict, prefix='Test')
        res = {k: torch.cat(v, 0).numpy() for k, v in res.items()}
        out_path = os.path.join(self.args.save_dir, 'test_features.npz')
        np.savez_compressed(out_path, **res)
        self.logger.info(f'Features saved: {out_path}')
        return res
