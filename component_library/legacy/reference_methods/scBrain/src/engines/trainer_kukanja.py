"""
Trainer for N-level hierarchical models on Kukanja data.
Generalizes TrainerEngine from 3-level to arbitrary N levels.
"""
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class KukanjaTrainerEngine:
    """N-level hierarchical trainer for Kukanja datasets."""

    def __init__(self, model, optimizer, scheduler, device, logger, args,
                 label_names=None, output_num=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.args = args
        self.best_val_loss = float('inf')
        self.label_names = label_names or args.label_names
        self.output_num = output_num or args.output_num
        self.n_levels = len(self.label_names)

    def _normalize(self, x):
        return torch.log1p(x)

    def compute_loss(self, batch_data):
        X = batch_data['X'].to(self.device)
        X_norm = self._normalize(X)

        # Collect labels for all levels
        y_list = [batch_data[f'y_{name}'].to(self.device)
                  for name in self.label_names]

        recon_X, logits_list, z = self.model(X_norm)

        loss_recon = F.mse_loss(recon_X, X_norm)

        # Classification loss per level (uniform weighting)
        level_losses = {}
        total_loss = loss_recon
        for i, name in enumerate(self.label_names):
            loss_i = F.cross_entropy(logits_list[i], y_list[i])
            level_losses[name] = loss_i.item()
            total_loss = total_loss + loss_i

        metrics = {'total': total_loss, 'recon': loss_recon.item()}
        metrics.update(level_losses)

        logits_dict = {name: logits_list[i] for i, name in enumerate(self.label_names)}
        return total_loss, loss_recon, metrics, logits_dict, z

    def train_epoch(self, dataloader):
        self.model.train()
        keys = ['total', 'recon'] + list(self.label_names)
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
        keys = ['total', 'recon'] + list(self.label_names)
        epoch_metrics = {k: 0.0 for k in keys}

        for batch_data in tqdm(dataloader, desc="Evaluating"):
            _, _, metrics, _, _ = self.compute_loss(batch_data)
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]

        return {k: v / len(dataloader) for k, v in epoch_metrics.items()}

    def _format_metrics(self, metrics, prefix):
        parts = [f"{prefix} total:{metrics['total']:.4f} recon:{metrics['recon']:.4f}"]
        for name in self.label_names:
            parts.append(f"{name}:{metrics[name]:.4f}")
        return " ".join(parts)

    def train(self, train_loader, val_loader):
        os.makedirs(self.args.save_dir, exist_ok=True)
        best_model_path = os.path.join(self.args.save_dir, "best_model.pth")
        patience_counter = 0
        best_val_metrics = None

        for epoch in range(1, self.args.max_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)
            self.scheduler.step()

            lr = self.scheduler.get_last_lr()[0]
            self.logger.info(
                f"Epoch [{epoch:03d}/{self.args.max_epochs}] "
                f"Time:{time.time()-t0:.1f}s LR:{lr:.2e}"
            )
            self.logger.info(f"  [Train] {self._format_metrics(train_metrics, '')}")
            self.logger.info(f"  [Val]   {self._format_metrics(val_metrics, '')}")
            if best_val_metrics is not None:
                self.logger.info(f"  [Best]  {self._format_metrics(best_val_metrics, '')}")

            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                best_val_metrics = val_metrics
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info("Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info("Early stopping triggered!")
                    break

    def log_classification_metrics(self, y_true_dict, y_pred_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} Classification Metrics ==========")
        for task in self.label_names:
            y_true = y_true_dict[task]
            y_pred = y_pred_dict[task]
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0)
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='micro', zero_division=0)
            per_p, per_r, per_f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0)

            self.logger.info(f"[{task.upper()}] Macro  - P:{macro_p:.4f} R:{macro_r:.4f} F1:{macro_f1:.4f}")
            self.logger.info(f"[{task.upper()}] Micro  - P:{micro_p:.4f} R:{micro_r:.4f} F1:{micro_f1:.4f}")

            parts = []
            for i in range(len(per_p)):
                if support[i] > 0 or per_p[i] > 0:
                    parts.append(f"C{i}(S={support[i]}):P={per_p[i]:.4f}/R={per_r[i]:.4f}")
            self.logger.info(f"[{task.upper()}] Per-class: {' | '.join(parts)}")
            self.logger.info("-" * 50)

    def log_auc_roc_metrics(self, y_true_dict, y_prob_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} AUC-ROC Metrics ==========")
        for i, task in enumerate(self.label_names):
            y_true = y_true_dict[task]
            y_prob = y_prob_dict[task]
            num_c = self.output_num[i]

            y_true_oh = np.zeros_like(y_prob)
            y_true_oh[np.arange(len(y_true)), y_true] = 1
            micro_auc = roc_auc_score(y_true_oh.ravel(), y_prob.ravel())

            valid_aucs = []
            for c in range(num_c):
                y_bin = (y_true == c).astype(int)
                if len(np.unique(y_bin)) == 2:
                    valid_aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
            macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

            self.logger.info(
                f"[{task.upper()}] AUC-ROC - Macro:{macro_auc:.4f} "
                f"Micro:{micro_auc:.4f} (Valid:{len(valid_aucs)}/{num_c})"
            )
            self.logger.info("-" * 50)

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("\nLoading best model for Test set evaluation...")
        best_model_path = os.path.join(self.args.save_dir, "best_model.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        keys = ['total', 'recon'] + list(self.label_names)
        epoch_metrics = {k: 0.0 for k in keys}
        all_y_true = {t: [] for t in self.label_names}
        all_y_pred = {t: [] for t in self.label_names}
        all_y_prob = {t: [] for t in self.label_names}
        res = {'latent': [], 'spatial': [], 'batch': []}

        for batch_data in test_loader:
            _, _, metrics, logits_dict, z = self.compute_loss(batch_data)
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            res['latent'].append(z.cpu())
            res['spatial'].append(batch_data['spatial'].cpu())
            res['batch'].append(batch_data['batch_id'].cpu())

            for task in self.label_names:
                all_y_true[task].append(batch_data[f'y_{task}'].cpu().numpy())
                all_y_pred[task].append(logits_dict[task].argmax(-1).cpu().numpy())
                all_y_prob[task].append(F.softmax(logits_dict[task], -1).cpu().numpy())

        test_metrics = {k: v / len(test_loader) for k, v in epoch_metrics.items()}
        self.logger.info(f"  [Test] {self._format_metrics(test_metrics, '')}")

        final_true = {k: np.concatenate(v) for k, v in all_y_true.items()}
        final_pred = {k: np.concatenate(v) for k, v in all_y_pred.items()}
        final_prob = {k: np.concatenate(v, axis=0) for k, v in all_y_prob.items()}

        self.log_classification_metrics(final_true, final_pred, prefix='Test')
        self.log_auc_roc_metrics(final_true, final_prob, prefix='Test')

        res = {k: torch.cat(v, 0).numpy() for k, v in res.items()}
        out_path = os.path.join(self.args.save_dir, 'test_features.npz')
        np.savez_compressed(out_path, **res)
        self.logger.info(f"Features saved: {out_path}")
        return res


class KukanjaTrainerEngine_2c(KukanjaTrainerEngine):
    """N-level trainer with FiLM disease conditioning (mjm_2c_nlevel)."""

    def _norm_spatial(self, st):
        mn = st.min(dim=0, keepdim=True).values
        mx = st.max(dim=0, keepdim=True).values
        return (st - mn) / (mx - mn + 1e-6)

    def compute_loss(self, batch_data):
        X = batch_data['X'].to(self.device)
        X_norm = self._normalize(X)

        disease = batch_data['disease_score'].to(self.device).unsqueeze(-1)
        spatial_n = batch_data['spatial_norm'].to(self.device)
        spatial_n = self._norm_spatial(spatial_n)

        y_list = [batch_data[f'y_{name}'].to(self.device)
                  for name in self.label_names]

        recon_X, logits_list, z = self.model(X_norm, disease, spatial_n)

        loss_recon = F.mse_loss(recon_X, X_norm)
        level_losses = {}
        total_loss = loss_recon
        for i, name in enumerate(self.label_names):
            loss_i = F.cross_entropy(logits_list[i], y_list[i])
            level_losses[name] = loss_i.item()
            total_loss = total_loss + loss_i

        metrics = {'total': total_loss, 'recon': loss_recon.item()}
        metrics.update(level_losses)
        logits_dict = {name: logits_list[i] for i, name in enumerate(self.label_names)}
        return total_loss, loss_recon, metrics, logits_dict, z


class KukanjaTrainerEngine_Attn(KukanjaTrainerEngine):
    """N-level trainer for mjm_attn_nlevel (needs spatial + batch_id)."""

    def _norm_spatial(self, st):
        mn = st.min(dim=0, keepdim=True).values
        mx = st.max(dim=0, keepdim=True).values
        return (st - mn) / (mx - mn + 1e-6)

    def compute_loss(self, batch_data):
        X = batch_data['X'].to(self.device)
        X_norm = self._normalize(X)

        spatial = batch_data['spatial'].to(self.device).float()
        spatial = self._norm_spatial(spatial)
        batch_id = batch_data['batch_id'].to(self.device)

        y_list = [batch_data[f'y_{name}'].to(self.device)
                  for name in self.label_names]

        recon_X, logits_list, z = self.model(X_norm, spatial, batch_id)

        loss_recon = F.mse_loss(recon_X, X_norm)
        level_losses = {}
        total_loss = loss_recon
        for i, name in enumerate(self.label_names):
            loss_i = F.cross_entropy(logits_list[i], y_list[i])
            level_losses[name] = loss_i.item()
            total_loss = total_loss + loss_i

        metrics = {'total': total_loss, 'recon': loss_recon.item()}
        metrics.update(level_losses)
        logits_dict = {name: logits_list[i] for i, name in enumerate(self.label_names)}
        return total_loss, loss_recon, metrics, logits_dict, z
