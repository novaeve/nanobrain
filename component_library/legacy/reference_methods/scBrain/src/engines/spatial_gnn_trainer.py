"""
Training engine for Spatial GNN cell-type annotation.

Key differences from base TrainerEngine:
  - Operates on full graph (not mini-batches) per donor subgraph
  - Uses subgraph sampling: each batch is a donor's cells + their KNN graph
  - Model forward: (recon, logits, z) with edge_index argument
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class SpatialGNNTrainerEngine:
    def __init__(self, model, optimizer, scheduler, device, logger, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.args = args
        self.best_val_loss = float('inf')

    def _normalize(self, x):
        return torch.log1p(x)

    def compute_loss(self, X, y_c, y_sc, y_st, confidence, edge_index):
        X = self._normalize(X)
        recon, logits, z = self.model(X, edge_index)
        logit_c, logit_sc, logit_st = logits

        loss_recon = F.mse_loss(recon, X)
        loss_c = F.cross_entropy(logit_c, y_c)
        loss_sc = F.cross_entropy(logit_sc, y_sc)
        loss_st = (F.cross_entropy(logit_st, y_st, reduction='none') * confidence).mean()

        total_loss = loss_recon + loss_c + loss_sc + loss_st

        metrics = {
            'total': total_loss.item(), 'recon': loss_recon.item(),
            'class': loss_c.item(), 'subclass': loss_sc.item(), 'supertype': loss_st.item(),
        }
        logits_dict = {'class': logit_c, 'subclass': logit_sc, 'supertype': logit_st}
        return total_loss, metrics, logits_dict, z

    def _run_subgraphs(self, subgraphs, train=True):
        """Process donor subgraphs. Each subgraph is a dict with tensors + edge_index."""
        if train:
            self.model.train()
        else:
            self.model.eval()

        agg = {'total': 0.0, 'recon': 0.0, 'class': 0.0, 'subclass': 0.0, 'supertype': 0.0}
        n_batches = 0

        for sg in subgraphs:
            X = sg['X'].to(self.device)
            y_c = sg['y_class'].to(self.device)
            y_sc = sg['y_subclass'].to(self.device)
            y_st = sg['y_supertype'].to(self.device)
            conf = sg['confidence'].to(self.device)
            ei = sg['edge_index'].to(self.device)

            if train:
                self.optimizer.zero_grad()
                loss, metrics, _, _ = self.compute_loss(X, y_c, y_sc, y_st, conf, ei)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            else:
                with torch.no_grad():
                    _, metrics, _, _ = self.compute_loss(X, y_c, y_sc, y_st, conf, ei)

            for k in agg:
                agg[k] += metrics[k]
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in agg.items()}

    def train(self, train_subgraphs, val_subgraphs):
        os.makedirs(self.args.save_dir, exist_ok=True)
        best_path = os.path.join(self.args.save_dir, "best_model.pth")
        patience_counter = 0
        best_val_metrics = None

        for epoch in range(1, self.args.max_epochs + 1):
            t0 = time.time()
            train_m = self._run_subgraphs(train_subgraphs, train=True)
            val_m = self._run_subgraphs(val_subgraphs, train=False)
            self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.args.max_epochs:03d}] | "
                f"Time: {time.time()-t0:.1f}s | LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            self.logger.info(
                f"  [Train] Total: {train_m['total']:.4f} | Recon: {train_m['recon']:.4f} | "
                f"Class: {train_m['class']:.4f} | Subclass: {train_m['subclass']:.4f} | "
                f"Supertype: {train_m['supertype']:.4f}"
            )
            self.logger.info(
                f"  [Val]   Total: {val_m['total']:.4f} | Recon: {val_m['recon']:.4f} | "
                f"Class: {val_m['class']:.4f} | Subclass: {val_m['subclass']:.4f} | "
                f"Supertype: {val_m['supertype']:.4f}"
            )
            if best_val_metrics is not None:
                self.logger.info(
                    f"  [Best]  Total: {self.best_val_loss:.4f} | Recon: {best_val_metrics['recon']:.4f} | "
                    f"Class: {best_val_metrics['class']:.4f} | Subclass: {best_val_metrics['subclass']:.4f} | "
                    f"Supertype: {best_val_metrics['supertype']:.4f}"
                )

            if val_m['total'] < self.best_val_loss:
                self.best_val_loss = val_m['total']
                best_val_metrics = val_m
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
                self.logger.info("Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info("Early stopping triggered!")
                    break

    @torch.no_grad()
    def test(self, test_subgraphs):
        self.logger.info("\nLoading best model for test-set evaluation...")
        best_path = os.path.join(self.args.save_dir, "best_model.pth")
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        self.model.eval()

        all_y_true = {'class': [], 'subclass': [], 'supertype': []}
        all_y_pred = {'class': [], 'subclass': [], 'supertype': []}
        all_y_prob = {'class': [], 'subclass': [], 'supertype': []}
        all_z, all_spatial, all_batch, all_st, all_cps = [], [], [], [], []
        agg = {'total': 0.0, 'recon': 0.0, 'class': 0.0, 'subclass': 0.0, 'supertype': 0.0}
        n_batches = 0

        for sg in test_subgraphs:
            X = sg['X'].to(self.device)
            y_c = sg['y_class'].to(self.device)
            y_sc = sg['y_subclass'].to(self.device)
            y_st = sg['y_supertype'].to(self.device)
            conf = sg['confidence'].to(self.device)
            ei = sg['edge_index'].to(self.device)

            _, metrics, logits_dict, z = self.compute_loss(X, y_c, y_sc, y_st, conf, ei)

            for k in agg:
                agg[k] += metrics[k]
            n_batches += 1

            all_z.append(z.cpu())
            all_spatial.append(sg['spatial'])
            all_batch.append(sg['batch_id'])
            all_st.append(sg['y_supertype'])
            all_cps.append(sg['cps'])

            for task_key, logit_key in [('class', 'class'), ('subclass', 'subclass'), ('supertype', 'supertype')]:
                all_y_true[task_key].append(sg[f'y_{task_key}'].numpy())
                all_y_pred[task_key].append(logits_dict[task_key].argmax(dim=-1).cpu().numpy())
                all_y_prob[task_key].append(F.softmax(logits_dict[task_key], dim=-1).cpu().numpy())

        test_m = {k: v / max(n_batches, 1) for k, v in agg.items()}
        self.logger.info(
            f"  [Test] Total: {test_m['total']:.4f} | Recon: {test_m['recon']:.4f} | "
            f"Class: {test_m['class']:.4f} | Subclass: {test_m['subclass']:.4f} | "
            f"Supertype: {test_m['supertype']:.4f}"
        )

        final_true = {k: np.concatenate(v) for k, v in all_y_true.items()}
        final_pred = {k: np.concatenate(v) for k, v in all_y_pred.items()}
        final_prob = {k: np.concatenate(v, axis=0) for k, v in all_y_prob.items()}

        self._log_classification_metrics(final_true, final_pred)
        self._log_auc_roc_metrics(final_true, final_prob, {
            'class': self.args.output_num[0],
            'subclass': self.args.output_num[1],
            'supertype': self.args.output_num[2],
        })

        res = {
            'latent': torch.cat(all_z).numpy(),
            'spatial': torch.cat(all_spatial).numpy(),
            'batch': torch.cat(all_batch).numpy(),
            'supertype': torch.cat(all_st).numpy(),
            'cps': torch.cat(all_cps).numpy(),
        }
        out_path = os.path.join(self.args.save_dir, 'test_features.npz')
        np.savez_compressed(out_path, **res)
        self.logger.info(f"Features saved to: {out_path}")
        return res

    def _log_classification_metrics(self, y_true_dict, y_pred_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} Classification Metrics ==========")
        for task in ['class', 'subclass', 'supertype']:
            y_true, y_pred = y_true_dict[task], y_pred_dict[task]
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
            per_p, per_r, _, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            self.logger.info(f"[{task.upper()}] Macro  - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
            self.logger.info(f"[{task.upper()}] Micro  - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}")
            logs = [f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}" for i in range(len(per_p)) if support[i] > 0 or per_p[i] > 0]
            self.logger.info(f"[{task.upper()}] Per-class: {' | '.join(logs)}")
            self.logger.info("-" * 50)

    def _log_auc_roc_metrics(self, y_true_dict, y_prob_dict, num_classes_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} AUC-ROC Metrics ==========")
        for task in ['class', 'subclass', 'supertype']:
            y_true, y_prob = y_true_dict[task], y_prob_dict[task]
            num_c = num_classes_dict[task]
            y_onehot = np.zeros_like(y_prob)
            y_onehot[np.arange(len(y_true)), y_true] = 1
            micro_auc = roc_auc_score(y_onehot.ravel(), y_prob.ravel())
            valid_aucs = []
            for c in range(num_c):
                binary = (y_true == c).astype(int)
                if len(np.unique(binary)) == 2:
                    valid_aucs.append(roc_auc_score(binary, y_prob[:, c]))
            macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
            self.logger.info(f"[{task.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid: {len(valid_aucs)}/{num_c})")
            self.logger.info("-" * 50)
