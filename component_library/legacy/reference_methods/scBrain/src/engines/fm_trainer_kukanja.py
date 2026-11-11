"""
N-level FM training engine for Kukanja data.
Adapts ScGPTTrainerEngine for:
  - N classification levels (instead of hardcoded 3)
  - Kukanja dataloader format (y_{name} keys, no confidence weighting)
  - No reconstruction loss
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class FMTrainerKukanja:
    """N-level foundation model trainer for Kukanja datasets."""

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

    def compute_loss(self, batch_data):
        X = batch_data['X'].to(self.device)
        y_list = [batch_data[f'y_{name}'].to(self.device) for name in self.label_names]

        logits_list, cell_emb = self.model(X)

        total_loss = torch.tensor(0.0, device=self.device)
        level_losses = {}
        for i, name in enumerate(self.label_names):
            loss_i = F.cross_entropy(logits_list[i], y_list[i])
            level_losses[name] = loss_i.item()
            total_loss = total_loss + loss_i

        metrics = {'total': total_loss.item()}
        metrics.update(level_losses)
        logits_dict = {name: logits_list[i] for i, name in enumerate(self.label_names)}
        return total_loss, metrics, logits_dict, cell_emb

    def train_epoch(self, dataloader):
        self.model.train()
        keys = ['total'] + list(self.label_names)
        agg = {k: 0.0 for k in keys}
        for batch_data in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            loss, metrics, _, _ = self.compute_loss(batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            for k in agg:
                agg[k] += metrics[k]
        return {k: v / len(dataloader) for k, v in agg.items()}

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        keys = ['total'] + list(self.label_names)
        agg = {k: 0.0 for k in keys}
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            _, metrics, _, _ = self.compute_loss(batch_data)
            for k in agg:
                agg[k] += metrics[k]
        return {k: v / len(dataloader) for k, v in agg.items()}

    def _fmt(self, m):
        parts = [f"total:{m['total']:.4f}"]
        for name in self.label_names:
            parts.append(f"{name}:{m[name]:.4f}")
        return " ".join(parts)

    def train(self, train_loader, val_loader):
        os.makedirs(self.args.save_dir, exist_ok=True)
        best_path = os.path.join(self.args.save_dir, 'best_model.pth')
        patience_counter = 0
        best_val_m = None

        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            self.logger.info(f"[Resume] Loaded checkpoint from {best_path}")

        for epoch in range(1, self.args.max_epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m = self.eval_epoch(val_loader)
            self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.args.max_epochs}] "
                f"Time:{time.time()-t0:.1f}s LR:{self.scheduler.get_last_lr()[0]:.2e}"
            )
            self.logger.info(f"  [Train] {self._fmt(train_m)}")
            self.logger.info(f"  [Val]   {self._fmt(val_m)}")
            if best_val_m:
                self.logger.info(f"  [Best]  {self._fmt(best_val_m)}")

            if val_m['total'] < self.best_val_loss:
                self.best_val_loss = val_m['total']
                best_val_m = val_m
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
                self.logger.info("Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info("Early stopping triggered!")
                    break

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("\nLoading best model for test...")
        best_path = os.path.join(self.args.save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        self.model.eval()

        keys = ['total'] + list(self.label_names)
        agg = {k: 0.0 for k in keys}
        all_true = {t: [] for t in self.label_names}
        all_pred = {t: [] for t in self.label_names}
        all_prob = {t: [] for t in self.label_names}
        res = {'cell_emb': [], 'spatial': [], 'batch': []}

        for batch_data in test_loader:
            _, metrics, logits_dict, cell_emb = self.compute_loss(batch_data)
            for k in agg:
                agg[k] += metrics[k]
            res['cell_emb'].append(cell_emb.cpu())
            res['spatial'].append(batch_data['spatial'].cpu())
            res['batch'].append(batch_data['batch_id'].cpu())

            for t in self.label_names:
                all_true[t].append(batch_data[f'y_{t}'].cpu().numpy())
                all_pred[t].append(logits_dict[t].argmax(-1).cpu().numpy())
                all_prob[t].append(F.softmax(logits_dict[t], -1).cpu().numpy())

        test_m = {k: v / len(test_loader) for k, v in agg.items()}
        self.logger.info(f"  [Test] {self._fmt(test_m)}")

        final_true = {k: np.concatenate(v) for k, v in all_true.items()}
        final_pred = {k: np.concatenate(v) for k, v in all_pred.items()}
        final_prob = {k: np.concatenate(v, axis=0) for k, v in all_prob.items()}

        # Classification metrics
        self.logger.info("========== Test Classification Metrics ==========")
        for task in self.label_names:
            mp, mr, mf, _ = precision_recall_fscore_support(
                final_true[task], final_pred[task], average='macro', zero_division=0)
            _, _, mif, _ = precision_recall_fscore_support(
                final_true[task], final_pred[task], average='micro', zero_division=0)
            pp, pr, _, sup = precision_recall_fscore_support(
                final_true[task], final_pred[task], average=None, zero_division=0)
            self.logger.info(f"[{task.upper()}] Macro  - P:{mp:.4f} R:{mr:.4f} F1:{mf:.4f}")
            self.logger.info(f"[{task.upper()}] Micro  - F1:{mif:.4f}")
            parts = [f"C{i}(S={sup[i]}):P={pp[i]:.4f}/R={pr[i]:.4f}"
                     for i in range(len(pp)) if sup[i] > 0]
            self.logger.info(f"[{task.upper()}] Per-class: {' | '.join(parts)}")
            self.logger.info("-" * 50)

        # AUC-ROC
        self.logger.info("========== Test AUC-ROC Metrics ==========")
        for i, task in enumerate(self.label_names):
            y_t, y_p = final_true[task], final_prob[task]
            y_oh = np.zeros_like(y_p)
            y_oh[np.arange(len(y_t)), y_t] = 1
            mi_auc = roc_auc_score(y_oh.ravel(), y_p.ravel())
            va = []
            for c in range(self.output_num[i]):
                b = (y_t == c).astype(int)
                if len(np.unique(b)) == 2:
                    va.append(roc_auc_score(b, y_p[:, c]))
            ma_auc = np.mean(va) if va else 0.0
            self.logger.info(
                f"[{task.upper()}] AUC-ROC - Macro:{ma_auc:.4f} Micro:{mi_auc:.4f} "
                f"(Valid:{len(va)}/{self.output_num[i]})")
            self.logger.info("-" * 50)

        res_np = {k: torch.cat(v, 0).numpy() for k, v in res.items()}
        out_path = os.path.join(self.args.save_dir, 'test_features.npz')
        np.savez_compressed(out_path, **res_np)
        self.logger.info(f"Features saved: {out_path}")
        return res_np
