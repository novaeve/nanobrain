"""
Training engine for scGPT cell-type annotation.

Differences from MJM TrainerEngine:
  - No reconstruction loss (scGPT is classification-only here).
  - Model forward: (logits, cell_emb)  vs MJM's (recon_X, logits, z).
  - val / eval uses torch.no_grad() (pure evaluation, no gradient step).
  - test saves cell_emb as the latent representation.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class ScGPTTrainerEngine:
    def __init__(self, model, optimizer, scheduler, device, logger, args):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.logger    = logger
        self.args      = args
        self.best_val_loss = float("inf")

    # ── loss ─────────────────────────────────────────────────────────────────

    def compute_loss(self, batch_data):
        X          = batch_data["X"].to(self.device)
        y_c        = batch_data["y_class"].to(self.device)
        y_sc       = batch_data["y_subclass"].to(self.device)
        y_st       = batch_data["y_supertype"].to(self.device)
        confidence = batch_data["confidence"].to(self.device)

        logits, cell_emb = self.model(X)
        logit_c, logit_sc, logit_st = logits

        loss_c  = F.cross_entropy(logit_c,  y_c)
        loss_sc = F.cross_entropy(logit_sc, y_sc)

        # confidence-weighted supertype loss (same as MJM)
        loss_st = (F.cross_entropy(logit_st, y_st, reduction="none") * confidence).mean()

        total_loss = loss_c + loss_sc + loss_st

        metrics = {
            "total":     total_loss.item(),
            "class":     loss_c.item(),
            "subclass":  loss_sc.item(),
            "supertype": loss_st.item(),
        }
        logits_dict = {
            "class":     logit_c,
            "subclass":  logit_sc,
            "supertype": logit_st,
        }
        return total_loss, metrics, logits_dict, cell_emb

    # ── one epoch ────────────────────────────────────────────────────────────

    def train_epoch(self, dataloader):
        self.model.train()
        agg = {"total": 0.0, "class": 0.0, "subclass": 0.0, "supertype": 0.0}

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
        agg = {"total": 0.0, "class": 0.0, "subclass": 0.0, "supertype": 0.0}

        for batch_data in tqdm(dataloader, desc="Evaluating"):
            _, metrics, _, _ = self.compute_loss(batch_data)
            for k in agg:
                agg[k] += metrics[k]

        return {k: v / len(dataloader) for k, v in agg.items()}

    # ── full training loop ───────────────────────────────────────────────────

    def train(self, train_loader, val_loader):
        os.makedirs(self.args.save_dir, exist_ok=True)
        best_model_path   = os.path.join(self.args.save_dir, "best_model.pth")
        patience_counter  = 0
        best_val_metrics  = None

        # Resume from existing checkpoint if available
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            self.logger.info(f"[Resume] Loaded checkpoint from {best_model_path}")

        for epoch in range(1, self.args.max_epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m   = self.eval_epoch(val_loader)
            self.scheduler.step()

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.args.max_epochs:03d}] | "
                f"Time: {time.time()-t0:.1f}s | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            self.logger.info(
                f"  [Train] Total: {train_m['total']:.4f} | "
                f"Class: {train_m['class']:.4f} | "
                f"Subclass: {train_m['subclass']:.4f} | "
                f"Supertype: {train_m['supertype']:.4f}"
            )
            self.logger.info(
                f"  [Val]   Total: {val_m['total']:.4f} | "
                f"Class: {val_m['class']:.4f} | "
                f"Subclass: {val_m['subclass']:.4f} | "
                f"Supertype: {val_m['supertype']:.4f}"
            )
            if best_val_metrics is not None:
                self.logger.info(
                    f"  [Best Val] Total: {self.best_val_loss:.4f} | "
                    f"Class: {best_val_metrics['class']:.4f} | "
                    f"Subclass: {best_val_metrics['subclass']:.4f} | "
                    f"Supertype: {best_val_metrics['supertype']:.4f}"
                )

            if val_m["total"] < self.best_val_loss:
                self.best_val_loss = val_m["total"]
                best_val_metrics   = val_m
                patience_counter   = 0
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info("Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info("Early stopping triggered!")
                    break

    # ── test ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("\nLoading best model for test-set evaluation...")
        best_model_path = os.path.join(self.args.save_dir, "best_model.pth")
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
        self.model.eval()

        res        = {"cell_emb": [], "spatial": [], "batch": [], "supertype": [], "cps": []}
        agg        = {"total": 0.0, "class": 0.0, "subclass": 0.0, "supertype": 0.0}
        all_y_true = {"class": [], "subclass": [], "supertype": []}
        all_y_pred = {"class": [], "subclass": [], "supertype": []}
        all_y_prob = {"class": [], "subclass": [], "supertype": []}

        for batch_data in test_loader:
            _, metrics, logits_dict, cell_emb = self.compute_loss(batch_data)

            res["cell_emb"].append(cell_emb.cpu())
            res["spatial"].append(batch_data["spatial"].cpu())
            res["batch"].append(batch_data["batch_id"].cpu())
            res["supertype"].append(batch_data["y_supertype"].cpu())
            res["cps"].append(batch_data["cps"].cpu())

            for k in agg:
                agg[k] += metrics[k]

            all_y_true["class"].append(batch_data["y_class"].cpu().numpy())
            all_y_true["subclass"].append(batch_data["y_subclass"].cpu().numpy())
            all_y_true["supertype"].append(batch_data["y_supertype"].cpu().numpy())

            all_y_prob["class"].append(
                F.softmax(logits_dict["class"], dim=-1).cpu().numpy()
            )
            all_y_prob["subclass"].append(
                F.softmax(logits_dict["subclass"], dim=-1).cpu().numpy()
            )
            all_y_prob["supertype"].append(
                F.softmax(logits_dict["supertype"], dim=-1).cpu().numpy()
            )

            all_y_pred["class"].append(
                logits_dict["class"].argmax(dim=-1).cpu().numpy()
            )
            all_y_pred["subclass"].append(
                logits_dict["subclass"].argmax(dim=-1).cpu().numpy()
            )
            all_y_pred["supertype"].append(
                logits_dict["supertype"].argmax(dim=-1).cpu().numpy()
            )

        test_m = {k: v / len(test_loader) for k, v in agg.items()}
        self.logger.info(
            f"  [Test] Total: {test_m['total']:.4f} | "
            f"Class: {test_m['class']:.4f} | "
            f"Subclass: {test_m['subclass']:.4f} | "
            f"Supertype: {test_m['supertype']:.4f}"
        )

        final_y_true = {k: np.concatenate(v)        for k, v in all_y_true.items()}
        final_y_pred = {k: np.concatenate(v)        for k, v in all_y_pred.items()}
        final_y_prob = {k: np.concatenate(v, axis=0) for k, v in all_y_prob.items()}

        self.log_classification_metrics(final_y_true, final_y_pred, prefix="Test")
        self.log_auc_roc_metrics(
            final_y_true,
            final_y_prob,
            num_classes_dict={"class": 3, "subclass": 24, "supertype": 137},
            prefix="Test",
        )

        res_np = {k: torch.cat(v, dim=0).numpy() for k, v in res.items()}
        out_path = os.path.join(self.args.save_dir, "test_features.npz")
        np.savez_compressed(out_path, **res_np)
        self.logger.info(f"Features saved to: {out_path}")

        return res_np

    # ── metric helpers (identical to MJM TrainerEngine) ──────────────────────

    def log_classification_metrics(self, y_true_dict, y_pred_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} Classification Metrics ==========")
        for task in ["class", "subclass", "supertype"]:
            y_true = y_true_dict[task]
            y_pred = y_pred_dict[task]

            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="micro", zero_division=0
            )
            per_p, per_r, _, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            self.logger.info(
                f"[{task.upper()}] Macro  - P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}"
            )
            self.logger.info(
                f"[{task.upper()}] Micro  - P: {micro_p:.4f} | R: {micro_r:.4f} | F1: {micro_f1:.4f}"
            )

            per_class_logs = [
                f"C{i}(S={support[i]}): P={per_p[i]:.4f}/R={per_r[i]:.4f}"
                for i in range(len(per_p))
                if support[i] > 0 or per_p[i] > 0
            ]
            self.logger.info(f"[{task.upper()}] Per-class: {' | '.join(per_class_logs)}")
            self.logger.info("-" * 50)

    def log_auc_roc_metrics(self, y_true_dict, y_prob_dict, num_classes_dict, prefix="Test"):
        self.logger.info(f"========== {prefix} AUC-ROC Metrics ==========")
        for task in ["class", "subclass", "supertype"]:
            y_true  = y_true_dict[task]
            y_prob  = y_prob_dict[task]
            num_c   = num_classes_dict[task]

            y_true_onehot = np.zeros_like(y_prob)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            micro_auc = roc_auc_score(y_true_onehot.ravel(), y_prob.ravel())

            valid_aucs = []
            for c in range(num_c):
                binary = (y_true == c).astype(int)
                if len(np.unique(binary)) == 2:
                    valid_aucs.append(roc_auc_score(binary, y_prob[:, c]))
            macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

            self.logger.info(
                f"[{task.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | "
                f"Micro: {micro_auc:.4f} | "
                f"(Valid classes: {len(valid_aucs)}/{num_c})"
            )
            self.logger.info("-" * 50)
