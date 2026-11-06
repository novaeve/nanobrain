import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class TrainerEngine:
    def __init__(self, model, optimizer, scheduler, device, logger, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.args = args
        self.best_val_loss = float('inf')

    def _normalize(self, x, is_norm=True, method='log1p'):
        if is_norm:
            if method == 'log1p':
                return torch.log1p(x)
            else:
                raise ValueError(f"Unsupported normalization: {method}")
        else:
            if method == 'log1p':
                return torch.exp(x)-1
            else:
                raise ValueError(f"Unsupported normalization: {method}")



    def log_classification_metrics(self, y_true_dict, y_pred_dict, prefix="Test"):
        """
        计算并记录三个层级分类任务的 Precision, Recall, Macro/Micro F1 以及单类指标。
        
        Args:
            y_true_dict: dict, 包含 'class', 'subclass', 'supertype' 的真实标签 Numpy 数组
            y_pred_dict: dict, 包含 'class', 'subclass', 'supertype' 的预测标签 Numpy 数组
            prefix: str, 日志前缀，例如 'Val' 或 'Test'
        """
        self.logger.info(f"========== {prefix} Classification Metrics ==========")
        
        tasks = ['class', 'subclass', 'supertype']
        
        for task in tasks:
            y_true = y_true_dict[task]
            y_pred = y_pred_dict[task]
            
            # 计算 Macro 和 Micro 级别的总体指标
            # zero_division=0 极其重要，防止某些罕见细胞在当前 batch/epoch 没出现时报错
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='micro', zero_division=0
            )
            
            # 计算 Per-class (单类) 级别的指标
            per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # 1. 记录总体指标 (Macro / Micro)
            self.logger.info(f"[{task.upper()}] Macro  - Precision: {macro_p:.4f} | Recall: {macro_r:.4f} | F1: {macro_f1:.4f}")
            self.logger.info(f"[{task.upper()}] Micro  - Precision: {micro_p:.4f} | Recall: {micro_r:.4f} | F1: {micro_f1:.4f}")
            
            # 2. 紧凑记录单类指标 (Per-class)
            # 为了防止日志过长，只打印 Support > 0 (真实存在的类) 或有预测值的类
            per_class_logs = []
            for i in range(len(per_class_p)):
                # 过滤掉既没有真实样本，又没有被预测为该类的“空类”
                if support[i] > 0 or per_class_p[i] > 0:
                    per_class_logs.append(f"C{i}(S={support[i]}): P={per_class_p[i]:.4f}/R={per_class_r[i]:.4f}")
            
            # 将单类指标合并为一行长字符串记录（或者按固定数量换行）
            per_class_str = " | ".join(per_class_logs)
            self.logger.info(f"[{task.upper()}] Per-class P/R: {per_class_str}")
            self.logger.info("-" * 50)



    def log_auc_roc_metrics(self, y_true_dict, y_prob_dict, num_classes_dict, prefix="Test"):
        """
        鲁棒的多分类 AUC-ROC 计算模块。
        
        Args:
            y_true_dict: dict, 包含真实标签的 1D Numpy 数组
            y_prob_dict: dict, 包含经过 Softmax 的预测概率的 2D Numpy 数组 [N, C]
            num_classes_dict: dict, 各个任务的类别总数，例如 {'class': 3, 'subclass': 24, 'supertype': 137}
            prefix: str, 日志前缀
        """
        self.logger.info(f"========== {prefix} AUC-ROC Metrics ==========")
        
        tasks = ['class', 'subclass', 'supertype']
        
        for task in tasks:
            y_true = y_true_dict[task]
            y_prob = y_prob_dict[task]
            num_c = num_classes_dict[task]
            
            # 1. 计算 Micro AUC (全局看作一个巨大的二分类任务)
            # 将 y_true 转为 one-hot 格式，方便与概率矩阵直接计算
            y_true_onehot = np.zeros_like(y_prob)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            
            # Micro AUC: 展平所有维度直接算
            micro_auc = roc_auc_score(y_true_onehot.ravel(), y_prob.ravel())
            
            # 2. 鲁棒地计算 Macro AUC (One-vs-Rest)
            # 针对长尾分布：只对存在正样本且存在负样本的类进行计算，防止 sklearn 报错
            valid_auc_scores = []
            for c in range(num_c):
                y_true_binary = (y_true == c).astype(int)
                
                # 只有当该类同时存在 0 和 1 的时候（即既有正样本也有负样本），才能算 AUC
                if len(np.unique(y_true_binary)) == 2:
                    auc = roc_auc_score(y_true_binary, y_prob[:, c])
                    valid_auc_scores.append(auc)
                    
            # 取有效类的均值作为 Macro AUC
            macro_auc = np.mean(valid_auc_scores) if len(valid_auc_scores) > 0 else 0.0
            
            # 3. 记录日志
            self.logger.info(f"[{task.upper()}] AUC-ROC - Macro: {macro_auc:.4f} | Micro: {micro_auc:.4f} | (Valid classes: {len(valid_auc_scores)}/{num_c})")
            
            self.logger.info("-" * 50)



    def compute_loss(self, batch_data):
        X = batch_data['X'].to(self.device)
        # batch_id = batch_data['batch_id'].to(self.device)
        y_c = batch_data['y_class'].to(self.device)
        y_sc = batch_data['y_subclass'].to(self.device)
        y_st = batch_data['y_supertype'].to(self.device)
        confidence = batch_data['confidence'].to(self.device)

        # recon_X, logits, z = self.model(X, batch_id)
        X = self._normalize(X)  # log(1+x) 处理输入特征的数值范围
        recon_X, logits, z = self.model(X)

        logit_c, logit_sc, logit_st = logits

        loss_recon = F.mse_loss(recon_X, X)
        loss_c = F.cross_entropy(logit_c, y_c)
        loss_sc = F.cross_entropy(logit_sc, y_sc)
        
        # 带有 Confidence 的样本级加权 Loss
        loss_st_unweighted = F.cross_entropy(logit_st, y_st, reduction='none')
        loss_st = (loss_st_unweighted * confidence).mean()

        # total_loss = 1.0 * loss_recon + 0.5 * loss_c + 1.0 * loss_sc + 2.0 * loss_st
        total_loss = 1.0 * loss_recon + 1.0 * loss_c + 1.0 * loss_sc + 1.0 * loss_st
        
        metrics = {'total': total_loss, 'recon': loss_recon.item(), 'class': loss_c.item(), 'subclass': loss_sc.item(), 'supertype': loss_st.item()}
        logits_dict = {'class': logit_c, 'subclass': logit_sc, 'supertype': logit_st}
        return total_loss, loss_recon, metrics, logits_dict, z

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_metrics = {'total': 0.0, 'recon': 0.0, 'class': 0.0, 'subclass': 0.0, 'supertype': 0.0}
        
        for batch_data in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()
            loss, _, metrics, _, _ = self.compute_loss(batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            for k in epoch_metrics: epoch_metrics[k] += metrics[k]
            
        return {k: v / len(dataloader) for k, v in epoch_metrics.items()}

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        epoch_metrics = {'total': 0.0, 'recon': 0.0, 'class': 0.0, 'subclass': 0.0, 'supertype': 0.0}

        for batch_data in tqdm(dataloader, desc="Evaluating"):
            _, _, metrics, _, _ = self.compute_loss(batch_data)
            for k in epoch_metrics: epoch_metrics[k] += metrics[k]

        return {k: v / len(dataloader) for k, v in epoch_metrics.items()}

    def train(self, train_loader, val_loader):
        os.makedirs(self.args.save_dir, exist_ok=True)
        best_model_path = os.path.join(self.args.save_dir, "best_model.pth")
        patience_counter = 0

        for epoch in range(1, self.args.max_epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)
            self.scheduler.step()
            
            self.logger.info(f"Epoch [{epoch:03d}/{self.args.max_epochs:03d}] | Time: {time.time()-start_time:.1f}s | LR: {self.scheduler.get_last_lr()[0]:.2e}")
            self.logger.info(f"  [Train] Total: {train_metrics['total']:.4f} | Recon: {train_metrics['recon']:.4f} | Class: {train_metrics['class']:.4f} | Subclass: {train_metrics['subclass']:.4f} | Supertype: {train_metrics['supertype']:.4f}")
            self.logger.info(f"  [Val]   Total: {val_metrics['total']:.4f} | Recon: {val_metrics['recon']:.4f} | Class: {val_metrics['class']:.4f} | Subclass: {val_metrics['subclass']:.4f} | Supertype: {val_metrics['supertype']:.4f}")
            if epoch > 1:
                self.logger.info(f"  [Best Val] Total: {self.best_val_loss:.4f} | Recon: {best_val_metrics['recon']:.4f} | Class: {best_val_metrics['class']:.4f} | Subclass: {best_val_metrics['subclass']:.4f} | Supertype: {best_val_metrics['supertype']:.4f}")
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                best_val_metrics = val_metrics
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f"Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info("Early stopping triggered!")
                    break

    @torch.no_grad()
    def test(self, test_loader):
        self.logger.info("\nLoading best model for Test set evaluation...")
        best_model_path = os.path.join(self.args.save_dir, "best_model.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()
        
        res = {'latent': [], 'spatial': [], 'batch': [], 'supertype': [], 'cps': []}
        epoch_metrics = {'total': 0.0, 'recon': 0.0, 'class': 0.0, 'subclass': 0.0, 'supertype': 0.0}
        # 用于收集全 Epoch 的真实标签和预测结果
        all_y_true = {'class': [], 'subclass': [], 'supertype': []}
        all_y_pred = {'class': [], 'subclass': [], 'supertype': []}
        # 🌟 新增：用于收集预测概率 [N, C] 的容器
        all_y_prob = {'class': [], 'subclass': [], 'supertype': []}
        for batch_data in test_loader:
            _, _, metrics, logits_dict, z = self.compute_loss(batch_data)
            res['latent'].append(z.cpu())
            res['spatial'].append(batch_data['spatial'].cpu())
            res['batch'].append(batch_data['batch_id'].cpu())
            res['supertype'].append(batch_data['y_supertype'].cpu())
            res['cps'].append(batch_data['cps'].cpu())
            for k in epoch_metrics: epoch_metrics[k] += metrics[k]

            # 2. 提取真实的标签并转为 CPU Numpy
            all_y_true['class'].append(batch_data['y_class'].cpu().numpy())
            all_y_true['subclass'].append(batch_data['y_subclass'].cpu().numpy())
            all_y_true['supertype'].append(batch_data['y_supertype'].cpu().numpy())
            
            # 🌟 新增：对 logits 进行 Softmax 转为概率，并收集 [Batch, Num_Classes]
            all_y_prob['class'].append(F.softmax(logits_dict['class'], dim=-1).cpu().numpy())
            all_y_prob['subclass'].append(F.softmax(logits_dict['subclass'], dim=-1).cpu().numpy())
            all_y_prob['supertype'].append(F.softmax(logits_dict['supertype'], dim=-1).cpu().numpy())

            # 3. 从 logits 中提取预测类别 (argmax)，并转为 CPU Numpy
            # 假设 compute_loss 返回了 logit_c, logit_sc, logit_st 组成的 logits_dict
            # 这里为了演示，我假设你的 logits 已经拿到了
            logit_c = logits_dict['class']
            logit_sc = logits_dict['subclass']
            logit_st = logits_dict['supertype']
        
            all_y_pred['class'].append(logit_c.argmax(dim=-1).cpu().numpy())
            all_y_pred['subclass'].append(logit_sc.argmax(dim=-1).cpu().numpy())
            all_y_pred['supertype'].append(logit_st.argmax(dim=-1).cpu().numpy())

        test_metrics =  {k: v / len(test_loader) for k, v in epoch_metrics.items()}            
        self.logger.info(f"  [Test] Total: {test_metrics['total']:.4f} | Recon: {test_metrics['recon']:.4f} | Class: {test_metrics['class']:.4f} | Subclass: {test_metrics['subclass']:.4f} | Supertype: {test_metrics['supertype']:.4f}")
        # 将 List of Numpy arrays 拼接成一个完整的大一维 Array
        final_y_true = {k: np.concatenate(v) for k, v in all_y_true.items()}
        final_y_pred = {k: np.concatenate(v) for k, v in all_y_pred.items()}
        
        # 🌟 调用我们刚才写的日志函数计算并打印 P, R, F1
        self.log_classification_metrics(final_y_true, final_y_pred, prefix='test')

        # 🌟 新增：拼接概率矩阵 (沿着样本维度 dim=0 拼接)
        final_y_prob = {k: np.concatenate(v, axis=0) for k, v in all_y_prob.items()}
        num_classes_dict = {
            'class':     self.args.output_num[0],
            'subclass':  self.args.output_num[1],
            'supertype': self.args.output_num[2],
        }
        self.log_auc_roc_metrics(final_y_true, final_y_prob, num_classes_dict, prefix='Test')


        res = {k: torch.cat(v, dim=0).numpy() for k, v in res.items()}
        out_path = os.path.join(self.args.save_dir, 'test_features.npz')
        np.savez_compressed(out_path, **res)
        self.logger.info(f"Features saved to: {out_path}")


        return res