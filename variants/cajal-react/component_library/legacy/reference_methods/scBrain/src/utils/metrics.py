import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleWeightedCrossEntropyLoss(nn.Module):
    """
    定制版：支持样本级权重 (Sample-level weights) 的交叉熵损失函数。
    非常适合用于带有“置信度(Confidence)”或“噪声标签(Noisy Labels)”的单细胞数据分类。
    """
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): 'mean' (求加权平均), 'sum' (求加权和), 或 'none' (返回每个样本的 loss)
        """
        super(SampleWeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, sample_weights):
        """
        Args:
            logits: 模型的预测输出，形状为 [Batch_Size, Num_Classes]
            targets: 真实的类别索引，形状为 [Batch_Size]
            sample_weights: 每个细胞的权重/置信度，形状为 [Batch_Size]
        
        Returns:
            loss: 计算后的标量或向量
        """
        # 1. 计算基础的 Cross Entropy，强制设为 'none'，暴露出每个样本独立的 loss
        # unweighted_loss 形状: [Batch_Size]
        unweighted_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 2. 将每个样本的 loss 与其对应的置信度系数相乘
        # 注意：这里要求 sample_weights 也是 [Batch_Size] 的 1D Tensor
        weighted_loss = unweighted_loss * sample_weights
        
        # 3. 按照 reduction 策略返回
        if self.reduction == 'mean':
            # 也可以选择更严格的加权平均： weighted_loss.sum() / sample_weights.sum()
            # 但在流形学习中，直接 .mean() 通常表现更稳定，相当于把权重看作 learning rate 的标量
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss