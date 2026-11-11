import torch
import logging

logger = logging.getLogger(__name__)

def scgpt_binning_torch(x: torch.Tensor, n_bins: int = 51) -> torch.Tensor:
    """
    针对 Cell * Gene 矩阵的纯 PyTorch 向量化分箱 (scGPT-style)
    
    Args:
        x (torch.Tensor): 细胞 x 基因矩阵，形状为 (N, G)，必须是非负数。
        n_bins (int): 分箱总数。Bin 0 被硬性保留给表达量为 0 的基因。
        
    Returns:
        torch.Tensor: 分箱后的整数矩阵，形状为 (N, G)，值域为 [0, n_bins - 1]。
    """
    if n_bins <= 1:
        raise ValueError(f"n_bins 必须大于 1，当前为 {n_bins}")
    if x.min() < 0:
        raise ValueError(f"期望非负表达矩阵，但检测到最小值为 {x.min().item()}")

    N, G = x.shape
    binned_x = torch.zeros_like(x, dtype=torch.long)

    # 1. 隔离零值 (Zero Isolation)
    # 创建一个布尔掩码，记录哪些位置有真实的表达值
    non_zero_mask = x > 0

    # 检查警告：是否存在全是 0 的细胞（死细胞/空孔）
    row_max = x.max(dim=1).values
    if (row_max == 0).any():
        logger.warning("输入数据中包含全 0 表达的细胞，它们的分箱结果将全部为 0。")

    # 2. 构建 NaN 矩阵用于分位数计算
    # 将 0 强行替换为 NaN，这样 PyTorch 在算分位数时就会自动无视它们
    # 转为 float32 防止半精度/整数计算 nanquantile 报错
    x_nan = x.clone().to(torch.float32)
    x_nan[~non_zero_mask] = float('nan')

    # 3. 计算细胞级相对分位数边界 (Batched Quantiles)
    # q 的形状: [n_bins - 1]
    q = torch.linspace(0, 1, steps=n_bins - 1, dtype=torch.float32, device=x.device)

    # nanquantile 沿基因维度(dim=1)计算，结果形状为 [n_bins - 1, N]
    # 我们将其转置回 [N, n_bins - 1]，这就相当于为每个细胞生成了专属的尺子
    bins = torch.nanquantile(x_nan, q, dim=1).T
    
    # 全 0 细胞会算出 NaN 边界，我们将其补为 0 防止报错
    bins = torch.nan_to_num(bins, nan=0.0)

    # 4. 向量化分配档位 (Batched Searchsorted)
    # searchsorted 能够直接拿 2D 的特征和 2D 的专属尺子进行比对，瞬间返回分箱索引
    # right=True 确保边界值的分配逻辑与 np.digitize 保持一致
    digits = torch.searchsorted(bins, x.contiguous(), right=True)

    # 裁剪：确保所有的非零值，最少被分配到 Bin 1，最高到 n_bins - 1
    digits = torch.clamp(digits, min=1, max=n_bins - 1)

    # 5. 重组输出 (Reconstruct)
    # 只将原本非零基因的位置，替换为新算出来的 Bin 索引
    # 原本是 0 的位置，在 binned_x 初始化时就是 0，因此永远保持 Bin 0
    binned_x[non_zero_mask] = digits[non_zero_mask]

    return binned_x