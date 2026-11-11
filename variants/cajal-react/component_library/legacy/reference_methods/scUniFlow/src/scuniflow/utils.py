import math
import torch
import torch.nn.functional as F

def sinkhorn(scores, n_iter=20, temperature=1.0):
    """
    对数域 Sinkhorn 迭代，数值稳定，用于计算双随机分配矩阵 P。
    scores: (B, L, G)
    返回 P: (B, L, G)，行和=1，列和=1。
    """
    log_alpha = scores / temperature
    for _ in range(n_iter):
        # 归一化列与行
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha.exp()

def sparse_sinkhorn_routing(Q, K, V, k=200, n_iter=20, temperature=1.0):
    """
    基于 Top-K 的稀疏 Sinkhorn 路由，避免全序列级别的庞大 L×G 矩阵。
    Q: (L, D)
    K: (B, G, D)
    V: (B, G, D)
    k: 每个位置仅考虑亲和度 Top-K 的基因
    返回 M: (B, L, D) 路由后的序列特征
    """
    L, D = Q.shape
    B, G, _ = K.shape
    # 分块避免 OOM
    CHUNK = 1000
    outputs = []
    
    for start in range(0, L, CHUNK):
        Q_chunk = Q[start:start+CHUNK]  # (chunk, D)
        scores = torch.einsum("cd,bgd->bcg", Q_chunk, K)  # (B, chunk, G)
        scores = scores / math.sqrt(D)
        
        # 仅保留 top-k 最相关的基因
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # (B, chunk, k)
        
        # 在局部 Top-K 特征内进行 Softmax/Sinkhorn 的归一化（作为近似）
        P_chunk = F.softmax(topk_scores / temperature, dim=-1)  # (B, chunk, k)
        
        # 提取 V 对应的局部 Top-K
        V_topk = V.gather(1, topk_idx.view(B, -1, 1).expand(-1, -1, D)) # (B, chunk*k, D)
        V_topk = V_topk.view(B, -1, k, D)  # (B, chunk, k, D)
        
        # 加权路由
        out = (P_chunk.unsqueeze(-1) * V_topk).sum(dim=2)  # (B, chunk, D)
        outputs.append(out)
        
    return torch.cat(outputs, dim=1)

def morans_I_loss(expr_permuted):
    """
    近似 Moran's I 空间自相关损失。
    只在非零位置对上计算局部自相关，避免大量零值淹没信号。
    对非零值做 z-score 标准化后计算 Pearson 自相关。
    expr_permuted: (B, L) 的已排列的基因表达值序列
    """
    left = expr_permuted[:, :-1]
    right = expr_permuted[:, 1:]

    # 相邻对中至少一个非零才参与计算
    pair_mask = (left.abs() > 1e-8) | (right.abs() > 1e-8)  # (B, L-1)

    n_pairs = pair_mask.sum(dim=1, keepdim=True).clamp(min=1)

    # 在非零位置上做标准化（只用非零值算均值和标准差）
    nz_mask = expr_permuted.abs() > 1e-8
    nz_count = nz_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    nz_sum = (expr_permuted * nz_mask).sum(dim=1, keepdim=True)
    mean = nz_sum / nz_count
    diff = (expr_permuted - mean) * nz_mask
    std = (diff.pow(2).sum(dim=1, keepdim=True) / nz_count).sqrt() + 1e-8
    z = diff / std

    z_left = z[:, :-1]
    z_right = z[:, 1:]

    autocorr = ((z_left * z_right) * pair_mask).sum(dim=1) / n_pairs.squeeze(1)
    return -autocorr.mean()

def hilbert_to_2d(pi, grid_size=174):
    """
    提取位置基因的一维序列索引对应的 2D (row, col) 坐标，供位移评价使用。
    """
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
        order = math.ceil(math.log2(grid_size))
        hc = HilbertCurve(order, 2)
        # 将张量挪到 CPU 解算后再带回
        pi_np = pi.detach().cpu().numpy().tolist()
        pos = np.array(hc.points_from_distances(pi_np))
        return torch.tensor(pos, device=pi.device, dtype=torch.float32)
    except:
        # Fallback 策略：如果系统缺少该库或出错暂时线性推导
        y = pi // grid_size
        x = pi % grid_size
        return torch.stack([y.to(torch.float32), x.to(torch.float32)], dim=-1)

def compute_delta_pi(Q_current, Q_previous, K_matrix):
    """
    每 1000 步监测一次排列变化速度（即单个基因所在格点坐标的绝对距离偏移）。
    Q_current: (L, D) 当前的可学习位置编码
    Q_previous: (L, D) 过去的记录
    K_matrix: (G, D) 实际上应当去除 batch 直接评估平均位移即可，我们假定这里传入单一静态 K: (G, D)
    """
    with torch.no_grad():
        # K_matrix 形状期望是 (G, D)，即非 batched context 下评估
        scores_cur  = torch.einsum("ld,gd->lg", Q_current, K_matrix)
        scores_prev = torch.einsum("ld,gd->lg", Q_previous, K_matrix)
        
        # 获取最匹配的硬位置 (G,) 其中索引标识位置
        pi_cur  = scores_cur.argmax(dim=0)
        pi_prev = scores_prev.argmax(dim=0)
        
        # 二维格点转化计算 L2 位移距离
        pos_cur  = hilbert_to_2d(pi_cur)
        pos_prev = hilbert_to_2d(pi_prev)
        
        delta = (pos_cur - pos_prev).norm(dim=1).mean()
        
    return delta.item()
