import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


MASK_PAD = 0
MASK_KNOWN = 1
MASK_MASKED = 2
MASK_CLS = 3


def seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_t_stratified(batch_size: int, device: torch.device) -> torch.Tensor:
    bins = [
        (0.0, 0.2, int(batch_size * 0.40)),
        (0.2, 0.4, int(batch_size * 0.20)),
        (0.4, 0.6, int(batch_size * 0.15)),
        (0.6, 0.8, int(batch_size * 0.15)),
        (
            0.8,
            1.0,
            batch_size - int(batch_size * 0.40) - int(batch_size * 0.20) - int(batch_size * 0.15) - int(batch_size * 0.15),
        ),
    ]
    parts = []
    for low, high, n_items in bins:
        if n_items <= 0:
            continue
        parts.append(torch.rand(n_items, device=device) * (high - low) + low)
    t = torch.cat(parts)
    t = t[torch.randperm(t.shape[0], device=device)]
    return t.unsqueeze(1)


def build_random_mask(valid_mask: torch.Tensor, mask_ratio: float, generator: torch.Generator = None) -> torch.Tensor:
    batch_size, seq_len = valid_mask.shape
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=valid_mask.device)
    for b in range(batch_size):
        valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue
        n_mask = max(1, int(valid_idx.numel() * mask_ratio))
        perm = torch.randperm(valid_idx.numel(), generator=generator, device=valid_mask.device)
        chosen = valid_idx[perm[:n_mask]]
        mask[b, chosen] = True
    return mask


def build_deterministic_mask(valid_mask: torch.Tensor, cell_indices: torch.Tensor, mask_ratio: float, base_seed: int) -> torch.Tensor:
    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    for b in range(batch_size):
        valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue
        n_mask = max(1, int(valid_idx.numel() * mask_ratio))
        g = torch.Generator(device=device)
        g.manual_seed(int(base_seed + int(cell_indices[b].item())))
        perm = torch.randperm(valid_idx.numel(), generator=g, device=device)
        chosen = valid_idx[perm[:n_mask]]
        mask[b, chosen] = True
    return mask


def build_mask_states(
    padding_mask: torch.Tensor,
    mask_indicator: torch.Tensor,
    cls_mask: torch.Tensor = None,
) -> torch.Tensor:
    states = torch.full_like(mask_indicator, MASK_PAD, dtype=torch.long)
    known_mask = (~padding_mask) & (~mask_indicator)
    masked_mask = (~padding_mask) & mask_indicator
    states[known_mask] = MASK_KNOWN
    states[masked_mask] = MASK_MASKED
    if cls_mask is not None:
        states[cls_mask] = MASK_CLS
    return states


def sample_gaussian_prior(valid_mask: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    x0 = torch.randn(valid_mask.shape, device=valid_mask.device, dtype=torch.float32) * float(std) + float(mean)
    x0 = x0 * valid_mask.to(torch.float32)
    return x0


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2 or target.numel() < 2:
        return pred.new_tensor(0.0)
    pred_c = pred - pred.mean()
    tgt_c = target - target.mean()
    return 1.0 - (pred_c * tgt_c).sum() / (pred_c.norm() * tgt_c.norm() + 1e-8)


def cell_pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
    return_stats: bool = False,
):
    losses = []
    n_used = 0
    for b in range(pred.shape[0]):
        idx = supervision_mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        pred_b = pred[b, idx]
        tgt_b = target[b, idx]
        pred_c = pred_b - pred_b.mean()
        tgt_c = tgt_b - tgt_b.mean()
        corr = (pred_c * tgt_c).sum() / (pred_c.norm() * tgt_c.norm() + 1e-8)
        losses.append(1.0 - corr)
        n_used += 1

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = pred.new_tensor(0.0)

    if not return_stats:
        return loss

    n_total = int(pred.shape[0])
    stats = {
        "n_total": n_total,
        "n_used": int(n_used),
        "n_skipped": int(n_total - n_used),
        "used_ratio": float(n_used / max(n_total, 1)),
        "skipped_ratio": float((n_total - n_used) / max(n_total, 1)),
    }
    return loss, stats


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
    delta: float = 0.5,
    top_k: int = 20,
    top_weight: float = 3.0,
) -> torch.Tensor:
    base = F.huber_loss(pred, target, delta=delta, reduction="none")
    weights = supervision_mask.to(pred.dtype)
    for b in range(pred.shape[0]):
        idx = supervision_mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        k = min(top_k, idx.numel())
        top_local = target[b, idx].topk(k).indices
        chosen = idx[top_local]
        weights[b, chosen] = top_weight
    denom = supervision_mask.sum().clamp_min(1).to(pred.dtype)
    return (base * weights).sum() / denom


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    mask = supervision_mask.bool()
    if not mask.any():
        return logits.new_tensor(0.0)
    flat_logits = logits[mask]
    flat_targets = target_bins[mask]
    return F.cross_entropy(flat_logits, flat_targets)


def bin_expression_values(values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    if bin_edges.numel() == 0:
        return torch.zeros_like(values, dtype=torch.long)
    return torch.bucketize(values, bin_edges)


def _safe_pearson_np(x: np.ndarray, y: np.ndarray):
    if x.size < 2 or y.size < 2:
        return None
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return None
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return None
    return float(r)


def masked_metrics(pred: torch.Tensor, target: torch.Tensor, supervision_mask: torch.Tensor) -> dict:
    mask = supervision_mask.bool()
    if not mask.any():
        return {
            "val_mse": 0.0,
            "val_rmse": 0.0,
            "val_mae": 0.0,
            "val_pearson_all": None,
            "val_pearson_cell": None,
            "val_spearman_cell": None,
            "val_pearson_top20_cell": None,
        }

    diff = pred[mask].float() - target[mask].float()
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())

    pred_flat = pred[mask].float()
    tgt_flat = target[mask].float()
    pred_c = pred_flat - pred_flat.mean()
    tgt_c = tgt_flat - tgt_flat.mean()
    denom = float(pred_c.norm().item() * tgt_c.norm().item())
    pearson_all = None if denom <= 0 else float((pred_c * tgt_c).sum().item() / (denom + 1e-8))

    cell_corrs = []
    cell_spearmans = []
    cell_corrs_top20 = []
    for b in range(pred.shape[0]):
        idx = supervision_mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        pred_b = pred[b, idx].float()
        tgt_b = target[b, idx].float()
        pred_cb = pred_b - pred_b.mean()
        tgt_cb = tgt_b - tgt_b.mean()
        denom_b = float(pred_cb.norm().item() * tgt_cb.norm().item())
        if denom_b <= 0:
            continue
        cell_corr = float((pred_cb * tgt_cb).sum().item() / (denom_b + 1e-8))
        cell_corrs.append(cell_corr)

        pred_np = pred_b.detach().cpu().numpy().astype(np.float64)
        tgt_np = tgt_b.detach().cpu().numpy().astype(np.float64)
        if spearmanr is not None:
            sp_corr = spearmanr(pred_np, tgt_np).correlation
            if sp_corr is not None and not np.isnan(sp_corr):
                cell_spearmans.append(float(sp_corr))

        k = min(20, tgt_np.shape[0])
        if k >= 2:
            idx_top = np.argsort(tgt_np)[-k:]
            top_corr = _safe_pearson_np(pred_np[idx_top], tgt_np[idx_top])
            if top_corr is not None:
                cell_corrs_top20.append(top_corr)

    return {
        "val_mse": mse,
        "val_rmse": float(math.sqrt(mse)),
        "val_mae": mae,
        "val_pearson_all": pearson_all,
        "val_pearson_cell": float(np.mean(cell_corrs)) if cell_corrs else None,
        "val_spearman_cell": float(np.mean(cell_spearmans)) if cell_spearmans else None,
        "val_pearson_top20_cell": float(np.mean(cell_corrs_top20)) if cell_corrs_top20 else None,
    }


def cosine_direction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() < 2 or target.numel() < 2:
        return pred.new_tensor(0.0)
    pred_flat = pred.reshape(pred.shape[0], -1) if pred.ndim > 1 else pred.unsqueeze(0)
    target_flat = target.reshape(target.shape[0], -1) if target.ndim > 1 else target.unsqueeze(0)
    cos = F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-8)
    return (1.0 - cos).mean()


def type_classification_loss(type_logits: torch.Tensor, target_type_ids: torch.Tensor) -> torch.Tensor:
    if type_logits is None:
        return target_type_ids.new_tensor(0.0, dtype=torch.float32)
    return F.cross_entropy(type_logits, target_type_ids.long())


def type_classification_accuracy(type_logits: torch.Tensor, target_type_ids: torch.Tensor) -> float:
    if type_logits is None or type_logits.numel() == 0:
        return 0.0
    pred = type_logits.argmax(dim=-1)
    return float((pred == target_type_ids.long()).float().mean().item())


def pseudotime_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor = None,
    beta: float = 0.1,
) -> torch.Tensor:
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    if valid_mask is None:
        valid_mask = torch.isfinite(target)
    else:
        valid_mask = valid_mask.view(-1).bool() & torch.isfinite(target)
    if not bool(valid_mask.any()):
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[valid_mask], target[valid_mask], beta=float(beta), reduction="mean")


def pseudotime_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor = None,
    margin: float = 0.0,
) -> torch.Tensor:
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    if valid_mask is None:
        valid_mask = torch.isfinite(target)
    else:
        valid_mask = valid_mask.view(-1).bool() & torch.isfinite(target)
    if int(valid_mask.sum().item()) < 2:
        return pred.new_tensor(0.0)

    pred = pred[valid_mask]
    target = target[valid_mask]
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    pair_sign = torch.sign(target_diff)
    pair_mask = pair_sign != 0
    if not bool(pair_mask.any()):
        return pred.new_tensor(0.0)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    signed_margin = pair_sign * pred_diff
    losses = F.relu(float(margin) - signed_margin)
    return losses[pair_mask].mean()


def mmd_rbf_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor = None,
    min_group_size: int = 8,
    gamma: float = None,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    def _mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2 or y.shape[0] < 2:
            return x.new_tensor(0.0)
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        if gamma is None:
            with torch.no_grad():
                xy = torch.cat([x_flat, y_flat], dim=0)
                if xy.shape[0] > 1024:
                    perm = torch.randperm(xy.shape[0], device=xy.device)[:1024]
                    xy = xy[perm]
                d2 = torch.cdist(xy, xy, p=2.0).pow(2)
                positive = d2[d2 > 0]
                gamma_val = 1.0 / positive.median().clamp_min(1e-6) if positive.numel() > 0 else x.new_tensor(1.0)
            gamma_local = gamma_val
        else:
            gamma_local = x.new_tensor(float(gamma))

        k_xx = torch.exp(-gamma_local * torch.cdist(x_flat, x_flat, p=2.0).pow(2))
        k_yy = torch.exp(-gamma_local * torch.cdist(y_flat, y_flat, p=2.0).pow(2))
        k_xy = torch.exp(-gamma_local * torch.cdist(x_flat, y_flat, p=2.0).pow(2))
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    if group_ids is None:
        return _mmd(pred, target)

    losses = []
    for gid in torch.unique(group_ids):
        group_mask = group_ids == gid
        if int(group_mask.sum().item()) < int(min_group_size):
            continue
        losses.append(_mmd(pred[group_mask], target[group_mask]))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def pairwise_mean_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().reshape(x.shape[0], -1)
    y = y.float().reshape(y.shape[0], -1)
    feat_dim = max(x.shape[1], 1)
    return torch.cdist(x, y, p=2.0).pow(2) / float(feat_dim)


def sinkhorn_uniform_transport(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    if cost.ndim != 2:
        raise ValueError("sinkhorn cost must be a 2D matrix")
    n, m = cost.shape
    if n == 0 or m == 0:
        return cost.new_zeros((n, m))

    eps = max(float(epsilon), 1e-6)
    log_a = -math.log(float(n))
    log_b = -math.log(float(m))
    log_k = -cost / eps

    u = cost.new_zeros((n,))
    v = cost.new_zeros((m,))
    for _ in range(int(max_iter)):
        u_prev = u
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(1), dim=0)
        if torch.max(torch.abs(u - u_prev)).item() < float(tol):
            break

    log_pi = log_k + u.unsqueeze(1) + v.unsqueeze(0)
    pi = torch.exp(log_pi)
    pi = pi / pi.sum().clamp_min(1e-12)
    return pi


def sinkhorn_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return x.new_tensor(0.0)
    cost = pairwise_mean_sq_dist(x, y)
    plan = sinkhorn_uniform_transport(cost, epsilon=epsilon, max_iter=max_iter, tol=tol)
    return (plan * cost).sum()


def sample_transport_pairs(
    plan: torch.Tensor,
    n_samples: int,
    generator: torch.Generator = None,
):
    flat = plan.reshape(-1)
    if flat.numel() == 0 or n_samples <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=plan.device)
        return empty, empty
    probs = flat / flat.sum().clamp_min(1e-12)
    choice = torch.multinomial(probs, int(n_samples), replacement=True, generator=generator)
    src_idx = torch.div(choice, plan.shape[1], rounding_mode="floor")
    tgt_idx = choice % plan.shape[1]
    return src_idx, tgt_idx
