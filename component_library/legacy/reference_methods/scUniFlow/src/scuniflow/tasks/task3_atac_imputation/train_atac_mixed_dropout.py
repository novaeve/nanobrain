import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "model.py").exists() and (p / "data.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_atac  # noqa: E402


def _read_config_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=str(ROOT / "config_atac.yaml"))
    args, _ = parser.parse_known_args()
    with open(args.config) as f:
        return yaml.safe_load(f)


def _pearson_loss(pred, target):
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = pred_c.norm() * target_c.norm()
    if float(denom.detach().item()) <= 0:
        return pred.new_tensor(0.0)
    return 1.0 - (pred_c * target_c).sum() / (denom + 1e-8)


class MixedDropoutATACPipelineTrainer(train_atac.scATACPipelineTrainer):
    """
    Experimental trainer for scATAC dropout imputation.

    It leaves train_atac.py untouched and replaces only the training loss:
    - recon mode: original mask-and-recover objective
    - dropout mode: simulate observed dropout by zeroing true nonzero positions,
      train binary head under imbalanced positive/negative candidates, and train
      intensity head on the simulated dropout positives.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = _read_config_from_argv()
        mixed_cfg = cfg.get("training", {}).get("mixed_dropout", {})
        self.mixed_dropout_prob = float(mixed_cfg.get("prob", 0.5))
        self.dropout_nonzero_ratio = float(mixed_cfg.get("nonzero_dropout_ratio", 0.4))
        self.dropout_neg_sample_ratio = float(mixed_cfg.get("neg_sample_ratio", 10.0))
        self.dropout_intensity_t = float(mixed_cfg.get("intensity_t", 0.0))
        self.dropout_pos_weight_min = float(mixed_cfg.get("pos_weight_min", 1.0))
        self.dropout_pos_weight_max = float(mixed_cfg.get("pos_weight_max", 20.0))
        self.recon_pos_weight_min = float(mixed_cfg.get("recon_pos_weight_min", 0.5))
        self.recon_pos_weight_max = float(mixed_cfg.get("recon_pos_weight_max", 5.0))

    def _compute_train_losses(self, g_id, x_1, x_0, mask_ind, t):
        if torch.rand((), device=x_1.device).item() < float(self.mixed_dropout_prob):
            return self._compute_dropout_train_losses(g_id, x_1)
        return self._compute_recon_train_losses(g_id, x_1, x_0, mask_ind, t)

    def _compute_recon_train_losses(self, g_id, x_1, x_0, mask_ind, t):
        raw = self.raw_model
        preds, binary_logits, targets, x1_hats, x1_targets, _ = raw.forward_train_dual(
            g_id, x_1, mask_ind, t, x_0, temperature=0.0
        )
        flat_preds = torch.cat(preds, dim=-1)
        flat_binary_logits = torch.cat(binary_logits, dim=-1)
        flat_targets = torch.cat(targets, dim=-1)

        nz_preds = []
        nz_targets = []
        for pred, target in zip(x1_hats, x1_targets):
            nz = target > 0
            if int(nz.sum().item()) > 0:
                nz_preds.append(pred[nz])
                nz_targets.append(target[nz])

        if nz_targets:
            flat_nz_preds = torch.cat(nz_preds, dim=-1)
            flat_nz_targets = torch.cat(nz_targets, dim=-1)
        else:
            flat_nz_preds = flat_preds.new_empty(0)
            flat_nz_targets = flat_targets.new_empty(0)

        loss_x1 = F.huber_loss(flat_preds, flat_targets, delta=0.5)
        loss_weighted = (
            F.huber_loss(flat_nz_preds, flat_nz_targets, delta=0.5)
            if flat_nz_targets.numel() > 0
            else flat_preds.new_tensor(0.0)
        )
        loss_pearson = (
            _pearson_loss(flat_nz_preds, flat_nz_targets)
            if flat_nz_targets.numel() > 1
            else flat_preds.new_tensor(0.0)
        )

        cell_losses = []
        for pred, target in zip(nz_preds, nz_targets):
            if pred.numel() >= 2:
                cell_losses.append(_pearson_loss(pred, target))
        loss_cell_pearson = torch.stack(cell_losses).mean() if cell_losses else flat_preds.new_tensor(0.0)
        loss_pearson_nonzero = loss_pearson

        binary_target = (flat_targets > 0).to(flat_binary_logits.dtype)
        n_pos = binary_target.sum()
        n_neg = binary_target.numel() - n_pos
        pos_weight = (n_neg / (n_pos + 1e-6)).clamp(
            min=float(self.recon_pos_weight_min),
            max=float(self.recon_pos_weight_max),
        ).detach()
        loss_binary = F.binary_cross_entropy_with_logits(
            flat_binary_logits,
            binary_target,
            pos_weight=pos_weight,
        )

        loss_binary_rank = flat_preds.new_tensor(0.0)
        loss = (
            self.loss_weighted_x1 * loss_weighted
            + self.loss_pearson_weight * loss_pearson
            + self.loss_cell_pearson_weight * loss_cell_pearson
            + float(getattr(self, "loss_pearson_nonzero_weight", 0.0)) * loss_pearson_nonzero
            + float(getattr(self, "loss_binary_weight", 0.0)) * loss_binary
            + float(getattr(self, "loss_binary_rank_weight", 0.0)) * loss_binary_rank
        )
        return {
            "preds": preds,
            "targets": targets,
            "loss": loss,
            "x1_loss": loss_x1,
            "weighted_x1_loss": loss_weighted,
            "pearson_loss": loss_pearson,
            "cell_pearson_loss": loss_cell_pearson,
            "pearson_nonzero_loss": loss_pearson_nonzero,
            "binary_loss": loss_binary,
            "binary_rank_loss": loss_binary_rank,
        }

    def _sample_dropout_candidates(self, x_1):
        dropout_mask = torch.zeros_like(x_1, dtype=torch.bool)
        zero_sample_mask = torch.zeros_like(x_1, dtype=torch.bool)
        for b in range(x_1.shape[0]):
            nz = (x_1[b] > 0).nonzero(as_tuple=True)[0]
            z = (x_1[b] <= 0).nonzero(as_tuple=True)[0]
            if nz.numel() > 0:
                n_pos = max(1, int(round(nz.numel() * float(self.dropout_nonzero_ratio))))
                n_pos = min(n_pos, int(nz.numel()))
                pos = nz[torch.randperm(nz.numel(), device=x_1.device)[:n_pos]]
                dropout_mask[b, pos] = True
            else:
                n_pos = 0

            if z.numel() > 0 and n_pos > 0:
                n_neg = int(round(n_pos * float(self.dropout_neg_sample_ratio)))
                n_neg = max(1, min(n_neg, int(z.numel())))
                neg = z[torch.randperm(z.numel(), device=x_1.device)[:n_neg]]
                zero_sample_mask[b, neg] = True

        candidate_mask = dropout_mask | zero_sample_mask
        return dropout_mask, zero_sample_mask, candidate_mask

    def _compute_dropout_train_losses(self, g_id, x_1):
        raw = self.raw_model
        dropout_mask, _, candidate_mask = self._sample_dropout_candidates(x_1)
        observed = x_1.clone()
        observed[dropout_mask] = 0.0

        dropout_mask_f = dropout_mask.to(x_1.dtype)
        candidate_mask_f = candidate_mask.to(x_1.dtype)
        t_int = torch.full(
            (x_1.shape[0], 1),
            float(self.dropout_intensity_t),
            device=x_1.device,
            dtype=x_1.dtype,
        )
        t_binary = torch.zeros_like(t_int)

        intensity_context, _ = raw.compute_context(
            g_id,
            observed,
            dropout_mask_f,
            temperature=0.0,
        )
        binary_context, _ = raw.compute_context(
            g_id,
            observed,
            candidate_mask_f,
            temperature=0.0,
        )

        preds = []
        targets = []
        x1_hats = []
        x1_targets = []
        binary_logits = []
        binary_targets = []

        for b in range(x_1.shape[0]):
            pos_idx = dropout_mask[b].nonzero(as_tuple=True)[0]
            cand_idx = candidate_mask[b].nonzero(as_tuple=True)[0]

            if pos_idx.numel() > 0:
                ctx_pos = intensity_context[b, pos_idx, :].unsqueeze(0)
                xt_pos = observed[b, pos_idx].unsqueeze(0)
                pred = raw.vector_field(xt_pos, ctx_pos, t_int[b].unsqueeze(0)).squeeze(0)
                target = x_1[b, pos_idx]
                preds.append(pred)
                targets.append(target)
                x1_hats.append(pred)
                x1_targets.append(target)

            if cand_idx.numel() > 0:
                ctx_cand = binary_context[b, cand_idx, :].unsqueeze(0)
                xt_cand = observed[b, cand_idx].unsqueeze(0)
                logit = raw.vector_field.binary_forward(
                    xt_cand,
                    ctx_cand,
                    t_binary[b].unsqueeze(0),
                ).squeeze(0)
                binary_logits.append(logit)
                binary_targets.append((x_1[b, cand_idx] > 0).to(logit.dtype))

        if preds:
            flat_preds = torch.cat(preds, dim=-1)
            flat_targets = torch.cat(targets, dim=-1)
            loss_x1 = F.huber_loss(flat_preds, flat_targets, delta=0.5)
            loss_weighted = loss_x1
            loss_pearson = _pearson_loss(flat_preds, flat_targets)
            cell_losses = [
                _pearson_loss(pred, target)
                for pred, target in zip(preds, targets)
                if pred.numel() >= 2
            ]
            loss_cell_pearson = (
                torch.stack(cell_losses).mean()
                if cell_losses
                else flat_preds.new_tensor(0.0)
            )
        else:
            flat_preds = x_1.new_empty(0)
            flat_targets = x_1.new_empty(0)
            loss_x1 = x_1.new_tensor(0.0)
            loss_weighted = x_1.new_tensor(0.0)
            loss_pearson = x_1.new_tensor(0.0)
            loss_cell_pearson = x_1.new_tensor(0.0)

        loss_pearson_nonzero = loss_pearson

        if binary_logits:
            flat_binary_logits = torch.cat(binary_logits, dim=-1)
            flat_binary_targets = torch.cat(binary_targets, dim=-1)
            n_pos = flat_binary_targets.sum()
            n_neg = flat_binary_targets.numel() - n_pos
            pos_weight = (n_neg / (n_pos + 1e-6)).clamp(
                min=float(self.dropout_pos_weight_min),
                max=float(self.dropout_pos_weight_max),
            ).detach()
            loss_binary = F.binary_cross_entropy_with_logits(
                flat_binary_logits,
                flat_binary_targets,
                pos_weight=pos_weight,
            )
        else:
            loss_binary = x_1.new_tensor(0.0)

        loss_binary_rank = x_1.new_tensor(0.0)
        loss = (
            self.loss_weighted_x1 * loss_weighted
            + self.loss_pearson_weight * loss_pearson
            + self.loss_cell_pearson_weight * loss_cell_pearson
            + float(getattr(self, "loss_pearson_nonzero_weight", 0.0)) * loss_pearson_nonzero
            + float(getattr(self, "loss_binary_weight", 0.0)) * loss_binary
            + float(getattr(self, "loss_binary_rank_weight", 0.0)) * loss_binary_rank
        )

        return {
            "preds": preds if preds else [flat_preds],
            "targets": targets if targets else [flat_targets],
            "loss": loss,
            "x1_loss": loss_x1,
            "weighted_x1_loss": loss_weighted,
            "pearson_loss": loss_pearson,
            "cell_pearson_loss": loss_cell_pearson,
            "pearson_nonzero_loss": loss_pearson_nonzero,
            "binary_loss": loss_binary,
            "binary_rank_loss": loss_binary_rank,
        }


train_atac.scATACPipelineTrainer = MixedDropoutATACPipelineTrainer


if __name__ == "__main__":
    train_atac.main()
