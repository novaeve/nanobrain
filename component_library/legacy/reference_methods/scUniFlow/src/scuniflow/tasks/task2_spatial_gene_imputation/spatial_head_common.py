import json
import random
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "data.py").exists() and (parent / "model.py").exists():
            return parent
    raise FileNotFoundError(f"Could not locate scUniFlow repo root from: {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_scuniflow_model_class():
    gene_mamba_path = ROOT / 'gene_mamba.py'
    gene_mamba_spec = importlib.util.spec_from_file_location('gene_mamba', gene_mamba_path)
    if gene_mamba_spec is None or gene_mamba_spec.loader is None:
        raise ImportError(f'Failed to load scUniFlow gene_mamba module from {gene_mamba_path}')
    gene_mamba_module = importlib.util.module_from_spec(gene_mamba_spec)
    sys.modules['gene_mamba'] = gene_mamba_module
    gene_mamba_spec.loader.exec_module(gene_mamba_module)

    model_path = ROOT / 'model.py'
    spec = importlib.util.spec_from_file_location('scuniflow_task3_model', model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load scUniFlow model module from {model_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scRNAModel


scRNAModel = _load_scuniflow_model_class()

from common import transform_counts_array


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SpotSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    split_info: dict


def build_spot_train_val_split(n_spots: int, val_ratio: float, seed: int) -> SpotSplit:
    all_idx = np.arange(int(n_spots), dtype=np.int64)
    if n_spots <= 1 or val_ratio <= 0:
        return SpotSplit(
            train_idx=all_idx,
            val_idx=np.array([], dtype=np.int64),
            split_info={"split_mode": "all_train", "seed": int(seed), "val_ratio": float(val_ratio)},
        )

    rng = np.random.RandomState(int(seed))
    n_val = max(1, int(round(float(val_ratio) * n_spots)))
    n_val = min(n_val, n_spots - 1)
    val_idx = np.sort(rng.choice(n_spots, size=n_val, replace=False)).astype(np.int64)
    train_mask = np.ones(n_spots, dtype=bool)
    train_mask[val_idx] = False
    train_idx = all_idx[train_mask]
    return SpotSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        split_info={
            "split_mode": "random_spots",
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "n_train_spots": int(train_idx.shape[0]),
            "n_val_spots": int(val_idx.shape[0]),
            },
        )


def build_spot_spatial_block_split(
    spatial_xy: np.ndarray,
    val_ratio: float,
    corner: str = "auto",
) -> SpotSplit:
    xy = np.asarray(spatial_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected spatial coords of shape [N,2], got {xy.shape}")
    n_spots = int(xy.shape[0])
    all_idx = np.arange(n_spots, dtype=np.int64)
    if n_spots <= 1 or val_ratio <= 0:
        return SpotSplit(
            train_idx=all_idx,
            val_idx=np.array([], dtype=np.int64),
            split_info={"split_mode": "all_train", "val_ratio": float(val_ratio)},
        )

    target = float(val_ratio)
    valid_corners = ("lower_left", "lower_right", "upper_left", "upper_right")
    if corner not in {"auto", *valid_corners}:
        raise ValueError(f"Unsupported spatial block corner: {corner}")

    norm_xy = normalize_spatial_coords(xy)
    x = norm_xy[:, 0]
    y = norm_xy[:, 1]
    x_qs = np.linspace(0.15, 0.85, 29)
    y_qs = np.linspace(0.15, 0.85, 29)

    def _mask_for(corner_name: str, tx: float, ty: float) -> np.ndarray:
        if corner_name == "lower_left":
            return (x <= tx) & (y <= ty)
        if corner_name == "lower_right":
            return (x >= tx) & (y <= ty)
        if corner_name == "upper_left":
            return (x <= tx) & (y >= ty)
        if corner_name == "upper_right":
            return (x >= tx) & (y >= ty)
        raise AssertionError("unreachable")

    candidate_corners = valid_corners if corner == "auto" else (corner,)
    best = None
    best_payload = None
    for corner_name in candidate_corners:
        for qx in x_qs:
            tx = float(np.quantile(x, qx))
            for qy in y_qs:
                ty = float(np.quantile(y, qy))
                mask = _mask_for(corner_name, tx, ty)
                n_val = int(mask.sum())
                if n_val <= 0 or n_val >= n_spots:
                    continue
                frac = float(n_val / n_spots)
                val_xy = norm_xy[mask]
                span_x = float(val_xy[:, 0].max() - val_xy[:, 0].min()) if n_val > 1 else 0.0
                span_y = float(val_xy[:, 1].max() - val_xy[:, 1].min()) if n_val > 1 else 0.0
                bbox_area = span_x * span_y
                # Prefer fractions closest to target; then prefer a compact held-out region.
                score = (abs(frac - target), bbox_area, abs(qx - 0.5) + abs(qy - 0.5))
                if best is None or score < best:
                    best = score
                    best_payload = {
                        "corner": corner_name,
                        "threshold_x": tx,
                        "threshold_y": ty,
                        "quantile_x": float(qx),
                        "quantile_y": float(qy),
                        "mask": mask.copy(),
                        "frac": frac,
                        "bbox_area": bbox_area,
                    }

    if best_payload is None:
        raise RuntimeError("Failed to build a spatial block split.")

    val_idx = np.where(best_payload["mask"])[0].astype(np.int64)
    train_mask = np.ones(n_spots, dtype=bool)
    train_mask[val_idx] = False
    train_idx = all_idx[train_mask]
    return SpotSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        split_info={
            "split_mode": "spatial_block",
            "val_ratio": float(val_ratio),
            "n_train_spots": int(train_idx.shape[0]),
            "n_val_spots": int(val_idx.shape[0]),
            "block_corner": str(best_payload["corner"]),
            "block_quantile_x": float(best_payload["quantile_x"]),
            "block_quantile_y": float(best_payload["quantile_y"]),
            "block_threshold_x": float(best_payload["threshold_x"]),
            "block_threshold_y": float(best_payload["threshold_y"]),
            "block_bbox_area": float(best_payload["bbox_area"]),
        },
    )


def normalize_spatial_coords(spatial_xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(spatial_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected spatial coords of shape [N,2], got {xy.shape}")
    mn = xy.min(axis=0, keepdims=True)
    mx = xy.max(axis=0, keepdims=True)
    denom = np.maximum(mx - mn, 1e-6)
    return ((xy - mn) / denom).astype(np.float32, copy=False)


class SpatialSpotTargetDataset(Dataset):
    def __init__(
        self,
        count_matrix,
        gene_ids: np.ndarray,
        spatial_xy: np.ndarray,
        indices: np.ndarray,
        observed_idx: np.ndarray,
        target_idx: np.ndarray,
        expression_space: str = "logcpm",
    ):
        self.count_matrix = count_matrix
        self.sparse = sp.issparse(count_matrix)
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.spatial_xy = np.asarray(spatial_xy, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.expression_space = str(expression_space)
        self.observed_idx = np.asarray(observed_idx, dtype=np.int64)
        self.target_idx = np.asarray(target_idx, dtype=np.int64)

        n_genes = int(self.gene_ids.shape[0])
        self.missing_mask = np.ones((n_genes,), dtype=np.float32)
        self.missing_mask[self.observed_idx] = 0.0

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, item):
        spot_idx = int(self.indices[item])
        if self.sparse:
            raw = self.count_matrix.getrow(spot_idx).toarray().ravel().astype(np.float32)
        else:
            raw = np.asarray(self.count_matrix[spot_idx], dtype=np.float32).ravel()

        visible_raw = np.zeros_like(raw, dtype=np.float32)
        visible_raw[self.observed_idx] = raw[self.observed_idx]

        x_visible = transform_counts_array(visible_raw, expression_space=self.expression_space)
        y_target = transform_counts_array(raw[self.target_idx], expression_space=self.expression_space)
        log_total_visible = np.float32(np.log10(float(visible_raw.sum()) + 1.0))
        xy = self.spatial_xy[spot_idx]

        return (
            torch.from_numpy(self.gene_ids.copy()),
            torch.from_numpy(x_visible),
            torch.from_numpy(self.missing_mask.copy()),
            torch.from_numpy(self.target_idx.copy()),
            torch.from_numpy(y_target.astype(np.float32, copy=False)),
            torch.tensor(log_total_visible, dtype=torch.float32),
            torch.from_numpy(xy.astype(np.float32, copy=False)),
            torch.tensor(spot_idx, dtype=torch.long),
        )


class SpatialGeneTaskHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = int(d_model) + 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context_target: torch.Tensor,
        x_visible_target: torch.Tensor,
        log_total_visible: torch.Tensor,
        spatial_xy: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_target, _ = context_target.shape
        total_feat = log_total_visible.view(bsz, 1, 1).expand(-1, n_target, -1)
        xy_feat = spatial_xy.view(bsz, 1, 2).expand(-1, n_target, -1)
        x_feat = x_visible_target.unsqueeze(-1)
        feat = torch.cat([context_target, x_feat, total_feat, xy_feat], dim=-1)
        pred = self.net(feat).squeeze(-1)
        return torch.clamp(pred, min=0.0)


class ScUniFlowSpatialImputer(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int,
        grid_size: int,
        n_layers: int = 8,
        d_state_row: int = 64,
        d_state_col: int = 16,
        activation_checkpointing: bool = True,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = scRNAModel(
            n_genes=n_genes,
            d_model=d_model,
            grid_size=grid_size,
            n_layers=n_layers,
            d_state_row=d_state_row,
            d_state_col=d_state_col,
            activation_checkpointing=activation_checkpointing,
        )
        self.head = SpatialGeneTaskHead(
            d_model=d_model,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_mamba(self):
        for p in self.backbone.mamba_encoder.parameters():
            p.requires_grad = True

    def unfreeze_router(self):
        for p in self.backbone.router.parameters():
            p.requires_grad = True

    def load_backbone_checkpoint(self, checkpoint_path: str, device=None):
        ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "vector_field.binary_mlp.",
        )
        disallowed_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Incompatible backbone checkpoint. "
                f"Missing keys: {disallowed_missing}; unexpected keys: {list(unexpected)}"
            )
        return ckpt

    def forward(
        self,
        gene_ids: torch.Tensor,
        x_visible: torch.Tensor,
        mask_missing: torch.Tensor,
        target_idx: torch.Tensor,
        log_total_visible: torch.Tensor,
        spatial_xy: torch.Tensor,
    ) -> torch.Tensor:
        requires_backbone_grad = any(p.requires_grad for p in self.backbone.parameters())
        if requires_backbone_grad:
            context, _ = self.backbone.compute_context(gene_ids, x_visible, mask_missing)
        else:
            with torch.no_grad():
                context, _ = self.backbone.compute_context(gene_ids, x_visible, mask_missing)

        gather_idx = target_idx.unsqueeze(-1).expand(-1, -1, context.shape[-1])
        context_target = context.gather(1, gather_idx)
        x_visible_target = x_visible.gather(1, target_idx)
        return self.head(
            context_target=context_target,
            x_visible_target=x_visible_target,
            log_total_visible=log_total_visible,
            spatial_xy=spatial_xy,
        )
