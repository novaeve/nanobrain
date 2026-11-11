#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_mouse_development_temporal_datasets, _load_pair_index_from_npz  # noqa: E402
from model import scFiTModel  # noqa: E402
from utils import build_mask_states  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-5 validation checks for the fixed-HVG temporal pipeline.")
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="data/processed/neural_crest_pns_glia_hvg2048",
    )
    parser.add_argument(
        "--hvg-dir",
        type=str,
        default="results/cache/hvg_cache/neural_crest_pns_glia_rawcounts_seuratv3",
    )
    parser.add_argument(
        "--pair-table-path",
        type=str,
        default="results/cache/pair_cache/neural_crest_pns_glia_hvg2048_adjacent_k5.npz",
    )
    parser.add_argument(
        "--uce-npz",
        type=str,
        default="data/external/jax_uce_flow_data.npz",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config_mouse_temporal_smoke.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./experiments_mouse_temporal_smoke/smoke_temporal_latent_knn/checkpoints/final.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation/stage5_fixed_hvg2048",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def _ordered_timepoints(meta: pd.DataFrame):
    counts = meta["author_day"].astype(str).value_counts()
    available = counts[counts >= 100].index.tolist()
    preferred = ["E1000", "E1400", "P0000"]
    chosen = [tp for tp in preferred if tp in available]
    if len(chosen) >= 3:
        return chosen[:3]
    ordered = []
    time_means = meta.groupby("author_day")["author_day_numeric"].mean().sort_values()
    for tp in time_means.index.tolist():
        if tp in available and tp not in chosen:
            ordered.append(tp)
    while len(chosen) < min(3, len(available)) and ordered:
        idx = int(round((len(chosen) / max(2, 2)) * (len(ordered) - 1))) if len(ordered) > 1 else 0
        chosen.append(ordered.pop(idx))
    return chosen


def _marker_ranks(hvg_dir: Path, markers):
    ranked = np.loadtxt(hvg_dir / "hvg_ranked_gene_names.txt", dtype=str, ndmin=1)
    rank_map = {gene: idx + 1 for idx, gene in enumerate(ranked.tolist())}
    return {
        gene: {
            "rank": rank_map.get(gene),
            "in_top_2048": bool(rank_map.get(gene) is not None and rank_map[gene] <= 2048),
        }
        for gene in markers
    }


def _plot_distributions(expr: np.ndarray, meta: pd.DataFrame, output_path: Path, rng: np.random.RandomState):
    timepoints = _ordered_timepoints(meta)
    fig, axes = plt.subplots(1, len(timepoints), figsize=(5 * len(timepoints), 4), sharey=True)
    if len(timepoints) == 1:
        axes = [axes]

    summary = {}
    for ax, tp in zip(axes, timepoints):
        idx = np.flatnonzero(meta["author_day"].astype(str).to_numpy() == tp)
        sample_size = min(100, idx.shape[0])
        chosen = rng.choice(idx, size=sample_size, replace=False)
        values = np.asarray(expr[chosen], dtype=np.float32).reshape(-1)
        summary[tp] = {
            "n_cells_sampled": int(sample_size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }
        ax.hist(values, bins=60, color="#4C78A8", alpha=0.85, density=True)
        ax.set_title(tp)
        ax.set_xlabel("log1p normalized expression")
        ax.set_ylabel("density")
    fig.suptitle("Expression Distribution Across Time Bins (100 cells each)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary, timepoints


def _build_aligned_uce(preprocessed_dir: Path, uce_npz_path: Path):
    obs_names = np.load(preprocessed_dir / "obs_names.npy", allow_pickle=True).astype(str)
    meta = pd.read_pickle(preprocessed_dir / "cell_metadata.pkl").copy()
    uce = np.load(uce_npz_path, allow_pickle=True)
    uce_obs = uce["obs_names"].astype(str)
    z = np.asarray(uce["z"], dtype=np.float32)

    pre_row_by_obs = {name: idx for idx, name in enumerate(obs_names.tolist())}
    aligned_pre_idx = []
    aligned_uce_idx = []
    for u_idx, name in enumerate(uce_obs.tolist()):
        p_idx = pre_row_by_obs.get(name)
        if p_idx is not None:
            aligned_pre_idx.append(p_idx)
            aligned_uce_idx.append(u_idx)
    aligned_pre_idx = np.asarray(aligned_pre_idx, dtype=np.int64)
    aligned_uce_idx = np.asarray(aligned_uce_idx, dtype=np.int64)

    aligned_meta = meta.iloc[aligned_pre_idx].copy()
    aligned_meta["pre_idx"] = aligned_pre_idx
    aligned_meta["uce_idx"] = aligned_uce_idx
    aligned_meta["obs_name"] = obs_names[aligned_pre_idx]
    return aligned_meta.reset_index(drop=True), z[aligned_uce_idx]


def _plot_pair_umap(preprocessed_dir: Path, pair_table_path: Path, uce_npz_path: Path, output_path: Path, rng: np.random.RandomState):
    aligned_meta, aligned_z = _build_aligned_uce(preprocessed_dir, uce_npz_path)
    pair_table = _load_pair_index_from_npz(str(pair_table_path))

    transition_labels = np.char.add(
        np.char.add(pair_table["src_time_raw"].astype(str), "->"),
        pair_table["tgt_time_raw"].astype(str),
    )
    unique_transitions = np.unique(transition_labels)
    selected_pair_indices = []
    for transition in unique_transitions[:6]:
        idx = np.flatnonzero(transition_labels == transition)
        if idx.size > 0:
            selected_pair_indices.append(int(rng.choice(idx)))
    if not selected_pair_indices:
        raise ValueError("No pairs available for UMAP validation plot.")

    highlight_pre_idx = set()
    for pair_idx in selected_pair_indices:
        highlight_pre_idx.add(int(pair_table["src_index"][pair_idx]))
        highlight_pre_idx.add(int(pair_table["tgt_index"][pair_idx]))

    aligned_pre = aligned_meta["pre_idx"].to_numpy(dtype=np.int64)
    all_rows = np.arange(aligned_meta.shape[0], dtype=np.int64)
    background_size = min(5000, aligned_meta.shape[0])
    background_rows = rng.choice(all_rows, size=background_size, replace=False)
    keep_rows = set(background_rows.tolist())
    pre_to_row = {int(pre): row for row, pre in enumerate(aligned_pre.tolist())}
    for pre_idx in highlight_pre_idx:
        row = pre_to_row.get(int(pre_idx))
        if row is not None:
            keep_rows.add(int(row))
    keep_rows = np.array(sorted(keep_rows), dtype=np.int64)

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.25,
        metric="cosine",
        random_state=int(rng.randint(0, 1_000_000)),
    )
    embedding = reducer.fit_transform(aligned_z[keep_rows])
    row_to_emb = {int(row): emb for row, emb in zip(keep_rows.tolist(), embedding)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.reshape(-1)
    pair_summaries = []
    for ax, pair_idx in zip(axes, selected_pair_indices):
        src_pre = int(pair_table["src_index"][pair_idx])
        tgt_pre = int(pair_table["tgt_index"][pair_idx])
        src_row = pre_to_row.get(src_pre)
        tgt_row = pre_to_row.get(tgt_pre)
        ax.scatter(embedding[:, 0], embedding[:, 1], s=4, c="#D0D0D0", alpha=0.35, linewidths=0)
        if src_row is None or tgt_row is None or src_row not in row_to_emb or tgt_row not in row_to_emb:
            ax.set_title(f"pair {pair_idx}: missing in UMAP subset")
            ax.axis("off")
            continue
        src_xy = row_to_emb[src_row]
        tgt_xy = row_to_emb[tgt_row]
        ax.plot([src_xy[0], tgt_xy[0]], [src_xy[1], tgt_xy[1]], color="#555555", linewidth=1.0, alpha=0.9)
        ax.scatter(src_xy[0], src_xy[1], s=35, c="#1F77B4", label="src", zorder=3)
        ax.scatter(tgt_xy[0], tgt_xy[1], s=35, c="#D62728", label="tgt", zorder=3)
        src_meta = aligned_meta.iloc[src_row]
        tgt_meta = aligned_meta.iloc[tgt_row]
        ax.set_title(f"{src_meta['author_day']} -> {tgt_meta['author_day']}\n{src_meta['author_cell_type']}")
        ax.set_xticks([])
        ax.set_yticks([])
        pair_summaries.append(
            {
                "pair_idx": int(pair_idx),
                "src_pre_idx": src_pre,
                "tgt_pre_idx": tgt_pre,
                "src_time": str(src_meta["author_day"]),
                "tgt_time": str(tgt_meta["author_day"]),
                "cell_type": str(src_meta["author_cell_type"]),
                "umap_distance": float(np.linalg.norm(src_xy - tgt_xy)),
                "pair_weight": float(pair_table["pair_weight"][pair_idx]),
            }
        )
    for ax in axes[len(selected_pair_indices):]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4", markersize=7, label="src"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728", markersize=7, label="tgt"),
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle("Random Temporal kNN Pairs on UCE UMAP")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_aligned_cells": int(aligned_meta.shape[0]),
        "umap_subset_size": int(keep_rows.shape[0]),
        "sampled_pairs": pair_summaries,
    }


def _forward_smoke_test(preprocessed_dir: Path, pair_table_path: Path, config_path: Path, checkpoint_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_ds, _, _, meta = load_mouse_development_temporal_datasets(
        "ignored",
        input_mode="fixed_hvg",
        preprocessed_dir=str(preprocessed_dir),
        pair_table_path=str(pair_table_path),
        cell_type_col="author_cell_type",
        train_src_min_time=10.0,
        train_src_max_time=14.0,
        train_tgt_min_time=10.0,
        train_tgt_max_time=14.0,
        val_src_min_time=12.0,
        val_src_max_time=14.0,
        val_tgt_min_time=14.0,
        val_tgt_max_time=19.0,
    )

    batch_items = [train_ds[i] for i in range(4)]
    batch = [torch.stack([item[j] for item in batch_items], dim=0) for j in range(8)]
    gene_ids, src_expr, tgt_expr, valid_mask, t_src, t_tgt, cell_type_ids, pair_weight = batch
    del pair_weight

    padding_mask = ~valid_mask.bool()
    cls_mask = gene_ids.eq(meta["n_genes"] + 1)
    gene_valid_mask = valid_mask.bool() & (~cls_mask)
    mask_states = build_mask_states(padding_mask, gene_valid_mask, cls_mask=cls_mask)
    t_flow = torch.full((gene_ids.shape[0], 1), 0.5, dtype=torch.float32)
    x_t = (1.0 - t_flow) * src_expr + t_flow * tgt_expr

    model = scFiTModel(
        n_genes=int(meta["n_genes"]),
        top_k_genes=int(meta["fixed_panel_size"]),
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        d_ff=int(cfg["model"]["d_ff"]),
        nlayers=int(cfg["model"]["nlayers"]),
        dropout=float(cfg["model"]["dropout"]),
        value_max=float(cfg["model"].get("value_encoder", {}).get("max_value", 32.0)),
        share_adaln_projection=bool(cfg["model"].get("adaln", {}).get("share_projection", False)),
        aux_bin_ce=False,
        n_bins=0,
        n_cell_types=int(cfg["model"].get("n_cell_types", len(meta.get("cell_type_names", [])))),
        pseudotime_head=bool(cfg["model"].get("pseudotime_head", {}).get("enabled", False)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        pred_tgt, _, type_logits, _, _, _ = model.forward_with_aux(
            gene_ids,
            x_t,
            mask_states,
            padding_mask,
            t_flow,
            t_src=t_src.unsqueeze(1),
            t_tgt=t_tgt.unsqueeze(1),
            cell_type_ids=cell_type_ids,
            return_cls_attention=False,
        )
        rollout_pred = model.temporal_inference(
            gene_ids,
            src_expr,
            padding_mask=padding_mask,
            t_src=t_src.unsqueeze(1),
            t_tgt=t_tgt.unsqueeze(1),
            cell_type_ids=cell_type_ids,
            n_steps=4,
        )

    return {
        "batch_shape": {
            "gene_ids": list(gene_ids.shape),
            "src_expr": list(src_expr.shape),
            "tgt_expr": list(tgt_expr.shape),
            "valid_mask": list(valid_mask.shape),
        },
        "time_range": {
            "t_src_min": float(t_src.min().item()),
            "t_src_max": float(t_src.max().item()),
            "t_tgt_min": float(t_tgt.min().item()),
            "t_tgt_max": float(t_tgt.max().item()),
        },
        "gene_ids_consistent_across_batch": bool(torch.all(gene_ids.eq(gene_ids[0:1])).item()),
        "src_value_range": [float(src_expr.min().item()), float(src_expr.max().item())],
        "tgt_value_range": [float(tgt_expr.min().item()), float(tgt_expr.max().item())],
        "forward_pred_range": [float(pred_tgt.min().item()), float(pred_tgt.max().item())],
        "rollout_pred_range": [float(rollout_pred.min().item()), float(rollout_pred.max().item())],
        "forward_has_nan": bool(torch.isnan(pred_tgt).any().item()),
        "rollout_has_nan": bool(torch.isnan(rollout_pred).any().item()),
        "type_logits_has_nan": bool(False if type_logits is None else torch.isnan(type_logits).any().item()),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
    }


def main():
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    rng = np.random.RandomState(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = Path(args.preprocessed_dir)
    hvg_dir = Path(args.hvg_dir)
    pair_table_path = Path(args.pair_table_path)
    uce_npz_path = Path(args.uce_npz)

    expr = np.load(preprocessed_dir / "expression_log1p_clipped.npy", mmap_mode="r")
    meta = pd.read_pickle(preprocessed_dir / "cell_metadata.pkl")

    markers = ["Phox2b", "Sox10", "Ascl1"]
    marker_report = _marker_ranks(hvg_dir, markers)
    dist_summary, dist_timepoints = _plot_distributions(
        expr,
        meta,
        output_dir / "distribution_check.png",
        rng,
    )
    pair_umap_summary = _plot_pair_umap(
        preprocessed_dir,
        pair_table_path,
        uce_npz_path,
        output_dir / "pair_umap_check.png",
        rng,
    )
    smoke_summary = _forward_smoke_test(
        preprocessed_dir,
        pair_table_path,
        Path(args.config),
        Path(args.checkpoint),
    )

    summary = {
        "marker_report": marker_report,
        "distribution_check": {
            "timepoints": dist_timepoints,
            "summary": dist_summary,
            "figure": str(output_dir / "distribution_check.png"),
        },
        "pair_umap_check": {
            **pair_umap_summary,
            "figure": str(output_dir / "pair_umap_check.png"),
        },
        "forward_smoke_test": smoke_summary,
    }

    with (output_dir / "stage5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# Stage-5 Validation Summary",
        "",
        "## Gene Coverage",
        "",
    ]
    for gene, info in marker_report.items():
        lines.append(f"- `{gene}`: rank={info['rank']}, in_top_2048={info['in_top_2048']}")

    lines.extend(
        [
            "",
            "## Distribution Check",
            "",
            f"- figure: `{output_dir / 'distribution_check.png'}`",
        ]
    )
    for tp in dist_timepoints:
        stats = dist_summary[tp]
        lines.append(
            f"- `{tp}`: n={stats['n_cells_sampled']}, mean={stats['mean']:.4f}, "
            f"p50={stats['p50']:.4f}, p90={stats['p90']:.4f}, max={stats['max']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Pair UMAP Check",
            "",
            f"- figure: `{output_dir / 'pair_umap_check.png'}`",
            f"- aligned_cells={pair_umap_summary['n_aligned_cells']}, umap_subset_size={pair_umap_summary['umap_subset_size']}",
        ]
    )
    for item in pair_umap_summary["sampled_pairs"]:
        lines.append(
            f"- pair {item['pair_idx']}: {item['src_time']} -> {item['tgt_time']}, "
            f"{item['cell_type']}, weight={item['pair_weight']:.4f}, umap_distance={item['umap_distance']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interface Smoke Test",
            "",
            f"- gene_ids shape: `{smoke_summary['batch_shape']['gene_ids']}`",
            f"- src_expr shape: `{smoke_summary['batch_shape']['src_expr']}`",
            f"- gene_ids consistent across batch: `{smoke_summary['gene_ids_consistent_across_batch']}`",
            f"- src range: `{smoke_summary['src_value_range']}`",
            f"- tgt range: `{smoke_summary['tgt_value_range']}`",
            f"- forward pred range: `{smoke_summary['forward_pred_range']}`",
            f"- rollout pred range: `{smoke_summary['rollout_pred_range']}`",
            f"- forward has NaN: `{smoke_summary['forward_has_nan']}`",
            f"- rollout has NaN: `{smoke_summary['rollout_has_nan']}`",
            f"- type logits has NaN: `{smoke_summary['type_logits_has_nan']}`",
            f"- checkpoint: `{smoke_summary['checkpoint']}`",
        ]
    )

    with (output_dir / "stage5_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
