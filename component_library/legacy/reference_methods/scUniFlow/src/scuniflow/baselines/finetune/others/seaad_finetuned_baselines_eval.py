import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

import sys
sys.path.insert(0, "data/SEA-AD/MJM")

from donor_eval_utils import (
    DEFAULT_DONOR_CSV,
    DEFAULT_DONOR_DIR,
    pick_eval_donor,
    sample_single_donor_cells,
    split_val_donors,
)


DEFAULT_EXPERIMENT_DIR = Path("data/SEA-AD/MJM/experiments")
DEFAULT_MERFISH_H5AD = Path("data/SEA-AD/MTG/mtg_spatial_data/SEAAD_MTG_MERFISH.2024-12-11.h5ad")
DEFAULT_RESULT_DIR = Path("results/pretrain_eval")
DEFAULT_SCGPT_GENE_IDS = Path("data/SEA-AD/MJM/src/models/scGPT/merfish_indices.npy")
DEFAULT_SCGPT_BRAIN_CKPT = Path("external/scGPT/checkpoint/brain")
DEFAULT_SCGPT_HUMAN_CKPT = Path("external/scGPT/checkpoint/human")
DEFAULT_UCE_MODEL_DIR = Path("external/uce/model")
DEFAULT_NICHEFORMER_CKPT = Path(
    "external/model_zoo/"
    "4014c514594e479d8fd84d2c65dd94fc/Research/nicheformer.ckpt"
)
DEFAULT_NICHEFORMER_VOCAB = Path("external/nicheformer/data/model_means/model.h5ad")
DEFAULT_NICHEFORMER_MERFISH_MEAN = Path("external/nicheformer/data/model_means/merfish_mean_script.npy")
DEFAULT_GENE_NAME_TO_ENS = Path(
    "external/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl"
)
DEFAULT_GENEFORMER_CODE_DIR = Path("external/Geneformer")
DEFAULT_GENEFORMER_WEIGHTS = Path(
    "external/model_zoo/"
    "4014c514594e479d8fd84d2c65dd94fc/Research/model.safetensors"
)

TASK_SPECS = (
    ("Class", "class"),
    ("Subclass", "subclass"),
    ("Supertype", "supertype"),
)

MODEL_DIR_NAMES = {
    "scgpt_brain": "scGPT_brain",
    "scgpt_human": "scGPT_human",
    "uce": "UCE",
    "nicheformer": "Nicheformer",
    "geneformer": "Geneformer",
    "scanvi": "scAnVI",
}


def resolve_eval_donor(args):
    if args.eval_donor:
        donor_path = Path(args.donor_dir) / f"{args.eval_donor}.h5ad"
        if not donor_path.exists():
            raise FileNotFoundError(f"Eval donor file not found: {donor_path}")
        adata = sc.read_h5ad(donor_path, backed="r")
        return args.eval_donor, int(adata.n_obs)

    val_donors = split_val_donors(Path(args.donor_csv), val_ratio=0.1, seed=args.split_seed)
    return pick_eval_donor(val_donors, Path(args.donor_dir), min_cells=args.sample_n)


def load_merfish_panel_and_label_maps(merfish_h5ad: Path):
    adata = sc.read_h5ad(merfish_h5ad, backed="r")
    gene_panel = [str(g) for g in adata.var_names if not str(g).startswith("Blank")]
    label_maps = {}
    label_spaces = {}
    for obs_key, task_name in TASK_SPECS:
        cats = sorted(adata.obs[obs_key].astype("category").cat.categories.astype(str).tolist())
        label_maps[obs_key] = {name: idx for idx, name in enumerate(cats)}
        label_spaces[task_name] = cats
    return gene_panel, label_maps, label_spaces


def sample_donor_cells(donor_path: Path, sample_n: int, sample_seed: int):
    adata, _, _ = sample_single_donor_cells(
        donor_path=donor_path,
        sample_n=sample_n,
        sample_seed=sample_seed,
    )
    return adata


def build_panel_matrix(adata, gene_panel):
    source_gene_names = (
        adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in adata.var.columns
        else adata.var.index.astype(str).to_numpy()
    )
    source_index = {}
    for idx, gene in enumerate(source_gene_names):
        if gene not in source_index:
            source_index[gene] = idx

    matched_genes = [g for g in gene_panel if g in source_index]
    missing_genes = [g for g in gene_panel if g not in source_index]
    src_cols = [source_index[g] for g in matched_genes]
    dst_cols = [gene_panel.index(g) for g in matched_genes]

    X_sub = adata.X[:, src_cols]
    if issparse(X_sub):
        X_sub = X_sub.toarray()
    X_sub = X_sub.astype(np.float32)

    X_panel = np.zeros((adata.n_obs, len(gene_panel)), dtype=np.float32)
    X_panel[:, dst_cols] = X_sub
    return X_panel, matched_genes, missing_genes


def encode_labels(obs, label_maps):
    encoded = {}
    valid_mask = np.ones(obs.shape[0], dtype=bool)
    dropped = {}
    for obs_key, task_name in TASK_SPECS:
        values = obs[obs_key].astype(str).to_numpy()
        task_ids = np.array([label_maps[obs_key].get(v, -1) for v in values], dtype=np.int64)
        task_valid = task_ids >= 0
        if not np.all(task_valid):
            unseen = sorted(set(values[~task_valid].tolist()))
            dropped[task_name] = unseen
        valid_mask &= task_valid
        encoded[task_name] = task_ids

    for task_name in encoded:
        encoded[task_name] = encoded[task_name][valid_mask]
    return encoded, valid_mask, dropped


def resolve_checkpoint_path(args):
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    model_dir_name = MODEL_DIR_NAMES[args.model_name]
    ckpt_path = (
        Path(args.experiment_dir)
        / model_dir_name
        / "SEA_AD_MTP_ST"
        / "checkpoints"
        / f"{args.tuning_mode}_s{args.seed}"
        / "best_model.pth"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def load_finetuned_model(args, gene_panel, checkpoint_path: Path):
    model_name = args.model_name

    if model_name == "scgpt_brain":
        from src.models.scGPT.scGPT_annotation import scGPTForAnnotation

        model = scGPTForAnnotation(
            checkpoint_dir=str(args.scgpt_brain_pretrained_dir),
            gene_ids=torch.from_numpy(np.load(args.scgpt_gene_ids_path)).long(),
            output_num=[3, 24, 137],
            n_bins=51,
            dropout=0.2,
            freeze_backbone=False,
        )
    elif model_name == "scgpt_human":
        from src.models.scGPT.scGPT_annotation import scGPTForAnnotation

        model = scGPTForAnnotation(
            checkpoint_dir=str(args.scgpt_human_pretrained_dir),
            gene_ids=torch.from_numpy(np.load(args.scgpt_gene_ids_path)).long(),
            output_num=[3, 24, 137],
            n_bins=51,
            dropout=0.2,
            freeze_backbone=False,
        )
    elif model_name == "uce":
        from src.models.uce.uce_annotation import UCEForAnnotation

        model = UCEForAnnotation(
            model_dir=str(args.uce_model_dir),
            gene_names=gene_panel,
            output_num=[3, 24, 137],
            dropout=0.2,
            freeze_backbone=False,
        )
    elif model_name == "nicheformer":
        from src.models.nicheformer.nicheformer_annotation import NicheformerForAnnotation

        model = NicheformerForAnnotation(
            checkpoint_path=str(args.nicheformer_backbone_path),
            vocab_path=str(args.nicheformer_vocab_path),
            merfish_mean_path=str(args.nicheformer_merfish_mean_path),
            gene_name_to_ens_path=str(args.gene_name_to_ens_path),
            gene_names=gene_panel,
            output_num=[3, 24, 137],
            dropout=0.2,
            freeze_backbone=False,
        )
    elif model_name == "geneformer":
        from src.models.geneformer.geneformer_annotation import GeneformerForAnnotation

        model = GeneformerForAnnotation(
            code_dir=str(args.geneformer_code_dir),
            weights_path=str(args.geneformer_weights_path),
            gene_names=gene_panel,
            output_num=[3, 24, 137],
            dropout=0.2,
            freeze_backbone=False,
        )
    elif model_name == "scanvi":
        raise NotImplementedError(
            "scAnVI eval is not implemented yet in this script because there is no local "
            "training/eval wrapper or checkpoint layout to load from data/SEA-AD/MJM."
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def safe_auc(y_true, prob, num_classes):
    present_classes = np.unique(y_true)
    if present_classes.size < 2:
        return {"macro": None, "micro": None, "n_present_classes": int(present_classes.size)}

    prob_present = prob[:, present_classes]
    row_sum = prob_present.sum(axis=1, keepdims=True)
    prob_present = np.divide(prob_present, np.clip(row_sum, 1e-12, None))
    y_bin = label_binarize(y_true, classes=present_classes)

    if y_bin.ndim == 1:
        y_bin = np.stack([1 - y_bin, y_bin], axis=1)

    try:
        macro = float(roc_auc_score(y_bin, prob_present, average="macro", multi_class="ovr"))
    except ValueError:
        macro = None

    try:
        micro = float(roc_auc_score(y_bin, prob_present, average="micro", multi_class="ovr"))
    except ValueError:
        micro = None

    return {"macro": macro, "micro": micro, "n_present_classes": int(present_classes.size)}


def summarize_task_metrics(task_name, y_true, logits):
    y_pred = logits.argmax(axis=1)
    y_prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    auc = safe_auc(y_true, y_prob, num_classes=logits.shape[1])

    return {
        f"{task_name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{task_name}_macro_precision": float(macro_p),
        f"{task_name}_macro_recall": float(macro_r),
        f"{task_name}_macro_f1": float(macro_f1),
        f"{task_name}_micro_precision": float(micro_p),
        f"{task_name}_micro_recall": float(micro_r),
        f"{task_name}_micro_f1": float(micro_f1),
        f"{task_name}_weighted_precision": float(weighted_p),
        f"{task_name}_weighted_recall": float(weighted_r),
        f"{task_name}_weighted_f1": float(weighted_f1),
        f"{task_name}_auc_macro_present": auc["macro"],
        f"{task_name}_auc_micro_present": auc["micro"],
        f"{task_name}_n_present_classes": auc["n_present_classes"],
        f"{task_name}_n_pred_classes": int(np.unique(y_pred).size),
    }


def evaluate_model(model, X, labels, device, batch_size=64, show_progress=True):
    model.to(device)
    model.eval()

    total_samples = X.shape[0]
    total_loss = {"class": 0.0, "subclass": 0.0, "supertype": 0.0}
    logits_all = {"class": [], "subclass": [], "supertype": []}
    cell_embs = []

    starts = range(0, total_samples, batch_size)
    if show_progress:
        starts = tqdm(starts, total=(total_samples + batch_size - 1) // batch_size, desc="baseline eval", unit="batch")

    with torch.inference_mode():
        for start in starts:
            end = min(start + batch_size, total_samples)
            xb = torch.from_numpy(X[start:end]).to(device)
            y_class = torch.from_numpy(labels["class"][start:end]).to(device)
            y_subclass = torch.from_numpy(labels["subclass"][start:end]).to(device)
            y_supertype = torch.from_numpy(labels["supertype"][start:end]).to(device)

            logits, cell_emb = model(xb)
            logit_class, logit_subclass, logit_supertype = logits

            total_loss["class"] += float(torch.nn.functional.cross_entropy(logit_class, y_class, reduction="sum").item())
            total_loss["subclass"] += float(torch.nn.functional.cross_entropy(logit_subclass, y_subclass, reduction="sum").item())
            total_loss["supertype"] += float(torch.nn.functional.cross_entropy(logit_supertype, y_supertype, reduction="sum").item())

            logits_all["class"].append(logit_class.cpu().numpy())
            logits_all["subclass"].append(logit_subclass.cpu().numpy())
            logits_all["supertype"].append(logit_supertype.cpu().numpy())
            cell_embs.append(cell_emb.cpu().numpy())

    for task_name in logits_all:
        logits_all[task_name] = np.concatenate(logits_all[task_name], axis=0)
    cell_embs = np.concatenate(cell_embs, axis=0)

    mean_losses = {f"loss_{k}": v / max(total_samples, 1) for k, v in total_loss.items()}
    mean_losses["loss_total"] = sum(total_loss.values()) / max(total_samples, 1)

    metrics = {}
    for task_name in ("class", "subclass", "supertype"):
        metrics.update(summarize_task_metrics(task_name, labels[task_name], logits_all[task_name]))

    pred_class = logits_all["class"].argmax(axis=1)
    pred_subclass = logits_all["subclass"].argmax(axis=1)
    pred_supertype = logits_all["supertype"].argmax(axis=1)
    metrics["joint_exact_match"] = float(
        np.mean(
            (pred_class == labels["class"])
            & (pred_subclass == labels["subclass"])
            & (pred_supertype == labels["supertype"])
        )
    )

    return mean_losses, metrics, logits_all, cell_embs


def maybe_save_npz(save_path, cell_emb, labels, logits_all):
    if not save_path:
        return

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        cell_emb=cell_emb.astype(np.float32),
        y_class_true=labels["class"].astype(np.int64),
        y_subclass_true=labels["subclass"].astype(np.int64),
        y_supertype_true=labels["supertype"].astype(np.int64),
        y_class_pred=logits_all["class"].argmax(axis=1).astype(np.int64),
        y_subclass_pred=logits_all["subclass"].argmax(axis=1).astype(np.int64),
        y_supertype_pred=logits_all["supertype"].argmax(axis=1).astype(np.int64),
        logits_class=logits_all["class"].astype(np.float32),
        logits_subclass=logits_all["subclass"].astype(np.float32),
        logits_supertype=logits_all["supertype"].astype(np.float32),
    )


def build_result_path(args, donor_id: str) -> Path:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"seaad_baseline_eval_{args.model_name}_{args.tuning_mode}_"
        f"seed{args.seed}_{donor_id}_n{args.sample_n}.json"
    )
    filename = filename.replace("/", "_")
    return result_dir / filename


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEA-AD fine-tuned baseline checkpoints on an unseen donor.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["scgpt_brain", "scgpt_human", "uce", "nicheformer", "geneformer", "scanvi"],
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Direct path to best_model.pth. Overrides experiment_dir/tuning_mode.")
    parser.add_argument("--experiment_dir", type=str, default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--tuning_mode", type=str, default="freeze", choices=["freeze", "full"])
    parser.add_argument("--seed", type=int, default=3028)
    parser.add_argument("--donor_csv", type=str, default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", type=str, default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--merfish_h5ad", type=str, default=str(DEFAULT_MERFISH_H5AD))
    parser.add_argument("--eval_donor", type=str, default="", help="Optional fixed donor_id override.")
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--save_npz", type=str, default="", help="Optional path to save embeddings and predictions.")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))

    parser.add_argument("--scgpt_gene_ids_path", type=str, default=str(DEFAULT_SCGPT_GENE_IDS))
    parser.add_argument("--scgpt_brain_pretrained_dir", type=str, default=str(DEFAULT_SCGPT_BRAIN_CKPT))
    parser.add_argument("--scgpt_human_pretrained_dir", type=str, default=str(DEFAULT_SCGPT_HUMAN_CKPT))
    parser.add_argument("--uce_model_dir", type=str, default=str(DEFAULT_UCE_MODEL_DIR))
    parser.add_argument("--nicheformer_backbone_path", type=str, default=str(DEFAULT_NICHEFORMER_CKPT))
    parser.add_argument("--nicheformer_vocab_path", type=str, default=str(DEFAULT_NICHEFORMER_VOCAB))
    parser.add_argument("--nicheformer_merfish_mean_path", type=str, default=str(DEFAULT_NICHEFORMER_MERFISH_MEAN))
    parser.add_argument("--gene_name_to_ens_path", type=str, default=str(DEFAULT_GENE_NAME_TO_ENS))
    parser.add_argument("--geneformer_code_dir", type=str, default=str(DEFAULT_GENEFORMER_CODE_DIR))
    parser.add_argument("--geneformer_weights_path", type=str, default=str(DEFAULT_GENEFORMER_WEIGHTS))
    args = parser.parse_args()

    gene_panel, label_maps, label_spaces = load_merfish_panel_and_label_maps(Path(args.merfish_h5ad))
    checkpoint_path = resolve_checkpoint_path(args)
    donor_id, donor_cells_total = resolve_eval_donor(args)
    donor_path = Path(args.donor_dir) / f"{donor_id}.h5ad"

    adata = sample_donor_cells(donor_path, sample_n=args.sample_n, sample_seed=args.sample_seed)
    X_panel, matched_genes, missing_genes = build_panel_matrix(adata, gene_panel)
    labels, valid_mask, dropped_labels = encode_labels(adata.obs, label_maps)

    if not np.all(valid_mask):
        X_panel = X_panel[valid_mask]
        adata = adata[valid_mask].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned_model(args, gene_panel=gene_panel, checkpoint_path=checkpoint_path)
    mean_losses, metrics, logits_all, cell_emb = evaluate_model(
        model=model,
        X=X_panel,
        labels=labels,
        device=device,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )

    maybe_save_npz(args.save_npz, cell_emb=cell_emb, labels=labels, logits_all=logits_all)

    out = {
        "model_name": args.model_name,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "dataset": "SEA-AD MTG donor_h5ad",
        "task_type": "hierarchical_cell_type_classification",
        "donor_id": donor_id,
        "donor_cells_total": int(donor_cells_total),
        "sampled_cells_requested": int(args.sample_n),
        "sampled_cells_evaluated": int(X_panel.shape[0]),
        "split_rule": "scUniFlow donor split val_ratio=0.1 seed=42",
        "unseen_for_scuniflow_train": True,
        "gene_panel_size": int(len(gene_panel)),
        "matched_genes_in_donor": int(len(matched_genes)),
        "missing_genes_in_donor": missing_genes,
        "label_space_sizes_train": {
            "class": int(len(label_spaces["class"])),
            "subclass": int(len(label_spaces["subclass"])),
            "supertype": int(len(label_spaces["supertype"])),
        },
        "label_space_sizes_eval_present": {
            "class": int(np.unique(labels["class"]).size),
            "subclass": int(np.unique(labels["subclass"]).size),
            "supertype": int(np.unique(labels["supertype"]).size),
        },
        "dropped_unmapped_labels": dropped_labels,
        "cell_emb_dim": int(cell_emb.shape[1]),
        "batch_size": int(args.batch_size),
        **mean_losses,
        **metrics,
    }
    if args.save_npz:
        out["saved_npz"] = str(Path(args.save_npz))

    result_path = build_result_path(args, donor_id)
    out["result_json"] = str(result_path)
    result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
