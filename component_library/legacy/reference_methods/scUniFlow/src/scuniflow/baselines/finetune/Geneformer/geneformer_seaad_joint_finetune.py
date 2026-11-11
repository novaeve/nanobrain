import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import BertForMaskedLM


THIS_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = THIS_DIR.parent
EVAL_DIR = FINETUNE_DIR.parent
REPO_ROOT = EVAL_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(THIS_DIR))

from data import split_donors_train_val
from donor_eval_utils import DEFAULT_DONOR_CSV, DEFAULT_DONOR_DIR, sample_single_donor_cells


DEFAULT_GENEFORMER_CODE_DIR = Path("external/Geneformer")
DEFAULT_RESULT_ROOT = Path("./src/scuniflow/baselines/finetune/outputs")
DEFAULT_LABEL_KEYS = ("Class", "Subclass", "Supertype")
DEFAULT_TASK_NAMES = ("class", "subclass", "supertype")


def parse_csv_list(text: str):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def get_default_seq_len(model_version: str) -> int:
    if str(model_version).upper() == "V1":
        return 2048
    return 4096


def resolve_tokenizer_seq_len(model_version: str, requested_seq_len: int) -> int:
    if int(requested_seq_len) > 0:
        return int(requested_seq_len)
    return get_default_seq_len(model_version)


def load_geneformer_modules(code_dir: Path):
    code_dir = Path(code_dir).resolve()
    if not code_dir.exists():
        raise FileNotFoundError(f"Geneformer code directory not found: {code_dir}")
    pkg_dir = code_dir / "geneformer"
    if not pkg_dir.exists():
        raise FileNotFoundError(f"Geneformer package directory not found: {pkg_dir}")

    for name in ["geneformer.tokenizer", "geneformer"]:
        if name in sys.modules:
            del sys.modules[name]

    geneformer_pkg = types.ModuleType("geneformer")
    geneformer_pkg.__file__ = str(pkg_dir / "__init__.py")
    geneformer_pkg.__path__ = [str(pkg_dir)]
    geneformer_pkg.GENE_MEDIAN_FILE = pkg_dir / "gene_median_dictionary_gc104M.pkl"
    geneformer_pkg.TOKEN_DICTIONARY_FILE = pkg_dir / "token_dictionary_gc104M.pkl"
    geneformer_pkg.ENSEMBL_MAPPING_FILE = pkg_dir / "ensembl_mapping_dict_gc104M.pkl"
    geneformer_pkg.GENE_MEDIAN_FILE_30M = pkg_dir / "gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl"
    geneformer_pkg.TOKEN_DICTIONARY_FILE_30M = pkg_dir / "gene_dictionaries_30m/token_dictionary_gc30M.pkl"
    geneformer_pkg.ENSEMBL_MAPPING_FILE_30M = pkg_dir / "gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl"
    sys.modules["geneformer"] = geneformer_pkg

    if "loompy" not in sys.modules:
        loompy_stub = types.ModuleType("loompy")

        def _loompy_unavailable(*args, **kwargs):
            raise ModuleNotFoundError(
                "loompy is not installed. This script only supports Geneformer tokenization from .h5ad in the current environment."
            )

        loompy_stub.connect = _loompy_unavailable
        loompy_stub.create = _loompy_unavailable
        sys.modules["loompy"] = loompy_stub

    spec = importlib.util.spec_from_file_location(
        "geneformer.tokenizer",
        pkg_dir / "tokenizer.py",
    )
    tokenizer_module = importlib.util.module_from_spec(spec)
    sys.modules["geneformer.tokenizer"] = tokenizer_module
    spec.loader.exec_module(tokenizer_module)
    geneformer_pkg.tokenizer = tokenizer_module
    TranscriptomeTokenizer = tokenizer_module.TranscriptomeTokenizer

    return {
        "TranscriptomeTokenizer": TranscriptomeTokenizer,
        "V2": {
            "gene_median_file": Path(geneformer_pkg.GENE_MEDIAN_FILE),
            "token_dictionary_file": Path(geneformer_pkg.TOKEN_DICTIONARY_FILE),
            "gene_mapping_file": Path(geneformer_pkg.ENSEMBL_MAPPING_FILE),
        },
        "V1": {
            "gene_median_file": Path(geneformer_pkg.GENE_MEDIAN_FILE_30M),
            "token_dictionary_file": Path(geneformer_pkg.TOKEN_DICTIONARY_FILE_30M),
            "gene_mapping_file": Path(geneformer_pkg.ENSEMBL_MAPPING_FILE_30M),
        },
    }


def find_pretrained_weight_file(model_dir: Path):
    candidates = [
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_donor_cell_count_map(donor_csv: Path):
    donor_counts = {}
    with donor_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            donor_counts[row["donor_id"]] = int(row["n_cells"])
    return donor_counts


def allocate_cells_across_donors(donor_ids, donor_counts, total_cells: int, seed: int):
    available = []
    for donor_id in donor_ids:
        n = int(donor_counts.get(donor_id, 0))
        if n > 0:
            available.append((donor_id, n))
    if not available:
        return []

    total_available = sum(n for _, n in available)
    if total_cells is None or total_cells <= 0 or total_cells >= total_available:
        return [
            {"donor_id": donor_id, "alloc_cells": int(n), "donor_cells_total": int(n)}
            for donor_id, n in available
        ]

    donor_ids_arr = [x[0] for x in available]
    counts = np.array([x[1] for x in available], dtype=np.int64)
    raw = counts.astype(np.float64) * (float(total_cells) / float(total_available))
    alloc = np.floor(raw).astype(np.int64)
    remaining = int(total_cells - alloc.sum())

    frac = raw - alloc.astype(np.float64)
    rng = np.random.RandomState(seed)
    tie = rng.random(len(available))
    order = np.lexsort((-tie, -frac))
    for idx in order:
        if remaining <= 0:
            break
        if alloc[idx] < counts[idx]:
            alloc[idx] += 1
            remaining -= 1

    if remaining > 0:
        capacity = counts - alloc
        order2 = np.argsort(-capacity)
        for idx in order2:
            if remaining <= 0:
                break
            if capacity[idx] <= 0:
                continue
            take = min(int(capacity[idx]), remaining)
            alloc[idx] += take
            remaining -= take

    plan = []
    for donor_id, donor_total, donor_alloc in zip(donor_ids_arr, counts.tolist(), alloc.tolist()):
        if donor_alloc > 0:
            plan.append(
                {
                    "donor_id": donor_id,
                    "alloc_cells": int(donor_alloc),
                    "donor_cells_total": int(donor_total),
                }
            )
    return plan


def build_subset_from_donors(
    donor_ids,
    donor_counts,
    donor_dir: Path,
    total_cells: int,
    donor_seed: int,
    cell_seed: int,
    split_label: str,
):
    plan = allocate_cells_across_donors(
        donor_ids=donor_ids,
        donor_counts=donor_counts,
        total_cells=total_cells,
        seed=donor_seed,
    )
    if not plan:
        raise RuntimeError(f"No donors selected for {split_label}.")

    print(
        f"[data] {split_label}: total_cells_budget={total_cells}, "
        f"selected_donors={len(plan)}, donor_seed={donor_seed}, cell_seed={cell_seed}",
        flush=True,
    )

    adatas = []
    donor_stats = []
    donor_iter = tqdm(plan, total=len(plan), desc=f"loading {split_label} donors", unit="donor")
    for donor_idx, donor_plan in enumerate(donor_iter):
        donor_id = donor_plan["donor_id"]
        alloc_cells = int(donor_plan["alloc_cells"])
        donor_path = donor_dir / f"{donor_id}.h5ad"
        if not donor_path.exists():
            print(f"[warn] skip missing donor file: {donor_path}", flush=True)
            continue
        sampled_adata, donor_total, sampled_n = sample_single_donor_cells(
            donor_path=donor_path,
            sample_n=alloc_cells,
            sample_seed=int(cell_seed + donor_idx),
        )
        sampled_adata.obs = sampled_adata.obs.copy()
        sampled_adata.obs["_subset_donor_id"] = donor_id
        adatas.append(sampled_adata)
        donor_stats.append(
            {
                "donor_id": donor_id,
                "donor_cells_total": int(donor_total),
                "allocated_cells": int(alloc_cells),
                "sampled_cells": int(sampled_n),
                "sample_seed": int(cell_seed + donor_idx),
            }
        )

    if not adatas:
        raise RuntimeError(f"No donor data could be loaded for {split_label}.")

    print(f"[data] concatenating {split_label} donor subsets", flush=True)
    adata = (
        adatas[0].concatenate(
            *adatas[1:],
            join="inner",
            batch_key=f"_{split_label}_batch",
            index_unique=None,
        )
        if len(adatas) > 1
        else adatas[0]
    )
    print(f"[data] {split_label} subset ready: {adata.n_obs} cells × {adata.n_vars} genes", flush=True)
    return adata, donor_stats


def ensure_obs_n_counts(adata):
    if "n_counts" in adata.obs.columns:
        return
    x = adata.X
    if issparse(x):
        n_counts = np.asarray(x.sum(axis=1)).ravel()
    else:
        n_counts = np.asarray(x, dtype=np.float32).sum(axis=1)
    adata.obs = adata.obs.copy()
    adata.obs["n_counts"] = n_counts.astype(np.float32)


def ensure_unique_cell_ids(adata, split_label: str):
    adata.obs = adata.obs.copy()
    if "unique_cell_id" in adata.obs.columns:
        return
    obs_names = adata.obs_names.astype(str).tolist()
    adata.obs["unique_cell_id"] = [f"{split_label}:{x}" for x in obs_names]


def prepare_geneformer_adata(adata, gene_id_col: str, label_keys):
    ensure_obs_n_counts(adata)
    ensure_unique_cell_ids(adata, split_label="subset")
    adata.var = adata.var.copy()

    if gene_id_col == "index":
        gene_ids = adata.var_names.astype(str)
    else:
        if gene_id_col not in adata.var.columns:
            raise KeyError(f"Gene ID column not found in adata.var: {gene_id_col}")
        gene_ids = adata.var[gene_id_col].astype(str)
    adata.var["ensembl_id"] = gene_ids.astype(str)

    adata.obs = adata.obs.copy()
    for key in label_keys:
        if key not in adata.obs.columns:
            raise KeyError(f"Annotation column not found in adata.obs: {key}")
        adata.obs[key] = adata.obs[key].astype(str)
    return adata


def summarize_gene_overlap(adata, tokenizer):
    gene_ids = adata.var["ensembl_id"].astype(str).str.upper().to_numpy()
    mapped = np.array([tokenizer.gene_mapping_dict.get(g) for g in gene_ids], dtype=object)
    mapped_valid = mapped[mapped != None]  # noqa: E711
    in_vocab = np.array([g in tokenizer.gene_token_dict for g in mapped_valid], dtype=bool)
    in_median = np.array([g in tokenizer.gene_median_dict for g in mapped_valid], dtype=bool)
    usable = int(np.sum(in_vocab & in_median))
    return {
        "input_genes": int(len(gene_ids)),
        "mapped_genes": int(len(mapped_valid)),
        "usable_genes": int(usable),
        "unmapped_genes": int(len(gene_ids) - len(mapped_valid)),
    }


def tokenize_adata_to_dataset(
    adata,
    tokenizer,
    temp_h5ad_path: Path,
    metadata_cols,
    target_sum: float,
    save_to_disk_path: Path | None = None,
):
    temp_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(temp_h5ad_path)
    try:
        tokenized_cells, cell_metadata, tokenized_counts = tokenizer.tokenize_anndata(
            temp_h5ad_path,
            target_sum=target_sum,
            file_format="h5ad",
        )
        dataset = tokenizer.create_dataset(
            tokenized_cells=tokenized_cells,
            cell_metadata=cell_metadata,
            tokenized_counts=tokenized_counts,
            use_generator=False,
        )
        keep_cols = ["input_ids", "length"]
        for col in metadata_cols:
            if col in dataset.column_names:
                keep_cols.append(col)
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])
        if save_to_disk_path is not None:
            if save_to_disk_path.exists():
                shutil.rmtree(save_to_disk_path)
            dataset.save_to_disk(str(save_to_disk_path))
        return dataset
    finally:
        if temp_h5ad_path.exists():
            temp_h5ad_path.unlink()


def build_label_spaces(dataset: Dataset, label_keys, task_names):
    label_spaces = {}
    for obs_key, task_name in zip(label_keys, task_names):
        if obs_key not in dataset.column_names:
            raise KeyError(f"Annotation column missing from tokenized dataset: {obs_key}")
        values = [str(x) for x in dataset[obs_key]]
        label_spaces[task_name] = sorted(set(values))
    return label_spaces


def encode_annotation_labels(dataset: Dataset, label_keys, task_names, label_spaces):
    encoded = {}
    valid_mask = np.ones(len(dataset), dtype=bool)
    dropped = {}

    for obs_key, task_name in zip(label_keys, task_names):
        mapping = {name: idx for idx, name in enumerate(label_spaces[task_name])}
        values = np.array([str(x) for x in dataset[obs_key]], dtype=object)
        ids = np.array([mapping.get(v, -1) for v in values], dtype=np.int64)
        is_valid = ids >= 0
        if not np.all(is_valid):
            dropped[task_name] = sorted(set(values[~is_valid].tolist()))
        valid_mask &= is_valid
        encoded[task_name] = ids

    if np.all(valid_mask):
        return dataset, encoded, dropped

    keep_indices = np.where(valid_mask)[0].tolist()
    dataset = dataset.select(keep_indices)
    encoded = {task_name: ids[valid_mask] for task_name, ids in encoded.items()}
    return dataset, encoded, dropped


def mask_sequence(
    seq,
    rng: np.random.RandomState,
    mlm_probability: float,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: set[int],
):
    input_ids = np.asarray(seq, dtype=np.int64).copy()
    labels = np.full(input_ids.shape, -100, dtype=np.int64)
    candidates = np.array(
        [idx for idx, token_id in enumerate(input_ids.tolist()) if token_id not in special_token_ids],
        dtype=np.int64,
    )
    if candidates.size == 0:
        return input_ids, labels

    chosen = candidates[rng.rand(candidates.size) < mlm_probability]
    if chosen.size == 0:
        chosen = np.array([candidates[rng.randint(0, candidates.size)]], dtype=np.int64)

    labels[chosen] = input_ids[chosen]
    replace_prob = rng.rand(chosen.size)

    replace_mask = chosen[replace_prob < 0.8]
    input_ids[replace_mask] = mask_token_id

    random_mask = chosen[(replace_prob >= 0.8) & (replace_prob < 0.9)]
    if random_mask.size > 0:
        input_ids[random_mask] = rng.randint(0, vocab_size, size=random_mask.size, dtype=np.int64)

    return input_ids, labels


class PreparedGeneformerDataset(TorchDataset):
    def __init__(self, examples, task_names):
        self.examples = examples
        self.task_names = tuple(task_names)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        item = {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "mlm_labels": torch.tensor(ex["mlm_labels"], dtype=torch.long),
        }
        for task_name in self.task_names:
            item[f"labels_{task_name}"] = torch.tensor(ex[f"labels_{task_name}"], dtype=torch.long)
        return item


def build_prepared_examples(
    dataset: Dataset,
    encoded_labels,
    task_names,
    use_recon: bool,
    mlm_probability: float,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: set[int],
    seed: int,
):
    rng = np.random.RandomState(seed)
    input_ids_list = dataset["input_ids"]
    examples = []
    iterator = tqdm(
        range(len(input_ids_list)),
        total=len(input_ids_list),
        desc="building masked examples",
        unit="cell",
    )
    for idx in iterator:
        seq = input_ids_list[idx]
        if use_recon:
            masked_input_ids, mlm_labels = mask_sequence(
                seq=seq,
                rng=rng,
                mlm_probability=mlm_probability,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                special_token_ids=special_token_ids,
            )
        else:
            masked_input_ids = np.asarray(seq, dtype=np.int64)
            mlm_labels = np.full(masked_input_ids.shape, -100, dtype=np.int64)

        ex = {
            "input_ids": masked_input_ids.tolist(),
            "mlm_labels": mlm_labels.tolist(),
        }
        for task_name in task_names:
            ex[f"labels_{task_name}"] = int(encoded_labels[task_name][idx])
        examples.append(ex)
    return examples


def make_collate_fn(task_names, pad_token_id: int):
    task_names = tuple(task_names)

    def collate(examples):
        batch_size = len(examples)
        max_len = max(ex["input_ids"].shape[0] for ex in examples)
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        mlm_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
        }

        for task_name in task_names:
            batch[f"labels_{task_name}"] = torch.zeros(batch_size, dtype=torch.long)

        for row_idx, ex in enumerate(examples):
            seq_len = ex["input_ids"].shape[0]
            input_ids[row_idx, :seq_len] = ex["input_ids"]
            attention_mask[row_idx, :seq_len] = 1
            mlm_labels[row_idx, :seq_len] = ex["mlm_labels"]
            for task_name in task_names:
                batch[f"labels_{task_name}"][row_idx] = ex[f"labels_{task_name}"]

        return batch

    return collate


class AnnotationHead(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class GeneformerReconAnnotationModel(nn.Module):
    def __init__(
        self,
        pretrained_model_dir: Path,
        num_labels_by_task,
        annotation_head_dropout: float,
        pooling: str,
        freeze_layers: int,
        freeze_embeddings: bool,
    ):
        super().__init__()
        self.backbone = BertForMaskedLM.from_pretrained(str(pretrained_model_dir))
        self.pooling = pooling
        hidden_size = int(self.backbone.config.hidden_size)

        self.annotation_heads = nn.ModuleDict(
            {
                task_name: AnnotationHead(hidden_size, n_classes, annotation_head_dropout)
                for task_name, n_classes in num_labels_by_task.items()
            }
        )

        if freeze_embeddings:
            for param in self.backbone.bert.embeddings.parameters():
                param.requires_grad = False
        if freeze_layers > 0:
            for layer in self.backbone.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def encode(self, input_ids, attention_mask):
        outputs = self.backbone.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        mlm_logits = self.backbone.cls(sequence_output)

        if self.pooling == "cls":
            pooled = sequence_output[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
            pooled = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        annotation_logits = {
            task_name: head(pooled) for task_name, head in self.annotation_heads.items()
        }
        return mlm_logits, annotation_logits


def safe_auc(y_true, prob):
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
    auc = safe_auc(y_true, y_prob)

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


def aggregate_annotation_metrics(task_names, logits_all, labels_all):
    if not task_names:
        return {}

    metrics = {}
    macro_f1s = []
    accuracies = []
    for task_name in task_names:
        task_metrics = summarize_task_metrics(task_name, labels_all[task_name], logits_all[task_name])
        metrics.update(task_metrics)
        macro_f1s.append(task_metrics[f"{task_name}_macro_f1"])
        accuracies.append(task_metrics[f"{task_name}_accuracy"])

    metrics["annotation_mean_macro_f1"] = float(np.mean(macro_f1s)) if macro_f1s else None
    metrics["annotation_mean_accuracy"] = float(np.mean(accuracies)) if accuracies else None

    if len(task_names) > 1:
        exact = np.ones_like(labels_all[task_names[0]], dtype=bool)
        for task_name in task_names:
            exact &= logits_all[task_name].argmax(axis=1) == labels_all[task_name]
        metrics["annotation_joint_exact_match"] = float(np.mean(exact))
    return metrics


def run_model(model, batch, device, amp, use_recon, use_annotation, task_names):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    mlm_labels = batch["mlm_labels"].to(device)

    with torch.cuda.amp.autocast(enabled=amp):
        mlm_logits, annotation_logits = model.encode(input_ids, attention_mask)

        mlm_loss = None
        if use_recon:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.shape[-1]),
                mlm_labels.view(-1),
                ignore_index=-100,
            )

        annotation_losses = {}
        if use_annotation:
            for task_name in task_names:
                labels = batch[f"labels_{task_name}"].to(device)
                annotation_losses[task_name] = F.cross_entropy(annotation_logits[task_name], labels)

    return {
        "mlm_logits": mlm_logits,
        "annotation_logits": annotation_logits,
        "mlm_loss": mlm_loss,
        "annotation_losses": annotation_losses,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp,
    use_recon,
    use_annotation,
    task_names,
    mlm_loss_weight,
    annotation_loss_weight,
    grad_clip,
):
    model.train()

    total_loss = 0.0
    total_mlm = 0.0
    total_batches = 0
    total_samples = 0
    total_ann = {task_name: 0.0 for task_name in task_names}
    total_ann_acc = {task_name: 0.0 for task_name in task_names}

    it = tqdm(loader, total=len(loader), desc="train", unit="batch")
    for batch in it:
        batch_size = int(batch["input_ids"].shape[0])
        total_samples += batch_size

        optimizer.zero_grad(set_to_none=True)
        outputs = run_model(
            model=model,
            batch=batch,
            device=device,
            amp=amp,
            use_recon=use_recon,
            use_annotation=use_annotation,
            task_names=task_names,
        )

        loss = 0.0
        postfix = {}
        if use_recon:
            loss = loss + mlm_loss_weight * outputs["mlm_loss"]
            total_mlm += float(outputs["mlm_loss"].item())
            postfix["mlm"] = f"{outputs['mlm_loss'].item():.4f}"

        if use_annotation:
            ann_loss = sum(outputs["annotation_losses"].values()) / max(len(task_names), 1)
            loss = loss + annotation_loss_weight * ann_loss
            postfix["cls"] = f"{ann_loss.item():.4f}"
            for task_name in task_names:
                logits = outputs["annotation_logits"][task_name]
                labels = batch[f"labels_{task_name}"].to(device)
                total_ann[task_name] += float(
                    F.cross_entropy(logits, labels, reduction="sum").item()
                )
                total_ann_acc[task_name] += float((logits.argmax(dim=1) == labels).sum().item())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_batches += 1
        postfix["loss"] = f"{loss.item():.4f}"
        it.set_postfix(**postfix)

    metrics = {"train_loss": total_loss / max(total_batches, 1)}
    if use_recon:
        metrics["train_mlm_loss"] = total_mlm / max(total_batches, 1)
    if use_annotation:
        ann_losses = []
        ann_accs = []
        for task_name in task_names:
            metrics[f"train_loss_{task_name}"] = total_ann[task_name] / max(total_samples, 1)
            metrics[f"train_acc_{task_name}"] = total_ann_acc[task_name] / max(total_samples, 1)
            ann_losses.append(metrics[f"train_loss_{task_name}"])
            ann_accs.append(metrics[f"train_acc_{task_name}"])
        metrics["train_loss_annotation_mean"] = float(np.mean(ann_losses)) if ann_losses else None
        metrics["train_acc_annotation_mean"] = float(np.mean(ann_accs)) if ann_accs else None
    return metrics


def evaluate_epoch(
    model,
    loader,
    device,
    amp,
    use_recon,
    use_annotation,
    task_names,
):
    model.eval()

    total_mlm = 0.0
    total_batches = 0
    total_samples = 0
    total_ann = {task_name: 0.0 for task_name in task_names}
    logits_all = {task_name: [] for task_name in task_names}
    labels_all = {task_name: [] for task_name in task_names}

    with torch.no_grad():
        it = tqdm(loader, total=len(loader), desc="valid", unit="batch")
        for batch in it:
            batch_size = int(batch["input_ids"].shape[0])
            total_samples += batch_size
            outputs = run_model(
                model=model,
                batch=batch,
                device=device,
                amp=amp,
                use_recon=use_recon,
                use_annotation=use_annotation,
                task_names=task_names,
            )

            postfix = {}
            if use_recon:
                total_mlm += float(outputs["mlm_loss"].item())
                postfix["mlm"] = f"{outputs['mlm_loss'].item():.4f}"

            if use_annotation:
                for task_name in task_names:
                    logits = outputs["annotation_logits"][task_name]
                    labels = batch[f"labels_{task_name}"].to(device)
                    total_ann[task_name] += float(
                        F.cross_entropy(logits, labels, reduction="sum").item()
                    )
                    logits_all[task_name].append(logits.cpu().numpy())
                    labels_all[task_name].append(labels.cpu().numpy())
                ann_loss = sum(total_ann.values()) / max(total_samples * max(len(task_names), 1), 1)
                postfix["cls"] = f"{ann_loss:.4f}"

            total_batches += 1
            it.set_postfix(**postfix)

    metrics = {}
    if use_recon:
        metrics["valid_mlm_loss"] = total_mlm / max(total_batches, 1)
    if use_annotation:
        logits_cat = {task_name: np.concatenate(logits_all[task_name], axis=0) for task_name in task_names}
        labels_cat = {task_name: np.concatenate(labels_all[task_name], axis=0) for task_name in task_names}
        for task_name in task_names:
            metrics[f"valid_loss_{task_name}"] = total_ann[task_name] / max(total_samples, 1)
        metrics["valid_loss_annotation_mean"] = float(
            np.mean([metrics[f"valid_loss_{task_name}"] for task_name in task_names])
        )
        metrics.update(aggregate_annotation_metrics(task_names, logits_cat, labels_cat))
    return metrics


def make_selection_score(metrics, mode, use_recon, use_annotation):
    if mode == "valid_mlm_loss":
        return -float(metrics["valid_mlm_loss"])
    if mode == "annotation_mean_macro_f1":
        value = metrics.get("annotation_mean_macro_f1")
        return float("-inf") if value is None else float(value)
    if mode == "joint_exact_match":
        value = metrics.get("annotation_joint_exact_match")
        return float("-inf") if value is None else float(value)
    if mode == "auto":
        if use_annotation:
            value = metrics.get("annotation_mean_macro_f1")
            return float("-inf") if value is None else float(value)
        if use_recon:
            return -float(metrics["valid_mlm_loss"])
    raise ValueError(f"Unsupported model_selection_metric: {mode}")


def build_output_dir(args) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        label_tag = "-".join(args.annotation_task_names) if args.annotation_task_names else "nolabel"
        run_name = args.run_name or (
            f"geneformer_{args.model_version.lower()}_{args.training_tasks}_{label_tag}_"
            f"train{args.train_total_cells}_val{args.val_total_cells}_s{args.seed}"
        )
        out_dir = Path(args.result_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_best_checkpoint(
    out_dir: Path,
    model,
    tokenizer_files,
    tokenizer_seq_len,
    args,
    label_spaces,
    best_metrics,
    summary,
):
    backbone_dir = out_dir / "best_backbone"
    if backbone_dir.exists():
        shutil.rmtree(backbone_dir)
    model.backbone.save_pretrained(str(backbone_dir))

    tokenizer_manifest = {
        "model_version": args.model_version,
        "geneformer_code_dir": str(args.geneformer_code_dir),
        "gene_median_file": str(tokenizer_files["gene_median_file"]),
        "token_dictionary_file": str(tokenizer_files["token_dictionary_file"]),
        "gene_mapping_file": str(tokenizer_files["gene_mapping_file"]),
        "tokenizer_model_input_size": int(tokenizer_seq_len),
    }

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    torch.save(model.annotation_heads.state_dict(), out_dir / "annotation_heads.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "annotation_heads_state_dict": model.annotation_heads.state_dict(),
            "args": vars(args),
            "label_spaces": label_spaces,
            "best_metrics": best_metrics,
            "tokenizer_manifest": tokenizer_manifest,
        },
        out_dir / "best_joint_model.pt",
    )

    (out_dir / "label_spaces.json").write_text(json.dumps(label_spaces, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "finetune_config.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "tokenizer_manifest.json").write_text(json.dumps(tokenizer_manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Joint Geneformer fine-tuning on SEA-AD with masked reconstruction and/or annotation."
    )
    parser.add_argument("--geneformer_code_dir", type=str, default=str(DEFAULT_GENEFORMER_CODE_DIR))
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="",
        help="Directory containing a pretrained Geneformer checkpoint with config and weights.",
    )
    parser.add_argument("--model_version", type=str, default="V2", choices=["V1", "V2"])
    parser.add_argument("--donor_csv", type=str, default=str(DEFAULT_DONOR_CSV))
    parser.add_argument("--donor_dir", type=str, default=str(DEFAULT_DONOR_DIR))
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_total_cells", type=int, default=12000)
    parser.add_argument("--val_total_cells", type=int, default=2000)
    parser.add_argument("--train_donor_seed", type=int, default=1234)
    parser.add_argument("--val_donor_seed", type=int, default=2345)
    parser.add_argument("--train_cell_seed", type=int, default=3456)
    parser.add_argument("--val_cell_seed", type=int, default=4567)
    parser.add_argument("--gene_id_col", type=str, default="index", help="adata.var column with Ensembl IDs, or 'index'.")
    parser.add_argument("--target_sum", type=float, default=10000.0)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=0,
        help="Tokenizer model_input_size. 0 means use Geneformer default for the selected model version.",
    )
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--mask_seed", type=int, default=5678)
    parser.add_argument("--val_mask_seed", type=int, default=6789)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--freeze_layers", type=int, default=0)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--annotation_head_dropout", type=float, default=0.2)
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0)
    parser.add_argument("--annotation_loss_weight", type=float, default=1.0)
    parser.add_argument("--save_tokenized_dataset", action="store_true")
    parser.add_argument(
        "--training_tasks",
        type=str,
        default="joint",
        choices=["recon", "annotation", "joint"],
    )
    parser.add_argument(
        "--annotation_label_keys",
        type=str,
        default=",".join(DEFAULT_LABEL_KEYS),
        help="Comma-separated adata.obs columns used for annotation.",
    )
    parser.add_argument(
        "--annotation_task_names",
        type=str,
        default=",".join(DEFAULT_TASK_NAMES),
        help="Comma-separated task names aligned with --annotation_label_keys.",
    )
    parser.add_argument(
        "--model_selection_metric",
        type=str,
        default="auto",
        choices=["auto", "valid_mlm_loss", "annotation_mean_macro_f1", "joint_exact_match"],
    )
    args = parser.parse_args()

    args.geneformer_code_dir = str(Path(args.geneformer_code_dir).resolve())
    args.annotation_label_keys = tuple(parse_csv_list(args.annotation_label_keys))
    args.annotation_task_names = tuple(parse_csv_list(args.annotation_task_names))
    if len(args.annotation_label_keys) != len(args.annotation_task_names):
        raise ValueError("--annotation_label_keys and --annotation_task_names must have the same length.")

    use_recon = args.training_tasks in ("recon", "joint")
    use_annotation = args.training_tasks in ("annotation", "joint")
    if not use_recon and not use_annotation:
        raise ValueError("At least one task must be enabled.")
    if use_annotation and not args.annotation_task_names:
        raise ValueError("Annotation training requires at least one label task.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    module_info = load_geneformer_modules(Path(args.geneformer_code_dir))
    tokenizer_files = module_info[args.model_version]
    TranscriptomeTokenizer = module_info["TranscriptomeTokenizer"]
    tokenizer_seq_len = resolve_tokenizer_seq_len(args.model_version, args.seq_len)

    pretrained_model_dir = Path(args.pretrained_model_dir).resolve() if args.pretrained_model_dir else None
    if pretrained_model_dir is None:
        raise ValueError("--pretrained_model_dir is required because Geneformer_code_only does not include weights.")
    if not pretrained_model_dir.exists():
        raise FileNotFoundError(f"Pretrained model directory not found: {pretrained_model_dir}")
    if find_pretrained_weight_file(pretrained_model_dir) is None:
        raise FileNotFoundError(
            f"No pretrained weights found in {pretrained_model_dir}. "
            "Expected `pytorch_model.bin` or `model.safetensors`."
        )

    donor_dir = Path(args.donor_dir)
    donor_csv = Path(args.donor_csv)
    out_dir = build_output_dir(args)
    donor_counts = load_donor_cell_count_map(donor_csv)

    print(
        f"[setup] model_version={args.model_version} training_tasks={args.training_tasks} "
        f"epochs={args.epochs} batch_size={args.batch_size} tokenizer_seq_len={tokenizer_seq_len}",
        flush=True,
    )
    print(f"[setup] pretrained_model_dir={pretrained_model_dir}", flush=True)
    print(f"[setup] output_dir={out_dir}", flush=True)

    train_donors, val_donors, split_info = split_donors_train_val(
        str(donor_csv),
        val_ratio=0.1,
        seed=args.split_seed,
    )
    print(
        f"[split] train donors={len(train_donors)} val donors={len(val_donors)} "
        f"val_ratio_actual={split_info['val_ratio_actual']:.2%}",
        flush=True,
    )

    train_adata, train_donor_stats = build_subset_from_donors(
        donor_ids=train_donors,
        donor_counts=donor_counts,
        donor_dir=donor_dir,
        total_cells=args.train_total_cells,
        donor_seed=args.train_donor_seed,
        cell_seed=args.train_cell_seed,
        split_label="train",
    )
    val_adata, val_donor_stats = build_subset_from_donors(
        donor_ids=val_donors,
        donor_counts=donor_counts,
        donor_dir=donor_dir,
        total_cells=args.val_total_cells,
        donor_seed=args.val_donor_seed,
        cell_seed=args.val_cell_seed,
        split_label="val",
    )

    train_adata = prepare_geneformer_adata(train_adata, args.gene_id_col, args.annotation_label_keys)
    val_adata = prepare_geneformer_adata(val_adata, args.gene_id_col, args.annotation_label_keys)

    tokenizer_metadata_cols = ["unique_cell_id"]
    tokenizer_metadata_cols.extend(list(args.annotation_label_keys))
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={col: col for col in tokenizer_metadata_cols},
        nproc=args.nproc,
        chunk_size=args.chunk_size,
        model_input_size=tokenizer_seq_len,
        model_version=args.model_version,
        use_h5ad_index=False,
        keep_counts=False,
        gene_median_file=tokenizer_files["gene_median_file"],
        token_dictionary_file=tokenizer_files["token_dictionary_file"],
        gene_mapping_file=tokenizer_files["gene_mapping_file"],
    )

    train_gene_stats = summarize_gene_overlap(train_adata, tokenizer)
    val_gene_stats = summarize_gene_overlap(val_adata, tokenizer)
    print(f"[data] train gene summary: {json.dumps(train_gene_stats, ensure_ascii=False)}", flush=True)
    print(f"[data] val gene summary: {json.dumps(val_gene_stats, ensure_ascii=False)}", flush=True)

    tokenized_root = out_dir / "tokenized"
    tokenized_train_path = tokenized_root / "train.dataset" if args.save_tokenized_dataset else None
    tokenized_val_path = tokenized_root / "val.dataset" if args.save_tokenized_dataset else None

    print("[prep] tokenizing train subset with official Geneformer tokenizer", flush=True)
    tokenized_train = tokenize_adata_to_dataset(
        adata=train_adata,
        tokenizer=tokenizer,
        temp_h5ad_path=out_dir / "_tmp" / "train_subset.h5ad",
        metadata_cols=tokenizer_metadata_cols,
        target_sum=args.target_sum,
        save_to_disk_path=tokenized_train_path,
    )
    print("[prep] tokenizing val subset with official Geneformer tokenizer", flush=True)
    tokenized_val = tokenize_adata_to_dataset(
        adata=val_adata,
        tokenizer=tokenizer,
        temp_h5ad_path=out_dir / "_tmp" / "val_subset.h5ad",
        metadata_cols=tokenizer_metadata_cols,
        target_sum=args.target_sum,
        save_to_disk_path=tokenized_val_path,
    )

    print(
        f"[data] tokenized train cells={len(tokenized_train)} val cells={len(tokenized_val)} "
        f"train_max_len={max(tokenized_train['length']) if len(tokenized_train) else 0} "
        f"val_max_len={max(tokenized_val['length']) if len(tokenized_val) else 0}",
        flush=True,
    )

    label_spaces = {}
    train_labels = {}
    val_labels = {}
    dropped_train_labels = {}
    dropped_val_labels = {}
    if use_annotation:
        label_spaces = build_label_spaces(
            dataset=tokenized_train,
            label_keys=args.annotation_label_keys,
            task_names=args.annotation_task_names,
        )
        tokenized_train, train_labels, dropped_train_labels = encode_annotation_labels(
            dataset=tokenized_train,
            label_keys=args.annotation_label_keys,
            task_names=args.annotation_task_names,
            label_spaces=label_spaces,
        )
        tokenized_val, val_labels, dropped_val_labels = encode_annotation_labels(
            dataset=tokenized_val,
            label_keys=args.annotation_label_keys,
            task_names=args.annotation_task_names,
            label_spaces=label_spaces,
        )
        if len(tokenized_train) == 0 or len(tokenized_val) == 0:
            raise RuntimeError("No cells remain after annotation label filtering. Check label columns.")
        print(
            f"[annotation] train cells after label filtering={len(tokenized_train)} "
            f"val cells after label filtering={len(tokenized_val)}",
            flush=True,
        )
    else:
        train_labels = {}
        val_labels = {}

    num_labels_by_task = {task_name: len(label_spaces[task_name]) for task_name in args.annotation_task_names}
    model = GeneformerReconAnnotationModel(
        pretrained_model_dir=pretrained_model_dir,
        num_labels_by_task=num_labels_by_task if use_annotation else {},
        annotation_head_dropout=args.annotation_head_dropout,
        pooling=args.pooling,
        freeze_layers=args.freeze_layers,
        freeze_embeddings=args.freeze_embeddings,
    )

    pad_token_id = int(tokenizer.gene_token_dict["<pad>"])
    mask_token_id = int(tokenizer.gene_token_dict["<mask>"])
    cls_token_id = int(tokenizer.gene_token_dict.get("<cls>", -1))
    eos_token_id = int(tokenizer.gene_token_dict.get("<eos>", -1))
    special_token_ids = {pad_token_id, mask_token_id}
    if cls_token_id >= 0:
        special_token_ids.add(cls_token_id)
    if eos_token_id >= 0:
        special_token_ids.add(eos_token_id)

    fixed_val_examples = build_prepared_examples(
        dataset=tokenized_val,
        encoded_labels=val_labels,
        task_names=args.annotation_task_names,
        use_recon=use_recon,
        mlm_probability=args.mlm_probability,
        mask_token_id=mask_token_id,
        vocab_size=int(model.backbone.config.vocab_size),
        special_token_ids=special_token_ids,
        seed=args.val_mask_seed,
    )
    val_dataset = PreparedGeneformerDataset(fixed_val_examples, args.annotation_task_names)

    collate_fn = make_collate_fn(args.annotation_task_names, pad_token_id=pad_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and torch.cuda.is_available())
    model.to(device)
    print(
        f"[setup] device={device} amp={amp} freeze_layers={args.freeze_layers} "
        f"freeze_embeddings={args.freeze_embeddings}",
        flush=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    history = []
    history_path = out_dir / "history.jsonl"
    best_selection_score = float("-inf")
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch] {epoch}/{args.epochs}", flush=True)
        train_examples = build_prepared_examples(
            dataset=tokenized_train,
            encoded_labels=train_labels,
            task_names=args.annotation_task_names,
            use_recon=use_recon,
            mlm_probability=args.mlm_probability,
            mask_token_id=mask_token_id,
            vocab_size=int(model.backbone.config.vocab_size),
            special_token_ids=special_token_ids,
            seed=int(args.mask_seed + epoch * 100000),
        )
        train_dataset = PreparedGeneformerDataset(train_examples, args.annotation_task_names)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

        t0 = time.time()
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            use_recon=use_recon,
            use_annotation=use_annotation,
            task_names=args.annotation_task_names,
            mlm_loss_weight=args.mlm_loss_weight,
            annotation_loss_weight=args.annotation_loss_weight,
            grad_clip=args.grad_clip,
        )
        valid_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            device=device,
            amp=amp,
            use_recon=use_recon,
            use_annotation=use_annotation,
            task_names=args.annotation_task_names,
        )

        epoch_metrics = {
            "epoch": epoch,
            "seconds": round(time.time() - t0, 2),
            "training_tasks": args.training_tasks,
            "train_cells": len(tokenized_train),
            "val_cells": len(tokenized_val),
            "mlm_probability": float(args.mlm_probability if use_recon else 0.0),
            **train_metrics,
            **valid_metrics,
        }
        history.append(epoch_metrics)
        with history_path.open("a") as f:
            f.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")

        print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2), flush=True)

        selection_score = make_selection_score(
            metrics=epoch_metrics,
            mode=args.model_selection_metric,
            use_recon=use_recon,
            use_annotation=use_annotation,
        )
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_metrics = dict(epoch_metrics)
            summary = {
                "pretrained_model_dir": str(pretrained_model_dir),
                "geneformer_code_dir": str(args.geneformer_code_dir),
                "dataset": "SEA-AD MTG donor_h5ad",
                "split_rule": f"scUniFlow donor split val_ratio=0.1 seed={args.split_seed}",
                "model_version": args.model_version,
                "training_tasks": args.training_tasks,
                "annotation_label_keys": list(args.annotation_label_keys),
                "annotation_task_names": list(args.annotation_task_names),
                "label_spaces": label_spaces,
                "dropped_train_labels": dropped_train_labels,
                "dropped_val_labels": dropped_val_labels,
                "pooling": args.pooling,
                "freeze_layers": int(args.freeze_layers),
                "freeze_embeddings": bool(args.freeze_embeddings),
                "target_sum": float(args.target_sum),
                "chunk_size": int(args.chunk_size),
                "nproc": int(args.nproc),
                "seq_len": int(tokenizer_seq_len),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "mlm_probability": float(args.mlm_probability if use_recon else 0.0),
                "mlm_loss_weight": float(args.mlm_loss_weight),
                "annotation_loss_weight": float(args.annotation_loss_weight),
                "model_selection_metric": args.model_selection_metric,
                "train_total_cells_target": int(args.train_total_cells),
                "val_total_cells_target": int(args.val_total_cells),
                "train_donor_sampling": train_donor_stats,
                "val_donor_sampling": val_donor_stats,
                "train_gene_summary": train_gene_stats,
                "val_gene_summary": val_gene_stats,
                "best_metrics": best_metrics,
            }
            save_best_checkpoint(
                out_dir=out_dir,
                model=model,
                tokenizer_files=tokenizer_files,
                tokenizer_seq_len=tokenizer_seq_len,
                args=args,
                label_spaces=label_spaces,
                best_metrics=best_metrics,
                summary=summary,
            )

    print(f"[done] best selection metric={args.model_selection_metric} saved under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
