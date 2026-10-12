"""Frozen foundation-model adapters with train-only linear probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Callable

import numpy as np

from .base import AdapterContext, Availability, BaselineAdapter, TaskPrediction
from .common import array, labels, package_version, panel_pseudocounts
from .registry import register


def _catalog(data_root: Path) -> tuple[list[str], list[str]]:
    import pandas as pd

    panel = np.load(data_root / "panel_gene_index.npy", allow_pickle=False)
    table = pd.read_parquet(data_root.parent / "human_v3_genes.parquet").set_index("gene_index")
    ensembl = [str(table.loc[int(index), "ensembl_id"]).split(".", 1)[0] for index in panel]
    symbols = [str(table.loc[int(index), "gene_symbol"]) for index in panel]
    return ensembl, symbols


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_probe_predictions(
    context: AdapterContext,
    train_embedding: np.ndarray,
    validation_embedding: np.ndarray,
    *,
    method: str,
    notes: str,
) -> list[TaskPrediction]:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    train_annotation, train_annotation_valid = labels(context.train, "annotation")
    validation_annotation, validation_annotation_valid = labels(context.validation, "annotation")
    classes = np.arange(
        int(max(train_annotation[train_annotation_valid].max(), validation_annotation[validation_annotation_valid].max())) + 1,
        dtype=np.int64,
    )
    scaler = StandardScaler().fit(train_embedding[train_annotation_valid])
    train_x = scaler.transform(train_embedding[train_annotation_valid])
    validation_x = scaler.transform(validation_embedding[validation_annotation_valid])
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1.0e-4,
        max_iter=int(context.options.get("foundation_probe_epochs", 100)),
        tol=1.0e-4,
        class_weight="balanced",
        n_jobs=max(1, int(context.threads)),
        random_state=int(context.seed),
    )
    classifier.fit(train_x, train_annotation[train_annotation_valid].astype(np.int64))
    local_probability = np.asarray(classifier.predict_proba(validation_x), dtype=np.float32)
    probability = np.zeros((len(validation_x), len(classes)), dtype=np.float32)
    probability[:, np.asarray(classifier.classes_, dtype=np.int64)] = local_probability
    prediction = classes[np.argmax(probability, axis=1)]
    annotation_indices = np.flatnonzero(validation_annotation_valid)
    annotation = TaskPrediction(
        task="annotation",
        arrays={
            "y_true": validation_annotation[validation_annotation_valid].astype(np.int64),
            "y_pred": prediction.astype(np.int64),
            "y_prob": probability,
            "classes": classes,
            "eval_indices": annotation_indices.astype(np.int64),
        },
        fit_rows=int(train_annotation_valid.sum()),
        eval_rows=int(validation_annotation_valid.sum()),
        notes=f"{notes}; StandardScaler + train-only balanced SGD linear probe",
        training_regime="linear_probe",
    )

    biology, biology_valid = labels(context.validation, "annotation")
    batch, batch_valid = labels(context.validation, "integration")
    integration_valid = biology_valid & batch_valid
    integration_indices = np.flatnonzero(integration_valid)
    integration = TaskPrediction(
        task="integration",
        arrays={
            "embedding": validation_embedding[integration_valid].astype(np.float32),
            "biology_label": biology[integration_valid].astype(np.int64),
            "batch_label": batch[integration_valid].astype(np.int64),
            "eval_indices": integration_indices.astype(np.int64),
        },
        fit_rows=context.train.size,
        eval_rows=int(integration_valid.sum()),
        notes=f"{notes}; frozen cell embedding without label-aware correction",
        training_regime="zero_shot_embedding",
    )
    return [annotation, integration]


def _batched_embeddings(
    features: np.ndarray,
    encode: Callable[[Any], Any],
    *,
    batch_size: int,
    dimension: int,
) -> np.ndarray:
    output = np.empty((len(features), dimension), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        end = min(len(features), start + batch_size)
        output[start:end] = np.asarray(encode(features[start:end]), dtype=np.float32)
    return output


@register("geneformer_v1_10m")
class GeneformerV110MAdapter(BaselineAdapter):
    method_id = "geneformer_v1_10m"
    display_name = "Geneformer V1-10M"
    family = "foundation"
    tasks = ("annotation", "integration")

    def _root(self, context: AdapterContext) -> Path:
        return Path(context.options.get(
            "geneformer_root", context.data_root.parent / "foundation_models" / "geneformer"
        )).resolve()

    def availability(self, context: AdapterContext) -> Availability:
        root = self._root(context)
        required = [
            root / "Geneformer-V1-10M/config.json",
            root / "Geneformer-V1-10M/model.safetensors",
            root / "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
            root / "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        return Availability("external_asset_required", "; ".join(missing)) if missing else Availability("ready")

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "library": "transformers", "library_version": package_version("transformers")}

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import torch
        from safetensors.torch import load_file
        from transformers import BertConfig, BertModel

        root = self._root(context)
        dictionary_root = root / "geneformer/gene_dictionaries_30m"
        with (dictionary_root / "token_dictionary_gc30M.pkl").open("rb") as handle:
            token_dictionary = pickle.load(handle)
        with (dictionary_root / "gene_median_dictionary_gc30M.pkl").open("rb") as handle:
            median_dictionary = pickle.load(handle)
        ensembl, _ = _catalog(context.data_root)
        mapped = [
            (index, int(token_dictionary[gene]), float(median_dictionary[gene]))
            for index, gene in enumerate(ensembl)
            if gene in token_dictionary and gene in median_dictionary and float(median_dictionary[gene]) > 0
        ]
        if not mapped:
            raise RuntimeError("no panel genes map to the Geneformer V1 vocabulary")
        columns = np.asarray([item[0] for item in mapped], dtype=np.int64)
        token_ids = torch.tensor([item[1] for item in mapped], dtype=torch.long)
        medians = torch.tensor([item[2] for item in mapped], dtype=torch.float32)
        model_root = root / "Geneformer-V1-10M"
        config = BertConfig.from_pretrained(model_root, local_files_only=True)
        model = BertModel(config, add_pooling_layer=False)
        state = load_file(str(model_root / "model.safetensors"), device="cpu")
        state = {key.removeprefix("bert."): value for key, value in state.items() if key.startswith("bert.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        material_missing = [key for key in missing if not key.endswith("position_ids")]
        material_unexpected = [key for key in unexpected if not key.endswith("position_ids")]
        if material_missing or material_unexpected:
            raise RuntimeError(
                f"Geneformer weight mismatch: missing={material_missing[:5]}, unexpected={material_unexpected[:5]}"
            )
        device = torch.device(context.device)
        model.to(device).eval()
        token_ids = token_ids.to(device)
        medians = medians.to(device)
        torch.set_float32_matmul_precision("high")

        def encode(values: np.ndarray) -> np.ndarray:
            counts = torch.as_tensor(values[:, columns], device=device, dtype=torch.float32)
            normalized = counts / counts.sum(1, keepdim=True).clamp_min(1.0) * 10_000.0
            normalized = normalized / medians.unsqueeze(0)
            positive = normalized > 0
            scores = normalized.masked_fill(~positive, -torch.inf)
            order = torch.argsort(scores, dim=1, descending=True)
            input_ids = token_ids[order]
            nonzero = positive.sum(1)
            attention = torch.arange(len(mapped), device=device).unsqueeze(0) < nonzero.unsqueeze(1)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model(input_ids=input_ids, attention_mask=attention).last_hidden_state
                weight = attention.unsqueeze(-1).to(hidden.dtype)
                embedding = (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)
            return embedding.float().cpu().numpy()

        train_counts = panel_pseudocounts(context.train, context.data_root)
        validation_counts = panel_pseudocounts(context.validation, context.data_root)
        batch_size = int(context.options.get("geneformer_batch_size", 1024))
        train_embedding = _batched_embeddings(train_counts, encode, batch_size=batch_size, dimension=config.hidden_size)
        validation_embedding = _batched_embeddings(validation_counts, encode, batch_size=batch_size, dimension=config.hidden_size)
        weights = model_root / "model.safetensors"
        notes = (
            f"Geneformer V1-10M official weights; transformers {package_version('transformers')}; "
            f"{len(mapped)}/{len(ensembl)} panel genes mapped; median-normalized rank tokenization; "
            f"weights_sha256={_sha256(weights)}"
        )
        return _linear_probe_predictions(context, train_embedding, validation_embedding, method=self.method_id, notes=notes)


class _ScGPTAdapter(BaselineAdapter):
    family = "foundation"
    tasks = ("annotation", "integration")
    checkpoint_name: str

    def _root(self, context: AdapterContext) -> Path:
        option = f"{self.method_id}_root"
        return Path(context.options.get(
            option, context.data_root.parent / "foundation_models" / self.checkpoint_name
        )).resolve()

    def availability(self, context: AdapterContext) -> Availability:
        root = self._root(context)
        required = [root / "args.json", root / "vocab.json", root / "best_model.pt"]
        reference = context.data_root.parents[1] / "reference_methods/scBrain/src/models/scGPT/scGPT_annotation.py"
        missing = [str(path) for path in [*required, reference] if not path.is_file()]
        return Availability("external_asset_required", "; ".join(missing)) if missing else Availability("ready")

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "library": "scGPT reference snapshot", "library_version": "local-reference"}

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import torch

        reference_root = context.data_root.parents[1] / "reference_methods/scBrain"
        sys.path.insert(0, str(reference_root))
        try:
            from src.models.scGPT.scGPT_annotation import scGPTForAnnotation
        finally:
            if sys.path[0] == str(reference_root):
                sys.path.pop(0)

        checkpoint = self._root(context)
        vocab = json.loads((checkpoint / "vocab.json").read_text(encoding="utf-8"))
        _, symbols = _catalog(context.data_root)
        mapped = [(index, int(vocab[symbol])) for index, symbol in enumerate(symbols) if symbol in vocab]
        if not mapped:
            raise RuntimeError("no panel genes map to the scGPT vocabulary")
        columns = np.asarray([item[0] for item in mapped], dtype=np.int64)
        gene_ids = torch.tensor([item[1] for item in mapped], dtype=torch.long)
        model = scGPTForAnnotation(
            checkpoint_dir=str(checkpoint),
            gene_ids=gene_ids,
            output_num=[1, 1, 1],
            dropout=0.0,
            freeze_backbone=True,
        )
        device = torch.device(context.device)
        model.to(device).eval()
        torch.set_float32_matmul_precision("high")

        def encode(values: np.ndarray) -> np.ndarray:
            batch = torch.as_tensor(values[:, columns], device=device, dtype=torch.float32)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, embedding = model(batch)
            return embedding.float().cpu().numpy()

        train_counts = panel_pseudocounts(context.train, context.data_root)
        validation_counts = panel_pseudocounts(context.validation, context.data_root)
        batch_size = int(context.options.get("scgpt_batch_size", 256))
        dimension = int(json.loads((checkpoint / "args.json").read_text(encoding="utf-8"))["embsize"])
        train_embedding = _batched_embeddings(train_counts, encode, batch_size=batch_size, dimension=dimension)
        validation_embedding = _batched_embeddings(validation_counts, encode, batch_size=batch_size, dimension=dimension)
        notes = (
            f"{self.display_name} official perturblab checkpoint with local scBrain wrapper; "
            f"{len(mapped)}/{len(symbols)} panel genes mapped; frozen backbone; "
            f"weights_sha256={_sha256(checkpoint / 'best_model.pt')}"
        )
        return _linear_probe_predictions(context, train_embedding, validation_embedding, method=self.method_id, notes=notes)


@register("scgpt_brain")
class ScGPTBrainAdapter(_ScGPTAdapter):
    method_id = "scgpt_brain"
    display_name = "scGPT-brain"
    checkpoint_name = "scgpt-brain"


@register("scgpt_human")
class ScGPTHumanAdapter(_ScGPTAdapter):
    method_id = "scgpt_human"
    display_name = "scGPT-human"
    checkpoint_name = "scgpt-human"


@register("uce")
class UCEAdapter(BaselineAdapter):
    method_id = "uce"
    display_name = "UCE 4-layer"
    family = "foundation"
    tasks = ("annotation", "integration")

    def availability(self, context: AdapterContext) -> Availability:
        root = Path(context.options.get("uce_root", context.data_root.parent / "foundation_models/uce")).resolve()
        required = [
            root / "4layer_model.torch", root / "all_tokens.torch",
            root / "species_chrom.csv", root / "species_offsets.pkl",
            root / "model.py",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            return Availability(
                "external_asset_required",
                "official UCE assets/code are not present: " + "; ".join(missing),
            )
        return Availability("ready")

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        raise NotImplementedError("UCE execution is enabled only after the official asset bundle passes validation")


__all__ = ["GeneformerV110MAdapter", "ScGPTBrainAdapter", "ScGPTHumanAdapter", "UCEAdapter"]
