"""Integration adapters backed by maintained third-party libraries."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from .base import AdapterContext, Availability, BaselineAdapter, TaskPrediction
from .common import array, labels, log_expression, package_version, panel_pseudocounts, train_pca
from .registry import register


def _integration_arrays(context: AdapterContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    biology, biology_valid = labels(context.validation, "annotation")
    batch, batch_valid = labels(context.validation, "integration")
    valid = biology_valid & batch_valid
    return biology, batch, valid


def _bbknn_exact_backend(neighbors_within_batch: int) -> str:
    """Choose an exact backend that preserves the neighbor matrix rank."""

    # scipy.cKDTree.query squeezes its result to (n,) when k=1, while BBKNN
    # 1.6 expects (n, 1). sklearn's exact KDTree keeps that dimension.
    return "KDTree" if int(neighbors_within_batch) == 1 else "cKDTree"


def _prediction(context: AdapterContext, embedding: np.ndarray, *, fit_rows: int, notes: str, regime: str) -> TaskPrediction:
    biology, batch, valid = _integration_arrays(context)
    indices = np.flatnonzero(valid)
    return TaskPrediction(
        task="integration",
        arrays={
            "embedding": np.asarray(embedding[indices], dtype=np.float32),
            "biology_label": biology[indices].astype(np.int64),
            "batch_label": batch[indices].astype(np.int64),
            "eval_indices": indices.astype(np.int64),
        },
        fit_rows=int(fit_rows),
        eval_rows=len(indices),
        notes=notes,
        training_regime=regime,
    )


class _ImportChecked(BaselineAdapter):
    distribution: str
    module: str

    def availability(self, context: AdapterContext) -> Availability:
        if importlib.util.find_spec(self.module) is None:
            return Availability("dependency_missing", f"install {self.distribution}")
        return Availability("ready")

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "library": self.distribution, "library_version": package_version(self.distribution)}


@register("harmony")
class HarmonyAdapter(_ImportChecked):
    method_id = "harmony"
    display_name = "Harmony"
    family = "integration"
    tasks = ("integration",)
    distribution = "harmonypy"
    module = "harmonypy"

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import harmonypy
        import pandas as pd

        train_embedding, validation_embedding = train_pca(
            context.train, context.validation, seed=context.seed, dimensions=32
        )
        train_batch, train_valid = labels(context.train, "integration")
        validation_batch, validation_valid = labels(context.validation, "integration")
        combined = np.concatenate((train_embedding[train_valid], validation_embedding[validation_valid]), axis=0)
        batches = np.concatenate((train_batch[train_valid], validation_batch[validation_valid])).astype(str)
        frame = pd.DataFrame({"batch": batches})
        harmony_result = harmonypy.run_harmony(
            combined,
            frame,
            "batch",
            max_iter_harmony=int(context.options.get("harmony_epochs", 10)),
            max_iter_kmeans=10,
            random_state=int(context.seed),
            ncores=max(1, int(context.threads)),
            verbose=False,
        )
        corrected = np.asarray(harmony_result.Z_corr, dtype=np.float32)
        if corrected.shape != combined.shape and corrected.T.shape == combined.shape:
            corrected = corrected.T
        if corrected.shape != combined.shape:
            raise RuntimeError(
                f"unexpected Harmony output shape {corrected.shape}; expected {combined.shape}"
            )
        validation_corrected = validation_embedding.copy()
        validation_corrected[validation_valid] = corrected[int(train_valid.sum()):]
        notes = (
            f"harmonypy {package_version('harmonypy')}; train-fitted PCA; Harmony fit on train+validation "
            "features and batch IDs without validation biology labels"
        )
        return [_prediction(context, validation_corrected, fit_rows=len(combined), notes=notes, regime="transductive_fit")]


@register("scanorama")
class ScanoramaAdapter(_ImportChecked):
    method_id = "scanorama"
    display_name = "Scanorama"
    family = "integration"
    tasks = ("integration",)
    distribution = "scanorama"
    module = "scanorama"

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import scanorama

        expression = log_expression(context.validation, context.data_root)
        _, batch, valid = _integration_arrays(context)
        unique_batches = np.unique(batch[valid])
        groups = [np.flatnonzero(valid & (batch == value)) for value in unique_batches]
        genes = [f"panel_{index}" for index in range(expression.shape[1])]
        datasets = [expression[index] for index in groups]
        integrated, _ = scanorama.integrate(
            datasets,
            [genes] * len(datasets),
            dimred=32,
            approx=True,
            knn=min(20, max(2, min(len(index) for index in groups) - 1)),
            seed=int(context.seed),
            verbose=0,
        )
        embedding = np.zeros((context.validation.size, 32), dtype=np.float32)
        for index, values in zip(groups, integrated):
            width = min(embedding.shape[1], values.shape[1])
            embedding[index, :width] = values[:, :width]
        notes = (
            f"scanorama {package_version('scanorama')}; official integrate API on validation expression grouped "
            "by batch; no validation biology labels used"
        )
        return [_prediction(context, embedding, fit_rows=int(valid.sum()), notes=notes, regime="transductive_fit")]


@register("bbknn")
class BBKNNAdapter(_ImportChecked):
    method_id = "bbknn"
    display_name = "BBKNN"
    family = "integration"
    tasks = ("integration",)
    distribution = "bbknn"
    module = "bbknn"

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import anndata as ad
        import bbknn
        import scanpy as sc

        _, validation_embedding = train_pca(context.train, context.validation, seed=context.seed, dimensions=32)
        _, batch, valid = _integration_arrays(context)
        indices = np.flatnonzero(valid)
        adata = ad.AnnData(X=np.zeros((len(indices), 1), dtype=np.float32))
        adata.obsm["X_pca"] = validation_embedding[indices]
        adata.obs["batch"] = batch[indices].astype(str)
        min_batch = int(np.min(np.unique(batch[indices], return_counts=True)[1]))
        within = max(1, min(3, min_batch - 1))
        bbknn.bbknn(
            adata,
            batch_key="batch",
            use_rep="X_pca",
            neighbors_within_batch=within,
            n_pcs=validation_embedding.shape[1],
            computation=_bbknn_exact_backend(within),
            metric="euclidean",
        )
        sc.tl.diffmap(adata, n_comps=32, random_state=int(context.seed))
        local = np.asarray(adata.obsm["X_diffmap"], dtype=np.float32)
        embedding = np.zeros((context.validation.size, local.shape[1]), dtype=np.float32)
        embedding[indices] = local
        notes = (
            f"bbknn {package_version('bbknn')} graph plus scanpy {package_version('scanpy')} diffusion map; "
            "train-fitted PCA input; validation batch IDs only"
        )
        return [_prediction(context, embedding, fit_rows=len(indices), notes=notes, regime="transductive_fit")]


@register("scvi_scanvi")
class ScviScanviAdapter(_ImportChecked):
    """Shared scVI backbone producing integration and scANVI annotation outputs."""

    method_id = "scvi_scanvi"
    display_name = "scVI/scANVI"
    family = "deep"
    tasks = ("integration", "annotation")
    distribution = "scvi-tools"
    module = "scvi"

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import anndata as ad
        import pandas as pd
        import scvi

        scvi.settings.seed = int(context.seed)
        train_counts = panel_pseudocounts(context.train, context.data_root)
        validation_counts = panel_pseudocounts(context.validation, context.data_root)
        counts = np.concatenate((train_counts, validation_counts), axis=0)
        train_annotation, train_annotation_valid = labels(context.train, "annotation")
        validation_annotation, validation_annotation_valid = labels(context.validation, "annotation")
        train_batch, _ = labels(context.train, "integration")
        validation_batch, _ = labels(context.validation, "integration")
        annotation = np.concatenate(
            (
                np.where(train_annotation_valid, np.char.add("class_", train_annotation.astype(str)), "Unknown"),
                np.full(context.validation.size, "Unknown", dtype="<U32"),
            )
        )
        obs = pd.DataFrame(
            {
                "batch": np.concatenate((train_batch, validation_batch)).astype(str),
                "annotation": annotation,
            }
        )
        adata = ad.AnnData(X=counts, obs=obs)
        adata.var_names = [f"panel_{index}" for index in range(counts.shape[1])]
        scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
        vae = scvi.model.SCVI(
            adata,
            n_hidden=int(context.options.get("scvi_hidden", 128)),
            n_latent=int(context.options.get("scvi_latent", 32)),
            n_layers=2,
            gene_likelihood="nb",
        )
        accelerator = "gpu" if context.device.startswith("cuda") else "cpu"
        devices: Any = 1
        vae.train(
            max_epochs=int(context.options.get("scvi_epochs", 20)),
            batch_size=int(context.options.get("scvi_batch_size", 2048)),
            accelerator=accelerator,
            devices=devices,
            early_stopping=True,
            enable_progress_bar=False,
            check_val_every_n_epoch=5,
        )
        latent = np.asarray(vae.get_latent_representation(), dtype=np.float32)
        validation_latent = latent[context.train.size :]
        integration = _prediction(
            context,
            validation_latent,
            fit_rows=len(adata),
            notes=(
                f"scvi-tools {package_version('scvi-tools')}; official SCVI NB model; inverse-log1p rounded "
                "panel pseudo-count adaptation; train+validation features and batch IDs, no validation labels"
            ),
            regime="transductive_fit",
        )

        scanvi = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata,
            labels_key="annotation",
            unlabeled_category="Unknown",
        )
        scanvi.train(
            max_epochs=int(context.options.get("scanvi_epochs", 15)),
            batch_size=int(context.options.get("scvi_batch_size", 2048)),
            accelerator=accelerator,
            devices=devices,
            early_stopping=True,
            enable_progress_bar=False,
            check_val_every_n_epoch=5,
        )
        soft = scanvi.predict(adata[context.train.size :], soft=True)
        probabilities_raw = np.asarray(soft, dtype=np.float32)
        columns = list(getattr(soft, "columns", []))
        classes = np.arange(
            int(max(train_annotation[train_annotation_valid].max(), validation_annotation[validation_annotation_valid].max())) + 1,
            dtype=np.int64,
        )
        probabilities = np.zeros((context.validation.size, len(classes)), dtype=np.float32)
        for column_index, column in enumerate(columns):
            text = str(column)
            if text.startswith("class_"):
                class_index = int(text.removeprefix("class_"))
                if 0 <= class_index < len(classes):
                    probabilities[:, class_index] = probabilities_raw[:, column_index]
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1.0e-12)
        eval_indices = np.flatnonzero(validation_annotation_valid)
        predicted = classes[np.argmax(probabilities[eval_indices], axis=1)]
        annotation_prediction = TaskPrediction(
            task="annotation",
            arrays={
                "y_true": validation_annotation[eval_indices].astype(np.int64),
                "y_pred": predicted.astype(np.int64),
                "y_prob": probabilities[eval_indices],
                "classes": classes,
                "eval_indices": eval_indices.astype(np.int64),
            },
            fit_rows=int(train_annotation_valid.sum()),
            eval_rows=len(eval_indices),
            notes=(
                f"scvi-tools {package_version('scvi-tools')}; official SCANVI semi-supervised head; validation "
                "features marked Unknown; inverse-log1p rounded panel pseudo-count adaptation"
            ),
            training_regime="semi_supervised_transductive",
        )
        return [integration, annotation_prediction]


__all__ = ["BBKNNAdapter", "HarmonyAdapter", "ScanoramaAdapter", "ScviScanviAdapter"]
