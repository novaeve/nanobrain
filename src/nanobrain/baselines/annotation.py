"""Annotation adapters backed by maintained classifier libraries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .base import AdapterContext, Availability, BaselineAdapter, TaskPrediction
from .common import array, labels, log_expression, package_version
from .registry import register


def _annotation_data(context: AdapterContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_y_all, train_valid = labels(context.train, "annotation")
    validation_y_all, validation_valid = labels(context.validation, "annotation")
    classes = np.arange(
        int(max(train_y_all[train_valid].max(), validation_y_all[validation_valid].max())) + 1,
        dtype=np.int64,
    )
    return (
        train_y_all[train_valid].astype(np.int64),
        validation_y_all[validation_valid].astype(np.int64),
        train_valid,
        validation_valid,
        classes,
    )


def _task_prediction(
    context: AdapterContext,
    *,
    prediction: np.ndarray,
    probability: np.ndarray,
    fit_rows: int,
    notes: str,
    regime: str,
) -> TaskPrediction:
    _, validation_y, _, validation_valid, classes = _annotation_data(context)
    indices = np.flatnonzero(validation_valid)
    return TaskPrediction(
        task="annotation",
        arrays={
            "y_true": validation_y,
            "y_pred": np.asarray(prediction, dtype=np.int64),
            "y_prob": np.asarray(probability, dtype=np.float32),
            "classes": classes,
            "eval_indices": indices.astype(np.int64),
        },
        fit_rows=int(fit_rows),
        eval_rows=len(indices),
        notes=notes,
        training_regime=regime,
    )


def _panel_gene_symbols(data_root: Path) -> list[str]:
    import pandas as pd

    panel = np.load(data_root / "panel_gene_index.npy", allow_pickle=False)
    catalog = pd.read_parquet(data_root.parent / "human_v3_genes.parquet")
    indexed = catalog.set_index("gene_index")
    symbols: list[str] = []
    seen: dict[str, int] = {}
    for gene_index in panel:
        symbol = str(indexed.loc[int(gene_index), "gene_symbol"] or f"gene_{int(gene_index)}")
        occurrence = seen.get(symbol, 0)
        seen[symbol] = occurrence + 1
        symbols.append(symbol if occurrence == 0 else f"{symbol}__{occurrence}")
    return symbols


def _celltypist_expression(context: AdapterContext, dataset: Any) -> np.ndarray:
    log_values = log_expression(dataset, context.data_root)
    values = np.expm1(log_values)
    values *= 10_000.0 / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(values).astype(np.float32)


class _ImportChecked(BaselineAdapter):
    distribution: str
    module: str

    def availability(self, context: AdapterContext) -> Availability:
        if importlib.util.find_spec(self.module) is None:
            return Availability("dependency_missing", f"install {self.distribution}")
        return Availability("ready")

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "library": self.distribution, "library_version": package_version(self.distribution)}


@register("xgboost")
class XGBoostAdapter(_ImportChecked):
    method_id = "xgboost"
    display_name = "XGBoost"
    family = "classical"
    tasks = ("annotation",)
    distribution = "xgboost"
    module = "xgboost"

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        from xgboost import XGBClassifier

        train_y, _, train_valid, validation_valid, classes = _annotation_data(context)
        train_x = array(context.train.features, np.float32)[train_valid]
        validation_x = array(context.validation.features, np.float32)[validation_valid]
        learned_classes = np.unique(train_y)
        encode = {int(value): index for index, value in enumerate(learned_classes)}
        encoded_y = np.asarray([encode[int(value)] for value in train_y], dtype=np.int64)
        device = context.device if context.device.startswith("cuda") else "cpu"
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(learned_classes),
            n_estimators=int(context.options.get("xgboost_estimators", 200)),
            max_depth=int(context.options.get("xgboost_depth", 8)),
            learning_rate=float(context.options.get("xgboost_learning_rate", 0.08)),
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            tree_method="hist",
            device=device,
            n_jobs=max(1, int(context.threads)),
            random_state=int(context.seed),
            eval_metric="mlogloss",
        )
        model.fit(train_x, encoded_y, verbose=False)
        learned_probability = np.asarray(model.predict_proba(validation_x), dtype=np.float32)
        probability = np.zeros((len(validation_x), len(classes)), dtype=np.float32)
        probability[:, learned_classes] = learned_probability
        prediction = classes[np.argmax(probability, axis=1)]
        notes = (
            f"xgboost {package_version('xgboost')}; GPU histogram trees; {model.n_estimators} estimators; "
            "train rows only"
        )
        return [_task_prediction(
            context,
            prediction=prediction,
            probability=probability,
            fit_rows=len(train_y),
            notes=notes,
            regime="fit",
        )]


@register("celltypist")
class CellTypistAdapter(_ImportChecked):
    method_id = "celltypist"
    display_name = "CellTypist"
    family = "deep"
    tasks = ("annotation",)
    distribution = "celltypist"
    module = "celltypist"

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import anndata as ad
        import celltypist

        train_y, _, train_valid, validation_valid, classes = _annotation_data(context)
        genes = _panel_gene_symbols(context.data_root)
        train_x = _celltypist_expression(context, context.train)[train_valid]
        validation_x = _celltypist_expression(context, context.validation)[validation_valid]
        text_labels = np.char.add("class_", train_y.astype(str))
        model = celltypist.train(
            X=train_x,
            labels=text_labels,
            genes=genes,
            check_expression=False,
            use_SGD=True,
            mini_batch=True,
            batch_size=int(context.options.get("celltypist_batch_size", 4096)),
            epochs=int(context.options.get("celltypist_epochs", 30)),
            balance_cell_type=True,
            n_jobs=max(1, int(context.threads)),
            random_state=int(context.seed),
        )
        validation = ad.AnnData(X=validation_x)
        validation.var_names = genes
        result = celltypist.annotate(validation, model=model, majority_voting=False)
        matrix = result.probability_matrix
        probability = np.zeros((len(validation_x), len(classes)), dtype=np.float32)
        for column_index, column in enumerate(matrix.columns):
            text = str(column)
            if text.startswith("class_"):
                class_index = int(text.removeprefix("class_"))
                if 0 <= class_index < len(classes):
                    probability[:, class_index] = np.asarray(matrix.iloc[:, column_index], dtype=np.float32)
        probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1.0e-12)
        prediction = classes[np.argmax(probability, axis=1)]
        notes = (
            f"celltypist {package_version('celltypist')}; official SGD mini-batch trainer; panel expression "
            "renormalized to 10k/log1p; train rows only"
        )
        return [_task_prediction(
            context,
            prediction=prediction,
            probability=probability,
            fit_rows=len(train_y),
            notes=notes,
            regime="fine_tune",
        )]


__all__ = ["CellTypistAdapter", "XGBoostAdapter"]
