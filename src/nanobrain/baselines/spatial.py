"""Spatial representation adapters and explicit compatibility checks."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from .base import AdapterContext, Availability, BaselineAdapter, TaskPrediction
from .common import array, labels, log_expression, package_version
from .registry import register


def _spatial_rows(context: AdapterContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label, label_valid = labels(context.validation, "niche")
    coordinates = array(context.validation.coordinates, np.float32)
    valid = label_valid & np.isfinite(coordinates[:, :2]).all(axis=1)
    return label, coordinates, valid


@register("graphst")
class GraphSTAdapter(BaselineAdapter):
    method_id = "graphst"
    display_name = "GraphST"
    family = "deep"
    tasks = ("spatial_niche",)

    def availability(self, context: AdapterContext) -> Availability:
        return Availability("ready") if importlib.util.find_spec("GraphST") else Availability("dependency_missing", "install GraphST")

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "library": "GraphST", "library_version": package_version("GraphST")}

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        import anndata as ad
        import torch
        from GraphST.GraphST import GraphST
        from sklearn.cluster import MiniBatchKMeans

        label, coordinates, valid = _spatial_rows(context)
        indices = np.flatnonzero(valid)
        expression = log_expression(context.validation, context.data_root)[indices]
        adata = ad.AnnData(X=expression)
        adata.var_names = [f"panel_{index}" for index in range(expression.shape[1])]
        adata.var["highly_variable"] = True
        adata.obsm["spatial"] = coordinates[indices, :2]
        model = GraphST(
            adata,
            device=torch.device(context.device),
            epochs=int(context.options.get("graphst_epochs", 100)),
            dim_output=int(context.options.get("graphst_dimensions", 32)),
            random_seed=int(context.seed),
            datatype="Stereo",
        )
        fitted = model.train()
        embedding = np.asarray(fitted.obsm["emb"], dtype=np.float32)
        train_label, train_valid = labels(context.train, "niche")
        n_clusters = max(2, min(int(np.unique(train_label[train_valid]).size), 64))
        clusterer = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=2048,
            n_init=10,
            max_iter=200,
            random_state=int(context.seed),
        )
        clusters = clusterer.fit_predict(embedding)
        notes = (
            f"GraphST {package_version('GraphST')} official sparse KNN path; "
            f"{int(context.options.get('graphst_epochs', 100))} unsupervised epochs; "
            f"cluster count={n_clusters} fixed from train label cardinality"
        )
        return [TaskPrediction(
            task="spatial_niche",
            arrays={
                "embedding": embedding,
                "coordinates": coordinates[indices, :2].astype(np.float32),
                "labels": label[indices].astype(np.int64),
                "clusters": clusters.astype(np.int64),
                "eval_indices": indices.astype(np.int64),
            },
            fit_rows=len(indices),
            eval_rows=len(indices),
            notes=notes,
            training_regime="transductive_unsupervised",
        )]


@register("spagcn")
class SpaGCNAdapter(BaselineAdapter):
    method_id = "spagcn"
    display_name = "SpaGCN"
    family = "deep"
    tasks = ("spatial_niche",)

    def availability(self, context: AdapterContext) -> Availability:
        if importlib.util.find_spec("SpaGCN") is None:
            return Availability("dependency_missing", "install SpaGCN")
        _, _, valid = _spatial_rows(context)
        rows = int(valid.sum())
        dense_bytes = rows * rows * 8
        limit = int(context.options.get("spagcn_dense_limit_bytes", 512 * 1024 * 1024))
        if dense_bytes > limit:
            return Availability(
                "resource_contract_incompatible",
                f"SpaGCN 1.2.7 constructs a dense O(N^2) adjacency ({dense_bytes} bytes for {rows} rows), "
                f"above the configured {limit}-byte adapter limit",
            )
        return Availability("ready")

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        raise NotImplementedError("SpaGCN is run only when its dense-adjacency resource check passes")


@register("stagate")
class STAGATEAdapter(BaselineAdapter):
    method_id = "stagate"
    display_name = "STAGATE"
    family = "deep"
    tasks = ("spatial_niche",)

    def availability(self, context: AdapterContext) -> Availability:
        return Availability(
            "dependency_missing",
            "no maintained STAGATE package is installed in the unified Python 3.12/PyTorch 2.10 environment",
        )

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        raise NotImplementedError


@register("scuniflow")
class ScUniFlowAdapter(BaselineAdapter):
    method_id = "scuniflow"
    display_name = "scUniFlow"
    family = "deep"
    tasks = ("spatial_completion",)

    def availability(self, context: AdapterContext) -> Availability:
        checkpoint = context.options.get("scuniflow_checkpoint")
        if not checkpoint:
            return Availability(
                "external_asset_required",
                "local scUniFlow source is present, but no pretrained backbone checkpoint is bundled; "
                "training its published spatial head without that backbone would be a different method",
            )
        return Availability("ready")

    def run(self, context: AdapterContext) -> list[TaskPrediction]:
        raise NotImplementedError("scUniFlow adapter requires a validated checkpoint-specific loader")


__all__ = ["GraphSTAdapter", "ScUniFlowAdapter", "SpaGCNAdapter", "STAGATEAdapter"]
