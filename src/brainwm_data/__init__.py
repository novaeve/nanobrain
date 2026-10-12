"""Data utilities for the indexed Brain World Model pretraining corpus."""

from .age import parse_age

__all__ = [
    "CorpusSpec",
    "IndexedH5ADBatchReader",
    "MultiCorpus",
    "MixedIndexedH5ADBatchReader",
    "MultiCorpusIndexedH5ADBatchReader",
    "MouseMolecularBatchReader",
    "MouseSpatialBatchReader",
    "UnifiedGeneIndexAdapter",
    "UnifiedGeneTableAdapter",
    "GlobalBenchmarkIndex",
    "GlobalBenchmarkReader",
    "GlobalMultiCorpusReader",
    "BenchmarkCorpusSpec",
    "GlobalCorpusSpec",
    "assign_group_splits",
    "build_global_benchmark_split",
    "derive_split_groups",
    "qualify_row_id",
    "qualify_sample_key",
    "qualify_source_id",
    "validate_global_benchmark_split",
    "GlobalTaskLabelIndex",
    "build_global_task_labels",
    "canonicalize_annotation_frame",
    "validate_global_task_labels",
    "ROW_PLAN_COLUMNS",
    "ROW_PLAN_FORMAT",
    "TRAINING_SPLITS",
    "ROW_SELECTION_FORMAT",
    "build_global_row_plan",
    "validate_global_row_plan",
    "iter_global_row_plan_selection",
    "select_global_row_plan_rows",
    "GlobalRowPlanTrainingReader",
    "parse_age",
]


def __getattr__(name):
    if name in {
        "CorpusSpec",
        "MultiCorpus",
        "MixedIndexedH5ADBatchReader",
        "MultiCorpusIndexedH5ADBatchReader",
    }:
        from .mixed_indexed_h5ad import (
            CorpusSpec,
            MixedIndexedH5ADBatchReader,
            MultiCorpus,
            MultiCorpusIndexedH5ADBatchReader,
        )
        return {
            "CorpusSpec": CorpusSpec,
            "MultiCorpus": MultiCorpus,
            "MixedIndexedH5ADBatchReader": MixedIndexedH5ADBatchReader,
            "MultiCorpusIndexedH5ADBatchReader": MultiCorpusIndexedH5ADBatchReader,
        }[name]
    if name == "IndexedH5ADBatchReader":
        from .indexed_h5ad import IndexedH5ADBatchReader
        return IndexedH5ADBatchReader
    if name == "MouseSpatialBatchReader":
        from .mouse_spatial_index import MouseSpatialBatchReader
        return MouseSpatialBatchReader
    if name == "MouseMolecularBatchReader":
        from .mouse_molecular_index import MouseMolecularBatchReader
        return MouseMolecularBatchReader
    if name in {"UnifiedGeneIndexAdapter", "UnifiedGeneTableAdapter"}:
        from .unified_gene_index import (
            UnifiedGeneIndexAdapter,
            UnifiedGeneTableAdapter,
        )
        return {
            "UnifiedGeneIndexAdapter": UnifiedGeneIndexAdapter,
            "UnifiedGeneTableAdapter": UnifiedGeneTableAdapter,
        }[name]
    if name in {
        "GlobalBenchmarkIndex",
        "GlobalBenchmarkReader",
        "GlobalMultiCorpusReader",
        "BenchmarkCorpusSpec",
        "GlobalCorpusSpec",
        "assign_group_splits",
        "build_global_benchmark_split",
        "derive_split_groups",
        "qualify_row_id",
        "qualify_sample_key",
        "qualify_source_id",
        "validate_global_benchmark_split",
    }:
        from .global_split import (
            BenchmarkCorpusSpec,
            GlobalCorpusSpec,
            GlobalBenchmarkIndex,
            GlobalBenchmarkReader,
            GlobalMultiCorpusReader,
            assign_group_splits,
            build_global_benchmark_split,
            derive_split_groups,
            qualify_row_id,
            qualify_sample_key,
            qualify_source_id,
            validate_global_benchmark_split,
        )
        return {
            "BenchmarkCorpusSpec": BenchmarkCorpusSpec,
            "GlobalCorpusSpec": GlobalCorpusSpec,
            "GlobalBenchmarkIndex": GlobalBenchmarkIndex,
            "GlobalBenchmarkReader": GlobalBenchmarkReader,
            "GlobalMultiCorpusReader": GlobalMultiCorpusReader,
            "assign_group_splits": assign_group_splits,
            "build_global_benchmark_split": build_global_benchmark_split,
            "derive_split_groups": derive_split_groups,
            "qualify_row_id": qualify_row_id,
            "qualify_sample_key": qualify_sample_key,
            "qualify_source_id": qualify_source_id,
            "validate_global_benchmark_split": validate_global_benchmark_split,
        }[name]
    if name in {
        "GlobalTaskLabelIndex",
        "build_global_task_labels",
        "canonicalize_annotation_frame",
        "validate_global_task_labels",
    }:
        from .task_labels import (
            GlobalTaskLabelIndex,
            build_global_task_labels,
            canonicalize_annotation_frame,
            validate_global_task_labels,
        )
        return {
            "GlobalTaskLabelIndex": GlobalTaskLabelIndex,
            "build_global_task_labels": build_global_task_labels,
            "canonicalize_annotation_frame": canonicalize_annotation_frame,
            "validate_global_task_labels": validate_global_task_labels,
        }[name]
    if name in {
        "ROW_PLAN_COLUMNS",
        "ROW_PLAN_FORMAT",
        "TRAINING_SPLITS",
        "ROW_SELECTION_FORMAT",
        "build_global_row_plan",
        "validate_global_row_plan",
        "iter_global_row_plan_selection",
        "select_global_row_plan_rows",
        "GlobalRowPlanTrainingReader",
    }:
        from .global_row_plan import (
            GlobalRowPlanTrainingReader,
            ROW_PLAN_COLUMNS,
            ROW_PLAN_FORMAT,
            ROW_SELECTION_FORMAT,
            TRAINING_SPLITS,
            build_global_row_plan,
            iter_global_row_plan_selection,
            select_global_row_plan_rows,
            validate_global_row_plan,
        )
        return {
            "GlobalRowPlanTrainingReader": GlobalRowPlanTrainingReader,
            "ROW_PLAN_COLUMNS": ROW_PLAN_COLUMNS,
            "ROW_PLAN_FORMAT": ROW_PLAN_FORMAT,
            "ROW_SELECTION_FORMAT": ROW_SELECTION_FORMAT,
            "TRAINING_SPLITS": TRAINING_SPLITS,
            "build_global_row_plan": build_global_row_plan,
            "iter_global_row_plan_selection": iter_global_row_plan_selection,
            "select_global_row_plan_rows": select_global_row_plan_rows,
            "validate_global_row_plan": validate_global_row_plan,
        }[name]
    raise AttributeError(name)
