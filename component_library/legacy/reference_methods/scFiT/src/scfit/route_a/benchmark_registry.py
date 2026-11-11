from typing import Dict, List

from scfit.paths import project_root

ROOT = project_root()
BENCHMARK_DATA_ROOT = ROOT / "data" / "raw" / "benchmarks"
BENCHMARK_PREPROCESSED_ROOT = ROOT / "data" / "processed" / "benchmarks"
BENCHMARK_PROTOCOL_ROOT = ROOT / "configs" / "protocols" / "benchmark"


def _seq(start: int, stop: int) -> List[int]:
    return list(range(start, stop))


BENCHMARK_DATASETS: Dict[str, Dict] = {
    "wot": {
        "dataset_key": "wot",
        "display_name": "Schiebinger",
        "data_dir": BENCHMARK_DATA_ROOT / "Schiebinger2019" / "reduce_processed",
        "expr_suffix": "norm_data-hvg.csv",
        "gene_suffix": "var_genes_list.csv",
        "meta_suffix": "meta_data.csv",
        "time_label_col": "day",
        "obs_name_col": "index",
        "normalization_mode": "provided_log1p_hvg_clip_0_10",
        "split_defs": {
            "three_interpolation": {
                "protocol_name": "wot_three_interpolation_v1",
                "task_type": "three_interpolation",
                "direction_mode": "bidirectional",
                "source_policy": "bidirectional_linear",
                "train_idx": [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18],
                "test_idx": [5, 10, 15],
            },
            "three_forecasting": {
                "protocol_name": "wot_three_forecasting_v1",
                "task_type": "three_forecasting",
                "direction_mode": "forward_only",
                "source_policy": "previous_observed",
                "train_idx": _seq(0, 16),
                "test_idx": [16, 17, 18],
            },
            "remove_recovery": {
                "protocol_name": "wot_remove_recovery_v1",
                "task_type": "remove_recovery",
                "direction_mode": "bidirectional",
                "source_policy": "hybrid_observed",
                "train_idx": [0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14],
                "test_idx": [5, 7, 9, 11, 15, 16, 17, 18],
            },
        },
    },
    "drosophila": {
        "dataset_key": "drosophila",
        "display_name": "drosophila_embryonic",
        "data_dir": BENCHMARK_DATA_ROOT / "drosophila_embryonic" / "processed",
        "expr_suffix": "count_data-hvg.csv",
        "gene_suffix": "var_genes_list.csv",
        "meta_file": BENCHMARK_DATA_ROOT / "drosophila_embryonic" / "processed" / "subsample_meta_data.csv",
        "time_label_col": "time",
        "obs_name_col": "Unnamed: 0",
        "time_numeric_candidates": ["NNv2_age", "NNv1_age", "lasso_age"],
        "donor_id_col": "orig.ident",
        "normalization_mode": "librarysize_10000_log1p_clip_0_10_from_count_hvg",
        "split_defs": {
            "three_interpolation": {
                "protocol_name": "drosophila_three_interpolation_v1",
                "task_type": "three_interpolation",
                "direction_mode": "bidirectional",
                "source_policy": "bidirectional_linear",
                "train_idx": [0, 1, 2, 3, 5, 7, 9, 10],
                "test_idx": [4, 6, 8],
            },
            "three_forecasting": {
                "protocol_name": "drosophila_three_forecasting_v1",
                "task_type": "three_forecasting",
                "direction_mode": "forward_only",
                "source_policy": "previous_observed",
                "train_idx": [0, 1, 2, 3, 4, 5, 6, 7],
                "test_idx": [8, 9, 10],
            },
            "remove_recovery": {
                "protocol_name": "drosophila_remove_recovery_v1",
                "task_type": "remove_recovery",
                "direction_mode": "bidirectional",
                "source_policy": "hybrid_observed",
                "train_idx": [0, 1, 3, 5, 7],
                "test_idx": [2, 4, 6, 8, 9, 10],
            },
        },
    },
    "zebrafish": {
        "dataset_key": "zebrafish",
        "display_name": "zebrafish_embryonic",
        "data_dir": BENCHMARK_DATA_ROOT / "zebrafish_embryonic" / "new_processed",
        "expr_suffix": "count_data-hvg.csv",
        "gene_suffix": "var_genes_list.csv",
        "meta_file": BENCHMARK_DATA_ROOT / "zebrafish_embryonic" / "new_processed" / "meta_data.csv",
        "time_label_col": "stage.nice",
        "time_numeric_col": "HPF",
        "obs_name_col": "CELL",
        "donor_id_col": "SAMPLE",
        "normalization_mode": "librarysize_10000_log1p_clip_0_10_from_count_hvg",
        "split_defs": {
            "three_interpolation": {
                "protocol_name": "zebrafish_three_interpolation_v1",
                "task_type": "three_interpolation",
                "direction_mode": "bidirectional",
                "source_policy": "bidirectional_linear",
                "train_idx": [0, 1, 2, 3, 5, 7, 9, 10, 11],
                "test_idx": [4, 6, 8],
            },
            "two_forecasting": {
                "protocol_name": "zebrafish_two_forecasting_v1",
                "task_type": "two_forecasting",
                "direction_mode": "forward_only",
                "source_policy": "previous_observed",
                "train_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "test_idx": [10, 11],
            },
            "remove_recovery": {
                "protocol_name": "zebrafish_remove_recovery_v1",
                "task_type": "remove_recovery",
                "direction_mode": "bidirectional",
                "source_policy": "hybrid_observed",
                "train_idx": [0, 1, 3, 5, 7, 9],
                "test_idx": [2, 4, 6, 8, 10, 11],
            },
        },
    },
}


def list_supported_benchmark_tasks():
    rows = []
    for dataset_key, dataset_spec in BENCHMARK_DATASETS.items():
        for split_key, split_def in dataset_spec["split_defs"].items():
            rows.append(
                {
                    "dataset": dataset_key,
                    "split": split_key,
                    "protocol_name": split_def["protocol_name"],
                }
            )
    return rows


def get_benchmark_dataset_spec(dataset_key: str) -> Dict:
    dataset_key = str(dataset_key).strip()
    if dataset_key not in BENCHMARK_DATASETS:
        raise KeyError(f"unknown benchmark dataset: {dataset_key}")
    return BENCHMARK_DATASETS[dataset_key]


def get_benchmark_task_spec(dataset_key: str, split_key: str) -> Dict:
    dataset_spec = get_benchmark_dataset_spec(dataset_key)
    split_key = str(split_key).strip()
    if split_key not in dataset_spec["split_defs"]:
        raise KeyError(f"dataset={dataset_key} does not support split={split_key}")
    merged = dict(dataset_spec)
    merged.update(dataset_spec["split_defs"][split_key])
    merged["split_key"] = split_key
    return merged
