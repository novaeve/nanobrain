# Task 3: scATAC Imputation

This task adapts scUniFlow to peak accessibility imputation for scATAC data.

Core files:

- `prepare_atac_h5ad.py`: convert 10x `.h5` or `.h5ad` inputs into TF-IDF `.h5ad` files.
- `generate_atac_coords.py`: generate grid-scaled peak routing coordinates from genomic positions.
- `data_atac.py`: scATAC loading, splits, datasets, and prior statistics.
- `train_atac.py`: scATAC training entry point with AUROC/AUPRC validation.
- `impute_atac.py` and `impute_and_eval_atac.py`: checkpoint inference and evaluation orchestration.
- `benchmark/`: baseline conversion and evaluation helpers.

Use the repository-level entry point:

```bash
bash scripts/run_task3_atac.sh configs/task3_atac/config_atac.yaml --help
```

Configuration templates live under `configs/task3_atac/`. Local input files and generated peak coordinates should live under `data/task3_atac/` or be passed explicitly through CLI arguments.
