# Task 2: Spatial Gene Imputation

This task predicts held-out spatial genes from an observed spatial panel, using a scUniFlow backbone trained on matching scRNA/snRNA data.

Core files:

- `common.py`: reference and Visium loading, gene alignment, panel construction, metrics.
- `train_scuniflow_spatial_gene_imputation_head.py`: scUniFlow spatial head training.
- `eval_scuniflow_spatial_gene_imputation.py`: direct scUniFlow spatial imputation evaluation.
- `eval_scuniflow_spatial_gene_imputation_head.py`: task-head evaluation.
- `run_spage_mouse_brain.py`, `run_gimvi_mouse_brain.py`, `run_tangram_mouse_brain.py`, `run_seurat_mouse_brain.py`, `run_liger_mouse_brain.py`: baseline runners.

Use the repository-level entry point for the scUniFlow head:

```bash
bash scripts/run_task2_spatial.sh --help
```

Run baselines directly from this directory when needed, for example:

```bash
python src/scuniflow/tasks/task2_spatial_gene_imputation/run_spage_mouse_brain.py --help
```

Local datasets should be provided under `data/` or through explicit CLI paths.
