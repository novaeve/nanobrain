# Task 1: Read-Depth Enhancement

This directory contains the scUniFlow-specific read-depth enhancement head. It uses a pretrained scUniFlow backbone and trains a task head to map low-depth counts to high-depth counts.

Core files:

- `common.py`: split construction, downsampling, metrics, and shared model helpers.
- `train_scuniflow_read_depth_enhancement.py`: task-head training.
- `eval_scuniflow_read_depth_enhancement.py`: clustering and recovery evaluation.
- `eval_scuniflow_pcc.py`: Pearson-correlation diagnostics.

Use the repository-level entry point:

```bash
bash scripts/run_task1_read_depth.sh --help
```

Typical inputs are a pretrained backbone checkpoint, its config, and a local `.h5ad` file supplied through CLI arguments.
