# Annotation

## Goal

Predict the cell-type label attached to each eligible cell. The task head is trained from labels in the train split; validation selects the checkpoint. The test split must remain closed during tuning.

## Run

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --task annotation --steps 20 --batch-size 16 \
  --device cpu --precision fp32 --output results/annotation-smoke
```

Use the short run to verify data loading and the training loop. For GPU training, use a compatible PyTorch CUDA build and set `--device cuda`. The reference benchmark used 2,000 steps per seed; its scores are in the [Leaderboard](../../benchmarks/nanobrain_v3_100k/LEADERBOARD.md).

## Data and evaluation

The `annotation` label and eligibility mask are provided for each split. Unknown or train-unseen labels are not treated as ordinary training classes. Under [V3](../../METRICS_V3.md), Macro-F1 is the primary metric for both validation-best weight selection and final scoring: each ground-truth-supported evaluation class receives equal weight. Accuracy, Balanced Accuracy, and MCC remain diagnostics. This is a closed-set task, without a new unknown-type rejection benchmark.

Training output includes a JSON receipt with the seed, task, data manifest, validation diagnostics, checkpoint, and elapsed time. New runs should compare validation results only. Published test scores are historical descriptive results.
