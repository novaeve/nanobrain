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

The `annotation` label and eligibility mask are provided for each split. Unknown or train-unseen labels are not treated as ordinary training classes. Review accuracy together with macro F1 because class counts are imbalanced; the benchmark reports both alongside the remaining standard classification metrics.

Training output includes a JSON receipt with the seed, task, data manifest, validation diagnostics, checkpoint, and elapsed time. New runs should compare validation results only. Published test scores are historical descriptive results.
