# Development

## Goal

Predict a target cell population from a source population and a development condition. The fixed dataset stores explicit source/target population pairs; the model predicts the target population distribution using the Flow Matching Transformer path.

## Run

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --task development --steps 20 --batch-size 16 \
  --development-cells 64 --validation-development-groups 8 \
  --device cpu --precision fp32 --output results/development-smoke
```

For larger runs, increase the step budget and validation group count after the smoke run succeeds. The fixed reference campaign used 2,000 training steps per seed. See the [development leaderboard](../../benchmarks/nanobrain_v3_100k/LEADERBOARD.md#development).

## Evaluation

Population MSE measures the difference between target and predicted mean expression; lower is better. Gene-wise Spearman correlation measures whether genes retain a similar cross-population ordering; higher is better. The table also includes MMD and variance error. Development time is a condition, not a target label to copy into the prediction.
