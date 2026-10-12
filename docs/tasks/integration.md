# Integration

## Goal

Learn a cell representation that preserves observed expression while reducing batch signal. The integration objective uses batch labels for its adversarial branch and an expression reconstruction term; it does not use annotation, niche, or development labels as supervision.

## Run

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --task integration --steps 20 --batch-size 16 \
  --device cpu --precision fp32 --output results/integration-smoke
```

The smoke run checks the task path. The fixed benchmark used three nanoBrain seeds at 2,000 optimizer steps per seed. See the [integration leaderboard](../../benchmarks/nanobrain_v3_100k/LEADERBOARD.md#integration).

## Read the result

Compare the integration composite with biology-conservation and batch-removal component metrics in `overall_metrics.csv`. A single integrated score can hide a trade-off between mixing batches and preserving cell biology. Method budgets differ, so retain each run's configuration and seed when comparing a new architecture.
