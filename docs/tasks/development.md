# Development

## Goal

Predict a target cell population from a source population and a development condition. The fixed dataset stores explicit source/target population pairs; the model predicts the target population distribution using the Flow Matching Transformer path.

## Run

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_niche_v3 \
  --task development --steps 20 --batch-size 16 \
  --development-cells 64 --validation-development-groups 0 \
  --checkpoint-policy best_weights --eval-interval 10 \
  --device cpu --precision fp32 --output results/development-smoke
```

`--validation-development-groups 0` evaluates all validation groups, as required by the V3 campaign. The linked [development leaderboard](../../benchmarks/nanobrain_v3_100k/LEADERBOARD.md#development) contains historical upstream results, not V3 results.

## Evaluation

The [V3](../../METRICS_V3.md) primary is empirical `wasserstein2` (W₂), lower is better, in the shared train-normalized gene panel. Compute exact uniform-mass optimal transport on fixed samples of at most 256 cells per population, take the square root of average squared Euclidean transport cost, and average population groups equally. Validation selects weights with the same metric.

Population mean MSE, variance error, MMD, gene correlations, and separately named Sinkhorn divergence remain auxiliary. Sinkhorn uses a different cost normalization and is not W₂. Time is a model condition. The original split is retained; no whole-timepoint holdout or new unseen-time forecasting claim was introduced.
