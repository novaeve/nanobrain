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

The current [V3 protocol](../../METRICS_V3.md) reports seven established metrics using official `scib-metrics==0.6.0` implementations: Label ASW, Isolated-label ASW, KMeans NMI, KMeans ARI, within-type Batch ASW, iLISI, and Graph Connectivity. They appear individually in `standard_metrics.csv` and `STANDARD_METRICS.md`. BRAS and cell-type-conditional iLISI remain additional diagnostics. Sampling and seeds are fixed across candidates.

Validation-best checkpoint selection uses a predeclared project utility, retained under the compatibility key `scib_composite`: 60% biology and 40% batch removal over the selected subset. This is labeled `selection_only` and exported separately to `selection_scores.csv`; it is not an official scIB total or the scientific headline metric. The complete `benchmark_table.csv` identifies each metric's role. Formal leaderboards rank the standard metrics individually.

A single score can hide a trade-off between batch mixing and biological structure. Compare the component metrics and diagnostics, and retain each method's training budget and seed. The linked leaderboard is historical and cannot be compared directly with V3 scores.
