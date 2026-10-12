# Spatial Niche Prediction

## Goal

Predict the dataset's anatomy-derived niche label using expression and spatial context. These labels are an evaluation proxy attached to spatial rows; they are not presented as independently curated niche ground truth.

## Run

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --task spatial_niche --steps 20 --batch-size 16 \
  --device cpu --precision fp32 --output results/spatial-niche-smoke
```

The dataset only supplies neighbors for independently verified spatial sections. Rows without verified geometry use no cross-cell spatial neighbor. See the [spatial niche leaderboard](../../benchmarks/nanobrain_v3_100k/LEADERBOARD.md#spatial-niche).

## Evaluation

The fixed benchmark reports ARI, NMI, spatial autocorrelation, and enrichment metrics. ARI/NMI compare predicted groups with the provided anatomy-derived labels. Interpret them alongside the metadata policy and the number of eligible test rows; they do not establish general spatial biology performance.
