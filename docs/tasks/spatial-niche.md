# Spatial Niche Prediction

## Goal

Under [Metric V3](../../METRICS_V3.md), predict the surrounding cell-type proportions from center-cell expression at four fixed, train-calibrated radii. This is a composition regression task, with an additional neighbor-count head. Each independent model has a backbone and task heads.

The former anatomy-derived labels belong to the legacy region-proxy task. Its ARI/NMI scores and checkpoints do not evaluate the current task.

## Run

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_niche_v3 \
  --task spatial_niche --steps 20 --batch-size 16 \
  --checkpoint-policy best_weights --eval-interval 10 \
  --device cpu --precision fp32 --output results/spatial-niche-smoke
```

The V3 derivative is prepared on the experiment server; the original downloadable dataset does not contain these composition targets. Center rows and the 60k/20k/20k split are unchanged. Targets use full verified sections, exclude the center by identity, and retain an unknown-type bin. Valid centers number 14,872 / 1,612 / 1,293 in train/validation/test; individual radii can have fewer valid centers.

Coordinates, physical neighbors, and neighbor expression are hidden at both training and inference. The target labels are never model inputs. Counts are normalized by total neighbor count, rather than applying Softmax to counts.

## Evaluation

Primary: `composition_mae`, lower is better. Compute MAE over eligible centers and cell types at each radius, then average the four radii equally. Validation selects the best weights with this same metric. Auxiliary metrics include composition MSE/JSD, per-radius/type/section MAE, section-macro MAE, and neighbor-count MAE/R². The count head uses detached backbone embeddings so its loss does not dominate composition learning.

Use train-only global-mean and expression PCA+ridge controls. Domain-clustering adapters such as the existing native GraphST path require a matched composition probe before comparison; cluster IDs are not proportions. See the full [protocol and literature](../../METRICS_V3.md) for target provenance and coverage limits.
