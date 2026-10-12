# nanoBrain 100k Leaderboard

> [!IMPORTANT]
> These are descriptive results from the existing frozen test evaluation. The test split has been opened; it is not a new blind holdout. Methods use different published training regimes and run counts.

Dataset: `nanobrain_v3_100k` · split: `BRAINWM-HUMAN-GLOBAL-6-2-2-V3` · rows: 60k/20k/20k · source: frozen benchmark long table.

Each table ranks methods by the primary metric for that task. The complete set of overall metrics, including secondary metrics and provenance fields, is in [`overall_metrics.csv`](overall_metrics.csv).

## Annotation

Primary metric: Macro F1 (higher is better).

| Rank | Method | Score | SD | Runs | Evaluation cells |
|---:|---|---:|---:|---:|---:|
| 1 | Logistic regression (`logistic_regression`) | 0.3425 | — | 1 | 18547 |
| 2 | XGBoost (`xgboost`) | 0.3373 | — | 1 | 18547 |
| 3 | kNN (`knn`) | 0.2233 | — | 1 | 18547 |
| 4 | Multinomial NB (`multinomial_nb`) | 0.2184 | — | 1 | 18547 |
| 5 | scVI/scANVI (`scvi_scanvi`) | 0.2036 | — | 1 | 18547 |
| 6 | Linear SGD (`linear_sgd`) | 0.1576 | — | 1 | 18547 |
| 7 | Linear SVM (`linear_svm`) | 0.1567 | — | 1 | 18547 |
| 8 | Nearest centroid (`nearest_centroid`) | 0.1415 | — | 1 | 18547 |
| 9 | CellTypist (`celltypist`) | 0.1223 | — | 1 | 18547 |
| 10 | scGPT-brain (`scgpt_brain`) | 0.1217 | — | 1 | 18547 |
| 11 | scGPT-human (`scgpt_human`) | 0.1024 | — | 1 | 18547 |
| 12 | Geneformer V1-10M (`geneformer_v1_10m`) | 0.0857 | — | 1 | 18547 |
| 13 | nanoBrain FMT (`nanobrain_fmt`) | 0.0136 | 0.0012 | 3 | 18547 |
| 14 | Majority (`majority`) | 0.0025 | — | 1 | 18547 |

## Integration

Primary metric: Integration composite (higher is better).

| Rank | Method | Score | SD | Runs | Evaluation cells |
|---:|---|---:|---:|---:|---:|
| 1 | Batch-centered PCA (`batch_centered_pca`) | 0.5192 | — | 1 | 18547 |
| 2 | Scanorama (`scanorama`) | 0.5108 | — | 1 | 18547 |
| 3 | Raw PCA (`raw_pca`) | 0.4985 | — | 1 | 18547 |
| 4 | Harmony (`harmony`) | 0.4976 | — | 1 | 18547 |
| 5 | scVI/scANVI (`scvi_scanvi`) | 0.4907 | — | 1 | 18547 |
| 6 | Geneformer V1-10M (`geneformer_v1_10m`) | 0.4361 | — | 1 | 18547 |
| 7 | scGPT-brain (`scgpt_brain`) | 0.4311 | — | 1 | 18547 |
| 8 | scGPT-human (`scgpt_human`) | 0.4213 | — | 1 | 18547 |
| 9 | BBKNN (`bbknn`) | 0.3912 | — | 1 | 18547 |
| 10 | nanoBrain FMT (`nanobrain_fmt`) | 0.3764 | 0.0152 | 3 | 18547 |

## Development

Primary metric: Population MSE (lower is better).

| Rank | Method | Score | SD | Runs | Evaluation cells |
|---:|---|---:|---:|---:|---:|
| 1 | scFiT (`scfit`) | 0.1567 | — | 1 | 3219 |
| 2 | TrajectoryNet (`trajectorynet`) | 0.1619 | — | 1 | 3219 |
| 3 | Identity (`development_identity`) | 0.1732 | — | 1 | 3219 |
| 4 | Global delta (`development_global_delta`) | 0.1737 | — | 1 | 3219 |
| 5 | scNODE (`scnode`) | 0.1860 | — | 1 | 3219 |
| 6 | Ridge delta (`development_ridge_delta`) | 0.2169 | — | 1 | 3219 |
| 7 | nanoBrain FMT (`nanobrain_fmt`) | 0.2315 | 0.0076 | 3 | 3219 |

## Spatial Niche

Primary metric: ARI (higher is better).

| Rank | Method | Score | SD | Runs | Evaluation cells |
|---:|---|---:|---:|---:|---:|
| 1 | nanoBrain FMT (`nanobrain_fmt`) | 0.0601 | 0.0282 | 3 | 11483 |
| 2 | No spatial structure (`spatial_no_structure`) | 0.0558 | — | 1 | 11483 |
| 3 | Spatial neighbor mean (`spatial_neighbor_mean`) | 0.0430 | — | 1 | 11483 |
| 4 | Coordinate K-means (`spatial_coordinate_kmeans`) | 0.0215 | — | 1 | 11483 |
| 5 | GraphST (`graphst`) | 0.0096 | — | 1 | 11483 |

## Protocol

Scores are grouped over the `overall` stratum only; anatomy-specific rows are not mixed into task rankings. The table preserves source means and standard deviations without recomputing a composite across unrelated metrics. Undefined metrics and the legacy spatial-completion task are excluded from the four-task ranking.

Regenerate with `uv run python scripts/build_leaderboard.py --input PATH/TO/final_benchmark_long.csv`.
