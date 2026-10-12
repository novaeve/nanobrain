# nanoBrain Benchmark Status

The current public scope contains four tasks on the frozen 100k metadata-v2 dataset: integration, annotation, development, and spatial niche prediction. The [Leaderboard](benchmarks/nanobrain_v3_100k/LEADERBOARD.md) and [`overall_metrics.csv`](benchmarks/nanobrain_v3_100k/overall_metrics.csv) are generated from the archived unified benchmark table.

## Data and protocol

- Dataset: `nanobrain_v3_100k`, 60,000 train / 20,000 validation / 20,000 test.
- Split: `BRAINWM-HUMAN-GLOBAL-6-2-2-V3`; metadata version 2.
- Fitting and selection: train for parameter fitting, validation for model selection.
- Published result status: the fixed test partition has already been opened. These scores are descriptive historical results, not a new blind evaluation.
- nanoBrain reference: three seeds and 2,000 optimizer steps per seed in the completed fixed-model campaign.
- Legacy spatial-completion rows are excluded from the current four-task leaderboard.

## Reference scores

The following rows are the three-seed nanoBrain reference values from the frozen test evaluation. The generated leaderboard contains all available methods and metrics.

| Task | Metric | Mean | SD | Direction |
|---|---|---:|---:|:---:|
| Annotation | Accuracy | 0.2644 | 0.0101 | higher |
| Annotation | Macro F1 | 0.0136 | 0.0012 | higher |
| Integration | Integration composite | 0.3764 | 0.0152 | higher |
| Spatial niche | ARI | 0.0601 | 0.0282 | higher |
| Spatial niche | NMI | 0.1027 | 0.0138 | higher |
| Development | Population MSE | 0.2315 | 0.0076 | lower |
| Development | Gene Spearman | 0.5859 | 0.0340 | higher |

Scores reflect unequal method budgets and data coverage; use the detailed metric table and each method's protocol before making comparative claims.
