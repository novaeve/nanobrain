# nanoBrain

![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![Data](https://img.shields.io/badge/dataset-100k%20cells-2da44e)
![Tasks](https://img.shields.io/badge/benchmark-4%20tasks-0969da)

**nanoBrain is a compact virtual-cell research and teaching project.** It gives researchers and students a readable route from a fixed 100,000-cell dataset to model training, validation, and comparable results. It is a rapid experimentation entry point, not a small deployment of Brain World Model.

> [!IMPORTANT]
> This repository uses the frozen 100k dataset and its existing 60k/20k/20k train/validation/test split. It does not download or train on a 10M-cell dataset. The historical test split has already been evaluated and is not a new blind holdout.

**Language:** English | [简体中文](README_zh-CN.md) · [Tasks](#tasks) · [Leaderboard](benchmarks/nanobrain_v3_100k/LEADERBOARD.md) · [Data](#data) · [Reproducibility](#reproducibility)

## Quick Start

Supported quick-start environment: Linux x86_64, Python 3.12, `uv`, CPU. Install `uv` using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Then configure the company-hosted dataset URL in `.env` and run:

```bash
cp .env.example .env
# Set NANOBRAIN_DATA_URL and, when provided, NANOBRAIN_DATA_SHA256 in .env.
./scripts/bootstrap.sh
```

The setup script creates a project-local environment, installs the CPU PyTorch build and development tools, then downloads or verifies the frozen dataset at `assets/nanobrain_v3_100k_metadata_v2/`. If the data has already been placed there, only validation is performed.

Train a short CPU run for one task:

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --task annotation --steps 20 --batch-size 16 \
  --device cpu --precision fp32 --output results/annotation-smoke
```
/
Use `--task integration`, `--task development`, or `--task spatial_niche` for the other benchmark tasks. For a CUDA run, install a PyTorch wheel that matches the server's CUDA runtime using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then set `--device cuda`. A 20-step run is a smoke test, not a benchmark-quality training budget.

## Tasks

| Task | Training objective | Primary benchmark metric | Guide |
|---|---|---|---|
| Integration | Preserve observed expression while reducing batch signal | Integration composite | [Guide](docs/tasks/integration.md) |
| Annotation | Predict train-observed cell-type labels | Macro F1, accuracy | [Guide](docs/tasks/annotation.md) |
| Development | Predict a target population from source population and condition | Population MSE, gene Spearman | [Guide](docs/tasks/development.md) |
| Spatial niche | Predict anatomy-derived niche labels with verified spatial context | ARI, NMI | [Guide](docs/tasks/spatial-niche.md) |

Each task uses an independent model and optimizer. The training loader uses train and validation only; the test split is not opened during a normal training run. The model and task contracts are documented in the [component guide](docs/components.md).

To replace the encoder and train your own component, follow the [custom-model tutorial](docs/custom-model.md). It includes a short registered encoder, a configuration file, and a single training command.

## Data

The dataset has exactly 100,000 cells from the frozen `BRAINWM-HUMAN-GLOBAL-6-2-2-V3` split: 60,000 train, 20,000 validation, and 20,000 test cells. It contains a fixed 256-gene expression panel plus annotation, batch, development, and spatial metadata. Normalization is fit on train only. The metadata-v2 release includes repaired row identity and only independently verified spatial sections.

The bundle is intentionally not committed to Git. The download URL is configurable so the company can host the approved artifact separately. The preparation command verifies the manifest, per-file hashes, row counts, normalization provenance, and metadata version. It preserves the frozen partition; it does not randomly resplit rows.

```bash
uv run nanobrain-data prepare
# Or validate a local ZIP/TAR without a network request:
uv run nanobrain-data prepare --archive /path/to/nanobrain-100k-v2.tar.gz
```

See the [dataset contract](docs/data.md) and [data folder notes](assets/README.md).

## Leaderboard

The checked-in [Leaderboard](benchmarks/nanobrain_v3_100k/LEADERBOARD.md) summarizes the existing fixed 100k benchmark. The full machine-readable set of overall metrics is in [`overall_metrics.csv`](benchmarks/nanobrain_v3_100k/overall_metrics.csv). It includes the four current tasks, reports means, standard deviations, run counts, and evaluation counts, and records the test split as previously opened. Comparisons use different published training regimes; inspect each method's protocol before drawing conclusions. The optional legacy spatial-completion task is excluded from this four-task table.

Regenerate the summary from its checked-in source table with:

```bash
uv run python scripts/build_leaderboard.py \
  --input benchmarks/nanobrain_v3_100k/frozen_test_metrics.csv
```

## Reproducibility

- The 100k row selection, split, panel, and train-only normalization are recorded in the data manifest.
- Random seeds, step budgets, batch size, validation selection, and task are written to each training receipt.
- Final benchmark results aggregate three nanoBrain seeds. Other methods retain their own seed counts and evaluation coverage.
- The current fixed-test benchmark is descriptive because its labels have already been opened. New model selection must use train/validation; do not tune against the displayed test results.
- The baseline result table is generated from a checked-in common-format evaluation table, not from hand-entered values.

```bash
uv run pytest
uv run python scripts/build_leaderboard.py
```

## Project Layout

```text
src/nanobrain/       Model, data contracts, task objectives, and training
src/brainwm*/        Shared model and benchmark components used by nanoBrain
scripts/             Training, data preparation, and benchmark utilities
docs/tasks/           Task-by-task tutorials
benchmarks/           Frozen 100k leaderboard and source metrics
assets/               Local dataset location; data is not tracked by Git
```

## Scope

nanoBrain is intended for research, teaching, and prototype validation on modest hardware. It helps test encoders, task heads, losses, and sampling choices before deciding which ideas merit larger experiments. It is not a clinical decision system and does not make claims about performance on cohorts beyond this fixed dataset.

## Citation and License

Citation and license metadata will follow the company's release instructions. Third-party reuse notes are in [`docs/third_party.md`](docs/third_party.md).
