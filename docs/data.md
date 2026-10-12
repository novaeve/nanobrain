# Dataset Contract

## Frozen release

| Property | Value |
|---|---|
| Dataset directory | `assets/nanobrain_v3_100k_metadata_v2/` |
| Total cells | 100,000 |
| Train / validation / test | 60,000 / 20,000 / 20,000 |
| Split contract | `BRAINWM-HUMAN-GLOBAL-6-2-2-V3` |
| Metadata | v2 row identity repair |
| Expression panel | 256 genes |
| Normalization | train split only |

The row selection and three-way split are frozen. `nanobrain-data prepare` downloads a company-hosted bundle when `NANOBRAIN_DATA_URL` is configured, verifies it, and installs it at the relative project path above. A local archive can be supplied with `--archive`. The command does not resample or repartition cells.

The bundle root contains `manifest.json`, `panel_gene_index.npy`, train-only normalization arrays, an identity repair receipt, and one directory per split. Each split contains the feature matrix, task labels and masks, verified spatial-neighbor arrays where available, development-pair indices, and row metadata. The loader uses NumPy memory maps and does not open H5AD during training.

## Integrity and use

The preparation command checks the frozen split ID, exact row counts, metadata-v2 marker, train-only normalization provenance, and every file checksum declared in `manifest.json`. Training normally loads train and validation only. The historical test partition is present in the approved bundle for final evaluation and benchmark reproduction; its published results are descriptive because the split has already been evaluated.

The company data host and release license will be recorded here once supplied. The bundle itself is excluded from Git; only this contract and the empty destination marker are tracked.
