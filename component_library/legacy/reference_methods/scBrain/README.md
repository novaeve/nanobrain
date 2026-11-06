# scBrain

scBrain is a hierarchical neural model for spatial-transcriptomic brain cell annotation. It takes a `cell x gene` matrix, optionally uses spatial or disease covariates for ablations, and predicts nested cell labels from coarse to fine levels. The released code is source-only: datasets, pretrained foundation-model weights, and trained checkpoints are intentionally excluded.

## Paper Snapshot

The current paper bundle is included under [`paper/`](paper/), including the compiled PDF and the full LaTeX source used for the latest preprint-style manuscript:

- PDF: [`paper/neurips_2026_preprint.pdf`](paper/neurips_2026_preprint.pdf)
- TeX source: [`paper/neurips_2026_preprint.tex`](paper/neurips_2026_preprint.tex)
- Bibliography: [`paper/references.bib`](paper/references.bib)

![scBrain overview](paper/figures/fig_scbrain_overview.png)

![Primary benchmark summary](paper/figures/fig_primary_overview.png)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `paper/` | Compiled manuscript PDF, LaTeX source, bibliography, appendix fragments, and selected paper figures. |
| `src/models/` | scBrain model definitions and baseline model wrappers. |
| `src/engines/` | Training, evaluation, metric logging, and specialized ablation trainers. |
| `src/utils/` | Dataloaders, argument parser, logging, and metric helpers. |
| `experiments/scbrain/` | Public SEA-AD scBrain entry point. |
| `experiments/kukanja/` | Public Kukanja MS/EAE entry points and baseline runners. |
| `experiments/*/SEA_AD_MTP_ST/*.log` | Lightweight logs used to audit paper results. |
| `scripts/preprocess/` | Raw h5ad to packed NPZ preprocessing scripts. |
| `paper_tools/` | Utilities for regenerating result tables and appendix summaries. |
| `configs/` | Canonical command records for the reported scBrain runs. |
| `data/`, `weights/`, `outputs/` | Local-only placeholders; contents are ignored by git. |

## Environment Setup

Use Python 3.10+ with CUDA-enabled PyTorch for GPU training.

```bash
git clone https://github.com/novaeve/scBrain.git
cd scBrain
pip install -r requirements.txt
```

Conda users can instead run:

```bash
conda env create -f environment.yml
conda activate scbrain
```

Optional foundation-model baselines require third-party code and pretrained weights. See `docs/ASSET_MANIFEST.md` before running Geneformer, scGPT, UCE, scFoundation, Stack, Nicheformer, or scSimilarity experiments.

## Data Sources

The code expects local h5ad files to be converted to compact `.npz` files before training. The following links are the recommended starting points:

| Dataset | What to download | Reference link |
| --- | --- | --- |
| SEA-AD MTG MERFISH | `SEAAD_MTG_MERFISH.2024-12-11.h5ad` | Allen Brain Map SEA-AD download portal: https://portal.brain-map.org/explore/seattle-alzheimers-disease/seattle-alzheimers-disease-brain-cell-atlas-download |
| SEA-AD public bucket | Same SEA-AD atlas assets, when mirrored on AWS Open Data | AWS Open Data Registry: https://registry.opendata.aws/allen-sea-ad-atlas/ |
| Kukanja MS Xenium | `MS_xenium_data_v5_with_images_tmap.h5ad` | Obtain from the Kukanja dataset release or internal shared data store. |
| Kukanja EAE Xenium | `EAE_rev_ingest_only_new_v4_with_images_tmap.h5ad` | Obtain from the Kukanja dataset release or internal shared data store. |

If the Kukanja files are not publicly available in your environment, ask the data owner for the exact h5ad files above. The preprocessing scripts assume those h5ad schemas.

## Expected Local Data Layout

After downloading and preprocessing, place files as follows:

```text
data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad
data/raw/kukanja/MS_xenium_data_v5_with_images_tmap.h5ad
data/raw/kukanja/EAE_rev_ingest_only_new_v4_with_images_tmap.h5ad
data/global_data.npz
data/global_data_v2.npz
data/merfish_gene_names.json
data/kukanja/kukanja_ms.npz
data/kukanja/kukanja_eae.npz
```

`data/global_data.npz` is used by the main SEA-AD scBrain run. The same SEA-AD packing script also writes the auxiliary fields required by contextual ablations, so it can be copied or written directly as `data/global_data_v2.npz`.

## Preprocess Raw Data

SEA-AD:

```bash
python scripts/preprocess/pack_seaad.py \
  --input-h5ad data/raw/SEAAD_MTG_MERFISH.2024-12-11.h5ad \
  --output-npz data/global_data.npz \
  --gene-names-json data/merfish_gene_names.json

cp data/global_data.npz data/global_data_v2.npz
```

Kukanja MS:

```bash
python scripts/preprocess/pack_kukanja.py \
  --dataset ms \
  --input-h5ad data/raw/kukanja/MS_xenium_data_v5_with_images_tmap.h5ad \
  --output-npz data/kukanja/kukanja_ms.npz
```

Kukanja EAE:

```bash
python scripts/preprocess/pack_kukanja.py \
  --dataset eae \
  --input-h5ad data/raw/kukanja/EAE_rev_ingest_only_new_v4_with_images_tmap.h5ad \
  --output-npz data/kukanja/kukanja_eae.npz
```

The SEA-AD packer reads `Class`, `Subclass`, `Supertype`, `Donor ID`, `Supertype confidence`, `Continuous Pseudo-progression Score`, `obsm['spatial']`, and optionally `obsm['X_spatial_tiled']`, `Normalized depth from pia`, and `Cell volume`. The Kukanja packer reads the annotation hierarchy, sample IDs, spatial coordinates, and disease score or condition field.

## Train scBrain

SEA-AD main model:

```bash
python experiments/scbrain/main.py \
  --model_name scBrain \
  --npz_path data/global_data.npz \
  --device cuda:0 \
  --bs 512 --max_epochs 500 --patience 30 \
  --e_layers 3 --d_layers 1 \
  --enc_hidden_dim 256 --dec_hidden_dim 256 \
  --latent_dim 20 --residual_mode feature \
  --seed 3028
```

Kukanja MS:

```bash
python experiments/kukanja/main_scbrain.py \
  --npz_path data/kukanja/kukanja_ms.npz \
  --device cuda:0 \
  --bs 512 --max_epochs 500 --patience 30 \
  --e_layers 3 --d_layers 1 \
  --enc_hidden_dim 256 --dec_hidden_dim 128 \
  --latent_dim 20 --residual_mode feature \
  --seed 3028
```

Kukanja EAE:

```bash
python experiments/kukanja/main_scbrain.py \
  --npz_path data/kukanja/kukanja_eae.npz \
  --device cuda:0 \
  --bs 512 --max_epochs 500 --patience 30 \
  --e_layers 3 --d_layers 1 \
  --enc_hidden_dim 256 --dec_hidden_dim 256 \
  --latent_dim 20 --residual_mode feature \
  --seed 3028
```

Useful variations:

```bash
# Independent decoder ablation
python experiments/scbrain/main.py --model_name scBrain --residual_mode none --npz_path data/global_data.npz

# Logit-residual ablation
python experiments/scbrain/main.py --model_name scBrain --residual_mode logit --npz_path data/global_data.npz

# Contextual SEA-AD ablations
python experiments/mjm_2/main_a.py --npz_path data/global_data_v2.npz --imputer_path data/depth_imputer.pth
python experiments/mjm_2/main_b.py --npz_path data/global_data_v2.npz
python experiments/mjm_2/main_c.py --npz_path data/global_data_v2.npz
```

## Outputs and Metrics

Training scripts write logs and checkpoints under the corresponding `experiments/.../` directory by default. Each test run reports Macro Precision/Recall/F1, Micro Precision/Recall/F1, and AUC-ROC when available.

Parse all included logs:

```bash
python experiments/extract_all_metrics.py
python experiments/kukanja/collect_results.py
```

Regenerate paper-side result tables:

```bash
python paper_tools/build_main_results_table.py
python paper_tools/build_levelwise_appendix.py
python paper_tools/build_reproducibility_appendix.py
```

## Release and Safety Checks

Before pushing or sharing the repository, run:

```bash
python tools/check_release.py
```

This verifies that the repo does not contain local absolute paths, large packed datasets, checkpoints, pretrained weights, or generated caches. Large files should stay outside git under `data/`, `weights/`, `external/`, or `outputs/`.

Create a source tarball if needed:

```bash
bash tools/make_archive.sh
```

## Notes for Internal Users

Use `scBrain` as the public method name. Some legacy files and historical logs still contain older internal names because they are needed for exact reproducibility of completed experiments. New training commands should use `experiments/scbrain/main.py` and `experiments/kukanja/main_scbrain.py`.
