# scATAC imputation experiment settings

Last updated: 2026-04-30

This document records the experimental setup used for the scATAC imputation task in Table 3, including dataset paths, preprocessing, evaluation metrics, baseline settings, commands, output files, and caveats found during baseline debugging.

## 1. Task definition

The task evaluates whether an imputation method improves single-cell chromatin accessibility profiles while preserving peak-level accessibility structure.

Main dataset:

```text
Dataset name: mouse_top50k
Raw multiome 10x h5:
  ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.h5

ATAC TF-IDF top50k reference h5ad:
  ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad

Raw-count ATAC top50k aligned h5ad:
  ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/benchmark/outputs/mouse_top50k/mouse_top50k_raw_counts.h5ad

Cells: 4,818
Peaks: 50,000
```

The aligned raw-count input is generated from the 10x h5 and reordered to match the TF-IDF h5ad:

```bash
cd ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation
benchmark/run_benchmark_method.sh mouse_top50k prepare_raw
```

The benchmark scripts map `mouse_top50k` to:

```text
RAW_10X = data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.h5
TFIDF   = data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad
OUT     = benchmark/outputs/mouse_top50k
```

## 2. Evaluation metrics

### 2.1 Biological evaluation

Script:

```text
benchmark/eval_method.sh
scATAC/eval_atac_biological.py
```

Command template:

```bash
benchmark/eval_method.sh mouse_top50k \
  benchmark/outputs/mouse_top50k/<method>_eval.h5ad \
  benchmark/outputs/mouse_top50k/<method>.biological_eval.json
```

Reported biological metrics:

```text
ARI:
  Adjusted Rand Index between ATAC-derived Leiden clusters and RNA-derived reference Leiden labels.

Metacell Pearson:
  Mean cell-wise Pearson correlation between imputed ATAC profiles and raw RNA-label metacell ATAC profiles.

Peakwise Spearman:
  For each peak, compute Spearman(raw_atac[:, peak], imputed_atac[:, peak]) across cells, then average across peaks.
```

Peakwise Spearman measures whether each peak preserves its cell-specific accessibility ranking after imputation. A method can have high ARI but low peakwise Spearman if it smooths cells into good clusters while losing peak-level specificity.

### 2.2 Masked open/closed evaluation

Script:

```text
benchmark/eval_masked_method.py
```

This computes AUROC and AUPRC on the same validation split and mask policy used by the scUniFlow validation setup.

Default masked setting:

```text
val_cells: 500
split_seed: 42
max_eval_cells: 32
nonzero_mask_ratio: 0.2
zero_mask_ratio: 0.02
mask_seed: 1234
```

Command template:

```bash
python benchmark/eval_masked_method.py \
  --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --imputed_atac_h5ad benchmark/outputs/mouse_top50k/<method>_eval.h5ad \
  --output_json benchmark/outputs/mouse_top50k/<method>.masked_eval.json
```

For scUniFlow, AUROC/AUPRC are taken from:

```text
${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/experiments/mouse_brain_top50k_dual_v3/best_metrics.json
```

because they are already computed during model validation.

## 3. Common conversion interface

Script:

```text
benchmark/convert_for_eval.py
```

Purpose:

```text
1. Load each method output.
2. Align cells and peaks to the TF-IDF reference h5ad.
3. Optionally select a specific matrix source, e.g. X, obsm:impute, obsm:binary.
4. Optionally apply row rank normalization.
5. Write a final evaluation h5ad.
```

Important arguments:

```text
--method:
  raw, MAGIC, SCALE, scOpen, scUniFlow

--matrix_key:
  X, obsm:<key>, or layer:<key>

--rank_normalize:
  Converts each cell's peak values into row-wise ranks in [0, 1].
  This keeps within-cell peak ordering but removes original value scale.
```

We use `no-rank` to mean that `--rank_normalize` is not used.

## 4. Method settings

### 4.1 Raw ATAC baseline

Purpose:

```text
No imputation. This is the lower-bound biological baseline.
```

Run:

```bash
cd ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation
benchmark/run_benchmark_method.sh mouse_top50k raw
benchmark/eval_method.sh mouse_top50k \
  benchmark/outputs/mouse_top50k/mouse_top50k_raw_eval.h5ad \
  benchmark/outputs/mouse_top50k/mouse_top50k_raw.biological_eval.json
```

Output:

```text
benchmark/outputs/mouse_top50k/mouse_top50k_raw_eval.h5ad
benchmark/outputs/mouse_top50k/mouse_top50k_raw.biological_eval.json
```

### 4.2 MAGIC

Script:

```text
benchmark/run_magic.py
```

Input:

```text
TF-IDF top50k h5ad
```

Setting:

```text
max_features_dense = 60000 for mouse_top50k
```

Run:

```bash
cd ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation
benchmark/run_benchmark_method.sh mouse_top50k magic
benchmark/eval_method.sh mouse_top50k \
  benchmark/outputs/mouse_top50k/mouse_top50k_magic_eval.h5ad \
  benchmark/outputs/mouse_top50k/mouse_top50k_magic.biological_eval.json
python benchmark/eval_masked_method.py \
  --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --imputed_atac_h5ad benchmark/outputs/mouse_top50k/mouse_top50k_magic_eval.h5ad \
  --output_json benchmark/outputs/mouse_top50k/mouse_top50k_magic.masked_eval.json
```

Output:

```text
benchmark/outputs/mouse_top50k/mouse_top50k_magic.h5ad
benchmark/outputs/mouse_top50k/mouse_top50k_magic_eval.h5ad
benchmark/outputs/mouse_top50k/mouse_top50k_magic.biological_eval.json
benchmark/outputs/mouse_top50k/mouse_top50k_magic.masked_eval.json
```

### 4.3 SCALE

Environment:

```text
Python: ${SCALE_PY:-python}
SCALE:  ${SCALE_CMD:-SCALE.py}
```

Required extra dependencies:

```bash
${SCALE_PY:-python} -m pip install leidenalg igraph
```

Input:

```text
Raw-count ATAC h5ad aligned to the TF-IDF reference.
```

SCALE preprocessing behavior:

```text
SCALE internally binarizes counts:
  adata.X[adata.X > 1] = 1

It then filters cells/peaks and trains an autoencoder.
```

SCALE output matrix options:

```text
obsm['impute']:
  Continuous decoder reconstruction.

obsm['binary']:
  SCALE's official binarized imputation output:
  binarization(imputed, raw)
```

Run SCALE top50k from scratch:

```bash
cd ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation
benchmark/run_benchmark_method.sh mouse_top50k scale
```

If training already produced:

```text
benchmark/outputs/mouse_top50k/scale_output/model.pt
```

but failed during postprocessing due to missing `leidenalg`, regenerate output without retraining:

```bash
${SCALE_PY:-python} benchmark/run_scale.py \
  --input benchmark/outputs/mouse_top50k/mouse_top50k_raw_counts.h5ad \
  --outdir benchmark/outputs/mouse_top50k/scale_output \
  --output benchmark/outputs/mouse_top50k/mouse_top50k_scale.h5ad \
  --n_feature -1 \
  --scale_cmd ${SCALE_CMD:-SCALE.py} \
  --extra_args --pretrain benchmark/outputs/mouse_top50k/scale_output/model.pt
```

Evaluate `obsm['impute']`:

```bash
python benchmark/convert_for_eval.py \
  --method SCALE \
  --input benchmark/outputs/mouse_top50k/mouse_top50k_scale.h5ad \
  --matrix_key obsm:impute \
  --reference_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --output benchmark/outputs/mouse_top50k/mouse_top50k_scale_impute_eval.h5ad \
  --summary_json benchmark/outputs/mouse_top50k/mouse_top50k_scale_impute_eval.summary.json

benchmark/eval_method.sh mouse_top50k \
  benchmark/outputs/mouse_top50k/mouse_top50k_scale_impute_eval.h5ad \
  benchmark/outputs/mouse_top50k/mouse_top50k_scale_impute.biological_eval.json

python benchmark/eval_masked_method.py \
  --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --imputed_atac_h5ad benchmark/outputs/mouse_top50k/mouse_top50k_scale_impute_eval.h5ad \
  --output_json benchmark/outputs/mouse_top50k/mouse_top50k_scale_impute.masked_eval.json
```

Evaluate official `obsm['binary']`:

```bash
python benchmark/convert_for_eval.py \
  --method SCALE \
  --input benchmark/outputs/mouse_top50k/mouse_top50k_scale.h5ad \
  --matrix_key obsm:binary \
  --reference_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --output benchmark/outputs/mouse_top50k/mouse_top50k_scale_binary_eval.h5ad \
  --summary_json benchmark/outputs/mouse_top50k/mouse_top50k_scale_binary_eval.summary.json

benchmark/eval_method.sh mouse_top50k \
  benchmark/outputs/mouse_top50k/mouse_top50k_scale_binary_eval.h5ad \
  benchmark/outputs/mouse_top50k/mouse_top50k_scale_binary.biological_eval.json

python benchmark/eval_masked_method.py \
  --raw_atac_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --imputed_atac_h5ad benchmark/outputs/mouse_top50k/mouse_top50k_scale_binary_eval.h5ad \
  --output_json benchmark/outputs/mouse_top50k/mouse_top50k_scale_binary.masked_eval.json
```

SCALE sanity observations:

```text
1. obs/var order matches the reference for mouse_top50k.
2. SCALE raw binary X still has cell-type structure:
   raw binary X ARI vs RNA = 0.2077
3. SCALE latent clustering collapses:
   latent ARI vs RNA = -0.0009
4. Both impute and binary output have near-random masked AUROC and near-zero peakwise Spearman.
```

Top5k sanity run:

```text
SCALE top5k impute:
  ARI = -0.0001
  AUROC = 0.5191
  AUPRC = 0.6982
  Peakwise Spearman = -0.0003

SCALE top5k binary:
  ARI = 0.0029
  AUROC = 0.5187
  AUPRC = 0.6980
  Peakwise Spearman = -0.0045
```

Conclusion:

```text
SCALE performs poorly under this imputed-peak-matrix evaluation.
The failure is not due to top50k dimensionality alone, because top5k also collapses.
The likely reason is mismatch between SCALE's binary low-dimensional reconstruction objective and this benchmark's peak-level preservation / RNA-aligned clustering evaluation.
```

### 4.4 scOpen

Environment:

```text
Python: ${SCOPEN_PY:-python}
Package entry: ${SCOPEN_CMD:-scopen}
```

Important implementation note:

```text
The installed scOpen CLI does not directly write h5ad output.
benchmark/run_scopen.py therefore calls scOpen's TF-IDF and NMF functions directly, starting from the aligned raw-count h5ad, and writes an aligned h5ad.
```

Input:

```text
benchmark/outputs/mouse_top50k/mouse_top50k_raw_counts.h5ad
```

scOpen model settings:

```text
estimate_rank: true
alpha: 1.0
max_iter: 500
random_state: 42
selected n_components: 11
```

Run:

```bash
cd ${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation
benchmark/run_benchmark_method.sh mouse_top50k scopen
```

This creates:

```text
benchmark/outputs/mouse_top50k/mouse_top50k_scopen.h5ad
benchmark/outputs/mouse_top50k/mouse_top50k_scopen.summary.json
benchmark/outputs/mouse_top50k/scopen_output/mouse_top50k_peaks.txt
benchmark/outputs/mouse_top50k/scopen_output/mouse_top50k_barcodes.txt
benchmark/outputs/mouse_top50k/scopen_output/mouse_top50k_error.txt
benchmark/outputs/mouse_top50k/scopen_output/mouse_top50k_error.pdf
```

Rank-normalized evaluation:

```bash
python benchmark/convert_for_eval.py \
  --method scOpen \
  --input benchmark/outputs/mouse_top50k/mouse_top50k_scopen.h5ad \
  --reference_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --output benchmark/outputs/mouse_top50k/mouse_top50k_scopen_eval.h5ad \
  --rank_normalize \
  --summary_json benchmark/outputs/mouse_top50k/mouse_top50k_scopen_eval.summary.json
```

No-rank evaluation:

```bash
python benchmark/convert_for_eval.py \
  --method scOpen \
  --input benchmark/outputs/mouse_top50k/mouse_top50k_scopen.h5ad \
  --reference_h5ad data/pbmc_5k_mouse_brain/e18_mouse_brain_fresh_5k_atac_tfidf_top50k.h5ad \
  --output benchmark/outputs/mouse_top50k/mouse_top50k_scopen_norank_eval.h5ad \
  --summary_json benchmark/outputs/mouse_top50k/mouse_top50k_scopen_norank_eval.summary.json
```

Recommended scOpen reporting setting:

```text
Use no-rank for Table 3.

Reason:
  It directly evaluates scOpen's original dense NMF reconstruction.
  Rank-normalization changes cell-wise peak scale and can inflate clustering behavior.
```

scOpen interpretation:

```text
scOpen has strong cell-type smoothing / clustering behavior.
It can achieve high ARI, but peak-level Spearman remains much lower than scUniFlow.
This means it recovers coarse cell structure but does not preserve fine peak-specific accessibility patterns as well.
```

### 4.5 scUniFlow

Primary model output used for Table 3:

```text
benchmark/outputs/mouse_top50k/mouse_top50k_scuniflow_dual_v3_depthtopk005_specalpha1_val500_eval.h5ad
benchmark/outputs/mouse_top50k/mouse_top50k_scuniflow_dual_v3_depthtopk005_specalpha1_val500.biological_eval.json
```

Training/validation metrics:

```text
experiments/mouse_brain_top50k_dual_v3/best_metrics.json
```

Key scUniFlow setting:

```text
dataset: mouse_top50k
validation cells: 500
model: dual_v3
inference: one-step imputation
gate strategy: depth-scaled top-k
top_k_base_rate: 0.05
specificity prior: RNA Leiden specificity prior
specificity_alpha: 1.0
```

Relevant config:

```text
config_mouse_brain_top50k_impute_onestep_depthtopk005_specalpha1.yaml
```

Validation setting:

```text
val_cells: 500
split_seed: 42
validation max_batches: 32
nonzero_mask_ratio: 0.2
zero_mask_ratio: 0.02
mask_seed: 1234
```

## 5. Current result summary

### 5.1 Full-cell results

These use all 4,818 cells unless the method filters cells.

```text
Method          ARI       AUROC     AUPRC     Peakwise Spearman   Metacell Pearson
MAGIC           0.1656    0.8322    0.8165    0.2136              0.8614
SCALE impute   -0.0001    0.5207    0.5606    0.0002              0.1529
SCALE binary    0.0008    0.5191    0.5443    0.0000              0.2683
scOpen rank     0.4291    0.7911    0.7680    0.1179              0.7493
scOpen no-rank  0.2290    0.8565    0.8451    0.2174              0.7791
```

### 5.2 Same 500 validation cells as scUniFlow

For direct biological comparison with scUniFlow, scOpen was also subset to the same 500 cells used by the scUniFlow output.

```text
Method               ARI       AUROC     AUPRC     Peakwise Spearman   Metacell Pearson
scOpen rank val500   0.5103    --        --        0.1151              0.6390
scOpen no-rank val500 0.3208   0.8565    0.8451    0.2216              0.6640
scUniFlow val500     0.3440    0.8515    0.8579    0.9807              0.3250
```

Recommended Table 3 row for scOpen:

```text
scOpen no-rank val500:
  ARI = 0.3208
  AUROC = 0.8565
  AUPRC = 0.8451
  Peakwise Spearman = 0.2216
```

Reason:

```text
This uses the same 500-cell biological evaluation subset as scUniFlow and avoids row rank normalization.
The AUROC/AUPRC are from the same validation mask policy used by the benchmark.
```

Recommended Table 3 row for SCALE:

```text
SCALE official binary:
  ARI = 0.0008
  AUROC = 0.5191
  AUPRC = 0.5443
  Peakwise Spearman = 0.0000
```

Reason:

```text
obsm['binary'] is SCALE's official binarized imputation output.
obsm['impute'] is also reported internally, but it performs similarly poorly.
```

## 6. Notes on interpretation

### 6.1 Why scOpen ARI can be high

scOpen performs TF-IDF followed by NMF low-rank reconstruction. This is a strong smoothing operation. It can improve coarse cell-type separation, which increases Leiden ARI against RNA-derived labels. However, its peakwise Spearman remains low, indicating weak preservation of fine peak-specific cell accessibility patterns.

In this experiment:

```text
scOpen no-rank val500:
  ARI = 0.3208
  Peakwise Spearman = 0.2216

scUniFlow val500:
  ARI = 0.3440
  Peakwise Spearman = 0.9807
```

Thus, scOpen is competitive for clustering but much weaker for peak-level fidelity.

### 6.2 Why SCALE ARI is poor

SCALE's poor ARI is not caused by cell/peak misalignment:

```text
SCALE obs names match the reference.
SCALE var names match the reference.
Raw binary X from SCALE still gives ARI = 0.2077.
```

The failure happens after SCALE model reconstruction:

```text
SCALE latent ARI vs RNA = -0.0009
SCALE impute ARI      = -0.0001
SCALE binary ARI      =  0.0008
```

Likely reasons:

```text
1. SCALE binarizes input and discards count/TF-IDF intensity information.
2. Its low-dimensional reconstruction does not preserve this mouse multiome cell-type structure.
3. Its official binary thresholding is based on row/column means and does not recover peak-specific accessibility patterns under this benchmark.
4. The issue persists on top5k, so it is not only a top50k dimensionality problem.
```

### 6.3 Rank normalization caveat

Rank normalization is useful when method outputs have incompatible scales, but it changes the meaning of the matrix:

```text
With --rank_normalize:
  Each cell's peaks are replaced by within-cell ranks in [0, 1].

Without --rank_normalize:
  The method's original output values are evaluated directly.
```

For scOpen, rank normalization increases biological ARI but lowers peakwise Spearman relative to no-rank. For Table 3, no-rank is the cleaner reporting choice unless the text explicitly states that all non-TF-IDF baselines are rank-normalized.

## 7. Reproduction checklist

From `${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation`:

```bash
# Prepare raw-count input
benchmark/run_benchmark_method.sh mouse_top50k prepare_raw

# MAGIC
benchmark/run_benchmark_method.sh mouse_top50k magic

# SCALE
benchmark/run_benchmark_method.sh mouse_top50k scale

# scOpen
benchmark/run_benchmark_method.sh mouse_top50k scopen
```

Check logs:

```bash
tail -f benchmark/outputs/mouse_top50k/logs/scale_full.log
tail -f benchmark/outputs/mouse_top50k/logs/scopen_full.log
```

If SCALE fails after training with missing Leiden dependency:

```bash
${SCALE_PY:-python} -m pip install leidenalg igraph
```

Then regenerate from the saved model:

```bash
${SCALE_PY:-python} benchmark/run_scale.py \
  --input benchmark/outputs/mouse_top50k/mouse_top50k_raw_counts.h5ad \
  --outdir benchmark/outputs/mouse_top50k/scale_output \
  --output benchmark/outputs/mouse_top50k/mouse_top50k_scale.h5ad \
  --n_feature -1 \
  --scale_cmd ${SCALE_CMD:-SCALE.py} \
  --extra_args --pretrain benchmark/outputs/mouse_top50k/scale_output/model.pt
```

