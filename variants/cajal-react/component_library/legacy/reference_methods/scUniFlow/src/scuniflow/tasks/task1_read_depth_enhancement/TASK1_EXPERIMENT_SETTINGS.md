# Task2 读深增强实验设置

本文档记录 NeurIPS 2026 草稿 Table 1 中 Task2 PBMC68k read depth enhancement 的完整实验设置，包括数据来源、低深度任务构造、细胞训练/验证划分、统一 full-gene 评估流程，以及每个方法的具体运行设置。

## 任务目标

Task2 评估低读深 scRNA-seq 表达谱的读深增强能力。对每个细胞，方法输入由原始 raw counts 经过 4× binomial downsampling 得到的 low-depth profile，并预测对应的 high-depth raw-count profile。最终评估关注增强后的表达矩阵是否更接近 high-depth 表达，并且是否恢复更清晰的细胞类型结构。

本实验使用：

- 数据集：`${SCUNIFLOW_DATA:-data}/pbmc68k_raw_counts.h5ad`
- 数据规模：`68579` cells × `32738` genes
- 任务构造：4× binomial downsampling
- Keep probability：`0.25`
- 训练池细胞数：`61721`
- 验证细胞数：`6858`
- scUniFlow/Novaeve-Flow task-head 训练子集：`12000` cells
- 评估基因空间：`32738` full genes
- Classical imputation baseline 增强基因数：`2000` HVGs
- Reference labels：high-depth validation counts 上的 Leiden labels
- Reference Leiden resolution：`0.4`
- Reference cluster 数：`9`

## 数据与低读深构造

所有方法使用同一个 PBMC68k raw-count h5ad。对每个 high-depth count vector `x_high`，代码通过 binomial downsampling 构造 low-depth 输入 `x_low`，即每个 count 按 `keep probability = 0.25` 保留。High-depth raw counts 只作为训练目标和评估真值。

共享评估文件：

- 评估细胞索引：`read_depth_enhancement/shared_eval_indices/pbmc68k_scuniflow_val_seed42_indices.npy`
- 共享 low/high cache：`read_depth_enhancement/shared_eval_indices/pbmc68k_scuniflow_val_fold4_seed42_low_high_cache/`
- 共享 reference labels：`read_depth_enhancement/shared_eval_indices/pbmc68k_scuniflow_val_seed42_ref_labels_res0.4_n9.npy`

相关实现：

- `read_depth_enhancement/common.py`
- `read_depth_enhancement/prepare_shared_pbmc_rde_eval_indices.py`
- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`
- `scuniflow_read_depth_enhancement/common.py`
- `scuniflow_read_depth_enhancement/eval_scuniflow_read_depth_enhancement.py`

## 训练/验证划分

PBMC68k 主实验使用固定 cell-level split：

- 划分来源：`experiments/pbmc68k_augmented_181`
- Split mode：random cells
- 总细胞数：`68579`
- 训练池：`61721` cells
- 验证集：`6858` cells
- Validation ratio：约 `0.1`

这里需要区分两类细胞集合：

1. 模型训练集合：用于训练 supervised read-depth enhancement head。
2. 统一评估集合：固定的 `6858` 个 validation cells，用于所有方法的 Table 1 评估。

scUniFlow/Novaeve-Flow 的 RDE head 只使用训练池中的 `12000` 个细胞训练，和统一评估验证集没有重叠。Classical baselines 不训练 supervised task head，而是在固定 PBMC validation low-depth cells 上直接拟合或运行 imputation。scVI 在同一批 validation low-depth cells 上训练并输出 full-gene imputed matrix。

相关记录：

- scUniFlow split：`scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head/split_info.json`
- scUniFlow train subset：`read_depth_enhancement/shared_eval_indices/pbmc68k_scuniflow_train12k_seed20260430_indices.npy`
- scUniFlow validation subset：`read_depth_enhancement/shared_eval_indices/pbmc68k_scuniflow_val_seed42_indices.npy`

## 基因空间与 full-gene 评估

所有方法最终都在同一个 `6858 × 32738` full-gene matrix 上评估。不同方法的增强范围不同：

- Low-depth baseline：直接使用 4× downsampled full-gene counts，不做增强。
- MAGIC、SAVER、scImpute：只增强 `2000` 个 highly variable genes。增强后的 HVG values 写回 full-gene matrix，非 HVG genes 保留 low-depth values。
- scVI：在 full-gene matrix 上运行，输出 `32738` genes。
- scFoundation：模型输入受 scFoundation vocabulary 限制，记录中使用 `19264` target genes、`17695` present genes；评估时对齐到 full-gene pipeline。
- scUniFlow/Novaeve-Flow：full-gene forward，输出 `32738` genes。

因此 Table 1 的比较不是把所有方法限制到 HVG 空间，而是统一考察最终增强矩阵在 full-transcriptome 下的下游表现。HVG-only baselines 在其可计算范围内运行，非增强基因保留 low-depth 值。

## 统一评估流程

所有方法使用相同的 validation cells、low-depth/high-depth cache、reference labels 和 full-gene clustering/PCC pipeline。

Clustering 评估流程：

- 输入矩阵：每个方法输出的 full-gene enhanced matrix
- Normalize total counts 到 `10000`
- `log1p`
- 选择 `2000` HVGs
- Scale
- PCA
- Neighbors
- Leiden clustering
- 搜索 Leiden resolution，使预测 cluster 数与 reference labels 的 cluster 数一致

记录的指标包括：

- NMI
- ARI
- Target-label silhouette
- Predicted-cluster silhouette
- Calinski-Harabasz
- Davies-Bouldin
- Cell-level Pearson correlation mean/median
- Gene-level Pearson correlation mean/median

论文 Table 1 当前展示：

- NMI
- ARI
- Davies-Bouldin
- Mean PCC
- Median PCC

相关实现：

- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`
- `read_depth_enhancement/eval_baseline_read_depth_enhancement.py`
- `scuniflow_read_depth_enhancement/eval_scuniflow_read_depth_enhancement.py`
- `scuniflow_read_depth_enhancement/eval_scuniflow_pcc.py`

## scUniFlow / Novaeve-Flow 设置

主实验 run：

- 训练目录：`scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head`
- 最终评估目录：`scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head/eval_scuniflow_val_shared_common`
- Backbone 实验：`${SCUNIFLOW_RESULTS:-results}/pretrain/pbmc68k_augmented_181`
- Backbone checkpoint：`${SCUNIFLOW_RESULTS:-results}/pretrain/pbmc68k_augmented_181/checkpoints/checkpoint_best.pt`
- 数据：`${SCUNIFLOW_DATA:-data}/pbmc68k_raw_counts.h5ad`
- Downsample fold：`4.0`
- Train cells：`12000`
- Eval cells：`6858`
- Genes：`32738`
- Epochs：`8`
- Batch size：`24`
- Head learning rate：`1e-3`
- Backbone learning rate：`1e-5`
- Weight decay：`1e-4`
- Head hidden dimension：`256`
- Head dropout：`0.1`
- `train_mamba=False`
- `train_router=False`
- Loss：full-gene read-depth reconstruction loss in the task head

task head 的输入为 low-depth expression profile 和 backbone 产生的 full-gene context；训练目标为 high-depth expression profile。主实验冻结 router 和 Mamba backbone，只训练 read-depth enhancement head。

Table 1 结果：

| NMI | ARI | D-B | Mean PCC | Median PCC |
|---:|---:|---:|---:|---:|
| 0.5063 | 0.4261 | 2.1888 | 0.8420 | 0.8494 |

补充记录中也保留了 scUniFlow full-gene 与 HVG-only output 的对照：

| Method | Enhanced genes | NMI | ARI | C-H | D-B | Mean PCC | Median PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| scUniFlow full-gene | 32738 | 0.4870 | 0.4039 | 129.7029 | 2.1888 | 0.8420 | 0.8494 |
| scUniFlow HVG-only output | 2000 HVGs | 0.3848 | 0.3311 | 106.6378 | 2.5694 | 0.6983 | 0.7030 |

## Low-depth baseline 设置

运行脚本：

- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`

主要输出：

- `read_depth_enhancement/runs/pbmc68k_classical_lowdepth_val_shared_common`

设置：

- 输入：4× downsampled low-depth counts
- 增强：无
- Gene space：`32738` full genes
- Evaluation cells：固定 `6858` validation cells
- Reference labels：共享 high-depth Leiden labels

Table 1 结果：

| NMI | ARI | D-B | Mean PCC | Median PCC |
|---:|---:|---:|---:|---:|
| 0.3531 | 0.3249 | 3.1323 | 0.7030 | 0.7076 |

## MAGIC 设置

运行脚本：

- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`

主要输出：

- `read_depth_enhancement/runs/pbmc68k_magic_hvg2000_val_shared_common`

设置：

- 输入：PBMC validation low-depth cells
- Enhanced genes：`2000` HVGs
- Non-HVG genes：保留 low-depth values
- Input space：`log1p_norm`
- `knn=5`
- `t=3`
- `n_pca=100`
- Solver：`exact`
- 输出转换回估计 high-depth count scale 后进入 full-gene evaluation pipeline
- 不使用 supervised train/validation split

Table 1 结果：

| NMI | ARI | D-B | Mean PCC | Median PCC |
|---:|---:|---:|---:|---:|
| 0.3952 | 0.3465 | 2.4039 | 0.7016 | 0.7064 |

## SAVER 设置

运行脚本：

- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`

主要输出：

- `read_depth_enhancement/runs/pbmc68k_saver_hvg2000_val_shared_common`

设置：

- 输入：PBMC validation low-depth cells
- Enhanced genes：`2000` HVGs
- Non-HVG genes：保留 low-depth values
- R runner：`${RSCRIPT:-Rscript}`
- `do.fast=TRUE`
- `ncores=8`
- `npred=NULL`
- 使用默认 size-factor normalization
- 不使用 supervised train/validation split

Table 1 结果：

| NMI | ARI | D-B | Mean PCC | Median PCC |
|---:|---:|---:|---:|---:|
| 0.3475 | 0.2979 | 2.7906 | 0.6984 | 0.7040 |

## scImpute 设置

运行脚本：

- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`

主要输出：

- `read_depth_enhancement/runs/pbmc68k_scimpute_hvg2000_val_shared_common`

设置：

- 输入：PBMC validation low-depth cells
- Enhanced genes：`2000` HVGs
- Non-HVG genes：保留 low-depth values
- R runner：`${RSCRIPT:-Rscript}`
- `Kcluster=9`
- `drop_thre=0.5`
- `ncores=8`
- 不使用 supervised train/validation split

Table 1 结果：

| NMI | ARI | D-B | Mean PCC | Median PCC |
|---:|---:|---:|---:|---:|
| 0.3963 | 0.3561 | 2.5650 | 0.7044 | 0.7084 |

## scVI 设置

运行脚本：

- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`

主要输出：

- `read_depth_enhancement/runs/pbmc68k_scvi_hvg2000_val_shared_common`

设置：

- 输入：PBMC validation low-depth cells
- Gene mode：`all`
- Enhanced genes：`32738`
- Epochs：`100`
- Batch size：`256`
- Hidden dimension：`128`
- Latent dimension：`20`
- Layers：`2`
- Gene likelihood：`zinb`
- `n_samples=1`
- 输出 normalized expression，并转换回估计 high-depth count scale
- 不使用 scUniFlow 的 supervised RDE train split

Table 1 结果：

| NMI | ARI | D-B | Mean PCC | Median PCC |
|---:|---:|---:|---:|---:|
| 0.5041 | 0.3583 | 1.6900 | 0.6660 | 0.6721 |

## scFoundation 设置

运行脚本：

- `read_depth_enhancement/train_scfoundation_read_depth_enhancement.py`
- `read_depth_enhancement/eval_scfoundation_read_depth_enhancement.py`

设置：

- Pretrain checkpoint：`${SCFOUNDATION_ROOT:-external/scFoundation}/models.ckpt`
- Gene index：`${SCFOUNDATION_ROOT:-external/scFoundation}/OS_scRNA_gene_index.19264.tsv`
- 模型 target genes：`19264`
- Present genes：`17695`
- 数据：PBMC68k raw counts
- Downsample fold：`4.0`
- Evaluation cells：固定 `6858` validation cells
- Evaluation gene space：full-gene clustering pipeline

当前论文 Table 1 使用的记录值来自已有 RDE 结果文档和 registry；该版本对齐 full-gene evaluation pipeline：

| NMI | ARI | D-B | Mean PCC | Median PCC |
|---:|---:|---:|---:|---:|
| 0.3992 | 0.3533 | 2.4283 | 0.7950 | 0.8002 |

当前 `task2/read_depth_enhancement/runs/pbmc68k_scfoundation_rde_scuniflow_train12k_val6858_e1` 目录保留了一版和 scUniFlow train/val cells 对齐的 scFoundation RDE training run，但该目录中主要落盘的是训练 summary；论文表格中的 scFoundation 数值来自早期完整评估记录。

## Table 1 数值汇总

| Method | NMI | ARI | D-B | Mean PCC | Median PCC |
|---|---:|---:|---:|---:|---:|
| Low-depth | 0.3531 | 0.3249 | 3.1323 | 0.7030 | 0.7076 |
| MAGIC | 0.3952 | 0.3465 | 2.4039 | 0.7016 | 0.7064 |
| SAVER | 0.3475 | 0.2979 | 2.7906 | 0.6984 | 0.7040 |
| scImpute | 0.3963 | 0.3561 | 2.5650 | 0.7044 | 0.7084 |
| scVI | 0.5041 | 0.3583 | 1.6900 | 0.6660 | 0.6721 |
| scFoundation | 0.3992 | 0.3533 | 2.4283 | 0.7950 | 0.8002 |
| scUniFlow / Novaeve-Flow | 0.5063 | 0.4261 | 2.1888 | 0.8420 | 0.8494 |

## 可复现实验文件

核心脚本：

- `read_depth_enhancement/common.py`
- `read_depth_enhancement/prepare_shared_pbmc_rde_eval_indices.py`
- `read_depth_enhancement/eval_classical_imputation_read_depth_enhancement.py`
- `read_depth_enhancement/eval_baseline_read_depth_enhancement.py`
- `read_depth_enhancement/train_scfoundation_read_depth_enhancement.py`
- `read_depth_enhancement/eval_scfoundation_read_depth_enhancement.py`
- `scuniflow_read_depth_enhancement/common.py`
- `scuniflow_read_depth_enhancement/train_scuniflow_read_depth_enhancement.py`
- `scuniflow_read_depth_enhancement/eval_scuniflow_read_depth_enhancement.py`
- `scuniflow_read_depth_enhancement/eval_scuniflow_pcc.py`

关键结果记录：

- RDE summary：`read_depth_enhancement/rde_results_summary.md`
- Low-depth summary：`read_depth_enhancement/runs/pbmc68k_classical_lowdepth_val_shared_common/summary.json`
- MAGIC summary：`read_depth_enhancement/runs/pbmc68k_magic_hvg2000_val_shared_common/summary.json`
- SAVER summary：`read_depth_enhancement/runs/pbmc68k_saver_hvg2000_val_shared_common/summary.json`
- scImpute summary：`read_depth_enhancement/runs/pbmc68k_scimpute_hvg2000_val_shared_common/summary.json`
- scVI summary：`read_depth_enhancement/runs/pbmc68k_scvi_hvg2000_val_shared_common/summary.json`
- scFoundation training summary：`read_depth_enhancement/runs/pbmc68k_scfoundation_rde_scuniflow_train12k_val6858_e1/summary.json`
- scUniFlow run args：`scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head/run_args.json`
- scUniFlow split：`scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head/split_info.json`
- scUniFlow shared eval summary：`scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head/eval_scuniflow_val_shared_common/summary.json`
- scUniFlow HVG-only eval summary：`scuniflow_read_depth_enhancement/runs/pbmc68k_fold4_head/eval_scuniflow_val_shared_common_hvg2000/summary.json`
