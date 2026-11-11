# Task3 空间基因补全实验设置

本文档记录 NeurIPS 2026 草稿 Table 2 中 Task3 mouse brain spatial gene imputation 的完整实验设置，包括数据来源、基因划分、spot 训练/验证划分，以及每个方法的具体运行设置。

## 任务目标

Task3 评估 Visium mouse brain spot 上的空间基因补全能力。对每个空间 spot，方法只能看到固定的 observed gene panel；另一组不重叠的 target gene panel 会从空间输入中隐藏，只在预测后作为真值用于评估。

本实验使用：

- Reference sc/snRNA-seq：`${SCUNIFLOW_DATA:-data}/all_cells_20200625.h5ad`
- Spatial Visium 数据根目录：`${SCUNIFLOW_DATA:-data}/mouse_brain_visium/mouse_brain_visium_wo_cloupe_data/rawdata`
- 主实验 slide：`ST8059048`
- Spatial spot 过滤：`in_tissue_only=True`
- Reference cell 子采样：`max_ref_cells=8000`
- Spatial spot 数量：`2987`
- 对齐后的基因宇宙：`31053`
- Observed spatial genes：`1024`
- Held-out target genes：`256`
- Panel seed：`42`
- Reference/spot subsampling seed：`42`
- 合格基因的最小非零 spatial spot 数：`10`
- 合格基因的最小非零 reference cell 数：`20`
- 评估表达空间：log-CPM，即 `log1p(count / library_size * 1e6)`

## 数据对齐

所有方法使用同一个 Ensembl gene universe。Visium 数据通过 `scanpy.read_visium` 读取；若启用 `in_tissue_only`，则只保留 `in_tissue == 1` 的 spot。随后按照 reference h5ad 的基因顺序重排 Visium 矩阵。代码会检查 reference 和 Visium 的基因集合是否完全一致；如果不一致，实验不会继续。

相关实现：

- `common.py`：`load_reference_metadata`、`load_visium_mouse_brain`
- `benchmark_common.py`：`prepare_benchmark_data`

## 基因面板划分

主 benchmark 采用的是 gene-level holdout，而不是 spot-level train/test split。也就是说，所有方法都在同一批 ST8059048 spot 上做预测，但只能使用 observed genes；target genes 从空间输入中隐藏，用于最终评估。

合格基因必须同时满足以下条件：

- 在选中的 reference cells 中至少 `20` 个 cell 非零
- 在选中的 spatial spots 中至少 `10` 个 spot 非零
- 在选中的 reference cells 中方差大于 0
- 在选中的 spatial spots 中方差大于 0

在所有合格基因中，代码使用 `panel_seed=42` 无放回采样 `1024 + 256 = 1280` 个基因。采样结果中的前 `256` 个基因作为 held-out target genes，之后按基因索引排序；剩余 `1024` 个基因作为 observed genes，也按索引排序。

对所有方法都保持一致：

- Spatial observed input：ST8059048 中 `1024` 个 observed genes 的表达。
- Spatial target truth：ST8059048 中 `256` 个 target genes 的表达，只用于评估。
- Reference input：选中的 `8000` 个 reference cells；根据方法需要，reference 中可以包含 observed genes 和 target genes。
- Spatial target-gene expression 在拟合或推理时不会提供给任何 baseline 方法。它只会写入 `truth_target_logcpm.csv.gz`，并由 `write_benchmark_outputs` 读取以计算评估指标。

相关实现：

- `benchmark_common.py`：`pick_gene_panels`、`prepare_benchmark_data`、`write_benchmark_outputs`
- 每个 run 目录下的面板记录：`panel_info.json`

## Spot 训练/验证划分

这里需要区分两类划分：

1. 基因划分：所有方法都使用同一组 `1024` observed genes 和 `256` target genes，这是 Table 2 的核心 holdout 设置。
2. Spot 划分：只有 scUniFlow 的 supervised task head 训练阶段使用 spot-level train/validation split；其他 classical baselines 不使用 supervised spot train/validation split。

Table 2 中的 classical baselines，包括 SpaGE、gimVI、Tangram、Seurat 和 LIGER，都不训练一个以 hidden target genes 为监督信号的 spatial task head。因此它们没有 spot-level train/validation split。它们在全部 `2987` 个 in-tissue ST8059048 spots 上运行，只能看到 observed genes，然后在全部 spots 的 hidden target genes 上评估。

scUniFlow task head 使用 spot-level split 来选择最佳 task-head checkpoint：

- Split mode：random spots
- Split seed：`42`
- Validation ratio：`0.2`
- 总 spot 数：`2987`
- 训练 spot 数：`2390`
- 验证 spot 数：`597`
- 训练集和验证集使用同一组 `1024` observed genes 与 `256` target genes。
- task-head 训练时，每个样本是一个 spatial spot；输入中 observed genes 可见，target genes 被 mask。
- 验证集 target values 只用于计算 validation loss 和选择 best checkpoint。
- Table 2 中 scUniFlow 的最终数值来自 `split=all` 评估，即使用验证集选出的 best checkpoint 在全部 `2987` 个 spot 上评估。

相关文件：

- `spatial_head_common.py`：`build_spot_train_val_split`、`SpatialSpotTargetDataset`
- `train_scuniflow_spatial_gene_imputation_head.py`
- `eval_scuniflow_spatial_gene_imputation_head.py`
- 划分记录：`runs/scuniflow_spatial_head_ST8059048_v5/split_info.json`

## 统一评估指标

所有方法都在相同的 target log-CPM 矩阵上评估，矩阵形状为 `spots x target_genes`。

记录的指标包括：

- MSE
- MAE
- RMSE
- Pearson All：所有 spot-gene target entries 拉平后的全局 Pearson correlation
- Spot Pearson Mean：逐 spot 计算 Pearson 后取平均
- Spot Spearman Mean：逐 spot 计算 Spearman 后取平均
- Gene-wise Pearson/Spearman 也会记录在 `summary.json` 中，但不展示在 Table 2 里。

相关实现：

- `common.py`：`summarize_target_metrics`

## scUniFlow 设置

主实验 run：

- 训练目录：`runs/scuniflow_spatial_head_ST8059048_v5`
- 最终全 spot 评估目录：`runs/scuniflow_spatial_head_ST8059048_v5/eval_all_2987`
- Backbone 实验：`${SCUNIFLOW_RESULTS:-results}/pretrain/mouse_brain_snrna_augmented_177_run`
- Backbone checkpoint：`${SCUNIFLOW_RESULTS:-results}/pretrain/mouse_brain_snrna_augmented_177_run/checkpoints/checkpoint_best.pt`
- 表达空间：`logcpm`
- Observed genes：`1024`
- Target genes：`256`
- Train/validation split：`2390/597` random spots，seed `42`
- Epochs：`20`
- Batch size：`24`
- Head learning rate：`1e-3`
- Backbone learning rate：`1e-5`，但除非显式开启，否则 backbone 模块保持冻结
- `train_mamba=False`
- `train_router=False`
- Weight decay：`1e-4`
- Head hidden dimension：`256`
- Head dropout：`0.1`
- Loss：MSE + `0.1 * (1 - global Pearson)` + `0.1 * (1 - mean spot Pearson)`
- Gene Pearson loss weight：`0.0`
- Best checkpoint 选择标准：validation loss

task head 的输入包括 backbone 产生的 target-gene context、masked/visible expression vector、可见基因总 count 的 log 值，以及归一化后的二维 spot 坐标。

Table 2 结果来自 `eval_all_2987/summary.json`：

| MAE | RMSE | Pearson All | Spot Pearson Mean | Spot Spearman Mean |
|---:|---:|---:|---:|---:|
| 2.3546 | 3.2691 | 0.6058 | 0.5770 | 0.5301 |

## SpaGE 设置

运行脚本：

- `run_spage_mouse_brain.py`

主要输出：

- `runs/mouse_brain_benchmarks/spage`
- 等价 full run 也存在于 `runs/mouse_brain_baselines_full_v1/spage_full`

设置：

- Spatial input：`1024` 个 observed genes 的 log-CPM 矩阵
- Reference input：`1024` 个 observed genes + `256` 个 target genes 的 log-CPM 矩阵
- Reference cells：`8000`
- Principal vectors：`n_pv=30`
- 通过 `genes_to_predict` 指定需要预测的 spatial target genes
- 不使用 supervised spot train/validation split

Table 2 结果：

| MAE | RMSE | Pearson All | Spot Pearson Mean | Spot Spearman Mean |
|---:|---:|---:|---:|---:|
| 2.4769 | 4.1398 | 0.3303 | 0.2917 | 0.3588 |

## gimVI 设置

运行脚本：

- `run_gimvi_mouse_brain.py`

主要输出：

- `runs/mouse_brain_baselines_full_v1/gimvi_full`

设置：

- Reference AnnData 使用 observed + target genes 的 raw counts。
- Spatial AnnData 只使用 observed genes 的 raw counts。
- 模型：`scvi.external.GIMVI`
- Generative distributions：`["zinb", "nb"]`
- Model library size flags：`[True, False]`
- Latent dimension：`n_latent=10`
- Training epochs：`max_epochs=200`
- Batch size：`128`
- Device：脚本中为 `auto`，由 scvi 解析可用 accelerator
- Imputation：使用 `model.get_imputed_values` 的 deterministic normalized values
- 不使用 supervised spot train/validation split

Table 2 结果：

| MAE | RMSE | Pearson All | Spot Pearson Mean | Spot Spearman Mean |
|---:|---:|---:|---:|---:|
| 2.4792 | 4.7979 | 0.1949 | 0.1935 | 0.3480 |

## Tangram 设置

运行脚本：

- `run_tangram_mouse_brain.py`

主要输出：

- `runs/tangram_mouse_brain/ST8059048`
- `runs/mouse_brain_baselines_full_v1/tangram_full`

设置：

- Reference AnnData 使用所有对齐基因的 log-CPM。
- Spatial mapping AnnData 只使用 observed genes 的 log-CPM。
- 传给 Tangram 的 training genes：`1024` 个 observed genes。
- Mapping mode：`cells`
- Density prior：`rna_count_based`
- Epochs：`num_epochs=500`
- Learning rate：`0.1`
- Device：full run 使用 `cuda`
- Target gene predictions 由 `tg.project_genes` 得到
- 不使用 supervised spot train/validation split

Table 2 结果：

| MAE | RMSE | Pearson All | Spot Pearson Mean | Spot Spearman Mean |
|---:|---:|---:|---:|---:|
| 2.7123 | 4.1857 | 0.3150 | 0.2820 | 0.3675 |

## Seurat 设置

运行脚本：

- Python wrapper：`run_seurat_mouse_brain.py`
- R runner：`run_seurat_mouse_brain.R`

主要输出：

- `runs/mouse_brain_baselines_full_v1/seurat_full`

设置：

- Python wrapper 负责准备统一 benchmark 数据，并写出 R 输入文件。
- R 输入文件包括 spatial observed raw counts、reference observed+target raw counts 和 panel metadata。
- Seurat object 由转置后的 raw count matrix 创建。
- Reference 和 spatial object 都执行 `NormalizeData`。
- Anchor features：reference 与 spatial 的 observed genes 交集。
- Anchor method：`FindTransferAnchors(..., reduction="cca")`
- Transfer target data：reference 中 `256` 个 target genes 的 normalized expression
- Transfer method：`TransferData(..., weight.reduction="cca")`
- 不使用 supervised spot train/validation split

Table 2 结果：

| MAE | RMSE | Pearson All | Spot Pearson Mean | Spot Spearman Mean |
|---:|---:|---:|---:|---:|
| 2.5040 | 4.4442 | 0.2863 | 0.2689 | 0.3464 |

## LIGER 设置

运行脚本：

- Python wrapper：`run_liger_mouse_brain.py`
- R runner：`run_liger_mouse_brain.R`

主要输出：

- `runs/mouse_brain_benchmarks/liger`

设置：

- Python wrapper 负责准备统一 benchmark 数据，并写出 R 输入文件。
- R 输入文件包括 spatial observed raw counts、reference observed+target raw counts 和 panel metadata。
- 已完成 run 使用的 R 环境：`${RSCRIPT:-Rscript}`
- R package：脚本中优先检测 `liger`，若无则使用 `rliger`
- LIGER object datasets：`spatial` 和 `rna`
- Normalization：`normalize`
- Variable features：固定为 `1024` 个 observed genes
- Scaling：`scaleNotCenter`
- Integration：`runIntegration(method="iNMF", k=20, lambda=20)`，其中 `k` 会根据数据集大小做上限截断
- Quantile normalization：`nNeighbors=20`、`quantiles=50`、`maxSample=1000`、`refineKNN=TRUE`
- Prediction：在 LIGER embedding 中，从每个 spatial spot 找 reference cell 近邻并做加权平均
- Prediction KNN：`k=30`
- Neighbor weights：距离倒数权重，并按 spot 归一化
- Target expression source：reference target log-CPM
- 若 `rliger` 丢弃没有 observed-gene signal 的 spatial spots，脚本会用 reference target-gene mean 补回缺失 spot，保证输出矩阵和原始 benchmark spot 顺序一致。
- 不使用 supervised spot train/validation split

Table 2 结果：

| MAE | RMSE | Pearson All | Spot Pearson Mean | Spot Spearman Mean |
|---:|---:|---:|---:|---:|
| 2.6318 | 4.1944 | 0.2845 | 0.2810 | 0.3424 |

## Table 2 数值汇总

| Method | MAE | RMSE | Pearson All | Spot Pearson Mean | Spot Spearman Mean |
|---|---:|---:|---:|---:|---:|
| scUniFlow | 2.3546 | 3.2691 | 0.6058 | 0.5770 | 0.5301 |
| SpaGE | 2.4769 | 4.1398 | 0.3303 | 0.2917 | 0.3588 |
| Tangram | 2.7123 | 4.1857 | 0.3150 | 0.2820 | 0.3675 |
| gimVI | 2.4792 | 4.7979 | 0.1949 | 0.1935 | 0.3480 |
| Seurat | 2.5040 | 4.4442 | 0.2863 | 0.2689 | 0.3464 |
| LIGER | 2.6318 | 4.1944 | 0.2845 | 0.2810 | 0.3424 |

## 可复现实验文件

核心脚本：

- `common.py`
- `benchmark_common.py`
- `spatial_head_common.py`
- `train_scuniflow_spatial_gene_imputation_head.py`
- `eval_scuniflow_spatial_gene_imputation_head.py`
- `run_spage_mouse_brain.py`
- `run_gimvi_mouse_brain.py`
- `run_tangram_mouse_brain.py`
- `run_seurat_mouse_brain.py`
- `run_seurat_mouse_brain.R`
- `run_liger_mouse_brain.py`
- `run_liger_mouse_brain.R`

关键结果记录：

- scUniFlow run args：`runs/scuniflow_spatial_head_ST8059048_v5/run_args.json`
- scUniFlow spot split：`runs/scuniflow_spatial_head_ST8059048_v5/split_info.json`
- scUniFlow all-spot eval：`runs/scuniflow_spatial_head_ST8059048_v5/eval_all_2987/summary.json`
- SpaGE summary：`runs/mouse_brain_benchmarks/spage/summary.json`
- gimVI summary：`runs/mouse_brain_baselines_full_v1/gimvi_full/summary.json`
- Tangram summary：`runs/mouse_brain_baselines_full_v1/tangram_full/summary.json`
- Seurat summary：`runs/mouse_brain_baselines_full_v1/seurat_full/summary.json`
- LIGER summary：`runs/mouse_brain_benchmarks/liger/summary.json`
