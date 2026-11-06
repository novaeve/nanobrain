# scUniFlow scATAC Pipeline 详细说明

## 1. 这个任务到底在做什么

这个任务的目标是：  
基于 `scUniFlow` 现有框架，实现一个面向 `scATAC-seq peak accessibility imputation` 的训练与验证流程。

更直白地说，就是：

- 输入是一批单细胞 ATAC 数据
- 每个细胞有一长串 peak accessibility 特征
- 这些特征天然非常稀疏，很多位置是 0
- 这些 0 里面既有“真的关闭”，也有“本来开放但没测到”的假阴性
- 模型要学会根据同一个细胞里其余 peak 的模式，去恢复被遮住或没测到的 peak accessibility

这个任务本质上不是做分类，而是在做：

- 高维连续特征恢复
- 稀疏单细胞表观组学补全
- 利用局部结构与全局模式做 masked imputation

---

## 2. 为什么可以基于 scUniFlow 来做

`scUniFlow` 原本是为 scRNA 训练设计的，但它真正依赖的核心假设不是“输入一定是基因表达”，而是：

1. 输入是一个高维连续向量
2. 每个特征都可以被映射到一个固定的 2D 网格位置
3. 模型学习的是：从部分观测和噪声状态恢复目标值

只要 scATAC 数据也能被表示成连续值空间，整个主框架就仍然成立。

因此本次改造的核心思路是：

- 保留 `Flow Matching`
- 保留 `SpectralRouter`
- 保留 `GeneMamba`
- 保留 `VectorFieldNet`
- 保留现有训练范式
- 只替换 scRNA 特有的数据定义、mask 策略和验证指标

换句话说，这不是“重写模型”，而是“给现有模型接一条 scATAC 专用数据管线”。

---

## 3. 为什么 scATAC 不直接做 0/1 二分类

原始 scATAC 数据常被理解为 open / closed 的二值任务，但这里没有直接把它做成 binary classification，而是先把原始 counts 转成 `TF-IDF` 连续值，再做恢复。

这样做的原因有三层：

### 3.1 Flow Matching 适合连续空间

scUniFlow 当前训练的核心是连续插值：

`x_t = (1 - t) * x_0 + t * x_1`

这个定义天然属于连续空间。如果目标值是严格离散的 0/1，那么：

- 插值空间不自然
- 损失函数要重写
- 推理过程要改
- 整体工程复杂度会大很多

### 3.2 TF-IDF 让 ATAC 进入可训练的连续值空间

对 ATAC counts 做 TF-IDF 后，每个 peak 在每个细胞里的值变成连续正数，语义接近：

- 该 peak 在该细胞中的相对重要性
- 该 peak 的可及性强度

这与 scRNA 中常用的 `log1p(expression)` 在建模形式上是兼容的。

### 3.3 训练做连续恢复，评估再看 open/closed 判别

这里的策略是：

- 训练阶段：预测连续 TF-IDF 值
- 验证阶段：再把真实值按 `> 0` 转成二值 open/closed
- 用 AUROC / AUPRC 评估模型是否能把开放 peak 排在更前面

这样既保留了连续恢复能力，也保留了对二值生物学问题的评估。

---

## 4. 整体 pipeline 的结构

整个 scATAC pipeline 可以分成四大阶段：

1. 原始数据预处理
2. 路由坐标生成
3. 训练样本构造
4. 模型训练与验证

下面按执行顺序展开。

---

## 5. 阶段 A：原始 ATAC 数据预处理

实现脚本：

- [prepare_atac_h5ad.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/prepare_atac_h5ad.py)

### 5.1 输入

输入可以是：

- 10x `filtered_feature_bc_matrix.h5`
- 已有 `.h5ad`

如果输入是 10x 多组学矩阵，脚本会只保留：

- `feature_types == "Peaks"`

也就是说，最后只取 scATAC 那一部分。

### 5.2 做了什么

预处理脚本主要做下面几件事：

1. 读取原始 ATAC 矩阵
2. 过滤低质量细胞
3. 过滤低频 peak
4. 对矩阵做 TF-IDF
5. 解析 peak 的基因组坐标
6. 保存为新的 `.h5ad`

### 5.3 过滤规则

当前默认规则是：

- `min_peaks_per_cell = 200`
- `min_cells_per_peak = 10`

它们的作用分别是：

- 去掉 peak 数太少的细胞
- 去掉几乎不在细胞中出现的 peak

### 5.4 TF-IDF 变换的意义

TF-IDF 的目的是把原始 count 变成更适合建模的连续值。

它会同时考虑：

- 一个 peak 在当前 cell 中的相对占比
- 一个 peak 在所有 cell 中的普遍程度

结果是：

- 高频、缺乏区分度的 peak 权重变低
- 对特定细胞更有辨识度的 peak 权重更高

### 5.5 peak 坐标解析

脚本会把 peak 名称从：

- `chr1:10000-10500`

解析成：

- `chrom`
- `start`
- `end`

这些信息会写进 `adata.var`，供下一步生成 routing coordinates 使用。

### 5.6 输出

输出是一个新的 TF-IDF `.h5ad` 文件，例如：

- `pbmc_10k_atac_tfidf.h5ad`

这个文件的关键特征是：

- `.X` 中已经是连续 TF-IDF 值
- 不是原始 count
- 不是 binary 0/1
- `var` 里已经包含基因组坐标信息

---

## 6. 阶段 B：生成 scATAC 的 2D 路由坐标

实现脚本：

- [generate_atac_coords.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/generate_atac_coords.py)

### 6.1 为什么需要这一步

scUniFlow 不是直接把高维向量丢进普通 MLP，而是先把每个特征映射到 2D grid 上，再用 `GeneMamba` 在这个网格上做结构化扫描。

这一步需要每个 feature 都有一个固定位置。

在 scRNA 版本里，这个位置来自：

- STRING graph 的 spectral coordinates

但 scATAC peak 没有现成的 STRING 网络关系，所以这里换成：

- 基于 peak 基因组位置的坐标

### 6.2 坐标设计逻辑

这里的直觉是：

- 同一染色体上的相邻 peak，往往更可能具有相关染色质状态
- 如果它们在 2D 网格上也相邻，Mamba 的局部扫描就更容易学到这类模式

因此定义：

- 横轴：染色体顺序
- 纵轴：该 peak 在染色体内的相对位置

例如：

- `chr1` 靠左
- `chr22` 更靠右
- 同一条染色体内，start 小的靠上，start 大的靠下

### 6.3 一个很关键的实现细节

设计文档里写的是生成 `[0,1]^2` 坐标，但当前 `scUniFlow/model.py` 实际上会把坐标直接当成“网格坐标”来使用。

这意味着如果真的把所有 peak 坐标保存到 `[0,1]` 范围：

- 大量 peak 会在取整后挤到同一个很小的区域
- 路由质量会严重退化

所以现在脚本实际输出的是：

- 缩放到 `[0, grid_size - 1]` 的坐标

这一步是为了和当前模型实现严格对齐。

### 6.4 输出

输出文件是：

- `*.npy`：坐标矩阵，shape 为 `(n_peaks, 2)`
- `*.json`：元信息

其中每一行表示一个 peak 的二维坐标。

---

## 7. 阶段 C：把一个细胞构造成训练样本

实现文件：

- [data_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/data_atac.py)

这一层负责把预处理后的 ATAC 矩阵变成模型训练时真正使用的数据。

### 7.1 数据集读入

这里做了两种加载方式：

- `load_atac_h5ad`
- `load_atac_h5ad_split`

支持：

- 直接读单个 h5ad
- donor-level train/val split
- 如果没有 donor 信息，则退化为 cell-level random split

### 7.2 一个细胞会被拆成哪些对象

对每个细胞，最终会构造：

- `g_id`
- `x_1`
- `x_0`
- `mask_indicator`

它们分别代表：

#### `g_id`

每个 peak 的全局 feature id。  
模型需要知道每一列特征是谁。

#### `x_1`

真实目标值，也就是该 cell 的完整 peak accessibility 向量。  
这里默认是：

- TF-IDF 连续值
- 使用 `identity` 变换直接送入训练

#### `x_0`

初始先验噪声向量。  
它不是简单固定为 0，而是从经验高斯分布里采样。

#### `mask_indicator`

哪些位置要被遮住，让模型去恢复。

---

## 8. 阶段 D：`x_1`、`x_0` 和 mask 是怎么定义的

### 8.1 `x_1` 的定义

对于 scRNA，原始管线通常会做：

- `log1p(raw_counts)`

对于 scATAC，这里改成：

- 直接使用 TF-IDF 值

也就是：

- `x_1 = TF-IDF peak accessibility`

这是本次改造最核心的变化之一。

### 8.2 `x_0` 的定义

模型训练不是从空白开始，而是从一个噪声先验状态出发。

这里的做法是：

1. 在训练集里收集所有非零 TF-IDF 值
2. 计算它们的：
   - `prior_mean`
   - `prior_std`
3. 采样：
   - `x_0 ~ N(prior_mean, prior_std^2)`

这样做的好处是：

- 起点分布更贴近 scATAC 真实数值范围
- 比纯标准正态更稳定

### 8.3 为什么要 mask 非零值

这部分比较直观：

- 非零 peak 代表已观测到的开放状态
- 随机遮掉一部分，强迫模型根据上下文去恢复

当前默认：

- `nonzero_mask_ratio = 0.2`

也就是遮掉 20% 的非零 peak。

### 8.4 为什么还要 mask 一部分零值

这是 scATAC 比 scRNA 更重要的一点。

在 scATAC 里，0 不一定表示真关闭，它可能是：

- 真关闭
- dropout 假阴性

如果训练时从不 mask 零值，那么模型永远不会被要求处理“疑似缺失的关闭位点”，只能学会恢复本来就可见的开放位点。

所以当前默认还会：

- mask 10% 的零值 peak

也就是：

- `zero_mask_ratio = 0.1`

这会让模型被迫学习更接近真实 scATAC imputation 的问题。

---

## 9. 阶段 E：模型训练具体在做什么

实现文件：

- [train_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/train_atac.py)

训练过程本质上沿用原始 scUniFlow 的单阶段 epoch 训练。

### 9.1 保留了哪些原始模块

这里没有改动原始模型主体：

- [model.py](${SCUNIFLOW_ROOT:-.}/model.py)
- [gene_mamba.py](${SCUNIFLOW_ROOT:-.}/gene_mamba.py)

也就是说，scATAC 训练依然使用：

- `SpectralRouter`
- `GeneMamba`
- `VectorFieldNet`

### 9.2 训练时的一条样本路径

每次训练，一个 batch 大致经历以下过程：

1. 从数据集读取一批 cell
2. 得到每个 cell 的：
   - `x_1`
   - `x_0`
   - `mask_indicator`
3. 采样一个时间 `t`
4. 构造 flow matching 中间状态
5. 把特征按 routing coordinates 放进 2D 网格
6. 用 `GeneMamba` 编码上下文
7. 预测被 mask 位置的目标值
8. 计算损失
9. 反向传播更新参数

### 9.3 损失函数

目前沿用原管线里的组合损失，包括：

- weighted reconstruction loss
- Pearson loss
- cell-wise Pearson loss

它们仍然适用于 scATAC，因为目标空间仍然是连续值。

---

## 10. 推理时模型到底在恢复什么

训练结束后，模型做的事情不是单纯判别某个 peak 是 open 还是 closed，而是：

- 给出每个被遮住 peak 的连续预测值

这个值可以理解为：

- 该 peak 在该细胞里的预测 accessibility 强度

之后如果你想做二值层面的分析，可以再根据需要阈值化，但训练本身不强制离散化。

---

## 11. 验证阶段在评估什么

验证逻辑也沿用原始框架，但增加了更适合 scATAC 的指标。

### 11.1 保留的连续值恢复指标

这些指标衡量的是：

- 模型对真实 TF-IDF 值恢复得准不准
- 排序关系保留得好不好

包括：

- `val_mse`
- `val_rmse`
- `val_mae`
- `val_pearson_cell`
- `val_pearson_all`
- `val_spearman_cell`
- `val_pearson_top20`

### 11.2 新增的 scATAC 二值判别指标

新增：

- `val_auroc`
- `val_auprc`

计算逻辑是：

1. 收集所有 mask 位置上的真实值和预测值
2. 把真实值二值化：
   - `TF-IDF > 0` 记为 open
   - `TF-IDF == 0` 记为 closed
3. 预测值仍保持连续
4. 用连续预测值对 open / closed 做排序评估

这两个指标的意义分别是：

#### AUROC

模型能否总体上把开放 peak 排得比关闭 peak 更高。

#### AUPRC

在 ATAC 极度稀疏、正例稀少的情况下，更敏感地衡量模型对开放 peak 的识别能力。

### 11.3 为什么要同时保留两类指标

因为 scATAC imputation 有两层目标：

1. 恢复连续 accessibility 强度
2. 区分 open / closed 的生物学状态

只看相关性不够，只看分类也不够。  
所以现在验证会同时看：

- 连续恢复质量
- 二值开放状态判别能力

---

## 12. best checkpoint 是怎么选的

`train_atac.py` 里沿用了原始训练的“多数指标改进”策略，但把新指标也纳入比较。

也就是说，在 scATAC 训练里，best checkpoint 会综合考虑：

- 误差类指标是否下降
- 相关性类指标是否上升
- `val_auroc` 是否上升
- `val_auprc` 是否上升

这样比单独盯住某一个指标更稳妥。

---

## 13. donor split 在这个任务中的作用

如果数据里有：

- `obs["donor_id"]`

那么数据加载时会优先使用 donor-level train/val split。

这意味着：

- 训练集和验证集尽量来自不同 donor
- 验证更接近跨 donor 泛化能力

如果没有 donor 信息，则退化为：

- cell-level 随机划分

---

## 14. 这次实现和原项目的关系

这次实现不是直接去改原有 `scUniFlow` 的主入口，而是新建了一套 scATAC 专用分支，全部放在：

- [${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation)

核心文件包括：

- [data_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/data_atac.py)
- [train_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/train_atac.py)
- [launch_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/launch_atac.py)
- [prepare_atac_h5ad.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/prepare_atac_h5ad.py)
- [generate_atac_coords.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/generate_atac_coords.py)
- [config_atac.yaml](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/config_atac.yaml)

这样做的目的很明确：

- 不破坏原始 scRNA 训练路径
- 保持旧代码可用
- 把 scATAC 改造控制在新增文件范围内

---

## 15. 用一句话概括整条 pipeline

这条 pipeline 做的事情可以概括为：

先把原始 scATAC peak count 预处理成 TF-IDF 连续值，再根据 peak 的基因组位置生成固定 2D 路由坐标，把每个细胞的 peak accessibility 向量构造成 flow matching 训练样本，通过 mask 一部分非零和零值 peak，让 scUniFlow 学会根据上下文恢复被遮住的 accessibility，最后用连续恢复指标和 open/closed 判别指标共同评估模型效果。

---

## 16. 实际执行顺序

如果要真正跑这条 pipeline，建议按下面顺序执行：

### Step 1：准备 TF-IDF h5ad

```bash
python scATAC/prepare_atac_h5ad.py \
  --input /path/to/filtered_feature_bc_matrix.h5 \
  --output /path/to/pbmc_10k_atac_tfidf.h5ad \
  --top-n-peaks 5000
```

### Step 2：生成 2D 坐标

```bash
python scATAC/generate_atac_coords.py \
  --h5ad /path/to/pbmc_10k_atac_tfidf.h5ad \
  --grid_size 72 \
  --output_npy /path/to/top5k_atac_grid72_coords.npy
```

### Step 3：启动训练

```bash
python scATAC/launch_atac.py \
  --config scATAC/config_atac.yaml \
  --data /path/to/pbmc_10k_atac_tfidf.h5ad \
  --spectral_coords /path/to/top5k_atac_grid72_coords.npy \
  --run_name atac_top5k_smoke
```

---

## 17. 建议的实验推进方式

建议按三阶段推进：

### 第一阶段：top-5k peak smoke test

目标：

- 检查 pipeline 是否连通
- 检查是否有 shape mismatch / NaN / 路由错误

### 第二阶段：top-50k peak 中等规模实验

目标：

- 检查模型是否真正学到 accessibility 结构
- 观察 AUROC / AUPRC / Pearson 是否稳定提升

### 第三阶段：全量 peak 正式实验

目标：

- 获得最终模型
- 与 baseline 做正式对比

---

## 18. 当前准备使用的数据集与下载清单

当前准备先用于跑通 pipeline 的数据集信息如下：

- 数据集类型：10x Multiome
- 物种：小鼠
- 组织阶段：Mouse E18 brain
- 数据特征：同一份矩阵里同时包含 GEX 和 ATAC

对于当前这条 scATAC pipeline，实际上只需要下载两个文件，总体不到 100 MB。

### 18.1 必须下载的文件

#### 1. Filtered feature barcode matrix (HDF5)

- 大小：约 89.2 MB
- 作用：主数据矩阵，包含所有 `cell × feature` 的 count 数据
- 为什么必须下：这份 `.h5` 同时包含 gene expression 和 ATAC peaks，`prepare_atac_h5ad.py` 会通过 `sc.read_10x_h5(..., gex_only=False)` 读取它，再根据 `feature_types == "Peaks"` 提取 ATAC 部分

也就是说，这个文件是：

- Step 0 预处理的直接输入
- 训练数据矩阵的来源

#### 2. ATAC peak locations (BED)

- 大小：约 4.12 MB
- 作用：提供每个 peak 的基因组坐标信息，格式通常为：
  - `chrom start end`
- 为什么必须下：这是生成 scATAC 路由坐标最直接、最可靠的来源

有了这个文件，就可以明确知道每个 peak 的：

- 染色体
- 起始位置
- 终止位置

然后就可以构造二维坐标：

- `chrom_idx`
- `pos_norm`

这正是 Step 1 生成 routing / spectral coordinates 所需要的底层信息。

### 18.2 当前不需要下载的文件

下面这些文件都不是当前跑通 scATAC imputation pipeline 的必要条件：

| 文件 | 为什么不需要 |
|------|--------------|
| BAM / BAI（GEX + ATAC，共约 47 GB） | 原始比对文件，count 矩阵已经从它们生成出来了 |
| Fragment file (`.tsv.gz`，约 1.09 GB) | 原始 fragment 级数据，peak calling 已经完成 |
| BIGWIG（约 977 MB） | 主要用于可视化，不参与训练 |
| Raw feature barcode matrix (HDF5) | 未经过 cell filtering，当前优先用 filtered 版本即可 |
| Secondary analysis outputs | 主要是 Cell Ranger 下游分析结果，不影响 imputation 训练 |
| Loupe Browser 文件（约 761 MB） | 仅供 10x 可视化软件使用，不参与模型训练 |

### 18.3 这个数据集和设计文档的一个关键差异

之前设计文档里默认使用的人类 hg38 染色体长度不适用于这个数据集。

原因是：

- 设计文档默认参考的是人类数据
- 当前准备使用的数据集是小鼠胚胎脑（Mouse E18）
- 因此染色体长度应该改成小鼠 `mm10`

当前应使用的小鼠 `mm10` 染色体长度如下：

```python
CHROM_LENGTHS = {
    "chr1": 195471971, "chr2": 182113224, "chr3": 160039680,
    "chr4": 156508116, "chr5": 151834684, "chr6": 149736546,
    "chr7": 145441459, "chr8": 129401213, "chr9": 124595110,
    "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
    "chr13": 120421639, "chr14": 124902244, "chr15": 104043685,
    "chr16": 98207768,  "chr17": 94987271,  "chr18": 90702639,
    "chr19": 61431566,  "chrX": 171031299,  "chrY": 91744698,
}
```

### 18.4 对当前实现的影响

当前仓库里的 [generate_atac_coords.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/generate_atac_coords.py) 目前内置的是人类染色体长度表，因此如果后面直接用这个 Mouse E18 数据集训练，需要在正式生成坐标前切换为小鼠 `mm10` 版本，或者把脚本改成可配置参考基因组。

也就是说，当前这份数据集下载完成后，下一步不是立刻开始大规模训练，而是：

1. 先用下载好的 `.h5` 生成 TF-IDF `.h5ad`
2. 确认 peak 坐标来源
3. 将坐标脚本中的 `CHROM_LENGTHS` 切到小鼠 `mm10`
4. 再生成 scATAC 路由坐标
5. 最后启动 smoke test

### 18.5 BED 文件的额外价值

虽然当前预处理脚本也支持从 peak 名称字符串里解析：

- `chr:start-end`

但对这个数据集而言，BED 文件仍然更可靠，因为：

- BED 是官方提供的标准 peak 坐标表
- 不依赖 peak 名称字符串格式完全一致
- 更适合作为 `.var["chrom"] / .var["start"] / .var["end"]` 的外部基准

因此，当前建议是：

- `.h5` 作为主矩阵输入
- `.bed` 作为 peak 坐标参考输入

---

## 19. 当前实现里最关键的设计点

如果只保留最重要的几点，可以记住下面这几条：

1. 训练目标不是 binary ATAC，而是 TF-IDF 连续值恢复。
2. 原始模型骨架不动，改的是数据语义和验证逻辑。
3. `zero_mask_ratio = 0.1` 是 scATAC 任务成立的关键之一。
4. routing coordinates 必须输出成网格尺度，而不是简单 `[0,1]`。
5. 验证必须同时看连续恢复指标和 AUROC/AUPRC。

---

## 20. 相关实现文件

- 设计文档：[scUniFlow_scATAC_Adaptation_Design.md](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/scUniFlow_scATAC_Adaptation_Design.md)
- 本文档：[scUniFlow_scATAC_Pipeline_Overview.md](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/scUniFlow_scATAC_Pipeline_Overview.md)
- 数据处理：[data_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/data_atac.py)
- 预处理脚本：[prepare_atac_h5ad.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/prepare_atac_h5ad.py)
- 坐标脚本：[generate_atac_coords.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/generate_atac_coords.py)
- 训练入口：[train_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/train_atac.py)
- 启动器：[launch_atac.py](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/launch_atac.py)
- 配置模板：[config_atac.yaml](${SCUNIFLOW_ROOT:-.}/src/scuniflow/tasks/task3_atac_imputation/config_atac.yaml)
