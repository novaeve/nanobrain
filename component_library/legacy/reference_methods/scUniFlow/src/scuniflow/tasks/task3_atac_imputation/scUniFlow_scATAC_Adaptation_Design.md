# scUniFlow → scATAC-seq Imputation 改造设计文档

**文档版本**：v1.0  
**基础模型**：scUniFlow（TRAINING_PIPELINE.md 描述的当前实现）  
**目标任务**：scATAC-seq peak accessibility imputation  
**推荐数据集**：10x PBMC Multiome 10k（paired RNA + ATAC）

---

## 目录

1. [改造总体策略](#1-改造总体策略)
2. [Step 0：数据预处理（离线）](#2-step-0数据预处理离线)
3. [Step 1：Spectral Coordinates 重新生成（离线）](#3-step-1spectral-coordinates-重新生成离线)
4. [Step 2：`data.py` 修改](#4-step-2datapy-修改)
5. [Step 3：`config.yaml` 修改](#5-step-3configyaml-修改)
6. [Step 4：`train.py` 修改（新增验证指标）](#6-step-4trainpy-修改新增验证指标)
7. [不需要修改的部分](#7-不需要修改的部分)
8. [验证顺序建议（分阶段执行）](#8-验证顺序建议分阶段执行)
9. [后续扩展路径](#9-后续扩展路径)
10. [改动文件汇总](#10-改动文件汇总)

---

## 1. 改造总体策略

### 1.1 核心判断：保留 Flow Matching 框架，替换 x₁ 的定义

当前 scUniFlow 的 Flow Matching 框架（插值、x-prediction、Euler rollout）对 scATAC 完全适用，**前提是将 `x_1` 定义为 TF-IDF 归一化后的 peak accessibility 连续值**，而非 binary 0/1。

这个选择的逻辑链如下：

- TF-IDF(peak) 产出连续正值，与 log1p(gene) 在数学上同构
- 现有 Huber + Pearson 损失对连续值完全有效，无需改成 BCE 或 cross-entropy
- Flow Matching 的线性插值 `x_t = (1-t)x_0 + t·x_1` 只在连续空间成立；若改为 binary 建模，需换 discrete flow，工作量会增加 3-5 倍
- Binary accessibility 可在**验证阶段**通过阈值化连续预测来评估，无需改动训练目标

### 1.2 改动范围总览

| 模块 | 改动类型 | 改动内容概述 |
|------|----------|-------------|
| 离线预处理脚本 | 新增 | 生成 TF-IDF h5ad + genomic spectral coords |
| `data.py` | 修改 | x₁ 定义、prior 统计量、zero_mask_ratio |
| `config.yaml` | 修改 | n_genes、grid_size、mask 比例 |
| `train.py` | 修改 | 新增 AUROC、AUPRC 验证指标 |
| `model.py` | 不动 | 架构完全复用 |
| `gene_mamba.py` | 不动 | 编码器完全复用 |

---

## 2. Step 0：数据预处理（离线）

这一步在启动任何训练之前完成，产出一个标准 `h5ad` 文件，格式与现有 scRNA 数据一致。

### 2.1 输入

10x PBMC Multiome 10k 的 ATAC fragment 文件或 Cell Ranger ATAC 输出的 peak × cell count matrix（`.h5` 格式）。

### 2.2 预处理流程

```python
import scanpy as sc
import numpy as np
from sklearn.preprocessing import TfidfTransformer

# 1. 读取原始 peak × cell count matrix
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5", gex_only=False)
adata_atac = adata[:, adata.var["feature_types"] == "Peaks"].copy()

# 2. 基本质控：去掉 cell 中 open peak 数量过少的 cell，以及极低频 peak
sc.pp.filter_cells(adata_atac, min_genes=200)
sc.pp.filter_genes(adata_atac, min_cells=10)

# 3. TF-IDF 归一化（核心步骤）
# TF  = 每个 cell 中该 peak 的原始 count / 该 cell 所有 peak 总 count
# IDF = log(1 + 总 cell 数 / 含该 peak 的 cell 数)
# 最终值为连续正值，语义为"该 peak 在该 cell 中的相对 accessibility 重要性"
tfidf = TfidfTransformer(norm="l1", use_idf=True, smooth_idf=True)
X_tfidf = tfidf.fit_transform(adata_atac.X).toarray().astype(np.float32)

# 将 TF-IDF 结果作为 .X 存回
adata_atac.X = X_tfidf

# 4. 保存 peak 的基因组坐标到 .var（供后续生成 spectral coords 使用）
# peak 名称格式通常为 "chr1:10000-10500"
# 解析出 chrom, start, end 存入 .var
def parse_peak_name(peak_name):
    chrom, pos = peak_name.rsplit(":", 1)
    start, end = pos.split("-")
    return chrom, int(start), int(end)

chroms, starts, ends = [], [], []
for peak in adata_atac.var.index:
    c, s, e = parse_peak_name(peak)
    chroms.append(c)
    starts.append(s)
    ends.append(e)

adata_atac.var["chrom"] = chroms
adata_atac.var["start"] = starts
adata_atac.var["end"] = ends

# 5. 按 donor 分割（若 obs 中有 donor_id 字段则保留，训练时 data.py 会自动用它做 donor-level split）

# 6. 保存
adata_atac.write_h5ad("pbmc_10k_atac_tfidf.h5ad")
print(f"最终矩阵形状：{adata_atac.shape}")
# 预期输出：约 (10000, 107915)
```

### 2.3 关键注意事项

- `adata_atac.X` 中存的是 **TF-IDF 值（float32 连续值）**，不是原始 count，也不是 binary 0/1
- peak 名称格式必须与 Cell Ranger 输出一致（`chr:start-end`），否则第 Step 1 中的坐标解析会出错
- 若内存不足，可先将 peak 按方差 top-N（如 top-50000）过滤后再保存，用于 proof-of-concept 阶段

---

## 3. Step 1：Spectral Coordinates 重新生成（离线）

当前框架使用 STRING gene graph 来生成 2D spectral coordinates（储存在 `spectral_coords.npy`）。scATAC 中无法使用 STRING，需要用基因组坐标替代。

### 3.1 方案说明

利用每个 peak 在基因组上的物理位置直接生成 2D 坐标：

- **x 轴（坐标 0）**：染色体编号归一化值。chr1 → 0.0 … chrX → 0.95，chrY → 1.0
- **y 轴（坐标 1）**：该 peak 在所在染色体内的相对位置（start / chrom_total_length）

这样做的优势：Hilbert 路由会把**基因组上物理相邻的 peak 分配到网格中相邻的 slot**，而相邻 peak 往往共享转录因子结合位点和染色质状态，对 Mamba 的局部上下文扫描非常有利。

### 3.2 生成脚本

```python
import numpy as np
import scanpy as sc

# 人类 hg38 各染色体长度（bp）
CHROM_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345,  "chr17": 83257441,  "chr18": 80373285,
    "chr19": 58617616,  "chr20": 64444167,  "chr21": 46709983,
    "chr22": 50818468,  "chrX": 156040895,  "chrY": 57227415,
}
# 染色体顺序（决定 x 轴编号）
CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHROM_IDX = {c: i / (len(CHROM_ORDER) - 1) for i, c in enumerate(CHROM_ORDER)}

adata = sc.read_h5ad("pbmc_10k_atac_tfidf.h5ad")
n_peaks = adata.n_vars
coords = np.zeros((n_peaks, 2), dtype=np.float32)

for i, (chrom, start) in enumerate(zip(adata.var["chrom"], adata.var["start"])):
    if chrom not in CHROM_IDX or chrom not in CHROM_LENGTHS:
        # 未知染色体（alt contigs 等）放到角落
        coords[i] = [1.0, 1.0]
        continue
    coords[i, 0] = CHROM_IDX[chrom]
    coords[i, 1] = start / CHROM_LENGTHS[chrom]

np.save("atac_spectral_coords.npy", coords)
print(f"Saved coords shape: {coords.shape}")
# 预期输出：(107915, 2)
```

### 3.3 输出文件

`atac_spectral_coords.npy`，形状 `(n_peaks, 2)`，值域 `[0, 1]²`。这个文件对应 `config.yaml` 中的 `training.spectral_coords` 路径。

### 3.4 可选升级（后续迭代）

若需要更生物学合理的 2D 坐标，可用 Cicero 计算 peak co-accessibility 分数，构建 peak × peak 相似矩阵，再用 Laplacian eigenmaps 取前 2 个特征向量作为坐标，替换上面的 genomic coords。代码路径不变，只需替换 `.npy` 文件。

---

## 4. Step 2：`data.py` 修改

### 4.1 修改 1：x₁ 的定义

**位置**：`scRNADataset.__getitem__` 中构造 `x_1` 的代码行

**原代码**：
```python
x_1 = np.log1p(x_raw)
```

**改为**：
```python
# x_raw 此时已经是 TF-IDF 值（float32），直接使用，不再做 log1p
x_1 = x_raw.astype(np.float32)
```

**说明**：TF-IDF 预处理已在离线数据准备阶段完成（见 Step 0）。训练时 `h5ad` 的 `.X` 中存的就是 TF-IDF 连续值，无需在 `__getitem__` 里再做变换。如果代码里的变量名 `x_raw` 会引起混淆，可以改名为 `x_tfidf`，但逻辑不变。

---

### 4.2 修改 2：x₀ prior 的统计量来源

**位置**：`_compute_empirical_prior` 函数（或等价函数，用于在训练集上统计 `prior_mean` 和 `prior_std`）

**原逻辑**：统计训练集中所有非零 `log1p` 表达值的均值和标准差

**改为**：
```python
# 统计训练集中所有非零 TF-IDF accessibility 值的均值和标准差
# "非零" 意味着该 peak 在该 cell 中是开放状态（accessible）
nonzero_mask = X_tfidf > 0  # shape: (n_cells, n_peaks)
nonzero_values = X_tfidf[nonzero_mask]
prior_mean = float(nonzero_values.mean())
prior_std  = float(nonzero_values.std())
```

**说明**：机制与原实现完全一致，只是采样对象从 `log1p(gene expression)` 换成了 `TF-IDF(peak accessibility)`。验证时复用训练集统计量，行为不变。

---

### 4.3 修改 3：zero_mask_ratio 的默认值

**位置**：`scRNADataset.__getitem__` 中的 mask 生成逻辑，以及传入 Dataset 的配置参数

**原值**：
```python
zero_mask_ratio = 0.0   # 从不 mask 零值基因
```

**改为**：
```python
zero_mask_ratio = 0.1   # mask 10% 的关闭状态 peak
```

**为什么必须改**：scATAC 数据的零值有两种来源：（1）该 peak 真的是关闭状态；（2）该 peak 是开放的但因 dropout 没有测到（假阴性）。imputation 的核心任务之一就是恢复这些假阴性。如果 `zero_mask_ratio=0.0`，模型永远没有机会学习"恢复被遮掉的开放区域"，只能学习恢复已知开放区域，严重低估了任务难度。

**最终推荐配置**：
```python
nonzero_mask_ratio = 0.2   # 保持不变
zero_mask_ratio    = 0.1   # 从 0.0 改为 0.1
```

---

## 5. Step 3：`config.yaml` 修改

以下是所有需要修改的配置项：

```yaml
# ============================================================
# 以下字段需要修改
# ============================================================

# 特征数量：从基因数改为 peak 数
n_genes: 107915        # 原值：32738
                       # 对应 10x PBMC Multiome 10k 的 peak 数量
                       # 如使用 top-50k peak 版本则改为 50000

# 网格尺寸：需满足 grid_size² ≥ n_peaks
grid_size: 329         # 原值：181
                       # 329² = 108241 ≥ 107915，满足条件
                       # 如使用 top-50k peak 版本则改为 224（224²=50176）

# spectral coords 文件路径：指向新生成的 atac 坐标文件
training:
  spectral_coords: /path/to/atac_spectral_coords.npy   # 原：gene spectral coords

# mask 策略：启用对关闭 peak 的 mask
training:
  nonzero_mask_ratio: 0.2   # 不变
  zero_mask_ratio: 0.1      # 原值：0.0，改为 0.1

# ============================================================
# 以下字段建议根据显存情况调整
# ============================================================

# 由于序列长度变长（~3.3x），显存压力增加，可适当降低 batch size
batch_size_per_gpu: 8     # 原值：25；如 GPU 显存为 40GB，从 25 降至 8-12 起步

# ============================================================
# 以下字段不需要修改
# ============================================================

d_model: 512              # 不变
n_layers: 8               # 不变
d_state_row: 64           # 不变
d_state_col: 16           # 不变
epochs: 80                # 不变，视情况可增加
lr: 5.0e-4                # 不变
min_lr: 1.0e-6            # 不变
```

### grid_size 计算参考

| peak 数量 | 推荐 grid_size | grid_size² | 说明 |
|-----------|---------------|-----------|------|
| 107,915   | 329           | 108,241   | 全量 peak，正式实验 |
| 50,000    | 224           | 50,176    | top-50k peak，消融/快速验证 |
| 5,000     | 72            | 5,184     | top-5k peak，pipeline 连通性验证 |

---

## 6. Step 4：`train.py` 修改（新增验证指标）

### 6.1 新增 AUROC 和 AUPRC

**位置**：`_run_validation` 函数，在计算完现有 Pearson/Spearman 指标后追加。

**新增依赖**：
```python
from sklearn.metrics import roc_auc_score, average_precision_score
```

**新增代码（插入到现有指标计算结束、return 语句之前）**：

```python
# ——————————————————————————————————————————————
# 新增：AUROC 和 AUPRC（只在 mask 位置计算）
# 将连续 TF-IDF 预测值二值化为"预测开放/关闭"来评估
# ——————————————————————————————————————————————

# 收集所有 mask 位置的真实值和预测值
# （假设 all_x1_true 和 all_x1_pred 是已经在循环中收集好的 tensor 或 numpy array）
y_true_all = all_x1_true.cpu().numpy().ravel()   # 真实 TF-IDF 值，shape: (N,)
y_pred_all = all_x1_pred.cpu().numpy().ravel()   # 预测 TF-IDF 值，shape: (N,)

# 二值化真实值：TF-IDF > 0 视为 open（accessible），否则视为 closed
y_true_bin = (y_true_all > 0).astype(np.float32)

# 计算 AUROC：评估模型区分 open vs closed peak 的能力
# AUPRC：更适合极稀疏数据，对正例（open peaks）更敏感
try:
    val_auroc = float(roc_auc_score(y_true_bin, y_pred_all))
    val_auprc = float(average_precision_score(y_true_bin, y_pred_all))
except ValueError:
    # 当 mask 中没有正例或没有负例时（极端情况）给默认值
    val_auroc = 0.5
    val_auprc = float(y_true_bin.mean())

# 写入 metrics 字典（与现有指标一起返回）
metrics["val_auroc"] = val_auroc
metrics["val_auprc"] = val_auprc
```

### 6.2 将新指标纳入 best checkpoint 判断

**位置**：best checkpoint 选择逻辑（现有的"多数指标改进"判断）

**修改**：将 `val_auroc` 和 `val_auprc` 加入参与比较的指标集合。这两个指标越大越好，与 `val_pearson_cell` 等一致。

```python
# 原有指标集合（越小越好）
minimize_metrics = ["val_mse", "val_rmse", "val_mae"]

# 原有指标集合（越大越好）
maximize_metrics = [
    "val_pearson_cell", "val_pearson_all",
    "val_spearman_cell", "val_pearson_top20",
    "val_auroc",    # 新增
    "val_auprc",    # 新增
]
```

### 6.3 日志输出

`val_auroc` 和 `val_auprc` 会自动随其他 val 指标一起写入 `train_log.jsonl`，无需额外改动日志逻辑。

---

## 7. 不需要修改的部分

以下模块和逻辑**完全不需要改动**，可直接复用：

| 模块/逻辑 | 原因 |
|----------|------|
| `model.py`（SpectralRouter + VectorFieldNet） | 路由逻辑是通用的；输入维度通过 n_genes 和 grid_size 参数化，改 config 即可 |
| `gene_mamba.py`（GeneMamba 2D Hierarchical Mamba） | 编码器只关心序列长度，不关心特征的生物语义 |
| Flow Matching 框架（x_t 插值、x-prediction、Euler rollout） | 基于连续空间，TF-IDF 值满足条件 |
| Huber loss + Pearson loss + cell-wise Pearson loss | 对连续值完全有效 |
| 分层时间采样策略 | 与特征域无关 |
| Donor-level train/val 分割 | PBMC Multiome 10k 有 donor_id，可直接使用 |
| 多数指标 best checkpoint 选择 | 只需追加新指标（见 Step 4） |
| 确定性验证（固定 mask seed + noise seed） | 完全复用 |
| `checkpoint_best.pt` / `checkpoint_last.pt` 格式 | 不变 |

---

## 8. 验证顺序建议（分阶段执行）

### Phase 1：Pipeline 连通性验证（1-2 天）

**目标**：确认整个 pipeline 从数据读取到训练循环到指标输出完整跑通，没有 shape mismatch 或 NaN。

**配置**：
```yaml
n_genes: 5000          # top-5k peak（方差最高的 5000 个）
grid_size: 72          # 72² = 5184
batch_size_per_gpu: 4
epochs: 5
```

**通过标准**：5 个 epoch 内 `train_loss` 下降，`val_auroc` > 0.5（优于随机），无 NaN。

---

### Phase 2：中规模实验（3-5 天）

**目标**：在 top-50k peak 上验证模型是否能学到 accessibility 信号，初步评估 imputation 质量。

**配置**：
```yaml
n_genes: 50000
grid_size: 224
batch_size_per_gpu: 8
epochs: 30
```

**通过标准**：`val_auroc` > 0.70，`val_pearson_cell` > 0.30。

---

### Phase 3：全量实验（1-2 周）

**目标**：全量 ~107k peak，完整 80 epoch 训练，对标 scImpute / MAGIC 等 baseline。

**配置**：
```yaml
n_genes: 107915
grid_size: 329
batch_size_per_gpu: 8   # 视 GPU 显存调整
epochs: 80
```

**Baseline 对比**：建议同时跑 MAGIC（`magic.MAGIC()`）或 SCALE 在相同 mask 位置上的 imputation 结果，用相同的 AUROC/AUPRC/Pearson 指标比较。

---

## 9. 后续扩展路径

这里列出几个在主体改造完成后可以考虑的增强方向，供决策参考。

### 9.1 更强的 spectral coordinates（基于 co-accessibility）

用 Cicero 计算 peak 之间的 co-accessibility 分数（类似 scRNA 里的 PPI 共表达），构建 peak × peak 相似图，再用 Laplacian eigenmaps 生成 2D 坐标。替换 `atac_spectral_coords.npy` 即可，训练代码不动。

### 9.2 Cross-modal 验证（利用 Multiome 的 paired RNA）

10x PBMC Multiome 提供 paired RNA + ATAC。可以在 scATAC imputation 之后，通过 gene activity score（将 peak accessibility 聚合到 gene body）与同细胞的 RNA 表达做相关性，作为额外的生物学验证指标。

### 9.3 Chromosome-wise partition（超大规模扩展）

若未来需要处理远超 100k peaks 的数据，可以将 2D grid 按染色体分块，每条染色体独立跑一个 smaller grid 的 Mamba，最后拼接上下文。这需要修改 `model.py` 的 routing 逻辑。

---

## 10. 改动文件汇总

| 文件 | 是否修改 | 关键改动 |
|------|----------|---------|
| `pbmc_10k_atac_tfidf.h5ad`（新文件） | 新增 | TF-IDF 归一化后的 ATAC 数据 |
| `atac_spectral_coords.npy`（新文件） | 新增 | 基因组坐标驱动的 2D spectral coords |
| `data.py` | **修改** | x₁ 定义（去掉 log1p）、prior 统计量、zero_mask_ratio=0.1 |
| `config.yaml` | **修改** | n_genes、grid_size、spectral_coords 路径、zero_mask_ratio |
| `train.py` | **修改** | 新增 val_auroc、val_auprc 指标，纳入 best checkpoint 判断 |
| `model.py` | 不动 | — |
| `gene_mamba.py` | 不动 | — |

---

*文档依据 TRAINING_PIPELINE.md（scUniFlow 当前实现）和 10x PBMC Multiome 10k 数据集规格整理。执行前请确认 peak 数量与实际数据一致，并据此重新计算 grid_size。*
