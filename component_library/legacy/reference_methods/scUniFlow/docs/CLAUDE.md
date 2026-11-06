# scUniFlow

基于条件流匹配（CFM）的单细胞 RNA-seq 基因表达填充模型，使用 2D Mamba 编码器和可学习的基因排列模块。

## 模型架构

`scRNAModel`（[model.py](model.py)）由三个子模块组成：

- **`PermutationModule`** — 将 ~35K 基因软路由到 N×N 网格（如 190×190=36100 个槽位）。用 Top-K Gumbel-Sinkhorn 注意力为每个槽位分配基因特征的加权组合。可学习参数 `Q`（L×D）代表网格槽位嵌入。分块计算（每块 1000 槽位）+ 梯度检查点避免 OOM。
- **`GeneMamba`**（[gene_mamba.py](gene_mamba.py)）— 将路由后的网格当作 2D 空间序列处理。每个 `HierarchicalMambaBlock` 依次执行：
  1. `RowBiMamba`：行内双向 Mamba2（chunk_size = grid_size）
  2. `ColSSM`：列间双向 Mamba2
  3. FFN + Pre-Norm + 残差
- **`VectorFieldNet`** — MLP，输入 `(x_t, context, t)`，预测掩码基因位置的流速度场。

### Q 的谱嵌入初始化

`PermutationModule.init_from_spectral(gene_coords_grid)` 用基因的 2D 谱布局坐标初始化排列模块：

- `Q[l]` ← 槽位 l 的网格位置 `(row, col)` 的 Fourier 编码
- `gene_embedding[g]` ← 基因 g 目标网格坐标的 Fourier 编码
- `W_K.weight` ← 单位矩阵（保证初始时 `K[g] ≈ Q[目标槽位]`，路由从一开始就有意义）

坐标文件由 [string_spectral/step8_augmented_graph.py](string_spectral/step8_augmented_graph.py) 生成，已存放于：

```
data/gene_coords/augmented_gene_coords.npy   # shape (35483, 2)，值域 [0, 190)
```

覆盖策略：STRING thr=200（~53% 基因）→ GTEx 脑组织共表达 → 基因组物理相邻 → GO BP Jaccard → 孤立基因放网格边缘。

## 训练流程

三阶段训练，通过 `--phase all|1|2|3` 控制（[train.py](train.py)）：

| 阶段 | 训练内容 | 步数（flow_v3） | 说明 |
|------|---------|----------------|------|
| 1 — Warmup | 仅 Mamba + VectorField（Q 冻结） | 4000 | Huber loss 优化流速度场 |
| 2 — Joint | 全部参数；Q 以 lr×0.01 每 10 步更新一次 | 12000 | 温度从 1.0 退火到 0.1；记录 `delta_pi` 监控排列漂移 |
| 3 — Finetune | 仅 Mamba + VectorField（Q 再次冻结） | 24000 | 余弦 LR 衰减至 1e-6 |

CFM 目标：掩码位置的 `v* = x_1 - x_0`；插值路径 `x_t = (1-t)*x_0 + t*x_1`。

**掩码策略**（`scRNADataset.__getitem__`）：随机掩盖 20% 的非零基因、2% 的零基因。

## 数据

SEA-AD（西雅图阿尔茨海默症脑细胞图谱）h5ad 文件，通过 [data.py](data.py) 加载。

`DEFAULT_PATHS` 中的硬编码路径：
- `mtg_full` / `mtg_toy` — 中颞回：`${SEA_AD_ROOT:-data/SEA-AD}/MTG/`
- `dlpfc_full` / `dlpfc_toy` — 背外侧前额叶：`${SEA_AD_ROOT:-data/SEA-AD}/DLPFC/`

Train/val 按 **donor 粒度**划分（贪心装箱逼近 `val_ratio`）。需要项目根目录下存在 `donor_cell_counts*.csv`。基因 ID 是 Ensembl ID 字典序的 argsort 排名（非原始索引）。

## 启动命令

### 单卡训练

```bash
python train.py \
    --config experiments/flow_v3/config.yaml \
    --data mtg_full \
    --phase all \
    --spectral_coords data/gene_coords/augmented_gene_coords.npy
```

### 多卡训练（DDP，推荐）

```bash
torchrun --nproc_per_node=2 train.py \
    --config experiments/flow_v3/config.yaml \
    --data mtg_full \
    --phase all \
    --spectral_coords data/gene_coords/augmented_gene_coords.npy
```

### 单独运行某一阶段

```bash
# 只跑 Phase 1（热身）
torchrun --nproc_per_node=2 train.py \
    --config experiments/flow_v3/config.yaml \
    --data mtg_full \
    --phase 1 \
    --run_name flow_v4_p1 \
    --spectral_coords data/gene_coords/augmented_gene_coords.npy

# 只跑 Phase 2（联合训练）
torchrun --nproc_per_node=2 train.py \
    --config experiments/flow_v3/config.yaml \
    --data mtg_full \
    --phase 2 \
    --run_name flow_v4_p2
```

### 命令行参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--config` | `config.yaml` | YAML 配置文件路径 |
| `--data` | `mtg_toy` | 数据键名（`mtg_toy/mtg_full/dlpfc_toy/dlpfc_full`）或 h5ad 路径 |
| `--batch_size` | `32` | 单卡 batch size（总 batch = 此值 × GPU 数） |
| `--phase` | `all` | 训练阶段：`all` / `1` / `2` / `3` |
| `--run_name` | `{data}_{timestamp}` | 实验名称，结果存到 `experiments/<run_name>/` |
| `--spectral_coords` | 无 | 谱嵌入坐标文件路径（`.npy`），用于初始化 Q |

不传 `--spectral_coords` 则 Q 使用随机初始化（标准差 0.02 的正态分布）。

### 生成谱嵌入坐标（step8，可选）

已有现成文件，无需重跑。如需重新生成（例如换数据集或补充 GTEx/GO 数据后）：

```bash
cd string_spectral
conda activate scUniFlow
python step8_augmented_graph.py
```

GTEx 和 GO 文件需手动下载并放置：
- GTEx：`data/GTEx_median_tpm.gct.gz`（从 https://gtexportal.org 下载 v8 median TPM）
- GO：`data/goa_human.gaf.gz`（从 https://www.ebi.ac.uk/GOA/downloads 下载）

两者缺失时 step8 会跳过对应步骤，仍可生成基于 STRING + 基因组物理位置的坐标。

## 关键文件

| 文件 | 功能 |
|------|------|
| [train.py](train.py) | 三阶段训练器、DDP 初始化、数据分片 |
| [model.py](model.py) | `scRNAModel`、`PermutationModule`（含 `init_from_spectral`）、`VectorFieldNet` |
| [gene_mamba.py](gene_mamba.py) | `GeneMamba`、`HierarchicalMambaBlock`、`RowBiMamba`、`ColSSM` |
| [data.py](data.py) | SEA-AD 数据加载、donor 划分、`scRNADataset` |
| [utils.py](utils.py) | `morans_I_loss`、`compute_delta_pi`、`sinkhorn` |
| [string_spectral/](string_spectral/) | 离线流程：STRING 图 → 谱嵌入坐标（Q 初始化先验） |

## 实验记录

`experiments/flow_v{1,2,3}/` 各包含 `config.yaml`、`train_log.jsonl`（每步指标的 JSONL）、`split_info.json`、`checkpoints/`。当前最新为 `flow_v3`，已保存 `phase1_final.pt`。

日志字段：`phase`、`step`、`epoch`、`train_loss`/`cfm_loss`、`val_loss`、`val_corr`、`delta_pi`（阶段 2）、`tau`、`lr`。

## 依赖

- `mamba_ssm`（Mamba2）— 项目根目录有预编译 wheel（`causal_conv1d-1.6.1+...whl`）
- `scanpy`、`scipy.sparse`、`anndata` — h5ad 文件 I/O
- `pyarrow` — parquet 文件读取（`pip install pyarrow`）
- `torch` + CUDA — 全程使用 bfloat16 autocast
- `hilbertcurve`（可选）— `compute_delta_pi` 的回退实现
