# scFiT Pipeline And Mouse Development Dataset Summary

## 1. 这份文档的目的

这份文档总结两部分内容：

1. 当前 `scFiT` 代码里的模型 pipeline 是怎样工作的。
2. `scLatentFlow/` 里的小鼠发育时序数据集目前是什么结构，以及它和 `scFiT` 当前任务定义之间的差距在哪里。

这份总结的目标不是提出最终改法，而是把当前事实和约束讲清楚，方便后续决定怎么改模型。

## 2. 当前 scFiT 在做什么

`scFiT` 当前是一个基于 Transformer 的单细胞表达建模仓库，主干网络在：

- `model.py`
- `train.py`
- `data.py`
- `eval.py`

它现在的核心任务不是“跨时间预测细胞状态”，而是：

- 对每个细胞单独处理
- 从该细胞的表达谱里选出 top-k 高表达基因
- 把这些基因组成 token 序列
- 对其中一部分 token 做 mask
- 用 flow matching 的 `x-prediction` 方式去恢复被 mask 的表达值

所以它现在本质上更接近：

- gene-token 序列建模
- masked expression reconstruction
- flow matching rollout for imputation / reconstruction

而不是：

- temporal transition modeling
- source-time to target-time cell-state prediction
- developmental trajectory generation

## 3. 当前数据进入 scFiT 的方式

### 3.1 输入单位

当前 `scFiT` 的输入单位是“单个细胞”，不是“细胞对”或“时间转移样本”。

每个 cell 会从原始表达矩阵中取出一行：

- 原始输入是 count matrix
- 在 dataset 中先做 `log1p`
- 只保留该细胞中表达最高的 top-k 基因
- 再在序列最前面加一个 `CLS` token

最终单个样本是一个固定长度序列：

- 长度 = `top_k_genes + 1`
- 默认常见设置是 `2048 + 1`

### 3.2 每个 token 的内容

每个 token 由三部分嵌入相加：

1. `gene_embedding(gene_id)`
2. `expr_embedding(x_t)`
3. `mask_embedding(mask_state)`

其中：

- `gene_id` 表示基因身份
- `x_t` 是 flow matching 中间状态
- `mask_state` 表示这个位置是 `pad / known / masked / cls`

### 3.3 CLS token 的作用

序列第一个位置固定是 `CLS` token：

- `gene_id = cls_token_id`
- value 固定为 `0`
- mask state 固定为 `CLS`

模型里也提供了 `get_cell_embedding()`：

- 它返回最终 hidden state 的第一个 token
- 也就是 `CLS` 向量
- 这个向量可以视作 cell embedding

## 4. 当前模型结构

### 4.1 主干

主干是一个带 AdaLN 时间调制的 Transformer。

整体结构大致是：

1. gene embedding
2. continuous value encoder
3. mask embedding
4. time embedding
5. 多层 AdaLN Transformer block
6. final norm
7. expression head 输出每个 token 的表达预测

### 4.2 时间是怎么注入的

当前模型里的“时间”不是生物学时间 `author_day`，而是 flow matching 中的插值时间 `t in [0, 1]`。

也就是说：

- 它表示从噪声态 `x_0` 到真实表达 `x_1` 的生成过程进度
- 它不是细胞所处发育阶段
- 它不是 source time / target time

这是后续改成轨迹预测时最关键的区别之一。

### 4.3 输出头

当前主要输出是：

- `x1_head`: 预测最终表达 `x_1`

可选还有：

- `bin_head`: 辅助的表达分箱分类头

## 5. 当前训练目标

### 5.1 训练时输入

对每个 batch 中的每个细胞：

1. 取真实表达序列 `x_1`
2. 从高斯先验采样 `x_0`
3. 采样 flow 时间 `t`
4. 构造中间状态

```text
x_t = (1 - t) * x_0 + t * x_1
```

5. 对部分 gene token 做 mask
6. 对 masked token 输入 `x_t`
7. 对 unmasked token 输入真实 `x_1`

所以模型学习的是：

- 给定部分真实表达
- 给定部分中间 noisy 表达
- 预测完整的最终表达 `x_1`

### 5.2 损失函数

当前训练损失是几项加权和：

- weighted huber loss
- pearson loss
- cell-wise pearson loss
- optional cross entropy over bins

这些损失都围绕一个目标：

- 恢复被 mask 的基因表达

它们并不直接优化：

- 时间迁移准确性
- 跨 stage distribution matching
- 细胞发育方向一致性

## 6. 当前推理流程

推理时模型做的是 masked token 的 Euler rollout：

1. 对 masked 位置从高斯先验初始化
2. 重复多步：
   - 预测 `x_1`
   - 根据 `(x1_pred - x_t) / (1 - t)` 计算 velocity
   - 更新 masked token

最后输出的是被补全后的表达序列。

这依然是“同一个细胞内部的表达恢复”范式，不是“从早期细胞生成晚期细胞”。

## 7. 当前 scFiT 默认支持的数据设定

最初这套代码主要围绕 SEA-AD 数据写的：

- `mtg_full`
- `dlpfc_full`
- `mtg_toy`
- `dlpfc_toy`

原始默认切分方式是 donor-level split。

为了接你们这套发育数据，我已经扩展了数据层，现在支持：

- 通用 `.h5ad` 路径
- `donor` 切分
- `random` 切分
- `time_threshold` 切分

也就是说，`scFiT` 现在已经可以直接读你们的小鼠时序 h5ad，并按发育时间切 train / val。

## 8. 小鼠发育时序数据集的当前情况

### 8.1 数据文件

当前确认的数据文件是：

- `data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad`

### 8.2 数据规模

这个 h5ad 的形状是：

- `103,999 cells x 45,525 genes`

也就是说，它是原始 gene expression 空间，不是只有 latent embedding。

### 8.3 关键 obs 字段

已经确认存在的重要列包括：

- `author_day`
- `author_cell_type`
- `development_stage`
- `cell_type`
- `donor_id`

其中和时序最相关的是：

- `author_day`: 原始时间标签
- `author_cell_type`: 作者标注的细胞类型
- `development_stage`: 发育阶段

### 8.4 时间字段含义

这个数据集的时间标签是 JAX 风格：

- `E1000` 表示胚胎 10.0 天
- `E1325` 表示胚胎 13.25 天
- `P0000` 表示出生时，当前解析为 `19.0`

数据中共有 43 个时间 bin。

### 8.5 细胞类型情况

目前看到的主要 cell type 包括：

- `Neural crest (PNS neurons)`
- `Sympathetic neurons`
- `Dorsal root ganglion neurons`
- `Parasympathetic neurons`
- `Otic sensory neurons`
- `Enteric neurons`

这说明它不是单一细胞群的简单时间序列，而是带分支和谱系差异的发育数据。

## 9. 这个数据集的本质：snapshot，而不是 longitudinal

这是最重要的事实之一。

这个数据集不是：

- 同一个细胞在多个时间点被连续测量

而是：

- 每个时间点采到一批不同细胞
- 不同时间点形成的是分布级别的 snapshot

因此，所谓“预测发育轨迹”不能理解为：

- 对同一个 cell 的后续真实观测做逐帧回归

更合理的理解是：

- 学习从早期细胞状态分布到晚期细胞状态分布的迁移规律
- 在 cell type / lineage 条件下做时间推进
- 预测 trajectory 或 future state distribution

## 10. scLatentFlow 当前对这个数据集做了什么

`scLatentFlow` 里已经有一条与这个数据相关的时序建模路线，但那条路线主要在 latent 空间里工作。

当前可以看到的中间产物：

- `outputs/flow_data/jax_uce_flow_data.npz`

里面的键包括：

- `z`
- `time_raw`
- `time_numeric`
- `time_norm`
- `time_bin_index`
- `cell_type`
- `obs_names`
- `unique_time_bins`
- `unique_time_bin_numeric`

其中：

- `z` 的形状是 `(102239, 1280)`

说明 `scLatentFlow` 当前已有流程是：

1. 先把细胞映射到 latent embedding
2. 再在 latent space 做 temporal modeling

所以 `scLatentFlow` 当前主线和 `scFiT` 当前主线之间的区别是：

- `scLatentFlow`: latent space temporal modeling
- `scFiT`: gene-token space masked reconstruction

## 11. 现在已经为这套数据打通了什么

我已经把 `scFiT` 的数据入口扩展到能直接接这个 h5ad。

现在支持的配置方式示例是：

```yaml
data:
  source: data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad
  split:
    strategy: time_threshold
    time_col: author_day
    train_max_time: 14.0
    val_min_time: 14.0
    val_max_time: 19.0
```

对应示例配置文件已经放在：

- `config_mouse_neural_crest.yaml`

我也做过真实加载测试，确认以下流程已经是通的：

1. 读取这个 h5ad
2. 解析 `author_day`
3. 按时间阈值切分 train / val
4. 生成 top-k gene token 序列
5. 构造 `TopKExpressionDataset`

## 12. 现阶段最核心的任务错位

虽然数据已经能进 `scFiT`，但当前模型目标和你们想做的下游任务还没有对齐。

### 12.1 当前模型学到的东西

当前 `scFiT` 学的是：

- 在一个细胞内部，根据部分观察到的基因 token，恢复另外一部分基因 token

这是：

- conditional reconstruction
- masked imputation

### 12.2 你们想做的东西

你们想做的是：

- 预测细胞的发育轨迹

这通常意味着至少要支持下面一种或多种形式：

- 从 `t_src` 推到 `t_tgt`
- 从早期细胞状态生成未来细胞状态
- 在 cell type 或 lineage 条件下进行时间推进
- 在分布层面匹配真实晚期细胞群体

### 12.3 目前缺的关键信号

当前 `scFiT` 里还没有真正用到：

- 生物学时间 `author_day`
- source time / target time
- source cell / target cell pairing
- trajectory consistency loss
- temporal distribution matching objective

因此现在即使把这套发育数据喂进去，模型训练目标仍然主要是：

- “每个时间点各自做表达重建”

而不是：

- “学习发育方向和跨时间演化”

## 13. 如果后面要改成轨迹预测，最可能会动到哪些层

这里只列结构层面的改动点，不代表最终方案已经确定。

### 13.1 数据层

当前数据层输出是单细胞样本。

如果要做轨迹预测，通常需要改成至少一种形式：

- 单细胞 + 时间标签
- source-target 时间对
- source-target 细胞对
- source cell + target time

### 13.2 模型输入层

当前模型只有 flow 时间 `t`，没有生物学时间条件。

后续大概率要增加：

- `t_src`
- `t_tgt`
- 或者 `delta_t`

并明确区分：

- flow integration time
- biological developmental time

### 13.3 训练目标

后续大概率要从“mask reconstruction”转向某种 temporal objective，例如：

- 预测 target-time expression
- velocity / displacement prediction
- conditional flow matching over time-conditioned pairs
- distribution matching between predicted target cells and real target cells

### 13.4 评估

当前评估指标主要还是 reconstruction 指标：

- MSE
- MAE
- Pearson
- Spearman

如果是轨迹任务，后续更合适的评估会包括：

- 按时间点的分布距离
- 按 cell type 的 composition consistency
- 轨迹方向一致性
- trajectory centroid correlation
- source-to-target lineage plausibility

## 14. 目前可以明确得出的结论

### 结论 1

这个小鼠发育数据集可以直接接进 `scFiT`，因为它本身就是基因表达空间的 h5ad，不需要先依赖 latent `z`。

### 结论 2

当前 `scFiT` 的模型主干是可以复用的，特别是：

- gene tokenization
- Transformer backbone
- flow matching style training framework

### 结论 3

但当前训练任务定义还不等于“轨迹预测”。

眼下它更像：

- static cell reconstruction model

而不是：

- temporal developmental forecasting model

### 结论 4

真正要做“预测细胞发育轨迹”，接下来最关键的不是再换一个数据读法，而是决定：

- 时间条件怎么进模型
- source / target 样本怎么构造
- 训练目标是 token reconstruction、state transition，还是 distribution transport

## 15. 建议阅读顺序

如果要继续讨论怎么改模型，建议先基于下面三个问题做决策：

1. 我们到底想预测“单细胞未来状态”还是“目标时间点细胞分布”？
2. 我们要在 gene space 上直接做，还是参考 `scLatentFlow` 的 latent pairing 思路？
3. 我们是想保留 `scFiT` 现在的 top-k token 形式，还是改成更显式的时间条件生成任务？

等这三个点确定后，再去改 `data.py / model.py / train.py` 会更稳。
