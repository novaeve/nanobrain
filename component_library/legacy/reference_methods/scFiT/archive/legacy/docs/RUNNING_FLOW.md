# scFiT 代码运行流程

这份文档按“从启动命令到一次完整训练/验证/推理”的顺序，总结当前 `scFiT` 代码的实际运行流程。

当前仓库对应的是一个独立任务定义：

- 每个 cell 先从全基因表达里选出 top-`K` 高表达基因
- 在序列开头额外加入一个 `CLS` token
- 固定得到长度为 `K + 1` 的 token 序列，默认 `2048 + 1`
- 只在这个 2048 子集内部做 mask-and-reconstruct
- 使用 flow matching 的 `x-prediction` 目标训练
- 推理时在 masked token 上做 Euler rollout

---

## 1. 总体调用链

最常见的调用方式有两种：

```bash
python scripts/train.py --config config.yaml
```

或者：

```bash
python launch.py --config config.yaml
```

其中：

- `launch.py` 只是一个薄启动器，负责把参数转发给 `train.py`
- 真正的训练逻辑全部在 `train.py`

整体调用链如下：

```text
launch.py
  -> train.py: parse_args()
  -> 读取 config.yaml
  -> 设置随机种子
  -> 加载 SEA-AD 数据
  -> 构建 TopKExpressionDataset / DataLoader
  -> 构建 scFiTModel
  -> 创建 Trainer
  -> 进入 Trainer.train()
      -> 按 epoch 训练循环
      -> 按 step 定期验证
      -> 每个 epoch 结束后按配置保存 checkpoint
```

---

## 2. 配置文件的作用

配置文件在 [config.yaml](./config.yaml)。

它分为 5 块：

### 2.1 `data`

- `source`: 数据键名或 `.h5ad` 路径
- `max_cells`: 可选，调试时截断数据规模
- `cell_seed`: 数据子采样随机种子

### 2.2 `experiment`

- `output_dir`: 实验输出根目录
- `run_name`: 实验名；如果为空，会自动生成为 `{data_source}_{timestamp}`

### 2.3 `model`

- `top_k_genes`: 每个 cell 保留多少个高表达基因，默认 `2048`
- `d_model`: token hidden size
- `nhead`: 多头注意力头数
- `d_ff`: FFN 隐层宽度
- `nlayers`: Transformer block 层数
- `dropout`: dropout
- `value_encoder.max_value`: 连续表达编码器的截断上限
- `adaln.share_projection`: 是否跨层共享 AdaLN 的 time projection
- `aux_bin_ce.enabled`: 是否启用辅助 bin 分类损失
- `aux_bin_ce.n_bins`: bin 数量

### 2.4 `training`

- `batch_size`: 训练 batch size
- `lr`: 学习率
- `weight_decay`: AdamW 权重衰减
- `epochs`: 总训练轮数
- `seed`: 全局随机种子
- `mask_ratio`: 训练时在 2048 token 内 mask 的比例
- `log_every_steps`: 每隔多少步记录一次 batch 级日志
- `val_every_steps`: 每隔多少个 step 做一次验证
- `save_every_epochs`: 每隔多少个 epoch 保存一次 checkpoint
- `val_ratio`: donor 级别验证集比例
- `amp`: 是否启用 mixed precision
- `monitor_cls_attention`: 是否在日志步上额外统计 CLS 注意力
- `resume`: checkpoint 路径；为空表示从头训练
- `loss_weights.*`: 四项损失的权重

### 2.5 `validation` / `inference`

- `validation.mask_ratio`: 验证时 mask 比例
- `validation.mask_seed`: 验证 mask 的固定种子
- `validation.max_batches`: 最多验证多少个 batch
- `inference.n_steps`: 推理 Euler rollout 步数

---

## 3. 程序入口：`train.py`

入口在 [train.py](./train.py)。

主流程如下：

### 3.1 解析命令行参数

当前命令行只保留一个入口参数：

- `--config`

也就是说：

- 数据源由 YAML 的 `data.source` 控制
- run name 由 YAML 的 `experiment.run_name` 控制
- 恢复训练由 YAML 的 `training.resume` 控制
- 数据截断由 YAML 的 `data.max_cells` 控制

### 3.2 读取配置并设置随机种子

`seed_everything()` 会同时设置：

- Python `random`
- `numpy`
- `torch`
- `torch.cuda`

这样训练 run 的随机性可控。

### 3.3 创建实验目录

调用 `build_experiment_dir()`：

- 创建 `experiments/<run_name>/`
- 创建 `experiments/<run_name>/checkpoints/`
- 同时把当前使用的 `config.yaml` 复制到实验目录

输出文件主要有：

- `config.yaml`
- `train_log.jsonl`
- `checkpoints/*.pt`

---

## 4. 数据加载流程

数据逻辑在 [data.py](./data.py)。

### 4.1 支持的数据源

当前默认支持 4 个键：

- `mtg_full`
- `dlpfc_full`
- `mtg_toy`
- `dlpfc_toy`

这些键会映射到本机的 SEA-AD `.h5ad` 文件路径。

也支持直接传入 `.h5ad` 路径。

### 4.2 train / val 划分

当：

- 数据键是 `mtg_full` 或 `dlpfc_full`
- 或者 `val_ratio > 0`

程序会调用 `load_sea_ad_split()`。

它的流程是：

1. 读取仓库内的 donor 统计 CSV  
   即 [donor_cell_counts_full_mtg.csv](./donor_cell_counts_full_mtg.csv)
2. 调用 `split_donors_train_val()`
3. 按 donor 粒度把样本分为 train / val
4. 返回：
   - `train_X`
   - `val_X`
   - `gene_ids`
   - `meta`

划分策略是一个简单贪心过程：

- 先随机打乱 donor
- 再按 donor 细胞数从小到大排序
- 逐个加入 val，直到接近目标 `val_ratio`

### 4.3 不做 train / val 划分时

会调用 `load_sea_ad()`，直接返回全量矩阵。

### 4.4 prior 统计量

训练开始前会调用：

`compute_log1p_nonzero_prior_stats(train_X)`

它会在训练集的所有非零表达上做：

- `log1p`
- 计算均值 `prior_mean`
- 计算标准差 `prior_std`

这两个量后面会用来给 `x_0` 采样高斯先验。

---

## 5. 数据集构造：`TopKExpressionDataset`

这是当前任务定义里最关键的一层。

### 5.1 输入

每个样本来自一整行单细胞表达向量：

- 原始形状：`(n_genes,)`

### 5.2 预处理

在 `__getitem__()` 中：

1. 取出当前 cell 的原始 count
2. 计算 `x_log = log1p(x_raw)`
3. 找出非零位置 `nonzero_idx`

### 5.3 top-K 选择

如果这个 cell 有非零基因：

1. 取出非零基因的表达值
2. 取出对应的全局 `gene_ids`
3. 使用

```text
表达值降序
+ gene_id 升序
```

作为排序规则

4. 选出前 `top_k`

这一步的结果是“每个 cell 的 2048 个高表达 token 子集”。

### 5.4 padding 规则

如果非零基因少于 `top_k`：

- `subset_gene_ids` 用 `pad_token_id = n_genes` 补齐
- `subset_values` 用 `0` 补齐
- `valid_mask` 标记哪些位置是真实 token

最终每个样本输出：

- `gene_ids`: `(top_k + 1,)`
- `x_1`: `(top_k + 1,)`
- `valid_mask`: `(top_k + 1,)`
- 可选 `cell_idx`

其中第 `0` 个位置固定是 `CLS`。也就是说，后续模型永远只看到固定长度 `2049` 的序列。

---

## 6. DataLoader 输出到训练前的整理

在 [train.py](./train.py) 的 `Trainer._prepare_batch()` 中，
batch 会被整理为：

- `gene_ids`: `(B, K + 1)`
- `x_1`: `(B, K + 1)`
- `valid_mask`: `(B, K + 1)`
- `padding_mask = ~valid_mask`: `(B, K + 1)`
- 可选 `cell_idx`

其中：

- `valid_mask=True` 表示该位置是真实 token，包括开头的 `CLS`
- `padding_mask=True` 表示该位置是 padding

---

## 7. 模型结构：`scFiTModel`

模型定义在 [model.py](./model.py)。

它可以分成 5 层：

### 7.1 gene token 编码

`self.gene_encoder`

- `Embedding(n_genes + 2, d_model)`
- `n_genes` 作为 `pad_token_id`
- `n_genes + 1` 作为 `cls_token_id`

输出：

- `gene_emb`: `(B, K, d_model)`

### 7.2 value 编码

`ContinuousValueEncoder`

输入：

- `expr_values`: `(B, K)`

处理流程：

1. `unsqueeze(-1)` 变成 `(B, K, 1)`
2. 截断到 `[0, max_value]`
3. `Linear -> SiLU -> Linear -> LayerNorm -> Dropout`

输出：

- `value_emb`: `(B, K, d_model)`

### 7.3 mask state 编码

`mask_states` 不是简单二值，而是四值状态：

- `0 = pad`
- `1 = known`
- `2 = masked`
- `3 = cls`

通过 `build_mask_states()` 从：

- `padding_mask`
- `mask_indicator`

组合得到。

然后再经过：

- `self.mask_encoder`

得到：

- `mask_emb`: `(B, K, d_model)`

### 7.4 token 合成

每个 token 的最终输入为：

```text
x = gene_emb + value_emb + mask_emb
```

形状：

- `(B, K, d_model)`

### 7.5 时间嵌入

时间 `t` 的处理流程：

1. `SinusoidalTimeEmbedding`
2. `time_mlp`

得到：

- `t_emb`: `(B, d_model)`

这个 `t_emb` 会送入每个 Transformer block，用于 AdaLN 调制。
当前实现里 `CLS` 位置会保留普通 `LayerNorm` 输出，不直接乘上 AdaLN 的 `scale/shift`，避免把 cell embedding API 和时间步硬绑定。

---

## 8. AdaLN Transformer Block 的前向流程

单层 block 是 `AdaLNTransformerBlock`。

每层结构是：

```text
x = x + Attention(AdaLN(LN(x), t))
x = x + FFN(AdaLN(LN(x), t))
```

更细一点：

### 8.1 Attention 分支

1. 对输入做 `LayerNorm(elementwise_affine=False)`
2. 用 `attn_mod(t_emb)` 生成：
   - `scale1`
   - `shift1`
3. 执行：

```text
x_norm = LN(x) * (1 + scale1) + shift1
```

4. 把 `x_norm` 送入 `MultiheadAttention`
5. 用 residual 加回原输入

### 8.2 FFN 分支

同样步骤：

1. `LayerNorm`
2. `ffn_mod(t_emb)` 生成：
   - `scale2`
   - `shift2`
3. AdaLN 调制
4. `Linear -> GELU -> Dropout -> Linear`
5. residual

### 8.3 初始化策略

`attn_mod` 和 `ffn_mod` 都做了零初始化：

- 权重全零
- bias 全零

这意味着训练一开始，AdaLN 近似为恒等映射，模型起点接近普通 Pre-Norm Transformer。

---

## 9. 训练前向流程

训练逻辑在 `Trainer.train()`。

当前训练是标准的 epoch 制：

- 一个 epoch = 把整个 `train_loader` 完整遍历一遍
- 学习率调度仍然按总 batch 数展开
- 验证按全局 step 触发
- checkpoint 按 epoch 结束后触发

### 9.1 采样时间 `t`

使用 `sample_t_stratified()`：

- 在 `[0, 1]` 上按分层比例采样
- 返回形状 `(B, 1)`

这样做是为了降低不同 batch 之间的时间采样方差。

### 9.2 采样高斯先验 `x_0`

调用 `sample_gaussian_prior(valid_mask, mean, std)`：

- 在所有 valid token 上采样高斯
- padding 位置强制为 0

输出：

- `x_0`: `(B, K)`

### 9.3 构造训练 mask

调用 `build_random_mask(valid_mask, mask_ratio)`：

- 每个样本只在 valid token 上随机选一部分做 mask
- 至少 mask 1 个 token

输出：

- `mask_indicator`: `(B, K)`，布尔型

### 9.4 构造中间状态 `x_t`

在 `model.forward_train()` 中：

```text
x_t = (1 - t) * x_0 + t * x_1
```

然后构造真正送入 Transformer 的表达输入：

```text
input_expr = where(mask_indicator, x_t, x_1)
```

也就是：

- 被 mask 的位置喂 `x_t`
- 已知位置喂真实 `x_1`

### 9.5 编码并预测 `x̂_1`

然后模型执行：

1. 构建 `mask_states`
2. `encode()`
3. `x1_head()`

输出：

- `x1_pred`: `(B, K + 1)`

这个预测的是终点表达 `x̂_1`，不是速度。

---

## 10. 训练损失

训练损失由四部分组成：

### 10.1 `weighted_huber_loss`

只在 `mask_indicator=True` 的位置上监督。

逻辑：

1. 先算逐位置 Huber loss
2. 在每个样本的 masked token 里，挑真实值最高的 top-20
3. 这部分位置额外赋更大权重 `3.0`

目标：

- 让模型更关注高表达 token 的重建质量

### 10.2 `pearson_loss`

把所有 masked token 拉平成一个向量后，算全局 Pearson loss：

```text
1 - corr(pred, target)
```

目标：

- 防止模型只学均值回归

### 10.3 `cell_pearson_loss`

逐 cell 计算 masked token 上的 Pearson loss，再取平均。

注意：

- 如果某个样本被 mask 的 token 少于 2 个，会跳过

### 10.4 总损失

```text
L = 1.0 * weighted_huber
  + 0.1 * pearson
  + 0.1 * cell_pearson
  + 0.2 * cross_entropy
```

其中 `cross_entropy` 只在 `masked & valid` 的 gene token 上计算，不包含 padding，也不包含 `CLS`。

---

## 11. 训练循环细节

训练使用：

- `AdamW`
- `CosineAnnealingLR`
- 可选 `torch.autocast(..., bfloat16)`
- `GradScaler`
- 梯度裁剪 `max_norm=1.0`

每个 batch 的顺序是：

1. 取一个 batch
2. 构造 `t`
3. 构造 `x_0`
4. 构造 `mask_indicator`
5. 前向得到 `x1_pred`
6. 计算三项损失
7. 反向传播
8. 梯度裁剪
9. `optimizer.step()`
10. `scheduler.step()`

每 `log_every_steps` 步：

- 记录训练日志到 `train_log.jsonl`
- 如果 `monitor_cls_attention=true`，额外记录 CLS 接收到的平均注意力权重

每个 epoch 结束时：

- 汇总该 epoch 的平均训练损失

每 `val_every_steps` 步：

- 调用 `validate()`

每 `save_every_epochs` 个 epoch：

- 保存 checkpoint

---

## 12. 验证流程

验证逻辑在 `Trainer.validate()`。

### 12.1 验证 mask

验证不使用随机 mask，而是使用：

`build_deterministic_mask(valid_mask, cell_idx, mask_ratio, base_seed)`

这样同一个 cell 在同一个验证配置下总是被 mask 同样的位置。

### 12.2 构造 observed input

验证阶段不会额外把 masked 位置清零。

当前实现直接把完整的 `x_1` 传入 `inference()`，因为在 rollout 开始前，
所有 `masked + valid` 位置都会立即被高斯先验覆盖，所以这些位置在输入张量里
原本是什么值并不会影响实际推理。

### 12.3 调用 rollout 推理

然后进入：

`model.inference(...)`

输出整个 2048 token 子集上的最终预测 `pred`。

### 12.4 计算指标

`masked_metrics()` 只在 masked positions 上统计：

- `val_mse`
- `val_rmse`
- `val_mae`
- `val_pearson_all`
- `val_pearson_cell`

最后在 batch 间做简单平均。

---

## 13. 推理流程：Euler rollout

推理在 `scFiTModel.inference()` 中。

### 13.1 初始化

输入：

- `gene_ids`: `(B, K)`
- `observed_x1`: `(B, K)`  
  其中真正会被使用的是 known 位置；masked 位置会在 rollout 开始前被先验覆盖
- `padding_mask`
- `mask_indicator`

先复制：

```text
x_t = observed_x1.clone()
```

然后只对 masked 且 valid 的位置采样高斯初值：

```text
x_t[masked_valid] ~ N(prior_mean, prior_std^2)
```

### 13.2 rollout 主循环

设：

- `dt = 1 / n_steps`

每一步：

1. 构造当前时间 `t`
2. 根据 `padding_mask` 和 `mask_indicator` 构造 `mask_states`
3. 用当前 `x_t` 前向预测 `x1_pred`
4. 用 x-prediction 转速度：

```text
eps = 1e-3
v_pred = (x1_pred - x_t) / clamp_min(1 - t, eps)
```

5. Euler 更新：

```text
x_t = x_t + v_pred * dt
```

6. 对 masked 位置做 `clamp(min=0.0)`
7. 已知位置强制重置回 `observed_x1`
8. padding 位置强制为 0

### 13.3 输出

rollout 结束后返回：

- `x_t`: `(B, K)`

这就是最终的 token 级重建结果。

---

## 13.4 下游 `CLS` 表征接口

`model.get_cell_embedding()` 默认使用：

- `t = 1`
- 所有非 padding token 都按 `known` 构造 `mask_states`

所以这个接口的默认语义是“完整输入细胞的 CLS 表征”，而不是 rollout 某个中间时刻的瞬时状态。

---

## 14. 日志和 checkpoint

### 14.1 `train_log.jsonl`

每 `log_every_steps` 步记录一条 batch 级 JSON，包括：

- `epoch`
- `epoch_step`
- `global_step`
- `train_loss`
- `weighted_huber_loss`
- `pearson_loss`
- `cell_pearson_loss`
- `cross_entropy_loss`
- `cell_pearson_used_ratio`
- `lr`
- 可选验证指标
- 可选 `cls_attn_received_mean`
- 可选 `cls_attn_received_last`

每个 epoch 结束后还会额外记录一条 epoch 级 JSON，包括：

- `epoch`
- `global_step`
- `epoch_train_loss`
- `epoch_batches`
- `lr`

### 14.2 checkpoint

保存在：

- `experiments/<run_name>/checkpoints/`

内容包括：

- `epoch`
- `global_step`
- `model_state_dict`
- `optimizer_state_dict`
- 可选 `scheduler_state_dict`
- `prior_mean`
- `prior_std`

这些 prior 参数保存下来是为了恢复训练和推理时保持一致。

---

## 15. 当前代码的真实任务边界

当前这版 `scFiT` 不是全基因重建模型，而是：

**“每个 cell 的 top-2048 高表达 gene token 子集上的 flow-matching 重建模型”**

这意味着：

- 模型只看每个 cell 中最强的 2048 个非零基因
- 子集之外的基因不会进入当前模型
- mask / reconstruct / rollout 都只发生在这个子集内部

因此它最适合回答的问题是：

- 在高表达子集上，Transformer + AdaLN + FM 是否有效

而不是：

- 全基因空间是否被完整重建

---

## 16. 一次完整运行的时序总结

把所有步骤压缩成一条时序线，就是：

```text
读取 config
-> 设随机种子
-> 加载 SEA-AD
-> donor 划分 train/val
-> 统计 train 非零 log1p 的高斯先验
-> 每个 cell 取 top-2048 高表达 gene token
-> DataLoader 组成 batch
-> 训练时按 epoch 循环：
   -> 完整遍历一遍训练集
   -> 采样 mask
   -> 采样 x0
   -> 采样 t
   -> 构造 x_t
   -> gene/value/mask 三路 embedding 相加
   -> Transformer + AdaLN(time)
   -> 预测 x̂1
   -> 计算 weighted Huber + Pearson + cell-Pearson + CE
   -> 反向传播更新参数
   -> 按全局 step 定期验证
-> 每个 epoch 结束：
   -> 记录 epoch 平均训练损失
   -> 按配置保存 checkpoint
-> 验证时：
   -> 用固定 seed 构造 mask
   -> masked token 高斯初始化
   -> Euler rollout 多步推理
   -> 在 masked token 上统计 RMSE / MAE / Pearson
-> 定期保存 checkpoint 和日志
```

---

## 17. 你之后最可能改动的地方

如果后续要继续迭代，这几个位置通常最关键：

- [config.yaml](./config.yaml)  
  调整任务规模和训练超参数

- [data.py](./data.py)  
  改 top-K 选择规则，或者改 token 排序策略

- [model.py](./model.py)  
  改 Transformer block、AdaLN、decoder 头

- [utils.py](./utils.py)  
  改 mask 规则、损失函数、验证指标

- [train.py](./train.py)  
  改训练流程、日志、恢复策略、验证协议
