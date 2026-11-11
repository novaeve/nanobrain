# scFiT 方法文档：训练、推理与实现细节

本文档基于当前仓库 `.` 的主线实现，对 **时序单细胞发育分布预测** 方法进行系统整理，重点服务于论文方法部分的撰写。全文聚焦于任务定义、数据资产、协议机制、模型结构、训练目标、推理流程和评估指标，并尽量给出与代码一致的公式表达和实现细节。

---

# 1. 方法概述

scFiT 当前主线面向 **时序单细胞发育分布预测**：给定若干已观测发育时间点上的单细胞分布，学习一个条件动态模型，并据此预测未观测目标发育时间点上的细胞表达分布。当前仓库支持三类标准任务：

1. **Interpolation**：目标时间点位于两个已观测发育时间点之间。
2. **Near extrapolation**：目标时间点紧邻已观测时间边界。
3. **Far extrapolation**：目标时间点远离训练时间范围。

从实现上，整个方法由四个相互衔接的层次组成：

1. **固定表达空间构建**：先冻结一个全局 HVG 面板，再把所有细胞投影到统一的 fixed-HVG coordinate。
2. **任务协议定义**：通过 protocol 文件显式规定已观测发育时间点、未观测目标发育时间点、时间方向约束和目标时间生成策略。
3. **条件流匹配建模**：在已观测发育时间点之间动态构造 OT 软配对，并学习条件速度场。
4. **分布级生成与评估**：对未观测目标发育时间点生成整批细胞，再从几何、统计和生物学一致性三个层面进行评价。

---

# 2. 任务定义

## 2.1 数据对象

设固定 HVG 面板维度为 `d`。对于任意时间点 `t`，记该时间点的真实单细胞表达分布为

`X_t = {x_i^(t)}_(i=1)^(N_t),  x_i^(t) in R^d`

其中每个细胞都已经表示在同一套 fixed-HVG coordinate 中。

给定已观测发育时间集合

`T_obs = {t_1, ..., t_m}`

以及未观测目标时间集合

`T_target = {t'_1, ..., t'_n}`

目标是学习一个条件动态模型 `f_theta`，使其能够在任意 `(t_src, t_tgt)` 条件下刻画从源分布到目标分布的演化规律，并在 `t_tgt in T_target` 时生成尽量逼近真实 `X_(t_tgt)` 的细胞群体。

## 2.2 与传统 temporal pair regression 的区别

当前主线不是基于预先静态构造的 cell pair table 做逐对回归，而是采用在线分布配对：

1. 从两个已观测发育时间点各采样一个小批量细胞群体。
2. 在这两个小批量细胞群体之间计算 OT 传输计划。
3. 从传输计划中按概率抽样训练对。

因此，模型学习的是 **分布到分布的发育时间演化动力学**，而不是固定细胞配对的监督回归。

## 2.3 条件流匹配形式化

对一个训练步中采样得到的源细胞 `s` 和目标细胞 `u`，定义流时间 `tau in [0,1]`。中间态写作

`x_tau = (1 - tau) s + tau u`

真实速度定义为

`v^*(x_tau; s, u) = u - s`

模型学习一个条件速度场

`v_theta(x_tau, tau, t_src, t_tgt)`

其中：

1. `tau` 对应实现中的 `t_flow`
2. `t_src` 是源生物学时间
3. `t_tgt` 是目标生物学时间

训练时通过条件流匹配损失让 `v_theta` 逼近 `u - s`，推理时再通过多步 Euler 更新把源时间点样本推向目标时间点。

---

# 3. 数据表示与固定资产

## 3.1 Frozen Top-K HVG Panel

脚本 `scripts/prepare_hvg_panel.py` 负责从原始 `h5ad` 中计算全基因 HVG 排名，并冻结一个全局 top-K 面板。默认设置为：

1. 从 `layers["raw.X"]` 读取原始计数
2. 使用 Scanpy `seurat_v3` flavor
3. 在 raw counts 上直接做 HVG ranking

这个阶段的目的不是针对某个 split 单独选特征，而是建立一套所有样本共享的固定坐标系。若固定面板中的基因按顺序记为

`G = [g_1, g_2, ..., g_d]`

则任意细胞最终都被表示为

`x = [expr(g_1), expr(g_2), ..., expr(g_d)] in R^d`

主要输出包括：

1. `hvg_ranked_gene_indices.npy`
2. `hvg_ranked_gene_names.txt`
3. `hvg_ranked_var_names.txt`
4. `hvg_metadata.json`
5. 当指定 `top_k` 时，还会物化 `hvg_gene_indices.npy`、`hvg_gene_names.txt`、`hvg_var_names.txt`

## 3.2 固定 HVG 表达矩阵

脚本 `scripts/prepare_temporal_hvg_dataset.py` 会在固定 HVG 面板上构造统一的表达矩阵。对第 `i` 个细胞原始计数向量 `c_i`，预处理流程为：

1. 取固定 HVG 子矩阵 `c_i^(HVG)`
2. 做 library-size normalization 到目标总量 `S = 10000`
3. 做 `log1p`
4. 做数值裁剪

公式可写作

`tilde(c)_ig = c_ig * S / max(sum_h c_ih, eps)`

`x_ig = clip(log(1 + tilde(c)_ig), a, b)`

其中当前实现中 `a = 0`，`b = 10`，`eps = 1e-8`。

预处理后的表达矩阵以 `float32` memmap 形式保存为：

`expression_log1p_clipped.npy`

这样后续训练和评估都可以通过 `numpy.load(..., mmap_mode="r")` 直接随机访问，而无需一次性把全矩阵复制到内存。

## 3.3 预处理 bundle 的配套元数据

预处理目录还会同时保存：

1. `cell_metadata.pkl` / `cell_metadata.csv.gz`
2. `obs_names.npy`
3. `author_day_numeric.npy`
4. `hvg_gene_indices.npy`
5. `hvg_var_names.npy`
6. `hvg_gene_names.npy`
7. `dataset_metadata.json`

在代码实现 `route_a/data.py` 中，`load_preprocessed_temporal_bundle()` 负责读取这套资产，并执行以下校验：

1. metadata 行数与表达矩阵行数一致
2. `obs_names.npy` 长度与表达矩阵行数一致
3. `cell_metadata.pkl` 的 index 与 `obs_names.npy` 完全一致

函数最终返回

`(expression, metadata_df, bundle_meta)`

其中 `bundle_meta` 中额外记录：

1. `fixed_panel_size`
2. `hvg_gene_indices`
3. `hvg_gene_names`
4. `n_genes`
5. `time_norm_denom`

## 3.4 时间值解析

时间标签由代码实现 `route_a/data.py:parse_time_value()` 解析为浮点数。当前规则包括：

1. `E1000 -> 10.0`
2. `E1050 -> 10.5`
3. `E1433.3 -> 14.333`
4. `E1000-E1100 -> 10.5`
5. `P0` 或 `P0000 -> 19.0`

这些数值用于：

1. 协议中的时间顺序排序
2. 目标时间生成来源的权重计算
3. 模型条件输入 `t_src` 和 `t_tgt`

## 3.5 固定评估资产

`scripts/prepare_route_a_assets.py` 调用代码实现 `route_a/assets.py:prepare_route_a_assets()` 为每个 protocol 预计算一套固定评估资产。它们只在 **已观测发育时间点上的细胞** 上拟合，然后固定下来供后续所有验证与最终评估复用。主要包括：

1. PCA 模型 `pca_model.pkl`
2. 全体细胞的 PCA embedding `pca_embeddings_all_cells.npz`
3. cell type kNN 分类器 `cell_type_knn.pkl`
4. marker gene 字典 `marker_genes_by_cell_type.json`
5. 汇总信息 `asset_summary.json`

更具体地说：

1. PCA 默认保留 50 个主成分，仅在已观测发育时间点的细胞上拟合。
2. 全体细胞再按 chunk 投影到该固定 PCA 空间。
3. kNN 分类器默认在已观测发育时间点细胞的 PCA embedding 上训练，`k = 10`。
4. marker genes 通过已观测数据中每个细胞类型的均值差异进行筛选：

`score_g^(c) = mean_(i: y_i = c) x_ig - mean_(i: y_i != c) x_ig`

对每个 cell type 取 `score_g^(c)` 最大的 top-K 基因，默认 `K = 50`。

---

# 4. Protocol 机制

该任务的实验语义由 YAML protocol 显式定义，而不是由训练脚本隐式硬编码。核心实现位于 `route_a/protocol.py`。

## 4.1 protocol 的必需字段

当前实现要求 protocol 至少包含以下字段：

1. `protocol_name`
2. `data_bundle_path`
3. `eligible_time_bins`
4. `observed_time_bins`
5. `heldout_time_bins`
6. `task_type`
7. `source_policy`
8. `eval_time_bins`
9. `normalization`
10. `hvg_dim`

可选字段 `direction_mode` 若未提供，则会回退到 `task_type` 对应的语义。

## 4.2 protocol 校验规则

`_validate_protocol_manifest()` 会检查：

1. `eligible_time_bins`、`observed_time_bins`、`heldout_time_bins` 不含重复
2. 已观测发育时间点与未观测目标时间点互斥
3. `observed_time_bins` 和 `heldout_time_bins` 都是 `eligible_time_bins` 的子集
4. `eval_time_bins` 是 `heldout_time_bins` 的子集
5. `direction_mode` 合法

`load_protocol_bundle()` 还会进一步验证：

1. protocol 中引用的时间点在数据中真实存在
2. `bundle_meta["fixed_panel_size"] == protocol["hvg_dim"]`
3. 每个 eligible 时间点细胞数至少为 500
4. 所有 eligible 时间点均不早于 `E10`

## 4.3 ordered bins 与邻居映射

时间点会先按 `parse_time_value()` 后的数值排序，得到 `ordered_time_bins`。在此基础上，系统为每个时间点建立：

1. 最近的过去已观测发育时间 `prev_observed_time`
2. 最近的未来已观测发育时间 `next_observed_time`

这组邻居映射用于未观测目标时间点的生成来源解析。

## 4.4 未观测目标时间点的生成来源策略

当前实现支持两种未观测目标时间点生成来源策略。

### 4.4.1 `previous_observed`

目标时间点 `t` 只使用最近的过去已观测邻居 `t^-` 作为源，权重固定为 1：

`S(t) = {(t^-, 1)}`

该策略用于 near extrapolation 和 far extrapolation。

### 4.4.2 `bidirectional_linear`

目标时间点 `t` 同时使用左右两侧最近已观测邻居 `t^-` 和 `t^+` 作为源，权重按线性插值分配：

`w^- = (t^+ - t) / (t^+ - t^-)`

`w^+ = (t - t^-) / (t^+ - t^-)`

并做归一化后得到

`S(t) = {(t^-, w^-), (t^+, w^+)}`

该策略用于 interpolation。

## 4.5 三个内置 protocol

当前仓库主线提供三套 benchmark protocol：

1. `configs/protocols/mouse/interpolation.yaml`
2. `configs/protocols/mouse/near_extrapolation.yaml`
3. `configs/protocols/mouse/far_extrapolation.yaml`

其中：

1. interpolation 使用 `source_policy = bidirectional_linear`，`direction_mode = bidirectional`
2. near / far extrapolation 使用 `source_policy = previous_observed`，`direction_mode = forward_only`

---

# 5. 模型输入表示

尽管任务对象是时间分布，底层模型仍然使用 `model.py` 中的 `scFiTModel`。在当前时序单细胞发育分布预测任务中，输入表示由固定基因序列、连续表达值和时间条件三部分共同构成。

## 5.1 基因 token 序列

给定固定 HVG 面板大小为 `d` 的细胞表达向量 `x in R^d`，训练器会构造一个长度为 `d + 1` 的序列：

1. 第 0 个位置为序列级占位 token
2. 第 `1...d` 个位置依次对应固定 HVG 面板中的基因

在 `route_a/train.py` 中：

`gene_ids_np = [cls_token_id, hvg_gene_indices[0], ..., hvg_gene_indices[d-1]]`

因此所有样本共享同样的基因顺序，这一点对于时间点之间的直接比较至关重要。

## 5.2 表达值序列

对任意一个细胞，训练器构造：

`seq[0] = 0`

`seq[g] = x_g,  g = 1, ..., d`

也就是说，序列第 0 位只承担全局占位功能，不承载真实基因表达；真实基因表达仅出现在后 `d` 个位置。

## 5.3 mask-state 序列

训练器同时为每个位置提供 `mask_states`：

1. 第 0 位为 `MASK_CLS`
2. 其余位置为 `MASK_KNOWN`

当前主线任务不做 gene masking，因此对所有真实基因位点来说，输入均为已知状态。

## 5.4 时间条件输入

时间条件由三个标量组成：

1. `t_flow`：流时间，也就是连续插值中的 `tau`
2. `t_src`：源生物学时间
3. `t_tgt`：目标生物学时间

在代码中它们都以 shape `(batch, 1)` 的浮点张量形式输入模型。

---

# 6. 模型结构

## 6.1 总体架构

`scFiTModel` 可以视为一个由全局时间条件调制的 Transformer。输入 hidden state 由三部分相加得到：

`h^(0) = e_gene + e_value + e_mask`

其中：

1. `e_gene` 是基因 ID embedding
2. `e_value` 是连续表达值 embedding
3. `e_mask` 是 token 状态 embedding

随后 `h^(0)` 进入多层 `AdaLNTransformerBlock`，每一层都受到统一条件向量 `c_all` 的调制。

## 6.2 gene encoder

基因 ID 通过

`nn.Embedding(n_genes + 2, d_model, padding_idx=pad_token_id)`

编码，并在相加前经过一层 `LayerNorm`。其中模型内部保留了两个特殊位置：

1. `pad_token_id = n_genes`
2. `cls_token_id = n_genes + 1`

## 6.3 ContinuousValueEncoder

表达值编码器 `ContinuousValueEncoder` 直接建模连续数值，而不是先做离散分桶。对于单个表达值 `x`，其编码过程为：

`x' = clip(x, 0, x_max)`

`h_1 = SiLU(W_1 x' + b_1)`

`h_2 = W_2 h_1 + b_2`

`e_value = Dropout(LayerNorm(h_2))`

其中当前默认 `x_max = 32.0`。

这一设计使模型能够在实值表达空间中学习连续变化，而非把表达信号粗糙地离散化为 token 类别。

## 6.4 SequenceConditionEncoder

条件编码器 `SequenceConditionEncoder` 用于把 `t_flow`、`t_src`、`t_tgt` 编码为统一的全局条件向量 `c_all`。

### 6.4.1 Sinusoidal time embedding

对任意时间标量 `t`，正弦位置编码定义为

`freq_k = exp(- log(10000) * k / max(H-1, 1))`

`PE(t) = [sin(t freq_1), ..., sin(t freq_H), cos(t freq_1), ..., cos(t freq_H)]`

其中 `H = floor(d_model / 2)`。

### 6.4.2 流时间分支

`t_flow` 先做正弦嵌入，再过一个两层 MLP：

`c_flow = MLP_flow(PE(t_flow))`

其结构为

`Linear(d_model, 4 d_model) -> SiLU -> Linear(4 d_model, d_model)`

### 6.4.3 生物时间分支

`t_src` 和 `t_tgt` 分别嵌入后在最后一维拼接，再过另一套 MLP：

`c_bio = MLP_bio([PE(t_src) || PE(t_tgt)])`

其中：

`MLP_bio: Linear(2 d_model, 2 d_model) -> SiLU -> Dropout -> Linear(2 d_model, d_model)`

### 6.4.4 条件聚合

在当前任务设置中，不显式输入 cell type condition，因此全局条件向量为

`c_all = c_flow + c_bio`

如果未来启用 cell type 分支，则还会再加一个 `c_type`。

## 6.5 AdaLNTransformerBlock

每个 `AdaLNTransformerBlock` 包含：

1. 一个自注意力子层
2. 一个前馈子层
3. 两组由条件向量生成的 AdaLN 调制参数

### 6.5.1 自注意力子层

采用

`nn.MultiheadAttention(d_model, nhead, batch_first=True)`

### 6.5.2 前馈子层

当前实现中的 FFN 为

`Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)`

需要注意，当前代码中前馈激活是 `GELU`，而不是 `SiLU`。

### 6.5.3 AdaLN 调制

每层分别用两组线性投影从条件向量 `c_all` 生成注意力前和 FFN 前的调制参数：

`[gamma_attn, beta_attn] = W_attn c_all + b_attn`

`[gamma_ffn, beta_ffn] = W_ffn c_all + b_ffn`

对于任一子层输入 `h`，其 AdaLN 调制形式为

`AdaLN(h; gamma, beta) = LN(h) * (1 + gamma) + beta`

于是注意力和前馈两步分别写作：

`hat(h)_attn = AdaLN(h; gamma_attn, beta_attn)`

`h' = h + MHA(hat(h)_attn)`

`hat(h)_ffn = AdaLN(h'; gamma_ffn, beta_ffn)`

`h'' = h' + FFN(hat(h)_ffn)`

当前实现里，CLS 位置不参与调制，会保持原始 `LayerNorm` 输出。

### 6.5.4 AdaLN 参数共享选项

当前模型支持两种模式：

1. **per-layer projection**：每层使用独立的 `attn_mod` 和 `ffn_mod`
2. **shared projection**：若 `share_adaln_projection = true`，则所有层共享同一组调制投影

默认配置中该选项为 `false`。

## 6.6 输出头

主输出头为 `ExprHead`：

`Linear(d_model, d_model) -> SiLU -> Dropout -> Linear(d_model, 1)`

对每个 token 位置输出一个标量。

模型还实现了可选的：

1. `BinHead`
2. `TypeHead`
3. `PseudotimeHead`

但在当前主线配置中这些辅助头均未启用：

1. `aux_bin_ce = false`
2. `n_cell_types = 0`
3. `pseudotime_head = false`

## 6.7 当前主线中 `forward()` 的语义

在代码层面，`model.forward()` 先输出每个位置对终点表达的预测，再在当前训练器包装中把该输出解释为速度场项。训练器包装逻辑为：

`pred = model.forward(...)[:, 1:]`

`pred_tgt = x_tau + (1 - tau) pred`

并用 `pred` 与真实速度 `u - s` 做监督。

因此，在当前主线写法中，可以把 `pred[:, 1:]` 视为 **用于 Euler 更新的条件速度预测**。

---

# 7. 训练流程

训练逻辑实现于 `route_a/train.py:RouteATrainer`。

## 7.1 初始化

`RouteATrainer.__init__()` 负责完成：

1. 读取训练配置
2. 载入 protocol bundle
3. 构造固定长度序列与 mask state
4. 载入或自动生成评估资产
5. 读取 PCA 模型的 `mean_` 和 `components_` 到当前设备
6. 建立已观测发育时间点之间的训练时间对

## 7.2 已观测时间点之间的训练时间对

对已观测发育时间集合 `T_obs`，训练器构造所有满足方向约束的时间对：

`P_train = {(t_a, t_b): t_a in T_obs, t_b in T_obs, t_a != t_b, direction constraint}`

具体地：

1. 当 `direction_mode = forward_only` 时，仅保留 `t_a < t_b`
2. 当 `direction_mode = bidirectional` 时，保留双向时间对

这意味着 near/far extrapolation 训练时只学习正向时间演化，而 interpolation 训练时允许时间双向建模。

## 7.3 单步训练算法

下面给出一个训练步 `_train_step()` 的完整计算流程。

### Step 1：采样时间对

从 `P_train` 中均匀随机取一个 `(t_src, t_tgt)`。

### Step 2：采样源细胞群体与目标细胞群体

分别从 `X_(t_src)` 和 `X_(t_tgt)` 中采样 `B` 个细胞，构成源细胞群体与目标细胞群体，记为

`S = {s_i}_(i=1)^B`

`U = {u_j}_(j=1)^B`

若某个时间点的细胞数少于 `B`，则允许有放回采样。

### Step 3：定义两组细胞群体之间的代价矩阵

对任意 `s_i` 和 `u_j`，使用平均平方欧氏距离：

`C_ij = ||s_i - u_j||_2^2 / d`

对应函数为 `utils.pairwise_mean_sq_dist()`。

### Step 4：计算熵正则 OT 计划

在均匀边缘分布下，求解

`Pi^* = argmin_(Pi in U(a,b)) <Pi, C> - epsilon H(Pi)`

其中：

1. `a = 1/B * 1`
2. `b = 1/B * 1`
3. `H(Pi) = - sum_ij Pi_ij (log Pi_ij - 1)`

实现函数 `sinkhorn_uniform_transport()` 采用 log-space Sinkhorn 迭代，默认参数：

1. `epsilon = 0.05`
2. `max_iter = 100`

### Step 5：从 OT 计划抽样训练对

将传输计划 `Pi^*` 展平为概率分布后，从中有放回采样 `B` 个索引，得到训练对

`{(s_i, u_j)}_(k=1)^B`

这一步由 `sample_transport_pairs()` 完成。

### Step 6：采样流时间

使用分层采样函数 `sample_t_stratified()` 生成 `tau`，采样分布为：

1. 40% 来自 `[0.0, 0.2]`
2. 20% 来自 `[0.2, 0.4]`
3. 15% 来自 `[0.4, 0.6]`
4. 15% 来自 `[0.6, 0.8]`
5. 10% 来自 `[0.8, 1.0]`

这使模型在接近源端的局部区域获得更高采样密度。

### Step 7：构造中间态与真实速度

对每个配对 `(s, u)`，定义

`x_tau = (1 - tau) s + tau u`

`v^* = u - s`

### Step 8：模型前向与目标重建

模型接收 `(x_tau, tau, t_src, t_tgt)`，输出预测速度

`v_theta = v_theta(x_tau, tau, t_src, t_tgt)`

训练器再根据

`hat(u) = x_tau + (1 - tau) v_theta`

构造目标表达预测，并对 `hat(u)` 做 `clamp(min=0)`，以保持表达非负。

---

# 8. 训练目标

总损失是多个项的加权和，分别约束局部配对一致性、高维分布几何、低维流形结构以及群体统计量。

## 8.1 条件流匹配损失

核心损失是速度回归：

`L_cfm = E ||v_theta - v^*||_2^2`

在实现中对应

`torch.mean((pred_vel - true_vel) ** 2)`

## 8.2 HVG 空间 Sinkhorn 距离

在高维 fixed-HVG 空间中，对预测目标细胞群体 `hat(U)` 与真实目标细胞群体 `U` 计算 Sinkhorn 距离：

`L_hvg = Sinkhorn(hat(U), U)`

当前 `sinkhorn_distance()` 的实现先用

`C_ij = ||hat(u)_i - u_j||_2^2 / d`

构造代价矩阵，再求

`sum_ij Pi_ij C_ij`

## 8.3 PCA 空间 Sinkhorn 距离

为了约束低维流形位置，还在固定 PCA 空间中计算分布距离。记固定 PCA 投影为

`PCA(x) = (x - mu_PCA) W_PCA^T`

则

`L_pca = Sinkhorn(PCA(hat(U)), PCA(U))`

其中 `mu_PCA` 和 `W_PCA` 都来自仅在已观测发育时间点细胞上拟合的固定 PCA 模型。

## 8.4 分布均值损失

为了匹配群体平均表达，定义

`L_mean = || mean(hat(U)) - mean(U) ||_2^2`

## 8.5 方差正则

当开启方差约束时，比较预测与真实细胞群体在每个基因上的方差。默认在 log-variance 空间中比较：

`sigma_pred^2 = Var(hat(U))`

`sigma_real^2 = Var(U)`

`L_var = mean((log(sigma_pred^2 + eps) - log(sigma_real^2 + eps))^2)`

当前默认：

1. `variance_loss_use_log = true`
2. `variance_loss_eps = 1e-4`
3. `loss_weights.variance = 0.0`

因此默认配置下该项不参与总损失。

## 8.6 总目标

总损失写作

`L_total = lambda_cfm L_cfm + lambda_hvg L_hvg + lambda_pca L_pca + lambda_mean L_mean + lambda_var L_var`

当前三套主配置的默认权重一致：

1. `lambda_cfm = 1.0`
2. `lambda_hvg = 0.5`
3. `lambda_pca = 0.1`
4. `lambda_mean = 0.1`
5. `lambda_var = 0.0`

---

# 9. 优化与训练管理

## 9.1 模型与训练默认配置

当前主线配置文件包括：

1. `config_mouse_route_a_interpolation.yaml`
2. `config_mouse_route_a_near.yaml`
3. `config_mouse_route_a_far.yaml`

默认模型超参数为：

1. `top_k_genes = 4096`
2. `d_model = 512`
3. `nhead = 8`
4. `d_ff = 2048`
5. `nlayers = 8`
6. `dropout = 0.1`
7. `value_encoder.max_value = 32.0`
8. `adaln.share_projection = false`

## 9.2 优化器与混合精度

训练器使用：

1. `AdamW`
2. `lr = 2e-4`
3. `weight_decay = 1e-4`
4. AMP 自动混合精度
5. `torch.autocast(..., dtype=torch.bfloat16)`
6. `torch.cuda.amp.GradScaler`

## 9.3 梯度裁剪

每步更新前会执行

`clip_grad_norm_(theta, 1.0)`

其中默认 `grad_clip = 1.0`。

## 9.4 训练循环与日志

当前默认：

1. `epochs = 10`
2. `iters_per_epoch = 500`
3. `log_every_steps = 20`
4. `val_every_steps = 500`
5. `save_every_epochs = 1`

训练过程中：

1. 每个训练步记录当前损失分项和 `src_time` / `tgt_time`
2. 每到验证步，会在未观测目标发育时间点上做一次小规模生成与评估
3. 训练结束后运行一次最终全量评估

日志写入：

`<exp_dir>/train_log.jsonl`

checkpoint 保存到：

`<exp_dir>/checkpoints/`

## 9.5 GPU 显存预留

`route_a/main.py` 中的 `GpuMemoryReservation` 支持在模型初始化前预留一部分 GPU 显存，以减少多进程环境下的资源争抢。该机制不是方法本身的一部分，但属于当前实现中的工程性稳定措施。

---

# 10. 推理与目标时间分布生成

## 10.1 基本思想

训练完成后，模型通过多步 Euler 更新把源时间点细胞逐步演化到目标时间点。设初始状态为源细胞表达 `x^(0) = x_src`，总步数为 `K`，则每一步：

`tau_k = k / K`

`v_k = v_theta(x^(k), tau_k, t_src, t_tgt)`

`x^(k+1) = clamp(x^(k) + (1 / K) v_k, 0, +inf)`

最终输出 `x^(K)` 作为目标时间点的生成表达。

## 10.2 RouteATrainer._rollout()

当前主线在训练器中使用 `_rollout()` 实现时间演化。默认：

1. `n_steps = 10`
2. `dt = 1 / n_steps`
3. 每一步都把当前状态作为新的输入

这对应一个显式 Euler 积分器。

## 10.3 未观测目标时间点的生成来源采样

对于某个未观测目标发育时间点 `t`：

1. 先由 protocol 解析得到生成来源集合 `S(t)`
2. 再按来源权重决定每个源时间点需要采样多少细胞
3. 对每个来源单独执行 `_rollout(src_expr, t_src, t_tgt, n_steps)`
4. 最后将各来源生成结果拼接，组成该目标时间点的预测细胞群体

若某个未观测目标时间点真实细胞数为 `N_t`，而各生成来源权重为 `w_1, ..., w_m`，则分配到第 `k` 个源时间点的生成样本数近似为

`N_t^(k) approx round(w_k N_t)`

最后一个来源会吸收舍入误差，以保证总样本数与 `N_t` 一致。

## 10.4 prediction artifact

生成结果会被写成 artifact，包括：

1. `pred_by_time`
2. `real_by_time`
3. `obs_names_by_time`
4. `artifact_arrays.npz`
5. `artifact_summary.json`

这使得训练、验证和离线评估可以完全解耦。

---

# 11. 评估指标

评估逻辑实现于 `route_a/eval.py`。所有针对未观测目标发育时间点的指标都在固定评估资产上完成。

## 11.1 Wasserstein-HVG

对每个未观测目标发育时间点，定义

`W_HVG(t) = Sinkhorn(X_t^pred, X_t^real)`

它衡量预测分布与真实分布在 fixed-HVG 高维表达空间中的几何距离。

## 11.2 Wasserstein-PCA50

记固定 PCA 投影为 `Phi`，则

`W_PCA(t) = Sinkhorn(Phi(X_t^pred), Phi(X_t^real))`

该指标更侧重于低维流形结构的一致性。

## 11.3 Distribution mean R^2

对每个未观测目标发育时间点，先计算预测细胞群体与真实细胞群体在每个基因上的均值：

`mu_t^pred = mean(X_t^pred)`

`mu_t^real = mean(X_t^real)`

再计算二者的 Pearson 相关系数平方：

`R_dist^2(t) = Corr(mu_t^pred, mu_t^real)^2`

## 11.4 NN-matched per-cell R^2

为比较单细胞层面的局部一致性，先对每个预测细胞在对应真实目标时间细胞集合中寻找最近邻：

`NN(x_i^pred) = argmin_(x in X_t^real) ||x_i^pred - x||_2`

再计算每个预测细胞与其最近真实邻居之间的 Pearson `R^2`，最后取平均：

`R_NN^2(t) = 1 / N_t^pred sum_i Corr(x_i^pred, NN(x_i^pred))^2`

## 11.5 Cell-type accuracy

先将预测样本投影到固定 PCA 空间，再用冻结的 kNN 分类器得到预测标签：

`hat(y)_i = kNN(Phi(x_i^pred))`

然后在真实目标时间点的 PCA embedding 中为每个预测样本找最近邻真实细胞，把其标签记为 `y_i^match`，计算：

`Acc_type(t) = 1 / N_t^pred sum_i 1[hat(y)_i = y_i^match]`

因此这里比较的是“预测细胞被归到的类型”和“它最近真实邻居的类型”是否一致。

## 11.6 Marker MMD

对于每个细胞类型 `c`，设其 marker gene 子空间索引集合为 `M_c`。在该子空间上，对预测和真实细胞类型群体计算 RBF-kernel MMD：

`MMD_c = MMD(X_t^pred(c, M_c), X_t^real(c, M_c))`

最终对可计算的细胞类型求平均：

`MarkerMMD(t) = mean_c MMD_c`

RBF 核宽度通过预测和真实样本合并后的 pairwise squared distance 的中位数自适应确定。

## 11.7 聚合方式

对所有未观测目标发育时间点逐个计算上述指标后，最终报告它们在目标时间集合上的均值，例如：

`mean_t W_HVG(t)`

`mean_t W_PCA(t)`

`mean_t R_dist^2(t)`

`mean_t R_NN^2(t)`

`mean_t Acc_type(t)`

`mean_t MarkerMMD(t)`

---

# 12. 当前实现中的关键默认设置

为了便于论文撰写，下面将当前主线实现最重要的默认设置集中列出：

1. 数据空间：fixed HVG，维度 `4096`
2. 表达预处理：library-size normalize 到 `10000`，`log1p`，clip 到 `[0, 10]`
3. 模型骨干：8 层 AdaLN-Transformer，`d_model = 512`，`nhead = 8`，`d_ff = 2048`
4. 时间条件：`t_flow`、`t_src`、`t_tgt`
5. OT 参数：`epsilon = 0.05`，`max_iter = 100`
6. 优化器：AdamW，`lr = 2e-4`，`weight_decay = 1e-4`
7. 训练批大小：`batch_size = 24`
8. Euler 推理步数：`n_steps = 10`
9. 验证时每个未观测目标发育时间点默认最多采样 `256` 个真实细胞
10. 最终评估默认使用全部目标时间点细胞

---

# 13. 方法小结

从方法论角度看，当前 `scFiT` 的时序单细胞发育分布预测方法可以概括为：

1. 先在一个冻结的 HVG 坐标系中统一表示所有细胞；
2. 再通过 protocol 明确已观测发育时间点与未观测目标时间点的划分，以及目标时间的生成来源；
3. 然后在已观测发育时间点之间使用 OT 软配对构造分布级时间监督；
4. 用带 AdaLN 时间调制的 Transformer 学习条件速度场；
5. 最后通过 Euler rollout 生成目标发育时间点的细胞群体，并在固定 PCA、kNN 和 marker 标尺下进行分布级评价。

如果将其写成论文中的一句话方法摘要，可以表述为：

> We formulate temporal single-cell developmental distribution prediction as conditional distributional transport in a fixed HVG expression space. scFiT learns a time-conditioned velocity field over single-cell populations by coupling observed source-target batches with entropic optimal transport and optimizing a conditional flow-matching objective. At inference, the learned dynamics are numerically integrated to evolve source populations toward held-out developmental times, and the generated populations are evaluated using both geometric and biologically informed distributional metrics.
