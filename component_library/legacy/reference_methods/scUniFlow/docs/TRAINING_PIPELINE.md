# scUniFlow Current Training Pipeline

本文档依据当前仓库实现整理，目标是给论文写作提供一份与代码一致的训练与推理说明。  
对应主实现文件：

- [train.py](${SCUNIFLOW_ROOT:-.}/train.py)
- [model.py](${SCUNIFLOW_ROOT:-.}/model.py)
- [gene_mamba.py](${SCUNIFLOW_ROOT:-.}/gene_mamba.py)
- [data.py](${SCUNIFLOW_ROOT:-.}/data.py)
- [config.yaml](${SCUNIFLOW_ROOT:-.}/config.yaml)

## 1. Task Definition

当前 `scUniFlow` 的主任务是单细胞表达重建：

- 输入单细胞全基因表达谱
- 随机 mask 一部分基因
- 将基因按照预先生成的二维谱坐标映射到固定网格
- 用二维层次化双向 Mamba 编码全局上下文
- 仅在被 mask 的基因位置预测终点表达 `x_1`
- 推理时将 `x_1` 预测转换为速度，并用 Euler rollout 完成重建

从当前代码看，系统已经不是早期的“软路由 + 两阶段训练”版本，而是：

- 固定硬路由 `hard spectral routing`
- 单阶段训练
- 以 `epoch` 为训练主调度单位
- 以 `step` 为日志与验证主调度单位

## 2. End-to-End Pipeline

当前训练流程如下：

1. 读取 `h5ad` 数据或预定义数据键名对应的数据集。
2. 在可能时按 donor 做 train/val 划分；否则回退为 cell-level 随机划分。
3. 对每个细胞构造：
   - `x_1`: `log1p` 后的真实表达
   - `x_0`: 经验高斯先验噪声
   - `mask_indicator`: 被 mask 的基因位置
4. 加载训练前离线生成好的 `spectral_coords`。
5. 根据二维坐标建立固定硬路由，把基因放到二维网格 slot。
6. 对每个训练 batch：
   - 分层采样时间 `t`
   - 构造 `x_t = (1-t)x_0 + t x_1`
   - 非 mask 位置保留真值，mask 位置放入 `x_t`
   - 经硬路由送入二维 GeneMamba 编码器提取上下文
   - 仅在被 mask 位置预测 `x_1`
   - 计算加权重建损失与 Pearson 类损失
7. 按 `val_every_steps` 在验证集上做 rollout 验证。
8. 保存 `checkpoint_best.pt`、`checkpoint_last.pt` 和 `best_metrics.json`。

## 3. Data Pipeline

### 3.1 Input Data

数据读取逻辑在 [data.py](${SCUNIFLOW_ROOT:-.}/data.py)。

当前支持两种输入方式：

- 预定义键名：
  - `mtg_full`
  - `dlpfc_full`
  - `mtg_toy`
  - `dlpfc_toy`
- 直接传入任意 `h5ad` 路径

原始输入矩阵记为：

$$
X_{\text{raw}} \in \mathbb{R}_{\ge 0}^{N \times G}
$$

其中：

- `N` 为细胞数
- `G` 为基因数

### 3.2 Gene IDs and Column Order

当前实现不对 `h5ad` 的列顺序做重排，而是保留原始列顺序。  
同时在 [data.py](${SCUNIFLOW_ROOT:-.}/data.py) 中为每一列生成一个全局 `gene_id`：

- `ensembl_ids = adata.var.index`
- `gene_ids[i]` 表示第 `i` 列基因在 Ensembl ID 字典序中的排名

因此：

- 表达矩阵列顺序保持原始 `h5ad`
- `gene_ids` 作为模型侧的基因身份编号

### 3.3 Train/Val Split

主入口优先使用 [data.py](${SCUNIFLOW_ROOT:-.}/data.py) 中的 `load_sea_ad_split(...)`。

当前划分逻辑为：

- 若 `adata.obs` 中存在 `donor_id` 且 donor 数量不少于 2，则执行 donor-level 划分
- 否则回退到 cell-level 随机划分

donor-level 划分使用贪心装箱策略，使验证集细胞数接近 `val_ratio`。  
这意味着当前默认验证设定是：

- 优先 donor-level unseen split
- 不满足条件时才退化为 random cell split

### 3.4 Per-Cell Sample

训练样本由 `scRNADataset` 构造。对每个细胞输出：

- `g_id`: 基因 ID 序列，形状 `(G,)`
- `x_1`: 真实终点表达
- `x_0`: 初始噪声
- `mask_indicator`: mask 指示向量

其中：

$$
x_1 = \log(1 + x_{\text{raw}})
$$

### 3.5 Prior for `x_0`

当前默认先验为经验高斯先验 `empirical`。  
其均值和标准差在训练集所有非零表达值的 `log1p` 空间中统计：

$$
x_0 \sim \mathcal{N}(\mu_{\text{nz}}, \sigma_{\text{nz}}^2)
$$

其中 `\mu_{\text{nz}}, \sigma_{\text{nz}}` 来自训练集非零表达的 `log1p` 分布。

验证时会复用训练集统计量，即：

$$
x_0^{\text{val}} \sim \mathcal{N}(\mu_{\text{train}}, \sigma_{\text{train}}^2)
$$

## 4. Mask Strategy

### 4.1 Training Mask

训练 mask 在 [data.py](${SCUNIFLOW_ROOT:-.}/data.py) 的 `__getitem__` 中生成。

默认配置：

- 非零基因 mask 比例：`0.2`
- 零值基因 mask 比例：`0.0`

因此训练时主要从非零基因中抽样 mask：

$$
\mathcal{M}_{\text{train}} \subset \{g \mid x_1(g) > 0\}
$$

并且：

$$
|\mathcal{M}_{\text{train}}| = \max\left(1, \left\lfloor 0.2 \cdot |\{g \mid x_1(g) > 0\}| \right\rfloor \right)
$$

### 4.2 Validation Mask

验证 mask 不复用数据集内部随机 mask，而是在 [train.py](${SCUNIFLOW_ROOT:-.}/train.py) 中由 `_make_val_mask(...)` 单独构造。

默认配置：

- `nonzero_mask_ratio = 0.2`
- `zero_mask_ratio = 0.0`
- `mask_seed = 1234`

验证时每个细胞的 mask 由 `val_mask_seed + cell_id` 确定，因此验证 mask 是可复现的。

## 5. Spectral Hard Routing

### 5.1 External Spectral Coordinates

当前训练不会在线生成谱坐标。  
主入口要求先提供一个离线生成好的二维坐标文件：

$$
C \in \mathbb{R}^{G \times 2}
$$

对应配置项为 `training.spectral_coords`。

若没有提供该文件，训练会直接报错。

### 5.2 Hilbert-Based Slot Assignment

当前 `scUniFlow` 使用二维谱坐标生成固定硬路由，核心逻辑在 [model.py](${SCUNIFLOW_ROOT:-.}/model.py)：

- `_hilbert_distances(...)`
- `build_hard_permutation_from_coords(...)`
- `SpectralRouter.init_from_spectral(...)`

当前初始化时：

- `slot_order_mode = "hilbert"`
- `gene_order_mode = "hilbert"`

因此二维坐标先被映射为 Hilbert 顺序，再映射到二维网格 slot。

### 5.3 Hard Routing Mapping

路由本质上是一个固定置换：

$$
\pi : \{1,\dots,G\} \to \{1,\dots,L\}
$$

其中：

- `G` 为真实基因数
- `L = H \times W` 为网格总 slot 数

表达值与 mask 都会被 scatter 到网格位置：

$$
x_{\text{perm}}[\pi(g)] = x[g]
$$

$$
m_{\text{perm}}[\pi(g)] = m[g]
$$

未被真实基因占据的 slot 作为 dummy slot 保留。

### 5.4 Gene Identity Embedding

路由器中显式包含基因身份嵌入：

$$
E_{\text{gene}} \in \mathbb{R}^{G \times D}
$$

训练时先根据固定路由将其排列到网格 slot，再送入 Mamba 编码器。

## 6. Model Architecture

当前整体模型 `scRNAModel` 由三部分组成：

1. `SpectralRouter`
2. `GeneMamba`
3. `VectorFieldNet`

### 6.1 Context Construction

训练时首先构造中间状态：

$$
x_t = (1 - t)x_0 + t x_1
$$

随后仅在 mask 位置使用 `x_t`，未 mask 位置继续使用真实值 `x_1`：

$$
\tilde{x}(g) =
\begin{cases}
x_t(g), & m(g)=1 \\
x_1(g), & m(g)=0
\end{cases}
$$

即：

$$
\tilde{x} = m \odot x_t + (1-m)\odot x_1
$$

### 6.2 GeneMamba Input

路由后的 Mamba 输入由三部分相加得到：

- 基因身份嵌入 `x_feat`
- 表达值嵌入 `expr_emb`
- mask 嵌入 `mask_emb`

表达值嵌入为：

$$
e_{\text{expr}} = \phi_{\text{expr}}(x_{\text{perm}})
$$

mask 嵌入为：

$$
e_{\text{mask}} = \phi_{\text{mask}}(m_{\text{perm}})
$$

最终输入：

$$
h^{(0)} = x_{\text{feat}} + e_{\text{expr}} + e_{\text{mask}}
$$

### 6.3 2-D Hierarchical Mamba Encoder

`GeneMamba` 将长度为 `L` 的序列 reshape 为二维网格：

$$
h^{(0)} \in \mathbb{R}^{B \times H \times W \times D}
$$

然后堆叠多个 `HierarchicalMambaBlock`。每个 block 包含：

1. 行方向双向 Mamba
2. 列方向双向 Mamba
3. 前馈网络 FFN

其形式可写为：

$$
h' = h + \text{RowBiMamba}(\text{LN}(h))
$$

$$
h'' = h' + \text{ColSSM}(\text{LN}(h'))
$$

$$
h^{\text{out}} = h'' + \text{FFN}(\text{LN}(h''))
$$

### 6.4 VectorFieldNet

`VectorFieldNet` 输入为：

- 当前 mask 位置状态 `x_t`
- 对应位置的上下文 `context`
- 时间 `t`

时间先做 MLP 嵌入：

$$
e_t = \phi_t(t)
$$

再与上下文和当前状态拼接：

$$
z = [\text{context}, e_t, x_t]
$$

最后输出 mask 位置的终点预测：

$$
\hat{x}_1 = f_\theta(z)
$$

## 7. Training Objective

### 7.1 x-Prediction

当前模型不直接训练速度场，而是直接预测被 mask 基因的终点表达：

$$
\hat{x}_1 = f_\theta(x_t, c_t, t)
$$

其中 `c_t` 是 GeneMamba 提供的上下文。

### 7.2 Total Loss

当前训练总损失为：

$$
\mathcal{L}
= \lambda_{\text{weighted}} \mathcal{L}_{\text{weighted-x1}}
+ \lambda_{\text{pearson}} \mathcal{L}_{\text{pearson}}
+ \lambda_{\text{cell-pearson}} \mathcal{L}_{\text{cell-pearson}}
$$

默认权重：

$$
\lambda_{\text{weighted}} = 1.0,\quad
\lambda_{\text{pearson}} = 0.1,\quad
\lambda_{\text{cell-pearson}} = 0.1
$$

### 7.3 Weighted Reconstruction Loss

`weighted_recon_loss` 先在所有 mask 位置计算 Huber loss：

$$
\ell_i = \text{Huber}(\hat{x}_{1,i}, x_{1,i}; \delta = 0.5)
$$

然后对每个细胞中真实表达最高的 `top_k = 20` 个 mask 基因赋予更高权重：

$$
w_i =
\begin{cases}
3, & i \in \text{TopK}(x_1) \\
1, & \text{otherwise}
\end{cases}
$$

最终：

$$
\mathcal{L}_{\text{weighted-x1}}
= \frac{\sum_i w_i \ell_i}{\sum_i \mathbf{1}_{i \in \text{mask}}}
$$

### 7.4 Global Pearson Loss

全局 Pearson 损失定义为：

$$
\mathcal{L}_{\text{pearson}} = 1 - \rho(\hat{x}_1, x_1)
$$

其中 Pearson 是在当前 batch 全部 mask 位置拼接后计算。

### 7.5 Cell-Wise Pearson Loss

先对每个细胞单独计算 mask 位置上的 Pearson：

$$
\rho_b = \rho(\hat{x}_1^{(b)}, x_1^{(b)})
$$

再求平均：

$$
\mathcal{L}_{\text{cell-pearson}} =
\frac{1}{B}\sum_{b=1}^B (1 - \rho_b)
$$

### 7.6 Auxiliary Logged Loss

代码中还会记录一个未加权的 Huber loss：

$$
\mathcal{L}_{x1} = \text{Huber}(\hat{x}_1, x_1; \delta=0.5)
$$

它会写入日志，但当前不参与总损失。

## 8. Time Sampling

训练时 `t` 采用分层采样，而不是简单均匀采样。  
当前各区间占比为：

- `[0.0, 0.2)`: 40%
- `[0.2, 0.4)`: 20%
- `[0.4, 0.6)`: 15%
- `[0.6, 0.8)`: 15%
- `[0.8, 1.0)`: 10%

因此训练显式偏向较小 `t` 区域。

## 9. Inference and Validation

### 9.1 Validation Initialization

验证阶段对 mask 位置先清零得到 `observed`，随后再构造固定初始值：

- mask 图样由 `val_mask_seed + cell_id` 决定
- 初始噪声由 `val_noise_seed + cell_id` 决定

因此当前验证是 deterministic 的：

- 同一个 checkpoint 多次验证结果应一致
- 不同 checkpoint 之间使用相同的验证 mask 和相同的 masked 初始噪声

### 9.2 Rollout Dynamics

推理时每一步先预测终点：

$$
\hat{x}_1 = f_\theta(x_t, c_t, t)
$$

再反推出速度：

$$
\hat{v}_t = \frac{\hat{x}_1 - x_t}{1 - t + \epsilon}
$$

其中：

$$
\epsilon = 10^{-3}
$$

### 9.3 Euler Update

若总 rollout 步数为 `N_{\text{steps}}`，则：

$$
\Delta t = \frac{1}{N_{\text{steps}}}
$$

Euler 更新为：

$$
x_{t+\Delta t} = x_t + \hat{v}_t \Delta t
$$

并在每一步后裁剪到非负：

$$
x_{t+\Delta t} = \max(x_{t+\Delta t}, 0)
$$

同时非 mask 位置在每一步都会被重新写回已知观测值。

### 9.4 Validation Metrics

当前训练内验证输出 7 个指标：

- `val_mse`
- `val_rmse`
- `val_mae`
- `val_pearson_cell`
- `val_pearson_all`
- `val_spearman_cell`
- `val_pearson_top20`

其中：

- `val_pearson_cell`: 每个细胞 mask 位置上的 Pearson，再对细胞求平均
- `val_pearson_all`: 所有 mask 位置拼接后的全局 Pearson
- `val_spearman_cell`: 每个细胞 mask 位置上的 Spearman，再对细胞求平均
- `val_pearson_top20`: 每个细胞取真实表达最高的 20 个 mask 基因后计算 Pearson，再对细胞求平均

## 10. Training Schedule

### 10.1 Single-Stage Epoch Training

当前训练入口已经重构为单阶段训练。  
主函数调用的是 `train_epochs(...)`，不再执行旧版 phase1/phase2 两阶段流程。

训练调度规则为：

- `epochs` 控制总训练轮数
- 一个 epoch 等于完整跑完一轮训练集
- 学习率调度使用 `CosineAnnealingLR`
- `scheduler.step()` 在每个 epoch 结束后执行一次

### 10.2 Step-Based Validation and Logging

虽然训练主单位是 epoch，但验证和日志仍以 step 为主：

- `log_every_steps`: 每多少 step 写一次训练日志
- `val_every_steps`: 每多少 step 运行一次验证
- `save_every_epochs`: 每多少 epoch 更新一次 `checkpoint_last.pt`

如果最后一个训练 step 没有刚好落在 `val_every_steps` 上，代码会在训练结束前额外补一次验证。

## 11. Best Checkpoint Selection

当前 best checkpoint 不是只看单一指标。  
代码对以下 7 个验证指标做“多数改进”判断：

- `val_mse`，越小越好
- `val_rmse`，越小越好
- `val_mae`，越小越好
- `val_pearson_cell`，越大越好
- `val_pearson_all`，越大越好
- `val_spearman_cell`，越大越好
- `val_pearson_top20`，越大越好

规则是：

- 若是第一次验证，直接保存 best
- 否则只有当“可比较指标中超过 50% 改进”时，才覆盖 `checkpoint_best.pt`

因此当前 `checkpoint_best.pt` 表示“多数指标同步改进”的最优点，而不是单独某一个指标最优点。

## 12. Checkpoint, Logging, and Outputs

### 12.1 Experiment Directory

实验目录为：

$$
\text{exp\_dir} = \text{training.experiment\_dir} / \text{run\_name}
$$

### 12.2 Files Produced by Training

当前训练会生成以下文件：

- `config.yaml`
- `split_info.json`
- `routing_info.json`
- `train_log.jsonl`
- `loss_curve.png`
- `best_metrics.json`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_last.pt`

当前不会生成旧版文档中的这些文件：

- `phase1_step*.pt`
- `phase1_final.pt`
- `phase2_step*.pt`
- `phase2_final.pt`
- `model_final.pt`

### 12.3 Checkpoint Content

当前 checkpoint payload 主要包含：

- `format_version`
- `training_mode = "single_stage_epoch"`
- `phase = "single"`
- `epoch`
- `step`
- `model_state_dict`
- `prior_mean`
- `prior_std`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `best_record`
- `best_selection_summary`

### 12.4 Logging

当前只有一个主日志文件：`train_log.jsonl`。  
它以 step 为粒度记录训练和验证信息。

训练 step 记录通常包括：

- `epoch`
- `epoch_step`
- `step`
- `lr`
- `train_loss`
- `x1_loss`
- `weighted_x1_loss`
- `pearson_loss`
- `cell_pearson_loss`

若该 step 同时触发验证，还会附加：

- `val_mse`
- `val_rmse`
- `val_mae`
- `val_pearson_cell`
- `val_pearson_all`
- `val_spearman_cell`
- `val_pearson_top20`
- `best_selection_summary`

## 13. Current Default Hyperparameters

按当前 [config.yaml](${SCUNIFLOW_ROOT:-.}/config.yaml) 的默认值：

- `n_genes = 32738`
- `grid_size = 181`
- `d_model = 512`
- `d_state_row = 64`
- `d_state_col = 16`
- `n_layers = 8`
- `batch_size_per_gpu = 25`
- `train nonzero mask = 0.2`
- `train zero mask = 0.0`
- `epochs = 80`
- `lr = 5e-4`
- `min_lr = 1e-6`
- `val_every_epochs = 1`
- `save_every_epochs = 10`
- `log_every_steps = 100` 是代码默认值，不在当前 YAML 中显式写出
- `validation.n_steps = 10`
- `validation.mask_seed = 1234`
- `validation.noise_seed = 5678`
- `validation.max_batches = 10`
- `validation.batch_size = 25`

需要注意：训练入口会把 `val_every_epochs` 换算成 `val_every_steps`。  
若 YAML 里直接提供 `training.val_every_steps`，则会覆盖这一换算结果。

## 14. Code-Level Caveats for Writing

以下几点对于论文写作很重要，因为它们是当前代码的真实行为。

### 14.1 Current Code Is Not Two-Stage

当前主训练流程已经不是两阶段训练。  
论文如果写“phase1 warmup + phase2 finetune”将与当前实现不符，除非你明确是在描述历史版本。

### 14.2 Validation Is Deterministic

当前训练内验证不是每次重新随机采样：

- mask 固定
- masked 初始噪声固定

这使得不同 checkpoint 之间的验证结果可以直接比较，减少了验证噪声。

### 14.3 Best Model Selection Uses Majority Rule

当前 `checkpoint_best.pt` 不是 “val_rmse 最优模型”，而是“多数验证指标改进时覆盖保存”的模型。  
写论文时如果要描述 early model selection，最好明确这一点。

### 14.4 Total Loss Does Not Include Plain Huber

日志中虽然存在 `x1_loss`，但它只是记录项，不参与总损失。  
总损失只由：

- `weighted_x1`
- `pearson`
- `cell_pearson`

三项组成。

### 14.5 Routing Temperature Is Kept Only for Compatibility

当前 `temperature` 参数虽然保留在接口中，但对路由本身已经不起作用。  
当前路由是固定、确定性的硬路由。

### 14.6 Some Model Hyperparameters in YAML Are Not Fully Wired

当前 `config.yaml` 中虽然写有：

- `n_layers`
- `d_state_row`
- `d_state_col`

但 `scRNAPipelineTrainer` 构造 `scRNAModel` 时实际只显式传入了：

- `n_genes`
- `d_model`
- `grid_size`

因此当前代码运行时，`GeneMamba` 仍会使用类定义中的默认值：

- `n_layers = 8`
- `d_state_row = 64`
- `d_state_col = 16`

在当前默认配置下，YAML 数值与类默认值恰好一致，所以实验行为没有偏差；  
但若论文要写“这些参数由配置文件动态控制”，那在当前代码下并不严格准确。

## 15. Suggested Paper Description

如果需要一句尽量贴近当前实现的论文式概述，可以写成：

> We train scUniFlow in a single-stage epoch-based regime on full-gene single-cell expression matrices. For each cell, 20% of nonzero genes are masked, genes are deterministically routed onto a fixed 2-D spectral grid using Hilbert-ordered hard routing, and a hierarchical bidirectional GeneMamba backbone encodes the global context. The model predicts masked terminal expressions under an x-prediction objective optimized by a weighted Huber reconstruction loss together with global and cell-wise Pearson losses. During validation and inference, masked genes are reconstructed by converting predicted terminal states into velocities and integrating them with Euler rollout from a deterministic masked initialization.

## 16. Minimal Symbol Table

- `x_raw`: 原始 count
- `x_1`: `log1p` 后的终点表达
- `x_0`: 初始噪声
- `t`: 连续时间
- `x_t = (1-t)x_0 + tx_1`
- `m`: mask indicator
- `\tilde{x}`: 非 mask 位置填真值、mask 位置填 `x_t` 的混合输入
- `\pi`: 基因到网格 slot 的固定硬路由
- `c_t`: GeneMamba 编码后的上下文
- `\hat{x}_1`: 终点表达预测
- `\hat{v}_t`: 由 `\hat{x}_1` 反推得到的速度

---

如果后续又改了训练入口、验证策略或 checkpoint 规则，这份文档也需要同步更新，因为它现在描述的是“当前代码实现”，不是“历史版本”或“设计目标版本”。
