# nanoBrain 组件 API、数据划分与演化核对

2026-09-20：本文保留 V1 固定位置组件接口的历史说明。可配置计算图、细粒度单元和 objective API 的当前实现见 [V2 架构接口](NANOBRAIN_GRAPH_API.md)。旧演化已停止并保留。

更新：2026-09-17。使用固定 100k，不扩展全量数据。正式比较采用任务分别训练。

## 本轮实际完成

原有 baseline adapter 只解决整套方法的运行与评分。本轮新增 15 个可训练组件，注册表共 32 个组件；分类头与表达解码头也能由 JSON 选择。构建器检查编码器、注意力的维度契约，参数取值范围进入 LLM 提示词，未知配置字段直接报错。参考实现按原文件导入，实验冻结这些文件的 SHA-256 和依赖版本。

| 接口位置 | 可选择组件 | 上游实现与适配 |
| --- | --- | --- |
| encoder | linear、MLP、residual MLP、scBrain SwiGLU、scNODE MLP、scVI FC、panel patches | scBrain `FFN`；scNODE `LinearNet`；scVI `FCLayers`；PyTorch Linear |
| attention | identity、SDPA、differential linear、scFiT block、BERT、PyTorch Transformer | scFiT `AdaLNTransformerBlock` 无条件版本；Transformers `BertLayer`；PyTorch `TransformerEncoderLayer` |
| spatial | identity、neighbor mean、scBrain GCN、GraphST sparse | scBrain `GCNConv`；GraphST 官方编码权重＋PyTorch sparse 图传播；没有有效邻居时输出精确退化为输入 |
| residual | none、N-BEATS correction | 保留已有 backcast/forecast 校正 |
| pairing | identity、Sinkhorn | 已有 BrainWM OT 实现，配对目标停止梯度 |
| flow | none、MLP、FMT、scFiT AdaLN、scNODE time MLP、TrajectoryNet ODE、scUniFlow field | scFiT 时间条件编码器＋AdaLN block；scNODE `LinearNet`、TrajectoryNet `ODEnet`、scUniFlow `VectorFieldNet` 条件适配；正式发育任务禁用 none |
| classification head | linear、scBrain、scFiT | scBrain 层级解码器的单层配置；scFiT `TypeHead` |
| regression head | normalized linear、scFiT MLP | 输出宽度适配为当前基因 panel；最后一层不加 softmax |

统一顺序：表达 → encoder → attention → token 均值池化（有 token 时）→ spatial → residual → 当前任务头。发育任务另外调用 pairing、flow、表达解码器和 Heun/Euler 积分。

`panel_patches` 将固定的 256 基因 panel 分为 16 个有身份编码的 patch。这样 attention 能在多个 token 之间计算；其余 encoder 仍输出一个细胞向量。scFiT flow adapter 对每个细胞的 state/context 两个 token 计算条件注意力，保留每个细胞自己的插值时间、源年龄和目标年龄；原 BrainWM FMT 则在群体细胞之间计算注意力。两者在目录与提示词中明确区分。

这些是随机初始化后重新训练的组件组合。BERT block 不包含 Geneformer 预训练权重；scVI FC 不等于完整的 scVI VAE/NB 目标；单层 scBrain head 不宣称完成多级标签监督。scUniFlow 只延迟导入完整模型所需的 Mamba，让独立速度场可直接导入；数学实现保留在原源文件。其 field 是固定内部深度，flow_depth 不改变这个组件。原有整套 baseline 的结果与命名保持独立。

## baseline 接口覆盖的准确范围

| 方法族 | 整套方法评估 API | 本轮进入神经模型组合的核心 |
| --- | --- | --- |
| scBrain | 本地参考实现 | SwiGLU 编码、GCN、分类解码器 |
| scFiT | 已有发育 adapter | Transformer block、时间条件、条件 flow、输出头 |
| scNODE | 已有发育 adapter | LinearNet 编码和条件速度场 |
| scVI/scANVI | 已有整合／注释 adapter | FCLayers；完整概率目标仍由原方法 adapter 管理 |
| Geneformer、scGPT | 已有冻结 backbone／linear probe adapter | BERT / PyTorch Transformer 的可训练 block 家族；预训练 tokenizer/权重不作为任意拼接模块 |
| XGBoost、CellTypist | 已有 fit/predict adapter | 保持分类器算法接口，不作为反向传播层 |
| GraphST | 已有原方法 adapter | 稀疏图编码算子；完整对比损失留在原 baseline |
| Harmony、Scanorama、BBKNN | 已有原方法 adapter | 数据变换／图算法 API，不作为可反向传播层塞进神经主干 |
| TrajectoryNet | 已有发育 adapter | ODEnet 条件速度场；完整 CNF 密度目标留在原方法入口 |
| scUniFlow | 完整 baseline 的资产限制仍在 | 已接入 VectorFieldNet，以 latent velocity 训练；不声称使用 Mamba 或预训练模型 |
| UCE、SpaGCN、STAGATE | 未进入原冻结可运行方法集 | 不提供虚假的可选项；原资产／依赖限制仍需另外解决 |

因此，当前可选神经组件已经有实际组合、反向传播和序列化验证；不能表述成“原 HTML 所有 baseline 的所有内部层都已经任意互换”。新增组件仍沿用统一训练器，不复制训练循环。

## 数据边界与来源

| 数据 | 行数 | 用途 |
| --- | ---: | --- |
| train | 60,000 | 每个任务独立更新模型参数；预处理的 panel 选择和标准化只在 train 拟合 |
| validation | 20,000 | 计算各任务指标；反馈给 LLM；筛选、晋级、多 seed 确认与最终选择 |
| test | 20,000 | 选择冻结后的描述性复评，不输入提案或晋级过程 |

6:2:2 来自用户之前对 BrainWM 的全局要求。原 nanoBrain 提示词要求从既有完整划分按比例抽取约 100k；蓝图要求固定划分和 held-out 评估，但没有另行规定 6:2:2。当前数据没有重新随机切分。

本轮审计通过所有 compact 文件哈希、panel/标准化哈希、三份行数、split code、各 split 内 row reference 唯一性和范围；原全局 `samples.parquet` 中每个 sample 与 split_group 只属于一个 split。训练器启动也会验证 train/validation 文件与预处理哈希。

原 ESSD 上的 V3 目录已升级为 V4；原 V3 NAS 副本仍在，定位器已修正。已用原始 `source_row_ref` 查回全部 100,000 行的 `evaluation_sample`，与冻结 sample 表逐行关联并验证 split/source。恢复后 train / validation / test 的 sample 数分别为 1,613 / 701 / 628，split_group 数为 486 / 228 / 275；三份数据的真实 sample、split_group 和已知 donor（带 dataset 命名空间）交集均为空。缺失 donor 保持缺失。

修复数据独立保存在 `assets/nanobrain_v3_100k_metadata_v2/`。同一批细胞、同一划分、同一表达与标签；只有身份和邻居 sidecar 更新。原数据所有文件哈希已复核不变。大文件以硬链接复用，替换 sidecar 时使用原子重命名，避免覆盖原文件。

空间边界按已有独立核验的 section 证据构建，不把所有 sample 名称自动当作真实切片。train 的 14,872 行、validation 的 1,612 行有已验证几何，分别保存 118,976 / 12,896 条边。新 loader 会带上当前 split 内完整的一跳邻居表达，空间模块保持可反向传播；只在目标行上计算损失，不附带邻居标签。有效边不再因随机 batch 而丢失。

仍需准确保留的数据范围：

- test 的切片几何尚无同等级独立证据，因此 test 空间邻居仍退化为 self-loop；不能声称完成 test 的完整空间图验证。
- niche 标签仍是 anatomy 代理标签，不是独立 niche ground truth。
- test 已在历史实验中公布，当前只允许选择冻结后的描述性复评。

## 修正补全任务的信息泄漏

原 compact 表达文件与补全 target 保存的是相同的完整表达；此前的模型输入未统一遮住 held-out 值。现在独立补全任务在训练和推理入口都用同一 mask 清除这些输入，并更新 observed mask，目标仍独立保留。测试证明：任意改写 held-out 输入值，不会改变模型的补全表示。

历史 joint 模型的 completion 数值不再作为合规的 masked-completion 对照，需要在修正协议下重训。其他四项任务的历史记录不因此改写。历史文件原样保留，不用新协议覆盖旧结果。

## 用什么演化

使用 `.env` 中配置的第三方 LLM API。LLM 收到父配置、组件目录和参数范围、validation 指标、运行耗时与过去的失败反馈；输出模块选择、超参数、主攻任务和修改理由。训练器实际运行候选，分数不是 LLM 自报。

一轮流程：父配置 → LLM 给出若干候选 → 验证配置 → 每候选四任务分别从头训练 → validation 评分 → 合格者多 seed 复验 → 保存过关者 → 最佳配置进入下一轮。权重不跨任务共享，下一轮也不继承训练后的权重。可选补全为第五份独立模型。

现在默认 100 轮、每轮 12 候选，初筛每任务 600 steps，确认 2,000 steps；每批至少 4 个候选更换组件家族。综合分与各任务退化阈值仍按固定配置执行。轮数、预算、综合分公式是工程配置，不是蓝图规定的数字。没有过关者时保留 incumbent 并继续，来自用户后续澄清；失败方案不会传播。

## 验证与入口

- 自动测试：75 项通过；含各新增组件的实际梯度更新、组合维度、checkpoint 重载、时间条件、补全遮罩和多轮晋级逻辑。
- 数据审计与机器可读组件目录：`results/nanobrain/component-api-audit-20260917-metadata2/`。
- 五个任务真实 GPU smoke：`results/nanobrain/component-api-preflight-20260917-metadata2/`，每任务 2 steps，BF16，GPU 3，每进程 CPU 2 线程。五项任务均实际更新参数，非当前任务参数保持不变；178 条 validation 指标无错误。最终状态见该目录的 campaign 与 verification 记录。另用 GraphST / scUniFlow / scBrain 的组合运行五任务验证，记录在 `results/nanobrain/upstream-api-preflight-20260917-metadata2/`。
- 上述是接口和训练可用性验证，尚未跑出新的多代独立训练冠军。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
.venv/bin/python scripts/run_nanobrain_evolution.py \
  --config configs/nanobrain_components_smoke.json --task-suite-only --gpus 3 \
  --output results/nanobrain/my-component-smoke
```

正式演化用 `configs/nanobrain_evolution_independent_100k.json`，已固定到 metadata_v2 数据。新空间图、补全输入与独立任务协议均改变了比较条件，reference 和候选必须在这一版本下重新训练，不直接沿用历史 joint 分数晋级。

2026-09-17 正式演化已启动，当前运行入口见 `runtime/nanobrain/active_evolution.json`；其冻结配置和实际运行状态以输出目录中的 `protocol.json` / `campaign.json` 为准。
