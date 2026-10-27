# nanoBrain：可组合架构 API V2

更新：2026-09-20。此次完成模块化改造和接口验证。原 100 轮演化保持停止；没有启动新的 LLM 搜索。

## 架构如何配置

模型由 JSON 中的计算图构建。主干、速度场、各任务解码器的结构可以独立配置；训练器继续复用原数据加载、优化器、日志、checkpoint 和 evaluator。

| 内容 | V1 | V2 |
| --- | --- | --- |
| 主干顺序 | Python 固定 encoder → attention → pool → spatial → residual | JSON 显式声明节点和连接；可插入、删除、替换和重排兼容节点 |
| 分支 | 固定串行流程 | add / subtract / concat / stack 连接并行路径和跳连 |
| 内部层 | 主要选择整块 encoder / Transformer | scBrain SwiGLU、scFiT attention、FFN、归一化、激活、投影、池化独立组合 |
| 层数 | 部分层数绑定总配置 | 每节点 repeat 使用独立权重；各节点单独设置宽度、head 数、dropout、expansion |
| 发育速度场 | 选一种完整 flow adapter | 可复用原 flow 模块，也可用时间编码、条件融合、attention、FFN 构建速度场图 |
| 任务头 | 多任务共享头类型设置 | 各任务单独选头，可配置自己的 pre 子图，输出维度从数据契约推导 |
| 损失 | 逻辑分散在模型和 task trainer | 独立 objective 注册接口；可选回归损失、label smoothing 和损失项权重 |
| 无效参数 | 部分构造器忽略额外参数 | JSON 只接受构造器显式支持的参数；不支持的参数直接报错 |
| 演化接口 | 枚举固定位置和数值 | LLM 提交完整图配置；结构变更配额识别连接、层数和模块变化 |

编码主干接收 `features[B,G]`，输出 `state[B,H]`；速度场接收 `state[P,H]`、`context[P,H]`、`time[P,3]`，输出 `velocity[P,H]`。时间三列为插值时间、源年龄（年）和年龄间隔（年）。任务头的可选 `pre` 子图接收 `state[B,H]`，允许输出另一个隐藏宽度，再由选择的 readout 输出类别或表达。

## 可以直接修改的配置

- [decomposed.json](configs/nanobrain_graph_v2/decomposed.json)：显式 attention / SwiGLU / norm / residual；带 backcast、残差相减和 forecast 路径；发育端为独立时间编码与条件 Transformer 组合。
- [parallel.json](configs/nanobrain_graph_v2/parallel.json)：平行 attention 分支、concat 后投影、重复速度输出层，以及可配置的分类头子图。
- [migrated_incumbent.json](configs/nanobrain_graph_v2/migrated_incumbent.json)：旧冠军配置转换为图形式的结构参考。未搬运权重，也不把旧分数归为新实现的结果。
- [smoke.json](configs/nanobrain_graph_v2/smoke.json)：四任务、每任务 3 steps 的接口验收配置。
- [evolution.json](configs/nanobrain_graph_v2/evolution.json)：100 轮、12 候选的新图搜索配置，已准备，未运行。

一个节点示例：

```json
{
  "id": "swiglu",
  "kind": "operation",
  "name": "scbrain_ffn",
  "inputs": ["normalized"],
  "params": {"expansion": 2.67, "dropout": 0.05},
  "repeat": 2
}
```

列表按依赖顺序排列，`inputs` 引用前面节点的 `id`。`repeat: 2` 构建两份独立层。`add` / `subtract` 要求形状相同；`concat` 拼特征维度，`stack` 从多个细胞状态创建 token 维度；pool 只作用于 token 维度。

`hidden_dim` 只在允许改变宽度的投影或编码器上生效。`input_dim` 和最终 `output_dim` 自动推导。head 数必须整除该节点宽度。未知字段、错误 rank / width、环、悬空输入、无用节点和无效 repeat 均在构建前拒绝。

## baseline 的复用方式

继续调用 PyTorch、Transformers、scVI 库，以及项目保存的 scBrain、scFiT、scNODE、GraphST、TrajectoryNet、scUniFlow 参考实现。

新增细粒度适配直接取 scBrain `SwiGLU`、scFiT `AdaLNTransformerBlock.self_attn`、`.ffn`、`SequenceConditionEncoder`。SwiGLU 单元不隐含原 FFN 外部的残差与归一化，这些连接由配置描述。完整 baseline block 仍可作为一个节点选用。

这是组件组合接口，不代表所有 baseline 的完整算法、损失、预训练权重都能够任意互换。传统分类器、图构造和完整 VAE 等方法仍有各自的数据与算法契约。

## 训练目标 API

`objectives.py` 提供注册的任务目标和公共 masked-loss 函数。行任务接口：

```python
compute(model, batch, embedding, params, task) -> (loss, diagnostics)
```

群体发育目标接口：

```python
compute(model, population_batch, *, tau=None) -> (loss, diagnostics)
```

新增目标通过 `register_objective(name, tasks=..., compute=..., defaults=..., population=...)` 接入，不复制训练循环。配置只能选择已注册实现。当前内置分类交叉熵及 smoothing、整合表达重建＋batch 对抗、masked completion、conditional flow matching。回归项复用 PyTorch MSE / SmoothL1 / L1。

发育速度、终点、表达项的权重独立设置；整合的重建权重、batch 权重和梯度反转强度也独立设置。必要训练项保持正权重。评估指标和晋级门槛固定，训练 loss 的调整不会修改 evaluator。

四任务各自训练独立模型。整合、发育、补全各有独立表达解码头。非当前任务参数冻结。初始化按随机种子和模块路径隔离，改动 flow 不会改变注释模块的随机初始化。

## 演化和记录

新候选包含 `architecture` 和 `learning_rate`，随机种子由运行器分配。也支持单父候选的标准 RFC6902 JSON Patch，复用 `jsonpatch` 库执行插入、删除、替换、移动等修改；运行器展开后保存完整配置与原 patch，减少每次重写整个 JSON 的成本。patch 只能作用于架构和学习率。

LLM 得到图契约、真实组件参数、父配置和 validation 结果。候选先做形状、连接、参数和血缘检查，再进入原来的独立任务训练、多 seed 复验和晋级流程。

节点仅改名、显式补默认值不算新架构。历史候选 ID 和旧版 checkpoint 读取保持兼容。新日志记录完整图，receipt 记录版本、有效节点参数、展开层数、输出 rank / width 和节点参数量；各头与损失配置保存在 genome 中。

基因面板、60k/20k/20k 划分、标签访问边界、validation 选择和冻结后 test 复评规则沿用原协议。此次验收只读 train / validation，不调用 LLM，不打开 test。

## 保留的接口约束

- 有向无环图，每图最多 32 个节点、64 个展开层；配置不执行任意 Python。
- 当前 loader 提供一跳空间上下文。同一路径最多一次非 identity 空间聚合，并行一跳分支可用；连续多跳需先升级邻居数据契约。
- 缺少空间信息仍使用原来的精确 self-loop 退化。
- rank-2 编码器给 attention 一个 token；跨基因交互需选 patch 编码器或显式创建 token。空间节点要求先池化。
- 旧 FMT primitive 的偶数层要求保留，规范化配置写出实际层数；拆分的图层数由节点和 repeat 决定。
- 新图的初始化规则和可选目标有变化，正式比较必须建立新的同预算 reference；旧结果不能直接充当新协议对照。

## 检查和短训练

只校验配置：

```bash
.venv/bin/python scripts/check_nanobrain_architecture.py configs/nanobrain_graph_v2/decomposed.json
```

单任务短训练仍用原入口：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --genome-json configs/nanobrain_graph_v2/decomposed.json \
  --task annotation --steps 3 --batch-size 128 --precision bf16 \
  --device cuda:6 --output results/nanobrain/my-new-graph-smoke
```

四任务验收用 `run_nanobrain_evolution.py --task-suite-only` 和 `smoke.json`，只训练和评估给定配置，不生成候选。输出必须是新目录。

## 历史资产

旧目录 `results/nanobrain/llm-evolution-100g12-20260917T081327Z/` 保留全部候选、训练日志、指标、receipt、checkpoint、提案与源码快照。旧数据和划分也保留。

此次输出写在 `results/nanobrain/component-graph-v2-20260920/`。其中 `preserved_campaign_manifest.json` 记录改造开始时旧实验的 37,984 个文件：文本和元数据有 SHA-256，checkpoint 记录大小和修改时间，并沿用历史 receipt 中的哈希。

## 本次验收

最终图专项测试 28 项通过；回归集合 78 项通过（两组有重叠）。GPU 6 上完成 6 次 BF16 短训练，覆盖四个核心任务和可选补全，每次 3 steps、batch 128、CPU 2 线程；参数更新均通过，323 条 validation 指标全部产出。短训练证明接口和梯度可用，不代表收敛或性能提升。

旧实验 37,984 个文件校验无变更，旧冠军 checkpoint 严格重载和推理通过；正式演化和验收子进程均已停止。详细记录见 [验收报告](results/nanobrain/component-graph-v2-20260920/REPORT.md)。
