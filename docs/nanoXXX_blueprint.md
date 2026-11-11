# nanoXXX：面向其他领域的极简研究实验场设计蓝图

## 1. 核心定位

`nanoXXX` 是一个 NanoGPT 风格的最小研究实验场。它不是生产系统的缩小版，而是把一个领域中最重要的研究闭环压缩到一个小型、可运行、可修改、可复现的代码库中：

```text
准备最小数据
→ 构建模型或算法
→ 运行固定训练/搜索预算
→ 评估结果
→ 保存实验记录
→ 修改模块或配置
→ 继续比较
```

它的价值在于降低实验成本和修改成本，让研究者可以快速回答：

- 一个新模块是否值得继续做？
- 一个结构改动是否真的改善了目标指标？
- 一个数据处理方法是否有效？
- 一个超参数或任务组合是否影响结果？
- 一个自动搜索策略能否找到更好的候选？

验证有效后，再把方案迁移到完整系统。

## 2. 设计原则

### 小而完整

代码可以只有几百行，但必须覆盖数据、核心算法、训练或搜索、评估和记录。不要只做一个孤立的模型 demo。

### 固定协议，开放实现

输入格式、输出格式、数据切分和评估指标尽量固定；模型模块、算法步骤、参数和搜索策略可以替换。

### 配置与代码分离

实验结构写在 JSON/YAML 中，程序负责解析和构建。研究者可以先改配置，只有新增能力时才改 Python 代码。

### 每次实验都留下 receipt

每个实验至少记录：配置、随机种子、版本、数据摘要、训练预算、指标、运行时间和结果文件位置。结果必须能够被重新检查。

### 先短预算筛选，再扩大验证

第一阶段用很短的训练或运行预算筛选候选；第二阶段用多个随机种子、固定 held-out 数据和相同预算重新验证。

### 失败时停止传播

自动搜索或自动演化不能因为一次偶然提升就持续迭代。候选必须满足明确的目标指标、全局回归限制和复现条件，否则保留上一代并停止。

## 3. 推荐的分层架构

```text
                ┌──────────────────────────┐
                │      Experiment CLI      │
                └────────────┬─────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   Data Contract        Module Graph         Search Loop
        │                    │                    │
   Data Loader        Registry + Builder    Candidate Mutator
   Splitter            JSON Specification    Budget Controller
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                  Trainer / Executor
                             │
                    Evaluator + Receipt
```

建议拆成以下组件：

```text
nanoXXX/
├── main.py              # 最小 CLI 和端到端入口
├── data.py              # 数据集、切分和任务定义
├── modules.py           # 模块注册表和基础模块
├── architecture.py      # JSON 架构解析与构建
├── train.py             # 固定预算训练或执行循环
├── evaluate.py          # 指标、聚合和比较
├── search.py             # AutoResearch / AutoML / 演化搜索
├── schemas.py            # 配置和 receipt 的结构定义
├── architecture.json     # 默认架构
├── configs/              # 可复用实验配置
├── experiments/          # 机器可读的实验 receipt
├── reports/              # 人类可读的实验报告
└── tests/                # 接口和关键行为测试
```

如果希望保持 NanoGPT 的极简感，可以先把 `data.py`、`modules.py`、`train.py` 和 `evaluate.py` 合并进一个 `nano_xxx.py`；当模块数量增加后再拆分文件。

## 4. 统一模块 API

所有模块都应有明确的输入输出接口。接口首先约束张量或对象的关键维度，其他细节可以由模块自己处理。

```python
class Module:
    def forward(self, x, *, mask=None, metadata=None):
        """Return an object compatible with the next module."""
        raise NotImplementedError
```

每个模块最好声明：

```python
MODULE_SPEC = {
    "name": "example_block",
    "input_contract": {"rank": 3, "axes": ["batch", "item", "hidden"]},
    "output_contract": {"rank": 3, "axes": ["batch", "item", "hidden"]},
    "parameters": {"width": 64, "dropout": 0.0},
}
```

模块需要满足：

1. 输入输出类型明确；
2. 关键维度可检查；
3. 参数有默认值和合法范围；
4. 不依赖隐藏的全局状态；
5. 可以单独实例化和测试；
6. 能够被 JSON 配置引用。

## 5. Registry 和 JSON 架构

Registry 将配置中的字符串名称映射到实际实现：

```python
class ModuleRegistry:
    def __init__(self):
        self._items = {}

    def register(self, name, factory):
        self._items[name] = factory

    def build(self, name, **kwargs):
        if name not in self._items:
            raise KeyError(f"Unknown module: {name}")
        return self._items[name](**kwargs)
```

配置文件只描述模块顺序和参数：

```json
{
  "version": 1,
  "input": {"kind": "default", "width": 64},
  "modules": [
    {"name": "normalize", "params": {}},
    {"name": "encoder", "params": {"depth": 2}},
    {"name": "head", "params": {"output_size": 10}}
  ],
  "output": {"kind": "prediction"}
}
```

固定程序执行：

```python
def build_from_json(path, registry):
    spec = load_json(path)
    validate_spec(spec, registry)
    modules = [
        registry.build(item["name"], **item.get("params", {}))
        for item in spec["modules"]
    ]
    return Sequential(modules)
```

这样新增模块时，只需实现模块、注册名称、补充测试，不必重写训练主循环。

## 6. 最小训练/执行程序

```python
def run_experiment(config):
    set_seed(config["seed"])
    data = build_data(config["data"])
    model = build_from_json(config["architecture"], REGISTRY)
    optimizer = build_optimizer(model, config["optimizer"])

    for step in range(config["budget"]["steps"]):
        batch = data.train_batch()
        output = model(batch.x)
        loss = objective(output, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = evaluate(model, data.fixed_validation())
    receipt = make_receipt(config, metrics)
    save_json(receipt, config["output"])
    return receipt
```

核心训练循环应该保持稳定。新想法通过模块、数据构造器、loss 或配置注入，而不是复制一套新的训练脚本。

## 7. 数据设计

第一版建议使用一个很小、确定性强、可快速生成的数据集：

- 内置生成器，不依赖下载；
- 固定 train/validation/test 切分；
- 支持随机种子；
- 能生成至少两类有代表性的任务；
- 能控制难度、噪声、长度、类别或结构特征；
- 保留一份固定 held-out 集，搜索过程中不能修改。

数据接口可以统一为：

```python
class Dataset:
    def train_batch(self): ...
    def validation(self): ...
    def test(self): ...
    def describe(self): ...
```

后续接入真实数据时，只新增 source adapter，不改变模型和评估接口。

## 8. 评估与结果记录

评估器应同时输出：

- 主指标；
- 分任务或分子集指标；
- 参数量、计算量或运行时间；
- 失败状态和异常信息；
- 与 baseline 的差值。

推荐的 receipt 结构：

```json
{
  "run_id": "2026-09-14-example-001",
  "code_version": "git-sha",
  "seed": 7,
  "data": {"name": "tiny-v1", "train_size": 1024},
  "architecture": {"path": "architecture.json"},
  "budget": {"steps": 100, "samples": 4096},
  "metrics": {"primary": 0.42, "secondary": 0.18},
  "runtime_s": 12.4,
  "status": "complete"
}
```

机器读取 `experiments/*.json` 或 `*.jsonl`，人类阅读 `reports/*.md`。报告应说明实验问题、对照、结果、限制和下一步。

## 9. AutoResearch 和演化接口

AutoResearch 的抽象不是“让程序随便改代码”，而是定义一个受约束的候选空间：

```python
candidate = {
    "architecture": architecture_json,
    "hyperparameters": hyperparameter_dict,
    "data_policy": data_policy_dict,
}
```

候选生成可以分为：

- random：随机采样；
- mutation：对当前候选做小幅修改；
- crossover：组合两个已有候选；
- informed：根据失败指标或结构特征定向生成。

统一搜索循环：

```python
population = initialize_candidates()
for generation in range(max_generations):
    receipts = [run_with_fixed_budget(c) for c in population]
    survivors = select(receipts, constraints)
    if not survivors:
        break                 # fail-closed
    population = mutate_and_sample(survivors)
```

候选晋级至少需要：

1. 主目标指标改善；
2. 不超过全局回归限制；
3. 在固定 held-out 集上评估；
4. 在多个 seed 上重复；
5. 记录父候选和变更内容。

## 10. 如何迁移到其他领域

把 NanoSimons 改造成另一个 `nanoXXX` 时，优先替换以下部分：

| 时序版本 | 其他领域对应物 |
| --- | --- |
| time-series sample | 领域中的一个样本、任务或 episode |
| forecasting target | 分类、生成、控制、检索或决策目标 |
| temporal block | 领域核心处理模块 |
| task mixture | 多任务或多数据源比例 |
| MASE/WQL 等指标 | 领域主指标和约束指标 |
| synthetic generator | 可控的最小环境或数据生成器 |
| DGP mutation | 数据、任务或环境的定向修改 |

迁移时保持以下部分不变：

- 固定预算的实验运行器；
- Registry 和配置驱动的模块构建；
- held-out 评估协议；
- receipt 和报告机制；
- baseline、对照和 fail-closed 规则。

## 11. 推荐的开发顺序

1. 写一个能端到端运行的单文件 baseline；
2. 固定最小数据和评估指标；
3. 把核心步骤拆成统一模块接口；
4. 增加 Registry 和 JSON 配置；
5. 加入 receipt、日志和最小测试；
6. 做一个小规模参数或结构 sweep；
7. 加入 AutoResearch 候选生成和多 seed 重测；
8. 扩充真实数据或更复杂任务；
9. 只有在证据充分后，才迁移到完整生产系统。

## 12. 最小验收标准

一个可交付的 `nanoXXX` 至少应满足：

- 一条命令可以运行 smoke experiment；
- 不依赖私有 token、API key 或隐藏服务；
- 新模块可以通过 Registry 注册；
- JSON 配置可以构建并校验架构；
- 固定 seed 能得到可比较的结果；
- 失败实验不会覆盖成功实验；
- receipt 能说明代码、数据、配置、预算和指标；
- README 能让新用户在几分钟内完成第一次运行；
- 测试覆盖数据接口、模块接口、配置解析和评估逻辑。

## 13. 一句话定义

`nanoXXX` 是一个用固定协议保证可比较、用模块 API 保证可替换、用 JSON 保证可组合、用短预算搜索保证快速迭代、用 receipt 保证可复现的领域研究实验场。
