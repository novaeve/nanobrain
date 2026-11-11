# V3 ReAct 代码演化策略（2026-11-05 修订，正式运行入口已启用）

服务器独立目录：`/mnt/aliyunsb/zim_rider/nanobrain/try/v3-react-code-evolution-20261103`。

独立环境固定官方 `xgboost-cpu==3.0.5`，XGBoost 对照明确使用 CPU histogram 算法，避免 GPU 构建在本环境 CPU 隔离检查中触发 CUDA。仅修改新环境；其他神经模型仍使用 GPU 0–4。运行时库版本也写入冻结契约。

本分支保留 V3 的任务、数据划分和标准指标，增加受约束的代码演化、baseline 前置阶段，以及演化完成后自动执行的数据规模实验。2026-11-03 曾只完成准备。2026-11-05 按用户新指令修订并启动完整流程：先完成基线，覆盖通过后自动演化 50 代，再冻结冠军并执行规模实验。实时状态以 `results/research-v3-react/status.json` 和 `runtime/active_run.json` 为准。 `preparation_checks/` 内的数值和模型文件都来自微型合成数据，不能用作科研成绩。

## 一、先建立固定 100k benchmark

先完成 baseline，再允许演化。数据沿用 V3：60k train / 20k validation / 20k test；原数据只读复用。

- 20 个完整方法/明确任务适配入口保持原生任务的训练配方与已有成绩；15 个有可部署表征编码器的入口补齐四任务。原论文没有做过的任务明确标为下游迁移适配，不声称原论文已有这些结果。
- 取消全部“拆出组件再单独跑四任务”的 benchmark 和规模曲线。组件库仍可供 LLM 组合和编程使用；N-BEATS-inspired 等纯组件不再单列方法成绩。历史组件作业仅保留审计记录，不进入报告或 LLM baseline 摘要。
- XGBoost、CellTypist 是专用分类器；Harmony、Scanorama、BBKNN 保留原生整合任务。这五个入口缺少独立可部署表征，其他任务标记不适用。
- Foundation baseline 保留加载真实预训练 backbone 的 train-only 线性 probe；另新增 Geneformer V1-10M、scGPT brain、scGPT human 的全主干微调入口，使用官方 cell classifier / 原 Transformer+CLS 代码，按 validation Macro-F1 选权重。scFiT/scNODE/TrajectoryNet 使用官方模型代码和 V3 发育任务适配配方，配方、预算、权重文件和训练方式写入结果。
- XGBoost、CellTypist、foundation probe 和发育 baseline 保存 train-only 模型，测试阶段重载冻结结果。Harmony/Scanorama/BBKNN、scVI/scANVI 则明确标记无标签 eval 表达参与的 transductive 配方，配置在测试前固定；不得声称它们是完全相同的 inductive 训练。
- 新增完整 scBrain 级联网络；GraphST、SpaGCN、STAGATE 的原网络及原自监督目标；scUniFlow 完整 router + GeneMamba + VectorFieldNet 的训练集预训练。空间域方法先在训练集几何上无标签预训练，然后冻结为 self-loop 中心表达编码器，再训练邻域组成头；验证/测试不读取目标邻居或坐标。它们明确标记为任务适配，不声称原论文空间域聚类就是本任务。scUniFlow 缺外部谱坐标时用原代码的确定性 Hilbert 路由；无匹配 CUDA wheel 时采用上游 SSD 论文的 PyTorch 算法与不融合的 Mamba-2 运算，保留 Apache-2.0 许可。UCE 不属于当前组件库必需来源。
- 所有必需 seed/任务和标准指标完整、有限，coverage gate 才通过。缺依赖、缺权重、失败或未跑的成绩保持 null/failed，不能用 0 冒充。

结果先写 `baseline-*.json`、`baseline-summary.json` 和 coverage receipt，之后进入演化。现在是 20×3 个原方法作业，加上 15×3 个缺失任务迁移作业，共 105 个作业；每个迁移作业可产生多个缺失任务成绩。原生与迁移结果按方法、种子合并，禁止重复覆盖原生任务。

### 固定的完整方法迁移协议

冻结完整原方法主干，以其原生训练/预训练权重提取中心细胞表征；不以拆出的算子、随机替代网络代替原模型。Geneformer/scGPT 的冻结权重与 annotation fine-tuned 权重是分开的入口，后者明确带有训练集分类监督。GraphST/SpaGCN/STAGATE 延续 train-only 空间预训练、所有下游 split 的 self-loop 编码。scFiT 使用零时间条件 CLS，scNODE 使用 latent mean，TrajectoryNet 使用固定 unit-time canonical CNF map。

- 注释：128 隐层 GELU MLP 分类头，交叉熵，validation Macro-F1 选择权重。
- 整合：直接评估冻结表征，不人为加分类器来伪造整合；报告各项标准 scIB 指标。
- niche：128 隐层 GELU MLP → 每半径 cell-type softmax，train-only 软标签交叉熵、四半径等权，validation 组成 MAE 选择权重。
- 发育：两层 128 隐层、时间条件的 conditional flow-matching 头；仅训练批次中用精确等质量 OT 配对未配对群体，固定 Heun 10 步从 source expression 推进，validation 精确 W₂ 选择权重。推理只用源细胞表征、源表达、起止时间，不看目标群体或邻居信息。
- 可迁移任务采用固定头结构/学习率/训练预算；初始每头 10000 steps、AdamW lr=1e-3、weight decay=1e-4，500 steps 验证一次。特征标准化仅在训练表征拟合。所有头落盘并封存后才打开测试集，测试不重新优化。
- scVI 旧 transductive runner 未导出可冻结的编码器，因此缺失任务另用官方 NB VAE 的 train-only、batch-agnostic inductive 适配（原层数/宽度/epoch 配方），单独保存主干。该差异逐任务标注；不冒充同一 transductive 权重的迁移。
- 三个配对种子必须完整。coverage matrix 区分 native 与 transfer；每个任务的 regime、主干来源、头结构、预算及验证选择标准保留在 receipt 中。

本次属于 generation zero 前的显式 benchmark 修订：旧完整方法 runner 的 AST、公共评估器和数据契约保持不变。`benchmark-revision.json` 记录前后冻结契约、文件差异、复用作业及保留的三个 scUniFlow 进程。旧控制器暂停接单，已取消组件作业；三个原方法进程由新控制器接管，完成后退休旧控制器，不从零重跑。

## 二、允许修改的范围与输出契约

仍然是主干＋任务头，各任务各自初始化和训练权重。可以改计算图、分支、残差、重复层、任务头、宽度、配对/流场、已支持的损失权重、学习率、优化器及调度器，也可以提交新的 PyTorch 模块、损失和优化器代码。

候选输出必须是 `schemas/research-candidate.schema.json` 对应的完整 JSON bundle。必需字段：

```
schema_version, parent_ids, target_tasks, hypothesis,
genome: {architecture, learning_rate},
optimizer: {name, params, weight_decay, grad_clip, schedule},
plugins, files
```

不得添加自报 score/metrics、数据路径、seed、评估器或标签修改字段。父 ID 必须来自本轮可用谱系。源码限 `modules/`、`losses/`、`optimizers/`，插件数和源码文件数均无数量上限；每文件 32KiB、总计 128KiB。所有插件及参数必须显式声明类型、默认值和范围，并实际接入被评估的任务。未声明参数不静默忽略。

- 学习率 `1e-6..1e-2`，weight decay `0..0.2`，gradient clip `0.1..10`。
- 标准 AdamW/Adam/SGD/RMSprop/Adagrad；constant/linear/cosine 调度、最多 2000 warmup steps。可另写 `Optimizer`，必须覆盖所有可训练参数且不重复。
- 自定义 factory 只收到维度与声明参数；forward 只收到张量。分类 loss 收到本任务已按可信 mask 筛选的预测和标签；niche loss 保持真实组成、四半径等权和可信 density 辅助项；发育自定义 regression loss 保持真实 population flow target。
- 新增代码不可以改变数据划分、采样 seed、测试流程、mask、邻域组成定义、W₂ 公共空间或标准指标。niche 始终隐藏目标邻居表达与坐标。
- 验收链：严格 JSON/schema → 来源/父系/参数检查 → AST Tensor API 检查 → 独立 Linux bwrap CPU 沙箱 → 四任务合成 shape/loss/gradient/update/reload 检查 → 去重 → 提交 → 正式多种子评估。
- 沙箱断网、清空凭据环境、只读源码/数据、仅工作目录可写，限制时间/内存/输出文件。正式候选最多 20M 参数。GPU 白名单仅 0–4，内部使用选定 GPU UUID 映射为 `cuda:0`。
- 内容 SHA 固定候选。AST/图 ID、注释、文档字符串、文件重命名和部分优化器默认值等简单等价变化会被去重；**这不是任意 Python 程序语义等价的证明**。合成验收也不能保证真实训练永远不失败。失败/缺 seed/指标无定义的候选写拒绝记录，不进入父代或冠军选择。

接口示例见 `examples/research/`；示例父 ID 必须替换为工具返回的真实可用父 ID。完整图协议仍见 `NANOBRAIN_GRAPH_API.md`。AST 检查与 OS 沙箱是互补措施，不把 AST 规则单独称为完整 Python 安全沙箱。

## 三、ReAct 与更宽松的演化

初始输入包含完整演化 DAG 的大纲：所有已生成节点及全部父子边，包括未入选、验证失败和训练失败分支，不做 top-k 或最近节点截断。节点仅保留唯一短 ID、代数、状态、四任务主指标、实际改动大纲、简短假设和自定义插件数。它提供类似树搜索的全局视野；搜索策略仍为质量/Pareto/结构探索混合归档，不宣称变成 UCT/MCTS。模型可按任意节点短 ID 调用工具逐页读取完整 bundle、源码、验收和全部种子结果；父池是推荐列表，所有完整评估且通过的历史节点也可被选作父代。baseline 初始摘要仅给标准指标的均值/标准差/种子数，逐类型和全部运行详情通过工具读取。详情工具：

`inspect_contract`、`list_components`、`baseline_detail`、`read_baseline_artifact`、`read_artifact`、`lineage`、`write_candidate`、`validate_candidate`、`submit_candidate`。

不提供任意 shell、网络、文件系统写入或指标修改工具。artifact 读取分页、限定种类和长度。每个 proposal 最多 24 次工具轮、4 次验收修复；没有通过 `submit_candidate` 的自然语言回答不算候选。调用记录和 token usage 单独留存，不要求公开模型隐藏推理过程。

父代池混合质量精英、Pareto 方案与结构不同的有效探索方案；不强制每个子代四任务同时超过父代。较差但有效、有结构新意的候选仍可作为探索父代。最终冠军仍用已有 V3 内部 utility 排序，正式报告列标准任务指标。

默认准备配置：50 代、每代 6 候选、每候选 3 个相同 seed、4 任务，即每代 72 次独立任务训练。模型 `gpt-6.1-sol`，reasoning `max`；每任务 10000 steps。GPU 0–4，每卡最多 4 个 worker，总上限 20；另按显存预留、其它实验实际占用和主机可用内存动态准入。CPU 原方法最多 5 个并发，不占 GPU。worker 分配不同 CPU 核，避免同卡全部进程挤在同两个核上。OOM 扩大预留后重试；所有作业与代级结果支持恢复，已提交提案也会复用。

继续沿用用户选定的测试驱动策略：**validation 选择最佳 checkpoint；test 结果反馈给 LLM 选择架构**。测试已经参与搜索，因此这些分数及后续规模曲线是探索性结果，不能称为新的未触碰盲测。

## 四、演化完成后的自动行动

完成第 50 代后先写 `frozen-champion.json`，冻结冠军代码、图、损失、优化器、baseline 配方、配置哈希、可信源码哈希和指标协议。然后自动执行 `run_scaling_action`：

1. 名义数据集规模：100k → 1M → 5M → 10M → 20M → 全量。固定原 20k validation 和 20k test 后，对应训练行数为 60k → 960k → 4.96M → 9.96M → 19.96M → 全量 training split（25,702,876 行）；最后一档总计 25,742,876 个纳入本实验的细胞，不将其它 holdout 并入训练。曲线横轴为总规模，CSV 同时保留准确训练量和各任务有效量。
2. train 集严格嵌套，包含原 60k；原 20k validation / 20k test 逐文件校验不变。不能把全量 validation/test 纳入训练。
3. 固定 256-gene panel、原 train normalization、label codebook、niche 词表及半径；不重采 holdout、不重拟合这些定义。CSR 分块写入 memmap，不重开 H5AD。
4. niche 使用 V3 已验证的 full-section cache 和固定几何支持；未验证区域保持 mask false，不回退 anatomy。随规模记录各任务有效训练行和 population 数量；niche 有效量可能提前饱和。
5. 冠军和完整方法 baseline 在每档按相同 seed 从头训练；不跨规模带入旧权重，外部 foundation 初始化不变。冠军与新增迁移头默认 10 个采样 epoch 的行任务预算；发育头按有效 population pairs 计算。原方法保留各自 epochs/迭代配方。不再跑独立组件对照。
6. 输出每个 seed 的长表 CSV，以及标准任务指标随规模变化的 PNG/PDF（均值±样本标准差）。正式曲线不使用自定义总分或 `scib_composite`。
7. 缺数据、缓存来源不匹配、方法达到内存/能力上限、任一必需结果不完整时写失败/阻断信息，不把不完整全量实验标记为完成。

评估指标保持 V3：annotation **Macro-F1**；integration **Label ASW、Isolated-label ASW、KMeans NMI、KMeans ARI、within-type Batch ASW、iLISI、Graph Connectivity**；spatial niche **真实邻域组成 MAE**；development **公共 256 基因空间的精确经验 W₂**。`scib_composite` 只作内部选权重/搜索 utility。

## 五、入口与当前状态

默认仅打印计划（不会调用 LLM 或开始训练）：

```bash
.venv/bin/python scripts/nanobrain_research.py
.venv/bin/python scripts/nanobrain_research.py --audit
```

2026-11-05 用户已授权启动，正式入口使用 `--execute`；后台运行记录和 PID 写入 `runtime/active_run.json`。先基线覆盖，再 50 代，最后自动规模行动；缺失成绩不能被补成 0 或伪装完成。

核验记录见 `PREPARATION_VERIFICATION.json`、`preparation_checks/availability.json` 与合成测试日志。原 V3 和另外两条旧演化的代码、配置、进程均不由本分支修改。复制来的旧 launch/verification 文档属于来源记录，不代表本目录启动了那些实验。


## 2026-11-05 基线配方与来源核对

完整清单由 `benchmark.coverage_catalog()` 写入冻结契约。`baseline-coverage-matrix.csv` 按完整方法×任务×种子列状态，明确 native/transfer，不适用项写原因；`baseline-standard-metrics.csv` 和 `BASELINE_BENCHMARK.md` 输出正式指标。所有原生及可迁移任务齐全后才开放 LLM 演化。

| 方法/新增入口 | 原配方与必要适配 |
|---|---|
| Geneformer fine-tuned | 官方 V1-10M 权重、median-normalized rank tokenization、BertForSequenceClassification；全主干微调 3 epochs，lr=5e-5；固定本实验基因 panel 和划分。 |
| scGPT brain/human fine-tuned | 原 Transformer 与 CLS decoder，预训练权重、51-bin 表达、全主干微调 10 epochs，lr=1e-4，batch=32；无完整原分层标签时仅训练最细的规范标签头。 |
| scBrain original | 完整三级级联、feature residual、reconstruction+classification；缺少两层真实 ontology 时上两层设 singleton，保留规范细胞分类。最多 500 epochs、patience=30，validation Macro-F1 选权重。 |
| GraphST composition | 原 Encoder_sparse、reconstruction 与双 contrastive BCE（10:1:1）；600 个采样 epoch 的训练集邻域子图预训练，保留稀疏算子；中心表达 self-loop 下游组成头。 |
| SpaGCN composition | 原 GCN+DEC、KMeans 初始化、Gaussian 空间边权；无 histology，使用训练集稀疏邻域子图作资源适配，最多 5000 次更新；中心表达组成头。 |
| STAGATE composition | 原 GAT autoencoder、tied decoder attention/weights、reconstruction；1000 个采样 epoch 的训练集子图预训练，再接中心表达组成头。官方 Tensor edge_index 路径不要求 torch_sparse；可选 SparseTensor 路径仍明确报缺依赖。 |
| scUniFlow pretrain/readout | 完整 8×512 主干、原 weighted Huber/Pearson 目标、20% 非零掩码、80 个采样 epoch、batch=25、cosine 5e-4→1e-6；保留冻结主干的下游 readout 模式，适配 annotation/integration/niche。 |
| scVI/scANVI | 原 scvi-tools 的 NB VAE 与半监督 SCANVI，400/20 epochs；panel pseudo-counts 与 transductive 无标签表达适配按现有 adapter 明确记录。 |
| 其余原方法 | XGBoost、CellTypist、Harmony、Scanorama、BBKNN、scFiT、scNODE、TrajectoryNet 和冻结 foundation probes 保留各自原代码接口；具体训练参数、拟合行数与适配说明随每个 job 保存。 |

冠军搜索仍为每任务每 seed 10000 steps；这不是所有原方法统一强制的训练步数。原方法按各自 epochs/迭代配方；新增迁移头采用固定共同配方，规模阶段随有效行数/群体数增加预算，不产生组件曲线。

原始来源：
- [Geneformer cell classifier](https://geneformer.readthedocs.io/en/latest/geneformer.classifier.html)
- [scGPT annotation fine-tuning](https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html)
- [GraphST](https://github.com/JinmiaoChenLab/GraphST)、[SpaGCN](https://github.com/jianhuupenn/SpaGCN)、[STAGATE PyG](https://github.com/RucDongLab/STAGATE_pyG)
- [Mamba 官方 SSD 论文算法](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py)
- scBrain 与 scUniFlow 使用本 checkout 的 `reference_methods/` 原源码、原训练目标和随附文稿；各适配更改在上述表格及结果 notes 中记录。

2026-11-06 按用户指令仅调整调度：本 ReAct 分支每卡 4 个 worker、合计上限 20。四条演化总上限依次为旧测试驱动 5、固定头 5、V3 指标版 20、本分支 20。源码、数据、训练配方不随 ReAct 并发配置改变；新的运行契约与前后配置归档在 runtime/concurrency-* 中，已完成作业和三个原方法预训练进程保留。
