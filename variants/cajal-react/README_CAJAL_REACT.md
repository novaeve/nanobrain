# Cajal v1 + ReAct：合并组件库，单 seed，开放训练参数

2026-11-10 独立训练修订。30 代、每代 10 候选、最多 16 个并行阶段，使用 GPU 0–7；实际进度以 research-runs/cajal-react-self-contained-20261110/status.json 为准。
配置见 `configs/cajal_react.json`，完整范围由 `cajal_research/contracts.py` 和候选 schema 定义。

## 种子架构与数据

起点仍为现成的 `configs/experiments/cajal_v1.toml`：基因 token 编码器、循环 MoE、N-BEATS
残差、层级分类头、NB 解码器、邻域模型，以及 grad_equal 和分组 Muon + AdamW。
初始候选的 model/train/data 改动、components、plugins、files 全为空；新组件均为可选。
代码锚点是 `0ae20748c3b684b7aabecc07e29cbd1ed6767267`；没有把上游候选实验误作正式种子。

每候选只有一份共享模型，联合优化 gene / integration / cell_type / anatomy / spatial 五个目标。
同一 checkpoint 给出全部任务成绩。初始参考直接复用 Cajal 已有的 cajal_v1@10000 checkpoint 和成绩；新候选从现成架构随机初始化训练，不继承旧演化结果。

| 用途 | Cajal 当前数据 | 细胞数 |
|---|---|---:|
| 联合预训练 | `/data/brainwm/unified_1m` | 1,015,388 |
| 注释、脑区、基因插补 | `/data/brainwm/eval_10k` | 9,960 |
| 整合 | `/data/brainwm/eval_integration_10k` | 9,867 |
| 空间插补 | `/data/brainwm/eval_spatial` | 40,789 |

不采用旧 nanoBrain 的 100k / 256-gene 数据。保留 Cajal 的 65,918 基因词表、测量 panel、counts、
标签层级、行身份和全部预训练数据。候选不可修改 data.root/seed/cell_fraction/neighbors/masks；
cell_fraction 固定 1。Cajal 本身从 V4 train+validation 分片构造预训练和评估数据，unified_1m 不再内部切 train/val/test；评估细胞与预训练细胞不重叠。封存 V4 test 不挂载。数据指纹见 `reference/input_fingerprints.json`，
初始架构及训练文档见 `reference/initial_cajal_v1.json`。

## 单 seed 与预算

按本次用户要求移除三 seed：使用 Cajal 原本的 **seed=0**，评估抽样 seed 也固定 0。
30 代、每代 10 候选，每代 10 次联合训练，初始参考直接复用已有结果，最多 300 次新训练。没有末尾三 seed 确认。
结果中 seed_count=1、标准差为 null，不声称统计显著性。

比较预算仍为 30,000 步学习率日程，在 10,000 步停止，评估固定结束 checkpoint。
较大 batch 改变样本曝光量和成本，回执与结果表保留实际参数。现有 Cajal 评估集反复参与搜索，
是探索性反馈，不作为新的盲测。

## 放宽的训练参数

Cajal 的 batch 由表达细胞数量和空间组数量构成，每空间组含中心和 16 个邻居。
这些范围只改变训练采样/优化，不改变数据集或评估定义。

| 参数 | Cajal 默认 | 搜索范围 |
|---|---:|---|
| 表达细胞 `dissociated_cells` | 128 | 16–512 |
| 空间组 `spatial_groups` | 16 | 2–64 |
| 每细胞非零 token 上限 | 4096 | 512–8192 |
| 每细胞测量零值采样 | 64 | 8–256 |
| 学习率 | 1e-3 | 1e-6–2e-2 |
| Weight decay | 0.01 | 0–0.3 |
| 梯度裁剪 | 1 | 0.1–20 |
| Warmup | 1000 | 0–4000 |
| 学习率日程 | cosine | cosine / linear / constant |
| 最小学习率比例 | 0.1 | 0.01–0.5 |
| KL warmup | 5000 | 100–10000 |
| KL 权重 / capacity | 1 / 0 | 0.001–20 / 0–300 |
| 各任务损失权重 | 1 | 0.01–20，五任务保持开启 |
| 归一化初始步数 | 200 | 20–1000 |

支持 initial / grad_equal / FAMO 及其实际生效参数。优化器支持 grouped Muon + AdamW、AdamW、
Adam、SGD、自定义 Optimizer，开放 Adam betas/epsilon、Muon momentum/NS steps、SGD momentum。
字段必须适用于选中算法，不静默忽略。额外目标支持 HSIC、SIGReg、batch swap、batch adversary，
以及 Cajal 已有的 start / budget / dual_lr 机制。

候选首次训练必须随机初始化。父代只传递架构、源码和超参数，不传递权重、预测或教师表征。
禁止历史模型或额外预训练模型参与初始化、蒸馏、拼接；组件复用实现代码。只允许同一候选、同一
seed、同一协议和训练配置恢复自己的断点，训练入口核对 training-identity.json 后才允许 --resume。
自监督目标、损失、优化器和训练日程仍可搜索，但必须在既定预算内、仅凭声明的训练数据独立复现。

## 合并后的组件库

`component_library/legacy/` 冻结原 Graph V3 组件库、上游源码和许可。其 package 入口只加载神经
组件，旧数据读取器、训练器、100k assets 和历史结果没有接入。原方法代码与 Cajal 之间仅做明确的
张量/遮罩/维度适配，组件混合不冒充原论文完整方法。

完整目录是 `component_library/catalog.json`：共 111 条登记，52 个结构组件通过全部适用插槽的 CPU 合成五任务检查（67 个插槽接口），另保留整方法基线和配置型方法。每条含来源、哈希、参数、接口、可用性和核验记录。
可组合插槽：

- value_embedding：Cajal 数值 MLP、scFoundation AutoBin、scPRINT 表达嵌入，仍叠加原基因身份。
- token：原库 MLP、residual、attention、FFN 等，逐细胞处理，不能跨细胞混合。
- block_attention / block_mixer：原 attention/MoE，以及适配的 attention、SwiGLU、FFN。
- latent：原 μ/logvar 线性层或 scVI 官方 VAE Encoder。
- classifier：原层级头、scANVI 官方分类器、旧库分类头/原型头，保留全部标签层级。
- decoder：原 NB/unified decoder 或 scVI 官方 decoder，仍用 Cajal measured-panel NB 目标。
- neighborhood：Cajal Transformer、GraphST、STAGATE、SEDR、SpaGCN、原库图聚合。
  目标 latent 先被遮掉，图只在原 Cajal 空间组内构建，不读取目标表达。

另把 Cajal MoE、循环编码、N-BEATS、层级头、NB likelihood、MoG prior、梯度均衡/FAMO、KL
capacity、预算权重、对抗目标、分组 Muon 等登记到统一目录和配置接口。
教师 Gram 源码仅作为历史实现保留，组件目录标为不可用，不进入搜索。
当前五任务不用的发育 flow/endpoint 等仍保留源码与不适用原因。缺依赖或未通过检查的条目不能进入
搜索，没有同名替代实现。新增依赖安装在项目自己的 `.component_deps/`，不修改共享 `.venv`；
版本与 Torch/CUDA ABI 记录在 `component_library/dependencies.json`。

## 基线与已有性能

使用 Cajal 已完成的整方法和成绩：LogReg、XGBoost、scVI、scANVI、PCA、同 panel kNN、gene mean、
两种 neighbour mean。也保留 Pearson PCA、Harmony、transductive 参考，按原 regime 标注。
整方法接口复用原 Cajal scripts；树/kNN 作为整方法基线，不当作可微 Transformer 层。

`reference/opponents.json` 保存全部 13 个完整基线及选优对手；`reference/current_method_scores.json`
保存现有 63 个方法/实验的 checkpoint 轨迹，baseline_detail 工具可按方法查看。
`reference/cajal_existing_five_tasks.csv` 是原表快照。没有导入旧 100k benchmark 成绩；新组件组合
须另行训练评估，不能继承来源整方法的分数。 发给 LLM 的历史性能和比较门槛只使用 Cajal 现有结果；旧 nanoBrain 在不同数据、划分或指标下的实验成绩不进入该输入，原组件实现继续复用。演化 worker 使用 predictions-only 评分入口，不重训或重算原基线；整合 PCR 指标继续使用原缓存 PCA 作为度量参考，不拟合新 PCA。完整数据与评估口径见 `reference/evaluation_protocol.json`。

## ReAct、验收与恢复

保留完整 DAG、全部父系、失败/未选分支、工具式提交、质量/Pareto/结构探索父池及受限源码扩展。
候选相对冻结 Cajal v1 声明完整 model/train/data/components；不是相对父代的局部 patch。
工具：inspect_contract、list_components（分页）、baseline_detail、read_artifact、lineage、
write_candidate、validate_candidate、submit_candidate。

Schema、参数和源码检查后，bwrap CPU 沙箱做合成五目标梯度、更新和重载验证，无真实数据、GPU、
网络或凭据。正式 worker 只读指定输入。模型最多 200M 参数，源码每文件 32KiB、合计 128KiB，
不限制插件/文件数量。初始候选不启用替换，参数/前向与原 Cajal 一致。

单 seed 晋级使用预先声明的 0.005 工程改善/退化门槛和 joint-frontier 排序，不作显著性推断。
六项指标必须完整且有限；资源不足单独记录，不用零补成绩。最多 16 个并行阶段，GPU 0–7 每卡最多 3 个 worker；每代仍为 10 个候选，实际还受候选数量与显存限制。训练任务预留 32GiB，评估任务预留 8GiB，另留每卡 4GiB 余量；预留与实际显存占用共同决定准入，资源不足自动排队。同代父代池冻结，晋级决定按候选序号应用；恢复复用已完成的提案和结果。恢复核对代码、配置、输入、候选身份和结果，不重复已完成工作。

## 查看与验证

```bash
.venv/bin/python -B scripts/cajal_react.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests --basetemp=preparation/pytest-tmp
```

默认只打印计划，不读取 API 凭据、不创建 campaign、不调用 LLM 或真实训练。核验记录在
`preparation/verification.json` 与 `component_library/verification.json`。明确启动/恢复时执行
`.venv/bin/python -B scripts/cajal_react.py --execute`。结果在 research-runs/cajal-react-self-contained-20261110/，
完成后不自动扩量、不更新上游 Cajal 配方或网站。

2026-11-10：按用户要求只保留本演化，将调度上限从 10 提至 12。通过 scheduler-handoff.json 接管正在训练的沙箱，完成后退休旧控制器；已完成候选和提案复用。训练数据、30 代、每代 10 候选、seed=0、1 万步预算、指标与晋级规则保持不变。

2026-11-10 独立训练修订：关闭历史教师依赖的提示、schema、组件目录和训练挂载入口。旧 research-runs/cajal-react/ 保留为教师辅助协议的历史档案，不作为新协议父池或候选性能。前15代150个启动失败和第16代蒸馏分支不计入新协议有效代数；独立协议从第1代计数，仍为30代、每代10候选、seed=0、固定1万步、12并发、GPU 0–4。无教师的旧候选仅迁移结构和参数并重新通过CPU校验，不迁移权重。最终 frozen-champion.json 同时导出完整候选源码、解析后的模型/训练配置和空的外部权重依赖清单。

2026-11-10 全卡调度修订：用户解除显卡限制，允许 GPU 0–7。总阶段上限由12调至16，CPU合成验收由2调至4，每候选最多2条独立的任务预测/评分流水线并行。全局阶段信号量包含已接管的运行进程，并由每卡3任务上限、空闲显存和主存共同限制。每代仍10候选，最多10个候选同时推进；各任务复用同一训练checkpoint，按固定任务顺序合并原始指标，不改变训练、评分或晋级规则。旧控制器停在交接点，已运行的训练/预测/评分进程保留，结果和断点不重跑。
