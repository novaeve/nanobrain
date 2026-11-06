# Cajal v1：共享模型五目标联合训练的 ReAct 演化

状态：代码准备与 CPU 合成验证；**尚未启动演化、LLM 调用或真实数据训练**。

本目录独立于原 Cajal 和正在运行的四条 nanoBrain 演化。Cajal 源码冻结于
`0ae20748c3b684b7aabecc07e29cbd1ed6767267`；ReAct 的工具循环、完整谱系、
候选验收、质量/Pareto/新颖性混合父池及断点恢复来自
`../v3-react-code-evolution-20261103`。来源与哈希见 `reference/provenance.json`。

## 初始参考与训练

- 初始候选直接解析现成的 `configs/experiments/cajal_v1.toml` 及其继承链；
  初始 `model/train` 改动项、插件和源码都是空的，绝不使用 nanoBrain 的旧图作起点。
- 保留基因 token、循环 MoE、分组 Muon + AdamW、`grad_equal`，以及五个共同优化的目标：
  gene / integration / cell_type / anatomy / spatial。
- **每候选每 seed 只有一份共享模型、一份联合 checkpoint**。该 checkpoint 同时评估全部任务，
  不拼接任务各自的最优模型。
- 沿用 Cajal 方法实验协议：30,000 步余弦日程，在 10,000 步停止，用固定结束权重评估；
  没有移植 nanoBrain 的 validation-best + test 驱动权重选择。
- 从随机初始化按现成架构/配方重训初始参考与候选；不续训已有 30k 权重。三个配对 seed
  为 202609180/181/182。50 代 × 6 候选 × 3 seed，最多 903 次联合训练（含初始参考 3 次）。
- 五类任务报告六列标准结果（整合分 bio/batch），与现成 Cajal 数学评估脚本一致。
  固定 eval seed=0、三个现有评估集；评估集反复参与研究反馈，不能当作新的盲测。
  封存 V4 test 从不挂载给 worker。

## 演化范围

允许的 Cajal 模型字段与范围由 `cajal_research/contracts.py` 唯一定义：层数、循环次数、
MoE、路由、宽度、潜变量、解码器、邻域模块；以及学习率、正则、loss 平衡与五任务权重等。
数据/标签词表、原始 counts、测量 panel、遮罩定义、采样种子、训练步数、评估器固定。
所有五个任务必须保持正权重。

ReAct 也可提交 `modules/`、`losses/`、`optimizers/` 下的受限 PyTorch 源码。
每文件 32KiB、总计 128KiB、模型最多 200M 参数；没有插件或文件数量上限。
参数必须声明类型、默认值和界限。完整候选 schema 见 `schemas/cajal-candidate.schema.json`，
配置示例见 `examples/research/candidate.json`；初始架构与实际训练文档见 `reference/initial_cajal_v1.json`。
当前源码扩展接口为：

- `token_transform`：在原编码器输出后添加可学习模块；一次仅收到一个细胞的 `[1,L,D]`
  基因 token，不接触跨细胞信息或标签；训练与全部推理路径均使用同一实现。
- `gene_loss`：替换遮罩基因 MSE；`cell_type_loss` / `anatomy_loss`：在所有层级上应用自定义分类损失。
  整合 NB/KL 与空间目标继续使用可信 Cajal 实现，不能改目标定义。
- `optimizer`：覆盖全部可训练参数一次的自定义 Optimizer；也可选择现成 grouped Muon、AdamW、Adam、SGD。

这些接口支持扩展现成 Cajal，不是 nanoBrain Graph V3 的任意图接口。更换整条原始 encoder
源码或任意训练器代码不在候选权限内。初始候选不启用任何扩展，参数和前向数值与原 Cajal 一致。

## ReAct、结果与恢复

工具：inspect_contract / list_components / baseline_detail / read_artifact / lineage /
write_candidate / validate_candidate / submit_candidate。完整 DAG 保留失败与未入选分支；
artifact 详情分页读取。自然语言回答不算候选，验收成功且工具提交后才进入真实训练。
候选相对**冻结 Cajal v1**声明完整改动；继承父候选的变更必须显式保留。

验收链：严格 schema 与参数 → AST Tensor API → 无网络/凭据/真实数据/GPU 的 bwrap CPU 合成
五目标梯度、优化更新及重载 → 去重 → 正式多 seed 训练。正式训练、推理也在 bwrap 中，
只读指定输入，仅本作业输出可写。AST 不是完整安全沙箱；OS 隔离不可省略。

父池采用质量/Pareto/结构探索混合，允许有效但未获胜的候选继续探索。
冠军晋级另行约束：目标改善超过配对重复实验 spread 与绝对 0.005 下限，所有指标不得
超过该 spread 退化，最差对手差距/平均差距排序提高。初始参考始终保留。
三个 seed 或任一标准指标缺失/失败，则候选不进入有效父池，不用零补值。

现成 Cajal 对手结果作为冻结参考，见 `reference/opponents.json` 与来源表；不将历史 nanoBrain
组件分数混入 Cajal。原 Cajal 特征/基线缓存的轻量文件单独快照，用同一评估器计算候选指标；
不复制历史训练权重或运行状态。初始 Cajal 参考会按三个配对种子重新训练。

仅一个活动 Cajal worker，GPU 白名单 0–4，至少 64GiB 空闲显存才准入，避免把小 nanoBrain
的多进程并行预算照搬到大模型。OOM 保留已完成部分并重试，已完成训练和评估可复用。
恢复时核对代码、配置、数据 manifest、候选及结果文件哈希；重复控制器由文件锁拒绝。

## 查看与验证（不启动演化）

```bash
.venv/bin/python -B scripts/cajal_react.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_cajal_research.py
```

默认入口只输出计划，不创建 campaign，不读取 API 凭据，不调用 LLM，不训练。
CPU 验证证据见 `preparation/verification.json` 和 `preparation/` 日志。

未来明确启动/恢复才使用：

```bash
.venv/bin/python -B scripts/cajal_react.py --execute
```

本次没有执行上述启动命令。相同目录自动恢复；改协议/源码后需新结果目录。
输出为 `research-runs/cajal-react/` 下的 run-contract、campaign、完整 lineage、proposals、
jobs、标准逐 seed CSV 与冻结冠军。完成 50 代不自动扩量、不自动更新上游 Cajal 配方或网站。

原始 Cajal README 保留供模型、任务与基础设施说明；本文件说明此独立演化副本。
