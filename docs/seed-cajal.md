# Cajal v3 ReAct 联合演化

`cajal-v3-seed` 分支默认使用 [Cajal v3 campaign](../campaigns/cajal-v3/campaign.toml) 和 [演化配置](../configs/cajal_react_v3.json)。仓库中的配置标记为 `prepared_not_started`。启动后，运行状态写入 `research-runs/cajal-v3-react/status.json`。[原分支](https://github.com/novaeve/nanobrain/tree/main)保留 Cajal v1 实验。

## 种子与训练预算

一份共享模型联合优化 gene、integration、cell_type、anatomy、spatial 五个目标，同一 checkpoint 输出六项报告指标。v3 增加了同基因的空间邻居表达作为基因补全上下文。预训练沿用约 101 万健康人脑细胞、65,918 个基因的词表、原始 counts 与逐细胞测量面板。评估沿用原来的三个研究评估集和六项指标。

演化按 30 代、每代 10 个候选、seed=0 配置。每候选采用 30,000 步学习率日程，在 10,000 步固定停止。参考模型是已有的 Cajal v3 第 10,000 步权重，记录为 `scale_v3_1m@10000`。候选从随机初始化的权重开始训练，搜索范围涵盖架构组件、损失均衡、优化器和训练采样参数。

协议 `CAJAL-V3-JOINT-REACT-UNIFIED-1M` 采用 `random_init_self_contained`。父代传递架构、源码和超参数。断点恢复核对候选、seed、协议及训练配置的身份，只复用自身 checkpoint。v3 配方的 SelfGram 从模型自身的分类头计算目标。最终冠军导出包含完整候选源码、解析配置和空的外部权重依赖清单。

调度允许 GPU 0–4，全局最多 16 个并行阶段，每卡最多 3 个 worker。CPU 合成验收上限为 4，每候选最多并行 2 条任务预测与评分流水线。每代最多 10 个候选同时推进，实际并发还受显存和主存约束。控制器交接会接管已运行的阶段，恢复时复用已完成提案、结果和自身断点。

## 环境与计划检查

训练环境为 Linux、Python 3.11+、PyTorch 2.9+。正式 GPU 训练使用 CUDA、FlashAttention-3 等扩展，组件依赖记录在 `component_library/dependencies.json`。

```bash
# 在 cajal-v3-seed 分支的仓库根目录执行。
python -m pip install -e '.[dev]'
python -m brainwm_unified.cli show configs/experiments/cajal_v3.toml
python -B scripts/cajal_react.py --config configs/cajal_react_v3.json
```

默认入口打印计划。候选协议、搜索范围、验收和晋级细节见 [演化协议](react-protocol.md)。

## 数据下载与校验

```bash
# 在本目录执行。下载器使用 Python 3.12+。
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

默认下载到仓库根目录的 `assets/cajal/`。发布标签为 `cajal-react-data-20261106`，四个数据集的完整解压体积约 16 GB。分卷、归档和每个数据文件均有 SHA-256 校验。

| 数据目录 | 细胞数 | 用途 |
|---|---:|---|
| `unified_1m` | 1,015,388 | 联合预训练 |
| `eval_10k` | 9,960 | 类型、脑区、基因插补评估 |
| `eval_integration_10k` | 9,867 | 整合评估 |
| `eval_spatial` | 40,789 | 空间插补评估 |

数据发布清单位于 `reference/data-release.json`。原始行身份、空间邻居、标签层级和面板按源数据保存，原始 manifest 指纹与 `reference/input_fingerprints.json` 对应。

## 训练与完整演化的部署边界

完整演化在独立的 Git 工作目录中运行。部署时可直接克隆本分支。

```bash
git clone --branch cajal-v3-seed https://github.com/novaeve/nanobrain.git /work/nanobrain-v3
cd /work/nanobrain-v3
```

普通 Cajal 训练入口允许显式设置数据路径。例如，安装 GPU 训练环境后，可用已下载的数据在 10,000 步停止。

```bash
python -m brainwm_unified.train \
  --config configs/experiments/cajal_v3.toml \
  --out runs/cajal-v3-local \
  --set data.root=assets/cajal/unified_1m train.stop_at=10000
```

完整 ReAct 演化会核对参考权重与固定输入。部署需要以下资源。

- 数据集挂载到 `/data/brainwm/unified_1m`、`/data/brainwm/eval_10k`、`/data/brainwm/eval_integration_10k` 和 `/data/brainwm/eval_spatial`，并通过已有指纹检查。下载器也支持 `--destination /data/brainwm`。
- [v3 参考记录](../reference/existing_cajal_v3_10k.json)声明的服务器 checkpoint `/data/brainwm/BrainWM-Unified/runs/scaling/v3_1m/step10000.pt`、解析配置与训练回执。文件路径与 SHA-256 会逐项核对。该权重用作性能参照。
- 整合评估使用的已有 HVG/scaler/PCA 缓存。
- 与组件环境记录匹配的项目依赖、Linux bwrap 和可见 GPU。训练阶段初始预留 32 GiB 显存，评估阶段预留 8 GiB，每卡另留 4 GiB 余量。控制器还要求至少 32 GiB 可用主存，OOM 重试会提高任务预留。
- 控制器进程的 API 环境变量。配置中的 `api_env_file` 保留了来源路径，可通过已有 `NANOBRAIN_API_KEY`、`NANOBRAIN_API_BASE_URL` 等进程环境变量提供凭据。

本次公开数据包包含四个细胞数据集。参考 checkpoint、GPU 扩展二进制和运行目录由完整演化部署环境提供。`reference/` 中保留了核验所需的既有性能快照与协议定义。

依赖和固定路径准备齐全后，可执行冻结和演化入口。

```bash
nanobrain --config configs/cajal_react_v3.json --freeze
python -B scripts/cajal_react.py --config configs/cajal_react_v3.json --execute
```

`--execute` 会调用 LLM 并启动或恢复真实训练。

## 评估与研究记录

| 任务 | 指标 |
|---|---|
| 细胞类型 | Fine macro-F1 |
| 脑区 | Fine macro-F1 |
| 整合 | Integration bio、Integration batch |
| 基因插补 | 每细胞 Pearson 相关中位数 |
| 空间插补 | 每细胞 Pearson 相关中位数 |

三个研究评估集与预训练细胞无重叠，演化会反复读取其结果。单 seed 采用 `seed_count=1` 和 null 标准差。源协议保留完整 DAG、父系、失败分支、所有候选源码与训练回执。

`reference/opponents.json` 保存 13 个既有整方法基线，`reference/current_method_scores.json` 保留历史 checkpoint 轨迹。v3 参照分数写入 `reference/cajal_v3_five_tasks.csv`。新协议的候选成绩随运行写入 `research-runs/cajal-v3-react/`。

## 目录与来源

```text
brainwm_unified/     Cajal 模型、数据加载、联合训练与目标
campaigns/cajal-v3/  v3 演化定义、智能体说明与锁文件
nanobrain/           ReAct 工具、候选验收、沙箱、选择与恢复
component_library/  Cajal 与旧 nanoBrain 组件、来源哈希和依赖
configs/            Cajal 配方、演化参数和实验配置
reference/          固定评估协议、既有参考、来源与数据发布清单
scripts/            训练、预测、评估、演化和数据下载入口
data_build/         数据构建脚本及定义
tests/              来源项目的 CPU、GPU、边界与回归测试
```

组件目录中的上游文件保留原有许可和来源声明。加速库的来源提交见 `THIRD_PARTY.txt`。历史实验方法的开发规范见 `CLAUDE.md`。
