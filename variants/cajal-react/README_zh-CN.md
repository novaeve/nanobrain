# Cajal ReAct 联合演化变体

本目录发布 Cajal v1 共享模型上的 ReAct 研究变体。源提交为 `df8976caf24e993c1b33407f45d956b9843d5cdd`，源目录为服务器上的 `cajal-react-joint-20261106`。发布时的配置状态为 `prepared_not_started`。

## 与原 nanoBrain 路线的区别

一份共享模型联合优化 gene、integration、cell_type、anatomy、spatial 五个目标，同一 checkpoint 输出六项报告指标。输入是约 101 万健康人脑细胞、65,918 基因词表、原始 count 与逐细胞测量面板。评估使用 Cajal 既有的三个研究评估集。

演化按 50 代、每代 6 个候选、seed=0 配置。每候选采用 30,000 步学习率日程，在 10,000 步固定停止。参考模型为已有的 `cajal_v1@10000`，候选从架构随机初始化。候选搜索同时覆盖架构组件、损失均衡、优化器和训练采样参数。

## 环境与计划检查

运行环境为 Linux、Python 3.11+、PyTorch 2.9+。正式 GPU 训练沿用源项目的 CUDA、FlashAttention-3 等扩展环境，组件依赖记录在 `component_library/dependencies.json`。

```bash
cd variants/cajal-react
python -m pip install -e '.[dev]'
python -m brainwm_unified.cli show configs/experiments/cajal_v1.toml
python scripts/cajal_react.py
```

默认入口打印计划，配置为 `configs/cajal_react.json`。候选协议、开放训练参数、验收和晋级细节见 [完整策略说明](README_CAJAL_REACT.md)。

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

普通 Cajal 训练入口允许显式设置数据路径。例如，在安装 GPU 训练环境后，可将原配方的数据路径指向本地数据目录，并将停止步数设为 10,000。

```bash
python -m brainwm_unified.train \
  --config configs/experiments/cajal_v1.toml \
  --out runs/cajal-v1-local \
  --set data.root=../../assets/cajal/unified_1m train.stop_at=10000
```

完整 ReAct 演化还会核对历史参考与固定输入。这个服务器源码快照含有原始绝对路径约束，完整复现需要准备以下资源。

- 数据集挂载到 `/data/brainwm/unified_1m`、`/data/brainwm/eval_10k`、`/data/brainwm/eval_integration_10k` 和 `/data/brainwm/eval_spatial`，并通过已有指纹检查。下载器也支持 `--destination /data/brainwm`。
- `reference/existing_cajal_v1_10k.json` 声明的原参考 checkpoint、原配方和回执文件。其工件路径和 SHA-256 会逐项校验。
- `configs/cajal_react.json` 中固定的 cell-type 教师 checkpoint，以及整合评估使用的已有 HVG/scaler/PCA 缓存。
- 与组件环境记录匹配的项目依赖、Linux bwrap、可见 GPU，以及源协议要求的至少 64 GiB 空闲显存。
- 控制器进程的 API 环境变量。配置中的 `api_env_file` 保留了来源路径，可通过已有 `NANOBRAIN_API_KEY`、`NANOBRAIN_API_BASE_URL` 等进程环境变量提供凭据。

本次公开数据包包含四个细胞数据集。参考 checkpoint、教师权重、GPU 扩展二进制和运行目录由完整演化部署环境提供。`reference/` 中保留了核验所需的既有性能快照与协议定义。

依赖和路径准备齐全后，正式入口为 `python scripts/cajal_react.py --execute`。它将调用 LLM 并开始真实训练。

## 评估与研究记录

| 任务 | 指标 |
|---|---|
| 细胞类型 | Fine macro-F1 |
| 脑区 | Fine macro-F1 |
| 整合 | Integration bio、Integration batch |
| 基因插补 | 每细胞 Pearson 相关中位数 |
| 空间插补 | 每细胞 Pearson 相关中位数 |

三个研究评估集与预训练细胞无重叠，演化会反复读取其结果。单 seed 采用 `seed_count=1` 和 null 标准差。源协议保留完整 DAG、父系、失败分支、所有候选源码与训练回执。

`reference/opponents.json` 保存 13 个既有整方法基线，`reference/current_method_scores.json` 保存来源项目 63 个方法/实验的 checkpoint 轨迹。它们属于冻结参考数据。新变体的训练成绩需要由后续正式运行产生。

## 目录与来源

```text
brainwm_unified/     Cajal 模型、数据加载、联合训练与目标
cajal_research/      ReAct 工具、候选验收、沙箱、选择与恢复
component_library/  Cajal 与旧 nanoBrain 组件、来源哈希和依赖
configs/            Cajal 配方、演化参数和实验配置
reference/          固定评估协议、既有参考、来源与数据发布清单
scripts/            训练、预测、评估、演化和数据下载入口
data_build/         数据构建脚本及定义
tests/              来源项目的 CPU、GPU、边界与回归测试
```

组件目录中的上游文件保留原有许可和来源声明。加速库的来源提交见 `THIRD_PARTY.txt`。历史实验方法的开发规范见 `CLAUDE.md`。
