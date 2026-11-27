# Cajal v3 ReAct 联合演化

`cajal-v3-seed` 分支默认使用 Cajal v3 campaign。运行状态记录在所选输出目录的 `status.json`。原分支保留 Cajal v1 实验。 [Cajal v3 campaign](../campaigns/cajal-v3/campaign.toml) · [演化配置](../configs/cajal_react_v3.json) · [原分支](https://github.com/novaeve/nanobrain/tree/main)

## 种子与训练预算

一份共享模型联合优化 gene、integration、cell_type、anatomy、spatial 五个目标，同一 checkpoint 输出六项报告指标。v3 增加了同基因的空间邻居表达作为基因补全上下文。预训练沿用约 101 万健康人脑细胞、65,918 个基因的词表、原始 counts 与逐细胞测量面板。评估沿用原来的三个研究评估集和六项指标。

演化按 30 代、每代 10 个候选、seed=0 配置。每候选采用 30,000 步学习率日程，在 10,000 步固定停止。参考模型是已有的 Cajal v3 第 10,000 步权重，记录为 `scale_v3_1m@10000`。候选从随机初始化的权重开始训练，搜索范围涵盖架构组件、损失均衡、优化器和训练采样参数。

协议 `CAJAL-V3-JOINT-REACT-UNIFIED-1M-R2` 采用 `random_init_self_contained`。父代传递架构、源码和超参数。断点恢复核对候选、seed、协议及训练配置的身份，只复用自身 checkpoint。v3 配方的 SelfGram 从模型自身的分类头计算目标。最终冠军导出包含完整候选源码、解析配置和空的外部权重依赖清单。

CUDA 模式允许使用 GPU 0–4，全局最多 16 个并行阶段，每卡最多 3 个 worker。CPU 模式同时只执行一个训练或评估阶段。两种模式的合成验收最多使用 4 个 CPU worker，恢复时复用同一运行内已完成的提案、结果和断点。CUDA 控制器交接可接管已运行的阶段。

## 环境与计划检查

完整演化需要 Linux、Python 3.11+、PyTorch 2.9+ 和 bubblewrap。在仓库根目录为所选设备安装依赖。

- **CPU** 使用 PyTorch 的 CPU 构建，训练和预测使用 PyTorch 原生注意力，整合评估依赖 CPU 版 JAX 和 scikit-learn。运行完整预算可能耗时很长。
- **CUDA** 使用与 GPU、CUDA 和 PyTorch 版本匹配的 FlashAttention-3 等扩展。 [组件依赖](../component_library/dependencies.json)

组件需要额外依赖时，按目录声明将所需软件包安装到本工作区的 `.component_deps/`。CPU 模式的运行契约记录实际安装版本和文件指纹，CUDA 模式按组件环境清单核对。

```bash
# CPU 环境
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.9'
python -m pip install -e '.[eval-cpu,scvi,dev]'

# CUDA 环境
python -m pip install -e '.[eval,scvi,dev]'
```

用所选环境查看模型配置和运行计划。

```bash
python -m brainwm_unified.cli show configs/experiments/cajal_v3.toml
python -B scripts/cajal_react.py --device cpu
python -B scripts/cajal_react.py --device cuda
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

普通 Cajal 训练入口也接受 `--device cpu` 或 `--device cuda`，默认使用 CUDA。以下命令使用本地已下载的数据，从随机初始化开始训练，在 10,000 步停止。

```bash
python -m brainwm_unified.train \
  --config configs/experiments/cajal_v3.toml \
  --device cpu \
  --out runs/cajal-v3-local-cpu \
  --set data.root=assets/cajal/unified_1m train.stop_at=10000
```

完整 ReAct 演化会核对参考权重与固定输入。部署需要以下资源。

- 数据集挂载到 `/data/brainwm/unified_1m`、`/data/brainwm/eval_10k`、`/data/brainwm/eval_integration_10k` 和 `/data/brainwm/eval_spatial`，并通过已有指纹检查。下载器也支持 `--destination /data/brainwm`。
- [v3 参考记录](../reference/existing_cajal_v3_10k.json)声明的服务器 checkpoint `/data/brainwm/BrainWM-Unified/runs/scaling/v3_1m/step10000.pt`、解析配置与训练回执。文件路径与 SHA-256 会逐项核对。该权重用作性能参照。
- 整合评估使用的已有 HVG/scaler/PCA 缓存。
- CPU 和 CUDA 模式都需要 Linux bubblewrap 及足够的主存。CUDA 模式需要配置的 GPU 可见。其训练阶段初始预留 32 GiB 显存，评估阶段预留 8 GiB，每卡另留 4 GiB 余量。控制器还要求至少 32 GiB 可用主存，OOM 重试会提高任务预留。
- 控制器进程的 API 环境变量。配置中的 `api_env_file` 保留了来源路径，可通过已有 `NANOBRAIN_API_KEY`、`NANOBRAIN_API_BASE_URL` 等进程环境变量提供凭据。

公开数据包包含四个细胞数据集。参考 checkpoint 由完整演化部署环境提供。`reference/` 中保留了核验所需的既有性能快照与协议定义。

依赖和固定路径准备齐全后，可执行冻结和演化入口。

```bash
nanobrain --config configs/cajal_react_v3.json --freeze

# 二选一。在 NVIDIA GPU 上运行
python -B scripts/cajal_react.py --execute --device cuda

# 在 CPU 上运行
python -B scripts/cajal_react.py --execute --device cpu
```

`--execute` 会调用 LLM 并启动或恢复真实训练。CUDA 和 CPU 的默认输出目录分别为 `research-runs/cajal-v3-react/` 和 `research-runs/cajal-v3-react-cpu/`。可以用 `--output` 指定自定义目录。运行契约会记录设备模式，打开已有输出目录续跑时须沿用原模式。两种模式使用同一参考分数、数据、seed 和每候选 10,000 步训练预算。

CPU 和 CUDA 沿用相同的评估指标。计算后端和精度的差异可能使同一组预测的评分略有不同。

## 评估与研究记录

| 任务 | 指标 |
|---|---|
| 细胞类型 | Fine macro-F1 |
| 脑区 | Fine macro-F1 |
| 整合 | Integration bio、Integration batch |
| 基因插补 | 每细胞 Pearson 相关中位数 |
| 空间插补 | 每细胞 Pearson 相关中位数 |

三个研究评估集与预训练细胞无重叠，演化会反复读取其结果。单 seed 采用 `seed_count=1` 和 null 标准差。源协议保留完整 DAG、父系、失败分支、所有候选源码与训练回执。

`reference/opponents.json` 保存 13 个既有整方法基线，`reference/current_method_scores.json` 保留历史 checkpoint 轨迹。v3 参照分数写入 `reference/cajal_v3_five_tasks.csv`。新协议的候选成绩随运行写入所选输出目录。

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
