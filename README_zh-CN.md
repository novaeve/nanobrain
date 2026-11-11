# nanoBrain

**nanoBrain 是面向科研、教学和原型验证的小型虚拟细胞建模项目。** 它提供一条从固定的 10 万细胞数据集到模型训练、验证和结果对比的可读路径，是快速实验入口，不是 Brain World Model 的缩小部署版本。

> [!IMPORTANT]
> 本项目继续使用既有的 10 万细胞数据及 6 万 / 2 万 / 2 万训练、验证、测试划分，不切换到 10M 数据集。历史测试集已经评测，不是新的盲测集。

**Language:** [English](README.md) | 简体中文 · [四项任务](#四项任务) · [Leaderboard](benchmarks/nanobrain_v3_100k/LEADERBOARD.md) · [数据](#数据集) · [复现约定](#复现约定)

## 快速开始

当前的一键环境脚本面向 Linux x86_64、Python 3.12、CPU。按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/)安装 uv，将公司提供的数据下载地址写入本地 `.env`，然后运行：

```bash
cp .env.example .env
# 在 .env 中设置 NANOBRAIN_DATA_URL；如有归档校验值，再设置 NANOBRAIN_DATA_SHA256。
./scripts/bootstrap.sh
```

脚本会创建项目专属环境、安装 CPU 版 PyTorch 和开发依赖，并下载或校验 `assets/nanobrain_v3_100k_metadata_v2/` 下的数据。若数据已放到这个位置，则只做校验。

运行一个短的 CPU 冒烟训练：

```bash
uv run python scripts/train_nanobrain.py \
  --data assets/nanobrain_v3_100k_metadata_v2 \
  --task annotation --steps 20 --batch-size 16 \
  --device cpu --precision fp32 --output results/annotation-smoke
```

其余任务将 `--task` 改为 `integration`、`development` 或 `spatial_niche`。CUDA 环境应按服务器 CUDA 运行时，从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装对应 wheel，再使用 `--device cuda`。20 步只用于检查流程，不代表正式 benchmark 的训练预算。

## 四项任务

| 任务 | 训练目标 | 主要指标 | 教程 |
|---|---|---|---|
| 整合 | 保持观测表达，同时降低批次信号 | Integration composite | [教程](docs/tasks/integration.md) |
| 注释 | 预测训练集中观测到的细胞类型标签 | Macro F1、accuracy | [教程](docs/tasks/annotation.md) |
| 发育 | 根据源群体和条件预测目标群体 | Population MSE、gene Spearman | [教程](docs/tasks/development.md) |
| Niche 预测 | 利用验证过的空间邻域预测解剖来源 niche | ARI、NMI | [教程](docs/tasks/spatial-niche.md) |

四项任务分别训练模型和优化器。常规训练只读取训练集和验证集，不会打开测试集。模型与任务约定见[组件指南](docs/components.md)。

要替换编码器并训练自己的模型，请看[自定义模型教程](docs/custom-model.md)，其中包含小型编码器示例、配置文件和可直接运行的训练命令。

## 数据集

固定数据共 100,000 个细胞，来自 `BRAINWM-HUMAN-GLOBAL-6-2-2-V3` 划分：训练集 60,000，验证集 20,000，测试集 20,000。每个细胞包含固定的 256 基因面板，以及注释、批次、发育和空间元数据；归一化只在训练集上拟合。metadata-v2 修复了行身份信息，空间邻域仅使用独立验证过的 section。

数据包不会提交到 Git。下载地址可配置，供公司后续托管经过批准的数据包。准备命令会验证 manifest、文件哈希、行数、归一化来源和元数据版本，并保留现有固定划分，不会随机重分行。

```bash
uv run nanobrain-data prepare
# 或用本地 ZIP/TAR 校验并安装：
uv run nanobrain-data prepare --archive /path/to/nanobrain-100k-v2.tar.gz
```

详见[数据契约](docs/data.md)与[数据目录说明](assets/README.md)。

## Leaderboard

[Leaderboard](benchmarks/nanobrain_v3_100k/LEADERBOARD.md)汇总既有 100k 固定基准，完整整体指标见 [`overall_metrics.csv`](benchmarks/nanobrain_v3_100k/overall_metrics.csv)。表格只含当前四项任务，并记录均值、标准差、种子数、评测数量和测试集已公开评测的状态。不同方法的训练预算不同，解读结果时需查看方法协议。历史上可选的 spatial completion 不属于当前四项任务，因此不列入此表。

从原始汇总表重新生成：

```bash
uv run python scripts/build_leaderboard.py \
  --input benchmarks/nanobrain_v3_100k/frozen_test_metrics.csv
```

## 复现约定

- 数据清单记录固定抽样、划分、基因面板和仅用训练集拟合的归一化参数。
- 训练回执记录随机种子、步数、批量大小、验证集选择、任务和数据版本。
- nanoBrain 最终基准由三个独立随机种子汇总；其他方法保留自己的种子数和评测覆盖范围。
- 当前固定测试集属于描述性结果，因为标签已经开放。新的模型搜索应只使用训练集和验证集，不能根据展示的测试结果调参。
- baseline 表格由统一的评测记录生成，不手工填写指标。

```bash
uv run pytest
uv run python scripts/build_leaderboard.py
```

## 项目定位

nanoBrain 面向小型服务器、工作站和教学环境，帮助研究人员先验证编码器、任务头、损失函数和采样策略，再决定是否投入更大规模实验。它不是临床决策系统，也不声称结果可直接外推到当前固定数据集以外的人群。

引用与许可证信息按公司发布要求补充；第三方代码说明见 [`docs/third_party.md`](docs/third_party.md)。
