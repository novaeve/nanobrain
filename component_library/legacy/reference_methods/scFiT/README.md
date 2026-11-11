# scFiT

[English](#english) | [中文](#中文)

## English

scFiT is a research prototype for time-conditioned single-cell expression trajectory modeling, focused on held-out timepoint prediction in developmental single-cell time series.

The maintained open-source path is **Route A**: train on observed time bins, generate held-out timepoint expression distributions, and evaluate the predicted distributions against real held-out cells. Older top-k reconstruction experiments, GPU stress utilities, and exploratory variants are preserved under `archive/` for traceability but are not part of the primary release workflow.

Code status: **research prototype, not production-ready**.

### What Is Included

- scFiT model implementation.
- Route A training, prediction artifact generation, and evaluation code.
- Mouse neural crest/PNS glia configs and protocols.
- Benchmark configs/protocols for WOT/Schiebinger, Drosophila, and Zebrafish tasks.
- Baseline wrappers for scNODE, PRESCIENT, and MIOFlow where the corresponding third-party code is available.
- A synthetic smoke test that runs the core pipeline without downloading data.

Data matrices, model checkpoints, logs, `.env`, and generated results are intentionally excluded.

### Repository Layout

- `src/`: importable Python package for the scFiT model and maintained Route A code.
- `configs/`: YAML training configs, ablations, benchmark configs, and protocol manifests.
- `data/`: expected local data location; only documentation is tracked.
- `scripts/`: command-line entrypoints for smoke tests, training, evaluation, preprocessing, baselines, and figures.
- `notebooks/`: optional notebooks; outputs must remain cleared before commit.
- `results/`: default local output location for checkpoints, artifacts, metrics, caches, and figures.
- `third_party/`: vendored external baseline/reference code used for comparisons.
- `archive/`: legacy or exploratory code outside the maintained path.
- `docs/`: historical command logs, result summaries, and development notes.

### Installation

Recommended Python version: 3.10. Tested environment details are in `ENVIRONMENT.md`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

### Smoke Test

Run this first after installation. It creates a tiny synthetic dataset under `/tmp`, prepares evaluation assets, trains a one-step tiny model, writes a prediction artifact, and evaluates it.

```bash
PYTHONPATH=src python scripts/smoke_test.py --work-dir /tmp/scfit_smoke
```

Successful completion means the maintained pipeline can import, resolve paths, train, save artifacts, and evaluate metrics in the current environment.

### Data Layout

Datasets are not included. The configs use repository-relative paths so the project can be moved between machines.

Expected mouse layout:

```text
data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad
data/processed/neural_crest_pns_glia_hvg4096/
```

Expected benchmark layout:

```text
data/raw/benchmarks/
data/processed/benchmarks/
```

Processed bundles must contain:

- `expression_log1p_clipped.npy`
- `cell_metadata.pkl`
- `obs_names.npy`
- `hvg_gene_indices.npy`
- `hvg_var_names.npy`
- `hvg_gene_names.npy`
- `dataset_metadata.json`

See `data/README.md` and the preparation scripts for the expected structure.

### Main Mouse Pipeline

Prepare evaluation assets:

```bash
python scripts/prepare_route_a_assets.py \
  --protocol configs/protocols/mouse/interpolation.yaml \
  --output-dir results/eval_assets/mouse/interpolation
```

Train scFiT:

```bash
python scripts/train.py \
  --config configs/mouse/config_mouse_route_a_interpolation.yaml
```

Evaluate a saved prediction artifact:

```bash
python scripts/evaluate.py \
  --protocol configs/protocols/mouse/interpolation.yaml \
  --artifact-dir results/experiments/mouse/neural_crest_route_a_interpolation_v1/eval_artifacts/final \
  --assets-dir results/eval_assets/mouse/interpolation
```

The same pattern applies to:

- `configs/mouse/config_mouse_route_a_near.yaml`
- `configs/mouse/config_mouse_route_a_far.yaml`

### Benchmarks And Baselines

Benchmark configs live in `configs/benchmark/`; benchmark protocols live in `configs/protocols/benchmark/`.

Example scFiT benchmark run:

```bash
python scripts/train.py \
  --config configs/benchmark/wot_three_interpolation.yaml
```

Example scNODE baseline run:

```bash
python scripts/run_scnode_protocol.py \
  --protocol configs/protocols/mouse/interpolation.yaml \
  --scnode-root third_party/scNODE-main \
  --output-dir results/eval_results/mouse/scnode_interpolation \
  --assets-dir results/eval_assets/mouse/interpolation \
  --device cuda
```

Batch run scripts are provided in `scripts/run_scfit_*.sh` and `scripts/run_route_a_baselines_*.sh`. They assume the required datasets, assets, CUDA environment, and third-party dependencies are already available.

### Configuration

Most experiment behavior is controlled by YAML:

- `protocol.path`: held-out timepoint protocol.
- `experiment.output_dir`: where checkpoints and artifacts are written.
- `model`: transformer size and conditioning settings.
- `training`: batch size, learning rate, epoch count, seed, loss weights, and checkpoint cadence.
- `validation`: validation metric subset and cell sampling limits.
- `inference`: rollout steps and micro-batch size.
- `evaluation.assets_dir`: PCA/kNN/marker assets used by evaluation.

Random seeds are set from `training.seed`; maintained configs currently use `42`.

### Security And Release Hygiene

- `.env` is ignored; `.env.example` documents optional variables.
- Notebook outputs and execution counts are cleared.
- Generated data, checkpoints, metrics, logs, and caches are ignored.
- Current-file security scan notes are in `SECURITY_AUDIT.md`.

## 中文

scFiT 是一个面向单细胞发育时间序列建模的科研原型代码库，核心任务是在已观测时间点上训练模型，并预测未观测时间点的单细胞表达分布。

当前维护的开源主线是 **Route A**：使用观测时间 bin 训练模型，生成 held-out 时间点的表达分布预测结果，并与真实 held-out 细胞进行评估。早期 top-k 重构实验、GPU 压测工具和探索性代码保留在 `archive/` 中，仅用于追溯，不属于主要复现实验流程。

代码状态：**research prototype, not production-ready**。

### 包含内容

- scFiT 模型实现。
- Route A 训练、预测 artifact 生成和评估代码。
- 小鼠 neural crest/PNS glia 任务的配置与 protocol。
- WOT/Schiebinger、Drosophila、Zebrafish benchmark 的配置与 protocol。
- scNODE、PRESCIENT、MIOFlow baseline 的封装脚本，前提是对应第三方代码和依赖可用。
- 一个不需要下载数据的 synthetic smoke test，用于检查核心 pipeline 是否可运行。

数据矩阵、模型权重、日志、`.env` 和生成结果均不包含在本仓库中。

### 目录结构

- `src/`：可安装的 Python 包，包含 scFiT 模型和 Route A 主线代码。
- `configs/`：训练配置、消融配置、benchmark 配置和 protocol manifest。
- `data/`：本地数据预期放置位置，仓库中只保留说明文档。
- `scripts/`：smoke test、训练、评估、预处理、baseline 和画图入口脚本。
- `notebooks/`：可选 notebook；提交前必须清空输出。
- `results/`：本地默认输出目录，用于保存 checkpoint、artifact、metric、cache 和 figure。
- `third_party/`：用于 baseline 对比的外部参考代码。
- `archive/`：非主线的 legacy 或探索性代码。
- `docs/`：历史命令、结果总结和开发说明。

### 安装

推荐 Python 版本：3.10。更详细的环境信息见 `ENVIRONMENT.md`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

开发模式安装：

```bash
pip install -e .
```

### 冒烟测试

安装后建议先运行冒烟测试。该测试会在 `/tmp` 下生成一个很小的 synthetic 数据集，准备评估资源，训练一个单步小模型，生成预测 artifact，并执行评估。

```bash
PYTHONPATH=src python scripts/smoke_test.py --work-dir /tmp/scfit_smoke
```

如果该命令成功结束，说明当前环境下主线 pipeline 可以正常 import、解析路径、训练、保存 artifact 并计算评估指标。

### 数据组织

仓库不包含数据。配置文件使用相对路径，因此项目可以在不同机器间移动。

小鼠任务预期目录：

```text
data/raw/neural_crest_pns_glia/Neural_crest_PNS_glia.h5ad
data/processed/neural_crest_pns_glia_hvg4096/
```

Benchmark 任务预期目录：

```text
data/raw/benchmarks/
data/processed/benchmarks/
```

预处理后的 bundle 需要包含：

- `expression_log1p_clipped.npy`
- `cell_metadata.pkl`
- `obs_names.npy`
- `hvg_gene_indices.npy`
- `hvg_var_names.npy`
- `hvg_gene_names.npy`
- `dataset_metadata.json`

具体结构见 `data/README.md` 和相关预处理脚本。

### 小鼠主线流程

准备评估资源：

```bash
python scripts/prepare_route_a_assets.py \
  --protocol configs/protocols/mouse/interpolation.yaml \
  --output-dir results/eval_assets/mouse/interpolation
```

训练 scFiT：

```bash
python scripts/train.py \
  --config configs/mouse/config_mouse_route_a_interpolation.yaml
```

评估已保存的预测 artifact：

```bash
python scripts/evaluate.py \
  --protocol configs/protocols/mouse/interpolation.yaml \
  --artifact-dir results/experiments/mouse/neural_crest_route_a_interpolation_v1/eval_artifacts/final \
  --assets-dir results/eval_assets/mouse/interpolation
```

同样的模式也适用于：

- `configs/mouse/config_mouse_route_a_near.yaml`
- `configs/mouse/config_mouse_route_a_far.yaml`

### Benchmark 与 Baseline

Benchmark 配置位于 `configs/benchmark/`，benchmark protocol 位于 `configs/protocols/benchmark/`。

scFiT benchmark 示例：

```bash
python scripts/train.py \
  --config configs/benchmark/wot_three_interpolation.yaml
```

scNODE baseline 示例：

```bash
python scripts/run_scnode_protocol.py \
  --protocol configs/protocols/mouse/interpolation.yaml \
  --scnode-root third_party/scNODE-main \
  --output-dir results/eval_results/mouse/scnode_interpolation \
  --assets-dir results/eval_assets/mouse/interpolation \
  --device cuda
```

批量运行脚本位于 `scripts/run_scfit_*.sh` 和 `scripts/run_route_a_baselines_*.sh`。这些脚本默认所需数据、评估资源、CUDA 环境和第三方依赖都已准备好。

### 配置说明

主要实验行为由 YAML 控制：

- `protocol.path`：held-out 时间点 protocol。
- `experiment.output_dir`：checkpoint 和 artifact 输出目录。
- `model`：transformer 尺寸和条件输入设置。
- `training`：batch size、learning rate、epoch 数、随机种子、loss 权重和 checkpoint 保存频率。
- `validation`：验证指标子集和采样上限。
- `inference`：rollout 步数和 micro-batch 大小。
- `evaluation.assets_dir`：评估使用的 PCA/kNN/marker 资源目录。

随机种子由 `training.seed` 设置；当前维护配置使用 `42`。

### 安全与发布说明

- `.env` 已加入忽略列表，`.env.example` 只记录可选变量。
- Notebook 输出和执行计数已清空。
- 生成的数据、checkpoint、metric、日志和 cache 均被忽略。
- 当前文件级安全扫描记录见 `SECURITY_AUDIT.md`。

## Citation

Paper link: TODO.

```bibtex
@inproceedings{sun2026learning,
  title = {Learning Single-Cell Developmental Dynamics with Target-State Flow Matching},
  author = {Yuyang Sun and Jiaming Ma and Jiachang Zhan and Pingxing Wang and Zhe Zhao and Yu Li and Yimin Fan and Lei Li},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```
