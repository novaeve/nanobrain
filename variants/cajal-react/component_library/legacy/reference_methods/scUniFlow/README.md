# scUniFlow

[English](#english) | [中文](#中文)

<a id="english"></a>

## English

scUniFlow is a research prototype for single-cell multimodal expression modeling with conditional flow matching and 2D Mamba-style gene or feature layouts.

Paper: **scUniFlow: A Unified Flow Matching Framework for Sparse Omics Enhancement**. This repository is a cleaned research-code release, not production-ready software.

### Task Definition

Mask reconstruction is treated as the pretraining objective, not as a downstream task.

The public downstream task numbering is:

| Public name | Directory | Purpose |
| --- | --- | --- |
| Pretraining | `src/scuniflow/pretrain_eval/` and `src/scuniflow/train.py` | Masked gene-expression reconstruction |
| Task 1 | `src/scuniflow/tasks/task1_read_depth_enhancement/` | Read-depth enhancement |
| Task 2 | `src/scuniflow/tasks/task2_spatial_gene_imputation/` | Spatial gene imputation |
| Task 3 | `src/scuniflow/tasks/task3_atac_imputation/` | scATAC imputation |

### Repository Layout

| Path | Role |
| --- | --- |
| `src/` | Model, data loading, pretraining, downstream task code, and baseline adapters |
| `configs/` | Versioned YAML templates for pretraining and task-specific experiments |
| `data/` | Local datasets and generated coordinate files; ignored by Git |
| `scripts/` | Unified shell entry points for common train/eval commands |
| `notebooks/` | Optional notebooks; clear outputs before committing |
| `results/` | Local checkpoints, logs, metrics, and benchmark outputs; ignored by Git |
| `docs/` | Design notes and historical method documentation |

### Setup

Python 3.10 is recommended. CUDA requirements depend on the PyTorch and `mamba_ssm` wheels available on your machine.

```bash
conda env create -f scUniFlow_environment.yml
conda activate scUniFlow
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` only for local data locations or external baseline repositories. `.env` is ignored by Git.

### Data

This repository does not include datasets, checkpoints, generated arrays, or benchmark outputs.

Expected local inputs include:

- SEA-AD or other scRNA-seq `.h5ad` files for pretraining and read-depth enhancement
- generated gene coordinate arrays under `data/gene_coords/`
- spatial transcriptomics references for Task 2
- scATAC `.h5ad` or 10x inputs for Task 3
- optional external baseline checkpoints under `external/`

Most scripts accept explicit `--data`, `--h5ad_path`, `--checkpoint`, or `--spectral_coords` arguments. Prefer CLI arguments over editing code.

### Quick Start

Run the synthetic smoke test without external data:

```bash
python scripts/create_smoke_data.py
CUDA_VISIBLE_DEVICES=0 bash scripts/train_pretrain.sh \
  configs/pretrain/smoke.yaml data/smoke/smoke.h5ad \
  --run_name smoke_test --max_cells 8
```

The downstream entry points can be inspected with:

```bash
bash scripts/run_task1_read_depth.sh --help
bash scripts/run_task2_spatial.sh --help
bash scripts/run_task3_atac.sh configs/task3_atac/config_atac.yaml --help
```

### End-to-End Pipeline

The intended release pipeline is:

1. pretrain a gene-expression backbone with masked reconstruction;
2. use the pretraining checkpoint for Task 1 and Task 2 task-head fine-tuning;
3. train or resume the Task 3 scATAC model with ATAC feature coordinates;
4. run the matching evaluation scripts for each task.

#### 1. Prepare Inputs

This repository does not ship data. Put local inputs under `data/` or pass absolute paths on the command line.

Required pretraining inputs:

- an scRNA-seq `.h5ad` with cells x genes counts;
- a gene coordinate array with shape `(n_genes, 2)`, for example `data/gene_coords/seaad_augmented_gene_coords.npy`;
- a YAML config whose `model.n_genes`, `model.grid_size`, and `training.spectral_coords` match the data.

Task-specific inputs:

- Task 1 read-depth enhancement uses the same scRNA-seq `.h5ad` universe as the pretrained backbone.
- Task 2 spatial gene imputation needs a reference scRNA-seq `.h5ad` and a 10x Visium directory. The Visium `var['gene_ids']` must match the reference Ensembl IDs.
- Task 3 scATAC imputation needs an ATAC `.h5ad` and a peak/feature coordinate array with shape `(n_features, 2)`.

#### 2. Pretrain

Run masked reconstruction pretraining:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_pretrain.sh \
  configs/pretrain/default.yaml /path/to/scRNA.h5ad \
  --run_name seaad_pretrain \
  --max_cells 0
```

Main outputs:

- `results/pretrain/seaad_pretrain/config.yaml`
- `results/pretrain/seaad_pretrain/checkpoints/checkpoint_best.pt`
- `results/pretrain/seaad_pretrain/checkpoints/checkpoint_last.pt`

Use `checkpoint_best.pt` for downstream task-head fine-tuning when it exists; otherwise use `checkpoint_last.pt`.

Optional masked reconstruction evaluation:

```bash
bash scripts/eval_pretrain_mask_recon.sh \
  --checkpoint results/pretrain/seaad_pretrain/checkpoints/checkpoint_best.pt \
  --data /path/to/scRNA.h5ad \
  --config results/pretrain/seaad_pretrain/config.yaml \
  --output_dir results/pretrain_eval/seaad_pretrain
```

#### 3. Task 1: Read-Depth Enhancement

Task 1 fine-tunes a read-depth enhancement head on top of the pretrained scUniFlow backbone.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_task1_read_depth.sh \
  --exp_dir results/pretrain/seaad_pretrain \
  --checkpoint results/pretrain/seaad_pretrain/checkpoints/checkpoint_best.pt \
  --data /path/to/scRNA.h5ad \
  --output_dir results/task1_read_depth/seaad \
  --epochs 20 \
  --batch_size 4
```

By default, the script freezes the pretrained backbone and trains only the task head. Add `--train_mamba` or `--train_router` only when you intentionally want joint fine-tuning.

The training script can automatically run the read-depth evaluation after training. To skip that step, pass `--skip_eval_after_train`.

Standalone evaluation:

```bash
python src/scuniflow/tasks/task1_read_depth_enhancement/scuniflow_read_depth_enhancement/eval_scuniflow_read_depth_enhancement.py \
  --exp_dir results/pretrain/seaad_pretrain \
  --task_ckpt results/task1_read_depth/seaad/best_model.pt \
  --data /path/to/scRNA.h5ad \
  --output_dir results/task1_read_depth/seaad/eval_val \
  --split val
```

#### 4. Task 2: Spatial Gene Imputation

Task 2 fine-tunes a spatial imputation head using a pretrained scUniFlow backbone, a reference scRNA-seq atlas, and Visium spots.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_task2_spatial.sh \
  --exp_dir results/pretrain/seaad_pretrain \
  --checkpoint results/pretrain/seaad_pretrain/checkpoints/checkpoint_best.pt \
  --output_dir results/task2_spatial/mouse_brain \
  --ref_h5ad /path/to/reference_scRNA.h5ad \
  --visium_root /path/to/visium_root \
  --sample_ids ST8059048 \
  --observed_gene_count 1024 \
  --target_gene_count 256 \
  --epochs 20 \
  --batch_size 32
```

`--visium_root` should contain one subdirectory per sample in standard 10x Visium format. `--sample_ids` is comma-separated; omit it to use all sample directories.

Standalone evaluation:

```bash
python src/scuniflow/tasks/task2_spatial_gene_imputation/eval_scuniflow_spatial_gene_imputation_head.py \
  --task_ckpt results/task2_spatial/mouse_brain/best_model.pt \
  --output_dir results/task2_spatial/mouse_brain/eval_val \
  --split val
```

#### 5. Task 3: scATAC Imputation

Task 3 uses the same flow-matching/Mamba framework for ATAC features. It is not a direct RNA checkpoint fine-tune unless the feature universe is identical; normally it is trained with an ATAC `.h5ad` and ATAC peak coordinates.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_task3_atac.sh \
  configs/task3_atac/config_atac.yaml \
  --data /path/to/atac_tfidf_or_counts.h5ad \
  --run_name atac_imputation \
  --max_cells 0
```

Use `--resume /path/to/checkpoint.pt` to continue an ATAC run with the same feature universe and config.

Imputation and biological evaluation:

```bash
python src/scuniflow/tasks/task3_atac_imputation/impute_and_eval_atac.py \
  --data /path/to/atac_tfidf_or_counts.h5ad \
  --checkpoint results/task3_atac/atac_imputation/checkpoints/checkpoint_last.pt \
  --output results/task3_atac/atac_imputation/imputed.h5ad \
  --mode all_zero \
  --expr_transform tfidf_log1p_1e4
```

#### Smoke-Test Coverage

The cleaned release was checked in the local `scUniFlow` environment with synthetic data:

| Stage | Status | Scope |
| --- | --- | --- |
| Pretraining | Passed | synthetic h5ad, 16 genes, 1 epoch |
| Task 1 | Passed | loaded pretraining smoke checkpoint and fine-tuned read-depth head |
| Task 2 | Passed | synthetic reference h5ad plus synthetic 10x Visium directory |
| Task 3 | Passed | synthetic ATAC-style h5ad, 16 features, 1 epoch |

These smoke tests verify import paths, configs, checkpoint loading, training loops, validation loops, and output writing. They do not validate scientific performance.

### Reproducibility Notes

Random seeds are configured through each YAML file under `training.seed`, plus task-specific validation seeds where applicable, such as `validation.mask_seed` and `validation.noise_seed`.

Outputs are written under `results/` by default. The code keeps generated artifacts out of version control via `.gitignore`.

### Security And Release Hygiene

- No datasets, weights, logs, generated arrays, notebooks with outputs, or paper-writing files are included.
- Local paths are routed through `.env`, environment variables, configs, or CLI arguments.
- Run secret scanning before publishing a Git history, for example with `trufflehog git file://$PWD`.

### Citation

```bibtex
@misc{sun2026scuniflow,
  title  = {scUniFlow: A Unified Flow Matching Framework for Sparse Omics Enhancement},
  author = {Yuyang Sun and Yimin Fan and Jiaming Ma and Jiachang Zhan and Pingxing Wang and Zhe Zhao and Yu Li and Lei Li},
  year   = {2026},
  note   = {Research prototype},
  url    = {https://github.com/Canchen-AI-Group/scUniFlow}
}
```

<a id="中文"></a>

## 中文

scUniFlow 是一个用于单细胞多组学稀疏信号增强的科研原型代码库。它使用条件 flow matching 和二维 Mamba 风格的基因或特征布局，对稀疏、低深度或部分观测的分子谱进行恢复。

论文标题：**scUniFlow: A Unified Flow Matching Framework for Sparse Omics Enhancement**。本仓库是整理后的科研代码发布版本，不是生产级软件。

### 任务定义

Mask reconstruction 在本项目中是预训练目标，不作为下游任务编号。公开版本的任务编号如下：

| 名称 | 目录 | 目的 |
| --- | --- | --- |
| 预训练 | `src/scuniflow/pretrain_eval/` 和 `src/scuniflow/train.py` | Masked gene-expression reconstruction |
| Task 1 | `src/scuniflow/tasks/task1_read_depth_enhancement/` | Read-depth enhancement |
| Task 2 | `src/scuniflow/tasks/task2_spatial_gene_imputation/` | Spatial gene imputation |
| Task 3 | `src/scuniflow/tasks/task3_atac_imputation/` | scATAC imputation |

### 目录结构

| 路径 | 用途 |
| --- | --- |
| `src/` | 模型、数据读取、预训练、下游任务和 baseline 适配代码 |
| `configs/` | 预训练和任务实验的 YAML 配置模板 |
| `data/` | 本地数据和生成的坐标文件；默认被 Git 忽略 |
| `scripts/` | 常用训练和评测命令入口 |
| `notebooks/` | 可选 notebook；提交前应清空输出 |
| `results/` | 本地 checkpoint、日志、指标和 benchmark 输出；默认被 Git 忽略 |
| `docs/` | 设计说明和历史方法文档 |

### 环境安装

推荐使用 Python 3.10。CUDA 版本取决于本机 PyTorch 和 `mamba_ssm` wheel 的兼容性。

```bash
conda env create -f scUniFlow_environment.yml
conda activate scUniFlow
pip install -r requirements.txt
cp .env.example .env
```

只在 `.env` 中填写本地数据路径或外部 baseline 仓库路径。`.env` 已被 Git 忽略。

### 数据说明

本仓库不包含数据集、checkpoint、生成数组或 benchmark 输出。常见本地输入包括：

- 用于预训练和 read-depth enhancement 的 SEA-AD 或其他 scRNA-seq `.h5ad` 文件；
- `data/gene_coords/` 下的基因坐标数组；
- Task 2 所需的空间转录组 reference 和 Visium 数据；
- Task 3 所需的 scATAC `.h5ad` 或 10x 输入；
- `external/` 下可选的外部 baseline checkpoint。

多数脚本支持通过 `--data`、`--h5ad_path`、`--checkpoint` 或 `--spectral_coords` 显式传参。优先使用命令行参数，不建议直接改代码里的路径。

### 快速开始

无需外部数据即可运行合成数据 smoke test：

```bash
python scripts/create_smoke_data.py
CUDA_VISIBLE_DEVICES=0 bash scripts/train_pretrain.sh \
  configs/pretrain/smoke.yaml data/smoke/smoke.h5ad \
  --run_name smoke_test --max_cells 8
```

查看下游任务入口：

```bash
bash scripts/run_task1_read_depth.sh --help
bash scripts/run_task2_spatial.sh --help
bash scripts/run_task3_atac.sh configs/task3_atac/config_atac.yaml --help
```

### 完整 Pipeline

推荐流程如下：

1. 使用 masked reconstruction 预训练 gene-expression backbone；
2. 将预训练 checkpoint 用于 Task 1 和 Task 2 的 task-head 微调；
3. 使用 ATAC feature coordinates 训练或恢复 Task 3 scATAC 模型；
4. 对每个任务运行对应评测脚本。

#### 1. 准备输入

仓库不随附数据。将本地输入放到 `data/`，或在命令行中传入绝对路径。

预训练需要：

- cells x genes 的 scRNA-seq `.h5ad`；
- 形状为 `(n_genes, 2)` 的基因坐标数组，例如 `data/gene_coords/seaad_augmented_gene_coords.npy`；
- 与数据匹配的 YAML 配置，其中 `model.n_genes`、`model.grid_size` 和 `training.spectral_coords` 必须一致。

任务输入：

- Task 1 使用与预训练 backbone 相同基因空间的 scRNA-seq `.h5ad`；
- Task 2 需要 reference scRNA-seq `.h5ad` 和 10x Visium 目录，Visium 的 `var['gene_ids']` 需要与 reference Ensembl ID 对齐；
- Task 3 需要 ATAC `.h5ad` 和形状为 `(n_features, 2)` 的 peak 或 feature 坐标数组。

#### 2. 预训练

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_pretrain.sh \
  configs/pretrain/default.yaml /path/to/scRNA.h5ad \
  --run_name seaad_pretrain \
  --max_cells 0
```

主要输出：

- `results/pretrain/seaad_pretrain/config.yaml`
- `results/pretrain/seaad_pretrain/checkpoints/checkpoint_best.pt`
- `results/pretrain/seaad_pretrain/checkpoints/checkpoint_last.pt`

下游微调优先使用 `checkpoint_best.pt`；如果不存在，则使用 `checkpoint_last.pt`。

#### 3. Task 1: Read-Depth Enhancement

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_task1_read_depth.sh \
  --exp_dir results/pretrain/seaad_pretrain \
  --checkpoint results/pretrain/seaad_pretrain/checkpoints/checkpoint_best.pt \
  --data /path/to/scRNA.h5ad \
  --output_dir results/task1_read_depth/seaad \
  --epochs 20 \
  --batch_size 4
```

默认冻结预训练 backbone，只训练 task head。只有在明确需要联合微调时才添加 `--train_mamba` 或 `--train_router`。

#### 4. Task 2: Spatial Gene Imputation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_task2_spatial.sh \
  --exp_dir results/pretrain/seaad_pretrain \
  --checkpoint results/pretrain/seaad_pretrain/checkpoints/checkpoint_best.pt \
  --output_dir results/task2_spatial/mouse_brain \
  --ref_h5ad /path/to/reference_scRNA.h5ad \
  --visium_root /path/to/visium_root \
  --sample_ids ST8059048 \
  --observed_gene_count 1024 \
  --target_gene_count 256 \
  --epochs 20 \
  --batch_size 32
```

`--visium_root` 应包含标准 10x Visium 格式的样本子目录。`--sample_ids` 支持逗号分隔；省略时使用所有样本目录。

#### 5. Task 3: scATAC Imputation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_task3_atac.sh \
  configs/task3_atac/config_atac.yaml \
  --data /path/to/atac_tfidf_or_counts.h5ad \
  --run_name atac_imputation \
  --max_cells 0
```

Task 3 通常不是直接从 RNA checkpoint 微调，除非 feature universe 完全一致。一般应使用 ATAC `.h5ad` 和 ATAC peak coordinates 单独训练或 resume。

### 可复现性说明

随机种子通过 YAML 中的 `training.seed` 设置，部分任务还包含 `validation.mask_seed`、`validation.noise_seed` 等验证种子。输出默认写入 `results/`，并通过 `.gitignore` 排除在版本控制之外。

### 安全与发布说明

- 本仓库不包含数据集、权重、日志、生成数组、带输出的 notebook 或论文写作文件。
- 本地路径通过 `.env`、环境变量、配置文件或命令行参数传入。
- 发布 Git 历史前建议运行 secret scanning，例如 `trufflehog git file://$PWD`。

### 引用

```bibtex
@misc{sun2026scuniflow,
  title  = {scUniFlow: A Unified Flow Matching Framework for Sparse Omics Enhancement},
  author = {Yuyang Sun and Yimin Fan and Jiaming Ma and Jiachang Zhan and Pingxing Wang and Zhe Zhao and Yu Li and Lei Li},
  year   = {2026},
  note   = {Research prototype},
  url    = {https://github.com/Canchen-AI-Group/scUniFlow}
}
```
