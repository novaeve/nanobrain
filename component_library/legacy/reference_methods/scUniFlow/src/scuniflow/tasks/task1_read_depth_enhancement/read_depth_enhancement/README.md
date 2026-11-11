# Read Depth Enhancement

这个目录现在支持把你已经微调好的 `scFoundation` checkpoint 接到真正的 `read depth enhancement` 任务上。

核心脚本是：

- `train_scfoundation_read_depth_enhancement.py`
- `eval_scfoundation_read_depth_enhancement.py`
- `prepare_seaad_eval_subset.py`

## 任务定义

当前实现是：

- 输入：`x_low`
  - 由原始 `h5ad` raw counts 通过 binomial downsampling 从 `x_high` 构造
- 目标：`x_high`
  - 训练目标是高深度版本的归一化 log-expression
- 模型：scFoundation encoder-decoder
- 初始化：
  - 基座用 `--pretrain_ckpt`
  - 如果你已经有 PBMC 微调好的 checkpoint，再通过 `--init_ckpt` 继续接着训
- 损失：全基因 `MSE`

这样更适合直接拿来和 `scUniFlow` 的 enhancement 版本做对比。

## SEA-AD 子集准备

如果直接在 `SEA-AD MTG full` 上评估，计算量会比较大。现在可以先生成一个 donor-balanced 的验证子集：

```bash
python ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/prepare_seaad_eval_subset.py \
  --per_donor_n 512 \
  --subset_name seaad_mtg_rde_eval_balanced_val_512pd
```

默认会从验证 donor 中每个 donor 采样 `512` 个细胞，输出：

- `${SCUNIFLOW_DATA:-data}/seaad_mtg_rde_eval_balanced_val_512pd.h5ad`
- `${SCUNIFLOW_DATA:-data}/seaad_mtg_rde_eval_balanced_val_512pd.manifest.json`

这个子集适合做快速、可复现的 `read depth enhancement` 对比评估。
如果评估时直接传这个子集 h5ad，建议使用 `--split all`，避免再做一次 train/val 二次拆分。

## 训练示例

```bash
CUDA_VISIBLE_DEVICES=0 python ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/train_scfoundation_read_depth_enhancement.py \
  --pretrain_ckpt ${SCFOUNDATION_ROOT:-external/scFoundation}/models/models.ckpt \
  --init_ckpt /path/to/your/pbmc_finetuned_scfoundation_best_model.pt \
  --data_path ${SCUNIFLOW_DATA:-data}/pbmc68k_raw_counts.h5ad \
  --save_dir ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/runs/pbmc68k_scfoundation_rde \
  --downsample_fold 4.0 \
  --epochs 20 \
  --batch_size 24 \
  --num_workers 0 \
  --val_ratio 0.1 \
  --tgt_highres 4.0
```

## 评估示例

评估会同时输出两层结果：

- reconstruction 指标：
  - `rmse`
  - `mae`
  - `cell/gene PCC`
- clustering 指标：
  - `NMI`
  - `ARI`
  - `silhouette`
  - `calinski_harabasz`
  - `davies_bouldin`

如果 `h5ad.obs` 里有真实标签列：

```bash
CUDA_VISIBLE_DEVICES=0 python ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/eval_scfoundation_read_depth_enhancement.py \
  --ckpt_path ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/runs/pbmc68k_scfoundation_rde/best_model.pt \
  --data_path ${SCUNIFLOW_DATA:-data}/pbmc68k_raw_counts.h5ad \
  --save_dir ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/runs/pbmc68k_scfoundation_rde/eval_val \
  --split val \
  --num_workers 0 \
  --label_col cell_type
```

如果没有标签，就用高深度 reference Leiden：

```bash
CUDA_VISIBLE_DEVICES=0 python ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/eval_scfoundation_read_depth_enhancement.py \
  --ckpt_path ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/runs/pbmc68k_scfoundation_rde/best_model.pt \
  --data_path ${SCUNIFLOW_DATA:-data}/pbmc68k_raw_counts.h5ad \
  --save_dir ${SCUNIFLOW_ROOT:-.}/eval/read_depth_enhancement/runs/pbmc68k_scfoundation_rde/eval_val \
  --split val \
  --num_workers 0 \
  --ref_resolution 0.4
```

## 输出文件

训练目录会生成：

- `args.json`
- `split_info.json`
- `train_log.jsonl`
- `best_model.pt`
- `last_model.pt`
- `summary.json`

评估目录会生成：

- `summary.json`
- `cell_metrics.csv`
- `gene_metrics.csv`
- `cell_pccs.npy`
- `gene_pccs.npy`
- `run_args.json`
