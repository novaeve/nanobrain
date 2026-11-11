# scUniFlow 使用手册

## 1. 环境安装

```bash
conda create -n scUniFlow python=3.10 -y
conda activate scUniFlow

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install causal-conv1d>=1.2.0
MAX_JOBS=8 pip install mamba-ssm --no-build-isolation
pip install triton>=2.1.0

pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0 networkx>=3.1 \
    umap-learn>=0.5.5 scikit-learn>=1.2.0 \
    matplotlib>=3.7.0 seaborn>=0.12.0 \
    scanpy>=1.9.0 anndata>=0.9.0 \
    PyYAML>=6.0 tqdm>=4.65.0

python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); import mamba_ssm; print('mamba_ssm OK'); import scanpy; print('scanpy OK')"
```

## 2. 数据说明

SEA-AD 数据存放在 `${SEA_AD_ROOT:-data/SEA-AD}/`：

| 键名 | 路径 | 说明 |
|---|---|---|
| `mtg_toy` | `${SEA_AD_ROOT:-data/SEA-AD}/MTG/sea_ad_mtg_toy_50k.h5ad` | MTG 5 万细胞子集（试跑用） |
| `dlpfc_toy` | `${SEA_AD_ROOT:-data/SEA-AD}/DLPFC/sea_ad_dlpfc_toy_50k.h5ad` | DLPFC 5 万细胞子集 |
| `mtg_full` | `${SEA_AD_ROOT:-data/SEA-AD}/MTG/92b37feb-...h5ad` | MTG 全量 ~130 万细胞 |
| `dlpfc_full` | `${SEA_AD_ROOT:-data/SEA-AD}/DLPFC/dad4819b-...h5ad` | DLPFC 全量 |

Donor 级细胞计数：`donor_cell_counts_full_mtg.csv`（90 个 donor，用于 train/val 划分）。

## 3. 训练

```bash
conda activate scUniFlow
cd ${SCUNIFLOW_ROOT:-.}
```

### 3.1 快速试跑（toy 数据）

```bash
python train.py --data mtg_toy --batch_size 32
```

### 3.2 全量训练（自动按 donor 划分 9:1 train/val）

```bash
python train.py --data mtg_full --batch_size 64
```

全量数据会自动读取 `donor_cell_counts_full_mtg.csv`，按 donor 粒度划分验证集（~10% 细胞），并在训练过程中定期验证。划分信息保存在实验目录中。

### 3.3 单阶段 epoch 训练

训练现在按 `epoch` 驱动，一个 `epoch` 就是完整遍历一轮训练集，不再区分 phase 1 / phase 2。

### 3.4 指定实验名称

```bash
python train.py --data mtg_full --run_name my_experiment_v1
```

### 3.4.1 小样本快速验证

```bash
python train.py \
  --config experiments/tiny_sanity_xpred/config.yaml \
  --data mtg_toy \
  --run_name tiny_sanity_mtg
```

也可以临时覆盖子采样规模：

```bash
python train.py \
  --config experiments/tiny_sanity_xpred/config.yaml \
  --data mtg_toy \
  --max_cells 2048 \
  --run_name tiny_sanity_2k
```

### 3.5 后台训练

```bash
nohup python train.py --data mtg_full --batch_size 64 > train.log 2>&1 &
tail -f train.log
```

### 3.6 参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | `mtg_toy` | 数据源键名或 h5ad 路径 |
| `--batch_size` | `32` | 批大小 |
| `--config` | `config.yaml` | 配置文件路径 |
| `--device` | `cuda` | 训练设备 |
| `--run_name` | 自动生成 | 实验名称（决定输出子目录名） |

### 3.7 配置文件关键项（config.yaml）

```yaml
training:
  experiment_dir: experiments   # 实验输出根目录
  epochs: 80                    # 完整训练轮数
  lr: 0.0005                    # 初始学习率
  min_lr: 0.000001              # cosine 调度最小学习率
  val_every_epochs: 1           # 每隔多少个 epoch 验证
  save_every_epochs: 10         # 每隔多少个 epoch 保存 checkpoint
  val_ratio: 0.1                # 验证集占比（按 donor 划分）
  seed: 42
  loss_weights:
    weighted_x1: 1.0
    pearson: 0.1
    cell_pearson: 0.1
```

## 4. 实验输出结构

所有过程文件存放在 `experiments/` 下，每次训练一个子目录：

```
experiments/
└── mtg_full_20260321_143000/
    ├── config.yaml              # 本次实验的完整配置快照
    ├── split_info.json          # donor 划分详情（哪些 donor 用于验证）
    ├── train_log.jsonl          # 逐 epoch 训练日志（JSON Lines 格式）
    ├── model_final.pt           # 最终模型权重
    └── checkpoints/
        ├── checkpoint_epoch_010.pt
        ├── checkpoint_epoch_020.pt
        └── checkpoint_last.pt
```

- `train_log.jsonl`：每行一条 JSON，包含 epoch、step、train_loss、val_rmse 等，方便用 pandas 读取绘制曲线。
- `split_info.json`：记录了哪些 donor 用于训练、哪些用于验证，保证实验可复现。

## 5. 推理

```python
import torch
import yaml
from model import scRNAModel
from data import load_sea_ad, get_dataloader

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

count_matrix, gene_ids, gene_names, meta = load_sea_ad("mtg_toy")
dataloader = get_dataloader(count_matrix, gene_ids, batch_size=16, shuffle=False)

model = scRNAModel(
    n_genes=meta["n_genes"],
    d_model=cfg["model"]["d_model"],
    grid_size=cfg["model"]["grid_size"],
).to("cuda")
model.load_state_dict(torch.load("experiments/xxx/model_final.pt", map_location="cuda"))
model.eval()

with torch.no_grad():
    for g_id, x_1, x_0, mask_ind in dataloader:
        g_id, x_1, mask_ind = g_id.cuda(), x_1.cuda(), mask_ind.cuda()
        x_pred = model.inference(
            g_id, x_1, mask_ind,
            n_steps=cfg["inference"]["n_steps"],
            temperature=cfg["inference"]["temperature"],
        )
        break
```

## 6. STRING 谱嵌入流水线

```bash
cd ${SCUNIFLOW_ROOT:-.}/string_spectral
python run_all.py            # 完整流水线
python run_all.py --from 3   # 从第 3 步恢复
```

## 7. Python API 参考

```python
from data import load_sea_ad, load_sea_ad_split, load_sea_ad_by_donor, get_dataloader

# 加载全量
count_matrix, gene_ids, gene_names, meta = load_sea_ad("mtg_full")

# 按 donor 划分 train/val
train_data, val_data, gene_names, meta = load_sea_ad_split("mtg_full", val_ratio=0.1)
train_loader = get_dataloader(*train_data, batch_size=64)
val_loader = get_dataloader(*val_data, batch_size=64, shuffle=False)

# 按 donor 单独加载
donors = load_sea_ad_by_donor()
data = load_sea_ad_by_donor(donor_id=donors[0])
```

## 8. 项目结构

```
scUniFlow/
├── config.yaml                       # 模型与训练超参数
├── train.py                          # 训练入口（单阶段 epoch + 验证 + 实验管理）
├── model.py                          # PermutationModule + GeneMamba + VectorFieldNet
├── gene_mamba.py                     # GeneMamba 架构
├── data.py                           # 数据加载 / donor 划分 / Dataset
├── utils.py                          # Moran's I、Δπ 等工具函数
├── requirements.txt                  # 依赖列表
├── donor_cell_counts_full_mtg.csv    # donor 级细胞计数（用于 train/val 划分）
├── experiments/                      # 训练产出（checkpoint、日志、划分信息）
└── string_spectral/                  # STRING 谱嵌入子流水线
```
