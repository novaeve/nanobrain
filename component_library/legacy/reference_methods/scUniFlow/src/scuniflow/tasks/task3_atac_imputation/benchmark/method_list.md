整理好了，按优先级排列，每个方法给出完整的安装和运行命令。

---

## 方法一：MAGIC（最简单，先跑通）

```bash
# 安装
pip install magic-impute

# 运行（直接在 Python 里）
import magic
import scanpy as sc
import numpy as np

adata = sc.read_h5ad("pbmc_10k_atac_tfidf.h5ad")

# MAGIC 接受原始 TF-IDF，不需要 log1p
magic_op = magic.MAGIC(
    knn=5,          # 近邻数，scATAC 推荐 5-10
    t=3,            # 扩散步数，建议先用 'auto'
    n_pca=100,
    random_state=42,
    n_jobs=-1       # 用所有 CPU
)

X_magic = magic_op.fit_transform(adata.X.toarray())

adata_magic = adata.copy()
adata_magic.X = X_magic.astype(np.float32)
adata_magic.write_h5ad("pbmc_10k_atac_magic.h5ad")
```

**注意**：MAGIC 对 100k peaks 的全量矩阵内存压力较大，建议先在 top-5k 上验证，再用全量。如果 OOM，加 `solver='approximate'` 参数。

---

## 方法二：SCALE（最重要的对比方法）

```bash
# 安装（需要独立 conda 环境，SCALE 依赖较老的 PyTorch）
conda create -n scale_env python=3.8
conda activate scale_env
pip install scale

# 或者从源码装（更稳定）
git clone https://github.com/jsxlei/SCALE.git
cd SCALE
pip install -e .
```

```bash
# 运行（命令行）
SCALE.py \
    --input pbmc_10k_atac_raw_counts.h5ad \
    --outdir ./scale_output/ \
    --impute \                    # 输出 imputed 矩阵
    --n_feature 107915 \          # peak 数量，设 -1 禁用 HVG 筛选
    --gpu 0 \
    --seed 42
```

**重要**：SCALE 的输入需要**原始 count 矩阵**（不是 TF-IDF），它内部自己做归一化。所以要保存一份原始 count 的 h5ad 作为 SCALE 的输入，和你的模型输入分开。输出在 `scale_output/adata.h5ad` 的 `obsm['imputed']` 字段里。

---

## 方法三：scOpen

```bash
# 安装
conda create -n scopen_env python=3.8
conda activate scopen_env
pip install scopen
```

```bash
# 运行（命令行，接受 10x 格式或稀疏矩阵）
scopen \
    --input ./filtered_feature_bc_matrix/ \  # Cell Ranger 输出目录
    --input_format 10X \
    --output_dir ./scopen_output/ \
    --output_prefix pbmc_atac \
    --output_format h5ad \
    --estimate_rank \             # 自动估计 NMF rank，推荐开启
    --verbose 1
```

scOpen 输出的是**开放概率矩阵**（值域 0-1），语义是"该 peak 在该 cell 中开放的概率"，和你的 TF-IDF 预测值在数值上不可直接比较，但用于计算 AUROC/meta-cell 相关性时，先把两者都 rank 归一化再比较就行。

---

## 方法四：Raw TF-IDF baseline（零代码成本）

不需要安装任何东西，直接用预处理完的原始矩阵在评估脚本里跑一遍 `eval_atac_biological.py`，作为"不做任何 imputation"的下限。这个 baseline 是必须报的。

---

## 统一评估接口

四个方法跑完后，输出格式不统一，需要对齐后再送进 `eval_atac_biological.py`。建议写一个转换脚本：

```python
def load_imputed_matrix(method: str, path: str):
    """统一加载各方法输出，返回 (n_cells, n_peaks) numpy array"""
    
    if method == "scUniFlow":
        adata = sc.read_h5ad(path)
        return adata.X.toarray()
    
    elif method == "SCALE":
        adata = sc.read_h5ad(path)
        return adata.obsm["imputed"]          # SCALE 结果在 obsm 里
    
    elif method == "scOpen":
        adata = sc.read_h5ad(path)
        return adata.X.toarray()              # scOpen 直接写 .X
    
    elif method == "MAGIC":
        adata = sc.read_h5ad(path)
        return adata.X.toarray()
    
    elif method == "raw":
        adata = sc.read_h5ad(path)
        return adata.X.toarray()              # 原始 TF-IDF，不做任何处理
```

---

## 环境隔离建议

SCALE 和 scOpen 的依赖版本互相冲突，建议用三个独立 conda 环境：

```
scatac_main    # 你的 scUniFlow + MAGIC（同一环境，依赖兼容）
scale_env      # SCALE 专用
scopen_env     # scOpen 专用
```

评估脚本统一在 `scatac_main` 环境里跑，只用各方法的输出 h5ad 文件，不需要导入各方法的包。