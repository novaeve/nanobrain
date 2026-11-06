# unified_1m：BrainWM 统一预训练快速迭代子集

从 V4 train+validation 分片中抽取，共 1,015,388 个细胞。每个细胞都同时有细胞类型和脑区标签，全部是健康人脑。不再切分 train/val/test，整个子集都用于预训练。

- 构建脚本：`/data/brainwm/unified_1m_build/build_unified_1m.py`（seed 20261103）
- 校验脚本：同目录下的 `validate_unified_1m.py`

## 组成

| 部分 | 细胞数 | 抽样方式 |
|---|---:|---|
| 空间（MERFISH，已核实切片身份） | 314,566 | 96 张切片，每张切片配额相同，按 500 µm 方块整块抽取，共 1,427 块，块内细胞全部保留 |
| 仅表达 | 700,822 | 按"细胞类型 × 脑区"组设上限 c≈1216；每个数据集最多 42k、最少 3k |

## 文件约定（n = 1,015,388）

**表达矩阵**：CSR 格式，基因词表沿用 V4 的 65,918 个基因。
- `values.npy`：float32，原始整数 count。
- `gene_index.npy`：int32。
- `row_ptr.npy`：int64，长度 n+1。
- 只存非零值。

**逐细胞标签**：int32，形状 (n,)，编码与 V4 的 `joint_metadata_vocab.json` 一致。
- 文件：`cell_type`、`anatomy`、`integration_batch`、`dataset`、`technology`、`assay`、`donor`、`section`、`life_stage`、`cl_id`、`uberon_id`。

**测量面板**：用来区分"没测"和"测了是 0"。
- `panel_id.npy`：每个细胞对应一个面板。
- 该细胞测了哪些基因：`panel_gene_index[panel_gene_ptr[p]:panel_gene_ptr[p+1]]`。

**空间信息**：
- `spatial.npy`：float32，(n,3)，单位 µm，非空间细胞为 NaN。
- `tile_id.npy`：int32，非空间细胞为 -1。
- `neighbors.npy`：int32，(n,16)，同一切片内最近的 16 个邻居在本子集中的行号，不含自身，非空间细胞为 -1。
- `neighbor_dist.npy`：float32，邻居距离，已按从近到远排序。

**溯源**：`source_row.npy`，int64，(n,2)，每行对应 V4 中的 (分片序号, 分片内行号)。

**标签层级**：
- `cell_type_vocab.json`：两级，lineage（22 类）→ cell_type（167 类）；每个类别附带它在 CL 中的全部 is_a 祖先。
- `anatomy_vocab.json`：三级，division（前脑/中脑/后脑/脊髓/其他）→ region（端脑/间脑/中脑/小脑/脑桥/延髓/脊髓/其他）→ anatomy（91 类）。层级按 UBERON 2026-10-01 的 is_a 和 part_of 关系推出；10 个推不出的条目手工映射，并标注为 `manual_override`。

`manifest.json` 记录了各维度的计数和抽样参数。

## 已知局限

- 已核实切片身份的空间细胞几乎全部来自一个数据集：HMBA-MERSCOPE 基底神经节，310k 个；另有约 4.4k 来自皮层 MERFISH。
- 皮层 MERFISH 的切片身份未经核实，那部分细胞只作为仅表达细胞进入子集。
- 166/167 个细胞类型出现在子集里，缺 `CL:4310143`。
- 82/91 个脑区、49/57 个批次出现在子集里；缺的都是没有完整标签的数据集。
