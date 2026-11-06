# Cajal 联合演化数据

数据通过本仓库的 GitHub Releases 发布，发布标签为 `cajal-react-data-20261106`。下载和校验入口为
`python variants/cajal-react/scripts/download_data.py`。默认将四个数据集解压到本目录。

| 目录 | 细胞数 | 用途 |
|---|---:|---|
| `unified_1m/` | 1,015,388 | 五目标联合预训练 |
| `eval_10k/` | 9,960 | 细胞类型、脑区、基因插补评估 |
| `eval_integration_10k/` | 9,867 | 批次整合评估 |
| `eval_spatial/` | 40,789 | 空间插补评估 |

完整解压体积约 16 GB。数据保存原始 count、65,918 基因词表、测量面板、标签层级、行身份和空间邻居。
各评估集与预训练集无共享细胞。原研究评估集会参与演化反馈，固定 V4 test 不包含在本发布中。

分卷、每个文件的 SHA-256、行数和原始 manifest 指纹记录在
`variants/cajal-react/reference/data-release.json`。数据二进制文件由下载器管理。
