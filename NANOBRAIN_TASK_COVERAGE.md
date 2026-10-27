# nanoBrain 100k Task Coverage

历史记录日期：2026-09-16。本页记录旧 joint 协议，当前组件／独立任务状态见 [组件与数据审计](NANOBRAIN_COMPONENT_API.md)。2026-09-17 发现旧 completion 未在模型入口遮住 held-out 输入，其数值不作为合规 masked-completion 对照；需按新协议重训。

## 固定任务

`nanobrain_v3_100k` 使用唯一的 60k / 20k / 20k train-validation-test 划分。四个高层任务为数据整合、细胞注释、空间建模和发育建模；空间建模由 spatial niche 与 spatial completion 两个 evaluator 组成，发育任务为 unpaired cross-sectional population forecasting。

最终协议为 `NANOBRAIN-100K-FMT-EMA-EQUAL-20260916`。模型和 baseline 在 test 开放前冻结，最终 test campaign 已完成，错误行数为 0。

## Final test 覆盖矩阵

| 方法 | 整合 | 细胞注释 | 空间 niche | 空间 completion | 发育 | 实验方式 |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| **nanoBrain FMT** | ✅ | ✅ | ✅ | ✅ | ✅ | 共享模型从头联合训练；3 seeds × 2,000 steps；validation 选 checkpoint |
| XGBoost | N/A | ✅ | N/A | N/A | N/A | train-only 拟合原生分类器 |
| CellTypist | N/A | ✅ | N/A | N/A | N/A | 官方 SGD trainer 在 train 上训练原生分类器 |
| Geneformer V1-10M | ✅ | ✅ | N/A | N/A | N/A | 冻结官方 backbone；整合 ZS embedding；注释 train-only LP |
| scGPT-brain | ✅ | ✅ | N/A | N/A | N/A | 冻结官方 backbone；整合 ZS embedding；注释 train-only LP |
| scGPT-human | ✅ | ✅ | N/A | N/A | N/A | 冻结官方 backbone；整合 ZS embedding；注释 train-only LP |
| scVI / scANVI | ✅ | ✅ | N/A | N/A | N/A | scVI latent；scANVI 半监督原生分类头 |
| Harmony | ✅ | N/A | N/A | N/A | N/A | 官方无监督 transductive integration |
| Scanorama | ✅ | N/A | N/A | N/A | N/A | 官方无监督 transductive integration |
| BBKNN | ✅ | N/A | N/A | N/A | N/A | 官方 batch-balanced graph 与 diffusion embedding |
| GraphST | N/A | N/A | ✅ | N/A | N/A | 官方 sparse spatial graph，无监督训练 100 epochs |
| scFiT | N/A | N/A | N/A | N/A | ✅ | train population pairs 训练 2,000 steps，冻结 CFM checkpoint |
| scNODE | N/A | N/A | N/A | N/A | ✅ | train population pairs 训练 2,000 steps，冻结 VAE latent ODE checkpoint |
| TrajectoryNet | N/A | N/A | N/A | N/A | ✅ | train population pairs 训练 2,000 steps，冻结 CNF checkpoint |
| Classical annotation controls | N/A | ✅ | N/A | N/A | N/A | majority、centroid、kNN、SGD、SVM、NB、logistic regression |
| Integration controls | ✅ | N/A | N/A | N/A | N/A | raw PCA、batch-centered PCA |
| Spatial controls | N/A | N/A | ✅ | ✅ | N/A | no-structure、neighbor mean、coordinate K-means；zero/gene/row fill |
| Development controls | N/A | N/A | N/A | N/A | ✅ | identity、global delta、ridge delta |

`N/A` 表示该方法没有对应的原生输入或任务输出，因此不强行接入 nanoBrain 的任务头。

## 指标覆盖

| 任务 | Final table 中的主要指标 |
| --- | --- |
| 细胞注释 | Accuracy、Balanced Accuracy、MCC、Cohen Kappa、Precision/Recall/F1 的 macro/micro/weighted、Top-3 Accuracy、NLL、ECE、Brier、AUPRC，以及 donor/anatomy/age strata |
| 数据整合 | Cell-type ASW、inverse batch ASW、graph connectivity、isolated-label F1、iLISI、cLISI、composite，以及 anatomy/age strata |
| Spatial niche | ARI、NMI、ASW、Moran's I、Geary's C、neighborhood enrichment |
| Spatial completion | RMSE、MAE、nonzero RMSE/Pearson、global/cell/gene Pearson、Spearman、cosine、held-out RMSE/MAE/Pearson |
| 发育 | Population MSE、mean error、variance error、MMD-RBF、cell-type proportion L1、cell-type JS、time correlation |

宏 AUROC 在 test 未覆盖全部 train classes 时不可定义，常数 completion prediction 的 Pearson 也可能不可定义。这些记录保持空值，并由 final receipt 单独计数。

## Foundation model 模式

| 模式 | 定义 | 当前完成情况 |
| --- | --- | --- |
| Zero-shot embedding (`ZS`) | 官方 checkpoint 冻结，直接评价 embedding | Geneformer、scGPT-brain、scGPT-human 的 integration 已完成 |
| Linear probe (`LP`) | 官方 backbone 冻结，只用 train embedding 和 label 拟合线性头 | Geneformer、scGPT-brain、scGPT-human 的 annotation 已完成 |
| Native head / fine-tune | 使用方法自己的训练器和任务头 | CellTypist、XGBoost、scANVI 已完成 |
| Population checkpoint | 只在 train population pairs 上训练，test 加载冻结权重 | scFiT、scNODE、TrajectoryNet 已完成 |

## 审计状态

- nanoBrain：3/3 seeds 完成，共 552 条 per-seed metric rows。
- Internal baselines：18 个方法，共 1,148 条 metric rows。
- External baselines：冻结的 13/13 方法完成，共 989 条 metric rows。
- 汇总：184 条 nanoBrain 聚合指标，2,321 条统一 final long-table 指标，error rows = 0。
- 所有 final run manifest 都使用数据 manifest SHA-256 `4a7ae3f0d8d919dc79e3273f3d9b0aa96fd682214e97d14d545471baf3a6ea73`。

UCE、SpaGCN、STAGATE 和 scUniFlow 没有进入本次冻结的 13-method final set：当前分别缺少完整官方资产、超出稠密邻接资源约束、缺少 Python 3.12 兼容依赖、缺少所需 pretrained checkpoint。它们不占用本次 final campaign 的 complete 计数。

上表是历史固定模型的任务覆盖，不能据此认定整个 AutoResearch 原计划已完成。组件搜索、多代独立任务演化与数据来源／section 审计的当前状态以最新审计为准。用户已排除全量数据扩展。
