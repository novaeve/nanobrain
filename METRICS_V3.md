# 四任务指标协议 V3

2026-11-03 新增的 50 代启动方案见 [EVOLUTION_V3_20261103.md](EVOLUTION_V3_20261103.md)。
用户已明确选择“验证集选权重、测试集反馈选架构”，该启动配置覆盖本文原默认的验证集架构选择政策；
数学指标、数据目标和正式报告的标准单项定义不变。旧验证集驱动配置保留作历史入口。

协议 ID：`NANOBRAIN-METRICS-V3-20261102`。本次在独立 checkout
`/mnt/aliyunsb/zim_rider/nanobrain/try/metrics-v2-20261102` 内更新；目录名保留创建时的 V2 名称。
GitHub 起点为 `novaeve/nanobrain` 的 `b5e6bbf01f8b9f14ec63de873f1fa950c079738c`。

2026-11-03 报告修订：`standard-task-metrics-20261103`。数学定义、数据目标和抽样保持不变，
正式结果与内部选权重/搜索评分分开导出。新证据见 `METRICS_V3_STANDARD_REPORT_VERIFICATION.json`；
此前的 `METRICS_V3_VERIFICATION.json` 保留其创建时的验证记录。

## 四任务与权重选择

| 任务 | 预测对象 | 正式评估的标准指标 | 方向 |
|---|---|---|---|
| annotation | 中心细胞的细胞类型 | `f1_macro` | 越大越好 |
| integration | 同时保留生物信息并降低批次差异的表征 | Label ASW、Isolated-label ASW、KMeans NMI/ARI；类型内 Batch ASW、iLISI、Graph Connectivity，分别报告 | 越大越好 |
| spatial_niche | 中心细胞周围的细胞类型比例 | `composition_mae` | 越小越好 |
| development | 给定来源群体及时间条件的目标细胞分布 | `wasserstein2` | 越小越好 |

每个任务有独立的主干、任务头、优化器和权重，架构可以共用。此次没有增加跨任务共享权重的预训练阶段。
`best_weights` 同时保存验证最佳权重和最终权重；演化复评读取 `best_checkpoint`。
注释、niche、发育分别按 Macro-F1、MAE、W₂ 选权重。整合同时存在生物保留与批次去除目标，
仍用预先声明的项目效用选权重；这个效用不作为正式 benchmark 主指标或官方 scIB 总分。
架构、超参数和权重只根据 train/validation 选择。冻结后再用 test 评估所选权重，作为该 seed 的测试成绩。
本次只做代码、数据标签构造与短训练核验，未启动正式演化或完整基线重训。

注释的 Macro-F1 对评估集中有真实样本的已标注类别等权平均，预测不能改变类别集合。
保留 Accuracy、Balanced Accuracy、MCC 等诊断；这仍是闭集注释任务，没有新增未知类型拒识。
可选的 `spatial_completion` 不是上述四任务之一，其独立主指标为 `heldout_rmse`。

## 空间 niche：真实邻域组成

当前任务采用 Nicheformer 式的中心表达预测邻域组成。输入仅为中心细胞表达，模型入口隐藏坐标、
邻居索引、邻居表达；目标邻居的类型标签只用于构造监督目标。原 anatomy 标签另存为历史代理，
不进入组成回归。只有聚类编号的 GraphST 空间域适配器不能直接参加这项任务的主指标比较。

- 在经过独立几何核验的完整切片/采集单元上查找邻居，避免只在 100k 抽样中心中找邻居。
- 按细胞身份排除中心自身，保留坐标重合的不同细胞。不跨切片、供体或数据划分组邻域。
- 预先固定四个尺度：在完整训练切片中抽样中心，以第 10/20/50/100 个邻居距离的中位数，
  分别确定各数据来源的四个半径。半径不读取 validation/test 来校准；它们不是每个中心自适应的固定 k 邻域。
- 类型词表只从完整训练切片建立。当前为 64 个已知类型，加 1 个 unknown/unseen 槽位，共 65 类。
- 目标为 `类型邻居数 / 全部邻居数`，未知类型的质量保留在 unknown 槽位；不是对计数做 Softmax。
  这项归一化是本项目的选择，与 Nicheformer 公开实现的 Softmax(counts) 有区别。
- 每个中心、每个半径至少有一个邻居，且训练词表内的已知邻居比例至少 0.8，才计入组成损失和指标。
- 组成头输出 `[N, 4, 65]` 概率。训练用 MSE，每个半径等权；主指标为每个半径内对有效中心及类型求 MAE，
  再对四个半径等权平均。不要把这个数直接读成“预测正确的百分比”。
- 辅助报告组成 MSE、JSD、各半径/类型/切片 MAE、切片等权 MAE，以及邻居数量的 MAE、R²。
  单独的数量头使用非负输出和 log1p MSE（默认权重 0.1），输入表征 detach，避免数量尺度主导组成主干。

新数据目录：`assets/nanobrain_v3_100k_niche_v3`。原始父数据为
`/mnt/aliyunsb/zim_rider/nanobrain/brain/assets/nanobrain_v3_100k_metadata_v2`，没有修改。
中心行、表达、行身份和划分保持原样；只在派生目录更新 niche 目标与 mask，增加数量目标和溯源文件。

| 划分 | 全部中心 | 至少一个尺度有效 | 四尺度各自有效数 |
|---|---:|---:|---|
| train | 60,000 | 14,872 | 14,843 / 14,869 / 14,871 / 14,872 |
| validation | 20,000 | 1,612 | 1,593 / 1,605 / 1,612 / 1,612 |
| test | 20,000 | 1,293 | 1,286 / 1,291 / 1,292 / 1,293 |

不能把全部 100k 都算作空间样本。训练空间样本来自来源 8 与 65，验证/测试当前仅来源 65；
它们尚不能代表跨平台泛化。来源 64 未通过相同几何核验，没有自动启用。

来源 65 的 O1 采集单元通过原文件的样本、细胞、供体身份及坐标核对后，才扩展到测试供体。
完整测试分片的类型 sidecar 原为 -1，本次仅在新私有缓存中从原始 H5AD 准备注释目标，
核对全部 571,528 个邻域细胞的坐标行序；没有改写共享分片，也没有将测试目标加载到训练/验证。
因此“训练未打开测试集”不表示目标构造阶段没有读取过测试注释。

`niche_target_receipt.json`、`niche_vocabulary.json`、`niche_geometry_extension.json` 保存半径、
词表、覆盖与几何证据，其 SHA256 与各目标数组一起纳入 manifest 校验。
对照包括仅由训练标签求得的全局组成均值，以及中心表达 PCA + ridge 组成探针。

## 整合：固定官方 scIB 子集

固定 `scib-metrics==0.6.0` 的官方实现，CPU JAX。生物信息项为 label ASW、isolated-label ASW、
KMeans NMI、KMeans ARI；批次项为类型内 batch ASW、iLISI、graph connectivity。
正式结果逐项报告这七个标准指标。内部选权重效用仍采用类内等权、`0.6 × biology + 0.4 × batch`，
字段保留旧名 `scib_composite` 以兼容搜索代码，角色明确为 `selection_only`。
负 ARI 仅在该内部效用中按零计，正式 ARI 保留原值。60/40 参考 scIB 的类别权重，
但所选子集和内部处理属于项目配置，因此不能称为原论文完整总分。

固定种子 20260915，按 label×batch 组合抽样最多 5,000 个细胞；抽样不读取候选表征。
近邻数上限 90，connectivity 上限 15，均含自身。不跨候选动态 min-max 缩放。
必要主指标未定义时显式报错，不靠丢弃指标调整权重。这是固定子集，不是原始 scIB 全部 14 项。

最终 evaluator 另报官方 BRAS 和按类型分别计算、再等权汇总的条件 iLISI（`cilisi`），
帮助检查高混合分是否掩盖生物结构损失。仅诊断项跳过样本不足的类型，并报告可评估类型数；
没有可评估类型时明确记录空值。验证选权重时省略这些额外诊断，七个标准单项实现完全相同。

`standard_metrics.csv` 与 `STANDARD_METRICS.md` 只列四任务的标准单项指标；
`selection_scores.csv` 单独保存 `scib_composite`、`biology_conservation`、`batch_removal` 项目汇总。
完整的 `benchmark_table.csv` 增加 `metric_role` 列（`benchmark` / `selection_only` / `diagnostic`）。
正式 leaderboard 对整合分别生成七张单项表，不再按项目综合分排一个整合总榜。
演化的最终报告同样将标准单项导出到 `final_standard_metrics_long.csv`，冻结后的测试表仅展示这些指标；
原始全部指标保留在 `final_metrics_long.csv`，搜索/晋级表明确标为内部选优目标。
代码中的 `PRIMARY_METRICS` 是内部 `SELECTION_METRICS` 的兼容别名，正式报告读取 `BENCHMARK_METRICS`；
协议 JSON 也明确区分 `primary_metrics`（科学结果）和 `selection_metrics`（选权重目标）。

## 发育：精确经验 W₂

在共同的训练归一化基因 panel 中，比较预测细胞群与目标细胞群的分布，不作行对行配对，
也不使用各模型自己的潜空间。每群最多 256 个细胞，固定种子 20260915。

运输代价为平方欧氏距离；等样本数用 SciPy 最优指派，不等样本数用 POT 的均匀质量精确 OT。
对最优平均运输代价开平方，得到经验 W₂。它在固定抽样群体上精确求解，不代表对全部原始细胞求解。
各来源/目标时间与解剖群体等权平均；正式配置验证全部群体。

保留 MMD、群体均值/方差误差、基因相关性及单独命名的 Sinkhorn divergence。
Sinkhorn 辅助项仍使用 `geomloss==0.3.1`、p=2、blur=0.05、scaling=0.9、debias=True，
其代价为 `0.5 * ||x-y||² / D`，数值单位和 W₂ 不同，不能互称。

本次保留原有全局数据划分，没有新增整个时间点留出实验。因此当前结果不自动证明未见时间点的
插值/外推能力；这需要另行设计时间留出评估。横截面来源/目标群体也不是同一细胞的真实配对轨迹。

## 演化与复现

入口配置：`configs/nanobrain_graph_v2/metrics_v3_20261102.json`，已设为演化脚本的默认配置。
保留 50 代、每代 12 候选、三 seed、每任务 10,000 steps，以及 `gpt-6.1-sol`、`reasoning_effort=max`。
旧协议和 anatomy 代理数据不能直接 resume 成 V3。新旧成绩不可直接横比。

搜索效用为 `mean(annotation F1, scIB, 1/(1+composition MAE), 1/(1+W₂))`，
同时记录原始指标、各任务退化约束与多 seed 确认。此效用变换是工程选择，不是论文通用分数。
LLM 接收四任务选优目标、标准单项及紧凑诊断，完整的每类型/切片细节留在结果文件。

这里的 `scIB` 指上文的项目整合选权重效用；四任务平均同样只是工程搜索规则，
不进入正式标准指标表，不能作为通行的科学指标与其他论文横比。

后续启动时使用新的空结果目录；下列命令本次没有执行：

```bash
cd /mnt/aliyunsb/zim_rider/nanobrain/try/metrics-v2-20261102
.venv/bin/python scripts/run_nanobrain_evolution.py \
  --config configs/nanobrain_graph_v2/metrics_v3_20261102.json \
  --output results/nanobrain/metrics-v3-20261102 --gpus 0,1,2,3,4,5,6,7
```

旧测试集已有公开成绩，也已在历史演化中使用；冻结后的测试分只能作描述性比较。
若要作新的独立泛化结论，仍需新的未参与搜索 holdout。此次未改动其他实验的源码、数据或运行进程。
环境为本 checkout 独立 `.venv`，通过路径复用已有依赖安装，源码解析优先指向本目录。

验证证据见 `METRICS_V3_VERIFICATION.json`：单元/集成测试、真实数据短训练、最佳权重重载复评、
数据同一性与 manifest 哈希检查。短训练只验证流程，不作为正式模型成绩。

## 文献依据与项目选择

| 论文 | 采用或参考的评估思路 |
|---|---|
| [scTab, Nature Communications 2024](https://www.nature.com/articles/s41467-024-51059-5) | 不平衡细胞类型注释的 Macro-F1 |
| [scIB, Nature Methods 2022](https://www.nature.com/articles/s41592-021-01336-8) | 同时衡量生物信息保留和批次去除 |
| [批次整合指标研究, Nature Biotechnology 2025](https://www.nature.com/articles/s41587-025-02743-4) | BRAS、条件 iLISI，检查 silhouette 与混合指标局限 |
| [Nicheformer, Nature Methods 2025](https://www.nature.com/articles/s41592-025-02814-z) | 中心表达预测邻域组成；组成 MAE、数量 MAE/R² |
| [Waddington-OT, Cell 2019](https://doi.org/10.1016/j.cell.2019.01.006) | 基于最优运输的细胞群体分布评估 |
| [TrajectoryNet, ICML 2020](https://proceedings.mlr.press/v119/tong20a.html) | 分布预测的 Earth Mover's Distance 评估 |
| [MIOFlow, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bfc03f077688d8885c0a9389d77616d0-Abstract-Conference.html) | Wasserstein 与 MMD 等群体分布指标 |

这些论文并非都使用 W₂，也未使用本项目完全相同的四任务协议。四尺度半径、真实计数比例、
scIB 子集与权重、固定抽样预算、经验 W₂ 求解和搜索效用，是本项目预先明确的评估选择。
整合单项调用 [scib-metrics 官方 API](https://scib-metrics.readthedocs.io/en/stable/api.html)，
版本固定为 0.6.0；不继续使用旧版近邻一致性冒名 `isolated_label_f1` 或简化 iLISI。
