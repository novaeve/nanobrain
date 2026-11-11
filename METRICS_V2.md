# 2026-11-02：四任务指标协议 V2

> 历史协议，已由 [METRICS_V3.md](METRICS_V3.md) 取代。V2 的 anatomy 分类不等同于蓝图的 niche 预测；
> 当前空间任务已改为真实邻域组成回归，发育主指标已改为精确经验 W₂。本文及 V2 验证回执仅保留历史记录，勿用于新演化。

本目录从 https://github.com/novaeve/nanobrain 克隆，起点提交为
`b5e6bbf01f8b9f14ec63de873f1fa950c079738c`，分支 `metrics-v2-20261102`。
服务器目录：`/mnt/aliyunsb/zim_rider/nanobrain/try/metrics-v2-20261102`。
协议 ID：`NANOBRAIN-METRICS-V2-20261102`。

## 主指标及权重选择

| 任务键 | 科学任务 | 主指标 | 方向 |
|---|---|---|---|
| annotation | 细胞类型分类 | f1_macro | 越大越好 |
| integration | 批次整合与生物信息保留 | scib_composite | 越大越好 |
| spatial_niche | 有监督解剖区域分类（anatomy 代理标签） | f1_macro | 越大越好 |
| development | 给定来源群体和时间的目标细胞分布预测 | sinkhorn_divergence | 越小越好 |

每个任务独立训练。每次验证使用对应的同一个主指标选择最佳权重；验证集也负责架构选择。
`best_weights` 策略保存验证最佳权重和最终权重，不保存每次验证的优化器快照。
演化评分、复评及冻结后的测试读取 `best_checkpoint`。新协议不接受旧指标表或旧实验的 resume。

分类 Macro-F1 仅对本评估集有真实样本的已标注类别作等权平均，类别集合不随预测改变。
保留 Accuracy、Balanced Accuracy、MCC 和其他分类诊断。未知真实标签继续通过原任务 mask 排除；
这仍是闭集注释评估，没有新增未知类型拒识任务。

## 整合：固定的官方 scIB 指标子集

调用 `scib-metrics==0.6.0` 的官方实现，使用 CPU JAX，避免额外占用训练 GPU。

- 生物信息：label ASW、isolated-label ASW、KMeans NMI、KMeans ARI（综合分中负 ARI 按 0 计，原始 ARI 单独报告）。
- 批次去除：按细胞类型计算的 batch ASW、iLISI、graph connectivity。
- 两类内各指标等权平均，综合分 = 0.6 × 生物信息 + 0.4 × 批次去除。
- iLISI 使用最多 90 个近邻，graph connectivity 使用最多 15 个近邻，包含自身。
- 最多 5,000 个细胞。按固定标签×批次组合保留至少一个样本，再固定随机抽取剩余名额；
  种子 20260915，抽样不读取候选嵌入。所有候选使用相同的评估行。
- 指标缺失、完全混杂而无法评估批次去除等情况明确报错，不通过丢弃某一项改变权重。
- 不在不同候选之间动态做 min-max 归一化；不沿用旧的 `integration_composite` 名称。

这是预先固定的官方指标子集，未声称复刻原始 scIB 的全部 14 项；未加入 kBET、PCR 等项目。
旧实现的邻域纯度不再以 `isolated_label_f1` 的名称输出。

## 空间：区域分类与聚类诊断分别评分

沿用 `spatial_niche` 程序键以兼容模型与数据接口，但它的实际标签来自 anatomy，不能称为真实生态位标签。
主指标 Macro-F1 必须读取有语义的 `y_pred`。nanoBrain 使用区域任务头，三个经典空间控制使用
仅由训练标签拟合的线性分类头；聚类编号不能直接当区域类别，也不能利用验证/测试标签拟合编号映射。

ARI、NMI 单独评估嵌入的无监督聚类分区；有 section 元数据时补充有效切片的等权平均。
只有聚类输出的外部适配器仍可报告 ARI/NMI，但不产生区域分类 F1、不参与这项主指标的比较。
移除原来对任意第一嵌入坐标计算的 Moran/Geary 等诊断，避免把它当成空间域质量。

## 发育：固定参数的去偏 Sinkhorn divergence

使用 `geomloss==0.3.1` 的 `SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9,
debias=True, backend="tensorized")`，CPU float64。
所有方法在同一训练集归一化的基因 panel 中计算；各特征除以维度平方根，因此运输代价为
`0.5 * ||x-y||² / D`。不使用各模型自己的潜空间，也不使用测试数据重新拟合缩放或 PCA。

每个群体最多 256 个细胞，固定抽样种子 20260915；按来源/目标时间与解剖组计算后等权平均。
正式新配置验证全部群体。辅助保留群体均值 MSE、方差误差、均值表达相关性与 MMD；
MMD 使用共享空间下固定 RBF 带宽，避免候选通过影响带宽掩盖误差。
该主指标是正则化平方代价的 Sinkhorn divergence，不是精确 Wasserstein 距离。

## 数据、结果与运行

不改变原训练/验证/测试划分。新配置直接引用已有物化数据，不复制数据，也不改数据文件。
该历史测试集已公开，且此前参与其他演化，冻结后的复评仅为描述性结果；真正独立的泛化结论
仍需另有未参与搜索的新 holdout。本次只完成代码与检查，未自动启动 50 代演化或重训基线。

新配置：`configs/nanobrain_graph_v2/metrics_v2_20261102.json`。
保留 50 代、每代 12 个候选、三 seed、每任务 10,000 steps 的预算，LLM 为
`gpt-6.1-sol`，`reasoning_effort=max`。结果必须写入新的空目录。
旧配置与旧复评脚本是历史材料，应使用这个新入口配置。

```bash
cd /mnt/aliyunsb/zim_rider/nanobrain/try/metrics-v2-20261102
# 后续明确启动时使用；需在进程环境提供已有 API 配置。
.venv/bin/python scripts/run_nanobrain_evolution.py \
  --config configs/nanobrain_graph_v2/metrics_v2_20261102.json \
  --output results/nanobrain/metrics-v2-20261102 --gpus 0,1,2,3,4,5,6,7
```

服务器 `.venv` 是独立环境，通过只读依赖路径复用已有安装；本项目源码优先。
它不会通过修改旧实验的环境来安装依赖。新机器可按 `pyproject.toml` 安装 `cpu,benchmark,dev` extras。
新 CSV、权重、训练回执、指标回执和实验指纹记录本指标协议。
旧预测可用新 evaluator 作附加复评，但没有保存的最佳 epoch 无法事后恢复；
缺少区域类别预测的旧聚类文件也无法凭聚类编号补出有监督 F1。

综合分仍使用四任务等权的固定工程效用：
`mean(F1_annotation, scIB, F1_region, 1/(1+Sinkhorn))`。
它仅用于本协议内的搜索，不能与旧综合分横比；原始四项指标始终单独报告。

## 文献依据

- scTab, Nature Communications 2024：https://doi.org/10.1038/s41467-024-51059-5
- scIB, Nature Methods 2022：https://doi.org/10.1038/s41592-021-01336-8
- STAGATE, Nature Communications 2022：https://doi.org/10.1038/s41467-022-29439-6
- scNODE, Bioinformatics 2024：https://doi.org/10.1093/bioinformatics/btae393

主指标组合、固定抽样预算、Sinkhorn 参数及效用归一化是本项目的工程选择；
不声称上述论文使用了完全相同的四任务协议。
