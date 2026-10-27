# nanoBrain：正式 LLM 模块演化

本页记录历史联合训练版 V1。根据最新用户确认，当前默认入口已改为任务独立训练版 V2，见 [当前流程与原计划核对](NANOBRAIN_INDEPENDENT_EVOLUTION.md)。历史实验结果和源码快照保留；以下配置和数字只描述 V1。

工程支持在固定 100k 数据上进行模块组合搜索、多 seed 自动晋级、较长预算确认和结果记录，不扩展全量数据。

配置为 `configs/nanobrain_evolution_100k.json`。实验目录是 `results/nanobrain/llm-evolution-20260917-v1/`；状态在 `campaign.json`，结果读取 `REPORT.md` 和 `completion_receipt.json`。

## 2026-09-17 实测完成情况

本轮已结束。最多 5 代的预算实际执行了 1 代、4 个真实 LLM 提案；最好的 validation 综合分提高 0.000802，低于预设 0.002 门槛，按蓝图停止传播并保留原模型。候选未触发多 seed 晋级验证。原模型完成了 3 × 600-step 短预算和 3 × 2,000-step 较长预算训练；加上 4 个候选，共完成 10 次真实训练。

较长预算三 seed 的五项评估产出 1,086 条 validation / test 复评指标，错误行 0。多代自动晋级路径另由模拟指标测试覆盖；本轮没有实际发生多代传播或模块种类替换。详见 [中文结果摘要](../nanobrain_evolution_results.md)。

历史 `campaign.json:test_reevaluation` 的部分摘要字段继承自 validation，独立 test 表与 `REPORT.md` 正确。后续代码已经修正，原始归档保持不变，正确 test 摘要与文件哈希见实验目录的 `post_run_artifact_audit.json`。

## 固定协议

| 项目 | 设定 |
| --- | --- |
| 数据 | 60,000 train / 20,000 validation / 20,000 test，256-gene panel，原 V3 划分 |
| 搜索预算 | 最多 5 代，每代 4 个 LLM 提案，每个 600 optimizer steps |
| 传播验证 | 每代初筛前 2 个合格候选补齐 3 个 seed；只有复现通过者可成为下一代父候选 |
| 最终确认 | 至多 2 个候选，各自 3 个新 seed × 2,000 steps；原始 FMT 同预算、同 seed 重训 |
| 优化器 | AdamW，BF16，EMA-equal 多任务损失平衡 |
| 主 batch | 256 cells；发育 population 每端最多 64 cells |
| 候选评价 | 固定最终 step checkpoint，五个统一 evaluator；训练中 proxy quality 只作诊断 |
| 评估抽样 | 所有候选和训练 seed 共用 population evaluation seed 20260915 |
| LLM 接口 | 用户 `.env` 中的第三方模型；失败持续重试，未成功不以本地变异冒充 LLM 提案 |
| 资源 | 默认 GPU 3、5、6、1，每进程 2 CPU threads；支持显式选择最多 8 张卡 |
| 历史 test | 已在 2026-09-16 公开；本轮选择仅使用 validation，冻结后 test 作为描述性复评 |

五个 evaluator 是注释、整合、空间 niche、空间补全和发育预测；仍对应四个高层科研任务。

综合指标固定为：

\[
Q=\frac15\left[\mathrm{Accuracy}+\mathrm{IntegrationComposite}
+\frac{\mathrm{ARI}+1}{2}+\frac{1}{1+\mathrm{RMSE}_{completion}}
+\frac{1}{1+\mathrm{MSE}_{population}}\right].
\]

候选需要比父候选以及原始参考模型提高至少 0.002。三个向上指标每项允许最多下降 0.02；两个误差指标允许最多增加 5%。多 seed 确认要求平均指标通过、至少两个配对 seed 的综合分提高、每个 seed 都未超过任务退步上限。最强单任务 baseline 继续作为对照，不作为每代必须全面超过的门槛。

不通过条件的候选保留记录，不传播为父候选；如果一代没有可传播方案，停止并保留上一代。完成有限预算不等于一定发现优于参考模型的新方法，`scientific_promotion_passed` 单独记录性能晋级是否成立。

## LLM 实际修改什么

LLM 组合已经注册的 encoder、attention、spatial、residual、pairing、flow 模块，并调整宽度、深度、头数、dropout、学习率、Sinkhorn 和 flow 参数。所有任务同时训练。本轮要求保留活动的 MLP flow 或 FMT。

每份提案包含真实父候选 ID、目标任务、改动假设、预期代价和完整 genome。程序检查参数范围、父子关系、是否重复，然后执行训练。多目标 Pareto 筛选只作用于通过改进与退步检查的候选。每一代的原始提示词、LLM 返回文本、重试原因、接纳结果和筛选结果均保存。

当前 compact encoder 先把 256 个基因映射为一个细胞向量，主 attention 接收长度为 1 的序列；FMT 的 attention 则处理整个细胞 population。前者不能宣称已经建模了基因 token 间的注意力。这一实现范围已写入 LLM 提示词。

## 运行与恢复

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/run_nanobrain_evolution.py \
  --config configs/nanobrain_evolution_100k.json \
  --gpus 3,5,6,1 \
  --output results/nanobrain/llm-evolution-20260917-v1
```

意外中断后，在代码、数据、配置完全相同时追加 `--resume`。已完成候选根据 receipt 和 checkpoint/table SHA-256 复用，未完成候选重新执行该固定预算。代码或协议变化应使用新的输出目录；既有实验不覆盖。

每个实验保存启动前的代码快照和数据 manifest 指纹；保存按 checkpoint 记录的训练/validation 日志。最终输出 search history、三 seed 指标、冻结的选择记录、test 描述性复评和完成 receipt。原来的 2026-09-16 固定 FMT benchmark 继续作为历史结果，不标注为自动演化冠军。
