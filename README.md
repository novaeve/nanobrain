# Graph V3 测试驱动演化：验证集选择最佳权重

创建日期：2026-10-29。目录：`/mnt/aliyunsb/zim_rider/nanobrain/try/test-driven-val-best-20261029`。

从 `brain-test-evolution-20261027` 复制源码、配置和测试，独立保存新结果；数据、参考组件、API 配置复用原目录的链接；Python 环境与新增组件实现复用 Graph V3 组件库。

## 当前流程

1. 沿用同一份数据及其固定 60,000 / 20,000 / 20,000 train / validation / test 划分。
2. 初始基线及每个候选结构，均对四个任务独立从头训练；每任务使用三个配对种子 `202609190/191/192`，每次完整训练 10,000 steps。
3. 每 500 steps 在 validation 上评估。按训练器已有 validation quality 选择每任务、每 seed 的最佳权重；分数相同保留较早的权重。具体标准为：annotation accuracy 最大、niche accuracy 最大、integration objective 最小、development population mean-expression MSE 最小。
4. 训练器只加载 train/validation；训练完成后，canonical evaluator 才对选好的 `*-best.pt` 权重计算 test 指标。这些 test 指标是该结构、该 seed 的最终成绩。
5. 将三个 seed 的逐任务 test 指标取平均，计算四任务等权综合分；沿用原有综合提升和逐任务退化门槛，与初始基线及当前父代比较。LLM 根据这些 test 反馈提出下一代。每代 12 个候选，共 50 代，无额外短预算筛选和末尾确认。

测试集持续参与候选结构选择；这里的修改只把训练权重的选择改成依赖 validation。

## 保存与失败处理

- 每任务每 seed 保留最佳 validation 权重和最终 step 权重，不保存每次评估的优化器快照。
- 训练回执记录 `best_step`、`best_validation`、`best_validation_quality`；任务结果和 checkpoint bundle 记录实际选择的权重及其哈希。
- 任一 seed 的训练失败或缺少主指标，则整个候选不晋级，保留失败原因，继续处理其他候选；初始基线失败则明确退出。
- 运行状态见 runtime/active_run.json 和正式结果目录的 campaign.json；未复制旧版运行结果。

## 运行

```bash
cd /mnt/aliyunsb/zim_rider/nanobrain/try/test-driven-val-best-20261029
bash scripts/run_val_best_evolution.sh --gpus 0,1,2,3,4,5,6,7
```

配置：`configs/nanobrain_graph_v2/test_driven_val_best_50g_20261029.json`。
正式结果：`results/nanobrain/test-driven-val-best-20261029`。
同一份代码、数据与配置继续执行时，在上述命令后增加 `--resume`。


## 本次 50 代实验

- 模型固定 gpt-6.1-sol，每次提案请求显式发送 reasoning_effort=max；当前网关连通测试已接受该参数并返回相同模型。
- 新增 12 个方法的 17 个结构组件已接入 Graph V3 的 registry、张量契约、训练器和 LLM graph catalog；已有 CellTypist/Harmony/Scanorama/BBKNN 保留。
- 组件来源、逐文件哈希和可搜索条目见 runtime/component_integration_receipt.json；代码和上游源码会随正式实验冻结。
- 新增方法：scTab、STAGATE、scPoli、MIOFlow、PRESCIENT、scNym、SEDR、SpaGCN、scFoundation、scPRINT、scBERT、CellOT。
- 相关检查：101 passed。真实数据短流程预检结果见 results/preflight-v3-val-best-20261029-v2。

## 2026-10-30 调度优化

正式任务已使用 --resume --adaptive-queue 续跑。各任务的训练和评估共用队列，工作完成即补位，不等待整批或整项任务结束。优先选用外部 GPU 显存占用不超过 16 GiB 的卡，每张低负载卡最多并行两个进程；所有卡均有外部重负载时，优先选外部负载较低的卡。

原有 116 份任务结果保留，第 0 代复用 104 份结果和 3 份已训练权重，并接管原来尚未完成的一个训练进程。模型、数据、训练预算、随机种子、验证选权重与测试评分规则均保持原样。源代码变更仅涉及两个调度文件，旧协议和代码已归档到结果目录的 runtime_migrations。

验证记录：runtime/scheduler_migration_receipt.json。动态队列日志：results/nanobrain/test-driven-val-best-20261029/scheduler_events.jsonl。运行状态见 campaign.json 的 scheduler 和 active_runs。

## 2026-10-30 并行上限调整

按用户要求，低负载 GPU 的并行上限由 2 调为 5。正式进程和启动脚本均已使用 --workers-per-gpu 5。已保存 155 份任务结果并接管未完成的训练进程；模型、种子、步数、数据划分和评分规则保持原样。验证记录：runtime/parallelism_5_receipt.json。

## 2026-11-02 显存中断恢复

第 5 代曾于北京时间 2026-11-01 02:32 因候选评估 OOM 导致协调器退出。已保留 638 份任务记录并复用已训练权重恢复。调度器加入训练/评估显存预留、评估失败重试和已完成评估复用；所有卡均被外部任务重度占用时，总并发上限为 5。评估 OOM 不作为模型性能失败，保留权重并等待显存。数值模型、数据、训练预算、种子和评分阈值保持原样。验证：31 passed；原出错权重使用原 batch size=2048 重评成功。记录：runtime/oom_recovery_receipt.json。
