# Test-driven single-stage evolution copy

创建日期：2026-10-27。

该目录基于 Graph V2 input-components 50代实验的源码和 reference genome，保留 60,000/20,000/20,000 划分。现策略是：

- 每个候选每个任务直接训练 10,000 steps；
- 每个候选使用 3 个配对 seed；
- 取消 3,000-step screen、短预算 propagation 和末尾 confirmation；
- 每代训练完后直接调用 canonical evaluator 的 test split；
- 四任务 test 指标均值用于综合分、逐任务回归门槛、父代晋级和下一代 LLM prompt；
- 50 代、每代 12 个候选；
- `test_used_for_selection=true`，这是用户指定的 test-driven exploratory search。结果不应作为 untouched holdout 泛化报告。

配置：`configs/nanobrain_graph_v2/test_driven_single_stage_50g_20261027.json`
运行器：`scripts/run_nanobrain_evolution.py`
短路径预检：`results/smoke-test-driven-20261027`，四任务真实数据 2-step 训练和 test-selection evaluator 路径通过。

## LLM 看到什么

当前 prompt 的 `parents`、`initial_reference`、`recent_generations` 和 `best_observed_candidates` 都包含逐任务 `metrics`、`diagnostics`、综合 `score` 以及 gate 信息。单阶段副本把这些字段填成 test split 的指标，因此 LLM 会看到每个任务的 test performance，并基于它们提出下一代架构。

原 50 代实验则只把 validation 指标放入 prompt；它的 test 只在冻结选择后做描述性复评。原目录保持不变。
