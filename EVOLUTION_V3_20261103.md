# V3：50 代测试驱动演化

本方案按用户 2026-11-03 的明确选择，复用旧测试驱动版的单阶段策略：验证集选每个任务的最佳权重，
测试集成绩反馈给 LLM 并选择下一代架构。测试结果属于参与搜索的描述性成绩，不是未触碰 holdout 的泛化结果。
本次只修改并启动新 V3 checkout；两条旧演化的源码、配置及进程保持原样。

配置：`configs/nanobrain_graph_v2/metrics_v3_test_driven_50g_20261103.json`。
结果：`results/nanobrain/metrics-v3-test-driven-50g-20261103`。
运行记录与日志：`runtime/v3-test-driven-20261103/active_run.json`、`evolution.log`。

| 项目 | 本次设置 |
|---|---|
| 代数 | 50；无改进也保留 incumbent 并继续 |
| 每代候选架构 | 6，原旧版为 12 |
| 每架构种子 | 3：202609190 / 202609191 / 202609192 |
| 任务 | annotation / integration / spatial_niche / development，各自独立权重 |
| 训练预算 | 每个任务、每个 seed 均为 10,000 steps |
| 每代训练次数 | 6 × 3 × 4 = 72，旧版为 144，减少 50% |
| 初始参考模型 | 同样预算、同样 3 个种子，共 12 次任务训练 |
| 验证间隔 | 1,000 steps，原 500；验证仍用完整规定指标，减少重复计算 |
| 权重保留 | validation-best 与 final weights；测试评估读取 best weights |
| LLM | gpt-6.1-sol，reasoning_effort=max |
| GPU 范围 | 仅物理 GPU 0 / 1 / 2 / 3 / 4；2026-11-03 按用户要求收紧 |
| 并行 | 每 GPU 最多 8 个任务，总计最多 40 个训练/评估作业 |
| 显存调度 | 按显存动态准入，训练预留 4 GiB、评估预留 8 GiB、额外保留 4 GiB余量；OOM 扩大预留后重试 |

所有候选直接使用最终训练预算和三个种子；不再额外进行短筛选、传播复训和末尾确认复训。
各架构用三个种子的平均成绩，同时满足相对父代和初始参考的改进/任务退化约束，才可晋级。
任意一个种子失败，该架构不得靠剩余成功种子的平均值晋级。
LLM 搜索范围沿用旧测试驱动版，允许调整主干、任务适配模块及训练参数；保留 V3 邻域组成任务约束。

训练、种子与评估任务进入同一个自适应队列，作业完成即补位。共享卡也按显存参与调度，
不再沿用“所有卡忙时整轮最多五个作业”的回退限制。总并发和显存预留均限制准入，实际并发随负载变化。
状态文件每十秒更新一次；代级摘要放入 campaign.json，完整候选和种子信息保留在各代 selection.json、各作业 result.json，
避免每秒重写随代数膨胀的大文件。已有训练回执和成功评估可在相同源码/数据/配置下复用恢复。

GPU 范围调整通过 runtime 启动参数生效，冻结的科学配置与源码指纹保持一致；原启动时的 48 作业上限由当前
40 作业上限覆盖，迁移记录写入 protocol.json 的 runtime_migrations 和 runtime/gpu-restriction-latest.json。
切换时接管了 0–4 号卡上 8 个仍在训练的进程、复用 2 个已完成结果；6 号卡的 2 个未完成训练已归档后从头重排。
5、7 号卡当时没有活跃 V3 任务。两条旧演化没有调整。

V3 的科学指标定义保持 [METRICS_V3.md](METRICS_V3.md) 的标准单项定义，内部选优综合分仍明确单独标注。
官方 scIB 单项、Macro-F1、邻域组成 MAE、W₂ 在每代测试成绩中记录；初始参考与当前所选模型的标准单项
另写入 `final_standard_metrics_long.csv`。旧 ARI anatomy 代理任务没有重新引入。

启动前核验包括完整测试、四任务真实数据 GPU/BF16 两步训练及测试复评、验证最佳权重重载，
以及 gpt-6.1-sol/max 的真实接口连通性。两步预检只用于核验管线，正式参考模型会重新从头训练到 10,000 steps。

启动命令使用 runtime 的凭据加载器，将原有 API 设置仅放入子进程环境，不将密钥写入运行记录：

```bash
cd /mnt/aliyunsb/zim_rider/nanobrain/try/metrics-v2-20261102
.venv/bin/python runtime/v3-test-driven-20261103/launch.py
# 同一源码、数据、配置下中断恢复：
.venv/bin/python runtime/v3-test-driven-20261103/launch.py --resume
```
