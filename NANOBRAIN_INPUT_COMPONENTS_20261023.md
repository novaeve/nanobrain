# Graph V2 输入组件扩展与 50 代演化

用户范围：加入 CellTypist、Harmony、Scanorama、BBKNN；不加入 XGBoost。提案模型 gpt-6.1-sol。

## 注册和执行

- `head/celltypist_linear`：逻辑回归的线性 decision function（系数和截距），随机初始化；复用统一训练器，不加载原模型或 atlas，不复制 CellTypist 训练策略。
- `operation/harmony`：官方 harmonypy.run_harmony 的 Z_corr。
- `operation/scanorama`：官方 scanorama.assemble 在固定表达特征上的 MNN 校正。
- `operation/bbknn`：固定 PCA（最多 32 维）后，官方 bbknn.matrix.bbknn 构建批次平衡图，随后用行归一化图做一次表达特征传播；不是完整 diffusion-map baseline。

后三者是数据集级、固定的输入组件。各自独立处理 train/validation 的表达和技术 batch ID，不读取注释、niche、年龄目标。计算结果按输入、batch、实现和依赖版本哈希缓存，逐 batch 按原行索引取回；空间邻居上下文同步取回。输出仍为 256 维。

它们仅允许在 backbone 中使用 inputs=["features"]、repeat=1，可通过 strength（0 到 1）混合原表达和组件结果，再接任意 encoder，也可做并行分支。它们不能用于学习到的 embedding、flow 或 head 子图。算法结果冻结，不伪造反向传播；后续编码器和任务头正常优化。基因表达重建和发育目标保留原始表达。

搜索不加载 test；最终冻结后 test 仅为历史已公开 test 的描述性复评。可选 masked completion 不允许使用这些全表达输入组件，防止目标泄漏。

## 演化协议

配置：`configs/nanobrain_graph_v2/evolution_components_50g_20261023.json`

- 50 代，每代 12 个 LLM 候选；初筛每任务 3000 steps。
- 四任务独立训练；晋级前 3 seeds；最终确认 10000 steps × 3 seeds。
- 原先 validation 选中架构 9b9d080756e228e9 作为新 reference，所有权重重新初始化和训练，不继承旧分数或权重。
- 每代候选池要求覆盖四个新组件；没有合法提案时按原有有界重试协议记账并保留 incumbent。
- 不改变评估指标、晋级阈值、数据划分。
- LLM 模型由配置 llm_model 覆盖，仅影响新实验，不修改 .env 的全局模型设置。

## 运行环境

原 .venv/bin/python 指向失效的 /mnt/.pm-agent 路径。已保留原链接为 python.before-20261023，并改指向 /mnt/aliyunsb/.pm-agent/default/data/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12。原依赖目录和 .pth 桥接保留。

修改前源码备份：runtime/extension-backup-20261023/before.tgz。
真实数据预检：results/nanobrain/components-50g-preflight-pca-20261023。最初直接在 256 维做精确构图的预检因耗时已主动停止，保留原目录作为诊断记录。
缓存：runtime/integration_components/，每份 npy 配有哈希校验 json。

## 验证与启动结果

- 最终完整测试：244 passed（含新增 7 项）。日志：runtime/extension-backup-20261023/tests-final.log。
- 真实 100k 数据四任务各 2 steps，四任务均实际更新参数、梯度审计及稳定性通过，canonical validation evaluator 全部完成，test_opened=false。
- 模型 gpt-6.1-sol 的真实 API 请求返回成功。
- 正式实验已启动，controller PID 1115071，GPU 0,1,2,3。
- 输出：results/nanobrain/llm-graph-v2-components-50g-gpt61sol-20261023。
- 日志：runtime/nanobrain/components-50g-gpt61sol-20261023.log。
- 启动记录：runtime/nanobrain/components-50g-gpt61sol-20261023-launch.json。
- 恢复时先确认原 controller 已退出，再使用相同 config/output 加 --resume；源码或依赖指纹改变时不要强制续写。
