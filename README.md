# nanoBrain：固定任务头，仅搜索表征主干与训练参数

本目录于 2026-10-30 从 `../test-driven-val-best-20261029` 独立复制。
代码、配置和结果目录独立；数据、参考源码、Python 环境和 API 配置复用原有链接。
不继承原实验的候选、权重、运行状态或成绩。正式 50 代实验已于 2026-10-30 18:15（北京时间）启动；运行记录见 runtime/active_run.json。

## 固定的任务适配结构

根据 nanoBrain 教程 `docs/components.md`、四任务教程，以及教程默认配置与组件实现：

| 用途 | 固定实现 |
| --- | --- |
| 细胞注释 `annotation` | `classification`：Linear(H, 类别数) |
| 空间生态位 `niche` | `classification`：Linear(H, 类别数) |
| 整合批次判别 `integration` | 梯度反转后使用 `classification` |
| 整合表达重建 `integration_reconstruction` | `regression`：LayerNorm(H) + Linear(H, 基因数) |
| 发育表达输出 `development` | `regression`：LayerNorm(H) + Linear(H, 基因数) |
| 可选表达补全 `completion` | 同上；不新增第五项正式评估任务 |
| 发育专用分支 | 默认 FMT，2 层、4 个注意力头；不探索流场拓扑或端点传输组件 |

所有任务头的 `params={}`，不允许额外 `pre` 适配计算图。任务头的结构固定，**权重仍正常参与各自任务训练**。
表征宽度改变时，头部输入维度和固定 FMT 的接口宽度自动适配；FMT 的层数、注意力头数保持固定。

教程来源位于 `/mnt/aliyunsb/zim_rider/nanobrain/nanobrain-github/docs/`；具体默认实现位于同项目 `src/nanobrain/modules.py`、`genome.py`。

## 允许搜索

- 表征主干的注册组件、拓扑、顺序、重复层数、分支/残差/拼接、组件参数及表征宽度。
- 学习率：`1e-3 / 5e-4 / 2e-4 / 1e-4`。
- 原先已开放的训练目标参数：回归损失类型、损失权重、标签平滑、对抗强度等；必须保留正权重的必需损失项。
- Sinkhorn 的 epsilon/iterations，以及 Euler/Heun 求解器和推演步数。

不开放优化器类型、训练 batch size、训练步数、种子、数据划分、评估器或晋级规则。正式预算继续使用每候选每任务 10,000 步、3 个配对种子、50 代、每代 12 个候选。

`search_scope=backbone_training_fixed_heads_v1` 同时约束提示词、JSON Patch 路径、完整 genome、缓存提案及训练入口。
候选必须通过任务头、FMT 分支、配对方法和目标族的固定性检查；不通过时保留拒绝原因并向 LLM 请求修正。
每代输入组件覆盖保留 harmony/scanorama/bbknn，移除与固定任务头冲突的 CellTypist 头覆盖要求。

沿用原来的验证集选择最佳权重、test 成绩反馈演化协议；它仍属于 test-driven 探索实验，不能当作未触碰测试集的泛化结果。

## 启动

```bash
cd /mnt/aliyunsb/zim_rider/nanobrain/try/backbone-train-fixed-heads-20261030
bash scripts/run_fixed_heads_evolution.sh --gpus 0,1,2,3,4,5,6,7
```

继续本副本同一实验时加 `--resume`。`run_val_best_evolution.sh` 在本副本中也转发到上述新入口。
正式配置：`configs/nanobrain_graph_v2/backbone_training_fixed_heads_50g_20261030.json`。
正式输出：`results/nanobrain/backbone-train-fixed-heads-20261030`。
验证记录：`runtime/fixed_heads_verification.json`；验证产生的短流程结果存于 `results/preflight-fixed-heads-20261030`，与正式结果分开。

## 验证结果

89 项搜索边界与相关回归检查通过；四任务均通过真实数据两步训练、验证集择优权重、权重重载和预测检查。上述预检未调用 LLM API；随后已按用户要求启动正式演化实验。

另有 2 项旧的多阶段演化测试在未修改的原目录中同样因 `KeyError: tasks` 失败，已从本次检查中排除；原版复核日志保存在 `runtime/original_legacy_test_baseline.log`。
