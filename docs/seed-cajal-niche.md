# Cajal v3 的空间 Niche 预测

Cajal v3 默认运行五项联合任务。选择 `campaigns/cajal-v3-niche` 后，每个候选从随机初始化开始，以同一个共享模型联合训练原五项任务及 Niche。训练仍使用 `unified_1m`、seed 0 和 30,000 步学习率日程，在 10,000 步固定停止。[默认 v3 协议](seed-cajal.md) · [Niche campaign 配置](../campaigns/cajal-v3-niche/campaign.toml)

## 输入、目标和评估

模型用中心细胞的基因表达生成潜在均值，再通过线性头和 softmax 输出 167 维细胞类型比例。训练目标由同一组织切片内、排除中心的 16 个最近邻的细胞类型标签计数得出。坐标用于定义近邻，邻居标签用于计算监督目标。Niche 预测头只读取中心细胞的潜在均值。

原六项历史指标继续使用 Cajal v3 的已有分数。新增的 **Niche composition MAE** 对所有评估中心和 167 个类型维度取预测比例的平均绝对误差，越低越好。Niche 的对照是用 `unified_1m` 中有完整已注释邻居的中心计算平均组成比例，对每个评估中心输出同一向量。该分数单独记为训练集先验。[原 v3 参考记录](../reference/existing_cajal_v3_10k.json) · [Niche 先验记录](../reference/cajal_v3_niche_prior.json)

Niche 在现有 `eval_spatial` 的 40,789 个空间中心评分。评估使用与预训练无重叠的空间块和细胞。评估切片及供体在预训练中出现过。分数衡量同一队列内留出空间块的预测性能。[空间评估集构建](../data_build/build_eval_set.py)

可用同一份训练集和评估集复算先验参考分数。[先验构建脚本](../scripts/build_niche_reference.py)

```bash
python -B scripts/build_niche_reference.py \
  --train /data/brainwm/unified_1m \
  --spatial /data/brainwm/eval_spatial \
  --out /tmp/cajal_v3_niche_prior.json
```

## 选择设备并运行

使用与默认 v3 相同的 Linux 环境、数据挂载和设备依赖。先选择 campaign 检查计划，按实际环境选用 CPU 或 CUDA。

```bash
export NANOBRAIN_CAMPAIGN=campaigns/cajal-v3-niche
python -B scripts/cajal_react.py --device cpu
python -B scripts/cajal_react.py --device cuda
```

输入和依赖就绪后，使用以下命令冻结协议，再选一条命令启动演化。两种设备模式使用相同的评估指标。

```bash
python -B scripts/cajal_react.py --freeze
python -B scripts/cajal_react.py --execute --device cuda
# 或在 CPU 上运行
python -B scripts/cajal_react.py --execute --device cpu
```

CUDA 输出默认写入 `research-runs/cajal-v3-niche-react/`，CPU 输出默认写入 `research-runs/cajal-v3-niche-react-cpu/`。续跑时沿用原设备模式和输出目录。使用 `unset NANOBRAIN_CAMPAIGN` 即可恢复默认选择。
