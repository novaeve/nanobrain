# nanoBrain Graph V3 组件库

更新时间：2026-10-27。

该目录是 Graph V2 的隔离扩展版本，注册了 12 个官方 baseline 的 17 个结构组件：

- scTab：TabNet encoder、GLU layer
- STAGATE：GAT spatial layer
- scPoli：conditional encoder、prototype head
- MIOFlow：ODE velocity field
- PRESCIENT：potential-gradient velocity field
- scNym：MLP encoder、residual block
- SEDR：expression block、GCN spatial layer
- SpaGCN：GCN spatial layer
- scFoundation：AutoBin token encoder
- scPRINT：nonnegative value encoder、classification decoder
- scBERT：Performer attention
- CellOT：ICNN endpoint transport

这些是固定 upstream commit 的组件适配，不是完整方法复现；不加载预训练权重，也不移植原方法训练策略。scPoli prototype 只使用训练集状态并写入 checkpoint；scPRINT 明确要求非负表达输入；CellOT 是 endpoint 组件而非 velocity 节点；PRESCIENT 需要局部 autograd 求势函数梯度。

输入契约新增 Graph V3：`batch_id[B,1]` 技术批次条件和 `nonnegative_features[B,G]` 冻结归一化输入。endpoint 组件要求 `architecture.version=3` 和 `endpoint_transport` objective。

## 验证

- 17 个条目专项梯度、重载和训练器验证：37 passed。
- 原工程完整回归测试：281 passed。
- 目录：`component_library_receipt.json`。
- 候选来源和源码：`candidate_components.json`、`upstream_source_manifest.json`、`upstream_components/`。

## 与正在运行的 50 代演化

正在运行的旧实验仍使用其冻结的 Graph V2 源码指纹，没有被这个目录修改。旧实验目录为：
`/mnt/aliyunsb/zim_rider/nanobrain/brain/results/nanobrain/llm-graph-v2-components-50g-gpt61sol-20261023`。

要使用这些新增组件，需要在本目录的新实验中建立 Graph V3 reference、重新做短训练和相同预算比较。
