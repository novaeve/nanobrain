# nanoBrain 新 baseline 与可注册组件清单

收集日期：2026-10-26。依据当前 Graph V2 的四任务、256 基因输入、一跳空间上下文和 state/context/time 速度场接口筛选。

已核对 **12 个官方仓库、108 份源码/元数据文件、17 个拟注册条目**。所有源码固定到提交 SHA，并记录 SHA-256。

**本清单中的名称均为拟注册标识，尚未加入运行中的注册表，也未做本地性能排名。源码已核实不等于已完成运行兼容验证。**

## 建议顺序

优先评估 scTab、STAGATE、scPoli、MIOFlow、PRESCIENT、scNym。这里的优先级是依据当前结构缺口、接口适配和重复度作出的工程判断，并非论文排名或已经测得的性能排序。

| 方法 | 适合任务 | 拟拆模块 | 接入成本 | 优先级 |
| --- | --- | --- | --- | --- |
| [scTab](https://github.com/theislab/scTab) | annotation, integration | `sctab_tabnet`、`sctab_glu` | 中 | 优先 |
| [STAGATE](https://github.com/RucDongLab/STAGATE_pyG) | spatial_niche, annotation | `stagate_gat` | 中 | 优先 |
| [scPoli](https://github.com/theislab/scarches) | annotation, integration | `scpoli_conditional`、`scpoli_prototype` | 中到高 | 优先 |
| [MIOFlow](https://github.com/KrishnaswamyLab/MIOFlow) | development | `mioflow_ode` | 中 | 优先 |
| [PRESCIENT](https://github.com/gifford-lab/prescient) | development | `prescient_potential` | 中到高 | 优先 |
| [scNym](https://github.com/calico/scnym) | annotation, integration | `scnym_mlp`、`scnym_resblock` | 低到中 | 优先 |
| [SEDR](https://github.com/JinmiaoChenLab/SEDR) | spatial_niche, integration | `sedr_mlp`、`sedr_gcn` | 中 | 第二批 |
| [SpaGCN](https://github.com/jianhuupenn/SpaGCN) | spatial_niche | `spagcn_gcn` | 中 | 第二批 |
| [scFoundation](https://github.com/biomap-research/scFoundation) | annotation, integration | `scfoundation_autobin` | 中 | 第二批 |
| [scPRINT](https://github.com/cantinilab/scPRINT) | annotation, integration | `scprint_value`、`scprint_classifier` | 中 | 第二批 |
| [scBERT](https://github.com/TencentAILabHealthcare/scBERT) | annotation, integration | `scbert_performer` | 低到中 | 第二批 |
| [CellOT](https://github.com/bunnech/cellot) | development | `cellot_icnn_transport` | 高 | 需先扩展接口 |

## 已有组件与筛选边界

现有注册表已含 scBrain/scFiT/scNODE/scVI、GraphST、TrajectoryNet、scUniFlow、BERT/通用 Transformer、FMT，以及新加入的 CellTypist/Harmony/Scanorama/BBKNN。本清单优先寻找新增结构能力，不重复列这些已注册条目。XGBoost 沿用用户决定，暂不加入。

当前搜索空间的薄弱方向是：基因特征选择、条件/原型表征、可训练空间注意力，以及势函数和端点运输等动力学结构。不同论文在各自数据上的成绩不能作为当前脑 100k benchmark 的直接排名。

以下输入输出和适配建议来自官方代码阅读及当前工程接口对照；没有移植原方法的损失、优化策略或预训练权重。

## 1. scTab

适合任务：annotation, integration。建议：优先；接入成本：中。

对基因特征做逐步选择的 TabNet，不是再增加一个普通 Transformer。

官方仓库：[scTab](https://github.com/theislab/scTab)；固定版本：`5ede7f2ba1f9618b86924f2ff587931de18f4ada`（提交日期 `2024-09-13T12:55:09+02:00`）。

源码许可：MIT。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://doi.org/10.1038/s41467-024-51059-5)。

### 拟注册 `encoder/sctab_tabnet`

源码：[`cellnet/tabnet/tab_network.py::TabNetEncoder`](https://github.com/theislab/scTab/blob/5ede7f2ba1f9618b86924f2ff587931de18f4ada/cellnet/tabnet/tab_network.py#L39)。

接口：`features[B,G] -> steps(list of [B,n_d]), mask_loss；适配后求和为 state[B,H]`。

适配工作：n_d 对齐 hidden_dim；按原 TabNet 聚合各 decision step；显式处理辅助返回值，不改变现有 loss。Ghost BatchNorm 要测试小批次。

### 拟注册 `operation/sctab_glu`

源码：[`cellnet/tabnet/tab_network.py::GLU_Layer`](https://github.com/theislab/scTab/blob/5ede7f2ba1f9618b86924f2ff587931de18f4ada/cellnet/tabnet/tab_network.py#L470)。

接口：`state[B,H] -> state[B,H]（选择相同输入/输出宽度）`。

适配工作：保留源实现 GLU 和归一化，映射参数名；shared_fc 不允许从 JSON 执行任意对象。

依赖：torch；numpy；官方 cellnet.tabnet.sparsemax。

范围说明：只引用 cellnet/tabnet 核心文件，不必引入 Merlin 全量数据加载栈。复用结构不代表继承论文大规模训练成绩。

## 2. STAGATE

适合任务：spatial_niche, annotation。建议：优先；接入成本：中。

以可训练邻居注意力替代当前 neighbor_mean，增加真正不同的空间聚合方式。

官方仓库：[STAGATE](https://github.com/RucDongLab/STAGATE_pyG)；固定版本：`ae1158ca8cf1eb6bb8ee198298552d44c9ac21db`（提交日期 `2023-11-10T10:49:42+08:00`）。

源码许可：MIT。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://doi.org/10.1038/s41467-022-29439-6)。

### 拟注册 `spatial/stagate_gat`

源码：[`STAGATE_pyG/gat_conv.py::GATConv`](https://github.com/RucDongLab/STAGATE_pyG/blob/ae1158ca8cf1eb6bb8ee198298552d44c9ac21db/STAGATE_pyG/gat_conv.py#L17)。

接口：`state[N,H], edge_index[2,E] -> state[N,H]`。

适配工作：由现有一跳 neighbor/mask 构造 edge_index，包含上下文行；先限定 heads=1 并保持宽度；无有效邻居保持原精确 identity 退化。

依赖：torch；torch_geometric；torch_sparse。

范围说明：官方自定义 GATConv 不是直接等同于 PyG 同名标准类。源码是旧 PyG 接口，需检查当前 torch/PyG 版本。仅拆一层，不声称完整 STAGATE 多层自编码器已复现。

## 3. scPoli

适合任务：annotation, integration。建议：优先；接入成本：中到高。

增加技术条件嵌入和原型距离分类，区别于现有普通 FC 编码器/线性头。

官方仓库：[scPoli](https://github.com/theislab/scarches)；固定版本：`d35cde61fff4dec6b86660817f4750fa35ff836d`（提交日期 `2026-06-26T11:03:19+02:00`）。

源码许可：BSD-3-Clause。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://doi.org/10.1038/s41592-023-02035-2)。

### 拟注册 `encoder/scpoli_conditional`

源码：[`scarches/models/scpoli/scpoli.py::Encoder`](https://github.com/theislab/scarches/blob/d35cde61fff4dec6b86660817f4750fa35ff836d/scarches/models/scpoli/scpoli.py#L363)。

接口：`features[B,G], technical_condition[B,K] -> mean[B,H], log_var[B,H]`。

适配工作：条件 nn.Embedding 在 scpoli 主类中，需要把条件嵌入与 Encoder 一同包装；声明用 mean 还是采样。当前有符号标准化输入应走 MSE 兼容分支，不能默认 log1p。

### 拟注册 `head/scpoli_prototype`

源码：[`scarches/models/scpoli/scpoli.py::scpoli.classify`](https://github.com/theislab/scarches/blob/d35cde61fff4dec6b86660817f4750fa35ff836d/scarches/models/scpoli/scpoli.py#L270)。

接口：`state[B,H], train-fitted prototypes[C,H] -> distances[B,C] / probabilities[B,C]`。

适配工作：转换为分类头约定的分数并持久化原型。源实现依赖训练阶段的原型状态，需新增只由 train 构建/更新原型的生命周期，不可使用 validation/test 标签。

依赖：torch；numpy；scarches 模型内部工具；完整包还会引入 scvi 等。

范围说明：prototype 不是任意初始化的线性层。整合任务不能把 cell-type 原型监督引入原先禁止生物标签监督的任务中。

## 4. MIOFlow

适合任务：development。建议：优先；接入成本：中。

显式时间输入的 Neural ODE 漂移网络；可与当前 flow 系列做结构比较。

官方仓库：[MIOFlow](https://github.com/KrishnaswamyLab/MIOFlow)；固定版本：`4e79ada3ce96feeb3f51ba02c7d999e095eaf7e8`（提交日期 `2026-10-08T12:26:40-04:00`）。

源码许可：Yale Non-Commercial License（以快照 LICENSE.md 为准）。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://github.com/KrishnaswamyLab/MIOFlow)。

### 拟注册 `flow/mioflow_ode`

源码：[`mioflow/core/models/ode_model.py::ODEFunc`](https://github.com/KrishnaswamyLab/MIOFlow/blob/4e79ada3ce96feeb3f51ba02c7d999e095eaf7e8/mioflow/core/models/ode_model.py#L5)。

接口：`原始 forward(t,x)：t 为标量或可扩展列，x[P,H] -> velocity[P,H]`。

适配工作：适配为 state/context/time 三输入，明确上下文和绝对年龄的条件融合；禁用 momentum_beta（设为 0），避免 previous_v 在不同 batch/solver 调用间造成隐藏状态。

依赖：torch（此 ODEFunc 文件）；完整方法另有 torchdiffeq/PHATE 等依赖，见快照元数据。

范围说明：当前仓库 2026-10 版本已改为小写 mioflow/ 路径，不能沿用旧教程 MIOFlow/models.py。条件包装后是组件混合，不等于原整套方法。

## 5. PRESCIENT

适合任务：development。建议：优先；接入成本：中到高。

以势函数对状态的梯度产生漂移，扩展直接输出速度之外的结构族。

官方仓库：[PRESCIENT](https://github.com/gifford-lab/prescient)；固定版本：`50971c7d495e8763eaa60f83af91f51555ed7ece`（提交日期 `2025-01-20T16:08:40-05:00`）。

源码许可：MIT。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://github.com/gifford-lab/prescient)。

### 拟注册 `flow/prescient_potential`

源码：[`prescient/train/model.py::AutoGenerator`](https://github.com/gifford-lab/prescient/blob/50971c7d495e8763eaa60f83af91f51555ed7ece/prescient/train/model.py#L36)。

接口：`_pot: state[P,H] -> scalar[P,1]；_drift: state[P,H] -> gradient[P,H]`。

适配工作：保留源实现 _drift 的符号（当前代码为正梯度）；评估器 no_grad 内需局部 enable_grad；训练需保留对势网络的高阶导数。若加入 context/time，须明确条件适配。

依赖：torch；numpy；原文件顶层导入 geomloss、sklearn、tqdm 等。

范围说明：不是把势函数标量当 velocity。先采用现有确定性 solver；随机扩散项属于后续 solver 能力，不在本次组件收集中默认引入。

## 6. scNym

适合任务：annotation, integration。建议：优先；接入成本：低到中。

已有成熟分类网络可作为结构参照；与当前 residual_mlp 有重叠，建议做结构去重。

官方仓库：[scNym](https://github.com/calico/scnym)；固定版本：`e19f4f74d726be8831ce4b381ac297fb90228054`（提交日期 `2026-04-04T14:56:24-07:00`）。

源码许可：Apache-2.0。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://github.com/calico/scnym)。

### 拟注册 `encoder/scnym_mlp`

源码：[`scnym/model.py::CellTypeCLF`](https://github.com/calico/scnym/blob/e19f4f74d726be8831ce4b381ac297fb90228054/scnym/model.py#L85)。

接口：`features[B,G] -> logits[B,C]；拆分类前主干后 -> state[B,H]`。

适配工作：复用原 embedding 主干，分类投影按现有 head 接口拆开；避免把全模型 logits 当成 embedding。

### 拟注册 `residual/scnym_resblock`

源码：[`scnym/model.py::ResBlock`](https://github.com/calico/scnym/blob/e19f4f74d726be8831ce4b381ac297fb90228054/scnym/model.py#L9)。

接口：`state[B,H] -> state[B,H]`。

适配工作：n_inputs 必须等于 n_hidden（源码直接加 identity，无跳连投影）；检查 BatchNorm 在训练小批次时的行为。

依赖：torch。

范围说明：不引入 MixMatch、域对抗训练策略；这里只收集网络结构。现有 residual_mlp 是 LayerNorm/GELU 系列，本实现为 BatchNorm/ReLU 系列。

## 7. SEDR

适合任务：spatial_niche, integration。建议：第二批；接入成本：中。

表达分支与图分支拼接可作为结构参照；单个 GCN 与已有 scBrain/GraphST 组件较接近。

官方仓库：[SEDR](https://github.com/JinmiaoChenLab/SEDR)；固定版本：`ef4836059a4ea49be3bf7c67008a44ffc16a2a0e`（提交日期 `2024-11-06T15:20:31+08:00`）。

源码许可：MIT。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://doi.org/10.1186/s13073-024-01283-x)。

### 拟注册 `encoder/sedr_mlp`

源码：[`SEDR/SEDR_module.py::full_block`](https://github.com/JinmiaoChenLab/SEDR/blob/ef4836059a4ea49be3bf7c67008a44ffc16a2a0e/SEDR/SEDR_module.py#L23)。

接口：`features[B,G] -> state[B,H]`。

适配工作：保留 Linear/BatchNorm/ELU/Dropout 顺序；可组合多个表达块。

### 拟注册 `spatial/sedr_gcn`

源码：[`SEDR/SEDR_module.py::GraphConvolution`](https://github.com/JinmiaoChenLab/SEDR/blob/ef4836059a4ea49be3bf7c67008a44ffc16a2a0e/SEDR/SEDR_module.py#L33)。

接口：`state[N,H], normalized_sparse_adj[N,N] -> state[N,H]`。

适配工作：复用一跳上下文构图和稀疏归一化，注意边方向；不能直接将完整多层 VGAE 包成单个一跳节点。

依赖：torch；完整管线还用 numpy/scipy/scanpy 等。

范围说明：decoder 中图重建、聚类目标不在本次结构收集范围。优先比较和已有 GCN 的重复度。

## 8. SpaGCN

适合任务：spatial_niche。建议：第二批；接入成本：中。

可补充带距离权重的空间卷积；纯 GraphConvolution 与已有算子重叠较高。

官方仓库：[SpaGCN](https://github.com/jianhuupenn/SpaGCN)；固定版本：`dc7a1c26ea0fdf4dfe7064adc7699be141b4871f`（提交日期 `2026-02-09T09:33:32-05:00`）。

源码许可：MIT。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://github.com/jianhuupenn/SpaGCN)。

### 拟注册 `spatial/spagcn_gcn`

源码：[`SpaGCN_package/SpaGCN/layers.py::GraphConvolution`](https://github.com/jianhuupenn/SpaGCN/blob/dc7a1c26ea0fdf4dfe7064adc7699be141b4871f/SpaGCN_package/SpaGCN/layers.py#L7)。

接口：`state[N,H], spatial_adjacency[N,N] -> state[N,H]`。

适配工作：保留权重和 bias，避免构建整张 dense N×N 图；若使用距离核，需把邻居权重及上下文几何接入数据契约。

依赖：torch；构图原管线还依赖 numpy/scipy/scanpy 等。

范围说明：当前缺少组织图像时只能标注空间位置版本，不能声称使用了 histology 信息。空间加权一跳版本与原完整图算法须区别记录。

## 9. scFoundation

适合任务：annotation, integration。建议：第二批；接入成本：中。

可训练软分箱表达嵌入，不是普通线性 patch 投影。

官方仓库：[scFoundation](https://github.com/biomap-research/scFoundation)；固定版本：`397631c495eddf9ad6644fc00c6ea8139e651245`（提交日期 `2025-11-23T08:31:05Z`）。

源码许可：Apache-2.0（代码；不涉及模型权重许可）。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://github.com/biomap-research/scFoundation)。

### 拟注册 `encoder/scfoundation_autobin`

源码：[`model/pretrainmodels/mae_autobin.py::AutoDiscretizationEmbedding2`](https://github.com/biomap-research/scFoundation/blob/397631c495eddf9ad6644fc00c6ea8139e651245/model/pretrainmodels/mae_autobin.py#L18)。

接口：`features[B,G,1] -> gene_tokens[B,G,H]`。

适配工作：增加最后一维并保留基因身份编码；显式设置不与真实表达值重合的 mask/pad sentinel。形状推导目前只特殊支持 panel_patches，需要扩展通用 token encoder 的输出长度。

依赖：torch；numpy；einops；原文件若触发其余模型导入需隔离选择。

范围说明：只复用 AutoBin 结构时可从头初始化，不必下载大模型权重。完整预训练模型仍有 19,264 基因面板和归一化契约，不能拿 256 基因输入直接声称等价。

## 10. scPRINT

适合任务：annotation, integration。建议：第二批；接入成本：中。

逐基因数值编码和可配置分类解码器，可补充 token/头结构。

官方仓库：[scPRINT](https://github.com/cantinilab/scPRINT)；固定版本：`e7c60766cbff58bb7280a1c8fe4b7461b1af1911`（提交日期 `2026-08-03T19:18:21+02:00`）。

源码许可：GPL-3.0（代码）。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://github.com/cantinilab/scPRINT)。

### 拟注册 `encoder/scprint_value`

源码：[`scprint/model/encoders.py::ContinuousValueEncoder`](https://github.com/cantinilab/scPRINT/blob/e7c60766cbff58bb7280a1c8fe4b7461b1af1911/scprint/model/encoders.py#L161)。

接口：`values[B,G] -> gene_tokens[B,G,H]`。

适配工作：源代码会 clamp(min=0)，当前有符号标准化表达不能直接输入；需单独声明非负 log-expression 输入分支，并扩展 token encoder 形状推导。

### 拟注册 `head/scprint_classifier`

源码：[`scprint/model/decoders.py::ClsDecoder`](https://github.com/cantinilab/scPRINT/blob/e7c60766cbff58bb7280a1c8fe4b7461b1af1911/scprint/model/decoders.py#L202)。

接口：`state[B,H] -> logits[B,C]`。

适配工作：映射 d_model、n_cls 与隐藏层参数；保留 LayerNorm/激活/Dropout 结构；源文件单独导入可避免完整 Lightning 模型依赖。

依赖：torch（所选两个文件）；完整框架依赖见 pyproject.toml。

范围说明：GeneEncoder 的基因 ID/词表不能省略；只提取分类头不代表复用了其基因网络推断能力或预训练知识。

## 11. scBERT

适合任务：annotation, integration。建议：第二批；接入成本：低到中。

Performer 随机特征线性注意力，区别于现有 SDPA/BERT 与自有 differential_linear。

官方仓库：[scBERT](https://github.com/TencentAILabHealthcare/scBERT)；固定版本：`262fd4b91f3f1c21a6e595d03d4ef423e16ffc99`（提交日期 `2023-12-13T16:42:32+08:00`）。

源码许可：GPL-3.0（代码）。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://doi.org/10.1038/s42256-022-00534-z)。

### 拟注册 `attention/scbert_performer`

源码：[`performer_pytorch/performer_pytorch.py::SelfAttention`](https://github.com/TencentAILabHealthcare/scBERT/blob/262fd4b91f3f1c21a6e595d03d4ef423e16ffc99/performer_pytorch/performer_pytorch.py#L336)。

接口：`tokens[B,T,H] -> tokens[B,T,H]`。

适配工作：选择纯 attention 层，不加载 PerformerLM 的固定基因词表/预训练权重；固定随机特征投影与重采样规则，验证其 checkpoint 可重载。

依赖：torch；einops；local_attention；以源文件 import 为准。

范围说明：当前 panel_patches 只有 16 tokens，线性注意力的计算优势未必明显；增加一种注意力结构不保证提高准确率。

## 12. CellOT

适合任务：development。建议：需先扩展接口；接入成本：高。

输入凸神经网络及其梯度运输映射，增加非 ODE 的端点预测结构。

官方仓库：[CellOT](https://github.com/bunnech/cellot)；固定版本：`522d2b953da8ad244fcf36f64521487fdf763788`（提交日期 `2024-11-01T00:34:58+01:00`）。

源码许可：BSD-3-Clause。原始 LICENSE 文件随快照保存；这是源码元数据记录，不替代完整条款。

论文/方法入口：[来源](https://github.com/bunnech/cellot)。

### 拟注册 `endpoint（新增 kind）/cellot_icnn_transport`

源码：[`cellot/networks/icnns.py::ICNN`](https://github.com/bunnech/cellot/blob/522d2b953da8ad244fcf36f64521487fdf763788/cellot/networks/icnns.py#L27)。

接口：`forward: state[P,H] -> scalar[P,1]；transport: state[P,H] -> endpoint[P,H]`。

适配工作：现有图仅有 flow+solver，需新增 endpoint/potential 契约；维持凸性参数化（例如源实现 softplus_W_kernels=True）。局部启用 autograd。不能把 transport 输出直接充作 velocity。

依赖：torch；numpy。

范围说明：细胞扰动和发育轨迹可提供结构参考，但与当前条件速度场任务不是同一输出语义。来源论文报告不等于在当前脑数据上已验证。

## 注册前需要完成的实际检查

1. 对照固定 SHA 直接复用上游模块，以 wrapper 做名称、输入输出和依赖隔离；不把同名通用层冒充特定方法组件。
2. 明确 token 长度、输出 rank/width、条件 metadata、原型状态或 endpoint 语义。模型内部的状态/随机性需要进入 checkpoint。
3. 验证梯度更新、独立任务冻结、重载一致性、一跳空间边界；需要 no_grad 中局部求导的模块要单独验证。
4. 新组件接入后先做合成数据和真实数据短训练，再以新实验目录建立比较；不改动正在运行的 50 代实验源码快照。

## 交付文件

- `candidate_components.json`：机器可读的候选注册清单，含拟注册 kind/name、真实源码符号、契约、适配说明及验证状态。
- `source_manifest.json` / `metadata/`：各仓库提交 SHA、提交时间、源码 URL、文件哈希、定义行号。
- `sources/`：已下载的官方模型源文件、README、依赖说明、许可证；是选择性源码快照，不是预装可运行环境。
- `BASELINE_COMPONENTS_20261026.md`：本文。
- `baseline_candidates_20261026.tar.gz`：上述资料打包，便于服务器端保存。

服务器资料目录：`/mnt/aliyunsb/zim_rider/nanobrain/baseline_candidates_20261026`。
现有 50 代实验目录：`/mnt/aliyunsb/zim_rider/nanobrain/brain/results/nanobrain/llm-graph-v2-components-50g-gpt61sol-20261023`。
